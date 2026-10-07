"""Triton quantization kernels for the gfx120x int8-QK / fp8-PV attention kernel.

Compiled ahead of time by build_kernels.py; the package launches the code objects itself.
Q and K are rotated by an unnormalized +-1 Hadamard matrix Hm of size RB before int8 quantization (block-diagonal
when head_dim = 2 * RB): Q K^T only scales by RB, and outlier channels are spread over the rotation block so the
scales lose less. RB is capped at 128 so the matrix operand fits in LDS.
"""
import triton
import triton.language as tl


@triton.jit
def _load_qk(X, b, h, rows, seq, sxb, sxs, sxh, Hm,
             D: tl.constexpr, DP: tl.constexpr, ROT: tl.constexpr, RB: tl.constexpr):
    # One [rows, D] tile of Q or K (any B/S/H strides, D contiguous) -> optional Hadamard, fp32.
    base = X + b * sxb + rows[:, None] * sxs + h * sxh
    rm = rows[:, None] < seq
    if ROT:
        r = tl.arange(0, RB)
        hm = tl.load(Hm + r[:, None] * RB + r[None, :])  # +-1 entries, products exact, fp32 accumulation
        if DP == RB:
            x = tl.dot(tl.load(base + r[None, :], mask=rm, other=0.0), hm)
        else:
            ya = tl.dot(tl.load(base + r[None, :], mask=rm, other=0.0), hm)
            yb = tl.dot(tl.load(base + RB + r[None, :], mask=rm, other=0.0), hm)
            x = tl.reshape(tl.permute(tl.join(ya, yb), (0, 2, 1)), (rows.shape[0], DP))  # [ya | yb]
    else:
        cols = tl.arange(0, DP)
        x = tl.load(base + cols[None, :], mask=rm & (cols[None, :] < D), other=0.0).to(tl.float32)
    return x


@triton.jit
def _quant_q(X, Y, S, Hm, pid_bh, pid_r, seq, sxb, sxs, sxh,
             H, D: tl.constexpr, DP: tl.constexpr, BS: tl.constexpr, ROT: tl.constexpr, RB: tl.constexpr):
    # per (batch, head, token) int8 scale; Y is BSHD contiguous, S is [B, H, seq]
    b = pid_bh // H
    h = pid_bh % H
    rows = pid_r * BS + tl.arange(0, BS)
    cols = tl.arange(0, DP)
    x = _load_qk(X, b, h, rows, seq, sxb, sxs, sxh, Hm, D, DP, ROT, RB)
    scale = tl.maximum(tl.max(tl.abs(x), axis=1), 1e-12) / 127.0
    y = x / scale[:, None]
    tl.store(Y + ((b * seq + rows[:, None]) * H + h) * D + cols[None, :], tl.where(y >= 0, y + 0.5, y - 0.5).to(tl.int8),
             mask=(rows[:, None] < seq) & (cols[None, :] < D))
    if ROT:
        scale = scale / RB  # unnormalized H: (QH)(KH)^T = RB * Q K^T
    tl.store(S + pid_bh * seq + rows, scale, mask=rows < seq)


@triton.jit
def _quant_kv(K, V, K8, KS, V8, VS, Hm, pid_bh, blk, seq, seq_pad, skb, sks, skh, svb, svs, svh,
              H, D: tl.constexpr, DP: tl.constexpr, BS: tl.constexpr, ROT: tl.constexpr, RB: tl.constexpr):
    # one int8 K scale per (batch, head, BS-token block) and the matching fp8 V block; pad rows become zeros.
    # V is scaled by 448 / its per-(batch, head, channel) amax VS so every channel uses the full E4M3 range.
    b = pid_bh // H
    h = pid_bh % H
    krows = blk * BS + tl.arange(0, BS)
    cols = tl.arange(0, DP)
    cm = cols[None, :] < D
    dst = ((b * seq_pad + krows[:, None]) * H + h) * D + cols[None, :]
    kx = _load_qk(K, b, h, krows, seq, skb, sks, skh, Hm, D, DP, ROT, RB)
    kscale = tl.maximum(tl.max(tl.abs(kx)), 1e-12) / 127.0
    ky = kx / kscale
    tl.store(K8 + dst, tl.where(ky >= 0, ky + 0.5, ky - 0.5).to(tl.int8), mask=cm)
    tl.store(KS + pid_bh * (seq_pad // BS) + blk, kscale)
    xv = tl.load(V + b * svb + krows[:, None] * svs + h * svh + cols[None, :], mask=(krows[:, None] < seq) & cm, other=0.0)
    vs = tl.maximum(tl.load(VS + pid_bh * D + cols, mask=cols < D, other=1.0), 1e-30)
    xv = tl.minimum(tl.maximum(xv.to(tl.float32) * (448.0 / vs)[None, :], -448.0), 448.0)
    tl.store(V8 + dst, xv.to(tl.float8e4nv), mask=cm)


@triton.jit
def _quant_q_kernel(X, Y, S, Hm, seq, sxb, sxs, sxh,
                    H, D: tl.constexpr, DP: tl.constexpr, BS: tl.constexpr, ROT: tl.constexpr, RB: tl.constexpr):
    _quant_q(X, Y, S, Hm, tl.program_id(0), tl.program_id(1), seq, sxb, sxs, sxh, H, D, DP, BS, ROT, RB)


@triton.jit
def _quant_kv_kernel(K, V, K8, KS, V8, VS, Hm, seq, seq_pad, skb, sks, skh, svb, svs, svh,
                     H, D: tl.constexpr, DP: tl.constexpr, BS: tl.constexpr, ROT: tl.constexpr, RB: tl.constexpr):
    _quant_kv(K, V, K8, KS, V8, VS, Hm, tl.program_id(0), tl.program_id(1), seq, seq_pad, skb, sks, skh, svb, svs, svh,
              H, D, DP, BS, ROT, RB)
