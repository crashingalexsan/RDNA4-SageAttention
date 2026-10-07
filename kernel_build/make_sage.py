"""Derive kernels/attention/flash_attn_sage_gfx120x.py from the PR's fp8 kernel.

QK^T: int8 WMMA, i32 accumulation chained over head_dim, per-token Q scale and
per-32-token-block K scale. PV: unchanged fp8 E4M3 path (per-tensor V descale).
"""
import sys
from pathlib import Path

root = Path(sys.argv[1])
src = (root / "kernels/attention/flash_attn_fp8_gfx120x.py").read_text()


def sub(old, new, count=1):
    global src
    n = src.count(old)
    if n != count:
        raise SystemExit(f"expected {count} match(es), got {n}: {old[:70]!r}")
    src = src.replace(old, new)


sub('''"""Flash Attention FP8 (E4M3FN) forward kernel for gfx120x (RDNA4; HW may report gfx1201).
''', '''"""SageAttention-style forward kernel for gfx120x: int8 QK^T, fp8 PV.

Derived from ``flash_attn_fp8_gfx120x``. GEMM1 runs iu8 WMMA and keeps the i32
accumulator across all head_dim K-steps, converting to f32 once per tile.
QDescale is a per-token fp32 scale [B, H, Sq]; KDescale is a per-block fp32
scale [B, Hkv, Skv_pad / 32] (one scale per BLOCK_N tile, applied to the
logits before the online softmax so tiles with different scales mix
correctly). V stays per-tensor E4M3 and P is cast to E4M3 as in the fp8 kernel.
Bias is not supported.

Original fp8 notes follow.
''')
sub('KERNEL_NAME = "flash_attn_func_fp8_gfx120x_kernel"', 'KERNEL_NAME = "flash_attn_func_sage_gfx120x_kernel"')
sub("def build_flash_attn_func_fp8_module_primary(", "def build_flash_attn_func_sage_module_primary(")
sub("    def flash_attn_func_fp8_kernel(", "    def flash_attn_func_sage_kernel(")
sub("        launcher = flash_attn_func_fp8_kernel(", "        launcher = flash_attn_func_sage_kernel(")
src = src.replace("launch_flash_attn_fp8_func", "launch_flash_attn_sage_func")

sub('''    if sm_scale is None:
        sm_scale = 1.0 / host_math.sqrt(head_dim)
''', '''    assert not _FP8_E5M2, "sage gfx120x FA keeps V/P in E4M3"
    assert not (has_attn_bias or has_per_head_bias), "sage gfx120x FA does not support bias"
    if sm_scale is None:
        sm_scale = 1.0 / host_math.sqrt(head_dim)
''')

# Only V keeps a scalar descale; Q/K scales are per-row / per-tile arrays.
sub('''        q_descale = _load_descale(QDescale)
        k_descale = _load_descale(KDescale)
        v_descale = _load_descale(VDescale)
''', '''        v_descale = _load_descale(VDescale)
        q_scale_ptr = fx.recast_iter(fx.PointerType.get(fx.Float32.ir_type, QDescale.address_space), QDescale)
        k_scale_ptr = fx.recast_iter(fx.PointerType.get(fx.Float32.ir_type, KDescale.address_space), KDescale)
''')

sub('''        def wmma_acc(a_v8: fx.Vector, b_v8: fx.Vector, c_v8: fx.Vector) -> fx.Vector:
''', '''        wmma_atom_i8 = fx.make_mma_atom(
            fx.rocdl.WMMA(WMMA_M, WMMA_N, WMMA_K, fx.Int8, fx.Int32, sign_a=True, sign_b=True, clamp=False)
        )

        def wmma_acc_i32(a_v8: fx.Vector, b_v8: fx.Vector, c_v8: fx.Vector) -> fx.Vector:
            a_frag = fx.make_rmem_tensor(8, fx.Int8)
            b_frag = fx.make_rmem_tensor(8, fx.Int8)
            c_frag = fx.make_rmem_tensor(8, fx.Int32)
            a_frag.store(Vec(a_v8))
            b_frag.store(Vec(b_v8))
            c_frag.store(Vec(c_v8))
            fx.gemm(wmma_atom_i8, c_frag, [a_frag], [b_frag], c_frag)
            return Vec(c_frag.load())

        def wmma_acc(a_v8: fx.Vector, b_v8: fx.Vector, c_v8: fx.Vector) -> fx.Vector:
''')

# Per-lane Q scale: each lane owns one query row (lane16).
sub('''        c_logit_scale = fx.Float32(sm_scale * _LOG2E) * q_descale * k_descale
        c_zero_v8f32 = Vec.filled(8, 0.0, fx.Float32)
''', '''        _q_scale_idx = (fx.Int64(batch_idx) * fx.Int64(NUM_HEADS) + fx.Int64(head_idx)) * fx.Int64(seq_len) + q_row_safe
        q_scale = fx.Float32(fx.ptr_load(q_scale_ptr + fx.Int32(_q_scale_idx)))
        c_logit_scale = fx.Float32(sm_scale * _LOG2E) * q_scale
        c_zero_v8f32 = Vec.filled(8, 0.0, fx.Float32)
        c_zero_v8i32 = Vec.filled(8, 0, fx.Int32)
        _k_scale_row = (fx.Int64(batch_idx) * fx.Int64(NUM_KV_HEADS) + fx.Int64(kv_head_idx)) * (
            fx.Int64(seq_len_kv) // fx.Int64(BLOCK_N)
        )
''')

sub('''            # S = K @ Q^T
            s_accs = [c_zero_v8f32 for _ in range(NUM_S_ACCS)]
''', '''            # S = K @ Q^T in i32; one f32 conversion per tile.
            s_accs = [c_zero_v8i32 for _ in range(NUM_S_ACCS)]
            k_scale = fx.Float32(
                fx.ptr_load(k_scale_ptr + fx.Int32(_k_scale_row + fx.Int64(kv_block_start) // fx.Int64(BLOCK_N)))
            )
''')
sub('''                        s_accs[acc_idx_a] = wmma_acc(k_pack_a, q_b_packs[ks], s_accs[acc_idx_a])
                        s_accs[acc_idx_b] = wmma_acc(k_pack_b, q_b_packs[ks], s_accs[acc_idx_b])
''', '''                        s_accs[acc_idx_a] = wmma_acc_i32(k_pack_a, q_b_packs[ks], s_accs[acc_idx_a])
                        s_accs[acc_idx_b] = wmma_acc_i32(k_pack_b, q_b_packs[ks], s_accs[acc_idx_b])
''')
sub('''                    s_raw.append(Vec(s_accs[st])[r])
''', '''                    s_raw.append(fx.Float32(Vec(s_accs[st])[r]) * k_scale)
''')

sub('''        qk_scale = fx.Float32(sm_scale) * q_descale * k_descale
''', '''        qk_scale = fx.Float32(sm_scale) * q_scale
''')

src = src.replace("build_flash_attn_func_fp8_module = build_flash_attn_func_fp8_module_primary",
                  "build_flash_attn_func_sage_module = build_flash_attn_func_sage_module_primary")
src = src.replace("build_flash_attn_func_fp8_module_gfx120x = build_flash_attn_func_fp8_module_primary",
                  "build_flash_attn_func_sage_module_gfx120x = build_flash_attn_func_sage_module_primary")
src = src.replace('f"fp8 gfx120x FA: seq_len_kv', 'f"sage gfx120x FA: seq_len_kv')

# V^T[d][kv] in LDS: the PV A-operand (8 consecutive kv for one d) becomes one
# 8-byte LDS read instead of 8 strided byte reads. Transpose happens at store time.
sub('''    LDS_V_TILE_SIZE = BLOCK_N * V_STRIDE
''', '''    VT_STRIDE = BLOCK_N + 8
    LDS_V_TILE_SIZE = HEAD_DIM * VT_STRIDE
''')
sub('''    NUM_V_VECS = NUM_BATCHES_KV * V_SUBVECS
''', '''    # V^T staging: one work item = 4 kv rows x 8 d columns, packed in registers
    # into 8 dwords (4 kv bytes each) and written with 8 ds_write_b32.
    VT_COL_GROUPS = HEAD_DIM // 8
    VT_ITEMS_TOTAL = (BLOCK_N // 4) * VT_COL_GROUPS
    VT_ITEMS = (VT_ITEMS_TOTAL + BLOCK_SIZE - 1) // BLOCK_SIZE
    VT_NEEDS_GUARD = VT_ITEMS * BLOCK_SIZE != VT_ITEMS_TOTAL
    NUM_V_VECS = VT_ITEMS * 4
''')
start = src.index("        def coop_load_v_global(")
end = src.index("        q_row = q_start + wave_q_offset + lane16")
src = src[:start] + '''        def _vt_item(it: int):
            item = tid + fx.Uint64(it * BLOCK_SIZE)
            if const_expr(VT_NEEDS_GUARD):
                item = (item < fx.Uint64(VT_ITEMS_TOTAL)).select(item, fx.Uint64(VT_ITEMS_TOTAL - 1))
            return fx.Int64(item // VT_COL_GROUPS) * 4, fx.Int64(item % VT_COL_GROUPS) * 8

        def coop_load_v_global(tile_start: fx.Int32 | fx.Int64 | int) -> list[fx.Vector]:
            tile_start = fx.Int64(tile_start)
            vecs = []
            for it in range_constexpr(VT_ITEMS):
                row0, col0 = _vt_item(it)
                for r in range_constexpr(4):
                    vecs.append(_load_global_i8_vec_via_i32(v_buf_ptr, v_idx(tile_start + row0 + fx.Int64(r), col0), 8))
            return vecs

        _c255 = fx.Int32(255)

        def _perm(hi, lo, sel: int):
            return fx.Int32(fx.rocdl.perm_b32(fx.Int32(hi).ir_value(), fx.Int32(lo).ir_value(), fx.Int32(sel).ir_value()))

        def _vt_store_item(v_base, row0, col0, rows4) -> None:
            # 4x4 byte transpose per dword column with v_perm_b32 (2 perms per output dword).
            words = [Vec(rv).bitcast(fx.Int32) for rv in rows4]
            for half in range_constexpr(2):
                w = [Vec(words[r])[half] for r in range_constexpr(4)]
                t01_lo = _perm(w[1], w[0], 0x05010400)
                t01_hi = _perm(w[1], w[0], 0x07030602)
                t23_lo = _perm(w[3], w[2], 0x05010400)
                t23_hi = _perm(w[3], w[2], 0x07030602)
                outs = [
                    _perm(t23_lo, t01_lo, 0x05040100),
                    _perm(t23_lo, t01_lo, 0x07060302),
                    _perm(t23_hi, t01_hi, 0x05040100),
                    _perm(t23_hi, t01_hi, 0x07060302),
                ]
                for jj in range_constexpr(4):
                    idx = v_base + (col0 + fx.Int64(half * 4 + jj)) * VT_STRIDE + row0
                    view = fx.make_view(_lds_i32_ptr() + fx.Int32(idx) // fx.Int32(4), fx.make_layout(1, 1))
                    view.store(Vec.from_elements([outs[jj]], fx.Int32))

        def coop_store_v_lds(vecs: list[fx.Vector], buf_id: fx.Int32 | int = 0) -> None:
            v_base = v_buf_base(buf_id)
            for it in range_constexpr(VT_ITEMS):
                row0, col0 = _vt_item(it)
                rows4 = vecs[it * 4:(it + 1) * 4]
                if const_expr(VT_NEEDS_GUARD):
                    if tid + fx.Uint64(it * BLOCK_SIZE) < fx.Uint64(VT_ITEMS_TOTAL):
                        _vt_store_item(v_base, row0, col0, rows4)
                else:
                    _vt_store_item(v_base, row0, col0, rows4)

''' + src[end:]
sub('''                d_pos = fx.Int64(dc_val * D_CHUNK) + lane16
                v_elems = []
                for k_sub in range_constexpr(8):
                    kv_row = fx.Int64(st_kv_base_val + pks_val * PV_K_STEP) + klane * WMMA_LANE_K + fx.Int64(k_sub)
                    v_lds_idx = v_base + kv_row * V_STRIDE + d_pos
                    v_elems.append(fx.ptr_load(lds_kv + fx.Int32(v_lds_idx)))
                return Vec.from_elements(v_elems, elem_dtype)
''', '''                d_pos = fx.Int64(dc_val * D_CHUNK) + lane16
                kv_row0 = fx.Int64(st_kv_base_val + pks_val * PV_K_STEP) + klane * WMMA_LANE_K
                return Vec(lds_load_i8(v_base + d_pos * VT_STRIDE + kv_row0, 8))
''')

# max(s * ks) == ks * max(s) for ks > 0: scale the row max once, fold ks into the exp2 FMA.
sub('''                    s_raw.append(fx.Float32(Vec(s_accs[st])[r]) * k_scale)
''', '''                    s_raw.append(fx.Float32(Vec(s_accs[st])[r]))
''')
sub('''            row_max = _fmax(local_max, peer_max)
''', '''            row_max = _fmul(_fmax(local_max, peer_max), k_scale)
''')
sub('''                diff = fx.math.fma(
                    s_raw[r],
                    c_logit_scale,
                    neg_scaled_max,
                )
''', '''                diff = fx.math.fma(
                    s_raw[r],
                    c_tile_scale,
                    neg_scaled_max,
                )
''')
sub('''            scaled_max = _fmul(c_logit_scale, m_new_raw)
''', '''            scaled_max = _fmul(c_logit_scale, m_new_raw)
            c_tile_scale = _fmul(c_logit_scale, k_scale)
''')

# Keep a stale running max until the tile max exceeds it by 2^8 (P <= 256 still fits
# E4M3), and skip the O rescale when no lane moved its max.
sub('''            m_new_raw = _fmax(m_running, row_max)
''', '''            m_cand = _fmax(m_running, row_max)
            need_rescale = _fmul(_fsub(m_cand, m_running), c_logit_scale) > fx.Float32(8.0)
            m_new_raw = need_rescale.select(m_cand, m_running)
''')
sub('''            corr_vec = Vec.from_elements([corr], fx.Float32).broadcast_to(8)
            for dc in range_constexpr(D_CHUNKS):
                o_accs[dc] = _fmul(o_accs[dc], corr_vec)

            coop_store_v_lds''', '''            if need_rescale:
                corr_vec = Vec.from_elements([corr], fx.Float32).broadcast_to(8)
                o_resc = []
                for dc in range_constexpr(D_CHUNKS):
                    o_resc.append(_fmul(o_accs[dc], corr_vec))
                o_accs = o_resc

            coop_store_v_lds''')

sub('''                    v = (v > _max).select(_max, (v < _nmax).select(_nmax, v))
''', '')

(root / "kernels/attention/flash_attn_sage_gfx120x.py").write_text(src)
print("wrote", root / "kernels/attention/flash_attn_sage_gfx120x.py")
