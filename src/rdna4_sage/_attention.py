"""int8 QK / fp8 PV attention for RDNA4: quantize Q, K, V and run the FlyDSL kernel."""
import json
import math
from importlib import resources

import torch

from . import _hip

_MANIFEST = json.loads(resources.files(__package__).joinpath("kernels/manifest.json").read_text())
KV_BLOCK = _MANIFEST["kv_block"]
HEAD_DIMS = tuple(_MANIFEST["head_dims"])
ARCHS = tuple(_MANIFEST["archs"])
DTYPES = {torch.bfloat16: "bf16", torch.float16: "f16"}
_I32_MAX = 2**31 - 1

_kernels = {}   # (device, kind, key) -> _hip.Kernel
_hadamard = {}  # (head_dim, dtype, device) -> +-1 matrix
_arch_cache = {}


class Unsupported(Exception):
    """The inputs fall outside what the packaged kernels handle; use another attention backend."""


def device_arch(device: torch.device) -> str:
    idx = device.index if device.index is not None else torch.cuda.current_device()
    if idx not in _arch_cache:
        _arch_cache[idx] = torch.cuda.get_device_properties(idx).gcnArchName.split(":")[0]
    return _arch_cache[idx]


def _kernel(device: torch.device, kind: str, key: str) -> _hip.Kernel:
    idx = device.index if device.index is not None else torch.cuda.current_device()
    k = _kernels.get((idx, kind, key))
    if k is None:
        arch = device_arch(device)
        meta = _MANIFEST["archs"][arch][kind][key]
        code = resources.files(__package__).joinpath(f"kernels/{arch}/{meta['file']}").read_bytes()
        k = _kernels[(idx, kind, key)] = _hip.Kernel(code, meta["name"], meta["args"], meta["threads"], meta["shared"], idx)
    return k


def _meta(device, kind, key):
    return _MANIFEST["archs"][device_arch(device)][kind][key]


def _hadamard_pm1(d, dtype, device):
    key = (d, dtype, device)
    h = _hadamard.get(key)
    if h is None:
        h = torch.ones(1, 1)
        while h.shape[0] < d:
            h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
        h = _hadamard[key] = h.to(device=device, dtype=dtype).contiguous()
    return h


def _bsh(t: torch.Tensor, layout: str):
    """(batch, seq, head) strides; the quantize kernels are built for 16-element aligned bases and strides."""
    if t.stride(-1) != 1 or t.data_ptr() % 16:
        t = t.contiguous()
    st = (t.stride(0), t.stride(2), t.stride(1)) if layout == "HND" else (t.stride(0), t.stride(1), t.stride(2))
    if any(s % 16 for s in st):
        t = t.contiguous()
        st = (t.stride(0), t.stride(2), t.stride(1)) if layout == "HND" else (t.stride(0), t.stride(1), t.stride(2))
    return t, st


def check_supported(q, k, v, layout, is_causal=False):
    """Raise Unsupported with the reason if the packaged kernels cannot run these inputs."""
    if layout not in ("HND", "NHD"):
        raise Unsupported(f"layout {layout!r}")
    if not (q.is_cuda and k.device == q.device and v.device == q.device):
        raise Unsupported("tensors must be on one ROCm GPU")
    if q.dtype not in DTYPES or k.dtype != q.dtype or v.dtype != q.dtype:
        raise Unsupported(f"dtype {q.dtype}/{k.dtype}/{v.dtype} (fp16 or bf16 required)")
    if q.dim() != 4 or k.dim() != 4 or v.dim() != 4:
        raise Unsupported("4D tensors required")
    arch = device_arch(q.device)
    if arch not in ARCHS:
        raise Unsupported(f"GPU {arch} is not RDNA4 (kernels built for {', '.join(ARCHS)} only)")
    hd = q.shape[-1]
    if hd not in HEAD_DIMS or k.shape[-1] != hd or v.shape[-1] != hd:
        raise Unsupported(f"head_dim {hd} (supported {HEAD_DIMS})")
    h_ax, s_ax = (1, 2) if layout == "HND" else (2, 1)
    if q.shape[0] != k.shape[0] or k.shape[0] != v.shape[0] or k.shape[h_ax] != v.shape[h_ax] or k.shape[s_ax] != v.shape[s_ax]:
        raise Unsupported("q/k/v batch, k/v heads and k/v lengths must match")
    if q.shape[h_ax] % k.shape[h_ax]:
        raise Unsupported(f"heads {q.shape[h_ax]} not a multiple of kv heads {k.shape[h_ax]}")
    if max(q.numel(), k.numel()) > _I32_MAX or k.shape[s_ax] == 0 or q.shape[s_ax] == 0:
        raise Unsupported("tensor too large for 32-bit indexing or empty")
    if is_causal and q.shape[s_ax] != k.shape[s_ax]:
        raise Unsupported("causal attention needs q_len == kv_len")


def attention(q, k, v, layout="HND", sm_scale=None, is_causal=False):
    """Attention without mask, optionally causal (q_len == kv_len). q: (B, H, Sq, D) for "HND" or (B, Sq, H, D) for
    "NHD"; k/v alike, with H a multiple of the kv head count. Returns (B, Sq, H, D) contiguous in q's dtype."""
    check_supported(q, k, v, layout, is_causal)
    dev = q.device
    if layout == "HND":
        B, H, Sq, D = q.shape
        Hkv, Skv = k.shape[1], k.shape[2]
    else:
        B, Sq, H, D = q.shape
        Skv, Hkv = k.shape[1], k.shape[2]
    key = f"d{D}_{DTYPES[q.dtype]}"
    q, sq_ = _bsh(q, layout)
    k, sk_ = _bsh(k, layout)
    v, sv_ = _bsh(v, layout)
    skv_pad = (Skv + KV_BLOCK - 1) // KV_BLOCK * KV_BLOCK
    stream = torch.cuda.current_stream(dev).cuda_stream

    q8 = torch.empty((B, Sq, H, D), device=dev, dtype=torch.int8)
    qs = torch.empty((B, H, Sq), device=dev, dtype=torch.float32)
    k8 = torch.empty((B, skv_pad, Hkv, D), device=dev, dtype=torch.int8)
    ks = torch.empty((B, Hkv, skv_pad // KV_BLOCK), device=dev, dtype=torch.float32)
    v8 = torch.empty((B, skv_pad, Hkv, D), device=dev, dtype=torch.float8_e4m3fn)
    # per-(batch, kv head, channel) V amax; the kernels store V * 448 / amax and undo it in the epilogue
    vs = torch.linalg.vector_norm(v, ord=float("inf"), dim=2 if layout == "HND" else 1, dtype=torch.float32)
    qm = _meta(dev, "quant_q", key)
    hm = _hadamard_pm1(qm["rot_block"], q.dtype, dev) if qm["rot"] else q

    _kernel(dev, "quant_q", key).launch(B * H, (Sq + qm["rows"] - 1) // qm["rows"], stream,
                                        q.data_ptr(), q8.data_ptr(), qs.data_ptr(), hm.data_ptr(), Sq, *sq_, H, 0, 0)
    _kernel(dev, "quant_kv", key).launch(B * Hkv, skv_pad // KV_BLOCK, stream,
                                         k.data_ptr(), v.data_ptr(), k8.data_ptr(), ks.data_ptr(), v8.data_ptr(), vs.data_ptr(),
                                         hm.data_ptr(), Skv, skv_pad, *sk_, *sv_, Hkv, 0, 0)
    if sm_scale is not None and not math.isclose(sm_scale, D ** -0.5, rel_tol=1e-6):
        qs.mul_(sm_scale * math.sqrt(D))  # the kernel applies 1/sqrt(D); fold any other scale into the Q scales

    out = torch.empty((B, Sq, H, D), device=dev, dtype=q.dtype)
    akey = key + "_causal" if is_causal else key
    am = _meta(dev, "attn", akey)
    p = qs.data_ptr()
    _kernel(dev, "attn", akey).launch(B * H * ((Sq + am["block_m"] - 1) // am["block_m"]), 1, stream,
                                     q8.data_ptr(), k8.data_ptr(), v8.data_ptr(), out.data_ptr(), Sq, skv_pad, Skv,
                                     qs.data_ptr(), ks.data_ptr(), vs.data_ptr(), p, p, p, H, Hkv)
    return out
