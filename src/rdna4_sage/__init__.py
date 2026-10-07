"""SageAttention-style int8 QK / fp8 PV attention for AMD RDNA4 (gfx1200 / gfx1201).

Ahead-of-time compiled FlyDSL attention + Triton quantization code objects, launched through the HIP
runtime of a ROCm build of torch. Works on Windows and Linux with no compiler, Triton or FlyDSL at runtime.

    import rdna4_sage
    out = rdna4_sage.sageattn(q, k, v, tensor_layout="HND")   # drop-in for sageattention.sageattn
"""
import os

import torch
from torch.nn.attention.bias import causal_lower_right

from . import ops
from ._attention import ARCHS, HEAD_DIMS, Unsupported, check_supported, device_arch

__all__ = ["sageattn", "attention", "is_available", "Unsupported", "ARCHS", "HEAD_DIMS"]
__version__ = "0.2.1"

# The C binding behind F.scaled_dot_product_attention. Apps such as SD.Next replace F.scaled_dot_product_attention
# with sageattn, so falling back through F would call back into sageattn forever.
_torch_sdpa = torch._C._nn.scaled_dot_product_attention

# Calls below these sizes run torch SDPA: the fixed quantization pass costs more than int8 attention saves.
MIN_WORK = int(os.environ.get("RDNA4_SAGE_MIN_WORK", 64 * 2**20))  # batch * heads * q_len * kv_len
MIN_KV = int(os.environ.get("RDNA4_SAGE_MIN_KV", 1024))           # short text contexts stay on SDPA


def is_available(device=None) -> bool:
    """True when the current (or given) GPU is RDNA4 (gfx1200 / gfx1201), the only GPUs the kernels are built for."""
    if not torch.cuda.is_available() or torch.version.hip is None:
        return False
    dev = torch.device("cuda", torch.cuda.current_device() if device is None else torch.device(device).index or 0)
    return device_arch(dev) in ARCHS


def attention(q, k, v, tensor_layout="HND", sm_scale=None, is_causal=False):
    """Run the int8/fp8 kernel; raises Unsupported if it cannot. Output in the input layout and dtype.
    Causal attention needs q_len == kv_len."""
    out = ops.attention(q, k, v, tensor_layout, float(sm_scale or 0.0), bool(is_causal))
    return out.transpose(1, 2) if tensor_layout == "HND" else out


def _sdpa(q, k, v, tensor_layout, is_causal, sm_scale, attn_mask):
    if tensor_layout == "NHD":
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))
    gqa = q.shape[1] != k.shape[1]
    Sq, Skv = q.shape[2], k.shape[2]
    if is_causal and (Sq != Skv or attn_mask is not None):
        # SageAttention / FlashAttention semantics: bottom-right aligned causal mask
        causal = causal_lower_right(Sq, Skv)._materialize(q.device)
        if attn_mask is None:
            attn_mask = causal
        else:
            attn_mask = (attn_mask.masked_fill(~causal, float("-inf")) if attn_mask.dtype != torch.bool else attn_mask & causal)
        is_causal = False
    out = _torch_sdpa(q, k, v, attn_mask=attn_mask, is_causal=is_causal, scale=sm_scale, enable_gqa=gqa)
    return out.transpose(1, 2) if tensor_layout == "NHD" else out


def _worth_it(q, k, tensor_layout, is_causal=False):
    if tensor_layout == "HND":
        B, H, Sq, _ = q.shape
        Skv = k.shape[2]
    else:
        B, Sq, H, _ = q.shape
        Skv = k.shape[1]
    work = B * H * Sq * Skv // (2 if is_causal else 1)
    return Skv >= MIN_KV and work >= MIN_WORK


def sageattn(q, k, v, tensor_layout="HND", is_causal=False, sm_scale=None, attn_mask=None, return_lse=False, **kwargs):
    """Drop-in for sageattention.sageattn. Uses the RDNA4 kernel when it applies and pays off, torch SDPA otherwise
    (masks, causal with q_len != kv_len, LSE, unsupported shapes or GPUs, small problems). Causal attention is
    bottom-right aligned as in SageAttention / FlashAttention. Extra SageAttention kwargs are ignored."""
    if return_lse:
        raise NotImplementedError("rdna4_sage: return_lse is not supported")
    if attn_mask is not None or not _worth_it(q, k, tensor_layout, is_causal):
        return _sdpa(q, k, v, tensor_layout, is_causal, sm_scale, attn_mask)
    try:
        check_supported(q, k, v, tensor_layout, is_causal)
    except Unsupported:
        return _sdpa(q, k, v, tensor_layout, is_causal, sm_scale, attn_mask)
    return attention(q, k, v, tensor_layout, sm_scale, is_causal)
