"""SageAttention 2.2 API on top of rdna4_sage (AMD RDNA4, gfx1200 / gfx1201).

Every public function of SageAttention 2.2 is provided with the same signature. They all run the same
rdna4_sage path (int8 Q K^T with Hadamard rotation, fp8 P V with per-channel V scales); kernel-selection
arguments such as qk_quant_gran, pv_accum_dtype, smooth_k, smooth_v or value_dtype are accepted and ignored.
Calls the RDNA4 kernels cannot run or would run slower (masks, small problems, unsupported head dims or GPUs)
use torch SDPA, so the functions work on any device torch supports.
"""
import math
from typing import Any, Optional

import torch
import torch.nn.functional as F

import rdna4_sage


def _to_hnd(t, tensor_layout):
    if tensor_layout not in ("HND", "NHD"):
        raise ValueError(f"tensor_layout must be 'HND' or 'NHD', got {tensor_layout!r}")
    return t if tensor_layout == "HND" else t.transpose(1, 2)


def _lse(q, k, tensor_layout, is_causal, sm_scale):
    """logsumexp over keys of Q K^T * sm_scale, shape [B, H, q_len], fp32 (SageAttention's return_lse output).
    Computed in chunks with torch; slower than the attention itself, as in callers that need it rarely."""
    q, k = _to_hnd(q, tensor_layout), _to_hnd(k, tensor_layout)
    B, H, Sq, D = q.shape
    Skv = k.shape[2]
    if k.shape[1] != H:
        k = k.repeat_interleave(H // k.shape[1], dim=1)
    scale = D ** -0.5 if sm_scale is None else sm_scale
    out = torch.empty((B, H, Sq), device=q.device, dtype=torch.float32)
    rows = max(1, (64 * 2**20) // max(1, Skv))  # ~256 MB of fp32 scores per chunk and head
    cols = torch.arange(Skv, device=q.device)
    for b in range(B):
        for h in range(H):
            kf = k[b, h].float()
            for s in range(0, Sq, rows):
                sc = q[b, h, s:s + rows].float() @ kf.T * scale
                if is_causal:  # bottom-right aligned, as the kernels
                    r = torch.arange(s, s + sc.shape[0], device=q.device)[:, None]
                    sc.masked_fill_(cols[None, :] > r + (Skv - Sq), float("-inf"))
                out[b, h, s:s + rows] = torch.logsumexp(sc, dim=-1)
    return out


def sageattn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    tensor_layout: str = "HND",
    is_causal: bool = False,
    sm_scale: Optional[float] = None,
    return_lse: bool = False,
    **kwargs: Any,
):
    """Attention for [B, H, L, D] ("HND") or [B, L, H, D] ("NHD") fp16 / bf16 tensors; returns the input layout and
    dtype, plus the row logsumexp [B, H, q_len] when return_lse is True. Causal masks are bottom-right aligned."""
    out = rdna4_sage.sageattn(q, k, v, tensor_layout=tensor_layout, is_causal=is_causal, sm_scale=sm_scale,
                             attn_mask=kwargs.get("attn_mask"))
    if return_lse:
        return out, _lse(q, k, tensor_layout, is_causal, sm_scale)
    return out


def sageattn_varlen(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    is_causal: bool = False,
    sm_scale: Optional[float] = None,
    smooth_k: bool = True,
    **kwargs: Any,
) -> torch.Tensor:
    """Packed variable-length attention: q [total_q, H, D], k / v [total_k, Hkv, D], cu_seqlens [batch + 1]."""
    cq = cu_seqlens_q.tolist()
    ck = cu_seqlens_k.tolist()
    out = torch.empty_like(q)
    for i in range(len(cq) - 1):
        if cq[i + 1] == cq[i]:
            continue
        o = sageattn(q[None, cq[i]:cq[i + 1]], k[None, ck[i]:ck[i + 1]], v[None, ck[i]:ck[i + 1]], tensor_layout="NHD",
                     is_causal=is_causal, sm_scale=sm_scale)
        out[cq[i]:cq[i + 1]] = o[0]
    return out


def _variant(q, k, v, tensor_layout="HND", is_causal=False, sm_scale=None, return_lse=False, **kwargs):
    return sageattn(q, k, v, tensor_layout=tensor_layout, is_causal=is_causal, sm_scale=sm_scale, return_lse=return_lse,
                    attn_mask=kwargs.get("attn_mask"))


def sageattn_qk_int8_pv_fp16_triton(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, tensor_layout: str = "HND",
                                    quantization_backend: str = "triton", is_causal: bool = False,
                                    attn_mask: Optional[torch.Tensor] = None, sm_scale: Optional[float] = None,
                                    smooth_k: bool = True, return_lse: bool = False, **kwargs: Any):
    return _variant(q, k, v, tensor_layout, is_causal, sm_scale, return_lse, attn_mask=attn_mask)


def sageattn_qk_int8_pv_fp16_cuda(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, tensor_layout: str = "HND",
                                  is_causal: bool = False, qk_quant_gran: str = "per_thread", sm_scale: Optional[float] = None,
                                  pv_accum_dtype: str = "fp32", smooth_k: bool = True, smooth_v: bool = False,
                                  return_lse: bool = False, **kwargs: Any):
    return _variant(q, k, v, tensor_layout, is_causal, sm_scale, return_lse)


def sageattn_qk_int8_pv_fp8_cuda(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, tensor_layout: str = "HND",
                                 is_causal: bool = False, qk_quant_gran: str = "per_thread", sm_scale: Optional[float] = None,
                                 pv_accum_dtype: str = "fp32+fp16", smooth_k: bool = True, smooth_v: bool = False,
                                 return_lse: bool = False, **kwargs: Any):
    return _variant(q, k, v, tensor_layout, is_causal, sm_scale, return_lse)


def sageattn_qk_int8_pv_fp8_cuda_sm90(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, tensor_layout: str = "HND",
                                      is_causal: bool = False, qk_quant_gran: str = "per_thread", sm_scale: Optional[float] = None,
                                      pv_accum_dtype: str = "fp32+fp32", smooth_k: bool = True, return_lse: bool = False,
                                      **kwargs: Any):
    return _variant(q, k, v, tensor_layout, is_causal, sm_scale, return_lse)


def sageattn_qk_int8_pv_gfx12_native(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, tensor_layout: str = "HND",
                                     is_causal: bool = False, qk_quant_gran: str = "per_warp", sm_scale: Optional[float] = None,
                                     pv_accum_dtype: Optional[str] = None, value_dtype: str = "fp8", smooth_k: bool = True,
                                     smooth_v: bool = False, return_lse: bool = False, **kwargs: Any):
    return _variant(q, k, v, tensor_layout, is_causal, sm_scale, return_lse)
