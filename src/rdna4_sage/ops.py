"""torch.library registration: torch.ops.rdna4_sage.attention, traceable by torch.compile."""
import torch

from . import _attention


@torch.library.custom_op("rdna4_sage::attention", mutates_args=())
def attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, layout: str, sm_scale: float, is_causal: bool) -> torch.Tensor:
    """Returns (B, Sq, H, D) contiguous. sm_scale <= 0 selects 1/sqrt(head_dim)."""
    return _attention.attention(q, k, v, layout, sm_scale if sm_scale > 0 else None, is_causal)


@attention.register_fake
def _(q, k, v, layout, sm_scale, is_causal):
    B = q.shape[0]
    H, Sq = (q.shape[1], q.shape[2]) if layout == "HND" else (q.shape[2], q.shape[1])
    return q.new_empty((B, Sq, H, q.shape[3]))
