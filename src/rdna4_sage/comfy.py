"""ComfyUI attention backend.

    import rdna4_sage.comfy
    rdna4_sage.comfy.register()   # adds "rdna4_sage" to comfy's attention registry

Then select it per model through transformer_options["optimized_attention_override"] or
preferred_attention="rdna4_sage", like any registered backend.
"""
from comfy.ldm.modules.attention import attention_pytorch, register_attention_function, wrap_attn

from . import sageattn


@wrap_attn
def attention_rdna4_sage(q, k, v, heads, mask=None, attn_precision=None, skip_reshape=False, skip_output_reshape=False, **kwargs):
    if kwargs.get("low_precision_attention", True) is False or mask is not None:
        return attention_pytorch(q, k, v, heads, mask=mask, attn_precision=attn_precision, skip_reshape=skip_reshape,
                                 skip_output_reshape=skip_output_reshape, **kwargs)
    if skip_reshape:
        b, _, _, dim_head = q.shape
        layout = "HND"
    else:
        b, _, dim = q.shape
        dim_head = dim // heads
        # (B, L, H*D) -> (B, L, H, D), a view when possible; k/v may carry fewer (GQA) heads, which the kernel handles
        q = q.unflatten(-1, (heads, dim_head))
        k = k.unflatten(-1, (-1, dim_head))
        v = v.unflatten(-1, (-1, dim_head))
        layout = "NHD"

    out = sageattn(q, k, v, tensor_layout=layout, sm_scale=kwargs.get("scale", None))

    if layout == "HND":
        return out if skip_output_reshape else out.transpose(1, 2).reshape(b, -1, heads * dim_head)
    return out.transpose(1, 2) if skip_output_reshape else out.reshape(b, -1, heads * dim_head)


def register(name="rdna4_sage"):
    register_attention_function(name, attention_rdna4_sage)
    return attention_rdna4_sage
