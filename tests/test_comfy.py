"""ComfyUI backend vs comfy's own attention_pytorch. Run from anywhere with COMFYUI_PATH pointing at a ComfyUI checkout."""
import os
import sys

here = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(here, "..", "src"), os.environ.get("COMFYUI_PATH", r"D:\ComfyUI")]
sys.argv = sys.argv[:1]  # comfy parses CLI args on import
import torch
import torch.nn.functional as F

from comfy.ldm.modules.attention import attention_pytorch, get_attention_function

import rdna4_sage.comfy

fn = rdna4_sage.comfy.register()
assert get_attention_function("rdna4_sage", None) is fn


def check(name, q, k, v, heads, **kw):
    ref = attention_pytorch(q, k, v, heads, **kw).float()
    out = fn(q, k, v, heads, **kw)
    assert out.shape == ref.shape and out.dtype == q.dtype, (name, out.shape, ref.shape, out.dtype)
    cos = F.cosine_similarity(out.float().flatten(), ref.flatten(), dim=0).item()
    print(f"{name:42s} out {tuple(out.shape)} cos {cos:.5f} {'OK' if cos > 0.998 else 'FAIL'}")
    assert cos > 0.998


B, H, D, L = 1, 24, 128, 4352
x = lambda *s, dt=torch.bfloat16: torch.randn(*s, device="cuda", dtype=dt)
check("NHD (B, L, H*D) Wan/SD3 style", x(B, L, H * D), x(B, L, H * D), x(B, L, H * D), H)
q, k, v = x(B, H, L, D), x(B, H, L, D), x(B, H, L, D)
check("HND skip_reshape (Flux style)", q, k, v, H, skip_reshape=True)
check("HND skip_reshape + skip_output_reshape", q, k, v, H, skip_reshape=True, skip_output_reshape=True)
qkv = x(B, L, 3 * H * D)
check("NHD non-contiguous q/k/v from one projection", *qkv.chunk(3, dim=-1), H)
check("fp16 SDXL-like, custom scale", x(2, 4096, 640, dt=torch.float16), x(2, 4096, 640, dt=torch.float16),
      x(2, 4096, 640, dt=torch.float16), 10, scale=0.1)
m = torch.zeros(B, 1, L, L, device="cuda", dtype=torch.bfloat16)
check("masked call -> pytorch fallback", q, k, v, H, skip_reshape=True, mask=m)
print("comfy backend: OK")
