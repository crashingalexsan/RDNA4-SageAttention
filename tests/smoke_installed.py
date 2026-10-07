"""Smoke test of an installed rdna4_sage (no source tree on sys.path)."""
import os
import sys

import torch
import torch.nn.functional as F

import rdna4_sage
import sageattention
from sageattention import sageattn

for m in (rdna4_sage, sageattention):
    assert "src" not in os.path.dirname(m.__file__), m.__file__
print("sageattention", sageattention.__version__, "+ rdna4_sage", rdna4_sage.__version__, "from",
      os.path.dirname(os.path.dirname(rdna4_sage.__file__)), "| os:", os.name, "| available:", rdna4_sage.is_available())
for layout, dt, D in [("HND", torch.bfloat16, 128), ("NHD", torch.float16, 64), ("HND", torch.bfloat16, 96)]:
    shp = (1, 24, 4352, D) if layout == "HND" else (1, 4352, 24, D)
    q, k, v = (torch.randn(shp, device="cuda", dtype=dt) for _ in range(3))
    th = (lambda t: t) if layout == "HND" else (lambda t: t.transpose(1, 2))
    ref = F.scaled_dot_product_attention(th(q).float(), th(k).float(), th(v).float())
    out = th(sageattn(q, k, v, tensor_layout=layout)).float()
    cos = F.cosine_similarity(out.flatten(), ref.flatten(), dim=0).item()
    print(f"  {layout} {dt} D={D}: cos {cos:.5f}")
    assert cos > 0.998
print("installed wheel: OK")
