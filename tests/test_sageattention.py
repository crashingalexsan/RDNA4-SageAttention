"""The shim behaves like SageAttention 2.2: same functions, layouts, causal, return_lse and varlen."""
import inspect
import os
import sys

here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(here, "..", "src"))
import torch
import torch.nn.functional as F

import sageattention
from sageattention import sageattn, sageattn_varlen

COS = 0.998


def cos(a, b):
    return F.cosine_similarity(a.float().flatten(), b.float().flatten(), dim=0).item()


def ref(q, k, v, layout, causal=False, scale=None):
    th = (lambda t: t) if layout == "HND" else (lambda t: t.transpose(1, 2))
    rep = th(q).shape[1] // th(k).shape[1]
    o = F.scaled_dot_product_attention(th(q).float(), th(k).float().repeat_interleave(rep, 1),
                                       th(v).float().repeat_interleave(rep, 1), is_causal=causal, scale=scale)
    return th(o)


def test_api_surface():
    names = ["sageattn", "sageattn_varlen", "sageattn_qk_int8_pv_fp16_triton", "sageattn_qk_int8_pv_fp16_cuda",
             "sageattn_qk_int8_pv_fp8_cuda", "sageattn_qk_int8_pv_fp8_cuda_sm90", "sageattn_qk_int8_pv_gfx12_native"]
    for n in names:
        assert callable(getattr(sageattention, n)), n
    # ComfyUI decides mask support from the signature; SageAttention 2.2's sageattn has no attn_mask parameter
    assert "attn_mask" not in inspect.signature(sageattn).parameters


def test_layouts_dtypes_causal():
    for layout, dt, causal, D in [("HND", torch.bfloat16, False, 128), ("NHD", torch.float16, False, 64),
                                  ("HND", torch.float16, True, 128), ("NHD", torch.bfloat16, True, 96)]:
        shp = (1, 24, 4352, D) if layout == "HND" else (1, 4352, 24, D)
        q, k, v = (torch.randn(shp, device="cuda", dtype=dt) for _ in range(3))
        o = sageattn(q, k, v, tensor_layout=layout, is_causal=causal)
        assert o.dtype == dt and o.shape == q.shape
        c = cos(o, ref(q, k, v, layout, causal))
        assert c > COS, (layout, dt, causal, c)


def test_variants_and_kwargs():
    q, k, v = (torch.randn(2, 16, 4096, 128, device="cuda", dtype=torch.bfloat16) for _ in range(3))
    r = ref(q, k, v, "HND")
    for fn in (sageattention.sageattn_qk_int8_pv_fp16_triton, sageattention.sageattn_qk_int8_pv_fp16_cuda,
               sageattention.sageattn_qk_int8_pv_fp8_cuda, sageattention.sageattn_qk_int8_pv_fp8_cuda_sm90,
               sageattention.sageattn_qk_int8_pv_gfx12_native):
        assert cos(fn(q, k, v, tensor_layout="HND", smooth_k=True, pv_accum_dtype="fp32"), r) > COS, fn.__name__
    # ComfyUI's call
    assert cos(sageattn(q, k, v, is_causal=False, tensor_layout="HND", sm_scale=None, smooth_k=False), r) > COS
    # explicit scale
    assert cos(sageattn(q, k, v, sm_scale=0.05), ref(q, k, v, "HND", scale=0.05)) > COS


def test_return_lse():
    q, k, v = (torch.randn(1, 4, 2048, 64, device="cuda", dtype=torch.float16) for _ in range(3))
    for causal in (False, True):
        o, lse = sageattn(q, k, v, is_causal=causal, return_lse=True)
        sc = q.float() @ k.float().transpose(-1, -2) * 64 ** -0.5
        if causal:
            sc.masked_fill_(torch.ones(2048, 2048, dtype=torch.bool, device="cuda").triu(1), float("-inf"))
        assert lse.shape == (1, 4, 2048) and lse.dtype == torch.float32
        torch.testing.assert_close(lse, torch.logsumexp(sc, -1), atol=2e-3, rtol=1e-4)
        assert cos(o, ref(q, k, v, "HND", causal)) > COS


def test_varlen():
    H, D, lens_q, lens_k = 8, 128, [3000, 1200, 2048], [3000, 512, 4096]
    cq = torch.tensor([0] + list(torch.tensor(lens_q).cumsum(0)), dtype=torch.int32, device="cuda")
    ck = torch.tensor([0] + list(torch.tensor(lens_k).cumsum(0)), dtype=torch.int32, device="cuda")
    q = torch.randn(sum(lens_q), H, D, device="cuda", dtype=torch.bfloat16)
    k, v = (torch.randn(sum(lens_k), H, D, device="cuda", dtype=torch.bfloat16) for _ in range(2))
    out = sageattn_varlen(q, k, v, cq, ck, max(lens_q), max(lens_k))
    assert out.shape == q.shape
    for i in range(3):
        s = slice(int(cq[i]), int(cq[i + 1])); t = slice(int(ck[i]), int(ck[i + 1]))
        assert cos(out[s], ref(q[None, s], k[None, t], v[None, t], "NHD")[0]) > COS, i


if __name__ == "__main__":
    print("sageattention", sageattention.__version__, "from", os.path.dirname(sageattention.__file__))
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print(f"{name}: OK", flush=True)
