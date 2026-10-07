"""Correctness of rdna4_sage against fp32 SDPA. Run: python tests/test_attention.py (or pytest)."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import torch
import torch.nn.functional as F

import rdna4_sage

CASES = [  # name, B, Sq, Skv, H, Hkv, D, layout, dtype[, causal]
    ("SDXL self", 2, 4096, 4096, 10, 10, 64, "NHD", torch.float16),
    ("SD3.5 joint", 1, 4250, 4250, 24, 24, 64, "NHD", torch.float16),
    ("Flux joint", 1, 4352, 4352, 24, 24, 128, "HND", torch.bfloat16),
    ("Flux.2 joint", 1, 4608, 4608, 48, 48, 128, "HND", torch.bfloat16),
    ("Qwen-Image 1328", 1, 6889, 6889, 24, 24, 128, "HND", torch.bfloat16),
    ("Neta Lumina", 1, 4352, 4352, 24, 24, 96, "HND", torch.bfloat16),
    ("Lumina GQA 24/8", 1, 4096, 4096, 24, 8, 96, "HND", torch.bfloat16),
    ("Wan 1.3B 480p", 1, 32760, 32760, 12, 12, 128, "NHD", torch.float16),
    ("Wan cross 512", 1, 32760, 512, 12, 12, 128, "NHD", torch.float16),
    ("LTXV 2B", 1, 4992, 4992, 32, 32, 64, "NHD", torch.bfloat16),
    ("odd lengths", 3, 777, 1031, 6, 6, 128, "HND", torch.float16),
    ("AuraFlow joint d256", 1, 4360, 4360, 12, 12, 256, "HND", torch.float16),
    ("d256 odd lengths", 2, 777, 1031, 4, 4, 256, "NHD", torch.bfloat16),
    ("causal d128", 1, 4096, 4096, 24, 24, 128, "HND", torch.bfloat16, True),
    ("causal d64 odd", 2, 1000, 1000, 8, 8, 64, "NHD", torch.float16, True),
    ("causal d96 GQA 24/8", 1, 2048, 2048, 24, 8, 96, "HND", torch.bfloat16, True),
    ("causal d256", 1, 3000, 3000, 8, 8, 256, "NHD", torch.float16, True),
    ("causal d128 long", 1, 16384, 16384, 12, 12, 128, "HND", torch.bfloat16, True),
]


def run_case(name, B, Sq, Skv, H, Hkv, D, layout, dt, causal=False):
    g = torch.Generator(device="cuda").manual_seed(0)
    shape = (lambda s, h: (B, h, s, D)) if layout == "HND" else (lambda s, h: (B, s, h, D))
    q = torch.randn(shape(Sq, H), device="cuda", dtype=dt, generator=g)
    k = torch.randn(shape(Skv, Hkv), device="cuda", dtype=dt, generator=g)
    v = torch.randn(shape(Skv, Hkv), device="cuda", dtype=dt, generator=g)
    out = rdna4_sage.attention(q, k, v, tensor_layout=layout, is_causal=causal)  # kernel only, no SDPA routing
    assert out.dtype == dt and out.shape == q.shape, (out.dtype, out.shape)
    th = (lambda t: t) if layout == "HND" else (lambda t: t.transpose(1, 2))
    rep = H // Hkv
    ref = F.scaled_dot_product_attention(th(q).float(), th(k).float().repeat_interleave(rep, 1), th(v).float().repeat_interleave(rep, 1),
                                         is_causal=causal)  # q_len == kv_len for causal cases: top-left == bottom-right
    o = th(out).float()
    cos = F.cosine_similarity(o.flatten(), ref.flatten(), dim=0).item()
    rel = ((o - ref).norm() / ref.norm()).item()
    return cos, rel


def test_cases():
    for case in CASES:
        cos, rel = run_case(*case)
        assert cos > 0.998, (case[0], cos)


def test_sm_scale_and_routing():
    q, k, v = (torch.randn(1, 8, 2048, 128, device="cuda", dtype=torch.bfloat16) for _ in range(3))
    ref = F.scaled_dot_product_attention(q.float(), k.float(), v.float(), scale=0.05)
    out = rdna4_sage.attention(q, k, v, "HND", sm_scale=0.05).float()
    assert F.cosine_similarity(out.flatten(), ref.flatten(), dim=0).item() > 0.998
    # small problems and masks go to SDPA through the public drop-in
    m = torch.zeros(1, 1, 2048, 2048, device="cuda", dtype=torch.bfloat16)
    assert torch.allclose(rdna4_sage.sageattn(q, k, v, attn_mask=m).float(), F.scaled_dot_product_attention(q, k, v, attn_mask=m).float())
    # causal with q_len != kv_len -> SDPA with a bottom-right mask (SageAttention / FlashAttention semantics)
    qs = q[:, :, :1500]
    keep = torch.ones(1500, 2048, dtype=torch.bool, device="cuda").tril(2048 - 1500)
    ref = F.scaled_dot_product_attention(qs, k, v, attn_mask=keep)
    torch.testing.assert_close(rdna4_sage.sageattn(qs, k, v, is_causal=True), ref)


def test_compile():
    q, k, v = (torch.randn(1, 24, 4352, 128, device="cuda", dtype=torch.bfloat16) for _ in range(3))
    f = torch.compile(lambda a, b, c: rdna4_sage.attention(a, b, c, "HND") * 2, fullgraph=True)
    torch.testing.assert_close(f(q, k, v), rdna4_sage.attention(q, k, v, "HND") * 2)


if __name__ == "__main__":
    print(torch.__version__, torch.cuda.get_device_name(0), rdna4_sage.device_arch(torch.device("cuda")), "os:", os.name)
    for case in CASES:
        cos, rel = run_case(*case)
        print(f"{case[0]:20s} B={case[1]} Sq={case[2]:5d} Skv={case[3]:5d} H={case[4]}/{case[5]} D={case[6]:3d} {case[7]} "
              f"{str(case[8])[6:]:8s}{' causal' if case[9:] and case[9] else '       '} cos {cos:.5f} rel {rel:.4f} "
              f"{'OK' if cos > 0.998 else 'FAIL'}", flush=True)
    test_sm_scale_and_routing(); print("sm_scale + SDPA routing (mask, causal q_len != kv_len): OK")
    test_compile(); print("torch.compile(fullgraph=True): OK")
