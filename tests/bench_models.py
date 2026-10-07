"""Per-call attention time for common ComfyUI model shapes: torch SDPA vs rdna4_sage (kernel forced) vs routed sageattn.

Times include quantization and launch overhead with a sync per call (pessimistic for rdna4_sage on small shapes).
"""
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import torch
import torch.nn.functional as F

import rdna4_sage

SHAPES = [  # name, B, Sq, Skv, H, D, layout, dtype[, causal]   (B=2 where ComfyUI batches cond/uncond)
    ("SD1.5 512 (d40->sdpa)", 2, 4096, 4096, 8, 40, "NHD", torch.float16),
    ("SDXL 1024 lvl1 self", 2, 4096, 4096, 10, 64, "NHD", torch.float16),
    ("SDXL 1024 lvl2 self", 2, 1024, 1024, 20, 64, "NHD", torch.float16),
    ("SDXL 1024 lvl1 cross77", 2, 4096, 77, 10, 64, "NHD", torch.float16),
    ("SD3.5L joint", 2, 4250, 4250, 38, 64, "NHD", torch.float16),
    ("Flux 1024 joint", 1, 4352, 4352, 24, 128, "HND", torch.bfloat16),
    ("Flux 768 joint", 1, 2560, 2560, 24, 128, "HND", torch.bfloat16),
    ("Flux 512 joint", 1, 1280, 1280, 24, 128, "HND", torch.bfloat16),
    ("Flux.2 1024 joint", 1, 4608, 4608, 48, 128, "HND", torch.bfloat16),
    ("Qwen-Image 1328 joint", 1, 6989, 6989, 24, 128, "HND", torch.bfloat16),
    ("HiDream 1024 joint", 1, 4480, 4480, 20, 128, "NHD", torch.bfloat16),
    ("Neta Lumina 1024", 1, 4352, 4352, 24, 96, "HND", torch.bfloat16),
    ("Anima 1024 self", 2, 4096, 4096, 16, 128, "HND", torch.bfloat16),
    ("Anima 1024 cross512", 2, 4096, 512, 16, 128, "HND", torch.bfloat16),
    ("Wan1.3B 480p self", 1, 32760, 32760, 12, 128, "NHD", torch.float16),
    ("Wan1.3B 480p cross512", 1, 32760, 512, 12, 128, "NHD", torch.float16),
    ("Wan14B 480p self", 1, 32760, 32760, 40, 128, "NHD", torch.float16),
    ("Wan2.2-5B 704p self", 1, 27280, 27280, 24, 128, "NHD", torch.float16),
    ("HunyuanVideo 544p", 1, 67320, 67320, 24, 128, "HND", torch.bfloat16),
    ("LTXV-2B 768x512x97", 1, 4992, 4992, 32, 64, "NHD", torch.bfloat16),
    ("LTXV-2B cross128", 1, 4992, 128, 32, 64, "NHD", torch.bfloat16),
    ("CogVideoX-5B joint", 1, 17776, 17776, 48, 64, "NHD", torch.bfloat16),
    ("AuraFlow 1024 joint d256", 2, 4360, 4360, 12, 256, "HND", torch.float16),
    ("causal d128 4k", 1, 4096, 4096, 32, 128, "HND", torch.bfloat16, True),
    ("causal d128 16k", 1, 16384, 16384, 32, 128, "HND", torch.bfloat16, True),
    ("causal d64 8k", 1, 8192, 8192, 32, 64, "HND", torch.bfloat16, True),
]


def t(fn, n=10):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(n):
        s = time.perf_counter(); fn(); torch.cuda.synchronize(); ts.append(time.perf_counter() - s)
    ts.sort()
    return ts[len(ts) // 2] * 1000


def main():
    only = sys.argv[1:]
    print(torch.__version__, torch.cuda.get_device_name(0), "os:", os.name)
    print(f"{'shape':26s} {'sdpa':>9s} {'rdna4_sage':>10s} {'speedup':>8s} {'routed':>9s}  route")
    for name, B, Sq, Skv, H, D, layout, dt, *causal in SHAPES:
        causal = bool(causal and causal[0])
        if only and not any(o.lower() in name.lower() for o in only):
            continue
        mk = (lambda s: (B, H, s, D)) if layout == "HND" else (lambda s: (B, s, H, D))
        q = torch.randn(mk(Sq), device="cuda", dtype=dt)
        k, v = torch.randn(mk(Skv), device="cuda", dtype=dt), torch.randn(mk(Skv), device="cuda", dtype=dt)
        th = (lambda x: x) if layout == "HND" else (lambda x: x.transpose(1, 2))
        a = t(lambda: F.scaled_dot_product_attention(th(q), th(k), th(v), is_causal=causal))
        try:
            b = t(lambda: rdna4_sage.attention(q, k, v, layout, is_causal=causal))
            sp = f"{a / b:7.2f}x"
        except rdna4_sage.Unsupported as e:
            b, sp = float("nan"), "  n/a"
        r = t(lambda: rdna4_sage.sageattn(q, k, v, tensor_layout=layout, is_causal=causal))
        route = "kernel" if rdna4_sage._worth_it(q, k, layout, causal) and not (b != b) else "sdpa"
        print(f"{name:26s} {a:8.3f}  {b:9.3f}  {sp}  {r:8.3f}  {route}", flush=True)
        del q, k, v
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
