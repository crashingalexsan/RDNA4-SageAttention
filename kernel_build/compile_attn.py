"""Compile one attention variant with FlyDSL for a target arch (run in a fresh process).

usage: compile_attn.py <arch> <head_dim> <block_m> <row_subtiles> <out_dtype> <causal 0|1> <flydsl_checkout> <dump_dir>
The code object lands in <dump_dir>/flash_attn_func_sage_kernel_0/20_gpu_module_to_binary.mlir.
"""
import os
import sys

arch, head_dim, block_m, row_subtiles, out_dtype, causal, flydsl, dump = (
    sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), sys.argv[5], sys.argv[6] == "1", sys.argv[7], sys.argv[8])
os.environ.update({"ARCH": arch, "FLYDSL_GPU_ARCH": arch, "COMPILE_ONLY": "1", "FLYDSL_RUNTIME_ENABLE_CACHE": "0",
                   "FLYDSL_DUMP_IR": "1", "FLYDSL_DUMP_DIR": dump})
sys.path.insert(0, flydsl)

import torch  # noqa: E402

from kernels.attention.flash_attn_sage_gfx120x import build_flash_attn_func_sage_module  # noqa: E402

kern = build_flash_attn_func_sage_module(num_heads=1, head_dim=head_dim, causal=causal, block_m=block_m, block_n=32,
                                         waves_per_eu=2, num_kv_heads=1, row_subtiles=row_subtiles, out_dtype=out_dtype)
# Tiny placeholder tensors: only their dtypes/addresses reach the tracer; in COMPILE_ONLY mode nothing runs.
dev = "cuda" if torch.cuda.is_available() else "cpu"
S = 64
q8 = torch.zeros(1, S, 1, head_dim, dtype=torch.int8, device=dev)
v8 = torch.zeros(1, S, 1, head_dim, dtype=torch.float8_e4m3fn, device=dev)
o = torch.zeros(1, S, 1, head_dim, dtype=torch.bfloat16 if out_dtype == "bf16" else torch.float16, device=dev)
f = torch.ones(S * head_dim, dtype=torch.float32, device=dev)
try:
    kern(q8, q8, v8, o, 1, S, S, S, f, f, f, f, f, f, 1, 1, stream=torch.cuda.current_stream() if dev == "cuda" else None)
except Exception as e:  # launching a foreign-arch object fails after the dump was written
    print("note: launch skipped/failed after compile:", type(e).__name__, str(e).splitlines()[0][:120] if str(e) else "")
