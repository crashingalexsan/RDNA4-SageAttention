"""Build every rdna4_sage code object for gfx1200 / gfx1201 (Linux or WSL, GPU not required).

Requires the FlyDSL checkout with the gfx120x kernels (PR #1223 base + build/make_sage*.py), Triton 3.8
for ROCm and ROCm's llvm-readelf. Writes src/rdna4_sage/kernels/<arch>/*.hsaco and kernels/manifest.json.

usage: python build_kernels.py --flydsl ~/flydsl-pr [--archs gfx1200,gfx1201]
"""
import argparse
import json
import os
import re
import shutil
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "src" / "rdna4_sage" / "kernels"

# head_dim -> (block_m rows per workgroup, row sub-tiles of 16 rows per wave). Tuned on gfx1201; the kernel builder
# rejects combinations whose K loader would not tile the KV block (head_dim 96 needs 256 threads).
# 64: SDXL / SD3 / LTXV-2B / CogVideoX, 96: Lumina2 / Neta Lumina, 128: Flux / Qwen / Wan / Hunyuan / Cosmos ...,
# 256: AuraFlow (16 rows per wave: two sub-tiles of 256-wide accumulators would not fit in registers).
ATTN_CONFIG = {64: (256, 2), 96: (256, 2), 128: (256, 2), 256: (128, 1)}
HADAMARD_MAX = 128  # rotation block cap: a larger +-1 matrix would not fit in LDS as a tl.dot operand
OUT_DTYPES = ("bf16", "f16")
IN_DTYPES = {"bf16": "bf16", "f16": "fp16"}  # package name -> Triton type name
KV_BLOCK = 32
QUANT_ROWS = 32
QUANT_WARPS = 4

# Kernel argument blocks, as the runtime packs them (struct format, little endian).
ATTN_ARGS = "<4Q3i4x6Q2i"     # Q K V O | seq_len seq_len_kv seq_len_kv_valid | QS KS VAmax Bias LSE Sink | heads kv_heads
QUANT_Q_ARGS = "<4Q5i4x2Q"    # X Y S Hm | seq sxb sxs sxh H | Triton global/profile scratch
QUANT_KV_ARGS = "<7Q9i4x2Q"   # K V K8 KS V8 VS Hm | seq seq_pad skb sks skh svb svs svh H | scratch


def rotates(head_dim):
    return head_dim & (head_dim - 1) == 0


def readelf_notes(path):
    out = subprocess.run([os.environ.get("LLVM_READELF", "/opt/rocm/llvm/bin/llvm-readelf"), "--notes", str(path)],
                         check=True, capture_output=True, text=True).stdout
    kernarg = int(re.search(r"\.kernarg_segment_size:\s+(\d+)", out).group(1))
    target = re.search(r"amdhsa\.target:\s+(\S+)", out).group(1)
    name = re.search(r"\.name:\s+(\S+)", out).group(1)
    return kernarg, target, name


def check(path, fmt, arch):
    kernarg, target, name = readelf_notes(path)
    want = struct.calcsize(fmt)
    if kernarg != want:
        raise SystemExit(f"{path.name}: kernarg {kernarg} bytes, runtime packs {want} ({fmt})")
    if not target.endswith(arch):
        raise SystemExit(f"{path.name}: built for {target}, expected {arch}")
    return name


def generate_kernel_source(flydsl):
    for step in ("make_sage.py", "make_sage_r2.py", "make_sage_pkg.py"):
        subprocess.run([sys.executable, str(HERE / step), str(flydsl)], check=True, stdout=subprocess.DEVNULL)


def build_attention(flydsl, arch, head_dim, out_dtype, causal, dst):
    block_m, rs = ATTN_CONFIG[head_dim]
    with tempfile.TemporaryDirectory() as dump:
        subprocess.run([sys.executable, str(HERE / "compile_attn.py"), arch, str(head_dim), str(block_m), str(rs),
                        out_dtype, "1" if causal else "0", str(flydsl), dump], check=True, stdout=subprocess.DEVNULL)
        mlir = next(Path(dump).glob("*/20_gpu_module_to_binary.mlir"))
        subprocess.run([sys.executable, str(HERE / "extract_hsaco.py"), str(mlir), str(dst)], check=True, stdout=subprocess.DEVNULL)
    name = check(dst, ATTN_ARGS, arch)
    return {"file": dst.name, "name": name, "args": ATTN_ARGS, "block_m": block_m, "threads": block_m // (16 * rs) * 32,
            "shared": 0}


def build_quant(arch, head_dim, in_dtype, which, dst):
    import triton
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource

    sys.path.insert(0, str(HERE))
    import sage_quant

    t = IN_DTYPES[in_dtype]
    if which == "q":
        fn, fmt = sage_quant._quant_q_kernel, QUANT_Q_ARGS
        sig = {"X": f"*{t}", "Y": "*i8", "S": "*fp32", "Hm": f"*{t}", "seq": "i32", "sxb": "i32", "sxs": "i32", "sxh": "i32",
               "H": "i32"}
        aligned = ("X", "Y", "S", "Hm", "sxb", "sxs", "sxh")
        bs = QUANT_ROWS
    else:
        fn, fmt = sage_quant._quant_kv_kernel, QUANT_KV_ARGS
        sig = {"K": f"*{t}", "V": f"*{t}", "K8": "*i8", "KS": "*fp32", "V8": "*fp8e4nv", "VS": "*fp32", "Hm": f"*{t}",
               "seq": "i32", "seq_pad": "i32", "skb": "i32", "sks": "i32", "skh": "i32", "svb": "i32", "svs": "i32", "svh": "i32",
               "H": "i32"}
        aligned = ("K", "V", "K8", "KS", "V8", "VS", "Hm", "seq_pad", "skb", "sks", "skh", "svb", "svs", "svh")
        bs = KV_BLOCK
    # Hadamard rotation needs a power-of-two head_dim; other dims run unrotated on a power-of-two tile
    dp = 1 << (head_dim - 1).bit_length()
    const = {"D": head_dim, "DP": dp, "BS": bs, "ROT": rotates(head_dim), "RB": min(dp, HADAMARD_MAX)}
    sig.update({k: "constexpr" for k in const})
    # pointers and strides are 16-divisible (checked by the runtime before launch), which lets Triton vectorize
    attrs = {(fn.arg_names.index(a),): [["tt.divisibility", 16]] for a in aligned}
    ck = triton.compile(ASTSource(fn, sig, const, attrs), target=GPUTarget("hip", arch, 32), options={"num_warps": QUANT_WARPS})
    md = ck.metadata
    assert md.global_scratch_size == 0 and md.profile_scratch_size == 0, "kernel needs Triton scratch buffers"
    dst.write_bytes(ck.asm["hsaco"])
    name = check(dst, fmt, arch)
    return {"file": dst.name, "name": name, "args": fmt, "rows": bs, "threads": md.num_warps * md.warp_size, "shared": md.shared,
            "rot": const["ROT"], "rot_block": const["RB"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--flydsl", required=True, type=Path)
    ap.add_argument("--archs", default="gfx1200,gfx1201")
    a = ap.parse_args()
    flydsl = a.flydsl.expanduser().resolve()
    generate_kernel_source(flydsl)
    commit = subprocess.run(["git", "-C", str(flydsl), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    manifest = {"flydsl_commit": commit, "kv_block": KV_BLOCK, "head_dims": sorted(ATTN_CONFIG), "archs": {}}
    for arch in a.archs.split(","):
        d = OUT / arch
        shutil.rmtree(d, ignore_errors=True)
        d.mkdir(parents=True)
        entry = {"attn": {}, "quant_q": {}, "quant_kv": {}}
        for hd in sorted(ATTN_CONFIG):
            for dt in OUT_DTYPES:
                key = f"d{hd}_{dt}"
                for causal in (False, True):
                    akey = key + ("_causal" if causal else "")
                    entry["attn"][akey] = build_attention(flydsl, arch, hd, dt, causal, d / f"attn_{akey}.hsaco")
                entry["quant_q"][key] = build_quant(arch, hd, dt, "q", d / f"quant_q_{key}.hsaco")
                entry["quant_kv"][key] = build_quant(arch, hd, dt, "kv", d / f"quant_kv_{key}.hsaco")
                print(f"{arch} {key}: ok", flush=True)
        manifest["archs"][arch] = entry
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print("wrote", OUT / "manifest.json")


if __name__ == "__main__":
    main()
