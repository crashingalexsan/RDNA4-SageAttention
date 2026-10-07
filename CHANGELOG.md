# Changelog

Versions follow `2.2.0+rdna4.X.Y.Z`: `2.2.0` is the SageAttention API version this package implements, and
`X.Y.Z` is the `rdna4_sage` version. Wheels are on the
[Releases page](https://github.com/crashingalexsan/RDNA4-SageAttention/releases).

## Unreleased

### Changed
- README: recommend PyTorch builds for ROCm 10.1; note that `torch 2.12.0+rocm10.0.0` fails on RDNA4 for causal and
  masked SDPA.

## 2.2.0+rdna4.0.2.1 — 2026-10-07

### Fixed
- `RecursionError: maximum recursion depth exceeded` in apps that replace
  `torch.nn.functional.scaled_dot_product_attention` with `sageattn`, such as SD.Next (reported in its prompt parser
  / text encoder). The SDPA fallback for small and masked calls now calls torch's attention binding directly instead
  of looping back into `sageattn`.

### Changed
- README: release download, SD.Next usage.

## 2.2.0+rdna4.0.2.0 — 2026-10-07

First public release.

### Added
- SageAttention 2.2 API (`sageattn`, `sageattn_varlen`, `sageattn_qk_int8_pv_*`) under the `sageattention` name, as a
  drop-in replacement for an existing SageAttention install.
- Ahead-of-time int8 Q·Kᵀ / fp8 P·V kernels for gfx1200 and gfx1201: head dims 64, 96, 128 and 256, bf16 and fp16,
  causal, grouped-query attention. One pure-Python wheel for Windows and Linux.
- Automatic fallback to PyTorch SDPA for masks, unsupported head dims or GPUs, and small problems
  (`RDNA4_SAGE_MIN_KV`, `RDNA4_SAGE_MIN_WORK`).
- `torch.ops.rdna4_sage.attention` (`torch.compile(fullgraph=True)` compatible) and the `rdna4_sage` ComfyUI
  attention backend.

### Changed
- Import package renamed from `rdna_sage` to `rdna4_sage` (also the torch op namespace, the ComfyUI backend name and
  the `RDNA4_SAGE_*` environment variables).
