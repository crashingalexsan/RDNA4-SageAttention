"""SageAttention 2.2 drop-in for AMD RDNA4 (gfx1200 / gfx1201), backed by rdna4_sage.

Same import paths and functions as SageAttention 2.2, so it substitutes the install:
    from sageattention import sageattn
"""
from .core import sageattn, sageattn_varlen
from .core import sageattn_qk_int8_pv_fp16_triton
from .core import sageattn_qk_int8_pv_fp16_cuda
from .core import sageattn_qk_int8_pv_fp8_cuda
from .core import sageattn_qk_int8_pv_fp8_cuda_sm90
from .core import sageattn_qk_int8_pv_gfx12_native

__version__ = "2.2.0+rdna4.0.2.0"
