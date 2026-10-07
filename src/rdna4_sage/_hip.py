"""Minimal HIP driver API access through the HIP runtime torch already loaded."""
import ctypes
import os
import struct

import torch

_lib = None


def _loaded_hip_path():
    # Reuse the exact runtime instance torch uses, so modules and streams share one device context.
    if os.name == "nt":
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.GetModuleHandleW.restype = ctypes.c_void_p
        k32.GetModuleFileNameW.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p, ctypes.c_uint32]
        for name in ("amdhip64_7.dll", "amdhip64_6.dll", "amdhip64.dll"):
            h = k32.GetModuleHandleW(name)
            if h:
                buf = ctypes.create_unicode_buffer(32768)
                k32.GetModuleFileNameW(h, buf, len(buf))
                return buf.value
        return None
    with open("/proc/self/maps") as f:
        for line in f:
            if "libamdhip64.so" in line:
                return line.split(None, 5)[-1].strip()
    return None


def lib():
    global _lib
    if _lib is None:
        torch.cuda.init()
        path = _loaded_hip_path()
        if path is None:
            raise RuntimeError("rdna4_sage: torch has not loaded a HIP runtime (ROCm build of torch required)")
        hip = ctypes.CDLL(path)
        hip.hipModuleLoadData.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p]
        hip.hipModuleGetFunction.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_char_p]
        hip.hipModuleLaunchKernel.argtypes = [ctypes.c_void_p] + [ctypes.c_uint] * 7 + [ctypes.c_void_p] * 3
        hip.hipGetErrorString.restype = ctypes.c_char_p
        _lib = hip
    return _lib


def _check(err, what):
    if err != 0:
        raise RuntimeError(f"rdna4_sage: {what} failed: {lib().hipGetErrorString(err).decode()} ({err})")


class Kernel:
    """One kernel of a code object, loaded on one device."""

    def __init__(self, code: bytes, name: str, args_fmt: str, threads: int, shared: int, device: int):
        hip = lib()
        self._code = ctypes.create_string_buffer(code, len(code))  # must outlive the module
        mod, fn = ctypes.c_void_p(), ctypes.c_void_p()
        with torch.cuda.device(device):
            _check(hip.hipModuleLoadData(ctypes.byref(mod), self._code), "hipModuleLoadData")
            _check(hip.hipModuleGetFunction(ctypes.byref(fn), mod, name.encode()), f"hipModuleGetFunction({name})")
        self._fn, self._threads, self._shared = fn, threads, shared
        self._args = struct.Struct(args_fmt)
        self._size = ctypes.c_size_t(self._args.size)
        self._buf = ctypes.create_string_buffer(self._args.size)
        # HIP_LAUNCH_PARAM_BUFFER_POINTER, buf, HIP_LAUNCH_PARAM_BUFFER_SIZE, &size, HIP_LAUNCH_PARAM_END
        self._extra = (ctypes.c_void_p * 5)(1, ctypes.addressof(self._buf), 2, ctypes.addressof(self._size), 3)

    def launch(self, grid_x: int, grid_y: int, stream: int, *args):
        # The packed argument block is copied by hipModuleLaunchKernel, so the buffer can be reused right away.
        self._args.pack_into(self._buf, 0, *args)
        _check(lib().hipModuleLaunchKernel(self._fn, grid_x, grid_y, 1, self._threads, 1, 1, self._shared, stream, None,
                                           self._extra), "hipModuleLaunchKernel")
