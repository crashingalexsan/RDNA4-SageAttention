#!/bin/bash
# Run the package tests with a Linux ROCm torch (e.g. the WSL ComfyUI venv). No FlyDSL / Triton build needed.
PY=${PY:-$HOME/ComfyUI/.venv/bin/python}
cd "$(dirname "$0")/.." && $PY tests/test_attention.py 2>&1 | grep -vE "Warning|USDT"
