#!/bin/bash
# Install the built wheel into a scratch dir and smoke-test it with a Linux ROCm torch.
PY=${PY:-$HOME/ComfyUI/.venv/bin/python}
T=$(mktemp -d)
$PY -m pip install --no-deps -q --target "$T" "$(dirname "$0")"/../dist/sageattention-*rdna*.whl && cd "$T" && PYTHONPATH="$T" $PY "$(cd "$(dirname "$0")" && pwd)"/smoke_installed.py 2>&1 | grep -vE "Warning|USDT"
rm -rf "$T"
