#!/bin/bash
# Build all code objects in WSL/Linux. Needs the FlyDSL venv (~/flypr) and checkout (~/flydsl-pr).
set -e
source ~/flypr/bin/activate
export ROCM_PATH=${ROCM_PATH:-/opt/rocm}
cd "$(dirname "$0")"
python build_kernels.py --flydsl ~/flydsl-pr "$@" 2>&1 | grep -vE "Warning|USDT"
