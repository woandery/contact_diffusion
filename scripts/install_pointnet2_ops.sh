#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
POINTNET_DIR="${PROJECT_ROOT}/third_party/pointnet2_ops"

export CC="${CC:-/usr/bin/g++}"
export CXX="${CXX:-/usr/bin/g++}"
export CUDAHOSTCXX="${CUDAHOSTCXX:-/usr/bin/g++}"
# RTX 4090 (8.9) and H100 (9.0). Override this variable when targeting a
# different GPU generation.
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.9;9.0}"

python -m pip install --force-reinstall --no-deps --no-build-isolation "${POINTNET_DIR}"
