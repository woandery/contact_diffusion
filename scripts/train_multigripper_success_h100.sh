#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONFIG="${CONFIG:-${PROJECT_ROOT}/configs/contact_diffusion_contact_format_multigripper_success_h100.yaml}"
NUM_GPUS="${NUM_GPUS:-1}"

cd "${PROJECT_ROOT}"
mkdir -p "${PROJECT_ROOT}/.cache/torch/kernels"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export NVIDIA_TF32_OVERRIDE="${NVIDIA_TF32_OVERRIDE:-1}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTORCH_KERNEL_CACHE_PATH="${PROJECT_ROOT}/.cache/torch/kernels"

if ! nvidia-smi --query-gpu=name --format=csv,noheader | grep -q "H100"; then
    echo "WARNING: visible GPUs are not H100 GPUs:" >&2
    nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader >&2
fi

python -c "from pointnet2_ops.pointnet2_modules import PointnetSAModule" \
    || { echo "PointNet++ is unavailable. Run: TORCH_CUDA_ARCH_LIST=9.0 bash scripts/install_pointnet2_ops.sh" >&2; exit 1; }

echo "Launching ContactDiffusion with ${NUM_GPUS} GPU(s) and config ${CONFIG}"
torchrun --standalone --nproc_per_node="${NUM_GPUS}" train.py --config "${CONFIG}" "$@"
