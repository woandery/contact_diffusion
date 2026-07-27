#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONFIG="${CONFIG:-${PROJECT_ROOT}/configs/contact_diffusion_contact_format_multigripper_success_resume_4090x2.yaml}"
NUM_GPUS="${NUM_GPUS:-2}"
CONTACTDIFF_ENV="${CONTACTDIFF_ENV:-/inspire/qb-ilm2/project/zhanghanbo/public/mck/miniconda3/envs/contactdiff}"
PYTHON_BIN="${PYTHON_BIN:-${CONTACTDIFF_ENV}/bin/python}"
TORCHRUN_BIN="${TORCHRUN_BIN:-${CONTACTDIFF_ENV}/bin/torchrun}"

cd "${PROJECT_ROOT}"
mkdir -p "${PROJECT_ROOT}/.cache/torch/kernels"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export NVIDIA_TF32_OVERRIDE="${NVIDIA_TF32_OVERRIDE:-1}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTORCH_KERNEL_CACHE_PATH="${PROJECT_ROOT}/.cache/torch/kernels"

if [[ ! -x "${PYTHON_BIN}" || ! -x "${TORCHRUN_BIN}" ]]; then
    echo "contactdiff executables not found under ${CONTACTDIFF_ENV}" >&2
    exit 1
fi

"${PYTHON_BIN}" -c "from pointnet2_ops.pointnet2_modules import PointnetSAModule" \
    || { echo "PointNet++ is unavailable. Rebuild it for CUDA architectures 8.9 and 9.0." >&2; exit 1; }

echo "Resuming ContactDiffusion on ${NUM_GPUS} RTX 4090 GPU(s) with ${CONFIG}"
"${TORCHRUN_BIN}" --standalone --nproc_per_node="${NUM_GPUS}" train.py --config "${CONFIG}" "$@"
