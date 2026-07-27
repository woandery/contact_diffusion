#!/usr/bin/env bash
set -eo pipefail

MCK_ROOT="/inspire/qb-ilm2/project/zhanghanbo/public/mck"
PROJECT_ROOT="${MCK_ROOT}/contact_diffusion"

if [[ "${OMNI_KIT_ACCEPT_EULA:-}" != "YES" ]]; then
    echo "Isaac Sim requires acceptance of the NVIDIA Omniverse EULA."
    echo "After reviewing it, run with: OMNI_KIT_ACCEPT_EULA=YES bash $0 <python-script> [args...]"
    exit 2
fi
if [[ $# -lt 1 ]]; then
    echo "Usage: OMNI_KIT_ACCEPT_EULA=YES bash $0 <python-script> [args...]" >&2
    exit 2
fi

mkdir -p \
    "${MCK_ROOT}/.cache/pip" \
    "${MCK_ROOT}/.cache/xdg" \
    "${MCK_ROOT}/.cache/xdg-runtime" \
    "${MCK_ROOT}/.cache/cuda" \
    "${MCK_ROOT}/.cache/torch_extensions" \
    "${MCK_ROOT}/.cache/torch/kernels" \
    "${MCK_ROOT}/.cache/torchinductor" \
    "${MCK_ROOT}/.cache/ov" \
    "${MCK_ROOT}/.config/isaacsim" \
    "${MCK_ROOT}/.config/ov" \
    "${MCK_ROOT}/.local/share/isaacsim" \
    "${MCK_ROOT}/.local/share/isaacsim/portable" \
    "${MCK_ROOT}/.local/share/ov/data" \
    "${MCK_ROOT}/.local/share/ov/logs"

export PIP_CACHE_DIR="${MCK_ROOT}/.cache/pip"
export XDG_CACHE_HOME="${MCK_ROOT}/.cache/xdg"
export XDG_CONFIG_HOME="${MCK_ROOT}/.config/isaacsim"
export XDG_DATA_HOME="${MCK_ROOT}/.local/share/isaacsim"
export XDG_RUNTIME_DIR="${MCK_ROOT}/.cache/xdg-runtime"
export CUDA_CACHE_PATH="${MCK_ROOT}/.cache/cuda"
export TORCH_EXTENSIONS_DIR="${MCK_ROOT}/.cache/torch_extensions"
export TORCHINDUCTOR_CACHE_DIR="${MCK_ROOT}/.cache/torchinductor"
export PYTORCH_KERNEL_CACHE_PATH="${MCK_ROOT}/.cache/torch/kernels"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.9}"
export OMNI_USER="${MCK_ROOT}/.local/share/ov"
export OMNI_CACHE_BASE="${MCK_ROOT}/.cache/ov"
export OMNI_DATA_BASE="${MCK_ROOT}/.local/share/ov/data"
export OMNI_LOGS="${MCK_ROOT}/.local/share/ov/logs"
export OMNI_CONFIG="${MCK_ROOT}/.config/ov"
export OMNI_CONFIG_PATH="${MCK_ROOT}/.config/ov"
export CONTACTDIFF_OMNI_PORTABLE_ROOT="${CONTACTDIFF_OMNI_PORTABLE_ROOT:-${MCK_ROOT}/.local/share/isaacsim/portable}"
mkdir -p "${CONTACTDIFF_OMNI_PORTABLE_ROOT}"
# The platform-provided ICD points at libGLX_nvidia, whose Vulkan negotiation
# fails in this headless container.  The matching NVIDIA EGL driver exposes
# the same Vulkan ICD API without requiring the GLX path.
export VK_DRIVER_FILES="${PROJECT_ROOT}/configs/nvidia_egl_icd.json"
chmod 700 "${XDG_RUNTIME_DIR}"

source "${MCK_ROOT}/miniconda3/bin/activate" contactdiff

# The pip distribution loads USD from independent cache packages.  The Python
# binding cannot discover these few private libraries through the conda
# environment's default linker path.  Keep this list narrow: adding every
# extension directory can shadow Kit's own Vulkan loader.
ISAAC_PACKAGE_ROOT="${CONDA_PREFIX}/lib/python3.10/site-packages/isaacsim"
ISAAC_LIBRARY_PATHS="${CONDA_PREFIX}/lib:${CONDA_PREFIX}/lib/python3.10/site-packages/omni:${CONDA_PREFIX}/lib/python3.10/site-packages/omni/kernel/plugins"
shopt -s nullglob
for isaac_library_dir in \
    "${ISAAC_PACKAGE_ROOT}"/extscache/omni.usd.core-*/bin \
    "${ISAAC_PACKAGE_ROOT}"/extscache/omni.usd.libs-*/bin \
    "${ISAAC_PACKAGE_ROOT}"/extscache/omni.usd.schema.audio-*/lib; do
    ISAAC_LIBRARY_PATHS="${ISAAC_LIBRARY_PATHS}:${isaac_library_dir}"
done
shopt -u nullglob
export LD_LIBRARY_PATH="${ISAAC_LIBRARY_PATHS}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

set -u
cd "${PROJECT_ROOT}"
exec python "$@"
