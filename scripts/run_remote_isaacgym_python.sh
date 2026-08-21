#!/usr/bin/env bash
# Launch the compute-platform Isaac Gym Python with its isolated libraries.
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mck_root="$(cd "${project_root}/.." && pwd)"
isaac_root="${CONTACTDIFF_REMOTE_ISAAC_ROOT:-${mck_root}/IsaacGym}"
export PYTHONNOUSERSITE=1
export TORCH_EXTENSIONS_DIR="${isaac_root}/.torch_extensions"
export MAX_JOBS="${MAX_JOBS:-4}"
export LD_LIBRARY_PATH="${isaac_root}/.conda-env/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
exec "${isaac_root}/.conda-env/bin/python" "$@"
