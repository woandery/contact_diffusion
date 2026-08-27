#!/usr/bin/env bash
# Run an Isaac Gym script in either the local or compute-platform environment.
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mck_root="$(cd "${project_root}/.." && pwd)"
if [[ -d "${mck_root}/IsaacGymLocal" ]]; then
  isaac_root="${mck_root}/IsaacGymLocal"
else
  isaac_root="${mck_root}/IsaacGym"
fi
env_root="${isaac_root}/.conda-env"

export PATH="${env_root}/bin:${PATH}"
export LD_LIBRARY_PATH="${env_root}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export PYTHONPATH="${isaac_root}/isaacgym/python:${mck_root}/GenDexGrasp${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONNOUSERSITE=1
export TORCH_EXTENSIONS_DIR="${isaac_root}/.torch_extensions"
export MAX_JOBS="${MAX_JOBS:-4}"

exec "${env_root}/bin/python" "$@"
