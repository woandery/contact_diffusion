#!/usr/bin/env bash
# Run the ten OOD objects with the original D(R,O) Isaac Gym protocol.
set -euo pipefail

workspace_root="${MCK_ROOT:-/inspire/qb-ilm2/project/zhanghanbo/public/mck}"
dro_root="$workspace_root/dro_grasp_reproduction/DRO-Grasp"
isaac_root="$workspace_root/IsaacGym"
python_path="$isaac_root/.conda-env/bin/python"
run_root="${RUN_ROOT:-$workspace_root/contact_diffusion/outputs/contactdiff_ood10_dro_isaacgym}"
candidates="${CANDIDATES:-$run_root/inputs/barrett_matched64x32_fk800.json}"
robot_name="${ROBOT_NAME:-barrett}"

case "$robot_name" in
  barrett|shadowhand) ;;
  *)
    printf 'Unsupported ROBOT_NAME: %s\n' "$robot_name" >&2
    exit 2
    ;;
esac

if [[ ! -f "$candidates" ]]; then
  printf 'Candidate file does not exist: %s\n' "$candidates" >&2
  exit 2
fi

mkdir -p "$run_root/inputs" "$run_root/results" "$run_root/logs" "$run_root/status"

run_object() {
  local object_id="$1"
  local physical_gpu="$2"
  printf 'running\n' > "$run_root/status/${object_id}.status"
  if ! env \
    PYTHONNOUSERSITE=1 \
    TORCH_EXTENSIONS_DIR="$isaac_root/.torch_extensions" \
    MAX_JOBS=4 \
    CUDA_VISIBLE_DEVICES="$physical_gpu" \
    LD_LIBRARY_PATH="$isaac_root/.conda-env/lib:${LD_LIBRARY_PATH:-}" \
    "$python_path" "$dro_root/scripts/validate_contactdiff_ood10_dro_isaacgym.py" \
    --candidates "$candidates" \
    --robot-name "$robot_name" \
    --object-id "$object_id" \
    --gpu 0 \
    --output "$run_root/results/${object_id}.json" \
    > "$run_root/logs/${object_id}.log" 2>&1; then
    printf 'failed\n' > "$run_root/status/${object_id}.status"
    return 1
  fi
  printf 'complete\n' > "$run_root/status/${object_id}.status"
}

objects=(
  contactdb_apple
  contactdb_camera
  contactdb_cylinder_medium
  contactdb_door_knob
  contactdb_rubber_duck
  contactdb_water_bottle
  ycb_055_baseball
  ycb_016_pear
  ycb_010_potted_meat_can
  ycb_005_tomato_soup_can
)

printf 'running\n' > "$run_root/status/all.status"
pids=()
failed=0
for index in "${!objects[@]}"; do
  run_object "${objects[$index]}" "$((index % 4))" &
  pids+=("$!")
  if (( ${#pids[@]} == 4 )); then
    for pid in "${pids[@]}"; do
      if ! wait "$pid"; then failed=1; fi
    done
    pids=()
  fi
done
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then failed=1; fi
done
if (( failed )); then
  printf 'failed\n' > "$run_root/status/all.status"
  exit 1
fi
printf 'complete\n' > "$run_root/status/all.status"
