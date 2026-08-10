#!/usr/bin/env bash
# Re-evaluate the current 640 Barrett D(R,O) candidates under the full old Gym protocol.
set -euo pipefail

mck_root="${MCK_ROOT:-/inspire/qb-ilm2/project/zhanghanbo/public/mck}"
project_root="${mck_root}/contact_diffusion"
dro_root="${mck_root}/dro_grasp_reproduction/DRO-Grasp"
gendex_root="${mck_root}/GenDexGrasp"
gym_python="${mck_root}/IsaacGym/.conda-env/bin/python"
run_root="${RUN_ROOT:-outputs/barrett_dro_oldparams_isaacgym}"
prepared="outputs/multidex_ood10_barrett64x32_top1_steps400_dro_gendex/prepared/barrett.json"
hand_root="${dro_root}/data/data_urdf/robot/barrett"
objects=(
  contactdb_apple contactdb_camera contactdb_cylinder_medium
  contactdb_door_knob contactdb_rubber_duck contactdb_water_bottle
  ycb_055_baseball ycb_016_pear
  ycb_010_potted_meat_can ycb_005_tomato_soup_can
)

cd "${project_root}"
mkdir -p "${run_root}"/{results,logs,status}

run_object() {
  local object_name="$1" gpu="$2"
  PYTHONNOUSERSITE=1 \
  TORCH_EXTENSIONS_DIR="${mck_root}/IsaacGym/.torch_extensions" \
  MAX_JOBS=4 CUDA_VISIBLE_DEVICES="${gpu}" \
  LD_LIBRARY_PATH="${mck_root}/IsaacGym/.conda-env/lib:${LD_LIBRARY_PATH:-}" \
  "${gym_python}" scripts/validate_native_shadowhand_isaacgym_oldparams_dro.py \
    --prepared "${prepared}" \
    --output "${run_root}/results/${object_name}.json" \
    --gendex-root "${gendex_root}" \
    --native-hand-root "${hand_root}" \
    --native-hand-urdf model.urdf \
    --device-id 0 --only-object "${object_name}" --resume \
    >"${run_root}/logs/${object_name}.log" 2>&1
}

printf 'running\n' >"${run_root}/status/validation.status"
for start in 0 2 4 6 8; do
  pids=()
  for offset in 0 1; do
    index=$((start + offset))
    run_object "${objects[index]}" "${offset}" &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do
    wait "${pid}"
  done
done

"${gym_python}" scripts/summarize_barrett_dro_oldparams_isaacgym.py \
  --results "${run_root}/results/*.json" \
  --output-json "${run_root}/summary.json" \
  --output-md "${run_root}/REPORT.md" \
  >"${run_root}/logs/summary.log" 2>&1
printf 'complete\n' >"${run_root}/status/validation.status"

