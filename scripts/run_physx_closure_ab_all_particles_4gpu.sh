#!/usr/bin/env bash
# Controlled full-particle A/B: dynamic closure versus fixed-until-inner closure.
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mck_root="$(cd "${project_root}/.." && pwd)"
python_path="${CONTACTDIFF_PYTHON:-${mck_root}/miniconda3/envs/contactdiff_fk_localalign/bin/python}"
isaac_runner="${CONTACTDIFF_ISAAC_RUNNER:-${project_root}/scripts/run_remote_isaacgym_python.sh}"
source_run_root="${CONTACTDIFF_AB_SOURCE_ROOT:-${project_root}/outputs/basic_experiment_balanced_n35_step50k_all_particles_o10i20_v1_remote4090}"
run_root="${CONTACTDIFF_AB_RUN_ROOT:-${project_root}/outputs/basic_experiment_balanced_n35_step50k_closure_ab_all_particles_v1}"
batch_size="${CONTACTDIFF_AB_BATCH_SIZE:-512}"
inner_hold_steps="${CONTACTDIFF_AB_INNER_HOLD_STEPS:-100}"
num_gpus="${CONTACTDIFF_AB_GPUS:-4}"
smoke="${CONTACTDIFF_AB_SMOKE:-0}"
total_samples="${CONTACTDIFF_AB_SAMPLES_PER_HAND_OBJECT:-2048}"
selection="${CONTACTDIFF_AB_SELECTION:-all_particles}"
report_title="${CONTACTDIFF_AB_REPORT_TITLE:-step-50k 全粒子闭合阶段 PhysX A/B}"

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
if [[ "${smoke}" == "1" ]]; then
  objects=(contactdb_apple)
  total_samples=4
fi
conditions=(A_dynamic B_fixed)

if [[ ! -x "${python_path}" ]]; then
  echo "missing Python: ${python_path}" >&2
  exit 1
fi
if [[ ! -x "${isaac_runner}" ]]; then
  echo "missing Isaac Gym runner: ${isaac_runner}" >&2
  exit 1
fi
if [[ ! -d "${source_run_root}/prepared" ]]; then
  echo "missing frozen prepared manifests: ${source_run_root}/prepared" >&2
  exit 1
fi
if [[ ! "${total_samples}" =~ ^[1-9][0-9]*$ ]]; then
  echo "invalid samples per hand/object: ${total_samples}" >&2
  exit 1
fi
if ((batch_size < 1 || inner_hold_steps < 0 || num_gpus < 1)); then
  echo "invalid batch/hold/GPU configuration" >&2
  exit 1
fi

tasks=()
for hand in barrett shadowhand; do
  for object_id in "${objects[@]}"; do
    tasks+=("${hand}:${object_id}")
  done
done

mkdir -p "${run_root}/supervisor"
date -u +%Y-%m-%dT%H:%M:%SZ >"${run_root}/supervisor/started_at_utc"
printf 'running\n' >"${run_root}/supervisor/status"
printf '%s\n' \
  "source_run_root=${source_run_root}" \
  "run_root=${run_root}" \
  "smoke=${smoke}" \
  "samples_per_hand_object=${total_samples}" \
  "paired_unique_trials=$((${#tasks[@]} * total_samples))" \
  "total_physx_trials=$((2 * ${#tasks[@]} * total_samples))" \
  "batch_size=${batch_size}" \
  "inner_hold_steps=${inner_hold_steps}" \
  "gpus=${num_gpus}" \
  "selection=${selection}" \
  "pairing=same candidate, batch order, and GPU; A then B" \
  >"${run_root}/supervisor/config"

worker() {
  local gpu="$1" index task hand object_id condition mode
  local prepared hand_root hand_urdf result_dir telemetry_dir log_dir
  local start count output
  printf 'running\n' >"${run_root}/supervisor/gpu${gpu}.status"
  for ((index=gpu; index<${#tasks[@]}; index+=num_gpus)); do
    task="${tasks[index]}"
    hand="${task%%:*}"
    object_id="${task#*:}"
    if [[ "${hand}" == "barrett" ]]; then
      hand_root="${project_root}/outputs/local_dro_isaacgym_twohands/full/assets/barrett_adagrasp"
      hand_urdf="model_extended.urdf"
    else
      hand_root="${project_root}/outputs/shadow_dro_failure_visualizations/assets/robot/shadowhand"
      hand_urdf="shadow_hand_right_extended.urdf"
    fi
    prepared="${source_run_root}/prepared/${hand}/${object_id}.json"
    if [[ ! -s "${prepared}" ]]; then
      echo "missing prepared manifest: ${prepared}" >&2
      return 1
    fi

    # Keep the pair on one GPU and preserve the exact sample window order.
    for condition in "${conditions[@]}"; do
      if [[ "${condition}" == "A_dynamic" ]]; then
        mode="dynamic"
      else
        mode="fixed_until_inner"
      fi
      printf 'running %s %s %s\n' "${condition}" "${hand}" "${object_id}" \
        >"${run_root}/supervisor/gpu${gpu}.status"
      result_dir="${run_root}/results/${condition}/${hand}/${object_id}"
      telemetry_dir="${run_root}/telemetry/${condition}/${hand}/${object_id}"
      log_dir="${run_root}/logs/${condition}/${hand}/${object_id}"
      mkdir -p "${result_dir}" "${telemetry_dir}" "${log_dir}"
      for ((start=0; start<total_samples; start+=batch_size)); do
        count="${batch_size}"
        if ((start + count > total_samples)); then
          count=$((total_samples - start))
        fi
        output="${result_dir}/batch_$(printf '%04d' "${start}")_$(printf '%04d' "$((start + count - 1))").json"
        if [[ -s "${output}" ]] && "${python_path}" -c \
          'import json,sys; d=json.load(open(sys.argv[1])); raise SystemExit(not (d.get("status")=="complete" and len(d.get("results",[]))==int(sys.argv[2])))' \
          "${output}" "${count}"; then
          continue
        fi
        CUDA_VISIBLE_DEVICES="${gpu}" "${isaac_runner}" \
          "${project_root}/scripts/validate_native_shadowhand_isaacgym_oldparams_dro.py" \
          --prepared "${prepared}" --output "${output}" \
          --gendex-root "${mck_root}/GenDexGrasp" \
          --native-hand-root "${hand_root}" --native-hand-urdf "${hand_urdf}" \
          --native-hand-urdf-is-extended --only-object "${object_id}" \
          --sample-start "${start}" --max-samples-per-object "${count}" \
          --device-id 0 --progress-every 64 \
          --asset-profile dro --object-source dro \
          --dro-object-root "${project_root}/outputs/local_dro_params_only_coacd/assets/object" \
          --steps-per-second 100 --substeps 2 --closure-steps 100 \
          --closure-trajectory step --inner-hold-steps "${inner_hold_steps}" \
          --closure-ab-experiment --closure-object-mode "${mode}" \
          --closure-telemetry-dir "${telemetry_dir}" \
          --direction-seconds 1.0 --direction-order cedex \
          --success-mode final --threshold 0.02 --acceleration 0.5 \
          --robot-friction 3 --object-friction 3 --object-density 500 \
          --object-linear-damping -1 --object-angular-damping -1 \
          --joint-stiffness 1000 --joint-damping 200 \
          --joint-armature -1 --joint-velocity -1 \
          --pregrasp-open-fraction 0 --closure-overdrive-fraction 0 \
          --outer-settle-steps 0 \
          --virtual-root-stiffness 1000 --virtual-root-damping 200 \
          --solver-position-iterations 8 --solver-velocity-iterations 0 \
          --contact-offset 0.01 --rest-offset 0 --no-ground \
          >"${log_dir}/gym_${start}_$((start + count - 1)).log" 2>&1
      done
    done
  done
  printf 'complete\n' >"${run_root}/supervisor/gpu${gpu}.status"
}

pids=()
for ((gpu=0; gpu<num_gpus; gpu++)); do
  worker "${gpu}" &
  pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then failed=1; fi
done
if ((failed)); then
  printf 'failed\n' >"${run_root}/supervisor/status"
  exit 1
fi

summary_args=(
  --run-root "${run_root}"
  --expected-trials "$((${#tasks[@]} * total_samples))"
  --report-title "${report_title}"
)
"${python_path}" "${project_root}/scripts/summarize_physx_closure_ab_all_particles.py" \
  "${summary_args[@]}" >"${run_root}/supervisor/summary.log" 2>&1
printf 'complete\n' >"${run_root}/supervisor/status"
date -u +%Y-%m-%dT%H:%M:%SZ >"${run_root}/supervisor/completed_at_utc"
printf 'complete: %s\n' "${run_root}"
