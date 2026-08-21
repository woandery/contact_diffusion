#!/usr/bin/env bash
# Paired diffusion-contact vs geometry-matched-random guidance diagnostic.
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
workspace_root="$(cd "${project_root}/.." && pwd)"
python_path="${CONTACTDIFF_PYTHON:-python}"
isaac_runner="${CONTACTDIFF_ISAAC_RUNNER:-${project_root}/scripts/run_remote_isaacgym_python.sh}"
checkpoint="${CONTACTDIFF_CHECKPOINT:-${project_root}/outputs/contact_diffusion_multidex_filtered_barrett_shadow_balanced_n35_4x4090/checkpoints/step_00050000.pt}"
expected_checkpoint_sha="0934a15218f0f35cfd978d207e5c374556c0f8afbfb597e0489795d15ae97cc5"
protocol="${project_root}/configs/contact_set_guidance_ab_palm0_v4_protocol.yaml"
manifest="${project_root}/configs/basic_experiment_ood10_local_manifest.json"
run_root="${CONTACTDIFF_GUIDANCE_AB_RUN_ROOT:-${project_root}/outputs/contact_set_guidance_ab_palm0_v4_h100}"
start_stage="${CONTACTDIFF_GUIDANCE_AB_START_STAGE:-generate}"
generation_workers="${CONTACTDIFF_GUIDANCE_AB_GENERATION_WORKERS:-12}"
prepare_workers="${CONTACTDIFF_GUIDANCE_AB_PREPARE_WORKERS:-8}"
matched_random_candidates="${CONTACTDIFF_GUIDANCE_AB_MATCH_CANDIDATES:-4096}"
matched_random_seed_offset="${CONTACTDIFF_GUIDANCE_AB_MATCH_SEED_OFFSET:-7919}"
seed="${CONTACTDIFF_GUIDANCE_AB_SEED:-20260808}"
batch_size="${CONTACTDIFF_GUIDANCE_AB_GYM_BATCH_SIZE:-512}"
gpu_id_csv="${CONTACTDIFF_GPU_IDS:-0,1,2,3}"
status_root="${run_root}/supervisor"

IFS=',' read -r -a gpu_ids <<<"${gpu_id_csv}"
if ((${#gpu_ids[@]} != 4)); then
  printf 'CONTACTDIFF_GPU_IDS must contain exactly four comma-separated GPU IDs\n' >&2
  exit 2
fi
if ((batch_size != 512)); then
  printf 'v4 A/B protocol requires CONTACTDIFF_GUIDANCE_AB_GYM_BATCH_SIZE=512\n' >&2
  exit 2
fi
if ((generation_workers < 1 || prepare_workers < 1)); then
  printf 'worker counts must be positive\n' >&2
  exit 2
fi

case "${start_stage}" in
  generate) start_rank=0 ;;
  prepare) start_rank=1 ;;
  gym) start_rank=2 ;;
  report) start_rank=3 ;;
  *)
    printf 'CONTACTDIFF_GUIDANCE_AB_START_STAGE must be generate, prepare, gym, or report\n' >&2
    exit 2
    ;;
esac

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
tasks=()
generation_tasks=()
for variant in diffusion matched_random; do
  for hand in barrett shadowhand; do
    for object_index in "${!objects[@]}"; do
      tasks+=("${variant}:${hand}:${objects[object_index]}:${object_index}")
    done
  done
done
for hand in barrett shadowhand; do
  for object_index in "${!objects[@]}"; do
    generation_tasks+=("diffusion:${hand}:${objects[object_index]}:${object_index}")
  done
done

parse_task() {
  local task="$1" remainder
  parsed_variant="${task%%:*}"
  remainder="${task#*:}"
  parsed_hand="${remainder%%:*}"
  remainder="${remainder#*:}"
  parsed_object="${remainder%%:*}"
  parsed_object_index="${remainder##*:}"
}

hand_settings() {
  local hand="$1"
  case "${hand}" in
    barrett)
      resolved_gripper="Barrett"
      resolved_hand_index=0
      resolved_config="${project_root}/configs/multigripper_fk_multidex_ood10_barrett_steps400_palm0_v4.yaml"
      resolved_hand_root="${project_root}/outputs/local_dro_isaacgym_twohands/full/assets/barrett_adagrasp"
      resolved_hand_urdf="model_extended.urdf"
      ;;
    shadowhand)
      resolved_gripper="shadow_hand"
      resolved_hand_index=1
      resolved_config="${project_root}/configs/multigripper_fk_multidex_ood10_shadow_dro_steps400_palm0_v4.yaml"
      resolved_hand_root="${project_root}/outputs/shadow_dro_failure_visualizations/assets/robot/shadowhand"
      resolved_hand_urdf="shadow_hand_right_extended.urdf"
      ;;
    *)
      printf 'unsupported hand: %s\n' "${hand}" >&2
      return 2
      ;;
  esac
}

wait_for_workers() {
  local failed=0 pid
  for pid in "$@"; do
    if ! wait "${pid}"; then failed=1; fi
  done
  return "${failed}"
}

mkdir -p "${status_root}" "${run_root}/reports"
for variant in diffusion matched_random; do
  for hand in barrett shadowhand; do
    mkdir -p \
      "${run_root}/${variant}/candidates/${hand}" \
      "${run_root}/${variant}/prepared/${hand}" \
      "${run_root}/${variant}/results/${hand}" \
      "${run_root}/${variant}/logs/${hand}"
  done
done

if [[ ! -f "${checkpoint}" ]]; then
  printf 'filtered MF50 checkpoint not found: %s\n' "${checkpoint}" >&2
  exit 2
fi
actual_checkpoint_sha="$(sha256sum "${checkpoint}" | awk '{print $1}')"
if [[ "${actual_checkpoint_sha}" != "${expected_checkpoint_sha}" ]]; then
  printf 'checkpoint SHA256 mismatch: %s != %s\n' \
    "${actual_checkpoint_sha}" "${expected_checkpoint_sha}" >&2
  exit 2
fi
if [[ ! -x "${isaac_runner}" ]]; then
  printf 'Isaac Gym launcher is not executable: %s\n' "${isaac_runner}" >&2
  exit 2
fi

date -u +%Y-%m-%dT%H:%M:%SZ >"${status_root}/started_at_utc"
printf '%s\n' "${actual_checkpoint_sha}" >"${status_root}/checkpoint.sha256"
printf '%s\n' "${gpu_id_csv}" >"${status_root}/gpu_ids"
on_exit() {
  local exit_code=$?
  if ((exit_code != 0)); then
    printf 'failed exit_code=%s\n' "${exit_code}" >"${status_root}/status"
  fi
}
trap on_exit EXIT

generate_one() {
  local variant="$1" hand="$2" object_id="$3" object_index="$4" gpu="$5"
  local output log source_args=()
  hand_settings "${hand}"
  output="${run_root}/${variant}/candidates/${hand}/${object_id}.json"
  log="${run_root}/${variant}/logs/${hand}/generate_${object_id}.log"
  if [[ "${variant}" == "matched_random" ]]; then
    source_args+=(
      --source-diffusion-candidates
      "${run_root}/diffusion/candidates/${hand}/${object_id}.json"
    )
  fi
  CUDA_VISIBLE_DEVICES="${gpu}" "${python_path}" \
    "${project_root}/scripts/infer_local_contactdiffusion_grasp.py" \
    --config "${resolved_config}" --checkpoint "${checkpoint}" \
    --manifest "${manifest}" --manifest-only --object-id "${object_id}" \
    --grippers "${resolved_gripper}" --sample-start 0 --samples-per-object 64 \
    --particles 32 --optimization-steps 400 --diffusion-steps 50 \
    --fk-initialization enveloping --envelope-approach-weight 2.0 \
    --selection-min-envelope-cosine 0.5 \
    --selection-min-approach-cosine 0.8 \
    --preferred-root-direction 0 0 1 \
    --selection-max-penetration 0.007 --top-k 32 \
    --contact-target-mode "${variant}" \
    --initialization-contact-source diffusion \
    --matched-random-candidates "${matched_random_candidates}" \
    --matched-random-seed-offset "${matched_random_seed_offset}" \
    "${source_args[@]}" \
    --device cuda:0 --seed "${seed}" \
    --hand-index-offset "${resolved_hand_index}" \
    --object-index-offset "${object_index}" --resume --output "${output}" \
    >"${log}" 2>&1
}

generation_worker() {
  local slot="$1" task_index task
  local gpu="${gpu_ids[$((slot % 4))]}"
  for ((task_index=slot; task_index<${#generation_tasks[@]}; task_index+=generation_workers)); do
    task="${generation_tasks[task_index]}"
    parse_task "${task}"
    generate_one diffusion "${parsed_hand}" "${parsed_object}" \
      "${parsed_object_index}" "${gpu}"
    generate_one matched_random "${parsed_hand}" "${parsed_object}" \
      "${parsed_object_index}" "${gpu}"
  done
}

if ((start_rank <= 0)); then
  printf 'generating_20_paired_object_hand_jobs\n' >"${status_root}/status"
  pids=()
  for ((slot=0; slot<generation_workers; slot++)); do
    generation_worker "${slot}" &
    pids+=("$!")
  done
  if ! wait_for_workers "${pids[@]}"; then
    printf 'failed_generation\n' >"${status_root}/status"
    exit 1
  fi
fi

audit_candidates() {
  "${python_path}" "${project_root}/scripts/summarize_contact_set_guidance_ab.py" \
    --run-root "${run_root}" --audit-only \
    --output "${run_root}/reports/pairing_audit.json" \
    >"${status_root}/pairing_audit.log" 2>&1
}

if ((start_rank <= 2)); then
  printf 'auditing_ab_pairing\n' >"${status_root}/status"
  audit_candidates
fi

prepare_one() {
  local variant="$1" hand="$2" object_id="$3"
  local log
  hand_settings "${hand}"
  log="${run_root}/${variant}/logs/${hand}/prepare_${object_id}.log"
  "${python_path}" "${project_root}/scripts/prepare_basic_experiment_isaacgym.py" \
    --candidates "${run_root}/${variant}/candidates/${hand}/${object_id}.json" \
    --config "${resolved_config}" --gripper "${resolved_gripper}" \
    --candidate-mode all --execution-protocol "${protocol}" \
    --allow-runtime-budget \
    --output "${run_root}/${variant}/prepared/${hand}/${object_id}.json" \
    >"${log}" 2>&1
}

preparation_worker() {
  local slot="$1" task_index task
  for ((task_index=slot; task_index<${#tasks[@]}; task_index+=prepare_workers)); do
    task="${tasks[task_index]}"
    parse_task "${task}"
    prepare_one "${parsed_variant}" "${parsed_hand}" "${parsed_object}"
  done
}

if ((start_rank <= 1)); then
  printf 'preparing_40_all_particle_manifests\n' >"${status_root}/status"
  pids=()
  for ((slot=0; slot<prepare_workers; slot++)); do
    preparation_worker "${slot}" &
    pids+=("$!")
  done
  if ! wait_for_workers "${pids[@]}"; then
    printf 'failed_preparation\n' >"${status_root}/status"
    exit 1
  fi
fi

result_is_complete() {
  local path="$1" expected="$2"
  [[ -s "${path}" ]] && "${python_path}" -c \
    'import json,sys; d=json.load(open(sys.argv[1])); raise SystemExit(not (d.get("status")=="complete" and len(d.get("results", []))==int(sys.argv[2])))' \
    "${path}" "${expected}"
}

run_gym_task() {
  local variant="$1" hand="$2" object_id="$3" gpu="$4"
  local prepared result_dir log_dir start output
  hand_settings "${hand}"
  prepared="${run_root}/${variant}/prepared/${hand}/${object_id}.json"
  result_dir="${run_root}/${variant}/results/${hand}/${object_id}"
  log_dir="${run_root}/${variant}/logs/${hand}/${object_id}"
  mkdir -p "${result_dir}" "${log_dir}"
  for ((start=0; start<2048; start+=512)); do
    output="${result_dir}/batch_$(printf '%04d' "${start}")_$(printf '%04d' "$((start + 511))").json"
    if result_is_complete "${output}" 512; then
      continue
    fi
    CUDA_VISIBLE_DEVICES="${gpu}" "${isaac_runner}" \
      "${project_root}/scripts/validate_native_shadowhand_isaacgym_oldparams_dro.py" \
      --prepared "${prepared}" --output "${output}" \
      --gendex-root "${workspace_root}/GenDexGrasp" \
      --native-hand-root "${resolved_hand_root}" \
      --native-hand-urdf "${resolved_hand_urdf}" \
      --native-hand-urdf-is-extended --only-object "${object_id}" \
      --sample-start "${start}" --max-samples-per-object 512 \
      --device-id 0 --envs-per-row 23 --progress-every 16 \
      --asset-profile dro --object-source dro \
      --dro-object-root "${project_root}/outputs/local_dro_params_only_coacd/assets/object" \
      --steps-per-second 100 --substeps 2 --closure-steps 100 \
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
      >"${log_dir}/gym_${start}_$((start + 511)).log" 2>&1
  done
}

gym_worker() {
  local gpu_slot="$1" task_index task
  local gpu="${gpu_ids[gpu_slot]}"
  printf 'running\n' >"${status_root}/gym_gpu${gpu_slot}.status"
  for ((task_index=gpu_slot; task_index<${#tasks[@]}; task_index+=4)); do
    task="${tasks[task_index]}"
    parse_task "${task}"
    printf 'running %s %s %s\n' \
      "${parsed_variant}" "${parsed_hand}" "${parsed_object}" \
      >"${status_root}/gym_gpu${gpu_slot}.status"
    run_gym_task "${parsed_variant}" "${parsed_hand}" "${parsed_object}" "${gpu}"
  done
  printf 'complete\n' >"${status_root}/gym_gpu${gpu_slot}.status"
}

if ((start_rank <= 2)); then
  printf 'validating_160_gpu_physx_batches\n' >"${status_root}/status"
  pids=()
  for gpu_slot in 0 1 2 3; do
    gym_worker "${gpu_slot}" &
    pids+=("$!")
  done
  if ! wait_for_workers "${pids[@]}"; then
    printf 'failed_gym\n' >"${status_root}/status"
    exit 1
  fi
fi

printf 'summarizing_paired_results\n' >"${status_root}/status"
for variant in diffusion matched_random; do
  "${python_path}" "${project_root}/scripts/summarize_basic_experiment_all_particles.py" \
    --run-root "${run_root}/${variant}" \
    --output "${run_root}/reports/${variant}_summary.json" \
    >"${status_root}/${variant}_summary.log" 2>&1
done
"${python_path}" "${project_root}/scripts/summarize_contact_set_guidance_ab.py" \
  --run-root "${run_root}" \
  --output "${run_root}/reports/paired_summary.json" \
  --output-md "${run_root}/reports/PAIRED_SUMMARY.md" \
  >"${status_root}/paired_summary.log" 2>&1
printf 'complete\n' >"${status_root}/status"
date -u +%Y-%m-%dT%H:%M:%SZ >"${status_root}/completed_at_utc"
