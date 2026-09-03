#!/usr/bin/env bash
# AR64k contacts -> frozen FK -> exact EAWQ Top-1 -> GPU PhysX OOD-10.
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mck_root="$(cd "${project_root}/.." && pwd)"
python_path="${CONTACTDIFF_PYTHON:-${mck_root}/miniconda3/envs/contactdiff_fk_localalign/bin/python}"
isaac_runner="${CONTACTDIFF_ISAAC_RUNNER:-${project_root}/scripts/run_remote_isaacgym_python.sh}"
checkpoint="${CONTACT_AR_BASIC_CHECKPOINT:-${project_root}/outputs/contact_ar_success_k128_epsilonauc_fullpc_norm_warmk32_gb768_64k_8x4090/model/checkpoints/step_00064000.pt}"
expected_checkpoint_sha="${CONTACT_AR_BASIC_EXPECTED_CHECKPOINT_SHA:-c58404d03d9a7049774086f62d0fee9a56e18e656c8a6182080e92edfdcab67b}"
manifest="${project_root}/configs/basic_experiment_ood10_local_manifest.json"
samples="${CONTACT_AR_BASIC_SAMPLES:-4}"
stage="${CONTACT_AR_BASIC_STAGE:-generate}"
gpu_ids_text="${CONTACT_AR_BASIC_GPU_IDS:-0,1,2,3,4,5,6,7}"
IFS=',' read -r -a gpu_ids <<<"${gpu_ids_text}"
gpu_count="${#gpu_ids[@]}"
generation_workers="${CONTACT_AR_BASIC_GENERATION_WORKERS:-${gpu_count}}"
source_candidate_root="${CONTACT_AR_BASIC_SOURCE_CANDIDATE_ROOT:-}"
ar_fk_target="${CONTACT_AR_BASIC_AR_FK_TARGET:-nearest_2048}"

if (( gpu_count < 1 || gpu_count > 8 )); then
  printf 'CONTACT_AR_BASIC_GPU_IDS must list between 1 and 8 devices\n' >&2
  exit 2
fi
if (( generation_workers < 1 || generation_workers > 20 )); then
  printf 'CONTACT_AR_BASIC_GENERATION_WORKERS must be in [1, 20]\n' >&2
  exit 2
fi

case "${samples}" in
  4)
    protocol="${CONTACT_AR_BASIC_PROTOCOL:-${project_root}/configs/basic_experiment_ar64k_eawq_o10i20_palm0_pilot4_protocol.yaml}"
    default_root="${project_root}/outputs/basic_experiment_ar64k_eawq_o10i20_palm0_pilot4_ood10"
    ;;
  32)
    protocol="${CONTACT_AR_BASIC_PROTOCOL:-${project_root}/configs/basic_experiment_partial_ar64k_eawq_o10i20_palm0_v5_protocol.yaml}"
    default_root="${project_root}/outputs/basic_experiment_partial_ar64k_eawq_o10i20_palm0_v5_ood10"
    ;;
  64)
    protocol="${CONTACT_AR_BASIC_PROTOCOL:-${project_root}/configs/basic_experiment_ar64k_eawq_o10i20_palm0_v1_protocol.yaml}"
    default_root="${project_root}/outputs/basic_experiment_ar64k_eawq_o10i20_palm0_v1_ood10"
    ;;
  *)
    printf 'CONTACT_AR_BASIC_SAMPLES must be 4, 32, or 64\n' >&2
    exit 2
    ;;
esac
run_root="${CONTACT_AR_BASIC_RUN_ROOT:-${default_root}}"
status_root="${run_root}/status"

objects=(
  contactdb_apple contactdb_camera contactdb_cylinder_medium
  contactdb_door_knob contactdb_rubber_duck contactdb_water_bottle
  ycb_055_baseball ycb_016_pear ycb_010_potted_meat_can
  ycb_005_tomato_soup_can
)
tasks=()
for hand in barrett shadowhand; do
  for object_index in "${!objects[@]}"; do
    tasks+=("${hand}:${objects[$object_index]}:${object_index}")
  done
done

case "${stage}" in
  generate) start_rank=0 ;;
  prepare) start_rank=1 ;;
  rank) start_rank=2 ;;
  gym) start_rank=3 ;;
  report) start_rank=4 ;;
  *) printf 'CONTACT_AR_BASIC_STAGE must be generate, prepare, rank, gym, or report\n' >&2; exit 2 ;;
esac

actual_checkpoint_sha="$(sha256sum "${checkpoint}" | awk '{print $1}')"
if [[ "${actual_checkpoint_sha}" != "${expected_checkpoint_sha}" ]]; then
  printf 'checkpoint SHA256 mismatch: %s != %s\n' \
    "${actual_checkpoint_sha}" "${expected_checkpoint_sha}" >&2
  exit 2
fi

mkdir -p "${status_root}" "${run_root}/logs" \
  "${run_root}/candidates/barrett" "${run_root}/candidates/shadowhand" \
  "${run_root}/prepared_all/barrett" "${run_root}/prepared_all/shadowhand" \
  "${run_root}/results/barrett" "${run_root}/results/shadowhand"
if [[ ! -s "${status_root}/started_at_utc" ]]; then
  date -u +%Y-%m-%dT%H:%M:%SZ >"${status_root}/started_at_utc"
fi
printf '%s\n' "${actual_checkpoint_sha}" >"${status_root}/checkpoint.sha256"
printf '%s\n' "${protocol}" >"${status_root}/protocol.path"
failed=0

task_fields() {
  local task="$1"
  TASK_HAND="${task%%:*}"
  local remainder="${task#*:}"
  TASK_OBJECT="${remainder%%:*}"
  TASK_OBJECT_INDEX="${remainder##*:}"
  if [[ "${TASK_HAND}" == "barrett" ]]; then
    TASK_GRIPPER=Barrett
    TASK_HAND_INDEX=0
    TASK_CONFIG="${project_root}/configs/multigripper_fk_multidex_ood10_barrett_steps400_palm0_v4.yaml"
    TASK_HAND_ROOT="${project_root}/outputs/local_dro_isaacgym_twohands/full/assets/barrett_adagrasp"
    TASK_HAND_URDF=model_extended.urdf
  else
    TASK_GRIPPER=shadow_hand
    TASK_HAND_INDEX=1
    TASK_CONFIG="${project_root}/configs/multigripper_fk_multidex_ood10_shadow_dro_steps400_palm0_v4.yaml"
    TASK_HAND_ROOT="${project_root}/outputs/shadow_dro_failure_visualizations/assets/robot/shadowhand"
    TASK_HAND_URDF=shadow_hand_right_extended.urdf
  fi
}

if (( start_rank <= 0 )); then
  printf 'generating\n' >"${status_root}/stage"
  generation_worker() {
    local worker="$1" gpu index task output log
    local -a generation_extra
    gpu="${gpu_ids[$((worker % gpu_count))]}"
    for ((index=worker; index<${#tasks[@]}; index+=generation_workers)); do
      task="${tasks[$index]}"; task_fields "${task}"
      output="${run_root}/candidates/${TASK_HAND}/${TASK_OBJECT}.json"
      log="${run_root}/logs/generate_${TASK_HAND}_${TASK_OBJECT}.log"
      printf 'running gpu=%s\n' "${gpu}" >"${status_root}/generate_${TASK_HAND}_${TASK_OBJECT}.status"
      generation_extra=(--autoregressive-fk-target "${ar_fk_target}")
      if [[ -n "${source_candidate_root}" ]]; then
        generation_extra+=(
          --source-diffusion-candidates
          "${source_candidate_root}/${TASK_HAND}/${TASK_OBJECT}.json"
          --replay-source-diffusion-rng
        )
      fi
      CUDA_VISIBLE_DEVICES="${gpu}" "${python_path}" \
        "${project_root}/scripts/infer_local_contactdiffusion_grasp.py" \
        --config "${TASK_CONFIG}" --checkpoint "${checkpoint}" \
        --manifest "${manifest}" --manifest-only --object-id "${TASK_OBJECT}" \
        --grippers "${TASK_GRIPPER}" --sample-start 0 --samples-per-object "${samples}" \
        --inference-object-observation "${CONTACT_AR_BASIC_OBSERVATION:-checkpoint}" \
        --autoregressive-fk-target nearest_2048 \
        --particles 32 --optimization-steps 400 --diffusion-steps 50 \
        --fk-initialization enveloping --envelope-approach-weight 2.0 \
        --selection-min-envelope-cosine 0.5 --selection-min-approach-cosine 0.8 \
        --preferred-root-direction 0 0 1 --selection-max-penetration 0.007 \
        --top-k 32 --device cuda:0 --seed 20260808 \
        "${generation_extra[@]}" \
        --hand-index-offset "${TASK_HAND_INDEX}" \
        --object-index-offset "${TASK_OBJECT_INDEX}" --resume --output "${output}" \
        >"${log}" 2>&1
      printf 'complete\n' >"${status_root}/generate_${TASK_HAND}_${TASK_OBJECT}.status"
    done
  }
  pids=()
  for worker in $(seq 0 $((generation_workers - 1))); do
    generation_worker "${worker}" & pids+=("$!")
  done
  for pid in "${pids[@]}"; do if ! wait "${pid}"; then failed=1; fi; done
  if (( failed )); then printf 'failed_generation\n' >"${status_root}/stage"; exit 1; fi
fi

if (( start_rank <= 1 )); then
  printf 'preparing\n' >"${status_root}/stage"
  pids=()
  for task in "${tasks[@]}"; do
    task_fields "${task}"
    extra=(--allow-runtime-budget)
    if (( samples == 4 )); then extra+=(--allow-incomplete); fi
    "${python_path}" "${project_root}/scripts/prepare_basic_experiment_isaacgym.py" \
      --candidates "${run_root}/candidates/${TASK_HAND}/${TASK_OBJECT}.json" \
      --config "${TASK_CONFIG}" --gripper "${TASK_GRIPPER}" --candidate-mode all \
      --execution-protocol "${protocol}" "${extra[@]}" \
      --output "${run_root}/prepared_all/${TASK_HAND}/${TASK_OBJECT}.json" \
      >"${run_root}/logs/prepare_${TASK_HAND}_${TASK_OBJECT}.log" 2>&1 &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do if ! wait "${pid}"; then failed=1; fi; done
  if (( failed )); then printf 'failed_preparation\n' >"${status_root}/stage"; exit 1; fi
fi

if (( start_rank <= 2 )); then
  printf 'ranking_eawq\n' >"${status_root}/stage"
  mkdir -p "${run_root}/eawq"
  CUDA_VISIBLE_DEVICES=0 "${python_path}" \
    "${project_root}/scripts/compute_eawq_rank_fusion_metrics.py" \
    --run-root "${run_root}" --output-dir "${run_root}/eawq/metrics" \
    --workers 8 --qp-device cuda:0 --expected-hands 2 \
    --expected-objects-per-hand 10 --expected-sets-per-object "${samples}" \
    --expected-particles-per-set 32 --ranking-only \
    >"${run_root}/logs/eawq_metrics.log" 2>&1
  "${python_path}" "${project_root}/scripts/prepare_eawq_rank_fusion_top1.py" \
    --particle-metrics "${run_root}/eawq/metrics/particle_metrics.csv.gz" \
    --prepared-root "${run_root}/prepared_all" --output-root "${run_root}/eawq" \
    --execution-protocol "${protocol}" --expected-hands 2 \
    --expected-objects-per-hand 10 --expected-sets-per-object "${samples}" \
    --expected-particles-per-set 32 \
    >"${run_root}/logs/eawq_select.log" 2>&1
fi

if (( start_rank <= 3 )); then
  printf 'validating_gpu_physx\n' >"${status_root}/stage"
  gym_worker() {
    local slot="$1" gpu index task output log
    gpu="${gpu_ids[slot]}"
    for ((index=slot; index<${#tasks[@]}; index+=gpu_count)); do
      task="${tasks[$index]}"; task_fields "${task}"
      output="${run_root}/results/${TASK_HAND}/${TASK_OBJECT}.json"
      log="${run_root}/logs/gym_${TASK_HAND}_${TASK_OBJECT}.log"
      if [[ -s "${output}" ]] && "${python_path}" -c \
        'import json,sys; d=json.load(open(sys.argv[1])); raise SystemExit(not (d.get("status")=="complete" and int(d.get("trials",0))==int(sys.argv[2])))' \
        "${output}" "${samples}"; then
        continue
      fi
      CUDA_VISIBLE_DEVICES="${gpu}" "${isaac_runner}" \
        "${project_root}/scripts/validate_native_shadowhand_isaacgym_oldparams_dro.py" \
        --prepared "${run_root}/eawq/prepared/${TASK_HAND}/${TASK_OBJECT}.json" \
        --output "${output}" --gendex-root "${mck_root}/GenDexGrasp" \
        --native-hand-root "${TASK_HAND_ROOT}" --native-hand-urdf "${TASK_HAND_URDF}" \
        --native-hand-urdf-is-extended --only-object "${TASK_OBJECT}" \
        --max-samples-per-object "${samples}" --device-id 0 --progress-every 1 \
        --asset-profile dro --object-source dro \
        --dro-object-root "${project_root}/outputs/local_dro_params_only_coacd/assets/object" \
        --steps-per-second 100 --substeps 2 --closure-steps 100 \
        --direction-seconds 1.0 --direction-order cedex --success-mode final \
        --threshold 0.02 --acceleration 0.5 --robot-friction 3 \
        --object-friction 3 --object-density 500 --object-linear-damping -1 \
        --object-angular-damping -1 --joint-stiffness 1000 --joint-damping 200 \
        --joint-armature -1 --joint-velocity -1 --pregrasp-open-fraction 0 \
        --closure-overdrive-fraction 0 --outer-settle-steps 0 \
        --virtual-root-stiffness 1000 --virtual-root-damping 200 \
        --solver-position-iterations 8 --solver-velocity-iterations 0 \
        --contact-offset 0.01 --rest-offset 0 --no-ground >"${log}" 2>&1
    done
  }
  pids=()
  for slot in $(seq 0 $((gpu_count - 1))); do
    gym_worker "${slot}" & pids+=("$!")
  done
  for pid in "${pids[@]}"; do if ! wait "${pid}"; then failed=1; fi; done
  if (( failed )); then printf 'failed_gym\n' >"${status_root}/stage"; exit 1; fi
fi

printf 'reporting\n' >"${status_root}/stage"
"${python_path}" "${project_root}/scripts/summarize_contact_ar_basic_ood10.py" \
  --run-root "${run_root}" --output "${run_root}/summary.json" \
  >"${run_root}/logs/summary.log" 2>&1
printf 'complete\n' >"${status_root}/stage"
if [[ ! -s "${status_root}/completed_at_utc" ]]; then
  date -u +%Y-%m-%dT%H:%M:%SZ >"${status_root}/completed_at_utc"
fi
