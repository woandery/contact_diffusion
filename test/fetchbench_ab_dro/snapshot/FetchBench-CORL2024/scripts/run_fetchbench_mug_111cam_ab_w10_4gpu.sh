#!/usr/bin/env bash
set -euo pipefail

# Mug, formal 111-camera sampling. Only views with >=5% observable-mesh
# coverage are processed. For every legal view:
#   A = segmented target partial point cloud from that view
#   B = SAM3D proxy reconstructed from the same view
# Both branches use that view's SAM3D proxy for object geometry/FK penetration
# and that view's scene partial point cloud for w=10 environment refinement.
# The contact-set and FK-particle budgets are configurable; every contact set
# retains one pose after FK and environment refinement.

project_root="${CONTACTDIFF_ROOT:-/inspire/qb-ilm2/project/zhanghanbo/public/mck/ContactDiffusion}"
fetchbench_root="${FETCHBENCH_ROOT:-/inspire/qb-ilm2/project/zhanghanbo/public/mck/FetchBench-CORL2024}"
sam3d_root="${SAM3D_ROOT:-/inspire/qb-ilm2/project/zhanghanbo/public/mck/sam-3d-objects}"
run_root="${RUN_ROOT:-${project_root}/outputs/fetchbench_mug_111cam_ab_w10_20260907}"
log_root="${LOG_ROOT:-${run_root}/logs}"
status_root="${STATUS_ROOT:-${run_root}/status}"
source_root="${SOURCE_ROOT:-${run_root}}"
condition_source_root="${CONDITION_SOURCE_ROOT:-${source_root}}"
object_prefix="${OBJECT_PREFIX:-mug}"
case_workers_per_gpu="${CASE_WORKERS_PER_GPU:-3}"
gpu_count="${GPU_COUNT:-4}"
gpu_offset="${GPU_OFFSET:-0}"
reuse_sam3d="${REUSE_SAM3D:-false}"
max_views="${MAX_VIEWS:-0}"
hand_filter="${HAND_FILTER:-all}"
contact_sets="${CONTACT_SETS:-32}"
particles="${PARTICLES:-32}"
fk_steps="${FK_STEPS:-400}"
env_steps="${ENV_STEPS:-400}"
env_learning_rate="${ENV_LEARNING_RATE:-0.002}"
python_bin="${CONTACTDIFF_PYTHON:-/inspire/qb-ilm2/project/zhanghanbo/public/mck/miniconda3/envs/contactdiff/bin/python}"
sam3d_python="${SAM3D_PYTHON:-/inspire/qb-ilm2/project/zhanghanbo/public/mck/miniconda3/envs/sam3d-objects/bin/python}"
sam3d_env="${SAM3D_CONDA_PREFIX:-${sam3d_python%/bin/python}}"
checkpoint_config="${SAM3D_CHECKPOINT_CONFIG:-${sam3d_root}/checkpoints/hf-download/checkpoints/pipeline.yaml}"
checkpoint_a="${CONTACTDIFF_CHECKPOINT_A:-${project_root}/outputs/contact_ar_success_k128_mixedfullpartial_normfull_from8k_lr2e4_gb768_56k_4x4090/model/checkpoints/step_00056000.pt}"
checkpoint_b="${CONTACTDIFF_CHECKPOINT_B:-${checkpoint_a}}"
refine_adapter="${REFINE_ADAPTER:-scripts/refine_contactdiffusion_environment_visible_scene.py}"
visibility_root="${source_root}/visibility"
sam3d_cache_root="${source_root}/sam3d"
condition_cache_root="${condition_source_root}/sam3d"
legal_json="${visibility_root}/legal_partial_views.json"

config_for_hand() {
  if [[ "$1" == "barrett" ]]; then
    printf '%s' "${CONFIG_BARRETT:-${project_root}/configs/multigripper_fk_multidex_ood10_barrett_steps400_palm0_v4.yaml}"
  else
    printf '%s' "${CONFIG_SHADOW:-${project_root}/configs/multigripper_fk_multidex_ood10_shadow_dro_steps400_palm0_v4.yaml}"
  fi
}

model_hand() {
  [[ "$1" == "barrett" ]] && printf 'Barrett' || printf 'shadow_hand'
}

hand_offset() {
  [[ "$1" == "barrett" ]] && printf '0' || printf '1'
}

json_has_records() {
  local path="$1" record_count="$2" candidate_count="$3"
  "${python_bin}" -c \
    'import json,sys; d=json.load(open(sys.argv[1])); r=int(sys.argv[2]); n=int(sys.argv[3]); assert len(d.get("records", [])) == r; assert all(len(x["fk"]["candidates"]) == n for x in d["records"])' \
    "${path}" "${record_count}" "${candidate_count}" 2>/dev/null
}

mkdir -p "${log_root}" "${status_root}" "${run_root}/sam3d" "${run_root}/views"
cd "${project_root}"
test -s "${checkpoint_a}"
test -s "${checkpoint_b}"
test -s "${checkpoint_config}"
test -s "${legal_json}"
test -x "${python_bin}"
test -x "${sam3d_python}"
if (( gpu_count < 1 )); then
  echo "GPU_COUNT must be at least 1" >&2
  exit 2
fi
if [[ "${hand_filter}" != "all" && "${hand_filter}" != "barrett" && "${hand_filter}" != "shadowhand" ]]; then
  echo "HAND_FILTER must be all, barrett, or shadowhand" >&2
  exit 2
fi

mapfile -t view_ids < <("${python_bin}" - "${legal_json}" <<'PY'
import json
import sys
from pathlib import Path

data = json.loads(Path(sys.argv[1]).read_text())
for row in data["views"]:
    print(Path(row["rgbd_capture_dir"]).name)
PY
)
if (( max_views > 0 && max_views < ${#view_ids[@]} )); then
  view_ids=("${view_ids[@]:0:max_views}")
fi
(( ${#view_ids[@]} > 0 ))

reconstruct_view() {
  local gpu="$1" view_id="$2" capture output geometry_output
  capture="${visibility_root}/rgbd_views/${view_id}"
  output="${condition_cache_root}/${view_id}"
  geometry_output="${sam3d_cache_root}/${view_id}"
  mkdir -p "${output}" "${geometry_output}"

  if [[ "${reuse_sam3d}" == "true" ]]; then
    test -s "${output}/camera_partial_sam3d_centered.npy"
    test -s "${output}/sam3d_fused_centered.npy"
    test -s "${geometry_output}/sam3d_fused_centered.json"
    test -s "${geometry_output}/sam3d_fused_robot_base.npy"
    return 0
  fi
  test -s "${capture}/metadata.json"
  test -s "${capture}/camera_00_rgb.png"
  test -s "${capture}/camera_00_target_mask.png"
  test -s "${capture}/camera_00_pointmap_robot_base.npy"
  test -s "${capture}/scene_partial_robot_base.npy"

  if "${sam3d_python}" -c \
    'import json,sys; d=json.load(open(sys.argv[1])); assert d["camera_index"] == 0; assert d.get("observed_source") == "selected_camera"' \
    "${output}/sam3d_fused_centered.json" 2>/dev/null \
    && test -s "${output}/camera_partial_sam3d_centered.npy" \
    && test -s "${output}/sam3d_fused_centered.npy" \
    && test -s "${output}/sam3d_fused_robot_base.npy"; then
    return 0
  fi

  (
    export CONDA_PREFIX="${sam3d_env}"
    export LD_LIBRARY_PATH="${sam3d_env}/lib:$(dirname "${python_bin}")/../lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
    export CUDA_HOME="${sam3d_env}"
    export PATH="${sam3d_env}/bin:${PATH}"
    export CPATH="${sam3d_env}/targets/x86_64-linux/include${CPATH:+:${CPATH}}"
    export LIBRARY_PATH="${sam3d_env}/targets/x86_64-linux/lib${LIBRARY_PATH:+:${LIBRARY_PATH}}"
    CUDA_VISIBLE_DEVICES="${gpu}" "${sam3d_python}" \
      "${fetchbench_root}/scripts/reconstruct_fetchbench_sam3d.py" \
      --capture-dir "${capture}" --sam3d-root "${sam3d_root}" \
      --checkpoint-config "${checkpoint_config}" --output-dir "${output}" \
      --camera-index 0 --observed-source selected_camera \
      --points 8192 --observed-fraction 0.25 --opacity-threshold 0.10 \
      --seed 20260905
  ) >"${log_root}/reconstruct_${view_id}.log" 2>&1 || return 1

  test -s "${output}/camera_partial_sam3d_centered.npy"
  test -s "${output}/sam3d_fused_centered.npy"
  test -s "${output}/sam3d_fused_robot_base.npy"
}

generate_view_case() {
  local gpu="$1" view_index="$2" view_id="$3" condition="$4" hand="$5"
  local capture samdir condition_samdir root config model adapter output transformed refined selected
  local -a condition_args
  capture="${visibility_root}/rgbd_views/${view_id}"
  samdir="${sam3d_cache_root}/${view_id}"
  condition_samdir="${condition_cache_root}/${view_id}"
  root="${run_root}/views/${view_id}/${condition}"
  config="$(config_for_hand "${hand}")"
  model="$(model_hand "${hand}")"
  output="${root}/raw_object/${object_prefix}_${hand}.json"
  transformed="${root}/candidates_robot/${object_prefix}_${hand}.json"
  refined="${root}/refined_w10/${object_prefix}_${hand}.json"
  selected="${root}/selected_w10/${object_prefix}_${hand}.json"
  mkdir -p "${root}/raw_object" "${root}/candidates_robot" \
    "${root}/refined_w10" "${root}/selected_w10"

  if ! json_has_records "${output}" "${contact_sets}" "${particles}"; then
    if [[ "${condition}" == "A" ]]; then
      adapter="scripts/infer_contactdiffusion_camera_partial_v5_six.py"
      condition_args=(
        --partial-object-pc "${condition_samdir}/camera_partial_sam3d_centered.npy"
        --partial-points 2048 --partial-sampling-seed 20260905
      )
    else
      adapter="scripts/infer_contactdiffusion_explicit_condition_v6.py"
      condition_args=(
        --condition-object-pc "${condition_samdir}/sam3d_fused_centered.npy"
        --condition-points 2048 --condition-sampling-seed 20260905
        --condition-mode sam3d_singleview_complete_proxy
      )
    fi
    CUDA_VISIBLE_DEVICES="${gpu}" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      "${python_bin}" "${adapter}" "${condition_args[@]}" \
      --config "${config}" --grippers "${model}" \
      --hand-index-offset "$(hand_offset "${hand}")" \
      --checkpoint "$([[ "${condition}" == A ]] && printf '%s' "${checkpoint_a}" || printf '%s' "${checkpoint_b}")" \
      --object-pc "${condition_samdir}/sam3d_fused_centered.npy" \
      --object-id "${object_prefix}_${view_id}" --samples-per-object "${contact_sets}" --sample-start 0 \
      --inference-object-observation full --autoregressive-fk-target nearest_2048 \
      --particles "${particles}" --optimization-steps "${fk_steps}" --diffusion-steps 50 \
      --fk-initialization enveloping --envelope-approach-weight 2.0 \
      --selection-min-envelope-cosine 0.5 --selection-min-approach-cosine 0.8 \
      --preferred-root-direction 0 0 1 --selection-max-penetration 0.007 \
      --disable-palm-selection-gate \
      --top-k "${contact_sets}" --device cuda:0 --seed 20260808 \
      --object-index-offset "$((view_index + ${VIEW_INDEX_OFFSET:-0}))" --output "${output}" \
      >"${log_root}/generate_${view_id}_${condition}_${hand}.log" 2>&1 || return 1
  fi
  json_has_records "${output}" "${contact_sets}" "${particles}" || return 1

  if ! json_has_records "${transformed}" "${contact_sets}" "${particles}"; then
    "${python_bin}" scripts/transform_contactdiffusion_candidates.py \
      --input "${output}" \
      --transform-metadata "${samdir}/sam3d_fused_centered.json" \
      --object-pc "${samdir}/sam3d_fused_robot_base.npy" \
      --source-frame object_sam3d_surface_centered --target-frame robot_base \
      --output "${transformed}" \
      >"${log_root}/transform_${view_id}_${condition}_${hand}.log" 2>&1 || return 1
  fi
  json_has_records "${transformed}" "${contact_sets}" "${particles}" || return 1

  if ! json_has_records "${refined}" "${contact_sets}" "${particles}"; then
    CUDA_VISIBLE_DEVICES="${gpu}" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      "${python_bin}" "${refine_adapter}" \
      --candidates "${transformed}" --config "${config}" \
      --object-pc "${samdir}/sam3d_fused_robot_base.npy" \
      --scene-pc "${capture}/scene_partial_robot_base.npy" \
      --output "${refined}" --steps "${env_steps}" --learning-rate "${env_learning_rate}" \
      --environment-weight 10 --environment-clearance 0.005 \
      --selection-max-environment-violation 0.0001 \
      --environment-feasibility-mode soft --selection-rank-mode normalized_constraints \
      --environment-cvar-fraction 0.10 --environment-closure-sweep-samples 4 \
      --closure-outer-fraction 0.10 --closure-inner-fraction 0.20 \
      --object-constraint-weight 4 --environment-constraint-weight 4 \
      --contact-constraint-weight 10 --contact-constraint-limit 0.010 \
      --require-contact-feasibility --restore-best-constraint-state \
      --scene-voxel-size 0.008 --max-scene-points 4096 --device cuda:0 \
      >"${log_root}/refine_${view_id}_${condition}_${hand}.log" 2>&1 || return 1
  fi
  json_has_records "${refined}" "${contact_sets}" "${particles}" || return 1

  if ! json_has_records "${selected}" "${contact_sets}" 1; then
    "${python_bin}" scripts/select_fullpc_environment_weight_ensemble_constraints.py \
      --inputs "${refined}" --output "${selected}" \
      >"${log_root}/select_${view_id}_${condition}_${hand}.log" 2>&1 || return 1
  fi
  json_has_records "${selected}" "${contact_sets}" 1 || return 1
}

reconstruct_worker() {
  local slot="$1" gpu=$((slot + gpu_offset)) index view_id failures=0
  for ((index=slot; index<${#view_ids[@]}; index+=gpu_count)); do
    view_id="${view_ids[index]}"
    printf 'reconstructing gpu=%s view_index=%s\n' "${gpu}" "${index}" \
      >"${status_root}/${view_id}.txt"
    if reconstruct_view "${gpu}" "${view_id}"; then
      printf 'reconstructed gpu=%s view_index=%s\n' "${gpu}" "${index}" \
        >"${status_root}/${view_id}.txt"
    else
      printf 'reconstruction_failed gpu=%s view_index=%s\n' "${gpu}" "${index}" \
        >"${status_root}/${view_id}.txt"
      failures=$((failures + 1))
    fi
  done
  printf 'complete gpu=%s failures=%s\n' "${gpu}" "${failures}" \
    >"${status_root}/reconstruct_worker${slot}.txt"
  (( failures == 0 ))
}

case_worker() {
  local worker="$1" gpu="$2" total_workers="$3" slot case_index view_index rem
  local view_id condition hand failures=0
  # Iterate a condition/hand-major ordering. The earlier view-major ordering,
  # combined with 12 workers, pinned all ShadowHand cases to GPUs 1/3 and all
  # Barrett cases to GPUs 0/2 because both strides were divisible by four.
  # This permutation preserves every case exactly once while distributing each
  # hand/condition block evenly across all configured GPUs.
  for ((slot=worker; slot<${#view_ids[@]}*4; slot+=total_workers)); do
    rem=$((slot / ${#view_ids[@]}))
    view_index=$((slot % ${#view_ids[@]}))
    case_index=$((view_index * 4 + rem))
    view_id="${view_ids[view_index]}"
    if (( rem < 2 )); then condition=A; else condition=B; fi
    if (( rem % 2 == 0 )); then hand=barrett; else hand=shadowhand; fi
    if [[ "${hand_filter}" != "all" && "${hand}" != "${hand_filter}" ]]; then
      continue
    fi
    printf 'running gpu=%s worker=%s\n' "${gpu}" "${worker}" \
      >"${status_root}/${view_id}_${condition}_${hand}.txt"
    if generate_view_case "${gpu}" "${view_index}" "${view_id}" "${condition}" "${hand}"; then
      printf 'complete gpu=%s worker=%s\n' "${gpu}" "${worker}" \
        >"${status_root}/${view_id}_${condition}_${hand}.txt"
    else
      printf 'failed gpu=%s worker=%s\n' "${gpu}" "${worker}" \
        >"${status_root}/${view_id}_${condition}_${hand}.txt"
      failures=$((failures + 1))
    fi
  done
  printf 'complete worker=%s gpu=%s failures=%s\n' "${worker}" "${gpu}" "${failures}" \
    >"${status_root}/case_worker${worker}.txt"
  (( failures == 0 ))
}

if [[ "${hand_filter}" == "all" ]]; then hand_count=2; else hand_count=1; fi
total_final_poses=$(( ${#view_ids[@]} * 2 * hand_count * contact_sets ))
printf 'running hand=%s legal_views=%s total_final_poses=%s\n' \
  "${hand_filter}" "${#view_ids[@]}" "${total_final_poses}" \
  >"${status_root}/generation_master.txt"

pids=()
for ((slot=0; slot<gpu_count; slot++)); do
  reconstruct_worker "${slot}" >"${log_root}/reconstruct_worker${slot}.log" 2>&1 &
  pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do
  wait "${pid}" || failed=1
done
if (( failed )); then
  printf 'reconstruction_failed\n' >"${status_root}/generation_master.txt"
  exit 1
fi

if [[ "${RECONSTRUCT_ONLY:-false}" == "true" ]]; then
  printf 'reconstruction_complete\n' >"${status_root}/generation_master.txt"
  exit 0
fi
total_case_workers=$((gpu_count * case_workers_per_gpu))
printf 'generating hand=%s legal_views=%s case_workers=%s total_final_poses=%s\n' \
  "${hand_filter}" "${#view_ids[@]}" "${total_case_workers}" "${total_final_poses}" \
  >"${status_root}/generation_master.txt"
pids=()
for ((worker=0; worker<total_case_workers; worker++)); do
  gpu=$((worker % gpu_count + gpu_offset))
  case_worker "${worker}" "${gpu}" "${total_case_workers}" \
    >"${log_root}/case_worker${worker}.log" 2>&1 &
  pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do
  wait "${pid}" || failed=1
done
if (( failed )); then
  printf 'generation_failed\n' >"${status_root}/generation_master.txt"
  exit 1
fi

printf 'complete hand=%s legal_views=%s total_final_poses=%s\n' \
  "${hand_filter}" "${#view_ids[@]}" "${total_final_poses}" \
  >"${status_root}/generation_master.txt"
echo "Mug 111-camera A/B ${hand_filter} generation complete: ${run_root}"
