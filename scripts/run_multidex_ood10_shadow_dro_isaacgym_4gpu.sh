#!/usr/bin/env bash
# MultiDex ShadowHand OOD10 evaluation in the original D(R,O) Isaac Gym
# protocol: 64 contact sets x 32 FK particles x rank-0, 400 FK steps.
set -euo pipefail

mck_root="${MCK_ROOT:-/inspire/qb-ilm2/project/zhanghanbo/public/mck}"
project_root="${mck_root}/contact_diffusion"
python_path="${mck_root}/miniconda3/envs/contactdiff/bin/python"
config="configs/multigripper_fk_multidex_ood10_shadow_dro_steps400.yaml"
checkpoint="outputs/contact_diffusion_multidex_seen48_success_n235_4x4090/checkpoints/best_val.pt"
manifest="configs/baseline10_remote_manifest.json"
run_root="${RUN_ROOT:-outputs/multidex_ood10_shadow64x32_top1_steps400_dro}"

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

cd "${project_root}"
mkdir -p \
  "${run_root}/candidates/shards" \
  "${run_root}/results/isaacgym" \
  "${run_root}/logs" \
  "${run_root}/status"

generate_shard() {
  local gpu="$1"
  local shard="$2"
  local sample_start="$3"
  local sample_count="$4"
  shift 4
  local object_arguments=()
  local object_id
  for object_id in "$@"; do
    object_arguments+=(--object-id "${object_id}")
  done
  CUDA_VISIBLE_DEVICES="${gpu}" "${python_path}" \
    scripts/infer_local_contactdiffusion_grasp.py \
    --config "${config}" \
    --checkpoint "${checkpoint}" \
    --manifest "${manifest}" \
    --manifest-only \
    "${object_arguments[@]}" \
    --grippers shadow_hand \
    --sample-start "${sample_start}" \
    --samples-per-object "${sample_count}" \
    --particles 32 \
    --optimization-steps 400 \
    --diffusion-steps 50 \
    --fk-initialization enveloping \
    --envelope-side-weight 5.0 \
    --envelope-approach-weight 2.0 \
    --envelope-cosine-margin 0.5 \
    --selection-min-envelope-cosine 0.5 \
    --selection-min-approach-cosine 0.8 \
    --preferred-root-direction 0 0 1 \
    --selection-max-penetration 0.007 \
    --top-k 1 \
    --device cuda:0 \
    --seed 20260808 \
    --resume \
    --output "${run_root}/candidates/shards/${shard}.json"
}

wait_all() {
  local failed=0
  local pid
  for pid in "$@"; do
    if ! wait "${pid}"; then
      failed=1
    fi
  done
  (( failed == 0 ))
}

printf 'generating_16_shards\n' > "${run_root}/status/supervisor.status"
pids=()
generation_inputs=()
for gpu in 0 1 2 3; do
  case "${gpu}" in
    0) gpu_objects=("${objects[@]:0:3}") ;;
    1) gpu_objects=("${objects[@]:3:3}") ;;
    2) gpu_objects=("${objects[@]:6:2}") ;;
    3) gpu_objects=("${objects[@]:8:2}") ;;
  esac
  for range in 0 1 2 3; do
    sample_start=$((range * 16))
    shard="gpu${gpu}_range${range}"
    generation_inputs+=(--input "${run_root}/candidates/shards/${shard}.json")
    generate_shard \
      "${gpu}" "${shard}" "${sample_start}" 16 "${gpu_objects[@]}" \
      > "${run_root}/logs/generation_${shard}.log" 2>&1 &
    pids+=("$!")
  done
done
if ! wait_all "${pids[@]}"; then
  printf 'failed_generation\n' > "${run_root}/status/supervisor.status"
  exit 1
fi

printf 'merging\n' > "${run_root}/status/supervisor.status"
merge_objects=()
for object_id in "${objects[@]}"; do
  merge_objects+=(--object-id "${object_id}")
done
"${python_path}" scripts/merge_contactdiffusion_generation_shards.py \
  "${generation_inputs[@]}" \
  --output "${run_root}/candidates/shadow.json" \
  "${merge_objects[@]}" \
  --gripper shadow_hand \
  --samples-per-object 64 \
  --particles 32 \
  --optimization-steps 400 \
  > "${run_root}/logs/merge.log" 2>&1

printf 'validating_dro_isaacgym\n' > "${run_root}/status/supervisor.status"
if ! MCK_ROOT="${mck_root}" \
  RUN_ROOT="${project_root}/${run_root}/results/isaacgym" \
  CANDIDATES="${project_root}/${run_root}/candidates/shadow.json" \
  ROBOT_NAME=shadowhand \
  bash scripts/run_contactdiff_ood10_dro_isaacgym_4gpu.sh; then
  printf 'failed_isaacgym\n' > "${run_root}/status/supervisor.status"
  exit 1
fi

printf 'complete\n' > "${run_root}/status/supervisor.status"
