#!/usr/bin/env bash
# Generate MultiDex ContactDiffusion Barrett OOD10 64x32 top1 candidates with
# 400 FK steps in the exact GenDex/D(R,O) Barrett convention.  Stage the
# candidates through the existing GenDex A/B adapter, validate them with the
# original D(R,O) Isaac Gym protocol, and replay the same grasps in Isaac Sim.
set -euo pipefail

mck_root="${MCK_ROOT:-/inspire/qb-ilm2/project/zhanghanbo/public/mck}"
project_root="${mck_root}/contact_diffusion"
cedex_root="${mck_root}/CEDex-Grasp"
dro_root="${mck_root}/dro_grasp_reproduction/DRO-Grasp"
python_path="${mck_root}/miniconda3/envs/contactdiff/bin/python"
isaac_python="${mck_root}/IsaacGym/.conda-env/bin/python"
config="configs/multigripper_fk_multidex_ood10_barrett_steps400.yaml"
checkpoint="outputs/contact_diffusion_multidex_seen48_success_n235_4x4090/checkpoints/best_val.pt"
manifest="configs/baseline10_remote_manifest.json"
old_sim_root="outputs/gendex_ood10_barrett_fk_step_ablation_historical_isaacsim601/steps400/results"
run_root="${RUN_ROOT:-outputs/multidex_ood10_barrett64x32_top1_steps400_dro_gendex}"
dro_validator="${dro_root}/scripts/validate_contactdiff_ood10_dro_isaacgym.py"
sim_validator="${cedex_root}/scripts/validate_isaacsim_six_direction.py"

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
  "${run_root}/prepared" \
  "${run_root}/results/isaacgym_new" \
  "${run_root}/results/isaacsim_new" \
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
    --grippers Barrett \
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
  --output "${run_root}/candidates/barrett.json" \
  "${merge_objects[@]}" \
  --gripper Barrett \
  --samples-per-object 64 \
  --particles 32 \
  --optimization-steps 400 \
  > "${run_root}/logs/merge.log" 2>&1

printf 'staging_dro_gendex\n' > "${run_root}/status/supervisor.status"
"${python_path}" scripts/stage_barrett_dro_ab_gendex.py \
  --barrett "${run_root}/candidates/barrett.json" \
  --optimization-steps 400 \
  --dataset-suffix MultiDex45k-DROAB \
  > "${run_root}/logs/stage_dro_gendex.log" 2>&1

printf 'preparing\n' > "${run_root}/status/supervisor.status"
"${python_path}" scripts/prepare_gendex_ood10_matched_isaacsim.py \
  --hand barrett \
  --barrett "${run_root}/candidates/barrett.json" \
  --config "${config}" \
  --expected-optimization-steps 400 \
  --barrett-validator-hand gendex_barrett \
  --output-dir "${run_root}/prepared" \
  > "${run_root}/logs/preparation.log" 2>&1

printf 'validating_isaacgym_new\n' > "${run_root}/status/supervisor.status"
if ! MCK_ROOT="${mck_root}" \
  RUN_ROOT="${project_root}/${run_root}/results/isaacgym_new" \
  CANDIDATES="${project_root}/${run_root}/candidates/barrett.json" \
  bash scripts/run_contactdiff_ood10_dro_isaacgym_4gpu.sh; then
  printf 'failed_isaacgym\n' > "${run_root}/status/supervisor.status"
  exit 1
fi

historical_protocol=(
  --steps-per-second 100
  --substeps 2
  --grasp-steps 100
  --direction-seconds 1.0
  --acceleration 0.5
  --threshold 0.02
  --success-displacement-mode final
  --direction-order cedex
  --robot-friction 3
  --object-friction 3
  --object-density 500
  --object-linear-damping 0
  --object-angular-damping 0
  --joint-stiffness 1000
  --joint-damping 50
  --virtual-root-stiffness 1000000
  --virtual-root-damping 100000
  --joint-armature 0.001
  --solver-position-iterations 8
  --solver-velocity-iterations 0
)

run_sim_shard() {
  local shard="$1"
  local gpu="$2"
  shift 2
  local state="${run_root}/isaacsim/shard${shard}/state"
  local cache="${run_root}/isaacsim/shard${shard}/cache"
  local urdf_cache="${run_root}/isaacsim/shard${shard}/urdf_cache"
  local object_arguments=()
  local object_id
  for object_id in "$@"; do
    object_arguments+=(--only-object "${object_id}")
  done
  mkdir -p "${state}" "${cache}" "${urdf_cache}"
  printf 'running\n' > "${run_root}/status/isaacsim_shard${shard}.status"
  CONTACTDIFF_ISAAC_STATE_ROOT="${state}" \
  CONTACTDIFF_ISAAC_CACHE_ROOT="${cache}" \
  CONTACTDIFF_OMNI_PORTABLE_ROOT="${state}/portable" \
  OMNI_KIT_ACCEPT_EULA=YES \
  CONTACTDIFF_SKIP_VIEWPORT_WAIT=1 \
  CONTACTDIFF_HARD_EXIT_ISAAC=1 \
  bash scripts/run_contactdiff_isaacsim_601.sh \
    "${sim_validator}" \
    --prepared "${project_root}/${run_root}/prepared/barrett.json" \
    --output "${project_root}/${run_root}/results/isaacsim_new/shard${shard}.json" \
    --urdf-cache-dir "${project_root}/${urdf_cache}" \
    --resume \
    --gpu-id "${gpu}" \
    --progress-every 32 \
    "${historical_protocol[@]}" \
    "${object_arguments[@]}" \
    > "${run_root}/logs/isaacsim_shard${shard}.log" 2>&1
  printf 'complete\n' > "${run_root}/status/isaacsim_shard${shard}.status"
}

printf 'validating_isaacsim_new\n' > "${run_root}/status/supervisor.status"
printf 'running\n' > "${run_root}/status/isaacsim_new.status"
run_sim_shard 0 0 "${objects[@]:0:3}" & sim0=$!
run_sim_shard 1 1 "${objects[@]:3:3}" & sim1=$!
run_sim_shard 2 2 "${objects[@]:6:2}" & sim2=$!
run_sim_shard 3 3 "${objects[@]:8:2}" & sim3=$!
if ! wait_all "${sim0}" "${sim1}" "${sim2}" "${sim3}"; then
  printf 'failed_isaacsim\n' > "${run_root}/status/supervisor.status"
  exit 1
fi
printf 'complete\n' > "${run_root}/status/isaacsim_new.status"
printf 'complete\n' > "${run_root}/status/supervisor.status"

printf 'Old Sim baseline retained at %s\n' "${old_sim_root}" \
  > "${run_root}/logs/baseline_reference.log"
