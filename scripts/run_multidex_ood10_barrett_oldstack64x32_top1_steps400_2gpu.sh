#!/usr/bin/env bash
# MultiDex 45k OOD10 Barrett with the legacy dex-urdf hand/candidate convention
# and the native GenDex protocol. Gym consumes the same prepared samples and
# explicitly matches the Sim-visible control and material parameters.
set -euo pipefail

mck_root="${MCK_ROOT:-/inspire/qb-ilm2/project/zhanghanbo/public/mck}"
project_root="${mck_root}/contact_diffusion"
cedex_root="${mck_root}/CEDex-Grasp"
gendex_root="${mck_root}/GenDexGrasp"
python_path="${mck_root}/miniconda3/envs/contactdiff/bin/python"
gym_python="${mck_root}/IsaacGym/.conda-env/bin/python"
isaac_python="${mck_root}/miniconda3/envs/contactdiff_isaac601/bin/python"
config="configs/multigripper_fk_multidex_ood10_barrett_old_steps400.yaml"
checkpoint="outputs/contact_diffusion_multidex_seen48_success_n235_4x4090/checkpoints/best_val.pt"
manifest="configs/baseline10_remote_manifest.json"
run_root="${RUN_ROOT:-outputs/multidex_ood10_barrett_oldstack64x32_top1_steps400_aligned}"

objects=(
  contactdb_apple contactdb_camera contactdb_cylinder_medium
  contactdb_door_knob contactdb_rubber_duck contactdb_water_bottle
  ycb_055_baseball ycb_016_pear
  ycb_010_potted_meat_can ycb_005_tomato_soup_can
)

cd "${project_root}"
mkdir -p \
  "${run_root}/candidates/shards" "${run_root}/prepared" \
  "${run_root}/results/isaacgym" "${run_root}/results/isaacsim" \
  "${run_root}/logs" "${run_root}/status" "${run_root}/summary"

generate_range() {
  local gpu="$1" shard="$2" sample_start="$3" sample_count="$4"
  local object_args=() object_id
  for object_id in "${objects[@]}"; do
    object_args+=(--object-id "${object_id}")
  done
  local mish_fallback=0
  if [[ "${gpu}" == "1" ]]; then mish_fallback=1; fi
  CONTACTDIFF_MISH_COMPOSITE="${mish_fallback}" \
  CUDA_VISIBLE_DEVICES="${gpu}" "${python_path}" \
    scripts/infer_local_contactdiffusion_grasp.py \
      --config "${config}" --checkpoint "${checkpoint}" \
      --manifest "${manifest}" --manifest-only \
      "${object_args[@]}" --grippers Barrett \
      --sample-start "${sample_start}" --samples-per-object "${sample_count}" \
      --particles 32 --optimization-steps 400 \
      --diffusion-steps 50 --fk-initialization enveloping \
      --envelope-side-weight 5.0 --envelope-approach-weight 2.0 \
      --envelope-cosine-margin 0.5 \
      --selection-min-envelope-cosine 0.5 \
      --selection-min-approach-cosine 0.8 \
      --preferred-root-direction 0 0 1 \
      --selection-max-penetration 0.007 --top-k 1 \
      --device cuda:0 --seed 20260810 --resume \
      --output "${run_root}/candidates/shards/${shard}.json"
}

wait_all() {
  local failed=0 pid
  for pid in "$@"; do wait "${pid}" || failed=1; done
  (( failed == 0 ))
}

printf 'generating_old_native_candidates\n' >"${run_root}/status/pipeline.status"
generation_inputs=()
generation_pids=()
for range_index in 0 1 2 3 4 5 6 7; do
  gpu=$((range_index % 2))
  sample_start=$((range_index * 8))
  shard="range${range_index}"
  generation_inputs+=(--input "${run_root}/candidates/shards/${shard}.json")
  generate_range "${gpu}" "${shard}" "${sample_start}" 8 \
    >"${run_root}/logs/generation_${shard}.log" 2>&1 &
  generation_pids+=("$!")
done
wait_all "${generation_pids[@]}"

merge_args=()
for object_id in "${objects[@]}"; do merge_args+=(--object-id "${object_id}"); done
printf 'merging\n' >"${run_root}/status/pipeline.status"
"${python_path}" scripts/merge_contactdiffusion_generation_shards.py \
  "${generation_inputs[@]}" \
  --output "${run_root}/candidates/barrett.json" \
  "${merge_args[@]}" --gripper Barrett --samples-per-object 64 \
  --particles 32 --optimization-steps 400 \
  >"${run_root}/logs/merge.log" 2>&1

printf 'preparing_same_samples\n' >"${run_root}/status/pipeline.status"
"${python_path}" scripts/prepare_gendex_ood10_matched_isaacsim.py \
  --hand barrett --barrett "${run_root}/candidates/barrett.json" \
  --config "${config}" --expected-optimization-steps 400 \
  --barrett-validator-hand contactdiff_barrett \
  --output-dir "${run_root}/prepared" \
  >"${run_root}/logs/preparation.log" 2>&1

gym_object() {
  local object_name="$1" gpu="$2"
  PYTHONNOUSERSITE=1 \
  TORCH_EXTENSIONS_DIR="${mck_root}/IsaacGym/.torch_extensions" \
  MAX_JOBS=4 CUDA_VISIBLE_DEVICES="${gpu}" \
  LD_LIBRARY_PATH="${mck_root}/IsaacGym/.conda-env/lib:${LD_LIBRARY_PATH:-}" \
  "${gym_python}" scripts/validate_native_shadowhand_isaacgym_oldparams_dro.py \
    --prepared "${run_root}/prepared/barrett.json" \
    --output "${run_root}/results/isaacgym/${object_name}.json" \
    --gendex-root "${gendex_root}" \
    --native-hand-root "${project_root}/remote_assets/dex-urdf/robots/hands/barrett_hand" \
    --native-hand-urdf bhand_model.urdf --object-source prepared \
    --device-id 0 --only-object "${object_name}" --resume \
    --steps-per-second 60 --substeps 2 --closure-steps 200 \
    --direction-seconds 0.8333333333333334 --direction-order gendex \
    --success-mode per_direction --threshold 0.02 --acceleration 0.5 \
    --robot-friction 10 --object-friction 10 --object-density 10000 \
    --object-linear-damping 10 --object-angular-damping 100 \
    --joint-stiffness 400 --joint-damping 400 --joint-armature 0.001 \
    --joint-velocity -1 --virtual-root-stiffness 1000000 \
    --virtual-root-damping 100000 --solver-position-iterations 4 \
    --solver-velocity-iterations 0 --contact-offset 0.01 --rest-offset 0 \
    --no-ground
}

printf 'validating_isaacgym_aligned\n' >"${run_root}/status/pipeline.status"
for start in 0 2 4 6 8; do
  gym_object "${objects[start]}" 0 \
    >"${run_root}/logs/gym_${objects[start]}.log" 2>&1 & gym0=$!
  gym_object "${objects[start+1]}" 1 \
    >"${run_root}/logs/gym_${objects[start+1]}.log" 2>&1 & gym1=$!
  wait_all "${gym0}" "${gym1}"
done

sim_half() {
  local shard="$1" gpu="$2"
  shift 2
  local object_args=() object_name
  for object_name in "$@"; do object_args+=(--only-object "${object_name}"); done
  local state="${run_root}/isaacsim/${shard}/state"
  local cache="${run_root}/isaacsim/${shard}/cache"
  local urdf_cache="${run_root}/isaacsim/${shard}/urdf_cache"
  mkdir -p "${state}" "${cache}" "${urdf_cache}"
  CONTACTDIFF_ISAAC_STATE_ROOT="${state}" \
  CONTACTDIFF_ISAAC_CACHE_ROOT="${cache}" \
  CONTACTDIFF_OMNI_PORTABLE_ROOT="${state}/portable" \
  OMNI_KIT_ACCEPT_EULA=YES CONTACTDIFF_SKIP_VIEWPORT_WAIT=1 \
  CONTACTDIFF_HARD_EXIT_ISAAC=1 \
  bash scripts/run_contactdiff_isaacsim_601.sh \
    "${cedex_root}/scripts/validate_isaacsim_six_direction.py" \
    --prepared "${project_root}/${run_root}/prepared/barrett.json" \
    --output "${project_root}/${run_root}/results/isaacsim/${shard}.json" \
    --urdf-cache-dir "${project_root}/${urdf_cache}" --resume \
    --gpu-id "${gpu}" --progress-every 16 --record-physx-contacts \
    --steps-per-second 60 --substeps 2 --grasp-steps 200 \
    --direction-seconds 0.8333333333333334 --acceleration 0.5 \
    --threshold 0.02 --success-displacement-mode per_direction \
    --direction-order gendex --robot-friction 10 --object-friction 10 \
    --object-density 10000 --object-linear-damping 10 \
    --object-angular-damping 100 --joint-stiffness 400 \
    --joint-damping 400 --virtual-root-stiffness 1000000 \
    --virtual-root-damping 100000 --joint-armature 0.001 \
    --solver-position-iterations 4 --solver-velocity-iterations 0 \
    "${object_args[@]}"
}

printf 'validating_isaacsim_aligned\n' >"${run_root}/status/pipeline.status"
sim_half shard0 0 "${objects[@]:0:5}" \
  >"${run_root}/logs/sim_shard0.log" 2>&1 & sim0=$!
sim_half shard1 1 "${objects[@]:5:5}" \
  >"${run_root}/logs/sim_shard1.log" 2>&1 & sim1=$!
wait_all "${sim0}" "${sim1}"

printf 'summarizing\n' >"${run_root}/status/pipeline.status"
"${isaac_python}" scripts/summarize_barrett_oldstack_aligned_gym_sim.py \
  --run-root "${run_root}" \
  >"${run_root}/logs/summary.log" 2>&1
printf 'complete\n' >"${run_root}/status/pipeline.status"
echo "Complete: ${project_root}/${run_root}"
