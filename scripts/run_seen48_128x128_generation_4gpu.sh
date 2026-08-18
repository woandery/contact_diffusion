#!/usr/bin/env bash
# Generate and prepare both-hand seen48 128-set x 128-particle candidates.
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mck_root="$(cd "${project_root}/.." && pwd)"
python_path="${CONTACTDIFF_PYTHON:-${mck_root}/miniconda3/envs/contactdiff_fk_localalign/bin/python}"
checkpoint="${CONTACTDIFF_CHECKPOINT:-${project_root}/outputs/contact_diffusion_multidex_filtered_barrett_shadow_balanced_n35_4x4090/checkpoints/step_00050000.pt}"
run_root="${CONTACTDIFF_RUN_ROOT:-${project_root}/outputs/balanced50k_seen48_128set_128particle_physx_r1000_both_v1}"
asset_root="${CONTACTDIFF_ASSET_ROOT:-${project_root}/outputs/basic_experiment_balanced_n35_step50k_seen48_top1_o10i20_v1_remote4090/provenance/assets}"
manifest="${asset_root}/manifest.json"
status_root="${run_root}/supervisor"
shards_per_object=4
samples_per_shard=32
generation_workers="${CONTACTDIFF_GENERATION_WORKERS:-12}"

objects=(
  contactdb_alarm_clock contactdb_banana contactdb_binoculars contactdb_cell_phone
  contactdb_cube_large contactdb_cube_medium contactdb_cube_small contactdb_cylinder_large
  contactdb_cylinder_small contactdb_elephant contactdb_flashlight contactdb_hammer
  contactdb_light_bulb contactdb_mouse contactdb_piggy_bank contactdb_ps_controller
  contactdb_pyramid_large contactdb_pyramid_medium contactdb_pyramid_small contactdb_stanford_bunny
  contactdb_stapler contactdb_toothpaste contactdb_torus_large contactdb_torus_medium
  contactdb_torus_small contactdb_train ycb_bleach_cleanser ycb_cracker_box
  ycb_foam_brick ycb_gelatin_box ycb_hammer ycb_lemon
  ycb_master_chef_can ycb_mini_soccer_ball ycb_mustard_bottle ycb_orange
  ycb_peach ycb_pitcher_base ycb_plum ycb_power_drill
  ycb_pudding_box ycb_rubiks_cube ycb_sponge ycb_strawberry
  ycb_sugar_box ycb_toy_airplane ycb_tuna_fish_can ycb_wood_block
)

mkdir -p "${status_root}" "${run_root}/provenance" \
  "${run_root}/candidates/shards" "${run_root}/candidates/barrett" \
  "${run_root}/candidates/shadowhand" "${run_root}/prepared/barrett" \
  "${run_root}/prepared/shadowhand"

"${python_path}" "${project_root}/scripts/prepare_seen48_128x128_configs.py" \
  --run-root "${run_root}" --asset-root "${asset_root}" \
  --checkpoint "${checkpoint}" \
  --barrett-source "${project_root}/configs/multigripper_fk_multidex_ood10_barrett_steps400.yaml" \
  --shadow-source "${project_root}/configs/multigripper_fk_multidex_ood10_shadow_dro_steps400.yaml" \
  >"${status_root}/config.log" 2>&1

barrett_config="${run_root}/provenance/configs/barrett_seen48_128x128_steps400.yaml"
shadow_config="${run_root}/provenance/configs/shadowhand_seen48_128x128_steps400.yaml"
actual_step="$("${python_path}" -c 'import sys,torch; p=torch.load(sys.argv[1],map_location="cpu",weights_only=True); print(int(p.get("step",-1)))' "${checkpoint}")"
if [[ "${actual_step}" != 50000 ]]; then
  printf 'checkpoint step mismatch: %s\n' "${actual_step}" >&2
  exit 2
fi
sha256sum "${checkpoint}" >"${status_root}/checkpoint.sha256"
git -C "${project_root}" rev-parse HEAD >"${status_root}/git_commit" 2>/dev/null || true
date -u +%Y-%m-%dT%H:%M:%SZ >"${status_root}/started_at_utc"

tasks=()
for object_index in "${!objects[@]}"; do
  object_id="${objects[object_index]}"
  for hand in barrett shadowhand; do
    for ((shard=0; shard<shards_per_object; shard++)); do
      tasks+=("${hand}:${object_id}:${object_index}:${shard}")
    done
  done
done

generate_shard() {
  local hand="$1" object_id="$2" object_index="$3" shard="$4" gpu="$5"
  local gripper hand_index config sample_start output
  sample_start=$((shard * samples_per_shard))
  if [[ "${hand}" == barrett ]]; then
    gripper=Barrett; hand_index=0; config="${barrett_config}"
  else
    gripper=shadow_hand; hand_index=1; config="${shadow_config}"
  fi
  output="${run_root}/candidates/shards/${hand}_${object_id}_s$(printf '%02d' "${shard}").json"
  CUDA_VISIBLE_DEVICES="${gpu}" "${python_path}" \
    "${project_root}/scripts/infer_local_contactdiffusion_grasp.py" \
    --config "${config}" --checkpoint "${checkpoint}" \
    --manifest "${manifest}" --manifest-only --object-id "${object_id}" \
    --grippers "${gripper}" --sample-start "${sample_start}" \
    --samples-per-object "${samples_per_shard}" --particles 128 \
    --optimization-steps 400 --diffusion-steps 50 \
    --fk-initialization enveloping --envelope-side-weight 5.0 \
    --envelope-approach-weight 2.0 --envelope-cosine-margin 0.5 \
    --selection-min-envelope-cosine 0.5 \
    --selection-min-approach-cosine 0.8 \
    --preferred-root-direction 0 0 1 \
    --selection-max-penetration 0.007 --top-k 128 \
    --device cuda:0 --seed 20260808 --hand-index-offset "${hand_index}" \
    --object-index-offset "${object_index}" --resume --output "${output}"
}

printf 'generating_384_shards\n' >"${status_root}/status"
generation_worker() {
  local slot="$1" gpu=$((slot % 4)) index task hand object_id object_index shard
  for ((index=slot; index<${#tasks[@]}; index+=generation_workers)); do
    task="${tasks[index]}"; hand="${task%%:*}"; task="${task#*:}"
    object_id="${task%%:*}"; task="${task#*:}"
    object_index="${task%%:*}"; shard="${task##*:}"
    generate_shard "${hand}" "${object_id}" "${object_index}" "${shard}" "${gpu}" \
      >"${status_root}/generate_${hand}_${object_id}_s$(printf '%02d' "${shard}").log" 2>&1
  done
}
pids=()
for ((slot=0; slot<generation_workers; slot++)); do
  generation_worker "${slot}" & pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do if ! wait "${pid}"; then failed=1; fi; done
if ((failed)); then printf 'failed_generation\n' >"${status_root}/status"; exit 1; fi

printf 'merging_96_objects\n' >"${status_root}/status"
for hand in barrett shadowhand; do
  if [[ "${hand}" == barrett ]]; then gripper=Barrett; else gripper=shadow_hand; fi
  for object_id in "${objects[@]}"; do
    inputs=()
    for ((shard=0; shard<shards_per_object; shard++)); do
      inputs+=(--input "${run_root}/candidates/shards/${hand}_${object_id}_s$(printf '%02d' "${shard}").json")
    done
    "${python_path}" "${project_root}/scripts/merge_contactdiffusion_generation_shards.py" \
      "${inputs[@]}" --output "${run_root}/candidates/${hand}/${object_id}.json" \
      --object-id "${object_id}" --gripper "${gripper}" \
      --samples-per-object 128 --particles 128 --optimization-steps 400 \
      >"${status_root}/merge_${hand}_${object_id}.log" 2>&1
  done
done

printf 'preparing_96_all_particle_manifests\n' >"${status_root}/status"
prepare_tasks=()
for hand in barrett shadowhand; do
  for object_id in "${objects[@]}"; do prepare_tasks+=("${hand}:${object_id}"); done
done
prepare_worker() {
  local slot="$1" index task hand object_id gripper config
  for ((index=slot; index<${#prepare_tasks[@]}; index+=8)); do
    task="${prepare_tasks[index]}"; hand="${task%%:*}"; object_id="${task#*:}"
    if [[ "${hand}" == barrett ]]; then gripper=Barrett; config="${barrett_config}";
    else gripper=shadow_hand; config="${shadow_config}"; fi
    "${python_path}" "${project_root}/scripts/prepare_basic_experiment_isaacgym.py" \
      --candidates "${run_root}/candidates/${hand}/${object_id}.json" \
      --config "${config}" --gripper "${gripper}" --candidate-mode all \
      --closure-outer-fraction 0.10 --closure-inner-fraction 0.20 \
      --output "${run_root}/prepared/${hand}/${object_id}.json" \
      >"${status_root}/prepare_${hand}_${object_id}.log" 2>&1
  done
}
pids=()
for slot in $(seq 0 7); do prepare_worker "${slot}" & pids+=("$!"); done
failed=0
for pid in "${pids[@]}"; do if ! wait "${pid}"; then failed=1; fi; done
if ((failed)); then printf 'failed_preparation\n' >"${status_root}/status"; exit 1; fi

printf 'ready_for_physx_stability_r1000\n' >"${status_root}/status"
date -u +%Y-%m-%dT%H:%M:%SZ >"${status_root}/generation_completed_at_utc"
