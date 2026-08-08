#!/usr/bin/env bash
# Replay the completed MultiDex D(R,O)-ShadowHand 64x32 top-1 candidates in
# Isaac Sim, using the same six-direction reporting protocol as Barrett.
set -euo pipefail

mck_root="${MCK_ROOT:-/inspire/qb-ilm2/project/zhanghanbo/public/mck}"
project_root="${mck_root}/contact_diffusion"
cedex_root="${mck_root}/CEDex-Grasp"
dro_shadow_root="${mck_root}/dro_grasp_reproduction/DRO-Grasp/data/data_urdf/robot/shadowhand"
python_path="${mck_root}/miniconda3/envs/contactdiff/bin/python"
config="configs/multigripper_fk_multidex_ood10_shadow_dro_steps400.yaml"
run_root="${RUN_ROOT:-outputs/multidex_ood10_shadow64x32_top1_steps400_dro}"
candidate="${run_root}/candidates/shadow.json"
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
if [[ ! -f "${candidate}" ]]; then
  printf 'Candidate file does not exist: %s\n' "${candidate}" >&2
  exit 2
fi
if [[ ! -f "${dro_shadow_root}/shadow_hand_right_extended.urdf" ]]; then
  printf 'D(R,O) ShadowHand asset is missing: %s\n' "${dro_shadow_root}" >&2
  exit 2
fi
mkdir -p \
  "${run_root}/prepared" \
  "${run_root}/results/isaacsim" \
  "${run_root}/logs" \
  "${run_root}/status" \
  "${cedex_root}/data/urdf"
if [[ ! -e "${cedex_root}/data/urdf/shadowhand" ]]; then
  ln -s "${dro_shadow_root}" "${cedex_root}/data/urdf/shadowhand"
fi

printf 'preparing_isaacsim\n' > "${run_root}/status/isaacsim.status"
"${python_path}" scripts/prepare_gendex_ood10_matched_isaacsim.py \
  --hand shadow \
  --shadow "${candidate}" \
  --config "${config}" \
  --expected-optimization-steps 400 \
  --shadow-validator-hand shadowhand \
  --output-dir "${run_root}/prepared" \
  > "${run_root}/logs/isaacsim_preparation.log" 2>&1

protocol=(
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

run_shard() {
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
    --prepared "${project_root}/${run_root}/prepared/shadow.json" \
    --output "${project_root}/${run_root}/results/isaacsim/shard${shard}.json" \
    --urdf-cache-dir "${project_root}/${urdf_cache}" \
    --resume \
    --gpu-id "${gpu}" \
    --progress-every 32 \
    "${protocol[@]}" \
    "${object_arguments[@]}" \
    > "${run_root}/logs/isaacsim_shard${shard}.log" 2>&1
  printf 'complete\n' > "${run_root}/status/isaacsim_shard${shard}.status"
}

printf 'running\n' > "${run_root}/status/isaacsim.status"
run_shard 0 0 "${objects[@]:0:3}" & sim0=$!
run_shard 1 1 "${objects[@]:3:3}" & sim1=$!
run_shard 2 2 "${objects[@]:6:2}" & sim2=$!
run_shard 3 3 "${objects[@]:8:2}" & sim3=$!
if ! wait_all "${sim0}" "${sim1}" "${sim2}" "${sim3}"; then
  printf 'failed\n' > "${run_root}/status/isaacsim.status"
  exit 1
fi
printf 'complete\n' > "${run_root}/status/isaacsim.status"
