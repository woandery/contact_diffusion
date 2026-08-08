#!/usr/bin/env bash
# Gate experiment: D(R,O) candidates/URDF/exact closure under every old protocol parameter.
set -euo pipefail

mck_root="${MCK_ROOT:-/inspire/qb-ilm2/project/zhanghanbo/public/mck}"
project_root="${mck_root}/contact_diffusion"
cedex_root="${mck_root}/CEDex-Grasp"
dro_hand_root="${mck_root}/dro_grasp_reproduction/DRO-Grasp/data/data_urdf/robot/shadowhand"
gym_python="${mck_root}/IsaacGym/.conda-env/bin/python"
run_root="${RUN_ROOT:-outputs/shadow_dro_oldparams_recovery_gate}"
prepared="outputs/shadow_dro_closure_ab_ood10/manifests/exact_dro_25_15.json"
objects=(
  contactdb_apple contactdb_camera contactdb_cylinder_medium
  contactdb_door_knob contactdb_rubber_duck contactdb_water_bottle
  ycb_055_baseball ycb_016_pear
  ycb_010_potted_meat_can ycb_005_tomato_soup_can
)

cd "${project_root}"
mkdir -p "${run_root}"/{results/gym,results/sim,logs,status,state,cache,urdf_cache}

run_gym_object() {
  local object_name="$1" gpu="$2"
  PYTHONNOUSERSITE=1 \
  TORCH_EXTENSIONS_DIR="${mck_root}/IsaacGym/.torch_extensions" \
  MAX_JOBS=4 CUDA_VISIBLE_DEVICES="${gpu}" \
  LD_LIBRARY_PATH="${mck_root}/IsaacGym/.conda-env/lib:${LD_LIBRARY_PATH:-}" \
  "${gym_python}" scripts/validate_native_shadowhand_isaacgym_oldparams_dro.py \
    --prepared "${prepared}" \
    --output "${run_root}/results/gym/${object_name}.json" \
    --gendex-root "${mck_root}/GenDexGrasp" \
    --native-hand-root "${dro_hand_root}" \
    --native-hand-urdf shadow_hand_right.urdf \
    --device-id 0 --only-object "${object_name}" --resume \
    >"${run_root}/logs/gym_${object_name}.log" 2>&1
}

echo running >"${run_root}/status/gym.status"
for start in 0 2 4 6 8; do
  pids=()
  for offset in 0 1; do
    index=$((start + offset))
    run_gym_object "${objects[index]}" "${offset}" & pids+=("$!")
  done
  for pid in "${pids[@]}"; do wait "${pid}"; done
done
echo complete >"${run_root}/status/gym.status"

sim_protocol=(
  --steps-per-second 60 --substeps 2 --grasp-steps 200
  --direction-seconds 0.8333333333333334 --acceleration 0.5 --threshold 0.02
  --success-displacement-mode per_direction --direction-order gendex
  --robot-friction 10 --object-friction 10 --object-density 10000
  --object-linear-damping 10 --object-angular-damping 100
  --joint-stiffness 400 --joint-damping 400
  --virtual-root-stiffness 1000000 --virtual-root-damping 100000
  --joint-armature 0.001 --solver-position-iterations 4
  --solver-velocity-iterations 0 --progress-every 32
)

run_sim_shard() {
  local shard="$1" gpu="$2"; shift 2
  local only=() object_name
  for object_name in "$@"; do only+=(--only-object "${object_name}"); done
  CONTACTDIFF_ISAAC_STATE_ROOT="${project_root}/${run_root}/state/shard${shard}" \
  CONTACTDIFF_ISAAC_CACHE_ROOT="${project_root}/${run_root}/cache/shard${shard}" \
  CONTACTDIFF_OMNI_PORTABLE_ROOT="${project_root}/${run_root}/state/shard${shard}/portable" \
  OMNI_KIT_ACCEPT_EULA=YES CONTACTDIFF_SKIP_VIEWPORT_WAIT=1 \
  CONTACTDIFF_HARD_EXIT_ISAAC=1 \
  bash scripts/run_contactdiff_isaacsim_601.sh \
    "${cedex_root}/scripts/validate_isaacsim_six_direction.py" \
    --prepared "${project_root}/${prepared}" \
    --output "${project_root}/${run_root}/results/sim/shard${shard}.json" \
    --urdf-cache-dir "${project_root}/${run_root}/urdf_cache/shard${shard}" \
    --gpu-id "${gpu}" --resume "${sim_protocol[@]}" "${only[@]}" \
    >"${run_root}/logs/sim_shard${shard}.log" 2>&1
}

echo running >"${run_root}/status/sim.status"
run_sim_shard 0 0 "${objects[@]:0:5}" & sim0=$!
run_sim_shard 1 1 "${objects[@]:5:5}" & sim1=$!
failed=0
wait "${sim0}" || failed=1
wait "${sim1}" || failed=1
if ((failed)); then echo failed >"${run_root}/status/sim.status"; exit 1; fi
echo complete >"${run_root}/status/sim.status"

python scripts/summarize_shadow_dro_oldparams_gate.py \
  --gym "${run_root}/results/gym/*.json" \
  --sim "${run_root}/results/sim/*.json" \
  --output-json "${run_root}/summary.json" \
  --output-md "${run_root}/REPORT.md" \
  >"${run_root}/logs/summary.log" 2>&1
python -c 'import json,sys; d=json.load(open(sys.argv[1])); print(d["decision"])' \
  "${run_root}/summary.json" >"${run_root}/status/gate.status"
