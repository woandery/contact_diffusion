#!/usr/bin/env bash
# One object/contact set per hand: real v7 checkpoint -> AR -> FK -> EAWQ -> PhysX.
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mck_root="$(cd "${project_root}/.." && pwd)"
python_path="${CONTACT_V7_PYTHON:-/home/zhb1/miniconda3/envs/Dexgrasp/bin/python}"
isaac_runner="${CONTACT_V7_ISAAC_RUNNER:-${mck_root}/IsaacGymLocal/run_python_cpu.sh}"
checkpoint="${CONTACT_V7_CHECKPOINT:-${project_root}/weights/v7/best_val.pt}"
protocol="${project_root}/configs/basic_experiment_fetchbench_ar32k_eawq_o10i20_palm0_v7_smoke1_protocol.yaml"
manifest="${project_root}/configs/basic_experiment_ood10_local_manifest.json"
run_root="${CONTACT_V7_SMOKE_ROOT:-${project_root}/outputs/v7_local_end_to_end_smoke_verified}"
object_id="${CONTACT_V7_SMOKE_OBJECT:-contactdb_apple}"
device_id="${CONTACT_V7_SMOKE_DEVICE_ID:-0}"
hand_cooldown_seconds="${CONTACT_V7_SMOKE_HAND_COOLDOWN_SECONDS:-10}"
expected_sha="c55badbd2e1ce7bc9cda9b58832e68003b4b757ee02eedee47e01e697c7ec491"

actual_sha="$(sha256sum "${checkpoint}" | awk '{print $1}')"
if [[ "${actual_sha}" != "${expected_sha}" ]]; then
  printf 'checkpoint SHA256 mismatch: %s != %s\n' "${actual_sha}" "${expected_sha}" >&2
  exit 2
fi

mkdir -p "${run_root}/candidates/barrett" "${run_root}/candidates/shadowhand" \
  "${run_root}/prepared_all/barrett" "${run_root}/prepared_all/shadowhand" \
  "${run_root}/results/barrett" "${run_root}/results/shadowhand" \
  "${run_root}/logs"

generate_hand() {
  local hand="$1" gripper config hand_index
  hand="$1"
  if [[ "${hand}" == "barrett" ]]; then
    gripper=Barrett
    config="${project_root}/configs/multigripper_fk_multidex_ood10_barrett_steps400_palm0_v4.yaml"
    hand_index=0
  else
    gripper=shadow_hand
    config="${project_root}/configs/multigripper_fk_multidex_ood10_shadow_dro_steps400_palm0_v4.yaml"
    hand_index=1
  fi
  printf 'generate+FK: %s/%s\n' "${hand}" "${object_id}"
  CUDA_VISIBLE_DEVICES="${device_id}" "${python_path}" \
    "${project_root}/scripts/infer_local_contactdiffusion_grasp.py" \
    --config "${config}" --checkpoint "${checkpoint}" \
    --manifest "${manifest}" --manifest-only --object-id "${object_id}" \
    --grippers "${gripper}" --sample-start 0 --samples-per-object 1 \
    --inference-object-observation full --autoregressive-fk-target nearest_2048 \
    --particles 32 --optimization-steps 400 --diffusion-steps 50 \
    --fk-initialization enveloping --envelope-approach-weight 2.0 \
    --selection-min-envelope-cosine 0.5 --selection-min-approach-cosine 0.8 \
    --preferred-root-direction 0 0 1 --selection-max-penetration 0.007 \
    --top-k 32 --device cuda:0 --seed 20260808 \
    --hand-index-offset "${hand_index}" --object-index-offset 0 \
    --output "${run_root}/candidates/${hand}/${object_id}.json" \
    >"${run_root}/logs/generate_${hand}.log" 2>&1

  "${python_path}" "${project_root}/scripts/prepare_basic_experiment_isaacgym.py" \
    --candidates "${run_root}/candidates/${hand}/${object_id}.json" \
    --config "${config}" --gripper "${gripper}" --candidate-mode all \
    --execution-protocol "${protocol}" --allow-runtime-budget --allow-incomplete \
    --output "${run_root}/prepared_all/${hand}/${object_id}.json" \
    >"${run_root}/logs/prepare_${hand}.log" 2>&1
  printf 'prepared: %s/%s\n' "${hand}" "${object_id}"
}

generate_hand barrett
printf 'waiting %ss for local CUDA/process memory reclamation\n' "${hand_cooldown_seconds}"
sleep "${hand_cooldown_seconds}"
generate_hand shadowhand

printf 'exact ranking-only EAWQ: 2 hands x 1 set x 32 particles\n'
CUDA_VISIBLE_DEVICES="${device_id}" "${python_path}" \
  "${project_root}/scripts/compute_eawq_rank_fusion_metrics.py" \
  --run-root "${run_root}" --output-dir "${run_root}/eawq/metrics" \
  --workers 1 --qp-device cuda:0 --expected-hands 2 \
  --expected-objects-per-hand 1 --expected-sets-per-object 1 \
  --expected-particles-per-set 32 --ranking-only \
  >"${run_root}/logs/eawq_metrics.log" 2>&1

"${python_path}" "${project_root}/scripts/prepare_eawq_rank_fusion_top1.py" \
  --particle-metrics "${run_root}/eawq/metrics/particle_metrics.csv.gz" \
  --prepared-root "${run_root}/prepared_all" --output-root "${run_root}/eawq" \
  --execution-protocol "${protocol}" --expected-hands 2 \
  --expected-objects-per-hand 1 --expected-sets-per-object 1 \
  --expected-particles-per-set 32 \
  >"${run_root}/logs/eawq_select.log" 2>&1

validate_hand() {
  local hand="$1" hand_root hand_urdf
  if [[ "${hand}" == "barrett" ]]; then
    hand_root="${project_root}/outputs/local_dro_isaacgym_twohands/full/assets/barrett_adagrasp"
    hand_urdf=model_extended.urdf
  else
    hand_root="${project_root}/outputs/shadow_dro_failure_visualizations/assets/robot/shadowhand"
    hand_urdf=shadow_hand_right_extended.urdf
  fi
  printf 'CPU PhysX six-direction validation: %s/%s\n' "${hand}" "${object_id}"
  CUDA_VISIBLE_DEVICES="${device_id}" "${isaac_runner}" \
    "${project_root}/scripts/validate_native_shadowhand_isaacgym_oldparams_dro.py" \
    --prepared "${run_root}/eawq/prepared/${hand}/${object_id}.json" \
    --output "${run_root}/results/${hand}/${object_id}.json" \
    --gendex-root "${mck_root}/GenDexGrasp" \
    --native-hand-root "${hand_root}" --native-hand-urdf "${hand_urdf}" \
    --native-hand-urdf-is-extended --only-object "${object_id}" \
    --max-samples-per-object 1 --device-id 0 --cpu-physics --progress-every 1 \
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
    --contact-offset 0.01 --rest-offset 0 --no-ground \
    >"${run_root}/logs/physx_${hand}.log" 2>&1
}

validate_hand barrett
validate_hand shadowhand

"${python_path}" - "${run_root}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
summary = {"schema": "contactdiff-v7-local-smoke-v1", "run_root": str(root.resolve())}
for hand in ("barrett", "shadowhand"):
    candidate = json.loads(next((root / "candidates" / hand).glob("*.json")).read_text())
    result = json.loads(next((root / "results" / hand).glob("*.json")).read_text())
    generator = candidate["contact_generator"]
    summary[hand] = {
        "checkpoint_step": candidate["checkpoint_step"],
        "checkpoint_sha256": candidate["checkpoint_sha256"],
        "condition_observation": generator["condition_observation"],
        "normalization_reference": generator["normalization_reference"],
        "contact_sets": len(candidate["records"]),
        "particles_per_set": candidate["particles"],
        "fk_steps": candidate["optimization_steps"],
        "physx_status": result.get("status"),
        "physx_trials": result.get("trials"),
        "physx_successes": result.get("successes"),
    }
(root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
PY
