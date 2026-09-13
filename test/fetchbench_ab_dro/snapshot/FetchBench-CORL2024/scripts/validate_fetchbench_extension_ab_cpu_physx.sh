#!/usr/bin/env bash
set -euo pipefail

# Execute every selected pose from the 111-camera Mug A/B experiment.
# Validation intentionally uses headless CPU PhysX: the four GPUs remain a
# generation resource, while the execution backend matches the earlier Mug
# A/B table (direct q_outer -> q_inner -> 25 cm lift, no video).

repo_root="${FETCHBENCH_ROOT:-/inspire/qb-ilm2/project/zhanghanbo/public/mck/FetchBench-CORL2024}"
trap '"${contact_python}" "${repo_root}/scripts/summarize_extension3_111.py" --run "$(dirname "$(dirname "$(dirname "${run_root}")")")" || true' EXIT
contact_root="${CONTACTDIFF_ROOT:-/inspire/qb-ilm2/project/zhanghanbo/public/mck/ContactDiffusion}"
run_root="${RUN_ROOT:-${contact_root}/outputs/fetchbench_mug_111cam_ab_w10_20260907}"
source_root="${SOURCE_ROOT:-${run_root}}"
object_prefix="${OBJECT_PREFIX:?required}"
scene_alias="${SCENE_ALIAS:?required}"
task_index="${TASK_INDEX:?required}"
contact_python="${CONTACTDIFF_PYTHON:-/inspire/qb-ilm2/project/zhanghanbo/public/mck/miniconda3/envs/contactdiff/bin/python}"
fetchbench_python="${FETCHBENCH_PYTHON:-/inspire/qb-ilm2/project/zhanghanbo/public/mck/miniconda3/envs/fetchbench/bin/python}"
validation_workers="${VALIDATION_WORKERS:-16}"
hand_filter="${HAND_FILTER:-all}"
condition_filter="${CONDITION_FILTER:-all}"
poses_per_case="${POSES_PER_CASE:-32}"
max_views="${MAX_VIEWS:-0}"
legal_json="${source_root}/visibility/legal_partial_views.json"
validation_root="${VALIDATION_ROOT:-${run_root}/cpu_physx_all${poses_per_case}}"

config_for_hand() {
  if [[ "$1" == "barrett" ]]; then
    printf '%s' "${CONFIG_BARRETT:-${contact_root}/configs/multigripper_fk_multidex_ood10_barrett_steps400_palm0_v4.yaml}"
  else
    printf '%s' "${CONFIG_SHADOW:-${contact_root}/configs/multigripper_fk_multidex_ood10_shadow_dro_steps400_palm0_v4.yaml}"
  fi
}

gripper_for_hand() {
  [[ "$1" == "barrett" ]] && printf 'Barrett' || printf 'shadow_hand'
}

task_for_hand() {
  [[ "$1" == "barrett" ]] && printf 'FetchPtdDRORenderBarrett' || printf 'FetchPtdDRORenderShadow'
}

urdf_for_hand() {
  [[ "$1" == "barrett" ]] \
    && printf 'contactdiff_v4_barrett_physics.urdf' \
    || printf 'contactdiff_v4_shadowhand_physics.urdf'
}

decode_case() {
  local case_index="$1" view_index rem
  view_index=$((case_index / 4))
  rem=$((case_index % 4))
  CASE_VIEW_INDEX="${view_index}"
  CASE_VIEW_ID="${view_ids[view_index]}"
  if (( rem < 2 )); then CASE_CONDITION=A; else CASE_CONDITION=B; fi
  if (( rem % 2 == 0 )); then CASE_HAND=barrett; else CASE_HAND=shadowhand; fi
}

prepared_has_expected() {
  "${contact_python}" -c \
    'import json,sys; d=json.load(open(sys.argv[1])); assert len(d["objects"]) == 1; assert len(d["objects"][0]["samples"]) == int(sys.argv[2])' \
    "$1" "${poses_per_case}" 2>/dev/null
}

summary_has_expected() {
  "${fetchbench_python}" -c \
    'import json,sys; d=json.load(open(sys.argv[1])); assert int(d["validated_candidates"]) == int(sys.argv[2])' \
    "$1" "${poses_per_case}" 2>/dev/null
}

mkdir -p "${validation_root}/prepared" "${validation_root}/simulation" \
  "${validation_root}/logs" "${validation_root}/status" "${validation_root}/results"
test -s "${legal_json}"
test -x "${contact_python}"
test -x "${fetchbench_python}"
if [[ "${hand_filter}" != "all" && "${hand_filter}" != "barrett" && "${hand_filter}" != "shadowhand" ]]; then
  echo "HAND_FILTER must be all, barrett, or shadowhand" >&2
  exit 2
fi
if [[ "${condition_filter}" != "all" && "${condition_filter}" != "A" && "${condition_filter}" != "B" ]]; then
  echo "CONDITION_FILTER must be all, A, or B" >&2
  exit 2
fi

mapfile -t view_ids < <("${contact_python}" - "${legal_json}" <<'PY'
import json
import sys
from pathlib import Path
for row in json.loads(Path(sys.argv[1]).read_text())["views"]:
    print(Path(row["rgbd_capture_dir"]).name)
PY
)
if (( max_views > 0 && max_views < ${#view_ids[@]} )); then
  view_ids=("${view_ids[@]:0:max_views}")
fi
total_cases=$(( ${#view_ids[@]} * 4 ))
if [[ "${hand_filter}" == "all" ]]; then hand_count=2; else hand_count=1; fi
if [[ "${condition_filter}" == "all" ]]; then condition_count=2; else condition_count=1; fi
filtered_cases=$(( ${#view_ids[@]} * condition_count * hand_count ))
filtered_poses=$((filtered_cases * poses_per_case))
stage_status="${validation_root}/status/master_${condition_filter}_${hand_filter}.txt"

prepare_one() {
  local case_index="$1" selected prepared config gripper root
  decode_case "${case_index}"
  root="${run_root}/views/${CASE_VIEW_ID}/${CASE_CONDITION}"
  selected="${root}/selected_w10/${object_prefix}_${CASE_HAND}.json"
  prepared="${validation_root}/prepared/${CASE_VIEW_ID}/${CASE_CONDITION}/${object_prefix}_${CASE_HAND}.json"
  config="$(config_for_hand "${CASE_HAND}")"
  gripper="$(gripper_for_hand "${CASE_HAND}")"
  mkdir -p "$(dirname "${prepared}")"
  if prepared_has_expected "${prepared}"; then return 0; fi
  test -s "${selected}"
  cd "${contact_root}"
  "${contact_python}" scripts/prepare_basic_experiment_isaacgym.py \
    --candidates "${selected}" --config "${config}" --gripper "${gripper}" \
    --candidate-mode all --allow-filtered-candidates --allow-runtime-budget \
    --closure-outer-fraction 0.10 --closure-inner-fraction 0.20 \
    --output "${prepared}" \
    >"${validation_root}/logs/prepare_${CASE_VIEW_ID}_${CASE_CONDITION}_${CASE_HAND}.log" 2>&1
  prepared_has_expected "${prepared}"
}

prepare_worker() {
  local worker="$1" case_index failures=0
  for ((case_index=worker; case_index<total_cases; case_index+=validation_workers)); do
    decode_case "${case_index}"
    if [[ "${hand_filter}" != "all" && "${CASE_HAND}" != "${hand_filter}" ]]; then continue; fi
    if [[ "${condition_filter}" != "all" && "${CASE_CONDITION}" != "${condition_filter}" ]]; then continue; fi
    if ! prepare_one "${case_index}"; then
      decode_case "${case_index}"
      printf 'prepare_failed\n' >"${validation_root}/status/${CASE_VIEW_ID}_${CASE_CONDITION}_${CASE_HAND}.txt"
      failures=$((failures + 1))
    fi
  done
  (( failures == 0 ))
}

printf 'preparing hand=%s cases=%s poses=%s\n' "${hand_filter}" "${filtered_cases}" "${filtered_poses}" \
  >"${stage_status}"
pids=()
for ((worker=0; worker<validation_workers; worker++)); do
  prepare_worker "${worker}" & pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do wait "${pid}" || failed=1; done
if (( failed )); then
  printf 'prepare_failed\n' >"${stage_status}"
  exit 1
fi

export ASSET_PATH="${repo_root}"
export PYTHONPATH="${repo_root}/third_party/isaacgym/python:${repo_root}/InfiniGym${PYTHONPATH:+:${PYTHONPATH}}"
export LD_LIBRARY_PATH="$(dirname "${fetchbench_python}")/../lib:/usr/local/cuda-12.8/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-${contact_root}/outputs/fetchbench_singleview_ab_w10_20260907/mug_isaacgym_remote/torch_extensions}"
export MAX_JOBS="${MAX_JOBS:-4}"

validate_one() {
  local case_index="$1" prepared artifact log task asset_root urdf summary
  decode_case "${case_index}"
  prepared="${validation_root}/prepared/${CASE_VIEW_ID}/${CASE_CONDITION}/${object_prefix}_${CASE_HAND}.json"
  artifact="${validation_root}/simulation/${CASE_VIEW_ID}/${CASE_CONDITION}/${CASE_HAND}"
  log="${validation_root}/logs/validate_${CASE_VIEW_ID}_${CASE_CONDITION}_${CASE_HAND}.log"
  task="$(task_for_hand "${CASE_HAND}")"
  asset_root="${repo_root}/InfiniGym/assets/contactdiff_hands/${CASE_HAND}"
  urdf="$(urdf_for_hand "${CASE_HAND}")"
  mkdir -p "${artifact}"
  test -s "${prepared}"
  test -s "${asset_root}/${urdf}"
  summary="$(find "${artifact}" -path '*/lift_validation/summary.json' -type f -print -quit)"
  if [[ -n "${summary}" ]] && summary_has_expected "${summary}"; then return 0; fi
  printf 'validating\n' >"${validation_root}/status/${CASE_VIEW_ID}_${CASE_CONDITION}_${CASE_HAND}.txt"

  cd "${repo_root}/InfiniGym"
  CUDA_VISIBLE_DEVICES='' "${fetchbench_python}" isaacgymenvs/validate_dro_lift.py \
    task="${task}" scene="benchmark_eval/${scene_alias}" \
    task.solution.task_index="${task_index}" task.solution.visualize_top_k="${poses_per_case}" \
    task.solution.physics_only=true task.solution.lift.record_video=false \
    task.solution.goal_pointcloud_override="${source_root}/sam3d/${CASE_VIEW_ID}/sam3d_fused_robot_base.npy" \
    task.solution.external_pointcloud="${source_root}/sam3d/${CASE_VIEW_ID}/sam3d_fused_robot_base.npy" \
    task.solution.external_prepared="${prepared}" \
    task.solution.external_object_name="${object_prefix}_${CASE_VIEW_ID}" \
    task.solution.external_require_precomputed_environment=false \
    task.solution.external_validate_infeasible=true \
    task.solution.reject_runtime_environment_contacts=true \
    task.solution.target_closeup_max_tracking_displacement=0.30 \
    task.solution.lift.direct_closure=true \
    task.solution.lift.max_preclosure_object_displacement=0.02 \
    task.solution.lift.height=0.25 task.solution.lift.success_height=0.10 \
    task.env.enableCameraSensors=false \
    task.env.robot.asset_root="${asset_root}" task.env.robot.urdf_file="${urdf}" \
    task.solution.artifact_dir="${artifact}" \
    seed=20260808 num_threads=4 pipeline=cpu sim_device=cpu rl_device=cpu \
    graphics_device_id=-1 headless=true force_render=false \
    >"${log}" 2>&1

  summary="$(find "${artifact}" -path '*/lift_validation/summary.json' -type f -print -quit)"
  test -n "${summary}"
  summary_has_expected "${summary}"
  printf 'complete summary=%s\n' "${summary}" \
    >"${validation_root}/status/${CASE_VIEW_ID}_${CASE_CONDITION}_${CASE_HAND}.txt"
}

validate_worker() {
  local worker="$1" case_index failures=0
  for ((case_index=worker; case_index<total_cases; case_index+=validation_workers)); do
    decode_case "${case_index}"
    if [[ "${hand_filter}" != "all" && "${CASE_HAND}" != "${hand_filter}" ]]; then continue; fi
    if [[ "${condition_filter}" != "all" && "${CASE_CONDITION}" != "${condition_filter}" ]]; then continue; fi
    if ! validate_one "${case_index}"; then
      decode_case "${case_index}"
      printf 'validation_failed\n' >"${validation_root}/status/${CASE_VIEW_ID}_${CASE_CONDITION}_${CASE_HAND}.txt"
      failures=$((failures + 1))
    fi
  done
  (( failures == 0 ))
}

printf 'validating hand=%s cases=%s poses=%s workers=%s\n' \
  "${hand_filter}" "${filtered_cases}" "${filtered_poses}" "${validation_workers}" \
  >"${stage_status}"
pids=()
for ((worker=0; worker<validation_workers; worker++)); do
  validate_worker "${worker}" >"${validation_root}/logs/validation_worker${worker}.log" 2>&1 &
  pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do wait "${pid}" || failed=1; done
if (( failed )); then
  printf 'validation_failed\n' >"${stage_status}"
  exit 1
fi

"${fetchbench_python}" - "${legal_json}" "${validation_root}" "${hand_filter}" "${poses_per_case}" "${condition_filter}" "${max_views}" <<'PY'
import json
import sys
from collections import defaultdict
from pathlib import Path

legal_path = Path(sys.argv[1])
root = Path(sys.argv[2])
hand_filter = sys.argv[3]
poses_per_case = int(sys.argv[4])
condition_filter = sys.argv[5]
max_views = int(sys.argv[6])
rows = json.loads(legal_path.read_text())["views"]
if max_views > 0:
    rows = rows[:max_views]

def view_id(row):
    return Path(row["rgbd_capture_dir"]).name

def difficulty(coverage):
    if coverage < 0.10:
        return "extreme"
    if coverage <= 0.25:
        return "hard"
    if coverage <= 0.45:
        return "medium"
    return "easy"

trials = []
for row in rows:
    vid = view_id(row)
    conditions = ("A", "B") if condition_filter == "all" else (condition_filter,)
    for condition in conditions:
        hands = ("barrett", "shadowhand") if hand_filter == "all" else (hand_filter,)
        for hand in hands:
            artifact = root / "simulation" / vid / condition / hand
            matches = list(artifact.rglob("lift_validation/summary.json"))
            if len(matches) != 1:
                raise RuntimeError(f"Expected one summary under {artifact}, found {len(matches)}")
            data = json.loads(matches[0].read_text())
            trials.append({
                "view_id": vid,
                "radius_m": float(row["radius_m"]),
                "elevation_deg": float(row["elevation_deg"]),
                "azimuth_deg": float(row["azimuth_deg"]),
                "coverage_observable": float(row["fraction_observable"]),
                "difficulty": difficulty(float(row["fraction_observable"])),
                "condition": condition,
                "hand": hand,
                "validated": int(data["validated_candidates"]),
                "successes": int(data["successes"]),
                "success_rate": float(data["success_rate"]),
                "summary": str(matches[0]),
            })

def aggregate(keys):
    groups = defaultdict(lambda: {"validated": 0, "successes": 0, "view_count": 0})
    for row in trials:
        key = tuple(row[k] for k in keys)
        groups[key]["validated"] += row["validated"]
        groups[key]["successes"] += row["successes"]
        groups[key]["view_count"] += 1
    result = []
    for key in sorted(groups, key=lambda x: tuple(str(v) for v in x)):
        item = dict(zip(keys, key))
        item.update(groups[key])
        item["success_rate"] = item["successes"] / item["validated"] if item["validated"] else 0.0
        result.append(item)
    return result

payload = {
    "experiment": f"Mug 111视角抓取抬升验证, {hand_filter}, all {poses_per_case} poses per legal view",
    "legal_views": len(rows),
    "poses_per_view_condition_hand": poses_per_case,
    "expected_executions": len(rows) * (2 if condition_filter == "all" else 1) * (2 if hand_filter == "all" else 1) * poses_per_case,
    "protocol": "q_outer -> q_inner -> 25 cm lift; CPU PhysX; no video",
    "aggregate_condition_hand": aggregate(["condition", "hand"]),
    "aggregate_condition": aggregate(["condition"]),
    "aggregate_hand": aggregate(["hand"]),
    "aggregate_difficulty_condition_hand": aggregate(["difficulty", "condition", "hand"]),
    "aggregate_radius_condition_hand": aggregate(["radius_m", "condition", "hand"]),
    "views": trials,
}
suffix = ""
if condition_filter != "all":
    suffix += f"_{condition_filter}"
if hand_filter != "all":
    suffix += f"_{hand_filter}"
(root / "results" / f"summary{suffix}.json").write_text(json.dumps(payload, indent=2) + "\n")

lines = [
    f"# Mug 111视角抓取抬升验证：A/B all-{poses_per_case} CPU PhysX results ({hand_filter})",
    "",
    f"Legal views: {len(rows)}; executions: {payload['expected_executions']}.",
    "",
    "| Condition | Hand | Success | Rate |",
    "|---|---|---:|---:|",
]
for row in payload["aggregate_condition_hand"]:
    lines.append(
        f"| {row['condition']} | {row['hand']} | {row['successes']}/{row['validated']} | {row['success_rate']:.2%} |"
    )
lines.extend(["", "| Condition | Success | Rate |", "|---|---:|---:|"])
for row in payload["aggregate_condition"]:
    lines.append(f"| {row['condition']} | {row['successes']}/{row['validated']} | {row['success_rate']:.2%} |")
(root / "results" / f"REPORT{suffix}.md").write_text("\n".join(lines) + "\n")
print(json.dumps(payload["aggregate_condition_hand"], indent=2))
PY

printf 'complete hand=%s cases=%s poses=%s\n' "${hand_filter}" "${filtered_cases}" "${filtered_poses}" \
  >"${stage_status}"
echo "Mug 111-camera ${hand_filter} all-${poses_per_case} CPU PhysX validation complete: ${validation_root}"
