#!/usr/bin/env bash
set -euo pipefail

FETCHBENCH_ROOT="${FETCHBENCH_ROOT:-/inspire/qb-ilm2/project/zhanghanbo/public/mck/FetchBench-CORL2024}"
CONTACT_ROOT="${CONTACT_ROOT:-/inspire/qb-ilm2/project/zhanghanbo/public/mck/ContactDiffusion}"
DRO_REPRO_ROOT="${DRO_REPRO_ROOT:-/inspire/qb-ilm2/project/zhanghanbo/public/mck/dro_grasp_reproduction}"
DRO_ROOT="${DRO_ROOT:-${DRO_REPRO_ROOT}/DRO-Grasp}"
DRO_PYDEPS="${DRO_PYDEPS:-${DRO_REPRO_ROOT}/pydeps_py310}"
DRO_PYTHON="${DRO_PYTHON:-/inspire/qb-ilm2/project/zhanghanbo/public/mck/miniconda3/envs/contactdiff/bin/python}"
FETCHBENCH_PYTHON="${FETCHBENCH_PYTHON:-/inspire/qb-ilm2/project/zhanghanbo/public/mck/miniconda3/envs/fetchbench/bin/python}"
INPUT_ROOT="${INPUT_ROOT:-${CONTACT_ROOT}/outputs/fetchbench_candle_111cam_ab_barrett_8x8_fk200_env200_highlr_basefix_20260908/corrected_inputs}"
RUN_ROOT="${RUN_ROOT:-${CONTACT_ROOT}/outputs/fetchbench_candle_111cam_dro_partial_envfilter8_cpu_physx_20260908}"
CHECKPOINT="${CHECKPOINT:-${DRO_ROOT}/ckpt/model/model_3robots_partial.pth}"
GPU_COUNT="${GPU_COUNT:-4}"
WORKERS_PER_GPU="${WORKERS_PER_GPU:-3}"
VALIDATION_WORKERS="${VALIDATION_WORKERS:-24}"
CLEARANCE="${CLEARANCE:-0.005}"
CANDIDATES="${DRO_CANDIDATES:-8}"
GENERATION_ATTEMPTS="${GENERATION_ATTEMPTS:-4}"
BASE_SEED="${BASE_SEED:-20260808}"
TASK_INDEX="${TASK_INDEX:-39}"
SCENE_CONFIG="${SCENE_CONFIG:-RigidObjRoundTable_8}"
SCENE_FACTORY="${SCENE_FACTORY:-RoundTableSceneFactory_33}"
OBJECT_LABEL="${OBJECT_LABEL:-Candle}"
EXPECTED_LEGAL_VIEWS="${EXPECTED_LEGAL_VIEWS:-71}"
MAX_VIEWS="${MAX_VIEWS:-0}"
TASK_DIR="$(printf 'task_%03d' "${TASK_INDEX}")"

LEGAL_JSON="${INPUT_ROOT}/visibility/legal_partial_views.json"
mkdir -p "${RUN_ROOT}/logs" "${RUN_ROOT}/status" "${RUN_ROOT}/results"
test -s "${LEGAL_JSON}"
test -s "${CHECKPOINT}"
test -d "${DRO_ROOT}/model"
test -d "${DRO_ROOT}/data/data_urdf/robot"
test -x "${DRO_PYTHON}"
test -x "${FETCHBENCH_PYTHON}"

mapfile -t VIEW_IDS < <("${DRO_PYTHON}" - "${LEGAL_JSON}" <<'PY'
import json, sys
from pathlib import Path
for row in json.load(open(sys.argv[1]))["views"]:
    print(Path(row["rgbd_capture_dir"]).name)
PY
)
if [[ "${#VIEW_IDS[@]}" -ne "${EXPECTED_LEGAL_VIEWS}" ]]; then
  echo "Expected ${EXPECTED_LEGAL_VIEWS} legal views, found ${#VIEW_IDS[@]}" >&2
  exit 1
fi
if ((MAX_VIEWS > 0 && MAX_VIEWS < ${#VIEW_IDS[@]})); then
  VIEW_IDS=("${VIEW_IDS[@]:0:MAX_VIEWS}")
fi

raw_is_complete() {
  "${DRO_PYTHON}" -c \
    'import json,sys; d=json.load(open(sys.argv[1])); assert d["hand"]==sys.argv[2]; assert len(d["records"])==int(sys.argv[3]); assert d["sampled_points"]==512; assert d["optimization_steps"]==64' \
    "$1" "$2" "${CANDIDATES}" 2>/dev/null
}

filter_is_complete() {
  "${DRO_PYTHON}" -c \
    'import json,sys; d=json.load(open(sys.argv[1])); m=d["environment_filter"]; assert m["generated_candidates"]==int(sys.argv[3]); assert len(d["records"])==m["retained_candidates"]; assert abs(m["clearance_m"]-float(sys.argv[2]))<1e-12' \
    "$1" "${CLEARANCE}" "${CANDIDATES}" 2>/dev/null
}

generate_case() {
  local gpu="$1" view_index="$2" hand="$3"
  local view_id="${VIEW_IDS[view_index]}"
  local goal_pc="${INPUT_ROOT}/visibility/rgbd_views/${view_id}/target_partial_robot_base.npy"
  local scene_pc="${INPUT_ROOT}/visibility/rgbd_views/${view_id}/scene_partial_robot_base.npy"
  local artifact="${RUN_ROOT}/views/${view_id}/simulation/${SCENE_FACTORY}/${TASK_DIR}/${hand}"
  local raw="${artifact}/dro_candidates.json"
  local filtered="${artifact}/dro_candidates_environment_filtered.json"
  local inference_seed=$((BASE_SEED + TASK_INDEX))
  local attempt attempt_seed
  mkdir -p "${artifact}"

  if ! raw_is_complete "${raw}" "${hand}"; then
    # cvxpylayers/SCS occasionally reports an inaccurate-unbounded solve for
    # one sampled initialization.  Retry a fixed, predeclared seed schedule;
    # this is numerical-failure recovery, not success-conditioned resampling.
    for ((attempt=0; attempt<GENERATION_ATTEMPTS; attempt++)); do
      attempt_seed=$((inference_seed + attempt * 1000003))
      if CUDA_VISIBLE_DEVICES="${gpu}" \
        LANG=C.UTF-8 LC_ALL=C.UTF-8 \
        PYTHONPATH="${DRO_PYDEPS}:${DRO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
        MPLCONFIGDIR="${RUN_ROOT}/mpl-cache" \
        "${DRO_PYTHON}" "${FETCHBENCH_ROOT}/scripts/dro_generate_grasps.py" \
        --dro-root "${DRO_ROOT}" --checkpoint "${CHECKPOINT}" \
        --input "${goal_pc}" --output "${raw}" --hand "${hand}" \
        --candidates "${CANDIDATES}" --points 512 --optimization-steps 64 \
        --seed "${attempt_seed}" --device cuda:0; then
        printf 'generation_seed=%s attempt=%s\n' "${attempt_seed}" "$((attempt + 1))" \
          >"${artifact}/generation_retry_audit.txt"
        break
      fi
      printf 'generation_retry seed=%s attempt=%s/%s\n' \
        "${attempt_seed}" "$((attempt + 1))" "${GENERATION_ATTEMPTS}" >&2
    done
  fi
  raw_is_complete "${raw}" "${hand}" || return 1

  if ! filter_is_complete "${filtered}"; then
    "${DRO_PYTHON}" "${FETCHBENCH_ROOT}/scripts/filter_dro_environment_candidates.py" \
      --input "${raw}" --scene-pointcloud "${scene_pc}" \
      --clearance "${CLEARANCE}" --output "${filtered}" || return 1
  fi
  filter_is_complete "${filtered}" || return 1
}

total_views="${#VIEW_IDS[@]}"
total_cases=$((total_views * 2))
worker_count=$((GPU_COUNT * WORKERS_PER_GPU))

generation_worker() {
  local worker="$1" slot hand_idx view_index hand view_id failures=0
  local gpu=$((worker % GPU_COUNT + ${GPU_OFFSET:-0}))
  for ((slot=worker; slot<total_cases; slot+=worker_count)); do
    hand_idx=$((slot / total_views))
    view_index=$((slot % total_views))
    [[ "${hand_idx}" -eq 0 ]] && hand=barrett || hand=shadowhand
    view_id="${VIEW_IDS[view_index]}"
    printf 'generating gpu=%s worker=%s view=%s hand=%s\n' "${gpu}" "${worker}" "${view_id}" "${hand}" \
      >"${RUN_ROOT}/status/${view_id}_${hand}.txt"
    if generate_case "${gpu}" "${view_index}" "${hand}"; then
      printf 'generated_and_filtered\n' >"${RUN_ROOT}/status/${view_id}_${hand}.txt"
    else
      printf 'generation_failed\n' >"${RUN_ROOT}/status/${view_id}_${hand}.txt"
      failures=$((failures + 1))
    fi
  done
  ((failures == 0))
}

mkdir -p "${RUN_ROOT}/mpl-cache"
printf 'generating cases=%s raw_poses=%s workers=%s\n' "${total_cases}" "$((total_cases * CANDIDATES))" "${worker_count}" \
  >"${RUN_ROOT}/status/generation_master.txt"
pids=()
for ((worker=0; worker<worker_count; worker++)); do
  generation_worker "${worker}" >"${RUN_ROOT}/logs/generation_worker_${worker}.log" 2>&1 &
  pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do wait "${pid}" || failed=1; done
if ((failed)); then
  printf 'generation_failed\n' >"${RUN_ROOT}/status/generation_master.txt"
  exit 1
fi
printf 'generation_complete cases=%s raw_poses=%s\n' "${total_cases}" "$((total_cases * CANDIDATES))" \
  >"${RUN_ROOT}/status/generation_master.txt"

export ASSET_PATH="${FETCHBENCH_ROOT}"
export PYTHONPATH="${FETCHBENCH_ROOT}/third_party/isaacgym/python:${FETCHBENCH_ROOT}/InfiniGym${PYTHONPATH:+:${PYTHONPATH}}"
export LD_LIBRARY_PATH="$(dirname "${FETCHBENCH_PYTHON}")/../lib:/usr/local/cuda-12.8/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-${CONTACT_ROOT}/outputs/fetchbench_singleview_ab_w10_20260907/candle_isaacgym_remote/torch_extensions}"
export MAX_JOBS="${MAX_JOBS:-4}"

summary_is_complete() {
  "${FETCHBENCH_PYTHON}" -c \
    'import json,sys; d=json.load(open(sys.argv[1])); assert d["validated_candidates"]==int(sys.argv[2])' \
    "$1" "$2" 2>/dev/null
}

validate_case() {
  local view_index="$1" hand="$2"
  local view_id="${VIEW_IDS[view_index]}"
  local goal_pc="${INPUT_ROOT}/visibility/rgbd_views/${view_id}/target_partial_robot_base.npy"
  local scene_pc="${INPUT_ROOT}/visibility/rgbd_views/${view_id}/scene_partial_robot_base.npy"
  local simulation_root="${RUN_ROOT}/views/${view_id}/simulation"
  local artifact="${simulation_root}/${SCENE_FACTORY}/${TASK_DIR}/${hand}"
  local filtered="${artifact}/dro_candidates_environment_filtered.json"
  local summary="${artifact}/lift_validation/summary.json"
  local retained task urdf
  retained="$("${DRO_PYTHON}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["environment_filter"]["retained_candidates"])' "${filtered}")"

  if [[ "${retained}" -eq 0 ]]; then
    printf 'complete retained=0 validated=0\n' >"${RUN_ROOT}/status/${view_id}_${hand}.txt"
    return 0
  fi
  if summary_is_complete "${summary}" "${retained}"; then return 0; fi
  if [[ "${hand}" == barrett ]]; then
    task=FetchPtdDRORenderBarrett
    urdf=urdf/dexterous/dro_barrett_physics.urdf
  else
    task=FetchPtdDRORenderShadow
    urdf=urdf/dexterous/dro_shadowhand_physics.urdf
  fi
  printf 'validating retained=%s\n' "${retained}" >"${RUN_ROOT}/status/${view_id}_${hand}.txt"
  cd "${FETCHBENCH_ROOT}/InfiniGym"
  CUDA_VISIBLE_DEVICES='' "${FETCHBENCH_PYTHON}" isaacgymenvs/validate_dro_lift.py \
    task="${task}" scene="benchmark_eval/${SCENE_CONFIG}" \
    task.solution.task_index="${TASK_INDEX}" task.solution.physics_only=true \
    task.solution.goal_pointcloud_override="${goal_pc}" \
    task.solution.goal_pointcloud_override_source=fetchbench_same_view_partial \
    task.solution.scene_pointcloud_override="${scene_pc}" \
    task.solution.visualize_top_k="${retained}" task.solution.min_scene_clearance="${CLEARANCE}" \
    task.solution.reject_runtime_environment_contacts=true \
    task.solution.lift.record_video=false task.solution.lift.direct_closure=true \
    task.solution.lift.max_preclosure_object_displacement=0.02 \
    task.solution.lift.height=0.25 task.solution.lift.success_height=0.10 \
    task.solution.dro.root="${DRO_ROOT}" task.solution.dro.python="${DRO_PYTHON}" \
    task.solution.dro.inference_script="${FETCHBENCH_ROOT}/scripts/dro_generate_grasps.py" \
    task.solution.dro.checkpoint="${CHECKPOINT}" task.solution.dro.candidates="${CANDIDATES}" \
    task.solution.dro.points=512 task.solution.dro.optimization_steps=64 \
    task.solution.dro.seed="$((BASE_SEED + TASK_INDEX))" \
    task.solution.dro.device=cpu task.solution.dro.reuse_cache=true \
    task.env.enableCameraSensors=false \
    task.env.robot.asset_root="${FETCHBENCH_ROOT}/InfiniGym/assets" \
    task.env.robot.urdf_file="${urdf}" \
    task.solution.artifact_dir="${simulation_root}" \
    seed="${BASE_SEED}" num_threads=4 pipeline=cpu sim_device=cpu rl_device=cpu \
    graphics_device_id=-1 headless=true force_render=false || return 1
  summary_is_complete "${summary}" "${retained}" || return 1
  printf 'complete retained=%s validated=%s\n' "${retained}" "${retained}" \
    >"${RUN_ROOT}/status/${view_id}_${hand}.txt"
}

validation_worker() {
  local worker="$1" slot hand_idx view_index hand failures=0
  for ((slot=worker; slot<total_cases; slot+=VALIDATION_WORKERS)); do
    hand_idx=$((slot / total_views)); view_index=$((slot % total_views))
    [[ "${hand_idx}" -eq 0 ]] && hand=barrett || hand=shadowhand
    if ! validate_case "${view_index}" "${hand}"; then
      printf 'validation_failed\n' >"${RUN_ROOT}/status/${VIEW_IDS[view_index]}_${hand}.txt"
      failures=$((failures + 1))
    fi
  done
  ((failures == 0))
}

printf 'validating cases=%s workers=%s\n' "${total_cases}" "${VALIDATION_WORKERS}" \
  >"${RUN_ROOT}/status/validation_master.txt"
pids=()
for ((worker=0; worker<VALIDATION_WORKERS; worker++)); do
  validation_worker "${worker}" >"${RUN_ROOT}/logs/validation_worker_${worker}.log" 2>&1 &
  pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do wait "${pid}" || failed=1; done
if ((failed)); then
  printf 'validation_failed\n' >"${RUN_ROOT}/status/validation_master.txt"
  exit 1
fi

"${DRO_PYTHON}" - "${LEGAL_JSON}" "${RUN_ROOT}" "${CLEARANCE}" \
  "${SCENE_FACTORY}" "${TASK_DIR}" "${OBJECT_LABEL}" "${MAX_VIEWS}" "${CANDIDATES}" <<'PY'
import json, sys
from collections import defaultdict
from pathlib import Path

legal, root, clearance = Path(sys.argv[1]), Path(sys.argv[2]), float(sys.argv[3])
scene_factory, task_dir, object_label = sys.argv[4:7]
views = [Path(x["rgbd_capture_dir"]).name for x in json.loads(legal.read_text())["views"]]
max_views = int(sys.argv[7])
candidates = int(sys.argv[8])
if max_views > 0:
    views = views[:max_views]
rows = []
for view in views:
    for hand in ("barrett", "shadowhand"):
        artifact = root / "views" / view / "simulation" / scene_factory / task_dir / hand
        filtered = json.loads((artifact / "dro_candidates_environment_filtered.json").read_text())
        env = filtered["environment_filter"]
        retained = int(env["retained_candidates"])
        if retained:
            summary = json.loads((artifact / "lift_validation" / "summary.json").read_text())
            validated = int(summary["validated_candidates"])
            successes = int(summary["successes"])
        else:
            validated = successes = 0
        if validated != retained:
            raise RuntimeError(f"{view}/{hand}: retained={retained}, validated={validated}")
        rows.append({
            "view_id": view, "hand": hand, "generated": candidates,
            "environment_retained": retained, "environment_deleted": candidates - retained,
            "validated": validated, "successes": successes,
            "conditional_physx_success_rate": successes / validated if validated else None,
            "end_to_end_success_rate": successes / candidates,
        })

aggregate = []
for hand in ("barrett", "shadowhand"):
    group = [x for x in rows if x["hand"] == hand]
    generated = sum(x["generated"] for x in group)
    retained = sum(x["environment_retained"] for x in group)
    successes = sum(x["successes"] for x in group)
    aggregate.append({
        "hand": hand, "views": len(group), "generated": generated,
        "environment_retained": retained, "environment_deleted": generated - retained,
        "environment_survival_rate": retained / generated,
        "validated": retained, "successes": successes,
        "conditional_physx_success_rate": successes / retained if retained else None,
        "end_to_end_success_rate": successes / generated,
        "views_with_any_success": sum(x["successes"] > 0 for x in group),
    })
generated = sum(x["generated"] for x in rows)
retained = sum(x["environment_retained"] for x in rows)
successes = sum(x["successes"] for x in rows)
payload = {
    "experiment": f"{object_label} {len(views)} legal single views, D(R,O), {candidates} candidates/view/hand",
    "input": "same-view target partial point cloud in corrected robot-base coordinates",
    "environment_filter": "outer-preshape hand points vs same-view partial scene, nearest-neighbor clearance",
    "environment_clearance_m": clearance,
    "physics": "CPU PhysX, q_outer -> q_inner -> lift 0.25 m, no video",
    "aggregate_hand": aggregate,
    "aggregate_all": {
        "views_hand_units": len(rows), "generated": generated,
        "environment_retained": retained, "environment_deleted": generated-retained,
        "environment_survival_rate": retained/generated,
        "validated": retained, "successes": successes,
        "conditional_physx_success_rate": successes/retained if retained else None,
        "end_to_end_success_rate": successes/generated,
    },
    "views": rows,
}
(root / "results" / "summary.json").write_text(json.dumps(payload, indent=2) + "\n")
lines = [
    f"# {object_label} {len(views)}视角 D(R,O) 环境筛选 + CPU PhysX", "",
    f"每个视角、每只手生成{candidates}姿态；5 mm partial-scene门限删除环境碰撞后，仅执行保留姿态。", "",
    "| Hand | Generated | Retained | Deleted | Survival | Success/Executed | Conditional | Success/Generated | End-to-end |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
]
for x in aggregate:
    lines.append(
        f"| {x['hand']} | {x['generated']} | {x['environment_retained']} | {x['environment_deleted']} | {x['environment_survival_rate']:.2%} | "
        f"{x['successes']}/{x['validated']} | {x['conditional_physx_success_rate']:.2%} | {x['successes']}/{x['generated']} | {x['end_to_end_success_rate']:.2%} |"
    )
(root / "results" / "REPORT.md").write_text("\n".join(lines) + "\n")
print(json.dumps(payload["aggregate_hand"], indent=2))
PY

printf 'complete cases=%s\n' "${total_cases}" >"${RUN_ROOT}/status/validation_master.txt"
echo "D(R,O) ${EXPECTED_LEGAL_VIEWS}-view environment-filtered CPU PhysX experiment complete: ${RUN_ROOT}"
