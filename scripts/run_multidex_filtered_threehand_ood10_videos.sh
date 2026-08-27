#!/usr/bin/env bash
# Replay up to 32 unique filtered successes per hand/object, eight poses per video.
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mck_root="$(cd "${project_root}/.." && pwd)"
runner="${CONTACTDIFF_ISAAC_RUNNER:-${project_root}/scripts/run_isaacgym_cpu_compat.sh}"
if [[ -x "${mck_root}/IsaacGymLocal/.conda-env/bin/python" ]]; then
  default_selection_python="${mck_root}/IsaacGymLocal/.conda-env/bin/python"
else
  default_selection_python="${mck_root}/IsaacGym/.conda-env/bin/python"
fi
selection_python="${CONTACTDIFF_SELECTION_PYTHON:-${default_selection_python}}"
dro_root="${CONTACTDIFF_DRO_ROOT:-${mck_root}/DRO-Grasp}"
object_key="${CONTACTDIFF_OBJECT_KEY:-validate}"
output_root="${CONTACTDIFF_OUTPUT_ROOT:-${project_root}/outputs/multidex_filtered_threehand_${object_key}_32x4_videos_lowmem}"
robots_string="${CONTACTDIFF_ROBOTS:-barrett shadowhand ezgripper}"
read -r -a robots <<<"${robots_string}"

mkdir -p "${output_root}/logs" "${output_root}/status"
printf 'running\n' >"${output_root}/status/all.status"

for robot_name in "${robots[@]}"; do
  hand_root="${output_root}/${robot_name}"
  selection="${output_root}/selection_${robot_name}.json"
  mkdir -p "${hand_root}/logs" "${hand_root}/status"
  printf 'running\n' >"${hand_root}/status/all.status"

  if [[ ! -s "${selection}" ]]; then
    "${selection_python}" \
      "${project_root}/scripts/prepare_multidex_filtered_shadowhand_seen48_selection.py" \
      --robot-name "${robot_name}" --object-key "${object_key}" \
      --filtered "${dro_root}/data/MultiDex_filtered/${robot_name}/${robot_name}.pt" \
      --raw "${dro_root}/data/MultiDex/${robot_name}/${robot_name}.pt" \
      --samples-per-object 32 --output "${selection}" \
      >"${hand_root}/logs/prepare_selection.log" 2>&1
  fi

  mapfile -t objects < <("${selection_python}" -c \
    'import json,sys; print("\n".join(x["object_name"] for x in json.load(open(sys.argv[1]))["objects"]))' \
    "${selection}")

  : >"${hand_root}/logs/run.log"
  for object_name in "${objects[@]}"; do
    "${runner}" \
      "${project_root}/scripts/record_multidex_filtered_shadowhand_seen48.py" \
      --dro-root "${dro_root}" \
      --robot-name "${robot_name}" --selection-file "${selection}" \
      --only-object "${object_name}" --output-root "${hand_root}" \
      --samples-per-object 32 --samples-per-video 8 --resume \
      --width 256 --height 192 --video-fps 15 --video-stride 7 "$@" \
      >>"${hand_root}/logs/run.log" 2>&1
  done

  "${selection_python}" \
    "${project_root}/scripts/summarize_multidex_filtered_shadowhand_seen48_videos.py" \
    --selection "${selection}" --output-root "${hand_root}" \
    >"${hand_root}/logs/summary.log" 2>&1
  printf 'complete\n' >"${hand_root}/status/all.status"
done

"${selection_python}" \
  "${project_root}/scripts/summarize_multidex_filtered_threehand_videos.py" \
  --output-root "${output_root}" --robots "${robots[@]}" \
  >"${output_root}/logs/summary.log" 2>&1
printf 'complete\n' >"${output_root}/status/all.status"
