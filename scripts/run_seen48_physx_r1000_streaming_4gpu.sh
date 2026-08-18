#!/usr/bin/env bash
# Wait for 128x128 candidates, then stream 1.57B repeated GPU PhysX trials.
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mck_root="$(cd "${project_root}/.." && pwd)"
python_path="${CONTACTDIFF_PYTHON:-${mck_root}/miniconda3/envs/contactdiff_fk_localalign/bin/python}"
isaac_runner="${CONTACTDIFF_ISAAC_RUNNER:-${project_root}/scripts/run_remote_isaacgym_python.sh}"
run_root="${CONTACTDIFF_RUN_ROOT:-${project_root}/outputs/balanced50k_seen48_128set_128particle_physx_r1000_both_v1}"
source_assets="${project_root}/outputs/basic_experiment_balanced_n35_step50k_seen48_top1_o10i20_v1_remote4090/provenance/assets"
object_map="${source_assets}/object_map.json"
repeats="${CONTACTDIFF_STABILITY_REPEATS:-1000}"
batch_size="${CONTACTDIFF_STABILITY_BATCH_SIZE:-512}"
status_root="${run_root}/supervisor/stability"

mkdir -p "${status_root}" "${run_root}/streaming"
printf 'waiting_for_generation\n' >"${status_root}/status"
while true; do
  generation_status="$(cat "${run_root}/supervisor/status" 2>/dev/null || true)"
  if [[ "${generation_status}" == ready_for_physx_stability_r1000 ]]; then break; fi
  if [[ "${generation_status}" == failed_generation || "${generation_status}" == failed_preparation ]]; then
    printf 'blocked_by_%s\n' "${generation_status}" >"${status_root}/status"
    exit 1
  fi
  sleep 60
done

date -u +%Y-%m-%dT%H:%M:%SZ >"${status_root}/started_at_utc"
printf 'running_1572864000_trials\n' >"${status_root}/status"
pids=()
for gpu in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES="${gpu}" "${isaac_runner}" \
    "${project_root}/scripts/run_seen48_physx_streaming_worker.py" \
    --prepared-root "${run_root}/prepared" --object-map "${object_map}" \
    --output-db "${run_root}/streaming/gpu${gpu}.sqlite" \
    --status "${status_root}/gpu${gpu}.json" --mck-root "${mck_root}" \
    --gpu-index "${gpu}" --num-gpus 4 --repeats "${repeats}" \
    --batch-size "${batch_size}" --expected-samples 16384 \
    >"${status_root}/gpu${gpu}.log" 2>&1 &
  pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do if ! wait "${pid}"; then failed=1; fi; done
if ((failed)); then printf 'failed\n' >"${status_root}/status"; exit 1; fi

printf 'summarizing\n' >"${status_root}/status"
"${python_path}" "${project_root}/scripts/summarize_seen48_physx_streaming.py" \
  --run-root "${run_root}" --object-map "${object_map}" \
  --repeats "${repeats}" >"${status_root}/summary.log" 2>&1
printf 'complete\n' >"${status_root}/status"
date -u +%Y-%m-%dT%H:%M:%SZ >"${status_root}/completed_at_utc"
