#!/usr/bin/env bash
# Run the frozen seen-48, 128-set x 128-particle, 10-repeat protocol on 8 GPUs.
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mck_root="$(cd "${project_root}/.." && pwd)"
python_path="${CONTACTDIFF_PYTHON:-${mck_root}/miniconda3/envs/contactdiff_fk_localalign/bin/python}"
isaac_runner="${CONTACTDIFF_ISAAC_RUNNER:-${project_root}/scripts/run_remote_isaacgym_python.sh}"
source_root="${CONTACTDIFF_SOURCE_RUN_ROOT:-${project_root}/outputs/balanced50k_seen48_128set_128particle_physx_r1000_both_v1}"
run_root="${CONTACTDIFF_RUN_ROOT:-${project_root}/outputs/balanced50k_seen48_128set_128particle_physx_r10_both_8x4090_v1}"
source_assets="${project_root}/outputs/basic_experiment_balanced_n35_step50k_seen48_top1_o10i20_v1_remote4090/provenance/assets"
object_map="${source_assets}/object_map.json"
prepared_root="${source_root}/prepared"
repeats="${CONTACTDIFF_STABILITY_REPEATS:-10}"
batch_size="${CONTACTDIFF_STABILITY_BATCH_SIZE:-512}"
num_gpus="${CONTACTDIFF_NUM_GPUS:-8}"
status_root="${run_root}/supervisor/stability"

if [[ "$(find "${prepared_root}" -mindepth 2 -maxdepth 2 -name '*.json' | wc -l)" -ne 96 ]]; then
  echo "Expected 96 prepared manifests under ${prepared_root}" >&2
  exit 2
fi
if [[ "$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)" -lt "${num_gpus}" ]]; then
  echo "Fewer than ${num_gpus} GPUs are visible" >&2
  exit 2
fi

mkdir -p "${status_root}" "${run_root}/streaming" "${run_root}/provenance"
cp "${source_root}/provenance/protocol.json" "${run_root}/provenance/source_protocol.json"
printf '%s\n' "${source_root}" >"${run_root}/provenance/prepared_source_root"
date -u +%Y-%m-%dT%H:%M:%SZ >"${status_root}/started_at_utc"
printf 'running_%s_trials_on_%s_gpus\n' "$((2 * 48 * 128 * 128 * repeats))" "${num_gpus}" >"${status_root}/status"

pids=()
for ((gpu=0; gpu<num_gpus; gpu++)); do
  CUDA_VISIBLE_DEVICES="${gpu}" "${isaac_runner}" \
    "${project_root}/scripts/run_seen48_physx_streaming_worker.py" \
    --prepared-root "${prepared_root}" --object-map "${object_map}" \
    --output-db "${run_root}/streaming/gpu${gpu}.sqlite" \
    --status "${status_root}/gpu${gpu}.json" --mck-root "${mck_root}" \
    --gpu-index "${gpu}" --num-gpus "${num_gpus}" --repeats "${repeats}" \
    --batch-size "${batch_size}" --expected-samples 16384 \
    >"${status_root}/gpu${gpu}.log" 2>&1 &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then failed=1; fi
done
if ((failed)); then
  printf 'failed\n' >"${status_root}/status"
  exit 1
fi

printf 'summarizing\n' >"${status_root}/status"
"${python_path}" "${project_root}/scripts/summarize_seen48_physx_streaming.py" \
  --run-root "${run_root}" --object-map "${object_map}" \
  --repeats "${repeats}" >"${status_root}/summary.log" 2>&1
printf 'complete\n' >"${status_root}/status"
date -u +%Y-%m-%dT%H:%M:%SZ >"${status_root}/completed_at_utc"
