#!/usr/bin/env bash
# Frozen v7: FetchBench-trained AR32k, full-cloud inference, v6 FK/EAWQ budget.
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
bundled_checkpoint="${project_root}/weights/v7/best_val.pt"
remote_checkpoint="/inspire/qb-ilm2/project/zhanghanbo/public/mck/ContactDiffusion/outputs/contact_ar_success_k128_real60_synth20_full20_from_mixed56k_lr5e5_gb768_32k_4x4090/model/checkpoints/best_val.pt"
if [[ -f "${bundled_checkpoint}" ]]; then
  default_checkpoint="${bundled_checkpoint}"
else
  default_checkpoint="${remote_checkpoint}"
fi

export CONTACT_AR_BASIC_CHECKPOINT="${CONTACT_V7_CHECKPOINT:-${default_checkpoint}}"
export CONTACT_AR_BASIC_EXPECTED_CHECKPOINT_SHA="c55badbd2e1ce7bc9cda9b58832e68003b4b757ee02eedee47e01e697c7ec491"
export CONTACT_AR_BASIC_PROTOCOL="${project_root}/configs/basic_experiment_fetchbench_ar32k_eawq_o10i20_palm0_v7_protocol.yaml"
export CONTACT_AR_BASIC_SAMPLES=32
export CONTACT_AR_BASIC_OBSERVATION=full
export CONTACT_AR_BASIC_RUN_ROOT="${CONTACT_V7_RUN_ROOT:-${project_root}/outputs/basic_experiment_fetchbench_ar32k_eawq_o10i20_palm0_v7_ood10}"

exec bash "${project_root}/scripts/run_contact_ar64k_basic_ood10_8gpu.sh"
