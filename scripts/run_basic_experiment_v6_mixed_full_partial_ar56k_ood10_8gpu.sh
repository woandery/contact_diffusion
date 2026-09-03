#!/usr/bin/env bash
# Frozen v6: mixed-full/partial AR56k, full-cloud inference, v5 FK/EAWQ budget.
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
bundled_checkpoint="${project_root}/weights/v6/step_00056000.pt"
remote_checkpoint="/inspire/qb-ilm2/project/zhanghanbo/public/mck/ContactDiffusion/outputs/contact_ar_success_k128_mixedfullpartial_normfull_from8k_lr2e4_gb768_56k_4x4090/model/checkpoints/step_00056000.pt"
if [[ -f "${bundled_checkpoint}" ]]; then
  default_checkpoint="${bundled_checkpoint}"
else
  default_checkpoint="${remote_checkpoint}"
fi

export CONTACT_AR_BASIC_CHECKPOINT="${CONTACT_V6_CHECKPOINT:-${default_checkpoint}}"
export CONTACT_AR_BASIC_EXPECTED_CHECKPOINT_SHA="05bd542bb0c10a74ed8dafd7586a58ceedb17097dd08eb344011bb91e98516b4"
export CONTACT_AR_BASIC_PROTOCOL="${project_root}/configs/basic_experiment_mixed_full_partial_ar56k_eawq_o10i20_palm0_v6_protocol.yaml"
export CONTACT_AR_BASIC_SAMPLES=32
export CONTACT_AR_BASIC_OBSERVATION=full
export CONTACT_AR_BASIC_RUN_ROOT="${CONTACT_V6_RUN_ROOT:-${project_root}/outputs/basic_experiment_mixed_full_partial_ar56k_eawq_o10i20_palm0_v6_ood10}"

exec bash "${project_root}/scripts/run_contact_ar64k_basic_ood10_8gpu.sh"
