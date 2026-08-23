# FK contact-energy guidance diagnostic

This follow-up keeps the completed contact-set guidance A/B experiment frozen and
tests whether the FK optimizer actually uses the diffusion target contacts.

## Frozen controls

- MF50 checkpoint, OOD-10 objects, Barrett/ShadowHand, DDIM50 contacts;
- 64 contact sets per object/hand, 32 particles per set, 400 FK steps;
- identical diffusion contacts, sample seeds, and FK initial states;
- identical non-contact FK terms, candidate count, D(R,O) assets, O10/I20
  closure, GPU PhysX, and final/strict success definitions.

The only changed FK term is the contact Chamfer weight:

```text
contact_w000:   0
contact_w025:  25
contact_w100_4090: 100  (same-hardware paired reference)
contact_w200: 200
```

The completed diffusion-vs-matched-random A/B result remains a separate report.
Its FK candidates were generated on H100, so it is not mixed into the byte-exact
4090 initialization audit used by this one-factor weight sweep.

`contact_w000` means **no contact-Chamfer energy**. It is not a fully
contact-independent pipeline: diffusion contacts are deliberately retained for
the shared enveloping initialization and other frozen target-derived selection
geometry. This isolates deletion of the Chamfer energy instead of changing both
the initialization and objective at once.

All four weight arms are generated on the same 4090 platform. FK uses an explicit
per-sample generator; the audit requires their initialization tensor hashes to
match exactly. The earlier H100 w100 candidates cannot serve as this byte-exact
reference because Kabsch/SVD floating-point results differ across GPU
architectures even when seeds and contacts match.

## Metrics

- invalid-inclusive final/strict all-particle success;
- success@K for K=1,2,4,8,16,32;
- paired McNemar tests and object-cluster bootstrap confidence intervals against
  `contact_w100`;
- FK `contact_chamfer_m` and `assigned_contact_error_m`, including success/failure
  stratification.

The OOD-10 manifest contains geometry only. The model's seen48 train/validation
lists exclude these ten objects, and the annotated dataset root is absent from
the current compute-platform snapshot. Therefore a dataset-GT contact upper bound
is explicitly recorded as unavailable rather than reconstructed from PhysX
labels after observing outcomes.

## 4 x RTX 4090 launch

```bash
cd /path/to/ContactDiffusionAB
export CONTACTDIFF_PYTHON=/path/to/contactdiff_fk_localalign/bin/python
export CONTACTDIFF_ISAAC_RUNNER="$PWD/scripts/run_remote_isaacgym_python.sh"
export CONTACTDIFF_GPU_IDS=0,1,2,3
nohup bash scripts/run_contact_energy_diagnostic_4gpu.sh \
  > outputs/contact_energy_guidance_diagnostic_palm0_v4_4x4090/launcher.log 2>&1 &
```

The generation stage defaults to one worker per GPU. PhysX uses one worker per
GPU and produces 320 batches (163,840 trials) for the four same-hardware weight
arms. Existing complete arm files are resume-skipped.

Final reports are written to:

```text
outputs/contact_energy_guidance_diagnostic_palm0_v4_4x4090/reports/contact_energy_diagnostic.json
outputs/contact_energy_guidance_diagnostic_palm0_v4_4x4090/reports/CONTACT_ENERGY_DIAGNOSTIC.md
```
