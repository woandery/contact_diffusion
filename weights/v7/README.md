# Frozen v7 model

- File: `best_val.pt`; embedded step: **32000**
- Size: 86,705,982 bytes
- SHA256: `c55badbd2e1ce7bc9cda9b58832e68003b4b757ee02eedee47e01e697c7ec491`
- Training: 60% FetchBench simulated-camera partial, 20% synthetic crop, 20% full.
- Inference in the basic experiment: full 2048 x 3 cloud.
- Source: `/inspire/qb-ilm2/project/zhanghanbo/public/mck/ContactDiffusion/outputs/contact_ar_success_k128_real60_synth20_full20_from_mixed56k_lr5e5_gb768_32k_4x4090/model/checkpoints/best_val.pt`

On 2026-09-12, all model tensors were verified identical to the same run's
`step_00032000.pt` (SHA256
`614a64db3c8c8612cb3b991488ae979d2a6bc723af0dc57d9e65e570881b9a17`).
Serialization bytes differ. v7 pins the best-val file's hash, not a mutable
remote alias. A replacement must receive a new protocol identity.

The 32k step count refers to this continuation stage, warm-started from the
mixed56k best model. The camera data are simulated observations, not physical
robot camera captures.
