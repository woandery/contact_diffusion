# Basic experiment v6 checkpoint

- File: `step_00056000.pt`
- Size: `86,709,034` bytes
- SHA256: `05bd542bb0c10a74ed8dafd7586a58ceedb17097dd08eb344011bb91e98516b4`
- Model type: `autoregressive_contact_diffusion`
- Training observation: 50% complete point clouds and 50% synthetic partial point clouds
- Inference observation frozen by v6: complete `2048 x 3` point cloud

Source checkpoint:

```text
/inspire/qb-ilm2/project/zhanghanbo/public/mck/ContactDiffusion/outputs/contact_ar_success_k128_mixedfullpartial_normfull_from8k_lr2e4_gb768_56k_4x4090/model/checkpoints/step_00056000.pt
```

The binary is intentionally committed as the sole checkpoint exception in
`.gitignore`. It is below GitHub's 100 MiB per-file hard limit. Do not replace
it in place without updating the v6 protocol, audit, tests and this checksum.
