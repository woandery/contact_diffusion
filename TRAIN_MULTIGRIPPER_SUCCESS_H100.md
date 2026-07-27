# H100 MultiGripper success-only training

This setup trains one ContactDiffusion model from successful grasps across the
10 MultiGripperGrasp executors whose native contact counts are 2, 3, or 5.
Allegro is excluded because its native contact count is 4. The model is conditioned on the object point
cloud and native contact count, not on a gripper identity. Executors sharing a
contact count therefore contribute to one conditional mixture distribution.

## Included data

| Native contacts | Executors | Successful rows before the train/val/test split |
| --- | --- | ---: |
| 2 | fetch_gripper, franka_panda, h5_hand, sawyer, wsg_50 | 6,721,304 |
| 3 | Barrett, jaco_robot, robotiq_3finger | 4,657,641 |
| 5 | HumanHand, shadow_hand | 479,356 |

The configuration uses `success_only: true`, `native_n_filter: true`, and
uniform sampling over `n=[2,3,5]`. It deliberately leaves
`max_projection_distance` unset because a single global threshold would remove
very different fractions from different executors.

## 1. Enter an H100 allocation

From the platform, request one or more H100 GPUs and then verify the allocation:

```bash
nvidia-smi --query-gpu=index,name,memory.total --format=csv
```

The SSH node used while creating these files exposed four RTX 4090 GPUs, so the
full run has not been benchmarked on an H100 allocation yet.

## 2. Install the H100 PointNet++ extension

Run this inside the same Python environment that will launch training:

```bash
cd /inspire/qb-ilm2/project/zhanghanbo/public/mck/contact_diffusion
pip install -r requirements.txt
TORCH_CUDA_ARCH_LIST=9.0 bash scripts/install_pointnet2_ops.sh
python -c "from pointnet2_ops.pointnet2_modules import PointnetSAModule; print('PointNet++ OK')"
```

## 3. Build manifest caches once

The success filter must scan roughly 34 GB of JSONL manifests. Build its offset
caches once in a single process before starting DDP, so every rank does not scan
the same manifests concurrently:

```bash
python scripts/prepare_contactdiffusion_data_cache.py \
  --config configs/contact_diffusion_contact_format_multigripper_success_h100.yaml \
  --splits train val test
```

The generated files live under
`.cache/contactdiffusion/multigripper_success_h100/`. Re-run this step only after
the manifests or filter settings change.

## 4. Start training

Single H100:

```bash
NUM_GPUS=1 bash scripts/train_multigripper_success_h100.sh
```

Eight H100 GPUs:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 NUM_GPUS=8 \
  bash scripts/train_multigripper_success_h100.sh
```

`train.batch_size=256` is per GPU, so eight GPUs produce a global batch of 2048.
If the H100 allocation has less than 80 GB usable memory or PointNet++ reports
an out-of-memory error, lower it first to 192 and then to 128 before changing the model.

For a persistent background run:

```bash
mkdir -p outputs/contact_diffusion_multigripper_success_h100_n235
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 NUM_GPUS=8 nohup \
  bash scripts/train_multigripper_success_h100.sh \
  > outputs/contact_diffusion_multigripper_success_h100_n235/train.log 2>&1 &
```

Monitor it with:

```bash
tail -f outputs/contact_diffusion_multigripper_success_h100_n235/train.log
tensorboard --logdir outputs/contact_diffusion_multigripper_success_h100_n235/tensorboard
```

Checkpoints are saved every 5,000 steps under
`outputs/contact_diffusion_multigripper_success_h100_n235/checkpoints/`.

## Resume

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 NUM_GPUS=8 \
  bash scripts/train_multigripper_success_h100.sh \
  --resume outputs/contact_diffusion_multigripper_success_h100_n235/checkpoints/latest.pt
```

The launcher enables TF32, NCCL asynchronous error reporting, and expandable
CUDA allocator segments. The current trainer remains FP32/TF32; it does not use
BF16 autocast because the bundled PointNet++ extension has not been validated
for BF16 in this environment.
