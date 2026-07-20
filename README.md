# ContactDiffusion

Standalone contact-point diffusion project extracted from the mixed dex grasp workspace.

This project does not import `GraspGen`, `GenDexGrasp`, `SeqMultiGrasp`, `dex-urdf`, or `third_party` code.  The default object encoder is `simple_pointnet`, a lightweight local-token MLP encoder that preserves the Transformer cross-attention path without requiring PointNet++ CUDA extensions.

## Included

- Contact diffusion model
- Object point-cloud encoder
- Contact-to-object cross attention
- Transformer denoiser
- DDPM/DDIM sampling
- Chamfer set loss
- V0 `.npz` dataset loader
- Training and sampling scripts
- Minimal forward test

## Install

```bash
cd ContactDiffusion
pip install -r requirements.txt
```

Or with conda:

```bash
conda env create -f environment.yml
conda activate contactdiff
```

## Dataset format

Expected layout:

```text
dataset_root/
  train/n3/sample_00000000.npz
  val/n3/sample_00000000.npz
  test/n3/sample_00000000.npz
```

Each `.npz` must contain:

- `object_pc`: `(2048, 3)` float32
- `contacts`: `(n, 3)` float32
- `num_contacts`: scalar int
- `selected_indices`: `(n,)` int
- `object_name`: scalar string
- `robot_name`: scalar string

Large Contact Format datasets can be read directly from the manifest/shard layout
described in `/home/zhb1/mck/dexgrasp/README_dataset.md`. Use
`dataset.type: contact_format`, set `dataset.root_dir` to the `graspdata_end`
root, and set `dataset.dataset_dir` to the relative Contact Format bucket such
as `contact_format/v0/multidex/by_hand/barrett`.

## Run

Forward test:

```bash
cd ContactDiffusion
python tests/test_forward.py
```

Train with the default Barrett n=3 config:

```bash
cd ContactDiffusion
python train.py
```

Train directly from the large MultiDex Barrett Contact Format dataset:

```bash
python train.py --config configs/contact_diffusion_contact_format_multidex_barrett_n3.yaml
```

On a 4-GPU node, launch distributed training with `torchrun`:

```bash
torchrun --nproc_per_node=4 train.py \
  --config configs/contact_diffusion_contact_format_multidex_barrett_n3.yaml
```

Object encoder choices:

- `object_encoder_type: simple_pointnet` uses the standalone MLP token encoder.
  `object_num_tokens` controls how many sampled object points become tokens.
- `object_encoder_type: pointnet` uses the GraspGen PointNet++ local-token
  encoder and requires GraspGen's `pointnet2_ops` extension. Its local-token
  hierarchy is controlled by `pointnet_local_npoints`, for example
  `[256, 128, 64]` returns 64 final object tokens.

Override dataset path:

```bash
python train.py --dataset_dir /path/to/contact_dataset
```

Sample from a checkpoint:

```bash
python sample.py \
  --checkpoint outputs/contact_diffusion_barrett_n3/checkpoints/latest.pt \
  --n 3 \
  --output_dir outputs/samples
```

## Notes

- The original PointNet++ encoder was intentionally not migrated because it depends on GraspGen modules and local CUDA extensions.
- To preserve independence, all imports are local package imports such as `from models import ...` and `from datasets.contact_dataset import ...`.
