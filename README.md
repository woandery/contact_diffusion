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
