# ContactDiffusion

Standalone contact-point diffusion project extracted from the mixed dex grasp workspace.

This project does not import `GraspGen`, `GenDexGrasp`, `SeqMultiGrasp`, or
`dex-urdf` at runtime. The default object encoder is `simple_pointnet`, and an
optional bundled PointNet++ CUDA extension is available for
`object_encoder_type: pointnet`.

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

Install the optional PointNet++ CUDA extension from this repository when using
`object_encoder_type: pointnet`:

```bash
cd ContactDiffusion
bash scripts/install_pointnet2_ops.sh
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

Train all MultiGripperGrasp gripper buckets with n=2/3/5:

```bash
torchrun --nproc_per_node=4 train.py \
  --config configs/contact_diffusion_contact_format_multigripper_n235.yaml
```

For a single MultiGripperGrasp gripper, replace `dataset.dataset_dirs` in that
config with one concrete bucket such as
`contact_format/v0/multigrippergrasp/by_gripper/<name>`.

Object encoder choices:

- `object_encoder_type: simple_pointnet` uses the standalone MLP token encoder.
  `object_num_tokens` controls how many sampled object points become tokens.
- `object_encoder_type: pointnet` uses the PointNet++ local-token encoder and
  requires the bundled `pointnet2_ops` extension. Its local-token
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

## Multi-contact grasp quality

`MultiContactGraspEvaluator` scores diffusion-generated `(N, 3)` contact sets
without converting them to a parallel-jaw grasp. It supports arbitrary `N >= 2`,
SDF surface/normal validation, hard- and soft-finger wrench spaces,
Ferrari-Canny epsilon, force closure, target-wrench metrics, and reproducible
Monte Carlo evaluation:

```python
from utils import MultiContactGraspEvaluator

quality = MultiContactGraspEvaluator.evaluate(obj, generated_contacts,
                                              soft_fingers=True)
robust = MultiContactGraspEvaluator.evaluate_robust(
    obj, generated_contacts, soft_fingers=True, num_samples=100, seed=7)
training_label = quality["epsilon"] if quality["valid"] else 0.0
```

See `examples/evaluate_multicontact_grasp.py` for Dex-Net object loading and
2/3/5-contact labels. Nonzero point noise normally moves points off the SDF
surface; robust evaluation therefore requires an explicit trusted
`surface_projection_fn`, or records those samples as invalid.

For point-cloud training data, store an outward unit normal for every point and
evaluate the exact indexed contacts:

```python
from utils import PointCloudMultiContactGraspEvaluator

quality = PointCloudMultiContactGraspEvaluator.evaluate(
    batch["object_pc"][i].cpu().numpy(),
    batch["selected_indices"][i].cpu().numpy(),
    batch["object_normals"][i].cpu().numpy(),
    soft_fingers=True,
)
```

This path always gathers `object_pc[selected_indices]`; it does not project a
free 3-D prediction with nearest-neighbour search. Optional `object_normals`
fields are loaded and collated by the dataset readers. See
`examples/evaluate_point_cloud_contacts.py` for one-time PCA normal estimation,
robust evaluation, and scalar diffusion labels. Centroid-based normal
orientation and point-cloud-mean COM assume a complete object cloud; provide
pre-oriented normals and a physical COM for partial or single-view clouds.
Dataset `selected_indices` label the dataset contacts, not arbitrary continuous
diffusion predictions. Do not use ground-truth indices to score a different
predicted contact set; generated contacts must themselves be discrete indices,
or must first pass through an explicit, separately validated surface projector.

Training validation now performs that projection explicitly. It generates a
small deterministic DDIM sample, uses unique Hungarian point-cloud assignment,
rejects matches beyond the configured point-spacing gate, and writes the
following TensorBoard series per contact count and as an n-mean:

- `val/grasp_projection_valid_rate_*`
- `val/grasp_quality_valid_rate_*`
- `val/grasp_force_closure_rate_*` and its valid-only variant
- `val/grasp_epsilon_mean_*`, valid-only mean, and valid-only p10
- projection-distance mean/max and precomputed-normal usage rate

Configure its cost and contact model under `validation.grasp_quality`. The
all-prediction closure/epsilon metrics count rejected predictions as zero;
valid-only metrics are diagnostic and should not be used alone for checkpoint
selection. If `object_normals` are absent, validation estimates them with local
PCA; set `require_precomputed_normals: true` when centroid-oriented normals are
not appropriate for partial point clouds.

## Notes

- PointNet++ is optional. Use `simple_pointnet` when the CUDA extension is not installed.
- Runtime imports stay local to this project or to the bundled `pointnet2_ops` package.
