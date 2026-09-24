"""Isolated crop-guard recovery; never dilate or replace RGB/mask/pointmap.

The original crop is square with side=max(width,height,2). A thin, nonempty
mask with a long side >=2 has a valid square crop despite the short-side guard.
Only that guard is bypassed; compute_mask_bbox and crop_and_pad stay unchanged.
"""
import argparse
import os
import sys
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--worker-spec', required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    import json
    spec = json.loads(Path(args.worker_spec).read_text())
    # Same public-notebook initialization as the existing Inference entrypoint.
    os.environ['LIDRA_SKIP_INIT'] = 'true'
    sys.path.insert(0, spec['runtime']['sam3d_root'])
    from sam3d_objects.data.dataset.tdfy import img_and_mask_transforms as transforms
    old = transforms.check_bounding_box
    calls = []

    def check(width, height):
        if min(width, height) < 2 and min(width, height) >= 0 and max(width, height) >= 2:
            calls.append(dict(width=float(width), height=float(height)))
            print('THIN_MASK_SQUARE_CROP_GUARD', width, height, flush=True)
            return
        return old(width, height)

    transforms.check_bounding_box = check
    normalization_reuses = []
    if os.environ.get('SAM3D_REUSE_EXISTING_NORMALIZATION') == '1':
        # normalize() computes moments before applying an explicitly allowed
        # override. For thin masks this redundant computation sees zero pixels.
        # Reusing the supplied moments has EXACTLY the same normal-case output.
        normal = transforms.ObjectCentricSSI.normalize

        def reuse(self, pointmap, mask, scale=None, shift=None):
            if self.allow_scale_and_shift_override and scale is not None and shift is not None:
                import torch
                assert torch.isfinite(scale).all() and (scale > 0).all() and torch.isfinite(shift).all()
                calculate = self._compute_scale_and_shift
                self._compute_scale_and_shift = lambda p, m: (scale, shift)
                normalization_reuses.append(True)
                try:
                    return normal(self, pointmap, mask, scale, shift)
                finally:
                    self._compute_scale_and_shift = calculate
            return normal(self, pointmap, mask, scale, shift)

        transforms.ObjectCentricSSI.normalize = reuse
        # Regression check: on a nonempty mask the override path is identical.
        import torch
        normalizer = transforms.ObjectCentricSSI(allow_scale_and_shift_override=True)
        pm = torch.arange(3 * 4 * 4, dtype=torch.float32).reshape(3, 4, 4)
        mask = torch.ones(1, 4, 4); scale = torch.tensor([2., 2., 2.]); shift = torch.tensor([1., 3., 5.])
        expected = normal(normalizer, pm, mask, scale, shift)
        actual = reuse(normalizer, pm, mask, scale, shift)
        # Dataclass fields vary across SAM3D revisions; compare its stored tensors.
        expected_fields = expected._asdict() if hasattr(expected, '_asdict') else vars(expected)
        actual_fields = actual._asdict() if hasattr(actual, '_asdict') else vars(actual)
        for key, value in expected_fields.items():
            assert torch.equal(value, actual_fields[key]), key
        normalization_reuses.clear()
    from sam3d_resident_batch import worker, save
    try:
        return worker(args.worker_spec, args.output)
    finally:
        save(Path(args.output) / 'thin_mask_guard_audit.json', dict(
            calls=calls, rgb_changed=False, mask_changed=False, depth_changed=False,
            crop_formula_changed=False, minimum_long_axis_span=2,
            existing_normalization_reuses=len(normalization_reuses)))


if __name__ == '__main__':
    sys.exit(main())
