#!/usr/bin/env python3
"""Minimal forward-shape test.

Run from ContactDiffusion/:

    python tests/test_forward.py
"""

from pathlib import Path
import sys

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models import ContactSetDenoiser  # noqa: E402


def main():
    batch_size = 2
    num_points = 2048
    num_contacts = 3
    object_pc = torch.randn(batch_size, num_points, 3)
    contact_noise = torch.randn(batch_size, num_contacts, 3)
    timesteps = torch.randint(0, 1000, (batch_size,))

    model = ContactSetDenoiser(
        dc=3,
        d_model=64,
        num_layers=2,
        num_heads=4,
        num_diffusion_iters=1000,
        n_values=[num_contacts],
        object_encoder_type="simple_pointnet",
    )
    out = model(contact_noise, timesteps, object_pc, num_contacts)
    assert out.shape == (batch_size, num_contacts, 3), out.shape
    print(f"OK: output shape {tuple(out.shape)}")


if __name__ == "__main__":
    main()
