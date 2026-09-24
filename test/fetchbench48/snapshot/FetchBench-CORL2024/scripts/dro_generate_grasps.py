#!/usr/bin/env python3
"""Generate D(R,O) grasp candidates for a FetchBench target point cloud."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


BARRETT_MIMIC_JOINTS = {
    "bh_j21_joint": ("bh_j11_joint", 1.0),
    "bh_j13_joint": ("bh_j12_joint", 0.344262295082),
    "bh_j23_joint": ("bh_j22_joint", 0.344262295082),
    "bh_j33_joint": ("bh_j32_joint", 0.344262295082),
}


def _enforce_barrett_mimic(q: torch.Tensor, joint_names: list[str]) -> torch.Tensor:
    """Apply the mechanical coupling declared in the Barrett URDF.

    Both pytorch_kinematics and Isaac Gym expose URDF mimic joints as ordinary
    independent DOFs. D(R,O) consequently predicts non-physical values for
    them unless the coupling is restored explicitly.
    """
    q = q.clone()
    indices = {name: 6 + index for index, name in enumerate(joint_names)}
    for child, (parent, multiplier) in BARRETT_MIMIC_JOINTS.items():
        q[indices[child]] = q[indices[parent]] * multiplier
    return q


def _clamp_joint_limits(
    q: torch.Tensor, pk_chain, margin: float = 1.0e-5
) -> torch.Tensor:
    """Keep every value strictly inside its URDF interval."""
    lower, upper = pk_chain.get_joint_limits()
    lower = torch.as_tensor(lower, dtype=q.dtype, device=q.device)
    upper = torch.as_tensor(upper, dtype=q.dtype, device=q.device)
    safe_lower = lower + margin
    safe_upper = upper - margin
    return torch.maximum(torch.minimum(q, safe_upper), safe_lower)


def _link_positions(pk_chain, q: torch.Tensor) -> dict[str, list[float]]:
    """Serialize link-frame origins for an Isaac-vs-D(R,O) FK audit."""
    status = pk_chain.forward_kinematics(q.unsqueeze(0))
    return {
        name: transform.get_matrix()[0, :3, 3].detach().cpu().numpy().tolist()
        for name, transform in status.items()
    }


def _sample_points(points: np.ndarray, count: int, rng: np.random.Generator) -> np.ndarray:
    if len(points) == 0:
        raise ValueError("Target point cloud is empty")
    replace = len(points) < count
    indices = rng.choice(len(points), size=count, replace=replace)
    return points[indices]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dro-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hand", choices=("barrett", "shadowhand"), required=True)
    parser.add_argument("--candidates", type=int, default=8)
    parser.add_argument("--points", type=int, default=512)
    parser.add_argument("--optimization-steps", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    dro_root = args.dro_root.resolve()
    sys.path.insert(0, str(dro_root))
    from model.network import create_network
    from utils.controller import controller
    from utils.hand_model import create_hand_model
    from utils.multilateration import multilateration
    from utils.optimization import create_problem, optimization, process_transform
    from utils.se3_transform import compute_link_pose

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    requested_device = args.device
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        print(f"CUDA unavailable; falling back from {requested_device} to CPU")
        requested_device = "cpu"
    device = torch.device(requested_device)

    raw_pc = np.asarray(np.load(args.input), dtype=np.float32).reshape(-1, 3)
    raw_pc = raw_pc[np.isfinite(raw_pc).all(axis=1)]
    raw_pc = np.ascontiguousarray(raw_pc)
    input_sha256 = hashlib.sha256(raw_pc.tobytes()).hexdigest()
    center = raw_pc.mean(axis=0)
    object_pc_np = _sample_points(raw_pc - center, args.points, rng)
    object_pc = torch.from_numpy(object_pc_np).to(device).unsqueeze(0)

    network = create_network(
        SimpleNamespace(
            emb_dim=512,
            latent_dim=64,
            pretrain=None,
            center_pc=True,
            block_computing=True,
        ),
        mode="validate",
    ).to(device)
    state = torch.load(args.checkpoint, map_location=device)
    network.load_state_dict(state)
    network.eval()

    hand = create_hand_model(args.hand, device=device, num_points=args.points)
    joint_names = hand.pk_chain.get_joint_parameter_names()[6:]
    layer = None
    records = []
    for candidate_idx in range(args.candidates):
        initial_q = hand.get_initial_q().unsqueeze(0).to(device)
        robot_pc = hand.get_transformed_links_pc(initial_q[0])[:, :3].unsqueeze(0)
        with torch.no_grad():
            dro = network(robot_pc, object_pc)["dro"].detach()
            mlat_pc = multilateration(dro, object_pc)
            transform, _ = compute_link_pose(hand.links_pc, mlat_pc, is_train=False)
            optim_transform = process_transform(hand.pk_chain, transform, device=device)
        if layer is None:
            layer = create_problem(hand.pk_chain, optim_transform.keys())
        predict_q = optimization(
            hand.pk_chain,
            layer,
            initial_q,
            optim_transform,
            n_iter=args.optimization_steps,
        )[0]
        if args.hand == "barrett":
            predict_q = _enforce_barrett_mimic(predict_q, joint_names)
        # The upstream controller constructs several constant direction vectors
        # on CPU.  Keep this small geometric post-process on CPU so CUDA
        # inference does not hit a mixed-device matmul, then restore the
        # inference device for point-cloud transforms and serialization.
        outer_q, inner_q = controller(args.hand, predict_q.detach().cpu())
        outer_q = outer_q.to(device)
        inner_q = inner_q.to(device)
        if args.hand == "barrett":
            outer_q = _enforce_barrett_mimic(outer_q, joint_names)
            inner_q = _enforce_barrett_mimic(inner_q, joint_names)

        # Undo object centering. Axes are unchanged, so only root translation
        # needs the centroid added back.
        for q in (predict_q, outer_q, inner_q):
            q[:3] += torch.as_tensor(center, dtype=q.dtype, device=q.device)

        predict_q = _clamp_joint_limits(predict_q, hand.pk_chain)
        outer_q = _clamp_joint_limits(outer_q, hand.pk_chain)
        inner_q = _clamp_joint_limits(inner_q, hand.pk_chain)

        predict_pc = hand.get_transformed_links_pc(predict_q)[:, :3]
        outer_pc = hand.get_transformed_links_pc(outer_q)[:, :3]
        inner_pc = hand.get_transformed_links_pc(inner_q)[:, :3]
        records.append(
            {
                "candidate": candidate_idx,
                "predict_q": predict_q.detach().cpu().numpy().tolist(),
                "outer_q": outer_q.detach().cpu().numpy().tolist(),
                "inner_q": inner_q.detach().cpu().numpy().tolist(),
                "predict_pc": predict_pc.detach().cpu().numpy().tolist(),
                "outer_pc": outer_pc.detach().cpu().numpy().tolist(),
                "inner_pc": inner_pc.detach().cpu().numpy().tolist(),
                "predict_link_positions": _link_positions(hand.pk_chain, predict_q),
                "outer_link_positions": _link_positions(hand.pk_chain, outer_q),
                "inner_link_positions": _link_positions(hand.pk_chain, inner_q),
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "hand": args.hand,
                "joint_names": joint_names,
                "barrett_mimic_enforced": args.hand == "barrett",
                "joint_limits_clamped": True,
                "joint_limit_margin": 1.0e-5,
                "pose_pointclouds_serialized": True,
                "source_points": int(len(raw_pc)),
                "sampled_points": args.points,
                "optimization_steps": args.optimization_steps,
                "seed": args.seed,
                "checkpoint": str(args.checkpoint.resolve()),
                "input_sha256": input_sha256,
                "center": center.tolist(),
                "records": records,
            }
        )
    )
    print(args.output)


if __name__ == "__main__":
    main()
