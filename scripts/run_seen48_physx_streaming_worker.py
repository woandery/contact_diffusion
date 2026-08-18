#!/usr/bin/env python3
"""Stream repeated seen-48 GPU PhysX trials into a resumable SQLite aggregate."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import random
import sqlite3
import sys
import time


HANDS = ("barrett", "shadowhand")


def parse_worker_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-root", type=Path, required=True)
    parser.add_argument("--object-map", type=Path, required=True)
    parser.add_argument("--output-db", type=Path, required=True)
    parser.add_argument("--status", type=Path, required=True)
    parser.add_argument("--mck-root", type=Path, required=True)
    parser.add_argument("--gpu-index", type=int, required=True)
    parser.add_argument("--num-gpus", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--expected-samples", type=int, default=16384)
    parser.add_argument("--seed", type=int, default=20260817)
    parser.add_argument("--max-batches", type=int)
    parser.add_argument("--only-hand", choices=HANDS)
    parser.add_argument("--only-object")
    return parser.parse_args()


def repeat_seed(base_seed: int, repeat: int, hand: str, object_id: str) -> int:
    payload = f"{base_seed}:{repeat}:{hand}:{object_id}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def open_database(path: Path, metadata: dict) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, timeout=120.0)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=FULL")
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS batches (
            batch_key TEXT PRIMARY KEY,
            hand TEXT NOT NULL,
            object_id TEXT NOT NULL,
            repeat_index INTEGER NOT NULL,
            batch_index INTEGER NOT NULL,
            trials INTEGER NOT NULL,
            valid_trials INTEGER NOT NULL,
            final_successes INTEGER NOT NULL,
            strict_successes INTEGER NOT NULL,
            elapsed_seconds REAL NOT NULL
        );
        CREATE TABLE IF NOT EXISTS trials (
            hand TEXT NOT NULL,
            object_id TEXT NOT NULL,
            source_index INTEGER NOT NULL,
            candidate_rank INTEGER NOT NULL,
            repeats INTEGER NOT NULL,
            valid_count INTEGER NOT NULL,
            final_success_count INTEGER NOT NULL,
            strict_success_count INTEGER NOT NULL,
            final_displacement_sum REAL NOT NULL,
            final_displacement_sumsq REAL NOT NULL,
            final_displacement_min REAL NOT NULL,
            final_displacement_max REAL NOT NULL,
            maximum_segment_sum REAL NOT NULL,
            PRIMARY KEY (hand, object_id, source_index, candidate_rank)
        );
        CREATE INDEX IF NOT EXISTS batches_repeat_idx
            ON batches(hand, object_id, repeat_index);
        """
    )
    for key, value in metadata.items():
        encoded = json.dumps(value, sort_keys=True)
        row = connection.execute(
            "SELECT value FROM metadata WHERE key=?", (key,)
        ).fetchone()
        if row is not None and row[0] != encoded:
            raise ValueError(f"Database metadata mismatch for {key}: {row[0]} != {encoded}")
        connection.execute(
            "INSERT OR IGNORE INTO metadata(key,value) VALUES (?,?)",
            (key, encoded),
        )
    connection.commit()
    return connection


def load_validator(project_root: Path):
    path = project_root / "scripts" / "validate_native_shadowhand_isaacgym_oldparams_dro.py"
    spec = importlib.util.spec_from_file_location("seen48_streaming_validator", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validator_args(module, project_root: Path, mck_root: Path, hand: str) -> argparse.Namespace:
    if hand == "barrett":
        hand_root = project_root / "outputs/local_dro_isaacgym_twohands/full/assets/barrett_adagrasp"
        hand_urdf = "model_extended.urdf"
    else:
        hand_root = project_root / "outputs/shadow_dro_failure_visualizations/assets/robot/shadowhand"
        hand_urdf = "shadow_hand_right_extended.urdf"
    arguments = [
        str(project_root / "scripts/validate_native_shadowhand_isaacgym_oldparams_dro.py"),
        "--prepared", str(project_root / "unused_streaming_prepared.json"),
        "--output", str(project_root / "unused_streaming_output.json"),
        "--gendex-root", str(mck_root / "GenDexGrasp"),
        "--native-hand-root", str(hand_root),
        "--native-hand-urdf", hand_urdf,
        "--native-hand-urdf-is-extended",
        "--device-id", "0",
        "--progress-every", "0",
        "--asset-profile", "dro",
        "--object-source", "dro",
        "--dro-object-root", str(mck_root / "dro_grasp_reproduction/DRO-Grasp/data/data_urdf/object"),
        "--steps-per-second", "100",
        "--substeps", "2",
        "--closure-steps", "100",
        "--direction-seconds", "1.0",
        "--direction-order", "cedex",
        "--success-mode", "final",
        "--threshold", "0.02",
        "--acceleration", "0.5",
        "--robot-friction", "3",
        "--object-friction", "3",
        "--object-density", "500",
        "--object-linear-damping", "-1",
        "--object-angular-damping", "-1",
        "--joint-stiffness", "1000",
        "--joint-damping", "200",
        "--joint-armature", "-1",
        "--joint-velocity", "-1",
        "--pregrasp-open-fraction", "0",
        "--closure-overdrive-fraction", "0",
        "--outer-settle-steps", "0",
        "--virtual-root-stiffness", "1000",
        "--virtual-root-damping", "200",
        "--solver-position-iterations", "8",
        "--solver-velocity-iterations", "0",
        "--contact-offset", "0.01",
        "--rest-offset", "0",
        "--no-ground",
    ]
    previous = sys.argv
    try:
        sys.argv = arguments
        parsed = module.parse_args()
    finally:
        sys.argv = previous
    parsed.sample_start = 0
    parsed.max_samples_per_object = None
    return parsed


def batch_complete(connection: sqlite3.Connection, batch_key: str) -> bool:
    return connection.execute(
        "SELECT 1 FROM batches WHERE batch_key=?", (batch_key,)
    ).fetchone() is not None


def commit_batch(
    connection: sqlite3.Connection,
    batch_key: str,
    hand: str,
    object_id: str,
    repeat: int,
    batch_index: int,
    samples: list[dict],
    results: list[dict],
    elapsed: float,
) -> None:
    if len(samples) != len(results):
        raise ValueError(f"Batch result length mismatch: {len(samples)} != {len(results)}")
    with connection:
        if batch_complete(connection, batch_key):
            return
        valid_count = 0
        final_count = 0
        strict_count = 0
        for sample, row in zip(samples, results):
            if (
                int(sample["source_index"]) != int(row["source_index"])
                or int(sample.get("candidate_rank", 0)) != int(row.get("candidate_rank", 0))
            ):
                raise ValueError("Validator result order differs from prepared batch order")
            valid = bool(row.get("valid_simulation", False))
            final = valid and row.get("final_success") is True
            strict = valid and row.get("strict_six_direction_success") is True
            displacement = float(row["final_displacement_m"])
            maximum_segment = float(row["maximum_segment_displacement_m"])
            if not math.isfinite(displacement) or not math.isfinite(maximum_segment):
                valid = final = strict = False
            valid_count += int(valid)
            final_count += int(final)
            strict_count += int(strict)
            connection.execute(
                """
                INSERT INTO trials(
                    hand,object_id,source_index,candidate_rank,repeats,valid_count,
                    final_success_count,strict_success_count,final_displacement_sum,
                    final_displacement_sumsq,final_displacement_min,
                    final_displacement_max,maximum_segment_sum
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(hand,object_id,source_index,candidate_rank) DO UPDATE SET
                    repeats=repeats+1,
                    valid_count=valid_count+excluded.valid_count,
                    final_success_count=final_success_count+excluded.final_success_count,
                    strict_success_count=strict_success_count+excluded.strict_success_count,
                    final_displacement_sum=final_displacement_sum+excluded.final_displacement_sum,
                    final_displacement_sumsq=final_displacement_sumsq+excluded.final_displacement_sumsq,
                    final_displacement_min=min(final_displacement_min,excluded.final_displacement_min),
                    final_displacement_max=max(final_displacement_max,excluded.final_displacement_max),
                    maximum_segment_sum=maximum_segment_sum+excluded.maximum_segment_sum
                """,
                (
                    hand, object_id, int(sample["source_index"]),
                    int(sample.get("candidate_rank", 0)), 1, int(valid), int(final),
                    int(strict), displacement, displacement * displacement,
                    displacement, displacement, maximum_segment,
                ),
            )
        connection.execute(
            """
            INSERT INTO batches(
                batch_key,hand,object_id,repeat_index,batch_index,trials,
                valid_trials,final_successes,strict_successes,elapsed_seconds
            ) VALUES (?,?,?,?,?,?,?,?,?,?)
            """,
            (
                batch_key, hand, object_id, repeat, batch_index, len(results),
                valid_count, final_count, strict_count, elapsed,
            ),
        )


def main() -> None:
    args = parse_worker_args()
    project_root = Path(__file__).resolve().parents[1]
    prepared_root = args.prepared_root.resolve()
    execution_protocol_path = (
        prepared_root.parent / "provenance" / "protocol.json"
    ).resolve()
    if not execution_protocol_path.is_file():
        raise FileNotFoundError(execution_protocol_path)
    object_map = json.loads(args.object_map.resolve().read_text(encoding="utf-8"))
    objects = list(object_map)
    jobs = [(hand, object_id) for hand in HANDS for object_id in objects]
    if args.only_hand:
        jobs = [job for job in jobs if job[0] == args.only_hand]
    if args.only_object:
        jobs = [job for job in jobs if job[1] == args.only_object]
    if not args.only_hand and not args.only_object:
        jobs = jobs[args.gpu_index :: args.num_gpus]
    if not jobs:
        raise ValueError("No jobs assigned to this worker")

    metadata = {
        "schema": "contactdiff-seen48-physx-streaming-db-v1",
        "gpu_index": args.gpu_index,
        "num_gpus": args.num_gpus,
        "repeats": args.repeats,
        "batch_size": args.batch_size,
        "expected_samples": args.expected_samples,
        "seed": args.seed,
        "jobs": jobs,
    }
    connection = open_database(args.output_db.resolve(), metadata)
    module = load_validator(project_root)
    execution_protocol_sha256 = module.sha256(execution_protocol_path)
    module.OBJECT_MAP = {
        str(name): (str(parts[0]), str(parts[1]))
        for name, parts in object_map.items()
    }
    gym = module.gymapi.acquire_gym()
    parsed_by_hand = {
        hand: validator_args(module, project_root, args.mck_root.resolve(), hand)
        for hand in HANDS
    }

    batches_per_repeat = math.ceil(args.expected_samples / args.batch_size)
    total_batches = len(jobs) * args.repeats * batches_per_repeat
    completed = int(connection.execute("SELECT count(*) FROM batches").fetchone()[0])
    launched = 0
    started = time.monotonic()
    for hand, object_id in jobs:
        prepared_path = prepared_root / hand / f"{object_id}.json"
        prepared = json.loads(prepared_path.read_text(encoding="utf-8"))
        # The 128x128 preparation path uses explicit closure overrides, so the
        # generic preparer cannot also attach --execution-protocol. Bind the
        # immutable run protocol here before applying the frozen validator.
        # This changes provenance metadata only; candidate poses and physics
        # parameters remain exactly as materialized in the prepared manifest.
        if not prepared.get("execution_protocol_config"):
            prepared["execution_protocol_config"] = str(
                execution_protocol_path
            )
        if not prepared.get("execution_protocol_config_sha256"):
            prepared[
                "execution_protocol_config_sha256"
            ] = execution_protocol_sha256
        module.validate_frozen_basic_protocol(prepared, parsed_by_hand[hand])
        if len(prepared.get("objects", [])) != 1:
            raise ValueError(f"{prepared_path}: expected one object group")
        source_group = prepared["objects"][0]
        samples = list(source_group["samples"])
        if len(samples) != args.expected_samples:
            raise ValueError(
                f"{prepared_path}: {len(samples)} samples != {args.expected_samples}"
            )
        keys = {
            (int(sample["source_index"]), int(sample.get("candidate_rank", 0)))
            for sample in samples
        }
        if len(keys) != args.expected_samples:
            raise ValueError(f"{prepared_path}: duplicate trial keys")
        for repeat in range(args.repeats):
            order = list(range(len(samples)))
            random.Random(repeat_seed(args.seed, repeat, hand, object_id)).shuffle(order)
            for batch_index, start in enumerate(range(0, len(order), args.batch_size)):
                batch_key = f"{hand}:{object_id}:r{repeat:04d}:b{batch_index:04d}"
                if batch_complete(connection, batch_key):
                    continue
                selected = [samples[index] for index in order[start : start + args.batch_size]]
                group = {
                    "object_name": object_id,
                    "object_mesh": source_group["object_mesh"],
                    "samples": selected,
                    "_joint_names": list(prepared["joint_names"]),
                }
                batch_started = time.monotonic()
                results = module.validate_object(
                    gym, parsed_by_hand[hand], group
                )
                elapsed = time.monotonic() - batch_started
                commit_batch(
                    connection, batch_key, hand, object_id, repeat,
                    batch_index, selected, results, elapsed,
                )
                completed += 1
                launched += 1
                runtime = time.monotonic() - started
                rate = launched / runtime if runtime else 0.0
                remaining = total_batches - completed
                atomic_json(
                    args.status.resolve(),
                    {
                        "status": "running",
                        "gpu_index": args.gpu_index,
                        "hand": hand,
                        "object_id": object_id,
                        "repeat": repeat,
                        "batch": batch_index,
                        "completed_batches": completed,
                        "total_batches": total_batches,
                        "completed_trials": int(
                            connection.execute("SELECT coalesce(sum(trials),0) FROM batches").fetchone()[0]
                        ),
                        "batch_elapsed_seconds": elapsed,
                        "session_batch_rate": rate,
                        "session_eta_seconds": remaining / rate if rate else None,
                        "updated_at_unix": time.time(),
                    },
                )
                if args.max_batches is not None and launched >= args.max_batches:
                    connection.close()
                    return
    atomic_json(
        args.status.resolve(),
        {
            "status": "complete",
            "gpu_index": args.gpu_index,
            "completed_batches": completed,
            "total_batches": total_batches,
            "updated_at_unix": time.time(),
        },
    )
    connection.close()


if __name__ == "__main__":
    main()
