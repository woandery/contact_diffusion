#!/usr/bin/env python3
"""Run the fixed Shadow/Barrett/Franka friction benchmark across local GPUs."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import subprocess
import threading
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
MCK_ROOT = REPO_ROOT.parent
GRIPPERS = ("shadow_hand", "Barrett", "franka_panda")


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def friction_tag(value: float) -> str:
    return f"{round(value * 100):03d}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidates",
        default="outputs/isaacsim_validation/fixed_three_grippers_candidates_step35000.json",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/isaacsim_validation/fixed_three_grippers_friction_benchmark",
    )
    parser.add_argument("--frictions", nargs="+", type=float, default=[0.1, 0.3, 0.5, 0.8, 1.0])
    parser.add_argument("--gpu-ids", nargs="+", type=int, default=[0, 1, 2, 3])
    parser.add_argument("--candidate-rank", type=int, default=0)
    parser.add_argument("--mass", type=float, default=0.10)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    candidates_path = resolve(args.candidates)
    output_dir = resolve(args.output_dir)
    results_dir = output_dir / "results"
    logs_dir = output_dir / "logs"
    results_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    payload = json.loads(candidates_path.read_text(encoding="utf-8"))
    jobs = []
    for record_index, record in enumerate(payload["records"]):
        if record["gripper"] not in GRIPPERS:
            continue
        for friction in args.frictions:
            stem = f"record_{record_index:04d}_mu_{friction_tag(friction)}"
            jobs.append(
                {
                    "record_index": record_index,
                    "gripper": record["gripper"],
                    "object_id": record.get("object_id"),
                    "object_index": record.get("object_index"),
                    "sample_index": record.get("sample_index"),
                    "friction": float(friction),
                    "result": results_dir / f"{stem}.json",
                    "log": logs_dir / f"{stem}.log",
                }
            )

    manifest = {
        "candidates": str(candidates_path),
        "checkpoint": payload.get("checkpoint"),
        "checkpoint_step": payload.get("checkpoint_step"),
        "grippers": list(GRIPPERS),
        "frictions": list(args.frictions),
        "gpu_ids": list(args.gpu_ids),
        "num_jobs": len(jobs),
        "jobs": [{**job, "result": str(job["result"]), "log": str(job["log"])} for job in jobs],
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    lock = threading.Lock()
    completed = 0
    failed = 0
    started = time.monotonic()

    def run_job(job_index: int, gpu_id: int) -> dict:
        nonlocal completed, failed
        job = jobs[job_index]
        if args.resume and job["result"].exists():
            try:
                existing = json.loads(job["result"].read_text(encoding="utf-8"))
                if existing.get("status") == "ok":
                    with lock:
                        completed += 1
                    return {"returncode": 0, "skipped": True, **job}
            except (OSError, json.JSONDecodeError):
                pass

        command = [
            "bash",
            "scripts/run_contactdiff_isaacsim.sh",
            "scripts/validate_isaacsim_grasp.py",
            "--candidates",
            str(candidates_path),
            "--record-index",
            str(job["record_index"]),
            "--candidate-rank",
            str(args.candidate_rank),
            "--friction",
            str(job["friction"]),
            "--mass",
            str(args.mass),
            "--gpu-id",
            str(gpu_id),
            "--output",
            str(job["result"]),
        ]
        environment = os.environ.copy()
        environment["OMNI_KIT_ACCEPT_EULA"] = "YES"
        environment["CONTACTDIFF_OMNI_PORTABLE_ROOT"] = str(
            MCK_ROOT / ".local" / "share" / "isaacsim" / "portable" / f"gpu{gpu_id}"
        )
        with job["log"].open("w", encoding="utf-8") as log_handle:
            result = subprocess.run(
                command,
                cwd=REPO_ROOT,
                env=environment,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                check=False,
            )
        with lock:
            completed += 1
            if result.returncode != 0:
                failed += 1
            elapsed = time.monotonic() - started
            rate = completed / elapsed if elapsed > 0 else 0.0
            remaining = (len(jobs) - completed) / rate if rate > 0 else float("inf")
            print(
                f"[{completed}/{len(jobs)}] gpu={gpu_id} {job['gripper']} "
                f"object={job['object_index']} sample={job['sample_index']} "
                f"mu={job['friction']:.1f} rc={result.returncode} "
                f"failed={failed} eta_min={remaining / 60.0:.1f}",
                flush=True,
            )
        return {"returncode": result.returncode, "skipped": False, **job}

    def run_gpu_queue(gpu_id: int, indices: list[int]) -> list[dict]:
        return [run_job(index, gpu_id) for index in indices]

    object_groups: dict[tuple[str, str], list[int]] = {}
    for index, job in enumerate(jobs):
        object_groups.setdefault((job["gripper"], str(job["object_id"])), []).append(index)
    worker_queues: list[list[int]] = [[] for _ in args.gpu_ids]
    for group_index, group_indices in enumerate(object_groups.values()):
        worker_queues[group_index % len(args.gpu_ids)].extend(group_indices)

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(args.gpu_ids)) as pool:
        futures = [
            pool.submit(
                run_gpu_queue,
                gpu_id,
                worker_queues[worker_index],
            )
            for worker_index, gpu_id in enumerate(args.gpu_ids)
        ]
        for future in concurrent.futures.as_completed(futures):
            future.result()

    status = {
        "num_jobs": len(jobs),
        "num_completed": completed,
        "num_nonzero_returncodes": failed,
        "elapsed_seconds": time.monotonic() - started,
    }
    (output_dir / "run_status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    print(json.dumps(status, indent=2), flush=True)


if __name__ == "__main__":
    main()
