#!/usr/bin/env python3
"""Two-level batch runner: one long-lived SAM3D process per requested GPU.

All views keep their original seed and geometry parameters. A failed view is
recorded and the worker proceeds; a failed batch exits nonzero after all jobs.
"""
import argparse
import contextlib
import gc
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def cache_valid(job):
    """Accept only complete geometry with matching settings and array hashes."""
    import numpy as np
    out = Path(job["output_dir"])
    try:
        d = json.loads((out / "sam3d_fused_centered.json").read_text())
        if d.get('camera_pointmap_convention', 'isaac_gl_legacy') != os.environ.get('SAM3D_CAMERA_CONVENTION', 'isaac_gl_legacy'):
            return False
        expected = dict(camera_index=0, observed_source="selected_camera",
                        seed=20260905, fused_points=8192, observed_fraction=0.25,
                        opacity_threshold=0.10, layout_postprocess=False)
        if any(d.get(k) != v for k, v in expected.items()):
            return False
        if d.get("capture_dir") != str(Path(job["capture_dir"]).resolve()):
            return False
        for filename, key in [("sam3d_fused_robot_base.npy", "robot_base_sha256"),
                              ("sam3d_fused_centered.npy", "centered_sha256")]:
            a = np.load(out / filename, allow_pickle=False)
            if a.shape != (8192, 3) or not np.isfinite(a).all():
                return False
            if hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest() != d[key]:
                return False
        partial = np.load(out / "camera_partial_sam3d_centered.npy", allow_pickle=False)
        return partial.ndim == 2 and partial.shape[1] == 3 and len(partial) > 0 and bool(np.isfinite(partial).all())
    except (OSError, ValueError, KeyError, TypeError):
        return False


def worker(spec_path, log_root):
    import torch
    from reconstruct_fetchbench_sam3d import load_inference, reconstruct

    spec = json.loads(Path(spec_path).read_text())
    runtime = spec["runtime"]
    log_root = Path(log_root)
    log_root.mkdir(parents=True, exist_ok=True)
    inference = None
    loads = 0
    rows = []
    load_seconds = 0.0
    for job in spec["jobs"]:
        start = time.monotonic()
        row = dict(job=job, pid=os.getpid(), gpu=os.environ.get("CUDA_VISIBLE_DEVICES"))
        if not spec.get("force", False) and cache_valid(job):
            row["status"] = "cached"
        else:
            if inference is None:
                load_start = time.monotonic()
                inference = load_inference(Path(runtime["sam3d_root"]),
                                           Path(runtime["sam3d_config"]), log_root / "model")
                torch.cuda.synchronize()
                load_seconds = time.monotonic() - load_start
                loads += 1
                print(json.dumps(dict(event="MODEL_READY", pid=os.getpid(),
                                      model_load_count=loads, seconds=load_seconds)), flush=True)
            args = argparse.Namespace(
                capture_dir=Path(job["capture_dir"]), output_dir=Path(job["output_dir"]),
                sam3d_root=Path(runtime["sam3d_root"]),
                checkpoint_config=Path(runtime["sam3d_config"]), camera_index=0,
                observed_source="selected_camera", points=8192, observed_fraction=0.25,
                opacity_threshold=0.10, seed=20260905, layout_postprocess=False)
            row["model_identity"] = id(inference)
            work_start = time.monotonic()
            try:
                with (log_root / (job["id"] + ".log")).open("w") as stream:
                    with contextlib.redirect_stdout(stream):
                        result = reconstruct(args, inference=inference)
                torch.cuda.synchronize()
                row.update(status="success", centered_sha256=result["centered_sha256"])
            except Exception as error:
                row.update(status="failed", error=repr(error), traceback=traceback.format_exc())
                # No traceback/tensor references retained across jobs.
            finally:
                gc.collect()
                torch.cuda.empty_cache()
            row["view_seconds"] = time.monotonic() - work_start
            row["allocated_bytes"] = torch.cuda.memory_allocated()
            row["reserved_bytes"] = torch.cuda.memory_reserved()
        row.update(seconds=time.monotonic() - start, model_load_count=loads)
        rows.append(row)
        save(log_root / "summary.json", dict(pid=os.getpid(), model_load_count=loads,
             model_load_seconds=load_seconds, rows=rows, complete=False))
        print(json.dumps(row), flush=True)
    failed = sum(r["status"] == "failed" for r in rows)
    save(log_root / "summary.json", dict(pid=os.getpid(), model_load_count=loads,
         model_load_seconds=load_seconds, rows=rows, complete=True, failures=failed))
    return int(failed > 0)


def launch(manifest_path, output, gpus):
    manifest = json.loads(Path(manifest_path).read_text())
    runtime = manifest["runtime"]
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    jobs = []
    for case in manifest["cases"]:
        source = Path(case["observations"])
        legal = json.loads((source / "visibility/legal_partial_views.json").read_text())
        for view in legal["views"]:
            name = Path(view["rgbd_capture_dir"]).name
            jobs.append(dict(id=case["id"] + "__" + name,
                             capture_dir=str(source / "visibility/rgbd_views" / name),
                             output_dir=str(source / "sam3d" / name)))
    env_root = Path(runtime["sam3d_python"]).parent.parent
    children = []
    streams = []
    for index, gpu in enumerate(gpus):
        spec_path = output / f"gpu{gpu}_jobs.json"
        save(spec_path, dict(runtime=runtime, jobs=jobs[index::len(gpus)]))
        env = os.environ.copy()
        env.update(CUDA_VISIBLE_DEVICES=gpu, CONDA_PREFIX=str(env_root),
                   CUDA_HOME=str(env_root), OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                   OPENBLAS_NUM_THREADS="1", PYTHONUNBUFFERED="1")
        for key, prefix in dict(PATH=env_root / "bin", LD_LIBRARY_PATH=env_root / "lib",
                                CPATH=env_root / "targets/x86_64-linux/include",
                                LIBRARY_PATH=env_root / "targets/x86_64-linux/lib").items():
            env[key] = str(prefix) + ":" + env.get(key, "")
        env["LD_LIBRARY_PATH"] += ":" + str(Path(runtime["contact_python"]).parent.parent / "lib")
        stream = (output / f"gpu{gpu}.log").open("a")
        streams.append(stream)
        children.append(subprocess.Popen(
            [runtime["sam3d_python"], "-u", str(Path(__file__).resolve()),
             "--worker-spec", str(spec_path), "--output", str(output / f"gpu{gpu}")],
            env=env, stdout=stream, stderr=subprocess.STDOUT))
    save(output / "workers.json", [dict(gpu=g, pid=p.pid) for g, p in zip(gpus, children)])
    codes = [p.wait() for p in children]
    for stream in streams:
        stream.close()
    save(output / "batch_summary.json", dict(total_views=len(jobs), exit_codes=codes,
                                            complete=True, all_success=not any(codes)))
    return int(any(codes))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--manifest")
    group.add_argument("--worker-spec")
    parser.add_argument("--output", required=True)
    parser.add_argument("--gpus", default="0,1")
    args = parser.parse_args()
    if args.worker_spec:
        raise SystemExit(worker(args.worker_spec, args.output))
    gpu_ids = args.gpus.split(",")
    if not gpu_ids or len(set(gpu_ids)) != len(gpu_ids) or not all(g.isdigit() for g in gpu_ids):
        parser.error("gpus must be distinct numeric device IDs")
    raise SystemExit(launch(args.manifest, args.output, gpu_ids))
