"""Read-only audit of the frozen 48-task collaboration bundle."""

import ast
import hashlib
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parent
FROZEN = ROOT / "frozen"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def audit():
    errors = []
    source = read(ROOT / "SOURCE_SHA256.json")
    for relative, expected in source.items():
        path = ROOT / relative
        if not path.is_file() or sha(path) != expected:
            errors.append(f"Source missing or changed: {relative}")
        elif path.suffix == ".py":
            try:
                ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            except SyntaxError as exc:
                errors.append(f"Syntax error: {exc}")

    series = read(FROZEN / "SERIES.json")
    if sha(FROZEN / "JOINT_ALLOCATION.json") != series["allocation_sha256"]:
        errors.append("Joint allocation digest differs from SERIES.json")
    if len(series["cohorts"]) != 4:
        errors.append("Expected four cohorts")

    seen_assets = set()
    totals = Counter()
    for index in range(1, 5):
        folder = FROZEN / f"batch{index}"
        lock = read(folder / "DESIGN_LOCK.json")
        entry = series["cohorts"][index - 1]
        if sha(folder / "DESIGN_LOCK.json") != entry["design_lock_sha256"]:
            errors.append(f"Batch {index}: design lock digest differs from SERIES.json")
        for filename, expected in lock["sha256"].items():
            path = folder / filename
            if not path.is_file() or sha(path) != expected:
                errors.append(f"Batch {index}: frozen artifact missing/changed: {filename}")
        manifest = read(folder / "manifest.json")
        selection = read(folder / "TASK_SELECTION.json")
        protocol = read(folder / "protocol_v2.json")
        cases = manifest["cases"]
        if len(cases) != 48 or len(selection["tasks"]) != 48:
            errors.append(f"Batch {index}: expected 48 cases and task selections")
        assets = [case["asset_key"] for case in cases]
        if len(set(assets)) != 48 or seen_assets.intersection(assets):
            errors.append(f"Batch {index}: repeated canonical target asset")
        seen_assets.update(assets)
        environments = Counter(case["environment"] for case in cases)
        quota = Counter({key: 12 for key in ("table", "shelf_or_cabinet", "drawer", "basket")})
        if environments != quota:
            errors.append(f"Batch {index}: environment quota differs: {dict(environments)}")
        generation = protocol["generation"]
        expected = {"contact_sets": 4, "particles": 4, "fk_steps": 200,
                    "fk_lr": 0.0075, "diffusion_steps": 50, "env_steps": 0,
                    "dro_candidates": 64, "dro_points": 512,
                    "dro_optimization_steps": 64}
        for key, value in expected.items():
            if generation.get(key) != value:
                errors.append(f"Batch {index}: {key} changed")
        freeze = manifest["ranker_freeze"]
        for filename, field in (("ranker_unified_frozen.joblib", "unified_model_sha256"),
                                ("ranker_original_reference.joblib", "original_model_sha256")):
            if sha(folder / filename) != freeze[field]:
                errors.append(f"Batch {index}: {filename} digest changed")
        if protocol.get("joint_allocation_sha256") != series["allocation_sha256"]:
            errors.append(f"Batch {index}: joint allocation reference changed")
        totals.update(environments)

    if len(seen_assets) != 192:
        errors.append(f"Expected 192 unique target assets; found {len(seen_assets)}")
    return {"ok": not errors, "source_files": len(source), "cohorts": 4,
            "unique_assets": len(seen_assets), "environments": dict(totals),
            "errors": errors, "runtime_tested": False}


if __name__ == "__main__":
    result = audit()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(0 if result["ok"] else 1)
