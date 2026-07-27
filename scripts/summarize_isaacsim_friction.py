#!/usr/bin/env python3
"""Combine per-friction Isaac Sim result files into one compact report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    records = [json.loads(Path(path).read_text(encoding="utf-8")) for path in args.inputs]
    records.sort(key=lambda item: float(item["friction"]))
    fields = (
        "friction",
        "success",
        "valid_initialization",
        "displacement_m",
        "vertical_drop_m",
        "pre_release_drift_m",
        "actual_tip_chamfer_m",
    )
    rows = [{field: record.get(field) for field in fields} for record in records]
    payload = {
        "checkpoint": records[0].get("checkpoint"),
        "checkpoint_step": records[0].get("checkpoint_step"),
        "gripper": records[0].get("gripper"),
        "object_id": records[0].get("object_id"),
        "record_index": records[0].get("record_index"),
        "candidate_rank": records[0].get("candidate_rank"),
        "num_trials": len(rows),
        "num_successes": sum(bool(row["success"]) for row in rows),
        "success_rate": sum(bool(row["success"]) for row in rows) / max(len(rows), 1),
        "trials": rows,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
