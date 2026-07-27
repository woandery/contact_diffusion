#!/usr/bin/env python3
"""Combine one Isaac Sim screening result per gripper."""

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
    rows = []
    for record in records:
        rows.append(
            {
                key: record.get(key)
                for key in (
                    "gripper",
                    "status",
                    "error",
                    "success",
                    "valid_initialization",
                    "friction",
                    "contact_chamfer_m",
                    "actual_tip_chamfer_m",
                    "displacement_m",
                    "vertical_drop_m",
                )
            }
        )
    payload = {
        "checkpoint": records[0].get("checkpoint"),
        "checkpoint_step": records[0].get("checkpoint_step"),
        "num_grippers": len(rows),
        "num_simulated": sum(row["status"] == "ok" for row in rows),
        "num_import_errors": sum(row["status"] != "ok" for row in rows),
        "num_successes": sum(row["success"] is True for row in rows),
        "results": rows,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
