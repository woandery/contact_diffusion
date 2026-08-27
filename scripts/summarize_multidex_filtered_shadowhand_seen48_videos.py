#!/usr/bin/env python3
"""Summarize low-memory filtered MultiDex multi-hand recordings."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2


def video_info(path: Path) -> dict:
    capture = cv2.VideoCapture(str(path))
    result = {
        "exists": path.is_file(),
        "valid": capture.isOpened(),
        "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": float(capture.get(cv2.CAP_PROP_FPS)),
        "frames": int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
        "bytes": path.stat().st_size if path.is_file() else 0,
    }
    capture.release()
    result["valid"] = result["valid"] and result["frames"] > 0
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    is_cmap = selection.get("schema") == "cmap-filtered-shadowhand-object-selection-v1"
    robot_name = selection.get("robot_name", "shadowhand")
    target = int(selection.get("samples_per_object", 10))
    rows = []
    for selected in selection["objects"]:
        object_name = selected["object_name"]
        token = object_name.replace("+", "__")
        report_path = args.output_root / "reports" / f"{token}.json"
        status_path = args.output_root / "status" / f"{token}.status"
        status = status_path.read_text(encoding="utf-8").strip() if status_path.is_file() else "pending"
        row = {
            "object_name": object_name,
            "status": status,
            "filtered_samples_available": selected["filtered_samples_available"],
            "samples_selected": len(selected["samples"]),
        }
        if report_path.is_file():
            report = json.loads(report_path.read_text(encoding="utf-8"))
            video_paths = [Path(value) for value in report.get("videos", [])]
            if not video_paths and "video" in report:
                video_paths = [Path(report["video"])]
            row.update(
                {
                    "report": str(report_path.resolve()),
                    "videos": [str(path.resolve()) for path in video_paths],
                    "video_info": [video_info(path) for path in video_paths],
                    "local_final_successes": report["local_final_successes"],
                    "local_strict_six_direction_successes": report[
                        "local_strict_six_direction_successes"
                    ],
                    "samples": [
                        sample
                        for part in report.get("parts", [report])
                        for sample in part.get("samples", [])
                    ],
                }
            )
        rows.append(row)
    completed = [row for row in rows if row["status"] == "complete"]
    manifest = {
        "schema": (
            "cmap-filtered-shadowhand-recordings-v1-lowmem"
            if is_cmap
            else "multidex-filtered-multihand-recordings-v4-lowmem"
        ),
        "dataset_kind": "cmap_filtered" if is_cmap else "multidex_filtered",
        "robot_name": robot_name,
        "selection": str(args.selection.resolve()),
        "object_key": selection.get("object_key", "train"),
        "objects_requested": len(rows),
        "objects_with_at_least_target_filtered_successes": sum(
            row["filtered_samples_available"] >= target for row in rows
        ),
        "objects_completed": len(completed),
        "objects_completed_with_fewer_than_target": sum(
            0 < row["samples_selected"] < target and row["status"] == "complete"
            for row in rows
        ),
        "videos_valid": sum(
            info.get("valid", False)
            for row in rows
            for info in row.get("video_info", [])
        ),
        "poses_recorded": sum(row.get("samples_selected", 0) for row in completed),
        "local_final_successes": sum(row.get("local_final_successes", 0) for row in completed),
        "local_strict_six_direction_successes": sum(
            row.get("local_strict_six_direction_successes", 0) for row in completed
        ),
        "objects": rows,
    }
    destination = args.output_root / "manifest.json"
    destination.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in manifest.items() if key != "objects"}, indent=2))


if __name__ == "__main__":
    main()
