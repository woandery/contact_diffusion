"""Verify the archived upstream source against the original FK runtime lock."""

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PREFIX = "/inspire/qb-ilm2/project/zhanghanbo/public/mck/"


def audit():
    manifest = json.loads((ROOT / "frozen/upstream_fk_manifest.json").read_text())
    errors = []
    for absolute, expected in manifest["code_sha256"].items():
        if not absolute.startswith(PREFIX):
            errors.append("Unexpected path: " + absolute)
            continue
        path = ROOT / "snapshot" / absolute[len(PREFIX):]
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            errors.append("Frozen source missing or changed: " + str(path))
    return {"ok": not errors, "checked": len(manifest["code_sha256"]), "errors": errors}


if __name__ == "__main__":
    result = audit()
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["ok"] else 1)
