"""Offline provenance/syntax audit. Does not import training or Isaac Gym code."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parent


def audit(root=ROOT):
    source = json.loads((root / 'SOURCE_MANIFEST.json').read_text())
    errors = []
    for name, expected in source['files'].items():
        path = root / name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            errors.append('Snapshot missing or changed: ' + name)
    frozen = json.loads((root / 'configs/generation_code_hashes.json').read_text())
    for path, expected in frozen.items():
        relative = 'snapshot/' + path.split('/mck/', 1)[1]
        if source['files'].get(relative) != expected:
            errors.append('Generation-time hash mismatch: ' + relative)
    for path in root.rglob('*.py'):
        try:
            ast.parse(path.read_text(), filename=str(path))
        except SyntaxError as exc:
            errors.append(str(exc))
    for path in (root / 'snapshot').rglob('*.sh'):
        result = subprocess.run(['bash', '-n', str(path)], capture_output=True, text=True)
        if result.returncode:
            errors.append(result.stderr)
    return dict(ok=not errors, snapshot_files=len(source['files']),
                generation_time_hashes_checked=len(frozen), errors=errors,
                note='Static audit only; external assets, dependencies and physics are not tested.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    result = audit()
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result['ok'] else 1)
