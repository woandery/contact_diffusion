"""Export historical code into a NEW overlay directory, never overwrite a checkout."""
import argparse
import json
from pathlib import Path
import shutil

from audit_package import ROOT, audit


def export(destination):
    if not audit()['ok']:
        raise ValueError('Source audit failed')
    if destination.exists():
        raise FileExistsError('Destination must not exist; this tool never overlays a live checkout')
    destination.mkdir(parents=True, exist_ok=False)
    for name in ('ContactDiffusion', 'FetchBench-CORL2024'):
        shutil.copytree(ROOT / 'snapshot' / name, destination / name)
    shutil.copytree(ROOT / 'configs', destination / 'ContactDiffusion' / 'test_configs')
    shutil.copy2(ROOT / 'notices/FetchBench-LICENSE', destination / 'FetchBench-CORL2024/LICENSE')
    return {'overlay': str(destination), 'runnable_without_external_dependencies': False,
            'warning': 'Historical overlay only. See README for external repos/assets and finite-AB validation entry.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--destination', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(export(args.destination), indent=2))
