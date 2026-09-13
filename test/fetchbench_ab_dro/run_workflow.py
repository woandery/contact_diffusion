"""Plan by default; explicitly execute one phase on a provisioned historical runtime."""
import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

from audit_package import ROOT, audit


def plan(manifest, run, source, phase):
    project = Path(manifest['project_root'])
    fetch = project / 'FetchBench-CORL2024'
    contact = project / 'ContactDiffusion'
    rt = manifest['runtime']
    fetchpy = str(project / 'miniconda3/envs/fetchbench/bin/python')
    env = dict(SAM3D_CAMERA_CONVENTION='opencv', TORCH_HOME=str(source/'torch_cache'),
               SAM3D_DINO_REPOSITORY=str(source/'torch_cache/hub/facebookresearch_dinov2_main'),
               OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1',
               PYTHONUNBUFFERED='1', TORCH_EXTENSIONS_DIR=str(source/'torch_extensions'),
               LD_LIBRARY_PATH=f'{source}/runtime_lib:{project}/miniconda3/envs/contactdiff/lib:'+os.environ.get('LD_LIBRARY_PATH', ''))
    commands = []
    def add(cmd, extra=None):
        commands.append(dict(command=cmd, cwd=str(fetch), environment=dict(env, **(extra or {}))))
    if phase == 'sam3d':
        add([rt['contact_python'], str(fetch/'scripts/sam3d_resident_batch.py'), '--manifest', str(run/'manifest.json'), '--output', str(run/'resident_ab_first'), '--gpus', '0,1'])
    elif phase == 'ab-generate':
        add([rt['contact_python'], str(fetch/'scripts/fetchbench_ab_dro_queue.py'), '--manifest', str(run/'manifest.json'), '--output', str(run/'generation'), '--gpus', '0,1', '--workers-per-gpu', '3', '--prepare-workers-per-gpu', '1', '--sets', '4', '--particles', '4', '--methods', 'AB', '--run'])
    elif phase == 'ab-validate':
        add([rt['contact_python'], str(Path(__file__).resolve()), '--run', str(run), '--runtime-source', str(source), '--phase', '_finite-worker', '--execute'])
    elif phase == 'dro':
        for case in manifest['cases']:
            corrected = run/'dro_inputs'/case['id']
            add([rt['contact_python'], str(fetch/'scripts/materialize_fetchbench_world_to_robot_base.py'), '--source-root', case['observations'], '--output-root', str(corrected), '--task-config', case['task_config'], '--task-index', str(case['task_index']), '--skip-sam3d'])
            add(['bash', str(fetch/'scripts/run_extension3_dro64_worker.sh')], dict(
                FETCHBENCH_ROOT=str(fetch), CONTACT_ROOT=str(contact), DRO_REPRO_ROOT=str(project/'dro_grasp_reproduction'),
                DRO_ROOT=rt['dro_root'], DRO_PYDEPS=rt['dro_pydeps'], DRO_PYTHON=rt['dro_python'],
                FETCHBENCH_PYTHON=fetchpy, CHECKPOINT=rt['checkpoint_dro'], INPUT_ROOT=str(corrected),
                RUN_ROOT=str(run/'generation'/case['id']/'DRO_cpu_physx_all64'), TASK_INDEX=str(case['task_index']),
                SCENE_CONFIG=case['scene'], SCENE_FACTORY=case['scene_factory'], OBJECT_LABEL=case['id'],
                EXPECTED_LEGAL_VIEWS=str(case['legal_views']), GPU_COUNT='2', GPU_OFFSET='0', WORKERS_PER_GPU='2',
                VALIDATION_WORKERS='2', DRO_CANDIDATES='64', GENERATION_ATTEMPTS='4', BASE_SEED='20260808', CLEARANCE='0.005'))
    return commands


def preflight(manifest):
    result = audit()
    if not result['ok']:
        raise ValueError(result)
    project = Path(manifest['project_root'])
    provenance = json.loads((ROOT/'SOURCE_MANIFEST.json').read_text())
    for name, expected in provenance['files'].items():
        path = project / name.removeprefix('snapshot/')
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError('Runtime overlay missing/mismatched: '+str(path))
    for key, weight in provenance['weights'].items():
        path = Path(manifest['runtime'][key])
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != weight['sha256']:
            raise ValueError('Checkpoint missing/mismatched: '+str(path))
    import yaml
    for key, name in [('config_barrett', 'config_barrett.yaml'), ('config_shadow', 'config_shadow.yaml')]:
        actual = yaml.safe_load(Path(manifest['runtime'][key]).read_text())
        expected = yaml.safe_load((ROOT/'configs'/name).read_text())
        if actual != expected:
            raise ValueError('Config differs from frozen protocol: '+key)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, type=Path, help='Contains prepared manifest.json; use a new run for a new experiment')
    parser.add_argument('--runtime-source', required=True, type=Path, help='Historical source with runtime_lib/torch_cache/torch_extensions')
    parser.add_argument('--phase', required=True, choices=['sam3d', 'ab-generate', 'ab-validate', 'dro', '_finite-worker'])
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    manifest = json.loads((args.run/'manifest.json').read_text())
    protocol = manifest['experiment_protocol']
    if tuple(protocol[k] for k in ('sets', 'particles', 'fk_steps', 'env_steps', 'dro_candidates')) != (4, 4, 200, 200, 64):
        raise ValueError('Not the frozen 4x4 / DRO64 protocol')
    # The historical finite validator is intentionally fixed to these 172 views.
    if [(c['id'], c['task_index'], c['legal_views']) for c in manifest['cases']] != [
        ('cerealbox_shelf', 20, 79), ('shampoo_basket', 0, 45), ('shampoo_drawer', 10, 48)]:
        raise ValueError('This historical protocol requires the original task/view cohort')
    commands = plan(manifest, args.run, args.runtime_source, args.phase)
    if not args.execute:
        print(json.dumps({'execute': False, 'commands': commands}, indent=2))
        return
    preflight(manifest)
    # Lock only the outer dispatcher; the finite subprocess also has its own
    # historical launcher.lock. It must not reacquire this parent-held lock.
    if args.phase != '_finite-worker':
        lock = (args.run/'test_workflow.lock').open('a')
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if args.phase == 'dro':
        report = json.loads((args.run/'ab_finite_validation/summary.json').read_text())
        if not (report['complete'] and not report['errors'] and report['planned'] == 2752
                and report['finished_batches'] == 688 and report['finite'] == report['executed']
                and report['finite'] + report['invalid'] == 2752):
            raise ValueError('Finish and report finite AB validation before DRO')
    if args.phase == '_finite-worker':
        project = Path(manifest['project_root'])
        path = project/'FetchBench-CORL2024/scripts/validate_extension3_finite_ab.py'
        spec = importlib.util.spec_from_file_location('historical_finite_validation', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.P = project; module.R = project/'FetchBench-CORL2024'; module.C = project/'ContactDiffusion'
        module.RUN = args.run; module.SOURCE = args.runtime_source; module.OUT = args.run/'ab_finite_validation'
        sys.argv = [str(path), '--workers', '8']
        module.main()
        report = json.loads((module.OUT/'summary.json').read_text())
        if not report['complete'] or report['errors']:
            raise RuntimeError('Finite AB validation incomplete; inspect summary.json')
        return
    for step in commands:
        subprocess.run(step['command'], cwd=step['cwd'], env=dict(os.environ, **step['environment']), check=True)
        if '--skip-sam3d' in step['command']:
            output = Path(step['command'][step['command'].index('--output-root')+1])
            error = json.loads((output/'coordinate_fix_manifest.json').read_text())['maximum_roundtrip_error_m']
            if not 0 <= error < 1e-6:
                raise ValueError('World/base coordinate round-trip failed')


if __name__ == '__main__':
    main()
