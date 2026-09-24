"""Validate finite A/B poses independently; preserve invalids in the denominator audit."""
import argparse
import concurrent.futures
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np

R = Path(__file__).resolve().parents[1]
P = R.parent
C = P / 'ContactDiffusion'
RUN = C / 'outputs/extension3_111view_ab4x4_dro64_20260910'
SOURCE = C / 'outputs/extension3_shelf_basket_drawer_4x4_20260910_v2'
OUT = RUN / 'ab_finite_validation'


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, indent=2))
    tmp.replace(path)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def finite_record(row):
    fk = row['fk']
    assert len(fk['candidates']) == 1
    candidate = fk['candidates'][0]
    pose = np.asarray(candidate['root_pose'], dtype=float)
    joints = np.asarray(candidate['joint_positions'], dtype=float)
    assert pose.shape == (4, 4) and joints.shape == (len(fk['joint_names']),)
    return bool(np.isfinite(pose).all() and np.isfinite(joints).all())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--audit-only', action='store_true')
    args = parser.parse_args()
    assert 1 <= args.workers <= 8  # Measured five workers used about 10.5 of 20 CPU cores.
    OUT.mkdir(parents=True, exist_ok=True)
    lock = (RUN / 'launcher.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    m = json.loads((RUN / 'manifest.json').read_text())
    rt = m['runtime']
    jobs = []
    for case in m['cases']:
        legal = json.loads((Path(case['observations']) / 'visibility/legal_partial_views.json').read_text())['views']
        for view in legal:
            vid = Path(view['rgbd_capture_dir']).name
            for condition in ['A', 'B']:
                for hand in ['barrett', 'shadowhand']:
                    src = RUN / 'generation' / case['id'] / 'AB/views' / vid / condition / 'selected_w10' / f"{case['id']}_{hand}.json"
                    data = json.loads(src.read_text())
                    assert len(data['records']) == 4
                    good = [i for i, row in enumerate(data['records']) if finite_record(row)]
                    bad = [dict(record_index=i, record_id=row['record_id'], sample_index=row['sample_index'], reason='nonfinite_root_pose_or_joint_positions') for i, row in enumerate(data['records']) if i not in good]
                    job = dict(case=case, view=vid, condition=condition, hand=hand, source=str(src), source_sha256=digest(src), planned=4, valid=len(good), invalid=bad, retained_record_indices=good)
                    jobs.append(job)
    assert len(jobs) == 688
    audit = dict(planned=2752, finite=sum(j['valid'] for j in jobs), invalid=sum(len(j['invalid']) for j in jobs), jobs=jobs)
    save(OUT / 'input_audit.json', audit)
    print(json.dumps({k: v for k, v in audit.items() if k != 'jobs'}), flush=True)
    if args.audit_only:
        return
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1',
               ASSET_PATH=str(R), PYTHONPATH=f'{R}/third_party/isaacgym/python:{R}/InfiniGym',
               LD_LIBRARY_PATH=f'{P}/miniconda3/envs/fetchbench/lib:{SOURCE}/runtime_lib:{P}/miniconda3/envs/contactdiff/lib:' + env.get('LD_LIBRARY_PATH', ''),
               TORCH_EXTENSIONS_DIR=str(SOURCE / 'torch_extensions'), MAX_JOBS='4', CUDA_VISIBLE_DEVICES='')
    fetchpy = str(P / 'miniconda3/envs/fetchbench/bin/python')
    save(RUN / 'DRO_status.json', dict(stage='paused_until_ab_report', time=time.time()))
    save(RUN / 'AB_status.json', dict(stage='cpu_physx_finite', workers=args.workers, physx_threads_per_process=4, time=time.time()))
    save(RUN / 'pipeline_status.json', dict(stage='running', phase='AB_finite', time=time.time()))

    def work(job):
        start = time.monotonic()
        case, vid, cond, hand = job['case'], job['view'], job['condition'], job['hand']
        folder = OUT / 'cases' / case['id'] / vid / cond / hand
        folder.mkdir(parents=True, exist_ok=True)
        row = dict(case=case['id'], view=vid, condition=cond, hand=hand, planned=4, invalid=len(job['invalid']), valid=job['valid'], executed=0, height=0, strict=0)
        try:
            src = Path(job['source'])
            assert digest(src) == job['source_sha256'], 'Source candidates changed'
            if not job['valid']:
                row['status'] = 'skipped_all_nonfinite'
                return row
            payload = json.loads(src.read_text())
            payload['records'] = [payload['records'][i] for i in job['retained_record_indices']]
            payload['finite_validation_filter'] = dict(source=str(src), source_sha256=job['source_sha256'], excluded=job['invalid'])
            filtered = folder / 'finite_candidates.json'
            save(filtered, payload)
            config = rt['config_barrett' if hand == 'barrett' else 'config_shadow']
            gripper = 'Barrett' if hand == 'barrett' else 'shadow_hand'
            prepared = folder / 'prepared.json'
            def call(cmd, log, cwd):
                with (folder / log).open('a') as f:
                    subprocess.run(cmd, cwd=cwd, env=env, stdout=f, stderr=subprocess.STDOUT, check=True)
            if not prepared.exists():
                call([rt['contact_python'], str(C / 'scripts/prepare_basic_experiment_isaacgym.py'), '--candidates', str(filtered), '--config', config, '--gripper', gripper,
                      '--candidate-mode', 'all', '--allow-filtered-candidates', '--allow-runtime-budget', '--allow-incomplete', '--closure-outer-fraction', '0.10', '--closure-inner-fraction', '0.20', '--output', str(prepared)], 'prepare.log', C)
            d = json.loads(prepared.read_text())
            assert len(d['objects']) == 1 and len(d['objects'][0]['samples']) == job['valid']
            artifact = folder / 'simulation'
            summary = artifact / case['scene_factory'] / f"task_{case['task_index']:03d}" / hand / 'lift_validation/summary.json'
            if not summary.exists():
                objpc = RUN / 'generation' / case['id'] / 'corrected_inputs/sam3d' / vid / 'sam3d_fused_robot_base.npy'
                task = 'FetchPtdDRORenderBarrett' if hand == 'barrett' else 'FetchPtdDRORenderShadow'
                urdf = 'contactdiff_v4_barrett_physics.urdf' if hand == 'barrett' else 'contactdiff_v4_shadowhand_physics.urdf'
                call([fetchpy, 'isaacgymenvs/validate_dro_lift.py', f'task={task}', f"scene=benchmark_eval/{case['scene']}",
                      f"task.solution.task_index={case['task_index']}", f"task.solution.visualize_top_k={job['valid']}",
                      'task.solution.physics_only=true', 'task.solution.lift.record_video=false', f'task.solution.goal_pointcloud_override={objpc}', f'task.solution.external_pointcloud={objpc}',
                      f'task.solution.external_prepared={prepared}', f"task.solution.external_object_name={case['id']}_{vid}",
                      'task.solution.external_require_precomputed_environment=false', 'task.solution.external_validate_infeasible=true', 'task.solution.reject_runtime_environment_contacts=true',
                      'task.solution.target_closeup_max_tracking_displacement=0.30', 'task.solution.lift.direct_closure=true', 'task.solution.lift.max_preclosure_object_displacement=0.02',
                      'task.solution.lift.height=0.25', 'task.solution.lift.success_height=0.10', 'task.env.enableCameraSensors=false',
                      f'task.env.robot.asset_root={R}/InfiniGym/assets/contactdiff_hands/{hand}', f'task.env.robot.urdf_file={urdf}', f'task.solution.artifact_dir={artifact}',
                      'seed=20260808', 'num_threads=4', 'pipeline=cpu', 'sim_device=cpu', 'rl_device=cpu', 'graphics_device_id=-1', 'headless=true', 'force_render=false'], 'validate.log', R / 'InfiniGym')
            d = json.loads(summary.read_text())
            assert d['validated_candidates'] == job['valid'] and len(d['trials']) == job['valid']
            row.update(status='complete', executed=d['validated_candidates'], height=sum(t['final_object_lift_m'] >= .10 for t in d['trials']), strict=sum(bool(t['success']) for t in d['trials']), summary=str(summary))
        except Exception as exc:
            row.update(status='error', error=repr(exc))
        finally:
            row['seconds'] = time.monotonic() - start
            save(folder / 'status.json', row)
        return row

    rows = []
    def summarize():
        groups = {}
        for j in jobs:
            key = (j['case']['id'], j['condition'], j['hand'])
            g = groups.setdefault(key, dict(case=key[0], condition=key[1], hand=key[2], planned=0, invalid=0, finite=0, executed=0, height=0, strict=0))
            g['planned'] += 4; g['invalid'] += len(j['invalid']); g['finite'] += j['valid']
        for row in rows:
            g = groups[(row['case'], row['condition'], row['hand'])]
            for k in ['executed', 'height', 'strict']: g[k] += row[k]
        for g in groups.values():
            g['missing_finite'] = g['finite'] - g['executed']
            g['finite_height_rate'] = g['height'] / g['finite'] if g['finite'] and not g['missing_finite'] else None
            g['planned_height_rate'] = g['height'] / g['planned'] if not g['missing_finite'] else None
        complete = len(rows) == len(jobs) and all(r['status'] != 'error' for r in rows)
        result = dict(complete=complete, expected_batches=688, finished_batches=len(rows), planned=2752, invalid=audit['invalid'], finite=audit['finite'], executed=sum(g['executed'] for g in groups.values()), groups=list(groups.values()), errors=[r for r in rows if r['status']=='error'], rows=rows)
        save(OUT / 'summary.json', result)
        return result
    summarize()
    with concurrent.futures.ThreadPoolExecutor(args.workers) as pool:
        futures = [pool.submit(work, job) for job in jobs]
        for f in concurrent.futures.as_completed(futures):
            row = f.result(); rows.append(row); summarize()
            print(json.dumps(row), flush=True)
    result = summarize()
    save(RUN / 'AB_status.json', dict(stage='complete_finite' if result['complete'] else 'failed_finite', summary=str(OUT/'summary.json'), time=time.time()))
    save(RUN / 'pipeline_status.json', dict(stage='ab_complete_awaiting_report' if result['complete'] else 'failed', phase='AB_finite', time=time.time()))


if __name__ == '__main__':
    main()
