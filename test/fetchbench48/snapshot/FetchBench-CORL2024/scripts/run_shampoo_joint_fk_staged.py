"""Preregistered 8-view joint-FK pilot, gated 40-view holdout and transfer.

4 contact sets x 4 particles, FK200, ENV0; all particles in fixed ID order.
Only isolated output directories are written. No GUI/video or polling daemon.
"""
import argparse
import concurrent.futures as futures
from copy import deepcopy
import fcntl
import importlib.util
import inspect
import math
import os
from pathlib import Path
import queue
import shutil
import time

import numpy as np

P = Path('/inspire/qb-ilm2/project/zhanghanbo/public/mck')
C = P/'ContactDiffusion'; F = P/'FetchBench-CORL2024'
ROOT = C/'outputs/shampoo_joint_fk_env0_staged_20260914'
SOURCE = C/'outputs/shampoo_drawer_e3_all48views_s4p4_fk200_env200_20260914'
PAIRED = C/'outputs/shampoo_e3_all48_particle_paired_env0_env200_20260914'
PROBE = C/'outputs/shampoo_drawer_environment_probe_4x4_200_20260914'
HOOK = F/'scripts/shampoo_joint_fk_infer.py'
RUN = ROOT/'shampoo_drawer'
ARMS = {'J0':0., 'J1':1., 'J3':3., 'J10':10.}


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


paired = load(F/'scripts/run_shampoo_paired_env0_env200.py', 'joint_paired_validator')
base = paired.base


def folder(job):
    return RUN/'cases'/Path(*job)


# Reuse the verified fixed-order all-particle CPU PhysX function. Only the
# scene identity and output location become parameters for later transfer.
validation_source = inspect.getsource(paired.validate)
replacements = {
    "simulation/'DrawerShelfSceneFactory_28/task_010'/hand":
        "simulation/manifest['case']['scene_factory']/('task_%03d' % manifest['case']['task_index'])/hand",
    "base.OLD/'generation/shampoo_drawer/corrected_inputs/sam3d'":
        "base.OLD/'generation'/manifest['case']['id']/'corrected_inputs/sam3d'",
    "'scene=benchmark_eval/RigidObjDrawerShelf_6'": "f\"scene=benchmark_eval/{manifest['case']['scene']}\"",
    "'task.solution.task_index=10'": "f\"task.solution.task_index={manifest['case']['task_index']}\"",
    "f'task.solution.external_object_name=shampoo_drawer_{view}'":
        "f\"task.solution.external_object_name={manifest['case']['id']}_{view}\"",
}
for old, new in replacements.items():
    assert validation_source.count(old) == 1, old
    validation_source = validation_source.replace(old, new)
validation_scope = dict(paired.__dict__)
validation_scope.update(folder=folder)
exec(compile(validation_source, '<same-fixed-order-physics-generic-scene>', 'exec'), validation_scope)
validate = validation_scope['validate']


def prepare(case_id):
    global RUN
    RUN = ROOT/case_id
    RUN.mkdir(parents=True, exist_ok=True)
    validation_scope['RUN'] = RUN
    previous = base.read(SOURCE/'manifest.json')
    for path, value in previous['code_sha256'].items():
        assert base.sha(path) == value, path
    for key, value in previous['checkpoint_sha256'].items():
        assert base.sha(previous['runtime'][key]) == value, key
    if case_id == 'shampoo_drawer':
        case = previous['case']; selected = previous['selected_views']
        pilot = [v['view'] for v in base.read(PROBE/'manifest.json')['selected_views']]
    else:
        cases = base.read(base.OLD/'manifest.json')['cases']
        case = next(v for v in cases if v['id'] == case_id)
        legal = base.read(Path(case['observations'])/'visibility/legal_partial_views.json')['views']
        selected = [dict(view=Path(v['rgbd_capture_dir']).name, original_legal_index=i, metadata=v)
                    for i, v in enumerate(legal)]
        # Predetermined geometric coverage in the ordered legal-view inventory.
        ids = np.linspace(0, len(selected)-1, min(8,len(selected))).round().astype(int)
        pilot = [selected[i]['view'] for i in ids]
    assert len(pilot) == 8 and len(set(pilot)) == 8
    configs = {}; inputs = {}; frames = []
    for arm in ARMS:
        configs[arm] = {}
        for hand, src in previous['config_paths']['E3'].items():
            dst = RUN/'configs'/f'{arm}_{hand}.yaml'
            dst.parent.mkdir(parents=True, exist_ok=True)
            if not dst.exists(): shutil.copy2(src, dst)
            assert base.sha(src) == base.sha(dst)
            configs[arm][hand] = str(dst)
    for v in selected:
        view = v['view']
        if case_id != 'shampoo_drawer' and view not in pilot: continue
        corrected = base.OLD/'generation'/case_id/'corrected_inputs'
        geom = corrected/'sam3d'/view
        observed = Path(case['observations'])/'sam3d'/view
        frame = corrected/'visibility/rgbd_views'/view/'metadata.json'
        assert base.read(frame)['coordinate_frame'] == 'robot_base'
        md = base.read(geom/'sam3d_fused_centered.json')
        t = np.asarray(md.get('robot_from_pointcloud_frame', md.get('robot_from_object')))
        assert t.shape == (4,4) and np.allclose(t[3], [0,0,0,1])
        assert np.allclose(t[:3,:3].T @ t[:3,:3], np.eye(3), atol=1e-5)
        paths = [frame, geom/'sam3d_fused_centered.json', geom/'sam3d_fused_robot_base.npy',
                 corrected/'visibility/rgbd_views'/view/'scene_partial_robot_base.npy',
                 observed/'sam3d_fused_centered.npy', observed/'camera_partial_sam3d_centered.npy']
        for path in paths:
            if path.suffix == '.npy':
                arr = np.load(path)
                assert arr.ndim == 2 and arr.shape[1] == 3 and len(arr) and np.isfinite(arr).all(), path
        residual = float(np.abs(np.load(observed/'sam3d_fused_centered.npy') @ t[:3,:3].T + t[:3,3]
                               - np.load(geom/'sam3d_fused_robot_base.npy')).max())
        assert residual < 1e-5
        frames.append(dict(view=view, residual_m=residual))
        inputs[view] = {str(p):base.sha(p) for p in paths}
    prep = RUN/'prepare_ablation.py'
    if not prep.exists(): shutil.copy2(SOURCE/'prepare_ablation.py', prep)
    assert base.sha(prep) == base.sha(SOURCE/'prepare_ablation.py')
    manifest = dict(previous)
    manifest.update(case=case, selected_views=selected, pilot_views=pilot, config_paths=configs,
        arms=ARMS, inputs=inputs, frame_audit=frames, fk_steps=200, env_steps=0, env_lr=None,
        env_weight=None, fk_lr=.0075, no_quality_ranking=True, all_particles=True,
        numerical_mode='unchanged default math; same mode for every arm',
        expected_pilot_particles_per_arm=512, source_run=str(SOURCE),
        comparison='paired same contacts/initialization, all particles, height>=10cm plus strict success',
        source_code_sha256=previous['code_sha256'],
        joint_code_sha256={str(p):base.sha(p) for p in [Path(__file__),HOOK,
            F/'scripts/shampoo_environment_probe_infer_20260914.py',
            F/'scripts/run_shampoo_paired_env0_env200.py', paired.VALIDATOR]},
        protocol_id='joint-fk200-env0-e3-s4p4-closure4-paired-allparticles-v1',
        objective='J0=original FK; Jw=FK+w*Eenv+ramp*(4*Bobj+10*Bcontact+0.4*w*Benv)',
        objective_notes='Original FK terms not duplicated; barriers start after25%; env mean+CVaR active throughout; qouter .1 to qinner .2 four poses',
        gate=dict(min_net_success_gain=8, min_relative_gain=.10, strict_must_not_decrease=True,
                  minimum_positive_views_pilot=3, minimum_positive_views_holdout=12,
                  historical_env200_height_must_be_exceeded=True,
                  choice='pilot maximum height; then strict; then lower weight; heldout weight frozen'),
        generation_workers=8, validation_workers=12, physics_threads_per_process=4,
        conditional_transfer=['cerealbox_shelf', 'shampoo_basket'])
    manifest['config_sha256'] = {a:{h:base.sha(p) for h,p in cfg.items()} for a,cfg in configs.items()}
    if (RUN/'manifest.json').exists(): assert base.read(RUN/'manifest.json') == manifest
    else: base.save(RUN/'manifest.json', manifest)
    return manifest


def stages(job, gpu, manifest, names, files, commands, env, dest, old, config):
    arm, view, condition, hand = job
    case = manifest['case']['id']
    geom = base.OLD/'generation'/case/'corrected_inputs/sam3d'/view
    md = base.read(geom/'sam3d_fused_centered.json')
    source = (SOURCE/'cases/E3'/view/condition/hand/'raw_object.json') if case == 'shampoo_drawer' else (
        folder(('J0',view,condition,hand))/'raw_object.json' if arm != 'J0' else None)
    meta = dict(arm='E3', joint_arm=arm, contact_root=str(C), config=str(config),
        gripper='Barrett' if hand == 'barrett' else 'shadow_hand', joint_environment_weight=ARMS[arm],
        robot_from_object=md.get('robot_from_pointcloud_frame',md.get('robot_from_object')),
        scene=str(base.OLD/'generation'/case/'corrected_inputs/visibility/rgbd_views'/view/'scene_partial_robot_base.npy'),
        object_robot=str(geom/'sam3d_fused_robot_base.npy'), audit=str(dest/'intervention_audit.json'),
        output=str(files[0]), baseline_raw=str(source) if source else '')
    base.save(dest/'joint_metadata.json', meta)
    env.update(CONTACTDIFF_INFER_SOURCE=str(HOOK), ENV_PROBE_META=str(dest/'joint_metadata.json'))
    if source is not None:
        assert source.is_file()
        commands[0] += ['--source-diffusion-candidates',source,'--replay-source-diffusion-rng']
    timings = {}
    # Deliberately truncate the frozen stage commands BEFORE refinement/selection.
    for name, output, command in zip(names[:2], files[:2], commands[:2]):
        started = time.monotonic()
        if not output.exists(): base.execute(command, env, dest/(name+'.log'))
        data = base.read(output)
        assert data['config_sha256'] == base.sha(config)
        assert len(data['records']) == 4 and all(len(r['fk']['candidates']) == 4 for r in data['records'])
        timings[name] = time.monotonic()-started
    raw = base.read(files[0]); pairing = []
    if source is not None:
        reference = base.read(source)
        aa = sorted(reference['records'],key=lambda r:r['sample_index'])
        bb = sorted(raw['records'],key=lambda r:r['sample_index'])
        for a,b in zip(aa,bb):
            for key in ('sample_seed','target_contacts_sha256','source_diffusion_contacts_sha256'):
                assert a[key] == b[key], (job,key)
            assert a['fk']['initialization_state_sha256'] == b['fk']['initialization_state_sha256'], (job,'init')
            ac = sorted(a['fk']['candidates'],key=lambda c:c['particle'])
            bc = sorted(b['fk']['candidates'],key=lambda c:c['particle'])
            pairing.append(dict(sample=b['sample_index'],contacts_equal=True,initialization_equal=True,
                final_pose_joint_equal=[(x['root_pose'],x['joint_positions']) == (y['root_pose'],y['joint_positions']) for x,y in zip(ac,bc)]))
    payload = deepcopy(base.read(files[1])); payload['records'].sort(key=lambda r:r['sample_index'])
    for r in payload['records']:
        r['fk']['candidates'].sort(key=lambda c:c['particle'])
        for c in r['fk']['candidates']:
            assert np.isfinite(c['root_pose']).all() and np.isfinite(c['joint_positions']).all()
            c['paired_original_quality_rank'] = c['rank']; c['rank'] = int(c['particle'])
    payload['selection']['top_k'] = 4
    payload['joint_execution'] = dict(no_env_refinement=True, no_selection=True, arm=arm,
        ordering='sample_index then particle', all_particles=16)
    base.save(dest/'all_particles.json', payload)
    base.save(dest/'pairing_audit.json', dict(source=str(source),records=pairing))
    base.save(dest/'generation_timing.json', dict(gpu=gpu,seconds=timings))
    return dest/'all_particles.json'


def generation_function(manifest):
    source = inspect.getsource(base.generation)
    source = source[:source.index('    timings = {}')] + '    return _stages(job,gpu,manifest,stages,files,commands,env,dest,old,config)\n'
    scope = dict(base.__dict__)
    scope.update(RUN=RUN,CASE=manifest['case']['id'],ARMS={a:(4,4,200) for a in ARMS},folder=folder,_stages=stages)
    exec(compile(source,'<frozen-first-two-stages-joint-fk>','exec'),scope)
    return scope['generation']


def report(manifest):
    rows = [base.read(p) for p in (RUN/'cases').glob('*/*/*/*/validation_result.json')]
    pilot = set(manifest['pilot_views']); groups=[]
    for subset in ('pilot8','heldout40','all'):
        for arm in ARMS:
            for condition in ('A','B'):
                for hand in ('barrett','shadowhand'):
                    rs = [r for r in rows if (r['arm'],r['condition'],r['hand']) == (arm,condition,hand)
                          and (subset=='all' or (r['view'] in pilot)==(subset=='pilot8'))]
                    if rs: groups.append(dict(subset=subset,arm=arm,condition=condition,hand=hand,batches=len(rs),
                        **{k:sum(r[k] for r in rs) for k in ('executed','height','strict')}))
    base.save(RUN/'results.json',dict(groups=groups,rows=rows))
    lines=['# 联合 FK200、取消 ENV：分阶段配对实验','',
        '4接触集×4粒子；A/B、两只手；原初始化/接触目标配对；全部粒子CPU PhysX验证；无GUI/视频。',
        '主指标=最终提升≥10cm；严格成功另列。排名不影响执行顺序。',
        'J0原FK-only；J1/J3/J10=FK+环境与闭合轨迹约束。新增项详见manifest，不重复原FK能量。',
        'A/B沿用各自既有模型权重；这是各条件内部的优化消融，并非仅改变输入的同权重实验。','',
        '|范围|组|输入|手|批数|高度成功/执行|严格成功|','|---|---|---|---|---:|---:|---:|']
    for g in groups: lines.append(f"|{g['subset']}|{g['arm']}|{g['condition']}|{g['hand']}|{g['batches']}|{g['height']}/{g['executed']}|{g['strict']}|")
    for path in sorted(RUN.glob('gate_*.json')):
        lines += ['',f'## {path.stem}','', '```json',path.read_text().strip(),'```']
    (RUN/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    return rows


def run_jobs(manifest, views, arms, phase):
    generate = generation_function(manifest)
    slots = queue.Queue()
    for _ in range(2):
        for gpu in range(4): slots.put(gpu)
    def work(job):
        dest=folder(job)
        if (dest/'generation_done.json').exists(): return job
        gpu=slots.get();started=time.monotonic()
        try:
            generate(job,gpu,manifest)
            base.save(dest/'generation_done.json',dict(complete=True,gpu=gpu,seconds=time.monotonic()-started))
            return job
        finally: slots.put(gpu)
    jobs=[(a,v,c,h) for v in views for c in ('A','B') for h in ('barrett','shadowhand') for a in arms]
    base.save(RUN/'status.json',dict(stage=phase,pid=os.getpid(),jobs=len(jobs),generation_workers=8,validation_workers=12,started=time.time()))
    errors=[]
    with futures.ThreadPoolExecutor(12) as vp, futures.ThreadPoolExecutor(8) as gp:
        gs={gp.submit(work,j):j for j in jobs};vs={}
        for future in futures.as_completed(gs):
            job=gs[future]
            try: vs[vp.submit(validate,future.result(),manifest)]=job
            except Exception as error:
                entry=dict(job=job,error=repr(error));errors.append(entry);base.save(folder(job)/'error.json',entry)
        for future in futures.as_completed(vs):
            job=vs[future]
            try: future.result()
            except Exception as error:
                entry=dict(job=job,error=repr(error));errors.append(entry);base.save(folder(job)/'error.json',entry)
            report(manifest)
    if errors:
        base.save(RUN/'errors.json',errors)
        raise RuntimeError(f'{phase}: {len(errors)} failed batches; do not interpret missing results as algorithm failures')
    report(manifest)


def gate(manifest, views, arms, name):
    rows=report(manifest);selected=set(views)
    controls=[r for r in rows if r['arm']=='J0' and r['view'] in selected]
    assert len(controls)==len(views)*4
    before={k:sum(r[k] for r in controls) for k in ('height','strict','executed')}
    historical=None
    if manifest['case']['id']=='shampoo_drawer':
        old=[r for r in base.read(PAIRED/'results.json')['rows'] if r['arm']=='ENV200' and r['view'] in selected]
        assert len(old)==len(controls)
        historical={k:sum(r[k] for r in old) for k in ('height','strict','executed')}
    minimum=max(8,math.ceil(.10*before['height']))
    results=[]
    for arm in arms:
        after=[r for r in rows if r['arm']==arm and r['view'] in selected]
        assert len(after)==len(controls)
        scores={k:sum(r[k] for r in after) for k in ('height','strict','executed')}
        assert scores['executed']==before['executed']==len(views)*64
        deltas={v:sum(r['height'] for r in after if r['view']==v)-sum(r['height'] for r in controls if r['view']==v) for v in views}
        controls_by_key={(r['view'],r['condition'],r['hand']):r for r in controls}
        gained=lost=0
        for r in after:
            other=controls_by_key[(r['view'],r['condition'],r['hand'])]
            for a,b in zip(other['trials'],r['trials']):
                assert (a['sample'],a['particle'])==(b['sample'],b['particle'])
                gained+=int(not a['height'] and b['height']);lost+=int(a['height'] and not b['height'])
        checks=dict(height_gain=scores['height']-before['height']>=minimum,
            strict_preserved=scores['strict']>=before['strict'],
            distributed=sum(d>0 for d in deltas.values()) >= (3 if len(views)==8 else 12),
            beats_historical_env200=historical is None or scores['height']>historical['height'])
        results.append(dict(arm=arm,scores=scores,checks=checks,passed=all(checks.values()),
                            per_view_gain=deltas,gained=gained,lost=lost))
    passed=[r for r in results if r['passed']]
    winner=max(passed,key=lambda r:(r['scores']['height'],r['scores']['strict'],-ARMS[r['arm']]))['arm'] if passed else None
    result=dict(subset=name,control=before,historical_env200=historical,min_net_gain=minimum,arms=results,winner=winner,
        interpretation='Exploratory fixed-seed view-correlated pilot; no claim of statistical significance')
    base.save(RUN/f'gate_{name}.json',result);report(manifest)
    return winner


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--prepare-only',action='store_true');parser.add_argument('--smoke-only',action='store_true')
    args=parser.parse_args();ROOT.mkdir(parents=True,exist_ok=True)
    lock=(ROOT/'launcher.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    manifest=prepare('shampoo_drawer');pilot=manifest['pilot_views'];started=time.time()
    if args.prepare_only: print('Prepared staged joint FK protocol',flush=True);return
    # Two hands and baseline/high weight must finish before the full pilot.
    run_jobs(manifest,pilot[:1],['J0','J10'],'smoke')
    if args.smoke_only: print('Both-hand A/B joint smoke complete',flush=True);return
    run_jobs(manifest,pilot,list(ARMS),'pilot8')
    winner=gate(manifest,pilot,['J1','J3','J10'],'pilot8')
    if winner is None:
        base.save(ROOT/'status.json',dict(stage='complete_no_promotion',reason='No pilot arm passed preregistered gate',started=started,finished=time.time()))
        base.save(RUN/'status.json',dict(stage='complete',promoted=False,finished=time.time()));return
    holdout=[v['view'] for v in manifest['selected_views'] if v['view'] not in pilot]
    assert len(holdout)==40
    run_jobs(manifest,holdout,['J0',winner],'heldout40')
    promoted=gate(manifest,holdout,[winner],'heldout40')
    base.save(RUN/'status.json',dict(stage='complete',promoted=bool(promoted),finished=time.time()))
    if promoted:
        for case in manifest['conditional_transfer']:
            transfer=prepare(case)
            # E3-screened contacts are generated once by J0 in a new scene;
            # the frozen winning weight replays them, with an exact seed audit.
            run_jobs(transfer,transfer['pilot_views'],['J0'],'transfer_baseline')
            run_jobs(transfer,transfer['pilot_views'],[winner],'transfer_joint')
            gate(transfer,transfer['pilot_views'],[winner],'transfer8')
            base.save(RUN/'status.json',dict(stage='complete',finished=time.time()))
    base.save(ROOT/'status.json',dict(stage='complete',pilot_winner=winner,heldout_passed=bool(promoted),started=started,finished=time.time()))


if __name__=='__main__':
    try: main()
    except Exception as error:
        base.save(ROOT/'status.json',dict(stage='failed',error=repr(error),time=time.time()))
        raise
