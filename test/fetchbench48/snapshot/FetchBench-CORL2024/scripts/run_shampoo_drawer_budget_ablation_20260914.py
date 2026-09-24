"""Frozen 8-view 2x2 budget ablation; read-only baseline, four-GPU queue, CPU PhysX."""
import argparse
import concurrent.futures as futures
from copy import deepcopy
import fcntl
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import queue
import shutil
import subprocess
import threading
import time

import numpy as np
import yaml

P = Path('/inspire/qb-ilm2/project/zhanghanbo/public/mck')
C = P/'ContactDiffusion'
F = P/'FetchBench-CORL2024'
OLD = C/'outputs/extension3_111view_ab4x4_dro64_20260910'
SOURCE = C/'outputs/extension3_shelf_basket_drawer_4x4_20260910_v2'
RUN = C/'outputs/shampoo_drawer_budget_ablation_8views_20260914'
CASE = 'shampoo_drawer'
VIEWS = ['r1.0_el+30_az-90', 'r1.0_el+60_az+30', 'r1.5_el+30_az+0',
         'r1.5_el+60_az-60', 'r1.5_el+90_az+0', 'r2.0_el+0_az+60',
         'r2.0_el+30_az+90', 'r2.0_el+60_az+0']
ARMS = {'C0': (4, 4, 200), 'C1': (8, 8, 200), 'C2': (4, 4, 400), 'C3': (8, 8, 400)}
PROTOCOLS = {k: f'shampoo-drawer-ablation-{k.lower()}-s{s}p{p}-fk{n}-env{n}-w10-20260914'
             for k, (s, p, n) in ARMS.items()}
LOCK = threading.Lock()


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False)+'\n')
    tmp.replace(path)


def finite(record):
    candidate = record['fk']['candidates'][0]
    pose = np.asarray(candidate['root_pose'], dtype=float)
    joints = np.asarray(candidate['joint_positions'], dtype=float)
    assert pose.shape == (4, 4) and joints.shape == (len(record['fk']['joint_names']),)
    return bool(np.isfinite(pose).all() and np.isfinite(joints).all())


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prepare():
    original = read(OLD/'manifest.json')
    case = next(c for c in original['cases'] if c['id'] == CASE)
    legal = read(Path(case['observations'])/'visibility/legal_partial_views.json')['views']
    by_view = {Path(v['rgbd_capture_dir']).name: (i, v) for i, v in enumerate(legal)}
    assert all(v in by_view for v in VIEWS)
    assert case['task_index'] == 10 and case['object_index'] == 1
    config_paths = {}
    for arm, (sets, particles, steps) in ARMS.items():
        config_paths[arm] = {}
        for hand, key in [('barrett', 'config_barrett'), ('shadowhand', 'config_shadow')]:
            src = Path(original['runtime'][key]); dest = RUN/'configs'/f'{arm}_{hand}.yaml'
            cfg = yaml.safe_load(src.read_text())
            assert cfg['fk_optimization']['learning_rate'] == .0075
            expected = deepcopy(cfg)
            if arm != 'C0':
                expected['baseline'].update(protocol_id=PROTOCOLS[arm], contact_sets_per_object=sets,
                    particles_per_contact_set=particles, optimization_steps=steps)
                expected['fk_optimization'].update(particles=particles, steps=steps)
            text = src.read_text() if arm == 'C0' else yaml.safe_dump(expected, sort_keys=False)
            dest.parent.mkdir(parents=True, exist_ok=True)
            if dest.exists():
                assert dest.read_text() == text, 'Existing config changed'
            else:
                dest.write_text(text)
            config_paths[arm][hand] = str(dest)
    # Register only the three audited new IDs in an isolated preparation wrapper.
    # Production protocol helpers and generation scripts are not edited.
    wrapper = RUN/'prepare_ablation.py'
    wrapper_text = (
        'import importlib.util,sys\nfrom pathlib import Path\n'
        f'p=Path({str(C / "scripts/prepare_basic_experiment_isaacgym.py")!r})\n'
        'sys.path.insert(0,str(p.parents[1]))\n'
        's=importlib.util.spec_from_file_location("ablation_prepare",p)\n'
        'm=importlib.util.module_from_spec(s);s.loader.exec_module(m)\n'
        f'm.SUPPORTED_PROTOCOL_IDS=m.SUPPORTED_PROTOCOL_IDS | frozenset({list(PROTOCOLS.values())!r})\n'
        'm.main()\n')
    if wrapper.exists(): assert wrapper.read_text() == wrapper_text
    else: wrapper.write_text(wrapper_text)
    frozen = read(OLD/'test_source_export_20260913/SOURCE_MANIFEST.json') if (OLD/'test_source_export_20260913/SOURCE_MANIFEST.json').exists() else None
    if frozen is None:
        matches = list((OLD/'test_source_export_20260913').glob('**/SOURCE_MANIFEST.json'))
        assert len(matches) == 1, matches
        frozen = read(matches[0])
    source_hashes = {}
    for name, digest in frozen['files'].items():
        source = P/name.removeprefix('snapshot/')
        actual = sha(source)
        assert actual == digest, f'Historical runtime changed: {source}'
        source_hashes[str(source)] = actual
    for key in ('checkpoint_a', 'checkpoint_b'):
        assert sha(original['runtime'][key]) == frozen['weights'][key]['sha256']
    inputs = {}
    for view in VIEWS:
        corrected = OLD/'generation'/CASE/'corrected_inputs'
        md = read(corrected/'visibility/rgbd_views'/view/'metadata.json')
        assert md['coordinate_frame'] == 'robot_base'
        paths = [corrected/'visibility/rgbd_views'/view/'scene_partial_robot_base.npy',
                 corrected/'sam3d'/view/'sam3d_fused_robot_base.npy',
                 corrected/'sam3d'/view/'sam3d_fused_centered.json']
        paths += [Path(case['observations'])/'sam3d'/view/name for name in
                  ('camera_partial_sam3d_centered.npy', 'sam3d_fused_centered.npy')]
        inputs[view] = {str(p): sha(p) for p in paths}
    manifest = dict(case=case, runtime=original['runtime'], arms={k:list(v) for k,v in ARMS.items()}, config_paths=config_paths,
        selected_views=[dict(view=v, original_legal_index=by_view[v][0], metadata=by_view[v][1]) for v in VIEWS],
        view_selection='Fixed geometry-only strata across radius/elevation/azimuth, no outcome selection',
        source_run=str(OLD), inputs=inputs, code_sha256=source_hashes, driver_sha256=sha(__file__),
        checkpoint_sha256={k: frozen['weights'][k]['sha256'] for k in ('checkpoint_a','checkpoint_b')},
        config_sha256={a:{h:sha(p) for h,p in c.items()} for a,c in config_paths.items()},
        fk_lr=.0075, env_lr=.003, env_weight=10, diffusion_steps=50, closure=[.10,.20],
        final_lift_m=.10, physics_lift_m=.25, physics='CPU PhysX', record_video=False,
        comparison='All per-set Top1 poses validated; geometry-ranked Top1/Top4 compared at equal execution budget; nonfinite separately audited',
        sampling_pairing='Assert diffusion/target prefix identical; assert initialization hash identical for equal particle counts; different particle counts not claimed nested')
    dest = RUN/'manifest.json'
    if dest.exists(): assert read(dest) == manifest, 'Protocol changed: new run required'
    else: save(dest, manifest)
    return manifest


def execute(cmd, env, logfile, cwd=C):
    logfile.parent.mkdir(parents=True, exist_ok=True)
    with logfile.open('a') as log:
        log.write('\nCOMMAND '+json.dumps([str(c) for c in cmd])+'\n'); log.flush()
        subprocess.run([str(c) for c in cmd], cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)


def base_env():
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1',
        LD_LIBRARY_PATH=f'{SOURCE}/runtime_lib:{P}/miniconda3/envs/contactdiff/lib:'+env.get('LD_LIBRARY_PATH',''),
        PYTHONUNBUFFERED='1', PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
    return env


def folder(job):
    arm, view, condition, hand = job
    return RUN/'cases'/arm/view/condition/hand


def generation(job, gpu, manifest):
    arm, view, condition, hand = job
    dest = folder(job); dest.mkdir(parents=True, exist_ok=True)
    stages = ['raw_object','candidates_robot','refined_w10','selected_w10']
    files = [dest/(s+'.json') for s in stages]
    old = OLD/'generation'/CASE/'AB/views'/view/condition
    if arm == 'C0':
        for stage, target in zip(stages, files):
            source = old/stage/f'{CASE}_{hand}.json'
            if not target.exists(): shutil.copy2(source, target)
            assert sha(target) == sha(source)
        return files[-1]
    sets, particles, steps = ARMS[arm]
    config = manifest['config_paths'][arm][hand]
    rt = manifest['runtime']; py = rt['contact_python']
    geom = OLD/'generation'/CASE/'corrected_inputs/sam3d'/view
    observed = Path(manifest['case']['observations'])/'sam3d'/view
    scene = OLD/'generation'/CASE/'corrected_inputs/visibility/rgbd_views'/view/'scene_partial_robot_base.npy'
    index = next(v['original_legal_index'] for v in manifest['selected_views'] if v['view']==view)
    env = base_env(); env['CUDA_VISIBLE_DEVICES'] = str(gpu)
    config_hand = 'Barrett' if hand == 'barrett' else 'shadow_hand'
    if condition == 'A':
        adapter = 'infer_contactdiffusion_camera_partial_v5_six.py'
        args = ['--partial-object-pc', observed/'camera_partial_sam3d_centered.npy','--partial-points','2048','--partial-sampling-seed','20260905']
    else:
        adapter = 'infer_contactdiffusion_explicit_condition_v6.py'
        args = ['--condition-object-pc',observed/'sam3d_fused_centered.npy','--condition-points','2048','--condition-sampling-seed','20260905','--condition-mode','sam3d_singleview_complete_proxy']
    commands = [
        [py,C/'scripts'/adapter,*args,'--config',config,'--grippers',config_hand,
         '--hand-index-offset',str(0 if hand=='barrett' else 1),'--checkpoint',rt['checkpoint_a' if condition=='A' else 'checkpoint_b'],
         '--object-pc',observed/'sam3d_fused_centered.npy','--object-id',f'{CASE}_{view}','--samples-per-object',str(sets),'--sample-start','0',
         '--inference-object-observation','full','--autoregressive-fk-target','nearest_2048','--particles',str(particles),
         '--optimization-steps',str(steps),'--diffusion-steps','50','--fk-initialization','enveloping','--envelope-approach-weight','2.0',
         '--selection-min-envelope-cosine','0.5','--selection-min-approach-cosine','0.8','--preferred-root-direction','0','0','1',
         '--selection-max-penetration','0.007','--disable-palm-selection-gate','--top-k',str(sets),'--device','cuda:0',
         '--seed','20260808','--object-index-offset',str(index),'--output',files[0]],
        [py,C/'scripts/transform_contactdiffusion_candidates.py','--input',files[0],'--transform-metadata',geom/'sam3d_fused_centered.json',
         '--object-pc',geom/'sam3d_fused_robot_base.npy','--source-frame','object_sam3d_surface_centered','--target-frame','robot_base','--output',files[1]],
        [py,C/'scripts/refine_contactdiffusion_environment_visible_scene.py','--candidates',files[1],'--config',config,
         '--object-pc',geom/'sam3d_fused_robot_base.npy','--scene-pc',scene,'--output',files[2],'--steps',str(steps),'--learning-rate','0.003',
         '--environment-weight','10','--environment-clearance','0.005','--selection-max-environment-violation','0.0001',
         '--environment-feasibility-mode','soft','--selection-rank-mode','normalized_constraints','--environment-cvar-fraction','0.10',
         '--environment-closure-sweep-samples','4','--closure-outer-fraction','0.10','--closure-inner-fraction','0.20',
         '--object-constraint-weight','4','--environment-constraint-weight','4','--contact-constraint-weight','10','--contact-constraint-limit','0.010',
         '--require-contact-feasibility','--restore-best-constraint-state','--scene-voxel-size','0.008','--max-scene-points','4096','--device','cuda:0'],
        [py,C/'scripts/select_fullpc_environment_weight_ensemble_constraints.py','--inputs',files[2],'--output',files[3]]]
    timings = {}
    for stage, output, cmd in zip(stages,files,commands):
        start=time.monotonic()
        if not output.exists(): execute(cmd,env,dest/(stage+'.log'))
        data=read(output)
        assert data['config_sha256']==sha(config)
        assert len(data['records'])==sets
        assert all(len(r['fk']['candidates'])==(1 if stage=='selected_w10' else particles) for r in data['records'])
        timings[stage]=time.monotonic()-start
        if stage=='raw_object':
            original=read(old/'raw_object'/f'{CASE}_{hand}.json')
            for before,after in zip(original['records'],data['records']):
                assert before['sample_seed']==after['sample_seed']
                assert before['source_diffusion_contacts_sha256']==after['source_diffusion_contacts_sha256'], 'Diffusion samples changed'
                assert before['target_contacts_sha256']==after['target_contacts_sha256'], 'Projected contacts changed'
                if particles==4:
                    assert before['fk']['initialization_state_sha256']==after['fk']['initialization_state_sha256'], 'FK initialization changed'
    save(dest/'generation_timing.json',dict(gpu=gpu,seconds=timings))
    return files[-1]


def validate(job, manifest):
    arm,view,condition,hand=job;dest=folder(job)
    output=dest/'validation_result.json'
    if output.exists() and read(output).get('complete'): return read(output)
    data=read(dest/'selected_w10.json'); source_sha=sha(dest/'selected_w10.json')
    selector=load_module(C/'scripts/select_fullpc_environment_weight_ensemble_constraints.py','selector_'+hand)
    valid=sorted([r for r in data['records'] if finite(r)],key=lambda r:r['sample_index'])
    ranking=sorted(valid,key=lambda r:(selector.candidate_key(r['fk']['candidates'][0]),r['sample_index']))
    ordered=[r['sample_index'] for r in ranking]
    payload=deepcopy(data);payload['records']=valid
    payload['finite_validation_filter']=dict(source=str(dest/'selected_w10.json'),source_sha256=source_sha,
        excluded_sample_indices=[r['sample_index'] for r in data['records'] if not finite(r)])
    filtered=dest/'finite_candidates.json';save(filtered,payload)
    result=dict(arm=arm,view=view,condition=condition,hand=hand,planned=len(data['records']),
        invalid=len(data['records'])-len(valid),ranked_samples=ordered,source_sha256=source_sha)
    if not valid:
        result.update(complete=True,executed=0,height=0,strict=0,top1_executed=0,top1_height=0,top4_executed=0,top4_height=0,hit4=False)
        save(output,result);return result
    env=base_env(); env.update(CUDA_VISIBLE_DEVICES='',ASSET_PATH=str(F),
        PYTHONPATH=f'{F}/third_party/isaacgym/python:{F}/InfiniGym',
        LD_LIBRARY_PATH=f'{P}/miniconda3/envs/fetchbench/lib:'+env['LD_LIBRARY_PATH'],
        TORCH_EXTENSIONS_DIR=str(SOURCE/'torch_extensions'),MAX_JOBS='4')
    rt=manifest['runtime'];cfg=manifest['config_paths'][arm][hand];prepared=dest/'prepared.json'
    if not prepared.exists():
        execute([rt['contact_python'],RUN/'prepare_ablation.py','--candidates',filtered,'--config',cfg,
          '--gripper','Barrett' if hand=='barrett' else 'shadow_hand','--candidate-mode','all','--allow-filtered-candidates',
          '--allow-runtime-budget','--allow-incomplete','--closure-outer-fraction','0.10','--closure-inner-fraction','0.20','--output',prepared],env,dest/'prepare.log')
    samples=read(prepared)['objects'][0]['samples'];assert len(samples)==len(valid)
    simulation=dest/'simulation'
    summary=simulation/'DrawerShelfSceneFactory_28/task_010'/hand/'lift_validation/summary.json'
    if not summary.exists():
        objpc=OLD/'generation'/CASE/'corrected_inputs/sam3d'/view/'sam3d_fused_robot_base.npy'
        task='FetchPtdDRORenderBarrett' if hand=='barrett' else 'FetchPtdDRORenderShadow'
        urdf='contactdiff_v4_barrett_physics.urdf' if hand=='barrett' else 'contactdiff_v4_shadowhand_physics.urdf'
        execute([P/'miniconda3/envs/fetchbench/bin/python','isaacgymenvs/validate_dro_lift.py',f'task={task}',
          'scene=benchmark_eval/RigidObjDrawerShelf_6','task.solution.task_index=10',f'task.solution.visualize_top_k={len(valid)}',
          'task.solution.physics_only=true','task.solution.lift.record_video=false',f'task.solution.goal_pointcloud_override={objpc}',
          f'task.solution.external_pointcloud={objpc}',f'task.solution.external_prepared={prepared}',f'task.solution.external_object_name={CASE}_{view}',
          'task.solution.external_require_precomputed_environment=false','task.solution.external_validate_infeasible=true',
          'task.solution.reject_runtime_environment_contacts=true','task.solution.target_closeup_max_tracking_displacement=0.30',
          'task.solution.lift.direct_closure=true','task.solution.lift.max_preclosure_object_displacement=0.02',
          'task.solution.lift.height=0.25','task.solution.lift.success_height=0.10','task.env.enableCameraSensors=false',
          f'task.env.robot.asset_root={F}/InfiniGym/assets/contactdiff_hands/{hand}',f'task.env.robot.urdf_file={urdf}',
          f'task.solution.artifact_dir={simulation}','seed=20260808','num_threads=4','pipeline=cpu','sim_device=cpu',
          'rl_device=cpu','graphics_device_id=-1','headless=true','force_render=false'],env,dest/'validate.log',F/'InfiniGym')
    sim=read(summary);assert sim['validated_candidates']==len(valid) and len(sim['trials'])==len(valid)
    by_sample={}
    for trial in sim['trials']:
        index=int(trial['candidate']);assert 0<=index<len(samples)
        lift=float(trial['final_object_lift_m']);assert math.isfinite(lift)
        by_sample[int(samples[index]['source_index'])]=dict(height=lift>=.10,strict=bool(trial['success']),lift=lift)
    assert len(by_sample)==len(valid)
    result.update(complete=True,summary=str(summary),executed=len(valid),height=sum(t['height'] for t in by_sample.values()),
        strict=sum(t['strict'] for t in by_sample.values()),top1_executed=min(1,len(ordered)),
        top1_height=sum(by_sample[i]['height'] for i in ordered[:1]),top4_executed=min(4,len(ordered)),
        top4_height=sum(by_sample[i]['height'] for i in ordered[:4]),hit4=any(by_sample[i]['height'] for i in ordered[:4]),trials_by_sample=by_sample)
    save(output,result);return result


def summarize():
    rows=[read(p) for p in (RUN/'cases').glob('*/*/*/*/validation_result.json')]
    groups={}
    for row in rows:
        key=(row['arm'],row['condition'],row['hand'])
        g=groups.setdefault(key,dict(arm=key[0],condition=key[1],hand=key[2],views=0,planned=0,invalid=0,executed=0,height=0,strict=0,top1_height=0,top1_executed=0,top4_height=0,top4_executed=0,hit4=0))
        g['views']+=1
        for k in ('planned','invalid','executed','height','strict','top1_height','top1_executed','top4_height','top4_executed','hit4'):g[k]+=int(row[k])
    complete=len(rows)==128
    save(RUN/'results.json',dict(complete=complete,finished_batches=len(rows),expected_batches=128,groups=list(groups.values()),rows=rows))
    lines=['# Shampoo 抽屉：搜索预算 × 优化步数消融','',f'完成 {len(rows)}/128 批；最终高度 ≥10 cm；CPU PhysX；固定8视角。',
        '', '|配置|条件|手|视角完成|全部成功/执行|Top1成功/执行|Top4成功/执行|Top4命中视角|NaN|', '|---|---|---|---:|---:|---:|---:|---:|---:|']
    for g in sorted(groups.values(),key=lambda g:(g['arm'],g['condition'],g['hand'])):
        lines.append(f"|{g['arm']}|{g['condition']}|{g['hand']}|{g['views']}/8|{g['height']}/{g['executed']}|{g['top1_height']}/{g['top1_executed']}|{g['top4_height']}/{g['top4_executed']}|{g['hit4']}/{g['views']}|{g['invalid']}|")
    lines+=['','C0=4×4/200+200；C1=8×8/200+200；C2=4×4/400+400；C3=8×8/400+400。',
       '全部候选口径不同预算不可直接比较；Top1/Top4在执行前按几何约束排序，未使用仿真结果排序。',
       'A/B历史权重不同；此实验比较各条件内部参数，不是固定权重的输入消融。',
       '只有全部视角完成后才能解读为最终结果；C0复用生成候选，但在新节点重新执行以控制环境差异。']
    (RUN/'RESULTS.md').write_text('\n'.join(lines)+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare-only',action='store_true')
    parser.add_argument('--workers-per-gpu',type=int,default=3)
    parser.add_argument('--validation-workers',type=int,default=8)
    args=parser.parse_args()
    RUN.mkdir(parents=True,exist_ok=True)
    lock=(RUN/'launcher.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    manifest=prepare()
    if args.prepare_only:
        print(json.dumps(dict(prepared=True,run=str(RUN),views=VIEWS)));return
    devices=subprocess.check_output(['nvidia-smi','--query-gpu=index','--format=csv,noheader'],text=True).split()
    assert set(devices)=={'0','1','2','3'},devices
    assert 1<=args.workers_per_gpu<=4 and 1<=args.validation_workers<=16
    save(RUN/'status.json',dict(stage='cpu_physx_smoke',pid=os.getpid(),started=time.time()))
    # Both embodiments must load/execute successfully before the expensive queue.
    for hand in ('barrett','shadowhand'):
        job=('C0',VIEWS[0],'A',hand);generation(job,0,manifest);validate(job,manifest)
    slots=queue.Queue()
    for _ in range(args.workers_per_gpu):
        for gpu in range(4):slots.put(gpu)
    tasks=[(arm,v,c,h) for arm in ('C3','C1','C2','C0') for v in VIEWS for c in ('A','B') for h in ('barrett','shadowhand')]
    errors=[]
    def work(job):
        gpu=slots.get();start=time.monotonic()
        try:
            if not (folder(job)/'generation_done.json').exists():
                generation(job,gpu,manifest)
                save(folder(job)/'generation_done.json',dict(complete=True,gpu=gpu,elapsed_seconds=time.monotonic()-start,reused=job[0]=='C0'))
            return job
        finally:slots.put(gpu)
    save(RUN/'status.json',dict(stage='generation_and_cpu_validation',pid=os.getpid(),gpus=4,generation_workers=4*args.workers_per_gpu,validation_workers=args.validation_workers))
    with futures.ThreadPoolExecutor(args.validation_workers) as vp, futures.ThreadPoolExecutor(4*args.workers_per_gpu) as gp:
        generation_jobs={gp.submit(work,j):j for j in tasks}; validation_jobs={}
        for f in futures.as_completed(generation_jobs):
            job=generation_jobs[f]
            try: validation_jobs[vp.submit(validate,f.result(),manifest)]=job
            except Exception as e:
                errors.append(dict(job=job,stage='generation',error=repr(e)));save(folder(job)/'error.json',errors[-1])
            with LOCK:summarize()
        for f in futures.as_completed(validation_jobs):
            try:f.result()
            except Exception as e:
                job=validation_jobs[f];errors.append(dict(job=job,stage='validation',error=repr(e)));save(folder(job)/'error.json',errors[-1])
            with LOCK:summarize()
    save(RUN/'status.json',dict(stage='failed' if errors else 'complete',errors=errors,finished=time.time()))
    summarize()
    if errors:raise SystemExit(1)


if __name__=='__main__':main()
