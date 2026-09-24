"""Append-only recovery of the frozen six-family experiment on four GPUs.

Original successful artifacts are read-only symlinks. Failed jobs get new
directories and exactly one retry of the original protocol. SAM3D thin-mask
crop-guard recovery is identified separately. CPU PhysX keeps all thresholds.
"""
import argparse
import concurrent.futures as cf
import fcntl
import gzip
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import queue
import shutil
import subprocess
import sys
import threading
import time

HERE = Path(__file__).resolve().parent
F = HERE.parent
P = F.parent
C = P / 'ContactDiffusion'
BASE = C / 'outputs/fk_top1_extension6_families_ab_dro64_20260915'
OUT = BASE / 'supplement_20260916'
sys.path.insert(0, str(BASE / 'ranker_pydeps'))
os.environ.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', SAM3D_CAMERA_CONVENTION='opencv')
LOCK = threading.Lock()
EVENTS = []


def read(p): return json.loads(Path(p).read_text())
def save(p, d):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + '.tmp'); tmp.write_text(json.dumps(d, indent=2) + '\n'); tmp.replace(p)
def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''): h.update(b)
    return h.hexdigest()
def load(name, p):
    spec = importlib.util.spec_from_file_location(name, p); m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m; spec.loader.exec_module(m); return m
def link(p, target):
    p = Path(p); target = Path(target); p.parent.mkdir(parents=True, exist_ok=True)
    if p.is_symlink(): assert p.resolve() == target.resolve()
    elif not p.exists(): p.symlink_to(target, target_is_directory=target.is_dir())
    else: raise RuntimeError(f'Will not replace {p}')
def event(stage, **kw):
    with LOCK:
        row = dict(stage=stage, time=time.time(), **kw); EVENTS.append(row)
        save(OUT/'events.json', EVENTS); print(json.dumps(row), flush=True)
def execute(cmd, env, log, cwd=F):
    log = Path(log); log.parent.mkdir(parents=True, exist_ok=True)
    with log.open('a') as f:
        f.write('\nCOMMAND ' + json.dumps(list(map(str, cmd))) + '\n'); f.flush()
        subprocess.run(list(map(str, cmd)), cwd=cwd, env=env, stdin=subprocess.DEVNULL,
                       stdout=f, stderr=subprocess.STDOUT, check=True)
def attempt(stage, key, fn, *args):
    try:
        result = fn(*args); event(stage, key=key, ok=True); return result
    except Exception as ex:
        event(stage, key=key, ok=False, error=repr(ex)); return None


def setup():
    OUT.mkdir(parents=True, exist_ok=True)
    m = read(BASE/'manifest.json')
    for name in ('ranker_pydeps','ranker_unified_frozen.joblib','ranker_original_reference.joblib','dro_inputs'):
        link(OUT/name, BASE/name)
    driver = load('supplement_frozen_driver', HERE/'run_fk_top1_extension6_20260915.py')
    driver.OUT = OUT; driver.__file__ = str(Path(__file__).resolve())
    ref = driver.reference()
    env = ref.base.base_env()
    # Supply CUDA libraries from the new node; do not modify its system setup.
    env['LD_LIBRARY_PATH'] += ':/usr/local/cuda-12.8/lib64'
    hashes = {}
    fk_missing = []
    dro_jobs = []
    sam_jobs = []
    resident = load('supplement_resident', HERE/'sam3d_resident_batch.py')
    for c in m['cases']:
        cid = c['id']; obs = Path(c['observations']); newobs = OUT/'observations'/cid
        link(newobs/'visibility', obs/'visibility')
        legal = read(obs/'visibility/legal_partial_views.json')['views']
        for v in legal:
            view = Path(v['rgbd_capture_dir']).name
            src = obs/'sam3d'/view; dst = newobs/'sam3d'/view
            job = dict(id=cid+'__'+view, capture_dir=str(obs/'visibility/rgbd_views'/view), output_dir=str(src))
            if resident.cache_valid(job): link(dst, src)
            else: sam_jobs.append(dict(job, output_dir=str(dst)))
            for a in ('A','B'):
                for h in ('barrett','shadowhand'):
                    rel = Path('fk')/cid/'cases/J0'/view/a/h
                    old = BASE/rel; dest = OUT/rel
                    p = old/'validation_result.json'
                    if p.exists() and read(p).get('complete'):
                        link(dest, old)
                        for n in ('validation_result.json','raw_object.json','all_particles.json','prepared.json'):
                            if (old/n).exists(): hashes[str(old/n)] = sha(old/n)
                    else: fk_missing.append((cid,view,a,h))
            for hand in ('barrett','shadowhand'):
                rel = Path('dro')/cid/'views'/view/'simulation'/c['scene_factory']/('task_%03d'%c['task_index'])/hand
                src = BASE/rel; dst = OUT/rel; fil = src/'dro_candidates_environment_filtered.json'
                summary = src/'lift_validation/summary.json'
                retained = read(fil)['environment_filter']['retained_candidates'] if fil.exists() else None
                complete = retained == 0 or (retained is not None and summary.exists() and read(summary)['validated_candidates'] == retained)
                if complete: link(dst, src)
                else:
                    dst.mkdir(parents=True, exist_ok=True)
                    for n in ('dro_candidates.json','dro_candidates_environment_filtered.json'):
                        if (src/n).exists() and not (dst/n).exists(): shutil.copy2(src/n,dst/n)
                    dro_jobs.append(dict(case=c,view=view,hand=hand,artifact=str(dst)))
                for p in (src/'dro_candidates.json',fil,summary):
                    if p.exists(): hashes[str(p)] = sha(p)
    save(OUT/'source_sha256.json', hashes)
    save(OUT/'manifest.json', dict(m, source_run=str(BASE), recovery=dict(
        original_artifacts_immutable=True, sam3d_thin_mask_crop_guard_only=True,
        fk_retry_original_seed_and_budget=True, dro_cached_only=True,
        dro_validation_min_goal_points=1, environment_clearance_m=.005,
        preclosure_displacement_m=.02, success_height_m=.10)))
    save(OUT/'inventory.json', dict(sam_jobs=sam_jobs, fk_missing=fk_missing, dro_jobs=dro_jobs))
    event('inventory', sam_missing=len(sam_jobs), fk_missing=len(fk_missing), dro_pending_batches=len(dro_jobs))
    return m, driver, ref, env, sam_jobs, dro_jobs


def dro_valid(job, m, env):
    import numpy as np
    c = job['case']; view=job['view']; h=job['hand']; dest=Path(job['artifact'])
    raw=dest/'dro_candidates.json'; filt=dest/'dro_candidates_environment_filtered.json'
    d=read(raw); f=read(filt); retained=int(f['environment_filter']['retained_candidates'])
    assert len(d['records'])==64 and len(f['records'])==retained
    assert f['environment_filter']['clearance_m']==.005
    summary=dest/'lift_validation/summary.json'
    if retained==0: return dict(executed=0)
    if summary.exists() and read(summary)['validated_candidates']==retained: return read(summary)
    rt=m['runtime']; pc=BASE/'dro_inputs'/c['id']/'visibility/rgbd_views'/view
    a=np.asarray(np.load(pc/'target_partial_robot_base.npy'),dtype=np.float32).reshape(-1,3)
    a=np.ascontiguousarray(a[np.isfinite(a).all(1)])
    assert len(a)>0 and hashlib.sha256(a.tobytes()).hexdigest()==d['input_sha256']
    digest=sha(raw); e=env.copy()
    e.update(CUDA_VISIBLE_DEVICES='',ASSET_PATH=str(F),PYTHONPATH=f'{F}/third_party/isaacgym/python:{F}/InfiniGym',
             LD_LIBRARY_PATH=f'{P}/miniconda3/envs/fetchbench/lib:'+env['LD_LIBRARY_PATH'],
             TORCH_EXTENSIONS_DIR=str(C/'outputs/extension3_shelf_basket_drawer_4x4_20260910_v2/torch_extensions'),MAX_JOBS='2')
    opts={
        'task':'FetchPtdDRORenderBarrett' if h=='barrett' else 'FetchPtdDRORenderShadow',
        'scene':'benchmark_eval/'+c['scene'],'task.solution.task_index':c['task_index'],
        'task.solution.physics_only':'true','task.solution.goal_pointcloud_override':pc/'target_partial_robot_base.npy',
        'task.solution.goal_pointcloud_override_source':'fetchbench_same_view_partial',
        'task.solution.scene_pointcloud_override':pc/'scene_partial_robot_base.npy',
        'task.solution.visualize_top_k':retained,'task.solution.min_scene_clearance':.005,
        'task.solution.reject_runtime_environment_contacts':'true','task.solution.lift.record_video':'false',
        'task.solution.lift.direct_closure':'true','task.solution.lift.max_preclosure_object_displacement':.02,
        'task.solution.lift.height':.25,'task.solution.lift.success_height':.10,
        'task.solution.dro.root':rt['dro_root'],'task.solution.dro.python':rt['dro_python'],
        'task.solution.dro.inference_script':F/'scripts/dro_generate_grasps.py',
        'task.solution.dro.checkpoint':rt['checkpoint_dro'],'task.solution.dro.candidates':64,
        'task.solution.dro.points':512,'task.solution.dro.optimization_steps':64,
        'task.solution.dro.seed':d['seed'],'task.solution.dro.device':'cpu','task.solution.dro.reuse_cache':'true',
        'task.solution.dro.min_goal_points':1,'task.env.enableCameraSensors':'false',
        'task.env.robot.asset_root':F/'InfiniGym/assets',
        'task.env.robot.urdf_file':f'urdf/dexterous/dro_{h}_physics.urdf',
        'task.solution.artifact_dir':dest.parents[2], 'seed':20260808,'num_threads':4,
        'pipeline':'cpu','sim_device':'cpu','rl_device':'cpu','graphics_device_id':-1,'headless':'true','force_render':'false'}
    execute([P/'miniconda3/envs/fetchbench/bin/python',HERE/'validate_dro_cached_only.py',
             *[f'{k}={v}' for k,v in opts.items()]],e,dest/'recovery_validate.log',F/'InfiniGym')
    assert sha(raw)==digest, 'Cached poses mutated'
    s=read(summary); assert s['validated_candidates']==retained and len(s['trials'])==retained
    assert sorted(int(t['candidate']) for t in s['trials'])==sorted(f['environment_filter']['accepted_indices'])
    save(dest/'recovery_audit.json',dict(raw_sha256=digest, cached_only=True, actual_cache_seed=d['seed'],
                                      points=len(a), validated=retained, success_gates_unchanged=True))
    return s


def dro_generate(job, gpu, m, env):
    dest=Path(job['artifact']); rt=m['runtime']; c=job['case']; h=job['hand']
    raw=dest/'dro_candidates.json'; filt=dest/'dro_candidates_environment_filtered.json'
    pc=BASE/'dro_inputs'/c['id']/'visibility/rgbd_views'/job['view']
    e=env.copy(); e.update(CUDA_VISIBLE_DEVICES=str(gpu),LANG='C.UTF-8',LC_ALL='C.UTF-8',
        PYTHONPATH=rt['dro_pydeps']+':'+rt['dro_root'],MPLCONFIGDIR=str(OUT/'mpl-cache'))
    if not raw.exists():
        for attempt_index in range(4):
            seed=20260808+c['task_index']+attempt_index*1000003
            try:
                execute([rt['dro_python'],HERE/'dro_generate_grasps.py','--dro-root',rt['dro_root'],
                    '--checkpoint',rt['checkpoint_dro'],'--input',pc/'target_partial_robot_base.npy',
                    '--output',raw,'--hand',h,'--candidates',64,'--points',512,'--optimization-steps',64,
                    '--seed',seed,'--device','cuda:0'],e,dest/'recovery_generate.log')
                save(dest/'recovery_generation_seed.json',dict(seed=seed,attempt=attempt_index+1));break
            except subprocess.CalledProcessError:
                if attempt_index==3: raise
    assert len(read(raw)['records'])==64
    if not filt.exists():
        execute([rt['dro_python'],HERE/'filter_dro_environment_candidates.py','--input',raw,
            '--scene-pointcloud',pc/'scene_partial_robot_base.npy','--clearance',.005,'--output',filt],e,dest/'recovery_filter.log')
    return job


def sam_worker(gpu, jobs, m, env):
    if not jobs: return
    dest=OUT/'resident'/f'gpu{gpu}'; save(dest/'jobs.json',dict(runtime=m['runtime'],jobs=jobs))
    rt=m['runtime']; ve=Path(rt['sam3d_python']).parent.parent; e=env.copy()
    cache=C/'outputs/extension3_shelf_basket_drawer_4x4_20260910_v2/torch_cache'
    e.update(CUDA_VISIBLE_DEVICES=str(gpu),CONDA_PREFIX=str(ve),CUDA_HOME=str(ve),
             SAM3D_CAMERA_CONVENTION='opencv',TORCH_HOME=str(cache),SAM3D_DINO_REPOSITORY=str(cache/'hub/facebookresearch_dinov2_main'))
    for key,prefix in dict(PATH=ve/'bin',LD_LIBRARY_PATH=ve/'lib',CPATH=ve/'targets/x86_64-linux/include',LIBRARY_PATH=ve/'targets/x86_64-linux/lib').items():
        e[key]=str(prefix)+':'+e.get(key,'')
    execute([rt['sam3d_python'],'-u',HERE/'sam3d_thin_mask_recovery.py','--worker-spec',dest/'jobs.json','--output',dest],e,dest/'worker.log',C)


def prepare_fk(m, ref, env):
    resident=load('supplement_cache_check',HERE/'sam3d_resident_batch.py'); coverage=[]; manifests=[]
    for c in m['cases']:
        obs=OUT/'observations'/c['id']; legal=read(obs/'visibility/legal_partial_views.json'); good=[]; bad=[]
        for v in legal['views']:
            view=Path(v['rgbd_capture_dir']).name
            job=dict(capture_dir=str(obs/'visibility/rgbd_views'/view),output_dir=str(obs/'sam3d'/view))
            (good if resident.cache_valid(job) else bad).append(v)
        coverage.append(dict(case=c['id'],legal_views=len(legal['views']),reconstructed_views=len(good),
                              failed_reconstruction_views=[Path(v['rgbd_capture_dir']).name for v in bad]))
        partial=OUT/'finite_observations'/c['id'];link(partial/'visibility/rgbd_views',obs/'visibility/rgbd_views');link(partial/'sam3d',obs/'sam3d')
        save(partial/'visibility/legal_partial_views.json',dict(legal,views=good))
        oldcorrect=BASE/'generation'/c['id']/'corrected_inputs'; corrected=OUT/'generation'/c['id']/'corrected_inputs'
        oldviews=read(BASE/'finite_observations'/c['id']/'visibility/legal_partial_views.json')['views']
        if len(good)==len(oldviews): link(corrected,oldcorrect)
        else:
            if corrected.is_symlink():
                assert corrected.resolve()==oldcorrect.resolve()
                previous=corrected.with_name('corrected_inputs_before_thin_mask')
                assert not previous.exists();corrected.rename(previous)
                shutil.copytree(oldcorrect,corrected)
            execute([m['runtime']['contact_python'],HERE/'materialize_fetchbench_world_to_robot_base.py',
                '--source-root',partial,'--output-root',corrected,'--task-config',c['task_config'],'--task-index',c['task_index']],
                env,OUT/'logs'/f'materialize_{c["id"]}.log',C)
            import numpy as np
            for p in oldcorrect.glob('sam3d/*/*.npy'):
                assert np.array_equal(np.load(p),np.load(corrected/p.relative_to(oldcorrect))),p
        prior=OUT/'fk'/c['id']/'manifest.json'
        if prior.exists():
            old_selected={v['view'] for v in read(prior)['selected_views']}
            new_selected={Path(v['rgbd_capture_dir']).name for v in good}
            if old_selected!=new_selected:
                assert old_selected < new_selected
                backup=prior.with_name('manifest_before_thin_mask.json')
                assert not backup.exists();prior.rename(backup)
        manifest=ref.prepare_case(dict(c,observations=str(partial)),m);manifests.append(manifest)
    save(OUT/'coverage.json',dict(rows=coverage));return manifests,coverage


def fk_case(manifest, ref, slots, vp, workers=4):
    run=ref.run; run.RUN=OUT/'fk'/manifest['case']['id'];run.validation_scope['RUN']=run.RUN
    generation=run.generation_function(manifest);jobs=[];validations=[]
    for v in manifest['selected_views']:
        for a in ('A','B'):
            for h in ('barrett','shadowhand'):
                job=('J0',v['view'],a,h);dest=run.folder(job)
                if (dest/'validation_result.json').exists(): continue
                prior=dest/'recovery_attempt.json'
                if prior.exists() and not read(prior)['ok']: continue
                jobs.append(job)
    def work(job):
        dest=run.folder(job); done=dest/'recovery_attempt.json'
        if done.exists():
            if read(done)['ok']: return job
            raise RuntimeError('Original-budget retry already exhausted; no additional random draws')
        gpu=slots.get();start=time.time()
        try:
            generation(job,gpu,manifest)
            save(done,dict(ok=True,gpu=gpu,seconds=time.time()-start));return job
        except Exception as ex:
            save(done,dict(ok=False,gpu=gpu,error=repr(ex),seconds=time.time()-start));raise
        finally:slots.put(gpu)
    event('fk_case_start',case=manifest['case']['id'],missing_batches=len(jobs))
    with cf.ThreadPoolExecutor(workers) as gp:
        pending={gp.submit(work,j):j for j in jobs}
        for future in cf.as_completed(pending):
            j=pending[future]
            try:
                future.result();validations.append(vp.submit(attempt,'fk_validate',list(j),run.validate,j,manifest))
                event('fk_generate',case=manifest['case']['id'],job=j,ok=True)
            except Exception as ex:event('fk_generate',case=manifest['case']['id'],job=j,ok=False,error=repr(ex))
    for f in validations:f.result()
    ref.report(manifest)


def incremental_features(case, condition, hand):
    """Reuse frozen features only for byte-identical saved poses/geometry."""
    dest=OUT/'top1/fk'/case; old=BASE/'top1/fk'/case
    with gzip.open(dest/'source_dataset.json.gz','rt') as f:data=json.load(f)
    with gzip.open(old/'source_dataset.json.gz','rt') as f:previous=json.load(f)
    key=lambda r:tuple(r[k] for k in ('view','condition','hand','sample'))
    previous_rows={key(r):r for r in previous['rows']}
    reused=set(); pending=[]
    selected=[r for r in data['rows'] if (r['condition'],r['hand'])==(condition,hand)]
    for r in selected:
        oldrow=previous_rows.get(key(r))
        if oldrow is not None:
            for field in ('candidates','robot_from_object','object_summary','fk_metadata'):
                assert json.dumps(r[field],sort_keys=True)==json.dumps(oldrow[field],sort_keys=True),(key(r),field)
            assert sha(r['source_file'])==previous['source_sha256'][oldrow['source_file']]
            reused.add(key(r))
        else:pending.append(r)
    feature_name=f'closure_features_{condition}_{hand}.json'
    oldfeatures=read(old/feature_name)
    rows=[r for r in oldfeatures['rows'] if key(r) in reused]
    audit_row=next((r for r in selected if key(r) in reused),None)
    recompute=pending+([audit_row] if audit_row is not None else [])
    audit_max_difference=0.0
    if recompute:
        target=OUT/'incremental_features'/f'{condition}_{hand}'
        work=target/'fk'/case;work.mkdir(parents=True,exist_ok=True)
        shutil.copy2(dest/'manifest.json',work/'manifest.json')
        with gzip.open(work/'source_dataset.json.gz','wt') as f:json.dump(dict(data,rows=recompute),f)
        recovery=load('supplement_new_features',HERE/'recover_six_scene_existing_top1.py')
        recovery.feature_worker(target,case,condition,hand)
        computed=read(work/feature_name)['rows']
        if audit_row is not None:
            reference={(*key(r),r['particle']):r['features'] for r in rows}
            for r in computed:
                if key(r)!=key(audit_row):continue
                old_values=reference[(*key(r),r['particle'])]
                assert r['features'].keys()==old_values.keys()
                for name,value in r['features'].items():
                    difference=abs(value-old_values[name]);audit_max_difference=max(audit_max_difference,difference)
                    assert math.isclose(value,old_values[name],rel_tol=1e-6,abs_tol=1e-8),(case,condition,hand,name,difference)
        rows += [r for r in computed if key(r) not in reused]
    assert len(rows)==4*len(selected)
    assert len({(*key(r),r['particle']) for r in rows})==len(rows)
    save(dest/feature_name,dict(rows=rows,reused_sets=len(reused),computed_sets=len(pending),
        no_pose_changes=True,source_features_sha256=sha(old/feature_name)))
    save(dest/f'feature_recompute_audit_{condition}_{hand}.json',dict(
        checked_particles=4 if audit_row is not None else 0,max_absolute_difference=audit_max_difference,passed=True))


def main(smoke_only=False):
    OUT.mkdir(parents=True,exist_ok=True)
    lock=(OUT/'recovery.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    m,driver,ref,env,sam_jobs,dro_jobs=setup()
    save(OUT/'pipeline_status.json',dict(stage='running',pid=os.getpid(),host=os.uname().nodename,started=time.time()))
    cached=[j for j in dro_jobs if (Path(j['artifact'])/'dro_candidates_environment_filtered.json').exists()]
    missing=[j for j in dro_jobs if j not in cached]
    # One actual retained candidate batch for each hand before bulk validation.
    smoke=[]
    for h in ('barrett','shadowhand'):
        j=next(j for j in cached if j['hand']==h and read(Path(j['artifact'])/'dro_candidates_environment_filtered.json')['records'])
        dro_valid(j,m,env);smoke.append(j);event('physics_smoke',hand=h,ok=True)
    if smoke_only:return
    with cf.ThreadPoolExecutor(8) as dvp, cf.ThreadPoolExecutor(4) as fvp:
        validations=[dvp.submit(attempt,'dro_validate',j,dro_valid,j,m,env) for j in cached if j not in smoke]
        with cf.ThreadPoolExecutor(4) as gp:
            futures=[gp.submit(attempt,'sam3d_thin_mask',gpu,sam_worker,gpu,sam_jobs[gpu::4],m,env) for gpu in range(4)]
            for f in futures:f.result()
        with cf.ThreadPoolExecutor(4) as gp:
            futures={gp.submit(dro_generate,j,i%4,m,env):j for i,j in enumerate(missing)}
            for f in cf.as_completed(futures):
                j=futures[f]
                try:f.result();validations.append(dvp.submit(attempt,'dro_validate',j,dro_valid,j,m,env))
                except Exception as ex:event('dro_generate',key=j,ok=False,error=repr(ex))
        manifests,coverage=prepare_fk(m,ref,env)
        slots=queue.Queue()
        for _ in range(2):
            for gpu in range(4):slots.put(gpu)
        for manifest in manifests:
            fk_case(manifest,ref,slots,fvp,workers=8)
        for f in validations:f.result()
    for manifest in manifests:
        attempt('ranking',manifest['case']['id'],driver.score_case,manifest)
    errors=[r for r in EVENTS if r.get('ok') is False]
    # Ranking baseline files are copied only if no new labels were produced and
    # scoring failed; otherwise never pretend that new results were ranked.
    driver.summary(ref,m,coverage,errors)
    hashes=read(OUT/'source_sha256.json')
    for p,digest in hashes.items():assert sha(p)==digest,p
    save(OUT/'source_integrity.json',dict(unchanged=True,files=len(hashes),checked_at=time.time()))
    save(OUT/'pipeline_status.json',dict(stage='finished',pid=os.getpid(),finished=time.time(),
        complete=read(OUT/'RESULTS.json')['complete'],original_artifacts_unchanged=True,errors=len(errors)))
    event('finished',complete=read(OUT/'RESULTS.json')['complete'])


def append_reconstructions():
    """After the first queue finishes, add only newly recovered SAM3D views."""
    global EVENTS
    lock=(OUT/'recovery.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX)
    assert read(OUT/'pipeline_status.json')['stage']=='finished'
    EVENTS=read(OUT/'events.json')
    m=read(BASE/'manifest.json');c=next(c for c in m['cases'] if c['id']=='cellphone_deskwall')
    driver=load('supplement_expanded_driver',HERE/'run_fk_top1_extension6_20260915.py')
    driver.OUT=OUT;driver.__file__=str(Path(__file__).resolve());ref=driver.reference();env=ref.base.base_env()
    resident=load('supplement_expanded_cache',HERE/'sam3d_resident_batch.py')
    obs=OUT/'observations'/c['id'];legal=read(obs/'visibility/legal_partial_views.json')
    good=[]
    for v in legal['views']:
        view=Path(v['rgbd_capture_dir']).name
        if resident.cache_valid(dict(capture_dir=str(obs/'visibility/rgbd_views'/view),output_dir=str(obs/'sam3d'/view))):good.append(v)
    mp=OUT/'fk'/c['id']/'manifest.json';oldmanifest=read(mp)
    if len(good)==len(oldmanifest['selected_views']):
        event('append_reconstruction_no_new_views');return
    assert len(good)>len(oldmanifest['selected_views'])
    corrected=OUT/'generation'/c['id']/'corrected_inputs'
    assert corrected.is_symlink() and corrected.resolve()==(BASE/'generation'/c['id']/'corrected_inputs').resolve()
    # Preserve the previous link/manifest, then materialize to our own directory.
    previous=corrected.with_name('corrected_inputs_before_thin_mask')
    assert not previous.exists();corrected.rename(previous)
    shutil.copytree(previous.resolve(),corrected)
    mp.rename(mp.with_name('manifest_before_thin_mask.json'))
    save(OUT/'pipeline_status.json',dict(stage='appending_recovered_views',pid=os.getpid(),time=time.time()))
    manifests,coverage=prepare_fk(m,ref,env)
    manifest=next(d for d in manifests if d['case']['id']==c['id'])
    slots=queue.Queue()
    for _ in range(2):
        for gpu in range(4):slots.put(gpu)
    event('append_reconstructions',new_views=len(good)-len(oldmanifest['selected_views']),generation_workers=8)
    with cf.ThreadPoolExecutor(8) as vp:fk_case(manifest,ref,slots,vp,workers=8)
    driver.score_case(manifest)
    driver.summary(ref,m,coverage,[r for r in EVENTS if r.get('ok') is False])
    hashes=read(OUT/'source_sha256.json')
    for p,digest in hashes.items():assert sha(p)==digest,p
    save(OUT/'source_integrity.json',dict(unchanged=True,files=len(hashes),checked_at=time.time()))
    save(OUT/'pipeline_status.json',dict(stage='finished',pid=os.getpid(),finished=time.time(),
        complete=read(OUT/'RESULTS.json')['complete'],original_artifacts_unchanged=True,thin_mask_views_appended=True))
    event('finished_after_reconstruction_append',complete=read(OUT/'RESULTS.json')['complete'])


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--smoke-only',action='store_true')
    p.add_argument('--append-reconstructions',action='store_true')
    p.add_argument('--feature-case');p.add_argument('--condition');p.add_argument('--hand');args=p.parse_args()
    if args.feature_case:
        incremental_features(args.feature_case,args.condition,args.hand)
    elif args.append_reconstructions:
        append_reconstructions()
    else:
        try:main(args.smoke_only)
        except Exception as ex:
            save(OUT/'pipeline_status.json',dict(stage='failed',error=repr(ex),time=time.time()));raise
