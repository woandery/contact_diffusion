"""Manifest-based multi-scene A/B + DRO generation queue. Does not run PhysX.

Observation capture is an explicit upstream interface, not inferred from a name.
Run --check before reserving compute. Use --run only on the allocated node.
"""
import argparse, concurrent.futures, contextlib, fcntl, hashlib, json, os, queue, re, subprocess, sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
def gpu_slots(gpu_ids, workers):
    """Round-robin initial allocation, then reuse the first released slot."""
    slots=queue.Queue()
    for _ in range(workers):
        for g in gpu_ids:slots.put(g)
    return slots
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()
def dump(p,d):
    p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(d,indent=2,ensure_ascii=False));tmp.replace(p)
def read(p):return json.loads(Path(p).read_text())
def validate_manifest(m):
    ids=set();observations=set()
    for c in m['cases']:
        if not re.fullmatch('[a-zA-Z0-9_-]+',c['id']) or c['id'] in ids:raise ValueError('Invalid/duplicate case id')
        ids.add(c['id'])
        for k in ['scene','scene_factory','task_config','task_index','object_index','observations']:assert k in c,(c['id'],k)
        assert isinstance(c['task_index'],int) and c['task_index']>=0
        assert isinstance(c['object_index'],int) and c['object_index']>=0
        assert c['observations'] not in observations,'Each target needs independent observations'
        observations.add(c['observations'])
    assert ids,'Empty task manifest'
def legal(c):
    d=read(Path(c['observations'])/'visibility/legal_partial_views.json')
    assert d['task_index']==c['task_index'],c['id']
    assert d['scene']==c['scene'],(d['scene'],c['scene'])
    assert d['views'],f"{c['id']}: no legal views"
    seen=set()
    for v in d['views']:
        assert v['object_index']==c['object_index'],c['id']
        view=Path(v['rgbd_capture_dir']).name
        assert view not in seen,'Duplicate view';seen.add(view)
        assert re.fullmatch(r'[a-zA-Z0-9_.+\-]+',view),view
        p=Path(c['observations'])/'visibility/rgbd_views'/view
        md=read(p/'metadata.json')
        assert md['coordinate_frame']=='fetchbench_world','Use original WORLD capture; never re-correct base inputs'
        assert md['task_index']==c['task_index'] and md['target_object_index']==c['object_index']
        assert md['scene_config_path'].rstrip('/').split('/')[-1]==c['scene_factory']
        for f in ['camera_00_rgb.png','camera_00_target_mask.png','camera_00_depth.npy','camera_00_pointmap_robot_base.npy','target_partial_robot_base.npy','scene_partial_robot_base.npy']:
            assert (p/f).is_file(),str(p/f)
    return d
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    mode=p.add_mutually_exclusive_group(required=True);mode.add_argument('--plan',action='store_true');mode.add_argument('--check',action='store_true');mode.add_argument('--run',action='store_true')
    p.add_argument('--gpus',default='0,1,2,3,4,5,6,7');p.add_argument('--workers-per-gpu',type=int,default=6)
    p.add_argument('--prepare-workers-per-gpu',type=int,default=1,
                   help='Separate SAM3D preparation concurrency; never inherits generation concurrency')
    p.add_argument('--cpu-threads-per-worker',type=int,default=1)
    p.add_argument('--sets',type=int,default=4);p.add_argument('--particles',type=int,default=4);p.add_argument('--dro-candidates',type=int,default=4)
    p.add_argument('--max-views',type=int,default=0);p.add_argument('--methods',nargs='+',choices=['AB','DRO'],default=['AB','DRO'])
    a=p.parse_args();m=read(a.manifest);validate_manifest(m)
    assert min(a.sets,a.particles,a.dro_candidates,a.workers_per_gpu,a.prepare_workers_per_gpu,a.cpu_threads_per_worker)>0 and a.max_views>=0
    gpu_ids=a.gpus.split(',');assert all(x.isdigit() for x in gpu_ids) and len(set(gpu_ids))==len(gpu_ids)
    project=Path(m['project_root']);contact=project/'ContactDiffusion'
    paths={k:Path(v) for k,v in m['runtime'].items()}
    for c in m['cases']:
        print(f"{c['id']}: {c['scene']} task={c['task_index']} target={c['object_index']} observations={c['observations']}")
    print(f'A/B {a.sets}x{a.particles}, FK200 lr=.0075, ENV200 lr=.003 w10; DRO={a.dro_candidates}; GPUs={gpu_ids}')
    print(f'Generation slots={len(gpu_ids)*a.workers_per_gpu}; preparation slots={len(gpu_ids)*a.prepare_workers_per_gpu}; CPU threads/worker={a.cpu_threads_per_worker}')
    if a.plan:return
    required=['contact_python','sam3d_python','dro_python','dro_root','dro_pydeps','checkpoint_a','checkpoint_b','checkpoint_dro','config_barrett','config_shadow','sam3d_root','sam3d_config']
    for k in required:assert paths[k].exists(),f'Missing {k}: {paths[k]}'
    needed=[contact/'scripts'/f for f in ['infer_contactdiffusion_camera_partial_v5_six.py','infer_contactdiffusion_explicit_condition_v6.py','transform_contactdiffusion_candidates.py','select_fullpc_environment_weight_ensemble_constraints.py']]
    needed += [contact/'scripts'/f for f in ['refine_contactdiffusion_environment_visible_scene.py','refine_contactdiffusion_environment_six.py']]
    needed += [ROOT/'scripts'/f for f in ['reconstruct_fetchbench_sam3d.py','materialize_fetchbench_world_to_robot_base.py','dro_generate_grasps.py','filter_dro_environment_candidates.py']]
    for f in needed:assert f.is_file(),str(f)
    # Verify high-lr config without importing training code.
    cmd=[str(paths['contact_python']),'-c','import sys,yaml; d=yaml.safe_load(open(sys.argv[1])); f=d["fk_optimization"]; assert f["steps"]==200 and abs(f["learning_rate"]-.0075)<1e-10',str(paths['config_barrett'])]
    for k in ['config_barrett','config_shadow']:
        cmd[-1]=str(paths[k]);subprocess.run(cmd,check=True)
    legal_data={}
    for c in m['cases']:
        assert Path(c['task_config']).is_file(),c['task_config']
        legal_data[c['id']]=legal(c)
        subprocess.run([str(paths['contact_python']),'-c','import numpy as np,sys; d=np.load(sys.argv[1]); i=int(sys.argv[2]); assert i<len(d["task_init_state"]); assert int(d["task_obj_index"][i])==int(sys.argv[3])',c['task_config'],str(c['task_index']),str(c['object_index'])],check=True)
    print('CHECK PASSED. Observation identity, runtime files and task target bindings checked.')
    if a.check:return
    assert a.output.resolve()!=ROOT and ROOT/'Task' not in a.output.resolve().parents
    devices=subprocess.check_output(['nvidia-smi','--query-gpu=index','--format=csv,noheader'],text=True).split()
    assert set(gpu_ids)<=set(devices),(gpu_ids,devices)
    a.output.mkdir(parents=True,exist_ok=True)
    lock=(a.output/'queue.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    frozen=dict(manifest=m,sets=a.sets,particles=a.particles,dro_candidates=a.dro_candidates,max_views=a.max_views,methods=a.methods,
        hashes={k:sha(paths[k]) for k in ['checkpoint_a','checkpoint_b','checkpoint_dro','config_barrett','config_shadow']},
        code_hashes={str(f):sha(f) for f in needed+[Path(__file__),ROOT/'scripts/run_fetchbench_mug_111cam_ab_w10_4gpu.sh']},
        inputs={c['id']:dict(task=sha(c['task_config']),legal=legal_data[c['id']],files={str(f):sha(f) for v in legal_data[c['id']]['views'] for f in (Path(c['observations'])/'visibility/rgbd_views'/Path(v['rgbd_capture_dir']).name).iterdir() if f.is_file()}) for c in m['cases']})
    freeze=a.output/'frozen_manifest.json'
    if freeze.exists():assert read(freeze)==frozen,'Configuration/source changed: use a NEW output directory'
    else:dump(freeze,frozen)
    dump(a.output/'scheduling.json',dict(gpus=gpu_ids,workers_per_gpu=a.workers_per_gpu,
        prepare_workers_per_gpu=a.prepare_workers_per_gpu,cpu_threads_per_worker=a.cpu_threads_per_worker))
    slots=gpu_slots(gpu_ids,a.workers_per_gpu)
    prepare_slots=gpu_slots(gpu_ids,a.prepare_workers_per_gpu)
    def run_command(cmd,env,log):
        log.parent.mkdir(parents=True,exist_ok=True)
        with log.open('a') as f:
            f.write('\nCOMMAND '+json.dumps(cmd)+'\n');f.flush()
            subprocess.run(cmd,env=env,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
    def base_env(c,g):
        e=os.environ.copy();e.update(CONTACTDIFF_ROOT=str(contact),CONTACTDIFF_PYTHON=str(paths['contact_python']),FETCHBENCH_ROOT=str(ROOT),
            SAM3D_ROOT=str(paths['sam3d_root']),SAM3D_PYTHON=str(paths['sam3d_python']),SAM3D_CHECKPOINT_CONFIG=str(paths['sam3d_config']),
            CONTACTDIFF_CHECKPOINT_A=str(paths['checkpoint_a']),CONTACTDIFF_CHECKPOINT_B=str(paths['checkpoint_b']),CONFIG_BARRETT=str(paths['config_barrett']),CONFIG_SHADOW=str(paths['config_shadow']),
            CONTACT_SETS=str(a.sets),PARTICLES=str(a.particles),FK_STEPS='200',ENV_STEPS='200',ENV_LEARNING_RATE='.003',
            OBJECT_PREFIX=c['id'],GPU_COUNT='1',GPU_OFFSET=g,CASE_WORKERS_PER_GPU='1',HAND_FILTER='all',MAX_VIEWS='0')
        for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']:
            e[k]=str(a.cpu_threads_per_worker)
        return e
    def prepare(c):
        g=prepare_slots.get()
        try:
            out=a.output/c['id'];status=out/'prepare_done.json'
            if status.exists():return
            e=base_env(c,g);obs=Path(c['observations']);e.update(RUN_ROOT=str(out/'AB'),SOURCE_ROOT=str(obs),CONDITION_SOURCE_ROOT=str(obs),RECONSTRUCT_ONLY='true',REUSE_SAM3D='false')
            run_command(['bash',str(ROOT/'scripts/run_fetchbench_mug_111cam_ab_w10_4gpu.sh')],e,out/'prepare.log')
            run_command([str(paths['contact_python']),str(ROOT/'scripts/materialize_fetchbench_world_to_robot_base.py'),'--source-root',str(obs),'--output-root',str(out/'corrected_inputs'),'--task-config',c['task_config'],'--task-index',str(c['task_index'])],e,out/'prepare.log')
            audit=read(out/'corrected_inputs/coordinate_fix_manifest.json');assert audit['maximum_roundtrip_error_m']<1e-6
            dump(status,dict(complete=True,gpu=g))
        finally:prepare_slots.put(g)
    def case_view(job):
        c,v=job;g=slots.get();view=Path(v['rgbd_capture_dir']).name;out=a.output/c['id'];status=out/'status'/f'{view}.json'
        try:
            if status.exists() and read(status).get('complete'):
                saved=read(status).get('artifacts',{})
                if saved and all(Path(f).exists() and sha(f)==digest for f,digest in saved.items()):return
                raise RuntimeError('Completed artifacts changed/missing; use a new output root or inspect this case')
            e=base_env(c,g);correct=out/'corrected_inputs';sub=out/'view_inputs'/view
            (sub/'visibility').mkdir(parents=True,exist_ok=True)
            for link,target in [(sub/'visibility/rgbd_views',correct/'visibility/rgbd_views'),(sub/'sam3d',correct/'sam3d')]:
                if not link.exists():link.symlink_to(target.resolve(),target_is_directory=True)
            d=read(correct/'visibility/legal_partial_views.json');d['views']=[x for x in d['views'] if Path(x['rgbd_capture_dir']).name==view];assert len(d['views'])==1
            dump(sub/'visibility/legal_partial_views.json',d)
            e.update(RUN_ROOT=str(out/'AB'),SOURCE_ROOT=str(sub),CONDITION_SOURCE_ROOT=c['observations'],RECONSTRUCT_ONLY='false',REUSE_SAM3D='true')
            e.update(LOG_ROOT=str(out/'AB/logs'/view),STATUS_ROOT=str(out/'AB/worker_status'/view),
                     MPLCONFIGDIR=str(out/'mpl_cache'/view))
            # Original full legal-view index must survive single-view dispatch for seeds.
            e['VIEW_INDEX_OFFSET']=str(next(i for i,x in enumerate(legal_data[c['id']]['views']) if Path(x['rgbd_capture_dir']).name==view))
            if 'AB' in a.methods:run_command(['bash',str(ROOT/'scripts/run_fetchbench_mug_111cam_ab_w10_4gpu.sh')],e,out/'logs'/f'{view}_AB.log')
            if 'DRO' in a.methods:
                e.update(CUDA_VISIBLE_DEVICES=g,PYTHONPATH=str(paths['dro_pydeps'])+':'+str(paths['dro_root']))
                for h in ['barrett','shadowhand']:
                    dest=out/'DRO'/view/h;dest.mkdir(parents=True,exist_ok=True);raw=dest/'dro_candidates.json'
                    cmd=[str(paths['dro_python']),str(ROOT/'scripts/dro_generate_grasps.py'),'--dro-root',str(paths['dro_root']),'--checkpoint',str(paths['checkpoint_dro']),'--input',str(correct/'visibility/rgbd_views'/view/'target_partial_robot_base.npy'),'--output',str(raw),'--hand',h,'--candidates',str(a.dro_candidates),'--points','512','--optimization-steps','64','--seed',str(20260808+c['task_index']),'--device','cuda:0']
                    run_command(cmd,e,out/'logs'/f'{view}_DRO_{h}.log');assert len(read(raw)['records'])==a.dro_candidates
                    run_command([str(paths['dro_python']),str(ROOT/'scripts/filter_dro_environment_candidates.py'),'--input',str(raw),'--scene-pointcloud',str(correct/'visibility/rgbd_views'/view/'scene_partial_robot_base.npy'),'--clearance','.005','--output',str(dest/'dro_candidates_environment_filtered.json')],e,out/'logs'/f'{view}_DRO_{h}.log')
            artifacts=[]
            if 'AB' in a.methods:
                for condition in ['A','B']:
                    for h in ['barrett','shadowhand']:
                        f=out/'AB/views'/view/condition/'selected_w10'/f"{c['id']}_{h}.json"
                        d=read(f);assert len(d['records'])==a.sets
                        assert all(len(r['fk']['candidates'])==1 for r in d['records'])
                        artifacts.append(f)
            if 'DRO' in a.methods:
                for h in ['barrett','shadowhand']:
                    artifacts.extend((out/'DRO'/view/h).glob('*.json'))
            dump(status,dict(complete=True,view=view,gpu=g,methods=a.methods,artifacts={str(f):sha(f) for f in artifacts}));print('DONE',c['id'],view,flush=True)
        except Exception as exc:
            dump(status,dict(complete=False,error=str(exc)));raise
        finally:slots.put(g)
    with concurrent.futures.ThreadPoolExecutor(len(gpu_ids)*a.prepare_workers_per_gpu) as pool:
        list(pool.map(prepare,m['cases']))
    with concurrent.futures.ThreadPoolExecutor(len(gpu_ids)*a.workers_per_gpu) as pool:
        jobs=[(c,v) for c in m['cases'] for v in legal_data[c['id']]['views'][:a.max_views or None]]
        futures=[pool.submit(case_view,j) for j in jobs];errors=[]
        for f in concurrent.futures.as_completed(futures):
            try:f.result()
            except Exception as exc:errors.append(str(exc))
    dump(a.output/'summary.json',dict(complete=not errors,view_jobs=len(jobs),errors=errors,physics_validation_run=False))
    if errors:raise SystemExit(1)
if __name__=='__main__':main()
