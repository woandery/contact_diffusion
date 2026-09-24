"""New-scene frozen FK-only comparison: reconstruction -> FK/physics -> DRO64.

All writes are isolated. No fitting, videos, hidden mesh fallback or score search.
"""
import concurrent.futures
import fcntl
import gzip
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np

HERE=Path(__file__).resolve().parent
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec)
    sys.modules[name]=m;spec.loader.exec_module(m);return m
terms=load('six_separated',HERE/'run_shampoo_fk_terms_trajectory.py')
run=terms.run;base=run.base
OUT=run.C/'outputs/fkonly_ranked_six_scene_dro64_20260915'
run.ROOT=OUT/'fk';run.ARMS={'J0':0.};base.OLD=OUT

def status(stage,**kw):base.save(OUT/'pipeline_status.json',dict(stage=stage,time=time.time(),pid=os.getpid(),**kw))
def execute(cmd,env,log,cwd=None):return base.execute(cmd,env,log,cwd or run.C)

def prepare_case(c,m):
    run.RUN=run.ROOT/c['id'];run.RUN.mkdir(parents=True,exist_ok=True);run.validation_scope['RUN']=run.RUN
    old=base.read(run.SOURCE/'manifest.json');config={'J0':{}};inputs={};audits=[]
    for p,h in old['code_sha256'].items():assert base.sha(p)==h,p
    for k,h in old['checkpoint_sha256'].items():assert base.sha(m['runtime'][k])==h
    for hand,p in old['config_paths']['E3'].items():
        dest=run.RUN/'configs'/f'J0_{hand}.yaml';dest.parent.mkdir(exist_ok=True)
        if dest.exists():assert base.sha(dest)==base.sha(p)
        else:shutil.copy2(p,dest)
        config['J0'][hand]=str(dest)
    legal=base.read(Path(c['observations'])/'visibility/legal_partial_views.json')['views']
    original=base.read(OUT/'observations'/c['id']/'visibility/legal_partial_views.json')['views']
    indices={Path(v['rgbd_capture_dir']).name:i for i,v in enumerate(original)}
    selected=[]
    for v in legal:
        view=Path(v['rgbd_capture_dir']).name;selected.append(dict(view=view,original_legal_index=indices[view],metadata=v))
        corrected=OUT/'generation'/c['id']/'corrected_inputs';geom=corrected/'sam3d'/view
        obs=Path(c['observations'])/'sam3d'/view;frame=corrected/'visibility/rgbd_views'/view/'metadata.json'
        assert base.read(frame)['coordinate_frame']=='robot_base'
        md=base.read(geom/'sam3d_fused_centered.json');t=np.asarray(md['robot_from_pointcloud_frame'])
        assert t.shape==(4,4) and np.allclose(t[:3,:3].T@t[:3,:3],np.eye(3),atol=1e-5)
        residual=float(np.max(np.abs(np.load(obs/'sam3d_fused_centered.npy')@t[:3,:3].T+t[:3,3]-np.load(geom/'sam3d_fused_robot_base.npy'))))
        assert residual<1e-5
        paths=[frame,geom/'sam3d_fused_centered.json',geom/'sam3d_fused_robot_base.npy',
               corrected/'visibility/rgbd_views'/view/'scene_partial_robot_base.npy',obs/'sam3d_fused_centered.npy',obs/'camera_partial_sam3d_centered.npy']
        for p in paths:
            if p.suffix=='.npy':
                a=np.load(p);assert a.ndim==2 and a.shape[1]==3 and len(a)>0 and np.isfinite(a).all(),p
        inputs[view]={str(p):base.sha(p) for p in paths};audits.append(dict(view=view,residual_m=residual))
    prep=run.RUN/'prepare_ablation.py'
    if prep.exists():assert base.sha(prep)==base.sha(run.SOURCE/'prepare_ablation.py')
    else:shutil.copy2(run.SOURCE/'prepare_ablation.py',prep)
    manifest=dict(old);manifest.update(case=c,runtime=m['runtime'],arms={'J0':0},config_paths=config,inputs=inputs,
        selected_views=selected,pilot_views=[],frame_audit=audits,env_steps=0,env_weight=0,fk_steps=200,fk_lr=.0075,
        protocol_id='six-unseen-targets-frozen-fkonly-ranker-s4p4-fk200-env0-v1',
        config_sha256={a:{h:base.sha(p) for h,p in ps.items()} for a,ps in config.items()},
        expected_batches=len(selected)*4,expected_particles=len(selected)*64,expected_top1_sets=len(selected)*16,
        ranking_fit=False,additional_fk_environment=False,independent_env=False,record_video=False,
        driver_sha256=base.sha(__file__),hook_sha256=base.sha(run.HOOK))
    path=run.RUN/'manifest.json'
    if path.exists():assert base.read(path)==manifest
    else:base.save(path,manifest)
    return manifest

def report(manifest):
    rows=[base.read(p) for p in (run.RUN/'cases/J0').glob('*/*/*/validation_result.json')]
    for r in rows:assert r['complete'] and r['executed']==16
    base.save(run.RUN/'particle_results.json',dict(complete=len(rows)==manifest['expected_batches'],rows=rows,
        expected_batches=manifest['expected_batches'],height=sum(r['height'] for r in rows),
        strict=sum(r['strict'] for r in rows),executed=sum(r['executed'] for r in rows)))
    return rows
run.report=report

def export_score(manifest):
    out=run.RUN/'ranking';out.mkdir(exist_ok=True);rows=[];hashes={}
    results=base.read(run.RUN/'particle_results.json');assert results['complete']
    for result in results['rows']:
        view,condition,hand=[result[k] for k in ('view','condition','hand')]
        p=run.RUN/'cases/J0'/view/condition/hand/'raw_object.json';payload=base.read(p);hashes[str(p)]=base.sha(p)
        mdpath=next(Path(p) for p in manifest['inputs'][view] if p.endswith('sam3d_fused_centered.json'))
        t=base.read(mdpath)['robot_from_pointcloud_frame']
        pc=np.load(Path(manifest['case']['observations'])/'sam3d'/view/'sam3d_fused_centered.npy')
        ts={(t['sample'],t['particle']):t for t in result['trials']}
        for record in payload['records']:
            fk=record['fk'];cs=sorted(fk['candidates'],key=lambda c:c['particle']);assert [c['particle'] for c in cs]==[0,1,2,3]
            labels=[ts[(record['sample_index'],c['particle'])] for c in cs]
            for c,tlabel in zip(cs,labels):assert abs(c['max_finger_contact_error_m']-tlabel['max_contact_error_m'])<1e-8
            rows.append(dict(view=view,condition=condition,hand=hand,sample=record['sample_index'],candidates=cs,
                labels=labels,source_file=str(p),robot_from_object=t,
                object_summary=dict(mean=pc.mean(0).tolist(),lower=pc.min(0).tolist(),upper=pc.max(0).tolist()),
                fk_metadata={k:fk[k] for k in ('joint_names','joint_lower','joint_upper','palm_normal_axis','grasp_local_approach_axis','target_contacts','selection_rank_mode')}))
    with gzip.open(out/'source_dataset.json.gz','wt') as f:json.dump(dict(rows=rows,source_sha256=hashes),f)
    scoring=dict(manifest);scoring['config_paths']={'E3':manifest['config_paths']['J0']}
    base.save(out/'manifest.json',scoring)
    closure=load('six_closure',HERE/'score_fk_particle_closure.py')
    closure.CACHE.clear();closure.base.ROOT=out;closure.base.SOURCE=out
    with concurrent.futures.ProcessPoolExecutor(4) as pool:
        for item in pool.map(closure.base.worker,[('A','barrett'),('A','shadowhand'),('B','barrett'),('B','shadowhand')]):print(item,flush=True)

def export_comparison(m,coverage,errors):
    rows=[]
    for c in m['cases']:
        legal=base.read(Path(c['observations'])/'visibility/legal_partial_views.json')['views']
        for v in legal:
            view=Path(v['rgbd_capture_dir']).name
            for hand in ('barrett','shadowhand'):
                folder=OUT/'dro'/c['id']/'views'/view/'simulation'/c['scene_factory']/('task_%03d'%c['task_index'])/hand
                raw=folder/'dro_candidates.json';filtered=folder/'dro_candidates_environment_filtered.json'
                summary=folder/'lift_validation/summary.json'
                d=base.read(filtered) if filtered.exists() else None
                s=base.read(summary) if summary.exists() else None
                retained=d['environment_filter']['retained_candidates'] if d else None
                executed=int(s['validated_candidates']) if s else 0
                trials=s['trials'] if s else [];assert len(trials)==executed
                rows.append(dict(case=c['id'],view=view,hand=hand,planned=64,
                    generated=len(base.read(raw)['records']) if raw.exists() else 0,
                    retained=retained,executed=executed,
                    complete=bool(d is not None and executed==retained),
                    height=sum(float(t['final_object_lift_m'])>=.10 for t in trials),
                    strict=sum(bool(t['success']) for t in trials),
                    source_summary=str(summary) if summary.exists() else None,
                    source_summary_sha256=base.sha(summary) if summary.exists() else None))
    rankable=[c['id'] for c in m['cases'] if all((OUT/'fk'/c['id']/'ranking'/name).exists() for name in
              ['source_dataset.json.gz']+[f'closure_features_{a}_{h}.json' for a in ('A','B') for h in ('barrett','shadowhand')])]
    base.save(OUT/'comparison_inputs.json',dict(cases=m['cases'],coverage=coverage,dro=rows,errors=errors,fk_rankable_cases=rankable,
                                               all_dro_complete=all(r['complete'] for r in rows)))

def sam3d_smoke(m,env):
    c=m['cases'][0];obs=Path(c['observations'])
    legal=base.read(obs/'visibility/legal_partial_views.json')['views']
    sample=max(legal,key=lambda v:v['partial_raw_point_count']);view=Path(sample['rgbd_capture_dir']).name
    def worker(gpu):
        root=OUT/'sam3d_device_smoke'/f'gpu{gpu}';root.mkdir(parents=True,exist_ok=True)
        spec=dict(runtime=m['runtime'],jobs=[dict(id='smoke',capture_dir=str(obs/'visibility/rgbd_views'/view),
                                                 output_dir=str(root/'reconstruction'))])
        base.save(root/'jobs.json',spec)
        py=Path(m['runtime']['sam3d_python']);venv=py.parent.parent;e=env.copy()
        e.update(CUDA_VISIBLE_DEVICES=str(gpu),CONDA_PREFIX=str(venv),CUDA_HOME=str(venv))
        for key,prefix in dict(PATH=venv/'bin',LD_LIBRARY_PATH=venv/'lib',
                               CPATH=venv/'targets/x86_64-linux/include',LIBRARY_PATH=venv/'targets/x86_64-linux/lib').items():
            e[key]=str(prefix)+':'+e.get(key,'')
        error=None
        try:execute([py,'-u',HERE/'sam3d_resident_batch.py','--worker-spec',root/'jobs.json','--output',root/'logs'],e,root/'worker.log')
        except Exception as ex:error=repr(ex)
        result=base.read(root/'logs/summary.json') if (root/'logs/summary.json').exists() else {}
        success=result.get('complete') and result.get('failures')==0
        return dict(gpu=gpu,success=bool(success),error=error,source_view=view,summary=result)
    with concurrent.futures.ThreadPoolExecutor(4) as pool:results=list(pool.map(worker,range(4)))
    base.save(OUT/'sam3d_device_smoke.json',dict(rows=results,note='Same-frame smoke; failure alone does not diagnose bad hardware'))
    return [str(r['gpu']) for r in results if r['success']]

def main():
    lock=(OUT/'comparison.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    m=base.read(OUT/'manifest.json');runtime=m['runtime'];env=base.base_env()
    assert base.read(OUT/'capture_status.json')['stage']=='complete','Capture must finish first'
    # True geometry is never substituted when SAM3D fails.
    cache=run.C/'outputs/extension3_shelf_basket_drawer_4x4_20260910_v2/torch_cache'
    env.update(SAM3D_CAMERA_CONVENTION='opencv',TORCH_HOME=str(cache),SAM3D_DINO_REPOSITORY=str(cache/'hub/facebookresearch_dinov2_main'))
    status('sam3d_device_smoke',gpus=[0,1,2,3])
    gpus=sam3d_smoke(m,env)
    status('sam3d',gpus=gpus)
    reconstruction_error=None
    if gpus:
        try:execute([runtime['contact_python'],HERE/'sam3d_resident_batch.py','--manifest',OUT/'manifest.json',
                     '--output',OUT/'resident','--gpus',','.join(gpus)],env,OUT/'logs/sam3d.log')
        except subprocess.CalledProcessError as e:reconstruction_error=str(e)
    else:reconstruction_error='No GPU passed identical-frame reconstruction smoke; no true-mesh substitution'
    resident=load('six_resident_cache',HERE/'sam3d_resident_batch.py');os.environ['SAM3D_CAMERA_CONVENTION']='opencv'
    coverage=[];usable=[]
    for c in m['cases']:
        obs=Path(c['observations']);legal=base.read(obs/'visibility/legal_partial_views.json')
        good=[];bad=[]
        for v in legal['views']:
            view=Path(v['rgbd_capture_dir']).name;job=dict(capture_dir=str(obs/'visibility/rgbd_views'/view),output_dir=str(obs/'sam3d'/view))
            (good if resident.cache_valid(job) else bad).append(v)
        coverage.append(dict(case=c['id'],legal_views=len(legal['views']),reconstructed_views=len(good),
                             failed_reconstruction_views=[Path(v['rgbd_capture_dir']).name for v in bad]))
        # DRO always keeps the full legal-camera inventory, independent of SAM3D availability.
        if legal['views']:
            execute([runtime['contact_python'],HERE/'materialize_fetchbench_world_to_robot_base.py',
                '--source-root',obs,'--output-root',OUT/'dro_inputs'/c['id'],'--task-config',c['task_config'],
                '--task-index',c['task_index'],'--skip-sam3d'],env,OUT/'logs'/f"materialize_dro_{c['id']}.log")
        if not good:continue
        partial=OUT/'finite_observations'/c['id'];(partial/'visibility').mkdir(parents=True,exist_ok=True)
        for link,target in [(partial/'visibility/rgbd_views',obs/'visibility/rgbd_views'),(partial/'sam3d',obs/'sam3d')]:
            if not link.exists():link.symlink_to(target,target_is_directory=True)
            else:assert link.resolve()==target.resolve()
        base.save(partial/'visibility/legal_partial_views.json',dict(legal,views=good))
        execute([runtime['contact_python'],HERE/'materialize_fetchbench_world_to_robot_base.py',
            '--source-root',partial,'--output-root',OUT/'generation'/c['id']/'corrected_inputs',
            '--task-config',c['task_config'],'--task-index',c['task_index']],env,OUT/'logs'/f"materialize_fk_{c['id']}.log")
        usable.append(dict(c,observations=str(partial)))
    base.save(OUT/'coverage.json',dict(rows=coverage,reconstruction_process_error=reconstruction_error))
    errors=[]
    for c in usable:
        status('fk_generation_validation',case=c['id'],gpus=[0,1,2,3],generation_workers=8,validation_workers=12)
        try:
            manifest=prepare_case(c,m)
            run.run_jobs(manifest,[v['view'] for v in manifest['selected_views']],['J0'],'six_scene_fkonly')
            report(manifest);export_score(manifest)
        except Exception as e:
            errors.append(dict(stage='FK',case=c['id'],error=repr(e)));base.save(OUT/'errors.json',errors)
    # Independent DRO baseline, unchanged64-candidate budget and native qouter/qinner.
    for c in m['cases']:
        count=next(x['legal_views'] for x in coverage if x['case']==c['id'])
        if not count:continue
        status('dro_generation_validation',case=c['id'],gpus=[0,1,2,3])
        e=env.copy();e.update(FETCHBENCH_ROOT=str(run.F),CONTACT_ROOT=str(run.C),DRO_ROOT=runtime['dro_root'],
            DRO_PYDEPS=runtime['dro_pydeps'],DRO_PYTHON=runtime['dro_python'],FETCHBENCH_PYTHON=str(run.P/'miniconda3/envs/fetchbench/bin/python'),
            CHECKPOINT=runtime['checkpoint_dro'],INPUT_ROOT=str(OUT/'dro_inputs'/c['id']),RUN_ROOT=str(OUT/'dro'/c['id']),
            TASK_INDEX=str(c['task_index']),SCENE_CONFIG=c['scene'],SCENE_FACTORY=c['scene_factory'],OBJECT_LABEL=c['id'],
            EXPECTED_LEGAL_VIEWS=str(count),GPU_COUNT='4',GPU_OFFSET='0',WORKERS_PER_GPU='2',VALIDATION_WORKERS='12',
            DRO_CANDIDATES='64',GENERATION_ATTEMPTS='4',TORCH_EXTENSIONS_DIR=str(base.SOURCE/'torch_extensions'))
        try:execute(['bash',HERE/'run_extension3_dro64_worker.sh'],e,OUT/'logs'/f"dro_{c['id']}.log",run.F)
        except Exception as ex:errors.append(dict(stage='DRO',case=c['id'],error=repr(ex)));base.save(OUT/'errors.json',errors)
    export_comparison(m,coverage,errors)
    status('ready_for_local_ranking' if not errors else 'partial_ready_for_local_ranking',errors=errors)
    print('REMOTE_FINISHED',OUT,flush=True)

if __name__=='__main__':
    try:main()
    except Exception as e:status('failed',error=repr(e));raise
