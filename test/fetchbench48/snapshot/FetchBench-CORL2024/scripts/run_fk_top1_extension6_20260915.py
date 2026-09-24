"""Detached, resumable new-six-scene FK-only A/B Top1 + DRO64 pipeline.

Existing FK/physics implementations are reused without energy changes. Complete
sets are exported even when other batches fail; summaries are written per stage.
No external GUI/video, ranking training, or true-mesh reconstruction fallback.
"""
import argparse
import concurrent.futures
import fcntl
import gzip
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

os.environ.update(OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
HERE=Path(__file__).resolve().parent
F=HERE.parent;P=F.parent;C=P/'ContactDiffusion'
OUT=C/'outputs/fk_top1_extension6_families_ab_dro64_20260915'
sys.path.insert(0,str(OUT/'ranker_pydeps'))
KEYS=[(a,h) for a in ('A','B') for h in ('barrett','shadowhand')]


def read(p):return json.loads(Path(p).read_text())
def save(p,d):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    temp=p.with_suffix(p.suffix+'.tmp');temp.write_text(json.dumps(d,indent=2)+'\n');temp.replace(p)
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()
def load(name,p):
    spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec)
    sys.modules[name]=m;spec.loader.exec_module(m);return m
def status(stage,**kw):save(OUT/'pipeline_status.json',dict(stage=stage,pid=os.getpid(),time=time.time(),**kw))
def execute(cmd,env,log,cwd=F):
    log.parent.mkdir(parents=True,exist_ok=True)
    with log.open('a') as f:
        f.write('\nCOMMAND '+json.dumps([str(x) for x in cmd])+'\n');f.flush()
        subprocess.run([str(x) for x in cmd],env=env,cwd=cwd,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,check=True)


def reference():
    ref=load('extension_fixed_fk',HERE/'run_fkonly_six_scene_dro.py')
    ref.OUT=OUT;ref.run.ROOT=OUT/'fk';ref.base.OLD=OUT
    return ref


def preflight(m):
    import numpy as np
    import joblib,sklearn
    assert sklearn.__version__=='1.7.2'
    freeze=m['ranker_freeze']
    assert sha(OUT/'ranker_unified_frozen.joblib')==freeze['unified_model_sha256']
    assert sha(OUT/'ranker_original_reference.joblib')==freeze['original_model_sha256']
    model=joblib.load(OUT/'ranker_unified_frozen.joblib')
    assert model['name']=='hgb_diverse' and len(model['features'])==502
    assert np.isfinite(model['model'].predict_proba(np.zeros((1,502),dtype=np.float32))).all()
    for p,digest in freeze['historical_policy']['sha256'].items():
        if '/scripts/' in p:assert sha(HERE/Path(p).name)==digest,p
    roots=read(OUT/'asset_roots.json')['roots']
    for rel in roots:
        assert not Path(rel).is_absolute() and '..' not in Path(rel).parts
        assert (F/rel).exists(),rel
    for c in m['cases']:
        p=Path(c['task_config']);assert sha(p)==c['task_config_sha256']
        assert sha(p.parent/'asset_config.json')==c['asset_config_sha256']
        d=np.load(p);assert int(d['task_obj_index'][c['task_index']])==c['object_index']
        assert str(d['task_obj_label'][c['task_index']])==c['placement']
    for k,p in m['runtime'].items():assert Path(p).exists(),(k,p)
    icd=None
    for p in (Path('/usr/share/vulkan/icd.d/nvidia_icd.json'),Path('/etc/vulkan/icd.d/nvidia_icd.json')):
        if p.is_file() and p.stat().st_size:
            conf=read(p)
            if conf.get('ICD',{}).get('library_path'):icd=str(p);break
    assert icd,'No valid NVIDIA Vulkan ICD JSON; no system files will be modified'
    ref=reference();old=read(ref.run.SOURCE/'manifest.json')
    for p,h in old['code_sha256'].items():assert sha(p)==h,p
    for k,h in old['checkpoint_sha256'].items():assert sha(m['runtime'][k])==h,k
    save(OUT/'preflight.json',dict(complete=True,hostname=os.uname().nodename,vulkan_icd=icd,
        asset_roots=len(roots),sklearn=sklearn.__version__,model_verified=True,source_code_verified=True,
        driver_sha256=sha(__file__),time=time.time()))
    return icd,ref


def capture(c,gpu,icd):
    dest=Path(c['observations'])/'visibility';done=dest/'capture_complete.json'
    if done.exists():
        assert read(done)['complete'];return
    lib=C/'outputs/extension3_shelf_basket_drawer_4x4_20260910_v2/runtime_lib'
    env=os.environ.copy();env.update(ASSET_PATH=str(F),PYTHONPATH=f'{F}/third_party/isaacgym/python:{F}/InfiniGym',
        LD_LIBRARY_PATH=f'{P}/miniconda3/envs/fetchbench/lib:{lib}:/usr/local/cuda-12.8/lib64',
        VK_ICD_FILENAMES=icd,VK_LOADER_LAYERS_DISABLE='~implicit~',CAPTURE_OUTPUT=str(dest),
        CAPTURE_TARGET_RGB='1,1,1',PYTHONUNBUFFERED='1',MAX_JOBS='2',
        TORCH_EXTENSIONS_DIR=str(C/'outputs/extension3_shelf_basket_drawer_4x4_20260910_v2/torch_extensions'))
    cmd=[P/'miniconda3/envs/fetchbench/bin/python','isaacgymenvs/capture_extension_111.py',
        'task=FetchPtdDRORenderBarrett',f"scene=benchmark_eval/{c['scene']}",f"task.solution.task_index={c['task_index']}",
        f'task.env.robot.asset_root={F}/InfiniGym/assets/contactdiff_hands/barrett',
        'task.env.robot.urdf_file=contactdiff_v4_barrett_physics.urdf','pipeline=cpu','sim_device=cpu',
        'rl_device=cpu','headless=true','force_render=false',f'graphics_device_id={gpu}','seed=20260808']
    execute(cmd,env,OUT/'logs'/f"capture_{c['id']}.log",F/'InfiniGym')
    assert read(done)['complete'] and read(done)['candidate_views']==111
    legal=read(dest/'legal_partial_views.json')
    assert legal['task_index']==c['task_index']
    assert all(v['object_index']==c['object_index'] for v in legal['views'])
    save(OUT/'capture_progress'/f"{c['id']}.json",dict(complete=True,legal_views=len(legal['views']),gpu=gpu,time=time.time()))


def score_case(manifest):
    import numpy as np
    import joblib
    case=manifest['case']['id'];target=OUT/'top1';dest=target/'fk'/case
    recovery=load('extension_existing_export',HERE/'recover_six_scene_existing_top1.py')
    cov=recovery.export_case(OUT,target,manifest['case'])
    if cov['available_sets']==0:
        save(dest/'unavailable.json',dict(reason='No fully validated four-particle sets'));return
    def worker(key):
        a,h=key
        execute([sys.executable,__file__,'--feature-case',case,'--condition',a,'--hand',h],os.environ.copy(),dest/f'features_{a}_{h}.log')
    with concurrent.futures.ThreadPoolExecutor(4) as pool:list(pool.map(worker,KEYS))
    geom=load('extension_geometry',HERE/'probe_fk_particle_ranking_geometry.py')
    frozen=joblib.load(OUT/'ranker_original_reference.joblib')
    new=joblib.load(OUT/'ranker_unified_frozen.joblib')
    with gzip.open(dest/'source_dataset.json.gz','rt') as f:data=json.load(f)
    rows=sorted(data['rows'],key=lambda r:tuple(r[k] for k in ('view','condition','hand','sample')))
    env={tuple(v[k] for k in ('view','condition','hand','sample','particle')):v['features']
         for a,h in KEYS for v in read(dest/f'closure_features_{a}_{h}.json')['rows']}
    xs=[]
    for row in rows:
        x,names=geom.features({k:v for k,v in row.items() if k not in ('labels','source_file','validation_file')},env)
        assert names==frozen['features'];xs.append(x)
    x=np.asarray(xs);baseline=geom.score(frozen['name'],frozen['model'],x.reshape(-1,x.shape[-1])).reshape(-1,4)
    logit=np.log(np.clip(baseline,1e-6,1-1e-6)/np.clip(1-baseline,1e-6,1-1e-6))
    features=np.concatenate([x,logit[...,None]],axis=2).astype(np.float32)
    assert names+['frozen_external_model_logit']==new['features']
    pred=new['model'].predict_proba(features.reshape(-1,502))[:,1].reshape(-1,4)
    assert np.isfinite(pred).all()
    for label,scores in [('original_reference',baseline),('unified_top1',pred)]:
        result=geom.helper.evaluate(geom.converted(rows),scores)
        result['coverage']={k:v for k,v in cov.items() if k!='source_sha256'}
        result['hit4']=sum(any(t['height'] for t in r['labels']) for r in rows)
        for s,r,score in zip(result['selections'],rows,scores):
            s.update(scores_by_particle=score.tolist(),source_file=r['source_file'],validation_file=r['validation_file'])
        save(dest/f'{label}.json',result)
    for p,h in data['source_sha256'].items():assert sha(p)==h,p


def summary(ref,m,coverage,errors):
    ref.export_comparison(m,coverage,errors)
    snapshot=read(OUT/'comparison_inputs.json');cases=[]
    for c in m['cases']:
        dest=OUT/'top1/fk'/c['id'];p=dest/'unified_top1.json';rank=read(p) if p.exists() else None
        baseline=read(dest/'original_reference.json') if rank else None
        dro=[r for r in snapshot['dro'] if r['case']==c['id']]
        ds={k:sum((r.get(k) or 0) for r in dro) for k in ('planned','generated','retained','executed','height','strict')}
        ds['complete']=bool(dro) and all(r['complete'] for r in dro)
        # Common view/hand includes all four sets in A and B AND complete DRO.
        selections=rank['selections'] if rank else []
        complete_cd={a:{(s['view'],s['hand']) for s in selections if s['condition']==a} for a in ('A','B')}
        common=complete_cd['A']&complete_cd['B']&{(r['view'],r['hand']) for r in dro if r['complete']}
        matched={}
        for a in ('A','B'):
            ss=[s for s in selections if s['condition']==a and (s['view'],s['hand']) in common]
            assert len(ss)==4*len(common)
            matched[a]=dict(sets=len(ss),height=sum(s['height'] for s in ss))
        dd=[r for r in dro if (r['view'],r['hand']) in common]
        matched['dro']={k:sum((r.get(k) or 0) for r in dd) for k in ('generated','executed','height')}
        cases.append(dict(case=c['id'],title=c['title'],top1=rank,original_reference=baseline,dro=ds,common_complete=matched))
    save(OUT/'RESULTS.json',dict(cases=cases,coverage=coverage,errors=errors,ranker_freeze=m['ranker_freeze'],
        complete=all(c['top1'] is not None and not c['top1']['coverage']['missing_batches'] and c['dro']['complete'] for c in cases)))
    rate=lambda s,n:f'{s}/{n} ({100*s/n:.2f}%)' if n else '—'
    lines=['# 新增六场景：冻结 FK-only A/B Top1 与 DRO64','',
        '独立运行目录，统一排序器在新场景生成前已冻结；保留原排序器对照。所有统计仅以最终物体提升≥10cm为主指标。',
        '此文件随阶段更新，缺失生成/验证不当作已执行失败。DRO筛除计入成功/生成分母，不计入成功/执行分母。',
        'A/B各4接触集×4粒子，FK200/ENV0；DRO64，候选预算不同。无视频。原粒子独立CPU PhysX执行后关联Top1选择。','',
        '|场景|A Top1|B Top1|Hit@4合并|原排序Top1合并|DRO成功/执行|DRO成功/生成|DRO待验证|',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for item in cases:
        r=item['top1'];d=item['dro'];ab=[]
        for a in ('A','B'):
            g=[v for k,v in r['groups'].items() if k.startswith(a+'/')] if r else []
            ab.append(rate(sum(v['height'] for v in g),sum(v['sets'] for v in g)))
        h=rate(r['hit4'],r['sets']) if r else '—';orig=item['original_reference'];o=rate(orig['height'],orig['sets']) if orig else '—'
        lines.append(f"|{item['title']}|{ab[0]}|{ab[1]}|{h}|{o}|{rate(d['height'],d['executed'])}|{rate(d['height'],d['generated'])}|{d['retained']-d['executed']}|")
    lines+=['','## A/B/DRO共同完整视角与手型','', '|场景|A Top1|B Top1|DRO成功/执行|DRO成功/生成|','|---|---:|---:|---:|---:|']
    for item in cases:
        c=item['common_complete'];a,b,d=c['A'],c['B'],c['dro']
        lines.append(f"|{item['title']}|{rate(a['height'],a['sets'])}|{rate(b['height'],b['sets'])}|{rate(d['height'],d['executed'])}|{rate(d['height'],d['generated'])}|")
    lines+=['','错误与覆盖率详见RESULTS.json、errors.json；这是学习排序与原生DRO在不等预算下的比较，不是等计算量对比。']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')


def main(preflight_only=False):
    lock=(OUT/'pipeline.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    m=read(OUT/'manifest.json');status('preflight');icd,ref=preflight(m)
    if preflight_only:status('preflight_passed');return
    status('capture_smoke',case=m['cases'][0]['id']);capture(m['cases'][0],0,icd)
    status('capture_all',gpus=[0,1,2,3])
    def capworker(gpu):
        for c in m['cases'][1:][gpu::4]:capture(c,gpu,icd)
    with concurrent.futures.ThreadPoolExecutor(4) as pool:list(pool.map(capworker,range(4)))
    counts={c['id']:read(Path(c['observations'])/'visibility/capture_complete.json')['legal_views'] for c in m['cases']}
    save(OUT/'capture_status.json',dict(stage='complete',legal_views=counts,time=time.time()))
    env=ref.base.base_env();cache=C/'outputs/extension3_shelf_basket_drawer_4x4_20260910_v2/torch_cache'
    env.update(SAM3D_CAMERA_CONVENTION='opencv',TORCH_HOME=str(cache),SAM3D_DINO_REPOSITORY=str(cache/'hub/facebookresearch_dinov2_main'))
    # Use the first case with an eligible camera; no favorable result-based choice.
    nonempty=[c for c in m['cases'] if counts[c['id']]]
    errors=[]
    def error(stage,case,ex):
        errors.append(dict(stage=stage,case=case,error=repr(ex)));save(OUT/'errors.json',errors)
    status('sam3d_device_smoke');gpus=ref.sam3d_smoke(dict(m,cases=nonempty),env) if nonempty else []
    if gpus:
        status('sam3d',gpus=gpus)
        try:execute([m['runtime']['contact_python'],HERE/'sam3d_resident_batch.py','--manifest',OUT/'manifest.json',
                     '--output',OUT/'resident','--gpus',','.join(gpus)],env,OUT/'logs/sam3d.log',C)
        except Exception as ex:error('sam3d','all',ex)
    else:error('sam3d','all',RuntimeError('No successful identical-frame smoke or no legal camera; no mesh fallback'))
    resident=load('extension_resident',HERE/'sam3d_resident_batch.py');os.environ['SAM3D_CAMERA_CONVENTION']='opencv'
    coverage=[];usable=[]
    for c in m['cases']:
        obs=Path(c['observations']);legal=read(obs/'visibility/legal_partial_views.json');good=[];bad=[]
        for v in legal['views']:
            view=Path(v['rgbd_capture_dir']).name
            (good if resident.cache_valid(dict(capture_dir=str(obs/'visibility/rgbd_views'/view),output_dir=str(obs/'sam3d'/view))) else bad).append(v)
        coverage.append(dict(case=c['id'],legal_views=len(legal['views']),reconstructed_views=len(good),failed_reconstruction_views=[Path(v['rgbd_capture_dir']).name for v in bad]))
        if legal['views']:
            execute([m['runtime']['contact_python'],HERE/'materialize_fetchbench_world_to_robot_base.py','--source-root',obs,
                '--output-root',OUT/'dro_inputs'/c['id'],'--task-config',c['task_config'],'--task-index',c['task_index'],'--skip-sam3d'],env,OUT/'logs'/f"materialize_dro_{c['id']}.log",C)
        if not good:continue
        partial=OUT/'finite_observations'/c['id'];(partial/'visibility').mkdir(parents=True,exist_ok=True)
        for link,target in [(partial/'visibility/rgbd_views',obs/'visibility/rgbd_views'),(partial/'sam3d',obs/'sam3d')]:
            if not link.exists():link.symlink_to(target,target_is_directory=True)
            else:assert link.resolve()==target.resolve()
        save(partial/'visibility/legal_partial_views.json',dict(legal,views=good))
        execute([m['runtime']['contact_python'],HERE/'materialize_fetchbench_world_to_robot_base.py','--source-root',partial,
            '--output-root',OUT/'generation'/c['id']/'corrected_inputs','--task-config',c['task_config'],'--task-index',c['task_index']],env,OUT/'logs'/f"materialize_fk_{c['id']}.log",C)
        usable.append(dict(c,observations=str(partial)))
    save(OUT/'coverage.json',dict(rows=coverage));summary(ref,m,coverage,errors)
    for c in usable:
        status('fk_generation_validation',case=c['id'],gpus=[0,1,2,3],generation_workers=8,validation_workers=12)
        manifest=ref.prepare_case(c,m)
        try:ref.run.run_jobs(manifest,[v['view'] for v in manifest['selected_views']],['J0'],'extension6_fkonly')
        except Exception as ex:error('FK',c['id'],ex)
        ref.report(manifest)
        status('fixed_top1_ranking',case=c['id'],workers=4)
        try:score_case(manifest)
        except Exception as ex:error('Top1',c['id'],ex)
        summary(ref,m,coverage,errors)
    for c in m['cases']:
        if not counts[c['id']]:continue
        status('dro_generation_validation',case=c['id'],gpus=[0,1,2,3],generation_workers=8,validation_workers=12)
        runtime=m['runtime'];e=env.copy();e.update(FETCHBENCH_ROOT=str(F),CONTACT_ROOT=str(C),DRO_ROOT=runtime['dro_root'],
            DRO_PYDEPS=runtime['dro_pydeps'],DRO_PYTHON=runtime['dro_python'],FETCHBENCH_PYTHON=str(P/'miniconda3/envs/fetchbench/bin/python'),
            CHECKPOINT=runtime['checkpoint_dro'],INPUT_ROOT=str(OUT/'dro_inputs'/c['id']),RUN_ROOT=str(OUT/'dro'/c['id']),
            TASK_INDEX=str(c['task_index']),SCENE_CONFIG=c['scene'],SCENE_FACTORY=c['scene_factory'],OBJECT_LABEL=c['id'],
            EXPECTED_LEGAL_VIEWS=str(counts[c['id']]),GPU_COUNT='4',GPU_OFFSET='0',WORKERS_PER_GPU='2',VALIDATION_WORKERS='12',
            DRO_CANDIDATES='64',GENERATION_ATTEMPTS='4',TORCH_EXTENSIONS_DIR=str(ref.base.SOURCE/'torch_extensions'))
        try:execute(['bash',HERE/'run_extension3_dro64_worker.sh'],e,OUT/'logs'/f"dro_{c['id']}.log")
        except Exception as ex:error('DRO',c['id'],ex)
        summary(ref,m,coverage,errors)
    summary(ref,m,coverage,errors)
    status('complete' if read(OUT/'RESULTS.json')['complete'] and not errors else 'partial_complete',errors=errors)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--preflight-only',action='store_true')
    parser.add_argument('--feature-case');parser.add_argument('--condition');parser.add_argument('--hand');args=parser.parse_args()
    if args.feature_case:
        recovery=load('extension_feature_worker',HERE/'recover_six_scene_existing_top1.py')
        recovery.feature_worker(OUT/'top1',args.feature_case,args.condition,args.hand)
    else:
        try:main(args.preflight_only)
        except Exception as ex:status('failed',error=repr(ex));raise
