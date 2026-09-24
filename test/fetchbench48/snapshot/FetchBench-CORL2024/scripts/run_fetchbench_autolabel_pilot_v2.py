"""Isolated four-task automatic-label pilot; immutable v1 and historical outputs."""
import argparse
import concurrent.futures as cf
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import queue
import sys
import time
import xml.etree.ElementTree as ET

os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',SAM3D_CAMERA_CONVENTION='opencv',SAM3D_REUSE_EXISTING_NORMALIZATION='1')
HERE=Path(__file__).resolve().parent
F=HERE.parent;P=F.parent;C=P/'ContactDiffusion'
OUT=C/'outputs/fetchbench_v2_autolabel_pilot4_ab_dro64_20260920'
PRIOR=C/'outputs/fk_top1_extension6_round3_ab_dro64_20260916'
sys.path.insert(0,str(OUT/'ranker_pydeps'))
from run_fk_top1_extension6_20260915 import load,read,save,sha
from fetchbench_confirmatory_rules_v1 import select_views

def hand_audit():
    """Compare all URDF nodes semantically; resolve mesh references to content hashes."""
    rows=[]
    def signature(path):
        root=ET.parse(path).getroot();meshes={}
        for e in root.iter():
            if e.tag=='mesh':
                file=Path(e.get('filename'));file=file if file.is_absolute() else path.parent/file
                assert file.exists(),file
                digest=sha(file);meshes[str(file)]=digest;e.set('filename',digest)
        def canonical(e):return (e.tag,sorted(e.attrib.items()),[canonical(c) for c in e])
        nodes={(e.tag,e.get('name')):canonical(e) for e in root if e.tag in ('link','joint')}
        return nodes,meshes
    for hand in ('barrett','shadowhand'):
        a=F/f'InfiniGym/assets/contactdiff_hands/{hand}/contactdiff_v4_{hand}_physics.urdf'
        b=F/f'InfiniGym/assets/urdf/dexterous/dro_{hand}_physics.urdf'
        sa,ma=signature(a);sb,mb=signature(b)
        keys=set(sa)|set(sb);different=[list(k) for k in sorted(keys) if sa.get(k)!=sb.get(k)]
        # Dedicated sub-signatures distinguish physical-model changes from display materials.
        def physical(path):
            root=ET.parse(path).getroot()
            for link in root.findall('link'):
                for visual in link.findall('visual'):link.remove(visual)
            for mesh in root.findall('.//mesh'):
                f=Path(mesh.get('filename'));f=f if f.is_absolute() else path.parent/f
                mesh.set('filename',sha(f))
            def canon(e):return (e.tag,sorted(e.attrib.items()),[canon(c) for c in e])
            return {(e.tag,e.get('name')):canon(e) for e in root if e.tag in ('link','joint')}
        pa,pb=physical(a),physical(b)
        pdiff=[list(k) for k in sorted(set(pa)|set(pb)) if pa.get(k)!=pb.get(k)]
        def core(path):
            root=ET.parse(path).getroot()
            for link in root.findall('link'):
                for visual in link.findall('visual'):link.remove(visual)
                for collision in link.findall('collision'):
                    for coeff in collision.findall('contact_coefficients'):collision.remove(coeff)
            for joint in list(root.findall('joint')):
                if joint.get('name','').startswith('extra_tip_'):
                    assert joint.get('type')=='fixed'
                    name=joint.find('child').get('link');leaf=root.find(f"./link[@name='{name}']")
                    assert leaf is not None and not list(leaf),'Auxiliary link must be empty'
                    assert not any(j.find('parent').get('link')==name for j in root.findall('joint'))
                    root.remove(joint);root.remove(leaf)
            for mesh in root.findall('.//mesh'):
                f=Path(mesh.get('filename'));f=f if f.is_absolute() else path.parent/f;mesh.set('filename',sha(f))
            def canon(e):return (e.tag,sorted(e.attrib.items()),[canon(c) for c in e])
            return {(e.tag,e.get('name')):canon(e) for e in root if e.tag in ('link','joint')}
        core_equal=core(a)==core(b)
        rows.append(dict(hand=hand,cd_urdf=str(a),dro_urdf=str(b),cd_sha256=sha(a),dro_sha256=sha(b),
                         all_node_differences=different,physical_node_differences=pdiff,
                         physical_urdf_equivalent=not pdiff,core_geometry_inertia_joints_equivalent=core_equal,
                         inherited_difference_disclosure='Barrett: contact_coefficients tags and empty fixed auxiliary fingertip nodes; not changed. Runtime overwrites friction; auxiliary-body dynamics not proven equivalent.' if pdiff else None,
                         cd_meshes=ma,dro_meshes=mb))
    save(OUT/'HAND_MODEL_AUDIT.json',dict(rows=rows,scope='URDF collision/inertia/kinematics; runtime controllers inherited, not newly tuned'))
    if any(not r['core_geometry_inertia_joints_equivalent'] for r in rows):
        raise RuntimeError('Core physical URDF mismatch beyond disclosed baseline auxiliary nodes: stop')

def runtime_lock(m):
    hashes={str(Path(__file__)):sha(__file__)}
    for key,value in m['runtime'].items():
        path=Path(value)
        if path.is_file() and not key.endswith('python'):hashes[value]=sha(path)
    for p in sorted(HERE.glob('*.py')):hashes[str(p)]=sha(p)
    for c in m['cases']:
        path=Path(c['task_config']);hashes[str(path)]=sha(path);hashes[str(path.parent/'asset_config.json')]=sha(path.parent/'asset_config.json')
    result=dict(hashes=hashes,created=time.time(),host=os.uname().nodename,python=sys.version,hand_audit_sha256=sha(OUT/'HAND_MODEL_AUDIT.json'))
    path=OUT/'RUNTIME_LOCK.json'
    if path.exists():assert read(path)['hashes']==hashes,'Runtime changed since preflight'
    else:save(path,result)

def locked_views(m,rec):
    active=copy.deepcopy(m);selections=[]
    for c in active['cases']:
        raw=Path(c['observations']);payload=read(raw/'visibility/legal_partial_views.json')
        rows=[dict(v,view_id=Path(v['rgbd_capture_dir']).name,original_legal_index=i) for i,v in enumerate(payload['views'])]
        chosen=select_views(c['task_id'],rows);ids={v['view_id'] for v in chosen['views']}
        selected=[v for v in rows if v['view_id'] in ids]
        selected_root=OUT/'selected_observations'/c['id']
        (raw/'sam3d').mkdir(exist_ok=True)
        rec.link(selected_root/'visibility/rgbd_views',raw/'visibility/rgbd_views')
        # Capture reads/writes remain at raw; only selected observations reach inference.
        rec.link(selected_root/'sam3d',raw/'sam3d')
        result=dict(payload,views=selected,selection=chosen,source_legal_sha256=sha(raw/'visibility/legal_partial_views.json'))
        target=selected_root/'visibility/legal_partial_views.json'
        if target.exists():assert read(target)==result
        else:save(target,result)
        c['observations']=str(selected_root)
        selections.append(dict(case=c['id'],task_id=c['task_id'],selection=chosen,raw_legal_count=len(rows),selection_sha256=sha(target)))
    path=OUT/'VIEW_SELECTION_LOCK.json'
    if path.exists():assert read(path)['cases']==selections
    else:save(path,dict(cases=selections,locked_before_sam3d=True,seed=2026092002))
    save(OUT/'active_manifest.json',active)
    return active

def pilot_report(m,errors):
    import math
    snapshot=read(OUT/'comparison_inputs.json');rows=[]
    for c in m['cases']:
        selected=read(Path(c['observations'])/'visibility/legal_partial_views.json')['views'];views=len(selected)
        rankpath=OUT/'top1/fk'/c['id']/'unified_top1.json'
        selections=read(rankpath)['selections'] if rankpath.exists() else []
        ds=[r for r in snapshot['dro'] if r['case']==c['id']]
        for method in ('A','B','DRO'):
            for hand in ('barrett','shadowhand'):
                if method=='DRO':
                    rr=[r for r in ds if r['hand']==hand];success=sum(r['height'] for r in rr);executed=sum(r['executed'] for r in rr);planned=64*max(1,views)
                else:
                    rr=[r for r in selections if r['condition']==method and r['hand']==hand];success=sum(bool(r['height']) for r in rr);executed=len(rr);planned=4*max(1,views)
                rows.append(dict(case=c['id'],geometry=c['preliminary_geometry'],environment=c['environment'],method=method,hand=hand,
                                 views=views,success=success,planned=planned,executed=executed,sr10_plan=success/planned,sr10_exec=success/executed if executed else None))
    rates={a:sum(r['sr10_plan'] for r in rows if r['method']==a)/8 for a in ('A','B','DRO')}
    finalized=read(OUT/'RESULTS.json')['complete'] and not errors
    save(OUT/'PILOT_STATISTICS.json',dict(rows=rows,task_macro_plan=rates,errors=errors,finalized_no_pending_errors=finalized,classification_review_pending=True,confirmatory=False))
    lines=['# v2 首批自动初标探索实验','','分类待用户复核。4任务×最多8视角，不做确认性显著性检验；四几何与四环境的效应不能独立分离。',
           'A/B各4×4、FK200/ENV0、DRO64；CPU PhysX；唯一成功标签为最终抬升≥10cm。','',
           f'记录到 {len(errors)} 项阶段错误；有错误时以下仅为暂定统计，不能把基础设施待补全当算法失败。','',
           '|任务|方法|手|成功/计划|成功/实际执行|','|---|---|---|---:|---:|']
    for r in rows:lines.append(f"|{r['case']}|{r['method']}|{r['hand']}|{r['success']}/{r['planned']}|{r['success']}/{r['executed']}|")
    lines+=['','任务与两手等权的计划口径（有待补全时暂定）：'+', '.join(f'{a}={v:.2%}' for a,v in rates.items()),
            '既有REPORT.md是兼容旧脚本导出的字段；本试验范围、计划分母和分类状态以本文件及protocol_v2.json为准。']
    (OUT/'PILOT_REPORT.md').write_text('\n'.join(lines)+'\n')

def main(preflight_only=False):
    lock=(OUT/'pipeline.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    deps=OUT/'ranker_pydeps'
    if not deps.exists():deps.symlink_to(PRIOR/'ranker_pydeps',target_is_directory=True)
    old=load('v2_round3',HERE/'run_fk_top1_round3_20260916.py');old.OUT=OUT
    driver,rec=old.modules();driver.__file__=str(Path(__file__).resolve())
    m=read(OUT/'manifest.json');driver.status('preflight');icd,ref=driver.preflight(m)
    hand_audit();runtime_lock(m)
    if preflight_only:driver.status('preflight_passed');return
    driver.status('capture_all',gpus=[0,1,2,3],tasks=len(m['cases']))
    def capture_worker(gpu):
        for case in m['cases'][gpu::4]:driver.capture(case,gpu,icd)
    with cf.ThreadPoolExecutor(4) as pool:list(pool.map(capture_worker,range(4)))
    m=locked_views(m,rec)
    env=ref.base.base_env();env['LD_LIBRARY_PATH']+=':/usr/local/cuda-12.8/lib64';env['SAM3D_REUSE_EXISTING_NORMALIZATION']='1'
    jobs=[]
    for c in m['cases']:
        obs=Path(c['observations'])
        for v in read(obs/'visibility/legal_partial_views.json')['views']:
            name=Path(v['rgbd_capture_dir']).name;jobs.append(dict(id=c['id']+'__'+name,capture_dir=str(obs/'visibility/rgbd_views'/name),output_dir=str(obs/'sam3d'/name)))
    driver.status('sam3d',gpus=[0,1,2,3],selected_views=len(jobs))
    with cf.ThreadPoolExecutor(4) as pool:
        futures=[pool.submit(rec.attempt,'sam3d',gpu,rec.sam_worker,gpu,jobs[gpu::4],m,env) for gpu in range(4)]
        for f in futures:f.result()
    manifests,coverage=old.prepare(m,ref,rec,env)
    # Parent prepare_case obtains seeds from OUT/observations, i.e. the FULL original legal list.
    for manifest in manifests:
        selected=read(OUT/'selected_observations'/manifest['case']['id']/'visibility/legal_partial_views.json')['views']
        indices={v['view_id']:v['original_legal_index'] for v in selected}
        assert all(v['original_legal_index']==indices[v['view']] for v in manifest['selected_views'])
    slots=queue.Queue()
    for _ in range(2):
        for gpu in range(4):slots.put(gpu)
    with cf.ThreadPoolExecutor(12) as vp:
        for manifest in manifests:
            driver.status('fk_generation_validation',case=manifest['case']['id'],generation_workers=8,validation_workers=12,gpus=[0,1,2,3])
            rec.fk_case(manifest,ref,slots,vp,workers=8)
    # Publish completed A/B before DRO; no change to candidate or execution budgets.
    for manifest in manifests:
        driver.status('ranking',case=manifest['case']['id']);rec.attempt('ranking',manifest['case']['id'],driver.score_case,manifest)
    errors=[r for r in rec.EVENTS if r.get('ok') is False]
    driver.summary(ref,m,coverage,errors);pilot_report(m,errors)
    drojobs=[]
    for c in m['cases']:
        for v in read(Path(c['observations'])/'visibility/legal_partial_views.json')['views']:
            name=Path(v['rgbd_capture_dir']).name
            for hand in ('barrett','shadowhand'):
                dest=OUT/'dro'/c['id']/'views'/name/'simulation'/c['scene_factory']/('task_%03d'%c['task_index'])/hand
                drojobs.append(dict(case=c,view=name,hand=hand,artifact=str(dest)))
    driver.status('dro_generation_validation',generation_workers=8,validation_workers=12,gpus=[0,1,2,3],batches=len(drojobs))
    def gen(job):
        gpu=slots.get()
        try:return rec.dro_generate(job,gpu,m,env)
        finally:slots.put(gpu)
    with cf.ThreadPoolExecutor(8) as gp,cf.ThreadPoolExecutor(12) as vp:
        pending={gp.submit(gen,j):j for j in drojobs};validation=[]
        for future in cf.as_completed(pending):
            j=pending[future]
            try:
                future.result();rec.event('dro_generate',case=j['case']['id'],view=j['view'],hand=j['hand'],ok=True)
                validation.append(vp.submit(rec.attempt,'dro_validate',dict(case=j['case']['id'],view=j['view'],hand=j['hand']),rec.dro_valid,j,m,env))
            except Exception as ex:rec.event('dro_generate',case=j['case']['id'],view=j['view'],hand=j['hand'],ok=False,error=repr(ex))
        for future in validation:future.result()
    errors=[r for r in rec.EVENTS if r.get('ok') is False]
    driver.summary(ref,m,coverage,errors);pilot_report(m,errors)
    driver.status('finished',complete=read(OUT/'RESULTS.json')['complete'],errors=len(errors),exploratory=True,user_review_pending=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--preflight-only',action='store_true');p.add_argument('--feature-case');p.add_argument('--condition');p.add_argument('--hand');args=p.parse_args()
    if args.feature_case:
        worker=load('v2_features',HERE/'recover_six_scene_existing_top1.py');worker.feature_worker(OUT/'top1',args.feature_case,args.condition,args.hand)
    else:
        try:main(args.preflight_only)
        except Exception as ex:save(OUT/'pipeline_status.json',dict(stage='failed',error=repr(ex),time=time.time()));raise
