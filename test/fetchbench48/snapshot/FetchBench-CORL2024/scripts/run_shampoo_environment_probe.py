"""E0--E3 isolated environmental contact/initialization study, fixed 4x4/200+200."""
import argparse
import concurrent.futures as futures
import fcntl
import importlib.util
import inspect
import json
import os
from pathlib import Path
import queue
import shutil
import subprocess
import time


P = Path('/inspire/qb-ilm2/project/zhanghanbo/public/mck')
C = P/'ContactDiffusion'
F = P/'FetchBench-CORL2024'
PREVIOUS = C/'outputs/shampoo_drawer_budget_ablation_8views_20260914'
RUN = C/'outputs/shampoo_drawer_environment_probe_4x4_200_20260914'
SCRIPT = F/'scripts/run_shampoo_drawer_budget_ablation_20260914.py'
HOOK = F/'scripts/shampoo_environment_probe_infer_20260914.py'
spec = importlib.util.spec_from_file_location('frozen_budget_driver', SCRIPT)
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
base.RUN = RUN
base.ARMS = {a:(4,4,200) for a in ('E0','E1','E2','E3')}


def prepare():
    previous = base.read(PREVIOUS/'manifest.json')
    assert base.sha(SCRIPT) == previous['driver_sha256']
    for path,digest in previous['code_sha256'].items():
        assert base.sha(path)==digest, f'Historical runtime changed: {path}'
    for view,paths in previous['inputs'].items():
        for path,digest in paths.items(): assert base.sha(path)==digest, path
    for key,digest in previous['checkpoint_sha256'].items(): assert base.sha(previous['runtime'][key])==digest
    manifest = dict(previous)
    manifest.update(arms={a:[4,4,200] for a in base.ARMS}, source_run=str(PREVIOUS),
        driver_sha256=base.sha(__file__), hook_sha256=base.sha(HOOK),
        interventions=dict(E0='unchanged baseline',E1='5mm visible environment contact rejection; max32 draws for4 sets',
            E2='4-particle fixed rotation/joints, minimum feasible translation <=4cm; no extra optimizer steps',E3='E1+E2'),
        execution='CPU PhysX; original closure and all physics parameters; final height>=10cm',
        uncertainty='Observed geometry only; low seed collision not guaranteed; missing sets count in planned denominator')
    manifest['config_paths'] = {}
    for arm in base.ARMS:
        manifest['config_paths'][arm] = {}
        for hand in ('barrett','shadowhand'):
            src = Path(previous['config_paths']['C0'][hand])
            dst = RUN/'configs'/f'{arm}_{hand}.yaml'
            dst.parent.mkdir(parents=True, exist_ok=True)
            if not dst.exists(): shutil.copy2(src,dst)
            assert base.sha(dst)==base.sha(src)
            manifest['config_paths'][arm][hand]=str(dst)
    manifest['config_sha256']={a:{h:base.sha(p) for h,p in paths.items()} for a,paths in manifest['config_paths'].items()}
    shutil.copy2(PREVIOUS/'prepare_ablation.py', RUN/'prepare_ablation.py')
    if (RUN/'manifest.json').exists(): assert base.read(RUN/'manifest.json')==manifest
    else: base.save(RUN/'manifest.json', manifest)
    return manifest


def run_stages(job,gpu,manifest,stages,files,commands,env,dest,old,config):
    arm,view,condition,hand=job
    metadata=base.read(base.OLD/'generation/shampoo_drawer/corrected_inputs/sam3d'/view/'sam3d_fused_centered.json')
    geom=base.OLD/'generation/shampoo_drawer/corrected_inputs/sam3d'/view
    intervention=dict(arm=arm,contact_root=str(C),
        robot_from_object=metadata.get('robot_from_pointcloud_frame',metadata.get('robot_from_object')),
        scene=str(base.OLD/'generation/shampoo_drawer/corrected_inputs/visibility/rgbd_views'/view/'scene_partial_robot_base.npy'),
        object_robot=str(geom/'sam3d_fused_robot_base.npy'), audit=str(dest/'intervention_audit.json'),
        output=str(files[0]), baseline_raw=str(old/'raw_object'/f'shampoo_drawer_{hand}.json'))
    # Path construction kept explicit to avoid confusing frame-labelled assets.
    intervention['baseline_raw']=str(old/'raw_object'/f'shampoo_drawer_{hand}.json')
    base.save(dest/'intervention.json',intervention)
    env.update(CONTACTDIFF_INFER_SOURCE=str(HOOK),ENV_PROBE_META=str(dest/'intervention.json'))
    timings={}
    for stage,output,command in zip(stages,files,commands):
        start=time.monotonic()
        if not output.exists(): base.execute(command,env,dest/(stage+'.log'))
        data=base.read(output)
        assert data['config_sha256']==base.sha(config)
        assert len(data['records'])<=4
        assert all(len(r['fk']['candidates'])==(1 if stage=='selected_w10' else 4) for r in data['records'])
        timings[stage]=time.monotonic()-start
        if stage=='raw_object':
            if arm=='E2':
                original=base.read(old/'raw_object'/f'shampoo_drawer_{hand}.json')
                for before,after in zip(original['records'],data['records']):
                    assert before['sample_seed']==after['sample_seed']
                    assert before['source_diffusion_contacts_sha256']==after['source_diffusion_contacts_sha256']
                    assert before['target_contacts_sha256']==after['target_contacts_sha256']
            if not data['records']:
                for remaining in files[1:]: base.save(remaining,data)
                break
    base.save(dest/'generation_timing.json',dict(gpu=gpu,seconds=timings))
    return files[-1]


# Reuse the exact four historical stage commands; change only the intervention
# hook, arm naming, and the post-generation audit for replacement contact sets.
source=inspect.getsource(base.generation)
source=source.replace("if arm == 'C0':", "if arm == 'E0':")
start=source.index('    timings = {}')
source=source[:start]+'    return run_stages(job,gpu,manifest,stages,files,commands,env,dest,old,config)\n'
namespace=dict(base.__dict__)
namespace['run_stages']=run_stages
exec(compile(source,'<frozen-stage-commands-with-environment-hook>','exec'),namespace)
generate=namespace['generation']


def validate(job,manifest):
    result=base.validate(job,manifest)
    result['missing_contact_sets']=4-result['planned']
    result['planned_slots']=4
    base.save(base.folder(job)/'validation_result.json',result)
    return result


def summarize():
    rows=[base.read(p) for p in (RUN/'cases').glob('*/*/*/*/validation_result.json')]
    groups={}
    for r in rows:
        key=(r['arm'],r['condition'],r['hand'])
        g=groups.setdefault(key,dict(arm=key[0],condition=key[1],hand=key[2],batches=0,planned=0,executed=0,height=0,strict=0,invalid=0,missing=0,top1=0,hit4=0))
        g['batches']+=1;g['planned']+=4
        for k in ('executed','height','strict','invalid'):g[k]+=r[k]
        g['missing']+=r.get('missing_contact_sets',0)
        g['top1']+=r['top1_height'];g['hit4']+=int(r['hit4'])
    base.save(RUN/'results.json',dict(complete=len(rows)==128,finished_batches=len(rows),expected_batches=128,groups=list(groups.values()),rows=rows))
    lines=['# 抽屉 Shampoo：环境处理消融','',f'完成 {len(rows)}/128 批。4×4，FK200+ENV200，最终提升≥10cm；CPU PhysX。','',
        '|组|输入|手|完成视角|成功/实际执行|成功/计划|Top1成功|Top4命中视角|缺失接触集|NaN|',
        '|---|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for key,g in sorted(groups.items()):
        lines.append(f"|{g['arm']}|{g['condition']}|{g['hand']}|{g['batches']}/8|{g['height']}/{g['executed']}|{g['height']}/{g['planned']}|{g['top1']}/{g['batches']}|{g['hit4']}/{g['batches']}|{g['missing']}|{g['invalid']}|")
    lines+=['','E0原基线；E1接触集环境筛选补采样；E2低碰撞初始化；E3两者结合。',
        'E1/E3最多32次diffusion尝试补齐4组；不安全接触集不进入FK，不用危险候选补齐。',
        'E2/E3仅对原4粒子作最多4cm平移候选搜索，保留原旋转/关节，未增加FK优化步数；不能保证每个粒子无碰撞。',
        '环境仅来自当前视角partial scene；物体辅助几何仍为SAM3D，不是完全partial-only算法。',
        'A/B分别沿用原权重，属于条件内部环境处理消融，不是同权重输入消融。']
    (RUN/'RESULTS.md').write_text('\n'.join(lines)+'\n')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--prepare-only',action='store_true')
    parser.add_argument('--smoke-only',action='store_true')
    args=parser.parse_args()
    RUN.mkdir(parents=True,exist_ok=True)
    lock=(RUN/'launcher.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    manifest=prepare()
    if args.prepare_only: print('Prepared',RUN,flush=True);return
    devices=subprocess.check_output(['nvidia-smi','--query-gpu=index','--format=csv,noheader'],text=True).split()
    assert set(devices)=={'0','1','2','3'}
    base.save(RUN/'status.json',dict(stage='smoke',pid=os.getpid(),started=time.time()))
    # One E3 batch per hand exercises both new paths and downstream CPU physics.
    slots=queue.Queue()
    for _ in range(3):
        for gpu in range(4):slots.put(gpu)
    def work(job):
        gpu=slots.get();start=time.monotonic()
        try:
            if not (base.folder(job)/'generation_done.json').exists():
                generate(job,gpu,manifest)
                base.save(base.folder(job)/'generation_done.json',dict(complete=True,gpu=gpu,elapsed_seconds=time.monotonic()-start))
            return job
        finally:slots.put(gpu)
    smoke=[('E3',base.VIEWS[0],'A',hand) for hand in ('barrett','shadowhand')]
    with futures.ThreadPoolExecutor(2) as pool:
        for job in pool.map(work,smoke): validate(job,manifest)
    summarize()
    if args.smoke_only: print('Both-hand E3 smoke passed',flush=True);return
    base.save(RUN/'status.json',dict(stage='generation_and_cpu_validation',pid=os.getpid(),gpus=4,generation_workers=12,validation_workers=8))
    # Baseline remains untouched; repeat its CPU verification in this run.
    jobs=[(arm,v,c,h) for arm in ('E0','E1','E2','E3') for v in base.VIEWS for c in ('A','B') for h in ('barrett','shadowhand')]
    errors=[]
    with futures.ThreadPoolExecutor(8) as vp, futures.ThreadPoolExecutor(12) as gp:
        gjobs={gp.submit(work,j):j for j in jobs};vjobs={}
        for future in futures.as_completed(gjobs):
            job=gjobs[future]
            try:vjobs[vp.submit(validate,future.result(),manifest)]=job
            except Exception as error:
                errors.append(dict(job=job,stage='generation',error=repr(error)));base.save(base.folder(job)/'error.json',errors[-1])
            summarize()
        for future in futures.as_completed(vjobs):
            try:future.result()
            except Exception as error:
                job=vjobs[future];errors.append(dict(job=job,stage='validation',error=repr(error)));base.save(base.folder(job)/'error.json',errors[-1])
            summarize()
    # E1/E3 must use identical accepted contacts before interpreting init effect.
    for view in base.VIEWS:
        for condition in ('A','B'):
            for hand in ('barrett','shadowhand'):
                paths=[base.folder((a,view,condition,hand))/'raw_object.json' for a in ('E1','E3')]
                if all(p.exists() for p in paths):
                    hashes=[[r['target_contacts_sha256'] for r in base.read(p)['records']] for p in paths]
                    if hashes[0]!=hashes[1]:errors.append(dict(stage='pairing',view=view,condition=condition,hand=hand))
    base.save(RUN/'status.json',dict(stage='failed' if errors else 'complete',errors=errors,finished=time.time()))
    summarize()
    if errors:raise SystemExit(1)


if __name__=='__main__':
    try: main()
    except Exception as error:
        base.save(RUN/'startup_error.json',dict(error=repr(error),time=time.time()))
        raise
