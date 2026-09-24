"""Paired fresh deterministic F0 vs continuous-FK-environment F1 study."""
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
import time

P=Path('/inspire/qb-ilm2/project/zhanghanbo/public/mck')
C=P/'ContactDiffusion';F=P/'FetchBench-CORL2024'
ROOT=C/'outputs/shampoo_repro_and_fk_environment_20260914'
RUN=ROOT/'controlled'
PREVIOUS=C/'outputs/shampoo_drawer_budget_ablation_8views_20260914'
ENV_OLD=C/'outputs/shampoo_drawer_environment_probe_4x4_200_20260914'
HOOK=F/'scripts/shampoo_fk_trace_infer_20260914.py'
STAGE=F/'scripts/shampoo_deterministic_stage.py'
SCRIPT=F/'scripts/run_shampoo_drawer_budget_ablation_20260914.py'
spec=importlib.util.spec_from_file_location('controlled_base',SCRIPT)
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
base.RUN=RUN;base.ARMS={a:(4,4,200) for a in ('F0','F1')}


def prepare():
    previous=base.read(PREVIOUS/'manifest.json')
    for p,h in previous['code_sha256'].items():assert base.sha(p)==h,p
    for paths in previous['inputs'].values():
        for p,h in paths.items():assert base.sha(p)==h,p
    for k,h in previous['checkpoint_sha256'].items():assert base.sha(previous['runtime'][k])==h
    manifest=dict(previous)
    manifest.update(arms={a:[4,4,200] for a in base.ARMS},
        interventions={'F0':'fresh deterministic baseline, fixed historical contacts and original initialization',
                       'F1':'same baseline plus 10*(mean+CVaR) visible-environment energy at every FK update'},
        deterministic=True,contact_replay=str(ENV_OLD/'cases/E1'),
        driver_sha256=base.sha(__file__),hook_sha256=base.sha(HOOK),stage_wrapper_sha256=base.sha(STAGE),
        fk_environment_weight={'F0':0,'F1':10},expected_batches=64,expected_execution_slots=256)
    manifest['config_paths']={}
    for arm in base.ARMS:
        manifest['config_paths'][arm]={}
        for hand in ('barrett','shadowhand'):
            source=Path(previous['config_paths']['C0'][hand]);dest=RUN/'configs'/f'{arm}_{hand}.yaml'
            dest.parent.mkdir(parents=True,exist_ok=True)
            if not dest.exists():shutil.copy2(source,dest)
            assert base.sha(source)==base.sha(dest)
            manifest['config_paths'][arm][hand]=str(dest)
    manifest['config_sha256']={a:{h:base.sha(p) for h,p in ps.items()} for a,ps in manifest['config_paths'].items()}
    shutil.copy2(PREVIOUS/'prepare_ablation.py',RUN/'prepare_ablation.py')
    if (RUN/'manifest.json').exists():assert base.read(RUN/'manifest.json')==manifest
    else:base.save(RUN/'manifest.json',manifest)
    return manifest


def stages(job,gpu,manifest,names,files,commands,env,dest,old,config):
    arm,view,condition,hand=job
    geom=base.OLD/'generation/shampoo_drawer/corrected_inputs/sam3d'/view
    m=base.read(geom/'sam3d_fused_centered.json')
    meta=dict(directory=str(dest),contact_root=str(C),deterministic=True,
        environment_weight=10 if arm=='F1' else 0,repeat_backward=False,
        scene=str(base.OLD/'generation/shampoo_drawer/corrected_inputs/visibility/rgbd_views'/view/'scene_partial_robot_base.npy'),
        object_robot=str(geom/'sam3d_fused_robot_base.npy'),
        robot_from_object=m.get('robot_from_pointcloud_frame',m.get('robot_from_object')))
    base.save(dest/'trace_metadata.json',meta)
    env.update(CONTACTDIFF_INFER_SOURCE=str(HOOK),FK_TRACE_META=str(dest/'trace_metadata.json'),
               CUBLAS_WORKSPACE_CONFIG=':4096:8',PYTHONHASHSEED='0')
    contacts=ENV_OLD/'cases/E1'/view/condition/hand/'raw_object.json'
    commands[0]+=['--source-diffusion-candidates',contacts,'--replay-source-diffusion-rng']
    commands[2]=[commands[2][0],STAGE,*commands[2][1:]]
    timings={}
    for name,output,command in zip(names,files,commands):
        start=time.monotonic()
        if not output.exists():base.execute(command,env,dest/(name+'.log'))
        data=base.read(output)
        assert data['config_sha256']==base.sha(config) and len(data['records'])==4
        assert all(len(r['fk']['candidates'])==(1 if name=='selected_w10' else 4) for r in data['records'])
        if name=='raw_object':
            for x,y in zip(base.read(contacts)['records'],data['records']):
                assert x['target_contacts_sha256']==y['target_contacts_sha256']
                assert x['sample_seed']==y['sample_seed']
        timings[name]=time.monotonic()-start
    base.save(dest/'generation_timing.json',dict(gpu=gpu,seconds=timings))
    return files[-1]


source=inspect.getsource(base.generation)
source=source[:source.index('    timings = {}')]+'    return _stages(job,gpu,manifest,stages,files,commands,env,dest,old,config)\n'
scope=dict(base.__dict__);scope['_stages']=stages
exec(compile(source,'<frozen-generation-with-continuous-env>','exec'),scope)
generate=scope['generation']


def summarize():
    rows=[base.read(p) for p in (RUN/'cases').glob('*/*/*/*/validation_result.json')]
    groups={}
    for row in rows:
        key=(row['arm'],row['condition'],row['hand'])
        g=groups.setdefault(key,dict(arm=key[0],condition=key[1],hand=key[2],batches=0,executed=0,height=0,strict=0,invalid=0,top1=0,hit4=0))
        g['batches']+=1
        for k in ('executed','height','strict','invalid'):g[k]+=row[k]
        g['top1']+=row['top1_height'];g['hit4']+=int(row['hit4'])
    base.save(RUN/'results.json',dict(complete=len(rows)==64,finished_batches=len(rows),expected_batches=64,groups=list(groups.values()),rows=rows))
    lines=['# FK全程环境约束：可复现配对对照','',f'完成 {len(rows)}/64 批；固定4×4、FK200+ENV200；CPU PhysX；最终提升≥10cm。','',
        '|组|输入|手|完成|成功/执行|严格成功|Top1|Top4命中|NaN|','|---|---|---|---:|---:|---:|---:|---:|---:|']
    for key,g in sorted(groups.items()):lines.append(f"|{g['arm']}|{g['condition']}|{g['hand']}|{g['batches']}/8|{g['height']}/{g['executed']}|{g['strict']}|{g['top1']}|{g['hit4']}|{g['invalid']}|")
    lines+=['','F0新生成确定性基线；F1仅在每个FK更新中加入环境能量（权重10）。',
        '两组同源接触点回放、原初始化、相同物体与环境点云；FK学习率0.0075、ENV学习率0.003。',
        '不加入平移初始化、接触筛选、掌距门限或新闭合策略；ENV200及原选择、CPU验证保持不变。']
    (RUN/'RESULTS.md').write_text('\n'.join(lines)+'\n')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--prepare-only',action='store_true');parser.add_argument('--smoke-only',action='store_true')
    args=parser.parse_args();RUN.mkdir(parents=True,exist_ok=True)
    lock=(RUN/'launcher.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert (ROOT/'repro_gate.json').exists() and base.read(ROOT/'repro_gate.json')['passed'], 'Repeatability diagnosis must pass first'
    manifest=prepare()
    if args.prepare_only:print('Prepared',RUN);return
    base.save(RUN/'status.json',dict(stage='smoke',pid=os.getpid(),started=time.time()))
    slots=queue.Queue()
    for _ in range(2):
        for gpu in range(4):slots.put(gpu)
    def work(job):
        gpu=slots.get();start=time.monotonic()
        try:
            if not (base.folder(job)/'generation_done.json').exists():
                generate(job,gpu,manifest)
                base.save(base.folder(job)/'generation_done.json',dict(complete=True,gpu=gpu,elapsed_seconds=time.monotonic()-start))
            return job
        finally:slots.put(gpu)
    smoke=[(a,'r1.5_el+30_az+0','A',h) for a in ('F0','F1') for h in ('barrett','shadowhand')]
    with futures.ThreadPoolExecutor(4) as pool:
        for job in pool.map(work,smoke):base.validate(job,manifest)
    summarize()
    if args.smoke_only:return
    base.save(RUN/'status.json',dict(stage='generation_and_validation',pid=os.getpid(),gpus=4,generation_workers=8,validation_workers=8))
    jobs=[(arm,view,condition,hand) for view in base.VIEWS for condition in ('A','B') for hand in ('barrett','shadowhand') for arm in ('F0','F1')]
    errors=[]
    with futures.ThreadPoolExecutor(8) as vp,futures.ThreadPoolExecutor(8) as gp:
        gjobs={gp.submit(work,j):j for j in jobs};vjobs={}
        for future in futures.as_completed(gjobs):
            job=gjobs[future]
            try:vjobs[vp.submit(base.validate,future.result(),manifest)]=job
            except Exception as error:
                item=dict(job=job,stage='generation',error=repr(error));errors.append(item);base.save(base.folder(job)/'error.json',item)
            summarize()
        for future in futures.as_completed(vjobs):
            try:future.result()
            except Exception as error:
                job=vjobs[future];item=dict(job=job,stage='validation',error=repr(error));errors.append(item);base.save(base.folder(job)/'error.json',item)
            summarize()
    paired=[]
    for view in base.VIEWS:
        for condition in ('A','B'):
            for hand in ('barrett','shadowhand'):
                paths=[base.folder((a,view,condition,hand))/'raw_object.json' for a in ('F0','F1')]
                if all(p.exists() for p in paths):
                    aa,bb=[base.read(p)['records'] for p in paths]
                    for a,b in zip(aa,bb):
                        same=a['fk']['initialization_state_sha256']==b['fk']['initialization_state_sha256'] and a['target_contacts_sha256']==b['target_contacts_sha256']
                        paired.append(dict(view=view,condition=condition,hand=hand,sample=a['sample_index'],same=same))
                        if not same:errors.append(dict(stage='pairing',view=view,condition=condition,hand=hand,sample=a['sample_index']))
    base.save(RUN/'pairing.json',paired)
    base.save(RUN/'status.json',dict(stage='failed' if errors else 'complete',errors=errors,finished=time.time()))
    summarize()


if __name__=='__main__':
    try:main()
    except Exception as error:
        base.save(RUN/'startup_error.json',dict(error=repr(error),time=time.time()));raise
