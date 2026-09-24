"""Same-particle ENV0 vs ENV200 CPU PhysX experiment; no new grasp optimization."""
import argparse
import concurrent.futures as futures
from copy import deepcopy
import fcntl
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import time

import numpy as np

P=Path('/inspire/qb-ilm2/project/zhanghanbo/public/mck');C=P/'ContactDiffusion';F=P/'FetchBench-CORL2024'
SOURCE=C/'outputs/shampoo_drawer_e3_all48views_s4p4_fk200_env200_20260914'
RUN=C/'outputs/shampoo_e3_all48_particle_paired_env0_env200_20260914'
BASE=F/'scripts/run_shampoo_drawer_budget_ablation_20260914.py'
VALIDATOR=F/'scripts/validate_shampoo_particle_order.py'
spec=importlib.util.spec_from_file_location('paired_frozen_helpers',BASE)
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
WORKERS=12


def folder(job):
    arm,view,condition,hand=job
    return RUN/'cases'/arm/view/condition/hand


def prepare():
    previous=base.read(SOURCE/'manifest.json')
    assert base.read(SOURCE/'status.json')['stage']=='complete'
    for p,h in previous['code_sha256'].items():assert base.sha(p)==h,p
    for files in previous['inputs'].values():
        for p,h in files.items():assert base.sha(p)==h,p
    configs={a:{} for a in ('ENV0','ENV200')}
    for a in configs:
        for hand,p in previous['config_paths']['E3'].items():
            dst=RUN/'configs'/f'{a}_{hand}.yaml';dst.parent.mkdir(parents=True,exist_ok=True)
            if not dst.exists():shutil.copy2(p,dst)
            assert base.sha(p)==base.sha(dst)
            configs[a][hand]=str(dst)
    hashes={};pairing=[]
    for v in previous['selected_views']:
        view=v['view']
        for condition in ('A','B'):
            for hand in ('barrett','shadowhand'):
                src=SOURCE/'cases/E3'/view/condition/hand
                paths=[src/(s+'.json') for s in ('candidates_robot','refined_w10')]
                aa,bb=[base.read(p) for p in paths]
                hashes.update({str(p):base.sha(p) for p in paths})
                assert aa['particles']==bb['particles']==4
                assert len(aa['records'])==len(bb['records'])==4
                ar={r['sample_index']:r for r in aa['records']};br={r['sample_index']:r for r in bb['records']}
                assert set(ar)==set(br)=={0,1,2,3}
                for index in sorted(ar):
                    x,y=ar[index],br[index]
                    assert x['target_contacts_sha256']==y['target_contacts_sha256']
                    assert x['sample_seed']==y['sample_seed']
                    assert x['fk']['joint_names']==y['fk']['joint_names']
                    ac={z['particle']:z for z in x['fk']['candidates']};bc={z['particle']:z for z in y['fk']['candidates']}
                    assert set(ac)==set(bc)=={0,1,2,3}
                    for particle in sorted(ac):
                        for z in (ac[particle],bc[particle]):
                            assert np.asarray(z['root_pose']).shape==(4,4)
                            assert np.isfinite(z['root_pose']).all() and np.isfinite(z['joint_positions']).all()
                        pairing.append(dict(view=view,condition=condition,hand=hand,sample=index,particle=particle))
                for arm,payload,original_path in zip(('ENV0','ENV200'),(aa,bb),paths):
                    dest=folder((arm,view,condition,hand));dest.mkdir(parents=True,exist_ok=True)
                    data=deepcopy(payload)
                    data['records'].sort(key=lambda r:r['sample_index'])
                    for record in data['records']:
                        record['fk']['candidates'].sort(key=lambda c:c['particle'])
                        for candidate in record['fk']['candidates']:
                            candidate['paired_original_quality_rank']=candidate['rank']
                            candidate['rank']=int(candidate['particle'])
                    data['selection']['top_k']=4
                    data['paired_execution']=dict(arm=arm,source_file=str(original_path),source_sha256=hashes[str(original_path)],
                        ordering='sample_index then particle; rank aliases particle only for input format',no_selection=True,no_new_optimization=True)
                    path=dest/'all_particles.json'
                    if path.exists():assert base.read(path)==data
                    else:base.save(path,data)
    assert len(pairing)==3072
    prep=RUN/'prepare_ablation.py'
    if not prep.exists():shutil.copy2(SOURCE/'prepare_ablation.py',prep)
    assert base.sha(prep)==base.sha(SOURCE/'prepare_ablation.py')
    manifest=dict(previous)
    manifest.update(source_run=str(SOURCE),arms={'ENV0':'saved FK output before ENV','ENV200':'same particles after saved ENV200'},
        config_paths=configs,driver_sha256=base.sha(__file__),validator_sha256=base.sha(VALIDATOR),
        base_driver_sha256=base.sha(BASE),source_candidate_sha256=hashes,
        expected_batches=384,expected_execution_slots=6144,expected_pairs=3072,
        workers=WORKERS,physics_threads_per_process=4,no_new_generation=True,no_quality_ranking=True,
        execution_order='sample_index then particle, same order in both arms; reset scene before every particle')
    if (RUN/'manifest.json').exists():assert base.read(RUN/'manifest.json')==manifest
    else:base.save(RUN/'manifest.json',manifest)
    base.save(RUN/'input_pairing.json',pairing)
    return manifest


def validate(job,manifest):
    arm,view,condition,hand=job;dest=folder(job)
    result_path=dest/'validation_result.json'
    if result_path.exists() and base.read(result_path).get('complete'):return base.read(result_path)
    start=time.monotonic();candidate_file=dest/'all_particles.json';data=base.read(candidate_file)
    env=base.base_env();env.update(CUDA_VISIBLE_DEVICES='',ASSET_PATH=str(F),
        PYTHONPATH=f'{F}/third_party/isaacgym/python:{F}/InfiniGym',
        LD_LIBRARY_PATH=f'{P}/miniconda3/envs/fetchbench/lib:'+env['LD_LIBRARY_PATH'],
        TORCH_EXTENSIONS_DIR=str(base.SOURCE/'torch_extensions'),MAX_JOBS='4')
    cfg=manifest['config_paths'][arm][hand];prepared=dest/'prepared.json'
    if not prepared.exists():
        base.execute([manifest['runtime']['contact_python'],RUN/'prepare_ablation.py','--candidates',candidate_file,'--config',cfg,
            '--gripper','Barrett' if hand=='barrett' else 'shadow_hand','--candidate-mode','all','--allow-runtime-budget',
            '--closure-outer-fraction','0.10','--closure-inner-fraction','0.20','--output',prepared],env,dest/'prepare.log')
    samples=base.read(prepared)['objects'][0]['samples']
    expected=[(s,p) for s in range(4) for p in range(4)]
    assert [(int(s['source_index']),int(s['particle_index'])) for s in samples]==expected
    simulation=dest/'simulation'
    summary=simulation/'DrawerShelfSceneFactory_28/task_010'/hand/'lift_validation/summary.json'
    if not summary.exists():
        objpc=base.OLD/'generation/shampoo_drawer/corrected_inputs/sam3d'/view/'sam3d_fused_robot_base.npy'
        task='FetchPtdDRORenderBarrett' if hand=='barrett' else 'FetchPtdDRORenderShadow'
        urdf='contactdiff_v4_barrett_physics.urdf' if hand=='barrett' else 'contactdiff_v4_shadowhand_physics.urdf'
        command=[P/'miniconda3/envs/fetchbench/bin/python',VALIDATOR,f'task={task}',
            'scene=benchmark_eval/RigidObjDrawerShelf_6','task.solution.task_index=10','task.solution.visualize_top_k=16',
            'task.solution.physics_only=true','task.solution.lift.record_video=false',f'task.solution.goal_pointcloud_override={objpc}',
            f'task.solution.external_pointcloud={objpc}',f'task.solution.external_prepared={prepared}',f'task.solution.external_object_name=shampoo_drawer_{view}',
            'task.solution.external_require_precomputed_environment=false','task.solution.external_validate_infeasible=true',
            'task.solution.reject_runtime_environment_contacts=true','task.solution.target_closeup_max_tracking_displacement=0.30',
            'task.solution.lift.direct_closure=true','task.solution.lift.max_preclosure_object_displacement=0.02',
            'task.solution.lift.height=0.25','task.solution.lift.success_height=0.10','task.env.enableCameraSensors=false',
            f'task.env.robot.asset_root={F}/InfiniGym/assets/contactdiff_hands/{hand}',f'task.env.robot.urdf_file={urdf}',
            f'task.solution.artifact_dir={simulation}','seed=20260808','num_threads=4','pipeline=cpu','sim_device=cpu',
            'rl_device=cpu','graphics_device_id=-1','headless=true','force_render=false']
        base.execute(command,env,dest/'validate.log',F/'InfiniGym')
    sim=base.read(summary)
    assert sim['validated_candidates']==16 and len(sim['trials'])==16
    assert [int(t['candidate']) for t in sim['trials']]==list(range(16)), 'Validator changed particle execution order'
    rows=[]
    candidates={(r['sample_index'],c['particle']):c for r in data['records'] for c in r['fk']['candidates']}
    for trial in sim['trials']:
        index=int(trial['candidate']);sample,particle=expected[index]
        lift=float(trial['final_object_lift_m']);assert math.isfinite(lift)
        row=dict(sample=sample,particle=particle,height=lift>=.10,strict=bool(trial['success']),lift=lift,
            max_contact_error_m=candidates[(sample,particle)]['max_finger_contact_error_m'])
        for k in ('initial_environment_collision','environment_collision_steps','contact_steps','contact_lift_steps','max_preclosure_object_displacement_m'):
            if k in trial:row[k]=trial[k]
        rows.append(row)
    result=dict(complete=True,arm=arm,view=view,condition=condition,hand=hand,executed=16,
        height=sum(r['height'] for r in rows),strict=sum(r['strict'] for r in rows),trials=rows,
        source_sha256=base.sha(candidate_file),prepared_sha256=base.sha(prepared),summary=str(summary),seconds=time.monotonic()-start)
    base.save(result_path,result);return result


def summarize():
    rows=[base.read(p) for p in (RUN/'cases').glob('*/*/*/*/validation_result.json')]
    index={(r['arm'],r['view'],r['condition'],r['hand']):r for r in rows}
    groups=[]
    for arm in ('ENV0','ENV200'):
        for condition in ('A','B'):
            for hand in ('barrett','shadowhand'):
                rs=[r for r in rows if (r['arm'],r['condition'],r['hand'])==(arm,condition,hand)]
                groups.append(dict(arm=arm,condition=condition,hand=hand,batches=len(rs),executed=sum(r['executed'] for r in rs),
                    height=sum(r['height'] for r in rs),strict=sum(r['strict'] for r in rs)))
    pairs=[]
    for key,a in index.items():
        if key[0]!='ENV0' or ('ENV200',*key[1:]) not in index:continue
        b=index[('ENV200',*key[1:])]
        for x,y in zip(a['trials'],b['trials']):
            assert (x['sample'],x['particle'])==(y['sample'],y['particle'])
            pairs.append(dict(view=a['view'],condition=a['condition'],hand=a['hand'],sample=x['sample'],particle=x['particle'],
                before=x,after=y))
    transitions=[]
    for condition,hand in [('all','all'),('A','barrett'),('A','shadowhand'),('B','barrett'),('B','shadowhand')]:
        ps=[p for p in pairs if condition=='all' or (p['condition'],p['hand'])==(condition,hand)]
        t=dict(condition=condition,hand=hand,pairs=len(ps))
        for metric in ('height','strict'):
            t[metric]={name:sum(bool(p['before'][metric])==x and bool(p['after'][metric])==y for p in ps)
                for name,x,y in [('both_success',True,True),('lost',True,False),('gained',False,True),('both_failure',False,False)]}
        transitions.append(t)
    data=dict(complete=len(rows)==384 and len(pairs)==3072,finished_batches=len(rows),expected_batches=384,
        groups=groups,transitions=transitions,pairs=pairs,rows=rows)
    base.save(RUN/'results.json',data)
    lines=['# 同FK粒子：ENV前后CPU PhysX配对实验','',f'完成{len(rows)}/384批；已配对{len(pairs)}/3072个粒子。每组最多3072次执行。',
        '固定48视角、A/B、两只手及每接触集全部4粒子；不重新生成，不做Top-K筛选，不使用质量分数决定执行顺序。',
        'ENV0复用E3的FK200输出；ENV200复用对应原ENV200输出。两组固定接触集→粒子顺序，逐粒子重置场景。',
        '主指标为最终提升≥10cm，严格成功另列；所有原物理参数、关节闭合比例和抬升25cm流程不变。无GUI、无视频。','',
        '|组|条件|手|完成批数|成功/执行|严格成功|','|---|---|---|---:|---:|---:|']
    for g in groups:lines.append(f"|{g['arm']}|{g['condition']}|{g['hand']}|{g['batches']}/48|{g['height']}/{g['executed']}|{g['strict']}|")
    lines+=['','## 同粒子转换（高度成功）','','|条件|手|配对数|都成功|成功变失败|失败变成功|都失败|','|---|---|---:|---:|---:|---:|---:|']
    for t in transitions:
        h=t['height'];lines.append(f"|{t['condition']}|{t['hand']}|{t['pairs']}|{h['both_success']}|{h['lost']}|{h['gained']}|{h['both_failure']}|")
    lines+=['','所有粒子一视同仁进入物理验证，不因ENV几何指标或接触阈值不达标而丢弃。',
        '该实验估计保存的ENV阶段对同粒子的影响；不将FK几何接触达标等同于未经验证的物理成功。',
        '不和上一轮每接触集Top1的成功率直接混比；当前分母包含全部4粒子。']
    (RUN/'RESULTS.md').write_text('\n'.join(lines)+'\n')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--prepare-only',action='store_true')
    args=parser.parse_args();RUN.mkdir(parents=True,exist_ok=True)
    lock=(RUN/'launcher.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    manifest=prepare()
    if args.prepare_only:print('Prepared 3072 fixed particle pairs',flush=True);return
    started=time.time();base.save(RUN/'status.json',dict(stage='smoke',pid=os.getpid(),started=started))
    view=manifest['selected_views'][0]['view']
    smoke=[(arm,view,'A',hand) for arm in ('ENV0','ENV200') for hand in ('barrett','shadowhand')]
    with futures.ThreadPoolExecutor(4) as pool:list(pool.map(lambda j:validate(j,manifest),smoke))
    summarize()
    base.save(RUN/'status.json',dict(stage='cpu_validation',pid=os.getpid(),started=started,workers=WORKERS,physics_threads=4,expected_batches=384,expected_trials=6144))
    jobs=[(a,v['view'],c,h) for v in manifest['selected_views'] for c in ('A','B') for h in ('barrett','shadowhand') for a in ('ENV0','ENV200')]
    errors=[]
    with futures.ThreadPoolExecutor(WORKERS) as pool:
        pending={pool.submit(validate,j,manifest):j for j in jobs}
        for future in futures.as_completed(pending):
            try:future.result()
            except Exception as error:
                item=dict(job=pending[future],error=repr(error));errors.append(item);base.save(folder(pending[future])/'error.json',item)
            summarize()
    complete=base.read(RUN/'results.json')['complete']
    base.save(RUN/'status.json',dict(stage='complete' if complete and not errors else 'failed',errors=errors,started=started,finished=time.time()))


if __name__=='__main__':
    try:main()
    except Exception as error:
        base.save(RUN/'startup_error.json',dict(error=repr(error),time=time.time()));raise
