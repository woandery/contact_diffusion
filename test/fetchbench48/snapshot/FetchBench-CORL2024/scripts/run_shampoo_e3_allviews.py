"""Fresh E3 generation and CPU PhysX validation over every legal Shampoo view.

Reuses frozen E3 stage commands and hook; no FK-throughout environment term.
All outputs are isolated; checkpoints, historical configurations and assets are read-only.
"""
import argparse
import concurrent.futures as futures
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import queue
import shutil
import subprocess
import time

import numpy as np

P=Path('/inspire/qb-ilm2/project/zhanghanbo/public/mck')
C=P/'ContactDiffusion';F=P/'FetchBench-CORL2024'
RUN=C/'outputs/shampoo_drawer_e3_all48views_s4p4_fk200_env200_20260914'
OLD_PROBE=C/'outputs/shampoo_drawer_environment_probe_4x4_200_20260914'
PROBE=F/'scripts/run_shampoo_environment_probe_20260914.py'
spec=importlib.util.spec_from_file_location('allviews_frozen_e3',PROBE)
probe=importlib.util.module_from_spec(spec);spec.loader.exec_module(probe)
base=probe.base
base.RUN=RUN;probe.RUN=RUN
base.ARMS={'E3':(4,4,200)}
probe.generate.__globals__.update(RUN=RUN,ARMS=base.ARMS)


def prepare():
    old=base.read(OLD_PROBE/'manifest.json')
    assert base.sha(PROBE)==old['driver_sha256']
    assert base.sha(probe.HOOK)==old['hook_sha256']
    for p,h in old['code_sha256'].items():assert base.sha(p)==h,p
    for k,h in old['checkpoint_sha256'].items():assert base.sha(old['runtime'][k])==h,k
    obs=Path(old['case']['observations'])
    legal_path=obs/'visibility/legal_partial_views.json'
    legal=base.read(legal_path)
    views=[Path(v['rgbd_capture_dir']).name for v in legal['views']]
    assert len(views)==48 and len(set(views))==48
    base.VIEWS=views
    configs={'E3':{}}
    for hand in ('barrett','shadowhand'):
        src=Path(old['config_paths']['E3'][hand]);dst=RUN/'configs'/f'E3_{hand}.yaml'
        dst.parent.mkdir(parents=True,exist_ok=True)
        if not dst.exists():shutil.copy2(src,dst)
        assert base.sha(src)==base.sha(dst)
        configs['E3'][hand]=str(dst)
    inputs={};frame_audit=[]
    for view in views:
        corrected=base.OLD/'generation/shampoo_drawer/corrected_inputs'
        md=corrected/'visibility/rgbd_views'/view/'metadata.json'
        assert base.read(md)['coordinate_frame']=='robot_base',str(md)
        geom=corrected/'sam3d'/view
        metadata=base.read(geom/'sam3d_fused_centered.json')
        t=np.asarray(metadata.get('robot_from_pointcloud_frame',metadata.get('robot_from_object')))
        assert t.shape==(4,4) and np.allclose(t[3],[0,0,0,1])
        assert np.allclose(t[:3,:3].T@t[:3,:3],np.eye(3),atol=1e-5)
        paths=[md,corrected/'visibility/rgbd_views'/view/'scene_partial_robot_base.npy',
               geom/'sam3d_fused_robot_base.npy',geom/'sam3d_fused_centered.json',
               obs/'sam3d'/view/'camera_partial_sam3d_centered.npy',obs/'sam3d'/view/'sam3d_fused_centered.npy']
        arrays={str(p):np.load(p) for p in paths if p.suffix=='.npy'}
        for p,a in arrays.items():assert a.ndim==2 and a.shape[1]==3 and len(a)>0 and np.isfinite(a).all(),p
        x=arrays[str(obs/'sam3d'/view/'sam3d_fused_centered.npy')]
        y=arrays[str(geom/'sam3d_fused_robot_base.npy')]
        residual=float(np.max(np.abs(x@t[:3,:3].T+t[:3,3]-y)))
        assert residual<1e-5,(view,residual)
        frame_audit.append(dict(view=view,maximum_transform_residual_m=residual))
        inputs[view]={str(p):base.sha(p) for p in paths}
        for condition in ('A','B'):
            for hand in ('barrett','shadowhand'):
                fallback=base.OLD/'generation/shampoo_drawer/AB/views'/view/condition/'raw_object'/f'shampoo_drawer_{hand}.json'
                assert fallback.is_file(),str(fallback)
    manifest=dict(old)
    manifest.update(arms={'E3':[4,4,200]},config_paths=configs,
        config_sha256={'E3':{h:base.sha(p) for h,p in configs['E3'].items()}},
        source_run=str(OLD_PROBE),driver_sha256=base.sha(__file__),frozen_probe_driver_sha256=base.sha(PROBE),
        selected_views=[dict(view=v,original_legal_index=i,metadata=legal['views'][i]) for i,v in enumerate(views)],
        view_selection='All 48 legal views from the existing 111-camera observation protocol; no success-based selection',
        legal_views_sha256=base.sha(legal_path),inputs=inputs,frame_audit=frame_audit,
        expected_batches=192,expected_slots=768,fresh_generation=True,full_fk_environment=False,
        numerical_mode='Original E3 default math; do not mix in the F0/F1 strict-math intervention',
        generation_workers=12,validation_workers=8,
        comparison='Full-view E3 test; historical 8-view E3 comparison is an overlap reproducibility audit, not a new control arm')
    source=OLD_PROBE/'prepare_ablation.py';dst=RUN/'prepare_ablation.py'
    if not dst.exists():shutil.copy2(source,dst)
    assert base.sha(source)==base.sha(dst)
    if (RUN/'manifest.json').exists():assert base.read(RUN/'manifest.json')==manifest
    else:base.save(RUN/'manifest.json',manifest)
    base.save(RUN/'frame_audit.json',frame_audit)
    return manifest


def summarize(manifest):
    rows=[base.read(p) for p in (RUN/'cases').glob('E3/*/*/*/validation_result.json')]
    groups=[]
    for subset in ('all48','original8','additional40'):
        original={v['view'] for v in base.read(OLD_PROBE/'manifest.json')['selected_views']}
        selected=[r for r in rows if subset=='all48' or ((r['view'] in original)==(subset=='original8'))]
        for condition in ('A','B'):
            for hand in ('barrett','shadowhand'):
                rs=[r for r in selected if r['condition']==condition and r['hand']==hand]
                g=dict(subset=subset,condition=condition,hand=hand,batches=len(rs),planned=4*len(rs),
                       expected_slots=4*(48 if subset=='all48' else 8 if subset=='original8' else 40))
                for key in ('height','strict','executed','invalid','top1_height','hit4'):
                    g[key]=sum(int(r[key]) for r in rs)
                g['missing_contact_sets']=sum(r.get('missing_contact_sets',0) for r in rs)
                groups.append(g)
    audits=[base.read(p) for p in (RUN/'cases').glob('E3/*/*/*/intervention_audit.json')]
    screening=dict(batches=len(audits),draws=sum(len(a['draws']) for a in audits),
        rejected=sum(not d['accepted'] for a in audits for d in a['draws']),
        exhausted=sum(bool(a['budget_exhausted']) for a in audits))
    output=dict(complete=len(rows)==192,finished_batches=len(rows),expected_batches=192,groups=groups,rows=rows,screening=screening)
    base.save(RUN/'results.json',output)
    lines=['# 抽屉 Shampoo：E3 全48合法视角测试','',f'完成 {len(rows)}/192 批。每批4接触集×4粒子、FK200+ENV200，CPU PhysX，无GUI/视频。',
        'A为单视角partial条件，B为同视角SAM3D条件；辅助物体几何均仍为SAM3D，环境来自单视角partial scene。',
        'E3=5mm接触集筛选（最多32次补齐4组）+原4粒子固定关节/朝向的最多4cm平移初始化；不含FK全程环境项。',
        '主成功指标为最终提升≥10cm，严格成功另列。缺失接触集、NaN单列并计入计划分母；未完成批次不预先算失败。',
        '', '|范围|条件|手|完成批数|成功/已完成计划槽位|成功/实际执行|严格成功|Top1成功|Top4命中|缺失|NaN|',
        '|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for g in groups:
        lines.append(f"|{g['subset']}|{g['condition']}|{g['hand']}|{g['batches']}|{g['height']}/{g['planned']}|{g['height']}/{g['executed']}|{g['strict']}|{g['top1_height']}|{g['hit4']}|{g['missing_contact_sets']}|{g['invalid']}|")
    lines+=['',f"筛选审计：{screening['draws']}次尝试，拒绝{screening['rejected']}次，{screening['exhausted']}批预算耗尽。",
        '', '## 逐视角成功/已完成计划槽位','','|视角|A Barrett|A ShadowHand|B Barrett|B ShadowHand|','|---|---:|---:|---:|---:|']
    for v in manifest['selected_views']:
        cells=[]
        for c,h in [('A','barrett'),('A','shadowhand'),('B','barrett'),('B','shadowhand')]:
            rs=[r for r in rows if r['view']==v['view'] and r['condition']==c and r['hand']==h]
            cells.append(f"{rs[0]['height']}/4" if rs else '待完成')
        lines.append('|'+v['view']+'|'+'|'.join(cells)+'|')
    lines+=['','旧8视角20.31%不是全视角预期成功率，也不是相对新生成基线的显著收益证明。',
        '本轮不新跑DRO或F0；不同历史实验的成功率不能直接当作严格配对因果比较。',
        '原8视角与新增40视角分开统计，避免将视角组成变化误判为算法变化。']
    (RUN/'RESULTS.md').write_text('\n'.join(lines)+'\n')


def overlap_audit():
    rows=[]
    old=base.read(OLD_PROBE/'results.json')
    for prior in old['rows']:
        if prior['arm']!='E3':continue
        rel=Path('E3')/prior['view']/prior['condition']/prior['hand']
        current=RUN/'cases'/rel
        if not (current/'validation_result.json').exists():continue
        a=base.read(OLD_PROBE/'cases'/rel/'selected_w10.json')['records']
        b=base.read(current/'selected_w10.json')['records']
        rows.append(dict(view=prior['view'],condition=prior['condition'],hand=prior['hand'],
            target_hashes_same=[r['target_contacts_sha256'] for r in a]==[r['target_contacts_sha256'] for r in b],
            selected_poses_same=[(r['fk']['candidates'][0]['root_pose'],r['fk']['candidates'][0]['joint_positions']) for r in a]==[(r['fk']['candidates'][0]['root_pose'],r['fk']['candidates'][0]['joint_positions']) for r in b],
            historical_height=prior['height'],fresh_height=base.read(current/'validation_result.json')['height']))
    base.save(RUN/'overlap_audit.json',rows)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--prepare-only',action='store_true')
    args=parser.parse_args();RUN.mkdir(parents=True,exist_ok=True)
    lock=(RUN/'launcher.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    manifest=prepare()
    if args.prepare_only:print('Prepared all48',flush=True);return
    assert set(subprocess.check_output(['nvidia-smi','--query-gpu=index','--format=csv,noheader'],text=True).split())=={'0','1','2','3'}
    started=time.time();base.save(RUN/'status.json',dict(stage='smoke',pid=os.getpid(),started=started))
    slots=queue.Queue()
    for _ in range(3):
        for gpu in range(4):slots.put(gpu)
    def work(job):
        gpu=slots.get();t=time.monotonic()
        try:
            if not (base.folder(job)/'generation_done.json').exists():
                probe.generate(job,gpu,manifest)
                base.save(base.folder(job)/'generation_done.json',dict(complete=True,gpu=gpu,elapsed_seconds=time.monotonic()-t))
            return job
        finally:slots.put(gpu)
    smoke=[('E3',base.VIEWS[0],c,h) for c in ('A','B') for h in ('barrett','shadowhand')]
    with futures.ThreadPoolExecutor(4) as pool:
        for job in pool.map(work,smoke):probe.validate(job,manifest)
    summarize(manifest)
    base.save(RUN/'status.json',dict(stage='generation_and_cpu_validation',pid=os.getpid(),started=started,gpus=4,generation_workers=12,validation_workers=8,expected_batches=192,expected_slots=768))
    jobs=[('E3',v,c,h) for v in base.VIEWS for c in ('A','B') for h in ('barrett','shadowhand')]
    errors=[]
    with futures.ThreadPoolExecutor(8) as vp,futures.ThreadPoolExecutor(12) as gp:
        gjobs={gp.submit(work,j):j for j in jobs};vjobs={}
        for future in futures.as_completed(gjobs):
            job=gjobs[future]
            try:vjobs[vp.submit(probe.validate,future.result(),manifest)]=job
            except Exception as error:
                item=dict(job=job,stage='generation',error=repr(error));errors.append(item);base.save(base.folder(job)/'error.json',item)
            summarize(manifest)
        for future in futures.as_completed(vjobs):
            try:future.result()
            except Exception as error:
                job=vjobs[future];item=dict(job=job,stage='validation',error=repr(error));errors.append(item);base.save(base.folder(job)/'error.json',item)
            summarize(manifest)
    overlap_audit();summarize(manifest)
    complete=base.read(RUN/'results.json')['complete']
    base.save(RUN/'status.json',dict(stage='complete' if complete and not errors else 'failed',errors=errors,started=started,finished=time.time()))


if __name__=='__main__':
    try:main()
    except Exception as error:
        base.save(RUN/'startup_error.json',dict(error=repr(error),time=time.time()));raise
