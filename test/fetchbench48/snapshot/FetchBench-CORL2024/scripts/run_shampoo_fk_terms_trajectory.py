"""Three isolated objectives, five arms; same 8-view paired FK/CPU physics.

J0 original FK; E1/E3/E10 only environment mean+CVaR; OC only object/contact
quadratic barriers. Every arm records all 201 states; no ENV refinement.
"""
import argparse
import fcntl
import importlib.util
import inspect
import os
from pathlib import Path
import time


ORIGINAL=Path(__file__).with_name('run_shampoo_joint_fk_staged.py')
spec=importlib.util.spec_from_file_location('terms_frozen_pipeline',ORIGINAL)
run=importlib.util.module_from_spec(spec);spec.loader.exec_module(run)
base=run.base
run.ROOT=run.C/'outputs/shampoo_fk_terms_trajectory_8views_20260915'
run.RUN=run.ROOT/'shampoo_drawer'
run.HOOK=Path(__file__).with_name('shampoo_fk_terms_trajectory_infer.py')
run.ARMS={'J0':0.,'E1':1.,'E3':3.,'E10':10.,'OC':0.}
MODES={'J0':'baseline','E1':'environment','E3':'environment','E10':'environment','OC':'obstacles'}


def protocol():
    return dict(protocol_id='fk-terms-separated-s4p4-fk200-env0-trajectories201-v1',
        objectives={'J0':'unchanged FK only','E1/E3/E10':'FK+w*mean_cvar_environment; ALL new quadratic barriers zero',
                    'OC':'FK+ramp*(4*object_barrier+10*contact_barrier); ALL environment terms zero'},
        mode_by_arm=MODES,standalone_env_steps=0,weight_selection=False,promotion=False,
        true_recorded_steps=list(range(201)),trajectory_note='State 0 before Adam; state k after exactly k updates; no interpolation',
        execution_budget_per_arm=512,expected_batches=160,expected_execution_slots=2560,
        original_joint_driver_sha256=base.sha(ORIGINAL),driver_sha256=base.sha(__file__),
        hook_sha256=base.sha(run.HOOK))


run.__dict__['_separated_protocol']=protocol
run.__dict__['_mode_by_arm']=MODES
source=inspect.getsource(run.prepare)
anchor="    if (RUN/'manifest.json').exists():"
assert source.count(anchor)==1
source=source.replace(anchor,
    "    manifest.update(_separated_protocol())\n"
    "    manifest['gate'] = None\n"
    "    manifest['conditional_transfer'] = []\n"+anchor)
exec(compile(source,'<separated-fk-protocol>','exec'),run.__dict__)

source=inspect.getsource(run.stages)
anchor="    base.save(dest/'joint_metadata.json', meta)"
assert source.count(anchor)==1
source=source.replace(anchor,"    meta['separated_mode'] = _mode_by_arm[arm]\n"+anchor)
anchor="    raw = base.read(files[0]); pairing = []"
assert source.count(anchor)==1
source=source.replace(anchor,
    "    for slot in range(4):\n"
    "        trajectory = np.load(dest/f'trajectory_{slot}.npz')\n"
    "        assert trajectory['root_pose'].shape == (201,4,4,4)\n"
    "        assert list(trajectory['step']) == list(range(201))\n"+anchor)
exec(compile(source,'<paired-separated-stages>','exec'),run.__dict__)


def summary(manifest):
    rows=run.report(manifest)
    groups=[]
    for arm in run.ARMS:
        for condition,hand in [('all','all'),('A','barrett'),('A','shadowhand'),('B','barrett'),('B','shadowhand')]:
            rs=[r for r in rows if r['arm']==arm and (condition=='all' or (r['condition'],r['hand'])==(condition,hand))]
            groups.append(dict(arm=arm,condition=condition,hand=hand,batches=len(rs),
                **{k:sum(r[k] for r in rs) for k in ('height','strict','executed')}))
    result=dict(complete=len(rows)==160,groups=groups,rows=rows)
    base.save(run.RUN/'separated_results.json',result)
    lines=['# FK 分项消融：8视角、五组、真实200步轨迹','',
        'J0=原FK；E1/E3/E10=仅环境mean/CVaR（w=1/3/10）；OC=仅额外物体/接触平方惩罚。',
        '相同E3预处理、接触点、初始化、4×4粒子、FK200；无独立ENV。全部粒子CPU PhysX，无GUI/视频。',
        '主指标最终提升≥10cm；严格成功另列。每组512次；轨迹包含初始状态及每次真实更新，共201帧。','',
        '|组|输入|手|完成批数|高度成功/执行|严格成功|','|---|---|---|---:|---:|---:|']
    for g in groups:lines.append(f"|{g['arm']}|{g['condition']}|{g['hand']}|{g['batches']}|{g['height']}/{g['executed']}|{g['strict']}|")
    (run.RUN/'SEPARATED_RESULTS.md').write_text('\n'.join(lines)+'\n')
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--prepare-only',action='store_true');parser.add_argument('--smoke-only',action='store_true')
    args=parser.parse_args();run.ROOT.mkdir(parents=True,exist_ok=True)
    lock=(run.ROOT/'launcher.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    manifest=run.prepare('shampoo_drawer')
    if args.prepare_only:print('Prepared five-arm separated experiment',flush=True);return
    started=time.time();views=manifest['pilot_views']
    run.run_jobs(manifest,views[:1],['J0','E10','OC'],'smoke')
    summary(manifest)
    if args.smoke_only:return
    run.run_jobs(manifest,views,list(run.ARMS),'separated_generation_validation')
    result=summary(manifest);assert result['complete']
    status=dict(stage='complete',pid=os.getpid(),started=started,finished=time.time(),validated=2560)
    base.save(run.RUN/'status.json',status);base.save(run.ROOT/'status.json',status)


if __name__=='__main__':
    try:main()
    except Exception as error:
        base.save(run.ROOT/'status.json',dict(stage='failed',error=repr(error),time=time.time()))
        raise
