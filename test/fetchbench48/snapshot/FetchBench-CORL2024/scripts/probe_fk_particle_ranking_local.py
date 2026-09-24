"""Offline Top1 probe using ONLY cached pre-execution contact-error features.

No pose mutation or new simulation. Train/tune on the other 40 views; evaluate
once on the original eight, with all conditions/hands of a view kept together.
Previously inspected outcomes mean this is retrospective, not a blind test.
"""
import collections
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT=Path(__file__).resolve().parents[1]
FULL=ROOT/'outputs/shampoo_e3_all48_particle_paired_env0_env200_20260914/results.json'
PILOT=ROOT/'outputs/shampoo_joint_fk_env0_staged_20260914/shampoo_drawer/results.json'
OUT=ROOT/'outputs/fk_only_particle_ranking_20260915/contact_only_probe'


def read(path):return json.loads(path.read_text())


def sets(rows, arm):
    output=[]
    for r in rows:
        if r['arm']!=arm:continue
        groups=collections.defaultdict(list)
        for t in r['trials']:groups[t['sample']].append(t)
        assert set(groups)=={0,1,2,3}
        for sample,ts in sorted(groups.items()):
            ts=sorted(ts,key=lambda t:t['particle'])
            assert [t['particle'] for t in ts]==[0,1,2,3]
            assert all(t['height']==(t['lift']>=.10) for t in ts)
            output.append(dict(view=r['view'],condition=r['condition'],hand=r['hand'],sample=sample,trials=ts))
    return sorted(output,key=lambda s:(s['view'],s['condition'],s['hand'],s['sample']))


def features(s):
    # Explicit whitelist: never use contact_steps, observed displacement,
    # initial_environment_collision, lift or any other simulation measurement.
    e=np.array([t['max_contact_error_m'] for t in s['trials']])
    assert np.isfinite(e).all() and (e>=0).all()
    rank=np.argsort(np.argsort(e,kind='stable'),kind='stable')/3
    hand=float(s['hand']=='shadowhand');condition=float(s['condition']=='B')
    values=[]
    for i in range(4):
        v=[np.log1p(e[i]*1000), min(e[i],.1)*100,rank[i],
           np.log1p(max(0,e[i]-e.min())*1000),np.log1p(e.mean()*1000),
           np.log1p((e.max()-e.min())*1000), hand,condition]
        v += [x*hand for x in v[:4]]+[x*condition for x in v[:4]]
        values.append(v)
    return values


def evaluate(ss,scores):
    scores=np.asarray(scores).reshape(len(ss),4)
    choices=np.argmax(scores,axis=1);groups={};picked=[]
    for s,i in zip(ss,choices):
        t=s['trials'][int(i)]
        g=groups.setdefault(s['condition']+'/'+s['hand'],dict(sets=0,height=0,strict=0))
        g['sets']+=1;g['height']+=int(t['height']);g['strict']+=int(t['strict'])
        picked.append(dict(view=s['view'],condition=s['condition'],hand=s['hand'],sample=s['sample'],
            particle=int(i),height=bool(t['height']),strict=bool(t['strict'])))
    return dict(sets=len(ss),height=sum(x['height'] for x in picked),strict=sum(x['strict'] for x in picked),
        rate=sum(x['height'] for x in picked)/len(ss),groups=groups,selections=picked)


def baselines(ss):
    contact=np.asarray([[t['max_contact_error_m'] for t in s['trials']] for s in ss])
    return dict(min_contact=evaluate(ss,-contact),
        particle0=evaluate(ss,np.tile([1,0,0,0],(len(ss),1))),
        uniform_random_expected_successes=sum(sum(t['height'] for t in s['trials'])/4 for s in ss),
        oracle_hit4=sum(any(t['height'] for t in s['trials']) for s in ss),sets=len(ss))


def model(name):
    if name=='logistic':return make_pipeline(StandardScaler(),LogisticRegression(C=.1,max_iter=1000,class_weight='balanced',random_state=20260915))
    if name=='histgb':return HistGradientBoostingClassifier(max_iter=100,max_leaf_nodes=7,min_samples_leaf=30,l2_regularization=10,learning_rate=.05,early_stopping=False,random_state=20260915)
    if name=='forest':return RandomForestClassifier(n_estimators=128,max_depth=6,min_samples_leaf=16,class_weight='balanced',n_jobs=2,random_state=20260915)
    raise ValueError(name)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    allsets=sets(read(FULL)['rows'],'ENV0');test=sets(read(PILOT)['rows'],'J0')
    views={s['view'] for s in test};train=[s for s in allsets if s['view'] not in views]
    assert len(allsets)==768 and len(train)==640 and len(test)==128
    reference={(s['view'],s['condition'],s['hand'],s['sample']):s for s in allsets}
    for s in test:
        r=reference[(s['view'],s['condition'],s['hand'],s['sample'])]
        assert [(t['height'],t['max_contact_error_m']) for t in r['trials']]==[(t['height'],t['max_contact_error_m']) for t in s['trials']]
    X=np.asarray([features(s) for s in train]).reshape(-1,16)
    y=np.array([t['height'] for s in train for t in s['trials']],dtype=int)
    groups=np.repeat([s['view'] for s in train],4)
    folds=list(GroupKFold(5).split(X,y,groups));cv={}
    for name in ('logistic','histgb','forest'):
        scores=np.full(len(y),np.nan)
        for fit,valid in folds:
            assert set(groups[fit]).isdisjoint(groups[valid])
            est=model(name);est.fit(X[fit],y[fit]);scores[valid]=est.predict_proba(X[valid])[:,1]
        assert np.isfinite(scores).all()
        cv[name]=evaluate(train,scores)
    # The choice depends ONLY on out-of-fold Top1 in the forty training views.
    chosen=max(cv,key=lambda name:(cv[name]['height'],cv[name]['strict'],-list(cv).index(name)))
    est=model(chosen);est.fit(X,y)
    pred=est.predict_proba(np.asarray([features(s) for s in test]).reshape(-1,16))[:,1]
    result=dict(status='contact-only preliminary probe; full geometric features blocked by changed SSH host key',
        validation='40 historical views train/tune (5-fold view-group CV); original8 retrospective heldout. No same-view split across A/B/hands.',
        excluded_features='All simulation-state measurements; candidate ID is not a learned feature',
        sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (FULL,PILOT)},
        training_views=sorted(set(groups)),evaluation_views=sorted(views),train_cv=cv,chosen_model=chosen,
        evaluation=evaluate(test,pred),baselines_original8=baselines(test),baselines_all48=baselines(allsets))
    (OUT/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    joblib.dump(est,OUT/'contact_only_ranker.joblib')
    b=result['baselines_original8'];e=result['evaluation'];full=result['baselines_all48']
    lines=['# FK-only 四粒子 Top1：本地初步排序探究','',
        '只使用缓存中优化完成、仿真开始前的最大手指接触误差及其组内相对值。不使用执行后的接触、位移或抬升信息。',
        '未更改姿态，也未重新仿真；以下为在已有独立粒子仿真标签上的离线选择回放。',
        '训练/选模型使用其余40视角，5折按视角分组；原8视角仅做最终回顾性评估。历史标签此前已被检查，不能称为盲测或跨物体泛化。','',
        '|方法|原8视角成功/128|成功率|','|---|---:|---:|',
        f"|最小接触误差|{b['min_contact']['height']}/128|{100*b['min_contact']['rate']:.2f}%|",
        f"|接触特征排序器（{chosen}，其余40视角训练）|{e['height']}/128|{100*e['rate']:.2f}%|",
        f"|均匀随机选择（期望，不是实测整数成功数）|{b['uniform_random_expected_successes']}/128|{100*b['uniform_random_expected_successes']/128:.2f}%|",
        f"|事后oracle上限，不可部署|{b['oracle_hit4']}/128|{100*b['oracle_hit4']/128:.2f}%|",'',
        f"全部48视角的候选池oracle为{full['oracle_hit4']}/{full['sets']}={100*full['oracle_hit4']/full['sets']:.2f}%。",
        '原8视角严格超过30%需至少39/128，需从49个可成功接触集中选对39个（79.59%）。',
        '全部48视角严格超过30%需至少231/768，需从240个可成功接触集中选对231个（96.25%）。',
        '', '## 后续所需数据','',
        '完整FK候选文件中的力闭合/GraspQP、物体穿透、自碰撞、接触分布、关节及姿态；同视角partial scene闭合轨迹环境评分。',
        '不对测试集反复调阈值；完整特征方法需要重新明确冻结验证划分，并在新种子/未用于调参的视角上验证。']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(chosen=chosen,cv={k:v['height'] for k,v in cv.items()},test=e['height'],sets=e['sets'],min_contact=b['min_contact']['height'],oracle=b['oracle_hit4'])))


if __name__=='__main__':main()
