"""Geometry-only ranking, view-grouped model selection, fixed existing poses."""
import argparse
import gzip
import json
from pathlib import Path
import importlib.util

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier,HistGradientBoostingClassifier,RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'outputs/fk_only_particle_ranking_20260915'
spec=importlib.util.spec_from_file_location('contact_probe',Path(__file__).with_name('probe_fk_particle_ranking_local.py'))
helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)

SCALARS=['optimization_score','contact_chamfer_m','object_surface_distance_sum_m','assigned_contact_error_m',
    'max_finger_contact_error_m','mean_penetration_m','cvar_penetration_m','hinge_penetration_m',
    'max_penetration_m','raw_max_penetration_m','penetration_normal_confidence','penetrating_surface_fraction',
    'mean_self_collision_m','cvar_self_collision_m','max_self_collision_m','self_collision_pair_fraction',
    'envelope_side_cosine','contact_patch_side_cosine','palm_approach_cosine','palm_normal_to_contact_plane_cosine',
    'graspqp_score','graspqp_wrench_residual','graspqp_min_singular_value','selection_feasible']


def summary(a):
    a=np.asarray(a)
    return [float(a.min()),float(a.mean()),float(a.max()),float(a.std())]


def features(r,env):
    lo=np.asarray(r['fk_metadata']['joint_lower']);hi=np.asarray(r['fk_metadata']['joint_upper'])
    T=np.asarray(r['robot_from_object']);obj=r['object_summary'];center=np.asarray(obj['mean'])
    bounds=np.maximum(np.asarray(obj['upper'])-obj['lower'],.001)
    names=[];result=[]
    for c in r['candidates']:
        f={k:float(c[k]) for k in SCALARS}
        errors=np.array(c['per_finger_contact_errors_m'])
        for i,e in enumerate(np.pad(np.sort(errors),(0,5-len(errors)))):f[f'finger_sorted_error{i}']=float(e)
        for threshold in (.005,.010,.020):f[f'finger_fraction_within{threshold}']=float((errors<=threshold).mean())
        q=(np.array(c['joint_positions'])-lo)/np.maximum(hi-lo,1e-6)
        for i,e in enumerate(np.pad(q,(0,32-len(q)))):f[f'joint_normalized{i}']=float(e)
        for i,e in enumerate(summary(q)):f[f'joint_summary{i}']=e
        for i,e in enumerate(summary(np.minimum(q,1-q))):f[f'joint_limit_margin{i}']=e
        pose=np.array(c['root_pose']);rot=pose[:3,:3];position=pose[:3,3]
        for i,e in enumerate(((position-center)/bounds)):f[f'root_relative_xyz{i}']=float(e)
        for i,e in enumerate(rot.reshape(-1)):f[f'root_rotation{i}']=float(e)
        worldrot=T[:3,:3]@rot
        for i,e in enumerate(worldrot[2]):f[f'hand_axis_gravity_projection{i}']=float(e)
        matched=np.asarray(c['matched_contact_points']);target=np.asarray(c['matched_target_object_points'])
        normals=np.asarray(c['matched_contact_normals']);targetnormals=np.asarray(c['matched_target_object_normals'])
        assert matched.shape==target.shape==normals.shape==targetnormals.shape
        # Raw-object file: ALL of these vectors share the same frame.
        for label,values in [('normal_opposition',-np.sum(normals*targetnormals,axis=1)),
                             ('signed_contact_gap',np.sum((matched-target)*targetnormals,axis=1))]:
            for i,e in enumerate(summary(values)):f[f'{label}{i}']=e
        for label,points in [('matched',matched),('target',target)]:
            sv=np.linalg.svd((points-points.mean(0))/np.linalg.norm(bounds),compute_uv=False)
            for i,e in enumerate(sv):f[f'{label}_spread_sv{i}']=float(e)
            for i,e in enumerate((points.mean(0)-center)/bounds):f[f'{label}_relative_center{i}']=float(e)
        f['normal_vector_sum']=float(np.linalg.norm(targetnormals.mean(0)))
        f['hand_shadow']=float(r['hand']=='shadowhand');f['condition_B']=float(r['condition']=='B')
        if env is not None:f.update(env[(r['view'],r['condition'],r['hand'],r['sample'],c['particle'])])
        current=list(f)
        if names:assert current==names
        else:names=current
        result.append(list(f.values()))
    raw=np.asarray(result)
    assert np.isfinite(raw).all()
    # Within-set relative features carry no outcome information. Rank ties
    # use mean ranks rather than particle IDs.
    ranks=(raw[:,None,:]>raw[None,:,:]).sum(1)/3
    ranks+=(raw[:,None,:]==raw[None,:,:]).sum(1)/6-1/6
    centered=raw-np.median(raw,axis=0)
    out=np.concatenate((raw,ranks,centered),axis=1)
    return out,names+['within_rank_'+k for k in names]+['within_centered_'+k for k in names]


def make_model(name):
    if name.startswith('hist'):
        leaves=int(name.split('_')[1]);return HistGradientBoostingClassifier(max_iter=180,max_leaf_nodes=leaves,min_samples_leaf=16,l2_regularization=5,learning_rate=.05,early_stopping=False,random_state=20260915)
    if name.startswith('forest'):
        leaf=int(name.split('_')[1]);return RandomForestClassifier(n_estimators=160,min_samples_leaf=leaf,max_depth=12,max_features=.7,class_weight='balanced',n_jobs=2,random_state=20260915)
    if name.startswith('extra'):
        leaf=int(name.split('_')[1]);return ExtraTreesClassifier(n_estimators=200,min_samples_leaf=leaf,max_features=.7,class_weight='balanced',n_jobs=2,random_state=20260915)
    if name.startswith('logistic') or name.startswith('pairwise'):
        return make_pipeline(StandardScaler(),LogisticRegression(C=float(name.split('_')[1]),max_iter=2000,class_weight='balanced',random_state=20260915))
    raise ValueError(name)


def fit(name,X,y):
    model=make_model(name)
    if name.startswith('pairwise'):
        diffs=[];labels=[]
        for xx,yy in zip(X.reshape(-1,4,X.shape[1]),y.reshape(-1,4)):
            for i in np.flatnonzero(yy):
                for j in np.flatnonzero(1-yy):
                    diffs.extend([xx[i]-xx[j],xx[j]-xx[i]]);labels.extend([1,0])
        model.fit(np.asarray(diffs),labels)
    else:model.fit(X,y)
    return model


def score(name,model,X):
    if name.startswith('pairwise'):
        scaler=model.steps[0][1];coef=model.steps[1][1].coef_[0]/scaler.scale_
        return X@coef
    return model.predict_proba(X)[:,1]


def converted(rows):
    return [dict(view=r['view'],condition=r['condition'],hand=r['hand'],sample=r['sample'],trials=r['labels']) for r in rows]


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--environment',action='store_true');parser.add_argument('--closure',action='store_true');args=parser.parse_args()
    if args.closure:args.environment=True
    out=BASE/('geometry_closure_probe' if args.closure else 'geometry_environment_probe' if args.environment else 'geometry_probe');out.mkdir(parents=True,exist_ok=True)
    with gzip.open(BASE/'source_dataset.json.gz','rt') as f:data=json.load(f)
    views={r['view'] for r in helper.read(helper.PILOT)['rows']}
    rows=sorted(data['rows'],key=lambda r:(r['view'],r['condition'],r['hand'],r['sample']))
    train=[r for r in rows if r['view'] not in views];test=[r for r in rows if r['view'] in views]
    assert len(train)==640 and len(test)==128
    env=None
    if args.environment:
        env={}
        for c in ('A','B'):
            for h in ('barrett','shadowhand'):
                prefix='closure' if args.closure else 'environment'
                for row in helper.read(BASE/f'{prefix}_features_{c}_{h}.json')['rows']:
                    env[(row['view'],row['condition'],row['hand'],row['sample'],row['particle'])]=row['features']
        assert len(env)==3072
    names=features(train[0],env)[1]
    X=np.concatenate([features(r,env)[0] for r in train]);y=np.array([t['height'] for r in train for t in r['labels']],dtype=int)
    groups=np.repeat([r['view'] for r in train],4);folds=list(GroupKFold(5).split(X,y,groups))
    cv={};models=['hist_7','hist_15','forest_4','forest_12','extra_4','extra_12','logistic_0.01','pairwise_0.01','pairwise_0.1']
    if args.closure:models=['hist_7','forest_12','pairwise_0.01']
    # All families/parameters are fixed before evaluating the original8.
    for name in models:
        pred=np.full(len(y),np.nan)
        for tr,va in folds:
            assert set(groups[tr]).isdisjoint(groups[va])
            model=fit(name,X[tr],y[tr]);pred[va]=score(name,model,X[va])
        cv[name]=helper.evaluate(converted(train),pred)
        print('train40_cv',name,cv[name]['height'],cv[name]['sets'],flush=True)
    chosen=max(cv,key=lambda n:(cv[n]['height'],cv[n]['strict'],-models.index(n)))
    model=fit(chosen,X,y)
    xtest=np.concatenate([features(r,env)[0] for r in test]);pred=score(chosen,model,xtest)
    evaluation=helper.evaluate(converted(test),pred)
    baselines={}
    for key,sign in [('rank',-1),('graspqp_score',-1),('graspqp_wrench_residual',-1),('optimization_score',-1),('max_finger_contact_error_m',-1),('palm_approach_cosine',1)]:
        baselines[key]=helper.evaluate(converted(test),[[sign*c[key] for c in r['candidates']] for r in test])
    result=dict(chosen=chosen,evaluation=evaluation,train_cv=cv,baselines=baselines,features=names,
        training_views=sorted(set(groups)),evaluation_views=sorted(views),environment=args.environment,closure=args.closure,
        no_pose_changes=True,no_new_simulation=True,validation='40 train/tune views with five-fold grouped CV; 8 historical retrospective evaluation views, not blind',
        simulation_features_excluded=True,palm_distance_feature_excluded=True)
    (out/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    joblib.dump(dict(model=model,name=chosen,features=names),out/'ranker.joblib')
    lines=['# 固定 FK-only 候选：完整几何排序试验','',f'环境特征：{args.environment}；训练选择模型：{chosen}。',
        '其余40视角用于训练和5折分组模型选择，原8视角回顾性评估。没有重新生成/修改姿态，没有新跑仿真。',
        f"原8视角Top1：{evaluation['height']}/128 = {100*evaluation['rate']:.2f}%；严格成功：{evaluation['strict']}/128。",'',
        '|条件|成功/组数|严格成功|','|---|---:|---:|']
    for k,g in evaluation['groups'].items():lines.append(f"|{k}|{g['height']}/{g['sets']}|{g['strict']}|")
    lines+=['','模型选择只使用训练视角的OOF Top1；该最优CV数本身存在模型选择偏差，不能用作独立测试成功率。',
        '测试集历史标签已被查看且当前是探索性比较，不主张新种子或跨物体泛化。',
        '四候选oracle为49/128；若Top1未达39/128，未完成超过30%的目标。']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print('RESULT',json.dumps(dict(chosen=chosen,top1=evaluation['height'],strict=evaluation['strict'],sets=128)),flush=True)


if __name__=='__main__':main()
