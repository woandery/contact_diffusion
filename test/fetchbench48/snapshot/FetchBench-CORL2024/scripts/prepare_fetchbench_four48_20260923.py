"""Joint outcome-blind allocation of four disjoint 48-target cohorts."""
import collections
import hashlib
import importlib.util
import json
from pathlib import Path
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix

ROOT=Path(__file__).resolve().parents[1]
SERIES=ROOT/'outputs/fetchbench_four48_series_20260923'
spec=importlib.util.spec_from_file_location('prepare_base',ROOT/'scripts/prepare_fetchbench_newassets48_20260922.py')
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
read=base.read
SEED=2026092301
def priority(*parts):
    return hashlib.sha256(json.dumps([SEED,*parts]).encode()).hexdigest()

def main():
    assert not SERIES.exists(),SERIES
    prior_dirs=[base.OLD,ROOT/'outputs/fetchbench_v2_newassets48_ab_dro64_20260922']
    previous=sum([read(p/'TASK_SELECTION.json')['tasks'] for p in prior_dirs],[])
    excluded={r['asset_key'] for r in previous}
    invalid=set(read(base.OLD/'INVALID_ASSETS_PRE_OUTCOME.json')['scenes'])
    frame=read(base.OLD/'AUTO_LABELED_TASKS.json')['tasks']
    eligible=[r for r in frame if r['asset_key'] not in excluded and r['scene_instance'] not in invalid]
    def representatives(rows):
        d={}
        for r in sorted(rows,key=lambda r:(priority('task',r['task_id']),r['task_id'])):
            d.setdefault((r['scene_instance'],r['asset_key'],r['environment']),r)
        return list(d.values())
    candidates=representatives(eligible)
    assert [r['task_id'] for r in candidates]==[r['task_id'] for r in representatives([dict(r,ignored_outcome=1) for r in reversed(eligible)])]
    n=len(candidates);groups=[]
    for asset in sorted({r['asset_key'] for r in candidates}):
        groups.append(([b*n+i for b in range(4) for i,r in enumerate(candidates) if r['asset_key']==asset],0,1))
    for b in range(4):
        for env in base.ENV:groups.append(([b*n+i for i,r in enumerate(candidates) if r['environment']==env],12,12))
        for scene in sorted({r['scene_instance'] for r in candidates}):groups.append(([b*n+i for i,r in enumerate(candidates) if r['scene_instance']==scene],0,2))
    targets=[]
    for b in range(4):
        for g in base.GEO:targets.append(([b*n+i for i,r in enumerate(candidates) if r['geometry']==g],16,1000))
        for g in base.GEO:
            for e in base.ENV:targets.append(([b*n+i for i,r in enumerate(candidates) if r['geometry']==g and r['environment']==e],4,1))
    k=len(targets);a=lil_matrix((len(groups)+k,4*n+2*k));lo=[];hi=[]
    for j,(indices,l,u) in enumerate(groups):a[j,indices]=1;lo.append(l);hi.append(u)
    cost=np.zeros(4*n+2*k)
    for b in range(4):
        cost[b*n:(b+1)*n]=[int(priority('tie',b,r['task_id'])[:12],16)/16**12*1e-6 for r in candidates]
    for j,(indices,target,weight) in enumerate(targets):
        a[len(groups)+j,indices]=1;a[len(groups)+j,4*n+2*j]=-1;a[len(groups)+j,4*n+2*j+1]=1
        lo.append(target);hi.append(target);cost[4*n+2*j:4*n+2*j+2]=weight
    result=milp(cost,integrality=np.r_[np.ones(4*n),np.zeros(2*k)],bounds=Bounds(np.zeros(4*n+2*k),np.r_[np.ones(4*n),np.full(2*k,48)]),constraints=LinearConstraint(a.tocsc(),lo,hi),options={'time_limit':300,'mip_rel_gap':0})
    assert result.success,result.message
    cohorts=[[r for i,r in enumerate(candidates) if result.x[b*n+i]>.5] for b in range(4)]
    assert len({r['asset_key'] for batch in cohorts for r in batch})==192
    SERIES.mkdir(parents=True)
    allocation=dict(seed=SEED,objective=float(result.fun),outcome_blind=True,input_order_and_outcome_invariance_passed=True,prior_selection_hashes={str(p/'TASK_SELECTION.json'):base.sha(p/'TASK_SELECTION.json') for p in prior_dirs},cohorts=cohorts)
    base.save(SERIES/'JOINT_ALLOCATION.json',allocation)
    entries=[]
    for b,selected in enumerate(cohorts,1):
        assert len(selected)==48 and collections.Counter(r['environment'] for r in selected)==collections.Counter({e:12 for e in base.ENV})
        assert max(collections.Counter(r['scene_instance'] for r in selected).values())<=2
        name=f'fetchbench_v2_four48_batch{b}_ab_dro64_20260923'
        base.NAME=name;base.OUT=ROOT/'outputs'/name;base.REMOTE=base.P/'ContactDiffusion/outputs'/name;base.SEED=SEED+b
        def adjusted_read(p):return {'tasks':previous} if Path(p)==base.OLD/'TASK_SELECTION.json' else read(p)
        def selected_subset(rows):
            assert {r['task_id'] for r in selected}<={r['task_id'] for r in rows}
            return selected,float(result.fun)
        base.read=adjusted_read;base.choose=selected_subset;base.main()
        audit=read(base.OUT/'EXCLUSION_AUDIT.json');audit['prior_selection_hashes']=allocation['prior_selection_hashes'];audit['joint_allocation_sha256']=base.sha(SERIES/'JOINT_ALLOCATION.json');audit['selection_method']='Joint four-cohort MILP; input representative ordering invariant to outcome fields';base.save(base.OUT/'EXCLUSION_AUDIT.json',audit)
        protocol=read(base.OUT/'protocol_v2.json');protocol.update(task_seed=SEED,cohort_index=b,task_sampling='Joint four-cohort allocation:192 unique new assets,12 tasks/environment/cohort,max2 tasks/scene/cohort; geometry balance by availability; no outcome selection',historical_exclusions='All canonical assets from both prior48 cohorts and other cohorts in this series',joint_allocation_sha256=base.sha(SERIES/'JOINT_ALLOCATION.json'));base.save(base.OUT/'protocol_v2.json',protocol)
        manifest=read(base.OUT/'manifest.json');manifest['protocol_v2']=protocol;base.save(base.OUT/'manifest.json',manifest)
        selection=read(base.OUT/'TASK_SELECTION.json');selection['seed']=SEED;selection['joint_allocation_sha256']=base.sha(SERIES/'JOINT_ALLOCATION.json');base.save(base.OUT/'TASK_SELECTION.json',selection)
        review=(base.OUT/'REVIEW.md').read_text().replace('第二轮48个新目标',f'四批新实验第{b}批48目标').replace('目标资产与第一轮交集为0','目标资产与前两轮及本系列其他批次交集为0');(base.OUT/'REVIEW.md').write_text(review)
        lock=read(base.OUT/'DESIGN_LOCK.json');lock['sha256']={p.name:base.sha(p) for p in base.OUT.iterdir() if p.is_file() and p.name!='DESIGN_LOCK.json'};lock['preparer_sha256']=base.sha(__file__);base.save(base.OUT/'DESIGN_LOCK.json',lock)
        entries.append(dict(index=b,name=name,local=str(base.OUT),remote=str(base.REMOTE),design_lock_sha256=base.sha(base.OUT/'DESIGN_LOCK.json')))
        previous=previous+selected
    base.save(SERIES/'SERIES.json',dict(cohorts=entries,allocation_sha256=base.sha(SERIES/'JOINT_ALLOCATION.json'),order='complete A/B and DRO; audit and publish execution statistics; then next cohort',no_outcome_based_replacement=True))
    print('PREPARED',str(SERIES),flush=True)

if __name__=='__main__':main()
