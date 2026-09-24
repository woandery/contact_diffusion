"""Prepare a new-asset cohort without reading grasp outcomes or starting experiments."""
import copy
import hashlib
import json
from collections import Counter
from pathlib import Path
import shutil
import numpy as np
import yaml
from scipy.optimize import milp, Bounds, LinearConstraint
from scipy.sparse import lil_matrix

R = Path(__file__).resolve().parents[1]
P = Path('/inspire/qb-ilm2/project/zhanghanbo/public/mck')
NAME = 'fetchbench_v2_newassets48_ab_dro64_20260922'
OUT = R/'outputs'/NAME
REMOTE = P/'ContactDiffusion/outputs'/NAME
OLD = R/'outputs/fetchbench_v2_autolabel_full_ab_dro64_20260920'
SEED = 2026092201
ENV = ('table', 'shelf_or_cabinet', 'drawer', 'basket')
GEO = ('block', 'thin_or_slender', 'complex')
def read(p): return json.loads(Path(p).read_text())
def save(p,x): Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def priority(*parts): return hashlib.sha256(json.dumps([SEED,*parts]).encode()).hexdigest()

def choose(rows):
    reps = {}
    for r in sorted(rows,key=lambda r:(priority('task',r['task_id']),r['task_id'])):
        reps.setdefault((r['scene_instance'],r['asset_key'],r['environment']),r)
    candidates = list(reps.values()); n=len(candidates)
    groups=[('environment',e,12,12) for e in ENV]
    groups += [('asset_key',a,0,1) for a in sorted({r['asset_key'] for r in candidates})]
    groups += [('scene_instance',s,0,2) for s in sorted({r['scene_instance'] for r in candidates})]
    # First minimize deviation from 16 per available geometry, then from 4 per cell.
    # Maximum cell L1 is <=96, so weight100 makes the first objective dominant.
    targets=[('geometry',g,None,16,100) for g in GEO]
    targets += [('cell',g,e,4,1) for g in GEO for e in ENV]
    k=len(targets);a=lil_matrix((len(groups)+k,n+2*k));lo=[];hi=[]
    for j,(field,value,l,u) in enumerate(groups):
        for i,r in enumerate(candidates):
            if r[field]==value:a[j,i]=1
        lo.append(l);hi.append(u)
    cost=np.zeros(n+2*k)
    cost[:n]=[int(priority('tie',r['task_id'])[:12],16)/16**12*1e-6 for r in candidates]
    for j,(field,g,e,target,w) in enumerate(targets):
        for i,r in enumerate(candidates):
            if r['geometry']==g and (e is None or r['environment']==e):a[len(groups)+j,i]=1
        a[len(groups)+j,n+2*j]=-1;a[len(groups)+j,n+2*j+1]=1
        lo.append(target);hi.append(target);cost[n+2*j:n+2*j+2]=w
    result=milp(cost,integrality=np.r_[np.ones(n),np.zeros(2*k)],bounds=Bounds(np.zeros(n+2*k),np.r_[np.ones(n),np.full(2*k,48)]),constraints=LinearConstraint(a.tocsc(),lo,hi),options={'time_limit':180,'mip_rel_gap':0})
    assert result.success,result.message
    selected=[r for r,x in zip(candidates,result.x[:n]) if x>.5]
    assert len(selected)==len({r['asset_key'] for r in selected})==48
    assert max(Counter(r['scene_instance'] for r in selected).values())<=2
    assert Counter(r['environment'] for r in selected)==Counter({e:12 for e in ENV})
    return selected,float(result.fun)

def main():
    assert not OUT.exists(),f'Refusing to overwrite {OUT}'
    previous=read(OLD/'TASK_SELECTION.json')['tasks'];excluded={r['asset_key'] for r in previous}
    invalid=set(read(OLD/'INVALID_ASSETS_PRE_OUTCOME.json')['scenes'])
    frame=read(OLD/'AUTO_LABELED_TASKS.json')['tasks']
    rows=[r for r in frame if r['asset_key'] not in excluded and r['scene_instance'] not in invalid]
    assert all(r['geometry']!='axisymmetric' for r in rows)
    selected,obj=choose(rows)
    assert not excluded.intersection(r['asset_key'] for r in selected)
    # Outcome fields and input ordering must not influence selection.
    check,_=choose([dict(r,ignored_success=1) for r in reversed(rows)])
    assert [r['task_id'] for r in check]==[r['task_id'] for r in selected]
    protocol=copy.deepcopy(read(OLD/'protocol_v2.json'))
    protocol.update(protocol_id=NAME,parent='fetchbench-v2-autolabel-full48-ab-dro64-20260920',status='prepared_not_started',
        task_sampling='Exclude all first-cohort canonical target assets; allow scene reuse; env12 each; unique assets; max2 tasks/scene. Minimize geometry L1 from16 across three available classes, then cell L1 from4, then SHA seeded tie-cost.',
        tasks_per_geometry_environment_cell='Availability-adapted, empty cells allowed; not a four-geometry balanced cohort',
        task_seed=SEED,view_seed=2026092002,
        historical_exclusions='Original held-out frame exclusions plus all48 canonical target assets of first cohort; invalid scene assets unchanged',
        statistics={'primary':'SR10_exec: success/actual execution; A/B fixed Top1 only; DRO environment rejects excluded',
                    'secondary':'generation availability and filtered counts; descriptive task/hand equal-weight execution rates',
                    'cluster':'scene_instance across both cohorts if combining; no independent-view significance tests',
                    'interpretation':'Exploratory; auto labels pending review; no axisymmetric targets in this cohort; no legacy16-cell inference'},
        stopping='Fixed48 tasks; no significance-based stopping or outcome-based replacement',
        amendments_authorized_before_outcomes=['User selected new target assets, scene reuse, env12 each, geometry quotas by availability on20260922'])
    aliases={s.replace('benchmark_eval/',''):p.stem for p in (R/'InfiniGym/isaacgymenvs/config/scene/benchmark_eval').glob('*.yaml') for s in yaml.safe_load(p.read_text()).get('scene_list',[])}
    m=copy.deepcopy(read(OLD/'manifest.json'));m['cases']=[];deps=set()
    for i,r in enumerate(selected):
        scene=r['scene_instance'];rel=Path('Task/benchmark_eval')/scene;assets=read(R/rel/'asset_config.json')
        cid=f"t{i+1:02d}_{r['category'].lower()}_{r['environment']}"
        case=dict(id=cid,title=f"{scene} / {r['category']} / {r['geometry']} (待复核)",scene=aliases[scene],scene_factory=scene.split('/')[1],task_config=str(P/'FetchBench-CORL2024'/rel/'task_config.npz'),observations=str(REMOTE/'observations'/cid),candidate_views=111,task_config_sha256=sha(R/rel/'task_config.npz'),asset_config_sha256=sha(R/rel/'asset_config.json'),preliminary_geometry=r['geometry'])
        for key in ('scene_instance','task_index','object_index','category','placement','task_id','asset_id','asset_key','environment','feasibility'):case[key]=r[key]
        m['cases'].append(case);deps.add(str(rel));deps.add('InfiniGym/isaacgymenvs/config/scene/benchmark_eval/'+aliases[scene]+'.yaml')
        for a in [assets['scene_config'],*assets['object_config']]:
            marker='benchmark_scenes/' if 'benchmark_scenes/' in a['asset_root'] else 'benchmark_objects/'
            dep=marker+a['asset_root'].split(marker)[1]
            assert (R/dep/a['urdf_file']).is_file(),dep
            deps.add(dep)
    m.update(run_root=str(REMOTE),protocol_v2=protocol)
    m['experiment_protocol'].update(protocol_id=NAME,confirmatory=False,user_review_pending=True)
    OUT.mkdir(parents=True)
    for name in ('ranker_unified_frozen.joblib','ranker_original_reference.joblib','ranker_freeze.json','AUTO_LABELS.json'):shutil.copy2(OLD/name,OUT/name)
    import scipy
    save(OUT/'TASK_SELECTION.json',dict(tasks=selected,seed=SEED,objective=obj,scipy_version=scipy.__version__,outcome_blind=True,user_review_pending=True))
    save(OUT/'protocol_v2.json',protocol);save(OUT/'manifest.json',m)
    save(OUT/'EXCLUSION_AUDIT.json',dict(prior_selection_sha256=sha(OLD/'TASK_SELECTION.json'),frame_sha256=sha(OLD/'AUTO_LABELED_TASKS.json'),excluded_assets=sorted(excluded),excluded_scenes=sorted(invalid),asset_overlap=0,scene_overlap=len({r['scene_instance'] for r in selected}&{r['scene_instance'] for r in previous}),remaining_tasks=len(rows),remaining_assets=len({r['asset_key'] for r in rows}),input_order_and_outcome_invariance_passed=True))
    files={}
    for dep in sorted(deps):
        p=R/dep
        for f in ([p] if p.is_file() else sorted(x for x in p.rglob('*') if x.is_file())):files[str(f.relative_to(R))]={'size':f.stat().st_size}
    save(OUT/'ASSET_FILE_INVENTORY.json',dict(files=files))
    save(OUT/'asset_roots.json',dict(roots=sorted(deps)));(OUT/'asset_roots.txt').write_text('\n'.join(sorted(deps))+'\n')
    counts=Counter((r['geometry'],r['environment']) for r in selected)
    lines=['# 第二轮48个新目标（未启动）','','目标资产与第一轮交集为0；允许场景复用。自动几何标签待复核。','', '|几何|桌面|架/柜|抽屉|篮子|合计|','|---|---:|---:|---:|---:|---:|']
    for g in GEO:lines.append('|'+g+'|'+'|'.join(str(counts[g,e]) for e in ENV)+'|'+str(sum(counts[g,e] for e in ENV))+'|')
    lines+=['','|任务|场景/任务|物体|几何|环境|','|---|---|---|---|---|']
    for c in m['cases']:lines.append(f"|{c['id']}|{c['task_id']}|{c['category']}|{c['preliminary_geometry']}|{c['environment']}|")
    (OUT/'REVIEW.md').write_text('\n'.join(lines)+'\n')
    save(OUT/'DESIGN_LOCK.json',dict(status='prepared_not_started',sha256={p.name:sha(p) for p in OUT.iterdir() if p.is_file()},preparer_sha256=sha(__file__)))
    print(json.dumps(dict(output=str(OUT),scenes=len({r['scene_instance'] for r in selected}),cells={g+'/'+e:counts[g,e] for g in GEO for e in ENV},assets_files=len(files)),indent=2))
if __name__=='__main__':main()
