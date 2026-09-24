"""Separate missing work from filtered candidates; budgets are 4/4/64 per hand/view."""
import argparse,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args()
m=json.loads((a.run/'manifest.json').read_text());rows=[]
for c in m['cases']:
 views=json.loads((Path(c['observations'])/'visibility/legal_partial_views.json').read_text())['views']
 for method in ['A','B','DRO']:
  for hand in ['barrett','shadowhand']:
   budget=64 if method=='DRO' else 4
   row=dict(case=c['id'],method=method,hand=hand,planned=len(views)*budget,executed=0,filtered=0,strict=0,height=0,missing=0)
   for v in views:
    vid=Path(v['rgbd_capture_dir']).name
    if method=='DRO':
     d=a.run/'generation'/c['id']/'DRO_cpu_physx_all64/views'/vid/'simulation'/c['scene_factory']/f"task_{c['task_index']:03d}"/hand
     path=d/'dro_candidates_environment_filtered.json'
     if path.exists():
      f=json.loads(path.read_text())['environment_filter'];assert f['generated_candidates']==64
      row['filtered']+=64-f['retained_candidates']
    else:d=a.run/'generation'/c['id']/'AB/cpu_physx_all4/simulation'/vid/method/hand
    trials=list(d.glob('**/lift_validation/candidate_*.json'));assert len(trials)<=budget
    for f in trials:
     t=json.loads(f.read_text());row['executed']+=1;row['strict']+=bool(t['success']);row['height']+=t['final_object_lift_m']>=.10
   row['missing']=row['planned']-row['filtered']-row['executed'];assert row['missing']>=0
   rows.append(row)
complete=all(r['missing']==0 for r in rows)
payload=dict(complete=complete,rows=rows,criterion='final_object_lift_m >= 0.10',note='Missing work is NOT a completed failure; DRO environment rejection is an end-to-end failure.')
(a.run/'RESULTS_AB_DRO64.json').write_text(json.dumps(payload,indent=2))
lines=['# 三场景 A/B 4×4 与 DRO64 比较','',f'完成：{complete}。CPU PhysX；最终抬升≥10 cm。缺失执行不冒充算法失败，未完成时仅报告成功计数；DRO环境筛除计端到端失败。','',
 '|场景|方法|手|计划|执行|环境筛除|缺失|仅高度成功|严格成功|高度端到端率|','|---|---|---|---|---|---|---|---|---|---|']
for r in rows:
 rate=f"{r['height']/r['planned']:.2%}" if r['missing']==0 else '未完成'
 lines.append('|'+ '|'.join(str(r[k]) for k in ['case','method','hand','planned','executed','filtered','missing','height','strict'])+'|'+rate+'|')
lines+=['','A/B每视角每手4个接触集×4粒子，每集保留1姿态；DRO每视角每手64候选。不得直接比较成功数量，候选成功率也不代表预算匹配的生成器消融。']
(a.run/'RESULTS_AB_DRO64.md').write_text('\n'.join(lines)+'\n')
print(json.dumps(dict(complete=complete,planned=sum(r['planned'] for r in rows),executed=sum(r['executed'] for r in rows))))
