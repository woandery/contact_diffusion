"""Second cohort: existing algorithms, isolated outputs, execution-denominator reporting."""
import argparse
import json
import math
from pathlib import Path
import sys
import time
import run_fetchbench_autolabel_pilot_v2 as core

NAME='fetchbench_v2_newassets48_ab_dro64_20260922'
core.OUT=core.C/'outputs'/NAME
core.__file__=__file__
OUT=core.OUT
sys.path.insert(0,str(OUT/'ranker_pydeps'))

def verify_prepared():
    lock=core.read(OUT/'DESIGN_LOCK.json')
    for name,digest in lock['sha256'].items():
        assert core.sha(OUT/name)==digest,('Design artifact changed',name)
    manifest=core.read(OUT/'manifest.json')
    assert manifest['run_root']==str(OUT)
    assert len(manifest['cases'])==48
    assert len({c['asset_key'] for c in manifest['cases']})==48
    excluded=set(core.read(OUT/'EXCLUSION_AUDIT.json')['excluded_assets'])
    assert not excluded.intersection(c['asset_key'] for c in manifest['cases'])
    missing=[]
    for rel,meta in core.read(OUT/'ASSET_FILE_INVENTORY.json')['files'].items():
        p=core.F/rel
        if not p.is_file() or p.stat().st_size!=meta['size']:missing.append(rel)
    if missing:
        core.save(OUT/'REMOTE_ASSET_GAPS.json',dict(missing=missing))
        raise RuntimeError(f'{len(missing)} missing/wrong-size asset files; see REMOTE_ASSET_GAPS.json. No generation started.')

def report(m,errors):
    inp=core.read(OUT/'comparison_inputs.json');rows=[]
    for case in m['cases']:
        p=OUT/'top1/fk'/case['id']/'unified_top1.json'
        selected=core.read(p)['selections'] if p.exists() else []
        for method in ('A','B','DRO'):
            for hand in ('barrett','shadowhand'):
                if method=='DRO':
                    group=[r for r in inp['dro'] if r['case']==case['id'] and r['hand']==hand]
                    executed=sum(r['executed'] for r in group);success=0
                    for r in group:
                        if r.get('source_summary'):
                            for t in core.read(r['source_summary'])['trials']:
                                lift=float(t['final_object_lift_m']);success+=math.isfinite(lift) and lift>=.1
                else:
                    group=[r for r in selected if r['condition']==method and r['hand']==hand]
                    executed=len(group);success=sum(bool(r['height']) for r in group)
                rows.append(dict(case=case['id'],scene=case['scene_instance'],geometry=case['preliminary_geometry'],environment=case['environment'],method=method,hand=hand,success=success,executed=executed,rate=success/executed if executed else None))
    totals={}
    for method in ('A','B','DRO'):
        rr=[r for r in rows if r['method']==method];s=sum(r['success'] for r in rr);n=sum(r['executed'] for r in rr)
        totals[method]=dict(success=s,executed=n,rate=s/n if n else None)
    core.save(OUT/'EXECUTION_STATISTICS.json',dict(rows=rows,totals=totals,errors=errors,confirmatory=False,geometry_review_pending=True,dro_complete=inp.get('all_dro_complete'),note='No legacy16-cell inference; audit errors before finalizing'))
    lines=['# 新目标48任务：执行成功率','','最终有限净抬升≥10cm；A/B固定Top1；DRO仅成功/实际执行，筛除项不进分母。错误与缺失须另行核查，不将阶段性数据当作最终结果。','','|方法|成功|实际执行|成功率|','|---|---:|---:|---:|']
    for method,r in totals.items():
        rate=f"{r['rate']:.2%}" if r['rate'] is not None else 'NA'
        lines.append(f"|{method}|{r['success']}|{r['executed']}|{rate}|")
    lines+=['',f'阶段错误数：{len(errors)}。', '几何与环境分层不平衡且存在混杂；不与第一轮直接混合计数后声称显著性。详细分组数据见EXECUTION_STATISTICS.json。']
    (OUT/'EXECUTION_REPORT.md').write_text('\n'.join(lines)+'\n')

core.pilot_report=report
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--preflight-only',action='store_true');p.add_argument('--feature-case');p.add_argument('--condition');p.add_argument('--hand');args=p.parse_args()
    if args.feature_case:
        worker=core.load('newassets_features',core.HERE/'recover_six_scene_existing_top1.py')
        worker.feature_worker(OUT/'top1',args.feature_case,args.condition,args.hand)
    else:
        try:verify_prepared();core.main(args.preflight_only)
        except Exception as ex:
            core.save(OUT/'pipeline_status.json',dict(stage='failed',error=repr(ex),time=time.time()))
            raise
