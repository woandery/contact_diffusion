"""User-authorized continuation of cohorts 3 and 4, leaving cohort 2 unresolved."""
import fcntl
import json
import os
from pathlib import Path
import sys
import time

P=Path('/inspire/qb-ilm2/project/zhanghanbo/public/mck')
sys.path.insert(0,str(P/'FetchBench-CORL2024/scripts'))
import run_fetchbench_four48_series_20260923 as run

def main():
    root=run.ROOT
    lock=(root/'series.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    spec=run.read(root/'SERIES.json');assert run.sha(root/'JOINT_ALLOCATION.json')==spec['allocation_sha256']
    cohorts=[c for c in spec['cohorts'] if c['index'] in (3,4)]
    assert [c['index'] for c in cohorts]==[3,4]
    import torch
    assert torch.cuda.device_count()==4
    prior=[]
    for c in spec['cohorts'][:2]:
        out=Path(c['remote']);stats=run.read(out/'EXECUTION_STATISTICS.json')
        prior.append(dict(cohort=c['index'],totals=stats['totals'],statistics_sha256=run.sha(out/'EXECUTION_STATISTICS.json'),audit_passed=(out/'SERIES_COHORT_AUDIT.json').exists(),unexpected_errors_file=str(out/'SERIES_UNEXPECTED_ERRORS.json') if (out/'SERIES_UNEXPECTED_ERRORS.json').exists() else None))
    authorization=root/'RESUME34_AUTHORIZATION.json'
    if not authorization.exists():
        run.save(authorization,dict(time=time.time(),user_instruction='继续启动3,4批结果',previous_status=run.read(root/'status.json'),prior_cohorts=prior,note='Cohort2 exception remains unresolved; no changes to algorithm, tolerance, hashes, candidate budgets, or cohorts1/2. Future unknown errors still stop continuation.'))
    for c in cohorts:
        run.save(root/'status.json',dict(stage='preflight_resume34',cohort=c['index'],pid=os.getpid(),time=time.time(),cohort2_exception_unresolved=True))
        run.execute(c,True)
    results=[]
    for c in cohorts:
        run.save(root/'status.json',dict(stage='running_cohort',cohort=c['index'],pid=os.getpid(),time=time.time(),resume_scope=[3,4],cohort2_exception_unresolved=True))
        run.execute(c,False)
        result=run.audit_report(c);results.append(result)
        run.save(root/'RESUME34_RESULTS.json',dict(prior_cohorts=prior,continued_cohorts=results,finished=len(results)==2,cohort2_exception_unresolved=True))
        lines=['# 第3、4批继续执行结果','','第2批初始化能量断言异常保留，未解决或静默改记为通过。第1、2批结果未修改。','','|批次|方法|成功/实际执行|成功率|','|---|---|---:|---:|']
        for r in [*prior,*results]:
            for method,t in r['totals'].items():
                rate=f"{t['rate']:.2%}" if t['rate'] is not None else 'NA'
                lines.append(f"|{r['cohort']}|{method}|{t['success']}/{t['executed']}|{rate}|")
        (root/'RESUME34_REPORT.md').write_text('\n'.join(lines)+'\n')
    run.save(root/'status.json',dict(stage='finished_resume34',completed_continuation=[3,4],cohort2_exception_unresolved=True,time=time.time()))

if __name__=='__main__':
    try:main()
    except Exception as ex:
        run.save(run.ROOT/'status_resume34_error.json',dict(stage='blocked_resume34',error=repr(ex),time=time.time()))
        raise
