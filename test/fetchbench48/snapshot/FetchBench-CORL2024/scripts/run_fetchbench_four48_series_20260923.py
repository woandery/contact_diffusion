"""Sequential four-cohort pipeline. No polling service; each child is awaited."""
import argparse
import collections
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

P=Path('/inspire/qb-ilm2/project/zhanghanbo/public/mck')
F=P/'FetchBench-CORL2024';C=P/'ContactDiffusion'
ROOT=C/'outputs/fetchbench_four48_series_20260923'
PY=P/'miniconda3/envs/contactdiff/bin/python'
ENTRY=F/'scripts/run_fetchbench_four48_cohort_20260923.py'
def read(p):return json.loads(Path(p).read_text())
def save(p,data):
    p=Path(p);temp=p.with_suffix(p.suffix+'.tmp');temp.write_text(json.dumps(data,indent=2,ensure_ascii=False)+'\n');temp.replace(p)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def env_for(name):
    e=os.environ.copy();e.update(FETCHBENCH_SERIES_COHORT=name,CUDA_VISIBLE_DEVICES='0,1,2,3',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONUNBUFFERED='1');return e
def execute(cohort,preflight):
    out=Path(cohort['remote']);assert sha(out/'DESIGN_LOCK.json')==cohort['design_lock_sha256']
    with (out/('series_preflight.log' if preflight else 'driver.log')).open('a') as log:
        subprocess.run([str(PY),'-u',str(ENTRY),*(['--preflight-only'] if preflight else [])],env=env_for(cohort['name']),cwd=F,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,check=True)
def audit_report(cohort):
    out=Path(cohort['remote']);state=read(out/'pipeline_status.json');assert state['stage']=='finished',state
    stats=read(out/'EXECUTION_STATISTICS.json');inp=read(out/'comparison_inputs.json')
    assert inp['all_dro_complete'],'DRO incomplete; do not advance'
    assert len(list((out/'capture_progress').glob('*.json')))==48
    failures=collections.Counter();unexpected=[]
    for error in stats['errors']:
        if error.get('stage')!='fk_generate':unexpected.append(error);continue
        j=error['job'];dest=out/'fk'/error['case']/'cases'/j[0]/j[1]/j[2]/j[3]
        text='\n'.join(f.read_text(errors='replace')[-12000:] for f in dest.glob('*.log'))
        if 'ContactBudgetExhausted' in text:failures['contact_budget_exhausted']+=1
        elif 'No visible scene points remain after local crop' in text:failures['empty_local_scene_crop']+=1
        else:unexpected.append(dict(error=error,log_tail=text[-1000:]))
    if unexpected:
        save(out/'SERIES_UNEXPECTED_ERRORS.json',dict(errors=unexpected));raise RuntimeError('Unexpected infrastructure/numerical errors: stop before next cohort')
    cov=read(out/'coverage.json')['rows']
    result=dict(cohort=cohort['index'],name=cohort['name'],totals=stats['totals'],failures=dict(failures),views=sum(r['legal_views'] for r in cov),reconstructed_views=sum(r['reconstructed_views'] for r in cov),dro={k:sum(r[k] for r in inp['dro']) for k in ('planned','generated','retained','executed')},raw_complete=state.get('complete'),execution_report_sha256=sha(out/'EXECUTION_REPORT.md'),statistics_sha256=sha(out/'EXECUTION_STATISTICS.json'),reported_at=time.time(),note='Fixed-budget unavailable sets and empty scene crops are disclosed, not silently counted as executed failures; no extra draws.')
    lines=['# 四批顺序实验：第%d批结果'%cohort['index'],'','最终有限净抬升≥10cm；A/B为冻结排序Top1，DRO分母为筛选后实际执行。','','|方法|成功|实际执行|成功率|','|---|---:|---:|---:|']
    for m,r in stats['totals'].items():lines.append(f"|{m}|{r['success']}|{r['executed']}|{r['rate']:.2%}|" if r['rate'] is not None else f'|{m}|0|0|NA|')
    lines+=['','## 环境分组','','|环境|方法|成功|实际执行|成功率|','|---|---|---:|---:|---:|']
    for environment in sorted({r['environment'] for r in stats['rows']}):
        for method in ('A','B','DRO'):
            rr=[r for r in stats['rows'] if r['environment']==environment and r['method']==method];s=sum(r['success'] for r in rr);n=sum(r['executed'] for r in rr)
            rate=f'{s/n:.2%}' if n else 'NA';lines.append(f'|{environment}|{method}|{s}|{n}|{rate}|')
    lines+=['',f'终止性不可用批次：{dict(failures)}。',f"DRO生成/保留/执行：{result['dro']}。",'上述执行成功率不是统一候选分母的端到端成功率；不合并样本后声称独立或显著。',f"原始complete={state.get('complete')}；终止性不可用项单独披露，未扩大采样预算。"]
    (out/'SERIES_COHORT_REPORT.md').write_text('\n'.join(lines)+'\n');save(out/'SERIES_COHORT_AUDIT.json',result)
    return result
def publish(results):
    save(ROOT/'SERIES_RESULTS.json',dict(cohorts=results,finished=len(results)==4))
    lines=['# 四批48任务顺序实验汇总','','每批先完成执行、异常分类和报告落盘，才推进下一批。','','|批次|方法|成功/执行|成功率|','|---|---|---:|---:|']
    for row in results:
        for m,r in row['totals'].items():
            rate=f"{r['rate']:.2%}" if r['rate'] is not None else 'NA';lines.append(f"|{row['cohort']}|{m}|{r['success']}/{r['executed']}|{rate}|")
    (ROOT/'SERIES_REPORT.md').write_text('\n'.join(lines)+'\n')
def main(preflight):
    lock=(ROOT/'series.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    spec=read(ROOT/'SERIES.json');assert sha(ROOT/'JOINT_ALLOCATION.json')==spec['allocation_sha256']
    import torch
    assert torch.cuda.device_count()==4,'Requires four visible GPUs'
    cohorts=spec['cohorts']
    for cohort in cohorts:
        save(ROOT/'status.json',dict(stage='preflight',cohort=cohort['index'],pid=os.getpid(),time=time.time()));execute(cohort,True)
    if preflight:save(ROOT/'status.json',dict(stage='preflight_passed',time=time.time()));return
    results=[]
    for cohort in cohorts:
        out=Path(cohort['remote']);audit=out/'SERIES_COHORT_AUDIT.json'
        if audit.exists():
            prior=read(audit);assert prior['statistics_sha256']==sha(out/'EXECUTION_STATISTICS.json');results.append(prior);publish(results);continue
        save(ROOT/'status.json',dict(stage='running_cohort',cohort=cohort['index'],pid=os.getpid(),completed_cohorts=len(results),time=time.time()))
        execute(cohort,False)
        results.append(audit_report(cohort));publish(results)
    save(ROOT/'status.json',dict(stage='finished',completed_cohorts=4,time=time.time()))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--preflight-only',action='store_true');a=p.parse_args()
    try:main(a.preflight_only)
    except Exception as ex:
        save(ROOT/'status.json',dict(stage='blocked',error=repr(ex),time=time.time()));raise
