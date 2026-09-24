"""Outcome-free sampling utilities for the frozen confirmatory design; no launcher."""
import hashlib
import json
import math
from pathlib import Path
from collections import Counter

ROOT=Path(__file__).resolve().parents[1]
CONFIG=ROOT/'configs/fetchbench_confirmatory_v1_20260920.json'
def protocol():return json.loads(CONFIG.read_text())
def priority(seed,*parts):
    return hashlib.sha256(json.dumps([seed,*parts],ensure_ascii=True,separators=(',',':')).encode()).hexdigest()

def sample_tasks(eligible_rows):
    """Input must already be blind-annotated and eligibility-locked. No outcome keys used."""
    p=protocol();s=p['task_sampling'];classes=p['geometry']['classes'];envs=p['environment']['classes']
    fields=['task_id','scene_instance','asset_id','geometry','environment']
    rows=[{k:r[k] for k in fields} for r in eligible_rows]
    assert len({r['task_id'] for r in rows})==len(rows)
    assert all(r['geometry'] in classes and r['environment'] in envs for r in rows)
    for attempt in range(s['maximum_attempts']):
        counts=Counter();scenes=set();assets=set();selected=[]
        for r in sorted(rows,key=lambda r:(priority(s['seed'],'task',attempt,r['task_id']),r['task_id'])):
            cell=(r['geometry'],r['environment'])
            if counts[cell]>=3 or r['scene_instance'] in scenes or r['asset_id'] in assets:continue
            selected.append(r);counts[cell]+=1;scenes.add(r['scene_instance']);assets.add(r['asset_id'])
        if len(selected)==48:
            assert all(counts[(g,e)]==3 for g in classes for e in envs)
            return dict(attempt=attempt,tasks=selected)
    raise RuntimeError('No complete balanced sample found in 256 fixed attempts; stop, do not relax or substitute based on outcomes')

def select_views(task_id,all_views):
    p=protocol()['views'];seed=p['seed'];bins={k:[] for k in p['quotas']};legal=[]
    seen=set()
    for r in all_views:
        key=r['view_id'];assert key not in seen;seen.add(key)
        if r['fraction_observable']<.05 or r['partial_raw_point_count']<=0:continue
        f=r['fraction_full'];assert math.isfinite(f) and 0<=f<=1
        k='low' if f<.15 else 'medium' if f<.35 else 'high'
        row={n:r[n] for n in ('view_id','original_legal_index','fraction_full','fraction_observable','partial_raw_point_count')}
        row['bin']=k;bins[k].append(row);legal.append(row)
    if not legal:return dict(task_id=task_id,observation_unavailable=True,views=[],planned_null_slots=1)
    selected=[];take={}
    for k,rs in bins.items():
        take[k]=min(p['quotas'][k],len(rs))
        selected+=sorted(rs,key=lambda r:(priority(seed,'bin',task_id,k,r['view_id']),r['view_id']))[:take[k]]
    used={r['view_id'] for r in selected};remaining=[r for r in legal if r['view_id'] not in used]
    extra=min(8,len(legal))-len(selected)
    selected+=sorted(remaining,key=lambda r:(priority(seed,'fill',task_id,r['view_id']),r['view_id']))[:extra]
    # Inclusion probability under uniform pseudorandom priorities, conditional on bin counts.
    for r in selected:
        q=take[r['bin']]/len(bins[r['bin']]);r['inclusion_probability']=q+(1-q)*(extra/len(remaining) if remaining else 0)
    return dict(task_id=task_id,observation_unavailable=False,views=sorted(selected,key=lambda r:r['original_legal_index']),bin_counts={k:len(v) for k,v in bins.items()},fill_count=extra,planned_null_slots=0)

def task_score(method,views):
    """Each view: {'barrett': {'height_successes': n, 'infrastructure_pending': False}, ...}."""
    denom=64 if method=='DRO' else 4
    if method not in ('A','B','DRO'):raise ValueError(method)
    if not views:return 0.0
    sums=[]
    for v in views:
        for hand in ('barrett','shadowhand'):
            r=v[hand]
            if r.get('infrastructure_pending'):raise ValueError('Incomplete infrastructure data cannot be scored as failure')
            n=r['height_successes'];assert isinstance(n,int) and 0<=n<=denom;sums.append(n/denom)
    return sum(sums)/len(sums)

def stratified_difference(rows,method):
    """One immutable row/task: geometry, environment, scores{A,B,DRO}; approximate design-based t inference."""
    import numpy as np
    from scipy.stats import t
    p=protocol();cells=[(g,e) for g in p['geometry']['classes'] for e in p['environment']['classes']]
    assert len(rows)==48 and len({r['task_id'] for r in rows})==48
    means=[];v=[]
    for g,e in cells:
        d=np.array([r['scores'][method]-r['scores']['DRO'] for r in rows if (r['geometry'],r['environment'])==(g,e)])
        assert len(d)==3 and np.isfinite(d).all() and (abs(d)<=1).all()
        means.append(d.mean());v.append(d.var(ddof=1)/3/16**2)
    delta=float(np.mean(means));variance=float(sum(v))
    if variance<=0:raise ValueError('Degenerate variance: report descriptively; no automatic significance declaration')
    se=math.sqrt(variance);df=variance**2/sum(x*x/2 for x in v)
    return dict(delta=delta,se=se,df=df,p_raw=float(2*t.sf(abs(delta/se),df)),ci95=[delta-float(t.ppf(.975,df))*se,delta+float(t.ppf(.975,df))*se],ci975_simultaneous=[delta-float(t.ppf(.9875,df))*se,delta+float(t.ppf(.9875,df))*se])

def holm_two(results):
    assert set(results)=={'A','B'}
    ordered=sorted(results,key=lambda k:results[k]['p_raw']);previous=0.;out={}
    for i,key in enumerate(ordered):
        adjusted=min(1.,max(previous,(2-i)*results[key]['p_raw']));previous=adjusted
        out[key]=dict(results[key],p_holm=adjusted,superior=adjusted<.05 and results[key]['delta']>0)
    return out
