"""Freeze-rank already validated four-particle sets; never regenerate or simulate.

Remote mode exports read-only source snapshots and computes the unchanged CPU
ranking features. Local mode applies the pre-existing frozen Shampoo ranker.
All new outputs live in an isolated subdirectory of the original experiment.
"""
import argparse
import concurrent.futures
import gzip
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

os.environ.update(OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
HERE = Path(__file__).resolve().parent
NAME = 'fkonly_ranked_six_scene_dro64_20260915'
RECOVERY = 'top1_existing_20260915'
KEYS = [(a, h) for a in ('A', 'B') for h in ('barrett', 'shadowhand')]


def read(p):
    return json.loads(Path(p).read_text())


def save(p, value):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(p)


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def rowkey(r):
    return tuple(r[k] for k in ('view', 'condition', 'hand', 'sample'))


def export_case(source, target, case):
    import numpy as np
    folder = source / 'fk' / case['id']
    manifest_path = folder / 'manifest.json'
    manifest = read(manifest_path)
    hashes = {str(manifest_path): sha(manifest_path)}
    rows, batches = [], []
    for result_path in sorted(folder.glob('cases/J0/*/*/*/validation_result.json')):
        result = read(result_path)
        assert result['complete'] and result['executed'] == 16, result_path
        view, condition, hand = [result[k] for k in ('view', 'condition', 'hand')]
        assert tuple(result_path.parts[-4:-1]) == (view, condition, hand)
        raw = result_path.parent / 'raw_object.json'
        payload = read(raw)
        hashes[str(raw)] = sha(raw)
        hashes[str(result_path)] = sha(result_path)
        mdpath = next(Path(p) for p in manifest['inputs'][view]
                      if p.endswith('sam3d_fused_centered.json'))
        transform = read(mdpath)['robot_from_pointcloud_frame']
        pc = np.load(Path(manifest['case']['observations']) / 'sam3d' / view / 'sam3d_fused_centered.npy')
        assert np.isfinite(pc).all()
        labels = {(t['sample'], t['particle']): t for t in result['trials']}
        assert len(labels) == 16
        assert sorted(x['sample_index'] for x in payload['records']) == [0, 1, 2, 3]
        for record in payload['records']:
            fk = record['fk']
            candidates = sorted(fk['candidates'], key=lambda c: c['particle'])
            assert [c['particle'] for c in candidates] == [0, 1, 2, 3]
            ts = [labels[(record['sample_index'], c['particle'])] for c in candidates]
            for c, t in zip(candidates, ts):
                assert abs(c['max_finger_contact_error_m'] - t['max_contact_error_m']) < 1e-8
                assert bool(t['height']) == (t['lift'] >= .10)
            rows.append(dict(view=view, condition=condition, hand=hand,
                sample=record['sample_index'], candidates=candidates, labels=ts,
                source_file=str(raw), validation_file=str(result_path), robot_from_object=transform,
                object_summary=dict(mean=pc.mean(0).tolist(), lower=pc.min(0).tolist(), upper=pc.max(0).tolist()),
                fk_metadata={k: fk[k] for k in ('joint_names', 'joint_lower', 'joint_upper',
                    'palm_normal_axis', 'grasp_local_approach_axis', 'target_contacts', 'selection_rank_mode')}))
        batches.append((view, condition, hand))
    assert len({rowkey(r) for r in rows}) == len(rows)
    dest = target / 'fk' / case['id']
    dest.mkdir(parents=True, exist_ok=True)
    rows.sort(key=rowkey)
    with gzip.open(dest / 'source_dataset.json.gz', 'wt') as f:
        json.dump(dict(rows=rows, source_sha256=hashes), f)
    scoring = dict(manifest)
    scoring['config_paths'] = {'E3': manifest['config_paths']['J0']}
    save(dest / 'manifest.json', scoring)
    available = set(batches)
    missing = [(v['view'], a, h) for v in manifest['selected_views'] for a, h in KEYS
               if (v['view'], a, h) not in available]
    info = dict(case=case['id'], expected_batches=manifest['expected_batches'],
                available_batches=len(batches), available_sets=len(rows),
                missing_batches=missing, source_sha256=hashes)
    save(dest / 'coverage.json', info)
    return info


def feature_worker(target, case, condition, hand):
    module = load('recovery_closure', HERE / 'score_fk_particle_closure.py')
    module.CACHE.clear()
    module.base.ROOT = target / 'fk' / case
    module.base.SOURCE = module.base.ROOT
    result = module.base.worker((condition, hand))
    print(result, flush=True)


def remote(source):
    target = source / RECOVERY
    target.mkdir(exist_ok=False)
    policy = read(source / 'frozen_policy.json')
    for path, digest in policy['sha256'].items():
        if Path(path).name in ('score_fk_particle_closure.py', 'score_fk_particle_environment.py'):
            assert sha(HERE / Path(path).name) == digest
    for name in ('manifest.json', 'comparison_inputs.json', 'frozen_policy.json'):
        shutil.copy2(source / name, target / name)
    save(target / 'status.json', dict(stage='exporting_existing', started=time.time()))
    coverage = [export_case(source, target, c) for c in read(source / 'manifest.json')['cases']]
    save(target / 'coverage.json', coverage)
    tasks = [(c['case'], a, h) for c in coverage for a, h in KEYS]
    save(target / 'status.json', dict(stage='scoring_fixed_poses', workers=8,
                                    sets=sum(c['available_sets'] for c in coverage), time=time.time()))

    def worker(task):
        case, a, h = task
        log = target / 'fk' / case / f'features_{a}_{h}.log'
        with log.open('w') as f:
            result = subprocess.run([sys.executable, __file__, '--mode', 'feature-worker',
                '--root', str(source), '--case', case, '--condition', a, '--hand', h],
                stdout=f, stderr=subprocess.STDOUT)
        return dict(case=case, condition=a, hand=h, returncode=result.returncode)

    finished = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        for result in pool.map(worker, tasks):
            finished.append(result)
            save(target / 'feature_jobs.json', finished)
    errors = [x for x in finished if x['returncode']]
    # Audit that ranking did not mutate any source candidate or validation label.
    for c in coverage:
        for p, digest in c['source_sha256'].items():
            assert sha(p) == digest, p
    save(target / 'status.json', dict(stage='features_failed' if errors else 'ready_for_local_ranking',
                                    errors=errors, source_hashes_unchanged=True, finished=time.time()))
    if errors:
        raise RuntimeError(errors)


def evaluate(source):
    import joblib
    import numpy as np
    target = source / RECOVERY
    policy = read(target / 'frozen_policy.json')
    for path, digest in policy['sha256'].items():
        assert sha(path) == digest, path
    assert read(target / 'status.json')['stage'] == 'ready_for_local_ranking'
    geom = load('recovery_geometry', HERE / 'probe_fk_particle_ranking_geometry.py')
    model_path = HERE.parent / 'outputs/fk_only_particle_ranking_20260915/geometry_closure_probe/ranker.joblib'
    frozen = joblib.load(model_path)
    inputs = read(target / 'comparison_inputs.json')
    coverage = read(target / 'coverage.json')
    results = {}
    for c in coverage:
        case = c['case']
        folder = target / 'fk' / case
        with gzip.open(folder / 'source_dataset.json.gz', 'rt') as f:
            data = json.load(f)
        rows = sorted(data['rows'], key=rowkey)
        environment = {(v['view'], v['condition'], v['hand'], v['sample'], v['particle']): v['features']
                       for a, h in KEYS for v in read(folder / f'closure_features_{a}_{h}.json')['rows']}
        assert len(environment) == len(rows) * 4
        features = []
        for row in rows:
            # Execution labels and source file identities are absent from ranker input.
            feature_row = {k: v for k, v in row.items() if k not in ('labels', 'source_file', 'validation_file')}
            x, names = geom.features(feature_row, environment)
            assert names == frozen['features']
            features.append(x)
        scores = geom.score(frozen['name'], frozen['model'], np.concatenate(features)).reshape(-1, 4)
        assert np.isfinite(scores).all()
        output = geom.helper.evaluate(geom.converted(rows), scores)
        selected = []
        for r, s, selection in zip(rows, scores, output['selections']):
            assert rowkey(r) == rowkey(selection)
            i = int(np.argmax(s))
            selection['scores_by_particle'] = s.tolist()
            selection['source_file'] = r['source_file']
            selection['validation_file'] = r['validation_file']
            selected.append(dict(selection, candidate=r['candidates'][i],
                robot_from_object=r['robot_from_object'], fk_metadata=r['fk_metadata']))
        output['all_particles'] = dict(count=4*len(rows), height=sum(t['height'] for r in rows for t in r['labels']))
        output['oracle_sets'] = sum(any(t['height'] for t in r['labels']) for r in rows)
        output['coverage'] = {k: v for k, v in c.items() if k != 'source_sha256'}
        # Match exact view/hand/contact-set indices across A and B, no outcome-based gating.
        matchkeys = {a: {(r['view'], r['hand'], r['sample']) for r in rows if r['condition'] == a} for a in ('A', 'B')}
        common = matchkeys['A'] & matchkeys['B']
        output['matched_ab'] = {}
        for a in ('A', 'B'):
            picks = [s for s in output['selections'] if s['condition'] == a and (s['view'], s['hand'], s['sample']) in common]
            output['matched_ab'][a] = dict(sets=len(picks), height=sum(s['height'] for s in picks), strict=sum(s['strict'] for s in picks))
        save(folder / 'frozen_top1.json', output)
        save(folder / 'selected_candidates.json', dict(coordinate_frame='SAM3D_centered_object',
            transform_to_robot_base='robot_from_object; do not apply twice', rows=selected))
        results[case] = output
    save(target / 'RESULTS.json', dict(top1_ranking_complete=True, original_experiment_complete=False,
        protocol='Frozen hist_7, four particles -> one per contact set; no new poses/simulation/training',
        frozen_policy=policy, cases=results, dro=inputs['dro'], original_errors=inputs['errors']))
    render_report(target, results, inputs)
    save(target / 'local_status.json', dict(stage='complete_existing_top1', finished=time.time(),
         sets=sum(r['sets'] for r in results.values()), height=sum(r['height'] for r in results.values())))


def render_report(target, results, inputs):
    names = dict(bowl_cellshelfdesk='格架书桌 Bowl', stapler_shelfdesk='大架书桌 Stapler',
        gamecontroller_table='圆桌游戏手柄', beerbottle_basket='篮子 BeerBottle',
        tissuebox_cabinet='柜子 TissueBox', camera_eketshelf='Eket 架子 Camera')
    def rate(s, n):
        return f'{s}/{n}＝{100*s/n:.2f}%' if n else '—'
    def group(r, a):
        groups = [g for k, g in r['groups'].items() if k.startswith(a + '/')]
        return sum(g['height'] for g in groups), sum(g['sets'] for g in groups)
    lines = ['# 六场景：保留现有结果的冻结四粒子 Top1', '',
        '只选择已有完整四粒子接触集，模型/501维特征定义/权重均冻结，不重新训练、不重新生成、不更改原姿态、不重新仿真。',
        '成功仅指最终物体提升≥10 cm；严格成功另存于JSON。Top1根据执行前几何、环境和闭合代理特征选择；随后关联已有CPU PhysX标签。',
        '主表仅统计已有完整执行标签的候选组；缺失生成批次不算物理失败，另报覆盖率。这是回顾性固定候选选择，不是新增独立仿真。',
        'DRO列沿用已完成执行的条件成功率，环境淘汰不在此分母。DRO仍缺441个已保留候选的仿真，候选预算与视角覆盖也不同，不能作为最终公平对比结论。', '',
        '|场景|A Top1|B Top1|DRO 成功/已执行（未补齐）|可用组/计划组|', '|---|---:|---:|---:|---:|']
    for case, r in results.items():
        dro = [x for x in inputs['dro'] if x['case'] == case]
        c = r['coverage']
        lines.append(f"|{names[case]}|{rate(*group(r,'A'))}|{rate(*group(r,'B'))}|{rate(sum(x['height'] for x in dro),sum(x['executed'] for x in dro))}|{r['sets']}/{4*c['expected_batches']}|")
    lines += ['', '## 按手拆分', '', '|场景|手|A Top1|B Top1|', '|---|---|---:|---:|']
    for case, r in results.items():
        for h in ('barrett', 'shadowhand'):
            a, b = [r['groups'][x+'/'+h] for x in ('A','B')]
            lines.append(f"|{names[case]}|{h}|{rate(a['height'],a['sets'])}|{rate(b['height'],b['sets'])}|")
    lines += ['', '## A/B共同可用索引（相同视角、手、set编号）', '',
              'A/B接触点不同；这里只配对实验条件索引，不宣称接触点相同。', '',
              '|场景|A Top1|B Top1|', '|---|---:|---:|']
    for case, r in results.items():
        a, b = [r['matched_ab'][x] for x in ('A','B')]
        lines.append(f"|{names[case]}|{rate(a['height'],a['sets'])}|{rate(b['height'],b['sets'])}|")
    lines += ['', '## 排序收益与覆盖率', '',
        '|场景|全部粒子成功率（随机选一期望）|Top1成功率|事后oracle上限（不可部署）|缺失批次|', '|---|---:|---:|---:|---:|']
    for case, r in results.items():
        p=r['all_particles']
        lines.append(f"|{names[case]}|{rate(p['height'],p['count'])}|{rate(r['height'],r['sets'])}|{rate(r['oracle_sets'],r['sets'])}|{len(r['coverage']['missing_batches'])}|")
    total=sum(r['sets'] for r in results.values());success=sum(r['height'] for r in results.values())
    lines += ['', f'合并Top1：{rate(success,total)}。不同场景视角数不同，此为微平均，不是场景等权平均。', '',
        '每场景frozen_top1.json保留每组4个分数、所选particle、原候选路径与仿真标签路径；selected_candidates.json保存所选原姿态及坐标变换。',
        'coverage.json保存输入文件SHA-256及缺失批次；远端特征计算结束后已验证原候选与仿真结果哈希未改变。']
    (target / 'REPORT.md').write_text('\n'.join(lines)+'\n')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['remote','feature-worker','evaluate'], required=True)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--case');parser.add_argument('--condition');parser.add_argument('--hand')
    args=parser.parse_args()
    if args.mode=='remote':remote(args.root)
    elif args.mode=='feature-worker':feature_worker(args.root/RECOVERY,args.case,args.condition,args.hand)
    else:evaluate(args.root)


if __name__=='__main__':
    main()
