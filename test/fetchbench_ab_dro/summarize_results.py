"""Read final finite-AB audit and individual DRO trials; never use obsolete AB paths."""
import argparse
import json
import math
from collections import defaultdict
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def count_trials(trials):
    height = strict = 0
    for trial in trials:
        value = trial['final_object_lift_m']
        if not isinstance(value, (float, int)) or isinstance(value, bool) or not math.isfinite(value):
            raise ValueError('Nonfinite/invalid final_object_lift_m is not a completed failure')
        if not isinstance(trial['success'], bool):
            raise ValueError('strict success must be boolean')
        height += value >= .10
        strict += trial['success']
    return len(trials), height, strict


def finish(row):
    row = dict(row)
    keys = ('planned', 'invalid', 'filtered', 'executed', 'height', 'strict')
    if any(type(row[k]) is not int or row[k] < 0 for k in keys):
        raise ValueError('Counts must be nonnegative integers')
    row['missing'] = row['planned'] - row['invalid'] - row['filtered'] - row['executed']
    if row['missing'] < 0 or max(row['height'], row['strict']) > row['executed']:
        raise ValueError('Inconsistent candidate counts')
    row['complete'] = row['missing'] == 0
    # Incomplete groups expose counts, not a misleading final success rate.
    row['executed_height_rate'] = row['height'] / row['executed'] if row['complete'] and row['executed'] else None
    row['planned_height_rate'] = row['height'] / row['planned'] if row['complete'] and row['planned'] else None
    return row


def aggregate(rows, fields):
    groups = {}
    for row in rows:
        key = tuple(row[k] for k in fields)
        group = groups.setdefault(key, dict(zip(fields, key), **{k: 0 for k in (
            'planned', 'invalid', 'filtered', 'executed', 'height', 'strict')}))
        for k in ('planned', 'invalid', 'filtered', 'executed', 'height', 'strict'):
            group[k] += row[k]
    return [finish(group) for group in groups.values()]


def summarize(run):
    manifest = read(run / 'manifest.json')
    ab = read(run / 'ab_finite_validation/summary.json')
    rows = []
    expected_ab_keys = set()
    for case in manifest['cases']:
        views = read(Path(case['observations']) / 'visibility/legal_partial_views.json')['views']
        view_ids = [Path(v['rgbd_capture_dir']).name for v in views]
        if len(set(view_ids)) != len(view_ids):
            raise ValueError('Duplicate view IDs')
        for method in ('A', 'B'):
            for hand in ('barrett', 'shadowhand'):
                expected_ab_keys.add((case['id'], method, hand))
                found = [g for g in ab['groups'] if (g['case'], g['condition'], g['hand']) == (case['id'], method, hand)]
                if len(found) != 1 or found[0]['planned'] != len(views) * 4:
                    raise ValueError('Missing/duplicate AB group or wrong 4-pose budget')
                g = found[0]
                row = finish(dict(case=case['id'], method=method, hand=hand,
                    **{k: g[k] for k in ('planned', 'invalid', 'executed', 'height', 'strict')}, filtered=0))
                if g['finite'] != g['planned'] - g['invalid'] or g['missing_finite'] != row['missing']:
                    raise ValueError('AB finite audit mismatch')
                rows.append(row)
        for hand in ('barrett', 'shadowhand'):
            row = dict(case=case['id'], method='DRO', hand=hand, planned=len(views)*64,
                       invalid=0, filtered=0, executed=0, height=0, strict=0)
            for vid in view_ids:
                folder = run / 'generation' / case['id'] / 'DRO_cpu_physx_all64/views' / vid / 'simulation' / case['scene_factory'] / ('task_%03d' % case['task_index']) / hand
                filt = folder / 'dro_candidates_environment_filtered.json'
                paths = sorted((folder / 'lift_validation').glob('candidate_*.json'))
                if not filt.exists():
                    if paths:
                        raise ValueError('DRO trials without environment-filter provenance')
                    continue
                payload = read(filt)
                f = payload['environment_filter']
                retained = f['retained_candidates']
                if f['generated_candidates'] != 64 or not 0 <= retained <= 64 or len(payload['records']) != retained:
                    raise ValueError('DRO candidate budget/filter mismatch')
                if len(paths) > retained:
                    raise ValueError('Too many DRO executions')
                row['filtered'] += 64-retained
                count, height, strict = count_trials([read(p) for p in paths])
                row['executed'] += count; row['height'] += height; row['strict'] += strict
            rows.append(finish(row))
    if len(ab['groups']) != len(expected_ab_keys):
        raise ValueError('Unexpected AB groups')
    complete = ab['complete'] and not ab.get('errors') and all(r['complete'] for r in rows)
    return dict(complete=bool(complete), criterion='final_object_lift_m >= 0.10', rows=rows,
                by_scene=aggregate(rows, ('case', 'method')), by_method=aggregate(rows, ('method',)),
                provenance={'AB': 'ab_finite_validation/summary.json (finite validator trial aggregation)',
                            'DRO': 'individual lift_validation/candidate_*.json + filter records'},
                note='AB 4 retained poses vs DRO 64 raw candidates; input AND checkpoint differ for A/B. Not a budget-matched pure ablation.')


def markdown(result):
    lines = ['# FetchBench 三场景 A/B 与 DRO', '',
             '完成：%s；主判定：最终物体提升 ≥10 cm（不是过程最大提升）。' % result['complete'], '',
             '|场景|方法|手|计划|NaN 排除|环境筛除|执行|缺失|高度成功|严格成功|执行成功率|计划成功率|',
             '|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for row in result['rows']:
        values = [str(row[k]) for k in ('case', 'method', 'hand', 'planned', 'invalid', 'filtered', 'executed', 'missing', 'height', 'strict')]
        values += ['—' if row[k] is None else '%.2f%%' % (100*row[k]) for k in ('executed_height_rate', 'planned_height_rate')]
        lines.append('|'+'|'.join(values)+'|')
    lines.extend(['', '缺失不记作已完成失败；NaN 与环境筛除单列，计划分母包含二者。', '', result['note'], ''])
    return '\n'.join(lines)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path, help='New directory; never overwrite old reports')
    args = parser.parse_args()
    result = summarize(args.run)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'results.json').write_text(json.dumps(result, indent=2)+'\n')
    (args.output / 'RESULTS.md').write_text(markdown(result))
    print(json.dumps({'complete': result['complete'], 'by_method': result['by_method']}))
