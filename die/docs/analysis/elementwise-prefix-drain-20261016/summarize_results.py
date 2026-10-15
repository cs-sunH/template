#!/usr/bin/env python3
"""Preserve all metrics, scope and source identities for this checkpoint."""
import collections
import hashlib
import json
import pathlib

root = pathlib.Path(__file__).resolve().parents[4]
evidence = root / 'die/docs/evidence/elementwise-prefix-drain-20261016'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
summary = {'scope': 'synthetic independent single-Core oracle; not hardware calibration', 'suites': {}}
for name in ['baseline', 'longer-after-fix', 'original-after-fix', 'two-cohort-after-fix']:
    counts = {k: collections.Counter() for k in ['ledger', 'completion_busy', 'completion', 'first_output']}
    maxima = {}
    for path in sorted((evidence / name).glob('*.report.json')):
        report = json.loads(path.read_text())
        counts['ledger'][report['ledger']['status']] += 1
        for scene, result in report['comparisons'].items():
            comparison = result['invocation_barrier']['comparison']
            counts['completion_busy'][comparison['status']] += 1
            counts['completion']['target_10' if comparison['completion_target_10_percent'] else
                                 'within_20_unaccepted' if comparison['completion_within_20_percent'] else 'failed'] += 1
            first = comparison['first_output_readiness']
            counts['first_output']['target_10' if all(x['target_10_percent'] for x in first) else
                                   'within_20_unaccepted' if all(x['within_20_percent'] for x in first) else 'failed'] += 1
            for kind in ['completion', 'first_output_readiness', 'busy_original_definition']:
                metrics = comparison[kind]
                for metric, value in (metrics.items() if isinstance(metrics, dict) else enumerate(metrics)):
                    for measure in ['relative_error', 'absolute_error_s']:
                        number = value[measure]
                        if number is None:
                            continue
                        key = kind + ':' + measure
                        if key not in maxima or number > maxima[key]['value']:
                            maxima[key] = {'value': number, 'case': path.name, 'scene': scene,
                                           'metric': metric, 'actual': value['actual_s'], 'reference': value['reference_s']}
    summary['suites'][name] = {'counts': counts, 'maxima': maxima}
comparisons = []
def differences(old, new, prefix=''):
    if isinstance(old, dict):
        return [d for k in old for d in differences(old[k], new[k], prefix + '/' + str(k))]
    if isinstance(old, list):
        return [d for k, value in enumerate(old) for d in differences(value, new[k], prefix + '/' + str(k))]
    return [] if old == new else [{'path': prefix, 'before': old, 'after': new}]

for path in sorted((evidence / 'baseline').glob('*.report.json')):
    newer = evidence / 'longer-after-fix' / path.name
    old, new = json.loads(path.read_text()), json.loads(newer.read_text())
    changes = differences(old, new)
    comparisons.append({'case': path.name, 'all_reference_and_comparison_metrics_equal': old == new,
        'all_references_equal': all(old['comparisons'][scene][semantics]['reference'] == value['reference']
                                   for scene, results in new['comparisons'].items() for semantics, value in results.items()),
        'maximum_changed_actual_s': max((abs(d['after'] - d['before']) for d in changes
                                         if d['path'].endswith('/actual_s')), default=0),
        'changes': changes})
summary['longer_before_after'] = comparisons
benchmark = json.loads((evidence / 'benchmark.json').read_text())
summary['performance'] = {'status': benchmark['status'], 'failures': benchmark['failures'],
    'queries_p95_ns': {case['case']: {k: v['p95_ns'] for k, v in case['metrics'].items()}
                       for case in benchmark['query_fixtures']},
    'updates_p95_ns': {case['case']: case['metric']['p95_ns'] for case in benchmark['runtime_update_fixtures']}}
(evidence / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
files = [root / 'work/build-die-continue/coarse_elementwise_oracle_driver',
         root / 'work/build-die-continue/lib/libcoarse_die_cabi.so']
files += list((root / 'die/docs/analysis/elementwise-prefix-drain-20261016').glob('*.py'))
files += [root / 'die/validation/elementwise_service_oracle.py', root / 'die/validation/cabi.py']
(evidence / 'auxiliary-identity.json').write_text(json.dumps({str(p.relative_to(root)): sha(p) for p in files}, indent=2) + '\n')
print(json.dumps({'suite_counts': {k: v['counts'] for k, v in summary['suites'].items()},
                  'longer_before_after_metrics_all_equal': all(x['all_reference_and_comparison_metrics_equal'] for x in comparisons),
                  'performance': summary['performance']}))
