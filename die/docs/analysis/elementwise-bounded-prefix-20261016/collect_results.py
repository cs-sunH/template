#!/usr/bin/env python3
"""Bind actual results and optionally reference byte-identical Git artifacts."""
import argparse
import collections
import hashlib
import json
import pathlib

p = argparse.ArgumentParser()
p.add_argument('--reference-identical', action='store_true')
a = p.parse_args()
root = pathlib.Path(__file__).resolve().parents[4]
base = root / 'die/docs/evidence'
current = base / 'elementwise-bounded-prefix-20261016'
old = base / 'elementwise-prefix-drain-20261016'
digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
manifest_path = current / 'referenced-artifacts.json'
references = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
if a.reference_identical:
    for name, previous in [('original', 'original-after-fix'), ('two-cohort', 'two-cohort-after-fix')]:
        for path in sorted((current / name).glob('*.json')):
            if not path.name.endswith(('.raw.json', '.report.json')):
                continue
            earlier = old / previous / path.name
            if path.read_bytes() != earlier.read_bytes():
                continue
            key = str(path.relative_to(current))
            references[key] = {'reference': str(earlier.relative_to(root)), 'sha256': digest(path),
                               'actual_rerun_byte_identical': True}
            # Preserve the physical new-run copies in work as well. Git holds
            # the existing identical artifact plus this checked hash manifest.
            backup = root / 'work/elementwise-bounded-prefix/referenced-rerun-copies' / key
            backup.parent.mkdir(parents=True, exist_ok=True)
            if backup.exists():
                raise RuntimeError('refusing to overwrite existing rerun copy: ' + str(backup))
            path.rename(backup)
    manifest_path.write_text(json.dumps(references, indent=2) + '\n')

def read(path):
    if not path.exists():
        ref = references[str(path.relative_to(current))]
        path = root / ref['reference']
        if digest(path) != ref['sha256']:
            raise RuntimeError('reference SHA mismatch')
    return json.loads(path.read_text())

def report_names(name):
    names = {x.name for x in (current / name).glob('*.report.json')}
    names.update(pathlib.Path(key).name for key in references if key.startswith(name + '/') and key.endswith('.report.json'))
    return sorted(names)

summary = {'scope': 'independent synthetic single-Core oracle; not hardware calibration', 'suites': {}}
for name, previous in [('longer', 'longer-after-fix'), ('original', 'original-after-fix'), ('two-cohort', 'two-cohort-after-fix')]:
    counts = {key: collections.Counter() for key in ['ledger', 'completion_busy', 'completion', 'first_output']}
    maxima, comparisons = {}, []
    for filename in report_names(name):
        report = read(current / name / filename)
        before = json.loads((old / previous / filename).read_text())
        counts['ledger'][report['ledger']['status']] += 1
        refs_equal = True
        changes = []
        for scene, result in report['comparisons'].items():
            for semantics, value in result.items():
                refs_equal &= value['reference'] == before['comparisons'][scene][semantics]['reference']
            comparison = result['invocation_barrier']['comparison']
            earlier = before['comparisons'][scene]['invocation_barrier']['comparison']
            counts['completion_busy'][comparison['status']] += 1
            counts['completion']['target_10' if comparison['completion_target_10_percent'] else
                                 'within_20_unaccepted' if comparison['completion_within_20_percent'] else 'failed'] += 1
            first = comparison['first_output_readiness']
            counts['first_output']['target_10' if all(x['target_10_percent'] for x in first) else
                                   'within_20_unaccepted' if all(x['within_20_percent'] for x in first) else 'failed'] += 1
            changes.append({'scene': scene, 'before': earlier['status'], 'after': comparison['status']})
            for kind in ['completion', 'first_output_readiness', 'busy_original_definition']:
                values = comparison[kind]
                for metric, value in (values.items() if isinstance(values, dict) else enumerate(values)):
                    for measure in ['relative_error', 'absolute_error_s']:
                        number = value[measure]
                        if number is None:
                            continue
                        key = kind + ':' + measure
                        if key not in maxima or number > maxima[key]['value']:
                            maxima[key] = {'value': number, 'case': filename, 'scene': scene, 'metric': metric,
                                           'actual_s': value['actual_s'], 'reference_s': value['reference_s']}
        comparisons.append({'case': filename, 'all_references_equal': refs_equal,
                            'all_report_metrics_equal': report == before, 'completion_busy_changes': changes})
    summary['suites'][name] = {'counts': counts, 'maxima': maxima, 'before_after': comparisons}
benchmark = read(current / 'benchmark.json')
summary['performance'] = {'status': benchmark['status'], 'failures': benchmark['failures'],
    'query_p95_ns': {x['case']: {k: v['p95_ns'] for k, v in x['metrics'].items()} for x in benchmark['query_fixtures']},
    'update_p95_ns': {x['case']: x['metric']['p95_ns'] for x in benchmark['runtime_update_fixtures']}}
(current / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
paths = [root / 'die/src/Mapping.cc', root / 'die/tests/elementwise_service_test.cc',
         root / 'die/tests/elementwise_oracle_driver.cc', root / 'die/validation/elementwise_service_oracle.py',
         root / 'die/validation/cabi.py', root / 'work/build-die-continue/coarse_elementwise_oracle_driver',
         root / 'work/build-die-continue/lib/libcoarse_die_cabi.so']
paths += list(pathlib.Path(__file__).parent.glob('*.py'))
(current / 'auxiliary-identity.json').write_text(json.dumps({str(x.relative_to(root)): digest(x) for x in paths}, indent=2) + '\n')
print(json.dumps({'counts': {k: v['counts'] for k, v in summary['suites'].items()},
                  'referenced_identical_artifacts': len(references), 'performance': summary['performance']['status']}))
