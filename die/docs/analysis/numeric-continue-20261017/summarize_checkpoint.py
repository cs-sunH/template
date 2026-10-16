#!/usr/bin/env python3
"""Summarize actual observations without changing strict comparison verdicts."""
from decimal import Decimal
import hashlib
import json
import math
from pathlib import Path

root = Path(__file__).resolve().parents[4]
work = root / 'work/runtime-resume-20261017'
load = lambda name: json.loads((work / name).read_text())
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
api = load('api-comparison.json')
exact = [abs(Decimal(d['after']) - Decimal(d['before'])) for d in api['differences']
         if isinstance(d['before'], str)]
projection = [abs(d['after'] - d['before']) / math.ulp(d['before']) for d in api['differences']
              if isinstance(d['before'], float)]
coupled, asymmetric = [load('matrix-reviewed/' + n + '-results.json') for n in ('coupled', 'asymmetric')]
layered = load('matrix-reviewed/layered-assessment.json')
perf = {}
for name in ('example', 'engineering'):
    result = load('budget-' + name + '.json')
    record = load('performance-' + name + '.json')
    before, after = [dict(line.split() for line in record[k].splitlines()) for k in ('cpu_before', 'cpu_after')]
    perf[name] = {'status': result['status'], 'samples': result['samples_per_metric'],
                  'failures': result['failures'], 'new_throttled': int(after['nr_throttled']) - int(before['nr_throttled']),
                  'query_p95_ns': {row['case']: {k: v['p95_ns'] for k, v in row['metrics'].items()}
                                   for row in result['query_fixtures']},
                  'update_p95_ns': {row['case']: row['metric']['p95_ns'] for row in result['runtime_update_fixtures']}}
summary = {
    'scope': 'current Runtime numerical checkpoint; not main integration, SDK publication or engineering acceptance',
    'execution_sha256': sha(root / 'die/src/Execution.cc'),
    'header_sha256': sha(root / 'die/include/coarse_die/Execution.hh'),
    'libraries_sha256': {str(p.relative_to(root)): sha(p) for p in
                        [root / 'work/build-numeric-final-20261017/libcoarse_die.a',
                         root / 'work/build-numeric-final-20261017/lib/libcoarse_die_cabi.so']},
    'public_ctest': {'passed': 58, 'failed': 0, 'actual_record': 'public-58.json',
                     'record_return_code': load('public-58.json')['return_code']},
    'new_tests_baseline_red': load('baseline-new-tests-corrected/identity.json')['all_five_executed_red'],
    'invalid_initial_baseline_runner': 'baseline-new-tests/qualification.json; original aborts are tool failures',
    'public_api': {'strict_exact_comparison_passed': api['passed'], 'changed_fields': len(api['differences']),
                   'changes_only_estimate_busy_fields': all('/estimate/' in d['path'] and 'busy_ns' in d['path'] for d in api['differences']),
                   'live_trace_differences': sum('/live_traces/' in d['path'] for d in api['differences']),
                   'max_exact_busy_difference_ns': str(max(exact, default=Decimal(0))),
                   'max_binary64_busy_projection_ulp': max(projection, default=0),
                   'identical_cabi_symbols': api['symbols']['baseline'] == api['symbols']['candidate'],
                   'symbol_count': len(api['symbols']['baseline'])},
    'strict_matrix': {'passed': coupled['passed'] + asymmetric['passed'], 'failed': coupled['failed'] + asymmetric['failed'],
                       'strict_feasibility_violations': len(layered['strict_feasibility_violations']),
                       'machine_ledger_failures': len(layered['machine_ledger_failures'])},
    'snapshot_cross_command_return_code': load('snapshot-cross-command.json')['return_code'],
    'performance': perf,
    'scope_limits': [
        'Synthetic probes do not calibrate target hardware.',
        'Positive-time preflight may reject an unrelated slow curved participant; unsupported representation fails closed.',
        'Capacity1 Norm snapping/deadlock, structural busy error and R>1 readiness are not solved.',
        'Pure legacy progress closure remains unchanged; strict sub-ULP positive legacy tail oracle remains unresolved.',
        'Strict ceil rejects historical floor-valued public clocks with tiny positive fractions.',
        'Fractional error-state snapshot recovery is only tested near origin0; large-origin wire serialization can lose fractions.',
        'Current native evidence is A+B baseline, not this new Runtime.',
        'Formal canonical24 SDKs unchanged; publisher not run; main still old integration plus recovery pointer.',
        'Each performance configuration was measured once; historical failures remain valid.'
    ]}
(work / 'checkpoint-summary.json').write_text(json.dumps(summary, indent=2) + '\n')
print(json.dumps({k: summary[k] for k in ('execution_sha256', 'public_ctest', 'public_api', 'strict_matrix')}))
