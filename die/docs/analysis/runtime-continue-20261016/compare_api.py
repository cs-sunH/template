#!/usr/bin/env python3
"""Exact public API/trace comparison with the observed fork repair explicit."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import subprocess

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--baseline', type=Path, required=True)
p.add_argument('--candidate', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
if a.output.exists():
    p.error('fresh output required')
sha = lambda f: hashlib.sha256(f.read_bytes()).hexdigest()
old, new = [json.loads(gzip.decompress(f.read_bytes())) for f in (a.baseline, a.candidate)]
def differences(x, y, path=''):
    if type(x) != type(y):
        return [dict(path=path, before=x, after=y)]
    if isinstance(x, dict):
        result = []
        for k in sorted(x.keys() | y.keys()):
            if k not in x or k not in y:
                result.append(dict(path=path + '/' + k, before=x.get(k, '<missing>'), after=y.get(k, '<missing>')))
            else:
                result += differences(x[k], y[k], path + '/' + k)
        return result
    if isinstance(x, list):
        if len(x) != len(y):
            return [dict(path=path + '/length', before=len(x), after=len(y))]
        return [d for i, (u, v) in enumerate(zip(x, y)) for d in differences(u, v, path + '/' + str(i))]
    return [] if x == y else [dict(path=path, before=x, after=y)]
fork_keys = {'fork_failed', 'fork_snapshot', 'fork_final_counters'}
rows = []
assert len(old['results']) == len(new['results']) == 2
for x, y in zip(old['results'], new['results']):
    static = differences({k: v for k, v in x.items() if k != 'live_traces'},
                         {k: v for k, v in y.items() if k != 'live_traces'})
    traces = []
    assert len(x['live_traces']) == len(y['live_traces']) == 4
    for t, u in zip(x['live_traces'], y['live_traces']):
        ordinary = lambda trace: [r for r in trace['rows'] if not r.keys() & fork_keys]
        forks = lambda trace: [r for r in trace['rows'] if r.keys() & fork_keys]
        before, after = forks(t), forks(u)
        repaired = (x['config_name'] == 'tx8_compute_matched_hardware.json' and t['mode'] == 'split')
        if repaired:
            expected = len(before) == 1 and 'fork_failed' in before[0] and len(after) == 2
            expected = expected and 'fork_snapshot' in after[0] and 'fork_final_counters' in after[1]
            # The repaired fork must copy the exact source at the fork call.
            latest = None
            for r in u['rows']:
                if 'snapshot' in r:
                    latest = r['snapshot']
                if 'fork_snapshot' in r:
                    expected = expected and r['fork_snapshot'] == latest
        else:
            expected = before == after
        traces.append(dict(owners=t['owners'], mode=t['mode'],
                           ordinary_differences=differences(ordinary(t), ordinary(u)),
                           baseline_fork=before, candidate_fork=after,
                           repaired_baseline_failure=repaired, expected_fork_behavior=expected))
    rows.append(dict(config=x['config_name'], static_differences=static, traces=traces))
symbols = []
for report in (old, new):
    path = Path(report['library'])
    assert sha(path) == report['library_sha256'], 'actual library changed after capture'
    symbols.append(sorted(line.split()[-1] for line in subprocess.check_output(
        ['nm', '-D', '--defined-only', str(path)], text=True).splitlines()
        if line.split()[-1].startswith('die_coarse_')))
passed = symbols[0] == symbols[1] and all(
    not r['static_differences'] and all(not t['ordinary_differences'] and t['expected_fork_behavior']
                                      for t in r['traces']) for r in rows)
report = dict(scope='exact JSON and C ABI symbol comparison; no accuracy/performance certificate',
              input_sha256={str(f): sha(f) for f in (a.baseline, a.candidate)},
              tool_sha256=sha(Path(__file__)), configs=rows,
              cabi_symbols=dict(baseline=symbols[0], candidate=symbols[1], identical=symbols[0] == symbols[1]),
              passed=passed)
a.output.write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(dict(passed=passed, static_inputs_per_config=28, trace_count=8,
                     repaired_fork_count=sum(t['repaired_baseline_failure'] for r in rows for t in r['traces']),
                     cabi_symbol_count=len(symbols[0]))))
raise SystemExit(not passed)
