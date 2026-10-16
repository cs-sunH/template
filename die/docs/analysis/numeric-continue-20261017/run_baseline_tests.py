#!/usr/bin/env python3
"""Relink the exact new test objects to the actual freshly built A+B baseline."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--build', type=Path, required=True)
p.add_argument('--baseline', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
root = Path(__file__).resolve().parents[4]
build, library, out = a.build.resolve(), a.baseline.resolve(), a.output.resolve()
out.mkdir(parents=True, exist_ok=False)
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
report = {'scope': 'new test red controls against real A+B; not candidate acceptance',
          'baseline_library': str(library), 'baseline_library_sha256': sha(library), 'steps': []}

def run(label, command):
    row = {'label': label, 'command': list(map(str, command)), 'started_unix': time.time()}
    report['steps'].append(row)
    with (out / (label + '.log')).open('w') as log:
        try:
            row['return_code'] = subprocess.run(row['command'], cwd=root, stdout=log,
                                                stderr=subprocess.STDOUT, timeout=60).returncode
        except (OSError, subprocess.TimeoutExpired) as error:
            row.update(status='not_completed', error=str(error))
    row.update(finished_unix=time.time(), log_sha256=sha(out / (label + '.log')))
    (out / 'identity.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: row.get(k) for k in ('label', 'return_code', 'status')}), flush=True)
    return row.get('return_code')

red = []
for stem in ('positive_time', 'credit_representability', 'failure_snapshot', 'tiny_ready', 'subepsilon_event'):
    name = 'runtime_' + stem + '_test'
    obj = build / ('CMakeFiles/coarse_' + name + '.dir/tests/' + name + '.cc.o')
    exe = out / (name + '.exe')
    report.setdefault('test_inputs', {})[name] = {'object_sha256': sha(obj),
        'source_sha256': sha(root / 'die/tests' / (name + '.cc'))}
    if run(name + '-link', ['g++', obj, library, '-o', exe]) == 0:
        report['test_inputs'][name]['executable_sha256'] = sha(exe)
        code = run(name, [exe, out / name, 'fresh_ab_baseline'])
        observation = out / name / 'report.json'
        report['test_inputs'][name]['observation_written'] = observation.is_file()
        red.append(code == 1 and observation.is_file())
    else:
        red.append(False)
report['baseline_unchanged'] = sha(library) == report['baseline_library_sha256']
report['all_five_executed_red'] = len(red) == 5 and all(red)
(out / 'identity.json').write_text(json.dumps(report, indent=2) + '\n')
raise SystemExit(not (report['all_five_executed_red'] and report['baseline_unchanged']))
