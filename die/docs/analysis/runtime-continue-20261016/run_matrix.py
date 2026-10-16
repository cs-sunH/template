#!/usr/bin/env python3
"""Freshly compile archived immutable Runtime probes against an explicit library.

Original exact oracles keep their verdicts, including binary64 LP discrepancies.
All output is retained even if an oracle fails. No production source is edited.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--build', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
root = Path(__file__).resolve().parents[4]
build, out = a.build.resolve(), a.output.resolve()
out.mkdir(parents=True, exist_ok=False)
old = root / 'die/docs/analysis/core-resume-20261016'
source = old / 'runtime-subunit/tests'
runners = old / 'runtime-subunit/runners'
library = build / 'libcoarse_die.a'
sha = lambda f: hashlib.sha256(f.read_bytes()).hexdigest()
inputs = [library, root / 'die/tests/test_support.hh', Path(__file__)]
inputs += list((root / 'die/include').rglob('*.hh'))
inputs += list((root / 'die/configs').rglob('*.json'))
inputs += [source / (s + '_credit_driver.cc') for s in ('coupled', 'asymmetric')]
inputs += [runners / (s + '_credit_oracle.py') for s in ('coupled', 'asymmetric')]
inputs += [runners / 'assess_coupled.py', old / 'snapshot-import/guards.cc']
meta = dict(build=str(build), scope='actual explicit-library synthetic probes; no hardware calibration',
            source_sha256={str(f): sha(f) for f in inputs}, steps=[])
def save():
    (out / 'identity.json').write_text(json.dumps(meta, indent=2) + '\n')
def run(label, command):
    row = dict(label=label, command=list(map(str, command)), started_unix=time.time())
    meta['steps'].append(row)
    save()
    with (out / (label + '.stdout.log')).open('w') as stdout, (out / (label + '.stderr.log')).open('w') as stderr:
        try:
            row['return_code'] = subprocess.run(row['command'], cwd=root, stdout=stdout, stderr=stderr, timeout=120).returncode
        except subprocess.TimeoutExpired:
            row['status'] = 'timeout'
        except OSError as error:
            row.update(status='not_run', error=str(error))
    row.update(finished_unix=time.time(), stdout_sha256=sha(out / (label + '.stdout.log')),
               stderr_sha256=sha(out / (label + '.stderr.log')))
    save()
    print(json.dumps(row), flush=True)
    return row.get('return_code') == 0
flags = ['g++', '-std=c++17', '-O3', '-DNDEBUG', '-I', root / 'die/tests',
         '-I', root / 'die/include', '-I', root / 'astra-sim-wscllm/extern/helper',
         '-DCOARSE_DIE_SOURCE_DIR="' + str(root / 'die') + '"',
         '-DCOARSE_DIE_ROOT_DIR="' + str(root) + '"']
for stem in ('coupled', 'asymmetric'):
    executable = out / (stem + '_driver')
    if run(stem + '-compile', flags + [source / (stem + '_credit_driver.cc'), library, '-o', executable]):
        meta.setdefault('executable_sha256', {})[stem] = sha(executable)
        run(stem + '-driver', [executable])
        run(stem + '-oracle', [sys.executable, runners / (stem + '_credit_oracle.py'),
                              out / (stem + '-driver.stdout.log'), out / (stem + '-results.json')])
if all((out / (s + '-driver.stdout.log')).is_file() for s in ('coupled', 'asymmetric')):
    # This unchanged offline assessor writes relative to its own directory.
    shutil.copy2(runners / 'assess_coupled.py', out / 'assess_coupled.py')
    run('layered-assessment', [sys.executable, out / 'assess_coupled.py'])
executable = out / 'snapshot_guards'
if run('snapshot-guards-compile', flags + [old / 'snapshot-import/guards.cc', library, '-o', executable]):
    meta.setdefault('executable_sha256', {})['snapshot_guards'] = sha(executable)
    run('snapshot-guards', [executable])
meta['inputs_unchanged'] = all(sha(f) == meta['source_sha256'][str(f)] for f in inputs)
save()
print(json.dumps({'steps': {r['label']: r.get('return_code', r.get('status')) for r in meta['steps']},
                  'inputs_unchanged': meta['inputs_unchanged']}))
raise SystemExit(any(r.get('return_code') != 0 for r in meta['steps']) or not meta['inputs_unchanged'])
