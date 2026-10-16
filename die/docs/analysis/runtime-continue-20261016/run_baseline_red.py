#!/usr/bin/env python3
"""Compile the unchanged new regressions with the freshly built main baseline."""
import hashlib
import json
from pathlib import Path
import subprocess
import time

root = Path(__file__).resolve().parents[4]
out = root / 'work/runtime-continue/baseline-red'
out.mkdir(parents=True, exist_ok=False)
source = root / 'work/runtime-baseline-source/die'
build = root / 'work/build-runtime-baseline-release'
sha = lambda f: hashlib.sha256(f.read_bytes()).hexdigest()
inputs = [build / 'libcoarse_die.a', build / 'lib/libcoarse_die_cabi.so',
          root / 'die/tests/test_support.hh', source / 'include/coarse_die/Execution.hh',
          root / 'die/tests/fixtures/snapshot_split_owner.json', Path(__file__)]
names = ['runtime_subunit_credit_test', 'runtime_snapshot_split_owner_test']
inputs += [root / ('die/tests/' + name + '.cc') for name in names]
meta = dict(scope='fresh main baseline; unchanged new tests; failures retained',
            sha256={str(f): sha(f) for f in inputs}, steps=[])
for name in names:
    exe = out / name
    commands = [
        ['g++', '-std=c++17', '-O3', '-DNDEBUG', '-I', str(root / 'die/tests'),
         '-I', str(source / 'include'), '-I', str(root / 'astra-sim-wscllm/extern/helper'),
         '-DCOARSE_DIE_SOURCE_DIR="' + str(root / 'die') + '"',
         '-DCOARSE_DIE_ROOT_DIR="' + str(root) + '"',
         str(root / ('die/tests/' + name + '.cc')), str(build / 'libcoarse_die.a'),
         str(build / 'lib/libcoarse_die_cabi.so'), '-Wl,-rpath,' + str(build / 'lib'), '-o', str(exe)],
        [str(exe)]]
    for phase, command in zip(('compile', 'test'), commands):
        label = name + '-' + phase
        start = time.time()
        with (out / (label + '.log')).open('w') as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=120)
        meta['steps'].append(dict(label=label, command=command, return_code=result.returncode,
                                  started_unix=start, finished_unix=time.time(),
                                  log_sha256=sha(out / (label + '.log'))))
        if phase == 'compile' and result.returncode:
            break
    if exe.exists():
        meta.setdefault('executable_sha256', {})[name] = sha(exe)
        meta.setdefault('ldd', {})[name] = subprocess.check_output(['ldd', str(exe)], text=True)
meta['inputs_unchanged'] = all(sha(f) == meta['sha256'][str(f)] for f in inputs)
(out / 'identity.json').write_text(json.dumps(meta, indent=2) + '\n')
print(json.dumps({'steps': {r['label']: r['return_code'] for r in meta['steps']},
                  'inputs_unchanged': meta['inputs_unchanged']}))
# Expected red is recorded as red; a separate caller decides its meaning.
