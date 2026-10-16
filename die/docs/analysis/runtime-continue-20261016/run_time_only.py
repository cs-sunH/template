#!/usr/bin/env python3
"""Build an explicit mixed-object time-only experiment and keep every failure."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time

root = Path(__file__).resolve().parents[4]
out = root / 'work/runtime-positive-time'
baseline = root / 'work/build-runtime-ab-release'
sha = lambda f: hashlib.sha256(f.read_bytes()).hexdigest()
meta = dict(scope='time-only mixed-object research; no fresh all-source acceptance or adoption', steps=[],
            inputs_sha256={str(f): sha(f) for f in [out / 'Execution.cc', baseline / 'libcoarse_die.a',
                baseline / 'CMakeFiles/coarse_die_cabi.dir/src/Capi.cc.o', root / 'die/include/coarse_die/Execution.hh',
                root / 'die/src/Execution.cc', Path(__file__)]})
assert not (out / 'run-identity.json').exists()
def save():
    (out / 'run-identity.json').write_text(json.dumps(meta, indent=2) + '\n')
def run(label, command):
    command = list(map(str, command))
    row = dict(label=label, command=command, started_unix=time.time())
    meta['steps'].append(row)
    save()
    with (out / (label + '.log')).open('w') as log:
        try:
            row['return_code'] = subprocess.run(command, cwd=root, stdout=log, stderr=subprocess.STDOUT, timeout=180).returncode
        except subprocess.TimeoutExpired:
            row['status'] = 'timeout'
        except OSError as error:
            row.update(status='not_run', error=str(error))
    row.update(finished_unix=time.time(), log_sha256=sha(out / (label + '.log')))
    save()
    print(json.dumps(dict(label=label, return_code=row.get('return_code'), status=row.get('status'))), flush=True)
    return row.get('return_code') == 0
obj = out / 'Execution.cc.o'
flags = ['g++', '-std=gnu++17', '-O3', '-DNDEBUG', '-fPIC', '-Wall', '-Wextra', '-Wpedantic',
         '-I', root / 'die/include', '-I', root / 'astra-sim-wscllm/extern/helper', '-I', root / 'die/src']
if not run('compile-execution', flags + ['-c', out / 'Execution.cc', '-o', obj]):
    raise SystemExit(1)
archive = out / 'libcoarse_die.a'
shutil.copy2(baseline / 'libcoarse_die.a', archive)
if not run('remove-baseline-execution', ['ar', 'd', archive, 'Execution.cc.o']):
    raise SystemExit(1)
if not run('insert-new-execution', ['ar', 'r', archive, obj]):
    raise SystemExit(1)
meta['archive_members_sha256'] = {}
for name in subprocess.check_output(['ar', 't', archive], text=True).splitlines():
    data = subprocess.check_output(['ar', 'p', archive, name])
    meta['archive_members_sha256'][name] = hashlib.sha256(data).hexdigest()
    if name != 'Execution.cc.o':
        assert data == subprocess.check_output(['ar', 'p', baseline / 'libcoarse_die.a', name])
lib = out / 'lib/libcoarse_die_cabi.so'
lib.parent.mkdir()
if not run('link-cabi', ['g++', '-O3', '-DNDEBUG', '-shared', '-Wl,-soname,libcoarse_die_cabi.so',
                      baseline / 'CMakeFiles/coarse_die_cabi.dir/src/Capi.cc.o', archive, '-o', lib]):
    raise SystemExit(1)
meta['candidate_library_sha256'] = {str(f): sha(f) for f in (archive, lib)}
for name in ('runtime_subunit_credit_test', 'runtime_snapshot_split_owner_test', 'service_curve_runtime_test',
             'runtime_clock_translation_test', 'runtime_semantics_test', 'runtime_counters_test'):
    test_obj = baseline / ('CMakeFiles/coarse_' + name + '.dir/tests/' + name + '.cc.o')
    meta.setdefault('fresh_test_object_sha256', {})[str(test_obj)] = sha(test_obj)
    executable = out / ('coarse_' + name)
    if run(name + '-link', ['g++', '-O3', '-DNDEBUG', test_obj, archive, lib,
                           '-Wl,-rpath,' + str(lib.parent), '-o', executable]):
        meta.setdefault('executable_sha256', {})[name] = sha(executable)
        run(name, [executable])
source = root / 'work/norm-runtime-current/subepsilon_credit_driver.cc'
copy = out / source.name
shutil.copy2(source, copy)
meta['subepsilon_source_sha256'] = sha(copy)
executable = out / 'subepsilon_driver'
if run('subepsilon-compile', ['g++', '-std=c++17', '-O2', '-I', root / 'die/tests', '-I', root / 'die/include',
        '-I', root / 'astra-sim-wscllm/extern/helper', '-DCOARSE_DIE_SOURCE_DIR="' + str(root / 'die') + '"',
        '-DCOARSE_DIE_ROOT_DIR="' + str(root) + '"', copy, archive, '-o', executable]):
    meta.setdefault('executable_sha256', {})['subepsilon_driver'] = sha(executable)
    run('subepsilon-actual', [executable, out / 'subepsilon-observations', 'time-only'])
meta['inputs_unchanged'] = all(sha(Path(f)) == value for f, value in meta['inputs_sha256'].items())
save()
print(json.dumps({'steps': {r['label']: r.get('return_code', r.get('status')) for r in meta['steps']},
                  'inputs_unchanged': meta['inputs_unchanged']}))
raise SystemExit(any(r.get('return_code') != 0 for r in meta['steps']) or not meta['inputs_unchanged'])
