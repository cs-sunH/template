#!/usr/bin/env python3
"""Link a real R3 admission probe with a built consumer's exact objects.

Only its test main is replaced. Existing production/backend objects, archives,
SDK headers and compiler flags are reused and hashed; CMake and the standard
test executable are never modified. No simulator formula is reimplemented.
"""
import argparse
import hashlib
import json
import pathlib
import shlex
import subprocess
import sys
import time

p = argparse.ArgumentParser()
p.add_argument('--root', required=True)
p.add_argument('--build', required=True)
p.add_argument('--repo', required=True)
p.add_argument('--label', required=True)
p.add_argument('--kind', choices=['repeat', 'lifecycle', 'native_projection'], default='repeat')
p.add_argument('--fixture-source')
a = p.parse_args()
root = pathlib.Path(__file__).resolve().parents[4]
source = pathlib.Path(a.root).resolve()
build = pathlib.Path(a.build).resolve()
if pathlib.Path(a.label).name != a.label:
    p.error('simple fresh label required')
evidence = root / 'die/docs/evidence/consumer-availability-20261016' / a.label
evidence.mkdir(parents=True, exist_ok=False)
work = root / 'work/consumer-availability' / a.label
work.mkdir(parents=True, exist_ok=False)
joint = a.repo == 'astra-sim-joint'
stem = 'coarse_workspace_runtime_test' if joint else 'die_execution_lifecycle_test'
target = 'AstraSim_Analytical_Congestion_Aware_' + ('CoarseWorkspaceRuntimeTest' if joint else 'DieExecutionLifecycleTest')
if a.kind == 'native_projection':
    if not joint: p.error('native projection fixture belongs to JOINT')
    stem, target = 'native_coarse_projection_test', 'NativeCoarseProjectionTest'
fixture = source / a.repo / 'astra-sim/workload/execution_driven/tests' / (stem + '.cc')
if a.fixture_source:
    fixture = pathlib.Path(a.fixture_source).resolve()
template = pathlib.Path(__file__).with_name('repeated_prepare_probe.cc')
wrapper = work / 'probe.cc'
wrapper.write_text(('#define CONSUMER_FIXTURE_SOURCE ' + json.dumps(str(fixture)) + '\n'
                   '#define AVAILABILITY_SDK_ROOT ' + json.dumps(str(source / a.repo / 'extern/die_model/coarse')) + '\n'
                   + ('#define AVAILABILITY_JOINT\n' if joint else '')
                   + '#include ' + json.dumps(str(template)) + '\n') if a.kind == 'repeat'
                   else '#include ' + json.dumps(str(fixture)) + '\n')
(evidence / 'probe.cc').write_bytes(wrapper.read_bytes())
matches = list(build.rglob('CMakeFiles/' + target + '.dir/flags.make'))
if len(matches) != 1:
    raise RuntimeError('one configured target directory required')
metadata = matches[0].parent
directory = metadata.parent.parent
flags = {}
for line in (metadata / 'flags.make').read_text().splitlines():
    if line.startswith('CXX_') and ' = ' in line:
        key, value = line.split(' = ', 1)
        flags[key] = shlex.split(value)
cache = (build / 'CMakeCache.txt').read_text().splitlines()
compiler = next(line.split('=', 1)[1] for line in cache if line.startswith('CMAKE_CXX_COMPILER:FILEPATH='))
obj = work / 'probe.o'
binary = work / 'repeated-prepare-probe'
compile_cmd = [compiler, *flags['CXX_DEFINES'], *flags['CXX_INCLUDES'], *flags['CXX_FLAGS'],
               '-c', str(wrapper), '-o', str(obj)]
link = shlex.split((metadata / 'link.txt').read_text())
replaced = [i for i, value in enumerate(link) if value.endswith('/' + stem + '.cc.o')]
if len(replaced) != 1:
    raise RuntimeError('one original fixture main object required')
link[replaced[0]] = str(obj)
link[link.index('-o') + 1] = str(binary)
# Keep the standard target's link.d untouched as well as its executable.
link = [('-Wl,--dependency-file=' + str(work / 'link.d'))
        if x.startswith('-Wl,--dependency-file=') else x for x in link]
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
inputs = [template, fixture, wrapper, metadata / 'flags.make', metadata / 'link.txt', build / 'CMakeCache.txt']
for value in link:
    path = pathlib.Path(value)
    path = path if path.is_absolute() else directory / path
    if path.is_file(): inputs.append(path.resolve())
identity = {'repo': a.repo, 'source_root': str(source), 'build': str(build),
            'kind': a.kind, 'fixture_source': str(fixture),
            'commands': {'compile': compile_cmd, 'link': link}, 'cwd': str(directory),
            'input_sha256': {str(path): sha(path) for path in sorted(set(inputs))},
            'runner_sha256': sha(pathlib.Path(__file__)), 'started_unix': time.time()}
result = None
for step, command in [('compile', compile_cmd), ('link', link)]:
    with (evidence / (step + '.log')).open('w') as out:
        result = subprocess.run(command, cwd=directory, stdout=out, stderr=subprocess.STDOUT)
    identity[step + '_return_code'] = result.returncode
    if result.returncode:
        break
if result.returncode == 0:
    identity['probe_binary_sha256'] = sha(binary)
    record = pathlib.Path(__file__).with_name('record_cpp_run.py')
    result = subprocess.run([sys.executable, str(record), '--root', str(source),
        '--prefix', str(evidence / 'run'), '--build', str(build), '--', str(binary)], cwd=root)
    identity['run_return_code'] = result.returncode
else:
    identity['run_status'] = 'not_run_after_build_failure'
identity['finished_unix'] = time.time()
(evidence / 'probe-identity.json').write_text(json.dumps(identity, indent=2) + '\n')
print(json.dumps({'repo': a.repo, 'return_code': result.returncode, 'evidence': str(evidence)}))
raise SystemExit(result.returncode)
