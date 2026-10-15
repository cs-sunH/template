from __future__ import annotations
import hashlib, json, os, subprocess, sys, time
from pathlib import Path
repo = Path('/workspace/template')
work = repo / 'work/integration-input-20261015'
variant = sys.argv[1]
stage = work / 'stage-root' / variant
build = work / 'build' / variant
build.mkdir(parents=True, exist_ok=True)
logdir = work / 'logs' / variant
logdir.mkdir(parents=True, exist_ok=True)
cmake = str(repo / 'work/tooling/bin/cmake')
configure = [cmake, '-S', str(stage / 'extern/die_model'), '-B', str(build),
    '-DCMAKE_BUILD_TYPE=Release',
    '-DCOARSE_DIE_JSON_INCLUDE=' + str(stage / 'extern/helper'),
    '-DCOARSE_DIE_BUILD_TESTS=OFF']
compile_cmd = [cmake, '--build', str(build), '--parallel', '2', '--target',
    'die_model_cabi', 'coarse_die_cabi']
env = os.environ.copy()
env['PYTHONPATH'] = str(repo / 'work/tooling') + os.pathsep + env.get('PYTHONPATH', '')
results = []
for label, command in [('configure', configure), ('build', compile_cmd)]:
    started = time.time()
    result = subprocess.run(command, cwd=repo, env=env, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    output = result.stdout
    (logdir / f'{label}.log').write_text(output, encoding='utf-8')
    row = {'phase': label, 'command': command, 'cwd': str(repo),
        'returncode': result.returncode, 'elapsed_seconds': round(time.time()-started, 3),
        'log': str(logdir / f'{label}.log')}
    results.append(row)
    print(json.dumps(row), flush=True)
    if result.returncode:
        print(output, file=sys.stderr)
        break
libs = {}
if len(results) == 2 and all(row['returncode'] == 0 for row in results):
    for name in ('libdie_model_cabi.so', 'libcoarse_die_cabi.so'):
        path = build / 'lib' / name
        if not path.is_file():
            raise SystemExit(f'missing build output: {path}')
        libs[name] = {'path': str(path), 'size': path.stat().st_size,
                      'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
summary = {'variant': variant, 'source_commit': '89c2499c6e61de450291961d800594a8a83eeca7',
    'release_source_digest': json.loads((stage / 'extern/die_model/coarse/RELEASE.json').read_text())['source_digest'],
    'phases': results, 'libraries': libs}
(logdir / 'build-identity.json').write_text(json.dumps(summary, indent=2) + '\n')
if len(results) != 2 or any(row['returncode'] for row in results):
    raise SystemExit(1)
print(json.dumps({'variant': variant, 'status': 'built', 'libraries': libs}, sort_keys=True))
