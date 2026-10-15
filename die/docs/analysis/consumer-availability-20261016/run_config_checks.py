#!/usr/bin/env python3
"""Run a consumer's actual config suite against its two built ABI libraries."""
import argparse
import hashlib
import json
import os
import pathlib
import re
import subprocess
import sys
import time

p = argparse.ArgumentParser()
p.add_argument('--root', required=True)
p.add_argument('--build', required=True)
p.add_argument('--repo', required=True)
p.add_argument('--label', required=True)
a = p.parse_args()
root = pathlib.Path(__file__).resolve().parents[4]
source = pathlib.Path(a.root).resolve() / a.repo
build = pathlib.Path(a.build).resolve()
if pathlib.Path(a.label).name != a.label: p.error('simple fresh label required')
evidence = root / 'die/docs/evidence/consumer-availability-20261016' / a.label
evidence.mkdir(parents=True, exist_ok=False)
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
libraries = {name: build / 'lib' / ('lib' + filename + '.so') for name, filename in
    [('DIE_MODEL_CABI_PATH', 'die_model_cabi'), ('DIE_COARSE_CABI_PATH', 'coarse_die_cabi')]}
if not all(path.is_file() for path in libraries.values()):
    raise RuntimeError('both libraries must have been built for this consumer')
env = os.environ.copy()
env.update({name: str(path) for name, path in libraries.items()})
env['PYTHONDONTWRITEBYTECODE'] = '1'
command = [sys.executable, '-m', 'unittest', 'discover', '-s', str(source / 'sh_test_mesh/tests'),
           '-p', 'test_config_resolver*.py', '-v']
inputs = [source / 'sh_test_mesh/config_resolver.py', source / 'CMakeLists.txt',
          build / 'CMakeCache.txt', source / 'extern/die_model/coarse/RELEASE.json']
inputs += list((source / 'sh_test_mesh/tests').glob('test_config_resolver*.py'))
inputs += list((source / 'sh_test_mesh/hardware').rglob('*.json'))
inputs += [path for base in ['extern/die_model', 'extern/helper/json']
           for path in (source / base).rglob('*')
           if path.is_file() and path.suffix in ['.cc', '.hh', '.h', '.hpp', '.json']]
meta = {'repo': a.repo, 'source_root': str(source), 'build': str(build),
        'command': command, 'cwd': str(source), 'started_unix': time.time(),
        'library_sha256': {name: sha(path) for name, path in libraries.items()},
        'library_paths': {name: str(path) for name, path in libraries.items()},
        'source_sha256': {str(path): sha(path) for path in sorted(set(inputs)) if path.is_file()},
        'runner_sha256': sha(pathlib.Path(__file__))}
with (evidence / 'run.log').open('w') as log:
    result = subprocess.run(command, cwd=source, env=env, stdout=log, stderr=subprocess.STDOUT)
log = (evidence / 'run.log').read_text()
counts = re.findall(r'^Ran (\d+) tests? in ', log, re.MULTILINE)
meta.update(return_code=result.returncode, finished_unix=time.time(),
            tests_run=int(counts[-1]) if counts else None,
            skips_mentioned='skipped' in log.lower(), log_sha256=sha(evidence / 'run.log'))
meta['status'] = 'passed' if result.returncode == 0 and meta['tests_run'] and not meta['skips_mentioned'] else 'failed_or_incomplete'
(evidence / 'identity.json').write_text(json.dumps(meta, indent=2) + '\n')
print(json.dumps({k: meta[k] for k in ['repo', 'status', 'tests_run', 'skips_mentioned', 'return_code']}))
raise SystemExit(meta['status'] != 'passed')
