#!/usr/bin/env python3
"""Retain command, source/build identities and CPU scope for Norm work."""
import argparse
import hashlib
import json
import pathlib
import re
import subprocess
import time

p = argparse.ArgumentParser()
p.add_argument('--prefix', required=True)
p.add_argument('--build', default='work/build-die-continue')
p.add_argument('command', nargs=argparse.REMAINDER)
a = p.parse_args()
root = pathlib.Path(__file__).resolve().parents[4]
prefix = pathlib.Path(a.prefix).resolve()
if pathlib.Path(str(prefix) + '-identity.json').exists(): p.error('fresh prefix required')
prefix.parent.mkdir(parents=True, exist_ok=True)
build = pathlib.Path(a.build).resolve()
command = a.command[1:] if a.command[:1] == ['--'] else a.command
if not command: p.error('command required')
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
sources = [root / 'die/CMakeLists.txt', root / 'die/validation/norm_row_oracle.py',
           root / 'die/validation/elementwise_service_oracle.py']
for base in ['die/src', 'die/include', 'die/tests', 'die/configs', 'die/docs/analysis/norm-row-accounting-20261016',
             'die/docs/analysis/hotpath-allocation-20261016',
             'die/docs/analysis/compiler-ranking-20261016']:
    sources += [path for path in (root / base).rglob('*') if path.is_file()
                and path.suffix in ['.cc', '.hh', '.h', '.py', '.json', '.txt']]
binaries = list(build.rglob('*.a')) + list(build.rglob('*.so'))
binaries += [path for path in build.rglob('coarse_*') if path.is_file() and path.suffix == '']
if pathlib.Path(command[0]).is_file(): binaries.append(pathlib.Path(command[0]).resolve())
for value in command[1:]:
    path=pathlib.Path(value)
    if path.is_file() and path.suffix in ['.a','.so','.cc','.hh']: binaries.append(path.resolve())
cpu = pathlib.Path('/sys/fs/cgroup/cpu.stat')
meta = {'observed_git_head': subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
    'command': command, 'cwd': str(root), 'build': str(build),
    'source_sha256': {str(path.relative_to(root)): sha(path) for path in sorted(set(sources))},
    'binary_before_sha256': {str(path): sha(path) for path in sorted(set(binaries))},
    'cmake_cache_sha256': sha(build / 'CMakeCache.txt') if (build / 'CMakeCache.txt').is_file() else None,
    'started_unix': time.time(), 'cpu_before': cpu.read_text() if cpu.is_file() else None}
with pathlib.Path(str(prefix) + '.log').open('w') as log:
    try:
        result = subprocess.run(command, cwd=root, stdout=log, stderr=subprocess.STDOUT)
    except OSError as error:
        log.write(str(error)+'\n')
        meta['launch_error']=str(error)
        result=subprocess.CompletedProcess(command,127)
meta.update(return_code=result.returncode, finished_unix=time.time(),
    cpu_after=cpu.read_text() if cpu.is_file() else None,
    binary_after_sha256={str(path): sha(path) for path in sorted(set(binaries)) if path.is_file()})
if pathlib.Path(command[0]).name=='ctest':
    log=pathlib.Path(str(prefix)+'.log').read_text()
    count=re.search(r'Total Tests: (\d+)|out of (\d+)',log)
    meta['tests_run']=int(next(x for x in count.groups() if x is not None)) if count else 0
    meta['status']='passed' if result.returncode==0 and meta['tests_run']>0 else 'failed_or_not_run'
pathlib.Path(str(prefix) + '-identity.json').write_text(json.dumps(meta,indent=2) + '\n')
print(json.dumps({key: meta[key] for key in ['command', 'return_code', 'started_unix', 'finished_unix']}))
raise SystemExit(result.returncode)
