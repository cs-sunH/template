#!/usr/bin/env python3
"""Bind a command to the actual isolated source tree and newly built binaries."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import time

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--source-root', type=Path, required=True)
p.add_argument('--build', type=Path, required=True)
p.add_argument('--prefix', type=Path, required=True)
p.add_argument('command', nargs=argparse.REMAINDER)
a = p.parse_args()
repository = Path(__file__).resolve().parents[4]
source, build, prefix = a.source_root.resolve(), a.build.resolve(), a.prefix.resolve()
command = a.command[1:] if a.command[:1] == ['--'] else a.command
if not command: p.error('command required')
if Path(str(prefix) + '-identity.json').exists() or Path(str(prefix) + '.log').exists():
    p.error('fresh prefix required')
prefix.parent.mkdir(parents=True, exist_ok=True)
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
inputs = sorted(path for path in (source / 'die').rglob('*')
                if path.is_file() and '__pycache__' not in path.parts)
inputs += sorted(source.glob('*.docx')) + sorted(source.glob('*.md'))
inputs += sorted(source.glob('astra-sim-*/astra-sim/workload/CoarseDiePrediction.hh'))
source_hashes = {str(path.relative_to(source)): sha(path) for path in inputs}
def binaries():
    files = set(build.rglob('*.a')) | set(build.rglob('*.so'))
    files |= {path for path in build.rglob('coarse_*') if path.is_file() and not path.suffix}
    return {str(path): sha(path) for path in sorted(files)}
cpu = Path('/sys/fs/cgroup/cpu.stat')
meta = {'scope': 'actual isolated source command; engineering acceptance assessed separately',
    'observed_parent_git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repository, text=True).strip(),
    'command': command, 'cwd': str(source), 'source_tree_root': str(source),
    'build': str(build), 'source_sha256': source_hashes,
    'external_json_header_sha256': sha(repository / 'astra-sim-wscllm/extern/helper/json/json.hpp'),
    'recorder_sha256': sha(Path(__file__)), 'binaries_before_sha256': binaries(),
    'started_unix': time.time(), 'cpu_before': cpu.read_text()}
with Path(str(prefix) + '.log').open('w') as log:
    try:
        result = subprocess.run(command, cwd=source, stdout=log, stderr=subprocess.STDOUT)
    except OSError as error:
        log.write(str(error) + '\n')
        meta['launch_error'] = str(error)
        result = subprocess.CompletedProcess(command, 127)
meta.update(return_code=result.returncode, finished_unix=time.time(), cpu_after=cpu.read_text(),
    binaries_after_sha256=binaries(),
    source_unchanged=all(path.is_file() and sha(path) == source_hashes[str(path.relative_to(source))] for path in inputs))
if Path(command[0]).name == 'ctest':
    counts = re.findall(r'(\d+) tests? failed out of (\d+)', Path(str(prefix) + '.log').read_text())
    meta['ctest_counts'] = counts
    meta['tests_run'] = int(counts[-1][1]) if counts else 0
Path(str(prefix) + '-identity.json').write_text(json.dumps(meta, indent=2) + '\n')
print(json.dumps({k: meta[k] for k in ['command', 'return_code', 'source_unchanged', 'started_unix', 'finished_unix']}))
raise SystemExit(result.returncode if meta['source_unchanged'] else 1)
