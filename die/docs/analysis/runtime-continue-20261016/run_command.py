#!/usr/bin/env python3
"""Record actual source/binary identity and unfiltered command output."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--root', type=Path, default=Path.cwd())
p.add_argument('--build', type=Path, required=True)
p.add_argument('--prefix', type=Path, required=True)
p.add_argument('command', nargs=argparse.REMAINDER)
a = p.parse_args()
root, build, prefix = a.root.resolve(), a.build.resolve(), a.prefix.resolve()
command = a.command[1:] if a.command[:1] == ['--'] else a.command
if not command or prefix.with_suffix('.json').exists() or prefix.with_suffix('.log').exists():
    p.error('command and fresh output prefix required')
prefix.parent.mkdir(parents=True, exist_ok=True)
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
inputs = [root / 'die/CMakeLists.txt']
for directory in ('src', 'include', 'configs', 'tests', 'tools', 'validation'):
    inputs += [f for f in (root / 'die' / directory).rglob('*')
               if f.is_file() and '__pycache__' not in f.parts]
inputs += list(root.glob('*.docx')) + list(root.glob('*最粗粒度*.md'))
inputs += list(root.glob('astra-sim-*/astra-sim/workload/CoarseDiePrediction.hh'))
inputs += [root / 'astra-sim-wscllm/extern/helper/json/json.hpp']
sources = {str(f.relative_to(root)): sha(f) for f in sorted(inputs)}
def binaries():
    files = set(build.rglob('*.a')) | set(build.rglob('*.so'))
    files |= {f for f in build.rglob('coarse_*') if f.is_file() and not f.suffix}
    return {str(f.relative_to(build)): sha(f) for f in sorted(files)}
cpu = Path('/sys/fs/cgroup/cpu.stat')
meta = dict(command=command, cwd=str(root), build=str(build),
            git_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(),
            source_sha256=sources, recorder_sha256=sha(Path(__file__)),
            binaries_before=binaries(), started_unix=time.time(), cpu_before=cpu.read_text())
with prefix.with_suffix('.log').open('w') as log:
    try:
        result = subprocess.run(command, cwd=root, stdout=log, stderr=subprocess.STDOUT)
        meta['return_code'] = result.returncode
    except OSError as error:
        log.write(str(error) + '\n')
        meta.update(return_code=127, launch_error=str(error))
meta.update(finished_unix=time.time(), cpu_after=cpu.read_text(), binaries_after=binaries(),
            source_unchanged=all(f.is_file() and sha(f) == sources[str(f.relative_to(root))] for f in inputs))
prefix.with_suffix('.json').write_text(json.dumps(meta, indent=2) + '\n')
print(json.dumps({k: meta[k] for k in ('command', 'return_code', 'source_unchanged', 'started_unix', 'finished_unix')}))
raise SystemExit(meta['return_code'] if meta['source_unchanged'] else 1)
