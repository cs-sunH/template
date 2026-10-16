#!/usr/bin/env python3
"""Build current-source allocation probes; these timings are never budget tests."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--samples', type=int, default=5)
a = p.parse_args()
root = Path(__file__).resolve().parents[4]
output = a.output.resolve()
if output.exists() or not output.is_relative_to(root / 'work'): p.error('fresh repository work/ output required')
if not 1 <= a.samples <= 1000: p.error('samples must be in [1,1000]')
output.mkdir(parents=True)
relative_probe = Path('die/docs/analysis/hotpath-allocation-20261016/phase_profile.cc')
lab = root / 'work/core-perf-lab'
destination = lab / relative_probe
if destination.exists():
    if destination.read_bytes() != (root / relative_probe).read_bytes(): p.error('lab probe differs')
else:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(root / relative_probe, destination)
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
rows = []
for name, source, build in [('baseline', root, root / 'work/build-core-release'),
                            ('candidate', lab, lab / 'build')]:
    inputs = sorted(path for folder in ['src', 'include', 'configs', 'tests']
                    for path in (source / 'die' / folder).rglob('*') if path.is_file())
    inputs += [source / relative_probe, build / 'libcoarse_die.a', build / 'lib/libcoarse_die_cabi.so',
               root / 'astra-sim-wscllm/extern/helper/json/json.hpp']
    before = {str(path): sha(path) for path in inputs}
    binary = output / (name + '-phase-profile')
    command = ['/usr/bin/g++', '-O3', '-DNDEBUG', '-std=c++17',
               '-I' + str(source / 'die/include'), '-I' + str(root / 'astra-sim-wscllm/extern/helper'),
               '-DCOARSE_DIE_SOURCE_DIR="' + str(source / 'die') + '"',
               '-DCOARSE_DIE_ROOT_DIR="' + str(source) + '"', str(source / relative_probe),
               str(build / 'libcoarse_die.a'), str(build / 'lib/libcoarse_die_cabi.so'),
               '-Wl,-rpath,' + str(build / 'lib'), '-pthread', '-o', str(binary)]
    commands = [('build', command)]
    for config in ['example_hardware.json', 'tx8_compute_matched_hardware.json']:
        commands.append((config, [str(binary), '--samples', str(a.samples), '--config',
            str(source / 'die/configs' / config), '--output', str(output / (name + '-' + config))]))
    for label, argv in commands:
        prefix = output / (name + '-' + label)
        started = time.time()
        with Path(str(prefix) + '.log').open('w') as log:
            result = subprocess.run(argv, cwd=source, stdout=log, stderr=subprocess.STDOUT)
        row = {'scope': 'instrumented allocation and thread-CPU diagnosis; not performance acceptance; concurrent functional build/testing allowed',
               'name': name, 'label': label, 'command': argv, 'cwd': str(source),
               'return_code': result.returncode, 'started_unix': started, 'finished_unix': time.time(),
               'observed_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(),
               'input_sha256': before, 'inputs_unchanged': all(sha(path) == before[str(path)] for path in inputs),
               'binary_sha256': sha(binary) if binary.exists() else None,
               'tool_sha256': sha(Path(__file__)), 'log_sha256': sha(Path(str(prefix) + '.log'))}
        Path(str(prefix) + '-identity.json').write_text(json.dumps(row, indent=2) + '\n')
        rows.append(row)
        (output / 'summary.json').write_text(json.dumps(rows, indent=2) + '\n')
        print(json.dumps({key: row[key] for key in ['name', 'label', 'return_code', 'inputs_unchanged']}), flush=True)
        if result.returncode or not row['inputs_unchanged']: raise SystemExit(1)
