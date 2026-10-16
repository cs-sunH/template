#!/usr/bin/env python3
"""Compile an independent probe once, then run against explicit fresh libraries."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--source', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--libraries', nargs='+', required=True, help='label=archive')
args = parser.parse_args()
root = Path(__file__).resolve().parents[4]
dest = args.output.resolve()
dest.mkdir(parents=True, exist_ok=False)
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
source = args.source.resolve()
record = {'scope': 'fresh explicit-library synthetic numerical probe; no hardware calibration',
          'source': str(source), 'source_sha256': sha(source),
          'runner_sha256': sha(Path(__file__)), 'steps': [],
          'header_sha256': {str(p.relative_to(root)): sha(p) for directory in
                           ('die/include', 'die/tests') for p in (root / directory).rglob('*.hh')}}

def run(label, command):
    command = list(map(str, command))
    row = {'label': label, 'command': command, 'started_unix': time.time()}
    record['steps'].append(row)
    with (dest / (label + '.log')).open('w') as log:
        try:
            row['return_code'] = subprocess.run(command, cwd=root, stdout=log,
                                                stderr=subprocess.STDOUT, timeout=120).returncode
        except (OSError, subprocess.TimeoutExpired) as error:
            row.update(status='not_completed', error=str(error))
    row.update(finished_unix=time.time(), log_sha256=sha(dest / (label + '.log')))
    (dest / 'identity.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps({k: row.get(k) for k in ('label', 'return_code', 'status')}), flush=True)
    return row.get('return_code') == 0

obj = dest / 'probe.o'
if not run('compile', ['g++', '-std=c++17', '-O2', '-Wall', '-Wextra', '-Wpedantic',
                     '-I' + str(root / 'die/include'), '-I' + str(root / 'die/tests'),
                     '-I' + str(root / 'astra-sim-wscllm/extern/helper'),
                     '-DCOARSE_DIE_SOURCE_DIR="' + str(root / 'die') + '"',
                     '-DCOARSE_DIE_ROOT_DIR="' + str(root) + '"',
                     '-c', source, '-o', obj]):
    raise SystemExit(2)
record['object_sha256'] = sha(obj)
passed = True
for item in args.libraries:
    label, path = item.split('=', 1)
    library = Path(path).resolve()
    before = sha(library)
    exe = dest / (label + '-probe')
    record.setdefault('libraries', {})[label] = {'path': str(library), 'sha256': before}
    linked = run(label + '-link', ['g++', obj, library, '-o', exe])
    if linked:
        record['libraries'][label]['executable_sha256'] = sha(exe)
        passed = run(label, [exe, dest / label, label]) and passed
    else:
        passed = False
    record['libraries'][label]['unchanged'] = sha(library) == before
    passed = passed and record['libraries'][label]['unchanged']
record['source_unchanged'] = sha(source) == record['source_sha256']
record['passed'] = passed and record['source_unchanged']
(dest / 'identity.json').write_text(json.dumps(record, indent=2) + '\n')
raise SystemExit(not record['passed'])
