#!/usr/bin/env python3
"""Run identical CLI regression tests against preserved pre-fix/current tools."""
import argparse
import hashlib
import json
import pathlib
import shutil
import subprocess
import sys
import time

p = argparse.ArgumentParser()
p.add_argument('--mode', choices=['red', 'green'], required=True)
p.add_argument('--label', required=True)
p.add_argument('--library', required=True)
a = p.parse_args()
root = pathlib.Path(__file__).resolve().parents[4]
if pathlib.Path(a.label).name != a.label: p.error('simple fresh label required')
evidence = root / 'die/docs/evidence/consumer-availability-20261016/cli' / a.label
evidence.mkdir(parents=True, exist_ok=False)
stage = root / 'work/consumer-availability/cli' / a.label / 'die'
stage.mkdir(parents=True, exist_ok=False)
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
before = evidence.parent / 'estimate-before.py'
if before.read_bytes() != subprocess.check_output(['git', 'show', '9bb8c0a:die/tools/estimate.py'], cwd=root):
    raise RuntimeError('preserved pre-fix CLI does not match Git9bb')
(stage / 'tools').mkdir(); (stage / 'tests').mkdir()
shutil.copyfile(before if a.mode == 'red' else root / 'die/tools/estimate.py', stage / 'tools/estimate.py')
shutil.copyfile(root / 'die/tests/estimate_cli_test.py', stage / 'tests/estimate_cli_test.py')
shutil.copytree(root / 'die/configs', stage / 'configs')
library = pathlib.Path(a.library).resolve()
command = [sys.executable, str(stage / 'tests/estimate_cli_test.py'), '--library', str(library), '-v']
meta = {'mode': a.mode, 'command': command, 'cwd': str(root), 'started_unix': time.time(),
        'library': str(library), 'library_sha256': sha(library),
        'runner_sha256': sha(pathlib.Path(__file__)),
        'source_sha256': {str(path.relative_to(stage)): sha(path) for path in sorted(stage.rglob('*')) if path.is_file()}}
with (evidence / 'run.log').open('w') as log:
    result = subprocess.run(command, cwd=root, stdout=log, stderr=subprocess.STDOUT)
meta.update(return_code=result.returncode, finished_unix=time.time(), log_sha256=sha(evidence / 'run.log'))
(evidence / 'identity.json').write_text(json.dumps(meta, indent=2) + '\n')
print(json.dumps({'mode': a.mode, 'return_code': result.returncode, 'evidence': str(evidence)}))
raise SystemExit(result.returncode)
