#!/usr/bin/env python3
"""Run frozen consumer build/test commands; keep every attempted step."""
import argparse
import json
import pathlib
import subprocess
import sys

p = argparse.ArgumentParser()
p.add_argument('--mode', choices=['old', 'preview'], required=True)
p.add_argument('--freeze', required=True)
p.add_argument('--label', required=True)
p.add_argument('--repos', nargs='+', required=True)
p.add_argument('--jobs', type=int, default=2)
a = p.parse_args()
root = pathlib.Path(__file__).resolve().parents[4]
freeze_path = pathlib.Path(a.freeze).resolve()
freeze = json.loads(freeze_path.read_text())
stage = pathlib.Path(freeze['stage_root'])
if a.jobs < 1 or pathlib.Path(a.label).name != a.label:
    p.error('positive jobs and a simple fresh label required')
evidence = root / 'die/docs/evidence/consumer-availability-20261016' / a.label
evidence.mkdir(parents=True, exist_ok=False)
results = {'mode': a.mode, 'freeze': str(freeze_path), 'runs': {}}
recorder = pathlib.Path(__file__).with_name('record_cpp_run.py')
for repo in a.repos:
    commands = freeze['commands'][repo]
    preview_build = pathlib.Path(commands['build'][2])
    build = (root / 'work/consumer-availability' / ('old-' + repo + '-build')
             if a.mode == 'old' else preview_build)
    source = root if a.mode == 'old' else stage
    records = results['runs'][repo] = {}
    for step in ['configure', 'build', 'ctest']:
        command = [x.replace(str(preview_build), str(build)).replace(str(stage), str(source))
                   for x in commands[step]]
        if step == 'build':
            command[command.index('--parallel') + 1] = str(a.jobs)
        prefix = evidence / (repo + '-' + step)
        recorded = [sys.executable, str(recorder), '--root', str(source),
                    '--prefix', str(prefix), '--build', str(build), '--', *command]
        result = subprocess.run(recorded, cwd=root)
        records[step] = {'command': command, 'return_code': result.returncode,
                         'identity': str(prefix) + '-identity.json'}
        (evidence / 'matrix.json').write_text(json.dumps(results, indent=2) + '\n')
        print(json.dumps({'mode': a.mode, 'repo': repo, 'step': step,
                          'return_code': result.returncode}), flush=True)
        if result.returncode != 0 and step != 'ctest':
            records['remaining_steps'] = 'not_run_after_' + step + '_failure'
            break
(evidence / 'matrix.json').write_text(json.dumps(results, indent=2) + '\n')
raise SystemExit(any(step.get('return_code', 0) != 0
                     for run in results['runs'].values()
                     for step in run.values() if isinstance(step, dict)))
