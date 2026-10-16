#!/usr/bin/env python3
"""Report every public output difference; never silently approve changed fields."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import subprocess

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--baseline', type=Path, required=True)
p.add_argument('--candidate', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
if a.output.exists():
    p.error('fresh output required')
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
old, new = [json.loads(gzip.decompress(path.read_bytes())) for path in (a.baseline, a.candidate)]

def differences(x, y, path=''):
    if type(x) != type(y):
        return [{'path': path, 'before': x, 'after': y}]
    if isinstance(x, dict):
        return [d for key in sorted(x.keys() | y.keys()) for d in
                ([{'path': path + '/' + key, 'before': x.get(key, '<missing>'),
                   'after': y.get(key, '<missing>')}] if key not in x or key not in y else
                 differences(x[key], y[key], path + '/' + key))]
    if isinstance(x, list):
        if len(x) != len(y):
            return [{'path': path + '/length', 'before': len(x), 'after': len(y)}]
        return [d for i, (u, v) in enumerate(zip(x, y)) for d in differences(u, v, path + '/' + str(i))]
    return [] if x == y else [{'path': path, 'before': x, 'after': y}]

symbols = []
for capture in (old, new):
    library = Path(capture['library'])
    assert sha(library) == capture['library_sha256'], 'captured actual library changed'
    symbols.append(sorted(line.split()[-1] for line in subprocess.check_output(
        ['nm', '-D', '--defined-only', library], text=True).splitlines()
        if line.split()[-1].startswith('die_coarse_')))
diff = differences(old['results'], new['results'])
report = {'scope': 'all actual public fields and C ABI symbols; no accuracy/performance acceptance',
          'capture_sha256': {str(path): sha(path) for path in (a.baseline, a.candidate)},
          'differences': diff, 'symbols': {'baseline': symbols[0], 'candidate': symbols[1]},
          'passed': not diff and symbols[0] == symbols[1]}
a.output.write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps({'differences': len(diff), 'identical_symbols': symbols[0] == symbols[1],
                  'passed': report['passed']}))
raise SystemExit(not report['passed'])
