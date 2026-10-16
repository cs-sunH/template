#!/usr/bin/env python3
"""Prepare a separate C3 numerical candidate; never install it in production."""
import difflib
import hashlib
import json
from pathlib import Path
import shutil

root = Path(__file__).resolve().parents[4]
dest = root / 'work/runtime-resume-20261017/candidate-c3-source'
dest.mkdir(exist_ok=False)
for name in ('src', 'include'):
    shutil.copytree(root / 'die' / name, dest / name)
shutil.copy2(root / 'die/CMakeLists.txt', dest / 'CMakeLists.txt')
parent = root / 'work/runtime-positive-time-v2/Execution.cc'
source = parent.read_text()
before = '''long double execution_time_tolerance(const Execution& execution) {
  return has_credit_curves(*execution.plan)?0.0L:kTimeEpsilonNs;
}'''
assert source.count(before) == 1
source = source.replace(before, '''// Readiness is an exact external integer boundary for every Plan, including
// legacy activities competing with a curved execution.
long double execution_time_tolerance(const Execution&) { return 0.0L; }''')
before = '''        changed=changed || represented_progress_after(a,dt)>a.progress;'''
assert source.count(before) == 1
source = source.replace(before, '''        if(!(represented_progress_after(a,dt)>a.progress))
          throw std::runtime_error("numerical_failure: positive curved event cannot advance a participating represented physical progress");
        changed=true;''')
(dest / 'src/Execution.cc').write_text(source)
baseline = (root / 'die/src/Execution.cc').read_text()
(dest / 'candidate.patch').write_text(''.join(difflib.unified_diff(
    baseline.splitlines(True), source.splitlines(True),
    fromfile='a/die/src/Execution.cc', tofile='b/die/src/Execution.cc')))
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
(dest / 'identity.json').write_text(json.dumps({
    'scope': 'isolated C3, exact readiness and all selected credit participants; not adoption',
    'parent_source_sha256': sha(parent),
    'baseline_source_sha256': sha(root / 'die/src/Execution.cc'),
    'files_sha256': {str(p.relative_to(dest)): sha(p) for p in sorted(dest.rglob('*')) if p.is_file()},
}, indent=2) + '\n')
print(dest)
