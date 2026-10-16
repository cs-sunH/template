#!/usr/bin/env python3
"""Keep the first C4r immutable while preparing two independent review fixes."""
import difflib
import hashlib
import json
from pathlib import Path
import subprocess

root = Path(__file__).resolve().parents[4]
parent = root / 'work/runtime-resume-20261017/candidate-c4r-source/src/Execution.cc'
dest = root / 'work/runtime-resume-20261017/candidate-c4r-reviewed-final'
dest.mkdir(exist_ok=False)
s = parent.read_text()
before = '''    if(result<delta || (result==delta && strict)) {
      delta=result;strict_delta=strict;
    }'''
after = '''    if(result<delta) {
      delta=result;strict_delta=strict;
    } else if(strict && candidate==delta && candidate>0.0L && std::isfinite(candidate)) {
      strict_delta=true;
    }'''
assert s.count(before) == 1
s = s.replace(before, after)
before = '''      const long double next_progress=represented_progress_after(live,delta_ns);
      const long double delta_progress=next_progress-live.progress;
      if(delta_progress<=0.0L)continue;'''
after = '''      // Busy retains its original positive-service-rate semantics, even
      // when legacy progress quantizes to the same binary64 coordinate.
      if(static_cast<long double>(live.rate)*(delta_ns/kNsPerSecond)<=0.0L)continue;
      const long double next_progress=represented_progress_after(live,delta_ns);
      const long double delta_progress=next_progress-live.progress;'''
assert s.count(before) == 1
s = s.replace(before, after)
before = '  if (absolute_ns < now_ns_) throw std::invalid_argument("invalid_time: advance_to moved backwards");\n'
assert s.count(before) == 1
s = s.replace(before, before +
              '  if (absolute_ns == now_ns_ && time_until(absolute_ns) <= 0.0L) return;\n')
(dest / 'Execution.cc').write_text(s)
baseline = subprocess.check_output(['git', 'show', '6431f2d:die/src/Execution.cc'], cwd=root, text=True)
(dest / 'candidate.patch').write_text(''.join(difflib.unified_diff(
    baseline.splitlines(True), s.splitlines(True),
    fromfile='a/die/src/Execution.cc', tofile='b/die/src/Execution.cc')))
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
(dest / 'identity.json').write_text(json.dumps({
    'scope': 'review fixes: correct chosen-event tie; retain original busy at quantized-zero progress',
    'parent_sha256': sha(parent), 'source_sha256': sha(dest / 'Execution.cc'),
    'preparer_sha256': sha(Path(__file__)), 'baseline_sha256': hashlib.sha256(baseline.encode()).hexdigest(),
}, indent=2) + '\n')
print(dest)
