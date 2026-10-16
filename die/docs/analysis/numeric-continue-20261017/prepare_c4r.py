#!/usr/bin/env python3
"""Prepare the reviewed C4 revision in scratch; installation is explicit."""
import difflib
import hashlib
import json
from pathlib import Path
import shutil

root = Path(__file__).resolve().parents[4]
dest = root / 'work/runtime-resume-20261017/candidate-c4r-source'
dest.mkdir(exist_ok=False)
for name in ('src', 'include'):
    shutil.copytree(root / 'die' / name, dest / name)
shutil.copy2(root / 'die/CMakeLists.txt', dest / 'CMakeLists.txt')
parent = root / 'work/runtime-resume-20261017/candidate-c4-source/src/Execution.cc'
source = parent.read_text()

def replace(before, after, count=1):
    global source
    assert source.count(before) == count, (before, source.count(before))
    source = source.replace(before, after)

replace('''// Readiness is an exact external integer boundary for every Plan, including
// legacy activities competing with a curved execution.
long double execution_time_tolerance(const Execution&) { return 0.0L; }
''', '')
replace('execution_time_tolerance(execution)', '0.0L', 7)
replace('execution_time_tolerance(e)', '0.0L', 1)
replace('    if(buffer==3)return; // A real ready boundary needs clock, not premature service.\n', '')
replace('''    if(!changed)
      throw std::runtime_error("numerical_failure: positive curved event cannot advance its represented physical progress");''',
        '''    if(buffer!=3 && !changed)
      throw std::runtime_error("numerical_failure: positive curved event cannot advance its represented physical progress");''')
replace('next_delta > remaining_to_target + (strict_time?0.0L:kTimeEpsilonNs)',
        'next_delta > remaining_to_target')
begin = source.index('      long double delta_progress = static_cast<long double>(live.rate) * seconds;')
end = source.index('      live.progress = next_progress;', begin)
source = source[:begin] + '''      const long double next_progress=represented_progress_after(live,delta_ns);
      const long double delta_progress=next_progress-live.progress;
      if(delta_progress<=0.0L)continue;
''' + source[end:]
replace('  const long double seconds = delta_ns / kNsPerSecond;\n', '')
replace('// ISOLATED POSITIVE-TIME EXPERIMENT. No credit or allocation rule changes.',
        '// Curved service retains every positive event. Validate representability\n// before a tiny integration; credit and arbitration tolerances are unchanged.')
# Keep public event rounding consistent with the selected mathematical event.
begin = source.index('std::optional<std::uint64_t> Runtime::next_event() const {')
end = source.index('\nnlohmann::json Runtime::query_progress', begin)
part = source[begin:end]
part = part.replace('  long double delta = std::numeric_limits<long double>::infinity();', '''  long double delta = std::numeric_limits<long double>::infinity();
  bool strict_delta=false;
  const auto choose=[&](long double candidate,bool strict) {
    const long double result=min_positive(delta,candidate,strict);
    if(result<delta || (result==delta && strict)) {
      delta=result;strict_delta=strict;
    }
  };''')
part = part.replace('delta = std::min(delta, time_until(execution.ready_ns));',
                    'choose(time_until(execution.ready_ns),true);')
part = part.replace('delta = min_positive(delta, remain / live.rate * kNsPerSecond,!live.definition.credit_curve.empty());',
                    'choose(remain/live.rate*kNsPerSecond,!live.definition.credit_curve.empty());')
part = part.replace('delta = min_positive(delta, (capacity-credit)/rate*kNsPerSecond,epoch_has_curves(execution));',
                    'choose((capacity-credit)/rate*kNsPerSecond,epoch_has_curves(execution));')
part = part.replace('delta = min_positive(delta, credit/(-rate)*kNsPerSecond,epoch_has_curves(execution));',
                    'choose(credit/(-rate)*kNsPerSecond,epoch_has_curves(execution));')
part = part.replace('std::ceil(delta - kTimeEpsilonNs)', 'std::ceil(delta - (strict_delta?0.0L:kTimeEpsilonNs))')
source = source[:begin] + part + source[end:]
assert 'execution_time_tolerance' not in source
(dest / 'src/Execution.cc').write_text(source)
baseline = (root / 'die/src/Execution.cc').read_text()
(dest / 'candidate.patch').write_text(''.join(difflib.unified_diff(
    baseline.splitlines(True), source.splitlines(True),
    fromfile='a/die/src/Execution.cc', tofile='b/die/src/Execution.cc')))
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
(dest / 'identity.json').write_text(json.dumps({
    'scope': 'C4 reviewed revision, pending fresh all-source verification; not adoption',
    'parent_source_sha256': sha(parent), 'baseline_source_sha256': sha(root / 'die/src/Execution.cc'),
    'files_sha256': {str(p.relative_to(dest)): sha(p) for p in sorted(dest.rglob('*')) if p.is_file()},
}, indent=2) + '\n')
print(dest)
