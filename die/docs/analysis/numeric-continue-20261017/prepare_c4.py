#!/usr/bin/env python3
"""Prepare C4 with strict clock rounding and pre-mutation tiny-step validation."""
import difflib
import hashlib
import json
from pathlib import Path
import shutil

root = Path(__file__).resolve().parents[4]
dest = root / 'work/runtime-resume-20261017/candidate-c4-source'
dest.mkdir(exist_ok=False)
for name in ('src', 'include'):
    shutil.copytree(root / 'die' / name, dest / name)
shutil.copy2(root / 'die/CMakeLists.txt', dest / 'CMakeLists.txt')
parent = root / 'work/runtime-resume-20261017/candidate-c3-source/src/Execution.cc'
source = parent.read_text()

def replace(before, after, count=1):
    global source
    assert source.count(before) == count, (before, source.count(before))
    source = source.replace(before, after)

replace('event_owner=has_credit_curves(*execution.plan)?&execution:nullptr;',
        'event_owner=&execution;')
replace('if (exact_ns_ <= kTimeEpsilonNs) return exact_origin_ns_;',
        'if (exact_ns_ <= 0.0L) return exact_origin_ns_;')
replace('      next_fraction > kTimeEpsilonNs)', '      next_fraction > 0.0L)')
replace('''        if(has_credit_curves(*execution.plan) && exact_ns_>0.0L &&
           execution.finish_ns==exact_origin_ns_) {
          if(exact_origin_ns_==std::numeric_limits<std::uint64_t>::max())
            throw std::overflow_error("overflow: positive curved terminal exceeds uint64 time");
          execution.finish_ns=exact_origin_ns_+1;
        }
''', '')
replace('''    bool changed=false;
    for(const auto& [handle,e]:executions_) {
      (void)handle;
      if((owner && owner!=&e) || e.terminal || time_until(e.ready_ns)>execution_time_tolerance(e))continue;
      for(const auto& a:e.activities) {
        if(a.epoch_index!=e.epoch || a.done || a.rate<=0 ||
           (selected && selected!=&a))continue;
        if(buffer<2 && curve_derivative(a)[buffer]==0)continue;
        if(!owner && a.definition.credit_curve.empty())continue;
        if(!(represented_progress_after(a,dt)>a.progress))
          throw std::runtime_error("numerical_failure: positive curved event cannot advance a participating represented physical progress");
        changed=true;
      }
    }
    if(!changed)
      throw std::runtime_error("numerical_failure: positive curved event cannot advance its represented physical progress");''', '''    bool changed=false;
    for(const auto& [handle,e]:executions_) {
      (void)handle;
      if(e.terminal || time_until(e.ready_ns)>0.0L ||
         e.epoch>=e.plan->epochs.size())continue;
      const auto& epoch=e.plan->epochs[e.epoch];
      std::array<long double,2> predicted{{epoch.credit_initial[0],epoch.credit_initial[1]}};
      for(const auto& a:e.activities) {
        if(a.epoch_index!=e.epoch)continue;
        LiveActivity next=a;
        if(!a.done && a.rate>0.0) {
          next.progress=represented_progress_after(a,dt);
          // A tiny step is unsupported if any positively served curved role
          // cannot represent its progress. Never mutate a different role first.
          if(!a.definition.credit_curve.empty() && !(next.progress>a.progress))
            throw std::runtime_error("numerical_failure: positive curved event cannot advance a participating represented physical progress");
          if((!owner || owner==&e) && (!selected || selected==&a) &&
             (buffer>=2 || curve_derivative(a)[buffer]!=0.0))
            changed=changed || next.progress>a.progress;
        }
        const auto contribution=accumulated_credit(next);
        for(std::size_t b=0;b<2;++b)predicted[b]+=contribution[b];
      }
      // Exactly the existing recompute_credit arithmetic/order and guard.
      // This validates represented service, not just continuous net rate.
      for(std::size_t b=0;b<2;++b) {
        const long double capacity=epoch.credit_capacity[b];
        const long double tolerance=credit_boundary_tolerance(e,b,
            static_cast<long double>(kCreditTolerance)*std::max(1.0L,capacity));
        if(!std::isfinite(predicted[b]) || predicted[b]<-tolerance ||
           predicted[b]>capacity+tolerance)
          throw std::runtime_error("numerical_failure: represented tiny service would cross a finite buffer boundary");
      }
    }
    if(!changed)
      throw std::runtime_error("numerical_failure: positive curved event cannot advance its represented physical progress");''')
(dest / 'src/Execution.cc').write_text(source)
baseline = (root / 'die/src/Execution.cc').read_text()
(dest / 'candidate.patch').write_text(''.join(difflib.unified_diff(
    baseline.splitlines(True), source.splitlines(True),
    fromfile='a/die/src/Execution.cc', tofile='b/die/src/Execution.cc')))
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
(dest / 'identity.json').write_text(json.dumps({
    'scope': 'isolated C4: not production/adoption or general numerical acceptance',
    'parent_source_sha256': sha(parent), 'baseline_source_sha256': sha(root / 'die/src/Execution.cc'),
    'files_sha256': {str(p.relative_to(dest)): sha(p) for p in sorted(dest.rglob('*')) if p.is_file()},
}, indent=2) + '\n')
print(dest)
