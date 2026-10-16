#!/usr/bin/env python3
"""Prepare a separate time-only candidate with exact curved readiness."""
import difflib
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parents[4]
old = root / 'work/runtime-positive-time/Execution.cc'
out = root / 'work/runtime-positive-time-v2'
out.mkdir(parents=True, exist_ok=False)
original, s = old.read_text(), old.read_text()
def replace(before, after, count=1):
    global s
    assert s.count(before) == count, (before, s.count(before), count)
    s = s.replace(before, after)
replace('long double represented_progress_after(', '''long double execution_time_tolerance(const Execution& execution) {
  return has_credit_curves(*execution.plan)?0.0L:kTimeEpsilonNs;
}
long double represented_progress_after(''')
count = s.count('time_until(execution.ready_ns) > kTimeEpsilonNs')
assert count == 7, count
replace('time_until(execution.ready_ns) > kTimeEpsilonNs',
        'time_until(execution.ready_ns) > execution_time_tolerance(execution)', count)
replace('time_until(e.ready_ns)>kTimeEpsilonNs', 'time_until(e.ready_ns)>execution_time_tolerance(e)', 2)
replace('  const auto curved_service_live=[&]() {', '''  const auto curved_time_needed=[&]() {
    return std::any_of(executions_.begin(),executions_.end(),[](const auto& item) {
      return !item.second.terminal && has_credit_curves(*item.second.plan);
    });
  };
  const auto curved_service_live=[&]() {''')
replace('    bool changed=false;\n    for(const auto& [handle,e]:executions_) {',
        '    if(buffer==3)return; // A real ready boundary needs clock, not premature service.\n    bool changed=false;\n    for(const auto& [handle,e]:executions_) {')
replace('(curved_service_live()?0.0L:kTimeEpsilonNs)', '(curved_time_needed()?0.0L:kTimeEpsilonNs)', 2)
replace('next_delta=time_until(execution.ready_ns);event_owner=nullptr;\n          event_activity=nullptr;event_buffer=2;',
        'next_delta=time_until(execution.ready_ns);\n          event_owner=has_credit_curves(*execution.plan)?&execution:nullptr;\n          event_activity=nullptr;event_buffer=3;')
replace('const bool strict_time=curved_service_live();', 'const bool strict_time=curved_time_needed();')
replace('if(strict_time)preflight_tiny(remaining_to_target,nullptr,nullptr,2);',
        'if(strict_time)preflight_tiny(remaining_to_target,nullptr,nullptr,curved_service_live()?2:3);')
candidate = out / 'Execution.cc'
candidate.write_text(s)
baseline = root / 'die/src/Execution.cc'
(out / 'time-only-v2.patch').write_text(''.join(difflib.unified_diff(
    baseline.read_text().splitlines(True), s.splitlines(True), fromfile='a/die/src/Execution.cc', tofile='b/die/src/Execution.cc')))
(out / 'prepare-identity.json').write_text(json.dumps(dict(scope='isolated time-only v2; not production/adoption',
    parent_candidate_source_sha256=hashlib.sha256(old.read_bytes()).hexdigest(),
    baseline_source_sha256=hashlib.sha256(baseline.read_bytes()).hexdigest(),
    candidate_source_sha256=hashlib.sha256(candidate.read_bytes()).hexdigest(),
    preparer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    parent_candidate_unchanged=old.read_text()==original), indent=2)+'\n')
print(str(out))
