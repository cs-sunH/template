#!/usr/bin/env python3
"""Prepare an isolated positive-time experiment on A+B, never production.

This deliberately retains all credit/LP/witness/closure rules. It is a research
candidate until actual regressions, representation cases and budgets pass.
"""
import difflib
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parents[4]
out = root / 'work/runtime-positive-time'
out.mkdir(parents=True, exist_ok=False)
source = root / 'die/src/Execution.cc'
original = source.read_text()
assert hashlib.sha256(source.read_bytes()).hexdigest() == '6fcc637e13c876d240d92d8677e8f9d2e323b325aea46176848063ffe93e710e'
s = original
def replace(before, after, count=1):
    global s
    assert s.count(before) == count, (before, s.count(before), count)
    s = s.replace(before, after)
replace('long double min_positive(long double current, long double candidate) {\n  if (!std::isfinite(candidate) || candidate <= kTimeEpsilonNs) return current;',
        'long double min_positive(long double current, long double candidate, bool curved=false) {\n  if (!std::isfinite(candidate) || candidate <= (curved ? 0.0L : kTimeEpsilonNs)) return current;')
helper = '''
// ISOLATED POSITIVE-TIME EXPERIMENT. No credit or allocation rule changes.
bool epoch_has_curves(const Execution& execution) {
  if(execution.epoch>=execution.plan->epochs.size())return false;
  const auto& epoch=execution.plan->epochs[execution.epoch];
  return std::any_of(epoch.activities.begin(),epoch.activities.end(),
      [](const Activity& a){return !a.credit_curve.empty();});
}
long double represented_progress_after(const LiveActivity& live,long double delta_ns) {
  long double delta=static_cast<long double>(live.rate)*(delta_ns/kNsPerSecond);
  if(delta<=0)return live.progress;
  const double boundary=curve_boundary(live);
  const long double remaining=static_cast<long double>(boundary)-live.progress;
  const bool curved=!live.definition.credit_curve.empty();
  if(curved ? delta_ns>=remaining/live.rate*kNsPerSecond : delta>=remaining-1.0e-14L)
    delta=remaining;
  if(delta<=0)return live.progress;
  return curved ? (delta>=remaining ? static_cast<long double>(boundary)
                  : std::min(static_cast<long double>(boundary),live.progress+delta))
                : static_cast<long double>(std::min(boundary,
                    static_cast<double>(live.progress)+static_cast<double>(delta)));
}
'''
replace('bool same_or_one_ulp(double saved, double rebuilt) {', helper+'\nbool same_or_one_ulp(double saved, double rebuilt) {')
replace('execution.finish_ns = ceil_exact_time();', '''execution.finish_ns = ceil_exact_time();
        if(has_credit_curves(*execution.plan) && exact_ns_>0.0L &&
           execution.finish_ns==exact_origin_ns_) {
          if(exact_origin_ns_==std::numeric_limits<std::uint64_t>::max())
            throw std::overflow_error("overflow: positive curved terminal exceeds uint64 time");
          execution.finish_ns=exact_origin_ns_+1;
        }''')
replace('void Runtime::advance_to(std::uint64_t absolute_ns) {', '''void Runtime::advance_to(std::uint64_t absolute_ns) {
  const auto curved_service_live=[&]() {
    for(const auto& [handle,e]:executions_) {
      (void)handle;
      if(e.terminal || time_until(e.ready_ns)>kTimeEpsilonNs)continue;
      for(const auto& a:e.activities)
        if(a.epoch_index==e.epoch && !a.done && a.rate>0 &&
           !a.definition.credit_curve.empty())return true;
    }
    return false;
  };
  // Preflight happens before any counters/state mutate. Only tiny intervals
  // need this representability check; they must not turn into no-service loops.
  const auto preflight_tiny=[&](long double dt,const Execution* owner,
                                const LiveActivity* selected,std::size_t buffer) {
    if(!(dt>0 && dt<=kTimeEpsilonNs))return;
    const long double fraction=exact_ns_+dt;
    const auto carry=static_cast<std::uint64_t>(std::floor(fraction));
    const long double next_fraction=fraction-carry;
    if((carry==0 && next_fraction==exact_ns_) ||
       (exact_origin_ns_==std::numeric_limits<std::uint64_t>::max() &&
        (carry>0 || next_fraction>0)))
      throw std::runtime_error("numerical_failure: positive curved event cannot advance represented clock");
    bool changed=false;
    for(const auto& [handle,e]:executions_) {
      (void)handle;
      if((owner && owner!=&e) || e.terminal || time_until(e.ready_ns)>kTimeEpsilonNs)continue;
      for(const auto& a:e.activities) {
        if(a.epoch_index!=e.epoch || a.done || a.rate<=0 ||
           (selected && selected!=&a))continue;
        if(buffer<2 && curve_derivative(a)[buffer]==0)continue;
        if(!owner && a.definition.credit_curve.empty())continue;
        changed=changed || represented_progress_after(a,dt)>a.progress;
      }
    }
    if(!changed)
      throw std::runtime_error("numerical_failure: positive curved event cannot advance its represented physical progress");
  };''')
replace('if (absolute_ns == now_ns_ && time_until(absolute_ns) <= kTimeEpsilonNs) return;',
        'if (absolute_ns == now_ns_ && time_until(absolute_ns) <= (curved_service_live()?0.0L:kTimeEpsilonNs)) return;')
replace('while (time_until(absolute_ns) > kTimeEpsilonNs) {',
        'while (time_until(absolute_ns) > (curved_service_live()?0.0L:kTimeEpsilonNs)) {')
replace('long double next_delta = std::numeric_limits<long double>::infinity();', '''long double next_delta = std::numeric_limits<long double>::infinity();
    const Execution* event_owner=nullptr;
    const LiveActivity* event_activity=nullptr;
    std::size_t event_buffer=2;
    const auto choose=[&](long double candidate,bool curved,const Execution* owner,
                           const LiveActivity* activity,std::size_t buffer) {
      const long double result=min_positive(next_delta,candidate,curved);
      if(result<next_delta) {
        next_delta=result;event_owner=curved?owner:nullptr;
        event_activity=curved?activity:nullptr;event_buffer=buffer;
      }
    };''')
replace('next_delta = std::min(next_delta, time_until(execution.ready_ns));', '''if(time_until(execution.ready_ns)<next_delta) {
          next_delta=time_until(execution.ready_ns);event_owner=nullptr;
          event_activity=nullptr;event_buffer=2;
        }''')
replace('next_delta = min_positive(next_delta, remain / live.rate * kNsPerSecond);',
        'choose(remain / live.rate * kNsPerSecond,!live.definition.credit_curve.empty(),&execution,&live,2);')
replace('next_delta = min_positive(next_delta, (capacity - credit) / rate * kNsPerSecond);',
        'choose((capacity-credit)/rate*kNsPerSecond,epoch_has_curves(execution),&execution,nullptr,buffer);')
replace('next_delta = min_positive(next_delta, credit / (-rate) * kNsPerSecond);',
        'choose(credit/(-rate)*kNsPerSecond,epoch_has_curves(execution),&execution,nullptr,buffer);')
replace('const long double remaining_to_target = time_until(absolute_ns);\n    if (!std::isfinite(next_delta) || next_delta > remaining_to_target + kTimeEpsilonNs) {',
        'const long double remaining_to_target = time_until(absolute_ns);\n    const bool strict_time=curved_service_live();\n    if (!std::isfinite(next_delta) || next_delta > remaining_to_target + (strict_time?0.0L:kTimeEpsilonNs)) {')
replace('      integrate(remaining_to_target);',
        '      if(strict_time)preflight_tiny(remaining_to_target,nullptr,nullptr,2);\n      integrate(remaining_to_target);')
replace('    if (next_delta <= kTimeEpsilonNs) {', '    if (next_delta <= (event_owner?0.0L:kTimeEpsilonNs)) {')
replace('    integrate(next_delta);',
        '    if(event_owner)preflight_tiny(next_delta,event_owner,event_activity,event_buffer);\n    integrate(next_delta);')
replace('delta = min_positive(delta, remain / live.rate * kNsPerSecond);',
        'delta = min_positive(delta, remain / live.rate * kNsPerSecond,!live.definition.credit_curve.empty());')
replace('delta = min_positive(delta, (capacity - credit) / rate * kNsPerSecond);',
        'delta = min_positive(delta, (capacity-credit)/rate*kNsPerSecond,epoch_has_curves(execution));')
replace('delta = min_positive(delta, credit / (-rate) * kNsPerSecond);',
        'delta = min_positive(delta, credit/(-rate)*kNsPerSecond,epoch_has_curves(execution));')
candidate = out / 'Execution.cc'
candidate.write_text(s)
(out / 'time-only.patch').write_text(''.join(difflib.unified_diff(original.splitlines(True),s.splitlines(True),
                                              fromfile='a/die/src/Execution.cc',tofile='b/die/src/Execution.cc')))
(out / 'prepare-identity.json').write_text(json.dumps(dict(scope='isolated time-only research; not adoption',
    baseline_source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
    candidate_source_sha256=hashlib.sha256(candidate.read_bytes()).hexdigest(),
    preparer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    source_unchanged=source.read_text()==original),indent=2)+'\n')
print(str(out))
