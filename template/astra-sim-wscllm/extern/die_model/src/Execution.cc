/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

/*
Execution.cc -- the externally driven plan-advancement state machine
(contract §4 action protocol; design doc §6.3/§6.4/§8.1).

The machine is the "externally driven version" of the isolated list schedule
behind estimate_alone() (detail::evaluate_stages in CostModel.cc).  Semantics
that are shared verbatim with evaluate_stages (frozen in the stage A2 report):

  - flat (CSR) instance graph with instance id = offset[stage] + j; every
    DieStageDep reduces to at most one dependee instance (lag kAllDep = the
    predecessor's LAST instance, lag l > 0 = instance j-l, lag 0 = instance j);
  - Kahn list schedule; ready instances pop in (stage<<32)|instance key order,
    which keeps same-slot instances serialized in index order;
  - four resource slots PE/VEC/NOC/MEM conservatively serialize same-class
    stages; a MEM_NOC instance holds BOTH the MEM and the NOC slot from its
    start until the later of its two legs;
  - a stage's start = max(dependee ends, its slot's free time); internal
    stages (TIMER/PE/VEC/NOC) may legitimately be scheduled with a start in
    the future of the clock (a busy slot), exactly like the single pass.

What the external driver provides (design doc §8.1): the clock
(advance_time) and memory completion events (complete_memory).  Internal
events are timed by the machine itself: stage durations come from the same
detail::stage_instance_seconds() truth, converted once to integer ns.

Time base (design doc §6.4): the internal clock is uint64 nanoseconds.
Every positive stage duration is reserved as max(1, ceil(seconds*1e9)); the
accumulated rounding error is therefore bounded by (#positive reservations +
#memory links) x 1 ns, since each ceil adds strictly less than 1 ns per
reservation and each external memory link contributes one issue-time ceil
plus one service-duration ceil.  True zero-work durations reserve 0 ns and
complete synchronously inside the scheduling pass, so no zero-time event
chain is ever produced and next_action() always makes progress.

Memory protocol: at most ONE outstanding request per execution.  A MEM or
MEM_NOC instance is issued as a MemoryRequest only when it is the min-key
ready instance AND its computed start time has been reached by the clock;
the request carries the instance's DieStage::mem triple.  The next request
is issued only after complete_memory() for the current one.  A MEM_NOC
instance ends at max(external memory completion, start + NoC leg), and both
slots stay occupied until that end.

Pop discipline: the machine never pops past the min-key instance.  While a
request is outstanding, or while the min-key instance is a memory stage
whose start lies in the future, nothing else is popped; this preserves the
single-pass Kahn pop order (and therefore the slot assignment order) of
evaluate_stages under full-speed memory service.
*/

#include "DieModel.hh"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

namespace DieModel {

namespace {

// max(1, ceil(seconds*1e9)) for positive durations, 0 for zero work (§6.4).
std::uint64_t seconds_to_ns(double seconds) {
  if (!(seconds > 0.0)) {
    return 0;
  }
  const double ns = std::ceil(seconds * 1e9);
  if (ns < 1.0) {
    return 1;
  }
  if (ns >= 18446744073709549568.0) {  // 2^64 rounded down to a safe bound
    throw DieModelError(
        "Execution: stage duration exceeds the uint64 nanosecond range");
  }
  return static_cast<std::uint64_t>(ns);
}

}  // namespace

struct Execution::Impl {
  // --- immutable, built once in the constructor ---
  std::shared_ptr<const DiePlan> plan;
  const std::vector<DieStage>* stages = nullptr;
  const Hardware* hw = nullptr;
  std::vector<std::uint64_t> offset;         // stage -> flat instance range
  std::vector<std::uint32_t> stage_of;       // flat id -> stage index
  std::vector<std::uint32_t> dep_start;      // CSR dependees (per instance)
  std::vector<std::uint32_t> dep_to;
  std::vector<std::uint32_t> rdep_start;     // CSR reverse edges
  std::vector<std::uint32_t> rdep_to;
  std::vector<std::uint32_t> unresolved;     // unfinished dependees
  std::vector<std::uint64_t> dur_ns;         // internal stages (0 for MEM*)
  std::vector<std::uint64_t> noc_leg_ns;     // MEM_NOC NoC leg
  std::size_t total = 0;

  // --- scheduling state ---
  std::vector<std::uint64_t> heap;           // min-heap of (stage<<32)|inst
  std::vector<std::pair<std::uint64_t, std::uint32_t>> events;  // (end,id)
  std::vector<std::uint64_t> end;            // flat id -> end (once known)
  std::vector<bool> done;                    // flat id -> completed
  std::uint64_t now = 0;
  std::uint64_t elapsed = 0;
  std::size_t remaining = 0;
  std::uint64_t pe_free = 0, vec_free = 0, noc_free = 0, mem_free = 0;
  std::uint64_t next_mem_issue_at = UINT64_MAX;  // blocked min-key memory stage

  // --- the (at most one) outstanding memory request ---
  bool mem_outstanding = false;
  bool mem_issued_new = false;  // issued but not yet reported by next_action
  std::uint32_t mem_stage = 0;
  std::uint32_t mem_inst = 0;   // flat instance id
  bool mem_is_memnoc = false;
  std::uint64_t mem_noc_end = 0;
  double mem_actual_bytes = 0.0;
  double mem_service_units = 0.0;

  bool finished = false;

  void push_heap_key(std::uint64_t key) {
    heap.push_back(key);
    std::push_heap(heap.begin(), heap.end(), std::greater<std::uint64_t>());
  }
  std::uint64_t pop_heap_key() {
    std::pop_heap(heap.begin(), heap.end(), std::greater<std::uint64_t>());
    const std::uint64_t key = heap.back();
    heap.pop_back();
    return key;
  }
  void push_event(std::uint64_t end_ns, std::uint32_t id) {
    events.emplace_back(end_ns, id);
    std::push_heap(events.begin(), events.end(),
                   std::greater<std::pair<std::uint64_t, std::uint32_t>>());
  }

  void complete_instance(std::uint32_t id, std::uint64_t end_ns) {
    done[id] = true;
    --remaining;
    elapsed = std::max(elapsed, end_ns);
    for (std::uint32_t e = rdep_start[id]; e < rdep_start[id + 1]; ++e) {
      const std::uint32_t t = rdep_to[e];
      if (--unresolved[t] == 0) {
        const std::uint32_t ts = stage_of[t];
        push_heap_key((static_cast<std::uint64_t>(ts) << 32) |
                      static_cast<std::uint64_t>(t - offset[ts]));
      }
    }
  }

  void complete_due() {
    while (!events.empty() && events.front().first <= now) {
      const auto ev = events.front();
      std::pop_heap(events.begin(), events.end(),
                    std::greater<std::pair<std::uint64_t, std::uint32_t>>());
      events.pop_back();
      if (!done[ev.second]) {
        complete_instance(ev.second, ev.first);
      }
    }
  }

  // Start time of a memory instance = max(dependee ends, slot frees), the
  // same expression evaluate_stages uses at its min-key pop.
  std::uint64_t memory_start(std::uint32_t id, const DieStage& st) const {
    std::uint64_t start = 0;
    for (std::uint32_t e = dep_start[id]; e < dep_start[id + 1]; ++e) {
      start = std::max(start, end[dep_to[e]]);
    }
    start = std::max(start, mem_free);
    if (st.resource == StageResource::MEM_NOC) {
      start = std::max(start, noc_free);
    }
    return start;
  }

  void schedule_internal(std::uint32_t id, const DieStage& st) {
    std::uint64_t start = 0;
    for (std::uint32_t e = dep_start[id]; e < dep_start[id + 1]; ++e) {
      start = std::max(start, end[dep_to[e]]);
    }
    switch (st.resource) {
      case StageResource::PE:
        start = std::max(start, pe_free);
        break;
      case StageResource::VEC:
        start = std::max(start, vec_free);
        break;
      case StageResource::NOC:
        start = std::max(start, noc_free);
        break;
      case StageResource::MEM:
      case StageResource::MEM_NOC:
        break;  // unreachable; memory instances are issued, not scheduled
      case StageResource::TIMER:
        break;
    }
    const std::uint64_t fin = start + dur_ns[id];
    end[id] = fin;
    switch (st.resource) {
      case StageResource::PE:
        pe_free = fin;
        break;
      case StageResource::VEC:
        vec_free = fin;
        break;
      case StageResource::NOC:
        noc_free = fin;
        break;
      default:
        break;
    }
    push_event(fin, id);
  }

  void run() {
    next_mem_issue_at = UINT64_MAX;
    for (;;) {
      complete_due();
      if (mem_outstanding || heap.empty()) {
        break;  // strict pop discipline: never past the min-key instance
      }
      const std::uint64_t key = heap.front();
      const std::uint32_t s = static_cast<std::uint32_t>(key >> 32);
      const std::uint32_t j = static_cast<std::uint32_t>(key & 0xFFFFFFFFu);
      const std::uint32_t id =
          static_cast<std::uint32_t>(offset[s]) + j;
      const DieStage& st = (*stages)[s];
      if (st.resource == StageResource::MEM ||
          st.resource == StageResource::MEM_NOC) {
        const std::uint64_t start = memory_start(id, st);
        if (start > now) {
          next_mem_issue_at = start;  // future internal event: the issue time
          break;
        }
        pop_heap_key();
        mem_outstanding = true;
        mem_issued_new = true;
        mem_stage = s;
        mem_inst = id;
        mem_is_memnoc = st.resource == StageResource::MEM_NOC;
        mem_noc_end = start + noc_leg_ns[id];
        mem_actual_bytes = st.mem.actual_bytes;
        mem_service_units = st.mem.service_units;
        break;  // one outstanding request; nothing else pops before completion
      }
      pop_heap_key();
      schedule_internal(id, st);
    }
    if (remaining == 0) {
      finished = true;
    }
  }
};

// ---------------------------------------------------------------------------
// Construction: flatten the plan, reduce deps, validate, seed the heap.

Execution::Execution(std::shared_ptr<const DiePlan> p) : impl_(new Impl()) {
  if (!p || !p->valid()) {
    throw DieModelError("Execution: plan is empty");
  }
  Impl& m = *impl_;
  m.plan = std::move(p);
  m.stages = &m.plan->stages();
  m.hw = &m.plan->hardware();
  const std::size_t ns = m.stages->size();

  m.offset.assign(ns + 1, 0);
  std::size_t total = 0;
  for (std::size_t s = 0; s < ns; ++s) {
    total += static_cast<std::size_t>((*m.stages)[s].wave_repeat);
    m.offset[s + 1] = total;
  }
  m.total = total;
  m.remaining = total;
  if (total == 0) {
    m.finished = true;
    return;
  }
  m.stage_of.assign(total, 0);
  m.dur_ns.assign(total, 0);
  m.noc_leg_ns.assign(total, 0);
  m.end.assign(total, 0);
  m.done.assign(total, false);
  m.unresolved.assign(total, 0);
  for (std::size_t s = 0; s < ns; ++s) {
    for (std::size_t id = m.offset[s]; id < m.offset[s + 1]; ++id) {
      m.stage_of[id] = static_cast<std::uint32_t>(s);
    }
  }

  // Dependency reduction to single dependee instances (same reduction as
  // evaluate_stages) + durations from the shared §6.1 truth.
  m.dep_start.assign(total + 1, 0);
  std::vector<std::uint32_t> rdep_count(total, 0);
  auto dependee_of = [&](std::uint64_t j, const DieStageDep& dep) -> std::int64_t {
    if (dep.stage >= ns || (*m.stages)[dep.stage].wave_repeat == 0) {
      return -1;
    }
    if (dep.lag == DieStageDep::kAllDep) {
      return static_cast<std::int64_t>((*m.stages)[dep.stage].wave_repeat) - 1;
    }
    if (static_cast<std::int64_t>(j) >=
        static_cast<std::int64_t>(dep.lag)) {
      return static_cast<std::int64_t>(j) -
             static_cast<std::int64_t>(dep.lag);
    }
    return -1;
  };
  for (std::size_t s = 0; s < ns; ++s) {
    const DieStage& st = (*m.stages)[s];
    for (std::uint64_t j = 0; j < st.wave_repeat; ++j) {
      const std::size_t id = m.offset[s] + static_cast<std::size_t>(j);
      if (st.resource == StageResource::MEM ||
          st.resource == StageResource::MEM_NOC) {
        m.noc_leg_ns[id] = seconds_to_ns(detail::stage_noc_leg_seconds(st, *m.hw));
      } else {
        m.dur_ns[id] = seconds_to_ns(detail::stage_instance_seconds(st, *m.hw));
      }
      std::uint32_t d = 0;
      for (const DieStageDep& dep : st.deps) {
        const std::int64_t target = dependee_of(j, dep);
        if (target < 0) {
          continue;
        }
        ++d;
        ++rdep_count[m.offset[dep.stage] + static_cast<std::size_t>(target)];
      }
      m.unresolved[id] = d;
      m.dep_start[id + 1] = m.dep_start[id] + d;
    }
  }
  const std::size_t edges = m.dep_start[total];
  m.dep_to.assign(edges, 0);
  m.rdep_to.assign(edges, 0);
  m.rdep_start.assign(total + 1, 0);
  for (std::size_t id = 0; id < total; ++id) {
    m.rdep_start[id + 1] = m.rdep_start[id] + rdep_count[id];
  }
  {
    std::vector<std::uint32_t> dep_fill(m.dep_start.begin(),
                                        m.dep_start.end() - 1);
    std::vector<std::uint32_t> rdep_fill(m.rdep_start.begin(),
                                         m.rdep_start.end() - 1);
    for (std::size_t s = 0; s < ns; ++s) {
      for (std::uint64_t j = 0; j < (*m.stages)[s].wave_repeat; ++j) {
        const std::size_t id = m.offset[s] + static_cast<std::size_t>(j);
        for (const DieStageDep& dep : (*m.stages)[s].deps) {
          const std::int64_t target = dependee_of(j, dep);
          if (target < 0) {
            continue;
          }
          const std::size_t tid =
              m.offset[dep.stage] + static_cast<std::size_t>(target);
          m.dep_to[dep_fill[id]++] = static_cast<std::uint32_t>(tid);
          m.rdep_to[rdep_fill[tid]++] = static_cast<std::uint32_t>(id);
        }
      }
    }
  }

  // Kahn feasibility check (fail closed on cycles, like evaluate_stages).
  {
    std::vector<std::uint32_t> unresolved = m.unresolved;
    std::vector<std::uint32_t> stack;
    for (std::size_t id = 0; id < total; ++id) {
      if (unresolved[id] == 0) {
        stack.push_back(static_cast<std::uint32_t>(id));
      }
    }
    std::size_t reached = 0;
    while (!stack.empty()) {
      const std::uint32_t id = stack.back();
      stack.pop_back();
      ++reached;
      for (std::uint32_t e = m.rdep_start[id]; e < m.rdep_start[id + 1]; ++e) {
        if (--unresolved[m.rdep_to[e]] == 0) {
          stack.push_back(m.rdep_to[e]);
        }
      }
    }
    if (reached != total) {
      throw DieModelError("Execution: stage DAG has a dependency cycle");
    }
  }

  for (std::size_t s = 0; s < ns; ++s) {
    for (std::uint64_t j = 0; j < (*m.stages)[s].wave_repeat; ++j) {
      const std::size_t id = m.offset[s] + static_cast<std::size_t>(j);
      if (m.unresolved[id] == 0) {
        m.push_heap_key((static_cast<std::uint64_t>(s) << 32) | j);
      }
    }
  }
  m.run();
}

Execution::~Execution() = default;

bool Execution::finished() const { return impl_->finished; }

std::uint64_t Execution::elapsed_ns() const { return impl_->elapsed; }

// ---------------------------------------------------------------------------
// Action protocol

ExecutionAction Execution::next_action() {
  Impl& m = *impl_;
  m.run();
  ExecutionAction a;
  if (m.finished) {
    a.kind = ExecutionAction::Kind::Done;
    a.ns = m.elapsed;
    return a;
  }
  if (m.mem_issued_new) {
    // A fresh request was issued by run() (possibly already during an
    // advance_time()/complete_memory() call); report it before any timer so
    // the driver always learns about the handover.
    m.mem_issued_new = false;
    a.kind = ExecutionAction::Kind::MemoryRequest;
    a.stage_id = m.mem_stage;
    a.actual_bytes = m.mem_actual_bytes;
    a.service_units = m.mem_service_units;
    return a;
  }
  // Next internal event: the earliest scheduled-but-uncompleted stage end or
  // the issue time of the blocked min-key memory stage.  run() has already
  // completed everything <= now, so this is strictly in the future.
  std::uint64_t t = m.next_mem_issue_at;
  if (!m.events.empty()) {
    t = std::min(t, m.events.front().first);
  }
  if (t != UINT64_MAX && t > m.now) {
    a.kind = ExecutionAction::Kind::Timer;
    a.ns = t - m.now;
    return a;
  }
  if (m.mem_outstanding) {
    // Idempotent re-report of the SAME outstanding request (the driver owes
    // complete_memory for it; no new request may be issued before that).
    a.kind = ExecutionAction::Kind::MemoryRequest;
    a.stage_id = m.mem_stage;
    a.actual_bytes = m.mem_actual_bytes;
    a.service_units = m.mem_service_units;
    return a;
  }
  // Not finished, no internal events, no outstanding request: impossible
  // for an acyclic DAG (run() would have popped or issued something).
  throw DieModelError(
      "next_action: execution stalled without an internal event or an "
      "outstanding memory request");
}

void Execution::advance_time(std::uint64_t ns) {
  Impl& m = *impl_;
  if (ns > UINT64_MAX - m.now) {
    throw DieModelError("advance_time: clock overflow");
  }
  m.now += ns;
  m.run();
}

void Execution::complete_memory(std::uint64_t stage_id) {
  Impl& m = *impl_;
  if (!m.mem_outstanding) {
    throw DieModelError(
        "complete_memory: no outstanding memory request to complete");
  }
  if (stage_id != m.mem_stage) {
    throw DieModelError(
        "complete_memory: stage_id " + std::to_string(stage_id) +
        " does not match the outstanding request of stage " +
        std::to_string(m.mem_stage));
  }
  m.mem_outstanding = false;
  std::uint64_t end = m.now;  // the memory leg completes at the clock
  if (m.mem_is_memnoc && m.mem_noc_end > end) {
    end = m.mem_noc_end;  // later completion wins (dual occupancy, §6.1)
  }
  m.end[m.mem_inst] = end;
  m.mem_free = end;          // the MEM slot frees at the full end
  if (m.mem_is_memnoc) {
    m.noc_free = end;        // ... and so does the NOC slot of a MEM_NOC
  }
  m.push_event(end, m.mem_inst);
  m.run();
}

}  // namespace DieModel
