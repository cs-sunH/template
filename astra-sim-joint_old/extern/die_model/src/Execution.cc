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
#include <optional>
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

double stage_work(const DieStage& stage) {
  switch (stage.resource) {
    case StageResource::PE:
      return stage.pe_core_flop;
    case StageResource::VEC:
      return stage.vec_core_eqop;
    default:
      return 0.0;
  }
}

std::uint64_t physical_stage_min_ns(const DieStage& stage,
                                    const Hardware& hardware) {
  double work = 0.0;
  double peak = 0.0;
  if (stage.resource == StageResource::PE) {
    work = stage.pe_core_flop;
    peak = hardware.profile == Profile::Tx8ReferenceV1
        ? hardware.tx8_matrix_peak_flop_s
        : hardware.core_row.pe_gflops_per_core * 1e9;
  } else if (stage.resource == StageResource::VEC) {
    work = stage.vec_core_eqop;
    peak = hardware.profile == Profile::Tx8ReferenceV1
        ? hardware.tx8_vector_peak_eqop_s
        : hardware.core_row.vec_g_eqop_per_core * 1e9;
  } else {
    return 0;
  }
  if (work <= 0.0) return 0;
  if (!(peak > 0.0) || !std::isfinite(peak)) {
    throw DieModelError("Execution: missing physical PE/VEC peak for slice");
  }
  return seconds_to_ns(work / peak);
}

}  // namespace

struct Execution::Impl {
  // --- plan-derived data ---
  std::shared_ptr<const DiePlan> plan;
  const std::vector<DieStage>* stages = nullptr;
  const Hardware* hw = nullptr;
  mutable std::optional<std::string> template_key_cache;
  mutable std::uint64_t template_key_computations = 0;
  // These flattened shape/dependency/duration arrays are immutable after the
  // constructor finishes.
  std::vector<std::uint64_t> offset;         // stage -> flat instance range
  std::vector<std::uint32_t> stage_of;       // flat id -> stage index
  std::vector<std::uint32_t> dep_start;      // CSR dependees (per instance)
  std::vector<std::uint32_t> dep_to;
  std::vector<std::uint32_t> rdep_start;     // CSR reverse edges
  std::vector<std::uint32_t> rdep_to;
  std::vector<std::uint64_t> dur_ns;         // internal stages (0 for MEM*)
  std::vector<std::uint64_t> noc_leg_ns;     // MEM_NOC NoC leg
  std::size_t total = 0;

  // --- mutable dense per-run state ---
  std::vector<std::uint32_t> unresolved;     // unfinished dependees
  std::vector<std::uint64_t> start_ns;       // scheduled instance start
  std::vector<std::uint64_t> end;            // flat id -> end (once known)
  std::vector<std::uint64_t> completed_by_stage;
  std::vector<bool> done;

  // --- mutable scheduling frontier ---
  std::vector<std::uint64_t> heap;           // min-heap of (stage<<32)|inst
  std::vector<std::pair<std::uint64_t, std::uint32_t>> events;  // (end,id)
  std::vector<std::pair<std::uint64_t, std::uint32_t>> future_starts;
  std::vector<std::uint32_t> active_ids;     // insertion order of active frontier
  bool active_ids_dirty = false;             // completed IDs await batch compaction
  std::uint64_t active_id_compaction_calls = 0;
  std::uint64_t active_id_entries_examined = 0;
  std::uint64_t active_id_entries_moved = 0;
  std::uint64_t active_id_entries_removed = 0;
  std::uint64_t now = 0;
  std::uint64_t elapsed = 0;
  std::uint64_t frontier_generation = 1;
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
  std::uint64_t mem_start_ns = 0;
  double mem_actual_bytes = 0.0;
  double mem_service_units = 0.0;
  std::uint64_t external_memory_wait_elapsed_ns = 0;

  bool finished = false;

  struct PreviewUndoLog {
    enum class Field { Done, CompletedByStage, Unresolved, StartNs, End };
    struct Entry {
      Field field;
      std::size_t index;
      std::uint64_t old_value;
    };
    std::vector<Entry> entries;

    void record(const Field field, const std::size_t index,
                const std::uint64_t old_value) {
      entries.push_back(Entry{field, index, old_value});
    }

    void rollback(Impl& state) noexcept {
      for (auto it = entries.rbegin(); it != entries.rend(); ++it) {
        switch (it->field) {
          case Field::Done:
            state.done[it->index] = it->old_value != 0;
            break;
          case Field::CompletedByStage:
            state.completed_by_stage[it->index] = it->old_value;
            break;
          case Field::Unresolved:
            state.unresolved[it->index] =
                static_cast<std::uint32_t>(it->old_value);
            break;
          case Field::StartNs:
            state.start_ns[it->index] = it->old_value;
            break;
          case Field::End:
            state.end[it->index] = it->old_value;
            break;
        }
      }
      entries.clear();
    }
  };

  struct PreviewScalarState {
    bool active_ids_dirty;
    std::uint64_t active_id_compaction_calls;
    std::uint64_t active_id_entries_examined;
    std::uint64_t active_id_entries_moved;
    std::uint64_t active_id_entries_removed;
    std::uint64_t now;
    std::uint64_t elapsed;
    std::uint64_t frontier_generation;
    std::size_t remaining;
    std::uint64_t pe_free;
    std::uint64_t vec_free;
    std::uint64_t noc_free;
    std::uint64_t mem_free;
    std::uint64_t next_mem_issue_at;
    bool mem_outstanding;
    bool mem_issued_new;
    std::uint32_t mem_stage;
    std::uint32_t mem_inst;
    bool mem_is_memnoc;
    std::uint64_t mem_noc_end;
    std::uint64_t mem_start_ns;
    double mem_actual_bytes;
    double mem_service_units;
    std::uint64_t external_memory_wait_elapsed_ns;
    bool finished;

    explicit PreviewScalarState(const Impl& state)
        : active_ids_dirty(state.active_ids_dirty),
          active_id_compaction_calls(state.active_id_compaction_calls),
          active_id_entries_examined(state.active_id_entries_examined),
          active_id_entries_moved(state.active_id_entries_moved),
          active_id_entries_removed(state.active_id_entries_removed),
          now(state.now),
          elapsed(state.elapsed),
          frontier_generation(state.frontier_generation),
          remaining(state.remaining),
          pe_free(state.pe_free),
          vec_free(state.vec_free),
          noc_free(state.noc_free),
          mem_free(state.mem_free),
          next_mem_issue_at(state.next_mem_issue_at),
          mem_outstanding(state.mem_outstanding),
          mem_issued_new(state.mem_issued_new),
          mem_stage(state.mem_stage),
          mem_inst(state.mem_inst),
          mem_is_memnoc(state.mem_is_memnoc),
          mem_noc_end(state.mem_noc_end),
          mem_start_ns(state.mem_start_ns),
          mem_actual_bytes(state.mem_actual_bytes),
          mem_service_units(state.mem_service_units),
          external_memory_wait_elapsed_ns(
              state.external_memory_wait_elapsed_ns),
          finished(state.finished) {}

    void restore(Impl& state) const noexcept {
      state.active_ids_dirty = active_ids_dirty;
      state.active_id_compaction_calls = active_id_compaction_calls;
      state.active_id_entries_examined = active_id_entries_examined;
      state.active_id_entries_moved = active_id_entries_moved;
      state.active_id_entries_removed = active_id_entries_removed;
      state.now = now;
      state.elapsed = elapsed;
      state.frontier_generation = frontier_generation;
      state.remaining = remaining;
      state.pe_free = pe_free;
      state.vec_free = vec_free;
      state.noc_free = noc_free;
      state.mem_free = mem_free;
      state.next_mem_issue_at = next_mem_issue_at;
      state.mem_outstanding = mem_outstanding;
      state.mem_issued_new = mem_issued_new;
      state.mem_stage = mem_stage;
      state.mem_inst = mem_inst;
      state.mem_is_memnoc = mem_is_memnoc;
      state.mem_noc_end = mem_noc_end;
      state.mem_start_ns = mem_start_ns;
      state.mem_actual_bytes = mem_actual_bytes;
      state.mem_service_units = mem_service_units;
      state.external_memory_wait_elapsed_ns =
          external_memory_wait_elapsed_ns;
      state.finished = finished;
    }
  };

  struct PreviewBorrowGuard {
    Impl& state;
    std::vector<std::uint64_t> original_heap;
    std::vector<std::pair<std::uint64_t, std::uint32_t>> original_events;
    std::vector<std::pair<std::uint64_t, std::uint32_t>> original_future_starts;
    std::vector<std::uint32_t> original_active_ids;
    PreviewScalarState original_scalars;
    PreviewUndoLog undo;
    bool active = false;

    explicit PreviewBorrowGuard(Impl& value)
        : state(value),
          original_heap(value.heap),
          original_events(value.events),
          original_future_starts(value.future_starts),
          original_active_ids(value.active_ids),
          original_scalars(value) {
      state.heap.swap(original_heap);
      state.events.swap(original_events);
      state.future_starts.swap(original_future_starts);
      state.active_ids.swap(original_active_ids);
      active = true;
    }

    PreviewBorrowGuard(const PreviewBorrowGuard&) = delete;
    PreviewBorrowGuard& operator=(const PreviewBorrowGuard&) = delete;

    ~PreviewBorrowGuard() noexcept { rollback(); }

    void rollback() noexcept {
      if (!active) return;
      undo.rollback(state);
      original_scalars.restore(state);
      state.heap.swap(original_heap);
      state.events.swap(original_events);
      state.future_starts.swap(original_future_starts);
      state.active_ids.swap(original_active_ids);
      active = false;
    }
  };

  const std::string& template_key() const {
    if (!template_key_cache) {
      // Execution's mutable cursor requires a single driver; this lazy cache
      // follows that contract and is intentionally unsynchronized. Preserve
      // the original validation point instead of calculating in construction.
      template_key_cache = plan_cache_key(plan->spec(), *hw);
      ++template_key_computations;
    }
    return *template_key_cache;
  }

  void bump_frontier_generation() {
    if (frontier_generation == UINT64_MAX) {
      throw DieModelError("Execution: frontier generation overflows uint64");
    }
    ++frontier_generation;
  }

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

  void push_future_start(std::uint64_t start_time, std::uint32_t id) {
    future_starts.emplace_back(start_time, id);
    std::push_heap(future_starts.begin(), future_starts.end(),
                   std::greater<std::pair<std::uint64_t, std::uint32_t>>());
  }

  void activate_due_starts() {
    while (!future_starts.empty() && future_starts.front().first <= now) {
      std::pop_heap(future_starts.begin(), future_starts.end(),
                    std::greater<std::pair<std::uint64_t, std::uint32_t>>());
      const auto start = future_starts.back();
      future_starts.pop_back();
      if (!done[start.second] && end[start.second] > now) {
        active_ids.push_back(start.second);
      }
    }
  }

  void complete_instance(std::uint32_t id, std::uint64_t end_ns,
                         PreviewUndoLog* const preview_undo) {
    if (preview_undo != nullptr) {
      preview_undo->record(PreviewUndoLog::Field::Done, id,
                           done[id] ? 1 : 0);
    }
    done[id] = true;
    const std::uint32_t stage_id = stage_of[id];
    if (completed_by_stage[stage_id] == UINT64_MAX) {
      throw DieModelError("Execution: completed stage count overflows uint64");
    }
    if (preview_undo != nullptr) {
      preview_undo->record(PreviewUndoLog::Field::CompletedByStage,
                           stage_id, completed_by_stage[stage_id]);
    }
    ++completed_by_stage[stage_id];
    // The old per-instance erase/remove ran here. No observer or scheduler
    // callback can inspect active_ids until complete_due() returns, so defer
    // removal and compact once after all same-boundary completions. The dirty
    // flag is set at the same logical point as the old erase (after the
    // overflow guard and completed counter update).
    active_ids_dirty = true;
    bump_frontier_generation();
    --remaining;
    elapsed = std::max(elapsed, end_ns);
    for (std::uint32_t e = rdep_start[id]; e < rdep_start[id + 1]; ++e) {
      const std::uint32_t t = rdep_to[e];
      if (preview_undo != nullptr) {
        preview_undo->record(PreviewUndoLog::Field::Unresolved, t,
                             unresolved[t]);
      }
      if (--unresolved[t] == 0) {
        const std::uint32_t ts = stage_of[t];
        push_heap_key((static_cast<std::uint64_t>(ts) << 32) |
                      static_cast<std::uint64_t>(t - offset[ts]));
      }
    }
  }

  void compact_completed_active_ids() {
    std::size_t kept = 0;
    for (std::size_t i = 0; i < active_ids.size(); ++i) {
      const std::uint32_t id = active_ids[i];
      ++active_id_entries_examined;
      if (done[id]) {
        ++active_id_entries_removed;
        continue;
      }
      if (kept != i) {
        active_ids[kept] = id;
        ++active_id_entries_moved;
      }
      ++kept;
    }
    active_ids.resize(kept);
    ++active_id_compaction_calls;
    active_ids_dirty = false;
  }

  void complete_due(PreviewUndoLog* const preview_undo) {
    try {
      while (!events.empty() && events.front().first <= now) {
        const auto ev = events.front();
        std::pop_heap(events.begin(), events.end(),
                      std::greater<std::pair<std::uint64_t, std::uint32_t>>());
        events.pop_back();
        if (!done[ev.second]) {
          complete_instance(ev.second, ev.first, preview_undo);
        }
      }
    } catch (...) {
      if (active_ids_dirty) compact_completed_active_ids();
      throw;
    }
    if (active_ids_dirty) compact_completed_active_ids();
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

  void schedule_internal(std::uint32_t id, const DieStage& st,
                         PreviewUndoLog* const preview_undo) {
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
    if (dur_ns[id] > UINT64_MAX - start) {
      throw DieModelError("Execution: scheduled stage end overflows uint64");
    }
    const std::uint64_t fin = start + dur_ns[id];
    bump_frontier_generation();
    if (preview_undo != nullptr) {
      preview_undo->record(PreviewUndoLog::Field::StartNs, id, start_ns[id]);
    }
    start_ns[id] = start;
    if (preview_undo != nullptr) {
      preview_undo->record(PreviewUndoLog::Field::End, id, end[id]);
    }
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
    if (start <= now && fin > now) {
      active_ids.push_back(id);
    } else if (start > now) {
      push_future_start(start, id);
    }
  }

  void run(PreviewUndoLog* const preview_undo = nullptr) {
    next_mem_issue_at = UINT64_MAX;
    for (;;) {
      activate_due_starts();
      complete_due(preview_undo);
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
        bump_frontier_generation();
        mem_issued_new = true;
        mem_stage = s;
        mem_inst = id;
        mem_is_memnoc = st.resource == StageResource::MEM_NOC;
        mem_start_ns = start;
        if (preview_undo != nullptr) {
          preview_undo->record(PreviewUndoLog::Field::StartNs, id,
                               start_ns[id]);
        }
        start_ns[id] = start;
        mem_noc_end = start + noc_leg_ns[id];
        mem_actual_bytes = st.mem.actual_bytes;
        mem_service_units = st.mem.service_units;
        break;  // one outstanding request; nothing else pops before completion
      }
      pop_heap_key();
      schedule_internal(id, st, preview_undo);
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
  m.completed_by_stage.assign(ns, 0);
  if (total == 0) {
    m.finished = true;
    return;
  }
  m.stage_of.assign(total, 0);
  m.dur_ns.assign(total, 0);
  m.noc_leg_ns.assign(total, 0);
  m.start_ns.assign(total, 0);
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

std::optional<std::uint32_t> Execution::pending_memory_stage_id() const {
  const Impl& m = *impl_;
  if (!m.mem_outstanding) return std::nullopt;
  return m.mem_stage;
}

std::uint64_t Execution::template_key_computations_for_test() const {
  return impl_->template_key_computations;
}

InternalSliceInfo Execution::describe_internal_slice() const {
  const Impl& m = *impl_;
  InternalSliceInfo info;
  info.template_key = m.template_key();
  info.now_ns = m.now;

  std::uint64_t next_boundary = UINT64_MAX;
  if (!m.events.empty()) next_boundary = m.events.front().first;
  if (!m.future_starts.empty()) {
    next_boundary = std::min(next_boundary, m.future_starts.front().first);
  }
  if (m.next_mem_issue_at != UINT64_MAX) {
    next_boundary = std::min(next_boundary, m.next_mem_issue_at);
  }
  if (m.mem_outstanding && m.mem_is_memnoc && m.mem_noc_end > m.now) {
    next_boundary = std::min(next_boundary, m.mem_noc_end);
  }
  if (next_boundary != UINT64_MAX) {
    info.has_next_boundary = true;
    info.next_boundary_ns = next_boundary;
  }

  for (const std::uint32_t id : m.active_ids) {
    if (m.done[id] || m.start_ns[id] > m.now || m.end[id] <= m.now) continue;
    const std::uint32_t stage_id = m.stage_of[id];
    const std::uint64_t total_ns = m.end[id] - m.start_ns[id];
    const std::uint64_t remaining_ns = m.end[id] - m.now;
    const DieStage& stage = (*m.stages)[stage_id];
    const double work = stage_work(stage);
    InternalStageSlice slice;
    slice.stage_id = stage_id;
    slice.wave_index = static_cast<std::uint64_t>(id) - m.offset[stage_id];
    slice.resource = stage.resource;
    slice.start_ns = m.start_ns[id];
    slice.end_ns = m.end[id];
    slice.original_service_ns = total_ns;
    slice.remaining_service_ns = remaining_ns;
    slice.original_work = work;
    slice.remaining_work = total_ns == 0
        ? 0.0
        : work * static_cast<double>(remaining_ns) /
              static_cast<double>(total_ns);
    slice.hardware_min_ns = physical_stage_min_ns(stage, *m.hw);
    info.active_stages.push_back(slice);
  }

  if (m.mem_outstanding) {
    const DieStage& stage = (*m.stages)[m.mem_stage];
    InternalStageSlice slice;
    slice.stage_id = m.mem_stage;
    slice.wave_index = static_cast<std::uint64_t>(m.mem_inst) -
                       m.offset[m.mem_stage];
    slice.resource = stage.resource;
    slice.start_ns = m.mem_start_ns;
    slice.end_ns = m.mem_is_memnoc ? m.mem_noc_end : m.mem_start_ns;
    slice.original_service_ns = m.mem_is_memnoc
        ? m.noc_leg_ns[m.mem_inst] : 0;
    slice.remaining_service_ns = m.mem_is_memnoc && m.mem_noc_end > m.now
        ? m.mem_noc_end - m.now : 0;
    slice.memory_actual_bytes = m.mem_actual_bytes;
    slice.memory_service_units = m.mem_service_units;
    slice.memory_pending = true;
    info.active_stages.push_back(slice);
  }

  if (m.finished) {
    info.classification = InternalSliceClass::Idle;
    info.reason = "execution-finished";
    return info;
  }
  if (m.mem_outstanding) {
    info.classification = m.mem_is_memnoc
        ? InternalSliceClass::MemNocResidual
        : InternalSliceClass::MemoryService;
    info.reason = "external-memory-request-outstanding";
    return info;
  }
  if (info.active_stages.empty()) {
    info.classification = InternalSliceClass::Unknown;
    info.reason = "no-active-internal-service-frontier";
    return info;
  }
  if (info.active_stages.size() != 1) {
    info.classification = InternalSliceClass::MixedFrontier;
    info.reason = "multiple-internal-stages-active";
    return info;
  }

  const auto& current = info.active_stages.front();
  const std::uint32_t current_flat_id = static_cast<std::uint32_t>(
      m.offset[current.stage_id] + current.wave_index);
  if (m.events.empty() || m.events.front().second != current_flat_id) {
    info.classification = InternalSliceClass::MixedFrontier;
    info.reason = "another-event-precedes-current-stage-boundary";
    return info;
  }

  // Any other already scheduled event whose interval intersects this one
  // makes the timer a mixed frontier, even if that event starts later.
  for (const std::uint32_t id : m.active_ids) {
    if (id == current_flat_id || m.done[id]) continue;
    if (m.start_ns[id] < current.end_ns && m.end[id] > m.now) {
      info.classification = InternalSliceClass::MixedFrontier;
      info.reason = "overlapping-internal-stage-interval";
      return info;
    }
  }
  if (!m.future_starts.empty() &&
      m.future_starts.front().first < current.end_ns) {
    info.classification = InternalSliceClass::MixedFrontier;
    info.reason = "future-internal-stage-starts-before-current-boundary";
    return info;
  }
  if (m.next_mem_issue_at != UINT64_MAX &&
      m.next_mem_issue_at <= current.end_ns) {
    info.classification = InternalSliceClass::MixedFrontier;
    info.reason = "memory-request-boundary-before-stage-completion";
    return info;
  }

  if (current.resource == StageResource::PE) {
    info.classification = InternalSliceClass::PurePe;
  } else if (current.resource == StageResource::VEC) {
    info.classification = InternalSliceClass::PureVec;
  } else if (current.resource == StageResource::NOC) {
    info.classification = InternalSliceClass::InternalNoc;
    info.reason = "internal-noc-service-is-not-window-eligible";
    return info;
  } else {
    info.classification = InternalSliceClass::Unknown;
    info.reason = "resource-class-is-not-a-pure-compute-slice";
    return info;
  }
  if (current.start_ns > m.now || current.original_service_ns == 0) {
    info.classification = InternalSliceClass::Unknown;
    info.reason = "stage-is-not-in-positive-active-service";
    return info;
  }
  info.action_token.template_key = info.template_key;
  info.action_token.flat_instance_id = current_flat_id;
  info.action_token.frontier_generation = m.frontier_generation;
  info.action_token.valid = true;
  info.partial_service_supported = true;
  info.reason = "single-resource-stage-without-overlap-or-memory-boundary";
  return info;
}

std::vector<InternalTemplateSliceEntry> Execution::query_template_slice(
    const std::uint32_t stage_id, const std::uint64_t position,
    const std::uint64_t count) const {
  const Impl& m = *impl_;
  if (stage_id >= m.stages->size() || count == 0 || count > 8) {
    throw DieModelError(
        "query_template_slice: stage must exist and count must be in [1,8]");
  }
  const DieStage& stage = (*m.stages)[stage_id];
  if (stage.resource != StageResource::PE &&
      stage.resource != StageResource::VEC) {
    throw DieModelError(
        "query_template_slice: only typed PE/VEC service stages are queryable");
  }
  if (position > stage.wave_repeat || count > stage.wave_repeat - position) {
    throw DieModelError("query_template_slice: slice exceeds immutable wave range");
  }
  const double seconds = detail::stage_instance_seconds(stage, *m.hw);
  const std::uint64_t duration_ns = seconds_to_ns(seconds);
  const std::uint64_t min_ns = physical_stage_min_ns(stage, *m.hw);
  std::vector<InternalTemplateSliceEntry> result;
  result.reserve(static_cast<std::size_t>(count));
  for (std::uint64_t i = 0; i < count; ++i) {
    result.push_back(InternalTemplateSliceEntry{
        stage_id, position + i, stage.resource, stage_work(stage), duration_ns,
        min_ns});
  }
  return result;
}

InternalTimerWindowInfo Execution::query_internal_timer_window(
    const std::uint32_t max_count) const {
  if (max_count == 0 || max_count > 8) {
    throw DieModelError(
        "query_internal_timer_window: max_count must be in [1,8]");
  }
  const Impl& m = *impl_;
  const InternalSliceInfo frontier = describe_internal_slice();
  InternalTimerWindowInfo result;
  if (!frontier.partial_service_supported || !frontier.action_token.valid ||
      frontier.active_stages.size() != 1) {
    result.reason = frontier.reason.empty()
        ? "frontier_not_single_typed_timer"
        : frontier.reason;
    return result;
  }
  const InternalStageSlice& active = frontier.active_stages.front();
  if (active.stage_id >= m.stages->size()) {
    result.reason = "current_stage_out_of_range";
    return result;
  }
  if (active.resource != StageResource::PE &&
      active.resource != StageResource::VEC) {
    result.reason = "resource_not_pure_pe_or_vec";
    return result;
  }
  if (active.wave_index >= (*m.stages)[active.stage_id].wave_repeat ||
      active.start_ns != m.now ||
      active.remaining_service_ns != active.original_service_ns) {
    result.reason = "current_slice_already_partially_served";
    return result;
  }
  const std::uint64_t first_flat_wide =
      m.offset[active.stage_id] + active.wave_index;
  if (first_flat_wide >= m.total || first_flat_wide > UINT32_MAX) {
    result.reason = "current_instance_out_of_range";
    return result;
  }
  const std::uint32_t first_flat =
      static_cast<std::uint32_t>(first_flat_wide);
  if (m.active_ids.size() != 1 || m.active_ids.front() != first_flat ||
      m.events.size() != 1 || m.events.front().second != first_flat ||
      !m.future_starts.empty() || m.mem_outstanding) {
    result.reason = "other_scheduled_internal_boundary";
    return result;
  }

  const DieStage& stage = (*m.stages)[active.stage_id];
  const std::uint64_t available = std::min<std::uint64_t>(
      max_count, stage.wave_repeat - active.wave_index);
  result.slices = query_template_slice(active.stage_id,
                                       active.wave_index, available);
  if (result.slices.empty() ||
      result.slices.front().original_service_ns != active.original_service_ns) {
    result.slices.clear();
    result.reason = "immutable_template_does_not_match_active_frontier";
    return result;
  }

  if (m.dur_ns[first_flat] != active.original_service_ns) {
    result.slices.clear();
    result.reason = "current_timer_duration_differs_from_template";
    return result;
  }

  // Grow only while each next wave is released by its immediate predecessor
  // and no other pending operation becomes ready at an interior boundary.
  // Other dependents may become ready exactly at the final window boundary
  // (for example a MEM request waiting on all K waves); the caller will then
  // observe and dispatch that real action after the timer callback.
  std::size_t safe_count = 1;
  for (std::size_t i = 0; i + 1 < result.slices.size(); ++i) {
    const auto& current_slice = result.slices[i];
    const auto& next_slice = result.slices[i + 1];
    if (next_slice.stage_id != active.stage_id ||
        next_slice.wave_index != current_slice.wave_index + 1 ||
        next_slice.resource != active.resource ||
        next_slice.original_service_ns == 0) {
      result.reason = "template_slice_changes_typed_timer";
      break;
    }
    const std::uint64_t current_wide =
        m.offset[current_slice.stage_id] + current_slice.wave_index;
    const std::uint64_t next_wide =
        m.offset[next_slice.stage_id] + next_slice.wave_index;
    if (next_wide >= m.total || next_wide > UINT32_MAX) {
      result.reason = "future_instance_out_of_range";
      break;
    }
    const std::uint32_t current = static_cast<std::uint32_t>(current_wide);
    const std::uint32_t next = static_cast<std::uint32_t>(next_wide);
    if (m.unresolved[next] != 1 ||
        m.dep_start[next + 1] - m.dep_start[next] != 1 ||
        m.dep_to[m.dep_start[next]] != current) {
      result.reason = "future_wave_is_not_a_single_predecessor_chain";
      break;
    }
    bool has_expected_child = false;
    bool another_child_ready_at_boundary = false;
    for (std::uint32_t edge = m.rdep_start[current];
         edge < m.rdep_start[current + 1]; ++edge) {
      const std::uint32_t target = m.rdep_to[edge];
      if (target == next) {
        has_expected_child = true;
      } else if (m.unresolved[target] == 1) {
        // This target would be issued by run() at the same tick, ahead of
        // any later timer in the proposed window.
        another_child_ready_at_boundary = true;
      }
    }
    if (!has_expected_child) {
      result.reason = "future_wave_is_not_released_by_previous_wave";
      break;
    }
    if (another_child_ready_at_boundary) {
      result.reason = "another_action_becomes_ready_inside_window";
      break;
    }
    safe_count = i + 2;
  }
  if (safe_count < result.slices.size()) result.slices.resize(safe_count);

  std::uint64_t span_ns = 0;
  for (const auto& slice : result.slices) {
    if (slice.original_service_ns == 0 ||
        slice.original_service_ns >
            std::numeric_limits<std::uint64_t>::max() - span_ns) {
      throw DieModelError(
          "query_internal_timer_window: service span overflows uint64");
    }
    span_ns += slice.original_service_ns;
  }

  if (result.slices.empty()) {
    result.reason = "no_contiguous_typed_timer_slice";
    return result;
  }
  if (span_ns > UINT64_MAX - m.now) {
    throw DieModelError(
        "query_internal_timer_window: end clock overflows uint64");
  }
  if (m.next_mem_issue_at != UINT64_MAX &&
      m.next_mem_issue_at < m.now + span_ns) {
    result.slices.resize(1);
    result.reason = "memory_request_boundary_inside_window";
    span_ns = result.slices.front().original_service_ns;
  }
  result.supported = true;
  result.token.first_action = frontier.action_token;
  result.token.stage_id = active.stage_id;
  result.token.first_wave_index = active.wave_index;
  result.token.valid = true;
  result.token.original_service_ticks.reserve(result.slices.size());
  for (const auto& slice : result.slices) {
    result.token.original_service_ticks.push_back(
        slice.original_service_ns);
  }
  if (result.slices.size() == max_count) result.reason.clear();
  else if (result.reason.empty()) result.reason = "window_stops_at_typed_boundary";
  return result;
}

void Execution::advance_internal_service(const InternalActionToken& action_token,
                                         const std::uint64_t delta_ns) {
  const InternalSliceInfo info = describe_internal_slice();
  if (!info.partial_service_supported || !info.action_token.valid ||
      !action_token.valid ||
      action_token.template_key != info.action_token.template_key ||
      action_token.flat_instance_id != info.action_token.flat_instance_id ||
      action_token.frontier_generation !=
          info.action_token.frontier_generation) {
    throw DieModelError(
        "advance_internal_service: stale token or ineligible frontier");
  }
  const auto& stage = info.active_stages.front();
  const std::uint64_t until_boundary =
      info.next_boundary_ns >= info.now_ns
          ? info.next_boundary_ns - info.now_ns
          : 0;
  if (delta_ns > stage.remaining_service_ns || delta_ns > until_boundary) {
    throw DieModelError(
        "advance_internal_service: delta crosses the current service boundary");
  }
  advance_time(delta_ns);
}

void Execution::advance_internal_service_window(
    InternalServiceWindowToken& token, const std::uint64_t delta_ns) {
  if (!token.valid || !token.first_action.valid ||
      token.original_service_ticks.empty() ||
      token.original_service_ticks.size() > 8) {
    throw DieModelError(
        "advance_internal_service_window: invalid window token");
  }
  unsigned __int128 total_wide = 0;
  for (const auto service : token.original_service_ticks) {
    if (service == 0) {
      throw DieModelError(
          "advance_internal_service_window: token contains zero service");
    }
    total_wide += service;
  }
  if (total_wide > UINT64_MAX) {
    throw DieModelError(
        "advance_internal_service_window: token service total overflows uint64");
  }
  const std::uint64_t total = static_cast<std::uint64_t>(total_wide);
  if (token.service_already_applied > total ||
      delta_ns > total - token.service_already_applied) {
    throw DieModelError(
        "advance_internal_service_window: delta exceeds remaining window service");
  }
  if (delta_ns == 0) return;

  Impl& m = *impl_;
  if (token.stage_id >= m.stages->size()) {
    throw DieModelError(
        "advance_internal_service_window: stage id is out of range");
  }
  const std::uint64_t expected_first_flat =
      m.offset[token.stage_id] + token.first_wave_index;
  if (expected_first_flat >= m.total ||
      expected_first_flat > UINT32_MAX ||
      token.first_action.template_key !=
          m.template_key() ||
      token.first_action.flat_instance_id != expected_first_flat) {
    throw DieModelError(
        "advance_internal_service_window: token does not identify this execution");
  }

  std::uint64_t remaining_delta = delta_ns;
  while (remaining_delta != 0) {
    const InternalSliceInfo frontier = describe_internal_slice();
    const unsigned __int128 expected_generation_wide =
        static_cast<unsigned __int128>(
            token.first_action.frontier_generation) +
        static_cast<unsigned __int128>(2) * token.completed_slices;
    const unsigned __int128 expected_wave_wide =
        static_cast<unsigned __int128>(token.first_wave_index) +
        token.completed_slices;
    if (expected_generation_wide > UINT64_MAX ||
        expected_wave_wide > UINT64_MAX ||
        m.frontier_generation !=
            static_cast<std::uint64_t>(expected_generation_wide) ||
        !frontier.partial_service_supported ||
        frontier.active_stages.size() != 1 ||
        frontier.action_token.template_key != token.first_action.template_key ||
        frontier.active_stages.front().stage_id != token.stage_id ||
        frontier.active_stages.front().wave_index !=
            static_cast<std::uint64_t>(expected_wave_wide) ||
        token.completed_slices >= token.original_service_ticks.size()) {
      throw DieModelError(
          "advance_internal_service_window: stale or nonsequential frontier");
    }
    const InternalStageSlice& active = frontier.active_stages.front();
    const std::uint64_t slice_expected =
        token.original_service_ticks[token.completed_slices];
    if (active.original_service_ns != slice_expected ||
        active.remaining_service_ns > slice_expected) {
      throw DieModelError(
          "advance_internal_service_window: current wave differs from token");
    }
    unsigned __int128 expected_already_applied = 0;
    for (std::uint64_t i = 0; i < token.completed_slices; ++i) {
      expected_already_applied += token.original_service_ticks[i];
    }
    expected_already_applied +=
        slice_expected - active.remaining_service_ns;
    if (expected_already_applied > UINT64_MAX ||
        token.service_already_applied !=
            static_cast<std::uint64_t>(expected_already_applied)) {
      throw DieModelError(
          "advance_internal_service_window: token service cursor is stale");
    }
    const std::uint64_t take =
        std::min(remaining_delta, active.remaining_service_ns);
    if (take == 0) {
      throw DieModelError(
          "advance_internal_service_window: no remaining service at frontier");
    }
    advance_time(take);
    token.service_already_applied += take;
    remaining_delta -= take;
    if (take == active.remaining_service_ns) {
      ++token.completed_slices;
    }
  }
}

void Execution::advance_external_wait(
    const std::uint64_t delta_ns,
    const std::uint64_t expected_memory_stage) {
  Impl& m = *impl_;
  if (!m.mem_outstanding || m.mem_stage != expected_memory_stage) {
    throw DieModelError(
        "advance_external_wait: no matching outstanding memory stage");
  }
  if (delta_ns > UINT64_MAX - m.external_memory_wait_elapsed_ns) {
    throw DieModelError(
        "advance_external_wait: accumulated wait overflows uint64");
  }
  advance_time(delta_ns);
  m.external_memory_wait_elapsed_ns += delta_ns;
}

ExecutionProgress Execution::export_progress() const {
  const Impl& m = *impl_;
  ExecutionProgress progress;
  progress.now_ns = m.now;
  progress.elapsed_ns = m.elapsed;
  progress.finished = m.finished;
  progress.memory_pending = m.mem_outstanding;
  progress.memory_stage_id = m.mem_stage;
  progress.has_memory_instance_id = m.mem_outstanding;
  progress.memory_instance_id = m.mem_inst;
  progress.pending_memory_actual_bytes =
      m.mem_outstanding ? m.mem_actual_bytes : 0.0;
  progress.pending_memory_service_units =
      m.mem_outstanding ? m.mem_service_units : 0.0;
  progress.external_memory_wait_elapsed_ns =
      m.external_memory_wait_elapsed_ns;
  progress.active_id_compaction_calls = m.active_id_compaction_calls;
  progress.active_id_entries_examined = m.active_id_entries_examined;
  progress.active_id_entries_moved = m.active_id_entries_moved;
  progress.active_id_entries_removed = m.active_id_entries_removed;
  progress.frontier = describe_internal_slice();
  progress.stages.reserve(m.stages->size());
  for (std::size_t stage_id = 0; stage_id < m.stages->size(); ++stage_id) {
    const DieStage& stage = (*m.stages)[stage_id];
    const std::uint64_t completed = m.completed_by_stage[stage_id];
    const std::uint64_t total = static_cast<std::uint64_t>(
        m.offset[stage_id + 1] - m.offset[stage_id]);
    const std::uint64_t incomplete = total - completed;
    ExecutionProgress::StageCount state;
    state.stage_id = static_cast<std::uint32_t>(stage_id);
    state.resource = stage.resource;
    state.completed = completed;
    state.total = total;
    switch (stage.resource) {
      case StageResource::PE:
      case StageResource::VEC:
        state.remaining_work = stage_work(stage) *
                               static_cast<double>(incomplete);
        break;
      case StageResource::NOC:
        state.remaining_noc_bytes = stage.noc_edge_bytes *
                                    static_cast<double>(incomplete);
        break;
      case StageResource::MEM:
      case StageResource::MEM_NOC:
        state.remaining_actual_bytes = stage.mem.actual_bytes *
                                       static_cast<double>(incomplete);
        state.remaining_service_units = stage.mem.service_units *
                                        static_cast<double>(incomplete);
        if (stage.resource == StageResource::MEM_NOC) {
          state.remaining_noc_bytes = stage.noc_edge_bytes *
                                      static_cast<double>(incomplete);
        }
        break;
      case StageResource::TIMER:
        break;
    }
    std::uint64_t internal_ns_per_instance = 0;
    if (total != 0) {
      const std::size_t first = m.offset[stage_id];
      internal_ns_per_instance = stage.resource == StageResource::MEM_NOC
          ? m.noc_leg_ns[first]
          : m.dur_ns[first];
    }
    if (internal_ns_per_instance != 0 && incomplete != 0) {
      const unsigned __int128 remaining_ns =
          static_cast<unsigned __int128>(internal_ns_per_instance) * incomplete;
      if (remaining_ns > UINT64_MAX) {
        throw DieModelError("export_progress: internal service total overflows uint64");
      }
      state.remaining_internal_service_ns =
          static_cast<std::uint64_t>(remaining_ns);
    }
    progress.stages.push_back(state);
  }

  // Adjust aggregate stage totals only for the bounded active frontier. The
  // completed counters above are maintained at the original completion point,
  // so exporting progress never scans every wave in a repeated stage.
  for (const InternalStageSlice& active : progress.frontier.active_stages) {
    if (active.stage_id >= progress.stages.size()) continue;
    auto& state = progress.stages[active.stage_id];
    const DieStage& stage = (*m.stages)[active.stage_id];
    const std::uint64_t flat_id = static_cast<std::uint64_t>(m.offset[active.stage_id]) +
                                  active.wave_index;
    const bool current_memory = m.mem_outstanding &&
                                flat_id == m.mem_inst;
    if (active.resource == StageResource::PE ||
        active.resource == StageResource::VEC) {
      const double remaining = active.original_service_ns == 0
          ? 0.0
          : active.original_work *
                static_cast<double>(active.remaining_service_ns) /
                static_cast<double>(active.original_service_ns);
      const double full = stage_work(stage);
      state.remaining_work = std::max(0.0, state.remaining_work - full + remaining);
      if (state.remaining_internal_service_ns >= active.original_service_ns) {
        state.remaining_internal_service_ns -= active.original_service_ns;
        state.remaining_internal_service_ns += active.remaining_service_ns;
      }
    } else if (active.resource == StageResource::NOC) {
      const double remaining = active.original_service_ns == 0
          ? 0.0
          : stage.noc_edge_bytes *
                static_cast<double>(active.remaining_service_ns) /
                static_cast<double>(active.original_service_ns);
      state.remaining_noc_bytes = std::max(
          0.0, state.remaining_noc_bytes - stage.noc_edge_bytes + remaining);
      if (state.remaining_internal_service_ns >= active.original_service_ns) {
        state.remaining_internal_service_ns -= active.original_service_ns;
        state.remaining_internal_service_ns += active.remaining_service_ns;
      }
    } else if (active.resource == StageResource::MEM_NOC) {
      const std::uint64_t original_noc_ns = m.noc_leg_ns[flat_id];
      const std::uint64_t remaining_noc_ns = active.remaining_service_ns;
      const double remaining_noc_bytes = original_noc_ns == 0
          ? 0.0
          : stage.noc_edge_bytes * static_cast<double>(remaining_noc_ns) /
                static_cast<double>(original_noc_ns);
      state.remaining_noc_bytes = std::max(
          0.0, state.remaining_noc_bytes - stage.noc_edge_bytes +
                   remaining_noc_bytes);
      if (state.remaining_internal_service_ns >= original_noc_ns) {
        state.remaining_internal_service_ns -= original_noc_ns;
        state.remaining_internal_service_ns += remaining_noc_ns;
      }
      if (!current_memory) {
        state.remaining_actual_bytes = std::max(
            0.0, state.remaining_actual_bytes - stage.mem.actual_bytes);
        state.remaining_service_units = std::max(
            0.0, state.remaining_service_units - stage.mem.service_units);
      }
    } else if (active.resource == StageResource::MEM && !current_memory) {
      state.remaining_actual_bytes = std::max(
          0.0, state.remaining_actual_bytes - stage.mem.actual_bytes);
      state.remaining_service_units = std::max(
          0.0, state.remaining_service_units - stage.mem.service_units);
    } else if (active.resource == StageResource::TIMER &&
               state.remaining_internal_service_ns >= active.original_service_ns) {
      state.remaining_internal_service_ns -= active.original_service_ns;
      state.remaining_internal_service_ns += active.remaining_service_ns;
    }
  }
  return progress;
}

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

TimerStretchPreview Execution::preview_consecutive_internal_timers() const {
  // The preview walk is single-driver and Impl::run has no external callbacks
  // or owned payloads. Borrow the dense run arrays, journal their writes, and
  // copy only the four mutable scheduler frontiers. The guard restores the
  // original dense values, scalar cursor, and frontier buffers on every exit.
  Impl& preview = *impl_;
  Impl::PreviewBorrowGuard preview_guard(preview);
  preview.run(&preview_guard.undo);
  TimerStretchPreview result;
  if (preview.finished) {
    result.reaches_done = true;
    return result;
  }
  if (preview.mem_issued_new || preview.mem_outstanding) {
    result.reaches_memory_request = true;
    return result;
  }

  for (;;) {
    std::uint64_t next_tick = preview.next_mem_issue_at;
    if (!preview.events.empty()) {
      next_tick = std::min(next_tick, preview.events.front().first);
    }
    if (next_tick == UINT64_MAX || next_tick <= preview.now) {
      throw DieModelError(
          "preview_consecutive_internal_timers: missing future timer boundary");
    }
    const std::uint64_t delay = next_tick - preview.now;
    if (delay > UINT64_MAX - result.external_delay_ns) {
      throw DieModelError(
          "preview_consecutive_internal_timers: merged delay overflows uint64");
    }
    if (delay > UINT64_MAX - preview.now) {
      throw DieModelError(
          "preview_consecutive_internal_timers: internal clock overflows uint64");
    }
    result.external_delay_ns += delay;
    if (result.timer_action_count == UINT64_MAX) {
      throw DieModelError(
          "preview_consecutive_internal_timers: action count overflows uint64");
    }
    ++result.timer_action_count;
    preview.now = next_tick;
    preview.run(&preview_guard.undo);

    if (preview.finished) {
      result.reaches_done = true;
      return result;
    }
    if (preview.mem_issued_new || preview.mem_outstanding) {
      result.reaches_memory_request = true;
      return result;
    }
  }
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
  m.bump_frontier_generation();
  std::uint64_t end = m.now;  // the memory leg completes at the clock
  if (m.mem_is_memnoc && m.mem_noc_end > end) {
    end = m.mem_noc_end;  // later completion wins (dual occupancy, §6.1)
  }
  m.end[m.mem_inst] = end;
  if (end > m.now) {
    m.active_ids.push_back(m.mem_inst);
  }
  m.mem_free = end;          // the MEM slot frees at the full end
  if (m.mem_is_memnoc) {
    m.noc_free = end;        // ... and so does the NOC slot of a MEM_NOC
  }
  m.push_event(end, m.mem_inst);
  m.run();
}

}  // namespace DieModel
