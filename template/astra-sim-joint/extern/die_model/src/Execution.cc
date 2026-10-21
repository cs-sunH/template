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

Preview transactions (v9): preview_consecutive_internal_timers() runs the
SAME run() control flow on the authoritative scheduler state inside a
journal-based transaction instead of copying the four frontier vectors per
call. heap/events push/pop/slot writes are recorded before they happen and
replayed in reverse on exit (HeapJournal); the dense per-run arrays keep
the v8 PreviewUndoLog; the scalar cursor is snapshotted. The derived
frontiers (future_starts/active_ids) are not maintained inside a
transaction -- their only producer, activate_due_starts(), feeds slice
queries that neither run()'s scheduling decisions nor the preview return
value read -- so the walk's cost tracks the heap paths and instances it
actually touches rather than the entry frontier size.
*/

#include "DieModel.hh"


#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace DieModel {

namespace {

// Perf item 5.3 escape hatch: ASTRA_DIE_FLAT_SHAPE=0 disables the per-plan
// shared FlatShape cache; every Execution then builds a private shape
// (identical construction work and semantics, zero cross-Execution
// sharing).  Any other value (including unset) keeps the sharing on.
bool flat_shape_sharing_enabled() {
  const char* value = std::getenv("ASTRA_DIE_FLAT_SHAPE");
  return !(value != nullptr && std::string(value) == "0");
}

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
        : hardware.core_row.pe_int4_gops_per_core * 1e9;
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

// ---------------------------------------------------------------------------
// Journaled min-heap primitives (v9 preview transactions).
//
// sift_up/sift_pop are the single heap discipline shared by the authoritative
// run() and the preview transaction, so container behavior cannot drift
// between real advancement and preview. All three scheduling heaps
// (heap/events/future_starts) hold DISTINCT elements -- an instance id is
// scheduled, activated, issued or completed exactly once, and event pairs
// are keyed by that unique id -- so pop order and front() are independent of
// the internal array layout and identical to the previous
// std::push_heap/std::pop_heap discipline with the std::greater<> min-heap
// comparators.

// 1 + floor(log2(heap_size)): the depth of the deepest heap level one sift
// can touch (perf item 4.8).  GCC/Clang use the count-leading-zeros builtin
// (std::bit_width is C++20 and this library is C++17); other toolchains
// keep the original shift loop.  heap_size == 0 must yield 1 to match the
// loop (and would yield 0 from a naive bit_width).
inline std::size_t sift_depth(std::size_t heap_size) {
#if defined(__GNUC__) || defined(__clang__)
  if (heap_size == 0) {
    return 1;
  }
  return static_cast<std::size_t>(64 - __builtin_clzll(heap_size));
#else
  std::size_t depth = 1;
  while ((heap_size >>= 1) != 0) ++depth;
  return depth;
#endif
}

template <typename Vec>
class HeapJournal {
 public:
  using Value = typename Vec::value_type;

  struct Entry {
    enum class Op : std::uint8_t { kSlotWrite, kPushBack, kPopBack };
    Op op;
    std::size_t index;  // kSlotWrite only
    Value value;        // kSlotWrite: pre-write slot value;
                        // kPopBack: the removed tail element
  };

  static_assert(std::is_nothrow_copy_constructible_v<Value> &&
                    std::is_nothrow_move_assignable_v<Value>,
                "rollback restores heap values by copy-construct (push_back "
                "on spare capacity) and move-assign (slot write); no-throw "
                "variants keep the noexcept reverse replay provable");

  void begin(Vec& vec) {
    if (engaged_) {
      throw DieModelError("Execution: nested preview transaction");
    }
    engaged_ = true;
    vec_ = &vec;
    records_ = 0;
    peak_ = 0;
    entries_.clear();  // scratch capacity is reused across previews
  }

  bool engaged() const { return engaged_; }

  // Reserve for the worst-case journal length of ONE sift operation before
  // any heap slot is touched: a push journals one length record plus at
  // most one slot write per tree level; a pop journals one length record
  // plus at most two writes per level (descend plus reinsert climb).
  void reserve_for_sift(std::size_t heap_size, bool is_pop) {
    const std::size_t depth = sift_depth(heap_size);
    const std::size_t bound = is_pop ? 2 * depth + 2 : depth + 2;
    if (entries_.size() + bound > entries_.capacity()) {
      // Geometric growth: capacity is scratch, not model state (rollback only
      // ever writes onto already-reached capacity). Exact-size growth turned
      // each record into a full realloc+copy (O(N^2) per mega walk).
      const std::size_t geometric =
          entries_.capacity() == 0 ? 16 : entries_.capacity() * 2;
      entries_.reserve(std::max(entries_.size() + bound, geometric));
    }
  }

  void record_slot(std::size_t index, const Value& old_value) {
    entries_.push_back(Entry{Entry::Op::kSlotWrite, index, old_value});
    note_record();
  }
  void record_push_back() {
    entries_.push_back(Entry{Entry::Op::kPushBack, 0, Value{}});
    note_record();
  }
  void record_pop_back(const Value& removed) {
    entries_.push_back(Entry{Entry::Op::kPopBack, 0, removed});
    note_record();
  }

  // Reverse replay restores the exact logical content without allocating:
  // every pop-back restore is a push_back onto capacity this transaction
  // already reached (vector capacity never shrinks here), which cannot
  // throw for trivially copyable values.
  void rollback() noexcept {
    if (!engaged_) return;
    for (auto it = entries_.rbegin(); it != entries_.rend(); ++it) {
      switch (it->op) {
        case Entry::Op::kSlotWrite:
          // Move-assign: the journal entry is dead after this replay, and
          // move-assignment of the heap value types is provably no-throw.
          (*vec_)[it->index] = std::move(it->value);
          break;
        case Entry::Op::kPushBack:
          vec_->pop_back();
          break;
        case Entry::Op::kPopBack:
          vec_->push_back(it->value);
          break;
      }
    }
    entries_.clear();
    vec_ = nullptr;
    engaged_ = false;
  }

  std::uint64_t records() const { return records_; }
  std::uint64_t peak() const { return peak_; }

 private:
  void note_record() {
    ++records_;
    if (entries_.size() > peak_) peak_ = static_cast<std::uint64_t>(entries_.size());
  }

  Vec* vec_ = nullptr;
  bool engaged_ = false;
  std::vector<Entry> entries_;  // persistent scratch; capacity reused
  std::uint64_t records_ = 0;
  std::uint64_t peak_ = 0;
};

// Min-heap sift-up of the element at index i (record-before-write under an
// engaged journal; a journaling failure leaves this operation unwritten).
template <typename Vec>
void sift_up(Vec& v, std::size_t i, HeapJournal<Vec>* journal) {
  const auto value = v[i];
  std::size_t hole = i;
  while (hole > 0) {
    const std::size_t parent = (hole - 1) / 2;
    if (!(v[parent] > value)) break;  // parent <= value: min-heap holds
    if (journal != nullptr) journal->record_slot(hole, v[hole]);
    v[hole] = v[parent];
    hole = parent;
  }
  if (hole != i) {
    if (journal != nullptr) journal->record_slot(hole, v[hole]);
    v[hole] = value;
  }
}

// No-journal overload for the authoritative (non-transaction) path.
template <typename Vec>
void sift_up(Vec& v, std::size_t i) {
  sift_up(v, i, static_cast<HeapJournal<Vec>*>(nullptr));
}

// Min-heap pop: removes and returns the minimum (v[0]) by relocating the
// tail element into the tree (descend promoting the smaller child, then
// climb while smaller than the parent). The removed tail's slot beyond the
// new length keeps its old bytes; the journal restores both on rollback.
template <typename Vec>
typename Vec::value_type sift_pop(Vec& v, HeapJournal<Vec>* journal) {
  const auto result = v[0];
  const auto tail = v.back();
  if (journal != nullptr) journal->record_pop_back(tail);
  v.pop_back();
  if (v.empty()) return result;
  std::size_t hole = 0;
  const std::size_t limit = v.size();
  while (true) {
    std::size_t child = 2 * hole + 1;
    if (child >= limit) break;
    if (child + 1 < limit && v[child + 1] < v[child]) ++child;
    if (!(v[child] < tail)) break;
    if (journal != nullptr) journal->record_slot(hole, v[hole]);
    v[hole] = v[child];
    hole = child;
  }
  while (hole > 0) {
    const std::size_t parent = (hole - 1) / 2;
    if (!(v[parent] > tail)) break;
    if (journal != nullptr) journal->record_slot(hole, v[hole]);
    v[hole] = v[parent];
    hole = parent;
  }
  if (journal != nullptr) journal->record_slot(hole, v[hole]);
  v[hole] = tail;
  return result;
}

// No-journal overload for the authoritative (non-transaction) path.
template <typename Vec>
typename Vec::value_type sift_pop(Vec& v) {
  return sift_pop(v, static_cast<HeapJournal<Vec>*>(nullptr));
}

}  // namespace

// Joint perf R3 M0 (2026-10-17): process-wide preview diagnostics export
// (preview_diagnostics_totals, declared in DieTypes.hh).  Destroyed
// instances bank their E/T totals here at destruction and every completed
// preview transaction banks its per-transaction attribution inside the
// PreviewTxn::rollback merge; live instances are read through their public
// counter accessors at collection time instead.  The preview entry points
// deliberately never touch these variables (they are the hot path); the
// registry holds at most the concurrently alive instances (a handful under
// the single-flight driver contract), so the linear scans are free.
// Single-threaded per the Execution driver contract -- no locking.
namespace {

PreviewDiagnosticsTotals g_preview_diagnostics;
std::vector<const Execution*> g_live_preview_executions;

}  // namespace

struct Execution::Impl {
  // Bound once at construction: the plan-invariant flattened shape is
  // shared across every Execution of the same plan (perf item 5.3); the
  // const references below alias into it so every run()/preview/
  // next_action read point keeps its exact spelling and value.  The shape
  // never changes after construction and holds no per-run state.
  Impl(std::shared_ptr<const DiePlan> plan_in,
       std::shared_ptr<const FlatShape> shape_in)
      : plan(std::move(plan_in)),
        shape(std::move(shape_in)),
        offset(shape->offset),
        stage_of(shape->stage_of),
        dep_start(shape->dep_start),
        dep_to(shape->dep_to),
        rdep_start(shape->rdep_start),
        rdep_to(shape->rdep_to),
        dur_ns(shape->dur_ns),
        noc_leg_ns(shape->noc_leg_ns) {
    stages = &plan->stages();
    hw = &plan->hardware();
    total = shape->total;
  }

  // --- plan-derived data ---
  std::shared_ptr<const DiePlan> plan;
  const std::vector<DieStage>* stages = nullptr;
  const Hardware* hw = nullptr;
  mutable std::optional<std::string> template_key_cache;
  mutable std::uint64_t template_key_computations = 0;
  // Immutable-after-construction flattened shape (see the constructor above).
  std::shared_ptr<const FlatShape> shape;
  const std::vector<std::uint64_t>& offset;      // stage -> flat range
  const std::vector<std::uint32_t>& stage_of;    // flat id -> stage index
  const std::vector<std::uint32_t>& dep_start;   // CSR dependees (per inst)
  const std::vector<std::uint32_t>& dep_to;
  const std::vector<std::uint32_t>& rdep_start;  // CSR reverse edges
  const std::vector<std::uint32_t>& rdep_to;
  const std::vector<std::uint64_t>& dur_ns;      // internal stages (0 MEM*)
  const std::vector<std::uint64_t>& noc_leg_ns;  // MEM_NOC NoC leg
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
  // v8/v9 preview attribution (always-on integer diagnostics, same caliber
  // as the active_id_* family): E=preview_calls entry attempts, T=walked
  // timer boundaries, U=dense undo records merged at EVERY exit, F=frontier
  // snapshot copy entries (structurally zero since the v9 transactional
  // restructure; it would only grow if a copy path returned). The v9 §3.1
  // additions: J=successful frontier undo/overlay records (repeated
  // overwrites counted), per-container entry sizes S (total and max),
  // undo/journal peaks, exit categories and transaction capacity growth.
  std::uint64_t preview_calls = 0;
  std::uint64_t preview_walk_timers = 0;
  std::uint64_t preview_undo_entries = 0;
  std::uint64_t preview_frontier_entries = 0;
  std::uint64_t preview_j_entries = 0;
  std::uint64_t preview_s_heap = 0;
  std::uint64_t preview_s_events = 0;
  std::uint64_t preview_s_future_starts = 0;
  std::uint64_t preview_s_active_ids = 0;
  std::uint64_t preview_s_heap_max = 0;
  std::uint64_t preview_s_events_max = 0;
  std::uint64_t preview_s_future_starts_max = 0;
  std::uint64_t preview_s_active_ids_max = 0;
  std::uint64_t preview_copy_bytes = 0;            // frontier bytes copied (0)
  std::uint64_t preview_active_scan_entries = 0;   // active-id scans in preview (0)
  std::uint64_t preview_undo_peak = 0;
  std::uint64_t preview_journal_peak = 0;
  std::uint64_t preview_exit_memory = 0;
  std::uint64_t preview_exit_done = 0;
  std::uint64_t preview_exit_exception = 0;
  std::uint64_t preview_heap_growth_events = 0;
  std::uint64_t preview_heap_growth_moves = 0;
  std::uint64_t preview_events_growth_events = 0;
  std::uint64_t preview_events_growth_moves = 0;
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

  // Perf item 4.4 (2026-10-16): true iff an authoritative run() (txn ==
  // nullptr) has reached its fixed point and nothing outside run() has
  // mutated the state it reads since.  next_action() uses it to skip the
  // provably idempotent re-run; every out-of-run mutation (clock, memory
  // completion, frontier writes) and every preview entry invalidate it.
  // Conservative by construction: an unnecessary false only costs one
  // extra run(), which is itself a no-op at the fixed point.
  bool state_clean_ = false;

  // --- v11 Phase 4 (K2 epoch-keyed timer-window memo, 方案 §6.2) ---
  //
  // query_internal_timer_window is a read-only proof query ("Read-only
  // proof ... does not clone or replay", DieTypes.hh): it reads this Impl
  // through a const reference, and its only writes are the pre-existing
  // value-deterministic lazy caches (template_key_cache /
  // template_key_computations -- pure derivations of the immutable plan
  // data) plus the memo below. Therefore: identical Impl state + identical
  // max_count => identical return value and identical throw behavior (the
  // throwing paths depend only on state and arguments).
  //
  // state_epoch is bumped by EVERY public entry that can mutate Impl state
  // observable by that query. Audited mutation sites (keep in sync):
  //   1. Execution::next_action()          -- run() + mem_issued_new write
  //   2. Execution::advance_time()         -- now += ns; run()
  //   3. Execution::complete_memory()      -- mem_* clear, end/event, run()
  //   4. Execution::advance_internal_service()         -- via advance_time
  //   5. Execution::advance_internal_service_window()  -- via advance_time
  //   6. Execution::advance_external_wait()            -- via advance_time
  //   7. Execution::preview_consecutive_internal_timers() -- PreviewTxn
  // run() and every Impl helper (activate_due_starts / complete_due /
  // complete_instance / compact_completed_active_ids / schedule_internal /
  // push_* / pop_*) are called only from those entries (or from the
  // constructor, which runs before any memo exists). Fields read by the
  // query chain are: the immutable post-construction plan arrays (offset /
  // stage_of / dep_* / rdep_* / dur_ns / noc_leg_ns / total / stages / hw)
  // plus now / finished / mem_* / next_mem_issue_at / events /
  // future_starts / active_ids / done / start_ns / end / unresolved /
  // frontier_generation -- every write to the latter group happens inside
  // an entry that bumped the epoch first.
  //
  // PreviewTxn audit (v9 heritage): the preview walk mutates the
  // authoritative scheduler state through the journaled transaction and
  // rolls it back bit-exactly on exit (HeapJournal reverse replay +
  // PreviewUndoLog + PreviewScalarState). The epoch bump at ENTRY (7)
  // still invalidates the memo afterwards: the rollback restores queried
  // state, not the epoch, which only over-invalidates (a miss where a hit
  // would also have been correct) and can never serve a stale entry. No
  // query can observe the transaction's intermediate states: run() has no
  // callbacks and the driver is single-threaded. Preview capacity growth
  // (heap.reserve inside the transaction) persists past rollback, but the
  // query reads container CONTENTS only, and contents are restored.
  //
  // Perf item 4.4 interaction (union note, 2026-10-18): next_action()
  // bumps the epoch even when the 4.4 fixed-point skip avoids the actual
  // run() -- the skipped run() is provably a no-op, so the bump only
  // over-invalidates the memo, never under-invalidates it (the same safe
  // direction as every rejected/zero-delta entry above).
  std::uint64_t state_epoch = 0;
  struct TimerWindowMemoSlot {
    bool valid = false;
    std::uint64_t epoch = 0;
    InternalTimerWindowInfo value;
  };
  // Fixed 8 slots indexed by max_count-1 (the legal argument domain is
  // [1,8]); each slot self-invalidates lazily against state_epoch. mutable:
  // written by the const read-only query, like template_key_cache.
  TimerWindowMemoSlot timer_window_memo[8];
  mutable std::uint64_t timer_window_memo_hits = 0;
  mutable std::uint64_t timer_window_memo_misses = 0;

  // --- v12 Phase 4 (K2 structural negative cache, 方案 §6.2) ---
  //
  // P_struct(plan), computed ONCE at construction from the immutable
  // flattened arrays below: true iff every Execution of this plan is
  // finished at (or before) the constructor's run() -- i.e.
  //   P_A: total == 0 (constructor sets finished immediately), or
  //   P_B: every instance is a zero-duration internal stage (dur_ns == 0)
  //        and no MEM/MEM_NOC stage has instances (any memory instance
  //        would leave an outstanding request instead).
  // For such plans the reachable state set of ANY drive history is the
  // singleton {finished}, so query_internal_timer_window provably returns
  // the constant rejection {supported=false, reason="execution-finished",
  // default token, empty slices} -- the only rejection class whose exact
  // reason field is state-independent (Phase 4 audit,
  // docs/PHASE4_K2_CACHE_DESIGN.md). The flag gates both the probe and the
  // write in query_internal_timer_window, so the process-level negative
  // cache only ever fires where returning the stored value is byte-identical
  // to recomputation. Production fire-face: effectively empty (production
  // plans carry positive launch/compute/memory durations).
  bool structural_negative = false;
  mutable std::uint64_t negative_cache_hits = 0;
  mutable std::uint64_t negative_cache_writes = 0;
  struct PreviewUndoLog {
    enum class Field { Done, CompletedByStage, Unresolved, StartNs, End };
    struct Entry {
      Field field;
      std::size_t index;
      std::uint64_t old_value;
    };
    std::vector<Entry> entries;

    void begin() {
      if (engaged_) {
        throw DieModelError("Execution: nested preview transaction");
      }
      engaged_ = true;
      entries.clear();  // scratch capacity is reused across previews
    }

    bool engaged() const { return engaged_; }

    // Drop engagement without replay (transaction construction failed
    // before any record was made).
    void disengage() noexcept { engaged_ = false; }

    // Pre-reserve for the worst-case record count of one model call before
    // its first write, so a journaling failure leaves that call unwritten.
    void reserve_more(std::size_t n) {
      if (entries.size() + n > entries.capacity()) {
        // Geometric growth: capacity is scratch, not model state (rollback
        // only replays recorded writes). Exact-size growth turned each record
        // into a full realloc+copy (O(N^2) per mega walk).
        const std::size_t geometric =
            entries.capacity() == 0 ? 16 : entries.capacity() * 2;
        entries.reserve(std::max(entries.size() + n, geometric));
      }
    }

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
      engaged_ = false;
    }

   private:
    bool engaged_ = false;
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

  // v9 preview-transaction scratch (persistent across previews so the
  // steady-state transaction performs no scratch allocation; engaged only
  // inside preview_consecutive_internal_timers).
  HeapJournal<std::vector<std::uint64_t>> heap_journal;
  HeapJournal<std::vector<std::pair<std::uint64_t, std::uint32_t>>>
      events_journal;
  PreviewUndoLog preview_undo;

  // v9 preview transaction: exact journal-based rollback of the scheduler
  // state the walk actually touches -- heap/events slot writes, appends and
  // pop-backs (HeapJournal), the dense per-run arrays (PreviewUndoLog), and
  // the scalar cursor (PreviewScalarState). The two DERIVED frontiers are
  // proven walk-irrelevant and are simply not maintained inside a
  // transaction (real advancement keeps full maintenance):
  //   - future_starts is read only by activate_due_starts(), whose sole
  //     effects are popping it and appending active_ids;
  //   - active_ids is written only by activate_due_starts/schedule_internal/
  //     complete_memory and read only by the slice-query observers
  //     (describe_internal_slice), never by run()'s scheduling decisions
  //     or the preview return value.
  // Both skips therefore cannot change external_delay_ns, timer_action_count
  // or the Memory/Done stop boundary. This replaces the v8 per-call copy of
  // all four frontier vectors (the O(entry-size) F cost).
  // Rollback audit (v11 heritage, carried verbatim from the retired
  // epoch-keyed memo): the walk mutates the authoritative scheduler state
  // through the journaled transaction and rolls it back bit-exactly at
  // exit. No query can observe the transaction's intermediate states:
  // run() has no callbacks and the driver is single-threaded. Preview
  // capacity growth (heap.reserve inside the transaction) persists past
  // rollback, but observers read container CONTENTS only, and contents are
  // restored.
  struct PreviewTxn {
    enum class Exit : std::uint8_t { kPending, kMemoryRequest, kDone };

    Impl& state;
    PreviewScalarState original_scalars;
    Exit exit = Exit::kPending;

    explicit PreviewTxn(Impl& value)
        : state(value), original_scalars(value) {
      try {
        state.heap_journal.begin(state.heap);
        state.events_journal.begin(state.events);
        state.preview_undo.begin();
      } catch (...) {
        state.events_journal.rollback();  // no-ops unless engaged
        state.heap_journal.rollback();
        state.preview_undo.disengage();
        throw;
      }
    }

    PreviewTxn(const PreviewTxn&) = delete;
    PreviewTxn& operator=(const PreviewTxn&) = delete;

    ~PreviewTxn() { rollback(); }

    void note_exit(Exit kind) noexcept { exit = kind; }

    void rollback() noexcept {
      // Reverse replay of every touched container write, then the dense
      // undo log, then the scalar cursor; every step is non-throwing.
      const std::uint64_t undo_count =
          static_cast<std::uint64_t>(state.preview_undo.entries.size());
      state.events_journal.rollback();
      state.heap_journal.rollback();
      state.preview_undo.rollback(state);
      original_scalars.restore(state);
      // Attribution counters are diagnostics, not model state: U/J and the
      // peaks merge here at EVERY exit path, so exception exits count too.
      state.preview_undo_entries += undo_count;
      state.preview_j_entries += state.heap_journal.records() +
                                 state.events_journal.records();
      state.preview_undo_peak =
          std::max(state.preview_undo_peak, undo_count);
      state.preview_journal_peak =
          std::max(state.preview_journal_peak,
                   state.heap_journal.peak() + state.events_journal.peak());
      // Joint perf R3 M0: bank the same per-transaction attribution into
      // the process-wide diagnostics aggregate.  Pure counting alongside
      // the merge above (this runs exactly once per transaction because
      // rollback is invoked only by ~PreviewTxn); HeapJournal::rollback
      // keeps records_ until the next begin(), so records() still reports
      // this transaction's count here.  Integer arithmetic only, so the
      // noexcept contract of rollback is preserved.
      const std::uint64_t journal_records = state.heap_journal.records() +
                                            state.events_journal.records();
      g_preview_diagnostics.txn_exits += 1;
      g_preview_diagnostics.undo_sum += undo_count;
      g_preview_diagnostics.undo_max_per_txn =
          std::max(g_preview_diagnostics.undo_max_per_txn, undo_count);
      g_preview_diagnostics.journal_sum += journal_records;
      g_preview_diagnostics.journal_max_per_txn =
          std::max(g_preview_diagnostics.journal_max_per_txn,
                   journal_records);
      switch (exit) {
        case Exit::kMemoryRequest:
          ++state.preview_exit_memory;
          break;
        case Exit::kDone:
          ++state.preview_exit_done;
          break;
        case Exit::kPending:  // destructor without a normal exit: exception
          ++state.preview_exit_exception;
          break;
      }
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

  void push_heap_key(std::uint64_t key, PreviewTxn* txn = nullptr) {
    if (txn == nullptr) {
      heap.push_back(key);
      sift_up(heap, heap.size() - 1);
      return;
    }
    // Reserve the worst-path journal BEFORE touching the heap; on failure
    // this push has not happened at all.
    heap_journal.reserve_for_sift(heap.size() + 1, /*is_pop=*/false);
    if (heap.size() == heap.capacity()) {
      // Capacity is not model state: growth persists past rollback and is
      // reported instead of being undone.
      const std::size_t moved = heap.size();
      heap.reserve(std::max<std::size_t>(heap.capacity() * 2, 16));
      ++preview_heap_growth_events;
      preview_heap_growth_moves += moved;
    }
    heap_journal.record_push_back();
    heap.push_back(key);
    sift_up(heap, heap.size() - 1, &heap_journal);
  }
  std::uint64_t pop_heap_key(PreviewTxn* txn = nullptr) {
    if (txn == nullptr) {
      return sift_pop(heap);
    }
    heap_journal.reserve_for_sift(heap.size(), /*is_pop=*/true);
    return sift_pop(heap, &heap_journal);
  }
  void push_event(std::uint64_t end_ns, std::uint32_t id,
                  PreviewTxn* txn = nullptr) {
    if (txn == nullptr) {
      events.emplace_back(end_ns, id);
      sift_up(events, events.size() - 1);
      return;
    }
    events_journal.reserve_for_sift(events.size() + 1, /*is_pop=*/false);
    if (events.size() == events.capacity()) {
      const std::size_t moved = events.size();
      events.reserve(std::max<std::size_t>(events.capacity() * 2, 16));
      ++preview_events_growth_events;
      preview_events_growth_moves += moved;
    }
    events_journal.record_push_back();
    events.emplace_back(end_ns, id);
    sift_up(events, events.size() - 1, &events_journal);
  }
  std::pair<std::uint64_t, std::uint32_t> pop_event(PreviewTxn* txn = nullptr) {
    if (txn == nullptr) {
      return sift_pop(events);
    }
    events_journal.reserve_for_sift(events.size(), /*is_pop=*/true);
    return sift_pop(events, &events_journal);
  }

  void push_future_start(std::uint64_t start_time, std::uint32_t id) {
    future_starts.emplace_back(start_time, id);
    sift_up(future_starts, future_starts.size() - 1);
  }

  void activate_due_starts(PreviewTxn* txn = nullptr) {
    if (txn != nullptr) {
      // Derived-frontier feed only (see PreviewTxn): inside a preview
      // transaction the pops and appends are skipped entirely -- the walk
      // neither reads future_starts nor active_ids afterwards.
      return;
    }
    while (!future_starts.empty() && future_starts.front().first <= now) {
      const auto start = sift_pop(future_starts);
      if (!done[start.second] && end[start.second] > now) {
        active_ids.push_back(start.second);
      }
    }
  }

  // S sampling for the v9 attribution (§3.1): the four frontier entry sizes
  // at preview entry, kept separate from the (now structural zero) copy F.
  void note_preview_entry_sizes() {
    const std::uint64_t h = static_cast<std::uint64_t>(heap.size());
    const std::uint64_t e = static_cast<std::uint64_t>(events.size());
    const std::uint64_t f = static_cast<std::uint64_t>(future_starts.size());
    const std::uint64_t a = static_cast<std::uint64_t>(active_ids.size());
    preview_s_heap += h;
    preview_s_events += e;
    preview_s_future_starts += f;
    preview_s_active_ids += a;
    preview_s_heap_max = std::max(preview_s_heap_max, h);
    preview_s_events_max = std::max(preview_s_events_max, e);
    preview_s_future_starts_max = std::max(preview_s_future_starts_max, f);
    preview_s_active_ids_max = std::max(preview_s_active_ids_max, a);
  }

  void complete_instance(std::uint32_t id, std::uint64_t end_ns,
                         PreviewTxn* const txn) {
    if (txn != nullptr) {
      // Worst case below: Done + CompletedByStage + one Unresolved per
      // reverse edge; reserved before the first write of this call.
      preview_undo.reserve_more(
          2 + (rdep_start[id + 1] - rdep_start[id]));
      preview_undo.record(PreviewUndoLog::Field::Done, id,
                          done[id] ? 1 : 0);
    }
    done[id] = true;
    const std::uint32_t stage_id = stage_of[id];
    if (completed_by_stage[stage_id] == UINT64_MAX) {
      throw DieModelError("Execution: completed stage count overflows uint64");
    }
    if (txn != nullptr) {
      preview_undo.record(PreviewUndoLog::Field::CompletedByStage,
                          stage_id, completed_by_stage[stage_id]);
    }
    ++completed_by_stage[stage_id];
    // The old per-instance erase/remove ran here. No observer or scheduler
    // callback can inspect active_ids until complete_due() returns, so defer
    // removal and compact once after all same-boundary completions. Inside a
    // preview transaction the live-frontier bookkeeping is skipped: the walk
    // never reads active_ids, and the dirty flag is part of the scalar
    // snapshot restored at rollback.
    if (txn == nullptr) {
      active_ids_dirty = true;
    }
    bump_frontier_generation();
    --remaining;
    elapsed = std::max(elapsed, end_ns);
    for (std::uint32_t e = rdep_start[id]; e < rdep_start[id + 1]; ++e) {
      const std::uint32_t t = rdep_to[e];
      if (txn != nullptr) {
        preview_undo.record(PreviewUndoLog::Field::Unresolved, t,
                            unresolved[t]);
      }
      if (--unresolved[t] == 0) {
        const std::uint32_t ts = stage_of[t];
        push_heap_key((static_cast<std::uint64_t>(ts) << 32) |
                      static_cast<std::uint64_t>(t - offset[ts]), txn);
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

  void complete_due(PreviewTxn* const txn) {
    try {
      while (!events.empty() && events.front().first <= now) {
        const auto ev = pop_event(txn);
        if (!done[ev.second]) {
          complete_instance(ev.second, ev.first, txn);
        }
      }
    } catch (...) {
      // The catch-branch compaction is real-path derived maintenance; a
      // preview transaction skips the scan and restores the flag via the
      // scalar snapshot, leaving the real container untouched either way.
      if (active_ids_dirty && txn == nullptr) compact_completed_active_ids();
      throw;
    }
    if (active_ids_dirty && txn == nullptr) compact_completed_active_ids();
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
                         PreviewTxn* const txn) {
    if (txn != nullptr) {
      preview_undo.reserve_more(2);  // StartNs + End below
    }
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
    if (txn != nullptr) {
      preview_undo.record(PreviewUndoLog::Field::StartNs, id, start_ns[id]);
    }
    start_ns[id] = start;
    if (txn != nullptr) {
      preview_undo.record(PreviewUndoLog::Field::End, id, end[id]);
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
    push_event(fin, id, txn);
    if (txn == nullptr) {
      // Real-path derived-frontier maintenance only; a preview transaction
      // skips both branches (see PreviewTxn).
      if (start <= now && fin > now) {
        active_ids.push_back(id);
      } else if (start > now) {
        push_future_start(start, id);
      }
    }
  }

  void run(PreviewTxn* const txn = nullptr) {
    if (txn == nullptr) {
      // Perf item 4.4: entry invalidation; a mid-run throw leaves the flag
      // false (the machine is then re-run at the next next_action).
      state_clean_ = false;
    }
    next_mem_issue_at = UINT64_MAX;
    for (;;) {
      activate_due_starts(txn);
      complete_due(txn);
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
        pop_heap_key(txn);
        mem_outstanding = true;
        bump_frontier_generation();
        mem_issued_new = true;
        mem_stage = s;
        mem_inst = id;
        mem_is_memnoc = st.resource == StageResource::MEM_NOC;
        mem_start_ns = start;
        if (txn != nullptr) {
          preview_undo.reserve_more(1);
          preview_undo.record(PreviewUndoLog::Field::StartNs, id,
                              start_ns[id]);
        }
        start_ns[id] = start;
        mem_noc_end = start + noc_leg_ns[id];
        mem_actual_bytes = st.mem.actual_bytes;
        mem_service_units = st.mem.service_units;
        break;  // one outstanding request; nothing else pops before completion
      }
      pop_heap_key(txn);
      schedule_internal(id, st, txn);
    }
    if (remaining == 0) {
      finished = true;
    }
    if (txn == nullptr) {
      // Perf item 4.4: the authoritative exit leaves the scheduler at its
      // fixed point (heap/events/memory all quiescent for the current
      // clock), so an immediate re-run is provably a no-op.  Preview
      // transactions deliberately do NOT set the flag: their rollback is
      // verified separately and clean correctness must not depend on it.
      state_clean_ = true;
    }
  }
};

// ---------------------------------------------------------------------------
// Construction: flatten the plan, reduce deps, validate, seed the heap.

namespace detail {

// Perf item 5.3: the plan-invariant construction work of Execution, moved
// verbatim from the stage A3 constructor -- CSR flattening, the single-
// dependee dependency reduction, §6.1 integer durations and the Kahn
// feasibility check.  Built once per DiePlan (lazily, by the first
// Execution of that plan, or privately per Execution when sharing is
// disabled via ASTRA_DIE_FLAT_SHAPE=0).  The result holds no per-run state,
// so sharing it between Executions cannot couple their runs; the Kahn
// check keeps the cyclic-plan throw at the first Execution construction
// with the original message.
std::shared_ptr<const FlatShape> build_flat_shape(
    const std::vector<DieStage>& stages, const Hardware& hw) {
  std::shared_ptr<FlatShape> built = std::make_shared<FlatShape>();
  FlatShape& s = *built;
  const std::size_t ns = stages.size();

  s.offset.assign(ns + 1, 0);
  std::size_t total = 0;
  for (std::size_t st = 0; st < ns; ++st) {
    total += static_cast<std::size_t>(stages[st].wave_repeat);
    s.offset[st + 1] = total;
  }
  s.total = total;
  s.stage_count = ns;
  if (total == 0) {
    return built;
  }
  s.stage_of.assign(total, 0);
  s.dur_ns.assign(total, 0);
  s.noc_leg_ns.assign(total, 0);
  s.initial_unresolved.assign(total, 0);
  for (std::size_t st = 0; st < ns; ++st) {
    for (std::size_t id = s.offset[st]; id < s.offset[st + 1]; ++id) {
      s.stage_of[id] = static_cast<std::uint32_t>(st);
    }
  }

  // Dependency reduction to single dependee instances (same reduction as
  // evaluate_stages) + durations from the shared §6.1 truth.
  s.dep_start.assign(total + 1, 0);
  std::vector<std::uint32_t> rdep_count(total, 0);
  auto dependee_of = [&](std::uint64_t j, const DieStageDep& dep) -> std::int64_t {
    if (dep.stage >= ns || stages[dep.stage].wave_repeat == 0) {
      return -1;
    }
    if (dep.lag == DieStageDep::kAllDep) {
      return static_cast<std::int64_t>(stages[dep.stage].wave_repeat) - 1;
    }
    if (static_cast<std::int64_t>(j) >=
        static_cast<std::int64_t>(dep.lag)) {
      return static_cast<std::int64_t>(j) -
             static_cast<std::int64_t>(dep.lag);
    }
    return -1;
  };
  for (std::size_t st = 0; st < ns; ++st) {
    const DieStage& stage = stages[st];
    for (std::uint64_t j = 0; j < stage.wave_repeat; ++j) {
      const std::size_t id = s.offset[st] + static_cast<std::size_t>(j);
      if (stage.resource == StageResource::MEM ||
          stage.resource == StageResource::MEM_NOC) {
        s.noc_leg_ns[id] =
            seconds_to_ns(detail::stage_noc_leg_seconds(stage, hw));
      } else {
        s.dur_ns[id] =
            seconds_to_ns(detail::stage_instance_seconds(stage, hw));
      }
      std::uint32_t d = 0;
      for (const DieStageDep& dep : stage.deps) {
        const std::int64_t target = dependee_of(j, dep);
        if (target < 0) {
          continue;
        }
        ++d;
        ++rdep_count[s.offset[dep.stage] + static_cast<std::size_t>(target)];
      }
      s.initial_unresolved[id] = d;
      s.dep_start[id + 1] = s.dep_start[id] + d;
    }
  }
  const std::size_t edges = s.dep_start[total];
  s.edge_count = edges;
  s.dep_to.assign(edges, 0);
  s.rdep_to.assign(edges, 0);
  s.rdep_start.assign(total + 1, 0);
  for (std::size_t id = 0; id < total; ++id) {
    s.rdep_start[id + 1] = s.rdep_start[id] + rdep_count[id];
  }
  {
    std::vector<std::uint32_t> dep_fill(s.dep_start.begin(),
                                        s.dep_start.end() - 1);
    std::vector<std::uint32_t> rdep_fill(s.rdep_start.begin(),
                                         s.rdep_start.end() - 1);
    for (std::size_t st = 0; st < ns; ++st) {
      for (std::uint64_t j = 0; j < stages[st].wave_repeat; ++j) {
        const std::size_t id = s.offset[st] + static_cast<std::size_t>(j);
        for (const DieStageDep& dep : stages[st].deps) {
          const std::int64_t target = dependee_of(j, dep);
          if (target < 0) {
            continue;
          }
          const std::size_t tid =
              s.offset[dep.stage] + static_cast<std::size_t>(target);
          s.dep_to[dep_fill[id]++] = static_cast<std::uint32_t>(tid);
          s.rdep_to[rdep_fill[tid]++] = static_cast<std::uint32_t>(id);
        }
      }
    }
  }

  // Kahn feasibility check (fail closed on cycles, like evaluate_stages).
  {
    std::vector<std::uint32_t> unresolved = s.initial_unresolved;
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
      for (std::uint32_t e = s.rdep_start[id]; e < s.rdep_start[id + 1]; ++e) {
        if (--unresolved[s.rdep_to[e]] == 0) {
          stack.push_back(s.rdep_to[e]);
        }
      }
    }
    if (reached != total) {
      throw DieModelError("Execution: stage DAG has a dependency cycle");
    }
  }
  return built;
}

}  // namespace detail

Execution::Execution(std::shared_ptr<const DiePlan> p) {
  if (!p || !p->valid()) {
    throw DieModelError("Execution: plan is empty");
  }
  // The flatten + reduce + Kahn-validate work is plan-invariant and shared
  // through the plan (perf item 5.3); ASTRA_DIE_FLAT_SHAPE=0 builds a
  // private shape per Execution instead.  Either way a cyclic plan throws
  // here -- the first Execution construction -- with the original message.
  std::shared_ptr<const FlatShape> shape =
      flat_shape_sharing_enabled()
          ? p->flat_shape_handle()
          : detail::build_flat_shape(p->stages(), p->hardware());
  impl_.reset(new Impl(std::move(p), std::move(shape)));
  Impl& m = *impl_;

  // --- mutable per-run state, seeded from the shared shape ---
  m.remaining = m.total;
  m.completed_by_stage.assign(m.shape->stage_count, 0);
  if (m.total == 0) {
    m.finished = true;
    // P_A of the v12 Phase 4 structural negative predicate (see Impl).
    m.structural_negative = true;
    // Joint perf R3 M0: register for preview_diagnostics_totals(); both
    // construction success exits register exactly once (every throwing
    // path above stayed unregistered, and a throwing constructor never
    // reaches a destructor, so the registry holds no dangling pointer).
    g_live_preview_executions.push_back(this);
    return;
  }
  m.start_ns.assign(m.total, 0);
  m.end.assign(m.total, 0);
  m.done.assign(m.total, false);
  m.unresolved = m.shape->initial_unresolved;  // per-run copy
  for (std::size_t s = 0; s < m.shape->stage_count; ++s) {
    for (std::uint64_t j = 0; j < (*m.stages)[s].wave_repeat; ++j) {
      const std::size_t id = m.offset[s] + static_cast<std::size_t>(j);
      if (m.unresolved[id] == 0) {
        m.push_heap_key((static_cast<std::uint64_t>(s) << 32) | j);
      }
    }
  }
  // P_B of the v12 Phase 4 structural negative predicate (see Impl): no
  // memory instance exists and every internal instance is zero-duration, so
  // the run() below drains the whole DAG synchronously (§6.4 zero-time
  // rule) and leaves finished == true before any public entry.  The loops
  // read the shared FlatShape through the Impl aliases (perf item 5.3
  // adaptation: dur_ns/stage_count live in the shape, not per-Execution).
  m.structural_negative = true;
  for (std::size_t s = 0; s < m.shape->stage_count; ++s) {
    const DieStage& st = (*m.stages)[s];
    if (st.wave_repeat == 0) continue;
    if (st.resource == StageResource::MEM ||
        st.resource == StageResource::MEM_NOC) {
      m.structural_negative = false;
      break;
    }
  }
  for (std::size_t id = 0; id < m.total && m.structural_negative; ++id) {
    if (m.dur_ns[id] != 0) {
      m.structural_negative = false;
    }
  }
  m.run();
  // Joint perf R3 M0: tail registration, see the early-return note above.
  g_live_preview_executions.push_back(this);
}

Execution::~Execution() {
  // Joint perf R3 M0: first segment, before any member teardown -- drop
  // this instance from the live registry (linear scan; a not-found scan is
  // a safe no-op) and bank its E/T totals into the process-wide aggregate.
  // impl_ is still alive here (the unique_ptr member destroys after this
  // body runs).
  const auto registry_it =
      std::find(g_live_preview_executions.begin(),
                g_live_preview_executions.end(), this);
  if (registry_it != g_live_preview_executions.end()) {
    g_live_preview_executions.erase(registry_it);
  }
  g_preview_diagnostics.entries += impl_->preview_calls;
  g_preview_diagnostics.walk_timers += impl_->preview_walk_timers;
}

bool Execution::finished() const { return impl_->finished; }

std::uint64_t Execution::elapsed_ns() const { return impl_->elapsed; }

std::optional<std::uint32_t> Execution::pending_memory_stage_id() const {
  const Impl& m = *impl_;
  if (!m.mem_outstanding) return std::nullopt;
  return m.mem_stage;
}

std::uint64_t Execution::preview_calls() const {
  return impl_->preview_calls;
}
std::uint64_t Execution::preview_walk_timers() const {
  return impl_->preview_walk_timers;
}
std::uint64_t Execution::preview_undo_entries() const {
  return impl_->preview_undo_entries;
}
std::uint64_t Execution::preview_frontier_entries() const {
  return impl_->preview_frontier_entries;
}

PreviewCounters Execution::preview_counters() const {
  const Impl& m = *impl_;
  PreviewCounters c;
  c.calls = m.preview_calls;
  c.walk_timers = m.preview_walk_timers;
  c.undo_entries = m.preview_undo_entries;
  c.frontier_entries = m.preview_frontier_entries;
  c.j_entries = m.preview_j_entries;
  c.s_heap_total = m.preview_s_heap;
  c.s_events_total = m.preview_s_events;
  c.s_future_starts_total = m.preview_s_future_starts;
  c.s_active_ids_total = m.preview_s_active_ids;
  c.s_heap_max = m.preview_s_heap_max;
  c.s_events_max = m.preview_s_events_max;
  c.s_future_starts_max = m.preview_s_future_starts_max;
  c.s_active_ids_max = m.preview_s_active_ids_max;
  c.copy_bytes = m.preview_copy_bytes;
  c.active_scan_entries = m.preview_active_scan_entries;
  c.undo_peak = m.preview_undo_peak;
  c.journal_peak = m.preview_journal_peak;
  c.exit_memory = m.preview_exit_memory;
  c.exit_done = m.preview_exit_done;
  c.exit_exception = m.preview_exit_exception;
  c.heap_growth_events = m.preview_heap_growth_events;
  c.heap_growth_moves = m.preview_heap_growth_moves;
  c.events_growth_events = m.preview_events_growth_events;
  c.events_growth_moves = m.preview_events_growth_moves;
  return c;
}

PreviewDiagnosticsTotals preview_diagnostics_totals() {
  // Joint perf R3 M0: static aggregates (banked by destroyed instances)
  // plus a live read of every still-alive Execution through the existing
  // public counter accessors -- exact at any call point and independent of
  // destruction order.  Single-threaded per the driver contract.
  PreviewDiagnosticsTotals totals = g_preview_diagnostics;
  for (const Execution* instance : g_live_preview_executions) {
    totals.entries += instance->preview_calls();
    totals.walk_timers += instance->preview_walk_timers();
  }
  return totals;
}

std::uint64_t Execution::template_key_computations_for_test() const {
  return impl_->template_key_computations;
}

std::uint64_t Execution::timer_window_memo_hits_for_test() const {
  return impl_->timer_window_memo_hits;
}

std::uint64_t Execution::timer_window_memo_misses_for_test() const {
  return impl_->timer_window_memo_misses;
}

std::uint64_t Execution::negative_cache_hits_for_test() const {
  return impl_->negative_cache_hits;
}

std::uint64_t Execution::negative_cache_writes_for_test() const {
  return impl_->negative_cache_writes;
}

bool Execution::structural_negative_for_test() const {
  return impl_->structural_negative;
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

bool Execution::partial_api_probe(const std::uint64_t required_service_ns) const {
  // O(1) conservative eligibility probe for the internal-timer arm (perf
  // item 4.5).  Each clause below is a same-state copy of one rejection
  // clause of describe_internal_slice() (or the caller's boundary
  // sufficiency condition); a true result implies describe would reach its
  // eligible terminal state -- a valid PurePe/PureVec token with
  // remaining_service_ns >= required_service_ns.  The only failure
  // direction is a conservative false, and the caller's fallback
  // (advance_time) has an identical machine effect.  No side effects:
  // template_key() is deliberately NOT touched.
  const Impl& m = *impl_;
  // describe: finished / outstanding-memory rejections.
  if (m.finished || m.mem_outstanding) {
    return false;
  }
  // describe: exactly one active stage.  In the post-run state every
  // active_ids member passes describe's activity filter (done members are
  // compacted away; ends never pass `now` without completing), so the
  // filtered size equals the container size.  Checking the filter on the
  // single member explicitly keeps the probe sound in any state.
  if (m.active_ids.size() != 1) {
    return false;
  }
  const std::uint32_t id = m.active_ids[0];
  if (m.done[id] || m.start_ns[id] > m.now || m.end[id] <= m.now) {
    return false;
  }
  // describe: the single active stage must be the next completion event.
  // Events and instances correspond one-to-one (an instance is scheduled
  // exactly once), so the front event's payload id fixes both the boundary
  // time and the instance.  This also subsumes describe's overlap clause:
  // the only frontier member is the current stage itself.
  if (m.events.empty() || m.events.front().second != id) {
    return false;
  }
  const std::uint64_t current_end = m.end[id];
  // describe: future-start / memory-issue boundaries must not precede the
  // current stage's completion.
  if (!m.future_starts.empty() &&
      m.future_starts.front().first < current_end) {
    return false;
  }
  if (m.next_mem_issue_at != UINT64_MAX &&
      m.next_mem_issue_at <= current_end) {
    return false;
  }
  // describe: only pure compute classes reach the eligible terminal state.
  const DieStage& stage = (*m.stages)[m.stage_of[id]];
  if (stage.resource != StageResource::PE &&
      stage.resource != StageResource::VEC) {
    return false;
  }
  // describe: positive in-service stage (start <= now is established above
  // by the activity filter; a zero original service is end == start).
  if (current_end <= m.start_ns[id]) {
    return false;
  }
  // Caller boundary condition.  With every clause above passed, describe's
  // next_boundary is exactly current_end: the memory source is gone
  // (mem_outstanding rejected), and future_starts.front and
  // next_mem_issue_at are either >= current_end or absent.  An end of
  // UINT64_MAX would leave describe's has_next_boundary false, so reject
  // it; otherwise eligible-with-remaining >= required_service_ns reduces
  // to the subtraction below (end > now holds from the activity filter).
  if (current_end == UINT64_MAX) {
    return false;
  }
  return current_end - m.now >= required_service_ns;
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
    throw DieModelError("query_template_slice: only typed PE/VEC service stages are queryable");
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
  // v11 Phase 4 (K2 epoch-keyed memo, 方案 §6.2). Equivalence: the memo
  // returns exactly what the pure computation below returned for the SAME
  // (Impl state, max_count) pair -- the epoch is bumped by every mutating
  // public entry (audit list at Impl::state_epoch), so a slot hit proves
  // the state is unchanged since the stored value was computed, and by the
  // read-only-proof purity argument the recomputation would return the
  // same value. The frozen-snapshot projection phase (no mutations at all)
  // therefore hits on every repeated (execution, max_count) query, while
  // the drive path (state changes between queries) misses every time --
  // zero hits where they would be unsafe, all hits where they are free.
  // Exceptions are never memoized (a throwing computation simply retries,
  // deterministically, like the unmemoized baseline).
  Impl& memo_state = *impl_;
  Impl::TimerWindowMemoSlot& slot =
      memo_state.timer_window_memo[max_count - 1];

  // v12 Phase 4 (方案 §6.2): process-level structural negative cache. The
  // probe runs only for plans whose P_struct holds -- for every other plan
  // this is a single false bool test and the v11 behavior below is
  // bit-identical (counters included). A hit is possible only when the
  // stored rejection equals the full computation's return value in EVERY
  // reachable state of ANY Execution of this plan (reason field included;
  // see the Impl::structural_negative audit note), so serving it cannot
  // change any accept/reject result. The epoch slot is refreshed as well to
  // keep the v11 postcondition (the slot holds the value last returned for
  // the current epoch). Preview transactions never reach this code with
  // transactional intermediate state: preview_consecutive_internal_timers
  // does not call this query at all (v9 audit note at Impl::state_epoch).
  if (memo_state.structural_negative) {
    K2TimerWindowNegativeCache& negative =
        k2_timer_window_negative_cache();
    if (negative.enabled()) {
      const std::optional<InternalTimerWindowInfo> cached =
          negative.find(memo_state.template_key());
      if (cached.has_value()) {
        ++memo_state.negative_cache_hits;
        slot.valid = true;
        slot.epoch = memo_state.state_epoch;
        slot.value = *cached;
        return *cached;
      }
    }
  }

  if (slot.valid && slot.epoch == memo_state.state_epoch) {
    ++memo_state.timer_window_memo_hits;
    return slot.value;
  }
  ++memo_state.timer_window_memo_misses;
  const auto compute = [this, max_count]() -> InternalTimerWindowInfo {
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
  };  // compute (the pre-Phase-4 query body, unchanged)
  const InternalTimerWindowInfo result = compute();
  // v12 Phase 4 (方案 §6.2): negative-only write rule. supported=true is
  // NEVER stored -- the accept path always recomputes. A rejection is stored
  // only when P_struct holds, which is exactly when the stored value is
  // provably what the recomputation returns in every reachable state, so a
  // later hit (any Execution, any drive history) returns byte-identical
  // output. Exceptions are never stored (a throwing computation simply
  // retries, like the unmemoized baseline).
  if (!result.supported && memo_state.structural_negative) {
    K2TimerWindowNegativeCache& negative =
        k2_timer_window_negative_cache();
    if (negative.enabled()) {
      ++memo_state.negative_cache_writes;
      negative.insert(memo_state.template_key(), result);
    }
  }
  slot.valid = true;
  slot.epoch = memo_state.state_epoch;
  slot.value = result;
  return result;
}

std::uint64_t Execution::current_frontier_generation() const {
  return impl_->frontier_generation;
}

void Execution::advance_internal_service(const InternalActionToken& action_token,
                                         const std::uint64_t delta_ns) {
  // K2 memo epoch (v11 Phase 4): entry #4 of the audit at Impl::state_epoch
  // (bump before validation: a rejected call only over-invalidates).
  ++impl_->state_epoch;
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
  // K2 memo epoch (v11 Phase 4): entry #5 of the audit at Impl::state_epoch
  // (bump before validation: a rejected or zero-delta call only
  // over-invalidates the memo, never under-invalidates it).
  ++impl_->state_epoch;
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
  // K2 memo epoch (v11 Phase 4): entry #6 of the audit at Impl::state_epoch.
  ++m.state_epoch;
  if (!m.mem_outstanding || m.mem_stage != expected_memory_stage) {
    throw DieModelError(
        "advance_external_wait: no matching outstanding memory stage");
  }
  if (delta_ns > UINT64_MAX - m.external_memory_wait_elapsed_ns) {
    throw DieModelError(
        "advance_external_wait: accumulated wait overflows uint64");
  }
  advance_time(delta_ns);
  // Perf item 4.4: the accumulation below is an out-of-run mutation that
  // happens after advance_time's run() re-validated the flag.
  m.state_clean_ = false;
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
  // K2 memo epoch (v11 Phase 4): entry #1 of the mutating-surface audit at
  // Impl::state_epoch.  Bumped BEFORE the 4.4 fixed-point skip below: the
  // skipped run() is provably a no-op, so this only over-invalidates the
  // memo (the safe direction); a performed run() mutates observable state
  // and the bump is required.
  ++m.state_epoch;
  // Perf item 4.4: skip the provably idempotent run() when the machine is
  // still at the fixed point left by the last authoritative run() and no
  // out-of-run mutation has happened since.
  if (!m.state_clean_) {
    m.run();
  }
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
    // Perf item 4.4: the consumption write above is an out-of-run mutation
    // of Impl state; invalidate conservatively.
    m.state_clean_ = false;
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
  // or owned payloads. A journal-based transaction (v9) lets the walk run on
  // the authoritative scheduler state itself: heap/events writes are undone
  // in reverse, the dense arrays roll back through the undo log, the scalar
  // cursor is restored from the entry snapshot, and the two derived
  // frontiers (future_starts/active_ids) are not maintained at all during
  // the walk. No frontier copy is taken: the walk's cost follows the heap
  // paths and instances it actually touches, not the entry frontier size.
  Impl& preview = *impl_;
  // K2 memo epoch (v11 Phase 4): entry #7 of the audit at Impl::state_epoch.
  // The transaction below restores every queried field bit-exactly at exit,
  // but the epoch is deliberately NOT restored: a post-preview query misses
  // and recomputes from the restored state, which is always correct. See
  // the PreviewTxn audit note at Impl::state_epoch.  Bumped before the B1
  // short-circuit below: that path performs no writes, so the bump only
  // over-invalidates (safe direction).
  ++preview.state_epoch;
  // Perf item 4.4: the preview walk runs transactions on the authoritative
  // Impl; invalidate conservatively before the first write so clean
  // correctness never depends on the rollback proof (a read-only
  // short-circuit below is over-invalidated, which is the safe direction).
  preview.state_clean_ = false;
  // Joint perf Fix B1 (2026-10-16): singleton short-circuit (strict
  // equivalence). If the ready-heap front is a MEM/MEM_NOC instance whose
  // issue time is strictly after now and not later than the next completion
  // event, the walk provably crosses exactly one boundary and stops at that
  // memory issue: the result is identical to the caller's fallback
  // (action.ns, count=1). Verified 0 mis-fires by the detection-mode probe
  // (sc_fire_bad=0 over the full 1s-window TJE smoke, coverage 96.8%).
  if (!preview.heap.empty() && !preview.mem_outstanding) {
    const std::uint64_t sc_key = preview.heap.front();
    const std::uint32_t sc_s = static_cast<std::uint32_t>(sc_key >> 32);
    const DieStage& sc_st = (*preview.stages)[sc_s];
    if (sc_st.resource == StageResource::MEM ||
        sc_st.resource == StageResource::MEM_NOC) {
      const std::uint32_t sc_id =
          static_cast<std::uint32_t>(preview.offset[sc_s]) +
          (sc_key & 0xFFFFFFFFu);
      const std::uint64_t sc_ms = preview.memory_start(sc_id, sc_st);
      const std::uint64_t sc_next_event =
          preview.events.empty() ? UINT64_MAX : preview.events.front().first;
      if (sc_ms > preview.now && sc_ms <= sc_next_event) {
        // The public preview_calls counter documents entry attempts (its
        // test-enforced contract), so the short-circuit still counts one.
        ++preview.preview_calls;
        TimerStretchPreview sc_result;
        sc_result.external_delay_ns = sc_ms - preview.now;
        sc_result.timer_action_count = 1;
        sc_result.reaches_memory_request = true;
        return sc_result;
      }
    }
  }
  // E counts at the function entry, before any snapshot or journal work,
  // so preparation failures are attempts too (v9 §3.1).
  preview.preview_calls += 1;
  // S: the four scheduler-frontier entry sizes, sampled before the
  // transaction prepares (kept separate from the structural-zero copy F).
  preview.note_preview_entry_sizes();
  Impl::PreviewTxn txn(preview);
  preview.run(&txn);
  TimerStretchPreview result;
  if (preview.finished) {
    result.reaches_done = true;
    txn.note_exit(Impl::PreviewTxn::Exit::kDone);
    return result;
  }
  if (preview.mem_issued_new || preview.mem_outstanding) {
    result.reaches_memory_request = true;
    txn.note_exit(Impl::PreviewTxn::Exit::kMemoryRequest);
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
    preview.preview_walk_timers += 1;
    preview.now = next_tick;
    preview.run(&txn);

    if (preview.finished) {
      result.reaches_done = true;
      txn.note_exit(Impl::PreviewTxn::Exit::kDone);
      return result;
    }
    if (preview.mem_issued_new || preview.mem_outstanding) {
      result.reaches_memory_request = true;
      txn.note_exit(Impl::PreviewTxn::Exit::kMemoryRequest);
      return result;
    }
  }
}

void Execution::advance_time(std::uint64_t ns) {
  Impl& m = *impl_;
  // K2 memo epoch (v11 Phase 4): entry #2 of the audit at Impl::state_epoch.
  ++m.state_epoch;
  if (ns > UINT64_MAX - m.now) {
    throw DieModelError("advance_time: clock overflow");
  }
  // Perf item 4.4: invalidate after the validation throw, before the clock
  // mutation (run()'s entry invalidation makes this belt-and-braces).
  m.state_clean_ = false;
  m.now += ns;
  m.run();
}

void Execution::complete_memory(std::uint64_t stage_id) {
  Impl& m = *impl_;
  // K2 memo epoch (v11 Phase 4): entry #3 of the audit at Impl::state_epoch.
  ++m.state_epoch;
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
  // Perf item 4.4: invalidate after the validation throws above, before the
  // completion mutations rewrite run()'s inputs.
  m.state_clean_ = false;
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

// ---------------------------------------------------------------------------
// v12 Phase 4 (方案 §6.2): process-level structural negative cache for
// query_internal_timer_window. LRU discipline mirrors PlanCache (same
// single-driver, no-locking caliber). The config fingerprint is embedded in
// every entry key, so conclusions computed under one window-approximation
// configuration are structurally invisible to probes under another (exact
// vs approx can never share an entry).

K2TimerWindowNegativeCache::K2TimerWindowNegativeCache(
    std::size_t capacity, std::string config_fingerprint)
    : capacity_(capacity == 0 ? 1 : capacity),
      fingerprint_(std::move(config_fingerprint)) {}

void K2TimerWindowNegativeCache::set_config_fingerprint(
    std::string fingerprint) {
  fingerprint_ = std::move(fingerprint);
}

void K2TimerWindowNegativeCache::clear() {
  index_.clear();
  lru_.clear();
}

std::size_t K2TimerWindowNegativeCache::size() const { return lru_.size(); }

std::string K2TimerWindowNegativeCache::entry_key(
    const std::string& plan_cache_key) const {
  // '\x1f' (unit separator) cannot appear in either component: the
  // fingerprint is an ASCII token and plan_cache_key is 16 lowercase hex
  // digits, so the composite key is collision-free.
  std::string key = fingerprint_;
  key += '\x1f';
  key += plan_cache_key;
  return key;
}

std::optional<InternalTimerWindowInfo> K2TimerWindowNegativeCache::find(
    const std::string& plan_cache_key) const {
  auto it = index_.find(entry_key(plan_cache_key));
  if (it == index_.end()) {
    return std::nullopt;
  }
  lru_.splice(lru_.begin(), lru_, it->second);
  return it->second->second;
}

void K2TimerWindowNegativeCache::insert(
    const std::string& plan_cache_key, InternalTimerWindowInfo rejection) {
  const std::string key = entry_key(plan_cache_key);
  auto it = index_.find(key);
  if (it != index_.end()) {
    it->second->second = std::move(rejection);
    lru_.splice(lru_.begin(), lru_, it->second);
    return;
  }
  lru_.emplace_front(key, std::move(rejection));
  index_[key] = lru_.begin();
  while (lru_.size() > capacity_) {
    index_.erase(lru_.back().first);
    lru_.pop_back();
  }
}

K2TimerWindowNegativeCache& k2_timer_window_negative_cache() {
  static K2TimerWindowNegativeCache cache;
  return cache;
}

}  // namespace DieModel
