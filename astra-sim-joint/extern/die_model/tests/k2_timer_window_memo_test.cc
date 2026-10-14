/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

/*
k2_timer_window_memo_test.cc -- v11 Phase 4 (方案 §6.2) equivalence fixture
for the epoch-keyed query_internal_timer_window memo.

The memo is a pure cache: same (Execution state, max_count) must return the
exact same InternalTimerWindowInfo. The fixture proves that with a TWIN
oracle -- two Executions built from the same plan and driven through the
identical [query -> mutate -> query -> ...] script; the memoized side
queries each state repeatedly (miss then hits), the oracle side queries
each state exactly once (always a fresh computation, i.e. the memo-"off"
behavior). Field-for-field equality of the two answers at every step is
the equivalence proof; the memo hit/miss counters (exposed through
ExecutionTestAccess) prove the memo actually engaged instead of the test
passing vacuously.

Covered:
  - an ACCEPTED window (pure sequential PE chain, the same shape as the
    internal_service_progress_test fixture) is repeated identically and the
    memo never swallows or alters an accept;
  - a REJECTED frontier (parallel pre-scheduled waves -> the dominant v10
    dm_k2 rejection shape) repeats identically;
  - every mutating public entry invalidates: next_action / advance_time /
    complete_memory / advance_internal_service_window;
  - preview_consecutive_internal_timers restores state bit-exactly AND
    still invalidates (miss + identical fresh answer);
  - the full drive loop to Done: interleaved query/mutate twin equality;
  - distinct max_count slots memoize independently;
  - caller mutation of a returned token cannot corrupt the stored entry.
*/

#include "DieModel.hh"

#include <cstdint>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "test_helpers.hh"

namespace DieModel {
struct ExecutionTestAccess {
  static std::uint64_t memo_hits(const Execution& execution) {
    return execution.timer_window_memo_hits_for_test();
  }
  static std::uint64_t memo_misses(const Execution& execution) {
    return execution.timer_window_memo_misses_for_test();
  }
};
}  // namespace DieModel

namespace {

int failures = 0;

void expect(bool condition, const std::string& message) {
  if (!condition) {
    std::cerr << "[k2_timer_window_memo_test] FAIL: " << message << '\n';
    ++failures;
  }
}

// Field-for-field equality of the complete query result (the memo stores
// and returns every output field).
bool same_token(const DieModel::InternalServiceWindowToken& a,
                const DieModel::InternalServiceWindowToken& b) {
  return a.valid == b.valid && a.stage_id == b.stage_id &&
         a.first_wave_index == b.first_wave_index &&
         a.service_already_applied == b.service_already_applied &&
         a.completed_slices == b.completed_slices &&
         a.first_action.template_key == b.first_action.template_key &&
         a.first_action.flat_instance_id == b.first_action.flat_instance_id &&
         a.first_action.frontier_generation ==
             b.first_action.frontier_generation &&
         a.first_action.valid == b.first_action.valid &&
         a.original_service_ticks == b.original_service_ticks;
}

bool same_slice(const DieModel::InternalTemplateSliceEntry& a,
                const DieModel::InternalTemplateSliceEntry& b) {
  return a.stage_id == b.stage_id && a.wave_index == b.wave_index &&
         a.resource == b.resource && a.original_work == b.original_work &&
         a.original_service_ns == b.original_service_ns &&
         a.hardware_min_ns == b.hardware_min_ns;
}

bool same_window(const DieModel::InternalTimerWindowInfo& a,
                 const DieModel::InternalTimerWindowInfo& b) {
  if (a.supported != b.supported || a.reason != b.reason ||
      a.slices.size() != b.slices.size() || !same_token(a.token, b.token)) {
    return false;
  }
  for (std::size_t i = 0; i < a.slices.size(); ++i) {
    if (!same_slice(a.slices[i], b.slices[i])) {
      return false;
    }
  }
  return true;
}

std::shared_ptr<const DieModel::DiePlan> sequential_pe_then_mem_plan(
    const std::uint64_t repeat) {
  const auto hw = die_test::hw_default();
  DieModel::DieStage pe;
  pe.label = "k2-window-pe";
  pe.resource = DieModel::StageResource::PE;
  pe.wave_repeat = repeat;
  pe.deps.emplace_back(0, 1);  // wave i follows wave i-1
  const double rate = hw.core_row.pe_gflops_per_core * 1e9 * hw.eta_pe;
  pe.pe_core_flop = 1000.0 * rate / 1e9 * (1.0 - 1e-12);

  DieModel::DieStage mem;
  mem.label = "k2-window-mem";
  mem.resource = DieModel::StageResource::MEM;
  mem.mem.actual_bytes = 64.0;
  mem.mem.service_units = 64.0;
  mem.deps.emplace_back(0, DieModel::DieStageDep::kAllDep);

  auto data = std::make_shared<DieModel::DiePlanData>();
  data->spec = std::make_shared<const DieModel::DieOpSpec>(
      die_test::spec_of(DieModel::OpKind::LocalRegion,
                        DieModel::Phase::Prefill, 4, 4, 0, 0));
  data->hardware = hw;
  data->stages = {pe, mem};
  return std::make_shared<const DieModel::DiePlan>(std::move(data));
}

std::shared_ptr<const DieModel::DiePlan> repeated_pe_plan() {
  const auto hw = die_test::hw_default();
  DieModel::DieStage pe;
  pe.label = "k2-parallel-pe";
  pe.resource = DieModel::StageResource::PE;
  pe.wave_repeat = 2;
  const double rate = hw.core_row.pe_gflops_per_core * 1e9 * hw.eta_pe;
  pe.pe_core_flop = 1000.0 * rate / 1e9 * (1.0 - 1e-12);
  auto data = std::make_shared<DieModel::DiePlanData>();
  data->spec = std::make_shared<const DieModel::DieOpSpec>(
      die_test::spec_of(DieModel::OpKind::LocalRegion,
                        DieModel::Phase::Prefill, 4, 4, 0, 0));
  data->hardware = hw;
  data->stages.push_back(pe);
  return std::make_shared<const DieModel::DiePlan>(std::move(data));
}

void test_accept_window_repeats_and_twin_equivalence() {
  using namespace DieModel;
  Execution memoed(sequential_pe_then_mem_plan(4));
  Execution fresh(sequential_pe_then_mem_plan(4));

  const ExecutionAction first_memoed = memoed.next_action();
  const ExecutionAction first_fresh = fresh.next_action();
  expect(first_memoed.kind == ExecutionAction::Kind::Timer &&
             first_memoed.ns == 1000 && first_fresh.ns == first_memoed.ns,
         "accept fixture: both twins begin with the same 1000ns timer");

  const auto q0 = memoed.query_internal_timer_window(4);
  const auto oracle = fresh.query_internal_timer_window(4);
  expect(q0.supported && q0.slices.size() == 4 &&
             q0.token.original_service_ticks ==
                 std::vector<std::uint64_t>({1000, 1000, 1000, 1000}),
         "accept fixture: the window proves four sequential PE slices");
  expect(same_window(q0, oracle),
         "accept fixture: memoized answer equals the fresh-compute twin");

  // Frozen-state repetition: every later query must be a pure hit returning
  // the identical full value.
  const std::uint64_t hits_before =
      ExecutionTestAccess::memo_hits(memoed);
  for (int repeat = 0; repeat < 5; ++repeat) {
    const auto again = memoed.query_internal_timer_window(4);
    expect(same_window(again, q0),
           "accept fixture: repeated frozen query is field-identical");
  }
  expect(ExecutionTestAccess::memo_hits(memoed) == hits_before + 5,
         "accept fixture: five repeated queries were all memo hits");

  // The stored entry is a copy: mutating a returned token must not poison
  // later answers.
  auto mutated = memoed.query_internal_timer_window(4);
  mutated.token.original_service_ticks.assign({1});
  mutated.reason = "caller-side-corruption";
  expect(same_window(memoed.query_internal_timer_window(4), q0),
         "accept fixture: caller token mutation cannot corrupt the memo");

  // advance_time invalidates: miss again, and the new answer still matches
  // an identically-driven fresh twin. A 250ns advance leaves wave 0
  // partially served, so the fresh answer is the deterministic rejection.
  memoed.advance_time(250);
  fresh.advance_time(250);
  const std::uint64_t misses_before =
      ExecutionTestAccess::memo_misses(memoed);
  const auto q1 = memoed.query_internal_timer_window(4);
  const auto oracle1 = fresh.query_internal_timer_window(4);
  expect(ExecutionTestAccess::memo_misses(memoed) == misses_before + 1,
         "accept fixture: advance_time invalidated the slot (one more miss)");
  expect(same_window(q1, oracle1),
         "accept fixture: post-advance answer equals the fresh twin");
  expect(!q1.supported &&
             q1.reason == "current_slice_already_partially_served",
         "accept fixture: the partially-served frontier rejects identically");
}

void test_reject_frontier_repeats() {
  using namespace DieModel;
  Execution memoed(repeated_pe_plan());
  const auto first = memoed.next_action();
  expect(first.kind == ExecutionAction::Kind::Timer,
         "reject fixture: parallel PE plan begins with a timer");
  const auto q0 = memoed.query_internal_timer_window(4);
  expect(!q0.supported && q0.slices.empty() &&
             q0.reason == "other_scheduled_internal_boundary",
         "reject fixture: parallel pre-scheduled waves are rejected");
  const std::uint64_t hits_before =
      ExecutionTestAccess::memo_hits(memoed);
  for (int repeat = 0; repeat < 4; ++repeat) {
    expect(same_window(memoed.query_internal_timer_window(4), q0),
           "reject fixture: repeated rejection is field-identical");
  }
  expect(ExecutionTestAccess::memo_hits(memoed) == hits_before + 4,
         "reject fixture: repeated rejections were all memo hits");
}

void test_per_max_count_slots_are_independent() {
  using namespace DieModel;
  Execution memoed(sequential_pe_then_mem_plan(6));
  (void)memoed.next_action();
  const auto q4 = memoed.query_internal_timer_window(4);
  const auto q2 = memoed.query_internal_timer_window(2);
  expect(q4.supported && q4.slices.size() == 4 && q2.supported &&
             q2.slices.size() == 2,
         "slots fixture: max_count 4 and 2 prove different window sizes");
  const std::uint64_t hits4_before =
      ExecutionTestAccess::memo_hits(memoed);
  expect(same_window(memoed.query_internal_timer_window(2), q2),
         "slots fixture: the 2-slot repeats exactly");
  expect(ExecutionTestAccess::memo_hits(memoed) == hits4_before + 1,
         "slots fixture: the 2-slot repeat hit its own slot");
  expect(same_window(memoed.query_internal_timer_window(4), q4),
         "slots fixture: the 4-slot is still intact beside the 2-slot");
  expect(ExecutionTestAccess::memo_hits(memoed) == hits4_before + 2,
         "slots fixture: the 4-slot repeat hit independently");
}

void test_preview_restores_state_and_still_invalidates() {
  using namespace DieModel;
  Execution memoed(sequential_pe_then_mem_plan(4));
  (void)memoed.next_action();
  const auto q0 = memoed.query_internal_timer_window(4);
  expect(q0.supported, "preview fixture: window accepted before preview");

  const TimerStretchPreview preview =
      memoed.preview_consecutive_internal_timers();
  expect(preview.reaches_memory_request && !preview.reaches_done &&
             preview.timer_action_count >= 1,
         "preview fixture: the stretch preview walks the timer chain and "
         "stops at the memory request");

  // The preview rolled the scheduler state back bit-exactly, so the answer
  // must be identical -- but the epoch bump forces a recompute (miss), and
  // the fresh twin proves the recomputed value.
  Execution fresh(sequential_pe_then_mem_plan(4));
  (void)fresh.next_action();
  (void)fresh.preview_consecutive_internal_timers();
  const std::uint64_t misses_before =
      ExecutionTestAccess::memo_misses(memoed);
  const auto q1 = memoed.query_internal_timer_window(4);
  const auto oracle = fresh.query_internal_timer_window(4);
  expect(ExecutionTestAccess::memo_misses(memoed) == misses_before + 1,
         "preview fixture: preview entry invalidated the memo (one miss)");
  expect(same_window(q1, q0),
         "preview fixture: restored state reproduces the identical window");
  expect(same_window(q1, oracle),
         "preview fixture: recomputed answer equals the fresh twin");
}

void test_interleaved_query_mutate_drive_loop() {
  using namespace DieModel;
  // The core K2 equivalence loop: both twins run the identical drive script
  // (Timer advances, service-window application, the memory request, and
  // Done); at every step the memoized side queries twice (the second must
  // be a hit and identical to the first) while the oracle side queries
  // once (always a fresh computation).
  auto run_step = [](Execution& memoed, Execution& fresh,
                     const std::uint32_t max_count) {
    const auto a1 = memoed.query_internal_timer_window(max_count);
    const auto a2 = memoed.query_internal_timer_window(max_count);
    const auto b = fresh.query_internal_timer_window(max_count);
    expect(same_window(a1, a2), "drive loop: immediate re-query is a hit");
    expect(same_window(a1, b),
           "drive loop: memoized answer equals the fresh oracle");
    return b.supported;
  };

  Execution memoed(sequential_pe_then_mem_plan(4));
  Execution fresh(sequential_pe_then_mem_plan(4));
  for (Execution* const execution : {&memoed, &fresh}) {
    const ExecutionAction first = execution->next_action();
    expect(first.kind == ExecutionAction::Kind::Timer && first.ns == 1000,
           "drive loop: sequential plan begins with the 1000ns timer");
  }
  expect(run_step(memoed, fresh, 4), "drive loop: initial window accepted");

  // Consume the whole window through the service-window entry. The CURSOR
  // tokens are taken once from the accepted answers and advanced in step
  // (a re-query mid-wave returns a fresh-but-unusable rejected token); the
  // re-queries after every apply exercise the rejection path on both twins
  // at every intermediate state.
  const auto window_memoed = memoed.query_internal_timer_window(4);
  const auto window_fresh = fresh.query_internal_timer_window(4);
  expect(window_memoed.supported && window_fresh.supported &&
             same_window(window_memoed, window_fresh),
         "drive loop: both twins hold the identical accepted window token");
  auto token_memoed = window_memoed.token;
  auto token_fresh = window_fresh.token;
  for (std::uint64_t applied = 0; applied < 4000; applied += 500) {
    expect(same_token(token_memoed, token_fresh),
           "drive loop: window cursor tokens stay in lockstep");
    memoed.advance_internal_service_window(token_memoed, 500);
    fresh.advance_internal_service_window(token_fresh, 500);
    run_step(memoed, fresh, 4);
  }
  const auto final_window_memoed = memoed.query_internal_timer_window(4);
  const auto final_window_fresh = fresh.query_internal_timer_window(4);
  expect(!final_window_memoed.supported &&
             final_window_memoed.reason ==
                 "external-memory-request-outstanding" &&
             same_window(final_window_memoed, final_window_fresh),
         "drive loop: the issued memory request terminates the window "
         "identically on both twins");

  // Memory phase: identical requests on both twins, queried mid-wait.
  const auto mem_memoed = memoed.next_action();
  const auto mem_fresh = fresh.next_action();
  expect(mem_memoed.kind == ExecutionAction::Kind::MemoryRequest &&
             mem_memoed.stage_id == mem_fresh.stage_id,
         "drive loop: both twins report the same memory request");
  run_step(memoed, fresh, 4);
  memoed.advance_external_wait(10, mem_memoed.stage_id);
  fresh.advance_external_wait(10, mem_fresh.stage_id);
  run_step(memoed, fresh, 4);
  memoed.complete_memory(mem_memoed.stage_id);
  fresh.complete_memory(mem_fresh.stage_id);
  run_step(memoed, fresh, 4);

  const auto done_memoed = memoed.next_action();
  const auto done_fresh = fresh.next_action();
  expect(done_memoed.kind == ExecutionAction::Kind::Done &&
             done_fresh.kind == ExecutionAction::Kind::Done &&
             done_memoed.ns == done_fresh.ns,
         "drive loop: both twins finish with the identical Done action");
  expect(memoed.finished() && fresh.finished() &&
             memoed.elapsed_ns() == fresh.elapsed_ns(),
         "drive loop: twins end finished with equal makespans");
  run_step(memoed, fresh, 4);
  expect(ExecutionTestAccess::memo_hits(memoed) > 0 &&
             ExecutionTestAccess::memo_misses(memoed) > 0,
         "drive loop: the memo demonstrably hit and missed during the run");
}

}  // namespace

int main() {
  test_accept_window_repeats_and_twin_equivalence();
  test_reject_frontier_repeats();
  test_per_max_count_slots_are_independent();
  test_preview_restores_state_and_still_invalidates();
  test_interleaved_query_mutate_drive_loop();
  if (failures != 0) {
    std::cerr << "[k2_timer_window_memo_test] " << failures
              << " failure(s)\n";
    return 1;
  }
  std::cout << "[k2_timer_window_memo_test] PASS\n";
  return 0;
}
