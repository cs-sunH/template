#include "DieModel.hh"

#include <cmath>
#include <cstring>
#include <iostream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "test_helpers.hh"

namespace DieModel {
struct ExecutionTestAccess {
  static std::uint64_t template_key_computations(const Execution& execution) {
    return execution.template_key_computations_for_test();
  }
};
}  // namespace DieModel

namespace {

int failures = 0;

void expect(bool condition, const std::string& message) {
  if (!condition) {
    std::cerr << "[internal_service_progress_test] FAIL: " << message << '\n';
    ++failures;
  }
}

template <typename Fn>
void expect_throw(Fn&& fn, const std::string& message) {
  try {
    fn();
    expect(false, message + " (did not throw)");
  } catch (const std::exception&) {
  }
}

std::shared_ptr<const DieModel::DiePlan> repeated_pe_plan() {
  const auto hw = die_test::hw_default();
  DieModel::DieStage pe;
  pe.label = "ignored-by-eligibility";
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

std::shared_ptr<const DieModel::DiePlan> repeated_pe_then_mem_plan() {
  const auto hw = die_test::hw_default();
  DieModel::DieStage pe;
  pe.label = "preview-pe";
  pe.resource = DieModel::StageResource::PE;
  pe.wave_repeat = 2;
  const double rate = hw.core_row.pe_gflops_per_core * 1e9 * hw.eta_pe;
  pe.pe_core_flop = 1000.0 * rate / 1e9 * (1.0 - 1e-12);

  DieModel::DieStage mem;
  mem.label = "preview-mem";
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

std::shared_ptr<const DieModel::DiePlan> sequential_pe_then_mem_plan(
    const std::uint64_t repeat) {
  const auto hw = die_test::hw_default();
  DieModel::DieStage pe;
  pe.label = "window-pe";
  pe.resource = DieModel::StageResource::PE;
  pe.wave_repeat = repeat;
  pe.deps.emplace_back(0, 1);  // wave i follows wave i-1
  const double rate = hw.core_row.pe_gflops_per_core * 1e9 * hw.eta_pe;
  pe.pe_core_flop = 1000.0 * rate / 1e9 * (1.0 - 1e-12);

  DieModel::DieStage mem;
  mem.label = "window-mem";
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

std::shared_ptr<const DieModel::DiePlan> repeated_timer_waves_then_mem_plan(
    const std::uint64_t repeat) {
  const auto hw = die_test::hw_default();
  DieModel::DieStage timer_parent;
  timer_parent.label = "same-end-timer-parent";
  timer_parent.resource = DieModel::StageResource::TIMER;
  timer_parent.wave_repeat = repeat;
  // Exactly representable binary seconds; multiplying by 1e9 yields an
  // integral nanosecond count without exercising ceil-rounding edge cases.
  timer_parent.timer_seconds = 1.0 / 512.0;

  DieModel::DieStage timer_child;
  timer_child.label = "same-end-timer-child";
  timer_child.resource = DieModel::StageResource::TIMER;
  timer_child.wave_repeat = repeat;
  timer_child.timer_seconds = 1.0 / 512.0;
  timer_child.deps.emplace_back(0, 0);  // child j follows parent j

  DieModel::DieStage mem;
  mem.label = "timer-wave-memory-boundary";
  mem.resource = DieModel::StageResource::MEM;
  mem.mem.actual_bytes = 64.125;
  mem.mem.service_units = 96.5;
  mem.deps.emplace_back(1, DieModel::DieStageDep::kAllDep);

  auto data = std::make_shared<DieModel::DiePlanData>();
  data->spec = std::make_shared<const DieModel::DieOpSpec>(
      die_test::spec_of(DieModel::OpKind::LocalRegion,
                        DieModel::Phase::Prefill, 4, 4, 0, 0));
  data->hardware = hw;
  data->stages = {timer_parent, timer_child, mem};
  return std::make_shared<const DieModel::DiePlan>(std::move(data));
}

std::shared_ptr<const DieModel::DiePlan> root_timer_mem_suffix_plan(
    const std::uint64_t suffix_repeat) {
  const auto hw = die_test::hw_default();
  DieModel::DieStage root_timer;
  root_timer.label = "preview-root-timer";
  root_timer.resource = DieModel::StageResource::TIMER;
  root_timer.timer_seconds = 1.0 / 512.0;

  DieModel::DieStage mem;
  mem.label = "preview-middle-memory";
  mem.resource = DieModel::StageResource::MEM;
  mem.mem.actual_bytes = 64.125;
  mem.mem.service_units = 96.5;
  mem.deps.emplace_back(0, DieModel::DieStageDep::kAllDep);

  DieModel::DieStage suffix;
  suffix.label = "unvisited-preview-suffix";
  suffix.resource = DieModel::StageResource::TIMER;
  suffix.wave_repeat = suffix_repeat;
  suffix.timer_seconds = 1.0 / 512.0;
  suffix.deps.emplace_back(1, DieModel::DieStageDep::kAllDep);

  auto data = std::make_shared<DieModel::DiePlanData>();
  data->spec = std::make_shared<const DieModel::DieOpSpec>(
      die_test::spec_of(DieModel::OpKind::LocalRegion,
                        DieModel::Phase::Prefill, 4, 4, 0, 0));
  data->hardware = hw;
  data->stages = {root_timer, mem, suffix};
  return std::make_shared<const DieModel::DiePlan>(std::move(data));
}

std::shared_ptr<const DieModel::DiePlan> overflowing_timer_preview_plan() {
  const auto hw = die_test::hw_default();
  DieModel::DieStage first;
  first.label = "overflow-preview-first-timer";
  first.resource = DieModel::StageResource::TIMER;
  first.timer_seconds = 9.3e9;

  DieModel::DieStage second;
  second.label = "overflow-preview-second-timer";
  second.resource = DieModel::StageResource::TIMER;
  second.timer_seconds = 9.3e9;
  second.deps.emplace_back(0, 0);

  auto data = std::make_shared<DieModel::DiePlanData>();
  data->spec = std::make_shared<const DieModel::DieOpSpec>(
      die_test::spec_of(DieModel::OpKind::LocalRegion,
                        DieModel::Phase::Prefill, 4, 4, 0, 0));
  data->hardware = hw;
  data->stages = {first, second};
  return std::make_shared<const DieModel::DiePlan>(std::move(data));
}

std::uint64_t double_bits(const double value) {
  static_assert(sizeof(double) == sizeof(std::uint64_t),
                "the DIE test requires binary64 doubles");
  std::uint64_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

bool same_internal_slice(const DieModel::InternalSliceInfo& left,
                         const DieModel::InternalSliceInfo& right) {
  if (left.action_token.template_key != right.action_token.template_key ||
      left.action_token.flat_instance_id !=
          right.action_token.flat_instance_id ||
      left.action_token.frontier_generation !=
          right.action_token.frontier_generation ||
      left.action_token.valid != right.action_token.valid ||
      left.template_key != right.template_key ||
      left.classification != right.classification ||
      left.now_ns != right.now_ns ||
      left.next_boundary_ns != right.next_boundary_ns ||
      left.has_next_boundary != right.has_next_boundary ||
      left.partial_service_supported != right.partial_service_supported ||
      left.reason != right.reason ||
      left.active_stages.size() != right.active_stages.size()) {
    return false;
  }
  for (std::size_t i = 0; i < left.active_stages.size(); ++i) {
    const auto& a = left.active_stages[i];
    const auto& b = right.active_stages[i];
    if (a.stage_id != b.stage_id || a.wave_index != b.wave_index ||
        a.resource != b.resource || a.start_ns != b.start_ns ||
        a.end_ns != b.end_ns ||
        a.original_service_ns != b.original_service_ns ||
        a.remaining_service_ns != b.remaining_service_ns ||
        double_bits(a.original_work) != double_bits(b.original_work) ||
        double_bits(a.remaining_work) != double_bits(b.remaining_work) ||
        double_bits(a.memory_actual_bytes) !=
            double_bits(b.memory_actual_bytes) ||
        double_bits(a.memory_service_units) !=
            double_bits(b.memory_service_units) ||
        a.memory_pending != b.memory_pending ||
        a.hardware_min_ns != b.hardware_min_ns) {
      return false;
    }
  }
  return true;
}

bool same_progress(const DieModel::ExecutionProgress& left,
                   const DieModel::ExecutionProgress& right) {
  if (left.now_ns != right.now_ns || left.elapsed_ns != right.elapsed_ns ||
      left.finished != right.finished ||
      left.memory_pending != right.memory_pending ||
      left.memory_stage_id != right.memory_stage_id ||
      left.has_memory_instance_id != right.has_memory_instance_id ||
      left.memory_instance_id != right.memory_instance_id ||
      double_bits(left.pending_memory_actual_bytes) !=
          double_bits(right.pending_memory_actual_bytes) ||
      double_bits(left.pending_memory_service_units) !=
          double_bits(right.pending_memory_service_units) ||
      left.external_memory_wait_elapsed_ns !=
          right.external_memory_wait_elapsed_ns ||
      left.active_id_compaction_calls !=
          right.active_id_compaction_calls ||
      left.active_id_entries_examined != right.active_id_entries_examined ||
      left.active_id_entries_moved != right.active_id_entries_moved ||
      left.active_id_entries_removed != right.active_id_entries_removed ||
      !same_internal_slice(left.frontier, right.frontier) ||
      left.stages.size() != right.stages.size()) {
    return false;
  }
  for (std::size_t i = 0; i < left.stages.size(); ++i) {
    const auto& a = left.stages[i];
    const auto& b = right.stages[i];
    if (a.stage_id != b.stage_id || a.resource != b.resource ||
        a.completed != b.completed || a.total != b.total ||
        double_bits(a.remaining_work) != double_bits(b.remaining_work) ||
        double_bits(a.remaining_noc_bytes) !=
            double_bits(b.remaining_noc_bytes) ||
        double_bits(a.remaining_actual_bytes) !=
            double_bits(b.remaining_actual_bytes) ||
        double_bits(a.remaining_service_units) !=
            double_bits(b.remaining_service_units) ||
        a.remaining_internal_service_ns !=
            b.remaining_internal_service_ns) {
      return false;
    }
  }
  return true;
}

void test_partial_progress_and_stale_tokens() {
  using namespace DieModel;
  Execution execution(repeated_pe_plan());
  const InternalSliceInfo start = execution.describe_internal_slice();
  const auto initial_progress = execution.export_progress();
  expect(start.classification == InternalSliceClass::PurePe &&
             start.action_token.valid &&
             start.action_token.flat_instance_id == 0 &&
             start.active_stages[0].remaining_service_ns == 1000,
         "initial typed slice is one 1000ns PE wave");
  expect(initial_progress.stages.size() == 1 &&
             initial_progress.stages[0].completed == 0 &&
             initial_progress.stages[0].remaining_internal_service_ns == 2000 &&
             initial_progress.stages[0].remaining_work ==
                 2.0 * start.active_stages[0].original_work,
         "compact export reports both repeated PE waves without scanning them");

  execution.advance_internal_service(start.action_token, 333);
  const InternalSliceInfo partial = execution.describe_internal_slice();
  expect(partial.action_token.template_key == start.action_token.template_key &&
             partial.action_token.flat_instance_id ==
                 start.action_token.flat_instance_id &&
             partial.action_token.frontier_generation ==
                 start.action_token.frontier_generation &&
             partial.now_ns == 333 &&
             partial.active_stages[0].remaining_service_ns == 667,
         "partial advancement retains token and exact unserved timer prefix");
  expect(std::abs(partial.active_stages[0].remaining_work -
                  start.active_stages[0].original_work * 0.667) < 1e-9,
         "remaining work follows only the applied original-service fraction");
  const auto partial_progress = execution.export_progress();
  expect(partial_progress.stages[0].completed == 0 &&
             partial_progress.stages[0].remaining_internal_service_ns == 1667 &&
             std::abs(partial_progress.stages[0].remaining_work -
                      start.active_stages[0].original_work * 1.667) < 1e-9,
         "progress export adjusts totals using only its active frontier");

  execution.advance_internal_service(partial.action_token, 667);
  const InternalSliceInfo next = execution.describe_internal_slice();
  expect(next.action_token.valid &&
             next.action_token.flat_instance_id !=
                 start.action_token.flat_instance_id &&
             next.action_token.frontier_generation !=
                 start.action_token.frontier_generation &&
             next.now_ns == 1000 &&
             next.active_stages[0].wave_index == 1,
         "completed stage advances to the next immutable wave token");
  expect_throw([&] {
    execution.advance_internal_service(start.action_token, 1);
  }, "completed action token is rejected as stale");
  execution.advance_internal_service(next.action_token, 1000);
  const auto progress = execution.export_progress();
  expect(progress.finished && progress.stages.size() == 1 &&
             progress.stages[0].completed == 2 && progress.stages[0].total == 2,
         "compact export reports both completed waves and terminal state");
}

void test_boundary_and_overflow_rejection() {
  using namespace DieModel;
  Execution execution(repeated_pe_plan());
  const InternalSliceInfo start = execution.describe_internal_slice();
  expect_throw([&] {
    execution.advance_internal_service(start.action_token, 1001);
  }, "partial-service API rejects crossing a stage boundary");
  expect(execution.export_progress().now_ns == 0,
         "rejected over-boundary service leaves the cursor unchanged");
}

void test_timer_stretch_preview_is_read_only_and_stops_at_memory() {
  using namespace DieModel;
  Execution execution(repeated_pe_then_mem_plan());
  const auto action = execution.next_action();
  expect(action.kind == ExecutionAction::Kind::Timer && action.ns == 1000,
         "first pure PE timer is reported normally");

  const TimerStretchPreview preview =
      execution.preview_consecutive_internal_timers();
  const auto before = execution.export_progress();
  expect(preview.external_delay_ns == 2000 &&
             preview.timer_action_count == 2 &&
             preview.reaches_memory_request && !preview.reaches_done,
         "read-only preview merges contiguous timers up to, but not past, memory issue");
  expect(before.now_ns == 0 && before.stages[0].completed == 0 &&
             !before.memory_pending,
         "preview leaves authoritative progress and memory ownership untouched");

  execution.advance_time(preview.external_delay_ns);
  const auto next = execution.next_action();
  expect(next.kind == ExecutionAction::Kind::MemoryRequest &&
             next.stage_id == 1 && execution.elapsed_ns() == 2000,
         "the real cursor reaches the same memory boundary when the merged timer expires");
}

void test_preview_preserves_full_state_with_unvisited_suffix() {
  using namespace DieModel;
  for (const std::uint64_t suffix_repeat : {8ULL, 32ULL, 128ULL}) {
    const auto plan = root_timer_mem_suffix_plan(suffix_repeat);
    Execution previewed(plan);
    Execution reference(plan);

    const ExecutionAction first = previewed.next_action();
    const ExecutionAction reference_first = reference.next_action();
    expect(first.kind == ExecutionAction::Kind::Timer &&
               first.ns == 1953125 &&
               reference_first.kind == ExecutionAction::Kind::Timer &&
               reference_first.ns == first.ns,
           "suffix fixture begins with the exact binary Timer interval");

    const auto before = previewed.export_progress();
    expect(same_progress(before, reference.export_progress()),
           "preview and non-preview executions start with identical full progress");
    const std::uint64_t key_computations_before =
        ExecutionTestAccess::template_key_computations(previewed);
    const TimerStretchPreview preview =
        previewed.preview_consecutive_internal_timers();
    const auto after = previewed.export_progress();
    expect(preview.external_delay_ns == first.ns &&
               preview.timer_action_count == 1 &&
               preview.reaches_memory_request && !preview.reaches_done,
           "preview visits only the root Timer and stops at the middle Memory stage");
    expect(same_progress(before, after),
           "preview leaves every exported clock, token, generation, frontier, and stage field unchanged");
    expect(ExecutionTestAccess::template_key_computations(previewed) ==
               key_computations_before,
           "preview does not mutate the authoritative lazy template-key cache");
    for (std::size_t repeat = 0; repeat < 3; ++repeat) {
      const auto again = previewed.preview_consecutive_internal_timers();
      expect(again.external_delay_ns == preview.external_delay_ns &&
                 again.timer_action_count == preview.timer_action_count &&
                 again.reaches_memory_request == preview.reaches_memory_request &&
                 again.reaches_done == preview.reaches_done &&
                 same_progress(before, previewed.export_progress()) &&
                 ExecutionTestAccess::template_key_computations(previewed) ==
                     key_computations_before,
             "repeated preview restores the same cursor and stable token every time");
    }

    // Also preserve the genuinely lazy state when no progress export has
    // asked for a template key yet.
    if (suffix_repeat == 8) {
      Execution lazy_preview(plan);
      static_cast<void>(lazy_preview.next_action());
      expect(ExecutionTestAccess::template_key_computations(lazy_preview) == 0,
             "next_action alone leaves a fresh execution's template key lazy");
      static_cast<void>(lazy_preview.preview_consecutive_internal_timers());
      expect(ExecutionTestAccess::template_key_computations(lazy_preview) == 0,
             "read-only preview leaves the fresh execution's key cache lazy");
    }

    previewed.advance_time(preview.external_delay_ns);
    reference.advance_time(reference_first.ns);
    const ExecutionAction memory = previewed.next_action();
    const ExecutionAction reference_memory = reference.next_action();
    const bool same_memory =
        memory.kind == ExecutionAction::Kind::MemoryRequest &&
        reference_memory.kind == ExecutionAction::Kind::MemoryRequest &&
        memory.ns == reference_memory.ns &&
        memory.stage_id == reference_memory.stage_id &&
        double_bits(memory.actual_bytes) ==
            double_bits(reference_memory.actual_bytes) &&
        double_bits(memory.service_units) ==
            double_bits(reference_memory.service_units);
    expect(same_memory && memory.stage_id == 1 &&
               double_bits(memory.actual_bytes) == double_bits(64.125) &&
               double_bits(memory.service_units) == double_bits(96.5),
           "preview and ordinary advancement issue the same exact middle MemoryRequest");
    expect(same_progress(previewed.export_progress(),
                         reference.export_progress()),
           "full execution progress agrees at the MemoryRequest boundary");

    constexpr std::uint64_t kExternalMemoryWaitNs = 37;
    previewed.advance_external_wait(kExternalMemoryWaitNs, memory.stage_id);
    reference.advance_external_wait(kExternalMemoryWaitNs,
                                    reference_memory.stage_id);
    expect(previewed.export_progress().external_memory_wait_elapsed_ns ==
               kExternalMemoryWaitNs,
           "external memory wait uses the real driver wait accounting");
    previewed.complete_memory(memory.stage_id);
    reference.complete_memory(reference_memory.stage_id);

    const auto advance_suffix_to_done = [](
        Execution& execution,
        std::vector<std::pair<ExecutionAction::Kind, std::uint64_t>>& trace) {
      for (std::size_t guard = 0; guard < 8; ++guard) {
        const ExecutionAction action = execution.next_action();
        trace.emplace_back(action.kind, action.ns);
        if (action.kind == ExecutionAction::Kind::Done) return true;
        if (action.kind != ExecutionAction::Kind::Timer) return false;
        execution.advance_time(action.ns);
      }
      return false;
    };
    std::vector<std::pair<ExecutionAction::Kind, std::uint64_t>>
        preview_trace;
    std::vector<std::pair<ExecutionAction::Kind, std::uint64_t>>
        reference_trace;
    const bool preview_done =
        advance_suffix_to_done(previewed, preview_trace);
    const bool reference_done =
        advance_suffix_to_done(reference, reference_trace);
    const auto preview_final = previewed.export_progress();
    const auto reference_final = reference.export_progress();
    expect(preview_done && reference_done &&
               preview_trace == reference_trace &&
               same_progress(preview_final, reference_final),
           "previewed and ordinary executions continue through the suffix to identical Done progress");
    expect(preview_final.finished && preview_final.stages.size() == 3 &&
               preview_final.stages[0].completed == 1 &&
               preview_final.stages[1].completed == 1 &&
               preview_final.stages[2].completed == suffix_repeat,
           "the untouched suffix completes all requested waves after Memory returns");

    std::cout << "[internal_service_progress_test] preview_suffix="
              << suffix_repeat << " timer_actions="
              << preview.timer_action_count << " final_ns="
              << preview_final.elapsed_ns << '\n';
  }
}

void test_preview_overflow_exception_is_read_only() {
  using namespace DieModel;
  Execution execution(overflowing_timer_preview_plan());
  const ExecutionAction first = execution.next_action();
  expect(first.kind == ExecutionAction::Kind::Timer && first.ns > 0,
         "overflow preview fixture begins with one representable Timer");
  const auto before = execution.export_progress();
  expect(before.stages.size() == 2 && before.stages[0].completed == 0 &&
             before.stages[1].completed == 0,
         "overflow preview begins before either Timer completes");

  bool overflow_rejected = false;
  try {
    static_cast<void>(execution.preview_consecutive_internal_timers());
  } catch (const DieModelError&) {
    overflow_rejected = true;
  }
  const auto after = execution.export_progress();
  expect(overflow_rejected,
         "preview rejects the second Timer whose end exceeds uint64");
  expect(same_progress(before, after),
         "overflow exception restores the complete authoritative progress state");
  expect(ExecutionTestAccess::template_key_computations(execution) == 1,
         "overflow rollback preserves the already-computed plan-key cache");
}

void test_bounded_window_query_and_sequential_service_cursor() {
  using namespace DieModel;
  Execution execution(sequential_pe_then_mem_plan(4));
  expect(ExecutionTestAccess::template_key_computations(execution) == 0,
         "execution construction leaves the plan-key cache lazy");
  expect(!execution.pending_memory_stage_id().has_value(),
         "no memory stage is pending before the first request");
  const auto first = execution.next_action();
  expect(first.kind == ExecutionAction::Kind::Timer && first.ns == 1000,
         "sequential PE plan begins with its original timer action");
  expect(ExecutionTestAccess::template_key_computations(execution) == 0,
         "reading the first timer does not eagerly calculate the plan key");

  InternalTimerWindowInfo window = execution.query_internal_timer_window(4);
  expect(window.supported && window.slices.size() == 4 &&
             window.token.valid &&
             window.token.original_service_ticks ==
                 std::vector<std::uint64_t>({1000, 1000, 1000, 1000}),
         "bounded query proves four repeated PE actions up to the memory boundary");
  expect(ExecutionTestAccess::template_key_computations(execution) == 1,
         "the first frontier query calculates the immutable plan key once");
  const auto first_frontier = execution.describe_internal_slice();
  const auto repeated_frontier = execution.describe_internal_slice();
  expect(first_frontier.template_key == repeated_frontier.template_key &&
             first_frontier.template_key ==
                 window.token.first_action.template_key &&
             ExecutionTestAccess::template_key_computations(execution) == 1,
         "frontier and action token reuse one cached plan key");
  const auto before = execution.export_progress();
  expect(before.now_ns == 0 && before.stages[0].completed == 0,
         "window lookahead leaves the authoritative execution untouched");

  InternalServiceWindowToken bad_key_token = window.token;
  bad_key_token.first_action.template_key += "x";
  const auto before_bad_token = execution.export_progress();
  expect_throw([&] {
    execution.advance_internal_service_window(bad_key_token, 1);
  }, "window token with a different cached plan key is rejected");
  const auto after_bad_token = execution.export_progress();
  expect(after_bad_token.now_ns == before_bad_token.now_ns &&
             after_bad_token.elapsed_ns == before_bad_token.elapsed_ns &&
             after_bad_token.frontier.action_token.frontier_generation ==
                 before_bad_token.frontier.action_token.frontier_generation &&
             after_bad_token.stages[0].completed ==
                 before_bad_token.stages[0].completed &&
             ExecutionTestAccess::template_key_computations(execution) == 1,
         "bad token rejection leaves service state and cached key unchanged");

  execution.advance_internal_service_window(window.token, 333);
  auto partial = execution.describe_internal_slice();
  expect(window.token.service_already_applied == 333 &&
             window.token.completed_slices == 0 &&
             partial.active_stages[0].wave_index == 0 &&
             partial.active_stages[0].remaining_service_ns == 667,
         "stable window token applies a partial original prefix without advancing waves");

  execution.advance_internal_service_window(window.token, 1000);
  partial = execution.describe_internal_slice();
  expect(window.token.service_already_applied == 1333 &&
             window.token.completed_slices == 1 &&
             partial.active_stages[0].wave_index == 1 &&
             partial.active_stages[0].remaining_service_ns == 667,
         "same token crosses one completed wave in order and preserves the next remainder");

  execution.advance_internal_service_window(window.token, 2667);
  const auto memory = execution.next_action();
  const auto progress = execution.export_progress();
  expect(memory.kind == ExecutionAction::Kind::MemoryRequest &&
             memory.stage_id == 1 && progress.now_ns == 4000 &&
             progress.stages[0].completed == 4 && progress.memory_pending,
         "window cursor stops exactly at the real memory request after all original PE service");
  expect(execution.pending_memory_stage_id() ==
             std::optional<std::uint32_t>{1},
         "constant-time query reports the real outstanding memory stage");
  expect_throw([&] {
    execution.advance_internal_service_window(window.token, 1);
  }, "consumed window token rejects service beyond its bounded end");
  execution.complete_memory(memory.stage_id);
  expect(!execution.pending_memory_stage_id().has_value(),
         "completing the request clears the pending-stage query");
}

void test_window_query_stops_at_mixed_future_action() {
  using namespace DieModel;
  Execution execution(repeated_pe_plan());
  const auto query = execution.query_internal_timer_window(4);
  expect(!query.supported && query.slices.empty() &&
             query.reason == "other_scheduled_internal_boundary",
         "parallel pre-scheduled PE waves are ineligible for a sequential window");
}

void test_same_end_timer_batch_compacts_once_and_preserves_boundary() {
  using namespace DieModel;
  constexpr std::uint64_t kTimerNs = 1953125;
  for (const std::uint64_t repeat : {8ULL, 32ULL, 128ULL}) {
    Execution execution(repeated_timer_waves_then_mem_plan(repeat));

    const ExecutionAction first = execution.next_action();
    expect(first.kind == ExecutionAction::Kind::Timer && first.ns == kTimerNs,
           "same-end repeated TIMER waves expose their original first boundary");

    execution.advance_time(first.ns);
    const auto after_parent_batch = execution.export_progress();
    expect(after_parent_batch.now_ns == kTimerNs &&
               after_parent_batch.stages.size() == 3 &&
               after_parent_batch.stages[0].completed == repeat &&
               after_parent_batch.stages[1].completed == 0 &&
               after_parent_batch.active_id_compaction_calls == 1 &&
               after_parent_batch.active_id_entries_examined == repeat &&
               after_parent_batch.active_id_entries_moved == 0 &&
               after_parent_batch.active_id_entries_removed == repeat,
           "one same-end completion batch performs one stable linear compaction");
    expect(after_parent_batch.frontier.active_stages.size() == repeat,
           "compaction retains the newly released child TIMER frontier");
    for (std::size_t i = 0; i < after_parent_batch.frontier.active_stages.size(); ++i) {
      const auto& active = after_parent_batch.frontier.active_stages[i];
      expect(active.stage_id == 1 && active.wave_index == i,
             "same-end child TIMER waves retain ready-key and insertion order");
    }

    const ExecutionAction second = execution.next_action();
    expect(second.kind == ExecutionAction::Kind::Timer && second.ns == kTimerNs,
           "released child timers retain their exact second event boundary");
    execution.advance_time(second.ns);
    const ExecutionAction memory = execution.next_action();
    const auto before_memory_completion = execution.export_progress();
    expect(memory.kind == ExecutionAction::Kind::MemoryRequest &&
               memory.stage_id == 2 &&
               before_memory_completion.now_ns == 2 * kTimerNs &&
               before_memory_completion.stages[1].completed == repeat &&
               before_memory_completion.memory_pending,
           "the all-child dependency becomes a memory request at the original boundary");
    expect(double_bits(memory.actual_bytes) == double_bits(64.125) &&
               double_bits(memory.service_units) == double_bits(96.5),
           "memory request preserves both binary64 payloads exactly");
    expect(before_memory_completion.active_id_compaction_calls == 2 &&
               before_memory_completion.active_id_entries_examined == 2 * repeat &&
               before_memory_completion.active_id_entries_moved == 0 &&
               before_memory_completion.active_id_entries_removed == 2 * repeat,
           "second same-end batch adds exactly one more linear compaction");

    std::cout << "[internal_service_progress_test] timer_waves=" << repeat
              << " compactions=" << before_memory_completion.active_id_compaction_calls
              << " examined=" << before_memory_completion.active_id_entries_examined
              << " moved=" << before_memory_completion.active_id_entries_moved
              << " removed=" << before_memory_completion.active_id_entries_removed
              << '\n';
  }
}

}  // namespace

int main() {
  test_partial_progress_and_stale_tokens();
  test_boundary_and_overflow_rejection();
  test_timer_stretch_preview_is_read_only_and_stops_at_memory();
  test_preview_preserves_full_state_with_unvisited_suffix();
  test_preview_overflow_exception_is_read_only();
  test_bounded_window_query_and_sequential_service_cursor();
  test_window_query_stops_at_mixed_future_action();
  test_same_end_timer_batch_compacts_once_and_preserves_boundary();
  if (failures != 0) {
    std::cerr << "[internal_service_progress_test] " << failures
              << " failure(s)\n";
    return 1;
  }
  std::cout << "[internal_service_progress_test] PASS\n";
  return 0;
}
