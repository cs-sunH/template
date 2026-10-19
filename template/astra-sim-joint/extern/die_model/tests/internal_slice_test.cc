#include "DieModel.hh"

#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "test_helpers.hh"

namespace {

int failures = 0;

void expect(bool condition, const std::string& message) {
  if (!condition) {
    std::cerr << "[internal_slice_test] FAIL: " << message << '\n';
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

constexpr double kNudge = 1.0 - 1e-12;

std::shared_ptr<const DieModel::DiePlan> plan_of(
    std::vector<DieModel::DieStage> stages) {
  auto data = std::make_shared<DieModel::DiePlanData>();
  data->spec = std::make_shared<const DieModel::DieOpSpec>(
      die_test::spec_of(DieModel::OpKind::LocalRegion,
                        DieModel::Phase::Prefill, 4, 4, 0, 0));
  data->hardware = die_test::hw_default();
  data->stages = std::move(stages);
  return std::make_shared<const DieModel::DiePlan>(std::move(data));
}

DieModel::DieStage pe_stage(const std::uint64_t repeat,
                            const std::uint64_t ns) {
  const auto hw = die_test::hw_default();
  DieModel::DieStage stage;
  stage.label = "test-pe";  // classification must use typed resource/work.
  stage.resource = DieModel::StageResource::PE;
  stage.wave_repeat = repeat;
  const double peak = hw.core_row.pe_int4_gops_per_core * 1e9 * hw.eta_pe;
  stage.pe_core_flop = static_cast<double>(ns) * peak / 1e9 * kNudge;
  return stage;
}

DieModel::DieStage mem_stage() {
  DieModel::DieStage stage;
  stage.label = "memory";
  stage.resource = DieModel::StageResource::MEM;
  stage.mem.service_units = 1.0;
  stage.mem.actual_bytes = 1.0;
  return stage;
}

void test_single_pe_frontier() {
  using namespace DieModel;
  Execution execution(plan_of({pe_stage(2, 1000)}));
  const InternalSliceInfo info = execution.describe_internal_slice();
  expect(info.classification == InternalSliceClass::PurePe,
         "one non-overlapping PE instance is typed PURE_PE");
  expect(info.partial_service_supported && info.action_token.valid &&
             info.action_token.template_key == info.template_key,
         "eligible frontier exposes stable token and partial service");
  expect(info.active_stages.size() == 1 &&
             info.active_stages[0].wave_index == 0 &&
             info.active_stages[0].remaining_service_ns == 1000,
         "frontier names its active stage/wave and exact remaining time");
  expect(info.active_stages[0].hardware_min_ns <=
             info.active_stages[0].original_service_ns,
         "physical PE lower bound does not exceed reference service");
  expect(execution.export_progress().now_ns == 0,
         "frontier inspection leaves the authoritative cursor unchanged");
}

void test_ineligible_mixed_and_memory_frontiers() {
  using namespace DieModel;
  DieStage vec;
  vec.label = "test-vec";
  vec.resource = StageResource::VEC;
  vec.wave_repeat = 1;
  const auto hw = die_test::hw_default();
  const double vec_peak = hw.core_row.vec_g_eqop_per_core * 1e9 * hw.eta_vec;
  vec.vec_core_eqop = 1000.0 * vec_peak / 1e9 * kNudge;
  Execution mixed(plan_of({pe_stage(1, 1000), vec}));
  expect(mixed.describe_internal_slice().classification ==
             InternalSliceClass::MixedFrontier,
         "concurrent PE+VEC events are never averaged together");
  expect_throw([&] {
    mixed.advance_internal_service(
        InternalActionToken{"wrong-plan", 0, 1, true}, 100);
  },
               "mixed frontier rejects partial-service mutation");

  Execution memory(plan_of({mem_stage()}));
  const auto request = memory.next_action();
  expect(request.kind == ExecutionAction::Kind::MemoryRequest,
         "memory plan issues the same external request action");
  const auto pending = memory.export_progress();
  expect(pending.memory_pending && pending.has_memory_instance_id &&
             pending.memory_stage_id == request.stage_id &&
             pending.pending_memory_actual_bytes == request.actual_bytes &&
             pending.pending_memory_service_units == request.service_units &&
             pending.frontier.active_stages.size() == 1 &&
             pending.frontier.active_stages[0].memory_pending,
         "progress exposes the outstanding memory instance and true sink duty");
  expect(memory.describe_internal_slice().classification ==
             InternalSliceClass::MemoryService,
         "external memory request is typed and excluded from PURE service");
  expect_throw([&] { memory.advance_external_wait(1, request.stage_id + 1); },
               "external wait requires the matching outstanding memory stage");
  memory.advance_external_wait(1, request.stage_id);
  expect(memory.export_progress().external_memory_wait_elapsed_ns == 1,
         "external memory wait is tracked independently from internal service");
  memory.complete_memory(request.stage_id);
  expect(memory.export_progress().finished &&
             memory.describe_internal_slice().classification ==
             InternalSliceClass::Idle,
         "completed memory stage retires through the original callback path");
}

}  // namespace

int main() {
  test_single_pe_frontier();
  test_ineligible_mixed_and_memory_frontiers();
  if (failures != 0) {
    std::cerr << "[internal_slice_test] " << failures << " failure(s)\n";
    return 1;
  }
  std::cout << "[internal_slice_test] PASS\n";
  return 0;
}
