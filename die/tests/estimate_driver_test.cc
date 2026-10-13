#include "test_support.hh"

#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>

using namespace coarse_die;
using namespace coarse_die_test;

namespace {

json gemm_task() {
  return {{"op_kind", "gemm"},
          {"tensor_shapes", {{"m", 7}, {"n", 5}, {"k", 9}}},
          {"storage_dtype", {{"weights", "int4"},
                             {"activations", "bf16"},
                             {"output", "bf16"}}},
          {"execution_profile", "int4_matrix_vec_fp32_assumed_v1"},
          {"layer_instances", 1},
          {"weight_passes", 1}};
}

std::uint64_t actual_runtime_finish_ns(
    const Hardware& hardware, std::shared_ptr<const Plan> plan) {
  Runtime runtime(hardware);
  const std::uint64_t handle = runtime.submit(std::move(plan), owner(1, 1, 1), 0);
  std::size_t events = 0;
  while (!runtime.drained()) {
    const auto next = runtime.next_event();
    require(next.has_value(), "actual Runtime has a next event until completion");
    require(*next > runtime.now_ns(), "actual Runtime events advance time");
    runtime.advance_to(*next);
    require(++events <= 1000000, "actual Runtime completion stays within its event budget");
  }
  const json progress = runtime.query_progress(handle);
  require(progress.at("state") == "finished",
          "actual Runtime terminal state is finished");
  return time_from_progress(progress, "finish_ns");
}

template <class Function>
void require_throws_prefix(Function&& function, const std::string& prefix,
                           const std::string& message) {
  try {
    function();
  } catch (const std::exception& error) {
    require(std::string(error.what()).find(prefix) == 0, message);
    return;
  }
  throw std::runtime_error(message + ": did not throw");
}

void low_cost_estimate_matches_report_and_runtime() {
  const Hardware hardware = example_hardware();
  const auto plan = compile(hardware, gemm_task());
  const auto immutable_hardware = std::make_shared<const Hardware>(hardware);
  require(plan != nullptr, "compiler returns a Plan");
  require(plan->hardware_fingerprint == hardware.fingerprint,
          "compiled Plan retains the exact hardware fingerprint");
  require(plan_to_json(*plan).at("fingerprint") == plan->fingerprint,
          "compiled Plan retains its canonical fingerprint");

  const std::uint64_t candidate_latency =
      detail::estimate_latency_ns(hardware, plan);
  const std::uint64_t shared_hardware_latency =
      detail::estimate_latency_ns(immutable_hardware, plan);
  const json full_estimate = estimate_alone(hardware, *plan);
  const std::uint64_t runtime_latency =
      actual_runtime_finish_ns(hardware, plan);
  require(candidate_latency == runtime_latency,
          "low-cost candidate estimate equals an actual Runtime completion");
  require(shared_hardware_latency == candidate_latency,
          "shared immutable hardware estimate equals the by-value path");
  require(full_estimate.at("estimated_latency_ns") == runtime_latency,
          "public standalone estimate keeps the same Runtime latency");
  require(full_estimate.at("execution").at("finish_ns") == runtime_latency,
          "full report keeps the completed execution record");
  require(full_estimate.at("hardware_fingerprint") == hardware.fingerprint &&
              full_estimate.at("plan_fingerprint") == plan->fingerprint,
          "full report preserves plan and hardware fingerprints");
  require(full_estimate.at("resource_domains").size() == resource_count &&
              full_estimate.at("domain_capacities").size() == resource_count &&
              full_estimate.at("domain_work").size() == resource_count &&
              full_estimate.contains("runtime_counters"),
          "full report retains its resource and runtime fields");

  const auto repeated_plan = compile(hardware, gemm_task());
  require(repeated_plan->fingerprint == plan->fingerprint &&
              repeated_plan->summary.at("selected_template") ==
                  plan->summary.at("selected_template") &&
              repeated_plan->summary.at("estimated_latency_ns") ==
                  plan->summary.at("estimated_latency_ns"),
          "shared candidate hardware preserves deterministic selection and hash");
}

void runtime_shared_hardware_keeps_capacity_state_private() {
  const Hardware hardware = example_hardware();
  const auto immutable_hardware = std::make_shared<const Hardware>(hardware);
  const auto plan = compile(hardware, gemm_task());
  Runtime throttled(immutable_hardware);
  Runtime baseline(immutable_hardware);
  const double peak = hardware.rates[COMPUTE_GATE];
  bool has_compute_work = false;
  for (const auto& epoch : plan->epochs) {
    for (const auto& activity : epoch.activities) {
      has_compute_work = has_compute_work || activity.work[COMPUTE_GATE] > 0.0;
    }
  }
  require(peak > 0.0 && has_compute_work,
          "example GEMM has positive compute work and capacity");
  throttled.update_capacity(COMPUTE_GATE, 0.0, 0);

  const auto slow_handle = throttled.submit(plan, owner(2, 2, 1), 0);
  const auto base_handle = baseline.submit(plan, owner(3, 3, 1), 0);
  require(!throttled.terminal_finish_ns(slow_handle).has_value() &&
              !baseline.terminal_finish_ns(base_handle).has_value(),
          "typed terminal time is empty while executions are live");
  std::size_t throttled_events = 0;
  while (const auto next = throttled.next_event()) {
    throttled.advance_to(*next);
    require(++throttled_events <= 1000000,
            "throttled Runtime reaches its resource block within the event budget");
  }
  require(!throttled.drained(),
          "zero capacity blocks unfinished work in the throttled Runtime");
  require(baseline.next_event().has_value(),
          "the other Runtime retains its independent positive capacity");
  while (!baseline.drained()) {
    const auto next = baseline.next_event();
    require(next.has_value(), "baseline Runtime has a next event until completion");
    baseline.advance_to(*next);
  }
  const auto base_finish = baseline.terminal_finish_ns(base_handle);
  require(!throttled.terminal_finish_ns(slow_handle).has_value() &&
              base_finish.has_value(),
          "typed terminal time is available only after an execution terminates");
  require(baseline.query_progress(base_handle).at("finish_ns") == *base_finish,
          "typed terminal time matches the public progress report");
  require(throttled.hardware().fingerprint == hardware.fingerprint &&
              baseline.hardware().fingerprint == hardware.fingerprint &&
              immutable_hardware->rates[COMPUTE_GATE] == peak &&
              throttled.hardware().rates[COMPUTE_GATE] == peak &&
              baseline.hardware().rates[COMPUTE_GATE] == peak &&
              throttled.counters().at("capacity").at(COMPUTE_GATE) == 0.0 &&
              baseline.counters().at("capacity").at(COMPUTE_GATE) == peak,
          "per-Runtime capacity overrides do not mutate shared hardware peaks");

  const auto cancel_time = throttled.now_ns();
  throttled.cancel(slow_handle, cancel_time);
  const auto cancelled_finish = throttled.terminal_finish_ns(slow_handle);
  require(cancelled_finish.has_value() && *cancelled_finish == cancel_time,
          "typed terminal time is available for a cancelled execution");
  baseline.destroy_execution(base_handle);
  require_throws_prefix(
      [&]() { baseline.terminal_finish_ns(base_handle); }, "stale_handle:",
      "typed terminal time preserves stale-handle errors");
}

void low_cost_estimate_still_validates_hardware_and_physical_work() {
  const Hardware hardware = example_hardware();
  const auto plan = compile(hardware, gemm_task());

  auto wrong_hardware = std::make_shared<Plan>(*plan);
  wrong_hardware->hardware_fingerprint = "wrong-hardware-fingerprint";
  require_throws_prefix(
      [&]() { detail::estimate_latency_ns(hardware, wrong_hardware); },
      "incompatible_fingerprint:",
      "low-cost estimate rejects a plan from different hardware");

  auto invalid_work = std::make_shared<Plan>(*plan);
  invalid_work->epochs.front().activities.front().work[COMPUTE_GATE] = -1.0;
  require_throws_prefix(
      [&]() { detail::estimate_latency_ns(hardware, invalid_work); },
      "invalid_plan:",
      "low-cost estimate still validates physical resource work");
}

void estimate_driver_preserves_overflow_failure() {
  const Hardware hardware = example_hardware();
  const auto plan = compile(hardware, gemm_task());
  auto late_plan = std::make_shared<Plan>(*plan);
  late_plan->launch_ns = std::numeric_limits<std::uint64_t>::max();
  late_plan->fingerprint.clear();
  late_plan->fingerprint =
      plan_to_json(*late_plan).at("fingerprint").get<std::string>();
  require_throws_prefix(
      [&]() { detail::estimate_latency_ns(hardware, late_plan); },
      "overflow:",
      "low-cost estimate preserves Runtime event-time overflow errors");
}

}  // namespace

int main() {
  return run_test("estimate_driver_test", [] {
    low_cost_estimate_matches_report_and_runtime();
    runtime_shared_hardware_keeps_capacity_state_private();
    low_cost_estimate_still_validates_hardware_and_physical_work();
    estimate_driver_preserves_overflow_failure();
  });
}
