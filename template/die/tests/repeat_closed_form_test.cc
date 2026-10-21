// P9+P7-A: estimate-channel closed form for declared serial repeat plans —
// equivalence matrix (research track, 2026-10-17).
//
// Layer 1 (certification, holds before AND after the fast path): for every
// R in the matrix, the estimate-channel latency equals, bit for bit, what a
// full runtime simulation of the same plan publishes (and what estimate_alone
// reports). This is the constructive requirement that the closed form outputs
// exactly the simulator's number.
//
// Layer 2 (engagement — the red marker before the fast path exists): at a
// large compiled R the estimate must complete far faster than the O(R^2)-wall
// full simulation P8 measured (R=1024 full simulation is seconds of wall on
// the P8 machine). The threshold is a generous order-of-magnitude liveness
// check, not a precision benchmark; comparative timing verdicts stay under
// the same-window interleaved A/B discipline.
//
// Design and guards: docs/analysis/research-track-20261017/P9-CLOSED-FORM/
// README.md. P8 measurement: ../P8-MEASUREMENT/README.md.
#include "test_support.hh"
#include "coarse_die/Execution.hh"
#include "coarse_die/Plan.hh"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <memory>

using namespace coarse_die;
using namespace coarse_die_test;

namespace {

using u64 = std::uint64_t;

json ffn_repeat_task(u64 layers) {
  return json{{"op_kind", "ffn_region"},
              {"tensor_shapes", {{"m", 1}, {"hidden", 8}, {"ffn", 16},
                                 {"mlp_variant", "swiglu"}}},
              {"fusion_policy", "materialized"},
              {"layer_instances", layers},
              {"weight_passes", std::uint64_t{1}},
              {"repeat_seams", {{"mode", "serial_on_drain"}}}};
}

u64 full_simulation_finish(const std::shared_ptr<const Hardware>& hardware,
                           std::shared_ptr<const Plan> plan) {
  Runtime runtime(hardware);
  const auto handle = runtime.submit(std::move(plan), owner(1, 1, 1), 0);
  unsigned guard = 0;
  while (!runtime.drained()) {
    const auto next = runtime.next_event();
    require(next.has_value() && *next > runtime.now_ns() && ++guard < 3000000,
            "closed-form fixture has a bounded advancing completion sequence");
    runtime.advance_to(*next);
  }
  const auto finish = runtime.terminal_finish_ns(handle);
  require(finish.has_value(), "closed-form fixture drains to a finish");
  return *finish;
}

void verify_compiled_matrix() {
  const auto hardware =
      std::make_shared<const Hardware>(example_hardware());
  for (const u64 layers : std::vector<u64>{
           2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 24, 96, 256}) {
    const auto plan = compile(*hardware, ffn_repeat_task(layers));
    require(plan->repeat && plan->repeat->count == layers,
            "compiled fixture declares the expected repeat count");
    const u64 simulated = full_simulation_finish(hardware, plan);
    const u64 estimated = detail::estimate_latency_ns(*hardware, plan);
    require(estimated == simulated,
            "estimate channel returns exactly the simulated finish");
    const auto report = estimate_alone(*hardware, *plan);
    require(report.at("estimated_latency_ns").get<u64>() == simulated,
            "estimate_alone publishes the same finish");
  }
}

// Synthetic two-epoch chain on unit-rate hardware: independent integer ns
// arithmetic (one unit of work = one ns at rate 1e9), a nonzero seam and a
// seam=0 variant, feasible up to the 4096 repeat ceiling.
std::shared_ptr<const Plan> chain_plan(const Hardware& hw, u64 count,
                                       u64 seam_ns) {
  Epoch compute;
  compute.name = "compute";
  compute.activities = {activity(1, Role::Compute, COMPUTE_GATE, 1.0)};
  Epoch store;
  store.name = "store";
  store.activities = {activity(2, Role::Compute, SRAM_READ, 1.0)};
  auto plan = std::make_shared<Plan>();
  plan->task = json{{"op_kind", "repeat_closed_form_fixture"}};
  plan->summary = json{{"status", "test_fixture"}};
  plan->hardware_fingerprint = hw.fingerprint;
  plan->launch_ns = 0;
  plan->repeat = PlanRepeat{count, seam_ns};
  plan->epochs = {compute, store};
  plan->fingerprint =
      plan_to_json(*plan).at("fingerprint").get<std::string>();
  return plan;
}

void verify_synthetic_matrix() {
  std::array<double, resource_count> rates{};
  rates[COMPUTE_GATE] = 1e9;
  rates[SRAM_READ] = 1e9;
  const auto hardware =
      std::make_shared<const Hardware>(manual_hardware(rates));
  // Hand-computable check first: 2 bodies of 2 ns (compute + store units)
  // plus one 5 ns seam.
  const auto small = chain_plan(*hardware, 2, 5);
  require(full_simulation_finish(hardware, small) == 9,
          "unit chain fixture keeps its hand-computed schedule");
  for (const auto& [count, seam] :
       std::vector<std::pair<u64, u64>>{{8, 0}, {8, 5}, {1024, 5}, {4096, 5}}) {
    const auto plan = chain_plan(*hardware, count, seam);
    const u64 simulated = full_simulation_finish(hardware, plan);
    const u64 estimated = detail::estimate_latency_ns(*hardware, plan);
    require(estimated == simulated,
            "synthetic estimate channel returns exactly the simulated finish");
  }
}

void verify_nonrepeat_channel_unchanged() {
  const auto hardware =
      std::make_shared<const Hardware>(example_hardware());
  // A legacy non-declared task and a gemm task keep the exact estimate the
  // full simulation produces (the fast path declines before any work).
  for (const auto& task : std::vector<json>{
           json{{"op_kind", "ffn_region"},
                {"tensor_shapes", {{"m", 1}, {"hidden", 8}, {"ffn", 16},
                                   {"mlp_variant", "swiglu"}}},
                {"fusion_policy", "materialized"},
                {"layer_instances", 3},
                {"weight_passes", std::uint64_t{1}}},
           json{{"op_kind", "gemm"},
                {"tensor_shapes", {{"m", 128}, {"n", 128}, {"k", 128}}},
                {"storage_dtype", {{"weights", "bf16"},
                                   {"activations", "bf16"},
                                   {"output", "bf16"}}},
                {"execution_profile", "int4_matrix_vec_fp32_assumed_v1"},
                {"layer_instances", 1},
                {"weight_passes", 1}}}) {
    const auto plan = compile(*hardware, task);
    require(!plan->repeat, "legacy fixture stays non-declared");
    const u64 simulated = full_simulation_finish(hardware, plan);
    require(detail::estimate_latency_ns(*hardware, plan) == simulated,
            "non-repeat estimate keeps the full-simulation result");
  }
}

void verify_engagement_red_marker() {
  const auto hardware =
      std::make_shared<const Hardware>(example_hardware());
  const auto plan = compile(*hardware, ffn_repeat_task(1024));
  const auto start = std::chrono::steady_clock::now();
  const u64 estimated = detail::estimate_latency_ns(*hardware, plan);
  const double seconds =
      std::chrono::duration_cast<std::chrono::duration<double>>(
          std::chrono::steady_clock::now() - start)
          .count();
  require(estimated == full_simulation_finish(hardware, plan),
          "engagement case stays bit-exact against the full simulation");
  require(seconds < 0.5,
          "large-R estimate completes in fast-path time (P8 measured the "
          "full simulation of this plan family at ~2.0 s wall for R=1024; "
          "the closed form is milliseconds)");
}

}  // namespace

int main() {
  verify_compiled_matrix();
  verify_synthetic_matrix();
  verify_nonrepeat_channel_unchanged();
  verify_engagement_red_marker();
  std::cout << "repeat_closed_form_test passed\n";
  return 0;
}
