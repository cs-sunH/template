// WP1 (roofline stage 2, 2026-10-19): typed repeat observation replaced the
// closed-form estimator's per-event export_snapshot(). This test pins the
// new Runtime::repeat_observation() contract:
//   1. field parity — the typed {round, ready_ns} equals, for the same state
//      and handle, exactly what export_snapshot() serializes for those two
//      fields (live states, round transitions, and the terminal state);
//   2. purity — the observation moves no counter, builds no JSON, and is
//      idempotent (the snapshot export here is the test oracle only);
//   3. missing-handle semantics — nullopt for unknown/destroyed handles,
//      matching the old snapshot-array scan that recorded nothing for a
//      missing handle (never a 0/0 record), while terminal_finish_ns keeps
//      its stale_handle throw convention;
//   4. estimator equivalence — the estimate channel still returns exactly
//      the fully simulated finish for repeat plans (seam 0/nonzero, at and
//      below the closed-form count threshold, compiled templates, and a
//      zero-work body that never produces three observations), and the
//      estimator's public observation loop observes three consecutive
//      rounds without any snapshot or progress construction (the production
//      estimate path's zero-snapshot proof is the M1 callgrind run).
// The full equivalence matrix for the closed form itself lives in
// repeat_closed_form_test.cc; this file only adds the observation contract.
#include "test_support.hh"
#include "coarse_die/Execution.hh"
#include "coarse_die/Plan.hh"

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

using namespace coarse_die;
using namespace coarse_die_test;

namespace {

using u64 = std::uint64_t;

// Same synthetic two-epoch unit-rate chain family as
// repeat_closed_form_test.cc (independent integer ns arithmetic, one unit of
// work = one ns at rate 1e9), kept self-contained here.
std::shared_ptr<const Plan> chain_plan(const Hardware& hw, u64 count,
                                       u64 seam_ns, double work_units = 1.0) {
  Epoch compute;
  compute.name = "compute";
  compute.activities = {activity(1, Role::Compute, COMPUTE_GATE, work_units)};
  Epoch store;
  store.name = "store";
  store.activities = {activity(2, Role::Compute, SRAM_READ,
                               2.0 * work_units)};
  auto plan = std::make_shared<Plan>();
  plan->task = json{{"op_kind", "repeat_observation_fixture"}};
  plan->summary = json{{"status", "test_fixture"}};
  plan->hardware_fingerprint = hw.fingerprint;
  plan->launch_ns = 0;
  plan->repeat = PlanRepeat{count, seam_ns};
  plan->epochs = {compute, store};
  plan->fingerprint =
      plan_to_json(*plan).at("fingerprint").get<std::string>();
  return plan;
}

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
            "observation fixture has a bounded advancing completion sequence");
    runtime.advance_to(*next);
  }
  const auto finish = runtime.terminal_finish_ns(handle);
  require(finish.has_value(), "observation fixture drains to a finish");
  return *finish;
}

// Oracle: the two snapshot fields the estimator used to read via a full
// export_snapshot() plus an array scan.
struct SnapshotFields {
  u64 round;
  u64 ready_ns;
};

SnapshotFields snapshot_fields(const Runtime& runtime, u64 handle) {
  const json snapshot = runtime.export_snapshot();
  const json& executions = snapshot.at("executions");
  require(executions.is_array(),
          "oracle snapshot executions must be an array");
  for (const auto& entry : executions) {
    if (entry.value("handle", u64{0}) != handle) continue;
    return {entry.value("round", u64{0}),
            entry.value("ready_ns", u64{0})};
  }
  throw std::runtime_error("oracle snapshot lacks the fixture handle");
}

std::array<double, resource_count> chain_rates() {
  std::array<double, resource_count> rates{};
  rates[COMPUTE_GATE] = 1e9;
  rates[SRAM_READ] = 1e9;
  return rates;
}

void verify_field_parity_and_purity() {
  const Hardware hw = manual_hardware(chain_rates());
  Runtime runtime(hw);
  const auto handle =
      runtime.submit(chain_plan(hw, 8, 5), owner(1, 1, 1), 0);
  bool saw_round_transition = false;
  u64 previous_round = 0;
  unsigned guard = 0;
  while (!runtime.drained() && guard++ < 4096) {
    const auto next = runtime.next_event();
    require(next.has_value() && *next > runtime.now_ns(),
            "parity fixture must keep advancing");
    runtime.advance_to(*next);
    const auto observation = runtime.repeat_observation(handle);
    require(observation.has_value(),
            "live fixture handle stays observable");
    const SnapshotFields fields = snapshot_fields(runtime, handle);
    require(static_cast<u64>(observation->round) == fields.round &&
                observation->ready_ns == fields.ready_ns,
            "typed observation equals the snapshot fields of the same state");
    if (observation->round != previous_round) {
      require(observation->round > previous_round,
              "fixture rounds advance monotonically");
      saw_round_transition = true;
      previous_round = observation->round;
    }
    // Purity: a second read is identical and moves no observation counter
    // (the wholesale counter object must be unchanged by the observation;
    // the oracle export above was billed before this snapshot was taken).
    const json counters_before = runtime.observation_counters();
    const auto again = runtime.repeat_observation(handle);
    require(again.has_value() && again->round == observation->round &&
                again->ready_ns == observation->ready_ns,
            "typed observation is idempotent");
    require(runtime.observation_counters() == counters_before,
            "typed observation moves no observation counter");
  }
  require(runtime.drained(), "parity fixture drains within its guard bound");
  require(saw_round_transition,
          "parity fixture exercised at least one round transition");
  const auto observation = runtime.repeat_observation(handle);
  require(observation.has_value(),
          "terminal execution stays observable after draining");
  const SnapshotFields fields = snapshot_fields(runtime, handle);
  require(static_cast<u64>(observation->round) == fields.round &&
              observation->ready_ns == fields.ready_ns,
          "typed observation equals the snapshot fields at the terminal state");
}

void verify_missing_handle_semantics() {
  Runtime runtime(example_hardware());
  require(!runtime.repeat_observation(999).has_value(),
          "unknown handle observes nothing; the old array scan recorded "
          "nothing for it the same way instead of a 0/0 record");
  bool threw = false;
  try {
    (void)runtime.terminal_finish_ns(999);
  } catch (const std::out_of_range&) {
    threw = true;
  }
  require(threw, "terminal_finish_ns keeps its stale_handle throw convention");
  const Hardware hw = manual_hardware(chain_rates());
  Runtime owner_runtime(hw);
  const auto handle =
      owner_runtime.submit(chain_plan(hw, 8, 5), owner(1, 1, 1), 0);
  owner_runtime.cancel(handle, 0);
  owner_runtime.destroy_execution(handle);
  require(!owner_runtime.repeat_observation(handle).has_value(),
          "destroyed handle observes nothing");
}

void verify_estimate_equivalence() {
  const auto hardware =
      std::make_shared<const Hardware>(manual_hardware(chain_rates()));
  for (const auto& [count, seam] :
       std::vector<std::pair<u64, u64>>{{8, 0}, {8, 5}, {9, 7}, {16, 5}}) {
    const auto plan = chain_plan(*hardware, count, seam);
    require(detail::estimate_latency_ns(*hardware, plan) ==
                full_simulation_finish(hardware, plan),
            "repeat estimate channel returns exactly the simulated finish");
  }
  // Below the closed-form count threshold the drain channel is unchanged.
  const auto below = chain_plan(*hardware, 7, 5);
  require(detail::estimate_latency_ns(*hardware, below) ==
              full_simulation_finish(hardware, below),
          "count=7 declines the closed form and still matches the drain");
  // A compiled template with a real declared repeat.
  const Hardware compiled_hw = example_hardware();
  const auto compiled = compile(compiled_hw, ffn_repeat_task(8));
  require(detail::estimate_latency_ns(compiled_hw, compiled) ==
              full_simulation_finish(
                  std::make_shared<const Hardware>(compiled_hw), compiled),
          "compiled repeat fixture matches the full simulation");
  // Zero-work bodies never produce three observations: the estimator falls
  // back to the same drain and both agree on the trivial finish.
  const auto zero = chain_plan(*hardware, 8, 5, 0.0);
  require(detail::estimate_latency_ns(*hardware, zero) ==
              full_simulation_finish(hardware, zero),
          "zero-work repeat falls back to the drain with the same finish");
}

void verify_estimator_loop_needs_no_snapshot() {
  // Replica of the estimator's observation loop over the public API: three
  // consecutive round observations without a single snapshot construction.
  // (The production loop itself is private; the definitive production-path
  // evidence is the M1 callgrind run on the real estimate entry points.)
  const Hardware hw = manual_hardware(chain_rates());
  Runtime runtime(hw);
  const auto handle =
      runtime.submit(chain_plan(hw, 8, 5), owner(1, 1, 1), 0);
  std::vector<u64> observed_rounds;
  u64 last_round = 0;
  unsigned guard = 0;
  while (!runtime.drained() && observed_rounds.size() < 3 &&
         guard++ < 4096) {
    const auto next = runtime.next_event();
    require(next.has_value() && *next > runtime.now_ns(),
            "loop replica must keep advancing");
    runtime.advance_to(*next);
    const auto observation = runtime.repeat_observation(handle);
    require(observation.has_value(), "loop replica handle stays observable");
    const u64 round = static_cast<u64>(observation->round);
    if (round > last_round) {
      observed_rounds.push_back(round);
      last_round = round;
    }
  }
  require(observed_rounds.size() == 3 &&
              observed_rounds[1] == observed_rounds[0] + 1 &&
              observed_rounds[2] == observed_rounds[1] + 1,
          "loop replica observes three consecutive rounds");
  // The replica itself constructs no snapshot or progress object by
  // construction (neither API is called); the production-path proof — zero
  // export_snapshot calls from the estimate entry points — is the M1
  // callgrind run on the real workload binary.
}

}  // namespace

int main() {
  return run_test("repeat_observation_test", [] {
    verify_field_parity_and_purity();
    verify_missing_handle_semantics();
    verify_estimate_equivalence();
    verify_estimator_loop_needs_no_snapshot();
  });
}
