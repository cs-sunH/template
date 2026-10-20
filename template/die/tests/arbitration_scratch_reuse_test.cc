#include "test_support.hh"

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

using namespace coarse_die;
using namespace coarse_die_test;

// Directed test for the arbitration scratch reuse (roofline stage 3, WP3.3):
// Runtime::allocate() pools its rows, refs, owner inputs, and simplex across
// calls. The pools are bounded by the solver's structural limits (at most 96
// variables, 13 resource rows, tier rows bounded by the variable count), so
// the high-water storage cannot grow with the number of events; the bound is
// by construction, observable here as output stability across arbitrarily
// many reused-scratch arbitrations. Every logical container is refilled
// before it is read within a call, which these checks pin from the outside:
//
// 1. A second identical batch inside one Runtime (all its arbitrations reuse
//    pooled scratch grown by the first batch) must produce the same
//    per-execution progress as the first batch's fresh allocations.
// 2. Redundant consecutive arbitration of an unchanged state — driven by
//    update_capacity() with an unchanged rate, which re-runs allocate() —
//    must not change any published rate, alpha, or completion time.
// 3. Many repeated batches of varying concurrent widths must keep producing
//    the reference outputs, ruling out cross-call contamination as the pool
//    high-water mark settles.
//
// The batch walks every arbitration path: a one-activity interval (singleton
// closed form), shared nonnegative rows across distinct owners (packing),
// identical owner blocks across executions with another owner present
// (symmetric-block merge with multiplicity), credit-buffered epochs with
// opposite deltas (signed boundary rows, lexicographic LP restarts/tiers and
// bottleneck probes), and a curved epoch (ratio fast path attempt plus the
// tangent-rounding certification).

namespace {
constexpr std::uint64_t kNsPerSecond = 1000000000ULL;
constexpr std::uint64_t kBatchDrainNs = 20 * kNsPerSecond;

Hardware arbitration_hardware() {
  std::array<double, resource_count> capacities{};
  capacities[COMPUTE_GATE] = 6.0;
  capacities[LOCAL] = 8.0;
  capacities[HBM0] = 2.0;
  capacities[HBM1] = 3.0;
  return manual_hardware(capacities);
}

std::shared_ptr<const Plan> finalize_plan(std::vector<Epoch> epochs,
                                          const Hardware& hardware,
                                          std::uint64_t launch_ns,
                                          const char* name) {
  auto plan = std::make_shared<Plan>(*manual_plan(hardware, std::move(epochs), name));
  plan->launch_ns = launch_ns;
  plan->fingerprint.clear();
  plan->fingerprint =
      plan_to_json(*plan).at("fingerprint").get<std::string>();
  return plan;
}

// One activity on the compute gate: whenever it is the only ready activity
// the singleton closed form runs.
std::shared_ptr<const Plan> singleton_plan(const Hardware& hardware,
                                           std::uint64_t launch_ns) {
  Epoch epoch;
  epoch.name = "scratch_singleton";
  auto task = activity(1, Role::Compute, COMPUTE_GATE, 12.0);
  task.canonical_q = 4.0;
  epoch.activities.push_back(task);
  return finalize_plan({epoch}, hardware, launch_ns, "scratch-singleton");
}

// Two activities on nonnegative shared rows: the packing solver path. Two
// executions carrying this plan under one owner tuple form equivalent blocks
// that the merge step collapses into mirrors with multiplicity while another
// owner keeps multiple_owners true.
std::shared_ptr<const Plan> packing_plan(const Hardware& hardware,
                                         std::uint64_t launch_ns) {
  Epoch epoch;
  epoch.name = "scratch_packing";
  auto load = activity(1, Role::Load, HBM0, 6.0);
  load.work[HBM1] = 2.0;
  load.hbm_bytes[0][0] = 6;
  load.canonical_q = 6.0;
  auto store = activity(2, Role::Store, HBM1, 3.0);
  store.hbm_bytes[1][1] = 3;
  store.canonical_q = 3.0;
  epoch.activities = {load, store};
  return finalize_plan({epoch}, hardware, launch_ns, "scratch-packing");
}

// Credit-buffered epoch with opposite deltas: at the declared initial credit
// the boundary rows are signed, so packing declines and the lexicographic
// signed LP (restarts, tiers, degenerate-face probes) arbitrates it.
std::shared_ptr<const Plan> credit_lp_plan(const Hardware& hardware,
                                           std::uint64_t launch_ns) {
  Epoch epoch;
  epoch.name = "scratch_credit_lp";
  epoch.credit_initial = {1.0, 1.0};
  epoch.credit_capacity = {2.0, 2.0};
  auto load = activity(1, Role::Load, HBM0, 6.0);
  load.hbm_bytes[0][0] = 6;
  load.credit_delta = {1.0, 0.0};
  load.canonical_q = 6.0;
  auto compute = activity(2, Role::Compute, COMPUTE_GATE, 2.0);
  compute.work[LOCAL] = 4.0;
  compute.credit_delta = {-1.0, 1.0};
  compute.canonical_q = 4.0;
  auto store = activity(3, Role::Store, HBM1, 6.0);
  store.hbm_bytes[1][1] = 6;
  store.credit_delta = {0.0, -1.0};
  store.canonical_q = 6.0;
  epoch.activities = {load, compute, store};
  return finalize_plan({epoch}, hardware, launch_ns, "scratch-credit-lp");
}

// Curved epoch with opposite cumulative credit: curved_allocation engages
// the ratio fast path (count<=3, zero-bound signed pair rows) before the LP
// fallback and the tangent-rounding certification loop.
std::shared_ptr<const Plan> curved_ratio_plan(const Hardware& hardware,
                                              std::uint64_t launch_ns) {
  Epoch epoch;
  epoch.name = "scratch_curved_ratio";
  epoch.credit_initial = {0.0, 0.0};
  epoch.credit_capacity = {1.0, 1.0};
  auto load = activity(1, Role::Load, HBM0, 4.0);
  load.hbm_bytes[0][0] = 4;
  load.canonical_q = 4.0;
  load.credit_delta = {1.0, 0.0};
  load.credit_curve = {CreditPoint{0.0, {0.0, 0.0}},
                       CreditPoint{0.5, {0.5, 0.0}},
                       CreditPoint{1.0, {1.0, 0.0}}};
  auto compute = activity(2, Role::Compute, COMPUTE_GATE, 2.0);
  compute.work[LOCAL] = 2.0;
  compute.canonical_q = 2.0;
  compute.credit_delta = {-1.0, 0.0};
  compute.credit_curve = {CreditPoint{0.0, {0.0, 0.0}},
                          CreditPoint{0.5, {0.0, 0.0}},
                          CreditPoint{1.0, {-1.0, 0.0}}};
  epoch.activities = {load, compute};
  return finalize_plan({epoch}, hardware, launch_ns, "scratch-curved-ratio");
}

std::vector<std::uint64_t> submit_batch(Runtime& runtime,
                                        const Hardware& hardware,
                                        std::uint64_t base_ns) {
  const std::uint64_t ms = 1000000ULL;
  std::vector<std::uint64_t> handles;
  handles.push_back(runtime.submit(singleton_plan(hardware, 0 * ms),
                                   owner(1, 1, 10, 1), base_ns));
  handles.push_back(runtime.submit(packing_plan(hardware, 50 * ms),
                                   owner(2, 5, 50, 1), base_ns));
  handles.push_back(runtime.submit(packing_plan(hardware, 100 * ms),
                                   owner(3, 5, 50, 1), base_ns));
  handles.push_back(runtime.submit(credit_lp_plan(hardware, 150 * ms),
                                   owner(4, 6, 60, 1), base_ns));
  handles.push_back(runtime.submit(curved_ratio_plan(hardware, 200 * ms),
                                   owner(5, 7, 70, 2), base_ns));
  return handles;
}

// Strip fields that legitimately differ between two identical batches
// (handle numbering, absolute times, and the runtime-global generation) so
// the remaining projection — state, elapsed, credit ledger, activity rates,
// alphas, progress, served bytes — is exactly the arbitration output.
json project_progress(json progress) {
  progress.erase("handle");
  progress.erase("issue_ns");
  progress.erase("ready_ns");
  progress.erase("finish_ns");
  progress.erase("generation");
  return progress;
}

json harvest(Runtime& runtime, const std::vector<std::uint64_t>& handles,
             bool interleave_redundant_allocate) {
  const std::uint64_t ms = 1000000ULL;
  const std::vector<std::uint64_t> checkpoints_ns = {
      10 * ms, 60 * ms, 120 * ms, 180 * ms, 240 * ms,
      400 * ms, 800 * ms, 1500 * ms, 3 * kNsPerSecond, kBatchDrainNs};
  json record = json::array();
  for (std::uint64_t checkpoint : checkpoints_ns) {
    runtime.advance_to(checkpoint);
    if (interleave_redundant_allocate) {
      // An unchanged capacity value re-runs arbitration on the same state:
      // a consecutive allocate() over already-pooled scratch.
      runtime.update_capacity(HBM0, runtime.hardware().rates[HBM0],
                              checkpoint);
    }
    json step = json::array();
    for (std::uint64_t handle : handles)
      step.push_back(project_progress(runtime.query_progress(handle)));
    record.push_back(std::move(step));
  }
  return record;
}

bool runtime_drained_check(const json& record) {
  const json& final_step = record.back();
  bool any_running = false;
  bool any_rate = false;
  for (const auto& step : record) {
    for (const auto& progress : step) {
      for (const auto& activity : progress.at("activities")) {
        if (activity.at("status") == "running") any_running = true;
        if (activity.at("rate").get<double>() > 0.0) any_rate = true;
      }
    }
  }
  for (const auto& progress : final_step)
    if (progress.at("state") != "finished") return false;
  // The checkpoints must actually observe arbitration-published rates.
  return any_running && any_rate;
}

// Curved two-activity epoch alone: count<=3 with a signed zero-bound pair
// row keeps the ratio fast path in play before the LP fallback.
void solo_ratio_reuse_checks(const Hardware& hardware) {
  json reference;
  {
    Runtime runtime(hardware);
    const std::uint64_t handle = runtime.submit(
        curved_ratio_plan(hardware, 0), owner(9, 9, 90, 1), 0);
    reference = harvest(runtime, {handle}, false);
  }
  require(runtime_drained_check(reference),
          "solo ratio reference reaches terminal states");
  {
    Runtime runtime(hardware);
    runtime.submit(curved_ratio_plan(hardware, 0), owner(9, 9, 90, 1), 0);
    runtime.advance_to(kBatchDrainNs);
    const std::uint64_t second = runtime.submit(
        curved_ratio_plan(hardware, 0), owner(9, 9, 90, 1), kBatchDrainNs);
    require(harvest(runtime, {second}, false) == reference,
            "reused-scratch solo ratio batch matches the fresh reference");
  }
  {
    Runtime runtime(hardware);
    const std::uint64_t handle = runtime.submit(
        curved_ratio_plan(hardware, 0), owner(9, 9, 90, 1), 0);
    require(harvest(runtime, {handle}, true) == reference,
            "a consecutive solo ratio arbitration changes nothing");
  }
}

void scratch_reuse_checks() {
  const Hardware hardware = arbitration_hardware();

  json reference;
  {
    Runtime runtime(hardware);
    const auto handles = submit_batch(runtime, hardware, 0);
    reference = harvest(runtime, handles, false);
  }
  require(runtime_drained_check(reference),
          "reference batch reaches terminal states at the final checkpoint");

  // 1. Second identical batch in one Runtime: every arbitration now reuses
  //    the pooled scratch the first batch grew.
  {
    Runtime runtime(hardware);
    submit_batch(runtime, hardware, 0);
    runtime.advance_to(kBatchDrainNs);
    const auto second = submit_batch(runtime, hardware, kBatchDrainNs);
    const json second_record = harvest(runtime, second, false);
    require(second_record == reference,
            "reused-scratch batch matches the fresh-allocation reference");
  }

  // 2. Redundant consecutive allocate() on an unchanged state.
  {
    Runtime runtime(hardware);
    const auto handles = submit_batch(runtime, hardware, 0);
    const json interleaved = harvest(runtime, handles, true);
    require(interleaved == reference,
            "a consecutive arbitration of the same state changes nothing");
  }

  // 3. Repeated batches: the pool high-water mark settles while concurrent
  //    widths vary; outputs must keep matching the reference.
  {
    Runtime runtime(hardware);
    for (int round = 0; round < 12; ++round) {
      const auto handles = submit_batch(runtime, hardware,
                                        round * kBatchDrainNs);
      const json record = harvest(runtime, handles, false);
      require(record == reference,
              "repeated reused-scratch rounds keep the reference outputs");
    }
    const auto observation = runtime.observation_counters();
    const std::uint64_t allocate_calls =
        observation.at("allocate_calls").get<std::uint64_t>();
    require(allocate_calls >= 100,
            "repetition actually exercised many reused arbitrations");
  }

  solo_ratio_reuse_checks(hardware);
}

}  // namespace

int main() {
  coarse_die_test::run_test("arbitration_scratch_reuse_test",
                            scratch_reuse_checks);
  return 0;
}
