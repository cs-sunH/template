#include "test_support.hh"

#include <array>
#include <cstdint>
#include <memory>
#include <vector>

using namespace coarse_die;
using namespace coarse_die_test;

namespace {
constexpr std::uint64_t kNsPerSecond = 1000000000ULL;

std::shared_ptr<const Plan> pipeline(const Hardware& hardware, int boundary) {
  Epoch epoch;
  epoch.name = "symmetric_owner_pipeline";
  const double credit = boundary == 0 ? 1 : boundary == 1 ? 0 : 2;
  epoch.credit_initial = {credit, credit};
  epoch.credit_capacity = {2, 2};
  auto load = activity(1, Role::Load, HBM0, 1000);
  load.work[HBM1] = 1000;
  load.hbm_bytes[0][0] = 500;
  load.hbm_bytes[1][0] = 500;
  load.credit_delta = {1, 0};
  auto compute = activity(2, Role::Compute, COMPUTE_GATE, 1);
  compute.work[LOCAL] = 2000;
  compute.credit_delta = {-1, 1};
  auto store = activity(3, Role::Store, HBM0, 1000);
  store.work[HBM1] = 1000;
  store.hbm_bytes[0][1] = 500;
  store.hbm_bytes[1][1] = 500;
  store.credit_delta = {0, -1};
  load.canonical_q = canonical_basis(load);
  compute.canonical_q = canonical_basis(compute);
  store.canonical_q = canonical_basis(store);
  epoch.activities = {load, compute, store};
  return manual_plan(hardware, {epoch});
}

std::shared_ptr<const Plan> phase_offset_plan(const Hardware& hardware,
                                              std::uint64_t launch_ns,
                                              const char* name) {
  Epoch epoch;
  epoch.name = "same_lp_different_progress_phase";
  epoch.credit_initial = {0.5, 0.5};
  epoch.credit_capacity = {1.0, 1.0};
  auto load = activity(1, Role::Load, HBM0, 3.0);
  load.hbm_bytes[0][0] = 3;
  load.canonical_q = 3.0;
  epoch.activities.push_back(load);

  auto plan = std::make_shared<Plan>(*manual_plan(hardware, {epoch}, name));
  plan->launch_ns = launch_ns;
  plan->fingerprint.clear();
  plan->fingerprint =
      plan_to_json(*plan).at("fingerprint").get<std::string>();
  return plan;
}

void rate(const Runtime& runtime, std::uint64_t handle,
          double memory_rate, double compute_rate) {
  const auto progress = runtime.query_progress(handle);
  const auto& activities = progress.at("activities");
  for (std::size_t stage = 0; stage < 3; ++stage) {
    require_close(activities.at(stage).at("rate").get<double>(),
                  stage == 1 ? compute_rate : memory_rate, 1e-11,
                  "rates match the independent symmetric max-min solution");
    require_close(activities.at(stage).at("alpha").get<double>(),
                  stage == 1 ? .5 : .25, 1e-12,
                  "each owner keeps its original within-owner Q share");
  }
}

void body() {
  std::array<double, resource_count> capacities{};
  capacities[COMPUTE_GATE] = 1;
  capacities[LOCAL] = 2000;
  capacities[HBM0] = capacities[HBM1] = 1000;
  const auto hardware = manual_hardware(capacities);
  for (const unsigned owners : {1u, 2u, 8u, 32u}) {
    for (const int boundary : {0, 1, 2}) {
      Runtime runtime(hardware);
      std::vector<std::uint64_t> handles;
      const auto plan = pipeline(hardware, boundary);
      for (unsigned i = 0; i < owners; ++i)
        handles.push_back(runtime.submit(plan, owner(i+1, i+1, i+1), 0));
      // Q/alpha=4000 for all three stages. The two equal HBM rows force
      // vL=vS=1/(2J). Interior compute reaches 1/J; empty/full signed credit
      // forces its rate to the same 1/(2J). These are closed LP optima.
      for (auto handle : handles)
        rate(runtime, handle, .5/owners,
             boundary == 0 ? 1.0/owners : .5/owners);
      const auto snapshot = runtime.export_snapshot();
      const auto fork = Runtime::import_snapshot(hardware, snapshot);
      for (auto handle : handles)
        require(fork->query_progress(handle) == runtime.query_progress(handle),
                "snapshot reconstruction preserves every distinct owner");
      runtime.advance_to(2ULL * owners * 1000000000ULL);
      require(runtime.drained(), "symmetric owners drain without pivot failure");
      for (auto handle : handles) {
        const auto served = runtime.query_progress(handle);
        require(served.at("served").at("hbm_bytes_by_row_direction_exact") ==
                    json::array({json::array({"500", "500"}),
                                 json::array({"500", "500"})}),
                "each mirrored activity retains its own physical ledger");
      }
    }
  }

  Runtime mixed(hardware);
  std::vector<std::uint64_t> handles;
  for (unsigned i = 0; i < 32; ++i)
    handles.push_back(mixed.submit(pipeline(hardware, i < 16 ? 1 : 0),
                                  owner(i+1, i+1, i+1), 0));
  // Empty-credit owners consume 16*(1/64)=1/4 compute capacity. The sixteen
  // interior owners share the remaining 3/4, retaining their own flow bound.
  for (unsigned i = 0; i < 32; ++i)
    rate(mixed, handles[i], 1.0/64, i < 16 ? 1.0/64 : 3.0/64);
  mixed.cancel(handles.front(), 0);
  for (unsigned i = 1; i < 32; ++i)
    rate(mixed, handles[i], 1.0/62,
         i < 16 ? 1.0/62 : (1.0 - 15.0/62)/16);

  // A starts one second earlier and reaches 1/3 progress before B joins.
  // While both are active they share the LP rate, then A exits at 5s and B
  // receives its single-owner rate for the final second. These exact event
  // times protect busy/shared accounting across a member-set change.
  std::array<double, resource_count> phase_capacities{};
  phase_capacities[HBM0] = 1.0;
  const auto phase_hardware = manual_hardware(phase_capacities);
  Runtime phase_offset(phase_hardware);
  const auto early = phase_offset.submit(
      phase_offset_plan(phase_hardware, 0, "phase-early"), owner(1, 1, 1), 0);
  const auto late = phase_offset.submit(
      phase_offset_plan(phase_hardware, kNsPerSecond, "phase-late"),
      owner(2, 2, 2), 0);

  phase_offset.advance_to(kNsPerSecond);
  const auto early_at_join = phase_offset.query_progress(early);
  const auto late_at_join = phase_offset.query_progress(late);
  require_close(early_at_join.at("activities").at(0).at("progress")
                    .get<double>(),
                1.0 / 3.0, 1e-12,
                "early owner progresses before the second owner becomes ready");
  require_close(late_at_join.at("activities").at(0).at("progress")
                    .get<double>(),
                0.0, 1e-12, "late owner joins at zero progress");
  for (const auto handle : {early, late}) {
    const auto allocation = phase_offset.query_progress(handle)
                                .at("activities").at(0);
    require_close(allocation.at("rate").get<double>(), 1.0 / 6.0, 1e-12,
                  "phase-offset mirrors share the two-owner service rate");
    require_close(allocation.at("alpha").get<double>(), 1.0, 1e-12,
                  "each single-flow owner keeps alpha one");
  }

  phase_offset.advance_to(5 * kNsPerSecond);
  require(phase_offset.query_progress(early).at("terminal").get<bool>(),
          "the earlier progress phase exits at its own completion event");
  const auto late_after_change = phase_offset.query_progress(late)
                                     .at("activities").at(0);
  require_close(late_after_change.at("progress").get<double>(), 2.0 / 3.0,
                1e-12, "surviving owner retains its integrated service");
  require_close(late_after_change.at("rate").get<double>(), 1.0 / 3.0,
                1e-12, "surviving owner receives its single-owner rate");

  phase_offset.advance_to(6 * kNsPerSecond);
  require(phase_offset.drained(),
          "both phase-offset executions finish after member change");
  require(phase_offset.terminal_finish_ns(early) == 5 * kNsPerSecond &&
              phase_offset.terminal_finish_ns(late) == 6 * kNsPerSecond,
          "member completion preserves the one-second phase offset");
  const auto phase_counters = phase_offset.counters();
  require(phase_counters.at("hbm_busy_ns_exact") == "6000000000" &&
              phase_counters.at("hbm_shared_ns_exact") == "4000000000",
          "phase-offset member change yields six-second busy and four-second shared unions");
}
}

int main() { return run_test("runtime_owner_symmetry_test", body); }
