#include "test_support.hh"

using namespace coarse_die;
using namespace coarse_die_test;

namespace {
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
}
}

int main() { return run_test("runtime_owner_symmetry_test", body); }
