#include "test_support.hh"
#include <limits>

using namespace coarse_die;
using namespace coarse_die_test;

namespace {
void body() {
  std::array<double, resource_count> capacities{};
  capacities[HBM0] = 100;
  const auto hardware = manual_hardware(capacities);
  Epoch epoch;
  epoch.name = "load";
  auto load = activity(1, Role::Load, HBM0, 100);
  load.hbm_bytes[0][0] = 100;
  load.canonical_q = 100;
  epoch.activities = {load};
  const auto valid = manual_plan(hardware, {epoch});
  // Legal finite positive input whose work/Q is beyond solver precision.
  // Runtime must report numerical_failure, with no hidden live execution.
  epoch.activities[0].canonical_q = std::numeric_limits<double>::denorm_min();
  const auto pathological = manual_plan(hardware, {epoch}, "precision_failure");
  Runtime runtime(hardware);
  const auto handle = runtime.submit(valid, owner(1, 1, 1), 0);
  const auto before = runtime.export_snapshot();
  bool rejected = false;
  try { (void)runtime.submit(pathological, owner(2, 2, 2), 0); }
  catch (const std::runtime_error& error) {
    rejected = std::string(error.what()).find("numerical_failure:") == 0;
  }
  require(rejected, "precision failure must reach the real allocator");
  require(runtime.export_snapshot() == before,
          "failed submit at the current tick preserves the complete world");

  Runtime reference(hardware);
  reference.submit(valid, owner(1, 1, 1), 0);
  reference.advance_to(400000000);
  rejected = false;
  try { (void)runtime.submit(pathological, owner(2, 2, 2), 400000000); }
  catch (const std::runtime_error&) { rejected = true; }
  require(rejected && runtime.export_snapshot() == reference.export_snapshot(),
          "failed arrival commits only prior service up to its requested tick");
  require(runtime.execution_handles() == std::vector<std::uint64_t>{handle},
          "failed submit leaves no handle requiring caller cleanup");
  require(runtime.next_event() == reference.next_event(),
          "old rates remain usable without re-solving on an exception path");
  runtime.advance_to(1000000000);
  require(runtime.drained() && runtime.terminal_finish_ns(handle) == 1000000000,
          "pre-existing execution completes after failed arrivals");
  require(runtime.query_progress(handle).at("hbm_served_bytes_exact").at(0).at(0) == "100",
          "failed submissions add no physical payload");
  // A later valid arrival reuses only the never-published handle and works.
  const auto second = runtime.submit(valid, owner(3, 3, 3), 1000000000);
  require(second == handle + 1, "failed submit does not consume published IDs");
  runtime.advance_to(2000000000);
  require(runtime.terminal_finish_ns(second) == 2000000000,
          "runtime accepts valid work after precision rejection");

  // The pathological plan alone uses the exact one-variable allocator. An
  // endpoint arrival forces the general solver and deterministically fails.
  // Its registry reservations must roll back together with the execution.
  Runtime endpoints(hardware);
  const auto original = endpoints.submit(pathological, owner(4, 4, 4), 0);
  const auto before_endpoint = endpoints.export_snapshot();
  const json endpoint{{"endpoint_id", 99}, {"transaction_id", 99},
      {"owner_kind", "comm"}, {"direction", "read"},
      {"physical_bytes", 100}, {"row_fraction", {1.0, 0.0}},
      {"path_class", "hbm_to_edge"}, {"eta_hbm", 1.0},
      {"route", {{"internal_route", false}}}};
  rejected = false;
  try { (void)endpoints.submit_endpoint(endpoint, owner(5, 5, 5), 0); }
  catch (const std::runtime_error&) { rejected = true; }
  require(rejected && endpoints.export_snapshot() == before_endpoint,
          "failed endpoint arrival restores both ownership registries");
  endpoints.cancel(original, 0);
  endpoints.destroy_execution(original);
  const auto accepted = endpoints.submit_endpoint(endpoint, owner(5, 5, 5), 0);
  endpoints.advance_to(1000000000);
  require(endpoints.terminal_finish_ns(accepted) == 1000000000,
          "rejected endpoint and transaction IDs remain reusable");
  endpoints.destroy_execution(accepted);
  require(endpoints.drained(), "endpoint lifetime drains after registry rollback");
}
}

int main() { return run_test("runtime_submit_transaction_test", body); }
