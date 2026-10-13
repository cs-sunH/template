#include "test_support.hh"

using namespace coarse_die;
using namespace coarse_die_test;

namespace {
json endpoint() {
  return {{"endpoint_id", 1}, {"transaction_id", 1}, {"owner_kind", "comm"},
          {"direction", "read"}, {"physical_bytes", 5},
          {"path_class", "hbm_to_edge"}, {"eta_hbm", 1.0},
          {"route", {{"internal_route", false}}}};
}

void finish(Runtime& runtime) {
  while (const auto next = runtime.next_event()) runtime.advance_to(*next);
}

void test_allocation() {
  const auto hw = example_hardware();
  Runtime runtime(hw);
  const auto handle = runtime.submit_endpoint(endpoint(), owner(1, 1, 1), 0);
  const auto snapshot = runtime.export_snapshot();
  const auto& plan = snapshot.at("executions")[0].at("plan");
  require(plan.at("summary").at("allocation_evidence") == "uniform_default",
          "implicit row placement carries evidence");
  auto fork = Runtime::import_snapshot(hw, snapshot);
  finish(runtime);
  finish(*fork);
  require(runtime.query_progress(handle) == fork->query_progress(handle),
          "odd default allocation survives snapshot replay");
  const auto payload = runtime.query_progress(handle).at("hbm_served_bytes");
  require(payload[0][0] == 3.0 && payload[1][0] == 2.0,
          "default odd byte belongs to row0");

  auto advisory_summary = snapshot;
  auto& advisory_plan = advisory_summary["executions"][0]["plan"];
  advisory_plan["summary"]["kind"] = "advisory_kind";
  advisory_plan.erase("fingerprint");
  advisory_plan["fingerprint"] = stable_hash(advisory_plan.dump());
  auto classified = Runtime::import_snapshot(hw, advisory_summary);
  finish(*classified);
  require(classified->counters().at("served_by_category").at("comm_send")
              .at("hbm_served_bytes_by_row_direction")[0][0] == 3.0 &&
              classified->counters().at("served_by_category").at("comp")
              .at("hbm_served_bytes_by_row_direction")[0][0] == 0.0,
          "canonical endpoint owner classifies service independently of advisory summary");

  Runtime explicit_runtime(hw);
  auto explicit_endpoint = endpoint();
  explicit_endpoint["row_fraction"] = {0.5, 0.5};
  const auto explicit_handle = explicit_runtime.submit_endpoint(
      explicit_endpoint, owner(1, 1, 1), 0);
  finish(explicit_runtime);
  const auto explicit_payload = explicit_runtime.query_progress(explicit_handle)
                                    .at("hbm_served_bytes");
  require(explicit_payload[0][0] == 2.0 && explicit_payload[1][0] == 3.0,
          "explicit fractions retain their floor-and-remainder convention");

  // Reconstruct an old v2 plan's implicit floor split. Its canonical payload
  // remains frozen rather than being changed by the new default resolver.
  auto legacy = snapshot;
  auto& old_plan = legacy["executions"][0]["plan"];
  old_plan["summary"].erase("allocation_evidence");
  old_plan["epochs"][0]["activities"][0]["hbm_bytes"] = {{2, 0}, {3, 0}};
  old_plan["epochs"][0]["activities"][0]["work"][HBM0] = 2.0;
  old_plan["epochs"][0]["activities"][0]["work"][HBM1] = 3.0;
  old_plan.erase("fingerprint");
  old_plan["fingerprint"] = stable_hash(old_plan.dump());
  // Rates are checked against a fresh allocation during import.
  legacy["executions"][0]["activities"][0]["rate"] = 0.0;
  legacy["executions"][0]["activities"][0]["alpha"] = 0.0;
  auto old_fork = Runtime::import_snapshot(hw, legacy);
  finish(*old_fork);
  const auto old_payload = old_fork->query_progress(handle).at("hbm_served_bytes");
  require(old_payload[0][0] == 2.0 && old_payload[1][0] == 3.0,
          "legacy implicit split imports without rewriting its payload");
}
}  // namespace

int main() { return run_test("endpoint_allocation_test", test_allocation); }
