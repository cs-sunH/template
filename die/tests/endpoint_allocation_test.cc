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
  require(!plan.at("summary").contains("initial_delay_ns"),
          "legacy endpoint summary omits the absent optional delay field");
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

void test_initial_delay_lifecycle() {
  const auto hw = example_hardware();
  auto delayed_endpoint = endpoint();
  delayed_endpoint["endpoint_id"] = 11;
  delayed_endpoint["transaction_id"] = 11;
  delayed_endpoint["initial_delay_ns"] = 100;

  Runtime runtime(hw);
  const auto handle = runtime.submit_endpoint(
      delayed_endpoint, owner(11, 11, 11), 0);
  const auto initial = runtime.query_progress(handle);
  require(initial.at("ready_ns") == 100 && initial.at("state") == "queued" &&
              initial.at("activities")[0].at("status") == "blocked" &&
              runtime.next_event() == std::optional<std::uint64_t>(100),
          "endpoint initial delay queues service until the exact ready tick");

  runtime.advance_to(40);
  const auto queued = runtime.query_progress(handle);
  require(queued.at("state") == "queued" &&
              queued.at("activities")[0].at("progress") == 0.0 &&
              runtime.counters().at("hbm_busy_ns") == 0.0,
          "pre-ready interval carries no HBM service or busy time");

  const auto snapshot = runtime.export_snapshot();
  auto fork = Runtime::import_snapshot(hw, snapshot);
  const auto& saved_plan = snapshot.at("executions")[0].at("plan");
  require(saved_plan.at("launch_ns") == 100 &&
              saved_plan.at("task").at("endpoint").at("initial_delay_ns") == 100 &&
              saved_plan.at("summary").at("initial_delay_ns") == 100 &&
              fork->query_progress(handle).at("ready_ns") == 100,
          "snapshot preserves endpoint delay in both canonical JSON and Plan");

  runtime.advance_to(99);
  fork->advance_to(99);
  require(runtime.query_progress(handle).at("state") == "queued" &&
              fork->query_progress(handle).at("state") == "queued" &&
              runtime.counters().at("hbm_busy_ns") == 0.0 &&
              fork->counters().at("hbm_busy_ns") == 0.0,
          "snapshot replay remains idle through the nanosecond before readiness");

  runtime.advance_to(100);
  fork->advance_to(100);
  require(runtime.query_progress(handle).at("state") == "running" &&
              fork->query_progress(handle).at("state") == "running" &&
              runtime.query_progress(handle).at("activities")[0].at("rate")
                      .get<double>() > 0.0 &&
              runtime.counters().at("hbm_busy_ns") == 0.0,
          "endpoint begins service at ready time without charging delay as HBM busy");
  runtime.advance_to(101);
  fork->advance_to(101);
  require(runtime.query_progress(handle).at("state") == "finished" &&
              runtime.query_progress(handle) == fork->query_progress(handle) &&
              runtime.counters() == fork->counters(),
          "ready endpoint and snapshot fork finish with identical service");

  auto cancel_endpoint = endpoint();
  cancel_endpoint["endpoint_id"] = 12;
  cancel_endpoint["transaction_id"] = 12;
  cancel_endpoint["physical_bytes"] = 1ULL << 30;
  cancel_endpoint["initial_delay_ns"] = 100;
  Runtime cancelled(hw);
  const auto cancelled_handle = cancelled.submit_endpoint(
      cancel_endpoint, owner(12, 12, 12), 0);
  cancelled.advance_to(99);
  require(cancelled.counters().at("hbm_busy_ns") == 0.0,
          "cancel fixture has no service during its startup delay");
  cancelled.advance_to(100);
  require(cancelled.query_progress(cancelled_handle).at("state") == "running",
          "large endpoint becomes serviceable after its startup delay");
  cancelled.advance_to(101);
  const auto partial = cancelled.query_progress(cancelled_handle);
  cancelled.cancel(cancelled_handle, 101);
  const auto cancelled_progress = cancelled.query_progress(cancelled_handle);
  require(cancelled_progress.at("state") == "cancelled" &&
              cancelled_progress.at("cancelled").get<bool>() &&
              cancelled_progress.at("served").at("hbm_bytes_by_row_direction_exact") !=
                  json::array({json::array({"0", "0"}), json::array({"0", "0"})}) &&
              partial.at("finish_ns").is_null() &&
              cancelled.counters().at("hbm_busy_ns").get<double>() <= 1.0,
          "ready endpoint cancellation retains only partial HBM service, excluding startup delay");
  const auto cancelled_snapshot = cancelled.export_snapshot();
  auto cancelled_fork = Runtime::import_snapshot(hw, cancelled_snapshot);
  require(cancelled_fork->query_progress(cancelled_handle) == cancelled_progress &&
              cancelled_fork->export_snapshot() == cancelled_snapshot,
          "cancelled endpoint snapshot round-trips its delay and terminal state");
  cancelled.destroy_execution(cancelled_handle);
  require(cancelled.counters().at("endpoint_registry_size") == 0 &&
              cancelled.counters().at("transaction_registry_size") == 0,
          "destroying a cancelled delayed endpoint releases endpoint registries");

  for (const auto& invalid : {json(1.5), json(-1), json(true),
                               json("18446744073709551616")}) {
    auto malformed = endpoint();
    malformed["initial_delay_ns"] = invalid;
    bool rejected = false;
    try {
      Runtime invalid_runtime(hw);
      (void)invalid_runtime.submit_endpoint(malformed, owner(13, 13, 13), 0);
    } catch (const std::invalid_argument&) {
      rejected = true;
    }
    require(rejected, "initial_delay_ns rejects values outside uint64 integer form");
  }
}
}  // namespace

int main() {
  return run_test("endpoint_allocation_test", [] {
    test_allocation();
    test_initial_delay_lifecycle();
  });
}
