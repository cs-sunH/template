#include "test_support.hh"

#include <array>
#include <cstdint>
#include <string>

using namespace coarse_die;
using namespace coarse_die_test;

namespace {
constexpr std::uint64_t kNs = 1000000000ULL;

Hardware counter_hardware() {
  std::array<double, resource_count> rates{};
  rates[COMPUTE_GATE] = 4.0;
  rates[HBM0] = 100.0;
  rates[HBM1] = 100.0;
  return manual_hardware(rates);
}

std::shared_ptr<const Plan> comp_plan(const Hardware& hw,
                                      std::uint64_t int4_ops = 1001,
                                      std::uint64_t vec_eqops = 2003) {
  Epoch epoch;
  epoch.name = "typed_comp";
  Activity compute = activity(1, Role::Compute, COMPUTE_GATE, 4.0);
  compute.work[HBM0] = 4.0;
  compute.work[HBM1] = 4.0;
  compute.hbm_bytes = {{{100, 300}, {200, 400}}};
  compute.int4_ops = int4_ops;
  compute.vec_eqops = vec_eqops;
  epoch.activities.push_back(compute);
  auto plan = std::make_shared<Plan>(*manual_plan(hw, {epoch}, "runtime_counter_comp"));
  // These are requested/plan summaries, not proof of executed typed work.
  // Runtime counters must use the actual Activity fields above.
  plan->summary["executed_matrix_ops"] = 99999;
  plan->summary["vec_equivalent_ops"] = 88888;
  plan->fingerprint.clear();
  plan->fingerprint = plan_to_json(*plan).at("fingerprint").get<std::string>();
  return plan;
}

void require_counter_close(const json& counters, const char* category,
                           const char* field, double expected,
                           const std::string& context) {
  require_close(counters.at("served_by_category").at(category).at(field)
                    .get<double>(),
                expected, 1.0e-8, context);
}

void require_hbm(const json& matrix, std::size_t row, std::size_t direction,
                 double expected, const std::string& context) {
  require_close(matrix.at(row).at(direction).get<double>(), expected, 1.0e-8,
                context);
}

json endpoint(std::uint64_t id, std::uint64_t transaction,
              const std::string& owner_kind, const std::string& direction,
              std::uint64_t bytes, const std::string& path) {
  json out{{"endpoint_id", id}, {"transaction_id", transaction},
           {"owner_kind", owner_kind}, {"direction", direction},
           {"physical_bytes", bytes}, {"path_class", path}};
  if (path == "hbm_to_edge" || path == "edge_to_hbm") {
    out["eta_hbm"] = 1.0;
    out["route"] = {{"internal_route", false}};
  }
  return out;
}

void finish_all(Runtime& runtime) {
  for (std::size_t event = 0; event < 128; ++event) {
    const auto next = runtime.next_event();
    if (!next) return;
    runtime.advance_to(*next);
  }
  throw std::runtime_error("runtime counter endpoint fixture exceeded event bound");
}

void test_partial_cancel_retire_and_v3_continuation() {
  const Hardware hw = counter_hardware();
  Runtime runtime(hw);
  const auto handle = runtime.submit(comp_plan(hw), owner(1, 1, 1), 0);
  runtime.advance_to(kNs / 4);

  const auto partial = runtime.counters();
  require_counter_close(partial, "comp", "int4_ops", 250.25,
                        "partial COMP int4 count comes from activity progress");
  require_counter_close(partial, "comp", "vec_eqops", 500.75,
                        "partial COMP vector count comes from activity progress");
  require_hbm(partial.at("served_by_category").at("comp")
                  .at("hbm_served_bytes_by_row_direction"),
              0, 0, 25.0, "partial COMP row0 read bytes");
  require_hbm(partial.at("served_by_category").at("comp")
                  .at("hbm_served_bytes_by_row_direction"),
              0, 1, 75.0, "partial COMP row0 write bytes");
  require(partial.at("served_by_category").at("comp")
              .at("int4_ops_exact_decimal") == "250.25" &&
              partial.at("served_by_category").at("comp")
              .at("vec_eqops_exact_decimal") == "500.75",
          "exact typed totals preserve fractional active work");
  require(partial.at("hardware_raw").at("peak_int4_ops_per_s") ==
              hw.derived.at("hardware_summary").at("peak_int4_ops_per_s") &&
              partial.at("hardware_raw").at("peak_vec_fp32_eq_ops_per_s") ==
              hw.derived.at("hardware_summary").at("peak_vec_fp32_eq_ops_per_s") &&
              partial.at("hardware_raw").at("hbm_row_peak_Bps") ==
              hw.derived.at("hardware_summary").at("hbm_row_peak_Bps"),
          "runtime exposes raw hardware denominators alongside served counts");
  require(partial.at("hbm_served_bytes_by_row_direction") ==
              partial.at("served_by_category").at("comp")
                  .at("hbm_served_bytes_by_row_direction"),
          "COMP category HBM bytes match the existing global served counters");
  require(partial.at("accounting_history_complete") == true &&
              partial.at("typed_totals_history_complete") == true &&
              partial.at("served_category_totals_semantics") ==
                  "complete_lifetime_totals",
          "new runtime accounting starts with complete history");

  const auto snapshot = runtime.export_snapshot();
  require(snapshot.at("schema") == "coarse_die_snapshot_v3" &&
              snapshot.at("snapshot_version") == 3,
          "categorized runtime counters export under explicit v3 schema");
  auto fork = Runtime::import_snapshot(hw, snapshot);
  require(fork->counters() == partial,
          "v3 import restores exact partial lifetime counters");

  runtime.advance_to(kNs / 2);
  fork->advance_to(kNs / 2);
  runtime.cancel(handle, kNs / 2);
  fork->cancel(handle, kNs / 2);
  const auto cancelled = runtime.counters();
  require(cancelled == fork->counters(),
          "v3 import continues partial integration and cancellation identically");
  require_counter_close(cancelled, "comp", "int4_ops", 500.5,
                        "cancelled COMP preserves only actual partial int4 work");
  require_counter_close(cancelled, "comp", "vec_eqops", 1001.5,
                        "cancelled COMP preserves only actual partial vector work");
  require(cancelled.at("served_by_category").at("comp").at("int4_ops") !=
              99999.0 &&
              cancelled.at("served_by_category").at("comp").at("vec_eqops") !=
              88888.0,
          "planned summary totals are never counted as served typed work");

  const auto before_destroy = runtime.counters();
  runtime.destroy_execution(handle);
  fork->destroy_execution(handle);
  require(runtime.counters() == before_destroy &&
              fork->counters() == before_destroy,
          "destroying a cancelled execution does not erase lifetime counters");
}

void test_endpoint_categories_and_retirement() {
  const Hardware hw = counter_hardware();
  Runtime runtime(hw);
  const std::array<json, 6> endpoints = {
      endpoint(10, 10, "comm", "read", 5, "hbm_to_edge"),
      endpoint(11, 11, "comm", "write", 7, "edge_to_hbm"),
      endpoint(12, 12, "restore", "write", 9, "edge_to_hbm"),
      endpoint(13, 13, "pool", "read", 11, "hbm_to_edge"),
      endpoint(14, 14, "die", "write", 13, "edge_to_hbm"),
      endpoint(15, 15, "pool", "read", 0, "remote_pool_only")};
  std::array<std::uint64_t, endpoints.size()> handles{};
  for (std::size_t i = 0; i < endpoints.size(); ++i) {
    handles[i] = runtime.submit_endpoint(
        endpoints[i], owner(20 + i, 20 + i, 20 + i), 0);
  }
  finish_all(runtime);

  auto counters = runtime.counters();
  const auto& categories = counters.at("served_by_category");
  require_hbm(categories.at("comm_send").at("hbm_served_bytes_by_row_direction"),
              0, 0, 3.0, "COMM read maps to send and row0");
  require_hbm(categories.at("comm_send").at("hbm_served_bytes_by_row_direction"),
              1, 0, 2.0, "COMM read maps to send and row1");
  require_hbm(categories.at("comm_recv").at("hbm_served_bytes_by_row_direction"),
              0, 1, 4.0, "COMM write maps to receive and row0");
  require_hbm(categories.at("comm_recv").at("hbm_served_bytes_by_row_direction"),
              1, 1, 3.0, "COMM write maps to receive and row1");
  require_hbm(categories.at("restore").at("hbm_served_bytes_by_row_direction"),
              0, 1, 5.0, "RESTORE writes use the restore category");
  require_hbm(categories.at("pool").at("hbm_served_bytes_by_row_direction"),
              0, 0, 6.0, "POOL source reads use the pool category");
  require_hbm(categories.at("die").at("hbm_served_bytes_by_row_direction"),
              1, 1, 6.0, "DIE writes use the die category");
  require(categories.at("restore").at("int4_ops") == 0.0 &&
              categories.at("pool").at("vec_eqops") == 0.0 &&
              categories.at("comm_send").at("int4_ops") == 0.0,
          "endpoint-only categories do not claim COMP operations");
  const json expected_global = {{9, 16}, {7, 13}};
  require(counters.at("hbm_served_bytes_by_row_direction") == expected_global,
          "category totals preserve the legacy global HBM matrix");
  const auto before_destroy = counters;
  for (const auto handle : handles) runtime.destroy_execution(handle);
  const auto after_destroy = runtime.counters();
  require(after_destroy.at("served_by_category") ==
              before_destroy.at("served_by_category") &&
              after_destroy.at("hbm_served_bytes_by_row_direction") ==
              before_destroy.at("hbm_served_bytes_by_row_direction"),
          "retirement and execution destruction preserve lifetime category totals");
}

void test_v2_migration_and_badimport_atomicity() {
  const Hardware hw = counter_hardware();
  Runtime retired(hw);
  const auto retired_handle = retired.submit(comp_plan(hw, 40, 80), owner(30, 1, 1), 0);
  finish_all(retired);
  retired.destroy_execution(retired_handle);
  const auto v3 = retired.export_snapshot();
  auto old_v2 = v3;
  old_v2["schema"] = "coarse_die_snapshot_v2";
  old_v2["snapshot_version"] = 2;
  old_v2["stats"].erase("served_by_category");
  old_v2["stats"].erase("legacy_unclassified_hbm");
  old_v2["stats"].erase("accounting_history_complete");
  old_v2["stats"].erase("typed_totals_history_complete");

  auto imported = Runtime::import_snapshot(hw, old_v2);
  const auto migrated = imported->counters();
  require(migrated.at("accounting_history_complete") == false &&
              migrated.at("typed_totals_history_complete") == false &&
              migrated.at("legacy_unclassified_typed_totals_unknown") == true &&
              migrated.at("served_category_totals_semantics") ==
                  "known_lower_bound_history_incomplete",
          "v2 import explicitly marks unrecoverable destroyed-work history incomplete");
  require(migrated.at("hbm_served_bytes_by_row_direction") ==
              migrated.at("legacy_unclassified_hbm_bytes_by_row_direction"),
          "v2 HBM is preserved as unclassified history when its execution was destroyed");
  require(migrated.at("served_by_category").at("comp").at("int4_ops") == 0.0 &&
              migrated.at("served_by_category").at("comp").at("vec_eqops") == 0.0,
          "v2 typed totals expose only the known lower bound after destroyed work");

  Epoch epoch;
  epoch.name = "continuing_after_v2";
  auto new_work = activity(99, Role::Compute, COMPUTE_GATE, 4.0);
  new_work.work[HBM0] = 4.0;
  new_work.hbm_bytes[0][0] = 8;
  new_work.int4_ops = 12;
  new_work.vec_eqops = 24;
  epoch.activities.push_back(new_work);
  imported->submit(manual_plan(hw, {epoch}, "v2_continuation"), owner(31, 1, 1),
                   imported->now_ns());
  imported->advance_to(imported->now_ns() + kNs);
  require(imported->counters().at("accounting_history_complete") == false,
          "v2 incompleteness persists while known lower bounds continue accumulating");
  const auto continued_counters = imported->counters();
  const auto continued_matrix = continued_counters.at("hbm_served_bytes_by_row_direction");
  const auto known_matrix = continued_counters.at("served_by_category").at("comp")
                                .at("hbm_served_bytes_by_row_direction");
  const auto unknown_matrix = continued_counters.at(
      "legacy_unclassified_hbm_bytes_by_row_direction");
  for (std::size_t row = 0; row < 2; ++row) {
    for (std::size_t direction = 0; direction < 2; ++direction) {
      require_close(continued_matrix[row][direction].get<double>(),
                    known_matrix[row][direction].get<double>() +
                        unknown_matrix[row][direction].get<double>(),
                    1.0e-8,
                    "v2 unclassified HBM remains conserved after continuation");
    }
  }
  const auto migrated_v3_snapshot = imported->export_snapshot();
  auto migrated_v3 = Runtime::import_snapshot(hw, migrated_v3_snapshot);
  require(migrated_v3->counters() == continued_counters,
          "v2 incomplete accounting remains explicit through v3 re-export/import");

  Runtime active(hw);
  const auto handle = active.submit(comp_plan(hw), owner(32, 1, 1), 0);
  active.advance_to(kNs / 4);
  const auto snapshot = active.export_snapshot();
  const auto before_counters = active.counters();
  const auto before_progress = active.query_progress(handle);
  auto corrupt = snapshot;
  corrupt["stats"]["served_by_category"]["comp"]
         ["hbm_served_bytes_by_row_direction_exact"][0][0] = "999";
  const auto corrupt_before = corrupt;
  bool rejected = false;
  try {
    (void)Runtime::import_snapshot(hw, corrupt);
  } catch (const std::exception&) {
    rejected = true;
  }
  require(rejected, "v3 import rejects nonconserving categorized HBM totals");
  require(active.counters() == before_counters &&
              active.query_progress(handle) == before_progress &&
              corrupt == corrupt_before,
          "failed import leaves its source runtime and supplied snapshot unchanged");
}

void test_runtime_counters() {
  test_partial_cancel_retire_and_v3_continuation();
  test_endpoint_categories_and_retirement();
  test_v2_migration_and_badimport_atomicity();
}
}  // namespace

int main() { return run_test("runtime_counters_test", test_runtime_counters); }
