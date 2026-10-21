#include "test_support.hh"

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

using namespace coarse_die;
using namespace coarse_die_test;

namespace {
constexpr std::uint64_t kNsPerSecond = 1000000000ULL;

// One HBM row load per plan: every integrate sweep of the runtime observes
// exactly the submitting execution's owner tuple on the chosen row, so the
// shared-row accumulators isolate the owner identity comparison.
std::shared_ptr<const Plan> row_load_plan(const Hardware& hardware,
                                          std::size_t row,
                                          std::uint64_t launch_ns,
                                          std::uint64_t id,
                                          const char* name) {
  Epoch epoch;
  epoch.name = "owner_dedup_row_load";
  epoch.credit_initial = {0.5, 0.5};
  epoch.credit_capacity = {1.0, 1.0};
  auto load = activity(id, Role::Load, row == 0 ? HBM0 : HBM1, 3.0);
  load.hbm_bytes[row][0] = 3;
  load.canonical_q = 3.0;
  epoch.activities.push_back(load);
  auto plan = std::make_shared<Plan>(*manual_plan(hardware, {epoch}, name));
  plan->launch_ns = launch_ns;
  plan->fingerprint.clear();
  plan->fingerprint =
      plan_to_json(*plan).at("fingerprint").get<std::string>();
  return plan;
}

Hardware row_hardware(std::size_t row) {
  std::array<double, resource_count> capacities{};
  capacities[row == 0 ? HBM0 : HBM1] = 1.0;
  return manual_hardware(capacities);
}

// Two simultaneously active row loads for one second. counters() derives
// row_shared_owners/active_owners from Owner::key() strings while
// row_shared_ns/hbm_shared_ns come from integrate's structured tuple dedup,
// so agreement between the two views pins both sides of the identity
// definition: exactly (node_id, execution_id, generation), request_id out.
void check_two_active_owners(const Owner& first, const Owner& second,
                             std::size_t row, bool expect_shared) {
  const auto hardware = row_hardware(row);
  Runtime runtime(hardware);
  runtime.submit(row_load_plan(hardware, row, 0, 1, "dedup-first"), first, 0);
  runtime.submit(row_load_plan(hardware, row, 0, 2, "dedup-second"), second, 0);
  runtime.advance_to(kNsPerSecond);
  const auto counters = runtime.counters();
  const json expected_owners = row == 0
      ? json::array({expect_shared ? 2u : 1u, 0u})
      : json::array({0u, expect_shared ? 2u : 1u});
  require(counters.at("row_shared_owners") == expected_owners,
          "row_shared_owners reports the distinct owner count per row");
  require(counters.at("active_owners") ==
              static_cast<std::size_t>(expect_shared ? 2 : 1),
          "active owner count follows the same key identity");
  require(counters.at("hbm_shared_ns_exact") ==
              json(expect_shared ? "1000000000" : "0"),
          "hbm shared time matches the two-owner overlap exactly");
  const auto& row_shared = counters.at("row_shared_ns_exact");
  const std::size_t other = 1 - row;
  require(row_shared.is_array() && row_shared.size() == 2 &&
              row_shared.at(row) ==
                  json(expect_shared ? "1000000000" : "0") &&
              row_shared.at(other) == json("0"),
          "shared time lands on the loaded row only");
  const json expected_busy = row == 0
      ? json::array({"1000000000", "0"})
      : json::array({"0", "1000000000"});
  require(counters.at("hbm_busy_ns_exact") == json("1000000000") &&
              counters.at("row_busy_ns_exact") == expected_busy,
          "busy accumulators stay independent of owner identity");
}

void same_owner_tuple_with_distinct_requests_is_not_shared() {
  // request_id is tracing metadata: two executions of one owner tuple must
  // not mint a second owner in integrate's dedup any more than in
  // Owner::key() used by the solvers.
  check_two_active_owners(owner(1, 4, 4, 1), owner(777, 4, 4, 1), 0, false);
  check_two_active_owners(owner(5, 9, 9, 3), owner(6, 9, 9, 3), 1, false);
}

void one_execution_with_repeated_row_activities_is_not_shared() {
  const auto hardware = row_hardware(0);
  Epoch epoch;
  epoch.name = "owner_dedup_same_execution_rows";
  epoch.credit_initial = {0.5, 0.5};
  epoch.credit_capacity = {1.0, 1.0};
  auto first = activity(1, Role::Load, HBM0, 3.0);
  auto second = activity(2, Role::Load, HBM0, 3.0);
  first.hbm_bytes[0][0] = 3;
  second.hbm_bytes[0][0] = 3;
  first.canonical_q = 3.0;
  second.canonical_q = 3.0;
  epoch.activities = {first, second};
  Runtime runtime(hardware);
  runtime.submit(manual_plan(hardware, {epoch}, "dedup-same-execution"),
                 owner(2, 8, 8, 1), 0);
  runtime.advance_to(kNsPerSecond);
  const auto counters = runtime.counters();
  require(counters.at("row_shared_owners") == json::array({1u, 0u}) &&
              counters.at("active_owners") == 1,
          "one execution observing its owner twice stays one owner");
  require(counters.at("hbm_shared_ns_exact") == json("0") &&
              counters.at("row_shared_ns_exact") ==
                  json::array({"0", "0"}),
          "repeated same-tuple observations within one execution never share");
  require(counters.at("hbm_busy_ns_exact") == json("1000000000"),
          "the row stays busy while the owner is single");
}

void each_tuple_dimension_alone_triggers_sharing() {
  // node_id differs, execution_id and generation equal.
  check_two_active_owners(owner(1, 5, 5, 1), owner(1, 6, 5, 1), 0, true);
  // execution_id differs, node_id and generation equal.
  check_two_active_owners(owner(1, 5, 5, 1), owner(1, 5, 6, 1), 0, true);
  // generation differs, node_id and execution_id equal.
  check_two_active_owners(owner(1, 5, 5, 1), owner(1, 5, 5, 2), 0, true);
  // Same dimensions on the second HBM row.
  check_two_active_owners(owner(1, 5, 5, 1), owner(1, 6, 5, 1), 1, true);
  check_two_active_owners(owner(1, 5, 5, 1), owner(1, 5, 6, 1), 1, true);
  check_two_active_owners(owner(1, 5, 5, 1), owner(1, 5, 5, 2), 1, true);
}

// Numeric invariance against the pre-change implementation: staggered
// launches give exact busy/shared unions. Splitting or merging the same
// canonical weight across requests never changes the schedule, so the
// same-tuple member change must reproduce the different-owner finish times
// while reporting zero shared time; the different-owner variant reports the
// four-second overlap union.
void phase_offset_shared_accounting() {
  const auto run = [](const Owner& early_owner, const Owner& late_owner,
                      bool expect_shared) {
    const auto hardware = row_hardware(0);
    Runtime runtime(hardware);
    const auto early = runtime.submit(
        row_load_plan(hardware, 0, 0, 1, "dedup-phase-early"), early_owner, 0);
    const auto late = runtime.submit(
        row_load_plan(hardware, 0, kNsPerSecond, 2, "dedup-phase-late"),
        late_owner, 0);
    runtime.advance_to(5 * kNsPerSecond);
    require(runtime.query_progress(early).at("terminal").get<bool>(),
            "early member exits at its own completion event");
    runtime.advance_to(6 * kNsPerSecond);
    require(runtime.query_progress(late).at("terminal").get<bool>() &&
              runtime.drained(),
            "late member drains one second after the early exit");
    const auto counters = runtime.counters();
    require(counters.at("hbm_busy_ns_exact") == json("6000000000") &&
                counters.at("row_busy_ns_exact") ==
                    json::array({"6000000000", "0"}),
            "six-second busy union is independent of owner identity");
    require(counters.at("hbm_shared_ns_exact") ==
                json(expect_shared ? "4000000000" : "0"),
            "shared union reports the exact member-set overlap");
    require(counters.at("row_shared_ns_exact") ==
                json::array({expect_shared ? "4000000000" : "0", "0"}),
            "row shared union matches the hbm union on row zero");
  };
  run(owner(1, 3, 3, 1), owner(901, 3, 3, 1), false);
  run(owner(1, 3, 3, 1), owner(1, 3, 4, 1), true);
}

void body() {
  same_owner_tuple_with_distinct_requests_is_not_shared();
  one_execution_with_repeated_row_activities_is_not_shared();
  each_tuple_dimension_alone_triggers_sharing();
  phase_offset_shared_accounting();
}
}  // namespace

int main() { return run_test("owner_dedup_integrate_test", body); }
