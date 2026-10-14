#include "test_support.hh"

#include <array>
#include <cstdint>
#include <iostream>
#include <memory>

using namespace coarse_die;
using namespace coarse_die_test;

namespace {
constexpr std::uint64_t kNs = 1000000000ULL;

std::shared_ptr<const Plan> delayed_plan(const Hardware& hardware,
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
  auto mutable_plan = std::make_shared<Plan>(
      *manual_plan(hardware, {epoch}, name));
  mutable_plan->launch_ns = launch_ns;
  mutable_plan->fingerprint =
      plan_to_json(*mutable_plan).at("fingerprint").get<std::string>();
  return mutable_plan;
}

void run() {
  std::array<double, resource_count> capacities{};
  capacities[HBM0] = 1.0;
  const Hardware hardware = manual_hardware(capacities);
  Runtime runtime(hardware);
  const auto early = runtime.submit(delayed_plan(hardware, 0, "early"),
                                    owner(1, 1, 1), 0);
  const auto late = runtime.submit(delayed_plan(hardware, kNs, "late"),
                                   owner(2, 2, 2), 0);

  runtime.advance_to(kNs);
  const auto early_at_join = runtime.query_progress(early);
  const auto late_at_join = runtime.query_progress(late);
  const auto& early_activity = early_at_join.at("activities").at(0);
  const auto& late_activity = late_at_join.at("activities").at(0);
  require_close(early_activity.at("progress").get<double>(), 1.0 / 3.0,
                1e-12, "early owner progressed before the second owner became ready");
  require_close(late_activity.at("progress").get<double>(), 0.0,
                1e-12, "late owner joins at zero progress");
  require_close(early_activity.at("rate").get<double>(), 1.0 / 6.0,
                1e-12, "symmetry reduction publishes the shared rate to early member");
  require_close(late_activity.at("rate").get<double>(), 1.0 / 6.0,
                1e-12, "symmetry reduction publishes the shared rate to late member");
  require_close(early_activity.at("alpha").get<double>(), 1.0,
                1e-12, "single-flow owner keeps alpha one");
  require_close(late_activity.at("alpha").get<double>(), 1.0,
                1e-12, "mirrored owner keeps its own alpha one");

  runtime.advance_to(5 * kNs);
  const auto early_finish = runtime.query_progress(early);
  const auto late_after_member_change = runtime.query_progress(late);
  require(early_finish.at("terminal").get<bool>(),
          "the earlier progress phase leaves at its own completion event");
  require_close(late_after_member_change.at("activities").at(0)
                    .at("progress").get<double>(),
                2.0 / 3.0, 1e-12,
                "the surviving mirror retains only its integrated service");
  require_close(late_after_member_change.at("activities").at(0)
                    .at("rate").get<double>(),
                1.0 / 3.0, 1e-12,
                "allocation is rebuilt for the surviving owner at the same tick");

  runtime.advance_to(7 * kNs);
  require(runtime.drained(), "both executions finish after membership change");
  require(runtime.terminal_finish_ns(early) == 5 * kNs &&
              runtime.terminal_finish_ns(late) == 7 * kNs,
          "member completion times preserve the phase offset");
  const auto early_done = runtime.query_progress(early);
  const auto late_done = runtime.query_progress(late);
  require(early_done.at("served").at("hbm_bytes_by_row_direction_exact") ==
              json::array({json::array({"3", "0"}), json::array({"0", "0"})}),
          "early activity keeps its complete physical payload");
  require(late_done.at("served").at("hbm_bytes_by_row_direction_exact") ==
              json::array({json::array({"3", "0"}), json::array({"0", "0"})}),
          "late activity keeps its complete physical payload");

  const auto counters = runtime.counters();
  require(counters.at("hbm_busy_ns_exact") == "7000000000",
          "HBM busy union spans the continuous seven-second service interval");
  require(counters.at("hbm_shared_ns_exact") == "4000000000",
          "shared HBM busy time is exactly the four-second mirrored overlap");
  require(counters.at("row_busy_ns_exact").at(0) == "7000000000" &&
              counters.at("row_shared_ns_exact").at(0) == "4000000000",
          "row-level busy and shared unions remain conserved at member change");
  std::cout << "PASS phase-offset mirror member-change service/busy conservation\n";
}
}  // namespace

int main() { return run_test("member_change_runtime", run); }
