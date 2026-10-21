#include "test_support.hh"

#include <array>
#include <cstdint>

using namespace coarse_die;
using namespace coarse_die_test;

namespace {
constexpr std::uint64_t kNs = 1000000000ULL;

Hardware rates(double compute, double hbm0, double hbm1 = 0.0) {
  std::array<double, resource_count> r{};
  r[COMPUTE_GATE] = compute;
  r[HBM0] = hbm0;
  r[HBM1] = hbm1;
  return manual_hardware(r);
}

nlohmann::json run_until_terminal(Runtime& runtime, std::uint64_t handle) {
  for (std::size_t event = 0; event < 128; ++event) {
    const auto p = runtime.query_progress(handle);
    if (p.at("terminal").get<bool>()) return p;
    const auto next = runtime.next_event();
    require(next.has_value(), "live work must have a finite next event");
    require(*next > runtime.now_ns(), "runtime event must advance absolute time");
    runtime.advance_to(*next);
  }
  throw std::runtime_error("runtime did not complete within 128 events");
}

void equal_owner_fairness_and_split_invariance() {
  const auto hw = rates(0.0, 100.0);
  Runtime runtime(hw);
  Epoch epoch;
  epoch.name = "one_hbm_phase";
  epoch.activities.push_back(activity(1, Role::Load, HBM0, 100.0));
  const auto plan = manual_plan(hw, {epoch}, "equal_owner_a");
  Epoch epoch_b;
  epoch_b.name = "one_hbm_phase";
  epoch_b.activities.push_back(activity(2, Role::Load, HBM0, 100.0));
  const auto plan_b = manual_plan(hw, {epoch_b}, "equal_owner_b");
  const auto ha = runtime.submit(plan, owner(1, 1, 1), 0);
  const auto hb = runtime.submit(plan_b, owner(2, 2, 2), 0);
  const auto before = runtime.query_progress(ha);
  require_close(before.at("activities").at(0).at("rate").get<double>(),
                0.5, 1e-12, "equal owners receive equal HBM progress rates");
  const auto pa = run_until_terminal(runtime, ha);
  const auto pb = runtime.query_progress(hb);
  require(pa.at("finish_ns").get<std::uint64_t>() == 2 * kNs &&
              pb.at("finish_ns").get<std::uint64_t>() == 2 * kNs,
          "two equal HBM owners each finish 100 B in two seconds at 100 B/s");
  require_close(runtime.counters().at("hbm_shared_ns").get<double>(),
                2.0 * kNs, 1.0, "shared HBM time tracks actual overlap");

  Runtime split_runtime(hw);
  Epoch split_epoch;
  split_epoch.name = "split_owner";
  auto a0 = activity(10, Role::Load, HBM0, 50.0);
  auto a1 = activity(11, Role::Load, HBM0, 50.0);
  a0.canonical_q = 0.5;
  a1.canonical_q = 0.5;
  split_epoch.activities = {a0, a1};
  const auto split_plan = manual_plan(hw, {split_epoch}, "split_owner");
  Epoch single_epoch;
  single_epoch.name = "single_owner";
  single_epoch.activities.push_back(activity(12, Role::Load, HBM0, 100.0));
  const auto single_plan = manual_plan(hw, {single_epoch}, "single_owner");
  const auto hs = split_runtime.submit(split_plan, owner(3, 1, 1), 0);
  const auto ho = split_runtime.submit(single_plan, owner(4, 2, 2), 0);
  const auto split_progress = split_runtime.query_progress(hs);
  require_close(split_progress.at("activities").at(0).at("rate").get<double>(),
                0.5, 1e-12, "split activity receives half its normalized owner share");
  const auto split_done = run_until_terminal(split_runtime, hs);
  const auto single_done = split_runtime.query_progress(ho);
  require(split_done.at("finish_ns").get<std::uint64_t>() == 2 * kNs &&
              single_done.at("finish_ns").get<std::uint64_t>() == 2 * kNs,
          "splitting the same canonical owner weight does not increase its aggregate HBM share");
}

void independent_resources_epoch_barrier_and_repeat_audit() {
  const auto hw = rates(1.0, 1.0);
  Epoch first;
  first.name = "C1_plus_L2";
  first.repeat_count = 2;
  first.activities = {activity(1, Role::Compute, COMPUTE_GATE, 5.0),
                      activity(2, Role::Load, HBM0, 1.0)};
  Epoch second;
  second.name = "C2_plus_S1";
  second.repeat_count = 2;
  second.activities = {activity(3, Role::Compute, COMPUTE_GATE, 1.0),
                       activity(4, Role::Store, HBM0, 10.0)};
  const auto plan = manual_plan(hw, {first, second}, "W2_15s");
  Runtime runtime(hw);
  const auto handle = runtime.submit(plan, owner(10, 1, 1), 0);
  const auto end = run_until_terminal(runtime, handle);
  require(end.at("finish_ns").get<std::uint64_t>() == 15 * kNs,
          "independent compute/HBM service with an epoch barrier takes max(5,1)+max(1,10)=15 s");
  require(end.at("activities").size() == 4,
          "progress preserves one activity per declared epoch activity");
  for (const auto& item : end.at("activities"))
    require(item.at("done").get<bool>(),
            "historical activities remain queryable after later epochs finish");
  const auto counters = runtime.counters();
  require(counters.at("internal_events").get<std::uint64_t>() < 20 &&
              counters.at("allocation_updates").get<std::uint64_t>() < 20,
          "repeat_count is audit metadata and does not create per-wave runtime events");

  Epoch stable;
  stable.name = "long_stable_repeat";
  stable.repeat_count = 1000000000ULL;
  stable.activities.push_back(activity(5, Role::Store, HBM0, 2.0));
  Runtime long_repeat(hw);
  const auto stable_handle = long_repeat.submit(
      manual_plan(hw, {stable}, "billion_repeat_constant_events"),
      owner(11, 1, 1), 0);
  const auto stable_done = run_until_terminal(long_repeat, stable_handle);
  require(stable_done.at("finish_ns").get<std::uint64_t>() == 2 * kNs &&
              long_repeat.counters().at("internal_events").get<std::uint64_t>() < 8,
          "a billion-repeat stable activity integrates as one service interval, not a billion wave events");
}

void hbm_short_activity_releases_before_compute_finishes() {
  const auto hw = rates(1.0, 1.0);
  Epoch overlap;
  overlap.name = "compute_100s_load_50s";
  auto compute = activity(1, Role::Compute, COMPUTE_GATE, 100.0);
  auto load = activity(2, Role::Load, HBM0, 50.0);
  compute.canonical_q = 1.0;
  load.canonical_q = 50.0;
  load.hbm_bytes[0][0] = 50;
  overlap.activities = {compute, load};
  const auto main_plan = manual_plan(hw, {overlap}, "release_hbm_at_50s");
  Runtime runtime(hw);
  const auto main = runtime.submit(main_plan, owner(20, 1, 1), 0);
  runtime.advance_to(50 * kNs);
  const auto midway = runtime.query_progress(main);
  require_close(midway.at("activities").at(0).at("progress").get<double>(),
                0.5, 1e-12, "100-service-second compute remains halfway after 50 s");
  require(midway.at("activities").at(1).at("done").get<bool>() &&
              midway.at("activities").at(1).at("status") == "done",
          "the independent 50 B HBM load is complete at t=50 s");

  const json endpoint_spec{
      {"endpoint_id", 903}, {"transaction_id", 903}, {"owner_kind", "comm"},
      {"direction", "read"}, {"physical_bytes", 20},
      {"row_fraction", {1.0, 0.0}}, {"path_class", "hbm_to_edge"},
      {"eta_hbm", 0.5}, {"route", {{"internal_route", false}}}};
  auto missing_eta = endpoint_spec;
  missing_eta.erase("eta_hbm");
  bool eta_rejected = false;
  try { (void)runtime.submit_endpoint(missing_eta, owner(22, 1, 1), 60 * kNs); }
  catch (const std::invalid_argument&) { eta_rejected = true; }
  require(eta_rejected,
          "local HBM endpoints reject a missing eta_hbm before mutating runtime state");
  auto missing_route = endpoint_spec;
  missing_route.erase("route");
  bool route_rejected = false;
  try { (void)runtime.submit_endpoint(missing_route, owner(23, 1, 1), 60 * kNs); }
  catch (const std::invalid_argument&) { route_rejected = true; }
  require(route_rejected,
          "HBM edge endpoints reject a missing explicit route classification");

  const auto endpoint = runtime.submit_endpoint(
      endpoint_spec, owner(21, 2, 2), 60 * kNs);
  const auto endpoint_start = runtime.query_progress(endpoint);
  require_close(endpoint_start.at("activities").at(0).at("planned_work").at(HBM0).get<double>(),
                40.0, 1e-12,
                "endpoint service work divides physical payload by eta_hbm");
  require(runtime.query_progress(endpoint).at("finish_ns").is_null(),
          "new endpoint is active after its t=60 s arrival");
  const auto done = run_until_terminal(runtime, endpoint);
  const auto main_done = runtime.query_progress(main);
  require(done.at("finish_ns").get<std::uint64_t>() == 100 * kNs &&
              main_done.at("finish_ns").get<std::uint64_t>() == 100 * kNs,
          "new 40 B endpoint uses released HBM and finishes at t=100 s");
  require_close(runtime.counters().at("hbm_busy_ns").get<double>(),
                90.0 * kNs, 2.0,
                "HBM is busy for 50 s then 40 s, with no phantom service from 50 to 60 s");
  require_close(runtime.counters().at("hbm_shared_ns").get<double>(), 0.0,
                1.0, "the later endpoint does not overlap the finished load");
}

void finite_credit_lp_handles_both_boundary_directions() {
  const auto hw = rates(100.0, 100.0);
  Epoch empty;
  empty.name = "empty_buffer_requires_consumer_bound";
  empty.credit_initial = {0.0, 0.0};
  empty.credit_capacity = {10.0, 10.0};
  auto producer = activity(1, Role::Compute, COMPUTE_GATE, 10.0);
  auto consumer = activity(2, Role::Load, HBM0, 10.0);
  producer.credit_delta[0] = 1.0;
  consumer.credit_delta[0] = -2.0;
  producer.canonical_q = 1.0;
  consumer.canonical_q = 3.0;
  empty.activities = {producer, consumer};
  Runtime empty_runtime(hw);
  const auto empty_handle = empty_runtime.submit(
      manual_plan(hw, {empty}, "credit_empty_lp"), owner(30, 1, 1), 0);
  const auto empty_progress = empty_runtime.query_progress(empty_handle);
  const double p_rate = empty_progress.at("activities").at(0).at("rate").get<double>();
  const double c_rate = empty_progress.at("activities").at(1).at("rate").get<double>();
  require(p_rate > 0.0 && c_rate > 0.0,
          "empty-buffer constrained LP must assign positive progress to both feasible activities");
  require(p_rate + 1e-12 >= 2.0 * c_rate,
          "empty input credit cannot decrease: producer must cover the larger consumer delta");

  Epoch full;
  full.name = "full_buffer_requires_producer_bound";
  full.credit_initial = {10.0, 0.0};
  full.credit_capacity = {10.0, 10.0};
  auto full_producer = activity(3, Role::Compute, COMPUTE_GATE, 10.0);
  auto full_consumer = activity(4, Role::Load, HBM0, 10.0);
  full_producer.credit_delta[0] = 2.0;
  full_consumer.credit_delta[0] = -1.0;
  full_producer.canonical_q = 3.0;
  full_consumer.canonical_q = 1.0;
  full.activities = {full_producer, full_consumer};
  Runtime full_runtime(hw);
  const auto full_handle = full_runtime.submit(
      manual_plan(hw, {full}, "credit_full_lp"), owner(31, 1, 1), 0);
  const auto full_progress = full_runtime.query_progress(full_handle);
  const double fp_rate = full_progress.at("activities").at(0).at("rate").get<double>();
  const double fc_rate = full_progress.at("activities").at(1).at("rate").get<double>();
  require(fp_rate > 0.0 && fc_rate > 0.0,
          "full-buffer constrained LP must assign positive progress to both feasible activities");
  require(2.0 * fp_rate <= fc_rate + 1e-12,
          "full output credit cannot increase: consumer must cover the larger producer delta");
}

void cancel_capacity_update_and_snapshot_replay() {
  const auto hw = rates(0.0, 1.0);
  Epoch epoch;
  epoch.name = "hbm_100";
  auto hbm = activity(1, Role::Load, HBM0, 100.0);
  hbm.hbm_bytes[0][0] = 100;
  epoch.activities.push_back(hbm);
  const auto plan = manual_plan(hw, {epoch}, "cancel_capacity_snapshot");

  Runtime cancelled(hw);
  const auto ch = cancelled.submit(plan, owner(40, 1, 1), 0);
  cancelled.cancel(ch, 25 * kNs);
  const auto cp = cancelled.query_progress(ch);
  require(cp.at("state") == "cancelled" &&
              cp.at("finish_ns").get<std::uint64_t>() == 25 * kNs,
          "cancel settles through the requested absolute time and records it");
  require_close(cp.at("activities").at(0).at("progress").get<double>(),
                0.25, 1e-12, "cancel preserves completed partial service");
  require_close(cp.at("served").at("hbm_bytes_by_row_direction").at(0).at(0).get<double>(),
                25.0, 1e-9, "cancel retains HBM payload served before the cancellation tick");

  const auto resize_hw = rates(0.0, 2.0);
  Runtime resized(resize_hw);
  const auto resize_plan = manual_plan(resize_hw, {epoch}, "capacity_change");
  const auto rh = resized.submit(resize_plan, owner(41, 1, 1), 0);
  resized.update_capacity(HBM0, 1.0, 0);
  resized.update_capacity(HBM0, 2.0, 20 * kNs);
  const auto resized_done = run_until_terminal(resized, rh);
  require(resized_done.at("finish_ns").get<std::uint64_t>() == 60 * kNs,
          "capacity update first serves 20 s at 1 B/s, then 80 B at 2 B/s");

  Runtime original(hw);
  const auto original_handle = original.submit(plan, owner(42, 1, 1), 0);
  original.advance_to(7 * kNs);
  const auto frozen = original.query_progress(original_handle);
  const auto snapshot = original.export_snapshot();
  auto fork = Runtime::import_snapshot(hw, snapshot);
  auto corrupt = snapshot;
  corrupt["schema"] = "wrong_schema";
  const auto before = original.counters();
  bool rejected = false;
  try { (void)Runtime::import_snapshot(hw, corrupt); }
  catch (const std::invalid_argument&) { rejected = true; }
  require(rejected, "snapshot import rejects unsupported schema");
  require(original.counters() == before &&
              original.query_progress(original_handle) == frozen,
          "rejected snapshot import leaves the source runtime unchanged");
  const auto original_done = run_until_terminal(original, original_handle);
  const auto fork_handle = fork->execution_handles().at(0);
  const auto fork_done = run_until_terminal(*fork, fork_handle);
  require(original_done == fork_done,
          "snapshot fork replays the same event order and exact execution progress");
  require(original.counters() == fork->counters(),
          "snapshot fork replay produces identical service and event counters");
}

void test_runtime_semantics() {
  equal_owner_fairness_and_split_invariance();
  independent_resources_epoch_barrier_and_repeat_audit();
  hbm_short_activity_releases_before_compute_finishes();
  finite_credit_lp_handles_both_boundary_directions();
  cancel_capacity_update_and_snapshot_replay();
  std::cout << "fairness=2s, W2_finish_s=15, hbm_release_finish_s=100, "
               "credit_lp=checked, snapshot_replay=exact\n";
}
}  // namespace

int main() { return run_test("runtime_semantics_test", test_runtime_semantics); }
