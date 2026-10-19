// P1 canonical: current (round, epoch) activity window semantics.
//
// Event-path scans read only the execution's [view_begin, view_end) window
// while the full activity history stays for lifetime accounting. These
// checks pin: done items inside the window still feed the credit ledger
// across epoch and round switches; items that left the window publish zero
// rate and zero alpha afterwards; round seams, snapshot continuation, and
// cancellation keep their exact trajectories; and runtime counters stay
// linear in the declared repeat count.
#include "test_support.hh"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <memory>
#include <vector>

using namespace coarse_die;
using namespace coarse_die_test;

namespace {

using u64 = std::uint64_t;

// Per-second unit rates: one unit of work takes one second on its resource.
Hardware unit_hardware() {
  std::array<double, resource_count> rates{};
  rates[COMPUTE_GATE] = 1.0;
  rates[SRAM_READ] = 1.0;
  rates[HBM0] = 1.0;
  return manual_hardware(rates);
}

Activity unit_activity(u64 id, std::size_t resource) {
  Activity activity;
  activity.id = id;
  activity.role = Role::Compute;
  activity.work[resource] = 1.0;
  activity.canonical_q = 1.0;
  return activity;
}

// Epoch 0 charges one credit unit through compute while the store drains
// two HBM units. The store is credit-blocked until the compute completes
// (the buffer starts empty at its low boundary), so epoch 0 spans exactly
// 1s (compute) + 2s (store) and the ledger between t=1s and t=3s is carried
// entirely by the done compute item. Epoch 1 is a credit-free sram tail.
std::shared_ptr<const Plan> window_plan(const Hardware& hw, u64 repeat_count,
                                        u64 seam_ns) {
  auto plan = std::make_shared<Plan>();
  plan->task = json{{"op_kind", "view_window_fixture"}};
  plan->summary = json::object();
  plan->hardware_fingerprint = hw.fingerprint;
  plan->launch_ns = 0;
  if (repeat_count >= 2) plan->repeat = PlanRepeat{repeat_count, seam_ns};
  Epoch charge;
  charge.name = "charge_and_drain";
  charge.credit_capacity = {{1, 0}};
  Activity compute = unit_activity(1, COMPUTE_GATE);
  compute.credit_delta = {{1, 0}};
  Activity store = unit_activity(2, HBM0);
  store.role = Role::Store;
  store.work[HBM0] = 2.0;
  store.hbm_bytes[0][1] = 2;
  store.credit_delta = {{-1, 0}};
  charge.activities = {compute, store};
  Epoch tail;
  tail.name = "tail";
  tail.activities = {unit_activity(3, SRAM_READ)};
  plan->epochs = {charge, tail};
  plan->fingerprint =
      plan_to_json(*plan).at("fingerprint").get<std::string>();
  return plan;
}

// Curved fixture (service-curve shape): compute charges the buffer through
// a curve that only moves after half of its work, the store drains it, and
// a third short sram activity makes the window the maximal three entries.
std::shared_ptr<const Plan> curved_triple_plan(const Hardware& hw,
                                               u64 repeat_count = 0,
                                               u64 seam_ns = 0) {
  auto plan = std::make_shared<Plan>();
  plan->task = json{{"op_kind", "view_window_curved_fixture"}};
  plan->summary = json::object();
  plan->hardware_fingerprint = hw.fingerprint;
  plan->launch_ns = 0;
  if (repeat_count >= 2) plan->repeat = PlanRepeat{repeat_count, seam_ns};
  Epoch epoch;
  epoch.name = "curved_triple";
  epoch.credit_capacity = {{0, 1}};
  Activity compute = unit_activity(1, COMPUTE_GATE);
  compute.int4_ops = 1;
  compute.credit_delta = {{0, 2.0}};
  compute.credit_curve = {{0, {{0, 0}}},
                          {0.5, {{0, 0}}},
                          {1, {{0, 2.0}}}};
  Activity store = unit_activity(2, HBM0);
  store.role = Role::Store;
  store.work[HBM0] = 2.0;
  store.hbm_bytes[0][1] = 2;
  store.canonical_q = 2.0;
  store.credit_delta = {{0, -2.0}};
  store.credit_curve = {{0, {{0, 0}}}, {1, store.credit_delta}};
  Activity side = unit_activity(4, SRAM_READ);
  side.work[SRAM_READ] = 0.25;
  epoch.activities = {compute, store, side};
  plan->epochs = {epoch};
  plan->fingerprint =
      plan_to_json(*plan).at("fingerprint").get<std::string>();
  return plan;
}

void drain(Runtime& runtime) {
  unsigned guard = 0;
  while (!runtime.drained()) {
    const auto next = runtime.next_event();
    require(next.has_value() && *next > runtime.now_ns(),
            "fixture keeps a finite advancing next event");
    runtime.advance_to(*next);
    require(++guard < 1000, "fixture drains in a bounded event count");
  }
}

const json& activity_entry(const json& progress, u64 activity_id) {
  const auto& activities = progress.at("activities");
  for (const auto& entry : activities)
    if (entry.at("activity_id") == activity_id) return entry;
  throw std::runtime_error("activity missing from progress");
}

void require_window_cleared(const json& progress, u64 activity_id) {
  const json& entry = activity_entry(progress, activity_id);
  require(entry.at("rate") == 0 && entry.at("alpha") == 0,
          "activity outside the current window keeps zero rate and alpha");
}

void verify_window_lifecycle() {
  const auto hw = unit_hardware();
  Runtime runtime(hw);
  const auto handle = runtime.submit(window_plan(hw, 2, 1000000000ULL),
                                     owner(1, 1, 1), 0);

  // t=1.25s: compute completed at 1s while the store drained concurrently
  // (the low-boundary row only caps the draining rate at the charging
  // rate). progress_store = 0.625, so the ledger 1*1.0 - 1*0.625 = 0.375 is
  // carried entirely by the done compute item's contribution.
  runtime.advance_to(1250000000ULL);
  const auto mid = runtime.query_progress(handle);
  require(mid.at("activities").size() == 3,
          "round 0 holds its three live activities");
  require(activity_entry(mid, 1).at("status") == "done" &&
              activity_entry(mid, 1).at("rate") == 0,
          "compute completed at its unit-work instant");
  require(activity_entry(mid, 2).at("status") == "running",
          "store keeps draining after compute completed");
  require_close(mid.at("credit")[0].at("value").get<double>(), 0.375, 1e-12,
                "done activity still contributes its full credit");

  // t=2.5s: past the epoch 0 -> epoch 1 switch (epoch 0 spans 2s). The
  // departed window (compute, store) must read zero rate and zero alpha
  // while the tail epoch runs.
  runtime.advance_to(2500000000ULL);
  const auto tail = runtime.query_progress(handle);
  require(activity_entry(tail, 3).at("status") == "running",
          "tail epoch runs after the switch");
  require_window_cleared(tail, 1);
  require_window_cleared(tail, 2);

  // t=3.5s: round 0 finished at 3s (2s epoch 0 + 1s tail); round 1 becomes
  // ready at 4s (1s seam). Mid seam the execution holds no share and the
  // whole departed round reads zero.
  runtime.advance_to(3500000000ULL);
  const auto seam = runtime.query_progress(handle);
  require(seam.at("state") == "queued", "seam window reports queued");
  require_window_cleared(seam, 1);
  require_window_cleared(seam, 2);
  require_window_cleared(seam, 3);
  auto resumed = Runtime::import_snapshot(hw, runtime.export_snapshot());
  require(resumed && resumed->now_ns() == 3500000000ULL,
          "mid-seam fork imports at its own tick");

  // t=5.25s: round 1 epoch 0 — compute (id 1, round 1) completed at 5s and
  // the same done-item ledger reappears with the full round-0 history still
  // resident, and that history stays cleared.
  runtime.advance_to(5250000000ULL);
  const auto again = runtime.query_progress(handle);
  require(again.at("activities").size() == 6,
          "both declared rounds stay in the lifetime history");
  require_close(again.at("credit")[0].at("value").get<double>(), 0.375, 1e-12,
                "round 1 reproduces the same done-item credit ledger");
  require_window_cleared(again, 1);
  require_window_cleared(again, 2);
  require_window_cleared(again, 3);

  drain(runtime);
  drain(*resumed);
  require(*runtime.terminal_finish_ns(handle) == 7000000000ULL &&
              *resumed->terminal_finish_ns(handle) == 7000000000ULL,
          "two 3s rounds behind a 1s seam finish at 7s");
  require(resumed->export_snapshot() == runtime.export_snapshot(),
          "mid-seam resumed run is byte-identical after drain");
  const auto finished = runtime.query_progress(handle);
  require(finished.at("state") == "finished" &&
              finished.at("activities").size() == 6,
          "terminal report keeps the whole history");
  for (u64 id : {1, 2, 3}) require_window_cleared(finished, id);
  require(finished.at("served").at("hbm_bytes_by_row_direction")[0][1] == 4,
          "store served both rounds: 2 bytes per round");
}

void verify_cancel_after_switch() {
  const auto hw = unit_hardware();
  Runtime runtime(hw);
  const auto handle = runtime.submit(window_plan(hw, 2, 1000000000ULL),
                                     owner(2, 2, 2), 0);
  runtime.advance_to(3500000000ULL);
  runtime.cancel(handle, 3600000000ULL);
  const auto state = runtime.query_progress(handle);
  require(state.at("state") == "cancelled", "cancel mid-epoch stays cancelled");
  for (u64 id : {1, 2, 3}) require_window_cleared(state, id);
  auto resumed = Runtime::import_snapshot(hw, runtime.export_snapshot());
  require(resumed->export_snapshot() == runtime.export_snapshot(),
          "cancelled snapshot round-trips byte-identically");
  drain(*resumed);
  require(resumed->export_snapshot() == runtime.export_snapshot(),
          "drained import of a cancelled run stays identical");
}

void verify_repeat_curved_r1_retention() {
  // Declared repeat on a curved plan keeps the tiny-step preflight's
  // original whole-history credit prediction (retained R1 semantics), and
  // a fixed-step partition plus a mid-round import continuation still
  // reach the same drain as the event-stepped run.
  const auto hw = unit_hardware();
  const auto plan = curved_triple_plan(hw, 2, 0);
  Runtime whole(hw);
  const auto handle = whole.submit(plan, owner(3, 3, 3), 0);
  Runtime stepped(hw);
  const auto stepped_handle = stepped.submit(plan, owner(3, 3, 3), 0);
  unsigned guard = 0;
  for (u64 t = 300000000ULL; t < 4700000000ULL; t += 300000000ULL) {
    whole.advance_to(t);
    stepped.advance_to(t);
    require(++guard < 100, "fixed stepping stays bounded");
  }
  auto fork = Runtime::import_snapshot(hw, stepped.export_snapshot());
  drain(whole);
  drain(stepped);
  drain(*fork);
  require_close(
      static_cast<double>(*whole.terminal_finish_ns(handle)), 5000000000.0, 2.0,
      "curved repeat drains two 2.5s instances");
  require(std::llabs(static_cast<long long>(*stepped.terminal_finish_ns(
                         stepped_handle)) -
                     static_cast<long long>(*whole.terminal_finish_ns(handle))) <= 2,
          "fixed-step partition reaches the same completion instant");
  require(*fork->terminal_finish_ns(stepped_handle) ==
              *stepped.terminal_finish_ns(stepped_handle),
          "mid-round imported continuation agrees exactly");
  require(fork->export_snapshot() == stepped.export_snapshot(),
          "imported continuation is byte-identical after drain");
  require(stepped.query_progress(stepped_handle)
              .at("served").at("hbm_bytes_by_row_direction")[0][1] == 4,
          "curved store served both rounds exactly");
}

// True tiny-step retention variant of repeat_curved_r1_retention: the
// 300ms-step sibling never reaches advance_to's tiny preflight (its event
// intervals are far above the 1e-15 ns threshold). Here one curved
// activity's whole lifetime is 2^-100 s (~7.9e-22 ns) on a unit-rate gate,
// so every round emits two genuinely sub-1e-15 ns boundary events and the
// retained whole-history preflight really executes. Capacity 3 absorbs the
// whole-history prediction of the last round (two past rounds' full credit
// 1.0 each plus the current round's 1.0), so this scenario completes; the
// unabsorbed capacity-1 deviation itself is pinned in
// runtime_tiny_repeat_test.cc. Synthetic sub-nanosecond dyadic fixture, in
// the runtime_subepsilon_event_test family; not a hardware measurement.
std::shared_ptr<const Plan> tiny_curved_repeat_plan(const Hardware& hw,
                                                    u64 repeat_count) {
  auto plan = std::make_shared<Plan>();
  plan->task = json{{"op_kind", "view_window_tiny_retention_fixture"}};
  plan->summary = json::object();
  plan->hardware_fingerprint = hw.fingerprint;
  plan->launch_ns = 0;
  if (repeat_count >= 2) plan->repeat = PlanRepeat{repeat_count, 0};
  Epoch epoch;
  epoch.name = "tiny_curved";
  epoch.credit_capacity = {{3, 0}};
  epoch.credit_initial = {{0, 0}};
  Activity compute = unit_activity(1, COMPUTE_GATE);
  compute.work[COMPUTE_GATE] = std::ldexp(1.0, -100);
  compute.private_time_s = std::ldexp(1.0, -100);
  compute.credit_delta = {{1, 0}};
  compute.credit_curve = {{0, {{0, 0}}},
                          {0.5, {{0.5, 0}}},
                          {1, compute.credit_delta}};
  epoch.activities = {compute};
  plan->epochs = {epoch};
  plan->fingerprint =
      plan_to_json(*plan).at("fingerprint").get<std::string>();
  return plan;
}

void verify_repeat_curved_tiny_retention() {
  // Three declared rounds, each finishing inside its own 1ns public tick
  // through two tiny boundary events per round. The event-stepped run, the
  // fixed 1ns-step partition, and a mid-seam imported continuation must
  // keep the exact finish, the exact counters, and byte-identical drained
  // snapshots while the tiny preflight executes in every round.
  const auto hw = unit_hardware();
  const auto plan = tiny_curved_repeat_plan(hw, 3);
  Runtime whole(hw);
  const auto handle = whole.submit(plan, owner(6, 6, 6), 0);
  Runtime stepped(hw);
  const auto stepped_handle = stepped.submit(plan, owner(6, 6, 6), 0);
  stepped.advance_to(1);
  stepped.advance_to(2);
  auto fork = Runtime::import_snapshot(hw, stepped.export_snapshot());
  stepped.advance_to(3);
  drain(whole);
  require(whole.drained() && *whole.terminal_finish_ns(handle) == 3,
          "event-stepped tiny repeat finishes one round per tick");
  require(stepped.drained() &&
              *stepped.terminal_finish_ns(stepped_handle) == 3,
          "fixed-step partition finishes at the same tick");
  require(whole.counters() == stepped.counters(),
          "stepping does not perturb any runtime counter");
  require(whole.counters().at("internal_events").get<u64>() == 11,
          "two tiny boundary events per round plus seam, ready, and "
          "epoch-leave events");
  drain(*fork);
  require(*fork->terminal_finish_ns(stepped_handle) == 3,
          "mid-seam imported continuation keeps the exact finish");
  require(fork->export_snapshot() == stepped.export_snapshot(),
          "imported tiny-round continuation is byte-identical after drain");
  require(whole.export_snapshot() == stepped.export_snapshot(),
          "event-stepped and fixed-step tiny runs are byte-identical");
}

void verify_max_window_curved_completion() {
  // Three-activity epoch (maximal window) with curves and a finite buffer:
  // exact analytic completion, exact served endpoints, and a snapshot fork
  // at the curve knee.
  const auto hw = unit_hardware();
  Runtime runtime(hw);
  const auto handle = runtime.submit(curved_triple_plan(hw), owner(4, 4, 4), 0);
  require(runtime.next_event() == 250000000ULL,
          "the short sram side activity completes first");
  runtime.advance_to(500000000ULL);
  auto fork = Runtime::import_snapshot(hw, runtime.export_snapshot());
  drain(runtime);
  drain(*fork);
  require_close(static_cast<double>(*runtime.terminal_finish_ns(handle)),
                2500000000.0, 1.0,
                "maximal curved window keeps the analytic 2.5s completion");
  require(*fork->terminal_finish_ns(handle) ==
              *runtime.terminal_finish_ns(handle),
          "knee-time fork finishes identically");
  require(fork->export_snapshot() == runtime.export_snapshot(),
          "knee-time fork is byte-identical after drain");
  const auto progress = runtime.query_progress(handle);
  require(progress.at("served").at("hbm_bytes_by_row_direction")[0][1] == 2,
          "physical output byte endpoint is exact");
  require(activity_entry(progress, 4).at("status") == "done",
          "side activity completed");
}

void verify_repeat_counter_linearity() {
  // Same template at R instances: the trajectory stays the closed form and
  // the runtime counters grow by a constant per-round increment.
  const auto hw = unit_hardware();
  std::vector<u64> events, allocations;
  for (u64 rounds : {1, 2, 4, 8}) {
    Runtime runtime(hw);
    auto plan = std::make_shared<Plan>();
    plan->task = json{{"op_kind", "view_linearity_fixture"}};
    plan->summary = json::object();
    plan->hardware_fingerprint = hw.fingerprint;
    plan->launch_ns = 0;
    if (rounds >= 2) plan->repeat = PlanRepeat{rounds, 0};
    Epoch compute;
    compute.name = "compute";
    compute.activities = {unit_activity(1, COMPUTE_GATE)};
    Epoch store;
    store.name = "store";
    store.activities = {unit_activity(2, SRAM_READ)};
    plan->epochs = {compute, store};
    plan->fingerprint =
        plan_to_json(*plan).at("fingerprint").get<std::string>();
    const auto handle = runtime.submit(plan, owner(5, 5, 5), 0);
    drain(runtime);
    const auto finish = runtime.terminal_finish_ns(handle);
    require(finish.has_value() && *finish == 2000000000ULL * rounds,
            "unit chain of R rounds finishes at 2s per round");
    events.push_back(runtime.counters().at("internal_events"));
    allocations.push_back(runtime.counters().at("allocation_updates"));
  }
  const auto require_linear = [](const std::vector<u64>& values) {
    // Each template doubles the added rounds, so the counter increment
    // between consecutive templates must double exactly.
    const u64 first = values[1] - values[0];
    require(values[2] - values[1] == 2 * first,
            "counter increment doubles with the added rounds");
    require(values[3] - values[2] == 2 * (values[2] - values[1]),
            "counter increment doubles again with the added rounds");
  };
  require_linear(events);
  require_linear(allocations);
}

}  // namespace

int main() {
  int failures = 0;
  failures += run_test("window_lifecycle", verify_window_lifecycle);
  failures += run_test("cancel_after_switch", verify_cancel_after_switch);
  failures += run_test("repeat_curved_r1_retention",
                       verify_repeat_curved_r1_retention);
  failures += run_test("repeat_curved_tiny_retention",
                       verify_repeat_curved_tiny_retention);
  failures += run_test("max_window_curved_completion",
                       verify_max_window_curved_completion);
  failures += run_test("repeat_counter_linearity",
                       verify_repeat_counter_linearity);
  return failures == 0 ? 0 : 1;
}
