// P0 canonical: derived plan_curved cache equivalence.
//
// advance_to() consults has_credit_curves(plan) once per event iteration;
// the runtime now caches it as an Execution-side derived bool assigned at
// submit and rebuilt on snapshot import. These checks pin the strict-time
// curved trajectory (T4), the import rebuild (T5), and hand-constructed
// multi-epoch plans whose curve lives in a later epoch (T6) against the
// analytic milestones the uncached baseline produces for the same fixtures.
#include "test_support.hh"

#include <array>
#include <cstdint>
#include <memory>

using namespace coarse_die;
using namespace coarse_die_test;

namespace {

using u64 = std::uint64_t;

Hardware unit_curve_hardware() {
  std::array<double, resource_count> rates{};
  rates[COMPUTE_GATE] = 1.0;
  rates[HBM0] = 1.0;
  return manual_hardware(rates);
}

// Progress-delayed output fixture: compute charges the buffer through a
// curve that only moves after half of its work, store drains it linearly.
// Unit per-second rates make every threshold hand-computable.
Epoch delayed_output_epoch(std::uint64_t count = 1) {
  Epoch epoch;
  epoch.name = "progress_delayed_output";
  epoch.credit_capacity = {{0, 1}};
  auto compute = activity(1, Role::Compute, COMPUTE_GATE,
                          static_cast<double>(count));
  compute.int4_ops = count;
  compute.credit_delta = {{0, 2.0 * count}};
  compute.credit_curve = {{0, {{0, 0}}},
                          {0.5, {{0, 0}}},
                          {1, {{0, 2.0 * count}}}};
  auto store = activity(2, Role::Store, HBM0, 2.0 * count);
  store.hbm_bytes[0][1] = 2 * count;
  store.canonical_q = 2.0 * count;
  store.credit_delta = {{0, -2.0 * count}};
  store.credit_curve = {{0, {{0, 0}}}, {1, store.credit_delta}};
  epoch.activities = {compute, store};
  return epoch;
}

std::shared_ptr<const Plan> delayed_output(const Hardware& hw,
                                           std::uint64_t count = 1) {
  return manual_plan(hw, {delayed_output_epoch(count)},
                     "plan_curved_cache_fixture");
}

void drain(Runtime& runtime) {
  unsigned guard = 0;
  while (!runtime.drained()) {
    const auto next = runtime.next_event();
    require(next.has_value() && *next > runtime.now_ns(),
            "curved fixture keeps a finite advancing next event");
    runtime.advance_to(*next);
    require(++guard < 100, "curved fixture drains in a bounded event count");
  }
}

void verify_strict_time_trajectory() {
  const auto hw = unit_curve_hardware();
  Runtime runtime(hw);
  const auto handle = runtime.submit(delayed_output(hw), owner(1, 1, 1), 0);
  require(runtime.next_event() == 500000000ULL,
          "cached curved gate keeps the half-work output threshold");
  runtime.advance_to(400000000ULL);
  require(runtime.query_progress(handle)
              .at("served").at("hbm_bytes_by_row_direction")[0][1] == 0,
          "store stays idle below the curve knee");
  runtime.advance_to(500000000ULL);
  auto fork = Runtime::import_snapshot(hw, runtime.export_snapshot());
  require(fork->next_event() == runtime.next_event(),
          "imported fork agrees on the next strict event");
  drain(runtime);
  drain(*fork);
  require(*fork->terminal_finish_ns(handle) ==
                      *runtime.terminal_finish_ns(handle) &&
              *runtime.terminal_finish_ns(handle) == 2500000000ULL,
          "strict curved completion stays the analytical 2.5s drain");
  require(fork->export_snapshot() == runtime.export_snapshot(),
          "forked continuation is byte-identical");
  require(runtime.query_progress(handle)
              .at("served").at("hbm_bytes_by_row_direction")[0][1] == 2,
          "physical output byte endpoint is exact");
}

void verify_import_rebuild_mid_flight() {
  // T5: split a curved run at several mid-flight ticks; every resumed run
  // must reproduce the continuous trajectory byte-for-byte. plan_curved is
  // rebuilt from the decoded plan on import, so the resumed strict-time
  // gate matches the uninterrupted run at every later event.
  const auto hw = unit_curve_hardware();
  for (u64 split : {400000000ULL, 900000000ULL, 2000000000ULL}) {
    Runtime whole(hw);
    const auto handle =
        whole.submit(delayed_output(hw, 2), owner(2, 2, 2), 0);
    Runtime cut(hw);
    const auto cut_handle =
        cut.submit(delayed_output(hw, 2), owner(2, 2, 2), 0);
    whole.advance_to(split);
    cut.advance_to(split);
    auto resumed = Runtime::import_snapshot(hw, cut.export_snapshot());
    require(resumed && resumed->now_ns() == split,
            "fork imports at its own tick");
    whole.advance_to(split + 12345);
    resumed->advance_to(split + 12345);
    drain(whole);
    drain(*resumed);
    require(*resumed->terminal_finish_ns(cut_handle) ==
                *whole.terminal_finish_ns(handle),
            "resumed strict-time completion equals the continuous run");
    require(resumed->export_snapshot() == whole.export_snapshot(),
            "resumed snapshot is byte-identical after drain");
  }
}

void verify_hand_built_multi_epoch_plan() {
  // T6: plans constructed epoch-by-epoch (no compiler, no from_json) still
  // derive the curve cache on submit; the curve may live in a later epoch
  // while the first epoch stays flat.
  const auto hw = unit_curve_hardware();
  auto plan = std::make_shared<Plan>();
  plan->task = json{{"name", "hand_built_two_epoch"}};
  plan->summary = json{{"status", "test_fixture"}};
  plan->hardware_fingerprint = hw.fingerprint;
  Epoch flat;
  flat.name = "flat_gate";
  flat.activities = {activity(7, Role::Compute, COMPUTE_GATE, 1.0)};
  plan->epochs = {flat, delayed_output_epoch()};
  plan->fingerprint =
      plan_to_json(*plan).at("fingerprint").get<std::string>();

  Runtime runtime(hw);
  const auto handle = runtime.submit(plan, owner(3, 3, 3), 0);
  require(runtime.next_event() == 1000000000ULL,
          "flat first epoch drains a unit of work per second");
  runtime.advance_to(1000000000ULL);
  require(runtime.next_event() == 1500000000ULL,
          "curved second epoch keeps its half-work threshold after the seam");
  // Interleave a snapshot fork across the epoch boundary: the imported run
  // must keep the same strict-time milestones through the curved tail.
  auto fork = Runtime::import_snapshot(hw, runtime.export_snapshot());
  require(fork->next_event() == 1500000000ULL,
          "imported run keeps the curved threshold after the seam");
  drain(runtime);
  drain(*fork);
  require(*fork->terminal_finish_ns(handle) ==
                  *runtime.terminal_finish_ns(handle) &&
              runtime.terminal_finish_ns(handle).has_value(),
          "imported two-epoch continuation finishes identically");
  require_close(static_cast<double>(*runtime.terminal_finish_ns(handle)),
                3500000000.0, 1.0,
                "hand-built two-epoch plan finishes at 1s + 2.5s");
}

}  // namespace

int main() {
  int failures = 0;
  failures += run_test("strict_time_trajectory", verify_strict_time_trajectory);
  failures += run_test("import_rebuild_mid_flight", verify_import_rebuild_mid_flight);
  failures += run_test("hand_built_multi_epoch_plan",
                       verify_hand_built_multi_epoch_plan);
  return failures == 0 ? 0 : 1;
}
