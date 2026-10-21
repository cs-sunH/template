// Synthetic guard for the 2026-10-19 review §6 / analysis_P1 §4 R1 gap:
// advance_to's preflight_tiny whole-history ledger prediction under a
// declared repeat.
//
// preflight_tiny runs only for pending event intervals dt in (0, 1e-15] ns.
// It predicts each buffer's credit as credit_initial plus the
// accumulated_credit of every history entry whose epoch_index matches the
// active epoch, with NO round filter, while every other ledger consumer
// (recompute_credit, integrate's staging, settle's endpoint validation)
// filters epoch_index == epoch AND round == round. Under a declared repeat,
// past-round same-epoch activities stay done at progress 1.0 in the vector
// and contribute their full terminal credit to preflight_tiny's prediction
// only.
//
// KNOWN DEVIATION PINNED HERE (retained R1 whole-history semantics; not
// fixed in this batch, see
// runs/roofline_batch/manifest/remediation_tiny_repeat.md): with repeat >=
// 2, a genuine tiny event in round >= 1 throws
//   "numerical_failure: represented tiny service would cross a finite
//    buffer boundary"
// once (past-round same-epoch accumulated credit) + (current-round credit)
// exceeds capacity + tolerance, although the round-filtered prediction
// stays inside [0, capacity] by validate_plan's own endpoint contract. The
// fixtures below make the deviation exact and quantitative: the throwing
// event's predicted ledger is 1.5 against capacity 1 -- the round-filtered
// prediction 0.5 plus exactly one full copy of the old round's terminal
// credit 1.0 -- and the capacity-2 twin completes the identical event
// schedule, bracketing the surplus at exactly the old-round credit. A
// production fix (round filter inside preflight_tiny) is pending
// authorization; until it lands intentionally, these pins must keep
// passing against the current behavior.
//
// All fixtures are synthetic sub-nanosecond dyadic scales (the same family
// as runtime_subepsilon_event_test / runtime_tiny_ready_test): one curved
// COMPUTE_GATE activity whose whole lifetime is 2^-100 s (~7.9e-22 ns), so
// both per-round curve-knee boundary events are genuinely sub-1e-15 ns
// intervals and the tiny branch really executes in every round. They are
// not target hardware measurements.
#include "test_support.hh"

#include <cmath>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>

using namespace coarse_die;
using namespace coarse_die_test;

namespace {

using u64 = std::uint64_t;

// One curved activity on a unit-rate gate. Its lifetime is 2^-kTinyShift
// seconds, so each round emits two strictly positive boundary events
// (curve knee at progress 0.5, completion at progress 1.0), each with dt =
// 2^-(kTinyShift+1) s ~ 3.95e-22 ns <= 1e-15 ns: genuine tiny events.
constexpr int kTinyShift = 100;

Hardware tiny_hardware() {
  std::array<double, resource_count> rates{};
  rates[COMPUTE_GATE] = 1.0;
  return manual_hardware(rates);
}

// `credit` is the activity's full buffer-0 credit_delta (its curve ends
// there); `capacity` is the epoch's buffer-0 capacity. The throwing guard
// compares initial(0) + whole-history predicted against capacity, so the
// capacity choice relative to per-round credit decides whether the retained
// whole-history scan stays inside the tolerance.
std::shared_ptr<const Plan> tiny_curved_plan(const Hardware& hw,
                                             u64 repeat_count, double capacity,
                                             double credit) {
  auto plan = std::make_shared<Plan>();
  plan->task = json{{"op_kind", "tiny_repeat_guard_fixture"}};
  plan->summary = json::object();
  plan->hardware_fingerprint = hw.fingerprint;
  plan->launch_ns = 0;
  if (repeat_count >= 2) plan->repeat = PlanRepeat{repeat_count, 0};
  Epoch epoch;
  epoch.name = "tiny_curved_round";
  epoch.credit_capacity = {{capacity, 0}};
  epoch.credit_initial = {{0, 0}};
  Activity a = activity(1, Role::Compute, COMPUTE_GATE,
                        std::ldexp(1.0, -kTinyShift));
  a.private_time_s = std::ldexp(1.0, -kTinyShift);
  a.canonical_q = 1.0;
  a.credit_delta = {{credit, 0}};
  a.credit_curve = {{0, {{0, 0}}},
                    {0.5, {{credit / 2, 0}}},
                    {1, a.credit_delta}};
  epoch.activities = {a};
  plan->epochs = {epoch};
  plan->fingerprint =
      plan_to_json(*plan).at("fingerprint").get<std::string>();
  return plan;
}

void verify_tiny_single_instance_control() {
  // No repeat: the same two tiny boundary events must pass preflight_tiny
  // with an empty history. This is the control that isolates the repeat
  // history (not the tiny scale itself) as the throw's cause.
  const auto hw = tiny_hardware();
  Runtime runtime(hw);
  const auto handle =
      runtime.submit(tiny_curved_plan(hw, 0, 1.0, 1.0), owner(1, 1, 1), 0);
  require(runtime.next_event() == 1,
          "sub-tiny first knee maps to the next public tick");
  runtime.advance_to(1);
  require(runtime.drained(), "single tiny round drains inside its tick");
  require(*runtime.terminal_finish_ns(handle) == 1,
          "terminal finish lands on the enclosing tick");
  require(runtime.counters().at("internal_events").get<u64>() == 3,
          "two tiny boundary events plus the epoch leave fired, actual=" +
              std::to_string(
                  runtime.counters().at("internal_events").get<u64>()));
  const auto progress = runtime.query_progress(handle);
  require(progress.at("activities").size() == 1 &&
              progress.at("activities")[0].at("status") == "done" &&
              progress.at("activities")[0].at("progress") == 1.0,
          "tiny round completed its activity exactly");
  require_close(progress.at("credit")[0].at("value").get<double>(), 1.0,
                1e-12, "done activity leaves the ledger at capacity");
}

void verify_tiny_repeat_contamination_guard() {
  const auto hw = tiny_hardware();

  // Capacity 1 with per-round credit 1: round 0's tiny events pass (its
  // predicted ledger is 0.5 then 1.0, both legal). Round 1's first tiny
  // knee sees predicted = old round (1.0) + current knee (0.5) = 1.5
  // against capacity 1 and throws -- the pinned deviation.
  Runtime runtime(hw);
  const auto handle =
      runtime.submit(tiny_curved_plan(hw, 2, 1.0, 1.0), owner(2, 2, 2), 0);
  runtime.advance_to(1);
  const auto seam = runtime.query_progress(handle);
  require(seam.at("activities").size() == 2,
          "both declared rounds stay in the lifetime history");
  require(seam.at("activities")[0].at("status") == "done" &&
              seam.at("activities")[0].at("progress") == 1.0,
          "round 0 completed through its own tiny events");
  require(seam.at("activities")[1].at("progress") == 0.0,
          "round 1 has not started when round 0 drains");
  require_close(seam.at("credit")[0].at("value").get<double>(), 0.0, 1e-12,
                "round switch reset the reported ledger to the fresh "
                "round's initial credit");

  // Snapshot fork before the failing advance: the guard reads only the
  // exported state, so the imported twin must fail identically.
  auto forked = Runtime::import_snapshot(hw, runtime.export_snapshot());

  bool threw = false;
  std::string message;
  try {
    runtime.advance_to(2);
  } catch (const std::runtime_error& error) {
    threw = true;
    message = error.what();
  }
  require(threw, "round-1 tiny event hits the whole-history preflight guard");
  require(message ==
              "numerical_failure: represented tiny service would cross a "
              "finite buffer boundary",
          "the guard that fires is the predicted-ledger boundary check");
  require(runtime.now_ns() == 1,
          "preflight rejects before any time or progress mutation");
  const auto halted = runtime.query_progress(handle);
  require(halted.at("activities")[1].at("progress") == 0.0,
          "the failing tiny event advanced no round-1 progress");

  threw = false;
  try {
    forked->advance_to(2);
  } catch (const std::runtime_error& error) {
    threw = true;
    message = error.what();
  }
  require(threw && message ==
              "numerical_failure: represented tiny service would cross a "
              "finite buffer boundary",
          "imported snapshot reproduces the same deviation");

  // Capacity 2, identical trajectory: the same round-1 events predict
  // 1.5 (first knee) and 2.0 (completion) -- the latter is exactly the
  // old-round credit 1.0 plus the current full credit 1.0 -- and both stay
  // inside capacity + tolerance, so the run completes. Together with the
  // capacity-1 twin this brackets the surplus in the failing prediction at
  // exactly one old-round copy of the accumulated credit.
  Runtime absorber(hw);
  const auto absorber_handle =
      absorber.submit(tiny_curved_plan(hw, 2, 2.0, 1.0), owner(3, 3, 3), 0);
  absorber.advance_to(1);
  absorber.advance_to(2);
  require(absorber.drained() &&
              *absorber.terminal_finish_ns(absorber_handle) == 2,
          "capacity that absorbs old+current credit completes both rounds");
  require(absorber.counters().at("internal_events").get<u64>() == 7,
          "two tiny knees per round plus seam, ready, and epoch-leave "
          "events, actual=" +
              std::to_string(
                  absorber.counters().at("internal_events").get<u64>()));

  // Zero-credit curves, identical timing: past-round entries contribute
  // exactly 0, so the whole-history prediction equals the round-filtered
  // one and both tiny rounds complete. This is the quantitative zero point
  // of the contamination term.
  Runtime neutral(hw);
  const auto neutral_handle =
      neutral.submit(tiny_curved_plan(hw, 2, 0.0, 0.0), owner(4, 4, 4), 0);
  neutral.advance_to(1);
  neutral.advance_to(2);
  require(neutral.drained() && *neutral.terminal_finish_ns(neutral_handle) == 2,
          "zero old-round credit leaves the tiny rounds undisturbed");
  require(neutral.counters().at("internal_events").get<u64>() == 7,
          "zero-credit twin fires the same tiny event schedule, actual=" +
              std::to_string(
                  neutral.counters().at("internal_events").get<u64>()));
}

}  // namespace

int main() {
  int failures = 0;
  failures += run_test("tiny_single_instance_control",
                       verify_tiny_single_instance_control);
  failures += run_test("tiny_repeat_contamination_guard",
                       verify_tiny_repeat_contamination_guard);
  return failures == 0 ? 0 : 1;
}
