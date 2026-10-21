#include "test_support.hh"

#include <array>
#include <cmath>
#include <string>

using namespace coarse_die;
using namespace coarse_die_test;

namespace {
// One trillion synthetic operations over one second. A buffer has four units,
// starts one unit from a boundary and changes by two units per full activity.
// Hence exactly half a second is payable, in every declared credit unit.
std::shared_ptr<const Plan> fixture_plan(const Hardware& hw, double unit, bool filling) {
  Epoch epoch;
  epoch.name = "synthetic_subunit_credit_half_second";
  epoch.credit_capacity = {{4 * unit, 0}};
  epoch.credit_initial = {{(filling ? 3 : 1) * unit, 0}};
  auto work = activity(1, Role::Compute, COMPUTE_GATE, 1);
  work.private_time_s = 1;
  work.canonical_q = 1;
  work.vec_eqops = 1000000000000ULL;
  work.credit_delta = {{(filling ? 2 : -2) * unit, 0}};
  work.credit_curve = {{0, {{0, 0}}}, {1, work.credit_delta}};
  epoch.activities = {work};
  return manual_plan(hw, {epoch});
}

void check_progress(const Hardware& hw, const std::shared_ptr<const Plan>& plan) {
  Runtime source(hw);
  const auto handle = source.submit(plan, owner(1, 1, 1), 0);
  require(source.next_event() == 500000000ULL,
          "one unit of positive inventory/headroom pays exactly half a second");
  source.advance_to(250000000ULL);
  const auto prefix = source.query_progress(handle);
  require(prefix.at("activities")[0].at("progress") == 0.25 &&
              prefix.at("served").at("vec_eqops") == 250000000000ULL,
          "quarter-second progress pays a quarter trillion operations");

  const auto saved = source.export_snapshot();
  auto restored = Runtime::import_snapshot(hw, saved);
  require(restored->export_snapshot() == saved,
          "actual inventory and allocation survive an exact snapshot roundtrip");
  source.advance_to(500000000ULL);
  restored->advance_to(500000000ULL);
  require(!source.next_event() && !source.drained() &&
              source.query_progress(handle).at("activities")[0].at("progress") == 0.5,
          "the true boundary blocks the unpaid remaining half");
  source.advance_to(750000000ULL);
  restored->advance_to(750000000ULL);
  require(source.export_snapshot() == restored->export_snapshot() &&
              source.query_progress(handle).at("served").at("vec_eqops") == 500000000000ULL &&
              source.counters().at("compute_busy_ns") == 500000000ULL,
          "blocked work earns neither service nor busy and restores exactly");
  source.cancel(handle, 750000000ULL);
  require(source.drained() && source.query_progress(handle).at("cancelled") == true &&
              source.query_progress(handle).at("served").at("vec_eqops") == 500000000000ULL,
          "cancellation retains only the physically paid half trillion operations");
}

void check_snapshot_guards(const Hardware& hw, const std::shared_ptr<const Plan>& plan,
                           double unit) {
  Runtime source(hw);
  source.submit(plan, owner(1, 1, 1), 0);
  source.advance_to(250000000ULL);
  const auto saved = source.export_snapshot();
  auto restored = Runtime::import_snapshot(hw, saved);
  require(restored->export_snapshot() == saved, "unmodified snapshot restores exactly");
  // Separate from progress assertions, so a failed physical-unit test cannot
  // prevent execution of this independent inventory/Plan-integrity regression.
  for (bool change_capacity : {false, true}) {
    auto changed = saved;
    if (change_capacity) changed["executions"][0]["credit_capacity"][0] = 8 * unit;
    else changed["executions"][0]["credit"][0] = 0;
    bool rejected = false;
    try { (void)Runtime::import_snapshot(hw, changed); }
    catch (const std::invalid_argument&) { rejected = true; }
    require(rejected, "changing physical credit inventory or Plan capacity rejects");
    require(source.export_snapshot() == saved,
            "a rejected import leaves the original world unchanged");
  }
}

void verify() {
  std::array<double, resource_count> rates{};
  rates[COMPUTE_GATE] = 1;
  const auto hw = manual_hardware(rates);
  std::string failures;
  for (int exponent : {0, -10, -60}) for (bool filling : {false, true}) {
    const auto unit = std::ldexp(1.0, exponent);
    const auto plan = fixture_plan(hw, unit, filling);
    for (bool guards : {false, true}) {
      const std::string label = "unit=2^" + std::to_string(exponent) +
                                (filling ? " filling" : " emptying") +
                                (guards ? " snapshot guards" : " physical progress");
      try {
        if (guards) check_snapshot_guards(hw, plan, unit);
        else check_progress(hw, plan);
        std::cout << "PASS " << label << '\n';
      } catch (const std::exception& error) {
        std::cerr << "FAIL " << label << ": " << error.what() << '\n';
        failures += label + ": " + error.what() + "; ";
      }
    }
  }
  require(failures.empty(), failures);
}
}

int main() { return run_test("runtime_subunit_credit_test", verify); }
