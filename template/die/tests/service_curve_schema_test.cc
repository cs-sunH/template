#include "test_support.hh"

#include <array>
#include <cmath>
#include <limits>
#include <memory>
#include <string>
#include <vector>

using namespace coarse_die;
using namespace coarse_die_test;

namespace {

Plan bare_plan(const Hardware& hw, std::vector<Epoch> epochs,
               const std::string& name = "service_curve_test") {
  Plan plan;
  plan.task = json{{"name", name}};
  plan.summary = json{{"status", "test_fixture"}};
  plan.hardware_fingerprint = hw.fingerprint;
  plan.epochs = std::move(epochs);
  return plan;
}

Activity delayed_output_activity(std::uint64_t id = 1) {
  Activity output = activity(id, Role::Compute, COMPUTE_GATE, 1.0);
  output.credit_delta = {0.0, 2.0};
  output.credit_curve = {
      CreditPoint{0.0, {0.0, 0.0}},
      CreditPoint{0.5, {0.0, 0.0}},
      CreditPoint{1.0, {0.0, 2.0}}};
  return output;
}

Epoch epoch_with(std::vector<Activity> activities, std::string name = "curve") {
  Epoch epoch;
  epoch.name = std::move(name);
  epoch.activities = std::move(activities);
  epoch.credit_initial = {0.0, 0.0};
  epoch.credit_capacity = {1.0, 2.0};
  return epoch;
}

json expected_legacy_wire(const Plan& plan, bool fractional) {
  const Epoch& epoch = plan.epochs.front();
  const Activity& activity = epoch.activities.front();
  json work = json::array();
  for (double value : activity.work) work.push_back(value);
  json bytes = json::array();
  for (const auto& row : activity.hbm_bytes)
    bytes.push_back(json::array({row[0], row[1]}));
  json activity_json{{"id", activity.id},
                     {"role", "compute"},
                     {"work", std::move(work)},
                     {"hbm_bytes", std::move(bytes)},
                     {"int4_ops", activity.int4_ops},
                     {"vec_eqops", activity.vec_eqops},
                     {"canonical_q", activity.canonical_q},
                     {"private_time_s", activity.private_time_s},
                     {"credit_delta", {activity.credit_delta[0],
                                        activity.credit_delta[1]}}};
  if (fractional) {
    activity_json["vec_eqops_fraction"] =
        json{{"significand", 1}, {"exponent", -1}};
  }
  json payload{{"schema", fractional ? "coarse_die_plan_v3"
                                    : "coarse_die_plan_v2"},
               {"task", plan.task},
               {"summary", plan.summary},
               {"hardware_fingerprint", plan.hardware_fingerprint},
               {"launch_ns", plan.launch_ns},
               {"epochs", json::array({
                   json{{"name", epoch.name},
                        {"activities", json::array({activity_json})},
                        {"credit_initial", epoch.credit_initial},
                        {"credit_capacity", epoch.credit_capacity},
                        {"repeat_count", epoch.repeat_count}}})}};
  payload["fingerprint"] = stable_hash(payload.dump());
  return payload;
}

void delayed_curve_queries_use_right_segment_at_breakpoints() {
  const Activity output = delayed_output_activity();
  require(credit_at(output, 0.0) == std::array<double, 2>{0.0, 0.0},
          "curve starts with zero cumulative credit");
  require(credit_at(output, 0.5) == std::array<double, 2>{0.0, 0.0},
          "delayed output gate produces no credit before its threshold");
  require(credit_at(output, 0.75) == std::array<double, 2>{0.0, 1.0},
          "credit interpolation follows the segment after the gate");
  require(credit_at(output, 1.0) == output.credit_delta,
          "curve endpoint conserves the declared credit delta");
  require(credit_slope(output, 0.25) == std::array<double, 2>{0.0, 0.0},
          "slope before the delayed gate is zero");
  require(credit_slope(output, 0.5) == std::array<double, 2>{0.0, 4.0},
          "slope at the internal point selects the right segment");
  require(credit_slope(output, 1.0) == std::array<double, 2>{0.0, 4.0},
          "endpoint slope uses the final segment");
  require(next_credit_breakpoint(output, 0.25) == 0.5 &&
              next_credit_breakpoint(output, 0.5) == 1.0 &&
              next_credit_breakpoint(output, 1.0) == 1.0,
          "breakpoint lookup is strict and falls back to one");

  Activity legacy = activity(2, Role::Compute, COMPUTE_GATE, 1.0);
  legacy.credit_delta = {-2.0, 4.0};
  require(credit_at(legacy, 0.25) == std::array<double, 2>{-0.5, 1.0} &&
              credit_slope(legacy, 0.25) == legacy.credit_delta &&
              next_credit_breakpoint(legacy, 0.25) == 1.0,
          "empty curves preserve legacy linear credit_delta behavior");
}

void v4_curve_roundtrip_and_fractional_work() {
  const Hardware hw = example_hardware();
  Activity output = delayed_output_activity();
  output.vec_eqops = 7;
  output.vec_eqops_fraction = 0.5L;
  Plan plan = bare_plan(hw, {epoch_with({output})});
  const json wire = plan_to_json(plan);
  require(wire.at("schema") == "coarse_die_plan_v4" &&
              wire.at("epochs").at(0).at("activities").at(0)
                      .at("credit_curve").size() == 3 &&
              wire.at("epochs").at(0).at("activities").at(0)
                      .contains("vec_eqops_fraction"),
          "curve plans use v4 and retain fractional work");
  const auto restored = plan_from_json(wire);
  require(restored->fingerprint == wire.at("fingerprint").get<std::string>() &&
              plan_to_json(*restored) == wire &&
              restored->epochs[0].activities[0].credit_curve.size() == 3,
          "v4 curve payload round-trips canonically");
}

void v2_and_v3_wire_remain_unchanged() {
  const Hardware hw = example_hardware();
  for (bool fractional : {false, true}) {
    Activity compute = activity(7, Role::Compute, COMPUTE_GATE, 2.0);
    if (fractional) {
      compute.vec_eqops = 5;
      compute.vec_eqops_fraction = 0.5L;
    }
    Plan plan = bare_plan(hw, {epoch_with({compute}, "legacy")},
                          fractional ? "legacy_v3" : "legacy_v2");
    const json actual = plan_to_json(plan);
    const json expected = expected_legacy_wire(plan, fractional);
    require(actual.dump() == expected.dump(),
            fractional ? "legacy v3 wire remains byte-for-byte unchanged"
                       : "legacy v2 wire remains byte-for-byte unchanged");
  }
}

void invalid_curves_and_old_schema_curves_are_rejected() {
  const Activity valid = delayed_output_activity();
  const auto rejected = [](const Activity& candidate) {
    try {
      detail::validate_credit_curve(candidate);
    } catch (const std::invalid_argument&) {
      return true;
    }
    return false;
  };

  Activity too_short = valid;
  too_short.credit_curve.resize(1);
  require(rejected(too_short), "one-point curves are rejected");
  Activity too_long = valid;
  too_long.credit_curve.assign(18, CreditPoint{});
  require(rejected(too_long), "curves above 17 points are rejected");
  Activity wrong_start = valid;
  wrong_start.credit_curve.front().cumulative[1] = 1.0;
  require(rejected(wrong_start), "curves must start at zero cumulative credit");
  Activity wrong_end = valid;
  wrong_end.credit_curve.back().cumulative[1] = 3.0;
  require(rejected(wrong_end), "curves must end at credit_delta");
  Activity duplicate_progress = valid;
  duplicate_progress.credit_curve[1].progress = 0.0;
  require(rejected(duplicate_progress), "curve progress must strictly increase");
  Activity nonfinite_progress = valid;
  nonfinite_progress.credit_curve[1].progress =
      std::numeric_limits<double>::quiet_NaN();
  require(rejected(nonfinite_progress), "NaN progress is rejected");
  Activity nonfinite_credit = valid;
  nonfinite_credit.credit_curve[1].cumulative[1] =
      std::numeric_limits<double>::infinity();
  require(rejected(nonfinite_credit), "non-finite cumulative credit is rejected");
  Activity wrong_direction = valid;
  wrong_direction.credit_curve[1].cumulative[1] = 3.0;
  require(rejected(wrong_direction), "credit must be monotone in delta direction");
  Activity zero_delta = valid;
  zero_delta.credit_delta[1] = 0.0;
  zero_delta.credit_curve[2].cumulative[1] = 0.0;
  zero_delta.credit_curve[1].cumulative[1] = 0.25;
  require(rejected(zero_delta), "zero-delta channels remain zero throughout");

  const Hardware hw = example_hardware();
  for (bool fractional : {false, true}) {
    Activity legacy = activity(3, Role::Compute, COMPUTE_GATE, 1.0);
    if (fractional) legacy.vec_eqops_fraction = 0.5L;
    Plan plan = bare_plan(hw, {epoch_with({legacy})}, "old_schema_curve");
    json wire = plan_to_json(plan);
    wire["epochs"][0]["activities"][0]["credit_curve"] = {
        json{{"progress", 0.0}, {"cumulative", {0.0, 0.0}}},
        json{{"progress", 1.0}, {"cumulative", {0.0, 0.0}}}};
    bool threw = false;
    try {
      (void)plan_from_json(wire);
    } catch (const std::invalid_argument&) {
      threw = true;
    }
    require(threw, "v2/v3 schemas reject a credit_curve field");
  }

  Activity output = delayed_output_activity();
  Plan v4 = bare_plan(hw, {epoch_with({output})}, "malformed_v4_curve");
  json malformed = plan_to_json(v4);
  malformed["epochs"][0]["activities"][0]["credit_curve"][1]["unknown"] = 4;
  bool unknown_point_field_rejected = false;
  try {
    (void)plan_from_json(malformed);
  } catch (const std::invalid_argument&) {
    unknown_point_field_rejected = true;
  }
  require(unknown_point_field_rejected,
          "curve point schema rejects unknown fields");
}

void independent_breakpoints_consume_independent_budget() {
  const Hardware hw = example_hardware();
  Activity first = delayed_output_activity(1);
  Activity second = delayed_output_activity(2);
  Plan small = bare_plan(hw, {epoch_with({first, second})});
  require(service_description_count(small) == 4,
          "independent activities include their own internal and terminal switches");

  std::vector<Epoch> epochs;
  for (std::size_t i = 0; i < 64; ++i) {
    Epoch epoch;
    epoch.name = "budget_" + std::to_string(i);
    if (i == 0) epoch.activities.push_back(delayed_output_activity());
    epochs.push_back(std::move(epoch));
  }
  Plan over_budget = bare_plan(hw, std::move(epochs), "over_budget");
  require(service_description_count(over_budget) == 65,
          "description count includes empty closures and every independent curve segment");
  bool threw = false;
  try {
    detail::validate_plan_structure(over_budget);
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  require(threw, "combined service description count is capped at 64");
}

}  // namespace

int main() {
  return run_test("service_curve_schema_test", [] {
    delayed_curve_queries_use_right_segment_at_breakpoints();
    v4_curve_roundtrip_and_fractional_work();
    v2_and_v3_wire_remain_unchanged();
    invalid_curves_and_old_schema_curves_are_rejected();
    independent_breakpoints_consume_independent_budget();
  });
}
