#include "coarse_die/Execution.hh"

#include <array>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using coarse_die::Activity;
using coarse_die::Epoch;
using coarse_die::Hardware;
using coarse_die::Owner;
using coarse_die::Plan;
using coarse_die::Role;
using coarse_die::Runtime;
using Json = nlohmann::json;

constexpr double kOtherCapacity = 1.0e30;
constexpr const char* kFixtureScope =
    "synthetic Runtime allocator fixture; not a compiler certification";

const std::array<const char*, coarse_die::resource_count> kResourceNames = {
    "COMPUTE_GATE", "SRAM_READ", "SRAM_WRITE", "LOCAL", "HBM0",
    "HBM1", "INJECT0", "INJECT1", "XPOS", "XNEG", "YPOS", "YNEG",
    "HOP"};

struct ActivitySpec {
  std::uint64_t id = 0;
  Role role = Role::Compute;
  unsigned hbm_row = 0;
  std::uint64_t physical_bytes = 0;
  double local_work = 0.0;
  double sram_read_work = 0.0;
  double sram_write_work = 0.0;
  double private_time_s = 0.0;
};

struct OwnerSpec {
  std::uint64_t node_id = 0;
  std::uint64_t execution_id = 0;
  std::vector<ActivitySpec> activities;
};

struct Scenario {
  std::string name;
  std::array<double, coarse_die::resource_count> peak_capacities{};
  std::vector<OwnerSpec> owners;
  bool set_hbm0_capacity_to_zero_after_submit = false;
  std::uint64_t expected_zero_rate_activity_id = 0;
};

ActivitySpec compute(std::uint64_t id, double local_work,
                     double sram_read_work = 0.0,
                     double sram_write_work = 0.0,
                     double private_time_s = 0.0) {
  ActivitySpec out;
  out.id = id;
  out.role = Role::Compute;
  out.local_work = local_work;
  out.sram_read_work = sram_read_work;
  out.sram_write_work = sram_write_work;
  out.private_time_s = private_time_s;
  return out;
}

ActivitySpec load(std::uint64_t id, unsigned row, std::uint64_t bytes,
                  double local_work = 0.0, double sram_read_work = 0.0,
                  double sram_write_work = 0.0) {
  ActivitySpec out;
  out.id = id;
  out.role = Role::Load;
  out.hbm_row = row;
  out.physical_bytes = bytes;
  out.local_work = local_work;
  out.sram_read_work = sram_read_work;
  out.sram_write_work = sram_write_work;
  return out;
}

std::array<double, coarse_die::resource_count> capacities(double gate,
                                                          double hbm0,
                                                          double hbm1) {
  std::array<double, coarse_die::resource_count> rates{};
  rates.fill(kOtherCapacity);
  rates[coarse_die::COMPUTE_GATE] = gate;
  rates[coarse_die::HBM0] = hbm0;
  rates[coarse_die::HBM1] = hbm1;
  return rates;
}

std::vector<Scenario> scenarios() {
  Scenario hbm_zero{
      "zero_rate_hbm_activity_excluded_from_owner_denominator",
      capacities(10.0, 24.0, 18.0),
      {{1, 401, {load(41, 0, 128), compute(42, 80.0)}},
       {2, 402, {compute(43, 160.0)}}},
      true, 41};
  return {
      {"local_precedes_sram_for_compute_q", capacities(12.0, 20.0, 16.0),
       {{1, 101, {compute(11, 64.0, 512.0, 256.0)}},
        {2, 102, {compute(21, 128.0, 32.0, 16.0)}}}},
      {"sram_sum_is_fallback_q", capacities(10.0, 22.0, 14.0),
       {{1, 201, {compute(31, 0.0, 32.0, 16.0)}},
        {2, 202, {compute(41, 0.0, 96.0, 32.0)}}}},
      {"hbm_payload_precedes_local_and_sram", capacities(9.0, 24.0, 20.0),
       {{1, 301, {load(51, 0, 128, 256.0, 512.0), compute(52, 32.0)}},
        {2, 302, {load(61, 1, 256, 64.0, 128.0), compute(62, 96.0)}}}},
      std::move(hbm_zero),
  };
}

const char* role_name(Role role) {
  switch (role) {
    case Role::Load: return "load";
    case Role::Compute: return "compute";
    case Role::Store: return "store";
    case Role::Endpoint: return "endpoint";
  }
  return "unknown";
}

Activity make_activity(const ActivitySpec& spec) {
  Activity out;
  out.id = spec.id;
  out.role = spec.role;
  out.private_time_s = spec.private_time_s;
  out.credit_delta = {0.0, 0.0};
  if (spec.role == Role::Compute) {
    out.work[coarse_die::COMPUTE_GATE] = 1.0;
  } else {
    if (spec.hbm_row > 1 || spec.physical_bytes == 0) {
      throw std::runtime_error("invalid synthetic HBM stage");
    }
    const std::size_t resource = spec.hbm_row == 0 ? coarse_die::HBM0 : coarse_die::HBM1;
    out.work[resource] = static_cast<double>(spec.physical_bytes);
    out.hbm_bytes[spec.hbm_row][spec.role == Role::Load ? 0 : 1] = spec.physical_bytes;
  }
  out.work[coarse_die::LOCAL] = spec.local_work;
  out.work[coarse_die::SRAM_READ] = spec.sram_read_work;
  out.work[coarse_die::SRAM_WRITE] = spec.sram_write_work;
  // Use the production Plan.hh priority helper; do not hand-assign Q.
  out.canonical_q = coarse_die::canonical_basis(out);
  return out;
}

const char* q_basis(const Activity& activity) {
  long double hbm_payload = 0.0L;
  for (const auto& row : activity.hbm_bytes) {
    for (std::uint64_t bytes : row) hbm_payload += bytes;
  }
  if (hbm_payload > 0.0L) return "HBM_payload_bytes";
  if (activity.work[coarse_die::LOCAL] > 0.0) return "LOCAL_bytes";
  if (activity.work[coarse_die::SRAM_READ] + activity.work[coarse_die::SRAM_WRITE] > 0.0) {
    return "SRAM_READ_plus_SRAM_WRITE_bytes";
  }
  if (activity.work[coarse_die::COMPUTE_GATE] > 0.0) return "pure_compute_Q1";
  return "fallback_Q1";
}

Json activity_definition(const ActivitySpec& spec) {
  const Activity activity = make_activity(spec);
  Json work = Json::object();
  for (std::size_t resource = 0; resource < coarse_die::resource_count; ++resource) {
    if (activity.work[resource] > 0.0) work[kResourceNames[resource]] = activity.work[resource];
  }
  Json payload = Json::array({Json::array({0, 0}), Json::array({0, 0})});
  if (spec.role != Role::Compute) {
    payload[spec.hbm_row][spec.role == Role::Load ? 0 : 1] = spec.physical_bytes;
  }
  return Json{{"activity_id", spec.id},
              {"role", role_name(spec.role)},
              {"stage_physical_bytes", spec.role == Role::Compute ? Json(nullptr) : Json(spec.physical_bytes)},
              {"hbm_row", spec.role == Role::Compute ? Json(nullptr) : Json(spec.hbm_row)},
              {"resource_work", std::move(work)},
              {"canonical_q", activity.canonical_q},
              {"canonical_q_basis", q_basis(activity)},
              {"private_time_s", spec.private_time_s},
              {"credit_delta", {0.0, 0.0}},
              {"hbm_payload_bytes_by_row_direction", std::move(payload)}};
}

std::shared_ptr<const Plan> make_plan(const Hardware& hardware,
                                      const Scenario& scenario,
                                      const OwnerSpec& owner) {
  auto plan = std::make_shared<Plan>();
  plan->task = Json{{"schema", "synthetic_runtime_q_priority_fixture_v1"},
                    {"scenario", scenario.name},
                    {"owner_node_id", owner.node_id},
                    {"owner_execution_id", owner.execution_id}};
  plan->summary = Json{{"status", "synthetic_runtime_allocator_rate_fixture"},
                       {"fixture_scope", kFixtureScope}};
  plan->hardware_fingerprint = hardware.fingerprint;
  Epoch epoch;
  epoch.name = scenario.name;
  for (const ActivitySpec& spec : owner.activities) {
    epoch.activities.push_back(make_activity(spec));
  }
  plan->epochs.push_back(std::move(epoch));
  plan->fingerprint = coarse_die::plan_to_json(*plan).at("fingerprint").get<std::string>();
  return plan;
}

Hardware fixture_hardware(const std::string& config_path,
                          const std::array<double, coarse_die::resource_count>& rates) {
  std::ifstream input(config_path);
  if (!input) throw std::runtime_error("cannot read hardware config: " + config_path);
  Json config;
  input >> config;
  Hardware hardware = coarse_die::resolve_hardware(config);
  Json& resources = hardware.derived.at("resources");
  if (resources.size() != coarse_die::resource_count) {
    throw std::runtime_error("resolved fixture hardware has unexpected resource count");
  }
  for (std::size_t i = 0; i < rates.size(); ++i) {
    if (!std::isfinite(rates[i]) || rates[i] <= 0.0) {
      throw std::runtime_error("fixture capacity must be finite and positive");
    }
    // This synthetic allocator fixture makes every runtime domain available;
    // non-finite-in-practice domains receive 1e30 capacity and no activities
    // use them. This is not a physical topology or compiler claim.
    hardware.rates[i] = rates[i];
    resources[i]["enabled"] = true;
    resources[i]["rate"] = rates[i];
  }
  return hardware;
}

Json run_scenario(const Scenario& scenario, const std::string& config_path) {
  if (scenario.owners.size() != 2) {
    throw std::runtime_error("every fixture scenario must have exactly two owners");
  }
  Hardware hardware = fixture_hardware(config_path, scenario.peak_capacities);
  Runtime runtime(std::move(hardware));

  Json owner_definitions = Json::array();
  std::vector<std::pair<const OwnerSpec*, std::uint64_t>> handles;
  std::uint64_t request_id = 1;
  for (const OwnerSpec& owner_spec : scenario.owners) {
    if (owner_spec.activities.empty() || owner_spec.activities.size() > 3) {
      throw std::runtime_error("owner activity count must be between one and three");
    }
    Owner owner;
    owner.request_id = request_id++;
    owner.node_id = owner_spec.node_id;
    owner.execution_id = owner_spec.execution_id;
    owner.generation = 1;
    const std::uint64_t handle = runtime.submit(
        make_plan(runtime.hardware(), scenario, owner_spec), owner, 0);
    handles.emplace_back(&owner_spec, handle);

    Json activity_list = Json::array();
    for (const ActivitySpec& spec : owner_spec.activities) {
      activity_list.push_back(activity_definition(spec));
    }
    owner_definitions.push_back({{"node_id", owner.node_id},
                                 {"execution_id", owner.execution_id},
                                 {"generation", owner.generation},
                                 {"activities", std::move(activity_list)}});
  }

  Json capacity_updates = Json::array();
  if (scenario.set_hbm0_capacity_to_zero_after_submit) {
    runtime.update_capacity(coarse_die::HBM0, 0.0, 0);
    capacity_updates.push_back({{"resource", "HBM0"}, {"rate", 0.0}, {"at_ns", 0}});
  }
  const Json counters = runtime.counters();

  Json owner_rates = Json::array();
  bool saw_expected_zero_rate = false;
  for (const auto& [owner_spec, handle] : handles) {
    const Json progress = runtime.query_progress(handle);
    if (progress.at("state").get<std::string>() != "running") {
      throw std::runtime_error("submitted fixture execution is not running at time zero");
    }
    const Json& actual_activities = progress.at("activities");
    if (actual_activities.size() != owner_spec->activities.size()) {
      throw std::runtime_error("query_progress returned unexpected activity count");
    }
    Json rates = Json::array();
    for (std::size_t i = 0; i < actual_activities.size(); ++i) {
      const Json& actual = actual_activities[i];
      const ActivitySpec& definition = owner_spec->activities[i];
      const double rate = actual.at("rate").get<double>();
      const double alpha = actual.at("alpha").get<double>();
      const std::uint64_t id = actual.at("activity_id").get<std::uint64_t>();
      const bool is_expected_zero =
          scenario.set_hbm0_capacity_to_zero_after_submit &&
          id == scenario.expected_zero_rate_activity_id;
      if (is_expected_zero) {
        if (rate != 0.0 || alpha != 0.0 ||
            actual.at("status").get<std::string>() != "blocked") {
          throw std::runtime_error("zero-capacity HBM activity was not blocked/excluded");
        }
        saw_expected_zero_rate = true;
      } else if (!std::isfinite(rate) || rate <= 0.0 ||
                 !std::isfinite(alpha) || alpha <= 0.0 ||
                 actual.at("status").get<std::string>() != "running") {
        throw std::runtime_error("Runtime query_progress did not report a positive active rate/alpha");
      }
      rates.push_back({{"activity_id", actual.at("activity_id")},
                       {"status", actual.at("status")},
                       {"actual_rate", actual.at("rate")},
                       {"rate_unit", actual.at("rate_unit")},
                       {"alpha", actual.at("alpha")},
                       {"canonical_q", actual.at("canonical_q")},
                       {"definition_canonical_q", make_activity(definition).canonical_q}});
    }
    owner_rates.push_back({{"node_id", owner_spec->node_id},
                           {"execution_id", owner_spec->execution_id},
                           {"handle", handle},
                           {"activities", std::move(rates)}});
  }
  if (scenario.set_hbm0_capacity_to_zero_after_submit && !saw_expected_zero_rate) {
    throw std::runtime_error("expected zero-rate HBM activity was not present");
  }

  Json zero_rate_exclusion = nullptr;
  if (scenario.set_hbm0_capacity_to_zero_after_submit) {
    if (counters.at("active_owners").get<std::size_t>() != 2 ||
        counters.at("active_activities").get<std::size_t>() != 2) {
      throw std::runtime_error("zero-rate HBM activity should leave two active compute activities");
    }
    Json active_alpha_sums = Json::array();
    for (const Json& owner : owner_rates) {
      double sum = 0.0;
      for (const Json& activity : owner.at("activities")) {
        if (activity.at("status").get<std::string>() == "running") {
          sum += activity.at("alpha").get<double>();
        }
      }
      if (std::abs(sum - 1.0) > 1.0e-12) {
        throw std::runtime_error("zero-rate activity remained in its owner's active alpha denominator");
      }
      active_alpha_sums.push_back({{"node_id", owner.at("node_id")},
                                   {"active_alpha_sum", sum}});
    }
    zero_rate_exclusion = Json{{"status", "passed"},
                               {"disabled_activity_id", scenario.expected_zero_rate_activity_id},
                               {"disabled_activity_alpha", 0.0},
                               {"active_owner_alpha_sums", std::move(active_alpha_sums)},
                               {"active_activity_count_after_exclusion", 2}};
  }

  Json peak_capacities = Json::object();
  for (std::size_t i = 0; i < scenario.peak_capacities.size(); ++i) {
    peak_capacities[kResourceNames[i]] = scenario.peak_capacities[i];
  }
  Json current_capacities = Json::object();
  const Json& current_array = counters.at("capacity");
  if (!current_array.is_array() || current_array.size() != coarse_die::resource_count) {
    throw std::runtime_error("Runtime counters returned an invalid capacity vector");
  }
  for (std::size_t i = 0; i < coarse_die::resource_count; ++i) {
    current_capacities[kResourceNames[i]] = current_array.at(i);
  }
  return Json{{"scenario", scenario.name},
              {"fixture_scope", kFixtureScope},
              {"credit_boundary_limits", false},
              {"credit_delta_policy", "all activities have [0,0]; signed-credit constraints are inactive"},
              {"peak_capacities", std::move(peak_capacities)},
              {"capacity_updates", std::move(capacity_updates)},
              {"current_runtime_capacities", std::move(current_capacities)},
              {"active_owner_count", counters.at("active_owners")},
              {"active_activity_count", counters.at("active_activities")},
              {"zero_rate_activity_exclusion_check", std::move(zero_rate_exclusion)},
              {"owner_definitions", std::move(owner_definitions)},
              {"actual_query_progress_rates", std::move(owner_rates)}};
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc != 2) {
      throw std::runtime_error("usage: grouped_allocator_driver <hardware-config.json>");
    }
    Json results = Json::array();
    for (const Scenario& scenario : scenarios()) {
      results.push_back(run_scenario(scenario, argv[1]));
    }
    std::cout << Json{{"schema", "synthetic_grouped_allocator_runtime_q_priority_v1"},
                      {"source_scope", kFixtureScope},
                      {"runtime_entrypoint", "coarse_die::Runtime::submit + Runtime::query_progress"},
                      {"q_derivation", "coarse_die::canonical_basis(Activity) from Plan.hh"},
                      {"reported_rate_key", "query_progress.activities[].rate"},
                      {"reported_alpha_key", "query_progress.activities[].alpha"},
                      {"rate_unit_key", "query_progress.activities[].rate_unit"},
                      {"scenario_count", results.size()},
                      {"scenarios", std::move(results)}}.dump(2) << '\n';
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "grouped allocator driver failed: " << error.what() << '\n';
    return 1;
  }
}
