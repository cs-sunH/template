#include "test_support.hh"

#include <array>
#include <iomanip>
#include <iostream>
#include <map>
#include <string>
#include <vector>

using namespace coarse_die;
using namespace coarse_die_test;

namespace {
std::size_t resource_id(const std::string& name) {
  if (name == "COMPUTE_GATE") return COMPUTE_GATE;
  if (name == "HBM0") return HBM0;
  if (name == "HBM1") return HBM1;
  throw std::runtime_error("unexpected resource " + name);
}

std::shared_ptr<const Plan> make_plan(const Hardware& hw, const json& input,
                                      int owner_id) {
  Epoch epoch;
  epoch.name = "synthetic_three_stage_overlap";
  epoch.credit_initial = {{0.0, 0.0}};
  epoch.credit_capacity = {{1.0, 1.0}};
  epoch.repeat_count = 1;
  for (const auto& descriptor : input.at("activities")) {
    if (descriptor.at("owner").get<int>() != owner_id) continue;
    Activity activity;
    activity.id = descriptor.at("id").get<std::uint64_t>();
    const std::string stage = descriptor.at("stage").get<std::string>();
    if (stage == "L") activity.role = Role::Load;
    else if (stage == "C") activity.role = Role::Compute;
    else if (stage == "S") activity.role = Role::Store;
    else throw std::runtime_error("unexpected stage " + stage);

    if (stage == "C") {
      activity.work[COMPUTE_GATE] = descriptor.at("compute").get<double>();
      activity.private_time_s = descriptor.at("private_time").is_string()
          ? std::stod(descriptor.at("private_time").get<std::string>().substr(
                0, descriptor.at("private_time").get<std::string>().find('/'))) /
                std::stod(descriptor.at("private_time").get<std::string>().substr(
                descriptor.at("private_time").get<std::string>().find('/') + 1))
          : descriptor.at("private_time").get<double>();
      activity.canonical_q = canonical_basis(activity);
    } else {
      const std::uint64_t row0 = descriptor.at("row0").get<std::uint64_t>();
      const std::uint64_t row1 = descriptor.at("row1").get<std::uint64_t>();
      activity.work[HBM0] = static_cast<double>(row0);
      activity.work[HBM1] = static_cast<double>(row1);
      const std::size_t direction = stage == "L" ? 0 : 1;
      activity.hbm_bytes[0][direction] = row0;
      activity.hbm_bytes[1][direction] = row1;
      activity.canonical_q = canonical_basis(activity);
    }
    epoch.activities.push_back(activity);
  }
  if (epoch.activities.empty()) throw std::runtime_error("owner has no activities");
  return manual_plan(hw, {std::move(epoch)}, "grouped_rate_" +
                      input.at("name").get<std::string>() + "_owner_" +
                      std::to_string(owner_id));
}
}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc != 2) throw std::runtime_error("usage: runtime_rates CASES_JSON");
    const json all = read_json(argv[1]);
    std::cout << std::setprecision(17);
    for (const auto& scenario : all.at("cases")) {
      std::array<double, resource_count> capacities{};
      for (auto it = scenario.at("capacity").begin();
           it != scenario.at("capacity").end(); ++it)
        capacities[resource_id(it.key())] = it.value().get<double>();
      const Hardware hw = manual_hardware(capacities);
      Runtime runtime(hw);
      std::map<int, std::uint64_t> handles;
      for (const auto& descriptor : scenario.at("activities")) {
        const int owner_id = descriptor.at("owner").get<int>();
        if (handles.find(owner_id) != handles.end()) continue;
        const auto plan = make_plan(hw, scenario, owner_id);
        const std::uint64_t key = static_cast<std::uint64_t>(owner_id + 1);
        handles[owner_id] = runtime.submit(
            plan, owner(key, key, key), 0);
      }

      json record = {{"case", scenario.at("name")}, {"capacity", scenario.at("capacity")},
                     {"activities", json::array()}};
      for (const auto& [owner_id, handle] : handles) {
        const json progress = runtime.query_progress(handle);
        for (const auto& row : progress.at("activities")) {
          record["activities"].push_back({
              {"owner", owner_id},
              {"id", row.at("activity_id")},
              {"stage", row.at("role") == "load" ? "L" :
                         row.at("role") == "compute" ? "C" : "S"},
              {"q", row.at("canonical_q")},
              {"rate", row.at("rate")},
              {"alpha", row.at("alpha")},
              {"status", row.at("status")}});
        }
      }
      std::cout << record.dump() << '\n';
    }
  } catch (const std::exception& error) {
    std::cerr << "runtime_rates: " << error.what() << '\n';
    return 1;
  }
  return 0;
}
