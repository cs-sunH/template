#pragma once

#include "coarse_die/Execution.hh"

#include <cmath>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>

namespace coarse_die_test {
using nlohmann::json;

inline void require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}

inline void require_close(double actual, double expected, double tolerance,
                          const std::string& message) {
  if (!std::isfinite(actual) || std::abs(actual - expected) > tolerance) {
    throw std::runtime_error(message + ": actual=" + std::to_string(actual) +
                             " expected=" + std::to_string(expected) +
                             " tolerance=" + std::to_string(tolerance));
  }
}

inline std::string read_text(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("cannot open " + path);
  return std::string(std::istreambuf_iterator<char>(in),
                     std::istreambuf_iterator<char>());
}

inline json read_json(const std::string& path) {
  return json::parse(read_text(path));
}

inline std::string fixture(const std::string& name) {
  return std::string(COARSE_DIE_SOURCE_DIR) + "/tests/fixtures/" + name;
}

inline std::string source_doc(const std::string& name) {
  return std::string(COARSE_DIE_ROOT_DIR) + "/" + name;
}

inline coarse_die::Hardware manual_hardware(
    const std::array<double, coarse_die::resource_count>& rates) {
  const auto config = read_json(std::string(COARSE_DIE_SOURCE_DIR) +
                                "/configs/example_hardware.json");
  coarse_die::Hardware hw = coarse_die::resolve_hardware(config);
  hw.rates = rates;
  static const std::array<const char*, coarse_die::resource_count> ids = {
      "COMPUTE_GATE", "SRAM_READ", "SRAM_WRITE", "LOCAL", "HBM0",
      "HBM1", "INJECT0", "INJECT1", "XPOS", "XNEG", "YPOS", "YNEG",
      "HOP"};
  auto& resources = hw.derived.at("resources");
  for (std::size_t i = 0; i < rates.size(); ++i) {
    require(std::isfinite(rates[i]) && rates[i] >= 0.0,
            "manual hardware rates must be finite and nonnegative");
    require(resources.at(i).at("enabled").get<bool>() || rates[i] == 0.0,
            "manual fixture cannot enable a hardware-disabled resource");
    resources[i]["resource_id"] = i;
    resources[i]["id"] = ids[i];
    resources[i]["rate"] = rates[i];
  }
  return hw;
}

inline coarse_die::Hardware example_hardware() {
  return coarse_die::resolve_hardware(read_json(
      std::string(COARSE_DIE_SOURCE_DIR) + "/configs/example_hardware.json"));
}

inline coarse_die::Activity activity(std::uint64_t id,
                                     coarse_die::Role role,
                                     std::size_t resource,
                                     double work) {
  coarse_die::Activity a;
  a.id = id;
  a.role = role;
  a.work[resource] = work;
  return a;
}

inline std::shared_ptr<const coarse_die::Plan> manual_plan(
    const coarse_die::Hardware& hw, std::vector<coarse_die::Epoch> epochs,
    std::string name = "test_plan") {
  auto plan = std::make_shared<coarse_die::Plan>();
  plan->task = json{{"name", name}};
  plan->summary = json{{"status", "test_fixture"}};
  plan->hardware_fingerprint = hw.fingerprint;
  plan->epochs = std::move(epochs);
  // Runtime snapshots serialize plans through the same canonical public plan
  // format as compiled work. Give synthetic plans that canonical fingerprint
  // too, so a snapshot fork exercises real serialization and hash validation.
  plan->fingerprint = coarse_die::plan_to_json(*plan).at("fingerprint").get<std::string>();
  return plan;
}

inline coarse_die::Owner owner(std::uint64_t request, std::uint64_t node,
                               std::uint64_t execution,
                               std::uint64_t generation = 1) {
  coarse_die::Owner out;
  out.request_id = request;
  out.node_id = node;
  out.execution_id = execution;
  out.generation = generation;
  return out;
}

inline std::uint64_t time_from_progress(const json& p, const char* key) {
  if (!p.contains(key) || p.at(key).is_null())
    throw std::runtime_error(std::string("progress lacks ") + key);
  return p.at(key).get<std::uint64_t>();
}

inline int run_test(const char* test_name, void (*body)()) {
  try {
    body();
    std::cout << "PASS " << test_name << "\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "FAIL " << test_name << ": " << e.what() << "\n";
    return 1;
  }
}
}  // namespace coarse_die_test
