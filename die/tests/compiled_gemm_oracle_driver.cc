#include "coarse_die/Execution.hh"

#include <array>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>

#ifndef COARSE_DIE_SOURCE_DIR
#error "COARSE_DIE_SOURCE_DIR must point at die/"
#endif

namespace {
using coarse_die::Activity;
using coarse_die::Hardware;
using coarse_die::Owner;
using coarse_die::Plan;
using coarse_die::Role;
using coarse_die::Runtime;
using coarse_die::compile;
using coarse_die::resource_count;
using json = nlohmann::json;

constexpr std::array<const char*, resource_count> kResourceNames = {
    "COMPUTE_GATE", "SRAM_READ", "SRAM_WRITE", "LOCAL", "HBM0",
    "HBM1", "INJECT0", "INJECT1", "XPOS", "XNEG", "YPOS", "YNEG",
    "HOP"};
constexpr std::array<const char*, 3> kRoleNames = {"load", "compute", "store"};
constexpr std::uint64_t kNsPerSecond = 1000000000ULL;
constexpr double kHbmRowCapacityBps = 50000000.0;
constexpr double kNonBindingCapacity = 1.0e30;
constexpr double kVectorEfficiency = 0.02;

json read_json(const std::string& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) throw std::runtime_error("cannot open " + path);
  return json::parse(input);
}

Hardware synthetic_hardware() {
  json input = read_json(std::string(COARSE_DIE_SOURCE_DIR) +
                         "/configs/example_hardware.json");
  input["architecture"]["core_type"] = "small";
  input["architecture"]["sram_mib_per_core"] = 0.25;
  input["architecture"]["noc_width_bits"] = 64;
  input["architecture"]["core_columns_nx"] = 1;
  input["architecture"]["core_rows_ny"] = 1;
  input["architecture"]["die_geometry"] = "isotropic";
  input["performance_profile"]["eta_vec"] = kVectorEfficiency;
  input["performance_profile"]["eta_hbm_matrix"] = 1.0;
  input["performance_profile"]["granularity_validation"] = "not_run";

  Hardware hardware = coarse_die::resolve_hardware(input);
  const std::string source_fingerprint = hardware.fingerprint;
  for (std::size_t resource = 0; resource < resource_count; ++resource) {
    auto& descriptor = hardware.derived.at("resources").at(resource);
    const bool enabled = descriptor.at("enabled").get<bool>();
    double capacity = 0.0;
    if (enabled) {
      if (resource == coarse_die::HBM0 || resource == coarse_die::HBM1)
        capacity = kHbmRowCapacityBps;
      else if (resource == coarse_die::COMPUTE_GATE)
        capacity = 1.0;
      else
        capacity = kNonBindingCapacity;
    }
    hardware.rates[resource] = capacity;
    descriptor["rate"] = capacity;
  }
  const json fixture_identity = {
      {"kind", "synthetic_model_to_model_compiled_gemm_oracle_v1"},
      {"single_core", true},
      {"dataflow", "output_stationary_v2"},
      {"execution_profile", "fp32_vec_matrix_assumed_v1"},
      {"hbm_row_capacity_Bps", kHbmRowCapacityBps},
      {"compute_gate_capacity", 1.0},
      {"nonbinding_enabled_resource_capacity", kNonBindingCapacity},
      {"eta_vec", kVectorEfficiency},
      {"eta_hbm_matrix", 1.0}};
  hardware.fingerprint = "synthetic-model-to-model:" +
                         coarse_die::stable_hash(source_fingerprint +
                                                 fixture_identity.dump());
  return hardware;
}

json gemm_task() {
  return {{"op_kind", "gemm"},
          {"tensor_shapes", {{"m", 65}, {"n", 65}, {"k", 4097}}},
          {"storage_dtype", {{"weights", "fp32"},
                             {"activations", "fp32"},
                             {"output", "fp32"}}},
          {"execution_profile", "fp32_vec_matrix_assumed_v1"},
          {"layer_instances", 1},
          {"weight_passes", 1}};
}

const char* role_name(Role role) {
  switch (role) {
    case Role::Load: return kRoleNames[0];
    case Role::Compute: return kRoleNames[1];
    case Role::Store: return kRoleNames[2];
    case Role::Endpoint: return "endpoint";
  }
  return "unknown";
}

json activity_json(const Activity& activity) {
  json work = json::array();
  for (double value : activity.work) work.push_back(value);
  json hbm = json::array();
  for (const auto& row : activity.hbm_bytes)
    hbm.push_back({row[0], row[1]});
  return {{"id", activity.id},
          {"role", role_name(activity.role)},
          {"work_by_resource", std::move(work)},
          {"hbm_bytes_by_row_direction", std::move(hbm)},
          {"int4_ops", activity.int4_ops},
          {"vec_eqops", activity.vec_eqops},
          {"canonical_q", activity.canonical_q},
          {"private_time_s", activity.private_time_s},
          {"credit_delta", activity.credit_delta}};
}

json plan_json(const Plan& plan) {
  json epochs = json::array();
  for (const auto& epoch : plan.epochs) {
    json activities = json::array();
    for (const auto& activity : epoch.activities)
      activities.push_back(activity_json(activity));
    epochs.push_back({{"name", epoch.name},
                      {"repeat_count", epoch.repeat_count},
                      {"credit_initial", epoch.credit_initial},
                      {"credit_capacity", epoch.credit_capacity},
                      {"activities", std::move(activities)}});
  }
  return {{"compiler_path", "coarse_die::compile() -> detail::enumerate_mappings()"},
          {"fingerprint", plan.fingerprint},
          {"hardware_fingerprint", plan.hardware_fingerprint},
          {"task", plan.task},
          {"summary", plan.summary},
          {"E", plan.epochs.size()},
          {"epochs", std::move(epochs)}};
}

Owner owner(std::uint64_t id) {
  Owner result;
  result.request_id = id;
  result.node_id = id;
  result.execution_id = id;
  result.generation = 1;
  return result;
}

std::uint64_t advance_until_terminal(Runtime& runtime,
                                     const std::vector<std::uint64_t>& handles) {
  std::size_t events = 0;
  for (;;) {
    bool finished = true;
    for (std::uint64_t handle : handles) {
      if (!runtime.terminal_finish_ns(handle).has_value()) {
        finished = false;
        break;
      }
    }
    if (finished) return events;
    const auto next = runtime.next_event();
    if (!next || *next <= runtime.now_ns())
      throw std::runtime_error("actual Runtime has no advancing event before completion");
    runtime.advance_to(*next);
    if (++events > 1000000)
      throw std::runtime_error("actual Runtime exceeded the fixed event budget");
  }
}

json counters_json(const Runtime& runtime) {
  const json counters = runtime.counters();
  return {{"hbm_busy_ns", counters.at("hbm_busy_ns")},
          {"hbm_shared_ns", counters.at("hbm_shared_ns")},
          {"compute_busy_ns", counters.at("compute_busy_ns")},
          {"row_busy_ns", counters.at("row_busy_ns")},
          {"row_shared_ns", counters.at("row_shared_ns")},
          {"internal_events", counters.at("internal_events")},
          {"allocation_updates", counters.at("allocation_updates")},
          {"hbm_served_bytes_by_row_direction",
           counters.at("hbm_served_bytes_by_row_direction")}};
}

json run_standalone(const Hardware& hardware,
                    const std::shared_ptr<const Plan>& plan) {
  Runtime runtime(hardware);
  const std::uint64_t handle = runtime.submit(plan, owner(1), 0);
  const std::size_t events = advance_until_terminal(runtime, {handle});
  const auto finish = runtime.terminal_finish_ns(handle);
  if (!finish) throw std::runtime_error("standalone Runtime did not finish");
  return {{"request", "solo"},
          {"arrival_ns", 0},
          {"finish_ns", *finish},
          {"finish_s", static_cast<double>(*finish) / kNsPerSecond},
          {"event_advances", events},
          {"counters", counters_json(runtime)}};
}

json run_two_comp(const Hardware& hardware,
                  const std::shared_ptr<const Plan>& plan,
                  std::uint64_t phase_offset_ns) {
  Runtime runtime(hardware);
  const std::uint64_t first = runtime.submit(plan, owner(1), 0);
  runtime.advance_to(phase_offset_ns);
  const std::uint64_t second = runtime.submit(plan, owner(2), phase_offset_ns);
  const std::size_t events = advance_until_terminal(runtime, {first, second});
  const auto finish_first = runtime.terminal_finish_ns(first);
  const auto finish_second = runtime.terminal_finish_ns(second);
  if (!finish_first || !finish_second)
    throw std::runtime_error("2COMP Runtime did not finish both requests");
  return {{"request_names", {"comp0", "comp1"}},
          {"phase_offset_ns", phase_offset_ns},
          {"phase_offset_s", static_cast<double>(phase_offset_ns) / kNsPerSecond},
          {"finish_ns_by_request", {{"comp0", *finish_first},
                                     {"comp1", *finish_second}}},
          {"finish_s_by_request",
           {{"comp0", static_cast<double>(*finish_first) / kNsPerSecond},
            {"comp1", static_cast<double>(*finish_second) / kNsPerSecond}}},
          {"event_advances", events},
          {"counters", counters_json(runtime)}};
}

json synthetic_capacity_json(const Hardware& hardware) {
  json resources = json::array();
  for (std::size_t i = 0; i < resource_count; ++i) {
    const auto& descriptor = hardware.derived.at("resources").at(i);
    resources.push_back({{"resource_id", i},
                         {"id", kResourceNames[i]},
                         {"enabled", descriptor.at("enabled")},
                         {"capacity", hardware.rates[i]}});
  }
  return {{"fixture_kind", "synthetic_model_to_model"},
          {"source_core_table_sha256",
           hardware.derived.at("core_resource_table").at("rows_sha256")},
          {"synthetic_hardware_fingerprint", hardware.fingerprint},
          {"core_count", hardware.derived.at("die").at("core_count")},
          {"core_columns_nx", hardware.derived.at("die").at("core_columns_nx")},
          {"core_rows_ny", hardware.derived.at("die").at("core_rows_ny")},
          {"core_sram_bytes_per_core", hardware.derived.at("core").at("sram_bytes_per_core")},
          {"p_vec_per_core_ops_per_s",
           hardware.derived.at("core").at("P_vec_per_core_ops_per_s")},
          {"rho_sram", hardware.input.at("performance_profile").at("rho_sram")},
          {"eta_vec", hardware.input.at("performance_profile").at("eta_vec")},
          {"eta_hbm_matrix", hardware.input.at("performance_profile").at("eta_hbm_matrix")},
          {"resource_capacities", std::move(resources)}};
}

}  // namespace

int main() {
  try {
    const Hardware hardware = synthetic_hardware();
    const json task = gemm_task();
    const auto plan = compile(hardware, task);
    if (!plan) throw std::runtime_error("compile returned a null GEMM plan");
    const json solo = run_standalone(hardware, plan);
    const std::uint64_t offsets_ns[] = {0, 50000000ULL};
    json concurrent = json::array();
    for (std::uint64_t offset : offsets_ns)
      concurrent.push_back(run_two_comp(hardware, plan, offset));
    std::cout << json{{"schema_version", 1},
                      {"measurement_source", "compiled_plan_and_actual_cpp_runtime"},
                      {"synthetic_hardware", synthetic_capacity_json(hardware)},
                      {"compiled_plan", plan_json(*plan)},
                      {"standalone", solo},
                      {"two_comp", concurrent}}
                     .dump(2)
              << '\n';
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "compiled_gemm_oracle_driver: " << error.what() << '\n';
    return 1;
  }
}
