#include "test_support.hh"
#include "../src/GemmOSProgram.hh"

#include <iostream>
#include <string>
#include <vector>

namespace coarse_die::detail {
std::vector<std::shared_ptr<Plan>> enumerate_mappings(
    const Hardware&, const nlohmann::json&);
}

using namespace coarse_die;
using namespace coarse_die_test;

namespace {

Hardware one_core_hardware() {
  auto config = read_json(std::string(COARSE_DIE_SOURCE_DIR) +
                          "/configs/example_hardware.json");
  config["architecture"]["core_columns_nx"] = 1;
  config["architecture"]["core_rows_ny"] = 1;
  config["architecture"]["die_geometry"] = "isotropic";
  return resolve_hardware(config);
}

nlohmann::json program_task(std::uint64_t m = 3, std::uint64_t n = 5,
                           std::uint64_t k = 7,
                           std::uint64_t budget = 4096) {
  return {{"schema", "coarse_die_task_v2"},
          {"op_kind", "gemm"},
          {"tensor_shapes", {{"m", m}, {"n", n}, {"k", k}}},
          {"storage_dtype", {{"activations", "bf16"},
                              {"weights", "bf16"},
                              {"output", "bf16"}}},
          {"runtime_program", {{"kind", gemm_os_k_readiness_kind},
                               {"max_expanded_waves", budget}}}};
}

nlohmann::json reseal(nlohmann::json wire) {
  wire.erase("fingerprint");
  wire["fingerprint"] = stable_hash(wire.dump());
  return wire;
}

template <class F>
void rejects(F&& action, const std::string& message) {
  try {
    action();
  } catch (const std::exception&) {
    return;
  }
  throw std::runtime_error(message);
}

void body() {
  const Hardware hw = one_core_hardware();

  // The legacy path still emits its pre-Program v2 payload shape and parses it
  // back byte-for-byte. A fractional legacy Plan remains v3 with no Program.
  const auto ordinary = compile(
      hw, {{"op_kind", "gemm"},
           {"tensor_shapes", {{"m", 3}, {"n", 5}, {"k", 7}}}});
  const auto ordinary_wire = plan_to_json(*ordinary);
  require(ordinary_wire.at("schema") == "coarse_die_plan_v2" &&
              !ordinary_wire.contains("program") &&
              plan_to_json(*plan_from_json(ordinary_wire)) == ordinary_wire,
          "ordinary GEMM keeps the exact v2 serialization path");
  Plan fractional;
  fractional.task = {{"name", "legacy_fractional_fixture"}};
  fractional.summary = nlohmann::json::object();
  fractional.hardware_fingerprint = hw.fingerprint;
  Epoch fractional_epoch;
  fractional_epoch.name = "legacy_fractional";
  Activity fractional_activity;
  fractional_activity.id = 1;
  fractional_activity.vec_eqops = 2;
  fractional_activity.vec_eqops_fraction = 0.5L;
  fractional_epoch.activities.push_back(fractional_activity);
  fractional.epochs.push_back(fractional_epoch);
  const auto fractional_wire = plan_to_json(fractional);
  require(fractional_wire.at("schema") == "coarse_die_plan_v3" &&
              !fractional_wire.contains("program") &&
              plan_to_json(*plan_from_json(fractional_wire)) == fractional_wire,
          "fractional legacy Plan keeps the v3 serialization path");

  const auto program = compile(hw, program_task());
  const auto wire = plan_to_json(*program);
  require(wire.at("schema") == "coarse_die_plan_v4" &&
              wire.at("program").at("kind") == gemm_os_k_readiness_kind &&
              program->epochs.empty() && program->gemm_os_program.has_value(),
          "opt-in compiles to an epoch-free tagged Program v4 Plan");
  require(program->summary.at("runtime_model") ==
                  "gemm_os_k_readiness_v1" &&
              program->summary.at("validation_scope") ==
                  "bounded_expanded_diagnostic" &&
              program->summary.at("runtime_complexity") == "O(events)" &&
              program->summary.at("production_efficiency_certified") == false &&
              program->summary.contains("mapping_candidate_estimate_ns") &&
              !program->summary.contains("estimated_latency_ns") &&
              !program->summary.contains("alone_estimate_status"),
          "compile estimates are labeled as mapping-candidate estimates");
  require(detail::make_gemm_os_program(hw, *program) != nullptr,
          "factory reconstructs a valid bounded OS descriptor");
  require(plan_to_json(*plan_from_json(wire)) == wire,
          "Program v4 Plan round-trips canonically");

  const auto public_candidates = detail::enumerate_mappings(hw, program->task);
  require(!public_candidates.empty(),
          "public mapping enumeration returns Program candidates");
  for (const auto& candidate : public_candidates) {
    require(candidate->gemm_os_program.has_value() && candidate->epochs.empty() &&
                candidate->summary.at("dataflow_id") == "output_stationary_v2" &&
                plan_to_json(*candidate).at("schema") == "coarse_die_plan_v4",
            "public opt-in candidates preserve v4 and force OS mapping");
  }

  // Recompute the outer Plan hash to show that imported values are checked
  // against hardware-derived geometry, not trusted because their hash matches.
  auto forged_summary = wire;
  forged_summary["summary"]["hbm_read_bytes_total"] =
      forged_summary["summary"]["hbm_read_bytes_total"].get<std::uint64_t>() + 1;
  auto parsed_forgery = plan_from_json(reseal(std::move(forged_summary)));
  rejects([&] { (void)detail::make_gemm_os_program(hw, *parsed_forgery); },
          "factory rejects a self-consistently hashed but false ledger summary");

  auto budget_mismatch = wire;
  budget_mismatch["task"]["runtime_program"]["max_expanded_waves"] = 4095;
  rejects([&] { (void)plan_from_json(reseal(std::move(budget_mismatch))); },
          "wire Plan rejects disagreement between task and Program budgets");
  auto too_large = program_task();
  too_large["runtime_program"]["max_expanded_waves"] = 4097;
  rejects([&] { (void)compile(hw, too_large); },
          "normalized task rejects a budget above the hard expansion cap");
  rejects([&] { (void)compile(hw, program_task(3, 5, 4097, 1)); },
          "compiler rejects a shape with no candidate inside the wave budget");

  auto repeated = program_task();
  repeated["layer_instances"] = 2;
  rejects([&] { (void)compile(hw, repeated); },
          "Program rejects repeated layer composition");
  auto sharded = program_task();
  sharded["tp_slice"] = {{"tensor_parallel", 2},
                          {"tensor_parallel_rank", 0}};
  rejects([&] { (void)compile(hw, sharded); },
          "Program rejects multi-shard composition");
  rejects([&] { (void)compile(example_hardware(), program_task()); },
          "Program rejects multi-Core hardware");

  auto missing_tag = wire;
  missing_tag["schema"] = "coarse_die_plan_v2";
  missing_tag.erase("program");
  rejects([&] { (void)plan_from_json(reseal(std::move(missing_tag))); },
          "legacy Plan schema cannot carry the Program task marker");
  auto wrong_kind = wire;
  wrong_kind["program"]["kind"] = "other_program";
  rejects([&] { (void)plan_from_json(reseal(std::move(wrong_kind))); },
          "v4 wire rejects unsupported Program kinds");

  std::cout << "PASS: legacy v2/v3 serialization, bounded OS-only v4 schema, "
               "wire/facts rejection, and mapping-candidate estimate labels\n";
}

}  // namespace

int main() { return run_test("gemm_os_program_schema_test", body); }
