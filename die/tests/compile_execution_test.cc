#include "test_support.hh"
#include "coarse_die/Capi.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <memory>
#include <set>
#include <string>
#include <vector>

using namespace coarse_die;
using namespace coarse_die_test;

namespace coarse_die::detail {
std::vector<std::shared_ptr<Plan>> enumerate_mappings(
    const Hardware&, const nlohmann::json&);
}

namespace {

json gemm_task(std::uint64_t m, std::uint64_t n, std::uint64_t k,
               const std::string& dtype = "bf16") {
  return json{{"op_kind", "gemm"},
              {"tensor_shapes", {{"m", m}, {"n", n}, {"k", k}}},
              {"storage_dtype", {{"weights", dtype}, {"activations", dtype},
                                 {"output", "bf16"}}},
              {"execution_profile", "int4_matrix_vec_fp32_assumed_v1"},
              {"layer_instances", 1},
              {"weight_passes", 1}};
}

std::uint64_t u64_field(const json& object, const char* key) {
  require(object.contains(key), std::string("summary missing ") + key);
  return object.at(key).get<std::uint64_t>();
}

json take_cabi_json(const char* raw, const char* function) {
  require(raw != nullptr,
          std::string(function) + " failed: " + die_coarse_last_error_v2());
  const std::string text(raw);
  die_coarse_free_result_v2(raw);
  return json::parse(text);
}

void public_cabi_matches_cpp_plan_and_keeps_batch_siblings() {
  const auto hw = example_hardware();
  const auto task = gemm_task(3, 5, 7, "int4");
  const auto cpp = compile(hw, task);
  const std::string config = hw.input.dump();
  DieHandle model = 0;
  require(die_coarse_create_model_v2(config.c_str(), &model) == DIE_OK,
          std::string("C ABI model create failed: ") + die_coarse_last_error_v2());
  const json request{{"specs", json::array({task, json{{"op_kind", "bad_op"}}, task})}};
  const std::string batch = request.dump();
  const auto response = take_cabi_json(
      die_coarse_compile_batch_v2(model, batch.c_str()), "compile_batch");
  require(response.at("results").size() == 3 && response.at("errors").size() == 3,
          "C ABI batch responses and errors stay positional");
  require(response.at("results").at(1).is_null() &&
              !response.at("errors").at(1).is_null(),
          "one invalid task reports its own error without discarding successful siblings");
  const auto& first = response.at("results").at(0);
  const auto& last = response.at("results").at(2);
  require(first.at("plan_hash") == cpp->fingerprint &&
              last.at("plan_hash") == cpp->fingerprint,
          "C ABI and C++ produce the same canonical plan hash for identical inputs");
  for (const char* key : {"logical_tensor_bytes", "int4_input_bytes",
                          "hbm_read_bytes_total", "hbm_write_bytes_total"}) {
    require(first.at("summary").at(key) == cpp->summary.at(key),
            std::string("C ABI summary parity for ") + key);
  }
  require(die_coarse_release_plan_v2(first.at("plan_handle").get<DieHandle>()) == DIE_OK &&
              die_coarse_release_plan_v2(last.at("plan_handle").get<DieHandle>()) == DIE_OK,
          "C ABI returns releasable handles for each successful batch sibling");
  require(die_coarse_destroy_model_v2(model) == DIE_OK,
          "C ABI model releases after its retained plans are released");
}

Hardware hardware_with_sram(double sram_mib) {
  auto input = read_json(std::string(COARSE_DIE_SOURCE_DIR) +
                         "/configs/example_hardware.json");
  input["architecture"]["sram_mib_per_core"] = sram_mib;
  // Hold a legal, explicit outline fixed while varying SRAM capacity.
  input["architecture"]["core_columns_nx"] = 1;
  input["architecture"]["core_rows_ny"] = 1;
  return resolve_hardware(input);
}

void compile_gemm_int4_packing_hash_and_work_contract() {
  const auto hw = example_hardware();
  const auto task = gemm_task(3, 5, 7, "int4");
  const auto plan = compile(hw, task);
  const auto repeated = compile(hw, task);
  require(!plan->fingerprint.empty() && plan->fingerprint == repeated->fingerprint,
          "identical hardware and task inputs must produce the same public plan hash");
  require(plan->hardware_fingerprint == hw.fingerprint,
          "compiled plan records the exact hardware fingerprint");
  require(plan->summary.at("granularity_validation") == "not_run",
          "compile cannot claim the independent granularity certificate was run");
  require(u64_field(plan->summary, "source_matrix_ops") == 210,
          "GEMM source matrix count is exactly 2*3*5*7");
  require(u64_field(plan->summary, "executed_matrix_ops") >= 210,
          "mapped GEMM cannot execute fewer matrix operations than the source graph");
  require(u64_field(plan->summary, "executed_int4_ops") >= 210,
          "target INT4 execution records at least the unpadded matrix operations");
  require(u64_field(plan->summary, "int4_input_bytes") >= 29,
          "INT4 input packing charges ceil(21/2)+ceil(35/2)=29 bytes before tile reuse");
  require(u64_field(plan->summary, "logical_tensor_bytes") == 59,
          "logical GEMM tensor bytes preserve odd INT4 tails (11+18+30)");
  require(u64_field(plan->summary, "output_round_count") > 0 &&
              u64_field(plan->summary, "consumption_wave_count") >=
                  u64_field(plan->summary, "output_round_count"),
          "GEMM separates output rounds from K-consumption waves");
  require(u64_field(plan->summary, "candidate_count") > 0 &&
              plan->summary.at("selected_template") == plan->summary.at("template_id"),
          "the chosen candidate and template are exposed for review");
  const auto serialized = plan_to_json(*plan);
  const auto restored = plan_from_json(serialized);
  require(restored->fingerprint == plan->fingerprint &&
              restored->summary == plan->summary &&
              plan_to_json(*restored) == serialized,
          "public plan serialization round-trips canonically without changing its hash");
  const auto estimate = estimate_alone(hw, *plan);
  require(estimate.at("status") == "ok_assumed" &&
              estimate.at("estimated_latency_ns").get<std::uint64_t>() > 0,
          "actual library standalone execution returns a finite assumed estimate");

  bool overflow = false;
  try {
    (void)compile(hw, gemm_task(1, 1,
        std::numeric_limits<std::uint64_t>::max(), "bf16"));
  } catch (const std::invalid_argument& e) {
    overflow = std::string(e.what()).find("overflow") != std::string::npos;
  }
  require(overflow, "products outside uint64 outputs must be rejected with overflow");
}

void gemm_dataflow_candidates_are_reachable_and_audited() {
  const auto hw = example_hardware();
  const std::array<std::array<std::uint64_t, 3>, 4> shapes{{
      {{64, 4096, 1024}},
      {{4096, 64, 4096}},
      {{1024, 1024, 1024}},
      {{8192, 64, 1024}},
  }};
  std::set<std::string> candidate_flows;
  bool checked_os_runtime = false;
  bool checked_wr_runtime = false;
  for (const auto& shape : shapes) {
    const auto plan = compile(hw, gemm_task(shape[0], shape[1], shape[2]));
    const std::string dataflow = plan->summary.at("dataflow_id").get<std::string>();
    require(dataflow == "output_stationary_v2" ||
                dataflow == "weight_resident_v2",
            "GEMM identifies the selected OS/WR schedule");
    require(u64_field(plan->summary, "output_round_count") > 0 &&
                u64_field(plan->summary, "consumption_wave_count") > 0,
            "each selected dataflow exposes real output and consumption counts");

    const auto candidates = detail::enumerate_mappings(hw, plan->task);
    require(!candidates.empty() &&
                plan->summary.at("candidate_count") == candidates.size(),
            "compiler reports every mapping in the public candidate set");
    bool selected_candidate_present = false;
    for (const auto& candidate : candidates) {
      const std::string candidate_flow =
          candidate->summary.at("dataflow_id").get<std::string>();
      require(candidate_flow == "output_stationary_v2" ||
                  candidate_flow == "weight_resident_v2",
              "bounded GEMM candidate enumeration exposes a known dataflow");
      candidate_flows.insert(candidate_flow);

      const bool os = candidate_flow == "output_stationary_v2";
      bool& runtime_checked = os ? checked_os_runtime : checked_wr_runtime;
      if (!runtime_checked) {
        const auto latency = detail::estimate_latency_ns(hw, candidate);
        require(latency > 0,
                "a real OS/WR candidate completes through the Runtime estimator");
        runtime_checked = true;
      }

      bool matches_selected = candidate_flow == dataflow;
      for (const char* key : {"template_id", "tm", "tn", "tk"}) {
        matches_selected = matches_selected &&
            candidate->summary.at(key) == plan->summary.at(key);
      }
      selected_candidate_present = selected_candidate_present || matches_selected;
    }
    require(selected_candidate_present,
            "selected mapping is present in the complete candidate enumeration");
    require(detail::estimate_latency_ns(hw, plan) ==
                u64_field(plan->summary, "estimated_latency_ns"),
            "selected shape-specific Plan retains its actual Runtime latency");
  }
  require(candidate_flows.find("output_stationary_v2") != candidate_flows.end() &&
              candidate_flows.find("weight_resident_v2") != candidate_flows.end() &&
              checked_os_runtime && checked_wr_runtime,
          "shape set retains both OS and WR candidate families even when one wins every shape");
}

void ffn_recomputation_and_sram_capacity_are_explicit() {
  const auto hw = hardware_with_sram(2.0);
  const std::uint64_t m = 8, hidden = 512, ffn = 128;
  json task{{"op_kind", "swiglu"},
            {"tensor_shapes", {{"m", m}, {"hidden", hidden}, {"ffn", ffn}}},
            {"execution_profile", "int4_matrix_vec_fp32_assumed_v1"},
            {"storage_dtype", {{"weights", "bf16"}, {"activations", "bf16"},
                               {"output", "bf16"}}},
            {"fusion_policy", "fused"},
            {"layer_instances", 1},
            {"weight_passes", 1}};
  const auto fused = compile(hw, task);
  const std::uint64_t source = 6 * m * hidden * ffn;
  require(fused->summary.at("template_id") == "fused_output_shard_v2",
          "explicit fused SwiGLU task selects the fixed fused output-shard template");
  require(u64_field(fused->summary, "source_matrix_ops") == source,
          "SwiGLU source graph contains six MHF matrix operations");
  const std::uint64_t th = u64_field(fused->summary, "th");
  require(th > 0, "fused FFN reports its output hidden tile");
  const std::uint64_t rounds_h = (hidden + th - 1) / th;
  const std::uint64_t expected_executed = 4 * m * hidden * ffn * rounds_h +
                                          2 * m * hidden * ffn;
  require(u64_field(fused->summary, "executed_matrix_ops") == expected_executed &&
              u64_field(fused->summary, "recomputed_matrix_ops") ==
                  expected_executed - source,
          "fused up/gate recomputation is counted from the selected H-output shards");

  task["fusion_policy"] = "materialized";
  const auto materialized = compile(hw, task);
  require(materialized->summary.at("template_id") == "materialized_swiglu_v2",
          "explicit materialized SwiGLU task selects the fixed four-kernel path");
  require(u64_field(materialized->summary, "source_matrix_ops") == source &&
              u64_field(materialized->summary, "executed_matrix_ops") == source &&
              u64_field(materialized->summary, "recomputed_matrix_ops") == 0,
          "materialization avoids hidden matrix recomputation");

  const double rho = hw.input.at("performance_profile").at("rho_sram").get<double>();
  const double bytes = hw.derived.at("core").at("sram_bytes_per_core").get<double>();
  for (const auto& candidate : {fused, materialized}) {
    const auto peak = u64_field(candidate->summary, "sram_peak_bytes_per_core");
    require(static_cast<double>(peak) <= rho * bytes + 1e-6,
            "selected FFN peak live storage stays under the configured per-core SRAM capacity");
    require(candidate->summary.at("granularity_validation") == "not_run",
            "neither FFN strategy is called granularity-certified without the oracle report");
  }
}

void attention_gqa_causal_absolute_spans_and_compact_runs() {
  const auto hw = example_hardware();
  json task{{"op_kind", "attention"},
            {"tensor_shapes", {{"hidden", 8}, {"q_heads", 4},
                               {"kv_heads", 2}, {"head_dim", 2}}},
            {"spans", json::array({json{{"q_tokens", 3}, {"kv_tokens", 5},
                                        {"q_start_pos", 1}, {"kv_start_pos", 2}}})},
            {"causal", true},
            {"execution_profile", "int4_matrix_vec_fp32_assumed_v1"},
            {"storage_dtype", "bf16"}};
  const auto causal = compile(hw, task);
  require(u64_field(causal->summary, "valid_score_pairs") == 12,
          "GQA causal score pairs use key_abs<=query_abs: three valid pairs times four Q heads");
  require(u64_field(causal->summary, "executed_score_slots") == 60,
          "intersecting causal rectangle executes 3*5 score slots for each of four Q heads");
  require(u64_field(causal->summary, "source_matrix_ops") > 0 &&
              u64_field(causal->summary, "executed_matrix_ops") >=
                  u64_field(causal->summary, "source_matrix_ops"),
          "attention separately reports source and executed matrix work");

  task["spans"] = json::array({json{
      {"run", {{"q_len", 3}, {"kv_len", 5}, {"kv_stride", 2},
               {"count", 2}, {"q_start_pos", 2}, {"kv_start_pos", 0}}},
      {"request_id_prefix", "request"}, {"first_index", 0}}});
  task["causal"] = false;
  const auto compact = compile(hw, task);
  require(compact->task.at("spans").size() == 1 &&
              compact->task.at("spans").at(0).at("count").get<std::uint64_t>() == 2,
          "a compact absolute-span run remains one arithmetic progression instead of expanding requests");
  require(u64_field(compact->summary, "valid_score_pairs") == 144,
          "noncausal compact spans count (3x5+3x7) rectangles for four Q heads");
}

void test_compile_execution() {
  compile_gemm_int4_packing_hash_and_work_contract();
  public_cabi_matches_cpp_plan_and_keeps_batch_siblings();
  gemm_dataflow_candidates_are_reachable_and_audited();
  ffn_recomputation_and_sram_capacity_are_explicit();
  attention_gqa_causal_absolute_spans_and_compact_runs();
  std::cout << "GEMM_INT4_odd=checked, OS_WR=candidates, FFN_recompute=checked, "
               "attention_GQA_causal=checked, same_input_hash=stable\n";
}
}  // namespace

int main() { return run_test("compile_execution_test", test_compile_execution); }
