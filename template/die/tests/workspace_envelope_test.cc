#include "test_support.hh"
#include "coarse_die/Capi.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace coarse_die::detail {
std::vector<std::shared_ptr<coarse_die::Plan>> enumerate_mappings_for_compile(
    const coarse_die::Hardware&, const nlohmann::json&);
}

using namespace coarse_die;
using namespace coarse_die_test;
using json = nlohmann::json;

namespace {

json attention_task(std::uint64_t q = 64, std::uint64_t kv = 64,
                    std::uint64_t head_dim = 32) {
  return {{"schema", "coarse_die_task_v2"},
          {"op_kind", "attention_region"},
          {"phase", "prefill"},
          {"tensor_shapes", {{"m", q}, {"q_heads", 1}, {"kv_heads", 1},
                              {"head_dim", head_dim}}},
          {"spans", json::array({{{"q_tokens", q}, {"kv_tokens", kv},
                                  {"q_start_pos", kv >= q ? kv - q : 0},
                                  {"kv_start_pos", 0}}})},
          {"fusion_policy", "auto"}};
}

json capacity_attention_task(std::uint64_t q, std::uint64_t kv,
                             std::uint64_t head_dim,
                             const std::string& fusion = "auto",
                             std::uint64_t layers = 1,
                             std::uint64_t passes = 1) {
  return {{"schema", "coarse_die_task_v2"},
          {"op_kind", "attention_region"},
          {"phase", "prefill"},
          {"tensor_shapes", {{"q_heads", 1}, {"kv_heads", 1},
                              {"head_dim", head_dim}}},
          {"spans", json::array({{{"q_tokens", q}, {"kv_tokens", kv},
                                  {"q_start_pos", 0}, {"kv_start_pos", 0}}})},
          {"storage_dtype", {{"activations", "bf16"}, {"kv", "bf16"},
                              {"output", "bf16"}, {"scores", "fp32"},
                              {"accumulator", "fp32"}}},
          {"fusion_policy", fusion}, {"causal", false},
          {"layer_instances", layers}, {"weight_passes", passes}};
}

Hardware low_sram_hardware() {
  auto config = read_json(std::string(COARSE_DIE_SOURCE_DIR) +
                          "/configs/example_hardware.json");
  // example SRAM grew 0.25 -> 1.5 MiB/core (2026-10-21 die redesign); 1/24
  // keeps the usable SRAM at the historical 65536 B so this suite still
  // exercises the low-SRAM filtering boundary it was written for.
  config["performance_profile"]["rho_sram"] = 1.0 / 24.0;
  return resolve_hardware(config);
}

json ffn_task() {
  return {{"schema", "coarse_die_task_v2"},
          {"op_kind", "ffn_region"},
          {"phase", "prefill"},
          {"tensor_shapes", {{"m", 64}, {"hidden", 128}, {"ffn", 64},
                              {"mlp_variant", "swiglu"}}},
          {"fusion_policy", "auto"}};
}

std::uint64_t workspace(const Plan& plan) {
  return hbm_workspace_peak_bytes(plan);
}

std::uint64_t summary_u64(const Plan& plan, const char* key) {
  require(plan.summary.contains(key),
          std::string("Plan summary is missing ") + key);
  return plan.summary.at(key).get<std::uint64_t>();
}

std::uint64_t enumerated_max(const Hardware& hardware, const json& task,
                             std::uint64_t& count,
                             bool& has_materialized_candidate) {
  const auto normalized = compile_workspace_envelope(hardware, task).normalized_task;
  const auto candidates = detail::enumerate_mappings_for_compile(hardware, normalized);
  require(!candidates.empty(), "fixture has compiler mapping candidates");
  count = static_cast<std::uint64_t>(candidates.size());
  std::uint64_t maximum = 0;
  for (const auto& candidate : candidates) {
    const auto candidate_workspace = workspace(*candidate);
    maximum = std::max(maximum, candidate_workspace);
    has_materialized_candidate = has_materialized_candidate ||
        candidate_workspace > 0;
  }
  return maximum;
}

void compare_with_actual_mapping_candidates(const Hardware& hardware,
                                            const json& task) {
  const auto envelope = compile_workspace_envelope(hardware, task);
  std::uint64_t candidate_count = 0;
  bool has_materialized_candidate = false;
  const auto actual_max = enumerated_max(
      hardware, task, candidate_count, has_materialized_candidate);
  const auto selected = compile(hardware, task);
  const auto selected_workspace = workspace(*selected);
  require(envelope.candidate_count == candidate_count && candidate_count > 1,
          "envelope counts every actual mapping candidate");
  require(has_materialized_candidate,
          "auto candidate set retains its materialized mapping family");
  require(envelope.max_hbm_workspace_peak_bytes == actual_max,
          "envelope peak equals the maximum exact workspace in the actual mapping vector");
  if (envelope.op_kind == "attention_region")
    require(actual_max == 40960,
            "materialized BF16 attention fixture has an exact 40,960-byte maximum");
  else if (envelope.op_kind == "ffn_region")
    require(actual_max == 16384,
            "materialized BF16 SwiGLU fixture has an exact 16,384-byte maximum");
  require(envelope.max_hbm_workspace_peak_bytes >= selected_workspace,
          "selected Plan workspace is covered by the all-template envelope");
  auto fused_task = task;
  fused_task["fusion_policy"] = "fused";
  const auto forced_fused = compile(hardware, fused_task);
  require(actual_max > workspace(*forced_fused),
          "all-template envelope includes the materialized lifetime beyond a valid forced-fused Plan");
  require(envelope.normalized_task == selected->task &&
              envelope.hardware_fingerprint == hardware.fingerprint &&
              envelope.normalized_task_fingerprint ==
                  stable_hash(envelope.normalized_task.dump()),
          "envelope returns canonical task and hardware identities");
  require(envelope.op_kind == task.at("op_kind").get<std::string>() &&
              envelope.phase == "prefill",
          "envelope identifies the typed operation and phase");
}

json take(const char* raw, const std::string& api) {
  require(raw != nullptr, api + " failed: " + die_coarse_last_error_v2());
  std::unique_ptr<const char, void (*)(const char*)> owned(
      raw, &die_coarse_free_result_v2);
  return json::parse(raw);
}

json error_info(const char* context) {
  const char* raw = die_coarse_last_error_v2();
  require(raw != nullptr && raw[0] != '\0',
          std::string(context) + " must set last_error");
  return json::parse(raw);
}

void legacy_local_region() {
  const Hardware hardware = example_hardware();
  const json task{{"die_op_spec", {{"schema_version", 1},
                                   {"op_kind", "local_region"}}}};
  bool rejected = false;
  try {
    (void)compile_workspace_envelope(hardware, task);
  } catch (const std::exception& error) {
    rejected = std::string(error.what()).find(
                   "unsupported local_region: DieOpSpec v1 carries no typed mathematical graph")
               != std::string::npos;
  }
  require(rejected,
          "workspace query rejects legacy local_region exactly as normal compile does");
}

void exact_u64_and_overflow(const Hardware& hardware) {
  // One query row and 2^53 KV tokens produce 2^56+128 bytes for the full
  // materialized attention template. The result is above binary64's exact
  // integer range but remains within uint64.
  const auto huge = attention_task(1, std::uint64_t{1} << 53);
  const auto envelope = compile_workspace_envelope(hardware, huge);
  require(envelope.max_hbm_workspace_peak_bytes ==
              (std::uint64_t{1} << 56) + 128,
          "compiler envelope preserves the exact large workspace integer");

  const auto overflow = attention_task(1, std::uint64_t{1} << 62);
  bool rejected = false;
  try {
    (void)compile_workspace_envelope(hardware, overflow);
  } catch (const std::exception& error) {
    rejected = std::string(error.what()).find("overflow") != std::string::npos;
  }
  require(rejected, "workspace above uint64 fails closed during compilation");

  const auto repeat_overflow = capacity_attention_task(
      1, 1, 32, "materialized", std::numeric_limits<std::uint64_t>::max(),
      std::numeric_limits<std::uint64_t>::max());
  const auto repeat_bound =
      compile_workspace_capacity_bound(hardware, repeat_overflow);
  require(repeat_bound.hbm_workspace_upper_bound_bytes == 136,
          "sequential repeat factors do not inflate or overflow one attention workspace peak");
  rejected = false;
  try {
    (void)compile_workspace_envelope(hardware, repeat_overflow);
  } catch (const std::exception& error) {
    rejected = std::string(error.what()).find("overflow") != std::string::npos;
  }
  require(rejected,
          "the same extreme repeats still overflow checked physical work aggregation");

  auto unsupported_dtype = attention_task();
  unsupported_dtype["storage_dtype"] = {{"output", "fp16"}};
  rejected = false;
  try {
    (void)compile_workspace_capacity_bound(hardware, unsupported_dtype);
  } catch (const std::invalid_argument& error) {
    rejected = std::string(error.what()).find("unsupported dtype") !=
               std::string::npos;
  }
  require(rejected,
          "capacity query rejects unsupported storage dtypes during normalization");

  auto invalid_type = attention_task();
  invalid_type["spans"][0]["kv_tokens"] = 64.0;
  rejected = false;
  try {
    (void)compile_workspace_envelope(hardware, invalid_type);
  } catch (const std::invalid_argument& error) {
    rejected = std::string(error.what()).find("floating-point values are rejected") !=
               std::string::npos;
  }
  require(rejected, "workspace query rejects integral-looking floating dimensions");
}

void aliases_and_schema_equivalence(const Hardware& hardware) {
  const auto canonical = attention_task();
  auto aliases = canonical;
  aliases["tensor_shapes"] = {{"M", 64}, {"num_q_heads", 1},
                              {"num_kv_heads", 1}, {"head_dim", 32}};
  aliases["spans"] = json::array({{{"q_len", 64}, {"kv_len", 64},
                                   {"query_start", 0}, {"kv_start", 0}}});
  const auto left = compile_workspace_envelope(hardware, canonical);
  const auto right = compile_workspace_envelope(hardware, aliases);
  require(left.normalized_task == right.normalized_task &&
              left.normalized_task_fingerprint == right.normalized_task_fingerprint &&
              left.max_hbm_workspace_peak_bytes == right.max_hbm_workspace_peak_bytes &&
              left.candidate_count == right.candidate_count,
          "accepted dimension and span aliases share one canonical envelope identity");
}

void candidate_independent_capacity_bound() {
  const Hardware hardware = low_sram_hardware();
  const auto parent = capacity_attention_task(64, 64, 256);
  auto child = capacity_attention_task(8, 8, 256);
  child["layer_instances"] = 2;
  child["weight_passes"] = 3;
  const auto parent_envelope = compile_workspace_envelope(hardware, parent);
  const auto child_envelope = compile_workspace_envelope(hardware, child);
  const auto parent_candidates = detail::enumerate_mappings_for_compile(
      hardware, parent_envelope.normalized_task);
  const auto child_candidates = detail::enumerate_mappings_for_compile(
      hardware, child_envelope.normalized_task);
  const auto has_materialized = [](const auto& candidates) {
    return std::any_of(candidates.begin(), candidates.end(), [](const auto& p) {
      return p->summary.value("template_id", std::string()) ==
             "full_score_materialization_v3";
    });
  };
  require(parent_envelope.max_hbm_workspace_peak_bytes == 0 &&
              !has_materialized(parent_candidates),
          "full task SRAM filtering removes its materialized attention candidate");
  require(child_envelope.max_hbm_workspace_peak_bytes == 8704 &&
              has_materialized(child_candidates),
          "smaller legal child gains an 8,704-byte materialized attention candidate");

  const auto parent_bound = compile_workspace_capacity_bound(hardware, parent);
  const auto child_bound = compile_workspace_capacity_bound(hardware, child);
  require(parent_bound.hbm_workspace_upper_bound_bytes == 98304 &&
              child_bound.hbm_workspace_upper_bound_bytes == 8704 &&
              parent_bound.hbm_workspace_upper_bound_bytes >=
                  child_envelope.max_hbm_workspace_peak_bytes,
          "pre-SRAM lifetime bound covers child mappings hidden by parent feasibility filtering");
  require(parent_bound.normalized_task == parent_envelope.normalized_task &&
              parent_bound.hardware_fingerprint == hardware.fingerprint &&
              parent_bound.normalized_task_fingerprint ==
                  stable_hash(parent_bound.normalized_task.dump()),
          "capacity bound shares canonical task and hardware identities");

  auto materialized_only = parent;
  materialized_only["fusion_policy"] = "materialized";
  require(compile_workspace_capacity_bound(hardware, materialized_only)
                  .hbm_workspace_upper_bound_bytes == 98304,
          "capacity-only query succeeds without an SRAM-feasible materialized mapping");
  bool no_executable_mapping = false;
  try {
    (void)compile_workspace_envelope(hardware, materialized_only);
  } catch (const std::exception& error) {
    no_executable_mapping = std::string(error.what()).find(
        "no feasible fixed mapping candidate") != std::string::npos;
  }
  require(no_executable_mapping,
          "workspace envelope still requires at least one executable mapping");

  auto fused_attention = parent;
  fused_attention["fusion_policy"] = "fused";
  require(compile_workspace_capacity_bound(hardware, fused_attention)
                  .hbm_workspace_upper_bound_bytes == 0,
          "forced fused attention has no materialized HBM lifetime");
  auto none_attention = parent;
  none_attention["fusion_policy"] = "none";
  require(compile_workspace_capacity_bound(hardware, none_attention)
                  .hbm_workspace_upper_bound_bytes == 98304,
          "attention none policy still permits its materialized family");
  auto qk = parent;
  qk["op_kind"] = "qk";
  require(compile_workspace_capacity_bound(hardware, qk)
                  .hbm_workspace_upper_bound_bytes == 16384,
          "QK-only bound covers exactly one FP32 score rectangle");

  const auto ffn_bound = compile_workspace_capacity_bound(hardware, ffn_task());
  require(ffn_bound.hbm_workspace_upper_bound_bytes == 16384,
          "materialized SwiGLU capacity uses its reusable two-buffer peak");
  auto fused_ffn = ffn_task();
  fused_ffn["fusion_policy"] = "fused";
  require(compile_workspace_capacity_bound(hardware, fused_ffn)
                  .hbm_workspace_upper_bound_bytes == 0,
          "forced fused FFN has no materialized intermediate lifetime");
  auto none_ffn = ffn_task();
  none_ffn["fusion_policy"] = "none";
  require(compile_workspace_capacity_bound(hardware, none_ffn)
                  .hbm_workspace_upper_bound_bytes == 16384,
          "FFN none policy retains its materialized intermediate family");

  auto odd_int4_ffn = ffn_task();
  odd_int4_ffn["fusion_policy"] = "materialized";
  odd_int4_ffn["tensor_shapes"]["m"] = 1;
  odd_int4_ffn["tensor_shapes"]["ffn"] = 3;
  odd_int4_ffn["storage_dtype"]["activations"] = "int4";
  require(compile_workspace_capacity_bound(hardware, odd_int4_ffn)
                  .hbm_workspace_upper_bound_bytes == 4,
          "odd INT4 SwiGLU counts two separately byte-ceiled intermediate buffers");
  require(workspace(*compile(hardware, odd_int4_ffn)) == 4,
          "materialized FFN assignment uses the same odd INT4 byte-ceiling formula");

  for (const auto& invalid : {
           [] { auto task = capacity_attention_task(8, 8, 32);
                task["tensor_shapes"]["q_heads"] = 0; return task; }(),
           [] { auto task = capacity_attention_task(8, 8, 32);
                task["tensor_shapes"]["kv_heads"] = 0; return task; }(),
           [] { auto task = capacity_attention_task(8, 8, 32);
                task["spans"][0]["q_tokens"] = 0; return task; }(),
           [] { auto task = capacity_attention_task(8, 8, 32);
                task["spans"][0]["kv_tokens"] = 0; return task; }()}) {
    bool rejected = false;
    try {
      (void)compile_workspace_capacity_bound(hardware, invalid);
    } catch (const std::exception&) {
      rejected = true;
    }
    require(rejected,
            "capacity bound rejects empty local attention head/query/KV shards");
  }

  auto opaque_identity = capacity_attention_task(8, 8, 32);
  opaque_identity["math"] = {{"future_semantics", "opaque"}};
  opaque_identity["residency"] = {{"future_residency", "opaque"}};
  const auto opaque_bound =
      compile_workspace_capacity_bound(hardware, opaque_identity);
  require(opaque_bound.normalized_task.at("math") == opaque_identity.at("math") &&
              opaque_bound.normalized_task.at("residency") ==
                  opaque_identity.at("residency") &&
              opaque_bound.hbm_workspace_upper_bound_bytes > 0,
          "opaque capacity metadata stays in identity while the API remains capacity-only");
}

void repeated_attention_workspace_uses_sequential_peak() {
  const Hardware hardware = low_sram_hardware();
  const auto once_task = capacity_attention_task(64, 64, 32, "materialized");
  const auto repeated = capacity_attention_task(64, 64, 32, "materialized", 2, 3);
  const auto once_plan = compile(hardware, once_task);
  const auto plan = compile(hardware, repeated);
  const auto envelope = compile_workspace_envelope(hardware, repeated);
  const auto once_bound = compile_workspace_capacity_bound(hardware, once_task);
  const auto bound = compile_workspace_capacity_bound(hardware, repeated);
  require(workspace(*once_plan) == 40960 && workspace(*plan) == 40960 &&
              bound.hbm_workspace_upper_bound_bytes == 40960 &&
              bound.hbm_workspace_upper_bound_bytes ==
                  once_bound.hbm_workspace_upper_bound_bytes,
          "materialized workspace and capacity bound remain one peak across sequential layer/pass repeats");
  require(envelope.max_hbm_workspace_peak_bytes == workspace(*plan) &&
              bound.hbm_workspace_upper_bound_bytes >=
                  envelope.max_hbm_workspace_peak_bytes,
          "the revised candidate-independent bound covers every repeated-task candidate");
  require(plan->summary.at("template_id") == "full_score_materialization_v3" &&
              plan->summary.at("workspace_model_revision") ==
                  "workspace_lifetimes_v2" &&
              plan->fingerprint != once_plan->fingerprint &&
              plan->fingerprint !=
                  "3283ef35e93b58457e011f770a95647589b195fbe2c5335d1fcb6cdfaf7cc3a8",
          "corrected repeated-workspace Plans carry a new template/model identity and hash");
  for (const char* key : {"executed_score_slots", "executed_matrix_ops",
                          "vec_equivalent_ops", "hbm_read_bytes_total",
                          "hbm_write_bytes_total", "sram_read_bytes",
                          "sram_write_bytes", "local_bus_bytes"}) {
    require(summary_u64(*plan, key) == 6 * summary_u64(*once_plan, key),
            std::string("repeat aggregation still scales physical ") + key);
  }
  const Hardware default_hardware = example_hardware();
  auto repeated_ffn = ffn_task();
  repeated_ffn["fusion_policy"] = "materialized";
  repeated_ffn["layer_instances"] = 2;
  repeated_ffn["weight_passes"] = 3;
  const auto ffn_plan = compile(default_hardware, repeated_ffn);
  const auto ffn_bound = compile_workspace_capacity_bound(default_hardware,
                                                          repeated_ffn);
  require(workspace(*ffn_plan) == 16384 &&
              ffn_bound.hbm_workspace_upper_bound_bytes == 16384,
          "FFN intermediate capacity remains one reusable peak across repeats");
}

void cabi_contract(const Hardware& hardware) {
  const std::string config = hardware.input.dump();
  DieHandle model = 0;
  require(die_coarse_create_model_v2(config.c_str(), &model) == DIE_OK,
          std::string("create model: ") + die_coarse_last_error_v2());

  const auto large = attention_task(1, std::uint64_t{1} << 53);
  const std::string task_json = large.dump();
  const auto response = take(
      die_coarse_workspace_envelope_v2(model, task_json.c_str()),
      "workspace envelope query");
  require(response.at("schema_version") == 2 &&
              response.at("workspace_envelope_schema_version") == 1 &&
              response.at("candidate_count").is_number_unsigned() &&
              response.at("max_hbm_workspace_peak_bytes").is_number_unsigned() &&
              response.at("max_hbm_workspace_peak_bytes").get<std::uint64_t>() ==
                  (std::uint64_t{1} << 56) + 128 &&
              response.at("op_kind") == "attention_region" &&
              response.at("phase") == "prefill" &&
              response.at("normalized_task").is_object() &&
              response.at("hardware_fingerprint") == hardware.fingerprint,
          "C ABI emits exact uint64 workspace and canonical identity fields");
  const auto capacity_response = take(
      die_coarse_workspace_capacity_bound_v2(model, task_json.c_str()),
      "workspace capacity bound query");
  require(capacity_response.at("schema_version") == 2 &&
              capacity_response.at("workspace_capacity_bound_schema_version") == 1 &&
              capacity_response.at("capacity_model_revision") ==
                  "workspace_lifetimes_v2" &&
              capacity_response.at("capacity_bound_kind") ==
                  "all_template_lifetimes_without_sram_filter" &&
              capacity_response.at("hbm_workspace_upper_bound_bytes").is_number_unsigned() &&
              capacity_response.at("hbm_workspace_upper_bound_bytes").get<std::uint64_t>() ==
                  (std::uint64_t{1} << 56) + 128 &&
              capacity_response.at("normalized_task") == response.at("normalized_task") &&
              capacity_response.at("normalized_task_fingerprint") ==
                  response.at("normalized_task_fingerprint") &&
              capacity_response.at("hardware_fingerprint") == hardware.fingerprint,
          "capacity C ABI emits versioned exact bound and canonical identities");
  require(!capacity_response.contains("executable") &&
              !capacity_response.contains("is_executable"),
          "capacity response does not imply task executability");
  require(die_coarse_last_error_v2()[0] == '\0',
          "successful query clears the prior error state");

  const auto empty_cache = take(
      die_coarse_estimate_batch_v2(model, "{\"specs\":[]}"),
      "empty estimate after envelope query");
  require(empty_cache.at("cache_entries") == 0 &&
              empty_cache.at("cache_accounted_bytes") == 0,
          "envelope query neither inserts nor touches a model cache entry");

  require(die_coarse_workspace_envelope_v2(999999, task_json.c_str()) == nullptr,
          "invalid model handle must fail");
  require(error_info("invalid handle").at("status_code") == DIE_STALE_HANDLE,
          "invalid handle uses the existing C ABI error envelope");
  require(die_coarse_workspace_capacity_bound_v2(999999, task_json.c_str()) == nullptr,
          "invalid handle must fail on the capacity endpoint");
  require(error_info("capacity invalid handle").at("status_code") == DIE_STALE_HANDLE,
          "capacity endpoint uses the existing stale-handle error code");
  require(die_coarse_workspace_envelope_v2(model, "{") == nullptr,
          "malformed JSON must fail");
  require(error_info("bad JSON").at("status_code") == DIE_INVALID_INPUT,
          "malformed JSON reports invalid input");
  require(die_coarse_workspace_envelope_v2(model, nullptr) == nullptr,
          "null JSON must fail");
  require(error_info("null JSON").at("status_code") == DIE_INVALID_INPUT,
          "null JSON reports invalid input");

  auto mistyped = attention_task();
  mistyped["spans"][0]["kv_tokens"] = 64.0;
  const std::string mistyped_json = mistyped.dump();
  require(die_coarse_workspace_envelope_v2(model, mistyped_json.c_str()) == nullptr,
          "floating-point dimensions must fail");
  require(error_info("floating dimension").at("status_code") == DIE_INVALID_INPUT,
          "invalid exact-integer type reports invalid input");

  const std::string overflow_json =
      attention_task(1, std::uint64_t{1} << 62).dump();
  require(die_coarse_workspace_envelope_v2(model, overflow_json.c_str()) == nullptr,
          "workspace beyond uint64 must fail through the C ABI");
  require(error_info("uint64 overflow").at("status_code") == DIE_OVERFLOW,
          "workspace overflow uses the existing C ABI overflow code");
  require(die_coarse_workspace_capacity_bound_v2(model, overflow_json.c_str()) == nullptr,
          "capacity bound beyond uint64 must fail through the C ABI");
  require(error_info("capacity uint64 overflow").at("status_code") == DIE_OVERFLOW,
          "capacity overflow uses the existing C ABI overflow code");

  const std::string old_local =
      R"({"die_op_spec":{"schema_version":1,"op_kind":"local_region"}})";
  require(die_coarse_workspace_envelope_v2(model, old_local.c_str()) == nullptr,
          "unsupported legacy local_region must fail through C ABI");
  require(error_info("legacy local_region").at("reason").get<std::string>().find(
              "unsupported local_region") != std::string::npos,
          "C ABI preserves the compiler's local_region diagnostic");

  const std::string small_task = attention_task().dump();
  const std::string request = "{\"specs\":[" + small_task + "]}";
  const auto compiled = take(die_coarse_compile_batch_v2(model, request.c_str()),
                             "compile after envelope query");
  const auto plan_handle = compiled.at("results").at(0).at("plan_handle")
                               .get<DieHandle>();
  require(plan_handle == 2,
          "workspace query allocates no C ABI handle before the first Plan");
  require(die_coarse_release_plan_v2(plan_handle) == DIE_OK,
          "compiled comparison Plan releases normally");
  require(die_coarse_destroy_model_v2(model) == DIE_OK,
          "model can be destroyed after query and Plan release");
}

void body() {
  const Hardware hardware = example_hardware();
  compare_with_actual_mapping_candidates(hardware, attention_task());
  compare_with_actual_mapping_candidates(hardware, ffn_task());
  candidate_independent_capacity_bound();
  repeated_attention_workspace_uses_sequential_peak();
  aliases_and_schema_equivalence(hardware);
  exact_u64_and_overflow(hardware);
  legacy_local_region();
  cabi_contract(hardware);
}

}  // namespace

int main() { return run_test("workspace_envelope_test", body); }
