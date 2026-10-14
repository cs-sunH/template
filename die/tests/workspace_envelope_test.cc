#include "test_support.hh"
#include "coarse_die/Capi.h"

#include <algorithm>
#include <cstdint>
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

json attention_task(std::uint64_t q = 64, std::uint64_t kv = 64) {
  return {{"schema", "coarse_die_task_v2"},
          {"op_kind", "attention_region"},
          {"phase", "prefill"},
          {"tensor_shapes", {{"m", q}, {"q_heads", 1}, {"kv_heads", 1},
                              {"head_dim", 32}}},
          {"spans", json::array({{{"q_tokens", q}, {"kv_tokens", kv},
                                  {"q_start_pos", kv >= q ? kv - q : 0},
                                  {"kv_start_pos", 0}}})},
          {"fusion_policy", "auto"}};
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

std::uint64_t enumerated_max(const Hardware& hardware, const json& task,
                             std::uint64_t& count) {
  const auto normalized = compile_workspace_envelope(hardware, task).normalized_task;
  const auto candidates = detail::enumerate_mappings_for_compile(hardware, normalized);
  require(!candidates.empty(), "fixture has compiler mapping candidates");
  count = static_cast<std::uint64_t>(candidates.size());
  std::uint64_t maximum = 0;
  for (const auto& candidate : candidates)
    maximum = std::max(maximum, workspace(*candidate));
  return maximum;
}

void compare_with_actual_mapping_candidates(const Hardware& hardware,
                                            const json& task,
                                            bool expect_unselected_peak) {
  const auto envelope = compile_workspace_envelope(hardware, task);
  std::uint64_t candidate_count = 0;
  const auto actual_max = enumerated_max(hardware, task, candidate_count);
  const auto selected = compile(hardware, task);
  const auto selected_workspace = workspace(*selected);
  require(envelope.candidate_count == candidate_count && candidate_count > 1,
          "envelope counts every actual mapping candidate");
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
  if (expect_unselected_peak)
    require(envelope.max_hbm_workspace_peak_bytes > selected_workspace,
            "unselected materialized template can exceed the selected Plan");
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
  compare_with_actual_mapping_candidates(hardware, attention_task(), true);
  compare_with_actual_mapping_candidates(hardware, ffn_task(), true);
  aliases_and_schema_equivalence(hardware);
  exact_u64_and_overflow(hardware);
  legacy_local_region();
  cabi_contract(hardware);
}

}  // namespace

int main() { return run_test("workspace_envelope_test", body); }
