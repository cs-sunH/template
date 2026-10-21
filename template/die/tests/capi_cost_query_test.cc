#include "test_support.hh"
#include "coarse_die/Capi.h"

#include <cstdint>
#include <limits>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

using namespace coarse_die;
using namespace coarse_die_test;

namespace {

class CModel {
 public:
  explicit CModel(const Hardware& hardware) {
    const std::string config = hardware.input.dump();
    require(die_coarse_create_model_v2(config.c_str(), &handle_) == DIE_OK,
            std::string("C ABI model create failed: ") + die_coarse_last_error_v2());
  }
  CModel(const CModel&) = delete;
  CModel& operator=(const CModel&) = delete;
  ~CModel() {
    if (handle_) (void)die_coarse_destroy_model_v2(handle_);
  }
  DieHandle get() const { return handle_; }
  int destroy() {
    if (!handle_) return DIE_STALE_HANDLE;
    const int status = die_coarse_destroy_model_v2(handle_);
    if (status == DIE_OK) handle_ = 0;
    return status;
  }

 private:
  DieHandle handle_ = 0;
};

class PlanHandles {
 public:
  PlanHandles() = default;
  PlanHandles(const PlanHandles&) = delete;
  PlanHandles& operator=(const PlanHandles&) = delete;
  ~PlanHandles() {
    for (const auto handle : handles_)
      if (handle) (void)die_coarse_release_plan_v2(handle);
  }
  void track(const json& response) {
    for (const auto& row : response.at("results")) {
      if (row.is_object() && row.contains("plan_handle"))
        handles_.push_back(row.at("plan_handle").get<DieHandle>());
    }
  }
  std::size_t size() const { return handles_.size(); }
  DieHandle at(std::size_t index) const { return handles_.at(index); }
  int release_all() {
    int status = DIE_OK;
    for (auto& handle : handles_) {
      if (!handle) continue;
      const int current = die_coarse_release_plan_v2(handle);
      if (current == DIE_OK) handle = 0;
      else status = current;
    }
    return status;
  }

 private:
  std::vector<DieHandle> handles_;
};

json take_result(const char* raw, const char* api) {
  require(raw != nullptr,
          std::string(api) + " failed: " + die_coarse_last_error_v2());
  const std::string text(raw);
  die_coarse_free_result_v2(raw);
  return json::parse(text);
}

std::string request_text(const json& specs) {
  return json{{"specs", specs}}.dump();
}

json compile_batch(DieHandle model, const json& specs) {
  const std::string request = request_text(specs);
  return take_result(die_coarse_compile_batch_v2(model, request.c_str()),
                     "compile_batch");
}

json estimate_batch(DieHandle model, const json& specs) {
  const std::string request = request_text(specs);
  return take_result(die_coarse_estimate_batch_v2(model, request.c_str()),
                     "estimate_batch");
}

json estimate_cost_batch(DieHandle model, const json& specs) {
  const std::string request = request_text(specs);
  return take_result(
      die_coarse_estimate_cost_batch_v2(model, request.c_str()),
      "estimate_cost_batch");
}

json gemm_task(std::uint64_t m = 17, std::uint64_t n = 5,
               std::uint64_t k = 3) {
  return {{"op_kind", "gemm"},
          {"tensor_shapes", {{"m", m}, {"n", n}, {"k", k}}}};
}

json alias_gemm_task(std::uint64_t m = 17, std::uint64_t n = 5,
                     std::uint64_t k = 3) {
  return {{"schema", "coarse_die_task_v2"},
          {"op_kind", "gemm"},
          {"tensor_shapes", {{"K", k}, {"M", m}, {"N", n}}},
          {"phase", "prefill"},
          {"storage_dtype", "bf16"},
          {"execution_profile_id", "int4_matrix_vec_fp32_assumed_v1"},
          {"tp_slice", {{"tensor_parallel", 1}, {"tensor_parallel_rank", 0}}},
          {"layer_instances", 1},
          {"weight_passes", 1},
          {"fusion_policy", "auto"}};
}

json legacy_norm_task(const std::string& request_id) {
  return {{"die_op_spec",
           {{"schema_version", 1},
            {"op_kind", "norm"},
            {"phase", "prefill"},
            {"dtype", "bf16"},
            {"model_dims", {{"hidden", 128}, {"ffn", 256}, {"vocab", 512},
                            {"mlp_variant", "swiglu"}, {"norm_type", "rmsnorm"},
                            {"position_encoding", "rope"}, {"num_q_heads", 4},
                            {"num_kv_heads", 2}, {"head_dim", 32}}},
            {"tp_slice", {{"tensor_parallel", 1}, {"tensor_parallel_rank", 0},
                           {"q_heads_local", 4}, {"kv_heads_local", 2},
                           {"ffn_local", 256}, {"vocab_local", 512}}},
            {"spans", json::array({{{"request_id", request_id}, {"q_tokens", 4},
                                    {"kv_tokens", 16}, {"q_start_pos", 12},
                                    {"kv_start_pos", 0}}})},
            {"layer_count", 1},
            {"weight_passes", 1},
            {"kv_access", {{"identity", "local"}}},
            {"fusion_policy", "auto"}}}};
}

std::uint64_t direct_runtime_finish(const Hardware& hardware,
                                    std::shared_ptr<const Plan> plan) {
  Runtime runtime(hardware);
  const auto handle = runtime.submit(std::move(plan), owner(1, 7, 11), 0);
  std::size_t advances = 0;
  while (!runtime.drained()) {
    const auto next = runtime.next_event();
    require(next.has_value(), "isolated Runtime has a next event until drain");
    runtime.advance_to(*next);
    require(++advances <= 1000000, "isolated Runtime stays within its event budget");
  }
  const json progress = runtime.query_progress(handle);
  require(progress.at("state") == "finished",
          "direct isolated Runtime reaches finished");
  return time_from_progress(progress, "finish_ns");
}

void canonical_cache_and_fresh_compile_handles() {
  const Hardware hardware = example_hardware();
  CModel model(hardware);
  PlanHandles handles;

  const json response = compile_batch(
      model.get(), json::array({gemm_task(), alias_gemm_task(), gemm_task()}));
  handles.track(response);
  require(response.at("results").size() == 3 &&
              response.at("errors").size() == 3 &&
              response.at("errors").at(0).is_null() &&
              response.at("errors").at(1).is_null() &&
              response.at("errors").at(2).is_null(),
          "default, alias, and repeated spellings all compile successfully");
  const auto& rows = response.at("results");
  require(response.at("cache_entries") == 1 &&
              response.at("cache_accounted_bytes").get<std::size_t>() > 0,
          "semantically identical tasks share one normalized cache entry");
  require(rows.at(0).at("plan_hash") == rows.at(1).at("plan_hash") &&
              rows.at(0).at("plan_hash") == rows.at(2).at("plan_hash") &&
              rows.at(0).at("summary") == rows.at(1).at("summary") &&
              rows.at(0).at("summary") == rows.at(2).at("summary"),
          "normalized aliases preserve exact Plan fingerprint and summary");
  std::set<DieHandle> fresh_handles;
  for (const auto& row : rows)
    fresh_handles.insert(row.at("plan_handle").get<DieHandle>());
  require(handles.size() == 3 && fresh_handles.size() == 3 &&
              *fresh_handles.begin() != 0,
          "every occurrence, including a normalized-key hit, receives a fresh handle");

  const auto before_release = rows.at(0).at("plan_handle").get<DieHandle>();
  require(handles.release_all() == DIE_OK,
          "all fresh handles from canonical hits can be released");
  require(die_coarse_release_plan_v2(before_release) == DIE_STALE_HANDLE,
          "released handle is stale after the caller releases it");
}

void legacy_trace_fields_share_a_canonical_entry() {
  const Hardware hardware = example_hardware();
  CModel model(hardware);
  PlanHandles handles;
  const auto first = compile_batch(
      model.get(), json::array({legacy_norm_task("request-a")}));
  handles.track(first);
  const auto second = compile_batch(
      model.get(), json::array({legacy_norm_task("request-b")}));
  handles.track(second);
  require(first.at("results").at(0).at("plan_hash") ==
              second.at("results").at(0).at("plan_hash") &&
              first.at("cache_entries") == 1 &&
              second.at("cache_entries") == 1 && handles.size() == 2 &&
              first.at("results").at(0).at("plan_handle") !=
                  second.at("results").at(0).at("plan_handle"),
          "separate legacy request_id wires share a canonical Plan and fresh handles");
  require(first.at("results").at(0).at("summary") ==
                  second.at("results").at(0).at("summary") &&
              first.at("cache_accounted_bytes").get<std::size_t>() <
                  second.at("cache_accounted_bytes").get<std::size_t>() &&
              second.at("cache_accounted_bytes").get<std::size_t>() <= 64u*1024u*1024u,
          "two legacy raw encodings share one Plan and account both retained memos");
  const auto first_again=compile_batch(model.get(),json::array({legacy_norm_task("request-a")}));
  handles.track(first_again);
  const auto third=compile_batch(model.get(),json::array({legacy_norm_task("request-c")}));
  handles.track(third);
  require(first_again.at("cache_accounted_bytes")==second.at("cache_accounted_bytes") &&
      third.at("cache_accounted_bytes")==second.at("cache_accounted_bytes") &&
      third.at("results").at(0).at("summary")==first.at("results").at(0).at("summary") &&
      third.at("results").at(0).at("plan_handle")!=first_again.at("results").at(0).at("plan_handle"),
      "legacy memo hits and a third equal-length replacement keep two bounded memos and fresh handles");
  require(handles.release_all() == DIE_OK,
          "legacy canonical-hit handles release normally");
}

void meaningful_task_fields_remain_in_the_key() {
  const Hardware hardware = example_hardware();
  CModel model(hardware);
  const json base = gemm_task();
  json tp = base;
  tp["tp_slice"] = {{"tensor_parallel", 2}, {"tensor_parallel_rank", 0}};
  json precision = base;
  precision["storage_dtype"] = {{"weights", "int4"}};
  json spans = base;
  spans["spans"] = json::array({{{"request_id", "trace-one"},
                                 {"q_tokens", 1}, {"kv_tokens", 1}}});
  json ready = base;
  ready["readiness_scope"] = "after_local_store";
  json shape = gemm_task(18, 5, 3);
  const auto response = estimate_cost_batch(
      model.get(), json::array({base, tp, precision, spans, ready, shape}));
  const auto& rows = response.at("results");
  require(rows.size() == 6 && response.at("errors").size() == 6,
          "meaningful-field cost rows remain positional");
  for (std::size_t i = 0; i < rows.size(); ++i)
    require(rows.at(i).is_object() && response.at("errors").at(i).is_null(),
            "shape/TP/precision/span/readiness variants compile");
  std::set<std::string> hashes;
  for (const auto& row : rows)
    hashes.insert(row.at("plan_hash").get<std::string>());
  require(hashes.size() == rows.size(),
          "shape, TP, precision, span, and readiness changes do not collide");
  require(response.at("cache_entries") == rows.size() &&
              response.at("cache_accounted_bytes").get<std::size_t>() > 0,
          "each meaningful task variant retains its own accounted cache entry");
}

void cost_only_matches_full_estimate_and_runtime() {
  const Hardware hardware = example_hardware();
  const json task = gemm_task(7, 5, 9);
  CModel model(hardware);

  const auto cost = estimate_cost_batch(model.get(), json::array({task}));
  require(cost.size() == 6 && cost.at("schema_version") == 2 &&
              cost.at("results").size() == 1 &&
              cost.at("errors").at(0).is_null(),
          "cost-only API preserves the positional batch and cache envelope");
  const auto& cost_row = cost.at("results").at(0);
  require(cost_row.size() == 3 && cost_row.at("status") == "ok_assumed" &&
              cost_row.at("estimated_latency_ns").is_number_unsigned() &&
              cost_row.at("plan_hash").is_string(),
          "cost-only rows expose only exact cost, status, and Plan hash");
  const auto cost_ns = cost_row.at("estimated_latency_ns").get<std::uint64_t>();
  const auto cache_entries = cost.at("cache_entries");
  const auto cache_bytes = cost.at("cache_accounted_bytes");
  require(cache_entries == 1 && cache_bytes.get<std::size_t>() > 0,
          "cost query populates the same accounted Plan cache");

  const auto full = estimate_batch(model.get(),
                                   json::array({alias_gemm_task(7, 5, 9)}));
  const auto& report = full.at("results").at(0);
  require(report.at("estimated_latency_ns").get<std::uint64_t>() == cost_ns &&
              report.at("plan_fingerprint") == cost_row.at("plan_hash") &&
              report.at("status") == cost_row.at("status"),
          "lightweight cost matches the full isolated estimate report");
  require(full.at("cache_entries") == cache_entries &&
              full.at("cache_accounted_bytes").get<std::size_t>() > 0 &&
              full.at("cache_accounted_bytes").get<std::size_t>() <= 64u*1024u*1024u,
          "full estimate shares the normalized Plan while accounting its different raw-request memo");

  const auto compiled = compile(hardware, task);
  require(cost_ns == direct_runtime_finish(hardware, compiled),
          "lightweight selected cost matches an independent isolated Runtime");
  require(cost_row.at("plan_hash") == compiled->fingerprint,
          "lightweight cost returns the exact canonical Plan fingerprint");
}

void isolated_cost_errors_overflow_and_model_lifecycle() {
  const Hardware hardware = example_hardware();
  CModel model(hardware);
  const json overflowing = gemm_task(std::numeric_limits<std::uint64_t>::max(),
                                     std::numeric_limits<std::uint64_t>::max(),
                                     std::numeric_limits<std::uint64_t>::max());
  const auto response = estimate_cost_batch(
      model.get(), json::array({gemm_task(), overflowing}));
  require(response.at("results").at(0).is_object() &&
              response.at("errors").at(0).is_null() &&
              response.at("results").at(1).is_null() &&
              response.at("errors").at(1).is_object() &&
              response.at("errors").at(1).at("status_code") == DIE_OVERFLOW,
          "overflow stays isolated to its task and retains the established code");
  require(response.at("cache_entries") == 1 &&
              response.at("cache_accounted_bytes").get<std::size_t>() > 0,
          "failed overflow input does not enter the cache or corrupt its counters");
  const DieHandle stale_model = model.get();
  require(model.destroy() == DIE_OK,
          "cost-only queries allocate no Plan handles and permit model destruction");
  const std::string request = request_text(json::array({gemm_task()}));
  require(die_coarse_estimate_cost_batch_v2(stale_model, request.c_str()) == nullptr &&
              json::parse(die_coarse_last_error_v2()).at("status_code") ==
                  DIE_STALE_HANDLE,
          "cost query rejects a destroyed model with the established stale-handle error");
}

void body() {
  require(die_coarse_abi_version() == 2,
          "cost-only query is additive within the existing C ABI v2");
  canonical_cache_and_fresh_compile_handles();
  legacy_trace_fields_share_a_canonical_entry();
  meaningful_task_fields_remain_in_the_key();
  cost_only_matches_full_estimate_and_runtime();
  isolated_cost_errors_overflow_and_model_lifecycle();
}

}  // namespace

int main() { return run_test("capi_cost_query_test", body); }
