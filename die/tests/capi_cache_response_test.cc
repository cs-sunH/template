#include "test_support.hh"
#include "coarse_die/Capi.h"
#include "coarse_die/Memory.hh"

#include <cstdint>
#include <iostream>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

using namespace coarse_die;
using namespace coarse_die_test;

namespace {

json gemm_task(std::uint64_t m, std::uint64_t n = 1, std::uint64_t k = 1) {
  return {{"op_kind", "gemm"},
          {"tensor_shapes", {{"m", m}, {"n", n}, {"k", k}}},
          {"storage_dtype", {{"weights", "bf16"}, {"activations", "bf16"},
                             {"output", "bf16"}}},
          {"execution_profile", "int4_matrix_vec_fp32_assumed_v1"},
          {"layer_instances", 1},
          {"weight_passes", 1}};
}

json fractional_task() {
  return {{"op_kind", "gemm"},
          {"tensor_shapes", {{"m", 1}, {"n", 1}, {"k", 1}}},
          {"storage_dtype", {{"weights", "bf16"}, {"activations", "bf16"},
                             {"output", "int4"}}},
          {"execution_profile", "int4_matrix_vec_fp32_assumed_v1"}};
}

class CModel {
 public:
  explicit CModel(const Hardware& hardware) {
    const auto config = hardware.input.dump();
    require(die_coarse_create_model_v2(config.c_str(), &handle_) == DIE_OK,
            std::string("C ABI model create failed: ") + die_coarse_last_error_v2());
  }
  CModel(const CModel&) = delete;
  CModel& operator=(const CModel&) = delete;
  ~CModel() {
    if (handle_) (void)die_coarse_destroy_model_v2(handle_);
  }
  DieHandle get() const { return handle_; }

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
  DieHandle at(std::size_t index) const { return handles_.at(index); }
  std::size_t size() const { return handles_.size(); }
  int release(std::size_t index) {
    auto& handle = handles_.at(index);
    if (!handle) return DIE_STALE_HANDLE;
    const int status = die_coarse_release_plan_v2(handle);
    if (status == DIE_OK) handle = 0;
    return status;
  }

 private:
  std::vector<DieHandle> handles_;
};

json take_json(const char* raw, const char* api) {
  require(raw != nullptr,
          std::string(api) + " failed: " + die_coarse_last_error_v2());
  const std::string text(raw);
  die_coarse_free_result_v2(raw);
  const auto parsed = json::parse(text);
  require(parsed.dump() == text,
          std::string(api) + " keeps canonical compact JSON field order and encoding");
  return parsed;
}

json compile_batch(DieHandle model, const json& specs) {
  const auto request = json{{"specs", specs}}.dump();
  return take_json(die_coarse_compile_batch_v2(model, request.c_str()),
                   "compile_batch");
}

json estimate_batch(DieHandle model, const json& specs) {
  const auto request = json{{"specs", specs}}.dump();
  return take_json(die_coarse_estimate_batch_v2(model, request.c_str()),
                   "estimate_batch");
}

void check_batch_envelope(const json& response, std::size_t rows) {
  require(response.size() == 6 && response.at("schema_version") == 2 &&
              response.at("schema_version").is_number_integer(),
          "compile response keeps the same top-level keys and schema type");
  require(response.at("results").is_array() && response.at("results").size() == rows &&
              response.at("errors").is_array() && response.at("errors").size() == rows,
          "results and errors remain positional arrays");
  require(response.at("cache_entries").is_number_unsigned() &&
              response.at("cache_accounted_bytes").is_number_unsigned() &&
              response.at("cache_accounting") ==
                  "retained_container_capacities_with_allocator_allowances",
          "cache diagnostics preserve their public types and accounting label");
}

void estimate_prefill_mixed_rows_and_fractional_marker() {
  const Hardware hardware = example_hardware();
  const json task = gemm_task(17, 5, 3);
  const auto cpp_plan = compile(hardware, task);
  CModel model(hardware);
  PlanHandles handles;

  const auto estimate = estimate_batch(model.get(), json::array({task}));
  require(estimate.size() == 6 && estimate.at("schema_version") == 2 &&
              estimate.at("results").is_array() &&
              estimate.at("results").size() == 1 &&
              estimate.at("errors").is_array() &&
              estimate.at("errors").size() == 1 &&
              estimate.at("errors").at(0).is_null() &&
              estimate.at("cache_entries") == 1 &&
              estimate.at("cache_accounting") ==
                  "retained_container_capacities_with_allocator_allowances",
          "estimate_batch preserves its public envelope and still populates the cache");
  const auto cache_bytes = estimate.at("cache_accounted_bytes").get<std::size_t>();
  const std::string summary_json = cpp_plan->summary.dump();
  const std::string escaped_hash_json = json(cpp_plan->fingerprint).dump();
  const std::size_t minimum_retained = retained_plan_bytes(*cpp_plan) +
      summary_json.size() + 1 + escaped_hash_json.size() + 1 + 512;
  require(cache_bytes >= minimum_retained,
          "cache accounting includes the cached summary/hash JSON strings and object allowance");

  const json invalid{{"op_kind", "cache_response_test_unsupported"}};
  const auto batch = compile_batch(model.get(), json::array({task, invalid, task}));
  handles.track(batch);
  check_batch_envelope(batch, 3);
  require(batch.at("cache_entries") == estimate.at("cache_entries") &&
              batch.at("cache_accounted_bytes") == estimate.at("cache_accounted_bytes"),
          "compile hits the estimate-populated cache without changing its retained size");
  const auto& rows = batch.at("results");
  const auto& errors = batch.at("errors");
  require(rows.at(0).is_object() && rows.at(1).is_null() && rows.at(2).is_object() &&
              errors.at(0).is_null() && !errors.at(1).is_null() && errors.at(2).is_null(),
          "mixed compile success/failure siblings remain at their original indices");
  const std::size_t integer_row_keys=has_credit_curves(*cpp_plan) ? 4 : 3;
  require(rows.at(0).size() == integer_row_keys && rows.at(2).size() == integer_row_keys &&
              rows.at(0).at("plan_hash").is_string() &&
              rows.at(0).at("plan_handle").is_number_unsigned() &&
              rows.at(0).at("summary").is_object(),
          "integer compile rows preserve value types and advertise a new curve schema");
  if(has_credit_curves(*cpp_plan))
    require(rows.at(0).at("plan_schema")=="coarse_die_plan_v4" &&
                rows.at(2).at("plan_schema")=="coarse_die_plan_v4",
            "curve rows identify Plan v4 even without fractional work");
  require(rows.at(0).at("plan_hash") == cpp_plan->fingerprint &&
              rows.at(0).at("summary") == cpp_plan->summary &&
              rows.at(2).at("plan_hash") == cpp_plan->fingerprint &&
              rows.at(2).at("summary") == cpp_plan->summary,
          "cached response fragments preserve the exact C++ plan hash and summary");
  require(rows.at(0).at("plan_handle") != rows.at(2).at("plan_handle"),
          "every successful occurrence receives a fresh plan handle on a cache hit");
  require(errors.at(1).size() == 3 && errors.at(1).at("field_path").is_string() &&
              errors.at(1).at("reason").is_string() &&
              errors.at(1).at("status_code").is_number_integer() &&
              errors.at(1).at("status_code").get<int>() != DIE_OK,
          "per-row errors retain the field_path/reason/status_code siblings and types");
  const auto reason = errors.at(1).at("reason").get<std::string>();
  require(errors.at(1).at("field_path").get<std::string>() ==
              reason.substr(0, reason.find(':')),
          "per-row field_path remains the reason prefix");
  require(handles.size() == 2 && handles.release(0) == DIE_OK &&
              handles.release(1) == DIE_OK,
          "both fresh handles from repeated cache hits are releasable");

  auto fractional_config = hardware.input;
  fractional_config["performance_profile"]["kappa_quant_eqops"] = 1.25;
  fractional_config["performance_profile"]["kappa_acc_convert_eqops"] = 0.75;
  const Hardware fractional_hardware = resolve_hardware(fractional_config);
  const auto fractional_plan = compile(fractional_hardware, fractional_task());
  CModel fractional_model(fractional_hardware);
  PlanHandles fractional_handles;
  const auto fractional = compile_batch(fractional_model.get(),
      json::array({fractional_task()}));
  fractional_handles.track(fractional);
  check_batch_envelope(fractional, 1);
  const auto& row = fractional.at("results").at(0);
  require(row.size() == 5 && row.at("equivalent_work_schema") ==
              "dyadic_equivalent_work_v1" && row.at("plan_schema") ==
              (has_credit_curves(*fractional_plan) ? "coarse_die_plan_v4" : "coarse_die_plan_v3"),
          "fractional rows retain both schema markers and their original types");
  require(row.at("plan_hash") == fractional_plan->fingerprint &&
              row.at("summary") == fractional_plan->summary &&
              row.at("summary").at("vec_equivalent_ops") ==
                  fractional_plan->summary.at("vec_equivalent_ops"),
          "fractional cached summary preserves its exact public JSON value");
}

std::size_t one_entry_bytes(const Hardware& hardware, const json& task) {
  CModel model(hardware);
  const auto response = estimate_batch(model.get(), json::array({task,
      json{{"op_kind","cache_response_test_unsupported"}}}));
  require(response.at("cache_entries") == 1,
          "single-task footprint fixture retains exactly one cache entry");
  return response.at("cache_accounted_bytes").get<std::size_t>();
}

void raw_batch_memo_hit_invalidation_and_failed_siblings() {
  const Hardware hardware = example_hardware();
  const json first_task = gemm_task(17, 5, 3);
  const json inserted_task = gemm_task(19, 5, 3);
  const json invalid{{"op_kind", "cache_response_test_unsupported"}};
  CModel model(hardware);
  PlanHandles handles;

  // A failed sibling keeps this request out of the raw memo while still
  // compiling and retaining the successful task's Plan.
  const auto primed = estimate_batch(
      model.get(), json::array({first_task, invalid}));
  require(primed.at("results").at(0).is_object() &&
              primed.at("errors").at(0).is_null() &&
              primed.at("results").at(1).is_null() &&
              !primed.at("errors").at(1).is_null() &&
              primed.at("cache_entries") == 1,
          "a failed sibling leaves its successful Plan cached");
  const std::size_t plan_only_bytes =
      primed.at("cache_accounted_bytes").get<std::size_t>();

  const auto first = compile_batch(model.get(), json::array({first_task}));
  handles.track(first);
  const std::size_t memo_bytes =
      first.at("cache_accounted_bytes").get<std::size_t>();
  require(first.at("results").at(0).is_object() &&
              first.at("errors").at(0).is_null() &&
              first.at("cache_entries") == 1 && memo_bytes > plan_only_bytes,
          "an all-success raw batch retains its memo inside cache accounting");

  const auto memo_hit = compile_batch(model.get(), json::array({first_task}));
  handles.track(memo_hit);
  require(memo_hit.at("cache_accounted_bytes") == memo_bytes &&
              memo_hit.at("results").at(0).at("plan_hash") ==
                  first.at("results").at(0).at("plan_hash") &&
              memo_hit.at("results").at(0).at("summary") ==
                  first.at("results").at(0).at("summary") &&
              memo_hit.at("results").at(0).at("plan_handle") !=
                  first.at("results").at(0).at("plan_handle"),
          "an exact raw memo hit preserves the row and allocates a fresh Plan handle");

  // Inserting a new map entry invalidates pointers held by the previous memo.
  // The failing sibling then prevents this mixed request from installing a
  // replacement memo of its own.
  const auto inserted = compile_batch(
      model.get(), json::array({inserted_task, invalid}));
  handles.track(inserted);
  const std::size_t inserted_plan_bytes = one_entry_bytes(hardware, inserted_task);
  const std::size_t two_plan_bytes = plan_only_bytes + inserted_plan_bytes;
  require(inserted.at("results").at(0).is_object() &&
              inserted.at("errors").at(0).is_null() &&
              inserted.at("results").at(1).is_null() &&
              !inserted.at("errors").at(1).is_null() &&
              inserted.at("cache_entries") == 2 &&
              inserted.at("cache_accounted_bytes") == two_plan_bytes,
          "a miss clears the old memo before insertion and a failed batch retains no new memo");

  const auto failed_repeat = compile_batch(
      model.get(), json::array({inserted_task, invalid}));
  handles.track(failed_repeat);
  require(failed_repeat.at("cache_entries") == 2 &&
              failed_repeat.at("cache_accounted_bytes") == two_plan_bytes &&
              failed_repeat.at("results").at(0).at("plan_hash") ==
                  inserted.at("results").at(0).at("plan_hash") &&
              failed_repeat.at("results").at(0).at("plan_handle") !=
                  inserted.at("results").at(0).at("plan_handle"),
          "repeating a failed-sibling batch neither memoizes it nor reuses its handle");

  // Re-requesting the original successful batch installs its memo again after
  // the intervening map insertion invalidated it.
  const auto restored = compile_batch(model.get(), json::array({first_task}));
  handles.track(restored);
  require(restored.at("cache_accounted_bytes") == two_plan_bytes +
              (memo_bytes - plan_only_bytes) &&
              restored.at("results").at(0).at("plan_handle") !=
                  first.at("results").at(0).at("plan_handle"),
          "a successful batch can be memoized again after cache insertion");
  require(handles.size() == 5,
          "each successful memo, insert, and hit result owns an independent handle");
}

void submit_and_cancel_plan(DieHandle model, DieHandle plan) {
  DieHandle runtime = 0;
  require(die_coarse_create_runtime_v2(model, &runtime) == DIE_OK,
          std::string("runtime create failed: ") + die_coarse_last_error_v2());
  const DieOwnerV2 owner{sizeof(DieOwnerV2), 2, 71, 72, 73, 1};
  DieHandle execution = 0;
  require(die_coarse_submit_plan_v2(runtime, plan, &owner, 0, &execution) == DIE_OK,
          std::string("evicted plan handle is no longer usable: ") +
              die_coarse_last_error_v2());
  require(die_coarse_cancel_execution_v2(execution, 0) == DIE_OK,
          std::string("immediate cleanup cancellation failed: ") +
              die_coarse_last_error_v2());
  require(die_coarse_destroy_execution_v2(execution) == DIE_OK,
          "cancelled retained plan execution can be destroyed");
  require(die_coarse_destroy_runtime_v2(runtime) == DIE_OK,
          "empty runtime after handle-survival check can be destroyed");
}

void cache_capacity_lru_accounting_and_evicted_batch_rows() {
  const Hardware hardware = example_hardware();
  constexpr std::uint64_t capacity = 4096;
  constexpr std::size_t byte_limit = 64u * 1024u * 1024u;
  CModel model(hardware);
  PlanHandles handles;
  // Isolate the Plan LRU's exact byte accounting from the optional raw-batch
  // shortcut. A failed sibling is never memoized and must not affect the hit.
  const auto estimate_without_memo = [&](const json& task) {
    return estimate_batch(model.get(),json::array({task,
        json{{"op_kind","cache_response_test_unsupported"}}}));
  };

  // Keep fixtures tiny and vary a real GEMM dimension for unique cache keys.
  // The byte cap is reached before 4096 of these compiled Plans can fit, which
  // verifies that cached summary/hash strings count against retained storage.
  std::vector<std::uint64_t> dimensions;
  dimensions.reserve(capacity);
  for (std::uint64_t m = 1; m <= capacity; ++m) dimensions.push_back(m);
  require(dimensions.size() == capacity,
          "capacity fixture has exactly 4096 unique small GEMM tasks");
  json specs = json::array();
  for (const auto m : dimensions) specs.push_back(gemm_task(m));
  specs.push_back(gemm_task(1));
  specs.push_back(gemm_task(capacity + 1));

  const auto response = compile_batch(model.get(), specs);
  handles.track(response);
  check_batch_envelope(response, capacity + 2);
  const std::size_t entries_after_boundary =
      response.at("cache_entries").get<std::size_t>();
  const std::size_t bytes_after_boundary =
      response.at("cache_accounted_bytes").get<std::size_t>();
  require(entries_after_boundary > 0 && entries_after_boundary < capacity &&
              bytes_after_boundary <= byte_limit &&
              bytes_after_boundary > byte_limit - 64u * 1024u,
          "real insertion path reaches the retained-byte cap below 4096 entries (entries=" +
              std::to_string(response.at("cache_entries").get<std::size_t>()) +
              ", bytes=" +
              std::to_string(bytes_after_boundary) +
              ")");
  require(handles.size() == capacity + 2,
          "each successful result row commits its own handle after output allocation");
  const auto& rows = response.at("results");
  const auto& errors = response.at("errors");
  for (std::size_t i = 0; i < rows.size(); ++i) {
    require(rows.at(i).is_object() && errors.at(i).is_null(),
            "all small capacity-boundary GEMMs compile successfully");
  }
  const std::size_t repeated_first = capacity;
  const std::size_t final_new = capacity + 1;
  require(rows.at(0).at("plan_hash") == rows.at(repeated_first).at("plan_hash") &&
              rows.at(0).at("summary") == rows.at(repeated_first).at("summary") &&
              rows.at(0).at("plan_handle") != rows.at(repeated_first).at("plan_handle"),
          "LRU hit still creates a fresh handle and stable JSON row");
  const auto plan_for_m2 = compile(hardware, gemm_task(2));
  require(rows.at(1).at("summary") == plan_for_m2->summary &&
              rows.at(1).at("plan_hash") == plan_for_m2->fingerprint,
          "a row whose cache entry is evicted later in the same batch keeps its shared response template");
  require(rows.at(final_new).at("plan_hash") ==
              compile(hardware, gemm_task(capacity + 1))->fingerprint,
          "the capacity-crossing result serializes after the expected eviction");

  const auto first_hit = estimate_without_memo(gemm_task(1));
  require(first_hit.at("cache_entries") == entries_after_boundary &&
              first_hit.at("cache_accounted_bytes") == bytes_after_boundary,
          "the most-recently accessed task remains a true cache hit at the byte boundary");

  // The original row's plan handle owns a shared Plan independently of the
  // cache entry; use it after that entry was evicted and before reinsertion.
  submit_and_cancel_plan(model.get(), handles.at(1));

  // Insertions retain a suffix of the unique task order; the repeated m=1 hit
  // is moved to MRU before m=4097 is inserted. The count therefore identifies
  // the oldest suffix key that must be evicted on the next miss.
  const std::uint64_t oldest_suffix = 4099 - entries_after_boundary;
  require(oldest_suffix >= 3 && oldest_suffix <= capacity,
          "cache count identifies the retained suffix after the recent m=1 hit");
  const std::size_t bytes_for_m2 = one_entry_bytes(hardware, gemm_task(2));
  const std::size_t bytes_for_oldest =
      one_entry_bytes(hardware, gemm_task(oldest_suffix));
  require(bytes_for_m2 <= bytes_for_oldest &&
              bytes_after_boundary + bytes_for_m2 > byte_limit,
          "the evicted early task needs exactly one accounted LRU victim on reinsertion");
  const std::size_t expected_after_m2 =
      bytes_after_boundary + bytes_for_m2 - bytes_for_oldest;
  const auto miss_for_evicted = estimate_without_memo(gemm_task(2));
  require(miss_for_evicted.at("cache_entries") == entries_after_boundary &&
              miss_for_evicted.at("cache_accounted_bytes") == expected_after_m2,
          "access to the evicted task reinserts it and evicts exactly the oldest retained entry");
  const auto recent_again = estimate_without_memo(gemm_task(1));
  require(recent_again.at("cache_accounted_bytes") == expected_after_m2,
          "the recent entry stays cached after a boundary miss evicts the oldest suffix entry");
  const auto hit_for_m2 = estimate_without_memo(gemm_task(2));
  require(hit_for_m2.at("cache_accounted_bytes") == expected_after_m2,
          "reinserted entry becomes a cache hit without changing retained accounting");
}

void body() {
  require(die_coarse_abi_version() == 2,
          "response writer optimization preserves C ABI version 2");
  estimate_prefill_mixed_rows_and_fractional_marker();
  raw_batch_memo_hit_invalidation_and_failed_siblings();
  cache_capacity_lru_accounting_and_evicted_batch_rows();
}

}  // namespace

int main() { return run_test("capi_cache_response_test", body); }
