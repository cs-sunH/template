#include "test_support.hh"
#include "coarse_die/Capi.h"
#include "../src/CompilerInternal.hh"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <iterator>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace coarse_die::detail {
std::vector<std::shared_ptr<Plan>> enumerate_mappings(
    const Hardware&, const nlohmann::json& normalized_task);
}

namespace {
using namespace coarse_die;
using namespace coarse_die_test;
using Clock = std::chrono::steady_clock;
using Json = nlohmann::json;

constexpr std::uint64_t kCompileGateNs = 2'000'000;
constexpr std::uint64_t kHotQueryGateNs = 20'000;
constexpr std::uint64_t kCapacityUpdateGateNs = 50'000;
constexpr std::size_t kMaxSamples = 1000;
constexpr std::array<std::size_t, 3> kOwnerCounts{1, 8, 32};

struct Args {
  std::string config_path = std::string(COARSE_DIE_SOURCE_DIR) +
                            "/configs/example_hardware.json";
  std::string task_path = std::string(COARSE_DIE_SOURCE_DIR) +
                          "/configs/example_tasks.json";
  std::string output_path;
  std::size_t samples = 30;
};

struct OsBulkPlan {
  std::shared_ptr<const Plan> plan;
  std::size_t bulk_epoch = 0;
  std::size_t bulk_activity_count = 0;
  std::size_t credit_curve_activity_count = 0;
  std::string dataflow_id;
  std::string template_id;
};

struct CModel {
  DieHandle handle = 0;
  explicit CModel(const Json& config) {
    const std::string text = config.dump();
    require(die_coarse_create_model_v2(text.c_str(), &handle) == DIE_OK,
            std::string("C ABI model creation failed: ") +
                die_coarse_last_error_v2());
  }
  CModel(const CModel&) = delete;
  CModel& operator=(const CModel&) = delete;
  ~CModel() {
    if (handle) (void)die_coarse_destroy_model_v2(handle);
  }
};

std::string read_text(const std::string& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) throw std::runtime_error("cannot open input: " + path);
  return std::string(std::istreambuf_iterator<char>(input),
                     std::istreambuf_iterator<char>());
}

Json read_json_file(const std::string& path) {
  return Json::parse(read_text(path));
}

Args parse_args(int argc, char** argv) {
  Args args;
  for (int i = 1; i < argc; ++i) {
    const std::string option = argv[i];
    const auto next = [&]() -> std::string {
      if (i + 1 >= argc)
        throw std::runtime_error("missing value for " + option);
      return argv[++i];
    };
    if (option == "--samples") {
      const auto value = std::stoull(next());
      if (value == 0 || value > kMaxSamples)
        throw std::runtime_error("--samples must be in [1, 1000]");
      args.samples = static_cast<std::size_t>(value);
    } else if (option == "--output") {
      args.output_path = next();
    } else if (option == "--config") {
      args.config_path = next();
    } else if (option == "--task") {
      args.task_path = next();
    } else if (option == "--help") {
      std::cout << "usage: service_curve_benchmark --output PATH [--samples N] "
                   "[--config JSON] [--task JSON]\n";
      std::exit(0);
    } else {
      throw std::runtime_error("unknown argument: " + option);
    }
  }
  if (args.output_path.empty())
    throw std::runtime_error("--output PATH is required");
  return args;
}

std::uint64_t elapsed_ns(Clock::time_point begin, Clock::time_point end) {
  const auto count =
      std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count();
  return count < 0 ? 0 : static_cast<std::uint64_t>(count);
}

std::uint64_t nearest_rank(std::vector<std::uint64_t> values, double p) {
  if (values.empty()) return 0;
  std::sort(values.begin(), values.end());
  const auto rank = static_cast<std::size_t>(
      std::ceil(p * static_cast<double>(values.size())));
  return values[std::max<std::size_t>(1, rank) - 1];
}

Json metric_report(const std::vector<std::uint64_t>& samples,
                   std::uint64_t threshold_ns,
                   const std::string& threshold_basis) {
  const auto p50 = nearest_rank(samples, 0.50);
  const auto p95 = nearest_rank(samples, 0.95);
  const auto p99 = nearest_rank(samples, 0.99);
  const auto maximum = samples.empty()
      ? 0 : *std::max_element(samples.begin(), samples.end());
  // The frozen acceptance target is P95. P50/P99 remain diagnostics.
  const bool pass = !samples.empty() && p95 <= threshold_ns;
  return {{"sample_count", samples.size()},
          {"samples_ns", samples},
          {"p50_ns", p50},
          {"p95_ns", p95},
          {"p99_ns", p99},
          {"max_ns", maximum},
          {"fixed_threshold_ns", threshold_ns},
          {"threshold_basis", threshold_basis},
          {"threshold_checks", {{"p50", p50 <= threshold_ns},
                                 {"p95", p95 <= threshold_ns},
                                 {"p99", p99 <= threshold_ns}}},
          {"passed", pass}};
}

void fail_record(Json& failures, const std::string& where,
                 const std::string& reason) {
  failures.push_back({{"case", where}, {"reason", reason}});
}

std::string task_op(const Json& task) {
  if (!task.is_object()) return {};
  if (task.contains("die_op_spec") && task.at("die_op_spec").is_object())
    return task.at("die_op_spec").value("op_kind", std::string());
  return task.value("op_kind", std::string());
}

Json select_example_attention(const Json& input) {
  if (input.is_object() && input.contains("specs") &&
      input.at("specs").is_array()) {
    for (const auto& task : input.at("specs")) {
      const std::string op = task_op(task);
      if (op == "attention" || op == "attention_region") return task;
    }
  } else if (task_op(input) == "attention" ||
             task_op(input) == "attention_region") {
    return input;
  }
  throw std::runtime_error(
      "task input has no attention/attention_region spec; use the real "
      "example_tasks.json format or a single attention task");
}

Json gemm_task(std::uint64_t m, std::uint64_t n, std::uint64_t k,
               const std::string& profile) {
  return {{"op_kind", "gemm"},
          {"tensor_shapes", {{"m", m}, {"n", n}, {"k", k}}},
          {"storage_dtype", {{"weights", "bf16"},
                             {"activations", "bf16"},
                             {"output", "bf16"}}},
          {"execution_profile", profile},
          {"layer_instances", 1},
          {"weight_passes", 1}};
}

Json gemm_semantic_alias(const Json& task, const std::string& profile) {
  const auto& shapes = task.at("tensor_shapes");
  return {{"schema", "coarse_die_task_v2"},
          {"op_kind", "gemm"},
          {"phase", "prefill"},
          {"tensor_shapes", {{"M", shapes.at("m")},
                             {"N", shapes.at("n")},
                             {"K", shapes.at("k")}}},
          {"storage_dtype", "bf16"},
          {"execution_profile_id", profile},
          {"tp_slice", {{"tensor_parallel", 1},
                        {"tensor_parallel_rank", 0}}},
          {"layer_instances", 1},
          {"weight_passes", 1},
          {"fusion_policy", "auto"}};
}

Json semantic_trace_alias(const Json& task) {
  Json alias = task;
  if (alias.contains("die_op_spec")) {
    auto& spec = alias.at("die_op_spec");
    if (spec.contains("spans") && spec.at("spans").is_array()) {
      for (auto& span : spec["spans"]) {
        if (span.contains("request_id") && span.at("request_id").is_string())
          span["request_id"] = span.at("request_id").get<std::string>() +
                               "-alias";
        if (span.contains("request_id_prefix") &&
            span.at("request_id_prefix").is_string())
          span["request_id_prefix"] =
              span.at("request_id_prefix").get<std::string>() + "-alias";
      }
    }
  } else if (alias.contains("spans") && alias.at("spans").is_array()) {
    for (auto& span : alias["spans"]) {
      if (span.contains("request_id") && span.at("request_id").is_string())
        span["request_id"] = span.at("request_id").get<std::string>() +
                             "-alias";
      if (span.contains("request_id_prefix") &&
          span.at("request_id_prefix").is_string())
        span["request_id_prefix"] =
            span.at("request_id_prefix").get<std::string>() + "-alias";
    }
  }
  return alias;
}

std::string batch_text(const Json& task) {
  return Json{{"specs", Json::array({task})}}.dump();
}

Json take_result(const char* raw, const char* api) {
  require(raw != nullptr,
          std::string(api) + " failed: " + die_coarse_last_error_v2());
  const std::string text(raw);
  die_coarse_free_result_v2(raw);
  return Json::parse(text);
}

const Json& successful_row(const Json& response, const std::string& api) {
  require(response.at("results").size() == 1 &&
              response.at("errors").size() == 1 &&
              response.at("errors").at(0).is_null() &&
              response.at("results").at(0).is_object(),
          api + " returned a failed or non-positional task result");
  return response.at("results").at(0);
}

void release_row_plan(const Json& row, const std::string& api) {
  const DieHandle plan = row.at("plan_handle").get<DieHandle>();
  require(plan != 0, api + " returned a zero plan handle");
  require(die_coarse_release_plan_v2(plan) == DIE_OK,
          std::string(api) + " plan release failed: " +
              die_coarse_last_error_v2());
}

Json cache_stats(const Json& response) {
  return {{"entries", response.at("cache_entries")},
          {"accounted_bytes", response.at("cache_accounted_bytes")},
          {"accounting", response.at("cache_accounting")}};
}

template <class Call, class Validate>
Json measure_capi(std::size_t count, std::uint64_t threshold_ns,
                  const std::string& threshold_basis, Call&& call,
                  Validate&& validate) {
  std::vector<std::uint64_t> samples;
  samples.reserve(count);
  for (std::size_t i = 0; i < count; ++i) {
    const auto begin = Clock::now();
    const char* raw = call();
    const auto end = Clock::now();
    samples.push_back(elapsed_ns(begin, end));
    const Json response = take_result(raw, "C ABI benchmark call");
    validate(response);  // JSON parsing and handle release are outside timing.
  }
  return metric_report(samples, threshold_ns, threshold_basis);
}

Json compile_query_fixture(const Hardware& hardware, const Json& config,
                           const Json& task, const Json& alias,
                           const std::string& name, std::size_t count,
                           Json& failures) {
  Json result{{"case", name}, {"status", "ok"}};
  try {
    const auto warm_plan = compile(hardware, task);
    std::vector<std::uint64_t> cpp_samples;
    cpp_samples.reserve(count);
    bool cpp_hash_stable = true;
    for (std::size_t i = 0; i < count; ++i) {
      const auto begin = Clock::now();
      const auto plan = compile(hardware, task);  // direct API has no cache.
      const auto end = Clock::now();
      cpp_samples.push_back(elapsed_ns(begin, end));
      cpp_hash_stable = cpp_hash_stable &&
                        plan->fingerprint == warm_plan->fingerprint;
    }
    result["metrics"]["cpp_cold_compile_uncached"] =
        metric_report(cpp_samples, kCompileGateNs, "fixed_2ms_compile_gate");
    result["metrics"]["cpp_cold_compile_uncached"]["same_plan_hash_each_call"] =
        cpp_hash_stable;
    const auto op=task.value("op_kind",std::string());
    const bool regular=op=="gemm"||op=="norm"||op=="residual"||op=="mlp_activation";
    result["metrics"]["cpp_cold_compile_uncached"]["acceptance_applies"] = regular;
    result["metrics"]["cpp_cold_compile_uncached"]["scope"] =
        regular ? "regular_compile" : "composite_kernel_reported_separately_per_design_13";
    if (!cpp_hash_stable)
      fail_record(failures, name, "uncached C++ compile changed Plan hash");

    CModel model(config);
    const std::string canonical_request = batch_text(task);
    const std::string alias_request = batch_text(alias);
    Json plan_only_cache;
    {
      // A failed sibling prevents raw-batch memo installation while still
      // retaining the canonical Plan, giving this fixture a Plan-only byte
      // baseline for the later memo-accounting checks.
      CModel footprint_model(config);
      const Json invalid{{"op_kind", "benchmark_unsupported_sibling"}};
      const std::string footprint_request =
          Json{{"specs", Json::array({task, invalid})}}.dump();
      const Json footprint = take_result(
          die_coarse_estimate_batch_v2(footprint_model.handle,
                                       footprint_request.c_str()),
          "Plan-only cache footprint probe");
      require(footprint.at("results").at(0).is_object() &&
                  footprint.at("errors").at(0).is_null() &&
                  footprint.at("results").at(1).is_null() &&
                  !footprint.at("errors").at(1).is_null() &&
                  footprint.at("cache_entries") == 1,
              name + " Plan-only footprint probe did not isolate its bad sibling");
      plan_only_cache = cache_stats(footprint);
    }
    const std::size_t plan_only_bytes =
        plan_only_cache.at("accounted_bytes").get<std::size_t>();
    constexpr std::size_t cache_byte_limit = 64u * 1024u * 1024u;

    const auto warm_compile = take_result(
        die_coarse_compile_batch_v2(model.handle, canonical_request.c_str()),
        "compile_batch warmup");
    const Json canonical_row = successful_row(warm_compile, "compile_batch warmup");
    const std::string expected_hash = canonical_row.at("plan_hash").get<std::string>();
    release_row_plan(canonical_row, "compile_batch warmup");
    const Json initial_cache = cache_stats(warm_compile);
    const std::size_t canonical_bytes =
        initial_cache.at("accounted_bytes").get<std::size_t>();
    const bool canonical_memo_accounted =
        initial_cache.at("entries") == 1 && canonical_bytes > plan_only_bytes &&
        canonical_bytes <= cache_byte_limit;
    if (!canonical_memo_accounted)
      fail_record(failures, name,
                  "canonical raw memo was not counted within the 64 MiB cache bound");

    const auto warm_cost = take_result(
        die_coarse_estimate_cost_batch_v2(model.handle,
                                          canonical_request.c_str()),
        "estimate_cost_batch warmup");
    const Json cost_row = successful_row(warm_cost, "estimate_cost_batch warmup");
    const auto expected_cost =
        cost_row.at("estimated_latency_ns").get<std::uint64_t>();
    const auto expected_status = cost_row.at("status").get<std::string>();
    require(cost_row.at("plan_hash") == expected_hash,
            name + " cost warmup returned a different Plan hash");

    const auto warm_alias = take_result(
        die_coarse_compile_batch_v2(model.handle, alias_request.c_str()),
        "semantic alias warmup");
    const Json alias_row = successful_row(warm_alias, "semantic alias warmup");
    const bool alias_same_hash = alias_row.at("plan_hash") == expected_hash;
    release_row_plan(alias_row, "semantic alias warmup");
    const Json alias_cache = cache_stats(warm_alias);
    const std::size_t alias_bytes =
        alias_cache.at("accounted_bytes").get<std::size_t>();
    const bool alias_one_entry =
        initial_cache.at("entries") == 1 &&
        alias_cache.at("entries") == initial_cache.at("entries") &&
        alias_bytes >= plan_only_bytes && alias_bytes <= cache_byte_limit;
    if (!alias_same_hash || !alias_one_entry)
      fail_record(failures, name,
                  "semantic alias did not reuse one bounded canonical Plan entry");

    // One full estimate preflight is excluded; timed calls below still invoke
    // estimate_alone and construct its complete diagnostic report.
    const auto warm_full = take_result(
        die_coarse_estimate_batch_v2(model.handle, canonical_request.c_str()),
        "estimate_batch warmup");
    const Json full_row = successful_row(warm_full, "estimate_batch warmup");
    require(full_row.at("plan_fingerprint") == expected_hash &&
                full_row.at("estimated_latency_ns").get<std::uint64_t>() ==
                    expected_cost &&
                full_row.at("status") == expected_status,
            name + " full estimate differs from the cached selected cost");
    const Json steady_cache = cache_stats(warm_full);

    const auto hot_compile = measure_capi(
        count, kHotQueryGateNs, "fixed_20us_hot_cache_query_gate",
        [&] { return die_coarse_compile_batch_v2(model.handle,
                                                 canonical_request.c_str()); },
        [&](const Json& response) {
          const Json& row = successful_row(response, "hot compile_batch");
          const bool same_hash = row.at("plan_hash") == expected_hash;
          release_row_plan(row, "hot compile_batch");
          require(same_hash, name + " hot compile changed Plan hash");
        });
    result["metrics"]["cabi_hot_compile_batch"] = hot_compile;

    const auto hot_cost = measure_capi(
        count, kHotQueryGateNs, "fixed_20us_hot_cache_query_gate",
        [&] { return die_coarse_estimate_cost_batch_v2(
                   model.handle, canonical_request.c_str()); },
        [&](const Json& response) {
          const Json& row = successful_row(response, "hot cost query");
          require(!row.contains("plan_handle") &&
                      row.at("plan_hash") == expected_hash &&
                      row.at("estimated_latency_ns").get<std::uint64_t>() ==
                          expected_cost &&
                      row.at("status") == expected_status,
                  name + " hot cost changed selected Plan identity or cost");
        });
    result["metrics"]["cabi_hot_cost_only"] = hot_cost;

    const auto hot_alias = measure_capi(
        count, kHotQueryGateNs, "fixed_20us_hot_cache_query_gate",
        [&] { return die_coarse_compile_batch_v2(model.handle,
                                                 alias_request.c_str()); },
        [&](const Json& response) {
          const Json& row = successful_row(response, "hot semantic alias");
          const bool same_hash = row.at("plan_hash") == expected_hash;
          release_row_plan(row, "hot semantic alias");
          const Json stats = cache_stats(response);
          require(same_hash && stats.at("entries") == alias_cache.at("entries") &&
                      stats.at("accounted_bytes") ==
                          alias_cache.at("accounted_bytes") &&
                      stats.at("accounted_bytes").get<std::size_t>() <=
                          cache_byte_limit,
                  name + " exact semantic alias memo changed its bounded footprint");
        });
    result["metrics"]["cabi_hot_semantic_alias_compile_batch"] = hot_alias;

    // Switching between already validated representations must reuse the same
    // canonical Plan, including when the raw-batch shortcut has been displaced.
    std::size_t alternating_index=0;
    result["metrics"]["cabi_hot_alternating_alias_compile_batch"] = measure_capi(
        count,kHotQueryGateNs,"fixed_20us_hot_cache_query_gate",
        [&] { return die_coarse_compile_batch_v2(model.handle,
            (++alternating_index%2 ? canonical_request : alias_request).c_str()); },
        [&](const Json& response) {
          const auto& row=successful_row(response,"alternating semantic alias");
          release_row_plan(row,"alternating semantic alias");
          require(row.at("plan_hash")==expected_hash &&
              response.at("cache_entries")==1 &&
              response.at("cache_accounted_bytes").get<std::size_t>()<=cache_byte_limit,
              name+" alternating aliases changed Plan identity or exceeded the cache bound");
        });

    const auto full_estimate = measure_capi(
        count, kCompileGateNs, "fixed_2ms_full_estimate_comparison_gate",
        [&] { return die_coarse_estimate_batch_v2(
                   model.handle, canonical_request.c_str()); },
        [&](const Json& response) {
          const Json& row = successful_row(response, "full estimate_batch");
          require(row.at("plan_fingerprint") == expected_hash &&
                      row.at("estimated_latency_ns").get<std::uint64_t>() ==
                          expected_cost &&
                      row.at("status") == expected_status,
                  name + " full estimate changed selected Plan identity or cost");
        });
    result["metrics"]["cabi_hot_full_estimate_report"] = full_estimate;
    auto& full_metric=result["metrics"]["cabi_hot_full_estimate_report"];
    full_metric["acceptance_applies"]=false;
    full_metric["threshold_basis"]="diagnostic_full_report_no_stated_gate";
    full_metric.erase("fixed_threshold_ns");full_metric.erase("threshold_checks");full_metric.erase("passed");

    const Json final_cache = cache_stats(
        take_result(die_coarse_estimate_cost_batch_v2(
                        model.handle, canonical_request.c_str()),
                    "final cache diagnostics"));
    const bool final_cache_stable =
        final_cache.at("entries") == steady_cache.at("entries") &&
        final_cache.at("accounted_bytes") == steady_cache.at("accounted_bytes") &&
        final_cache.at("accounted_bytes").get<std::size_t>() > plan_only_bytes &&
        final_cache.at("accounted_bytes").get<std::size_t>() <= cache_byte_limit;
    if (!final_cache_stable)
      fail_record(failures, name,
                  "hot query samples changed the canonical cache entry count or bytes");
    result["plan_hash"] = expected_hash;
    result["selected_cost_ns"] = expected_cost;
    result["cache"] = {{"plan_only_baseline", plan_only_cache},
                        {"after_canonical_memo", initial_cache},
                        {"after_semantic_alias_memo", alias_cache},
                        {"after_all_request_warmups", steady_cache},
                        {"after_all_samples", final_cache},
                        {"alias_same_plan_hash", alias_same_hash},
                        {"canonical_memo_accounted_and_bounded", canonical_memo_accounted},
                        {"alias_reused_single_entry_and_bounded", alias_one_entry}};
    bool metric_gates_passed = true;
    for (const char* metric : {"cpp_cold_compile_uncached",
                               "cabi_hot_compile_batch",
                               "cabi_hot_cost_only",
                               "cabi_hot_semantic_alias_compile_batch",
                               "cabi_hot_alternating_alias_compile_batch"}) {
      if (!result.at("metrics").at(metric).value("acceptance_applies",true))continue;
      if (!result.at("metrics").at(metric).at("passed").get<bool>()) {
        metric_gates_passed = false;
        fail_record(failures, name + "." + metric,
                    "fixed percentile latency gate was exceeded");
      }
    }
    if (!metric_gates_passed || !cpp_hash_stable || !alias_same_hash ||
        !canonical_memo_accounted || !alias_one_entry || !final_cache_stable)
      result["status"] = "failed";
    result["coverage"] = {
        {"cpp_samples_are_uncached_compile_calls", cpp_hash_stable},
        {"compile_batch_handle_released_outside_timing", true},
        {"cost_api_returns_no_plan_handle", true},
        {"semantic_alias_same_plan_hash", alias_same_hash},
        {"canonical_memo_counted_and_bounded", canonical_memo_accounted},
        {"semantic_alias_plan_shared_and_bytes_bounded", alias_one_entry},
        {"alternating_semantic_aliases_measured", true},
        {"final_cache_entries_and_bytes_stable", final_cache_stable},
        {"full_estimate_cost_matches_cost_only", true},
        {"samples_per_metric", count}};
  } catch (const std::exception& error) {
    result["status"] = "failed";
    result["failure"] = error.what();
    fail_record(failures, name, error.what());
  }
  return result;
}

std::shared_ptr<const Plan> one_activity_plan(const Hardware& hardware,
                                               const Activity& activity_def,
                                               const std::string& name) {
  Epoch epoch;
  epoch.name = name;
  epoch.activities.push_back(activity_def);
  return manual_plan(hardware, {std::move(epoch)}, name);
}

Activity homogeneous_activity(const Hardware& hardware) {
  Activity result;
  result.id = 1;
  result.role = Role::Compute;
  result.work[HBM0] = 1000000.0;
  result.hbm_bytes[0][0] = 1000000;
  result.canonical_q = canonical_basis(result);
  result.private_time_s = 0.01;
  (void)hardware;
  return result;
}

Activity heterogeneous_activity(const Hardware& hardware, std::size_t variant) {
  static constexpr std::array<std::array<std::size_t, 4>, 4> kResources{{
      {{HBM0, COMPUTE_GATE, SRAM_READ, LOCAL}},
      {{HBM1, COMPUTE_GATE, SRAM_WRITE, HOP}},
      {{HBM0, HBM1, INJECT0, XPOS}},
      {{COMPUTE_GATE, SRAM_READ, SRAM_WRITE, YNEG}}}};
  Activity result;
  result.id = 1;
  result.role = Role::Compute;
  const auto& resources = kResources[variant % kResources.size()];
  for (std::size_t i = 0; i < resources.size(); ++i) {
    const auto resource = resources[i];
    if (hardware.rates[resource] <= 0.0) continue;
    result.work[resource] = 500000.0 + 10000.0 *
        static_cast<double>(variant + i + 1);
    if (resource == HBM0) result.hbm_bytes[0][0] = 500000;
    if (resource == HBM1) result.hbm_bytes[1][0] = 500000;
  }
  if (std::all_of(result.work.begin(), result.work.end(),
                  [](double work) { return work == 0.0; })) {
    throw std::runtime_error("heterogeneous fixture has no enabled resources");
  }
  result.canonical_q = canonical_basis(result);
  result.private_time_s = 0.002 + 0.001 * static_cast<double>(variant % 4);
  return result;
}

Activity multi_domain_activity(const Hardware& hardware, std::size_t variant) {
  Activity result;
  result.id = 1;
  result.role = Role::Compute;
  std::size_t used = 0;
  for (std::size_t resource = 0; resource < resource_count; ++resource) {
    if (hardware.rates[resource] <= 0.0) continue;
    result.work[resource] = 250000.0 +
        1000.0 * static_cast<double>((resource + variant) % 7);
    if (resource == HBM0) result.hbm_bytes[0][0] = 250000;
    if (resource == HBM1) result.hbm_bytes[1][0] = 250000;
    ++used;
  }
  if (used < 2)
    throw std::runtime_error("multi-domain fixture requires two enabled domains");
  result.canonical_q = canonical_basis(result);
  result.private_time_s = 0.005;
  return result;
}

Json endpoint_spec(std::uint64_t id) {
  return {{"endpoint_id", id},
          {"transaction_id", id},
          {"owner_kind", "comm"},
          {"direction", "read"},
          {"physical_bytes", 1000000000ULL},
          {"path_class", "hbm_to_edge"},
          {"eta_hbm", 1.0},
          {"row_fraction", {1.0, 0.0}},
          {"route", {{"internal_route", false}}}};
}

OsBulkPlan select_os_bulk_plan(const Hardware& hardware,
                               const Json& profile_task) {
  const Json normalized = detail::normalize_task(hardware, profile_task);
  const auto candidates = detail::enumerate_mappings(hardware, normalized);
  for (const auto& candidate : candidates) {
    if (!candidate ||
        candidate->summary.value("dataflow_id", std::string()) !=
            "output_stationary_v2" ||
        !candidate->summary.contains("service_curves"))
      continue;
    for (const auto& curve : candidate->summary.at("service_curves")) {
      const auto begin = curve.at("epoch_begin").get<std::size_t>();
      const auto end = begin+curve.at("epoch_count").get<std::size_t>();
      std::size_t bulk=end;
      for(auto epoch=begin;epoch<end && epoch<candidate->epochs.size();++epoch) {
        const auto& activities=candidate->epochs[epoch].activities;
        const bool has_load=std::any_of(activities.begin(),activities.end(),
            [](const auto& a){return a.role==Role::Load;});
        const bool has_compute=std::any_of(activities.begin(),activities.end(),
            [](const auto& a){return a.role==Role::Compute;});
        const bool has_store=std::any_of(activities.begin(),activities.end(),
            [](const auto& a){return a.role==Role::Store;});
        if(has_load&&has_compute&&has_store){bulk=epoch;break;}
      }
      if(bulk==end)continue;
      std::size_t credit_activities = 0;
      for (const auto& activity_def : candidate->epochs[bulk].activities)
        if (!activity_def.credit_curve.empty()) ++credit_activities;
      if (credit_activities == 0) continue;
      return {candidate, bulk, candidate->epochs[bulk].activities.size(),
              credit_activities,
              candidate->summary.value("dataflow_id", std::string()),
              candidate->summary.value("template_id", std::string())};
    }
  }
  throw std::runtime_error(
      "enumerate_mappings produced no output-stationary compiled candidate "
      "with an analytical bulk credit curve");
}

void advance_to_os_bulk(Runtime& runtime,
                        const std::vector<std::uint64_t>& handles,
                        std::size_t bulk_epoch) {
  std::size_t events = 0;
  while (runtime.query_progress(handles.front()).at("epoch_index")
             .get<std::size_t>() < bulk_epoch) {
    const auto next = runtime.next_event();
    require(next.has_value(),
            "OS first load must have a finite completion event");
    runtime.advance_to(*next);
    require(++events <= 16, "compiled OS first load stays event-bounded");
  }
  for (const auto handle : handles)
    require(runtime.query_progress(handle).at("epoch_index") == bulk_epoch,
            "all OS candidates enter the same cumulative-service bulk epoch");
}

enum class UpdateFixture { Homogeneous, Heterogeneous, Endpoint, MultiDomain, OsBulk };

std::string update_fixture_name(UpdateFixture fixture) {
  switch (fixture) {
    case UpdateFixture::Homogeneous: return "homogeneous";
    case UpdateFixture::Heterogeneous: return "heterogeneous";
    case UpdateFixture::Endpoint: return "endpoint_hbm";
    case UpdateFixture::MultiDomain: return "multi_domain";
    case UpdateFixture::OsBulk: return "compiled_os_credit_curve_bulk";
  }
  return "unknown";
}

const char* arbitration_path_name(ArbitrationDiagnostics::Path path) {
  switch (path) {
    case ArbitrationDiagnostics::Path::Empty: return "empty";
    case ArbitrationDiagnostics::Path::Singleton: return "singleton";
    case ArbitrationDiagnostics::Path::Packing: return "packing";
    case ArbitrationDiagnostics::Path::SignedLP: return "signed_lp";
    case ArbitrationDiagnostics::Path::Ratio: return "ratio";
  }
  return "unknown";
}

Json measure_capacity_update(const Hardware& hardware,
                             UpdateFixture fixture,
                             std::size_t owners,
                             std::size_t samples,
                             const std::shared_ptr<const Plan>& homogeneous,
                             const std::vector<std::shared_ptr<const Plan>>& heterogeneous,
                             const std::vector<std::shared_ptr<const Plan>>& multi_domain,
                             const OsBulkPlan& os_bulk,
                             Json& failures) {
  const std::string name = update_fixture_name(fixture) + "_j" +
                           std::to_string(owners);
  Json result{{"case", name}, {"status", "ok"}};
  std::vector<std::uint64_t> timings;
  std::vector<std::uint64_t> allocation_deltas;
  std::vector<std::uint64_t> active_before;
  std::vector<std::uint64_t> active_after;
  std::vector<std::uint64_t> internal_event_deltas;
  std::vector<ArbitrationDiagnostics> arbitration_samples;
  std::vector<bool> same_clock;
  timings.reserve(samples);
  allocation_deltas.reserve(samples);
  active_before.reserve(samples);
  active_after.reserve(samples);
  internal_event_deltas.reserve(samples);
  arbitration_samples.reserve(samples);
  same_clock.reserve(samples);
  const auto submit_plans = [&](Runtime& runtime,
                                const std::vector<std::shared_ptr<const Plan>>& plans) {
    std::vector<std::uint64_t> handles;
    handles.reserve(plans.size());
    for (std::size_t i = 0; i < plans.size(); ++i)
      handles.push_back(runtime.submit(plans[i], owner(i + 1, 1, i + 1), 0));
    return handles;
  };

  for (std::size_t sample = 0; sample < samples; ++sample) {
    try {
      Runtime runtime(hardware);
      std::vector<std::uint64_t> handles;
      if (fixture == UpdateFixture::Homogeneous) {
        handles = submit_plans(runtime,
            std::vector<std::shared_ptr<const Plan>>(owners, homogeneous));
      } else if (fixture == UpdateFixture::Heterogeneous) {
        std::vector<std::shared_ptr<const Plan>> plans;
        plans.reserve(owners);
        for (std::size_t i = 0; i < owners; ++i)
          plans.push_back(heterogeneous[i % heterogeneous.size()]);
        handles = submit_plans(runtime, plans);
      } else if (fixture == UpdateFixture::MultiDomain) {
        std::vector<std::shared_ptr<const Plan>> plans;
        plans.reserve(owners);
        for (std::size_t i = 0; i < owners; ++i)
          plans.push_back(multi_domain[i % multi_domain.size()]);
        handles = submit_plans(runtime, plans);
      } else if (fixture == UpdateFixture::Endpoint) {
        handles.reserve(owners);
        for (std::size_t i = 0; i < owners; ++i) {
          const auto id = static_cast<std::uint64_t>(i + 1);
          handles.push_back(runtime.submit_endpoint(
              endpoint_spec(id), owner(i + 1, 1, i + 1), 0));
        }
      } else {
        handles = submit_plans(
            runtime, std::vector<std::shared_ptr<const Plan>>(owners,
                                                               os_bulk.plan));
        advance_to_os_bulk(runtime, handles, os_bulk.bulk_epoch);
      }

      const Json before = runtime.counters();
      const auto before_allocations =
          before.at("allocation_updates").get<std::uint64_t>();
      const auto before_internal_events =
          before.at("internal_events").get<std::uint64_t>();
      const auto before_active = before.at("active_activities").get<std::uint64_t>();
      const auto before_now = runtime.now_ns();
      const auto same_clock_now = runtime.now_ns();
      const double same_capacity = hardware.rates[HBM0];
      const auto begin = Clock::now();
      runtime.update_capacity(HBM0, same_capacity, same_clock_now);
      const auto end = Clock::now();
      arbitration_samples.push_back(runtime.last_arbitration());
      const auto after_now = runtime.now_ns();
      const Json after = runtime.counters();
      const auto after_allocations =
          after.at("allocation_updates").get<std::uint64_t>();
      const auto after_internal_events =
          after.at("internal_events").get<std::uint64_t>();
      const auto after_active = after.at("active_activities").get<std::uint64_t>();
      timings.push_back(elapsed_ns(begin, end));
      allocation_deltas.push_back(after_allocations - before_allocations);
      internal_event_deltas.push_back(after_internal_events - before_internal_events);
      active_before.push_back(before_active);
      active_after.push_back(after_active);
      same_clock.push_back(before_now == after_now);
    } catch (const std::exception& error) {
      result["status"] = "failed";
      result["failure"] = error.what();
      fail_record(failures, name, error.what());
      break;
    }
  }

  const auto metric = metric_report(
      timings, kCapacityUpdateGateNs, "fixed_50us_single_public_update_gate");
  const bool one_allocate_each = !allocation_deltas.empty() &&
      std::all_of(allocation_deltas.begin(), allocation_deltas.end(),
                  [](std::uint64_t delta) { return delta == 1; });
  const bool same_clock_each = !same_clock.empty() &&
      std::all_of(same_clock.begin(), same_clock.end(),
                  [](bool value) { return value; });
  const bool same_clock_noop_each = same_clock_each &&
      !internal_event_deltas.empty() &&
      std::all_of(internal_event_deltas.begin(), internal_event_deltas.end(),
                  [](std::uint64_t delta) { return delta == 1; });
  const bool all_expected_active = active_before.size() == timings.size() &&
      active_after.size() == timings.size() &&
      std::all_of(active_before.begin(), active_before.end(),
                  [&](std::uint64_t value) {
                    return fixture == UpdateFixture::OsBulk
                        ? value >= owners : value == owners;
                  }) &&
      std::all_of(active_after.begin(), active_after.end(),
                  [&](std::uint64_t value) {
                    return fixture == UpdateFixture::OsBulk
                        ? value >= owners : value == owners;
                  });
  const bool arbitration_complete = arbitration_samples.size() == samples &&
      std::all_of(arbitration_samples.begin(), arbitration_samples.end(),
                  [](const ArbitrationDiagnostics& diagnostics) {
                    return diagnostics.path != ArbitrationDiagnostics::Path::Empty &&
                           diagnostics.activities > 0;
                  });
  const bool coverage_ok = timings.size() == samples && one_allocate_each &&
                           same_clock_noop_each && all_expected_active &&
                           arbitration_complete;
  Json arbitration_path_counts{{"empty", 0}, {"singleton", 0},
                               {"packing", 0}, {"signed_lp", 0}, {"ratio", 0}};
  Json arbitration_paths = Json::array();
  Json arbitration_activities = Json::array();
  Json arbitration_rows = Json::array();
  Json arbitration_pivots = Json::array();
  Json arbitration_tangent_rounds = Json::array();
  for (const auto& diagnostics : arbitration_samples) {
    const char* path = arbitration_path_name(diagnostics.path);
    arbitration_path_counts[path] =
        arbitration_path_counts[path].get<std::size_t>() + 1;
    arbitration_paths.push_back(path);
    arbitration_activities.push_back(diagnostics.activities);
    arbitration_rows.push_back(diagnostics.constraint_rows);
    arbitration_pivots.push_back(diagnostics.pivots);
    arbitration_tangent_rounds.push_back(diagnostics.tangent_rounds);
  }
  result["actual_arbitration"] = {
      {"path_counts", arbitration_path_counts},
      {"path_per_sample", arbitration_paths},
      {"activities_per_sample", arbitration_activities},
      {"constraint_rows_per_sample", arbitration_rows},
      {"pivots_per_sample", arbitration_pivots},
      {"tangent_rounds_per_sample", arbitration_tangent_rounds},
      {"read_from_runtime_last_arbitration_outside_timed_scope", true}};
  result["metric"] = metric;
  result["single_update_upper_bound"] = {
      {"public_call", "Runtime::update_capacity(HBM0, hardware HBM0 rate, current now_ns)"},
      {"timed_scope", "one update_capacity call only"},
      {"fixed_budget_ns", kCapacityUpdateGateNs},
      {"public_single_call_upper_bound_ns", kCapacityUpdateGateNs},
      {"p50_ns", metric.at("p50_ns")},
      {"p95_ns", metric.at("p95_ns")},
      {"p99_ns", metric.at("p99_ns")},
      {"observed_max_single_call_ns", metric.at("max_ns")},
      {"allocation_updates_delta_per_sample", allocation_deltas},
      {"internal_events_delta_per_sample", internal_event_deltas},
      {"same_clock_advance_preserved_now_ns_per_sample", same_clock},
      {"aggregate_time_divided_by_update_count", false}};
  result["coverage"] = {
      {"owners", owners},
      {"expected_active_activities", fixture == UpdateFixture::OsBulk
                                          ? Json(nullptr) : Json(owners)},
      {"compiled_os_bulk_epoch", fixture == UpdateFixture::OsBulk
                                     ? Json(os_bulk.bulk_epoch) : Json(nullptr)},
      {"compiled_os_plan_hash", fixture == UpdateFixture::OsBulk
                                    ? Json(os_bulk.plan->fingerprint) : Json(nullptr)},
      {"compiled_os_dataflow_id", fixture == UpdateFixture::OsBulk
                                      ? Json(os_bulk.dataflow_id) : Json(nullptr)},
      {"compiled_os_template_id", fixture == UpdateFixture::OsBulk
                                      ? Json(os_bulk.template_id) : Json(nullptr)},
      {"compiled_os_bulk_planned_activities_per_execution",
       fixture == UpdateFixture::OsBulk ? Json(os_bulk.bulk_activity_count)
                                        : Json(nullptr)},
      {"heterogeneous_pattern_count",
       fixture == UpdateFixture::Heterogeneous ? Json(heterogeneous.size())
                                               : Json(nullptr)},
      {"multi_domain_enabled_domain_count",
       fixture == UpdateFixture::MultiDomain
           ? Json(std::count_if(hardware.rates.begin(), hardware.rates.end(),
                                [](double rate) { return rate > 0.0; }))
           : Json(nullptr)},
      {"compiled_os_credit_curve_activities_per_execution",
       fixture == UpdateFixture::OsBulk
           ? Json(os_bulk.credit_curve_activity_count) : Json(nullptr)},
      {"active_activities_before_each_sample", active_before},
      {"active_activities_after_each_sample", active_after},
      {"all_samples_observed_exactly_one_allocate", one_allocate_each},
      {"same_clock_advance_was_noop", same_clock_noop_each},
      {"actual_arbitration_diagnostics_complete", arbitration_complete},
      {"sample_coverage_complete", timings.size() == samples},
      {"coverage_passed", coverage_ok}};
  if (!metric.at("passed").get<bool>()) {
    fail_record(failures, name, "fixed 50us single-update percentile gate was exceeded");
    result["status"] = "failed";
  }
  if (!coverage_ok)
    fail_record(failures, name,
                "coverage/counter/arbitration check failed; every sample must update once at the same clock with required active work and a non-empty measured arbitration path");
  if (!coverage_ok) result["status"] = "failed";
  return result;
}

}  // namespace

int main(int argc, char** argv) {
  Args args;
  Json report = Json::object();
  Json failures = Json::array();
  try {
    args = parse_args(argc, argv);
    const Json config = read_json_file(args.config_path);
    const Json tasks = read_json_file(args.task_path);
    const Hardware hardware = resolve_hardware(config);
    const std::string profile = config.value("performance_profile", Json::object())
                                    .value("execution_profile",
                                           std::string("int4_matrix_vec_fp32_assumed_v1"));
    const Json attention = select_example_attention(tasks);
    report = {{"benchmark_schema", "coarse_die_service_curve_benchmark_v1"},
              {"status", "running"},
              {"samples_per_metric", args.samples},
              {"config_path", args.config_path},
              {"task_source_path", args.task_path},
              {"config_hash", stable_hash(config.dump())},
              {"example_attention_task_hash", stable_hash(attention.dump())},
              {"hardware_fingerprint", hardware.fingerprint},
              {"fixed_thresholds_ns", {{"single_capacity_update", kCapacityUpdateGateNs},
                                       {"hot_cabi_query", kHotQueryGateNs},
                                       {"regular_uncached_compile", kCompileGateNs}}}};
    const std::vector<std::pair<std::string, Json>> query_fixtures = {
        {"gemm_1x128x128", gemm_task(1, 128, 128, profile)},
        {"gemm_128x128x128", gemm_task(128, 128, 128, profile)},
        {"residual_257", {{"op_kind","residual"},{"tensor_shapes",{{"m",1},{"hidden",257}}},
            {"storage_dtype","int4"},{"execution_profile",profile}}},
        {"swiglu_16x4097", {{"op_kind","mlp_activation"},{"tensor_shapes",{{"m",16},{"ffn",4097}}},
            {"storage_dtype","bf16"},{"execution_profile",profile}}},
        {"norm_8x4096_cached", {{"op_kind","norm"},{"tensor_shapes",{{"m",8},{"hidden",4096}}},
            {"storage_dtype","bf16"},{"execution_profile",profile}}},
        {"norm_128x19569_streamed_parameters", {{"op_kind","norm"},{"tensor_shapes",{{"m",128},{"hidden",19569}}},
            {"storage_dtype","bf16"},{"execution_profile",profile}}},
        {"example_attention", attention}};

    Json query_results = Json::array();
    for (const auto& [name, task] : query_fixtures) {
      const Json alias = task.value("op_kind",std::string())=="gemm"
          ? gemm_semantic_alias(task, profile) : semantic_trace_alias(task);
      query_results.push_back(compile_query_fixture(
          hardware, config, task, alias, name, args.samples, failures));
      report["query_fixtures"] = query_results;
    }

    if (hardware.rates[HBM0] <= 0.0)
      throw std::runtime_error("benchmark requires positive HBM0 capacity");
    const auto homogeneous = one_activity_plan(
        hardware, homogeneous_activity(hardware), "benchmark_homogeneous_hbm0");
    std::vector<std::shared_ptr<const Plan>> heterogeneous;
    std::vector<std::shared_ptr<const Plan>> multi_domain;
    std::set<std::array<double, resource_count>> heterogeneous_signatures;
    for (std::size_t i = 0; i < 4; ++i) {
      const Activity heterogeneous_def = heterogeneous_activity(hardware, i);
      heterogeneous_signatures.insert(heterogeneous_def.work);
      heterogeneous.push_back(one_activity_plan(
          hardware, heterogeneous_def,
          "benchmark_heterogeneous_" + std::to_string(i)));
      multi_domain.push_back(one_activity_plan(
          hardware, multi_domain_activity(hardware, i),
          "benchmark_multi_domain_" + std::to_string(i)));
    }
    if (heterogeneous_signatures.size() < 2)
      fail_record(failures, "heterogeneous_setup",
                  "heterogeneous workload patterns collapsed to one resource vector");
    const OsBulkPlan os_bulk = select_os_bulk_plan(
        hardware, gemm_task(128, 128, 128, profile));
    report["compiled_os_candidate"] = {
        {"plan_hash", os_bulk.plan->fingerprint},
        {"dataflow_id", os_bulk.dataflow_id},
        {"template_id", os_bulk.template_id},
        {"bulk_epoch", os_bulk.bulk_epoch},
        {"bulk_activity_count", os_bulk.bulk_activity_count},
        {"credit_curve_activity_count", os_bulk.credit_curve_activity_count},
        {"service_curve_summary", os_bulk.plan->summary.at("service_curves")}};

    Json update_results = Json::array();
    for (const auto fixture : {UpdateFixture::Homogeneous,
                               UpdateFixture::Heterogeneous,
                               UpdateFixture::Endpoint,
                               UpdateFixture::MultiDomain,
                               UpdateFixture::OsBulk}) {
      for (const auto owners : kOwnerCounts) {
        update_results.push_back(measure_capacity_update(
            hardware, fixture, owners, args.samples, homogeneous,
            heterogeneous, multi_domain, os_bulk, failures));
        report["runtime_update_fixtures"] = update_results;
      }
    }

    const auto has_query_coverage = [&](const std::string& name) {
      for (const auto& item : query_results) {
        if (item.value("case", std::string()) != name) continue;
        return item.contains("coverage") &&
               item.at("coverage").at("samples_per_metric") == args.samples;
      }
      return false;
    };
    const bool small_gemm_coverage = has_query_coverage("gemm_1x128x128");
    const bool large_gemm_coverage = has_query_coverage("gemm_128x128x128");
    const bool attention_coverage = has_query_coverage("example_attention");
    const bool runtime_coverage_complete = update_results.size() == 15 &&
        std::all_of(update_results.begin(), update_results.end(), [](const Json& item) {
          return item.contains("coverage") &&
                 item.at("coverage").at("sample_coverage_complete") == true &&
                 item.at("coverage").at("coverage_passed") == true;
        });
    std::size_t os_group_count = 0;
    bool os_curve_covered = true;
    for (const auto& item : update_results) {
      if (item.value("case", std::string()).find(
              "compiled_os_credit_curve_bulk_") != 0)
        continue;
      ++os_group_count;
      os_curve_covered = os_curve_covered && item.contains("coverage") &&
          item.at("coverage").at("sample_coverage_complete").get<bool>() &&
          item.at("coverage")
                  .at("compiled_os_credit_curve_activities_per_execution")
                  .get<std::size_t>() > 0;
    }
    os_curve_covered = os_curve_covered && os_group_count == kOwnerCounts.size();
    const bool passed = failures.empty();
    report = {{"benchmark_schema", "coarse_die_service_curve_benchmark_v1"},
              {"status", passed ? "ok" : "failed"},
              {"samples_per_metric", args.samples},
              {"config_path", args.config_path},
              {"task_source_path", args.task_path},
              {"config_hash", stable_hash(config.dump())},
              {"example_attention_task_hash", stable_hash(attention.dump())},
              {"hardware_fingerprint", hardware.fingerprint},
              {"compiled_os_candidate", {
                  {"plan_hash", os_bulk.plan->fingerprint},
                  {"dataflow_id", os_bulk.dataflow_id},
                  {"template_id", os_bulk.template_id},
                  {"bulk_epoch", os_bulk.bulk_epoch},
                  {"bulk_activity_count", os_bulk.bulk_activity_count},
                  {"credit_curve_activity_count",
                   os_bulk.credit_curve_activity_count},
                  {"service_curve_summary",
                   os_bulk.plan->summary.at("service_curves")}}},
              {"fixed_thresholds_ns", {{"single_capacity_update", kCapacityUpdateGateNs},
                                       {"hot_cabi_query", kHotQueryGateNs},
                                       {"regular_uncached_compile", kCompileGateNs}}},
              {"query_fixtures", std::move(query_results)},
              {"runtime_update_fixtures", std::move(update_results)},
              {"coverage", {{"required_gemm_1x128x128", small_gemm_coverage},
                            {"required_gemm_128x128x128", large_gemm_coverage},
                            {"required_real_example_attention_task", attention_coverage},
                            {"distinct_heterogeneous_activity_patterns",
                             heterogeneous_signatures.size()},
                            {"required_owner_counts", {1, 8, 32}},
                            {"required_runtime_groups",
                             {"homogeneous", "heterogeneous", "endpoint_hbm",
                              "multi_domain", "compiled_os_credit_curve_bulk"}},
                            {"full_estimate_report_measured_separately", true},
                            {"runtime_update_group_coverage_complete",
                             runtime_coverage_complete},
                            {"compiled_os_credit_curve_segment_covered",
                             os_curve_covered},
                            {"cpu_identity", "collected externally by orchestrator"}}},
              {"failures", std::move(failures)},
              {"notes", Json::array({
                   "C++ compile samples call compile() directly with resolved Hardware; there is no C++ cache.",
                   "C ABI cache calls use a warmed model; compile_batch handles are released after each timed call.",
                   "The cost-only endpoint reuses cached Plan.summary; full estimate_batch is reported separately and runs its standalone estimate path.",
                   "Each arbitration sample times exactly one same-clock update_capacity(HBM0, rate, now_ns) call; counters outside timing require allocation_updates to increase by exactly one.",
                   "The compiled OS fixture selects a real output_stationary_v2 candidate with an analytical cumulative credit curve, advances through the actual analytic first-output prefix, then samples the three-activity bulk epoch.",
                   "P50/P95/P99 use nearest-rank ceil(p*N); the fixed gates are never adjusted based on measurements."})}};
    const std::string output_text = report.dump(2) + "\n";
    std::ofstream output(args.output_path, std::ios::binary);
    if (!output) throw std::runtime_error("cannot open output: " + args.output_path);
    output << output_text;
    if (!output) throw std::runtime_error("failed writing output: " + args.output_path);
    std::cout << "service-curve benchmark status=" << report.at("status")
              << " output=" << args.output_path << "\n";
    return passed ? 0 : 1;
  } catch (const std::exception& error) {
    if (!args.output_path.empty()) {
      if (!report.is_object() || report.empty()) {
        report = {{"benchmark_schema", "coarse_die_service_curve_benchmark_v1"},
                  {"samples_per_metric", args.samples},
                  {"config_path", args.config_path},
                  {"task_source_path", args.task_path},
                  {"fixed_thresholds_ns", {{"single_capacity_update", kCapacityUpdateGateNs},
                                           {"hot_cabi_query", kHotQueryGateNs},
                                           {"regular_uncached_compile", kCompileGateNs}}},
                  {"coverage", {{"cpu_identity", "collected externally by orchestrator"}}}};
      }
      report["status"] = "failed";
      report["failure"] = error.what();
      if (!report.contains("failures")) report["failures"] = failures;
      report["failures"].push_back({{"case", "benchmark_setup_or_output"},
                                    {"reason", error.what()}});
      try {
        std::ofstream output(args.output_path, std::ios::binary);
        if (output) output << report.dump(2) << "\n";
      } catch (...) {
      }
    }
    std::cerr << "service_curve_benchmark failed: " << error.what() << "\n";
    return 2;
  }
}
