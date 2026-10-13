#include "coarse_die/Capi.h"
#include <json/json.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <sys/resource.h>
#include <vector>

namespace {
using Clock = std::chrono::steady_clock;
using json = nlohmann::json;

std::string read_file(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("cannot read " + path);
  return std::string(std::istreambuf_iterator<char>(in),
                     std::istreambuf_iterator<char>());
}

double elapsed_ns(Clock::time_point begin, Clock::time_point end) {
  return static_cast<double>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count());
}

double percentile(std::vector<double> values, double p) {
  if (values.empty()) return 0.0;
  std::sort(values.begin(), values.end());
  const auto index = static_cast<std::size_t>(std::ceil(p * static_cast<double>(values.size())) - 1.0);
  return values[std::min(index, values.size() - 1)];
}

json stats(const std::vector<double>& values) {
  json out;
  out["count"] = values.size();
  out["samples_ns"] = values;
  out["min_ns"] = values.empty() ? 0.0 : *std::min_element(values.begin(), values.end());
  out["p50_ns"] = percentile(values, 0.50);
  out["p95_ns"] = percentile(values, 0.95);
  out["max_ns"] = values.empty() ? 0.0 : *std::max_element(values.begin(), values.end());
  return out;
}

struct TimedJson {
  double elapsed = 0.0;
  json value;
};

template <class Call>
TimedJson timed_result(Call&& call, const char* name) {
  const auto begin = Clock::now();
  const char* raw = call();
  const auto end = Clock::now();
  if (!raw)
    throw std::runtime_error(std::string(name) + " failed: " + die_coarse_last_error_v2());
  const std::string text(raw);
  die_coarse_free_result_v2(raw);
  return {elapsed_ns(begin, end), json::parse(text)};
}

struct Args {
  std::string config;
  std::string task;
  std::string output;
  std::size_t samples = 30;
};

Args parse_args(int argc, char** argv) {
  Args args;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    auto next = [&]() -> std::string {
      if (i + 1 >= argc) throw std::runtime_error("missing value for " + arg);
      return argv[++i];
    };
    if (arg == "--config") args.config = next();
    else if (arg == "--task") args.task = next();
    else if (arg == "--output") args.output = next();
    else if (arg == "--samples") args.samples = std::stoull(next());
    else if (arg == "--help") {
      std::cout << "usage: cabi_cache_probe --config JSON --task JSON "
                   "--samples N --output JSON\n";
      std::exit(0);
    } else throw std::runtime_error("unknown argument: " + arg);
  }
  if (args.config.empty() || args.task.empty() || args.output.empty() || args.samples == 0)
    throw std::runtime_error("config, task, output, and positive samples are required");
  return args;
}

json compile_response(TimedJson& call, DieHandle model, const std::string& request,
                      const char* label) {
  call = timed_result([&] { return die_coarse_compile_batch_v2(model, request.c_str()); },
                      label);
  const json& response = call.value;
  if (!response.contains("results") || response.at("results").size() != 1 ||
      response.at("results").at(0).is_null() ||
      !response.contains("errors") || response.at("errors").size() != 1 ||
      !response.at("errors").at(0).is_null())
    throw std::runtime_error(std::string(label) + " returned a compile error: " +
                             response.dump());
  return response.at("results").at(0);
}

struct PlanGuard {
  DieHandle handle = 0;
  ~PlanGuard() { if (handle != 0) (void)die_coarse_release_plan_v2(handle); }
  int release() {
    if (handle == 0) return DIE_OK;
    const int status = die_coarse_release_plan_v2(handle);
    if (status == DIE_OK) handle = 0;
    return status;
  }
};

struct ModelGuard {
  DieHandle handle = 0;
  ~ModelGuard() { if (handle != 0) (void)die_coarse_destroy_model_v2(handle); }
  int destroy() {
    if (handle == 0) return DIE_OK;
    const int status = die_coarse_destroy_model_v2(handle);
    if (status == DIE_OK) handle = 0;
    return status;
  }
};
}  // namespace

int main(int argc, char** argv) {
  try {
    const Args args = parse_args(argc, argv);
    const std::string config_text = read_file(args.config);
    const std::string task_text = read_file(args.task);
    const json task = json::parse(task_text);
    const std::string request = json{{"specs", json::array({task})}}.dump();

    std::vector<double> model_create_ns, cold_miss_ns, hot_hit_ns;
    std::vector<double> cold_release_ns, hot_release_ns, model_destroy_ns;
    model_create_ns.reserve(args.samples);
    cold_miss_ns.reserve(args.samples);
    hot_hit_ns.reserve(args.samples);

    json samples = json::array();
    std::string stable_hash;
    std::uint64_t stable_bytes = 0;
    std::uint64_t stable_entries = 0;
    for (std::size_t i = 0; i < args.samples; ++i) {
      ModelGuard model_guard;
      const auto create_begin = Clock::now();
      const int create_status = die_coarse_create_model_v2(config_text.c_str(), &model_guard.handle);
      const auto create_end = Clock::now();
      if (create_status != DIE_OK)
        throw std::runtime_error("model create failed: " +
                                 std::string(die_coarse_last_error_v2()));
      model_create_ns.push_back(elapsed_ns(create_begin, create_end));

      TimedJson cold_call;
      const json cold = compile_response(cold_call, model_guard.handle, request, "cold compile miss");
      cold_miss_ns.push_back(cold_call.elapsed);
      PlanGuard cold_plan_guard{cold.at("plan_handle").get<DieHandle>()};
      const auto cold_handle = cold_plan_guard.handle;
      const std::string hash = cold.at("plan_hash").get<std::string>();
      const std::uint64_t entries = cold_call.value.at("cache_entries").get<std::uint64_t>();
      const std::uint64_t bytes = cold_call.value.at("cache_accounted_bytes").get<std::uint64_t>();
      if (entries != 1 || bytes == 0)
        throw std::runtime_error("cold miss did not leave exactly one accounted cache entry");

      TimedJson hot_call;
      const json hot = compile_response(hot_call, model_guard.handle, request, "same-task cache hit");
      hot_hit_ns.push_back(hot_call.elapsed);
      PlanGuard hot_plan_guard{hot.at("plan_handle").get<DieHandle>()};
      const auto hot_handle = hot_plan_guard.handle;
      const std::string hot_hash = hot.at("plan_hash").get<std::string>();
      const std::uint64_t hot_entries = hot_call.value.at("cache_entries").get<std::uint64_t>();
      const std::uint64_t hot_bytes = hot_call.value.at("cache_accounted_bytes").get<std::uint64_t>();
      if (hash != hot_hash || entries != hot_entries || bytes != hot_bytes)
        throw std::runtime_error("same-task cache hit changed plan hash or cache accounting");
      if (i == 0) {
        stable_hash = hash;
        stable_bytes = bytes;
        stable_entries = entries;
      } else if (hash != stable_hash || bytes != stable_bytes || entries != stable_entries) {
        throw std::runtime_error("cache accounting or plan hash changed across fresh models");
      }

      const auto cold_release_begin = Clock::now();
      if (cold_plan_guard.release() != DIE_OK)
        throw std::runtime_error("cold plan release failed: " + std::string(die_coarse_last_error_v2()));
      const auto cold_release_end = Clock::now();
      cold_release_ns.push_back(elapsed_ns(cold_release_begin, cold_release_end));
      const auto hot_release_begin = Clock::now();
      if (hot_plan_guard.release() != DIE_OK)
        throw std::runtime_error("hot plan release failed: " + std::string(die_coarse_last_error_v2()));
      const auto hot_release_end = Clock::now();
      hot_release_ns.push_back(elapsed_ns(hot_release_begin, hot_release_end));

      const auto destroy_begin = Clock::now();
      const int destroy_status = model_guard.destroy();
      const auto destroy_end = Clock::now();
      if (destroy_status != DIE_OK)
        throw std::runtime_error("model destroy failed: " +
                                 std::string(die_coarse_last_error_v2()));
      model_destroy_ns.push_back(elapsed_ns(destroy_begin, destroy_end));

      samples.push_back({{"sample", i},
                         {"create_model_ns", model_create_ns.back()},
                         {"cold_cache_miss_compile_batch_ns", cold_call.elapsed},
                         {"same_task_cache_hit_compile_batch_ns", hot_call.elapsed},
                         {"cold_plan_handle", cold_handle},
                         {"hot_plan_handle", hot_handle},
                         {"plan_hash", hash},
                         {"cache_entries_after_miss", entries},
                         {"cache_bytes_after_miss", bytes},
                         {"cache_entries_after_hit", hot_entries},
                         {"cache_bytes_after_hit", hot_bytes},
                         {"cold_plan_release_ns", cold_release_ns.back()},
                         {"hot_plan_release_ns", hot_release_ns.back()},
                         {"destroy_model_ns", model_destroy_ns.back()}});
    }

    struct rusage usage {};
    const long max_rss_kib = getrusage(RUSAGE_SELF, &usage) == 0 ? usage.ru_maxrss : -1;
    json report = {
        {"status", "ok"},
        {"tool", "work/performance-current/cabi_cache_probe.cc"},
        {"compiler", __VERSION__},
        {"cplusplus", __cplusplus},
        {"config_path", args.config},
        {"task_path", args.task},
        {"samples", args.samples},
        {"cache_hit_validation", {
            {"fresh_model_per_sample", true},
            {"same_request_reused_within_model", true},
            {"plan_hash_stable", true},
            {"cache_entries_and_accounted_bytes_stable_on_second_call", true},
            {"returned_plan_handles_released", true},
            {"entries_after_miss", stable_entries},
            {"bytes_after_miss", stable_bytes},
            {"plan_hash", stable_hash},
            {"implementation_evidence", "Capi.cc cached(): hash lookup returns retained Plan and moves LRU entry before compile() fallback"}}},
        {"timing_scope", {
            {"create_model_ns", "separate: includes C ABI model creation, config JSON parsing, and hardware resolution"},
            {"cold_cache_miss_compile_batch_ns", "times only die_coarse_compile_batch_v2; includes C ABI request JSON parse, key creation, actual compile on fresh model cache, cache insert, response JSON dump/allocation; excludes caller request construction, result JSON parse/string copy/free, plan release"},
            {"same_task_cache_hit_compile_batch_ns", "times only die_coarse_compile_batch_v2 for byte-identical request on same model; includes request parse, cache lookup/LRU touch, response dump/allocation; implementation returns cached Plan before compile() fallback; excludes caller request construction, result parse/string copy/free, handle release"},
            {"release_destroy", "measured separately and excluded from compile timings"}}},
        {"plan_hash", stable_hash},
        {"cache_entries", stable_entries},
        {"cache_accounted_bytes", stable_bytes},
        {"create_model", stats(model_create_ns)},
        {"cold_cache_miss_compile_batch", stats(cold_miss_ns)},
        {"same_task_cache_hit_compile_batch", stats(hot_hit_ns)},
        {"cold_plan_release", stats(cold_release_ns)},
        {"hot_plan_release", stats(hot_release_ns)},
        {"destroy_model", stats(model_destroy_ns)},
        {"max_rss_kib", max_rss_kib},
        {"raw_samples", samples}};

    std::ofstream out(args.output, std::ios::binary);
    if (!out) throw std::runtime_error("cannot write output " + args.output);
    out << report.dump(2) << '\n';
    std::cout << "C API cache probe status=ok output=" << args.output << '\n';
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "C API cache probe failed: " << error.what() << '\n';
    return 1;
  }
}
