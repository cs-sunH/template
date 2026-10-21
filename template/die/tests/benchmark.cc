#include "coarse_die/Execution.hh"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <ctime>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <sys/resource.h>
#include <vector>

namespace {
using Clock = std::chrono::steady_clock;
using nlohmann::json;

std::string read_text(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("cannot open input: " + path);
  return std::string(std::istreambuf_iterator<char>(in),
                     std::istreambuf_iterator<char>());
}

json read_json(const std::string& path) { return json::parse(read_text(path)); }

double percentile(std::vector<double> values, double p) {
  if (values.empty()) return 0.0;
  std::sort(values.begin(), values.end());
  const auto i = static_cast<std::size_t>(
      std::ceil(p * static_cast<double>(values.size())) - 1.0);
  return values[std::min(i, values.size() - 1)];
}

json stats(const std::vector<double>& ns) {
  json out;
  out["samples_ns"] = ns;
  out["count"] = ns.size();
  out["p50_ns"] = percentile(ns, 0.50);
  out["p95_ns"] = percentile(ns, 0.95);
  out["min_ns"] = ns.empty() ? 0.0 : *std::min_element(ns.begin(), ns.end());
  out["max_ns"] = ns.empty() ? 0.0 : *std::max_element(ns.begin(), ns.end());
  return out;
}

double elapsed_ns(Clock::time_point begin, Clock::time_point end) {
  return static_cast<double>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count());
}

struct Args {
  std::string config = std::string(COARSE_DIE_SOURCE_DIR) +
                       "/tests/fixtures/coarse_die_v2.json";
  std::string task =
      std::string(COARSE_DIE_SOURCE_DIR) + "/tests/fixtures/gemm_medium.json";
  std::string output;
  std::size_t iterations = 30;
  std::size_t owners = 8;
};

Args parse_args(int argc, char** argv) {
  Args a;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    auto value = [&]() -> std::string {
      if (i + 1 >= argc) throw std::runtime_error("missing value for " + arg);
      return argv[++i];
    };
    if (arg == "--config") a.config = value();
    else if (arg == "--task") a.task = value();
    else if (arg == "--output") a.output = value();
    else if (arg == "--iterations") a.iterations = std::stoull(value());
    else if (arg == "--owners") a.owners = std::stoull(value());
    else if (arg == "--help") {
      std::cout << "usage: coarse_die_benchmark [--config JSON] [--task JSON] "
                   "[--iterations N] [--owners J] [--output JSON]\n";
      std::exit(0);
    } else {
      throw std::runtime_error("unknown argument: " + arg);
    }
  }
  if (a.iterations == 0 || a.owners == 0)
    throw std::runtime_error("iterations and owners must be positive");
  return a;
}
}  // namespace

int main(int argc, char** argv) {
  try {
    const Args args = parse_args(argc, argv);
    const std::string config_text = read_text(args.config);
    const std::string task_text = read_text(args.task);
    const json config = json::parse(config_text);
    const json task = json::parse(task_text);

    std::vector<double> cold_compile, hot_compile, estimate_alone_samples;
    cold_compile.reserve(args.iterations);
    hot_compile.reserve(args.iterations);
    estimate_alone_samples.reserve(args.iterations);

    std::shared_ptr<const coarse_die::Plan> last_plan;
    coarse_die::Hardware hardware;
    json standalone;

    for (std::size_t i = 0; i < args.iterations; ++i) {
      const auto begin = Clock::now();
      const auto cold_config = json::parse(read_text(args.config));
      const auto cold_task = json::parse(read_text(args.task));
      auto cold_hardware = coarse_die::resolve_hardware(cold_config);
      auto cold_plan = coarse_die::compile(cold_hardware, cold_task);
      const auto end = Clock::now();
      cold_compile.push_back(elapsed_ns(begin, end));
      if (i + 1 == args.iterations) last_plan = std::move(cold_plan);
    }

    hardware = coarse_die::resolve_hardware(config);
    for (std::size_t i = 0; i < args.iterations; ++i) {
      const auto begin = Clock::now();
      last_plan = coarse_die::compile(hardware, task);
      const auto end = Clock::now();
      hot_compile.push_back(elapsed_ns(begin, end));
    }

    for (std::size_t i = 0; i < args.iterations; ++i) {
      const auto begin = Clock::now();
      standalone = coarse_die::estimate_alone(hardware, *last_plan);
      const auto end = Clock::now();
      estimate_alone_samples.push_back(elapsed_ns(begin, end));
    }

    std::vector<double> arbitration_wall, arbitration_cpu;
    std::uint64_t last_advance_events = 0;
    std::uint64_t last_finish_ns = 0;
    json last_counters = json::object();
    for (std::size_t iter = 0; iter < args.iterations; ++iter) {
      coarse_die::Runtime runtime(hardware);
      std::vector<std::uint64_t> handles;
      handles.reserve(args.owners);
      const auto cpu_begin = std::clock();
      const auto begin = Clock::now();
      for (std::size_t j = 0; j < args.owners; ++j) {
        coarse_die::Owner owner;
        owner.request_id = j + 1;
        owner.node_id = 1;
        owner.execution_id = iter * args.owners + j + 1;
        owner.generation = 1;
        handles.push_back(runtime.submit(last_plan, owner, 0));
      }
      std::uint64_t advances = 0;
      while (!runtime.drained()) {
        const auto next = runtime.next_event();
        if (!next.has_value())
          throw std::runtime_error("runtime has active work without a next event");
        runtime.advance_to(*next);
        ++advances;
        if (advances > 1000000)
          throw std::runtime_error("benchmark hit one million runtime events");
      }
      const auto end = Clock::now();
      const auto cpu_end = std::clock();
      arbitration_wall.push_back(elapsed_ns(begin, end));
      arbitration_cpu.push_back(
          1e9 * static_cast<double>(cpu_end - cpu_begin) / CLOCKS_PER_SEC);
      last_advance_events = advances;
      last_finish_ns = runtime.now_ns();
      last_counters = runtime.counters();
      for (const auto handle : handles) runtime.destroy_execution(handle);
    }

    struct rusage usage {};
    const long max_rss_kib = getrusage(RUSAGE_SELF, &usage) == 0
                                 ? usage.ru_maxrss
                                 : -1;
    json report = {
        {"status", "ok"},
        {"benchmark_version", 1},
        {"measurement", "actual coarse_die C++ library calls"},
        {"compiler", __VERSION__},
        {"cplusplus", __cplusplus},
        {"config_path", args.config},
        {"task_path", args.task},
        {"config_hash", coarse_die::stable_hash(config.dump())},
        {"task_hash", coarse_die::stable_hash(task.dump())},
        {"hardware_fingerprint", hardware.fingerprint},
        {"plan_hash", last_plan->fingerprint},
        {"iterations", args.iterations},
        {"contending_owners", args.owners},
        {"cold_create_parse_resolve_compile", stats(cold_compile)},
        {"hot_compile_same_hardware_same_task", stats(hot_compile)},
        {"estimate_alone", stats(estimate_alone_samples)},
        {"shared_runtime_arbitration", stats(arbitration_wall)},
        {"shared_runtime_arbitration_cpu", stats(arbitration_cpu)},
        {"runtime_advance_event_count", last_advance_events},
        {"runtime_finish_ns", last_finish_ns},
        {"runtime_counters", last_counters},
        {"standalone_result", standalone},
        {"max_rss_kib", max_rss_kib},
        {"notes", json::array({
             "cold includes JSON file reads, parse, hardware resolution, and compile",
             "hot reuses parsed task and resolved hardware; compile still runs each sample",
             "runtime event count is actual next_event/advance_to loop count",
             "latency is modeled device time; benchmark durations are host time"})}};

    const std::string output = report.dump(2) + "\n";
    if (args.output.empty()) std::cout << output;
    else {
      std::ofstream out(args.output, std::ios::binary);
      if (!out) throw std::runtime_error("cannot write output: " + args.output);
      out << output;
    }
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "coarse_die_benchmark failed: " << e.what() << "\n";
    return 2;
  }
}
