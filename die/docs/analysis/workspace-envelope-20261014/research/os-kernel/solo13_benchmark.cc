#include "solo13_kernel.hh"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdlib>
#include <cmath>
#include <fstream>
#include <iostream>
#include <numeric>
#include <string>
#include <thread>
#include <vector>

#include <sys/resource.h>
#include <sys/utsname.h>

namespace {

using namespace solo13;
using Clock = std::chrono::steady_clock;
volatile u64 benchmark_sink = 0;

json read_facts(const std::string& path) {
  std::ifstream stream(path, std::ios::binary);
  if (!stream) throw std::runtime_error("cannot read facts: " + path);
  return json::parse(stream);
}

void retain_result(const Kernel& kernel, Kernel::Status status) {
  benchmark_sink = benchmark_sink + kernel.events() +
      static_cast<u64>(kernel.skipped_waves()) +
      static_cast<u64>(kernel.skipped_outputs()) +
      static_cast<u64>(status == Kernel::Status::Passed);
}

u64 elapsed_ns(Clock::time_point start, Clock::time_point end) {
  return static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(
      end - start).count());
}

json stats_json(std::vector<u64> samples) {
  std::sort(samples.begin(), samples.end());
  const long double sum = std::accumulate(
      samples.begin(), samples.end(), static_cast<long double>(0));
  const std::size_t p95_index = (95 * samples.size() + 99) / 100 - 1;
  return {{"sample_count", samples.size()},
          {"mean_ns", static_cast<double>(sum / samples.size())},
          {"p95_ns_nearest_rank", samples.at(p95_index)},
          {"max_ns", samples.back()},
          {"min_ns", samples.front()}};
}

json host_json() {
  struct utsname info {};
  if (uname(&info) != 0) throw std::runtime_error("uname failed");
  std::string cpu_model = "unknown";
  std::ifstream cpuinfo("/proc/cpuinfo");
  std::string line;
  while (std::getline(cpuinfo, line)) {
    const std::string key = "model name\t: ";
    const std::size_t found = line.find(key);
    if (found != std::string::npos) {
      cpu_model = line.substr(found + key.size());
      break;
    }
  }
  struct rusage usage {};
  if (getrusage(RUSAGE_SELF, &usage) != 0)
    throw std::runtime_error("getrusage failed");
  return {{"system", info.sysname}, {"release", info.release},
          {"machine", info.machine}, {"cpu_model", cpu_model},
          {"hardware_concurrency", std::thread::hardware_concurrency()},
          {"process_peak_rss_kib", usage.ru_maxrss}};
}

struct Stops {
  Rational first;
  Rational second;
};

Stops make_stops(const json& facts) {
  Kernel reference(facts);
  if (reference.run_until() != Kernel::Status::Passed)
    throw std::runtime_error("reference run did not pass");
  const Rational finish = reference.ledger().elapsed;
  const Rational third = r_fraction("1/3");
  return {r_mul(finish, third), r_mul(finish, r_fraction("2/3"))};
}

void run_stop_resume(Kernel& kernel, const Stops& stops) {
  const auto first = kernel.run_until(&stops.first);
  const auto second = kernel.resume_until(&stops.second);
  const auto final = kernel.resume_until();
  if (first != Kernel::Status::Paused || second != Kernel::Status::Paused ||
      final != Kernel::Status::Passed)
    throw std::runtime_error("stop/resume benchmark did not reach two pauses and completion");
  retain_result(kernel, final);
}

json benchmark(const std::string& facts_path, std::size_t samples,
               const std::string& fixture) {
  const json parsed = read_facts(facts_path);
  const Stops stops = make_stops(parsed);
  std::vector<u64> parse_construct, cold_full, run_only_full;
  std::vector<u64> cold_stop_resume, run_only_stop_resume;
  parse_construct.reserve(samples);
  cold_full.reserve(samples);
  run_only_full.reserve(samples);
  cold_stop_resume.reserve(samples);
  run_only_stop_resume.reserve(samples);

  for (std::size_t i = 0; i < samples; ++i) {
    {
      const auto start = Clock::now();
      const json report = read_facts(facts_path);
      Kernel kernel(report);
      const auto end = Clock::now();
      parse_construct.push_back(elapsed_ns(start, end));
      benchmark_sink = benchmark_sink + kernel.facts().class_count;
    }
    {
      const auto start = Clock::now();
      const json report = read_facts(facts_path);
      Kernel kernel(report);
      const auto status = kernel.run_until();
      const auto end = Clock::now();
      if (status != Kernel::Status::Passed)
        throw std::runtime_error("cold full benchmark failed");
      cold_full.push_back(elapsed_ns(start, end));
      retain_result(kernel, status);
    }
    {
      const auto start = Clock::now();
      const json report = read_facts(facts_path);
      Kernel kernel(report);
      run_stop_resume(kernel, stops);
      const auto end = Clock::now();
      cold_stop_resume.push_back(elapsed_ns(start, end));
    }
  }

  const json reusable_facts = read_facts(facts_path);
  Kernel reusable_full(reusable_facts);
  Kernel reusable_resume(reusable_facts);
  for (std::size_t i = 0; i < samples; ++i) {
    reusable_full.reset();
    const auto start = Clock::now();
    const auto status = reusable_full.run_until();
    const auto end = Clock::now();
    if (status != Kernel::Status::Passed)
      throw std::runtime_error("run-only full benchmark failed");
    run_only_full.push_back(elapsed_ns(start, end));
    retain_result(reusable_full, status);

    reusable_resume.reset();
    const auto resume_start = Clock::now();
    run_stop_resume(reusable_resume, stops);
    const auto resume_end = Clock::now();
    run_only_stop_resume.push_back(elapsed_ns(resume_start, resume_end));
  }

  json output = {
      {"schema", "solo13_inprocess_benchmark_v1"},
      {"fixture", fixture}, {"samples", samples},
      {"timing_scope", "steady_clock in-process; each cold sample includes file read, JSON parse, Facts/geometry construction, and lifecycle call; run-only resets outside timed region"},
      {"timings_ns", {
          {"parse_construct_cold", stats_json(std::move(parse_construct))},
          {"cold_full", stats_json(std::move(cold_full))},
          {"run_only_full", stats_json(std::move(run_only_full))},
          {"cold_stop_resume", stats_json(std::move(cold_stop_resume))},
          {"run_only_stop_resume", stats_json(std::move(run_only_stop_resume))}}},
      {"host", host_json()},
      {"proof_scope", "scratch-only single-owner/single-core exact continuous-time kernel; event cap is this harness discovery bound, not Runtime solver pivots or a production certificate"}};

#ifdef SOLO13_PROFILE_EXACT
  Kernel profiled(reusable_facts);
  reset_exact_profile_stats();
  run_stop_resume(profiled, stops);
  const auto& p = exact_profile_stats();
  output["profiled_stop_resume_exact_ops"] = {
      {"add", {{"calls", p.add_calls}, {"inclusive_ns", p.add_ns}}},
      {"subtract", {{"calls", p.sub_calls}, {"inclusive_ns", p.sub_ns}}},
      {"multiply", {{"calls", p.mul_calls}, {"inclusive_ns", p.mul_ns}}},
      {"divide", {{"calls", p.div_calls}, {"inclusive_ns", p.div_ns}}},
      {"compare", {{"calls", p.compare_calls}, {"inclusive_ns", p.compare_ns}}},
      {"floor_u128", {{"calls", p.floor_calls}, {"inclusive_ns", p.floor_ns}}},
      {"u128_to_rational", {{"calls", p.u128_conversion_calls},
                            {"inclusive_ns", p.u128_conversion_ns}}},
      {"multiply_repeat", {{"calls", p.multiply_repeat_calls},
                            {"inclusive_ns", p.multiply_repeat_ns}}},
      {"timing_note", "operation scopes are inclusive and may overlap (floor includes internal compare/conversion work); profile-enabled timings are not benchmark samples"}};
#else
  output["profiled_stop_resume_exact_ops"] = nullptr;
#endif
  output["sink"] = static_cast<u64>(benchmark_sink);
  return output;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    std::string facts_path, fixture = "unspecified";
    std::size_t samples = 30;
    for (int i = 1; i < argc; ++i) {
      const std::string option = argv[i];
      if (option == "--help" || option == "-h") {
        std::cout << "usage: solo13_benchmark --facts FILE [--fixture NAME] [--samples N]\n";
        return 0;
      }
      if (i + 1 >= argc) throw std::invalid_argument(option + " requires an argument");
      const std::string value = argv[++i];
      if (option == "--facts") facts_path = value;
      else if (option == "--fixture") fixture = value;
      else if (option == "--samples") samples = static_cast<std::size_t>(std::stoul(value));
      else throw std::invalid_argument("unknown option: " + option);
    }
    if (facts_path.empty()) throw std::invalid_argument("--facts is required");
    if (samples == 0) throw std::invalid_argument("--samples must be positive");
#ifndef SOLO13_PROFILE_EXACT
    if (std::getenv("SOLO13_PROFILE_EXACT") != nullptr)
      throw std::invalid_argument("profile env set but binary was not compiled with SOLO13_PROFILE_EXACT");
#endif
    std::cout << benchmark(facts_path, samples, fixture).dump(2) << '\n';
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "benchmark failed: " << error.what() << '\n';
    return 2;
  }
}
