#include "/workspace/template/die/tests/test_support.hh"
#include "/workspace/template/die/src/CompilerInternal.hh"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <ctime>
#include <iostream>
#include <map>
#include <numeric>
#include <string>
#include <tuple>
#include <vector>

namespace coarse_die::detail {
std::vector<std::shared_ptr<Plan>> enumerate_mappings_for_compile(
    const Hardware&, const nlohmann::json&);
}

namespace {
using namespace coarse_die;
using namespace coarse_die_test;
using Json = nlohmann::json;
using Steady = std::chrono::steady_clock;

struct Stamp {
  std::uint64_t wall = 0;
  std::uint64_t cpu = 0;
};

std::uint64_t thread_cpu_ns() {
  timespec value{};
  if (clock_gettime(CLOCK_THREAD_CPUTIME_ID, &value) != 0)
    throw std::runtime_error("clock_gettime(CLOCK_THREAD_CPUTIME_ID) failed");
  return static_cast<std::uint64_t>(value.tv_sec) * 1'000'000'000ULL +
         static_cast<std::uint64_t>(value.tv_nsec);
}

Stamp stamp() {
  return {static_cast<std::uint64_t>(
              std::chrono::duration_cast<std::chrono::nanoseconds>(
                  Steady::now().time_since_epoch()).count()),
          thread_cpu_ns()};
}

struct Delta {
  std::uint64_t wall = 0;
  std::uint64_t cpu = 0;
};

Delta elapsed(const Stamp& begin, const Stamp& end) {
  return {end.wall - begin.wall, end.cpu - begin.cpu};
}

struct Samples {
  std::vector<std::uint64_t> wall;
  std::vector<std::uint64_t> cpu;

  void add(Delta value) {
    wall.push_back(value.wall);
    cpu.push_back(value.cpu);
  }
};

std::uint64_t percentile(std::vector<std::uint64_t> values, double p) {
  if (values.empty()) return 0;
  std::sort(values.begin(), values.end());
  const auto rank = static_cast<std::size_t>(std::ceil(p * values.size()));
  return values[std::max<std::size_t>(1, rank) - 1];
}

void print_metric(const char* name, const Samples& samples) {
  const auto mean = [](const std::vector<std::uint64_t>& values) {
    if (values.empty()) return std::uint64_t{0};
    return std::accumulate(values.begin(), values.end(), std::uint64_t{0}) /
           values.size();
  };
  const auto minimum = [](const std::vector<std::uint64_t>& values) {
    return values.empty() ? std::uint64_t{0}
                          : *std::min_element(values.begin(), values.end());
  };
  const auto maximum = [](const std::vector<std::uint64_t>& values) {
    return values.empty() ? std::uint64_t{0}
                          : *std::max_element(values.begin(), values.end());
  };
  std::cout << name << " wall_ns{mean=" << mean(samples.wall)
            << ",p50=" << percentile(samples.wall, .50)
            << ",p95=" << percentile(samples.wall, .95)
            << ",p99=" << percentile(samples.wall, .99)
            << ",min=" << minimum(samples.wall)
            << ",max=" << maximum(samples.wall) << "} cpu_ns{mean="
            << mean(samples.cpu) << ",p50=" << percentile(samples.cpu, .50)
            << ",p95=" << percentile(samples.cpu, .95)
            << ",p99=" << percentile(samples.cpu, .99)
            << ",min=" << minimum(samples.cpu)
            << ",max=" << maximum(samples.cpu) << "}\n";
}

std::uint64_t summary_u64(const Plan& plan, const char* key) {
  const auto it = plan.summary.find(key);
  return it == plan.summary.end() ? 0 : it->get<std::uint64_t>();
}

std::string summary_string(const Plan& plan, const char* key) {
  const auto it = plan.summary.find(key);
  return it != plan.summary.end() && it->is_string()
      ? it->get<std::string>() : std::string();
}

struct Ranked {
  std::shared_ptr<Plan> plan;
  std::uint64_t latency = 0;
  unsigned __int128 hbm = 0;
  std::uint64_t workspace = 0;
  std::uint64_t sram = 0;
  std::string template_id;
  std::uint64_t tm = 0, tn = 0, tk = 0, th = 0;
};

struct CandidateSamples {
  Samples timings;
};

Json gemm_task() {
  return {{"op_kind", "gemm"},
          {"tensor_shapes", {{"m", 128}, {"n", 128}, {"k", 128}}},
          {"storage_dtype", {{"weights", "bf16"},
                              {"activations", "bf16"},
                              {"output", "bf16"}}},
          {"execution_profile", "int4_matrix_vec_fp32_assumed_v1"},
          {"layer_instances", 1}, {"weight_passes", 1}};
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const unsigned repetitions = argc > 1 ? std::stoul(argv[1]) : 60;
    if (repetitions < 2 || repetitions > 1000)
      throw std::runtime_error("usage: profile-compile [repetitions 2..1000]");
    const Hardware hardware = example_hardware();
    const auto shared_hardware = std::make_shared<const Hardware>(hardware);
    const Json task = gemm_task();

    Samples normalize, enumerate, structure_validation, runtime_trials;
    Samples ranking_seal, actual_compile;
    std::map<std::string, CandidateSamples> by_candidate;
    std::size_t candidate_count = 0;
    for (unsigned iteration = 0; iteration < repetitions + 5; ++iteration) {
      Json normalized;
      auto begin = stamp();
      normalized = detail::normalize_task(hardware, task);
      auto end = stamp();
      if (iteration >= 5) normalize.add(elapsed(begin, end));

      std::vector<std::shared_ptr<Plan>> candidates;
      begin = stamp();
      candidates = detail::enumerate_mappings_for_compile(hardware, normalized);
      end = stamp();
      if (iteration >= 5) enumerate.add(elapsed(begin, end));
      candidate_count = candidates.size();

      std::vector<Ranked> ranked;
      ranked.reserve(candidates.size());
      for (auto& candidate : candidates) {
        begin = stamp();
        detail::validate_plan_structure(*candidate);
        end = stamp();
        if (iteration >= 5) structure_validation.add(elapsed(begin, end));

        begin = stamp();
        const std::uint64_t latency = detail::estimate_candidate_latency_ns(
            shared_hardware, candidate);
        end = stamp();
        if (iteration >= 5) {
          runtime_trials.add(elapsed(begin, end));
          const auto& summary = candidate->summary;
          const std::string label = summary.value("dataflow_id", std::string()) +
              " tm=" + std::to_string(summary.value("tm", std::uint64_t{0})) +
              " tn=" + std::to_string(summary.value("tn", std::uint64_t{0})) +
              " tk=" + std::to_string(summary.value("tk", std::uint64_t{0}));
          by_candidate[label].timings.add(elapsed(begin, end));
        }
        candidate->summary["estimated_latency_ns"] = latency;
        candidate->summary["alone_estimate_status"] = "ok_assumed";
        const auto hbm = static_cast<unsigned __int128>(
            summary_u64(*candidate, "hbm_read_bytes_total")) +
            summary_u64(*candidate, "hbm_write_bytes_total");
        ranked.push_back({candidate, latency, hbm,
                          summary_u64(*candidate, "hbm_workspace_peak_bytes"),
                          summary_u64(*candidate, "sram_peak_bytes_per_core"),
                          summary_string(*candidate, "template_id"),
                          summary_u64(*candidate, "tm"),
                          summary_u64(*candidate, "tn"),
                          summary_u64(*candidate, "tk"),
                          summary_u64(*candidate, "th")});
      }

      begin = stamp();
      const auto key = [](const Ranked& row) {
        return std::tie(row.latency, row.hbm, row.workspace, row.sram,
                        row.template_id, row.tm, row.tn, row.tk, row.th);
      };
      std::stable_sort(ranked.begin(), ranked.end(),
                       [&](const Ranked& lhs, const Ranked& rhs) {
                         return key(lhs) < key(rhs);
                       });
      auto selected = ranked.front().plan;
      selected->summary["candidate_count"] =
          static_cast<std::uint64_t>(ranked.size());
      selected->summary["selected_template"] =
          selected->summary.value("template_id", std::string());
      selected->summary["selection_rule"] =
          "ceil(estimate_alone_ns),hbm_payload_bytes,hbm_workspace_peak_bytes,sram_peak_bytes_per_core,template_id,tm,tn,tk,th";
      selected->summary["candidate_estimate_source"] =
          "estimate_alone(shared_runtime_engine)";
      selected->summary["activity_work_semantics"] =
          "activity work/bytes/ops are epoch total across repeat_count; repeat_count is audit metadata only; credit_delta is total signed credit change over progress 0..1";
      selected->summary["granularity_validation"] = "not_run";
      selected->summary["evidence_flags"] =
          {"engineering_assumption", "hardware_source_derived"};
      selected->fingerprint.clear();
      selected->fingerprint = plan_to_json(*selected).at("fingerprint").get<std::string>();
      end = stamp();
      if (iteration >= 5) ranking_seal.add(elapsed(begin, end));

      begin = stamp();
      const auto compiled = compile(hardware, task);
      end = stamp();
      if (iteration >= 5) actual_compile.add(elapsed(begin, end));
      if (compiled->fingerprint != selected->fingerprint)
        throw std::runtime_error("manual phase breakdown selected a different plan");

    }

    std::cout << "task=gemm_128x128x128 repetitions=" << repetitions
              << " candidates=" << candidate_count
              << " manual_trial_samples=" << runtime_trials.wall.size() << "\n";
    print_metric("normalize", normalize);
    print_metric("enumerate_candidates", enumerate);
    print_metric("one_candidate_plan_structure_validation", structure_validation);
    print_metric("one_candidate_estimate_candidate_latency", runtime_trials);
    print_metric("rank_sort_selected_summary_and_fingerprint", ranking_seal);
    print_metric("actual_compile_end_to_end", actual_compile);

    Samples estimate_total;
    estimate_total.wall.resize(runtime_trials.wall.size() / candidate_count);
    estimate_total.cpu.resize(runtime_trials.cpu.size() / candidate_count);
    for (std::size_t sample = 0; sample < estimate_total.wall.size(); ++sample) {
      for (std::size_t candidate = 0; candidate < candidate_count; ++candidate) {
        estimate_total.wall[sample] += runtime_trials.wall[sample * candidate_count + candidate];
        estimate_total.cpu[sample] += runtime_trials.cpu[sample * candidate_count + candidate];
      }
    }
    print_metric("all_candidate_runtime_trials_per_compile", estimate_total);
    std::vector<std::pair<std::uint64_t, std::string>> slowest;
    for (const auto& item : by_candidate) {
      const auto mean = std::accumulate(item.second.timings.wall.begin(),
                                        item.second.timings.wall.end(),
                                        std::uint64_t{0}) /
                        item.second.timings.wall.size();
      slowest.push_back({mean, item.first});
    }
    std::sort(slowest.begin(), slowest.end(), std::greater<>());
    std::cout << "slowest_candidate_estimate_means_ns (wall/cpu p50,p95):\n";
    for (std::size_t i = 0; i < std::min<std::size_t>(10, slowest.size()); ++i) {
      const auto& samples = by_candidate.at(slowest[i].second).timings;
      std::cout << "  " << slowest[i].second << " mean_wall=" << slowest[i].first
                << " wall_p50=" << percentile(samples.wall, .50)
                << " wall_p95=" << percentile(samples.wall, .95)
                << " cpu_p50=" << percentile(samples.cpu, .50)
                << " cpu_p95=" << percentile(samples.cpu, .95) << '\n';
    }

    // Isolate the fixed per-candidate Runtime constructor cost. This occurs
    // after compile timings so these diagnostic probes cannot warm or perturb
    // the end-to-end compile samples above.
    Samples runtime_constructor, empty_runtime_copy;
    Runtime empty_prototype(shared_hardware);
    const std::size_t constructor_samples = repetitions * candidate_count;
    for (std::size_t i = 0; i < constructor_samples; ++i) {
      auto begin = stamp();
      { Runtime trial(shared_hardware); }
      auto end = stamp();
      runtime_constructor.add(elapsed(begin, end));

      begin = stamp();
      { Runtime trial(empty_prototype); }
      end = stamp();
      empty_runtime_copy.add(elapsed(begin, end));
    }
    print_metric("one_fresh_runtime_constructor_including_hardware_validation",
                 runtime_constructor);
    print_metric("one_empty_runtime_prototype_copy", empty_runtime_copy);
    const auto savings_wall = static_cast<std::int64_t>(
        percentile(runtime_constructor.wall, .50)) -
        static_cast<std::int64_t>(percentile(empty_runtime_copy.wall, .50));
    const auto savings_cpu = static_cast<std::int64_t>(
        percentile(runtime_constructor.cpu, .50)) -
        static_cast<std::int64_t>(percentile(empty_runtime_copy.cpu, .50));
    std::cout << "estimated_28_candidate_constructor_only_savings_ns"
              << "{wall=" << savings_wall * candidate_count
              << ",cpu=" << savings_cpu * candidate_count << "}\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "profile-compile failed: " << error.what() << '\n';
    return 1;
  }
}
