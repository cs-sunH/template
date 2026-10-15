#include "/workspace/template/die/tests/test_support.hh"
#include "/workspace/template/die/src/CompilerInternal.hh"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

namespace coarse_die::detail {
std::vector<std::shared_ptr<Plan>> enumerate_mappings(
    const Hardware&, const nlohmann::json&);
}

extern "C" void die_profile_reset();
extern "C" void die_profile_read(std::uint64_t*, std::uint64_t*,
                                 std::uint64_t*, std::uint64_t*,
                                 std::uint64_t*);

namespace {
using namespace coarse_die;
using namespace coarse_die_test;
using Json = nlohmann::json;
using Clock = std::chrono::steady_clock;

constexpr std::size_t kStages = 10;
const char* kStageNames[kStages] = {
    "update_total", "allocate_total", "collect_raw_refs", "merge_owner_blocks",
    "sort_reduced_refs", "build_resource_private_credit_rows",
    "packing_sign_classification", "prepare_ratio_rows_and_owners",
    "ratio_solver_including_certification", "publish_to_mirrors"};

struct TimingSamples {
  std::vector<std::uint64_t> wall[kStages];
  std::vector<std::uint64_t> cpu[kStages];
  std::vector<std::uint64_t> call_wall;
  std::vector<std::uint64_t> call_cpu;
  std::uint64_t raw_refs = 0, reduced_refs = 0, rows = 0;
};

std::uint64_t percentile(std::vector<std::uint64_t> values, double p) {
  std::sort(values.begin(), values.end());
  const auto rank = static_cast<std::size_t>(std::ceil(p * values.size()));
  return values[std::max<std::size_t>(1, rank) - 1];
}

std::uint64_t mean(const std::vector<std::uint64_t>& values) {
  if (values.empty()) return 0;
  return std::accumulate(values.begin(), values.end(), std::uint64_t{0}) /
         values.size();
}

Json gemm_task(const std::string& profile) {
  return {{"op_kind", "gemm"},
          {"tensor_shapes", {{"m", 128}, {"n", 128}, {"k", 128}}},
          {"storage_dtype", {{"weights", "bf16"},
                              {"activations", "bf16"},
                              {"output", "bf16"}}},
          {"execution_profile", profile},
          {"layer_instances", 1}, {"weight_passes", 1}};
}

bool is_role(const Epoch& epoch, Role role) {
  return std::any_of(epoch.activities.begin(), epoch.activities.end(),
                     [role](const Activity& activity) {
                       return activity.role == role;
                     });
}

struct OsBulk {
  std::shared_ptr<const Plan> plan;
  std::size_t epoch = 0;
};

OsBulk select_os_bulk(const Hardware& hardware, const Json& task) {
  const Json normalized = detail::normalize_task(hardware, task);
  const auto candidates = detail::enumerate_mappings(hardware, normalized);
  for (const auto& candidate : candidates) {
    if (candidate->summary.value("dataflow_id", std::string()) !=
            "output_stationary_v2" ||
        !candidate->summary.contains("service_curves")) continue;
    for (const auto& curve : candidate->summary.at("service_curves")) {
      const std::size_t begin = curve.at("epoch_begin").get<std::size_t>();
      const std::size_t end = begin + curve.at("epoch_count").get<std::size_t>();
      for (std::size_t epoch = begin; epoch < end; ++epoch) {
        const auto& current = candidate->epochs.at(epoch);
        if (!is_role(current, Role::Load) || !is_role(current, Role::Compute) ||
            !is_role(current, Role::Store)) continue;
        if (std::none_of(current.activities.begin(), current.activities.end(),
                         [](const Activity& activity) {
                           return !activity.credit_curve.empty();
                         })) continue;
        return {candidate, epoch};
      }
    }
  }
  throw std::runtime_error("no OS bulk plan with three curved services");
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const std::size_t repetitions = argc > 1 ? std::stoul(argv[1]) : 120;
    if (repetitions < 20 || repetitions > 2000)
      throw std::runtime_error("usage: profile-j32 [repetitions 20..2000]");
    const Hardware hardware = example_hardware();
    const Json performance = hardware.input.value("performance_profile", Json::object());
    const std::string profile = performance.value(
        "execution_profile", std::string("int4_matrix_vec_fp32_assumed_v1"));
    const OsBulk selected = select_os_bulk(
        hardware, gemm_task(profile));
    Runtime runtime(hardware);
    std::vector<std::uint64_t> handles;
    for (std::uint64_t i = 0; i < 32; ++i) {
      handles.push_back(runtime.submit(
          selected.plan, Owner{i + 1, 1, i + 1, 1}, 0));
    }
    std::size_t events = 0;
    while (runtime.query_progress(handles.front()).at("epoch_index")
               .get<std::size_t>() < selected.epoch) {
      const auto next = runtime.next_event();
      if (!next) throw std::runtime_error("OS bulk setup has no next event");
      runtime.advance_to(*next);
      if (++events > 16) throw std::runtime_error("OS bulk setup event bound exceeded");
    }
    for (const auto handle : handles) {
      if (runtime.query_progress(handle).at("epoch_index") != selected.epoch)
        throw std::runtime_error("owners did not enter the same bulk epoch");
    }

    TimingSamples samples;
    for (std::size_t sample = 0; sample < repetitions + 10; ++sample) {
      die_profile_reset();
      const auto begin = Clock::now();
      runtime.update_capacity(HBM0, hardware.rates[HBM0], runtime.now_ns());
      const auto end = Clock::now();
      std::uint64_t wall[kStages]{}, cpu[kStages]{};
      die_profile_read(wall, cpu, &samples.raw_refs, &samples.reduced_refs,
                       &samples.rows);
      if (sample < 10) continue;
      samples.call_wall.push_back(static_cast<std::uint64_t>(
          std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count()));
      for (std::size_t stage = 0; stage < kStages; ++stage) {
        samples.wall[stage].push_back(wall[stage]);
        samples.cpu[stage].push_back(cpu[stage]);
      }
    }

    std::cout << "fixture=compiled_os_credit_curve_bulk_j32 plan="
              << selected.plan->summary.value("template_id", std::string())
              << " epoch=" << selected.epoch << " repetitions=" << repetitions
              << " raw_activity_refs=" << samples.raw_refs
              << " merged_solver_vars=" << samples.reduced_refs
              << " constraint_rows=" << samples.rows << '\n';
    std::cout << "public_update_wall_ns{mean=" << mean(samples.call_wall)
              << ",p50=" << percentile(samples.call_wall, .50)
              << ",p95=" << percentile(samples.call_wall, .95)
              << ",p99=" << percentile(samples.call_wall, .99) << "}\n";
    for (std::size_t stage = 0; stage < kStages; ++stage) {
      std::cout << kStageNames[stage] << " wall_ns{mean="
                << mean(samples.wall[stage]) << ",p50="
                << percentile(samples.wall[stage], .50) << ",p95="
                << percentile(samples.wall[stage], .95) << "} cpu_ns{mean="
                << mean(samples.cpu[stage]) << ",p50="
                << percentile(samples.cpu[stage], .50) << ",p95="
                << percentile(samples.cpu[stage], .95) << "}\n";
    }
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "profile-j32 failed: " << error.what() << '\n';
    return 1;
  }
}
