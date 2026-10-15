#include "test_support.hh"
#include "../src/CompilerInternal.hh"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace coarse_die::detail {
std::vector<std::shared_ptr<Plan>> enumerate_mappings(
    const Hardware&, const nlohmann::json& normalized_task);
}

namespace {
using coarse_die::Activity;
using coarse_die::Hardware;
using coarse_die::Plan;
using coarse_die::Runtime;
using coarse_die::detail::normalize_task;
using Json = nlohmann::json;

constexpr std::uint64_t kArrivalNs = 100;
constexpr std::uint64_t kEndpointBytes = 64;
constexpr std::size_t kDrainStepLimit = 100;
constexpr std::uint64_t kEventCountLimit = 100;
constexpr std::uint64_t kScalingSlack = 20;

struct Case {
  std::string name;
  std::uint64_t m = 16;
  std::uint64_t n = 64;
  std::uint64_t k = 64;
  std::uint64_t layer_instances = 1;
  std::uint64_t nx = 4;
  std::uint64_t ny = 4;
};

struct Measurement {
  Json report;
  std::uint64_t internal_events = 0;
  std::uint64_t allocation_updates = 0;
  std::size_t activity_records = 0;
  std::size_t description_count = 0;
};

std::uint64_t ceil_div(std::uint64_t value, std::uint64_t divisor) {
  return value / divisor + static_cast<std::uint64_t>(value % divisor != 0);
}

std::uint64_t exact_u64(const Json& value, const std::string& label) {
  if (value.is_number_unsigned()) return value.get<std::uint64_t>();
  if (value.is_number_integer()) {
    const auto signed_value = value.get<std::int64_t>();
    coarse_die_test::require(signed_value >= 0, label + " is negative");
    return static_cast<std::uint64_t>(signed_value);
  }
  if (value.is_string()) {
    const std::string text = value.get<std::string>();
    std::size_t consumed = 0;
    const auto parsed = std::stoull(text, &consumed, 10);
    coarse_die_test::require(consumed == text.size(), label + " is not an exact integer");
    return static_cast<std::uint64_t>(parsed);
  }
  throw std::runtime_error(label + " is not an exact unsigned integer");
}

std::uint64_t checked_add(std::uint64_t lhs, std::uint64_t rhs,
                          const std::string& label) {
  if (rhs > std::numeric_limits<std::uint64_t>::max() - lhs)
    throw std::overflow_error(label + " overflowed uint64");
  return lhs + rhs;
}

std::uint64_t sum_exact_row_direction(const Json& bytes,
                                      std::size_t direction,
                                      const std::string& label) {
  std::uint64_t sum = 0;
  coarse_die_test::require(bytes.is_array() && bytes.size() == 2,
                           label + " must contain two HBM rows");
  for (std::size_t row = 0; row < 2; ++row) {
    coarse_die_test::require(bytes.at(row).is_array() &&
                                 bytes.at(row).size() == 2,
                             label + " must contain read/write directions");
    sum = checked_add(sum,
                      exact_u64(bytes.at(row).at(direction), label), label);
  }
  return sum;
}

Json gemm_task(const Case& spec) {
  return {{"schema", "coarse_die_task_v2"},
          {"op_kind", "gemm"},
          {"phase", "prefill"},
          {"tensor_shapes", {{"m", spec.m}, {"n", spec.n}, {"k", spec.k}}},
          {"storage_dtype", {{"activations", "int4"},
                              {"weights", "int4"},
                              {"output", "bf16"}}},
          {"execution_profile", "int4_matrix_vec_fp32_assumed_v1"},
          {"layer_instances", spec.layer_instances},
          {"weight_passes", 1},
          {"fusion_policy", "auto"}};
}

Hardware hardware_for_grid(std::uint64_t nx, std::uint64_t ny) {
  Json config = coarse_die_test::read_json(
      std::string(COARSE_DIE_SOURCE_DIR) + "/configs/example_hardware.json");
  auto& architecture = config["architecture"];
  // A small square Core keeps every requested square grid within the source
  // file's original isotropic, aspect-ratio, area, and reticle constraints.
  architecture["core_type"] = "small";
  architecture["sram_mib_per_core"] = 0.25;
  architecture["core_columns_nx"] = nx;
  architecture["core_rows_ny"] = ny;
  architecture["die_geometry"] = "isotropic";
  // This fixture fixes active service at the arrival tick; region ready-time
  // behavior is covered separately by region_overhead_test.
  config["performance_profile"]["launch_ns_per_macro"]=0;
  config["performance_profile"]["memory_setup_ns_per_macro"]=0;
  return coarse_die::resolve_hardware(config);
}

std::shared_ptr<const Plan> fixed_os_plan(const Hardware& hardware,
                                          const Case& spec) {
  const Json normalized = normalize_task(hardware, gemm_task(spec));
  const std::uint64_t tm = std::min<std::uint64_t>(16, spec.m);
  const std::uint64_t tn = std::min<std::uint64_t>(64, spec.n);
  const std::uint64_t tk = std::min<std::uint64_t>(64, spec.k);
  const auto candidates = coarse_die::detail::enumerate_mappings(
      hardware, normalized);
  for (const auto& candidate : candidates) {
    const auto& summary = candidate->summary;
    if (summary.at("dataflow_id") == "output_stationary_v2" &&
        summary.at("tm").get<std::uint64_t>() == tm &&
        summary.at("tn").get<std::uint64_t>() == tn &&
        summary.at("tk").get<std::uint64_t>() == tk) {
      return candidate;
    }
  }
  throw std::runtime_error(spec.name + ": requested fixed OS tile mapping is unavailable");
}

Json local_hbm_endpoint() {
  return {{"endpoint_id", 1},
          {"transaction_id", 1},
          {"owner_kind", "comm"},
          {"direction", "read"},
          {"physical_bytes", kEndpointBytes},
          {"path_class", "hbm_to_edge"},
          {"eta_hbm", 1.0},
          {"route", {{"internal_route", false}}}};
}

Measurement run_case(const Case& spec) {
  const Hardware hardware = hardware_for_grid(spec.nx, spec.ny);
  const auto plan = fixed_os_plan(hardware, spec);
  const std::uint64_t tm = plan->summary.at("tm").get<std::uint64_t>();
  const std::uint64_t tn = plan->summary.at("tn").get<std::uint64_t>();
  const std::uint64_t tk = plan->summary.at("tk").get<std::uint64_t>();
  const std::uint64_t output_tiles =
      ceil_div(spec.m, tm) * ceil_div(spec.n, tn);
  const std::uint64_t expected_active_cores = std::min(
      hardware.derived.at("die").at("core_count").get<std::uint64_t>(),
      output_tiles);

  coarse_die_test::require(plan->epochs.size() >= 3 &&
                               plan->epochs.size() <= 4,
                           spec.name + ": OS v3 plan must use three or four epochs");
  coarse_die_test::require(
      plan->summary.at("active_cores").get<std::uint64_t>() ==
          expected_active_cores,
      spec.name + ": fixed mapping must expose the actual active Core count");
  const std::size_t descriptions = coarse_die::service_description_count(*plan);
  coarse_die_test::require(
      plan->summary.at("service_description_count").get<std::size_t>() ==
          descriptions,
      spec.name + ": summary and physical service description counts differ");
  coarse_die_test::require(descriptions <= 8,
                           spec.name + ": OS description count exceeds eight");
  if (output_tiles == 1 && spec.layer_instances == 1) {
    coarse_die_test::require(descriptions <= 5,
                             spec.name + ": single-output OS exceeds five descriptions");
  }

  std::size_t compiled_activity_records = 0;
  for (const auto& epoch : plan->epochs)
    compiled_activity_records += epoch.activities.size();

  Runtime runtime(hardware);
  runtime.advance_to(kArrivalNs);
  const auto workload_handle = runtime.submit(
      plan, coarse_die_test::owner(1, 1, 1), kArrivalNs);
  const auto endpoint_handle = runtime.submit_endpoint(
      local_hbm_endpoint(), coarse_die_test::owner(2, 2, 2), kArrivalNs);
  const Json arrival_counters = runtime.counters();
  const Json workload_at_arrival = runtime.query_progress(workload_handle);
  const Json endpoint_at_arrival = runtime.query_progress(endpoint_handle);
  coarse_die_test::require(
      workload_at_arrival.at("issue_ns").get<std::uint64_t>() == kArrivalNs &&
          endpoint_at_arrival.at("issue_ns").get<std::uint64_t>() == kArrivalNs,
      spec.name + ": workload and endpoint must share the fixed arrival tick");
  coarse_die_test::require(
      arrival_counters.at("active_owners").get<std::size_t>() == 2 &&
          arrival_counters.at("active_activities").get<std::size_t>() == 2,
      spec.name + ": active Runtime owner/activity count must stay fixed at two");
  coarse_die_test::require(
      endpoint_at_arrival.at("activities").size() == 1 &&
          endpoint_at_arrival.at("activities").at(0).at("role") == "endpoint",
      spec.name + ": fixed external event must be one physical endpoint activity");

  std::size_t drain_steps = 0;
  while (!runtime.drained()) {
    const auto next = runtime.next_event();
    coarse_die_test::require(next.has_value(),
                             spec.name + ": live work has no next Runtime event");
    coarse_die_test::require(*next > runtime.now_ns(),
                             spec.name + ": Runtime event did not advance time");
    coarse_die_test::require(++drain_steps <= kDrainStepLimit,
                             spec.name + ": Runtime exceeded fixed drain-step bound");
    runtime.advance_to(*next);
  }

  const Json workload_done = runtime.query_progress(workload_handle);
  const Json endpoint_done = runtime.query_progress(endpoint_handle);
  const Json workload_served = workload_done.at("served");
  const std::uint64_t actual_read = sum_exact_row_direction(
      workload_served.at("hbm_bytes_by_row_direction_exact"), 0,
      spec.name + " actual HBM reads");
  const std::uint64_t actual_write = sum_exact_row_direction(
      workload_served.at("hbm_bytes_by_row_direction_exact"), 1,
      spec.name + " actual HBM writes");
  const std::uint64_t actual_int4_ops = exact_u64(
      workload_served.at("int4_ops_exact_decimal"),
      spec.name + " actual INT4 ops");
  coarse_die_test::require(
      actual_read == plan->summary.at("hbm_read_bytes_total").get<std::uint64_t>() &&
          actual_write == plan->summary.at("hbm_write_bytes_total").get<std::uint64_t>() &&
          actual_int4_ops == plan->summary.at("executed_int4_ops").get<std::uint64_t>(),
      spec.name + ": exact physical terminal bytes/ops differ from Plan summary");
  coarse_die_test::require(
      sum_exact_row_direction(
          endpoint_done.at("served").at("hbm_bytes_by_row_direction_exact"), 0,
          spec.name + " endpoint HBM reads") == kEndpointBytes &&
          sum_exact_row_direction(
              endpoint_done.at("served").at("hbm_bytes_by_row_direction_exact"), 1,
              spec.name + " endpoint HBM writes") == 0,
      spec.name + ": fixed local HBM endpoint did not serve its exact payload");

  const Json final_counters = runtime.counters();
  const std::uint64_t internal_events =
      final_counters.at("internal_events").get<std::uint64_t>();
  const std::uint64_t allocation_updates =
      final_counters.at("allocation_updates").get<std::uint64_t>();
  coarse_die_test::require(internal_events < kEventCountLimit,
                           spec.name + ": internal event count exceeded fixed bound");
  coarse_die_test::require(allocation_updates < kEventCountLimit,
                           spec.name + ": allocation update count exceeded fixed bound");
  coarse_die_test::require(drain_steps < kDrainStepLimit,
                           spec.name + ": drain steps must remain below 100");

  const auto all_activity_records = compiled_activity_records +
      endpoint_at_arrival.at("activities").size();
  coarse_die_test::require(all_activity_records < 16,
                           spec.name + ": activity description count is not bounded");
  Measurement result;
  result.internal_events = internal_events;
  result.allocation_updates = allocation_updates;
  result.activity_records = all_activity_records;
  result.description_count = descriptions;
  result.report = {
      {"case", spec.name},
      {"shape", {{"m", spec.m}, {"n", spec.n}, {"k", spec.k}}},
      {"layer_instances", spec.layer_instances},
      {"core_grid", {{"nx", spec.nx}, {"ny", spec.ny},
                      {"core_count", hardware.derived.at("die").at("core_count")}}},
      {"fixed_os_tile", {{"tm", tm}, {"tn", tn}, {"tk", tk}}},
      {"active_cores", expected_active_cores},
      {"external_event", {{"type", "local_hbm_endpoint_read"},
                           {"arrival_ns", kArrivalNs},
                           {"physical_bytes", kEndpointBytes}}},
      {"active_owners_at_arrival", arrival_counters.at("active_owners")},
      {"active_activities_at_arrival", arrival_counters.at("active_activities")},
      {"runtime_events", {{"internal_events", internal_events},
                           {"allocation_updates", allocation_updates},
                           {"drain_steps", drain_steps}}},
      {"actual_counts", {{"compiled_plan_activity_records", compiled_activity_records},
                          {"endpoint_activity_records",
                           endpoint_at_arrival.at("activities").size()},
                          {"runtime_activity_records", all_activity_records},
                          {"compiled_plan_epochs", plan->epochs.size()},
                          {"compiled_service_descriptions", descriptions},
                          {"description_count_semantics",
                           "independent activity segments including terminal switch"}}},
      {"terminal_physical_totals", {{"hbm_read_bytes", actual_read},
                                     {"hbm_write_bytes", actual_write},
                                     {"int4_ops", actual_int4_ops},
                                     {"summary_matches_exact_runtime_service", true}}}};
  return result;
}

void require_bounded_scaling(const std::vector<Measurement>& measurements,
                             const std::vector<std::size_t>& indices,
                             const std::string& family,
                             bool require_same_state_shape = true) {
  coarse_die_test::require(indices.size() >= 2,
                           family + ": scaling family needs at least two points");
  std::uint64_t min_events = std::numeric_limits<std::uint64_t>::max();
  std::uint64_t max_events = 0;
  std::uint64_t min_allocations = std::numeric_limits<std::uint64_t>::max();
  std::uint64_t max_allocations = 0;
  const auto& first = measurements.at(indices.front());
  for (const auto index : indices) {
    const auto& sample = measurements.at(index);
    min_events = std::min(min_events, sample.internal_events);
    max_events = std::max(max_events, sample.internal_events);
    min_allocations = std::min(min_allocations, sample.allocation_updates);
    max_allocations = std::max(max_allocations, sample.allocation_updates);
    if (require_same_state_shape) {
      coarse_die_test::require(
          sample.activity_records == first.activity_records &&
              sample.description_count == first.description_count,
          family + ": plan activity/description count grew with workload size");
    }
  }
  coarse_die_test::require(max_events - min_events <= kScalingSlack,
                           family + ": internal events grew with the scaled dimension");
  coarse_die_test::require(max_allocations - min_allocations <= kScalingSlack,
                           family + ": allocation updates grew with the scaled dimension");
}

void verify() {
  std::vector<Case> cases = {
      {"base_m16_n64_k64", 16, 64, 64, 1, 4, 4},
      {"k4096", 16, 64, 4096, 1, 4, 4},
      {"k262144", 16, 64, 262144, 1, 4, 4},
      {"tiles_n4096", 16, 4096, 64, 1, 4, 4},
      {"tiles_n65536", 16, 65536, 64, 1, 4, 4},
      {"cores1_m1024_n1024_k4096", 1024, 1024, 4096, 1, 1, 1},
      {"cores4_m1024_n1024_k4096", 1024, 1024, 4096, 1, 2, 2},
      {"cores64_m1024_n1024_k4096", 1024, 1024, 4096, 1, 8, 8},
      {"repeat1024", 16, 64, 64, 1024, 4, 4},
      {"repeat1000000", 16, 64, 64, 1000000, 4, 4}};
  std::vector<Measurement> measurements;
  measurements.reserve(cases.size());
  for (const auto& spec : cases) measurements.push_back(run_case(spec));

  require_bounded_scaling(measurements, {0, 1, 2}, "K64_to_K262144", false);
  require_bounded_scaling(measurements, {3, 4}, "N4096_to_N65536");
  require_bounded_scaling(measurements, {5, 6, 7}, "Core1_to_Core64");
  require_bounded_scaling(measurements, {8, 9}, "repeat1024_to_repeat1000000");

  Json rows = Json::array();
  for (const auto& measurement : measurements)
    rows.push_back(measurement.report);
  std::cout << Json{{"test", "os_service_scaling_test"},
                    {"status", "pass"},
                    {"event_count_limit", kEventCountLimit},
                    {"drain_step_limit", kDrainStepLimit},
                    {"fixed_active_owners", 2},
                    {"cases", std::move(rows)}}
                   .dump()
            << '\n';
}
}  // namespace

int main() {
  return coarse_die_test::run_test("os_service_scaling_test", verify);
}
