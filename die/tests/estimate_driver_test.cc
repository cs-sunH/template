#include "test_support.hh"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace coarse_die::detail {
std::vector<std::shared_ptr<Plan>> enumerate_mappings(
    const Hardware&, const nlohmann::json&);
}

using namespace coarse_die;
using namespace coarse_die_test;

namespace {

json gemm_task() {
  return {{"op_kind", "gemm"},
          {"tensor_shapes", {{"m", 7}, {"n", 5}, {"k", 9}}},
          {"storage_dtype", {{"weights", "int4"},
                             {"activations", "bf16"},
                             {"output", "bf16"}}},
          {"execution_profile", "int4_matrix_vec_fp32_assumed_v1"},
          {"layer_instances", 1},
          {"weight_passes", 1}};
}

std::uint64_t actual_runtime_finish_ns(
    const Hardware& hardware, std::shared_ptr<const Plan> plan) {
  Runtime runtime(hardware);
  const std::uint64_t handle = runtime.submit(std::move(plan), owner(1, 1, 1), 0);
  std::size_t events = 0;
  while (!runtime.drained()) {
    const auto next = runtime.next_event();
    require(next.has_value(), "actual Runtime has a next event until completion");
    require(*next > runtime.now_ns(), "actual Runtime events advance time");
    runtime.advance_to(*next);
    require(++events <= 1000000, "actual Runtime completion stays within its event budget");
  }
  const json progress = runtime.query_progress(handle);
  require(progress.at("state") == "finished",
          "actual Runtime terminal state is finished");
  return time_from_progress(progress, "finish_ns");
}

template <class Function>
void require_throws_prefix(Function&& function, const std::string& prefix,
                           const std::string& message) {
  try {
    function();
  } catch (const std::exception& error) {
    require(std::string(error.what()).find(prefix) == 0, message);
    return;
  }
  throw std::runtime_error(message + ": did not throw");
}

using u64 = std::uint64_t;
using u128 = unsigned __int128;
using RankKey = std::tuple<u64, u128, u64, u64, std::string,
                           u64, u64, u64, u64>;

RankKey candidate_rank_key(const Plan& candidate, u64 latency) {
  const json& summary = candidate.summary;
  const u128 hbm = static_cast<u128>(summary.at("hbm_read_bytes_total").get<u64>()) +
                   summary.at("hbm_write_bytes_total").get<u64>();
  return {latency, hbm,
          summary.at("hbm_workspace_peak_bytes").get<u64>(),
          summary.at("sram_peak_bytes_per_core").get<u64>(),
          summary.at("template_id").get<std::string>(),
          summary.at("tm").get<u64>(), summary.at("tn").get<u64>(),
          summary.at("tk").get<u64>(), summary.at("th").get<u64>()};
}

std::size_t compare_compiled_gemm_with_sealed_enumerator(
    const Hardware& hardware, const json& raw_task, const std::string& label,
    bool require_os_and_wr = false) {
  std::shared_ptr<const Plan> compiled;
  try {
    compiled = compile(hardware, raw_task);
  } catch (const std::exception& error) {
    throw std::runtime_error(label + ": " + error.what());
  }
  require(compiled != nullptr && compiled->task.is_object(),
          label + ": compile returns the normalized task with its Plan");
  const auto candidates = detail::enumerate_mappings(hardware, compiled->task);
  require(!candidates.empty(), label + ": normalized task has feasible candidates");

  const auto immutable_hardware = std::make_shared<const Hardware>(hardware);
  struct RankedCandidate {
    std::size_t index;
    u64 latency;
    RankKey key;
  };
  std::vector<RankedCandidate> ranked;
  ranked.reserve(candidates.size());
  bool saw_os = false;
  bool saw_wr = false;
  const double sram_capacity =
      hardware.input.at("performance_profile").at("rho_sram").get<double>() *
      hardware.derived.at("core").at("sram_bytes_per_core").get<double>();
  for (std::size_t index = 0; index < candidates.size(); ++index) {
    const auto& candidate = candidates[index];
    require(candidate->task == compiled->task,
            label + ": public candidates use exactly the normalized task");
    require(!candidate->fingerprint.empty() &&
                plan_to_json(*candidate).at("fingerprint") == candidate->fingerprint,
            label + ": public enumerator returns each candidate fully sealed");
    const u64 sealed_latency = detail::estimate_latency_ns(hardware, candidate);
    auto unsealed_candidate = std::make_shared<Plan>(*candidate);
    unsealed_candidate->fingerprint.clear();
    const u64 trial_latency = detail::estimate_candidate_latency_ns(
        immutable_hardware, unsealed_candidate);
    require(trial_latency == sealed_latency,
            label + ": every unsealed trial matches its sealed Runtime latency");
    require(unsealed_candidate->fingerprint.empty(),
            label + ": compile-only trial does not seal or mutate the Plan");
    require(static_cast<double>(candidate->summary.at("sram_peak_bytes_per_core").get<u64>()) <=
                sram_capacity + 1e-6,
            label + ": enumerated candidates respect the configured SRAM threshold");

    const std::string flow = candidate->summary.at("dataflow_id").get<std::string>();
    saw_os = saw_os || flow == "output_stationary_v2";
    saw_wr = saw_wr || flow == "weight_resident_v2";
    ranked.push_back({index, sealed_latency,
                      candidate_rank_key(*candidate, sealed_latency)});
  }
  if (require_os_and_wr) {
    require(saw_os && saw_wr,
            label + ": SRAM-constrained candidate set retains feasible OS and WR paths");
  }
  std::stable_sort(ranked.begin(), ranked.end(),
      [](const RankedCandidate& left, const RankedCandidate& right) {
        return left.key < right.key;
      });

  const auto& winner = *candidates[ranked.front().index];
  require(compiled->summary.at("candidate_count") == candidates.size(),
          label + ": compile ranks the complete public candidate set");
  for (const char* key : {"template_id", "dataflow_id", "tm", "tn", "tk", "th"}) {
    require(compiled->summary.at(key) == winner.summary.at(key),
            label + ": sealed tuple winner is preserved at " + key);
  }
  require(compiled->summary.at("estimated_latency_ns") == ranked.front().latency,
          label + ": selected candidate retains its exact Runtime latency");

  // Recreate the documented winner-only summary decoration on a fully sealed
  // public candidate, then verify the compiler's complete summary and final
  // canonical fingerprint. This covers the seal boundary without depending on
  // an implementation-private candidate object.
  Plan expected = winner;
  expected.summary["estimated_latency_ns"] = ranked.front().latency;
  expected.summary["alone_estimate_status"] = "ok_assumed";
  expected.summary["candidate_count"] = static_cast<u64>(candidates.size());
  expected.summary["selected_template"] = expected.summary.value("template_id", std::string());
  expected.summary["selection_rule"] =
      "ceil(estimate_alone_ns),hbm_payload_bytes,hbm_workspace_peak_bytes,sram_peak_bytes_per_core,template_id,tm,tn,tk,th";
  expected.summary["candidate_estimate_source"] = "estimate_alone(shared_runtime_engine)";
  expected.summary["activity_work_semantics"] =
      "activity work/bytes/ops are epoch total across repeat_count; repeat_count is audit metadata only; credit_delta is total signed credit change over progress 0..1";
  expected.summary["granularity_validation"] = "not_run";
  expected.summary["evidence_flags"] = {"engineering_assumption", "hardware_source_derived"};
  expected.fingerprint.clear();
  expected.fingerprint = plan_to_json(expected).at("fingerprint").get<std::string>();
  require(compiled->summary == expected.summary,
          label + ": final winner summary exactly matches its sealed candidate");
  require(compiled->fingerprint == expected.fingerprint &&
              plan_to_json(*compiled).at("fingerprint") == compiled->fingerprint,
          label + ": winner is canonically sealed after trial ranking");
  return candidates.size();
}

Hardware small_core_hardware(double rho_sram, double kappa_quant = 4.0,
                             double kappa_acc_convert = 2.0) {
  const std::string source_dir = COARSE_DIE_SOURCE_DIR;
  json config = read_json(source_dir + "/configs/example_hardware.json");
  auto& architecture = config["architecture"];
  architecture["core_type"] = "small";
  architecture["sram_mib_per_core"] = 0.25;
  architecture["noc_width_bits"] = 64;
  architecture["core_columns_nx"] = 1;
  architecture["core_rows_ny"] = 1;
  architecture["die_geometry"] = "isotropic";
  config["performance_profile"]["rho_sram"] = rho_sram;
  config["performance_profile"]["kappa_quant_eqops"] = kappa_quant;
  config["performance_profile"]["kappa_acc_convert_eqops"] = kappa_acc_convert;
  return resolve_hardware(config);
}

json gemm_semantics_task(u64 m, u64 n, u64 k, const std::string& activations,
                         const std::string& weights, const std::string& output,
                         const std::string& execution_profile,
                         u64 layers = 1, u64 passes = 1) {
  return {{"op_kind", "gemm"},
          {"tensor_shapes", {{"m", m}, {"n", n}, {"k", k}}},
          {"storage_dtype", {{"activations", activations},
                             {"weights", weights}, {"output", output}}},
          {"execution_profile", execution_profile},
          {"layer_instances", layers}, {"weight_passes", passes}};
}

void cold_compile_candidate_trials_cover_semantic_boundaries() {
  const Hardware ordinary = small_core_hardware(0.9);
  const json odd_int4_tail = gemm_semantics_task(
      3, 5, 7, "int4", "int4", "int4",
      "int4_matrix_vec_fp32_assumed_v1");
  const auto odd_plan = compile(ordinary, odd_int4_tail);
  require(odd_plan->summary.at("source_matrix_ops") == 210 &&
              odd_plan->summary.at("int4_input_bytes") == 29 &&
              odd_plan->summary.at("logical_tensor_bytes") == 37,
          "odd INT4 GEMM retains ceil-packed input, output, and exact matrix work");
  compare_compiled_gemm_with_sealed_enumerator(
      ordinary, odd_int4_tail, "odd INT4 tails");

  const Hardware fractional = small_core_hardware(0.9, 1.25, 0.75);
  const json strict_fp32_fractional = gemm_semantics_task(
      3, 5, 7, "fp32", "fp32", "int4",
      "fp32_vec_matrix_assumed_v1");
  const auto strict_plan = compile(fractional, strict_fp32_fractional);
  require(strict_plan->task.at("execution_profile") ==
              "fp32_vec_matrix_assumed_v1" &&
              has_fractional_work(*strict_plan) &&
              plan_to_json(*strict_plan).at("schema") == "coarse_die_plan_v3",
          "strict FP32 GEMM retains fractional output-quantization work and Plan v3");
  compare_compiled_gemm_with_sealed_enumerator(
      fractional, strict_fp32_fractional, "strict FP32 with fractional kappa");

  const json repeated = gemm_semantics_task(
      128, 128, 128, "bf16", "bf16", "bf16",
      "int4_matrix_vec_fp32_assumed_v1", 2, 3);
  const Hardware tight_sram = small_core_hardware(0.15);
  const std::size_t tight_count = compare_compiled_gemm_with_sealed_enumerator(
      tight_sram, repeated, "WR under tight SRAM with repeated layers/passes", true);
  require(tight_count > 0,
          "a physically feasible candidate survives the constrained SRAM threshold");
  const auto repeated_plan = compile(tight_sram, repeated);
  require(repeated_plan->task.at("layer_instances") == 2 &&
              repeated_plan->task.at("weight_passes") == 3,
          "repeated GEMM task preserves layer and weight-pass multiplicity");
  const Hardware roomy_sram = small_core_hardware(0.9);
  const std::size_t roomy_count = compare_compiled_gemm_with_sealed_enumerator(
      roomy_sram, repeated, "WR under roomy SRAM with repeated layers/passes");
  require(roomy_count > tight_count,
          "larger SRAM threshold admits additional complete mapping candidates");
  require(repeated_plan->summary.at("source_matrix_ops") == 25165824,
          "two layers times three weight passes multiply the source matrix ledger exactly");
}

void low_cost_estimate_matches_report_and_runtime() {
  const Hardware hardware = example_hardware();
  const auto plan = compile(hardware, gemm_task());
  const auto immutable_hardware = std::make_shared<const Hardware>(hardware);
  require(plan != nullptr, "compiler returns a Plan");
  require(plan->hardware_fingerprint == hardware.fingerprint,
          "compiled Plan retains the exact hardware fingerprint");
  require(plan_to_json(*plan).at("fingerprint") == plan->fingerprint,
          "compiled Plan retains its canonical fingerprint");

  const std::uint64_t candidate_latency =
      detail::estimate_latency_ns(hardware, plan);
  const std::uint64_t shared_hardware_latency =
      detail::estimate_latency_ns(immutable_hardware, plan);
  const json full_estimate = estimate_alone(hardware, *plan);
  const std::uint64_t runtime_latency =
      actual_runtime_finish_ns(hardware, plan);
  require(candidate_latency == runtime_latency,
          "low-cost candidate estimate equals an actual Runtime completion");
  require(shared_hardware_latency == candidate_latency,
          "shared immutable hardware estimate equals the by-value path");
  require(full_estimate.at("estimated_latency_ns") == runtime_latency,
          "public standalone estimate keeps the same Runtime latency");
  require(full_estimate.at("execution").at("finish_ns") == runtime_latency,
          "full report keeps the completed execution record");
  require(full_estimate.at("hardware_fingerprint") == hardware.fingerprint &&
              full_estimate.at("plan_fingerprint") == plan->fingerprint,
          "full report preserves plan and hardware fingerprints");
  require(full_estimate.at("resource_domains").size() == resource_count &&
              full_estimate.at("domain_capacities").size() == resource_count &&
              full_estimate.at("domain_work").size() == resource_count &&
              full_estimate.contains("runtime_counters"),
          "full report retains its resource and runtime fields");

  const auto repeated_plan = compile(hardware, gemm_task());
  require(repeated_plan->fingerprint == plan->fingerprint &&
              repeated_plan->summary.at("selected_template") ==
                  plan->summary.at("selected_template") &&
              repeated_plan->summary.at("estimated_latency_ns") ==
                  plan->summary.at("estimated_latency_ns"),
          "shared candidate hardware preserves deterministic selection and hash");
}

void runtime_shared_hardware_keeps_capacity_state_private() {
  const Hardware hardware = example_hardware();
  const auto immutable_hardware = std::make_shared<const Hardware>(hardware);
  const auto plan = compile(hardware, gemm_task());
  Runtime throttled(immutable_hardware);
  Runtime baseline(immutable_hardware);
  const double peak = hardware.rates[COMPUTE_GATE];
  bool has_compute_work = false;
  for (const auto& epoch : plan->epochs) {
    for (const auto& activity : epoch.activities) {
      has_compute_work = has_compute_work || activity.work[COMPUTE_GATE] > 0.0;
    }
  }
  require(peak > 0.0 && has_compute_work,
          "example GEMM has positive compute work and capacity");
  throttled.update_capacity(COMPUTE_GATE, 0.0, 0);

  const auto slow_handle = throttled.submit(plan, owner(2, 2, 1), 0);
  const auto base_handle = baseline.submit(plan, owner(3, 3, 1), 0);
  require(!throttled.terminal_finish_ns(slow_handle).has_value() &&
              !baseline.terminal_finish_ns(base_handle).has_value(),
          "typed terminal time is empty while executions are live");
  std::size_t throttled_events = 0;
  while (const auto next = throttled.next_event()) {
    throttled.advance_to(*next);
    require(++throttled_events <= 1000000,
            "throttled Runtime reaches its resource block within the event budget");
  }
  require(!throttled.drained(),
          "zero capacity blocks unfinished work in the throttled Runtime");
  require(baseline.next_event().has_value(),
          "the other Runtime retains its independent positive capacity");
  while (!baseline.drained()) {
    const auto next = baseline.next_event();
    require(next.has_value(), "baseline Runtime has a next event until completion");
    baseline.advance_to(*next);
  }
  const auto base_finish = baseline.terminal_finish_ns(base_handle);
  require(!throttled.terminal_finish_ns(slow_handle).has_value() &&
              base_finish.has_value(),
          "typed terminal time is available only after an execution terminates");
  require(baseline.query_progress(base_handle).at("finish_ns") == *base_finish,
          "typed terminal time matches the public progress report");
  require(throttled.hardware().fingerprint == hardware.fingerprint &&
              baseline.hardware().fingerprint == hardware.fingerprint &&
              immutable_hardware->rates[COMPUTE_GATE] == peak &&
              throttled.hardware().rates[COMPUTE_GATE] == peak &&
              baseline.hardware().rates[COMPUTE_GATE] == peak &&
              throttled.counters().at("capacity").at(COMPUTE_GATE) == 0.0 &&
              baseline.counters().at("capacity").at(COMPUTE_GATE) == peak,
          "per-Runtime capacity overrides do not mutate shared hardware peaks");

  const auto cancel_time = throttled.now_ns();
  throttled.cancel(slow_handle, cancel_time);
  const auto cancelled_finish = throttled.terminal_finish_ns(slow_handle);
  require(cancelled_finish.has_value() && *cancelled_finish == cancel_time,
          "typed terminal time is available for a cancelled execution");
  baseline.destroy_execution(base_handle);
  require_throws_prefix(
      [&]() { baseline.terminal_finish_ns(base_handle); }, "stale_handle:",
      "typed terminal time preserves stale-handle errors");
}

void low_cost_estimate_still_validates_hardware_and_physical_work() {
  const Hardware hardware = example_hardware();
  const auto plan = compile(hardware, gemm_task());

  auto wrong_hardware = std::make_shared<Plan>(*plan);
  wrong_hardware->hardware_fingerprint = "wrong-hardware-fingerprint";
  require_throws_prefix(
      [&]() { detail::estimate_latency_ns(hardware, wrong_hardware); },
      "incompatible_fingerprint:",
      "low-cost estimate rejects a plan from different hardware");

  auto invalid_work = std::make_shared<Plan>(*plan);
  invalid_work->epochs.front().activities.front().work[COMPUTE_GATE] = -1.0;
  require_throws_prefix(
      [&]() { detail::estimate_latency_ns(hardware, invalid_work); },
      "invalid_plan:",
      "low-cost estimate still validates physical resource work");
}

void estimate_driver_preserves_overflow_failure() {
  const Hardware hardware = example_hardware();
  const auto plan = compile(hardware, gemm_task());
  auto late_plan = std::make_shared<Plan>(*plan);
  late_plan->launch_ns = std::numeric_limits<std::uint64_t>::max();
  late_plan->fingerprint.clear();
  late_plan->fingerprint =
      plan_to_json(*late_plan).at("fingerprint").get<std::string>();
  require_throws_prefix(
      [&]() { detail::estimate_latency_ns(hardware, late_plan); },
      "overflow:",
      "low-cost estimate preserves Runtime event-time overflow errors");
}

void compile_candidate_trials_match_sealed_gemm_ranking() {
  const std::string source_dir = COARSE_DIE_SOURCE_DIR;
  const Hardware hardware = resolve_hardware(read_json(
      source_dir + "/tests/fixtures/coarse_die_v2.json"));
  const json task = read_json(source_dir +
      "/docs/evidence/cache-lifecycle-20261014/performance/fixtures/gemm_128x128x128.json");
  const json normalized_task{{"schema", "coarse_die_task_v2"},
      {"op_kind", "gemm"}, {"phase", "prefill"},
      {"tensor_shapes", task.at("tensor_shapes")}, {"spans", json::array()},
      {"tp_slice", {{"tensor_parallel", 1}, {"tensor_parallel_rank", 0}}},
      {"storage_dtype", {{"weights", "bf16"}, {"activations", "bf16"},
          {"kv", "bf16"}, {"output", "bf16"}, {"parameters", "bf16"},
          {"scores", "fp32"}, {"accumulator", "fp32"}}},
      {"execution_profile", task.at("execution_profile")},
      {"layer_instances", 1}, {"weight_passes", 1},
      {"fusion_policy", "auto"}};
  const auto candidates = detail::enumerate_mappings(hardware, normalized_task);
  require(candidates.size() == 28,
          "public mapping enumerator retains all 28 feasible 128^3 GEMM candidates");

  using u64 = std::uint64_t;
  using u128 = unsigned __int128;
  using RankKey = std::tuple<u64, u128, u64, u64, std::string,
                             u64, u64, u64, u64>;
  struct RankedCandidate {
    std::size_t index;
    RankKey key;
  };
  const auto immutable_hardware =
      std::make_shared<const Hardware>(hardware);
  std::vector<u64> sealed_latencies;
  std::vector<RankedCandidate> ranked;
  sealed_latencies.reserve(candidates.size());
  ranked.reserve(candidates.size());
  for (std::size_t index = 0; index < candidates.size(); ++index) {
    const auto& candidate = candidates[index];
    require(!candidate->fingerprint.empty() &&
                plan_to_json(*candidate).at("fingerprint") == candidate->fingerprint,
            "public mapping enumeration continues to seal each candidate hash");
    const u64 sealed_latency = detail::estimate_latency_ns(hardware, candidate);
    auto unsealed_candidate = std::make_shared<Plan>(*candidate);
    unsealed_candidate->fingerprint.clear();
    const u64 trial_latency = detail::estimate_candidate_latency_ns(
        immutable_hardware, unsealed_candidate);
    require(trial_latency == sealed_latency,
            "unsealed compile-only Runtime trial exactly matches each sealed candidate latency");
    require(unsealed_candidate->fingerprint.empty(),
            "candidate Runtime trial does not seal or mutate its private Plan");
    sealed_latencies.push_back(sealed_latency);

    const json& summary = candidate->summary;
    const u128 hbm = static_cast<u128>(summary.at("hbm_read_bytes_total").get<u64>()) +
                     summary.at("hbm_write_bytes_total").get<u64>();
    ranked.push_back({index, RankKey{sealed_latency, hbm,
        summary.at("hbm_workspace_peak_bytes").get<u64>(),
        summary.at("sram_peak_bytes_per_core").get<u64>(),
        summary.at("template_id").get<std::string>(),
        summary.at("tm").get<u64>(), summary.at("tn").get<u64>(),
        summary.at("tk").get<u64>(), summary.at("th").get<u64>()}});
  }
  std::stable_sort(ranked.begin(), ranked.end(),
      [](const RankedCandidate& left, const RankedCandidate& right) {
        return left.key < right.key;
      });
  const auto compiled = compile(hardware, task);
  const auto& expected = *candidates[ranked.front().index];
  require(compiled->summary.at("candidate_count") == candidates.size(),
          "compile ranks the complete feasible GEMM candidate set");
  for (const char* key : {"template_id", "dataflow_id", "tm", "tn", "tk", "th"}) {
    require(compiled->summary.at(key) == expected.summary.at(key),
            std::string("compile winner follows the sealed-candidate tie tuple at ") + key);
  }
  require(compiled->summary.at("estimated_latency_ns") ==
              sealed_latencies[ranked.front().index],
          "compile publishes the same Runtime latency used to rank sealed candidates");
  const auto serialized_winner = plan_to_json(*compiled);
  require(serialized_winner.at("schema") == "coarse_die_plan_v4" &&
              compiled->summary.at("service_model_revision") ==
                  "analytical_cumulative_os_v3" &&
              serialized_winner.at("fingerprint") == compiled->fingerprint,
          "compile seals the ranked winner with its canonical Plan v4 service-model identity");

  auto unsealed = std::make_shared<Plan>(*candidates.front());
  unsealed->fingerprint.clear();
  Runtime public_runtime(hardware);
  require_throws_prefix(
      [&] { (void)public_runtime.submit(unsealed, owner(7, 7, 1), 0); },
      "invalid_plan:",
      "public Runtime::submit continues to reject an unsealed candidate");

  auto invalid_work = std::make_shared<Plan>(*unsealed);
  invalid_work->epochs.front().activities.front().work[COMPUTE_GATE] = -1.0;
  require_throws_prefix(
      [&] { (void)detail::estimate_candidate_latency_ns(
          immutable_hardware, invalid_work); },
      "plan.epochs[",
      "unsealed candidate trials still validate physical work");

  auto invalid_hbm_service = std::make_shared<Plan>(*unsealed);
  auto& invalid_hbm_activity =
      invalid_hbm_service->epochs.front().activities.front();
  invalid_hbm_activity.work[HBM0] = 0.0;
  invalid_hbm_activity.work[HBM1] = 0.0;
  invalid_hbm_activity.hbm_bytes[0][0] = 1;
  require_throws_prefix(
      [&] { (void)detail::estimate_candidate_latency_ns(
          immutable_hardware, invalid_hbm_service); },
      "invalid_plan:",
      "unsealed candidate trials still validate HBM payload against physical row service");

  auto invalid_hardware = std::make_shared<Plan>(*unsealed);
  invalid_hardware->hardware_fingerprint = "wrong-hardware-fingerprint";
  require_throws_prefix(
      [&] { (void)detail::estimate_candidate_latency_ns(
          immutable_hardware, invalid_hardware); },
      "incompatible_fingerprint:",
      "unsealed candidate trials still validate hardware identity");

  auto invalid_credit = std::make_shared<Plan>(*unsealed);
  invalid_credit->epochs.front().credit_initial[0] =
      invalid_credit->epochs.front().credit_capacity[0] + 1.0;
  require_throws_prefix(
      [&] { (void)detail::estimate_candidate_latency_ns(
          immutable_hardware, invalid_credit); },
      "plan.epochs[",
      "unsealed candidate trials still validate finite-buffer credit");

  auto invalid_task = std::make_shared<Plan>(*unsealed);
  invalid_task->task = nullptr;
  require_throws_prefix(
      [&] { (void)detail::estimate_candidate_latency_ns(
          immutable_hardware, invalid_task); },
      "plan.task:",
      "unsealed candidate trials still validate the task object");

  auto invalid_summary = std::make_shared<Plan>(*unsealed);
  invalid_summary->summary = nullptr;
  require_throws_prefix(
      [&] { (void)detail::estimate_candidate_latency_ns(
          immutable_hardware, invalid_summary); },
      "plan.summary:",
      "unsealed candidate trials still validate the summary object");

  auto invalid_epoch_name = std::make_shared<Plan>(*unsealed);
  invalid_epoch_name->epochs.front().name.clear();
  require_throws_prefix(
      [&] { (void)detail::estimate_candidate_latency_ns(
          immutable_hardware, invalid_epoch_name); },
      "plan.epochs[0].name:",
      "unsealed candidate trials still validate non-empty epoch names");

  auto invalid_duplicate_id = std::make_shared<Plan>(*unsealed);
  std::size_t first_nonempty = invalid_duplicate_id->epochs.size();
  std::size_t second_nonempty = invalid_duplicate_id->epochs.size();
  for (std::size_t index = 0; index < invalid_duplicate_id->epochs.size(); ++index) {
    if (invalid_duplicate_id->epochs[index].activities.empty()) continue;
    if (first_nonempty == invalid_duplicate_id->epochs.size())
      first_nonempty = index;
    else {
      second_nonempty = index;
      break;
    }
  }
  require(first_nonempty < invalid_duplicate_id->epochs.size() &&
              second_nonempty < invalid_duplicate_id->epochs.size(),
          "GEMM fixture has activities in at least two epochs for global ID validation");
  invalid_duplicate_id->epochs[second_nonempty].activities.front().id =
      invalid_duplicate_id->epochs[first_nonempty].activities.front().id;
  require_throws_prefix(
      [&] { (void)detail::estimate_candidate_latency_ns(
          immutable_hardware, invalid_duplicate_id); },
      "plan.epochs[",
      "unsealed candidate trials still validate globally unique activity IDs");

  auto invalid_idle_q = std::make_shared<Plan>(*unsealed);
  auto& idle = invalid_idle_q->epochs.front().activities.front();
  idle.work.fill(0.0);
  idle.hbm_bytes = {};
  idle.canonical_q = 0.0;
  require_throws_prefix(
      [&] { (void)detail::estimate_candidate_latency_ns(
          immutable_hardware, invalid_idle_q); },
      "plan.epochs[0].activities[0].canonical_q:",
      "unsealed candidate trials still validate canonical q for zero-work activities");
}

}  // namespace

int main() {
  return run_test("estimate_driver_test", [] {
    low_cost_estimate_matches_report_and_runtime();
    runtime_shared_hardware_keeps_capacity_state_private();
    low_cost_estimate_still_validates_hardware_and_physical_work();
    estimate_driver_preserves_overflow_failure();
    compile_candidate_trials_match_sealed_gemm_ranking();
    cold_compile_candidate_trials_cover_semantic_boundaries();
  });
}
