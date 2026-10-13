#include "test_support.hh"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <sstream>
#include <string>
#include <vector>

namespace coarse_die::detail {
std::vector<std::shared_ptr<Plan>> enumerate_mappings(
    const Hardware&, const nlohmann::json&);
}

using namespace coarse_die;
using namespace coarse_die_test;

namespace {
using u64 = std::uint64_t;
using u128 = unsigned __int128;

u64 ceildiv(u64 a, u64 b) { return a / b + static_cast<u64>(a % b != 0); }

std::vector<u64> extents(u64 length, u64 tile) {
  std::vector<u64> result;
  for (u64 offset = 0; offset < length; offset += tile)
    result.push_back(std::min(tile, length - offset));
  return result;
}

Hardware hardware_with_cores(u64 cores) {
  auto config = read_json(std::string(COARSE_DIE_SOURCE_DIR) +
                          "/configs/example_hardware.json");
  auto& architecture = config["architecture"];
  architecture["core_type"] = "small";
  architecture["sram_mib_per_core"] = 0.25;
  architecture["noc_width_bits"] = 64;
  const u64 columns = cores == 4 ? 2 : cores;
  architecture["core_columns_nx"] = columns;
  architecture["core_rows_ny"] = cores / columns;
  architecture["die_geometry"] = "isotropic";
  config["performance_profile"]["kappa_quant_eqops"] = 4.0;
  config["performance_profile"]["kappa_acc_convert_eqops"] = 2.0;
  return resolve_hardware(config);
}

json gemm_task(const std::string& profile, const std::string& a_dtype,
               const std::string& w_dtype, const std::string& o_dtype) {
  return { {"op_kind", "gemm"},
           {"tensor_shapes", {{"m", 17}, {"n", 65}, {"k", 65}}},
           {"storage_dtype", {{"activations", a_dtype},
                              {"weights", w_dtype},
                              {"output", o_dtype}}},
           {"execution_profile", profile},
           {"layer_instances", 2},
           {"weight_passes", 3} };
}

struct Accounting {
  bool int4_path = false;
  bool wr = false;
  u64 M = 0, N = 0, K = 0, tm = 0, tn = 0, tk = 0;
  u64 cores = 0, executions = 0;
  std::string a_dtype, w_dtype, o_dtype;
  long double p_pe = 0, p_vec = 0, eta_pe = 0, eta_vec = 0;
  long double a_coeff = 0, w_coeff = 0, acc_coeff = 0, output_coeff = 0;
  long double expected_pe_ops = 0, expected_vec_ops = 0;
  long double expected_gate_seconds = 0;
  u64 expected_rounds = 0, expected_waves = 0, expected_active_cores = 0;
};

long double rate_pe(const Accounting& a) { return a.p_pe * a.eta_pe; }
long double rate_vec(const Accounting& a) { return a.p_vec * a.eta_vec; }

struct TileWork {
  long double pe_ops = 0;
  long double vec_ops = 0;
  long double seconds = 0;
};

// Direct tile ledger used only by this compiler-facing test. It independently
// enumerates actual M/N tile extents and applies the declared real
// conversion coefficients independently of the production ledger.
TileWork tile_work(const Accounting& a, u64 m, u64 n,
                   bool include_weight_conversion) {
  const long double mn = static_cast<long double>(m) * n;
  const long double matrix_ops = 2.0L * mn * a.K;
  const u64 k_tiles = ceildiv(a.K, a.tk);
  TileWork work;
  if (a.int4_path) {
    work.pe_ops = matrix_ops;
    work.vec_ops += mn * k_tiles * a.acc_coeff;
    work.vec_ops += mn * (k_tiles - 1);
  } else {
    work.vec_ops += matrix_ops;
  }
  work.vec_ops += static_cast<long double>(m) * a.K * a.a_coeff;
  if (include_weight_conversion)
    work.vec_ops += static_cast<long double>(n) * a.K * a.w_coeff;
  work.vec_ops += mn * a.output_coeff;
  work.seconds = work.pe_ops / rate_pe(a) + work.vec_ops / rate_vec(a);
  return work;
}

struct Expected {
  long double gate = 0;
  long double pe_ops = 0;
  long double vec_ops = 0;
  u64 output_rounds = 0;
  u64 waves = 0;
  u64 last_active_cores = 0;
};

Expected brute_budget(const Accounting& a) {
  const auto m_tiles = extents(a.M, a.tm);
  const auto n_tiles = extents(a.N, a.tn);
  const u64 k_tiles = ceildiv(a.K, a.tk);
  Expected expected;
  const u64 tile_count = static_cast<u64>(m_tiles.size() * n_tiles.size());

  // The ops ledger is summed independently from the gate schedule. In WR,
  // each resident W column tile is converted once per group, not once per M.
  long double one_call_pe_ops = 0;
  long double one_call_vec_ops = 0;
  for (const u64 m : m_tiles) {
    for (const u64 n : n_tiles) {
      const TileWork tile = tile_work(a, m, n, !a.wr);
      one_call_pe_ops += tile.pe_ops;
      one_call_vec_ops += tile.vec_ops;
    }
  }
  if (a.wr) {
    for (const u64 n : n_tiles)
      one_call_vec_ops += static_cast<long double>(n) * a.K * a.w_coeff;
  }
  expected.pe_ops = one_call_pe_ops * a.executions;
  expected.vec_ops = one_call_vec_ops * a.executions;

  if (!a.wr) {
    // OS uses row-major q=iM*Tn+iN; each at-most-C consecutive tile cohort
    // contributes its slowest Core's full-K PE+Vec serial budget.
    std::vector<long double> tile_seconds;
    tile_seconds.reserve(static_cast<std::size_t>(tile_count));
    for (const u64 m : m_tiles)
      for (const u64 n : n_tiles)
        tile_seconds.push_back(tile_work(a, m, n, true).seconds);
    for (std::size_t start = 0; start < tile_seconds.size();
         start += static_cast<std::size_t>(a.cores)) {
      const std::size_t end = std::min(
          tile_seconds.size(), start + static_cast<std::size_t>(a.cores));
      long double slowest = 0;
      for (std::size_t i = start; i < end; ++i)
        slowest = std::max(slowest, tile_seconds[i]);
      expected.gate += slowest;
      expected.last_active_cores = static_cast<u64>(end - start);
    }
    expected.output_rounds = ceildiv(tile_count, a.cores);
  } else {
    // WR groups increasing N tile IDs. Each Core retains one W tile, performs
    // its M tiles serially, and pays that W conversion once before consumption.
    const u64 groups = ceildiv(static_cast<u64>(n_tiles.size()), a.cores);
    for (u64 group = 0; group < groups; ++group) {
      const u64 start = group * a.cores;
      const u64 end = std::min<u64>(static_cast<u64>(n_tiles.size()),
                                    start + a.cores);
      long double group_slowest = 0;
      for (u64 j = start; j < end; ++j) {
        const u64 n = n_tiles.at(static_cast<std::size_t>(j));
        long double core_seconds = 0;
        for (const u64 m : m_tiles)
          core_seconds += tile_work(a, m, n, false).seconds;
        core_seconds += static_cast<long double>(n) * a.K * a.w_coeff /
                        rate_vec(a);
        group_slowest = std::max(group_slowest, core_seconds);
      }
      expected.gate += group_slowest;
      expected.last_active_cores = end - start;
    }
    expected.output_rounds = groups * static_cast<u64>(m_tiles.size());
  }
  expected.gate *= a.executions;
  expected.waves = expected.output_rounds * k_tiles;
  return expected;
}

Accounting accounting_for(const Hardware& hw, const json& task,
                          const Plan& plan) {
  Accounting a;
  const auto& shape = task.at("tensor_shapes");
  const auto& storage = task.at("storage_dtype");
  a.M = shape.at("m").get<u64>();
  a.N = shape.at("n").get<u64>();
  a.K = shape.at("k").get<u64>();
  a.tm = plan.summary.at("tm").get<u64>();
  a.tn = plan.summary.at("tn").get<u64>();
  a.tk = plan.summary.at("tk").get<u64>();
  a.cores = hw.derived.at("die").at("core_count").get<u64>();
  a.executions = task.at("layer_instances").get<u64>() *
                 task.at("weight_passes").get<u64>();
  a.a_dtype = storage.at("activations").get<std::string>();
  a.w_dtype = storage.at("weights").get<std::string>();
  a.o_dtype = storage.at("output").get<std::string>();
  a.int4_path = task.at("execution_profile") ==
                "int4_matrix_vec_fp32_assumed_v1";
  a.wr = plan.summary.at("dataflow_id") == "weight_resident_v2";
  a.p_pe = hw.derived.at("core").at("P_int4_per_core_ops_per_s").get<double>();
  a.p_vec = hw.derived.at("core").at("P_vec_per_core_ops_per_s").get<double>();
  a.eta_pe = hw.input.at("performance_profile").at("eta_pe").get<double>();
  a.eta_vec = hw.input.at("performance_profile").at("eta_vec").get<double>();
  const double quant = hw.input.at("performance_profile").value("kappa_quant_eqops", 4.0);
  const double accumulator = hw.input.at("performance_profile").value("kappa_acc_convert_eqops", 2.0);
  const std::string execution_dtype = a.int4_path ? "int4" : "fp32";
  if (a.a_dtype != execution_dtype)
    a.a_coeff = a.int4_path || a.a_dtype == "int4" ? quant : 1;
  if (a.w_dtype != execution_dtype)
    a.w_coeff = a.int4_path || a.w_dtype == "int4" ? quant : 1;
  if (a.int4_path) a.acc_coeff = accumulator;
  if (a.o_dtype != "fp32")
    a.output_coeff = a.o_dtype == "int4" ? quant : 1;
  return a;
}

void verify_compiled_candidate(const Hardware& hw, const json& task,
                               const Plan& plan) {
  const Accounting accounting = accounting_for(hw, task, plan);
  const Expected expected = brute_budget(accounting);
  const std::string label = task.at("execution_profile").get<std::string>() +
      " M=" + std::to_string(accounting.M) + " N=" +
      std::to_string(accounting.N) + " K=" + std::to_string(accounting.K) +
      " C=" + std::to_string(accounting.cores) + " tm=" +
      std::to_string(accounting.tm) + " tn=" +
      std::to_string(accounting.tn) + " tk=" +
      std::to_string(accounting.tk) + " flow=" +
      (accounting.wr ? "WR" : "OS");

  long double gate_work = 0;
  u128 activity_pe_ops = 0;
  long double activity_vec_ops = 0;
  // Activity work values are already the aggregate quantities distributed by
  // append_epoch. repeat_count is audit metadata and must not be multiplied.
  for (const auto& epoch : plan.epochs) {
    for (const auto& activity : epoch.activities) {
      if (activity.role != Role::Compute) continue;
      gate_work += activity.work[COMPUTE_GATE];
      activity_pe_ops += activity.int4_ops;
      activity_vec_ops += planned_vec_eqops(activity);
    }
  }
  const auto close = [](long double actual, long double wanted) {
    const long double scale = std::max({1.0L, std::fabs(actual), std::fabs(wanted)});
    return std::isfinite(actual) && std::fabs(actual - wanted) <= scale * 2e-10L;
  };
  if (!close(gate_work, expected.gate)) {
    std::ostringstream message;
    message << label << " cumulative COMPUTE_GATE work got="
            << static_cast<double>(gate_work) << " expected="
            << static_cast<double>(expected.gate);
    throw std::runtime_error(message.str());
  }
  if (activity_pe_ops != static_cast<u128>(expected.pe_ops) ||
      activity_vec_ops != expected.vec_ops ||
      plan.summary.at("executed_int4_ops").get<u64>() !=
          static_cast<u64>(expected.pe_ops) ||
      plan.summary.at("vec_equivalent_ops").get<long double>() != expected.vec_ops) {
    throw std::runtime_error(label + " op ledger changed with gate budget update");
  }
  const u128 matrix_ops = static_cast<u128>(2) * accounting.M *
      accounting.N * accounting.K * accounting.executions;
  require(plan.summary.at("source_matrix_ops").get<u64>() ==
              static_cast<u64>(matrix_ops) &&
              plan.summary.at("executed_matrix_ops").get<u64>() ==
              static_cast<u64>(matrix_ops),
          label + " source/executed matrix-op counters changed");
  require(plan.summary.at("output_round_count").get<u64>() ==
              expected.output_rounds,
          label + " output-round count changed");
  require(plan.summary.at("consumption_wave_count").get<u64>() ==
              expected.waves,
          label + " consumption-wave count changed");
  require(plan.summary.at("active_cores").get<u64>() ==
              std::min(accounting.cores,
                       accounting.wr
                           ? static_cast<u64>(extents(accounting.N, accounting.tn).size())
                           : static_cast<u64>(extents(accounting.M, accounting.tm).size() *
                                              extents(accounting.N, accounting.tn).size())),
          label + " active-core summary changed");
}

void compiled_gemm_tail_budgets_match_independent_cohorts() {
  u64 candidates_checked = 0;
  bool saw_os = false, saw_wr = false, saw_m_tail = false;
  bool saw_n_tail = false, saw_k_tail = false, saw_fp32 = false;
  for (const u64 cores : {1ULL, 2ULL, 4ULL}) {
    for (const bool fractional : {false,true}) {
      auto config = hardware_with_cores(cores).input;
      if (fractional) {
        config["performance_profile"]["kappa_quant_eqops"] = 1.25;
        config["performance_profile"]["kappa_acc_convert_eqops"] = 0.75;
      }
      const Hardware hw = resolve_hardware(config);
      for (const json& task : {
               gemm_task("int4_matrix_vec_fp32_assumed_v1", "bf16", "bf16", "int4"),
               gemm_task("fp32_vec_matrix_assumed_v1", "bf16", "int4", "int4")}) {
        const auto plans = detail::enumerate_mappings(hw, task);
        require(!plans.empty(), "GEMM enumeration produced no compiled candidates");
        for (const auto& plan : plans) {
          verify_compiled_candidate(hw, task, *plan);
          const u64 tm = plan->summary.at("tm").get<u64>();
          const u64 tn = plan->summary.at("tn").get<u64>();
          const u64 tk = plan->summary.at("tk").get<u64>();
          saw_os = saw_os || plan->summary.at("dataflow_id") == "output_stationary_v2";
          saw_wr = saw_wr || plan->summary.at("dataflow_id") == "weight_resident_v2";
          saw_m_tail = saw_m_tail || 17 % tm != 0;
          saw_n_tail = saw_n_tail || 65 % tn != 0;
          saw_k_tail = saw_k_tail || 65 % tk != 0;
          saw_fp32 = saw_fp32 || task.at("execution_profile") ==
                                     "fp32_vec_matrix_assumed_v1";
          ++candidates_checked;
        }
      }
    }
  }
  require(saw_os && saw_wr && saw_m_tail && saw_n_tail && saw_k_tail && saw_fp32,
          "compiled test set must exercise OS/WR, all tails and strict FP32");
  require(candidates_checked > 100,
          "compiled oracle should cover the bounded OS/WR tile candidate set");
  std::cout << "compiled GEMM tail-budget candidates checked="
            << candidates_checked
            << " with 1/2/4 Cores, OS/WR, M/N/K tails, repeats and strict FP32\n";
}

}  // namespace

int main() {
  return run_test("gemm_tail_budget_test",
                  compiled_gemm_tail_budgets_match_independent_cohorts);
}
