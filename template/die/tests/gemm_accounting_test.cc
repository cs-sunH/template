#include "test_support.hh"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <map>
#include <set>
#include <sstream>
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
using u64 = std::uint64_t;

u64 ceildiv(u64 value, u64 divisor) {
  return value / divisor + (value % divisor != 0);
}

u64 bytes(u64 elements, const std::string& dtype) {
  if (dtype == "int4") return elements / 2 + (elements % 2 != 0);
  if (dtype == "bf16") return elements * 2;
  if (dtype == "fp32") return elements * 4;
  throw std::runtime_error("oracle received unsupported dtype " + dtype);
}

std::vector<u64> tile_extents(u64 length, u64 tile) {
  std::vector<u64> result;
  for (u64 offset = 0; offset < length; offset += tile)
    result.push_back(std::min(tile, length - offset));
  return result;
}

std::string case_label(const json& task) {
  const auto& shape = task.at("tensor_shapes");
  std::ostringstream out;
  out << task.at("execution_profile").get<std::string>() << " M="
      << shape.at("m").get<u64>() << " N=" << shape.at("n").get<u64>()
      << " K=" << shape.at("k").get<u64>() << " layers="
      << task.at("layer_instances").get<u64>() << " passes="
      << task.at("weight_passes").get<u64>();
  return out.str();
}

std::string candidate_label(const json& task, const Plan& plan) {
  std::ostringstream out;
  out << case_label(task) << " dataflow="
      << plan.summary.at("dataflow_id").get<std::string>() << " tm="
      << plan.summary.at("tm").get<u64>() << " tn="
      << plan.summary.at("tn").get<u64>() << " tk="
      << plan.summary.at("tk").get<u64>();
  return out.str();
}

void check_equal(const Plan& plan, const std::string& context,
                 const char* field, u64 actual, u64 expected) {
  if (actual != expected) {
    std::ostringstream out;
    out << context << " " << field << " got=" << actual
        << " expected=" << expected;
    throw std::runtime_error(out.str());
  }
}

u64 summary_u64(const Plan& plan, const char* key) {
  if (!plan.summary.contains(key))
    throw std::runtime_error(std::string("candidate summary missing ") + key);
  return plan.summary.at(key).get<u64>();
}

struct Oracle {
  u64 hbm_a = 0;
  u64 hbm_w = 0;
  u64 hbm_output = 0;
  u64 a_exec_loads = 0;
  u64 w_exec_loads = 0;
  u64 a_int4_loads = 0;
  u64 w_int4_loads = 0;
  u64 a_loaded_elements = 0;
  u64 w_loaded_elements = 0;
  u64 a_scale_tiles = 0;
  u64 w_scale_tiles = 0;
  u64 output_scale_tiles = 0;
  u64 t_m = 0;
  u64 t_n = 0;
  u64 t_k = 0;
};

Oracle enumerate_physical_tiles(const json& task, const Plan& plan) {
  const auto& shape = task.at("tensor_shapes");
  const auto& storage = task.at("storage_dtype");
  const u64 m = shape.at("m").get<u64>();
  const u64 n = shape.at("n").get<u64>();
  const u64 k = shape.at("k").get<u64>();
  const u64 tm = summary_u64(plan, "tm");
  const u64 tn = summary_u64(plan, "tn");
  const u64 tk = summary_u64(plan, "tk");
  const std::string ad = storage.at("activations").get<std::string>();
  const std::string wd = storage.at("weights").get<std::string>();
  const std::string od = storage.at("output").get<std::string>();
  const std::string profile = task.at("execution_profile").get<std::string>();
  const bool int4_path = profile == "int4_matrix_vec_fp32_assumed_v1";
  const std::string execution_dtype = int4_path ? "int4" : "fp32";
  const bool convert_a = ad != execution_dtype;
  const bool convert_w = wd != execution_dtype;
  const bool wr = plan.summary.at("dataflow_id") == "weight_resident_v2";
  const auto m_tiles = tile_extents(m, tm);
  const auto n_tiles = tile_extents(n, tn);
  const auto k_tiles = tile_extents(k, tk);
  Oracle totals;
  totals.t_m = m_tiles.size();
  totals.t_n = n_tiles.size();
  totals.t_k = k_tiles.size();

  // Enumerate every physical A tile load in the output-column loop. Each N
  // tile rereads A; each tile is packed independently, including odd tails.
  for (const u64 m_extent : m_tiles) {
    for (const u64 n_extent : n_tiles) {
      for (const u64 k_extent : k_tiles) {
        const u64 elements = m_extent * k_extent;
        totals.hbm_a += bytes(elements, ad);
        totals.a_exec_loads += bytes(elements, execution_dtype);
        totals.a_int4_loads += bytes(elements, "int4");
        totals.a_loaded_elements += elements;
        if (int4_path && convert_a) ++totals.a_scale_tiles;
      }
      totals.hbm_output += bytes(m_extent * n_extent, od);
      if (od == "int4") ++totals.output_scale_tiles;
    }
  }

  // OS reloads W for each M tile. WR loads a column group once, then consumes
  // that resident tile for each M tile. The loop intentionally models that
  // reuse independently from the Mapping.cc aggregate equations.
  const u64 weight_load_groups = wr ? 1 : static_cast<u64>(m_tiles.size());
  for (u64 group = 0; group < weight_load_groups; ++group) {
    for (const u64 n_extent : n_tiles) {
      for (const u64 k_extent : k_tiles) {
        const u64 elements = k_extent * n_extent;
        totals.hbm_w += bytes(elements, wd);
        totals.w_exec_loads += bytes(elements, execution_dtype);
        totals.w_int4_loads += bytes(elements, "int4");
        totals.w_loaded_elements += elements;
        if (int4_path && convert_w) ++totals.w_scale_tiles;
      }
    }
  }
  return totals;
}

struct Expected {
  u64 hbm_read = 0;
  u64 hbm_write = 0;
  u64 logical = 0;
  u64 int4_input = 0;
  u64 source_ops = 0;
  u64 int4_ops = 0;
  u64 conversion_elements = 0;
  u64 vec_ops = 0;
  u64 sram_read = 0;
  u64 sram_write = 0;
  u64 local = 0;
  u64 rounds = 0;
  u64 waves = 0;
  u64 active_cores = 0;
};

Expected independently_account(const json& task, const Hardware& hardware,
                               const Plan& plan, const Oracle& physical) {
  const auto& shape = task.at("tensor_shapes");
  const auto& storage = task.at("storage_dtype");
  const u64 m = shape.at("m").get<u64>();
  const u64 n = shape.at("n").get<u64>();
  const u64 k = shape.at("k").get<u64>();
  const u64 executions = task.at("layer_instances").get<u64>() *
                         task.at("weight_passes").get<u64>();
  const std::string ad = storage.at("activations").get<std::string>();
  const std::string wd = storage.at("weights").get<std::string>();
  const std::string od = storage.at("output").get<std::string>();
  const std::string profile = task.at("execution_profile").get<std::string>();
  const bool int4_path = profile == "int4_matrix_vec_fp32_assumed_v1";
  const std::string execution_dtype = int4_path ? "int4" : "fp32";
  const bool convert_a = ad != execution_dtype;
  const bool convert_w = wd != execution_dtype;
  const bool wr = plan.summary.at("dataflow_id") == "weight_resident_v2";
  const u64 weight_uses_after_load = wr ? physical.t_m : 1;
  const u64 cores = hardware.derived.at("die").at("core_count").get<u64>();
  const u64 tile_output_count = physical.t_m * physical.t_n;
  Expected out;
  out.hbm_read = (physical.hbm_a + physical.hbm_w) * executions;
  out.hbm_write = physical.hbm_output * executions;
  out.logical = (bytes(m * k, ad) + bytes(k * n, wd) + bytes(m * n, od)) *
                executions;
  out.int4_input = int4_path
      ? (physical.a_int4_loads + physical.w_int4_loads) * executions
      : 0;
  out.source_ops = 2 * m * n * k * executions;
  out.int4_ops = int4_path ? out.source_ops : 0;
  const u64 a_conversion_elements = convert_a
      ? physical.a_loaded_elements * executions : 0;
  const u64 w_conversion_elements = convert_w
      ? physical.w_loaded_elements * executions : 0;
  const u64 output_cast_elements = od == "fp32" ? 0 : m * n * executions;
  const u64 accumulator_elements = m * n * physical.t_k * executions;
  const u64 accumulator_adds = m * n * (physical.t_k - 1) * executions;
  if (int4_path) {
    out.conversion_elements = a_conversion_elements + w_conversion_elements +
                              accumulator_elements + output_cast_elements;
    const u64 kappa_quant = hardware.input.at("performance_profile")
                                .value("kappa_quant_eqops", 4.0);
    const u64 kappa_acc = hardware.input.at("performance_profile")
                              .value("kappa_acc_convert_eqops", 2.0);
    out.vec_ops = (a_conversion_elements + w_conversion_elements) * kappa_quant +
                  accumulator_elements * kappa_acc + accumulator_adds +
                  output_cast_elements * (od == "int4" ? kappa_quant : 1);
  } else {
    out.conversion_elements = a_conversion_elements + w_conversion_elements +
                              output_cast_elements;
    const u64 kappa_quant = hardware.input.at("performance_profile")
                                .value("kappa_quant_eqops", 4.0);
    out.vec_ops = out.source_ops +
                  a_conversion_elements * (ad == "int4" ? kappa_quant : 1) +
                  w_conversion_elements * (wd == "int4" ? kappa_quant : 1) +
                  output_cast_elements;
  }

  const u64 scale_write_bytes =
      (physical.a_scale_tiles + physical.w_scale_tiles) * 4 * executions;
  const u64 scale_read_bytes =
      (physical.a_scale_tiles + physical.w_scale_tiles * weight_uses_after_load) *
      4 * executions;
  const u64 converted_buffer_writes =
      ((convert_a ? physical.a_exec_loads : 0) +
       (convert_w ? physical.w_exec_loads : 0)) * executions +
      scale_write_bytes;
  const u64 conversion_source_reads =
      ((convert_a ? physical.hbm_a : 0) +
       (convert_w ? physical.hbm_w : 0)) * executions;
  const u64 operand_reads =
      (physical.a_exec_loads +
       physical.w_exec_loads * weight_uses_after_load) * executions;
  const u64 partial_bytes = int4_path ? accumulator_elements * 4 : 0;
  const u64 accumulator_writes = accumulator_elements * 4;
  const u64 accumulator_reads = accumulator_adds * 4;
  const u64 output_cast_reads = output_cast_elements * 4;
  const u64 output_cast_writes = output_cast_elements ? out.hbm_write : 0;
  const u64 output_scale_writes = physical.output_scale_tiles * 4 * executions;
  const u64 compute_sram_read = operand_reads + conversion_source_reads +
                                 partial_bytes + accumulator_reads +
                                 scale_read_bytes + output_cast_reads;
  const u64 compute_sram_write = converted_buffer_writes + partial_bytes +
                                 accumulator_writes + output_cast_writes +
                                 output_scale_writes;
  out.sram_read = compute_sram_read + out.hbm_write;
  out.sram_write = compute_sram_write + out.hbm_read;
  out.local = compute_sram_read + compute_sram_write;

  out.rounds = wr ? ceildiv(physical.t_n, cores) * physical.t_m
                  : ceildiv(tile_output_count, cores);
  out.waves = out.rounds * physical.t_k;
  out.active_cores = std::max<u64>(
      1, std::min(cores, wr ? physical.t_n : tile_output_count));
  return out;
}

void verify_activity_ledger(const json& task, const Hardware& hardware,
                            const Plan& plan,
                            const Expected& expected, bool verify_served) {
  const std::string context = candidate_label(task, plan);
  u64 hbm_read = 0, hbm_write = 0, int4_ops = 0, vec_ops = 0;
  double sram_read = 0.0, sram_write = 0.0, local = 0.0;
  double load_sram_write = 0.0, store_sram_read = 0.0;
  for (const auto& epoch : plan.epochs) {
    for (const auto& activity : epoch.activities) {
      for (const auto& row : activity.hbm_bytes) {
        hbm_read += row[0];
        hbm_write += row[1];
      }
      int4_ops += activity.int4_ops;
      vec_ops += activity.vec_eqops;
      sram_read += activity.work[SRAM_READ];
      sram_write += activity.work[SRAM_WRITE];
      local += activity.work[LOCAL];
      if (activity.role == Role::Load)
        load_sram_write += activity.work[SRAM_WRITE];
      if (activity.role == Role::Store)
        store_sram_read += activity.work[SRAM_READ];
    }
  }
  check_equal(plan, context, "summary HBM reads", summary_u64(plan, "hbm_read_bytes_total"), expected.hbm_read);
  check_equal(plan, context, "summary HBM writes", summary_u64(plan, "hbm_write_bytes_total"), expected.hbm_write);
  check_equal(plan, context, "Plan HBM reads", hbm_read, expected.hbm_read);
  check_equal(plan, context, "Plan HBM writes", hbm_write, expected.hbm_write);
  if (std::abs(load_sram_write - static_cast<double>(expected.hbm_read)) > 1e-6 ||
      std::abs(store_sram_read - static_cast<double>(expected.hbm_write)) > 1e-6) {
    std::ostringstream out;
    out << context << " load/store SRAM work got=" << load_sram_write << "/"
        << store_sram_read << " expected=" << expected.hbm_read << "/"
        << expected.hbm_write;
    throw std::runtime_error(out.str());
  }
  check_equal(plan, context, "summary logical bytes", summary_u64(plan, "logical_tensor_bytes"), expected.logical);
  check_equal(plan, context, "summary packed INT4 inputs", summary_u64(plan, "int4_input_bytes"), expected.int4_input);
  check_equal(plan, context, "summary source matrix ops", summary_u64(plan, "source_matrix_ops"), expected.source_ops);
  check_equal(plan, context, "summary executed matrix ops", summary_u64(plan, "executed_matrix_ops"), expected.source_ops);
  check_equal(plan, context, "summary INT4 ops", summary_u64(plan, "executed_int4_ops"), expected.int4_ops);
  check_equal(plan, context, "Plan INT4 ops", int4_ops, expected.int4_ops);
  check_equal(plan, context, "summary conversion elements", summary_u64(plan, "conversion_elements"), expected.conversion_elements);
  check_equal(plan, context, "summary Vec ops", summary_u64(plan, "vec_equivalent_ops"), expected.vec_ops);
  check_equal(plan, context, "Plan Vec ops", vec_ops, expected.vec_ops);
  check_equal(plan, context, "summary SRAM reads", summary_u64(plan, "sram_read_bytes"), expected.sram_read);
  check_equal(plan, context, "summary SRAM writes", summary_u64(plan, "sram_write_bytes"), expected.sram_write);
  check_equal(plan, context, "summary LOCAL bytes", summary_u64(plan, "local_bus_bytes"), expected.local);
  if (std::abs(sram_read - static_cast<double>(expected.sram_read)) > 1e-6 ||
      std::abs(sram_write - static_cast<double>(expected.sram_write)) > 1e-6 ||
      std::abs(local - static_cast<double>(expected.local)) > 1e-6) {
    std::ostringstream out;
    out << context << " Plan SRAM/LOCAL got=" << sram_read << "/"
        << sram_write << "/" << local << " expected=" << expected.sram_read
        << "/" << expected.sram_write << "/" << expected.local;
    throw std::runtime_error(out.str());
  }
  check_equal(plan, context, "output rounds", summary_u64(plan, "output_round_count"), expected.rounds);
  check_equal(plan, context, "consumption waves", summary_u64(plan, "consumption_wave_count"), expected.waves);
  check_equal(plan, context, "active cores", summary_u64(plan, "active_cores"), expected.active_cores);

  if (!verify_served) return;
  const auto estimate = estimate_alone(hardware, plan);
  const auto instances=plan.task.at("layer_instances").get<u64>()*
      plan.task.at("weight_passes").get<u64>();
  const bool incomplete=instances>1;
  if(estimate.at("status")!=(incomplete ? "not_available" : "ok_assumed"))
    throw std::runtime_error(context + " incorrect completeness of fixed region overhead timing");
  if(incomplete)require(plan.summary.at("fixed_region_overheads").at("modeled_instances")==0&&
      plan.summary.at("fixed_region_overheads").at("instances")==instances,
      "repeated GEMM body accounting must disclose missing instance startup windows");
  const auto& served = estimate.at("execution").at("served");
  const auto& served_hbm = served.at("hbm_bytes_by_row_direction");
  const double served_reads = served_hbm[0][0].get<double>() +
                              served_hbm[1][0].get<double>();
  const double served_writes = served_hbm[0][1].get<double>() +
                               served_hbm[1][1].get<double>();
  if (served_reads != static_cast<double>(expected.hbm_read) ||
      served_writes != static_cast<double>(expected.hbm_write) ||
      served.at("int4_ops").get<double>() != static_cast<double>(expected.int4_ops) ||
      served.at("vec_eqops").get<double>() != static_cast<double>(expected.vec_ops)) {
    std::ostringstream out;
    out << context << " estimate served reads/writes/INT4/Vec=" << served_reads
        << "/" << served_writes << "/" << served.at("int4_ops") << "/"
        << served.at("vec_eqops") << " expected=" << expected.hbm_read << "/"
        << expected.hbm_write << "/" << expected.int4_ops << "/"
        << expected.vec_ops;
    throw std::runtime_error(out.str());
  }
}

json gemm_task(u64 m, u64 n, u64 k, const std::string& activations,
               const std::string& weights, const std::string& output,
               const std::string& profile, u64 layers, u64 passes) {
  return json{{"op_kind", "gemm"},
              {"tensor_shapes", {{"m", m}, {"n", n}, {"k", k}}},
              {"storage_dtype", {{"activations", activations},
                                 {"weights", weights}, {"output", output}}},
              {"execution_profile", profile},
              {"layer_instances", layers}, {"weight_passes", passes}};
}

void check_candidate_set(const Hardware& hardware, const json& task,
                         bool verify_estimates, bool require_odd_packing) {
  const auto candidates = detail::enumerate_mappings(hardware, task);
  require(!candidates.empty(), case_label(task) + " generated no GEMM candidates");
  std::set<std::string> dataflows;
  std::map<std::tuple<u64, u64, u64>, std::array<u64, 2>> weight_reads_by_tile;
  std::array<bool, 2> verified_estimate{{false, false}};
  bool saw_odd_int4_tile_overhead = false;
  for (const auto& candidate : candidates) {
    const auto& summary = candidate->summary;
    const std::string dataflow = summary.at("dataflow_id").get<std::string>();
    if (dataflow != "output_stationary_v2" && dataflow != "weight_resident_v2")
      throw std::runtime_error(case_label(task) + " unexpected GEMM dataflow " + dataflow);
    const bool wr = dataflow == "weight_resident_v2";
    const u64 tm = summary.at("tm").get<u64>();
    const u64 tn = summary.at("tn").get<u64>();
    const u64 tk = summary.at("tk").get<u64>();
    const Oracle physical = enumerate_physical_tiles(task, *candidate);
    const Expected expected = independently_account(task, hardware, *candidate, physical);
    const double sram_limit =
        hardware.input.at("performance_profile").at("rho_sram").get<double>() *
        hardware.derived.at("core").at("sram_bytes_per_core").get<double>();
    if (static_cast<double>(summary.at("sram_peak_bytes_per_core").get<u64>()) >
        sram_limit + 1e-6) {
      throw std::runtime_error(candidate_label(task, *candidate) +
                               " candidate exceeds its declared SRAM capacity");
    }
    verify_activity_ledger(task, hardware, *candidate, expected,
                           verify_estimates && !verified_estimate[wr ? 1 : 0]);
    if (verify_estimates) verified_estimate[wr ? 1 : 0] = true;
    dataflows.insert(dataflow);
    auto& pair = weight_reads_by_tile[std::make_tuple(tm, tn, tk)];
    pair[wr ? 1 : 0] = physical.hbm_w;
    if (task.at("storage_dtype").at("activations") == "int4" &&
        physical.hbm_a > bytes(task.at("tensor_shapes").at("m").get<u64>() *
                                   task.at("tensor_shapes").at("k").get<u64>(),
                               "int4") * physical.t_n)
      saw_odd_int4_tile_overhead = true;
  }
  require(dataflows.count("output_stationary_v2") != 0,
          case_label(task) + " candidate enumeration missed OS");
  require(dataflows.count("weight_resident_v2") != 0,
          case_label(task) + " candidate enumeration missed WR");
  if (verify_estimates)
    require(verified_estimate[0] && verified_estimate[1],
            case_label(task) + " did not estimate a candidate from each dataflow");
  if (require_odd_packing)
    require(saw_odd_int4_tile_overhead,
            case_label(task) + " did not exercise per-tile odd INT4 packing overhead");

  bool saw_resident_weight_saving = false;
  const u64 m = task.at("tensor_shapes").at("m").get<u64>();
  for (const auto& item : weight_reads_by_tile) {
    const u64 tm = std::get<0>(item.first);
    const auto& loads = item.second;
    if (ceildiv(m, tm) > 1 && loads[0] && loads[1] && loads[1] < loads[0])
      saw_resident_weight_saving = true;
  }
  require(saw_resident_weight_saving,
          case_label(task) + " did not show WR weight reuse for a matched tile");
}

void gemm_accounting() {
  const auto hardware = example_hardware();
  check_candidate_set(
      hardware,
      gemm_task(17, 33, 65, "bf16", "bf16", "bf16",
                "int4_matrix_vec_fp32_assumed_v1", 2, 3),
      true, false);
  check_candidate_set(
      hardware,
      gemm_task(3, 5, 7, "int4", "int4", "int4",
                "int4_matrix_vec_fp32_assumed_v1", 1, 2),
      true, true);
  check_candidate_set(
      hardware,
      gemm_task(17, 33, 65, "bf16", "bf16", "bf16",
                "fp32_vec_matrix_assumed_v1", 2, 3),
      true, false);
  check_candidate_set(
      hardware,
      gemm_task(3, 5, 7, "int4", "int4", "bf16",
                "fp32_vec_matrix_assumed_v1", 1, 2),
      true, false);
  check_candidate_set(
      hardware,
      gemm_task(257, 129, 65, "bf16", "bf16", "bf16",
                "int4_matrix_vec_fp32_assumed_v1", 1, 1),
      false, false);
}

}  // namespace

int main() {
  return run_test("gemm_accounting_test", gemm_accounting);
}
