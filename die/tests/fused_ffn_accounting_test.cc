#include "test_support.hh"

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

using namespace coarse_die;
using namespace coarse_die_test;

namespace {
using u64 = std::uint64_t;

u64 ceildiv(u64 value, u64 divisor) {
  return value / divisor + (value % divisor != 0);
}

u64 bytes(u64 elements, const std::string& dtype) {
  if (dtype == "int4") return elements / 2 + (elements % 2 != 0);
  return elements * (dtype == "bf16" ? 2 : 4);
}

u64 tile_bytes(u64 rows, u64 cols, const std::string& dtype) {
  return bytes(rows * cols, dtype);
}

u64 tile_grid_bytes(u64 rows, u64 tile_rows, u64 cols, u64 tile_cols,
                    const std::string& dtype) {
  const u64 row_tiles = ceildiv(rows, tile_rows);
  const u64 col_tiles = ceildiv(cols, tile_cols);
  u64 result = 0;
  for (u64 r = 0; r < row_tiles; ++r) {
    const u64 current_rows = std::min(tile_rows, rows - r * tile_rows);
    for (u64 c = 0; c < col_tiles; ++c) {
      const u64 current_cols = std::min(tile_cols, cols - c * tile_cols);
      result += tile_bytes(current_rows, current_cols, dtype);
    }
  }
  return result;
}

json fused_task(u64 m, u64 hidden, u64 ffn, const std::string& activations,
                const std::string& weights, u64 layers, u64 passes,
                const std::string& profile =
                    "int4_matrix_vec_fp32_assumed_v1") {
  return json{{"op_kind", "ffn_region"},
              {"tensor_shapes", {{"m", m},
                                 {"hidden", hidden},
                                 {"ffn", ffn},
                                 {"mlp_variant", "swiglu"}}},
              {"storage_dtype", {{"activations", activations},
                                 {"weights", weights},
                                 {"output", "bf16"}}},
              {"execution_profile", profile},
              {"fusion_policy", "fused"},
              {"layer_instances", layers},
              {"weight_passes", passes}};
}

u64 live_peak_for_tk(const json& task, u64 tm, u64 tf, u64 th, u64 tk) {
  const auto& shape = task.at("tensor_shapes");
  const auto& types = task.at("storage_dtype");
  const u64 m = shape.at("m").get<u64>();
  const u64 h = shape.at("hidden").get<u64>();
  const u64 f = shape.at("ffn").get<u64>();
  const auto ad = types.at("activations").get<std::string>();
  const auto wd = types.at("weights").get<std::string>();
  const auto od = types.at("output").get<std::string>();
  const bool int4_path =
      task.at("execution_profile") == "int4_matrix_vec_fp32_assumed_v1";
  const bool fp32_path =
      task.at("execution_profile") == "fp32_vec_matrix_assumed_v1";
  auto live_tile = [&](u64 rows, u64 cols, const std::string& dtype) {
    const u64 raw = tile_bytes(rows, cols, dtype);
    if (int4_path && dtype != "int4")
      return raw + tile_bytes(rows, cols, "int4") + 4;
    if (fp32_path && dtype != "fp32")
      return raw + tile_bytes(rows, cols, "fp32");
    return raw;
  };
  const u64 max_m = std::min(m, tm), max_f = std::min(f, tf);
  const u64 max_h = std::min(h, th), max_k = std::min(h, tk);
  const u64 input_slots = 2 * live_tile(max_m, max_k, ad);
  const u64 up_gate_weights = 4 * live_tile(max_k, max_f, wd);
  const u64 down_weights = 2 * live_tile(max_f, max_h, wd);
  const u64 up_gate_fp32 = 2 * max_m * max_f * 4;
  const u64 up_gate_partial = int4_path ? up_gate_fp32 : 0;
  const u64 activation = int4_path
      ? (ad == "int4" ? tile_bytes(max_m, max_f, "int4") + 4
                      : live_tile(max_m, max_f, ad))
      : (fp32_path ? tile_bytes(max_m, max_f, "fp32")
                   : tile_bytes(max_m, max_f, ad));
  const u64 down_fp32 = max_m * max_h * 4;
  const u64 down_partial = int4_path ? down_fp32 : 0;
  const u64 pending_output = od == "fp32"
      ? 2 * tile_bytes(max_m, max_h, od)
      : 2 * (tile_bytes(max_m, max_h, od) + (od == "int4" ? 4 : 0));
  return input_slots + up_gate_weights + down_weights + up_gate_fp32 +
         up_gate_partial + activation + down_fp32 + down_partial +
         pending_output + 64;
}

void verify_selected_tk_is_first_feasible(const Hardware& hw,
                                          const json& task,
                                          const Plan& plan) {
  const auto& summary = plan.summary;
  const u64 h = task.at("tensor_shapes").at("hidden").get<u64>();
  const u64 tm = summary.at("tm").get<u64>();
  const u64 tf = summary.at("tf").get<u64>();
  const u64 th = summary.at("th").get<u64>();
  const u64 selected_tk = summary.at("tk").get<u64>();
  std::vector<u64> order;
  for (const u64 option : {64, 256, 1024, 4096}) {
    const u64 clipped = std::min(h, option);
    if (std::find(order.begin(), order.end(), clipped) == order.end())
      order.push_back(clipped);
  }
  std::sort(order.begin(), order.end(), std::greater<u64>());
  const u64 sram = summary.at("sram_capacity_bytes_per_core").get<u64>();
  const long double capacity =
      hw.input.at("performance_profile").at("rho_sram").get<double>() * sram;
  std::vector<u64> probes;
  u64 first_feasible = 0;
  for (const u64 candidate : order) {
    probes.push_back(candidate);
    if (static_cast<long double>(
            live_peak_for_tk(task, tm, tf, th, candidate)) <= capacity) {
      first_feasible = candidate;
      break;
    }
  }
  require(first_feasible == selected_tk,
          "selected tk must be the first capacity-feasible descending option");
  require(summary.at("fused_tk_capacity_probe_order").get<std::vector<u64>>() ==
              probes,
          "candidate summary preserves the descending capacity probe prefix");
  require(summary.at("sram_peak_bytes_per_core").get<u64>() ==
              live_peak_for_tk(task, tm, tf, th, selected_tk),
          "SRAM peak reserves ping-pong operands, two outputs, and metadata");
  const auto& components =
      summary.at("fused_sram_peak_components_bytes_per_core");
  const auto& types = task.at("storage_dtype");
  const auto ad = types.at("activations").get<std::string>();
  const auto wd = types.at("weights").get<std::string>();
  const auto od = types.at("output").get<std::string>();
  const bool int4_path =
      task.at("execution_profile") == "int4_matrix_vec_fp32_assumed_v1";
  const bool fp32_path =
      task.at("execution_profile") == "fp32_vec_matrix_assumed_v1";
  auto live_tile = [&](u64 rows, u64 cols, const std::string& dtype) {
    const u64 raw = tile_bytes(rows, cols, dtype);
    if (int4_path && dtype != "int4")
      return raw + tile_bytes(rows, cols, "int4") + 4;
    if (fp32_path && dtype != "fp32")
      return raw + tile_bytes(rows, cols, "fp32");
    return raw;
  };
  const u64 max_m = std::min<u64>(
      task.at("tensor_shapes").at("m").get<u64>(), tm);
  const u64 max_f = std::min<u64>(
      task.at("tensor_shapes").at("ffn").get<u64>(), tf);
  const u64 max_h = std::min<u64>(h, th);
  require(components.at("up_gate_weight_tiles").get<u64>() ==
              4 * live_tile(std::min<u64>(h, selected_tk), max_f, wd) &&
              components.at("down_weight_tile").get<u64>() ==
                  2 * live_tile(max_f, max_h, wd),
          "peak counts current and prefetched up/gate and down weight slots");
  require(components.at("pending_output").get<u64>() ==
              2 * (tile_bytes(max_m, max_h, od) + (od == "int4" ? 4 : 0)),
          "peak reserves two pending outputs without aliasing FP32 accumulation");
}

void verify_plan_and_estimate_conservation(const Hardware& hw,
                                           const Plan& plan) {
  u64 reads = 0, writes = 0, int4_ops = 0, vec_ops = 0;
  double sram_reads = 0.0, sram_writes = 0.0, local_bytes = 0.0;
  for (const auto& epoch : plan.epochs) {
    for (const auto& activity : epoch.activities) {
      for (const auto& row : activity.hbm_bytes) {
        reads += row[0];
        writes += row[1];
      }
      int4_ops += activity.int4_ops;
      vec_ops += activity.vec_eqops;
      sram_reads += activity.work[SRAM_READ];
      sram_writes += activity.work[SRAM_WRITE];
      local_bytes += activity.work[LOCAL];
    }
  }
  require(reads == plan.summary.at("hbm_read_bytes_total").get<u64>(),
          "summary HBM reads equal plan activities");
  require(writes == plan.summary.at("hbm_write_bytes_total").get<u64>(),
          "summary HBM writes equal plan activities");
  require(int4_ops == plan.summary.at("executed_int4_ops").get<u64>(),
          "summary INT4 ops equal plan activities");
  require(vec_ops == plan.summary.at("vec_equivalent_ops").get<u64>(),
          "summary Vec ops equal plan activities");
  require(sram_reads == static_cast<double>(
                            plan.summary.at("sram_read_bytes").get<u64>()) &&
              sram_writes == static_cast<double>(
                                 plan.summary.at("sram_write_bytes").get<u64>()) &&
              local_bytes == static_cast<double>(
                                 plan.summary.at("local_bus_bytes").get<u64>()),
          "summary SRAM and LOCAL bytes include the same Plan activity work");

  const auto estimate = estimate_alone(hw, plan);
  require(estimate.at("status") == "ok_assumed",
          "compiled fused plan completes a standalone estimate");
  const auto& served =
      estimate.at("execution").at("served").at("hbm_bytes_by_row_direction");
  require(served[0][0].get<double>() + served[1][0].get<double>() ==
              static_cast<double>(reads),
          "standalone estimate serves all planned read bytes");
  require(served[0][1].get<double>() + served[1][1].get<double>() ==
              static_cast<double>(writes),
          "standalone estimate serves all planned write bytes");
  require(estimate.at("execution").at("served").at("int4_ops").get<double>() ==
              static_cast<double>(int4_ops) &&
              estimate.at("execution").at("served").at("vec_eqops").get<double>() ==
                  static_cast<double>(vec_ops),
          "standalone estimate serves all planned INT4 and Vec operations");
}

void check_swiglu_case(const Hardware& hw, const json& task) {
  const auto plan = compile(hw, task);
  const auto& summary = plan->summary;
  const auto& shape = task.at("tensor_shapes");
  const auto& types = task.at("storage_dtype");
  const u64 m = shape.at("m").get<u64>();
  const u64 h = shape.at("hidden").get<u64>();
  const u64 f = shape.at("ffn").get<u64>();
  const u64 layers = task.at("layer_instances").get<u64>();
  const u64 passes = task.at("weight_passes").get<u64>();
  const u64 executions = layers * passes;
  const bool int4_path =
      task.at("execution_profile") == "int4_matrix_vec_fp32_assumed_v1";
  const bool fp32_path =
      task.at("execution_profile") == "fp32_vec_matrix_assumed_v1";
  const u64 tm = summary.at("tm").get<u64>();
  const u64 tf = summary.at("tf").get<u64>();
  const u64 th = summary.at("th").get<u64>();
  const u64 tk = summary.at("tk").get<u64>();
  const u64 Tm = ceildiv(m, tm), Tf = ceildiv(f, tf);
  const u64 Th = ceildiv(h, th), Tk = ceildiv(h, tk);
  const u64 ncore = hw.derived.at("die").at("core_count").get<u64>();
  const u64 rounds = ceildiv(Tm * Th, ncore);
  const auto ad = types.at("activations").get<std::string>();
  const auto wd = types.at("weights").get<std::string>();
  const auto od = types.at("output").get<std::string>();
  const u64 mh = m * h, hf = h * f, mhf = m * h * f;

  require(summary.at("template_id") == "fused_output_shard_v2" &&
              summary.at("fusion_policy") == "fused",
          "explicit SwiGLU fusion selects the fused-output-shard template");
  require(summary.at("candidate_count").get<u64>() > 0 &&
              summary.at("candidate_count").get<u64>() <= 32,
          "bounded fused candidate set contains at most 32 candidates");
  require(summary.at("source_matrix_ops").get<u64>() ==
              6 * mhf * executions,
          "source SwiGLU matrix ops are 6MHF per execution");
  require(summary.at("executed_matrix_ops").get<u64>() ==
              (4 * mhf * Th + 2 * mhf) * executions,
          "executed SwiGLU matrix ops include only H-shard up/gate recompute");
  require(summary.at("source_vec_ops").get<u64>() ==
              16 * m * f * executions,
          "source Vec ops count one source activation per execution");
  require(summary.at("fused_activation_eqops").get<u64>() ==
              16 * m * f * Th * executions,
          "executed activation Vec ops scale by H output shards");
  require(summary.at("executed_int4_ops").get<u64>() ==
              (int4_path ? summary.at("executed_matrix_ops").get<u64>() : 0),
          "only the declared INT4 execution profile charges matrix ops to PE");

  const u64 expected_reads =
      (tile_grid_bytes(m, tm, h, tk, ad) * Tf * Th +
       tile_grid_bytes(h, tk, f, tf, wd) * 2 * Tm * Th +
       tile_grid_bytes(f, tf, h, th, wd) * Tm) * executions;
  const u64 expected_writes =
      tile_grid_bytes(m, tm, h, th, od) * executions;
  require(summary.at("hbm_read_bytes_total").get<u64>() == expected_reads,
          "HBM reads match exact input/up-gate/down tile payloads");
  require(summary.at("hbm_write_bytes_total").get<u64>() == expected_writes,
          "fused HBM writes contain only the final output tensor");
  require(summary.at("fused_intermediate_hbm_bytes").get<u64>() == 0,
          "fused SwiGLU has no intermediate HBM payload");
  const u64 expected_logical =
      (bytes(mh, ad) + 3 * bytes(hf, wd) + bytes(mh, od)) * executions;
  require(summary.at("logical_tensor_bytes").get<u64>() == expected_logical,
          "logical bytes include input, all three weights, and final output once");
  require(summary.at("output_round_count").get<u64>() == rounds &&
              summary.at("consumption_wave_count").get<u64>() ==
                  rounds * Tf * Tk,
          "output rounds and F/K consumption waves are distinct and conserved");
  require(summary.at("active_cores").get<u64>() ==
              std::max<u64>(1, std::min(ncore, Tm * Th)),
          "active cores are the active output cohort, not output rounds");

  const u64 input_i4 = tile_grid_bytes(m, tm, h, tk, "int4") * Tf * Th;
  const u64 upgate_i4 =
      tile_grid_bytes(h, tk, f, tf, "int4") * 2 * Tm * Th;
  const u64 down_i4 = tile_grid_bytes(f, tf, h, th, "int4") * Tm;
  const u64 activation_i4 =
      tile_grid_bytes(m, tm, f, tf, "int4") * Th;
  const u64 expected_int4_bytes = int4_path
      ? (input_i4 + upgate_i4 + down_i4 + activation_i4) * executions
      : 0;
  require(summary.at("int4_input_bytes").get<u64>() == expected_int4_bytes,
          "INT4 cache bytes count one packed load per tile, including odd tails");
  const u64 expected_operand_reads = int4_path
      ? (2 * input_i4 + upgate_i4 + down_i4 + activation_i4) * executions
      : 0;
  require(summary.at("int4_operand_read_bytes").get<u64>() ==
              expected_operand_reads,
          "INT4 PE operand reads charge both up/gate uses of cached input");
  u64 quantized_elements = 0;
  if (int4_path) {
    if (ad != "int4") quantized_elements += m * h * Tf * Th;
    quantized_elements += m * f * Th;
    if (wd != "int4")
      quantized_elements += 2 * hf * Tm * Th + hf * Tm;
    if (od == "int4") quantized_elements += mh;
  }
  u64 fp32_conversion_elements = 0;
  if (fp32_path) {
    if (ad != "fp32") fp32_conversion_elements += m * h * Tf * Th;
    if (wd != "fp32")
      fp32_conversion_elements += 2 * hf * Tm * Th + hf * Tm;
  }
  const u64 output_bf16_conversion =
      (od != "fp32" && od != "int4") ? mh * executions : 0;
  const u64 converted_elements =
      (quantized_elements + fp32_conversion_elements) * executions +
      output_bf16_conversion;
  require(summary.at("conversion_elements").get<u64>() == converted_elements,
          "conversions count loaded tensors, fused activation, and final output");

  const u64 input_scales = int4_path && ad != "int4"
      ? Tm * Tk * Tf * Th : 0;
  const u64 upgate_scales = int4_path && wd != "int4"
      ? 2 * Tm * Th * Tk * Tf : 0;
  const u64 down_scales = int4_path && wd != "int4" ? Tm * Tf * Th : 0;
  const u64 activation_scales = int4_path ? Tm * Tf * Th : 0;
  require(summary.at("fused_conversion_scale_bytes").get<u64>() ==
              (input_scales + upgate_scales + down_scales +
               activation_scales) * executions * 4,
          "every converted tile buffer carries its 4-byte scale");
  const u64 quantized_buffers =
      int4_path ? ((ad == "int4" ? 0 : input_i4) +
                   (wd == "int4" ? 0 : upgate_i4 + down_i4) + activation_i4)
                : 0;
  require(summary.at("fused_quantized_buffer_bytes").get<u64>() ==
              quantized_buffers * executions,
          "quantized SRAM buffers preserve source and activation conversion payloads");

  const u64 updates =
      (2 * m * f * Th * Tk + m * h * Tf) * executions;
  const u64 add_ops =
      (2 * m * f * Th * (Tk - 1) + m * h * (Tf - 1)) * executions;
  const u64 expected_vec = int4_path
      ? 16 * m * f * Th * executions + 4 * quantized_elements * executions +
            2 * updates + add_ops + output_bf16_conversion
      : 16 * m * f * Th * executions +
            summary.at("executed_matrix_ops").get<u64>() +
            fp32_conversion_elements * executions + output_bf16_conversion;
  require(summary.at("vec_equivalent_ops").get<u64>() == expected_vec,
          "executed Vec work includes activation, quantization, accumulators, and output conversion");
  const u64 fp32_input_bytes = fp32_path && ad != "fp32"
      ? tile_grid_bytes(m, tm, h, tk, "fp32") * Tf * Th : 0;
  const u64 fp32_upgate_bytes = fp32_path && wd != "fp32"
      ? tile_grid_bytes(h, tk, f, tf, "fp32") * 2 * Tm * Th : 0;
  const u64 fp32_down_bytes = fp32_path && wd != "fp32"
      ? tile_grid_bytes(f, tf, h, th, "fp32") * Tm : 0;
  require(summary.at("fused_fp32_conversion_buffer_bytes").get<u64>() ==
              (fp32_input_bytes + fp32_upgate_bytes + fp32_down_bytes) * executions,
          "strict FP32 Vec conversion caches are sized from exact tiled payloads");
  require(summary.at("fused_sram_peak_components_bytes_per_core").is_object(),
          "SRAM peak exposes simultaneous buffer components for audit");
  u64 component_sum = 0;
  for (const auto& item : summary.at("fused_sram_peak_components_bytes_per_core"))
    component_sum += item.get<u64>();
  require(component_sum == summary.at("sram_peak_bytes_per_core").get<u64>(),
          "SRAM peak is the sum of live inputs, weights, states, partials, scales, output, and metadata");
  verify_selected_tk_is_first_feasible(hw, task, *plan);
  verify_plan_and_estimate_conservation(hw, *plan);
}

void fused_ffn_accounting() {
  const auto hw = example_hardware();
  check_swiglu_case(hw, fused_task(3, 7, 9, "bf16", "bf16", 2, 3));
  check_swiglu_case(hw, fused_task(17, 65, 33, "int4", "int4", 1, 2));
  check_swiglu_case(hw, fused_task(17, 65, 33, "bf16", "bf16", 2, 2,
                                   "fp32_vec_matrix_assumed_v1"));
  // The template grid is 32 combinations before SRAM filtering; the selected
  // candidate also exercises descending tk capacity selection.
  check_swiglu_case(hw, fused_task(256, 1025, 1024, "bf16", "bf16", 1, 1));
}

}  // namespace

int main() {
  return run_test("fused_ffn_accounting_test", fused_ffn_accounting);
}
