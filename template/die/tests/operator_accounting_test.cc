#include "test_support.hh"

#include <array>
#include <cstdint>

using namespace coarse_die;
using namespace coarse_die_test;

namespace {
using u64 = std::uint64_t;

// Check the reporting contract against the actual immutable work submitted to
// the runtime, rather than accepting a plausible standalone latency alone.
void conserved(const Hardware& hw, const Plan& plan) {
  u64 reads = 0, writes = 0, int4 = 0, vec = 0;
  for (const auto& epoch : plan.epochs) {
    for (const auto& activity : epoch.activities) {
      for (const auto& row : activity.hbm_bytes) {
        reads += row[0];
        writes += row[1];
      }
      int4 += activity.int4_ops;
      vec += activity.vec_eqops;
    }
  }
  const auto kind = plan.summary.at("kind").get<std::string>();
  require(reads == plan.summary.at("hbm_read_bytes_total").get<u64>(),
          kind + " summary reads must equal scheduled payload");
  require(writes == plan.summary.at("hbm_write_bytes_total").get<u64>(),
          kind + " summary writes must equal scheduled payload");
  require(int4 == plan.summary.at("executed_int4_ops").get<u64>(),
          kind + " summary INT4 ops must equal scheduled work");
  require(vec == plan.summary.at("vec_equivalent_ops").get<u64>(),
          kind + " summary Vec eqops must equal scheduled work");
  const auto estimate = estimate_alone(hw, plan);
  const auto& served = estimate.at("execution").at("served");
  const auto& payload = served.at("hbm_bytes_by_row_direction");
  require_close(payload[0][0].get<double>() + payload[1][0].get<double>(),
                static_cast<double>(reads), 1e-6, kind + " completed read service");
  require_close(payload[0][1].get<double>() + payload[1][1].get<double>(),
                static_cast<double>(writes), 1e-6, kind + " completed write service");
}

void materialized_ffn_writes_each_intermediate_once() {
  const auto hw = example_hardware();
  json task{{"op_kind", "ffn_region"},
            {"tensor_shapes", {{"m", 3}, {"hidden", 8}, {"ffn", 16},
                               {"mlp_variant", "swiglu"}}},
            {"storage_dtype", {{"activations", "bf16"}, {"weights", "bf16"},
                                {"output", "fp32"}}},
            {"fusion_policy", "materialized"}};
  const auto plan = compile(hw, task);
  // up, gate, and activation each write one BF16 3x16 intermediate;
  // down writes the FP32 3x8 output. Repetition and output tiling cannot
  // legitimately add extra intermediate writes to this materialized graph.
  require(plan->summary.at("hbm_write_bytes_total").get<u64>() ==
              3 * 3 * 16 * 2 + 3 * 8 * 4,
          "materialized SwiGLU intermediates use activation dtype and write once");
  conserved(hw, *plan);
}

void all_smoke_operator_reports_match_actual_work() {
  const auto hw = example_hardware();
  const auto tasks = read_json(std::string(COARSE_DIE_SOURCE_DIR) +
                               "/configs/example_tasks.json");
  for (const auto& task : tasks.at("specs")) conserved(hw, *compile(hw, task));
}

void rope_work_is_executed_and_none_is_preserved() {
  const auto hw = example_hardware();
  json task{{"op_kind", "qkv_rope"},
            {"tensor_shapes", {{"m", 3}, {"hidden", 8}, {"q_heads", 4},
                               {"kv_heads", 2}, {"head_dim", 2},
                               {"position_encoding", "none"}}},
            {"storage_dtype", "bf16"}};
  const auto projection = compile(hw, task);
  require(projection->summary.at("rope_eqops").get<u64>() == 0,
          "position_encoding=none does not charge RoPE");
  task["tensor_shapes"]["position_encoding"] = "rope";
  const auto rope = compile(hw, task);
  const u64 expected = 3 * 3 * (4 + 2) * 2;
  require(rope->summary.at("rope_eqops").get<u64>() == expected,
          "RoPE transforms Q and K, preserving V");
  require(rope->summary.at("vec_equivalent_ops").get<u64>() ==
              projection->summary.at("vec_equivalent_ops").get<u64>() + expected,
          "RoPE eqops enter the scheduled Vec work rather than summary alone");
  require(rope->summary.at("primitive_kernels").size() == 2 &&
              rope->summary.at("primitive_kernels")[1].at("template_id") ==
                  "rope_materialized_v2",
          "materialized QKV/RoPE reports its two serial primitives");
  require(rope->summary.at("logical_tensor_bytes").get<u64>() ==
              projection->summary.at("logical_tensor_bytes").get<u64>() + 2 * 3 * 2 * 2,
          "QKV/RoPE logical boundary adds cosine/sine once, excluding internal Q/K transfers");
  conserved(hw, *projection);
  conserved(hw, *rope);
}

void test_accounting() {
  materialized_ffn_writes_each_intermediate_once();
  rope_work_is_executed_and_none_is_preserved();
  all_smoke_operator_reports_match_actual_work();
}
}  // namespace

int main() { return run_test("operator_accounting_test", test_accounting); }
