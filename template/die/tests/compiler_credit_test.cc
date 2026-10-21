#include "test_support.hh"

#include <cmath>
#include <cstdint>

using namespace coarse_die;
using namespace coarse_die_test;

namespace {

json gemm_task() {
  return json{{"op_kind", "gemm"},
              {"tensor_shapes", {{"m", 3}, {"n", 5}, {"k", 7}}},
              {"storage_dtype", {{"weights", "int4"},
                                 {"activations", "int4"},
                                 {"output", "bf16"}}},
              {"execution_profile", "int4_matrix_vec_fp32_assumed_v1"},
              {"layer_instances", 1},
              {"weight_passes", 1}};
}

void compiled_credit_is_signed_and_plan_remains_valid() {
  const auto hw = example_hardware();
  const auto plan = compile(hw, gemm_task());

  bool saw_negative_credit = false;
  for (const auto& epoch : plan->epochs) {
    for (const auto& activity : epoch.activities) {
      require(std::isfinite(activity.private_time_s) &&
                  activity.private_time_s >= 0.0,
              "compiled private time remains finite and nonnegative");
      for (double work : activity.work) {
        require(std::isfinite(work) && work >= 0.0,
                "compiled physical work remains finite and nonnegative");
      }
      for (double delta : activity.credit_delta) {
        require(std::isfinite(delta),
                "compiled signed credit remains finite");
        saw_negative_credit = saw_negative_credit || delta < 0.0;
      }
    }
  }
  require(saw_negative_credit,
          "actual GEMM compilation emits a negative credit consumption delta");

  const auto estimate = estimate_alone(hw, *plan);
  require(estimate.at("status") == "ok_assumed" &&
              estimate.at("estimated_latency_ns").get<std::uint64_t>() > 0,
          "compiled plan with signed credit runs through standalone estimate");

  const auto serialized = plan_to_json(*plan);
  const auto restored = plan_from_json(serialized);
  require(restored->fingerprint == plan->fingerprint &&
              plan_to_json(*restored) == serialized,
          "signed credit plan round-trips canonically");
}

}  // namespace

int main() {
  return run_test("compiler_credit_test",
                  compiled_credit_is_signed_and_plan_remains_valid);
}
