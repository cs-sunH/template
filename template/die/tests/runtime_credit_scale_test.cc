#include "test_support.hh"

#include <array>
#include <cstdint>
#include <string>

using namespace coarse_die;
using namespace coarse_die_test;

namespace {

json gemm_task(std::uint64_t m, std::uint64_t n, std::uint64_t k) {
  return json{{"op_kind", "gemm"},
              {"tensor_shapes", {{"m", m}, {"n", n}, {"k", k}}},
              {"storage_dtype", {{"weights", "bf16"},
                                 {"activations", "bf16"},
                                 {"output", "bf16"}}},
              {"execution_profile", "int4_matrix_vec_fp32_assumed_v1"},
              {"layer_instances", 1},
              {"weight_passes", 1}};
}

void scaled_signed_credit_gemm_estimates_complete() {
  const auto hw = example_hardware();
  const std::array<std::array<std::uint64_t, 3>, 4> shapes{{
      {{4096, 64, 4096}},
      {{1024, 1024, 1024}},
      {{8192, 64, 1024}},
      {{64, 4096, 1024}},
  }};

  for (const auto& shape : shapes) {
    const auto plan = compile(hw, gemm_task(shape[0], shape[1], shape[2]));
    const auto estimate = estimate_alone(hw, *plan);
    require(estimate.at("status") == "ok_assumed" &&
                estimate.at("estimated_latency_ns").get<std::uint64_t>() > 0,
            "signed-credit GEMM completes through the shared Runtime engine for shape " +
                std::to_string(shape[0]) + "x" + std::to_string(shape[1]) + "x" +
                std::to_string(shape[2]));
  }
}

void large_logits_gemm_keeps_credit_feasible() {
  const auto hw = example_hardware();
  // Both vocab extents occur in the real TP=6 logits producer.
  for (const auto n : {21846ULL, 21845ULL}) {
    auto task = gemm_task(512, n, 16384);
    task["weight_passes"] = 8;
    // tm=1 candidates have 1988 interior repeats and canonical Q differs
    // by four orders of magnitude across L/C/S.
    const auto plan = compile(hw, task);
    const auto estimate = estimate_alone(hw, *plan);
    require(estimate.at("status") == "not_available" &&
                plan->summary.at("fixed_region_overheads").at("instances")==8 &&
                plan->summary.at("fixed_region_overheads").at("modeled_instances")==0 &&
                estimate.at("estimated_latency_ns").get<std::uint64_t>() > 0,
            "large logits body preserves finite credit while repeated startup timing remains unavailable");
  }
}

void singleton_matches_split_owner_and_respects_credit() {
  std::array<double,resource_count> rates{};
  rates[HBM0]=10.0;
  const auto hw=manual_hardware(rates);
  Epoch one; one.name="singleton";
  auto a=activity(1,Role::Endpoint,HBM0,100.0);
  a.hbm_bytes[0][0]=100;a.canonical_q=100.0;a.private_time_s=15.0;
  one.activities={a};
  const auto whole=estimate_alone(hw,*manual_plan(hw,{one}));
  Epoch two;two.name="same_owner_split";
  a.work[HBM0]=50.0;a.hbm_bytes[0][0]=50;a.canonical_q=50.0;
  two.activities={a};a.id=2;two.activities.push_back(a);
  const auto split=estimate_alone(hw,*manual_plan(hw,{two}));
  const auto finish=whole.at("estimated_latency_ns").get<std::uint64_t>();
  require(finish>=15000000000ULL&&finish<=15000000001ULL&&
          whole.at("estimated_latency_ns")==split.at("estimated_latency_ns"),
          "one-variable and signed-credit LP paths obey the same private bound: whole="+
          whole.at("estimated_latency_ns").dump()+", split="+split.at("estimated_latency_ns").dump());
  require(whole.at("runtime_counters").at("hbm_served_bytes_by_row_direction")==
          split.at("runtime_counters").at("hbm_served_bytes_by_row_direction"),
          "same-owner splitting preserves physical served bytes");
  Epoch blocked=one;blocked.name="empty_input_credit";
  blocked.credit_capacity={{1.0,0.0}};
  blocked.activities[0].credit_delta[0]=-1.0;
  Runtime runtime(hw);
  const auto handle=runtime.submit(manual_plan(hw,{blocked}),owner(1,1,1),0);
  require(!runtime.next_event().has_value(),
          "a sole activity cannot consume empty input credit");
  runtime.cancel(handle,0);
  require(runtime.query_progress(handle).at("served").at("hbm_bytes_by_row_direction")[0][0]==0.0,
          "blocked activity cancellation serves no payload");
}

}  // namespace

int main() {
  return run_test("runtime_credit_scale_test",
                  [] { scaled_signed_credit_gemm_estimates_complete();
                       large_logits_gemm_keeps_credit_feasible();
                       singleton_matches_split_owner_and_respects_credit(); });
}
