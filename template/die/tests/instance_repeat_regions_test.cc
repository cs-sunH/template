// Declared generative serial repeat on the typed instance regions — the
// generalization of the FFN materialized repeat contract (five-repo coarse
// enablement, 2026-10-18).
//
// Contract under test: a task with repeat_seams={"mode":"serial_on_drain"}
// on any typed region compiles ONE declared instance's template (per-instance
// work basis), attaches PlanRepeat{count=layer_instances*weight_passes}, and
// publishes a complete prediction (repeat_seam_modeled). The runtime
// materializes the instances serially behind real no-service seams priced
// from the per-instance region overhead; the estimate channel and a full
// runtime drain must agree bit for bit. Everything the FFN repeat path pins
// elsewhere stays pinned here for the generalized regions; the fused-FFN
// exclusion and the unsupported-op gate keep failing closed.
#include "test_support.hh"
#include "coarse_die/Execution.hh"
#include "coarse_die/Prediction.hh"

#include <cstdint>
#include <memory>
#include <vector>

using namespace coarse_die;
using namespace coarse_die_test;

namespace {

using u64 = std::uint64_t;

json repeat_task(const char* op_kind, u64 layers, u64 passes) {
  json task{{"op_kind", op_kind},
            {"layer_instances", layers},
            {"weight_passes", passes},
            {"repeat_seams", {{"mode", "serial_on_drain"}}}};
  if (std::string(op_kind) == "gemm") {
    task["tensor_shapes"] = {{"m", 7}, {"n", 5}, {"k", 9}};
  } else if (std::string(op_kind) == "attention_region") {
    task["tensor_shapes"] = {{"q_heads", 1}, {"kv_heads", 1}, {"head_dim", 8}};
    task["spans"] = json::array({{{"q_tokens", 1}, {"kv_tokens", 2}}});
    task["fusion_policy"] = "materialized";
  } else if (std::string(op_kind) == "qkv_rope") {
    task["tensor_shapes"] = {{"m", 1}, {"hidden", 8}, {"q_heads", 2},
                             {"kv_heads", 1}, {"head_dim", 4},
                             {"position_encoding", "rope"}};
  } else if (std::string(op_kind) == "mlp_activation") {
    task["tensor_shapes"] = {{"m", 1}, {"ffn", 16},
                             {"mlp_variant", "swiglu"}};
  } else if (std::string(op_kind) == "logits") {
    task["tensor_shapes"] = {{"m", 1}, {"hidden", 8}, {"vocab", 32}};
  } else if (std::string(op_kind) == "mlp_up") {
    task["tensor_shapes"] = {{"m", 1}, {"hidden", 8}, {"ffn", 16}};
  } else {
    task["tensor_shapes"] = {{"m", 1}, {"hidden", 257}};
  }
  return task;
}

u64 full_simulation_finish(const std::shared_ptr<const Hardware>& hardware,
                           std::shared_ptr<const Plan> plan) {
  Runtime runtime(hardware);
  const auto handle = runtime.submit(std::move(plan), owner(1, 1, 1), 0);
  unsigned guard = 0;
  while (!runtime.drained()) {
    const auto next = runtime.next_event();
    require(next.has_value() && *next > runtime.now_ns() && ++guard < 3000000,
            "instance-repeat fixture has a bounded advancing completion sequence");
    runtime.advance_to(*next);
  }
  const auto finish = runtime.terminal_finish_ns(handle);
  require(finish.has_value(), "instance-repeat fixture drains to a finish");
  return *finish;
}

void verify_complete_prediction_and_simulation() {
  const auto hardware =
      std::make_shared<const Hardware>(example_hardware());
  for (const char* op : {"norm", "residual", "qkv_rope", "gemm",
                         "attention_region", "o_proj", "mlp_up",
                         "mlp_activation", "logits"}) {
    const auto plan = compile(*hardware, repeat_task(op, 2, 3));
    require(plan->repeat && plan->repeat->count == 6,
            std::string(op) + ": compiled fixture declares repeat count 6");
    require(plan->summary.value("prediction_status", std::string()) ==
                "repeat_seam_modeled",
            std::string(op) + ": declared repeat publishes a complete prediction");
    require(plan->summary.at("alone_estimate_status") == "repeat_seam_modeled",
            std::string(op) + ": alone estimate status mirrors the prediction");
    require_complete_prediction(*plan);
    const u64 simulated = full_simulation_finish(hardware, plan);
    const u64 estimated = detail::estimate_latency_ns(*hardware, plan);
    require(estimated == simulated,
            std::string(op) + ": estimate channel returns exactly the simulated finish");
    const auto report = estimate_alone(*hardware, *plan);
    require(report.at("estimated_latency_ns").get<u64>() == simulated,
            std::string(op) + ": estimate_alone publishes the same finish");
  }
}

void verify_per_instance_work_basis() {
  const auto hardware =
      std::make_shared<const Hardware>(example_hardware());
  const auto repeated = compile(*hardware, repeat_task("norm", 2, 3));
  json one = repeat_task("norm", 1, 1);
  one.erase("repeat_seams");
  const auto single = compile(*hardware, one);
  // Activity work/bytes/ops describe one declared repeat instance; the
  // runtime, not the summary, owns the count multiplication.
  require(repeated->summary.value("hbm_read_bytes_total", u64{0}) ==
              single->summary.value("hbm_read_bytes_total", u64{1}),
          "norm repeat plan publishes one instance's HBM read bytes");
  require(repeated->summary.value("logical_tensor_bytes", u64{0}) ==
              single->summary.value("logical_tensor_bytes", u64{1}),
          "norm repeat plan publishes one instance's logical tensors");
  require(repeated->summary.contains("generative_repeat") &&
              repeated->summary.at("generative_repeat").at("count") == 6 &&
              repeated->summary.at("generative_repeat").at("credit_rule") ==
                  "reset" &&
              repeated->summary.at("generative_repeat").at("activity_work_basis") ==
                  "one_declared_instance",
          "norm repeat plan carries the generative repeat disclosure");
  // A repeated plan takes strictly longer than one instance behind the same
  // region overheads: the seams are real serialized barriers.
  const u64 repeated_ns =
      detail::estimate_latency_ns(*hardware, repeated);
  const u64 single_ns = detail::estimate_latency_ns(*hardware, single);
  require(repeated_ns > single_ns,
          "serial repeat materializes more than one instance of latency");
}

void verify_gates_still_fail_closed() {
  const auto hardware =
      std::make_shared<const Hardware>(example_hardware());
  json fused = repeat_task("ffn_region", 2, 1);
  fused["tensor_shapes"] = {{"m", 1}, {"hidden", 8}, {"ffn", 16},
                            {"mlp_variant", "swiglu"}};
  fused["fusion_policy"] = "fused";
  bool rejected = false;
  try {
    compile(*hardware, fused);
  } catch (const std::invalid_argument& e) {
    rejected = std::string(e.what()).find("materialized FFN template") !=
               std::string::npos;
  }
  require(rejected, "fused FFN stays excluded from declared serial repeat");
  json tiny = repeat_task("norm", 1, 1);
  rejected = false;
  try {
    compile(*hardware, tiny);
  } catch (const std::invalid_argument& e) {
    rejected = std::string(e.what()).find("layer_instances*weight_passes >= 2") !=
               std::string::npos;
  }
  require(rejected, "a single declared instance is not a serial repeat");
}

}  // namespace

int main() { return run_test("instance_repeat_regions_test", []() {
  verify_complete_prediction_and_simulation();
  verify_per_instance_work_basis();
  verify_gates_still_fail_closed();
}); }
