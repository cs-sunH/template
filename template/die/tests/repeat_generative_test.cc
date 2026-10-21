// Declared generative serial repeat: runtime semantics acceptance.
//
// The scheduling barrier counterexample from
// docs/analysis/repeat-availability-20261016/baseline-review.md is the exact
// acceptance: A=(Compute1, Store1) twice behind a declared serial_on_drain
// seam, B arrives at t=1 needing Compute1 on an independent unit-capacity
// resource. Correct: finish A=4, B=2 (the aggregate bulk fold would produce
// A=5, B=3). The seam must be a real no-service window that interference can
// see; nothing may be front-loaded.
#include "test_support.hh"
#include "coarse_die/Prediction.hh"

#include <algorithm>
#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

using namespace coarse_die;
using namespace coarse_die_test;

namespace {

using u64 = std::uint64_t;

// Independent unit capacities in nanoseconds: one unit of work on either
// resource is one nanosecond for a single active owner (rates are per
// second), so every expected finish below is hand-computable integer
// arithmetic.
Hardware unit_hardware() {
  std::array<double, resource_count> rates{};
  rates[COMPUTE_GATE] = 1e9;
  rates[SRAM_READ] = 1e9;
  return manual_hardware(rates);
}

Activity unit_activity(u64 id, std::size_t resource) {
  Activity activity;
  activity.id = id;
  activity.role = Role::Compute;
  activity.work[resource] = 1.0;
  activity.canonical_q = 1.0;
  return activity;
}

// One instance = the whole epoch sequence [compute unit, sram unit].
std::shared_ptr<const Plan> chain_plan(const Hardware& hw, u64 count,
                                       u64 seam_ns, bool declared = true) {
  auto plan = std::make_shared<Plan>();
  plan->task = json{{"op_kind", "repeat_generative_fixture"}};
  plan->summary = json::object();
  plan->hardware_fingerprint = hw.fingerprint;
  plan->launch_ns = 0;
  if (declared) plan->repeat = PlanRepeat{count, seam_ns};
  Epoch compute;
  compute.name = "compute";
  compute.activities = {unit_activity(1, COMPUTE_GATE)};
  Epoch store;
  store.name = "store";
  store.activities = {unit_activity(2, SRAM_READ)};
  plan->epochs = {compute, store};
  plan->fingerprint =
      plan_to_json(*plan).at("fingerprint").get<std::string>();
  return plan;
}

std::shared_ptr<const Plan> single_compute_plan(const Hardware& hw) {
  auto plan = std::make_shared<Plan>();
  plan->task = json{{"op_kind", "repeat_generative_fixture"}};
  plan->summary = json::object();
  plan->hardware_fingerprint = hw.fingerprint;
  plan->launch_ns = 0;
  Epoch compute;
  compute.name = "compute";
  compute.activities = {unit_activity(3, COMPUTE_GATE)};
  plan->epochs = {compute};
  plan->fingerprint =
      plan_to_json(*plan).at("fingerprint").get<std::string>();
  return plan;
}

void drain(Runtime& runtime) {
  unsigned count = 0;
  while (!runtime.drained()) {
    const auto next = runtime.next_event();
    require(next.has_value() && *next > runtime.now_ns() && ++count < 1000,
            "repeat fixture has a bounded advancing completion sequence");
    runtime.advance_to(*next);
  }
}

void verify_barrier_counterexample() {
  const auto hw = unit_hardware();
  Runtime runtime(hw);
  const auto a = runtime.submit(chain_plan(hw, 2, 0), owner(1, 1, 1), 0);
  runtime.advance_to(1);
  const auto b = runtime.submit(single_compute_plan(hw), owner(2, 2, 1), 1);
  drain(runtime);
  const auto a_finish = runtime.terminal_finish_ns(a);
  const auto b_finish = runtime.terminal_finish_ns(b);
  require(a_finish.has_value() && *a_finish == 4,
          "declared seam reproduces the barrier schedule: finish A=4");
  require(b_finish.has_value() && *b_finish == 2,
          "the t=1 arrival keeps its own response: finish B=2");
  const auto progress = runtime.query_progress(a);
  require(progress.at("state") == "finished" &&
              progress.at("activities").size() == 4,
          "both instances stay in the lifetime activity history");
}

void verify_seam_delay_and_idle_window() {
  const auto hw = unit_hardware();

  Runtime runtime(hw);
  const auto handle = runtime.submit(chain_plan(hw, 2, 5), owner(1, 1, 1), 0);
  drain(runtime);
  const auto finish_ns = runtime.terminal_finish_ns(handle);
  require(finish_ns.has_value() && *finish_ns == 9,
          "seam delay adds exactly once per boundary: 2 bodies (2ns) + 5ns");
  const auto report = estimate_alone(hw, *chain_plan(hw, 2, 5));
  require(report.at("estimated_latency_ns") == 9 &&
              report.at("status") == "ok_assumed",
          "standalone estimate executes the declared instances exactly");

  // The seam window itself accrues no service, share, or busy time. Only the
  // wall-clock field may differ between the two instants.
  Runtime early(hw);
  early.submit(chain_plan(hw, 2, 5), owner(1, 1, 1), 0);
  early.advance_to(2);
  auto at_drain = early.counters();
  at_drain.erase("now_ns");
  Runtime mid(hw);
  mid.submit(chain_plan(hw, 2, 5), owner(1, 1, 1), 0);
  mid.advance_to(4);
  auto at_mid = mid.counters();
  at_mid.erase("now_ns");
  require(at_mid == at_drain,
          "the no-service seam window changes no counter");
  const auto mid_progress = mid.query_progress(
      mid.execution_handles().front());
  require(mid_progress.at("state") == "queued",
          "a mid-seam execution reports queued, not running");
}

void verify_snapshot_round_trip_mid_seam() {
  const auto hw = unit_hardware();
  Runtime split(hw);
  const auto handle = split.submit(chain_plan(hw, 2, 5), owner(1, 1, 1), 0);
  split.advance_to(4);
  const auto snapshot = split.export_snapshot();
  auto resumed = Runtime::import_snapshot(hw, snapshot);
  require(resumed && resumed->now_ns() == 4,
          "mid-seam snapshot imports at its own tick");
  drain(split);
  drain(*resumed);
  require(*resumed->terminal_finish_ns(handle) == *split.terminal_finish_ns(handle) &&
              *split.terminal_finish_ns(handle) == 9,
          "resumed execution finishes identically after the seam");
  require(resumed->counters() == split.counters(),
          "resumed runtime publishes identical counters");
}

json rehash(json payload) {
  payload.erase("fingerprint");
  payload["fingerprint"] = stable_hash(payload.dump());
  return payload;
}

void require_invalid_json(const json& payload, const std::string& what) {
  bool rejected = false;
  try {
    (void)plan_from_json(payload);
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  require(rejected, "wire validation must reject: " + what);
}

void verify_wire_and_validation() {
  const auto hw = unit_hardware();
  const auto plan = chain_plan(hw, 3, 7);
  const auto wire = plan_to_json(*plan);
  require(wire.at("schema") == "coarse_die_plan_v5" &&
              wire.at("repeat") ==
                  json{{"count", 3}, {"seam_delay_ns", 7}},
          "declared repeat plans carry schema v5 and the repeat block");
  const auto back = plan_from_json(wire);
  require(plan_to_json(*back) == wire,
          "v5 wire round-trips byte-identically");

  const auto legacy = plan_to_json(*chain_plan(hw, 2, 0, false));
  require(legacy.at("schema") == "coarse_die_plan_v2" &&
              !legacy.contains("repeat"),
          "undeclared plans keep the legacy wire byte-for-byte");

  auto single = wire;
  single["repeat"]["count"] = 1;
  require_invalid_json(rehash(single), "count below two");
  auto huge = wire;
  huge["repeat"]["count"] = kMaxPlanRepeatCount + 1;
  require_invalid_json(rehash(huge), "count above the declared bound");
  auto misplaced = wire;
  misplaced["schema"] = "coarse_die_plan_v4";
  require_invalid_json(rehash(misplaced), "repeat block outside schema v5");
  auto missing = wire;
  missing.erase("repeat");
  require_invalid_json(rehash(missing), "schema v5 without a repeat block");

  bool struct_rejected = false;
  try {
    auto bad = std::make_shared<Plan>(*plan);
    bad->repeat = PlanRepeat{1, 0};
    bad->fingerprint.clear();
    (void)plan_to_json(*bad);
  } catch (const std::invalid_argument&) {
    struct_rejected = true;
  }
  require(struct_rejected, "structure validation rejects count below two");

  require_complete_prediction(*plan);
}

void verify_declared_events_scale() {
  // Two declared instances produce twice the internal event work of one;
  // the honest O(count) contract is visible in the event counter.
  const auto hw = unit_hardware();
  Runtime one(hw);
  one.submit(chain_plan(hw, 1, 0, false), owner(1, 1, 1), 0);
  drain(one);
  Runtime two(hw);
  two.submit(chain_plan(hw, 2, 0), owner(1, 1, 1), 0);
  drain(two);
  const u64 single = one.counters().at("internal_events").get<u64>();
  const u64 doubled = two.counters().at("internal_events").get<u64>();
  require(single > 0 && doubled > single,
          "declared instances add real scheduling events");
}

void verify_declared_compile() {
  auto cfg = read_json(std::string(COARSE_DIE_SOURCE_DIR) +
                       "/configs/example_hardware_explicit.json");
  cfg["architecture"]["core_columns_nx"] = 1;
  cfg["architecture"]["core_rows_ny"] = 1;
  cfg["performance_profile"]["launch_ns_per_macro"] = 750;
  cfg["performance_profile"]["memory_setup_ns_per_macro"] = 1250;
  const auto hw = resolve_hardware(cfg);
  const u64 per_instance_gate = 2000;  // ceil(750 + 1250)

  const auto ffn_task = [](u64 layers, u64 passes, bool declared) {
    json task = {{"op_kind", "ffn_region"},
                 {"tensor_shapes", {{("m"), 1}, {"hidden", 8}, {"ffn", 16},
                                    {"mlp_variant", "swiglu"}}},
                 {"fusion_policy", "materialized"},
                 {"layer_instances", layers},
                 {"weight_passes", passes}};
    if (declared)
      task["repeat_seams"] = json{{"mode", "serial_on_drain"}};
    return task;
  };

  const auto declared = compile(hw, ffn_task(2, 2, true));
  require(declared->summary.at("prediction_status") == "repeat_seam_modeled",
          "declared repeated FFN compiles to a complete modeled prediction");
  require_complete_prediction(*declared);
  require(declared->repeat && declared->repeat->count == 4 &&
              declared->repeat->seam_delay_ns == per_instance_gate &&
              declared->launch_ns == per_instance_gate,
          "four declared instances, seam equals the per-instance region gate");
  require(declared->summary.at("alone_estimate_status") == "repeat_seam_modeled",
          "the alone estimate carries the same modeled status");
  const auto& audit = declared->summary.at("fixed_region_overheads");
  require(audit.at("repeat_seam_modeled_instances") == 4 &&
              audit.at("repeat_seam_delay_ns") == per_instance_gate &&
              audit.at("timing_status") == "repeat_seams_modeled_serial_on_drain",
          "the overhead audit discloses the modeled seams");
  require(plan_to_json(*declared).at("schema") == "coarse_die_plan_v5",
          "declared compile emits plan schema v5");

  // The template is byte-identical to the true one-instance body compile.
  const auto single = compile(hw, ffn_task(1, 1, false));
  require(plan_to_json(*declared).at("epochs") ==
              plan_to_json(*single).at("epochs"),
          "declared activities describe exactly one physical instance");

  // Alone: launch + 4 bodies + 3 seams == 4 x (launch + one body).
  Runtime one(hw);
  const auto single_handle =
      one.submit(single, owner(1, 1, 1), 0);
  drain(one);
  const u64 single_finish = *one.terminal_finish_ns(single_handle);
  Runtime four(hw);
  const auto declared_handle = four.submit(declared, owner(1, 1, 1), 0);
  drain(four);
  const u64 declared_finish = *four.terminal_finish_ns(declared_handle);
  require(single_finish > per_instance_gate &&
              declared_finish == 4 * single_finish,
          "serial instances with the region seam finish at exactly 4x body");
  const auto report = estimate_alone(hw, *declared);
  require(report.at("status") == "repeat_seam_modeled" &&
              report.at("estimated_latency_ns") == declared_finish,
          "the standalone estimate executes the declared instances");

  // Legacy aggregate folds and out-of-scope declarations stay fail-closed.
  const auto legacy = compile(hw, ffn_task(2, 2, false));
  require(legacy->summary.at("prediction_status") == "not_available" &&
              !legacy->repeat,
          "undeclared xR keeps its not_available contract untouched");
  bool legacy_rejected = false;
  try {
    require_complete_prediction(*legacy);
  } catch (const std::invalid_argument&) {
    legacy_rejected = true;
  }
  require(legacy_rejected, "the guard still rejects the legacy bulk fold");

  std::vector<std::pair<json, std::string>> rejections = {
      {json{{"op_kind", "norm"},
            {"tensor_shapes", {{("m"), 1}, {"hidden", 8}}},
            {"repeat_seams", {{"mode", "serial_on_drain"}}}},
       "unsupported op_kind"},
      {ffn_task(1, 1, true), "single instance"},
      {ffn_task(2, 2, false), "undeclared stays bulk"},
  };
  for (auto& [task, what] : rejections) {
    if (what == "undeclared stays bulk") continue;
    bool rejected = false;
    try {
      (void)compile(hw, task);
    } catch (const std::exception&) {
      rejected = true;
    }
    require(rejected, "declaration out of v1 scope fails closed: " + what);
  }
  auto bad_mode = ffn_task(2, 2, false);
  bad_mode["repeat_seams"] = json{{"mode", "parallel_overlap"}};
  bool mode_rejected = false;
  try {
    (void)compile(hw, bad_mode);
  } catch (const std::exception&) {
    mode_rejected = true;
  }
  require(mode_rejected, "unknown seam modes fail closed");
}

}  // namespace

// P11 (2026-10-17): snapshot import must reject a round>0 ready_ns that is
// below the provable inductive lower bound issue + launch + round*seam
// (ready_{r+1} = ceil(drain_r) + seam >= ready_r + seam, base issue+launch).
// The tampered snapshot sets round-1 ready_ns to the old weak bound
// (issue + launch); the tightened contract must refuse it.
void verify_import_rejects_below_inductive_seam_bound() {
  const auto hw = unit_hardware();
  Runtime split(hw);
  split.submit(chain_plan(hw, 2, 5), owner(1, 1, 1), 0);
  split.advance_to(6);  // past the first drain: round 1 is live behind its seam gate
  const auto snapshot = split.export_snapshot();
  bool saw_round_one = false;
  nlohmann::json tampered = snapshot;
  for (auto& execution : tampered["executions"]) {
    if (execution.value("round", 0) != 1) continue;
    saw_round_one = true;
    const auto issue = execution.at("issue_ns").get<std::uint64_t>();
    const auto launch = execution.at("plan").at("launch_ns").get<std::uint64_t>();
    require(execution.at("ready_ns").get<std::uint64_t>() >= issue + launch + 5,
            "genuine round-1 ready gate respects the inductive bound");
    execution["ready_ns"] = issue + launch;  // legal under the old weak bound only
  }
  require(saw_round_one, "fixture reached round 1 before the seam");
  bool rejected = false;
  try {
    Runtime::import_snapshot(hw, tampered);
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  require(rejected,
          "import rejects round>0 ready_ns below issue+launch+round*seam");
}

int main() {
  verify_barrier_counterexample();
  verify_seam_delay_and_idle_window();
  verify_snapshot_round_trip_mid_seam();
  verify_import_rejects_below_inductive_seam_bound();
  verify_wire_and_validation();
  verify_declared_events_scale();
  verify_declared_compile();
  std::printf("[repeat_generative_test] ALL PASS\n");
  return 0;
}
