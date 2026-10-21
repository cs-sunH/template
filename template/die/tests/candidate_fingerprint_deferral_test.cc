// WP3.2 (roofline stage 3): candidate fingerprint deferral + typed
// closed-form eligibility — equivalence oracles.
//
// Deferral contract under test: the compile path may postpone a mapping
// candidate's canonical payload hash until the ranking winner is known, but
// (1) the public enumerator still returns fully sealed Plans,
// (2) a deferred candidate differs from its public counterpart by the
//     fingerprint alone (compact GEMM ranking trials are rebuilt through
//     materialize_gemm_compile_candidate and stay byte-identical to their
//     public candidates), and
// (3) compile()'s public output stays byte-identical to ranking the sealed
//     public candidates with the documented selection rule — the oracle that
//     reproduces the pre-deferral implementation from unchanged public APIs.
//
// Closed-form contract under test: the typed eligibility check admits exactly
// the declared-repeat plans the previous full plan_to_json inspection
// admitted (both facts it read from the payload are implied by the typed
// guard), structural rejection stays on the candidate trial, and public
// fingerprint-consistency checks stay on the canonical codecs.
#include "test_support.hh"
#include "../src/CompilerInternal.hh"

#include <array>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace coarse_die::detail {
std::vector<std::shared_ptr<Plan>> enumerate_mappings(const Hardware&,
                                                      const nlohmann::json&);
std::vector<std::shared_ptr<Plan>> enumerate_mappings_for_compile(
    const Hardware&, const nlohmann::json&);
}

using namespace coarse_die;
using namespace coarse_die_test;
using json = nlohmann::json;
using u64 = std::uint64_t;
using u128 = unsigned __int128;

namespace {

json spans_fixture() {
  return json::array({{{"q_tokens", 64},
                       {"kv_tokens", 64},
                       {"q_start_pos", 0},
                       {"kv_start_pos", 0}}});
}

json gemm_task() {
  return {{"op_kind", "gemm"}, {"tensor_shapes", {{"m", 64}, {"n", 128}, {"k", 64}}}};
}

json gemm_repeat_task(u64 layers) {
  auto task = gemm_task();
  task["layer_instances"] = layers;
  task["weight_passes"] = u64(1);
  task["repeat_seams"] = {{"mode", "serial_on_drain"}};
  return task;
}

json attention_task(const std::string& op, const std::string& fusion) {
  return {{"op_kind", op},
          {"phase", "prefill"},
          {"tensor_shapes",
           {{"m", 64}, {"q_heads", 2}, {"kv_heads", 2}, {"head_dim", 32}}},
          {"spans", spans_fixture()},
          {"fusion_policy", fusion}};
}

json qkv_rope_task(bool rope) {
  return {{"op_kind", "qkv_rope"},
          {"tensor_shapes",
           {{"m", 64},
            {"hidden", 128},
            {"q_heads", 2},
            {"kv_heads", 2},
            {"head_dim", 64},
            {"position_encoding", rope ? "rope" : "none"}}}};
}

json o_proj_task() {
  return {{"op_kind", "o_proj"}, {"tensor_shapes", {{"m", 64}, {"hidden", 128}}}};
}

json mlp_task(const char* op) {
  return {{"op_kind", op},
          {"tensor_shapes", {{"m", 64}, {"hidden", 128}, {"ffn", 64}}}};
}

json logits_task() {
  return {{"op_kind", "logits"},
          {"tensor_shapes", {{"m", 64}, {"hidden", 128}, {"vocab", 256}}}};
}

json ffn_task(const std::string& fusion, const std::string& variant) {
  return {{"op_kind", "ffn_region"},
          {"tensor_shapes",
           {{"m", 64}, {"hidden", 128}, {"ffn", 64}, {"mlp_variant", variant}}},
          {"fusion_policy", fusion}};
}

json norm_task(const char* op) {
  return {{"op_kind", op},
          {"tensor_shapes", {{"m", 64}, {"hidden", 128}, {"norm_type", "rmsnorm"}}}};
}

json softmax_task(const char* op) {
  return {{"op_kind", op},
          {"tensor_shapes", {{"q_heads", 2}, {"head_dim", 32}}},
          {"spans", spans_fixture()}};
}

json ffn_repeat_task(u64 layers) {
  auto task = ffn_task("materialized", "swiglu");
  task["layer_instances"] = layers;
  task["weight_passes"] = u64(1);
  task["repeat_seams"] = {{"mode", "serial_on_drain"}};
  return task;
}

u64 summary_number(const Plan& plan, const char* key) {
  const auto it = plan.summary.find(key);
  if (it == plan.summary.end()) return 0;
  return it->get<u64>();
}

std::string summary_text(const Plan& plan, const char* key) {
  const auto it = plan.summary.find(key);
  return it != plan.summary.end() && it->is_string() ? it->get<std::string>()
                                                     : std::string();
}

// Reproduce the pre-deferral compiler from unchanged public surfaces: rank
// the fully sealed public candidates with the documented selection rule and
// attach exactly the report fields compile() publishes, then seal.
std::shared_ptr<Plan> reconstruct_compiled(
    const Hardware& hw, const std::vector<std::shared_ptr<Plan>>& full) {
  struct Ranked {
    std::size_t index = 0;
    u64 latency = 0;
    u128 hbm = 0;
    u64 workspace = 0, sram = 0;
    std::string template_id;
    u64 tm = 0, tn = 0, tk = 0, th = 0;
  };
  std::vector<Ranked> ranked;
  ranked.reserve(full.size());
  for (std::size_t i = 0; i < full.size(); ++i) {
    const Plan& candidate = *full[i];
    Ranked r;
    r.index = i;
    r.latency = detail::estimate_latency_ns(hw, full[i]);
    r.hbm = static_cast<u128>(summary_number(candidate, "hbm_read_bytes_total")) +
            summary_number(candidate, "hbm_write_bytes_total");
    r.workspace = summary_number(candidate, "hbm_workspace_peak_bytes");
    r.sram = summary_number(candidate, "sram_peak_bytes_per_core");
    r.template_id = summary_text(candidate, "template_id");
    r.tm = summary_number(candidate, "tm");
    r.tn = summary_number(candidate, "tn");
    r.tk = summary_number(candidate, "tk");
    r.th = summary_number(candidate, "th");
    ranked.push_back(std::move(r));
  }
  const auto key = [](const Ranked& r) {
    return std::tie(r.latency, r.hbm, r.workspace, r.sram, r.template_id,
                    r.tm, r.tn, r.tk, r.th);
  };
  const auto winner = std::min_element(
      ranked.begin(), ranked.end(),
      [&](const Ranked& a, const Ranked& b) { return key(a) < key(b); });
  auto selected = std::make_shared<Plan>(*full[winner->index]);
  selected->summary["estimated_latency_ns"] = winner->latency;
  selected->summary["alone_estimate_status"] =
      selected->summary.value("prediction_status", std::string("ok_assumed"));
  selected->summary["candidate_count"] = static_cast<u64>(ranked.size());
  selected->summary["selected_template"] =
      selected->summary.value("template_id", std::string());
  selected->summary["selection_rule"] =
      "ceil(estimate_alone_ns),hbm_payload_bytes,hbm_workspace_peak_bytes,"
      "sram_peak_bytes_per_core,template_id,tm,tn,tk,th";
  selected->summary["candidate_estimate_source"] =
      "estimate_alone(shared_runtime_engine)";
  selected->summary["activity_work_semantics"] =
      selected->repeat
          ? "activity work/bytes/ops describe one declared repeat instance; "
            "repeat.count instances execute serially behind real no-service "
            "seams and credit resets per instance"
          : "activity work/bytes/ops are epoch total across repeat_count; "
            "repeat_count is audit metadata only; credit_delta is total "
            "signed credit change over progress 0..1";
  selected->summary["granularity_validation"] = "not_run";
  selected->summary["evidence_flags"] = {"engineering_assumption",
                                         "hardware_source_derived"};
  selected->fingerprint.clear();
  selected->fingerprint =
      plan_to_json(*selected).at("fingerprint").get<std::string>();
  return selected;
}

void deferral_equivalence(const Hardware& hw, const json& task,
                          const std::string& label) {
  const auto normalized = detail::normalize_task(hw, task);
  const auto full = detail::enumerate_mappings(hw, normalized);
  const auto trials = detail::enumerate_mappings_for_compile(hw, normalized);
  require(!full.empty(), label + ": fixture enumerates mapping candidates");
  require(full.size() == trials.size(),
          label + ": deferral preserves every candidate and its order");
  const bool gemm = normalized.at("op_kind").get<std::string>() == "gemm";
  for (std::size_t i = 0; i < full.size(); ++i) {
    require(full[i]->fingerprint.size() == 64 &&
                plan_to_json(*full[i]).at("fingerprint") == full[i]->fingerprint,
            label + ": public enumeration still seals every candidate hash");
    require(trials[i]->fingerprint.empty(),
            label + ": compile enumeration defers every candidate hash");
    if (gemm) {
      auto restored = detail::materialize_gemm_compile_candidate(hw, *trials[i]);
      detail::finalize_region_overhead_audit(hw, *restored);
      restored->fingerprint =
          plan_to_json(*restored).at("fingerprint").get<std::string>();
      require(plan_to_json(*restored) == plan_to_json(*full[i]),
              label + ": materialized GEMM trial stays byte-identical to its "
                      "public candidate");
    } else {
      require(plan_to_json(*trials[i]) == plan_to_json(*full[i]),
              label + ": deferred candidate differs from its public "
                      "counterpart by the fingerprint alone");
    }
  }
  const auto compiled = compile(hw, task);
  const auto expected = reconstruct_compiled(hw, full);
  require(plan_to_json(*expected) == plan_to_json(*compiled),
          label + ": compile() output is byte-identical to the documented "
                  "ranking over sealed public candidates");
}

std::shared_ptr<Plan> chain_plan(const Hardware& hw, u64 count, u64 seam) {
  Epoch compute;
  compute.name = "compute";
  compute.activities = {activity(1, Role::Compute, COMPUTE_GATE, 1.0)};
  Epoch store;
  store.name = "store";
  store.activities = {activity(2, Role::Compute, SRAM_READ, 1.0)};
  auto plan = std::make_shared<Plan>();
  plan->task = json{{"op_kind", "candidate_deferral_fixture"}};
  plan->summary = json{{"status", "test_fixture"}};
  plan->hardware_fingerprint = hw.fingerprint;
  plan->epochs = {compute, store};
  plan->repeat = PlanRepeat{count, seam};
  plan->fingerprint =
      plan_to_json(*plan).at("fingerprint").get<std::string>();
  return plan;
}

u64 drained_finish(const std::shared_ptr<const Hardware>& hw,
                   std::shared_ptr<const Plan> plan) {
  Runtime runtime(hw);
  const auto handle = runtime.submit(std::move(plan), owner(1, 1, 1), 0);
  u64 guard = 0;
  while (!runtime.drained()) {
    const auto next = runtime.next_event();
    require(next.has_value() && *next > runtime.now_ns() && ++guard < 3000000,
            "fixture has a bounded advancing completion sequence");
    runtime.advance_to(*next);
  }
  const auto finish = runtime.terminal_finish_ns(handle);
  require(finish.has_value(), "fixture drains to a finish");
  return *finish;
}

void typed_closed_form_eligibility() {
  std::array<double, resource_count> rates{};
  rates[COMPUTE_GATE] = 1e9;
  rates[SRAM_READ] = 1e9;
  const auto hw = std::make_shared<const Hardware>(manual_hardware(rates));
  // Eligible declared repeats: the unsealed candidate trial, the sealed
  // estimate and the full drain all agree bit for bit, so the typed check
  // admits exactly what the old serialized payload inspection admitted.
  for (const auto& [count, seam] :
       std::vector<std::pair<u64, u64>>{{8, 0}, {8, 5}, {9, 3}}) {
    const auto plan = chain_plan(*hw, count, seam);
    const u64 simulated = drained_finish(hw, plan);
    const u64 sealed_ns = detail::estimate_latency_ns(*hw, plan);
    auto unsealed = std::make_shared<Plan>(*plan);
    unsealed->fingerprint.clear();
    const u64 trial_ns = detail::estimate_candidate_latency_ns(hw, unsealed);
    require(sealed_ns == simulated && trial_ns == simulated,
            "typed closed-form eligibility keeps both estimate channels equal "
            "to the drained finish");
  }
  // Below-threshold declared repeats decline on both channels.
  for (const auto& [count, seam] :
       std::vector<std::pair<u64, u64>>{{2, 5}, {7, 0}}) {
    const auto plan = chain_plan(*hw, count, seam);
    const u64 simulated = drained_finish(hw, plan);
    auto unsealed = std::make_shared<Plan>(*plan);
    unsealed->fingerprint.clear();
    require(detail::estimate_latency_ns(*hw, plan) == simulated &&
                detail::estimate_candidate_latency_ns(hw, unsealed) ==
                    simulated,
            "below-threshold repeats keep the full-drain estimate");
  }
  // Plans without a declared repeat never enter the closed form.
  {
    auto plan = chain_plan(*hw, 8, 5);
    plan->repeat.reset();
    plan->fingerprint.clear();
    plan->fingerprint =
        plan_to_json(*plan).at("fingerprint").get<std::string>();
    const u64 simulated = drained_finish(hw, plan);
    require(detail::estimate_latency_ns(*hw, plan) == simulated,
            "non-repeat plans keep the full-drain estimate");
  }
  // Structural rejection stays on the candidate trial: deferring the hash
  // never defends against an invalid plan.
  auto duplicate = std::make_shared<Plan>(*chain_plan(*hw, 8, 5));
  duplicate->fingerprint.clear();
  duplicate->epochs.front().activities.push_back(
      duplicate->epochs.front().activities.front());
  bool rejected = false;
  try {
    (void)detail::estimate_candidate_latency_ns(hw, duplicate);
  } catch (const std::invalid_argument& error) {
    rejected = std::string(error.what()).find("duplicate activity id") !=
               std::string::npos;
  }
  require(rejected, "the candidate trial still runs every structural plan check");
  auto over = std::make_shared<Plan>(*chain_plan(*hw, 8, 5));
  over->fingerprint.clear();
  over->repeat = PlanRepeat{kMaxPlanRepeatCount + 1, 0};
  rejected = false;
  try {
    (void)detail::estimate_candidate_latency_ns(hw, over);
  } catch (const std::invalid_argument& error) {
    rejected = std::string(error.what()).find("plan.repeat.count") !=
               std::string::npos;
  }
  require(rejected,
          "the candidate trial still rejects out-of-range declared repeat counts");
  // Public fingerprint-consistency checks remain on the canonical codecs.
  auto corrupt = chain_plan(*hw, 8, 5);
  corrupt->fingerprint[0] = corrupt->fingerprint[0] == '0' ? '1' : '0';
  rejected = false;
  try {
    (void)plan_to_json(*corrupt);
  } catch (const std::invalid_argument& error) {
    rejected = std::string(error.what()).find(
                   "does not match canonical payload") != std::string::npos;
  }
  require(rejected,
          "plan_to_json still rejects a fingerprint that does not match its payload");
}

void body() {
  const Hardware hardware = example_hardware();
  deferral_equivalence(hardware, gemm_task(), "gemm");
  deferral_equivalence(hardware, gemm_repeat_task(4), "gemm_declared_repeat");
  deferral_equivalence(hardware, attention_task("attention", "auto"),
                       "attention_auto");
  deferral_equivalence(hardware, attention_task("attention_region", "materialized"),
                       "attention_materialized");
  deferral_equivalence(hardware, attention_task("attention_region", "fused"),
                       "attention_fused");
  deferral_equivalence(hardware, attention_task("qk", "auto"), "qk");
  deferral_equivalence(hardware, attention_task("av", "auto"), "av");
  deferral_equivalence(hardware, qkv_rope_task(true), "qkv_rope_rope");
  deferral_equivalence(hardware, qkv_rope_task(false), "qkv_rope_plain");
  deferral_equivalence(hardware, o_proj_task(), "o_proj");
  deferral_equivalence(hardware, mlp_task("mlp_up"), "mlp_up");
  deferral_equivalence(hardware, mlp_task("mlp_down"), "mlp_down");
  deferral_equivalence(hardware, logits_task(), "logits");
  deferral_equivalence(hardware, ffn_task("auto", "swiglu"), "ffn_auto");
  deferral_equivalence(hardware, ffn_task("materialized", "swiglu"),
                       "ffn_materialized");
  deferral_equivalence(hardware, ffn_task("materialized", "gelu"),
                       "ffn_materialized_gelu");
  deferral_equivalence(hardware, ffn_task("fused", "swiglu"), "ffn_fused");
  deferral_equivalence(hardware, norm_task("norm"), "norm");
  deferral_equivalence(hardware, norm_task("residual"), "residual");
  deferral_equivalence(hardware, softmax_task("softmax"), "softmax");
  deferral_equivalence(hardware, softmax_task("scale_mask"), "scale_mask");
  deferral_equivalence(hardware, ffn_repeat_task(8), "ffn_declared_repeat_8");
  deferral_equivalence(hardware, ffn_repeat_task(3), "ffn_declared_repeat_3");
  typed_closed_form_eligibility();
}

}  // namespace

int main() { return run_test("candidate_fingerprint_deferral_test", body); }
