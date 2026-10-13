#include "test_support.hh"

#include <cstdint>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

using namespace coarse_die;
using namespace coarse_die_test;
using json = nlohmann::json;

namespace {

constexpr std::uint64_t kQ = 2;
constexpr std::uint64_t kKvFirst = 4;
constexpr std::uint64_t kKvStride = 2;
constexpr std::uint64_t kRunCount = 2;
constexpr std::uint64_t kBatch = 3;
constexpr std::uint64_t kQHeads = 2;
constexpr std::uint64_t kHeadDim = 2;

json compact_span(std::uint64_t batch = kBatch) {
  return json{{"run", {{"q_len", kQ}, {"kv_len", kKvFirst},
                        {"kv_stride", kKvStride}, {"count", kRunCount}}},
              {"batch", batch}, {"request_id_prefix", "request"},
              {"first_index", 0}};
}

json explicit_spans() {
  json spans = json::array();
  for (std::uint64_t b = 0; b < kBatch; ++b) {
    for (std::uint64_t i = 0; i < kRunCount; ++i) {
      const std::uint64_t kv = kKvFirst + i * kKvStride;
      spans.push_back({{"q_len", kQ}, {"kv_len", kv}});
    }
  }
  return spans;
}

json attention_task(const json& spans, bool causal, const std::string& fusion) {
  return json{{"op_kind", "attention"},
              {"tensor_shapes", {{"q_heads", kQHeads}, {"kv_heads", 1},
                                 {"head_dim", kHeadDim}}},
              {"spans", spans}, {"causal", causal},
              {"fusion_policy", fusion}};
}

json transform_task(const std::string& op, const json& spans) {
  return json{{"op_kind", op}, {"tensor_shapes", {{"q_heads", kQHeads}}},
              {"spans", spans}};
}

json norm_task(const json& spans) {
  return json{{"op_kind", "norm"}, {"tensor_shapes", {{"hidden", 8}}},
              {"spans", spans}};
}

void rejected(const Hardware& hw, const json& task, const std::string& fragment);

std::uint64_t field(const json& summary, const char* name) {
  require(summary.contains(name), std::string("summary is missing ") + name);
  return summary.at(name).get<std::uint64_t>();
}

void summaries_match(const Plan& actual, const Plan& expected,
                     const std::vector<const char*>& fields,
                     const std::string& label) {
  for (const char* key : fields) {
    require(actual.summary.contains(key) && expected.summary.contains(key),
            label + " summaries must both expose " + key);
    require(actual.summary.at(key) == expected.summary.at(key),
            label + " compact and explicit values differ for " + key + ": " +
                actual.summary.at(key).dump() + " vs " + expected.summary.at(key).dump());
  }
}

std::pair<std::uint64_t, std::uint64_t> discrete_attention_counts(bool causal) {
  std::uint64_t slots = 0, valid = 0;
  for (std::uint64_t run = 0; run < kRunCount; ++run) {
    const std::uint64_t kv = kKvFirst + run * kKvStride;
    const std::uint64_t query_start = kv - kQ;
    for (std::uint64_t q = 0; q < kQ; ++q) {
      const std::uint64_t query_position = query_start + q;
      for (std::uint64_t key = 0; key < kv; ++key) {
        ++slots;
        if (!causal || key <= query_position) ++valid;
      }
    }
  }
  return {slots * kBatch * kQHeads, valid * kBatch * kQHeads};
}

void compare_runtime(const Hardware& hw, const Plan& compact,
                     const Plan& expanded, const std::string& label) {
  const json compact_run = estimate_alone(hw, compact);
  const json expanded_run = estimate_alone(hw, expanded);
  require(compact_run.at("status") == "ok_assumed" &&
              expanded_run.at("status") == "ok_assumed",
          label + " compiled plans must complete in the real Runtime");
  require(field(compact_run, "estimated_latency_ns") > 0 &&
              compact_run.at("estimated_latency_ns") ==
                  expanded_run.at("estimated_latency_ns"),
          label + " Runtime finish must match explicit per-request spans");
}

void attention_matches_discrete_requests_and_runtime() {
  const Hardware hw = example_hardware();
  const auto counts_causal = discrete_attention_counts(true);
  const auto counts_noncausal = discrete_attention_counts(false);
  require(counts_causal.first == 120 && counts_causal.second == 108,
          "independent causal loop computes 120 slots and 108 valid pairs");
  require(counts_noncausal.first == 120 && counts_noncausal.second == 120,
          "independent noncausal loop computes 120 slots and pairs");

  const std::vector<const char*> totals = {
      "source_matrix_ops", "executed_matrix_ops", "source_vec_ops",
      "vec_equivalent_ops", "logical_tensor_bytes", "physical_transfer_bytes",
      "hbm_read_bytes_total", "hbm_write_bytes_total", "hbm_workspace_peak_bytes",
      "sram_peak_bytes_per_core", "sram_read_bytes", "sram_write_bytes",
      "local_bus_bytes", "output_round_count", "consumption_wave_count",
      "active_cores", "executed_score_slots", "valid_score_pairs", "score_bytes",
      "probability_bytes", "conversion_elements", "int4_input_bytes",
      "candidate_count", "tm", "tn", "tk"};
  for (const bool causal : {true, false}) {
    const auto expected = causal ? counts_causal : counts_noncausal;
    for (const std::string fusion : {"fused", "materialized"}) {
      const auto compact = compile(hw, attention_task(
          json::array({compact_span()}), causal, fusion));
      const auto expanded = compile(hw, attention_task(
          explicit_spans(), causal, fusion));
      summaries_match(*compact, *expanded, totals,
                      std::string(causal ? "causal " : "noncausal ") + fusion);
      require(field(compact->summary, "executed_score_slots") == expected.first &&
                  field(compact->summary, "valid_score_pairs") == expected.second,
              "compiled attention counters must match the independent discrete oracle");
      require(field(compact->summary, "logical_tensor_bytes") == 432,
              "logical Q, KV, and output bytes count the six repeated requests");
      require(compact->task.at("spans").at(0).at("count") == kRunCount &&
                  compact->task.at("spans").at(0).at("batch") == kBatch,
              "canonical task keeps run count and batch as independent dimensions");
      require(field(compact->summary, "sram_peak_bytes_per_core") <=
                  field(compact->summary, "sram_capacity_bytes_per_core"),
              "per-core tile live storage remains within the hardware capacity");
      compare_runtime(hw, *compact, *expanded,
                      std::string(causal ? "causal " : "noncausal ") + fusion);
      const auto restored = plan_from_json(plan_to_json(*compact));
      require(restored->fingerprint == compact->fingerprint &&
                  plan_to_json(*restored) == plan_to_json(*compact),
              "batch plan keeps its normalized task and activity ledger across roundtrip");
    }
  }
}

void row_and_score_transforms_count_batch_once() {
  const Hardware hw = example_hardware();
  const auto compact = json::array({compact_span()});
  const auto expanded = explicit_spans();
  const std::vector<const char*> totals = {
      "source_vec_ops", "vec_equivalent_ops", "logical_tensor_bytes",
      "physical_transfer_bytes", "hbm_read_bytes_total", "hbm_write_bytes_total",
      "sram_peak_bytes_per_core", "sram_read_bytes", "sram_write_bytes",
      "local_bus_bytes", "output_round_count", "active_cores"};

  const auto compact_norm = compile(hw, norm_task(compact));
  const auto expanded_norm = compile(hw, norm_task(expanded));
  summaries_match(*compact_norm, *expanded_norm, totals, "Norm row batch");
  require(compact_norm->task.at("tensor_shapes").at("m") == kQ * kRunCount * kBatch,
          "row operators derive M from run count times batch");
  require(field(compact_norm->summary, "source_vec_ops") == 576,
          "Norm work includes all repeated query rows exactly once");

  for (const std::string op : {"scale_mask", "softmax"}) {
    const auto compact_plan = compile(hw, transform_task(op, compact));
    const auto expanded_plan = compile(hw, transform_task(op, expanded));
    summaries_match(*compact_plan, *expanded_plan, totals, op + " score batch");
    require(compact_plan->task.at("tensor_shapes").at("m") ==
                kQ * kRunCount * kBatch,
            op + " derives query rows from the repeated run");
    const std::uint64_t per_score_ops = op == "scale_mask" ? 2 : 10;
    require(field(compact_plan->summary, "source_vec_ops") ==
                120 * per_score_ops,
            op + " Vec work counts all batch score slots");
  }
}

void batch_does_not_change_single_request_capacity_or_legacy_hashes() {
  const Hardware hw = example_hardware();
  const auto one = compile(hw, attention_task(
      json::array({compact_span(1)}), true, "fused"));
  const auto many = compile(hw, attention_task(
      json::array({compact_span()}), true, "fused"));
  require(field(one->summary, "sram_peak_bytes_per_core") ==
              field(many->summary, "sram_peak_bytes_per_core"),
          "fused tile SRAM peak is per-core and does not grow with batch count");
  require(field(many->summary, "active_cores") >= field(one->summary, "active_cores"),
          "batch multiplicity increases available parallel request work");

  const auto full_one = compile(hw, attention_task(
      json::array({compact_span(1)}), true, "materialized"));
  const auto full_many = compile(hw, attention_task(
      json::array({compact_span()}), true, "materialized"));
  require(field(full_many->summary, "hbm_workspace_peak_bytes") ==
              3 * field(full_one->summary, "hbm_workspace_peak_bytes"),
          "materialized workspace includes all simultaneously live batch score/probability tensors");
  require(field(full_many->summary, "sram_peak_bytes_per_core") ==
              field(full_one->summary, "sram_peak_bytes_per_core"),
          "materialized per-core tile capacity remains independent of batch multiplicity");

  const auto hash_fixture = [&](bool explicit_default_batch) {
    json span{{"run", {{"q_len", 1}, {"kv_len", 4}, {"kv_stride", 2},
                       {"count", 2}}},
              {"request_id_prefix", "request"}, {"first_index", 0}};
    if (explicit_default_batch) span["batch"] = 1;
    return compile(hw, json{{"op_kind", "attention"},
        {"tensor_shapes", {{"q_heads", 1}, {"kv_heads", 1}, {"head_dim", 1}}},
        {"causal", true}, {"spans", json::array({span})}});
  };
  const auto no_batch = hash_fixture(false);
  const auto batch_one = hash_fixture(true);
  require(no_batch->fingerprint == batch_one->fingerprint &&
              no_batch->fingerprint ==
                  "138de6613111411b3ebd6c43ae5a59500e205d7b4809c979d97c2edd32491e1f",
          "batch=1 compact plans retain the pre-change canonical fingerprint");

  const json examples = read_json(std::string(COARSE_DIE_SOURCE_DIR) +
                                  "/configs/example_tasks.json");
  const auto legacy = compile(hw, examples.at("specs").at(1));
  require(legacy->fingerprint ==
              "945af35e59d26f58bda279450c6d53a35b25d8d8578d348a928c63a43cb4e2b0",
          "legacy v1 Plan identity is unchanged");
  const json default_gemm{{"op_kind", "gemm"},
      {"tensor_shapes", {{"m", 3}, {"n", 5}, {"k", 7}}},
      {"storage_dtype", {{"weights", "bf16"}, {"activations", "bf16"},
                          {"output", "bf16"}}},
      {"execution_profile", "int4_matrix_vec_fp32_assumed_v1"},
      {"layer_instances", 1}, {"weight_passes", 1}};
  require(compile(hw, default_gemm)->fingerprint ==
              "bdede6257f71948842252eb43a47686d0a2e40923b7d58230e2cb71ff89143fb",
          "default non-span Plan identity is unchanged");
}

void compact_count_and_position_overflows_are_rejected() {
  const Hardware hw = example_hardware();
  const std::uint64_t max = std::numeric_limits<std::uint64_t>::max();
  const auto attention = [&](const json& run, std::uint64_t batch) {
    return json{{"op_kind", "attention"},
        {"tensor_shapes", {{"q_heads", 1}, {"kv_heads", 1}, {"head_dim", 1}}},
        {"spans", json::array({json{{"run", run}, {"batch", batch}}})}};
  };
  rejected(hw, attention({{"q_len", 1}, {"kv_len", 1}, {"kv_stride", 0},
                          {"count", max}}, 2), "task.spans[0].count");
  rejected(hw, attention({{"q_len", 1}, {"kv_len", max - 1}, {"kv_stride", 2},
                          {"count", 2}}, 1), "kv length sequence overflows");
  rejected(hw, attention({{"q_len", 1}, {"kv_len", 1}, {"kv_stride", 1},
                          {"count", 2}, {"q_start_pos", max}}, 2),
           "query position sequence overflows");

  const json row_overflow{{"op_kind", "norm"}, {"tensor_shapes", {{"hidden", 8}}},
      {"spans", json::array({json{{"run", {{"q_len", 2}, {"kv_len", 2},
          {"kv_stride", 0}, {"count", max / 2 + 1}}}, {"batch", 1}}})}};
  rejected(hw, row_overflow, "task.spans query rows");
}

void rejected(const Hardware& hw, const json& task, const std::string& fragment) {
  try {
    (void)compile(hw, task);
  } catch (const std::invalid_argument& e) {
    require(std::string(e.what()).find(fragment) != std::string::npos,
            "overflow rejection must include '" + fragment + "': " + e.what());
    return;
  }
  throw std::runtime_error("overflowing compact span unexpectedly compiled: " + task.dump());
}

void test_compact_batch_semantics() {
  attention_matches_discrete_requests_and_runtime();
  row_and_score_transforms_count_batch_once();
  batch_does_not_change_single_request_capacity_or_legacy_hashes();
  compact_count_and_position_overflows_are_rejected();
  std::cout << "compact batch: discrete counts, Runtime, workspace, capacity, hashes, overflow\n";
}

}  // namespace

int main() { return run_test("compact_batch_semantics_test", test_compact_batch_semantics); }
