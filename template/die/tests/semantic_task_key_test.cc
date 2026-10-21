#include "test_support.hh"

#include <array>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

using namespace coarse_die;
using namespace coarse_die_test;

namespace {

// Minimal legacy DieOpSpec in the exact shape the FACE host emits: spans
// carry per-span request ids, and the wrapper carries provenance/config
// hashes. None of those fields survive normalization.
json old_ffn_spec(const std::string& span0_id, const std::string& span1_id,
                  const std::string& provenance, const std::string& config) {
  return json{{"schema_version", 1},
              {"op_kind", "ffn_region"},
              {"phase", "decode"},
              {"dtype", "bf16"},
              {"model_dims", {{"hidden", 64},
                              {"ffn", 128},
                              {"vocab", 256},
                              {"mlp_variant", "gelu"},
                              {"norm_type", "layernorm"},
                              {"position_encoding", "none"},
                              {"num_q_heads", 8},
                              {"num_kv_heads", 8},
                              {"head_dim", 8}}},
              {"tp_slice", {{"tensor_parallel", 2},
                             {"tensor_parallel_rank", 0},
                             {"q_heads_local", 4},
                             {"kv_heads_local", 4},
                             {"ffn_local", 64},
                             {"vocab_local", 128}}},
              {"spans", json::array({
                            json{{"request_id", span0_id},
                                 {"q_tokens", 1},
                                 {"kv_tokens", 1},
                                 {"q_start_pos", 0},
                                 {"kv_start_pos", 0}},
                            json{{"request_id", span1_id},
                                 {"q_tokens", 1},
                                 {"kv_tokens", 2},
                                 {"q_start_pos", 1},
                                 {"kv_start_pos", 0}},
                        })},
              {"kv_access", {{"identity", "local"}}},
              {"fusion_policy", "auto"},
              {"layer_count", 2},
              {"weight_passes", 2},
              {"provenance", provenance},
              {"config_hash", config}};
}

json v2_gemm_task() {
  return json{{"op_kind", "gemm"},
              {"phase", "prefill"},
              {"tensor_shapes", {{"m", 3}, {"n", 5}, {"k", 7}}},
              {"storage_dtype", json{{"weights", "int4"}, {"activations", "int4"},
                                      {"output", "bf16"}}},
              {"execution_profile", "int4_matrix_vec_fp32_assumed_v1"},
              {"layer_instances", 1},
              {"weight_passes", 1}};
}

void require_semantic_equality(const Hardware& hw, const json& a, const json& b,
                              const std::string& what) {
  const std::string key_a = semantic_task_key(hw, a);
  const std::string key_b = semantic_task_key(hw, b);
  require(key_a == key_b,
          what + ": identity-only change must preserve the semantic key");
  require(key_a == semantic_task_key(hw, a),
          what + ": repeated key computation is deterministic");
  require(json::parse(key_a) == json::parse(key_b),
          what + ": normalized tasks are equivalent objects");
}

void require_semantic_difference(const Hardware& hw, const json& a,
                                 const json& b, const std::string& what) {
  require(semantic_task_key(hw, a) != semantic_task_key(hw, b),
          what + ": a compiler-consumed field change must break the key");
}

void test_key_matches_public_normalized_fingerprint() {
  const auto hw = example_hardware();
  const auto task = v2_gemm_task();
  const auto envelope = compile_workspace_envelope(hw, task);
  const std::string key = semantic_task_key(hw, task);
  require(key == envelope.normalized_task.dump(),
          "semantic key is the normalized task text the envelope publishes");
  require(stable_hash(key) == envelope.normalized_task_fingerprint,
          "hashing the semantic key reproduces normalized_task_fingerprint");
  const auto legacy = old_ffn_spec("train_i0_1#s0", "train_i0_1#s1",
                                   "transformer_pass_aggregated",
                                   "3321c0cd5543414ad55b4601497027130d");
  const auto legacy_envelope = compile_workspace_envelope(
      hw, json{{"die_op_spec", legacy}});
  require(semantic_task_key(hw, json{{"die_op_spec", legacy}}) ==
              legacy_envelope.normalized_task.dump(),
          "legacy wrapper spelling folds onto the same canonical text");
}

void test_identity_only_changes_preserve_key() {
  const auto hw = example_hardware();
  const json base = json{{"die_op_spec", old_ffn_spec(
      "batch_train_i1_1#s0", "batch_train_i1_1#s1",
      "transformer_pass_aggregated",
      "3321c0cd5543414ad55b4601497027130d0e0a32994a7e52cae3716e13424cda")}};
  require_semantic_equality(
      hw, base,
      json{{"die_op_spec", old_ffn_spec(
           "batch_train_i1_2#s0", "batch_train_i1_2#s1",
           "transformer_pass_aggregated",
           "3321c0cd5543414ad55b4601497027130d0e0a32994a7e52cae3716e13424cda")}},
      "span request ids");
  require_semantic_equality(
      hw, base,
      json{{"die_op_spec", old_ffn_spec(
           "batch_train_i1_1#s0", "batch_train_i1_1#s1",
           "other_emitter",
           "3321c0cd5543414ad55b4601497027130d0e0a32994a7e52cae3716e13424cda")}},
      "provenance and config hash");

  // The compact run form: a prefix plus first index expanding into count
  // spans. Changing only those labels keeps the expanded geometry.
  const auto run_spec = [](const std::string& prefix, std::uint64_t first) {
    json spec = old_ffn_spec("unused#s0", "unused#s1", "p", "c");
    spec["spans"] = json::array({json{
        {"request_id_prefix", prefix},
        {"first_index", first},
        {"run", json{{"q_tokens", 1}, {"kv_first", 512},
                      {"kv_stride", 512}, {"count", 3}}}}});
    return json{{"die_op_spec", spec}};
  };
  require_semantic_equality(hw, run_spec("batch_train_i0_1#", 0),
                            run_spec("batch_train_i0_2#", 7),
                            "run-form prefix and first index");

  // v2 spelling: per-span request ids are accepted then dropped.
  const auto v2_attention = [](const std::string& request_id) {
    return json{{"op_kind", "attention"},
                {"tensor_shapes", {{"q_heads", 8}, {"kv_heads", 8},
                                    {"head_dim", 8}}},
                {"spans", json::array({json{
                    {"request_id", request_id},
                    {"q_tokens", 2}, {"kv_tokens", 4},
                    {"q_start_pos", 2}, {"kv_start_pos", 0}}})}};
  };
  require_semantic_equality(hw, v2_attention("train_a#s0"),
                            v2_attention("train_b#s9"),
                            "v2 span request id");
}

void test_semantic_changes_break_key() {
  const auto hw = example_hardware();
  const json base = v2_gemm_task();
  struct Mutation {
    const char* what;
    json task;
  };
  std::vector<Mutation> mutations;
  {
    json t = base; t["tensor_shapes"]["m"] = 4;
    mutations.push_back({"tensor_shapes.m", t});
  }
  {
    json t = base; t["tensor_shapes"]["n"] = 6;
    mutations.push_back({"tensor_shapes.n", t});
  }
  {
    json t = base; t["tensor_shapes"]["k"] = 8;
    mutations.push_back({"tensor_shapes.k", t});
  }
  {
    json t = base; t["storage_dtype"]["weights"] = "bf16";
    mutations.push_back({"storage_dtype.weights", t});
  }
  {
    json t = base; t["layer_instances"] = 2;
    mutations.push_back({"layer_instances", t});
  }
  {
    json t = base; t["weight_passes"] = 3;
    mutations.push_back({"weight_passes", t});
  }
  {
    json t = base; t["phase"] = "decode";
    mutations.push_back({"phase", t});
  }
  {
    json t = base; t["tp_slice"] = {{"tensor_parallel", 2},
                                     {"tensor_parallel_rank", 1}};
    mutations.push_back({"tp_slice rank", t});
  }
  {
    json t = base; t["causal"] = true;
    mutations.push_back({"causal", t});
  }
  {
    json t = base; t["execution_profile"] = "fp32_vec_matrix_assumed_v1";
    mutations.push_back({"execution_profile", t});
  }
  for (const auto& mutation : mutations)
    require_semantic_difference(hw, base, mutation.task, mutation.what);

  const auto old_base = json{{"die_op_spec", old_ffn_spec(
      "batch_train_i1_1#s0", "batch_train_i1_1#s1", "p", "c")}};
  std::vector<Mutation> old_mutations;
  {
    json t = old_base; t["die_op_spec"]["spans"][0]["q_tokens"] = 2;
    old_mutations.push_back({"legacy span q_tokens", t});
  }
  {
    json t = old_base; t["die_op_spec"]["spans"][1]["kv_tokens"] = 3;
    old_mutations.push_back({"legacy span kv_tokens", t});
  }
  {
    json t = old_base; t["die_op_spec"]["layer_count"] = 4;
    old_mutations.push_back({"legacy layer_count", t});
  }
  {
    json t = old_base; t["die_op_spec"]["weight_passes"] = 3;
    old_mutations.push_back({"legacy weight_passes", t});
  }
  {
    json t = old_base; t["die_op_spec"]["kv_access"]["identity"] = "staged";
    old_mutations.push_back({"legacy kv_access.identity", t});
  }
  {
    json t = old_base; t["die_op_spec"]["tp_slice"]["tensor_parallel_rank"] = 1;
    old_mutations.push_back({"legacy tp rank", t});
  }
  {
    json t = old_base; t["die_op_spec"]["dtype"] = "fp32";
    old_mutations.push_back({"legacy dtype", t});
  }
  {
    json t = old_base; t["die_op_spec"]["phase"] = "prefill";
    old_mutations.push_back({"legacy phase", t});
  }
  {
    // hidden = num_q_heads * head_dim must stay consistent, so widen both.
    json t = old_base;
    t["die_op_spec"]["model_dims"]["hidden"] = 72;
    t["die_op_spec"]["model_dims"]["head_dim"] = 9;
    old_mutations.push_back({"legacy model dims", t});
  }
  for (const auto& mutation : old_mutations)
    require_semantic_difference(hw, old_base, mutation.task, mutation.what);
}

void test_inherited_profile_is_semantic() {
  json config = read_json(std::string(COARSE_DIE_SOURCE_DIR) +
                          "/configs/example_hardware.json");
  const Hardware int4_hw = resolve_hardware(config);
  config["performance_profile"]["execution_profile"] =
      "fp32_vec_matrix_assumed_v1";
  const Hardware fp32_hw = resolve_hardware(config);
  // The task pins its profile explicitly, so the hardware inheritance has
  // no room to matter: both hardwares fold the task onto the same key.
  const json pinned = v2_gemm_task();
  require(semantic_task_key(int4_hw, pinned) ==
              semantic_task_key(fp32_hw, pinned),
          "explicit execution_profile dominates hardware inheritance");
  // Without a pinned profile the inherited hardware profile is a real
  // compile input and must separate the keys.
  const json inheriting = json{{"op_kind", "gemm"},
                               {"tensor_shapes", {{"m", 3}, {"n", 5}, {"k", 7}}}};
  require(semantic_task_key(int4_hw, inheriting) !=
              semantic_task_key(fp32_hw, inheriting),
          "inherited execution_profile participates in the semantic key");
}

Hardware counter_hardware() {
  std::array<double, resource_count> rates{};
  rates[COMPUTE_GATE] = 4.0;
  rates[HBM0] = 100.0;
  rates[HBM1] = 100.0;
  return manual_hardware(rates);
}

std::shared_ptr<const Plan> compute_plan(const Hardware& hw,
                                         std::uint64_t ops, double gate_work) {
  Epoch epoch;
  epoch.name = "semantic_key_counter_probe";
  Activity compute = activity(1, Role::Compute, COMPUTE_GATE, gate_work);
  compute.work[HBM0] = gate_work;
  compute.int4_ops = ops;
  epoch.activities.push_back(compute);
  return manual_plan(hw, {epoch}, "semantic_key_counter_probe");
}

void require_zero_observation_counters(const json& observation,
                                       const std::string& what) {
  require(observation.at("next_event_calls") == 0 &&
              observation.at("advance_to_calls") == 0 &&
              observation.at("allocate_calls") == 0 &&
              observation.at("solver_pivots_total") == 0,
          what + ": a fresh runtime reports zero observation counters");
}

void require_path_invariant(const json& observation,
                            const std::string& what) {
  const json& paths = observation.at("arbitration_paths");
  require(paths.at("empty").is_number_unsigned() &&
              paths.at("singleton").is_number_unsigned() &&
              paths.at("packing").is_number_unsigned() &&
              paths.at("ratio").is_number_unsigned() &&
              paths.at("signed_lp").is_number_unsigned(),
          what + ": arbitration path counts stay unsigned integers");
  const std::uint64_t sum =
      paths.at("empty").get<std::uint64_t>() +
      paths.at("singleton").get<std::uint64_t>() +
      paths.at("packing").get<std::uint64_t>() +
      paths.at("ratio").get<std::uint64_t>() +
      paths.at("signed_lp").get<std::uint64_t>();
  require(sum == observation.at("allocate_calls").get<std::uint64_t>(),
          what + ": every arbitration lands in exactly one path bucket");
}

void drain_counting(Runtime& runtime, std::uint64_t& queries,
                    std::uint64_t& advances) {
  for (std::size_t guard = 0; guard < 4096; ++guard) {
    const auto next = runtime.next_event();
    ++queries;
    if (!next) return;
    runtime.advance_to(*next);
    ++advances;
  }
  throw std::runtime_error("semantic key counter fixture exceeded event bound");
}

void test_observation_counters_track_calls() {
  const Hardware hw = counter_hardware();
  Runtime runtime(hw);
  require_zero_observation_counters(runtime.observation_counters(),
                                    "fresh runtime");
  runtime.submit(compute_plan(hw, 1001, 4.0), owner(1, 1, 1), 0);
  std::uint64_t queries = 0, advances = 0;
  drain_counting(runtime, queries, advances);
  require(runtime.drained(), "single-activity fixture drains");
  const json observation = runtime.observation_counters();
  // submit() advances to the submit tick once; the drain loop then advances
  // once per returned event and queries one final time to observe nullopt.
  require(observation.at("next_event_calls").get<std::uint64_t>() == queries,
          "next_event_calls equals the exact query count");
  require(observation.at("advance_to_calls").get<std::uint64_t>() ==
              1 + advances,
          "advance_to_calls equals submit tick plus drain advances");
  require_path_invariant(observation, "single-owner drain");
  require(observation.at("arbitration_paths").at("singleton")
              .get<std::uint64_t>() >= 1,
          "single live activity arbitrates through the singleton path");

  // Two owners with distinct work contend for the same resources, so their
  // joint arbitration is not foldable to the singleton closed form and must
  // exercise a multi-variable path at least once while the per-call path
  // invariant holds. (Identical activities would be owner-symmetry mirrors
  // and legitimately arbitrate through the singleton path.)
  Runtime contended(hw);
  contended.submit(compute_plan(hw, 997, 4.0), owner(1, 1, 1), 0);
  contended.submit(compute_plan(hw, 998, 6.0), owner(2, 2, 2), 0);
  std::uint64_t contended_queries = 0, contended_advances = 0;
  drain_counting(contended, contended_queries, contended_advances);
  require(contended.drained(), "two-owner fixture drains");
  const json contended_observation = contended.observation_counters();
  require(contended_observation.at("next_event_calls")
              .get<std::uint64_t>() == contended_queries,
          "two-owner next_event_calls equals the exact query count");
  require(contended_observation.at("advance_to_calls")
              .get<std::uint64_t>() == 2 + contended_advances,
          "two-owner advance_to_calls equals both submit ticks plus drains");
  require_path_invariant(contended_observation, "two-owner drain");
  const json& contended_paths = contended_observation.at("arbitration_paths");
  require(contended_paths.at("packing").get<std::uint64_t>() +
                  contended_paths.at("ratio").get<std::uint64_t>() +
                  contended_paths.at("signed_lp").get<std::uint64_t>() >= 1,
          "two contending owners leave the singleton path at least once");

  // Observation counters are call-history facts, deliberately absent from
  // counters() and from the snapshot wire.
  require(!runtime.counters().contains("next_event_calls") &&
              !runtime.counters().contains("observation_counters"),
          "counters() keeps its wire-stable field set");
  const json snapshot = runtime.export_snapshot();
  require(!snapshot.contains("next_event_calls") &&
              !snapshot.contains("observation_counters") &&
              !snapshot.contains("solver_pivots_total"),
          "observation counters never enter the snapshot wire");
}

}  // namespace

int main() {
  test_key_matches_public_normalized_fingerprint();
  test_identity_only_changes_preserve_key();
  test_semantic_changes_break_key();
  test_inherited_profile_is_semantic();
  test_observation_counters_track_calls();
  std::printf("semantic_task_key_test passed\n");
  return 0;
}
