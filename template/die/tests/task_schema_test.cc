#include "coarse_die/Plan.hh"

#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {
using coarse_die::compile;
using json = nlohmann::json;

void require(bool value, const std::string& message) {
  if (!value) throw std::runtime_error(message);
}

coarse_die::Hardware hardware() {
  std::ifstream in(std::string(COARSE_DIE_SOURCE_DIR) + "/configs/example_hardware.json");
  json input;
  in >> input;
  return coarse_die::resolve_hardware(input);
}

void rejected(const coarse_die::Hardware& hw, const json& task,
              const std::string& path) {
  try {
    (void)compile(hw, task);
  } catch (const std::invalid_argument& e) {
    require(std::string(e.what()).find(path) != std::string::npos,
            "rejection must identify " + path + ": " + e.what());
    return;
  } catch (const std::exception& e) {
    throw std::runtime_error("unexpected rejection type for " + task.dump() + ": " + e.what());
  }
  throw std::runtime_error("invalid task unexpectedly compiled: " + task.dump());
}

void required_dimensions(const coarse_die::Hardware& hw) {
  const json span = json::array({{{"q_tokens", 3}, {"kv_tokens", 5}}});
  for (const char* op : {"gemm", "norm", "residual", "qkv_rope", "mlp_up",
                         "mlp_down", "mlp_activation", "o_proj", "logits",
                         "ffn_region", "attention", "attention_region", "qk",
                         "av", "scale_mask", "softmax"}) {
    rejected(hw, {{"op_kind", op}, {"tensor_shapes", json::object()},
                  {"spans", span}}, "task.tensor_shapes");
  }
  json norm{{"op_kind", "norm"}, {"tensor_shapes", {{"m", 3}, {"hidden", 8}}}};
  for (const char* key : {"m", "hidden"}) {
    json missing = norm;
    missing["tensor_shapes"].erase(key);
    rejected(hw, missing, "task.tensor_shapes");
    for (const json& value : {json(0), json(-1), json(1.5), json(true), json("8")}) {
      json invalid = norm;
      invalid["tensor_shapes"][key] = value;
      rejected(hw, invalid, std::string("task.tensor_shapes.") + key);
    }
  }
  norm["tensor_shapes"]["hiddden"] = 8;
  rejected(hw, norm, "task.tensor_shapes");
  rejected(hw, {{"op_kind", "unimplemented"}, {"tensor_shapes", json::object()}},
           "task.op_kind");
}

void aliases_and_tp(const coarse_die::Hardware& hw) {
  json gemm{{"op_kind", "gemm"}, {"tensor_shapes", {{"m", 3}, {"n", 8}, {"k", 9}}}};
  json aliases = gemm;
  aliases["tensor_shapes"] = {{"M", 3}, {"N", 8}, {"K", 9}};
  require(compile(hw, gemm)->fingerprint == compile(hw, aliases)->fingerprint,
          "dimension spelling cannot change the canonical GEMM plan");
  aliases["tensor_shapes"]["m"] = 4;
  rejected(hw, aliases, "task.tensor_shapes.m");

  json attention{{"op_kind", "attention"},
      {"tensor_shapes", {{"q_heads", 2}, {"kv_heads", 1}, {"head_dim", 2}}},
      {"spans", json::array({{{"q_tokens", 3}, {"kv_tokens", 5}}})},
      {"tp_slice", {{"tensor_parallel", 2}, {"tensor_parallel_rank", 0},
                    {"q_heads_local", 2}, {"kv_heads_local", 1}}},
      {"causal", false}};
  const auto canonical = compile(hw, attention);
  require(canonical->task["tp_slice"]["tensor_parallel_rank"] == 0,
          "rank zero is a valid explicit TP rank");
  require(canonical->summary["source_matrix_ops"].get<std::uint64_t>() == 240,
          "TP metadata must not divide actual local head dimensions again");
  aliases = attention;
  aliases["tensor_shapes"] = {{"num_q_heads", 2}, {"kv_heads_local", 1}, {"head_dim", 2}};
  require(compile(hw, aliases)->fingerprint == canonical->fingerprint,
          "head aliases must reach Mapping with one canonical identity");
  aliases["tensor_shapes"]["q_heads"] = 3;
  rejected(hw, aliases, "task.tensor_shapes.q_heads");
  aliases = attention;
  aliases["tp_slice"]["q_heads_local"] = 1;
  rejected(hw, aliases, "task.tp_slice.q_heads_local");
  aliases = attention;
  aliases["tensor_shapes"].erase("q_heads");
  rejected(hw, aliases, "task.tensor_shapes");
  for (const char* key : {"tensor_parallel", "tensor_parallel_rank"}) {
    aliases = attention;
    aliases["tp_slice"][key] = key == std::string("tensor_parallel") ? 0 : 2;
    rejected(hw, aliases, "task.tp_slice");
  }
  aliases = attention;
  aliases["tp_slice"]["typo"] = 1;
  rejected(hw, aliases, "task.tp_slice");
  aliases = attention;
  aliases["execution_profile"] = "int4_matrix_vec_fp32_assumed_v1";
  aliases["execution_profile_id"] = "fp32_vec_matrix_assumed_v1";
  rejected(hw, aliases, "task.execution_profile");
}

void span_rows_and_legacy(const coarse_die::Hardware& hw) {
  json norm{{"op_kind", "norm"}, {"tensor_shapes", {{"hidden", 8}}},
      {"spans", json::array({{{"q_tokens", 3}, {"kv_tokens", 5}, {"batch", 2}},
                             {{"q_tokens", 1}, {"kv_tokens", 7}}})}};
  const auto from_spans = compile(hw, norm);
  require(from_spans->task["tensor_shapes"]["m"] == 7 &&
          from_spans->summary["source_vec_ops"].get<std::uint64_t>() == 7 * 8 * 6,
          "all declared query rows determine standalone vector work");
  norm["tensor_shapes"]["m"] = 7;
  require(compile(hw, norm)->fingerprint == from_spans->fingerprint,
          "explicit and derived actual row dimensions have identical plans");
  norm["tensor_shapes"]["m"] = 1;
  rejected(hw, norm, "task.tensor_shapes.m");

  json attention{{"op_kind", "attention"},
      {"tensor_shapes", {{"q_heads", 2}, {"kv_heads", 1}, {"head_dim", 2}}},
      {"spans", json::array({{{"q_tokens", 3}, {"kv_tokens", 5},
                             {"q_start_pos", 2}, {"kv_start_pos", 0}}})}};
  const auto canonical = compile(hw, attention);
  json aliases = attention;
  aliases["spans"][0]["q_len"] = 3;
  aliases["spans"][0]["kv_len"] = 5;
  aliases["spans"][0]["query_start"] = 2;
  aliases["spans"][0]["kv_start"] = 0;
  require(compile(hw, aliases)->fingerprint == canonical->fingerprint,
          "agreeing span aliases retain the canonical plan");
  for (const auto& conflict : {std::pair{"q_len", json(4)},
                               std::pair{"q_len", json(3.0)},
                               std::pair{"kv_len", json(6)},
                               std::pair{"query_start", json(1)},
                               std::pair{"kv_start", json(1)}}) {
    aliases = attention;
    aliases["spans"][0][conflict.first] = conflict.second;
    rejected(hw, aliases, "task.spans");
  }

  std::ifstream in(std::string(COARSE_DIE_SOURCE_DIR) + "/configs/example_tasks.json");
  json examples;
  in >> examples;
  for (const auto& task : examples["specs"])
    require(compile(hw, task) != nullptr, "legacy example tasks must keep compiling");
}
}

int main() {
  try {
    const auto hw = hardware();
    required_dimensions(hw);
    aliases_and_tp(hw);
    span_rows_and_legacy(hw);
    std::cout << "task_schema_test passed\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
