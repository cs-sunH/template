/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

/*
spec_parse_test.cc -- strict DieOpSpec JSON parsing (contract §2; design doc
§4).

Part A: a minimal legal spec parses and every field lands with the right
value/enum.  Part B: a battery of illegal inputs is rejected with
DieModelError, including unknown keys (message must name the key), missing
required keys, bad enum spellings, q_tokens = 0, empty spans, wrong types
(floats/negative/strings where unsigned integers are required),
schema_version != 1, and the §4.2 shape relations (hidden ==
num_q_heads*head_dim, kv heads <= q heads, rank < tensor_parallel, local
shards <= global dimensions).

Build (standalone): cmake -S extern/die_model -B <build> && cmake --build
<build> --target spec_parse_test.  Exit code 0 on ALL PASS.
*/

#include "DieModel.hh"

#include <cstdio>
#include <string>

namespace {

bool g_ok = true;

void expect(bool cond, const char* what) {
  if (!cond) {
    std::fprintf(stderr, "[spec_parse_test] FAIL: %s\n", what);
    g_ok = false;
  }
}

nlohmann::json base_spec() {
  nlohmann::json span0;
  span0["request_id"] = "req-0";
  span0["q_tokens"] = 1024u;
  span0["kv_tokens"] = 1024u;
  span0["q_start_pos"] = 0u;
  span0["kv_start_pos"] = 0u;
  nlohmann::json span1;
  span1["request_id"] = "req-1";
  span1["q_tokens"] = 1u;
  span1["kv_tokens"] = 2048u;
  span1["q_start_pos"] = 512u;
  span1["kv_start_pos"] = 0u;

  nlohmann::json j;
  // Unsigned literals on purpose: the strict parser requires unsigned
  // storage (is_number_unsigned), which parsed JSON text always provides.
  j["schema_version"] = 1u;
  j["op_kind"] = "ffn_region";
  j["phase"] = "prefill";
  j["dtype"] = "bf16";
  j["model_dims"] = {{"hidden", 4096u},        {"ffn", 14336u},
                     {"vocab", 152064u},       {"mlp_variant", "swiglu"},
                     {"norm_type", "rmsnorm"}, {"position_encoding", "rope"},
                     {"num_q_heads", 32u},     {"num_kv_heads", 8u},
                     {"head_dim", 128u}};
  j["tp_slice"] = {{"tensor_parallel", 1u},       {"tensor_parallel_rank", 0u},
                   {"q_heads_local", 32u},        {"kv_heads_local", 8u},
                   {"ffn_local", 14336u},         {"vocab_local", 152064u}};
  j["spans"] = nlohmann::json::array({span0, span1});
  j["layer_count"] = 1u;
  j["weight_passes"] = 2u;
  j["kv_access"] = {{"identity", "local"}};
  j["fusion_policy"] = "auto";
  j["provenance"] = "spec_parse_test";
  j["config_hash"] = "deadbeef";
  return j;
}

// Expects parse rejection; optionally checks the message for a substring
// (field path or key name).
void expect_reject(const nlohmann::json& j, const char* what,
                   const char* message_contains = nullptr) {
  try {
    DieModel::parse_die_op_spec(j);
    std::fprintf(stderr,
                 "[spec_parse_test] FAIL: %s (accepted, expected "
                 "DieModelError)\n",
                 what);
    g_ok = false;
  } catch (const DieModel::DieModelError& e) {
    if (message_contains != nullptr &&
        std::string(e.what()).find(message_contains) == std::string::npos) {
      std::fprintf(stderr,
                   "[spec_parse_test] FAIL: %s (message \"%s\" lacks \"%s\")\n",
                   what, e.what(), message_contains);
      g_ok = false;
    }
  } catch (const std::exception& e) {
    std::fprintf(stderr,
                 "[spec_parse_test] FAIL: %s (wrong exception type: %s)\n",
                 what, e.what());
    g_ok = false;
  }
}

template <typename Mutate>
void expect_reject_base(const char* what, Mutate mutate,
                        const char* message_contains = nullptr) {
  nlohmann::json j = base_spec();
  mutate(j);
  expect_reject(j, what, message_contains);
}

}  // namespace

int main() {
  // --- Part A: legal minimal spec round-trips into the right fields ---
  std::shared_ptr<const DieModel::DieOpSpec> spec;
  try {
    spec = DieModel::parse_die_op_spec(base_spec());
  } catch (const std::exception& e) {
    std::fprintf(stderr, "[spec_parse_test] FAIL: base spec rejected: %s\n",
                 e.what());
    return 1;
  }
  expect(spec->schema_version == 1, "schema_version");
  expect(spec->op_kind == DieModel::OpKind::FfnRegion, "op_kind ffn_region");
  expect(spec->phase == DieModel::Phase::Prefill, "phase prefill");
  expect(spec->dtype == DieModel::Dtype::Bf16, "dtype bf16");
  expect(spec->model_dims.hidden == 4096, "hidden");
  expect(spec->model_dims.ffn == 14336, "ffn");
  expect(spec->model_dims.vocab == 152064, "vocab");
  expect(spec->model_dims.mlp_variant == DieModel::MlpVariant::Swiglu,
         "mlp_variant swiglu");
  expect(spec->model_dims.norm_type == DieModel::NormType::Rmsnorm,
         "norm_type rmsnorm");
  expect(spec->model_dims.position_encoding ==
             DieModel::PositionEncoding::Rope,
         "position_encoding rope");
  expect(spec->model_dims.num_q_heads == 32, "num_q_heads");
  expect(spec->model_dims.num_kv_heads == 8, "num_kv_heads");
  expect(spec->model_dims.head_dim == 128, "head_dim");
  expect(spec->tp_slice.tensor_parallel == 1, "tensor_parallel");
  expect(spec->tp_slice.tensor_parallel_rank == 0, "tensor_parallel_rank");
  expect(spec->tp_slice.q_heads_local == 32, "q_heads_local");
  expect(spec->tp_slice.kv_heads_local == 8, "kv_heads_local");
  expect(spec->tp_slice.ffn_local == 14336, "ffn_local");
  expect(spec->tp_slice.vocab_local == 152064, "vocab_local");
  expect(spec->spans.size() == 2, "two spans");
  expect(spec->spans[0].request_id == "req-0", "span0 request_id");
  expect(spec->spans[0].q_tokens == 1024, "span0 q_tokens");
  expect(spec->spans[1].kv_tokens == 2048, "span1 kv_tokens");
  expect(spec->spans[1].q_start_pos == 512, "span1 q_start_pos");
  expect(spec->spans[1].kv_start_pos == 0, "span1 kv_start_pos");
  expect(spec->layer_count == 1, "layer_count");
  expect(spec->weight_passes == 2, "weight_passes");
  expect(spec->kv_access.identity == DieModel::KvIdentity::Local,
         "kv_access local");
  expect(spec->fusion_policy == DieModel::FusionPolicy::Auto,
         "fusion_policy auto");
  expect(spec->provenance == "spec_parse_test", "provenance");
  expect(spec->config_hash == "deadbeef", "config_hash");

  // Enum spellings: every op_kind round-trips through to_string/from_string.
  const char* op_kinds[15] = {
      "norm",   "qkv_rope", "qk",     "scale_mask", "softmax",
      "av",     "o_proj",   "residual", "attention_region", "mlp_up",
      "mlp_activation", "mlp_down", "ffn_region", "logits", "local_region"};
  for (const char* k : op_kinds) {
    nlohmann::json j = base_spec();
    j["op_kind"] = k;
    try {
      auto s = DieModel::parse_die_op_spec(j);
      expect(std::string(DieModel::to_string(s->op_kind)) == k,
             "op_kind round-trip");
    } catch (const std::exception& e) {
      std::fprintf(stderr,
                   "[spec_parse_test] FAIL: legal op_kind %s rejected: %s\n",
                   k, e.what());
      g_ok = false;
    }
  }

  // --- Part B: illegal inputs, each must be rejected ---
  // 1. unknown top-level key (message names the key).
  expect_reject_base("unknown top-level key", [](nlohmann::json& j) {
    j["surprise"] = 1;
  }, "surprise");
  // 2. missing required key.
  expect_reject_base("missing spans", [](nlohmann::json& j) { j.erase("spans"); },
                     "spans");
  // 3. invalid enum spelling.
  expect_reject_base("op_kind gemm", [](nlohmann::json& j) {
    j["op_kind"] = "gemm";
  }, "op_kind");
  // 4. q_tokens = 0.
  expect_reject_base("q_tokens 0", [](nlohmann::json& j) {
    j["spans"][0]["q_tokens"] = 0u;
  }, "spans[0].q_tokens");
  // 5. empty spans array.
  expect_reject_base("empty spans", [](nlohmann::json& j) {
    j["spans"] = nlohmann::json::array();
  }, "at least one span");
  // 6. float where an unsigned integer is required.
  expect_reject_base("layer_count float", [](nlohmann::json& j) {
    j["layer_count"] = 2.0;
  }, "layer_count");
  // 7. schema_version != 1.
  expect_reject_base("schema_version 2", [](nlohmann::json& j) {
    j["schema_version"] = 2u;
  }, "schema_version");
  // 8. unknown key inside model_dims.
  expect_reject_base("unknown model_dims key", [](nlohmann::json& j) {
    j["model_dims"]["quant"] = "int8";
  }, "quant");
  // 9. tensor_parallel_rank out of range.
  expect_reject_base("rank == tensor_parallel", [](nlohmann::json& j) {
    j["tp_slice"]["tensor_parallel"] = 2u;
    j["tp_slice"]["tensor_parallel_rank"] = 2u;
  }, "tensor_parallel_rank");
  // 10. hidden != num_q_heads * head_dim (§2.1/§4.2 convention).
  expect_reject_base("hidden mismatch", [](nlohmann::json& j) {
    j["model_dims"]["hidden"] = 4097u;
  }, "num_q_heads * head_dim");
  // 11. negative value (signed integer, not unsigned).
  expect_reject_base("negative q_start_pos", [](nlohmann::json& j) {
    j["spans"][0]["q_start_pos"] = -1;
  }, "q_start_pos");
  // 12. string where an unsigned integer is required.
  expect_reject_base("kv_tokens string", [](nlohmann::json& j) {
    j["spans"][0]["kv_tokens"] = "1024";
  }, "kv_tokens");
  // 13. invalid kv_access.identity.
  expect_reject_base("kv identity everywhere", [](nlohmann::json& j) {
    j["kv_access"]["identity"] = "everywhere";
  }, "identity");
  // 14. weight_passes = 0.
  expect_reject_base("weight_passes 0", [](nlohmann::json& j) {
    j["weight_passes"] = 0u;
  }, "weight_passes");
  // 15. num_kv_heads > num_q_heads (GQA constraint, §2.1/§4.2).
  expect_reject_base("num_kv_heads > num_q_heads", [](nlohmann::json& j) {
    j["model_dims"]["num_kv_heads"] = 33u;
  }, "num_kv_heads");
  // 16. local shard larger than the global dimension.
  expect_reject_base("q_heads_local > num_q_heads", [](nlohmann::json& j) {
    j["tp_slice"]["q_heads_local"] = 33u;
  }, "q_heads_local");
  // 17. missing kv_access object.
  expect_reject_base("missing kv_access", [](nlohmann::json& j) {
    j.erase("kv_access");
  }, "kv_access");
  // 18. invalid dtype spelling.
  expect_reject_base("dtype fp16", [](nlohmann::json& j) {
    j["dtype"] = "fp16";
  }, "dtype");
  // 19. top level not an object.
  expect_reject(nlohmann::json::array({1, 2, 3}), "top-level array");
  // 20. request_id with a non-string type.
  expect_reject_base("request_id number", [](nlohmann::json& j) {
    j["spans"][0]["request_id"] = 7;
  }, "request_id");
  // 21. missing schema_version.
  expect_reject_base("missing schema_version",
                     [](nlohmann::json& j) { j.erase("schema_version"); },
                     "schema_version");
  // 22. unknown key inside a span element.
  expect_reject_base("unknown span key", [](nlohmann::json& j) {
    j["spans"][0]["causal"] = true;
  }, "causal");
  // 23. empty local shard is legal (zero work, §4.2 rule 5): sanity
  // counter-check that we are not over-rejecting.
  {
    nlohmann::json j = base_spec();
    j["tp_slice"]["tensor_parallel"] = 64u;
    j["tp_slice"]["tensor_parallel_rank"] = 63u;
    j["tp_slice"]["q_heads_local"] = 0u;
    j["tp_slice"]["kv_heads_local"] = 0u;
    j["tp_slice"]["ffn_local"] = 224u;
    try {
      auto s = DieModel::parse_die_op_spec(j);
      expect(s->tp_slice.q_heads_local == 0, "empty shard accepted");
    } catch (const std::exception& e) {
      std::fprintf(stderr,
                   "[spec_parse_test] FAIL: empty shard rejected: %s\n",
                   e.what());
      g_ok = false;
    }
  }

  if (g_ok) {
    std::printf("[spec_parse_test] ALL PASS\n");
    return 0;
  }
  return 1;
}
