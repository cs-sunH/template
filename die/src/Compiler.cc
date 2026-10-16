#include "coarse_die/Plan.hh"
#include "CompilerInternal.hh"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace coarse_die {
namespace detail {
std::vector<std::shared_ptr<Plan>> enumerate_mappings_for_compile(
    const Hardware& hardware, const nlohmann::json& normalized_task);
}
namespace {

using u64 = std::uint64_t;
using u128 = unsigned __int128;

[[noreturn]] void fail(const std::string& path, const std::string& message) {
  throw std::invalid_argument(path + ": " + message);
}

void reject_unknown(const nlohmann::json& j,
                    std::initializer_list<const char*> allowed,
                    const std::string& path) {
  if (!j.is_object()) fail(path, "expected an object");
  for (auto it = j.begin(); it != j.end(); ++it) {
    bool found = false;
    for (const char* key : allowed) {
      if (it.key() == key) {
        found = true;
        break;
      }
    }
    if (!found) fail(path, "unknown key \"" + it.key() + "\"");
  }
}

const nlohmann::json& required(const nlohmann::json& j,
                               const char* key,
                               const std::string& path) {
  const auto it = j.find(key);
  if (it == j.end()) fail(path, "missing required key \"" + std::string(key) + "\"");
  return *it;
}

u64 uint_value(const nlohmann::json& j, const std::string& path,
               bool allow_zero = true) {
  u64 value = 0;
  if (j.is_number_unsigned()) {
    value = j.get<u64>();
  } else if (j.is_number_integer()) {
    const std::int64_t signed_value = j.get<std::int64_t>();
    if (signed_value < 0) fail(path, "expected a non-negative integer");
    value = static_cast<u64>(signed_value);
  } else {
    fail(path, "expected an integer (floating-point values are rejected)");
  }
  if (!allow_zero && value == 0) fail(path, "must be >= 1");
  return value;
}

std::string string_value(const nlohmann::json& j, const std::string& path) {
  if (!j.is_string()) fail(path, "expected a string");
  return j.get<std::string>();
}

u128 checked_add(u128 a, u128 b, const std::string& path) {
  constexpr u128 maximum = ~u128{0};
  if (b > maximum - a) fail(path, "overflow");
  return a + b;
}

u128 checked_mul(u128 a, u128 b, const std::string& path) {
  constexpr u128 maximum = ~u128{0};
  if (a != 0 && b > maximum / a) fail(path, "overflow");
  return a * b;
}

u64 u64_checked(u128 value, const std::string& path) {
  if (value > std::numeric_limits<u64>::max()) fail(path, "overflow (exceeds uint64 output range)");
  return static_cast<u64>(value);
}

void require_range_end(u64 start, u64 length, const std::string& path) {
  if (length == 0) fail(path, "must be >= 1");
  if (static_cast<u128>(start) + static_cast<u128>(length) - 1 >
      std::numeric_limits<u64>::max()) {
    fail(path, "absolute interval overflows uint64");
  }
}

std::string inherited_execution_profile(const Hardware& hardware) {
  const auto pp = hardware.input.find("performance_profile");
  if (pp == hardware.input.end() || !pp->is_object()) return {};
  const auto it = pp->find("execution_profile");
  if (it == pp->end() || !it->is_string()) return {};
  return it->get<std::string>();
}

nlohmann::json dtype_object(const nlohmann::json& raw,
                            const std::string& path) {
  nlohmann::json out = {{"weights", "bf16"},
                        {"activations", "bf16"},
                        {"kv", "bf16"},
                        {"output", "bf16"},
                        {"parameters", "bf16"},
                        {"scores", "fp32"},
                        {"accumulator", "fp32"}};
  if (raw.is_string()) {
    const std::string dtype = raw.get<std::string>();
    for (const char* role : {"weights", "activations", "kv", "output", "parameters"}) {
      out[role] = dtype;
    }
    return out;
  }
  if (!raw.is_object()) fail(path, "expected a dtype string or role object");
  reject_unknown(raw, {"weights", "activations", "kv", "output", "parameters", "scores", "accumulator"}, path);
  for (auto it = raw.begin(); it != raw.end(); ++it) {
    const std::string dtype = string_value(it.value(), path + "." + it.key());
    if (dtype != "int4" && dtype != "bf16" && dtype != "fp32" && dtype != "int32") {
      fail(path + "." + it.key(), "unsupported dtype \"" + dtype + "\"");
    }
    out[it.key()] = dtype;
  }
  return out;
}

nlohmann::json normalize_old_spans(const nlohmann::json& raw) {
  const std::string path = "die_op_spec.spans";
  if (!raw.is_array() || raw.empty()) fail(path, "expected a non-empty array");
  nlohmann::json spans = nlohmann::json::array();
  for (std::size_t i = 0; i < raw.size(); ++i) {
    const auto& s = raw[i];
    const std::string sp = path + "[" + std::to_string(i) + "]";
    if (!s.is_object()) fail(sp, "expected object");
    if (s.contains("run")) {
      reject_unknown(s, {"request_id_prefix", "first_index", "run"}, sp);
      const auto& run = required(s, "run", sp);
      reject_unknown(run, {"q_tokens", "kv_first", "kv_stride", "count"}, sp + ".run");
      const u64 q = uint_value(required(run, "q_tokens", sp + ".run"), sp + ".run.q_tokens", false);
      const u64 kv = uint_value(required(run, "kv_first", sp + ".run"), sp + ".run.kv_first", false);
      const u64 stride = uint_value(required(run, "kv_stride", sp + ".run"), sp + ".run.kv_stride", false);
      const u64 count = uint_value(required(run, "count", sp + ".run"), sp + ".run.count", false);
      (void)string_value(required(s, "request_id_prefix", sp), sp + ".request_id_prefix");
      (void)uint_value(required(s, "first_index", sp), sp + ".first_index");
      if (kv < q) fail(sp + ".run", "kv_first must be >= q_tokens");
      if (static_cast<u128>(kv) + static_cast<u128>(count - 1) * stride >
          std::numeric_limits<u64>::max()) fail(sp + ".run", "expanded kv length overflows uint64");
      spans.push_back({{"q_tokens", q},
                       {"kv_tokens", kv},
                       {"kv_stride", stride},
                       {"count", count},
                       {"q_start_pos", kv - q},
                       {"q_start_stride", stride},
                       {"kv_start_pos", 0},
                       {"kv_start_stride", 0}});
    } else {
      reject_unknown(s, {"request_id", "q_tokens", "kv_tokens", "q_start_pos", "kv_start_pos"}, sp);
      (void)string_value(required(s, "request_id", sp), sp + ".request_id");
      const u64 q = uint_value(required(s, "q_tokens", sp), sp + ".q_tokens", false);
      const u64 kv = uint_value(required(s, "kv_tokens", sp), sp + ".kv_tokens", false);
      const u64 qs = uint_value(required(s, "q_start_pos", sp), sp + ".q_start_pos");
      const u64 ks = uint_value(required(s, "kv_start_pos", sp), sp + ".kv_start_pos");
      require_range_end(qs, q, sp + ".q_start_pos");
      require_range_end(ks, kv, sp + ".kv_start_pos");
      spans.push_back({{"q_tokens", q}, {"kv_tokens", kv}, {"kv_stride", 0},
                       {"count", 1}, {"q_start_pos", qs}, {"q_start_stride", 0},
                       {"kv_start_pos", ks}, {"kv_start_stride", 0}});
    }
  }
  return spans;
}

nlohmann::json normalize_old_spec(const nlohmann::json& spec,
                                  const nlohmann::json& wrapper,
                                  const Hardware& hardware) {
  const std::string path = "die_op_spec";
  reject_unknown(spec, {"schema_version", "op_kind", "phase", "dtype", "model_dims", "tp_slice", "spans", "layer_count", "weight_passes", "kv_access", "fusion_policy", "provenance", "config_hash"}, path);
  if (uint_value(required(spec, "schema_version", path), path + ".schema_version", false) != 1) {
    fail(path + ".schema_version", "unsupported legacy DieOpSpec version; expected 1");
  }
  const std::string op = string_value(required(spec, "op_kind", path), path + ".op_kind");
  static const std::set<std::string> old_ops = {"norm", "qkv_rope", "qk", "scale_mask", "softmax", "av", "o_proj", "residual", "attention_region", "mlp_up", "mlp_activation", "mlp_down", "ffn_region", "logits", "local_region"};
  if (!old_ops.count(op)) fail(path + ".op_kind", "unsupported legacy op_kind \"" + op + "\"");
  if (op == "local_region") fail(path + ".op_kind", "unsupported local_region: DieOpSpec v1 carries no typed mathematical graph");
  const std::string phase = string_value(required(spec, "phase", path), path + ".phase");
  if (phase != "prefill" && phase != "decode") fail(path + ".phase", "unsupported phase \"" + phase + "\"");
  const std::string legacy_dtype = string_value(required(spec, "dtype", path), path + ".dtype");
  if (legacy_dtype != "bf16" && legacy_dtype != "fp32") fail(path + ".dtype", "unsupported legacy dtype \"" + legacy_dtype + "\"");

  const auto& md = required(spec, "model_dims", path);
  reject_unknown(md, {"hidden", "ffn", "vocab", "mlp_variant", "norm_type", "position_encoding", "num_q_heads", "num_kv_heads", "head_dim"}, path + ".model_dims");
  const u64 hidden = uint_value(required(md, "hidden", path + ".model_dims"), path + ".model_dims.hidden", false);
  const u64 ffn = uint_value(required(md, "ffn", path + ".model_dims"), path + ".model_dims.ffn", false);
  const u64 vocab = uint_value(required(md, "vocab", path + ".model_dims"), path + ".model_dims.vocab", false);
  const u64 qheads = uint_value(required(md, "num_q_heads", path + ".model_dims"), path + ".model_dims.num_q_heads", false);
  const u64 kvheads = uint_value(required(md, "num_kv_heads", path + ".model_dims"), path + ".model_dims.num_kv_heads", false);
  const u64 head_dim = uint_value(required(md, "head_dim", path + ".model_dims"), path + ".model_dims.head_dim", false);
  const std::string mlp = string_value(required(md, "mlp_variant", path + ".model_dims"), path + ".model_dims.mlp_variant");
  const std::string norm = string_value(required(md, "norm_type", path + ".model_dims"), path + ".model_dims.norm_type");
  const std::string position = string_value(required(md, "position_encoding", path + ".model_dims"), path + ".model_dims.position_encoding");
  if (mlp != "swiglu" && mlp != "gelu") fail(path + ".model_dims.mlp_variant", "unsupported value \"" + mlp + "\"");
  if (norm != "rmsnorm" && norm != "layernorm") fail(path + ".model_dims.norm_type", "unsupported value \"" + norm + "\"");
  if (position != "rope" && position != "none") fail(path + ".model_dims.position_encoding", "unsupported value \"" + position + "\"");
  if (kvheads > qheads) fail(path + ".model_dims.num_kv_heads", "must be <= num_q_heads");
  if (checked_mul(qheads, head_dim, path + ".model_dims") != hidden) fail(path + ".model_dims", "hidden must equal num_q_heads * head_dim");

  const auto& tp = required(spec, "tp_slice", path);
  reject_unknown(tp, {"tensor_parallel", "tensor_parallel_rank", "q_heads_local", "kv_heads_local", "ffn_local", "vocab_local"}, path + ".tp_slice");
  const u64 tp_degree = uint_value(required(tp, "tensor_parallel", path + ".tp_slice"), path + ".tp_slice.tensor_parallel", false);
  const u64 tp_rank = uint_value(required(tp, "tensor_parallel_rank", path + ".tp_slice"), path + ".tp_slice.tensor_parallel_rank");
  if (tp_rank >= tp_degree) fail(path + ".tp_slice.tensor_parallel_rank", "must be < tensor_parallel");
  const u64 qlocal = uint_value(required(tp, "q_heads_local", path + ".tp_slice"), path + ".tp_slice.q_heads_local");
  const u64 kvlocal = uint_value(required(tp, "kv_heads_local", path + ".tp_slice"), path + ".tp_slice.kv_heads_local");
  const u64 flocal = uint_value(required(tp, "ffn_local", path + ".tp_slice"), path + ".tp_slice.ffn_local");
  const u64 vlocal = uint_value(required(tp, "vocab_local", path + ".tp_slice"), path + ".tp_slice.vocab_local");
  if (qlocal > qheads || kvlocal > kvheads || flocal > ffn || vlocal > vocab) fail(path + ".tp_slice", "local shard exceeds its global dimension");
  const std::string fusion = string_value(required(spec, "fusion_policy", path), path + ".fusion_policy");
  if (fusion != "auto" && fusion != "none") fail(path + ".fusion_policy", "expected auto or none");
  const auto& kv_access = required(spec, "kv_access", path);
  reject_unknown(kv_access, {"identity"}, path + ".kv_access");
  const std::string kv_identity = string_value(required(kv_access, "identity", path + ".kv_access"), path + ".kv_access.identity");
  if (kv_identity != "local" && kv_identity != "staged" && kv_identity != "remote_ready") fail(path + ".kv_access.identity", "unsupported identity \"" + kv_identity + "\"");

  const u64 layers = uint_value(required(spec, "layer_count", path), path + ".layer_count", false);
  const u64 passes = uint_value(required(spec, "weight_passes", path), path + ".weight_passes", false);
  nlohmann::json shapes = {{"m", 0}, {"hidden", hidden}, {"ffn", flocal}, {"vocab", vlocal},
                           {"q_heads", qlocal}, {"kv_heads", kvlocal}, {"head_dim", head_dim},
                           {"mlp_variant", mlp}, {"norm_type", norm}, {"position_encoding", position}};
  auto spans = normalize_old_spans(required(spec, "spans", path));
  u128 m = 0;
  for (const auto& span : spans) {
    m = checked_add(m, checked_mul(uint_value(span.at("q_tokens"), "span.q_tokens"),
                                   uint_value(span.at("count"), "span.count"), path + ".spans"), path + ".spans");
  }
  shapes["m"] = u64_checked(m, path + ".spans");
  if (m == 0) fail(path + ".spans", "total query tokens must be non-zero");
  nlohmann::json dtypes = dtype_object(legacy_dtype, path + ".dtype");
  nlohmann::json normalized = {
      {"schema", "coarse_die_task_v2"},
      {"op_kind", op},
      {"phase", phase},
      {"tensor_shapes", std::move(shapes)},
      {"spans", std::move(spans)},
      {"tp_slice", {{"tensor_parallel", tp_degree}, {"tensor_parallel_rank", tp_rank},
                     {"q_heads_local", qlocal}, {"kv_heads_local", kvlocal},
                     {"ffn_local", flocal}, {"vocab_local", vlocal}}},
      {"storage_dtype", std::move(dtypes)},
      {"execution_profile", inherited_execution_profile(hardware)},
      {"layer_instances", layers},
      {"weight_passes", passes},
      {"fusion_policy", fusion},
      {"kv_identity", kv_identity}};
  for (const char* key : {"residency", "readiness_scope", "graph_identity"}) {
    const auto it = wrapper.find(key);
    if (it != wrapper.end()) normalized[key] = it.value();
  }
  return normalized;
}

void check_uint_aliases(const nlohmann::json& object,
                       std::initializer_list<const char*> keys,
                       const std::string& path, bool allow_zero = false) {
  bool present = false;
  u64 value = 0;
  for (const char* key : keys) {
    if (!object.contains(key)) continue;
    const u64 candidate = uint_value(object.at(key), path + "." + key, allow_zero);
    if (present && candidate != value) fail(path, "conflicting integer aliases");
    value = candidate;
    present = true;
  }
}

nlohmann::json normalize_new_spans(const nlohmann::json& raw) {
  const std::string path = "task.spans";
  if (!raw.is_array()) fail(path, "expected an array");
  nlohmann::json spans = nlohmann::json::array();
  for (std::size_t i = 0; i < raw.size(); ++i) {
    const auto& s = raw[i];
    const std::string sp = path + "[" + std::to_string(i) + "]";
    if (!s.is_object()) fail(sp, "expected object");
    const bool compact = s.contains("run");
    if (compact) {
      reject_unknown(s, {"run", "batch", "request_id_prefix", "first_index"}, sp);
      const auto& run = required(s, "run", sp);
      reject_unknown(run, {"q_len", "q_tokens", "kv_len", "kv_tokens", "kv_first", "kv_stride", "q_start_pos", "kv_start_pos", "count"}, sp + ".run");
      check_uint_aliases(run, {"q_len", "q_tokens"}, sp + ".run");
      check_uint_aliases(run, {"kv_len", "kv_tokens", "kv_first"}, sp + ".run");
      const char* qkey = run.contains("q_len") ? "q_len" : "q_tokens";
      const char* kvkey = run.contains("kv_len") ? "kv_len" : (run.contains("kv_first") ? "kv_first" : "kv_tokens");
      const u64 q = uint_value(required(run, qkey, sp + ".run"), sp + ".run." + qkey, false);
      const u64 kv = uint_value(required(run, kvkey, sp + ".run"), sp + ".run." + kvkey, false);
      const u64 stride = run.contains("kv_stride") ? uint_value(run.at("kv_stride"), sp + ".run.kv_stride") : 0;
      const u64 count = uint_value(required(run, "count", sp + ".run"), sp + ".run.count", false);
      const u64 batch = s.contains("batch") ? uint_value(s.at("batch"), sp + ".batch", false) : 1;
      const u64 qs = run.contains("q_start_pos") ? uint_value(run.at("q_start_pos"), sp + ".run.q_start_pos") : (kv >= q ? kv - q : 0);
      const u64 ks = run.contains("kv_start_pos") ? uint_value(run.at("kv_start_pos"), sp + ".run.kv_start_pos") : 0;
      require_range_end(qs, q, sp + ".run.q_start_pos");
      require_range_end(ks, kv, sp + ".run.kv_start_pos");
      if (static_cast<u128>(kv) + static_cast<u128>(count - 1) * stride > std::numeric_limits<u64>::max()) fail(sp + ".run", "kv length sequence overflows uint64");
      if (static_cast<u128>(qs) + static_cast<u128>(count - 1) * stride + q - 1 >
          std::numeric_limits<u64>::max()) fail(sp + ".run", "query position sequence overflows uint64");
      (void)u64_checked(checked_mul(count, batch, sp + ".count"), sp + ".count");
      nlohmann::json normalized = {{"q_tokens", q}, {"kv_tokens", kv}, {"kv_stride", stride},
                                   {"count", count}, {"q_start_pos", qs},
                                   {"q_start_stride", stride}, {"kv_start_pos", ks},
                                   {"kv_start_stride", 0}};
      // Keep the run's arithmetic progression intact. Batch repeats those
      // same request shapes; it does not extend the KV/query position series.
      // Omitting the default preserves existing batch=1 plan identities.
      if (batch > 1) normalized["batch"] = batch;
      spans.push_back(std::move(normalized));
      continue;
    }
    reject_unknown(s, {"batch", "q_len", "q_tokens", "kv_len", "kv_tokens", "q_start_pos", "query_start", "kv_start_pos", "kv_start", "count", "request_id"}, sp);
    check_uint_aliases(s, {"q_len", "q_tokens"}, sp);
    check_uint_aliases(s, {"kv_len", "kv_tokens"}, sp);
    check_uint_aliases(s, {"q_start_pos", "query_start"}, sp, true);
    check_uint_aliases(s, {"kv_start_pos", "kv_start"}, sp, true);
    check_uint_aliases(s, {"batch", "count"}, sp);
    const char* qkey = s.contains("q_len") ? "q_len" : "q_tokens";
    const char* kvkey = s.contains("kv_len") ? "kv_len" : "kv_tokens";
    const u64 q = uint_value(required(s, qkey, sp), sp + "." + qkey, false);
    const u64 kv = uint_value(required(s, kvkey, sp), sp + "." + kvkey, false);
    const char* qskey = s.contains("q_start_pos") ? "q_start_pos" : "query_start";
    const char* kskey = s.contains("kv_start_pos") ? "kv_start_pos" : "kv_start";
    const u64 qs = s.contains(qskey) ? uint_value(s.at(qskey), sp + "." + qskey) : (kv >= q ? kv - q : 0);
    const u64 ks = s.contains(kskey) ? uint_value(s.at(kskey), sp + "." + kskey) : 0;
    const u64 batch = s.contains("batch") ? uint_value(s.at("batch"), sp + ".batch", false) :
                      (s.contains("count") ? uint_value(s.at("count"), sp + ".count", false) : 1);
    if (s.contains("request_id")) (void)string_value(s.at("request_id"), sp + ".request_id");
    require_range_end(qs, q, sp + ".q_start_pos");
    require_range_end(ks, kv, sp + ".kv_start_pos");
    spans.push_back({{"q_tokens", q}, {"kv_tokens", kv}, {"kv_stride", 0},
                     {"count", batch}, {"q_start_pos", qs}, {"q_start_stride", 0},
                     {"kv_start_pos", ks}, {"kv_start_stride", 0}});
  }
  return spans;
}

void normalize_shape_alias(nlohmann::json& shapes, const char* canonical,
                           std::initializer_list<const char*> aliases) {
  bool present = false;
  u64 value = 0;
  for (const char* key : aliases) {
    if (!shapes.contains(key)) continue;
    const u64 candidate = uint_value(shapes.at(key),
        std::string("task.tensor_shapes.") + key, false);
    if (present && candidate != value)
      fail(std::string("task.tensor_shapes.") + canonical,
           "conflicting dimension aliases");
    value = candidate;
    present = true;
  }
  if (!present) return;
  for (const char* key : aliases) shapes.erase(key);
  shapes[canonical] = value;
}

nlohmann::json normalize_new_task(const nlohmann::json& raw,
                                  const Hardware& hardware) {
  reject_unknown(raw, {"schema", "op_kind", "region_ops", "phase", "tensor_shapes", "spans", "tp_slice", "storage_dtype", "execution_profile", "layer_instances", "weight_passes", "fusion_policy", "residency", "readiness_scope", "graph_identity", "causal", "math", "execution_profile_id"}, "task");
  if (raw.contains("schema") && string_value(raw.at("schema"), "task.schema") != "coarse_die_task_v2") fail("task.schema", "expected coarse_die_task_v2");
  if (raw.contains("region_ops")) fail("task.region_ops", "untyped region_ops are unsupported; provide a supported op_kind/typed region");
  const std::string raw_op = string_value(required(raw, "op_kind", "task"), "task.op_kind");
  const std::string op = raw_op == "swiglu" ? "ffn_region" : raw_op;
  const std::set<std::string> supported_ops = {
      "gemm", "norm", "residual", "mlp_activation", "scale_mask", "softmax",
      "qkv_rope", "mlp_up", "mlp_down", "o_proj", "logits", "ffn_region",
      "attention", "attention_region", "qk", "av"};
  if (!supported_ops.count(op)) fail("task.op_kind", "unsupported operator \"" + op + "\"");
  const std::string phase = raw.contains("phase") ? string_value(raw.at("phase"), "task.phase") : "prefill";
  if (phase != "prefill" && phase != "decode" && phase != "kv_recompute") fail("task.phase", "unsupported phase \"" + phase + "\"");
  const auto& shape_raw = required(raw, "tensor_shapes", "task");
  if (!shape_raw.is_object()) fail("task.tensor_shapes", "expected object");
  reject_unknown(shape_raw, {"m", "M", "n", "N", "k", "K", "hidden", "h",
      "q_heads", "num_q_heads", "q_heads_local", "kv_heads", "num_kv_heads",
      "kv_heads_local", "head_dim", "ffn", "ffn_local", "vocab", "vocab_local",
      "norm_type", "mlp_variant", "position_encoding"}, "task.tensor_shapes");
  nlohmann::json shapes = shape_raw;
  normalize_shape_alias(shapes, "m", {"m", "M"});
  normalize_shape_alias(shapes, "n", {"n", "N"});
  normalize_shape_alias(shapes, "k", {"k", "K"});
  normalize_shape_alias(shapes, "hidden", {"hidden", "h"});
  normalize_shape_alias(shapes, "q_heads", {"q_heads", "num_q_heads", "q_heads_local"});
  normalize_shape_alias(shapes, "kv_heads", {"kv_heads", "num_kv_heads", "kv_heads_local"});
  normalize_shape_alias(shapes, "ffn", {"ffn", "ffn_local"});
  normalize_shape_alias(shapes, "vocab", {"vocab", "vocab_local"});
  normalize_shape_alias(shapes, "head_dim", {"head_dim"});
  if (raw_op == "swiglu") {
    if (shapes.contains("mlp_variant") && shapes.at("mlp_variant") != "swiglu")
      fail("task.tensor_shapes.mlp_variant", "conflicts with op_kind=swiglu");
    shapes["mlp_variant"] = "swiglu";
  }
  const u64 layers = raw.contains("layer_instances") ? uint_value(raw.at("layer_instances"), "task.layer_instances", false) : 1;
  const u64 passes = raw.contains("weight_passes") ? uint_value(raw.at("weight_passes"), "task.weight_passes", false) : 1;
  const std::string profile = raw.contains("execution_profile") ? string_value(raw.at("execution_profile"), "task.execution_profile") :
                              (raw.contains("execution_profile_id") ? string_value(raw.at("execution_profile_id"), "task.execution_profile_id") : inherited_execution_profile(hardware));
  if (raw.contains("execution_profile_id") &&
      string_value(raw.at("execution_profile_id"), "task.execution_profile_id") != profile)
    fail("task.execution_profile", "conflicts with execution_profile_id");
  if (profile != "int4_matrix_vec_fp32_assumed_v1" && profile != "fp32_vec_matrix_assumed_v1") fail("task.execution_profile", "unsupported execution profile \"" + profile + "\"");
  nlohmann::json dtypes = raw.contains("storage_dtype") ? dtype_object(raw.at("storage_dtype"), "task.storage_dtype") : dtype_object("bf16", "task.storage_dtype");
  nlohmann::json tp = raw.contains("tp_slice") ? raw.at("tp_slice") : nlohmann::json::object();
  reject_unknown(tp, {"tensor_parallel", "tensor_parallel_rank", "q_heads_local",
      "kv_heads_local", "ffn_local", "vocab_local"}, "task.tp_slice");
  for (const char* key : {"tensor_parallel", "tensor_parallel_rank", "q_heads_local", "kv_heads_local", "ffn_local", "vocab_local"}) {
    if (tp.contains(key)) tp[key] = uint_value(tp.at(key), std::string("task.tp_slice.") + key, std::string(key) != "tensor_parallel");
  }
  for (const auto& pair : {std::pair{"q_heads", "q_heads_local"},
                          std::pair{"kv_heads", "kv_heads_local"},
                          std::pair{"ffn", "ffn_local"},
                          std::pair{"vocab", "vocab_local"}}) {
    if (!shapes.contains(pair.first)) continue;
    if (tp.contains(pair.second) && tp.at(pair.second) != shapes.at(pair.first))
      fail(std::string("task.tp_slice.") + pair.second,
           "must match the actual local tensor_shapes dimension");
    tp[pair.second] = shapes.at(pair.first);
  }
  if (!tp.contains("tensor_parallel")) tp["tensor_parallel"] = 1;
  if (!tp.contains("tensor_parallel_rank")) tp["tensor_parallel_rank"] = 0;
  if (uint_value(tp.at("tensor_parallel"), "task.tp_slice.tensor_parallel", false) <= uint_value(tp.at("tensor_parallel_rank"), "task.tp_slice.tensor_parallel_rank")) fail("task.tp_slice", "tensor_parallel_rank must be less than tensor_parallel");
  const nlohmann::json spans = raw.contains("spans") ? normalize_new_spans(raw.at("spans")) : nlohmann::json::array();
  if ((op == "attention" || op == "attention_region" || op == "qk" || op == "av" || op == "scale_mask" || op == "softmax") && spans.empty()) fail("task.spans", "attention requires at least one span");
  const bool row_op = op != "gemm" && op != "attention" &&
      op != "attention_region" && op != "qk" && op != "av";
  if (row_op && !spans.empty()) {
    u128 rows = 0;
    for (const auto& span : spans) {
      const u128 span_rows = checked_mul(span.at("q_tokens").get<u64>(),
          span.at("count").get<u64>(), "task.spans query rows");
      const u64 batch = span.contains("batch") ? span.at("batch").get<u64>() : 1;
      rows = checked_add(rows, checked_mul(span_rows, batch,
          "task.spans query rows"), "task.spans query rows");
    }
    const u64 local_rows = u64_checked(rows, "task.spans query rows");
    if (shapes.contains("m") && shapes.at("m") != local_rows)
      fail("task.tensor_shapes.m", "must match the query rows declared by spans");
    shapes["m"] = local_rows;
  }
  const auto need = [&](std::initializer_list<const char*> keys) {
    for (const char* key : keys)
      (void)uint_value(required(shapes, key, "task.tensor_shapes"),
                       std::string("task.tensor_shapes.") + key, false);
  };
  if (op == "gemm") need({"m", "n", "k"});
  else if (op == "norm" || op == "residual" || op == "o_proj") need({"m", "hidden"});
  else if (op == "mlp_activation") need({"m", "ffn"});
  else if (op == "mlp_up" || op == "mlp_down" || op == "ffn_region") need({"m", "hidden", "ffn"});
  else if (op == "logits") need({"m", "hidden", "vocab"});
  else if (op == "qkv_rope") need({"m", "hidden", "q_heads", "kv_heads", "head_dim"});
  else if (op == "scale_mask" || op == "softmax") need({"q_heads"});
  else need({"q_heads", "kv_heads", "head_dim"});
  for (const auto& variant : {std::pair{"norm_type", std::set<std::string>{"rmsnorm", "layernorm"}},
                             std::pair{"mlp_variant", std::set<std::string>{"swiglu", "gelu"}},
                             std::pair{"position_encoding", std::set<std::string>{"rope", "none"}}}) {
    if (shapes.contains(variant.first) &&
        !variant.second.count(string_value(shapes.at(variant.first), std::string("task.tensor_shapes.") + variant.first)))
      fail(std::string("task.tensor_shapes.") + variant.first, "unsupported variant");
  }
  if (raw.contains("causal") && !raw.at("causal").is_boolean())
    fail("task.causal", "expected boolean");
  const std::string fusion = raw.contains("fusion_policy") ? string_value(raw.at("fusion_policy"), "task.fusion_policy") : "auto";
  if (fusion != "auto" && fusion != "none" && fusion != "materialized" && fusion != "fused") fail("task.fusion_policy", "unsupported fusion policy \"" + fusion + "\"");
  nlohmann::json normalized = {{"schema", "coarse_die_task_v2"}, {"op_kind", op}, {"phase", phase},
                               {"tensor_shapes", std::move(shapes)}, {"spans", spans}, {"tp_slice", tp},
                               {"storage_dtype", dtypes}, {"execution_profile", profile},
                               {"layer_instances", layers}, {"weight_passes", passes},
                               {"fusion_policy", fusion}};
  for (const char* key : {"residency", "readiness_scope", "graph_identity", "causal", "math"}) {
    const auto it = raw.find(key);
    if (it != raw.end()) normalized[key] = it.value();
  }
  return normalized;
}

nlohmann::json normalize_task_impl(const Hardware& hardware,
                                   const nlohmann::json& task) {
  if (!task.is_object()) fail("task", "expected a JSON object");
  if (task.contains("die_op_spec")) {
    const auto& spec = task.at("die_op_spec");
    return normalize_old_spec(spec, task, hardware);
  }
  if (task.contains("model_dims") && task.contains("schema_version")) {
    return normalize_old_spec(task, task, hardware);
  }
  return normalize_new_task(task, hardware);
}

u64 summary_u64(const Plan& plan, const char* key) {
  const auto it = plan.summary.find(key);
  if (it == plan.summary.end()) return 0;
  return uint_value(*it, std::string("summary.") + key);
}

std::string summary_string(const Plan& plan, const char* key) {
  const auto it = plan.summary.find(key);
  return it != plan.summary.end() && it->is_string() ? it->get<std::string>() : std::string();
}

}  // namespace

namespace detail {

nlohmann::json normalize_task(const Hardware& hardware,
                              const nlohmann::json& task) {
  return normalize_task_impl(hardware, task);
}

}  // namespace detail

WorkspaceEnvelope compile_workspace_envelope(const Hardware& hardware,
                                              const nlohmann::json& task) {
  const nlohmann::json normalized = detail::normalize_task(hardware, task);
  std::vector<std::shared_ptr<Plan>> candidates =
      detail::enumerate_mappings_for_compile(hardware, normalized);
  if (candidates.empty()) {
    fail("compile", "unsupported_workspace: no feasible fixed mapping candidate");
  }
  if (candidates.size() > 256) fail("compile", "candidate set exceeds fixed upper bound 256");

  const std::string expected_hardware = hardware.fingerprint.empty()
      ? stable_hash(hardware.input.dump()) : hardware.fingerprint;
  const u64 capacity_upper=
      detail::workspace_capacity_upper_bound(hardware,normalized);
  u64 maximum = 0;
  for (const auto& candidate : candidates) {
    if (!candidate || candidate->task != normalized ||
        candidate->hardware_fingerprint != expected_hardware) {
      fail("compile", "invalid_plan: mapping candidate identity is incomplete");
    }
    maximum = std::max(maximum, hbm_workspace_peak_bytes(*candidate));
  }
  if(maximum>capacity_upper)
    fail("compile", "mapping_invariant: candidate workspace exceeds the all-template lifetime bound");

  WorkspaceEnvelope result;
  result.candidate_count = static_cast<u64>(candidates.size());
  result.max_hbm_workspace_peak_bytes = maximum;
  result.hardware_fingerprint = expected_hardware;
  result.normalized_task_fingerprint = stable_hash(normalized.dump());
  result.op_kind = normalized.at("op_kind").get<std::string>();
  result.phase = normalized.at("phase").get<std::string>();
  result.normalized_task = normalized;
  return result;
}

WorkspaceCapacityBound compile_workspace_capacity_bound(
    const Hardware& hardware,const nlohmann::json& task) {
  const nlohmann::json normalized=detail::normalize_task(hardware,task);
  WorkspaceCapacityBound result;
  result.hbm_workspace_upper_bound_bytes=
      detail::workspace_capacity_upper_bound(hardware,normalized);
  result.hardware_fingerprint=hardware.fingerprint.empty()
      ?stable_hash(hardware.input.dump()):hardware.fingerprint;
  result.normalized_task_fingerprint=stable_hash(normalized.dump());
  result.op_kind=normalized.at("op_kind").get<std::string>();
  result.phase=normalized.at("phase").get<std::string>();
  result.normalized_task=normalized;
  return result;
}

namespace {

std::shared_ptr<const Plan> compile_normalized_impl(
    const Hardware& hardware, const nlohmann::json& normalized) {
  if (!normalized.is_object() || !normalized.contains("op_kind")) {
    fail("task", "normalized task must be an object with op_kind");
  }
  std::vector<std::shared_ptr<Plan>> candidates =
      detail::enumerate_mappings_for_compile(hardware, normalized);
  if (candidates.empty()) {
    fail("compile", "unsupported_workspace: no feasible fixed mapping candidate");
  }
  if (candidates.size() > 256) fail("compile", "candidate set exceeds fixed upper bound 256");
  const auto immutable_hardware = std::make_shared<const Hardware>(hardware);
  const bool gemm_candidates_are_unsealed =
      normalized.at("op_kind").get<std::string>() == "gemm";

  struct Ranked {
    std::shared_ptr<Plan> plan;
    u64 latency = 0;
    u128 hbm = 0;
    u64 workspace = 0;
    u64 sram = 0;
    std::string template_id;
    u64 tm = 0, tn = 0, tk = 0, th = 0;
  };
  std::vector<Ranked> ranked;
  ranked.reserve(candidates.size());
  const std::string expected_hardware=hardware.fingerprint.empty()
      ? stable_hash(hardware.input.dump()) : hardware.fingerprint;
  for (auto& candidate : candidates) {
    // The compile-only GEMM enumerator defers candidate fingerprints because
    // losing candidates never leave this ranking pass. The dedicated Runtime
    // estimate path still performs all plan, hardware, work, and credit checks
    // while allowing only that internal trial to omit a not-yet-needed hash.
    // Public enumerate_mappings() continues to return fully sealed Plans.
    if(candidate->task != normalized || candidate->hardware_fingerprint != expected_hardware ||
       (gemm_candidates_are_unsealed ? !candidate->fingerprint.empty()
                                     : candidate->fingerprint.empty()))
      fail("compile", "invalid_plan: mapping candidate identity is incomplete");
    const u64 latency = gemm_candidates_are_unsealed
        ? detail::estimate_candidate_latency_ns(immutable_hardware, candidate)
        : detail::estimate_latency_ns(immutable_hardware, candidate);
    // Preserve status type validation for every candidate. Losing builders
    // never escape, so only the winner needs these report fields inserted.
    (void)candidate->summary.value("prediction_status",std::string("ok_assumed"));
    // These are still private mutable builders. Seal only the winner after
    // ranking; losing candidates never escape or enter a cache/runtime again.
    Ranked r;
    r.plan = candidate;
    r.latency = latency;
    r.hbm = checked_add(summary_u64(*candidate, "hbm_read_bytes_total"),
                        summary_u64(*candidate, "hbm_write_bytes_total"),
                        "candidate HBM payload");
    r.workspace = summary_u64(*candidate, "hbm_workspace_peak_bytes");
    r.sram = summary_u64(*candidate, "sram_peak_bytes_per_core");
    r.template_id = summary_string(*candidate, "template_id");
    r.tm = summary_u64(*candidate, "tm");
    r.tn = summary_u64(*candidate, "tn");
    r.tk = summary_u64(*candidate, "tk");
    r.th = summary_u64(*candidate, "th");
    ranked.push_back(std::move(r));
  }
  const auto key = [](const Ranked& r) {
    return std::tie(r.latency, r.hbm, r.workspace, r.sram,
                    r.template_id, r.tm, r.tn, r.tk, r.th);
  };
  // min_element retains the first equal minimum, exactly like the previous
  // stable_sort followed by front(), without reordering losing candidates.
  const auto winner=std::min_element(ranked.begin(),ranked.end(),[&](const Ranked& a,const Ranked& b) {
    return key(a)<key(b);
  });
  std::shared_ptr<Plan> selected=winner->plan;
  if(gemm_candidates_are_unsealed)
    selected=detail::materialize_gemm_compile_candidate(hardware,*selected);
  selected->summary["estimated_latency_ns"]=winner->latency;
  selected->summary["alone_estimate_status"]=
      selected->summary.value("prediction_status",std::string("ok_assumed"));
  selected->summary["candidate_count"] = static_cast<u64>(ranked.size());
  selected->summary["selected_template"] = selected->summary.value("template_id", std::string());
  selected->summary["selection_rule"] = "ceil(estimate_alone_ns),hbm_payload_bytes,hbm_workspace_peak_bytes,sram_peak_bytes_per_core,template_id,tm,tn,tk,th";
  selected->summary["candidate_estimate_source"] = "estimate_alone(shared_runtime_engine)";
  selected->summary["activity_work_semantics"] = "activity work/bytes/ops are epoch total across repeat_count; repeat_count is audit metadata only; credit_delta is total signed credit change over progress 0..1";
  selected->summary["granularity_validation"] = "not_run";
  selected->summary["evidence_flags"] = {"engineering_assumption", "hardware_source_derived"};
  if(gemm_candidates_are_unsealed)detail::finalize_region_overhead_audit(hardware,*selected);
  selected->fingerprint.clear();
  selected->fingerprint = plan_to_json(*selected).at("fingerprint").get<std::string>();
  return selected;
}

}  // namespace

namespace detail {

std::shared_ptr<const Plan> compile_normalized(
    const Hardware& hardware, const nlohmann::json& normalized_task) {
  return compile_normalized_impl(hardware, normalized_task);
}

}  // namespace detail

std::shared_ptr<const Plan> compile(const Hardware& hardware,
                                   const nlohmann::json& task) {
  const nlohmann::json normalized = detail::normalize_task(hardware, task);
  return detail::compile_normalized(hardware, normalized);
}

}  // namespace coarse_die
