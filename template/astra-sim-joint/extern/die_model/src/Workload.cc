/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

/*
Workload.cc -- DieOpSpec JSON strict parsing (contract §2; design doc §4) and
the §4.2 workload determination (derive_workload).

Stage A1 delivered parse_die_op_spec and the shared strict-JSON helpers;
stage A2 adds derive_workload in this same file.
*/

#include "DieModel.hh"

#include <string>
#include <vector>

namespace DieModel {

// ---------------------------------------------------------------------------
// detail helpers (declared in DieTypes.hh)

namespace detail {

void fail(const std::string& path, const std::string& message) {
  throw DieModelError(path + ": " + message);
}

void reject_unknown_keys(const nlohmann::json& obj,
                         const std::vector<const char*>& allowed,
                         const std::string& path) {
  for (auto it = obj.begin(); it != obj.end(); ++it) {
    bool known = false;
    for (const char* key : allowed) {
      if (it.key() == key) {
        known = true;
        break;
      }
    }
    if (!known) {
      fail(path, "unknown key \"" + it.key() + "\"");
    }
  }
}

const nlohmann::json& member(const nlohmann::json& obj, const char* key,
                             const std::string& path) {
  const auto it = obj.find(key);
  if (it == obj.end()) {
    fail(path, "missing required key \"" + std::string(key) + "\"");
  }
  return *it;
}

std::uint64_t require_uint(const nlohmann::json& obj, const char* key,
                           const std::string& path) {
  const std::string field_path = path + "." + key;
  const nlohmann::json& v = member(obj, key, path);
  if (!v.is_number_unsigned()) {
    std::string got = v.type_name();
    if (v.is_number_float()) {
      got += " (floating-point values are rejected here)";
    } else if (v.is_number_integer() && v.get<std::int64_t>() < 0) {
      got += " (negative values are rejected here)";
    } else {
      got += " (non-negative integers must be stored as unsigned)";
    }
    fail(field_path, "expected an unsigned integer, got " + got);
  }
  return v.get<std::uint64_t>();
}

double require_number(const nlohmann::json& obj, const char* key,
                      const std::string& path) {
  const nlohmann::json& v = member(obj, key, path);
  if (!v.is_number()) {
    fail(path + "." + key,
         "expected a number, got " + std::string(v.type_name()));
  }
  return v.get<double>();
}

std::string require_string(const nlohmann::json& obj, const char* key,
                           const std::string& path) {
  const nlohmann::json& v = member(obj, key, path);
  if (!v.is_string()) {
    fail(path + "." + key,
         "expected a string, got " + std::string(v.type_name()));
  }
  return v.get<std::string>();
}

std::string require_enum_string(const nlohmann::json& obj, const char* key,
                                const std::vector<const char*>& valid,
                                const std::string& path) {
  const std::string value = require_string(obj, key, path);
  for (const char* v : valid) {
    if (value == v) {
      return value;
    }
  }
  std::string list;
  for (const char* v : valid) {
    if (!list.empty()) {
      list += "|";
    }
    list += v;
  }
  fail(path + "." + key,
       "invalid enum value \"" + value + "\", expected one of " + list);
}

}  // namespace detail

// ---------------------------------------------------------------------------
// parse_die_op_spec (contract §2 schema)

namespace {

constexpr const char* kTopPath = "die_op_spec";

DieModelDims parse_model_dims(const nlohmann::json& j) {
  const std::string path = std::string(kTopPath) + ".model_dims";
  if (!j.is_object()) {
    detail::fail(path, "expected a JSON object");
  }
  detail::reject_unknown_keys(
      j,
      {"hidden", "ffn", "vocab", "mlp_variant", "norm_type",
       "position_encoding", "num_q_heads", "num_kv_heads", "head_dim"},
      path);

  DieModelDims md;
  md.hidden = detail::require_uint(j, "hidden", path);
  md.ffn = detail::require_uint(j, "ffn", path);
  md.vocab = detail::require_uint(j, "vocab", path);
  from_string(detail::require_enum_string(j, "mlp_variant", {"swiglu", "gelu"},
                                          path),
              md.mlp_variant);
  from_string(detail::require_enum_string(j, "norm_type",
                                          {"rmsnorm", "layernorm"}, path),
              md.norm_type);
  from_string(detail::require_enum_string(j, "position_encoding",
                                          {"rope", "none"}, path),
              md.position_encoding);
  md.num_q_heads = detail::require_uint(j, "num_q_heads", path);
  md.num_kv_heads = detail::require_uint(j, "num_kv_heads", path);
  md.head_dim = detail::require_uint(j, "head_dim", path);
  return md;
}

DieTpSlice parse_tp_slice(const nlohmann::json& j) {
  const std::string path = std::string(kTopPath) + ".tp_slice";
  if (!j.is_object()) {
    detail::fail(path, "expected a JSON object");
  }
  detail::reject_unknown_keys(
      j,
      {"tensor_parallel", "tensor_parallel_rank", "q_heads_local",
       "kv_heads_local", "ffn_local", "vocab_local"},
      path);

  DieTpSlice tp;
  tp.tensor_parallel = detail::require_uint(j, "tensor_parallel", path);
  tp.tensor_parallel_rank =
      detail::require_uint(j, "tensor_parallel_rank", path);
  tp.q_heads_local = detail::require_uint(j, "q_heads_local", path);
  tp.kv_heads_local = detail::require_uint(j, "kv_heads_local", path);
  tp.ffn_local = detail::require_uint(j, "ffn_local", path);
  tp.vocab_local = detail::require_uint(j, "vocab_local", path);
  return tp;
}

}  // namespace

std::shared_ptr<const DieOpSpec> parse_die_op_spec(const nlohmann::json& j) {
  if (!j.is_object()) {
    detail::fail(kTopPath, "expected a JSON object");
  }
  // provenance/config_hash are optional metadata; everything else in the
  // contract §2 schema is required.
  detail::reject_unknown_keys(
      j,
      {"schema_version", "op_kind", "phase", "dtype", "model_dims",
       "tp_slice", "spans", "layer_count", "weight_passes", "kv_access",
       "fusion_policy", "provenance", "config_hash", "repeat_seams"},
      kTopPath);

  auto spec = std::make_shared<DieOpSpec>();

  spec->schema_version = detail::require_uint(j, "schema_version", kTopPath);
  if (spec->schema_version != 1) {
    detail::fail(std::string(kTopPath) + ".schema_version",
                 "unsupported schema_version " +
                     std::to_string(spec->schema_version) + ", expected 1");
  }

  from_string(detail::require_enum_string(
                  j, "op_kind",
                  {"norm", "qkv_rope", "qk", "scale_mask", "softmax", "av",
                   "o_proj", "residual", "attention_region", "mlp_up",
                   "mlp_activation", "mlp_down", "ffn_region", "logits",
                   "local_region"},
                  std::string(kTopPath) + ".op_kind"),
              spec->op_kind);
  from_string(detail::require_enum_string(j, "phase", {"prefill", "decode"},
                                          std::string(kTopPath) + ".phase"),
              spec->phase);
  from_string(detail::require_enum_string(j, "dtype", {"bf16", "fp32"},
                                          std::string(kTopPath) + ".dtype"),
              spec->dtype);

  spec->model_dims = parse_model_dims(detail::member(j, "model_dims", kTopPath));
  spec->tp_slice = parse_tp_slice(detail::member(j, "tp_slice", kTopPath));

  // spans: at least one; each span is one physical-batch fragment
  // (contract §2; attention charges sum(q_i*k_i), never sum(q)*sum(k)).
  // v5 compact runs: an element may instead be a run record
  //   {"request_id_prefix", "first_index", "run": {q_tokens, kv_first,
  //    kv_stride, count}}
  // expanding to `count` spans with kv_i = kv_first + i*stride,
  // q_start_pos = kv_i - q_tokens, kv_start_pos = 0 and id
  // "<prefix><first_index + i>" -- numerically identical to the legacy
  // fully-expanded wire (lossless compression; derive_span_records is the
  // Python-side compressor).  Expansion happens once here, at the parse
  // boundary; plan compilation and billing consume the expanded
  // vector<DieSpan> unchanged.
  const std::string spans_path = std::string(kTopPath) + ".spans";
  const nlohmann::json& spans = detail::member(j, "spans", kTopPath);
  if (!spans.is_array()) {
    detail::fail(spans_path, "expected a JSON array");
  }
  if (spans.empty()) {
    detail::fail(spans_path, "expected at least one span");
  }
  for (std::size_t i = 0; i < spans.size(); ++i) {
    const std::string path =
        spans_path + "[" + std::to_string(i) + "]";
    const nlohmann::json& s = spans[i];
    if (!s.is_object()) {
      detail::fail(path, "expected a JSON object");
    }
    if (s.contains("run")) {
      detail::reject_unknown_keys(
          s, {"request_id_prefix", "first_index", "run"}, path);
      const std::string run_path = path + ".run";
      const nlohmann::json& run = detail::member(s, "run", path.c_str());
      detail::reject_unknown_keys(
          run, {"q_tokens", "kv_first", "kv_stride", "count"}, run_path);
      const std::uint64_t run_q =
          detail::require_uint(run, "q_tokens", run_path);
      const std::uint64_t kv_first =
          detail::require_uint(run, "kv_first", run_path);
      const std::uint64_t kv_stride =
          detail::require_uint(run, "kv_stride", run_path);
      const std::uint64_t count =
          detail::require_uint(run, "count", run_path);
      const std::string prefix =
          detail::require_string(s, "request_id_prefix", path);
      const std::uint64_t first_index =
          detail::require_uint(s, "first_index", path);
      if (run_q < 1) {
        detail::fail(run_path + ".q_tokens", "must be >= 1");
      }
      if (count < 1) {
        detail::fail(run_path + ".count", "must be >= 1");
      }
      if (kv_stride < 1) {
        detail::fail(run_path + ".kv_stride",
                     "compact runs require stride >= 1");
      }
      // Every expanded member must satisfy kv_tokens >= q_tokens (the
      // joint span invariant; kv grows with stride >= 1 so checking the
      // first member plus an overflow guard covers all).
      if (kv_first < run_q
          || count > (std::numeric_limits<std::uint64_t>::max() - kv_first)
                 / kv_stride) {
        detail::fail(run_path,
                     "expanded run violates kv_tokens == q_start_pos + "
                     "q_tokens for some member");
      }
      if (first_index > std::numeric_limits<std::uint64_t>::max() - count) {
        detail::fail(path + ".first_index", "expansion overflows indices");
      }
      spec->spans.reserve(spec->spans.size()
                          + static_cast<std::size_t>(count));
      for (std::uint64_t member = 0; member < count; ++member) {
        DieSpan span;
        span.request_id = prefix + std::to_string(first_index + member);
        span.q_tokens = run_q;
        span.kv_tokens = kv_first + member * kv_stride;
        span.q_start_pos = span.kv_tokens - span.q_tokens;
        span.kv_start_pos = 0;
        spec->spans.push_back(std::move(span));
      }
      continue;
    }
    detail::reject_unknown_keys(
        s,
        {"request_id", "q_tokens", "kv_tokens", "q_start_pos", "kv_start_pos"},
        path);
    DieSpan span;
    span.request_id = detail::require_string(s, "request_id", path);
    span.q_tokens = detail::require_uint(s, "q_tokens", path);
    span.kv_tokens = detail::require_uint(s, "kv_tokens", path);
    span.q_start_pos = detail::require_uint(s, "q_start_pos", path);
    span.kv_start_pos = detail::require_uint(s, "kv_start_pos", path);
    if (span.q_tokens < 1) {
      detail::fail(path + ".q_tokens", "must be >= 1");
    }
    if (span.kv_tokens < 1) {
      detail::fail(path + ".kv_tokens", "must be >= 1");
    }
    spec->spans.push_back(std::move(span));
  }

  spec->layer_count = detail::require_uint(j, "layer_count", kTopPath);
  if (spec->layer_count < 1) {
    detail::fail(std::string(kTopPath) + ".layer_count", "must be >= 1");
  }
  spec->weight_passes = detail::require_uint(j, "weight_passes", kTopPath);
  if (spec->weight_passes < 1) {
    detail::fail(std::string(kTopPath) + ".weight_passes", "must be >= 1");
  }

  // kv_access: exactly one key, its enum (contract §2).
  {
    const std::string path = std::string(kTopPath) + ".kv_access";
    const nlohmann::json& kv = detail::member(j, "kv_access", kTopPath);
    if (!kv.is_object()) {
      detail::fail(path, "expected a JSON object");
    }
    detail::reject_unknown_keys(kv, {"identity"}, path);
    from_string(detail::require_enum_string(
                    kv, "identity", {"local", "staged", "remote_ready"},
                    path + ".identity"),
                spec->kv_access.identity);
  }

  from_string(detail::require_enum_string(j, "fusion_policy", {"auto", "none"},
                                          std::string(kTopPath) +
                                              ".fusion_policy"),
              spec->fusion_policy);

  if (j.contains("provenance")) {
    spec->provenance = detail::require_string(j, "provenance", kTopPath);
  }
  if (j.contains("config_hash")) {
    spec->config_hash = detail::require_string(j, "config_hash", kTopPath);
  }

  // --- basic shape relations (design doc §4.2) ---
  const DieModelDims& md = spec->model_dims;
  const DieTpSlice& tp = spec->tp_slice;
  const std::string dims = std::string(kTopPath) + ".model_dims";
  const std::string tps = std::string(kTopPath) + ".tp_slice";
  if (md.hidden < 1 || md.ffn < 1 || md.vocab < 1 || md.head_dim < 1 ||
      md.num_q_heads < 1 || md.num_kv_heads < 1) {
    detail::fail(dims, "hidden/ffn/vocab/head_dim/num_q_heads/num_kv_heads "
                       "must all be >= 1");
  }
  // hidden == num_q_heads * head_dim: the workload model of design doc §2.1
  // (Hq*d == H) and the projection shapes of §4.2 use this convention; guard
  // against overflow before multiplying.
  if (md.num_q_heads != 0 &&
      md.head_dim > UINT64_MAX / md.num_q_heads) {
    detail::fail(dims, "num_q_heads * head_dim overflows");
  }
  if (md.num_q_heads * md.head_dim != md.hidden) {
    detail::fail(dims,
                 "hidden (" + std::to_string(md.hidden) +
                     ") must equal num_q_heads * head_dim (" +
                     std::to_string(md.num_q_heads * md.head_dim) +
                     ") (design doc §2.1/§4.2 convention)");
  }
  // GQA/MHA: at most as many KV heads as Q heads (design doc §2.1/§4.2).
  if (md.num_kv_heads > md.num_q_heads) {
    detail::fail(dims,
                 "num_kv_heads (" + std::to_string(md.num_kv_heads) +
                     ") must not exceed num_q_heads (" +
                     std::to_string(md.num_q_heads) + ")");
  }
  if (tp.tensor_parallel < 1) {
    detail::fail(tps + ".tensor_parallel", "must be >= 1");
  }
  if (tp.tensor_parallel_rank >= tp.tensor_parallel) {
    detail::fail(tps + ".tensor_parallel_rank",
                 "must be < tensor_parallel (" +
                     std::to_string(tp.tensor_parallel) + ")");
  }
  // Local shards may be zero (empty shard = legal zero work, §4.2 rule 5)
  // but never larger than the global dimension.
  if (tp.q_heads_local > md.num_q_heads) {
    detail::fail(tps + ".q_heads_local",
                 "must be <= model_dims.num_q_heads (" +
                     std::to_string(md.num_q_heads) + ")");
  }
  if (tp.kv_heads_local > md.num_kv_heads) {
    detail::fail(tps + ".kv_heads_local",
                 "must be <= model_dims.num_kv_heads (" +
                     std::to_string(md.num_kv_heads) + ")");
  }
  if (tp.ffn_local > md.ffn) {
    detail::fail(tps + ".ffn_local",
                 "must be <= model_dims.ffn (" + std::to_string(md.ffn) +
                     ")");
  }
  if (tp.vocab_local > md.vocab) {
    detail::fail(tps + ".vocab_local",
                 "must be <= model_dims.vocab (" + std::to_string(md.vocab) +
                     ")");
  }

  return spec;
}

// ---------------------------------------------------------------------------
// derive_workload (design doc §4.2)
//
// All quantities below are the MATHEMATICAL work of the op: matrix FLOP,
// vector eq-ops and the untiled byte floor (activations read/written once,
// K/V once, weights once per weight pass).  Tiling effects (reload counts,
// causal tile rectangles, FP32 partial sums, two-pass streaming) are added
// by src/Mapping.cc and may only increase these numbers.

namespace {

// Causally valid q*k pairs of one span with ABSOLUTE positions (design doc
// §4.2 rule 7): query token t sits at q_start_pos+t and attends to KV
// positions [kv_start_pos, kv_start_pos+kv_tokens) that are <= itself.
// Decode M=1 is never padded (rule 7); only pairs that are entirely in the
// future are absent -- everything on the diagonal side stays counted.
std::uint64_t valid_pairs_of(const DieSpan& s) {
  const std::int64_t offset =
      static_cast<std::int64_t>(s.q_start_pos) -
      static_cast<std::int64_t>(s.kv_start_pos);
  std::uint64_t total = 0;
  for (std::uint64_t t = 0; t < s.q_tokens; ++t) {
    const std::int64_t reach = offset + static_cast<std::int64_t>(t) + 1;
    if (reach <= 0) {
      continue;  // whole KV range is in this token's future
    }
    const std::uint64_t capped =
        static_cast<std::uint64_t>(reach) < s.kv_tokens
            ? static_cast<std::uint64_t>(reach)
            : s.kv_tokens;
    total += capped;
  }
  return total;
}

std::vector<DieAttentionSpanWork> attention_span_works(const DieOpSpec& spec) {
  std::vector<DieAttentionSpanWork> out;
  out.reserve(spec.spans.size());
  for (const DieSpan& s : spec.spans) {
    DieAttentionSpanWork w;
    w.q_tokens = s.q_tokens;
    w.kv_tokens = s.kv_tokens;
    w.q_start_pos = s.q_start_pos;
    w.kv_start_pos = s.kv_start_pos;
    w.valid_pairs = valid_pairs_of(s);
    w.rectangle_pairs = s.q_tokens * s.kv_tokens;
    out.push_back(w);
  }
  return out;
}

void finalize_totals(WorkloadQuantities& q) {
  const double layers = static_cast<double>(q.layer_count);
  const double passes = static_cast<double>(q.weight_passes);
  q.total_flops = q.flops_per_layer * layers;
  q.total_eqops = q.eqops_per_layer * layers;
  q.total_weight_bytes = q.weight_bytes_per_pass * passes * layers;
  q.total_matrix_bytes = q.matrix_bytes_per_layer * layers;
  q.total_vector_bytes = q.vector_bytes_per_layer * layers;
  q.launch_instances = q.layer_count * q.weight_passes;
}

// Attaches one GEMM [M,K]x[K,N] to the skeleton (design doc §4.2 rules 1/3:
// M is the merged token count, weight bytes per pass).
void add_gemm(WorkloadQuantities& q, std::uint64_t k, std::uint64_t n) {
  DieGemmShape g;
  g.m = q.m_tokens;
  g.k = k;
  g.n = n;
  q.gemms.push_back(g);
  const double m = static_cast<double>(q.m_tokens);
  const double kk = static_cast<double>(k);
  const double nn = static_cast<double>(n);
  const double e = static_cast<double>(q.elem_bytes);
  q.flops_per_layer += 2.0 * m * kk * nn;
  q.weight_bytes_per_pass += e * kk * nn;
  q.act_in_bytes_per_layer += e * m * kk;
  q.act_out_bytes_per_layer += e * m * nn;
}

}  // namespace

WorkloadQuantities derive_workload(const DieOpSpec& spec) {
  WorkloadQuantities q;
  q.op_kind = spec.op_kind;
  q.dtype = spec.dtype;
  q.elem_bytes = spec.dtype == Dtype::Fp32 ? 4 : 2;
  q.layer_count = spec.layer_count;
  q.weight_passes = spec.weight_passes;

  // M = sum(spans.q_tokens): one physical batch merged for linear ops
  // (design doc §4.2 rule 1).
  for (const DieSpan& s : spec.spans) {
    q.m_tokens += s.q_tokens;
  }

  const DieModelDims& md = spec.model_dims;
  const DieTpSlice& tp = spec.tp_slice;
  const double e = static_cast<double>(q.elem_bytes);
  const double m = static_cast<double>(q.m_tokens);
  const double H = static_cast<double>(md.hidden);
  const std::uint64_t d = md.head_dim;
  const std::uint64_t qh = tp.q_heads_local;
  const std::uint64_t kvh = tp.kv_heads_local;
  const std::uint64_t n_q = qh * d;
  const std::uint64_t n_kv = kvh * d;
  q.hidden = md.hidden;
  q.ffn_local = tp.ffn_local;
  q.vocab_local = tp.vocab_local;
  q.q_heads_local = qh;
  q.kv_heads_local = kvh;
  q.head_dim = d;
  q.mlp_variant = md.mlp_variant;
  q.norm_type = md.norm_type;
  q.position_encoding = md.position_encoding;
  q.attention_spans = attention_span_works(spec);
  double valid_pairs = 0.0;   // sum over spans
  double kv_tokens_sum = 0.0;
  for (const DieAttentionSpanWork& w : q.attention_spans) {
    valid_pairs += static_cast<double>(w.valid_pairs);
    kv_tokens_sum += static_cast<double>(w.kv_tokens);
  }
  // Per-head attention pair floor (GQA charges every local q head; K/V stay
  // at the compact kv-head count, design doc §4.2 rule 3 / §5.5).
  q.attention_pairs = static_cast<double>(qh) * valid_pairs;

  switch (spec.op_kind) {
    case OpKind::Norm: {
      // RMSNorm 6 / LayerNorm 8 eq-ops per element; gamma (+beta) is the
      // weight row; input once, output once (design doc §5.3, §2.2).
      const double w = spec.model_dims.norm_type == NormType::Rmsnorm ? 6.0 : 8.0;
      const double gamma_rows =
          spec.model_dims.norm_type == NormType::Rmsnorm ? 1.0 : 2.0;
      q.eqops_per_layer = w * m * H;
      q.weight_bytes_per_pass = e * gamma_rows * H;
      q.act_in_bytes_per_layer = e * m * H;
      q.act_out_bytes_per_layer = e * m * H;
      q.vector_bytes_per_layer = q.act_in_bytes_per_layer +
                                 q.act_out_bytes_per_layer +
                                 q.weight_bytes_per_pass;
      break;
    }
    case OpKind::QkvRope: {
      // Three real GEMMs on the merged M (design doc §4.2 rule 3):
      // [M,H]x[H,qh*d], [M,H]x[H,kvh*d] twice; GQA uses the true local head
      // counts, num_kv == num_q is simply the MHA reading of the same
      // formulas (rule 6).  The input tensor is read once for all three.
      add_gemm(q, md.hidden, n_q);
      add_gemm(q, md.hidden, n_kv);
      add_gemm(q, md.hidden, n_kv);
      q.act_in_bytes_per_layer = e * m * H;  // shared input, read once
      if (md.position_encoding == PositionEncoding::Rope) {
        // RoPE rotates only the NEW Q/K (q_tokens part); cos/sin are read
        // for the actual position set of each span: 2 tables x elem bytes x
        // d x positions (design doc §5.3, §2.2 RoPE row).
        const double rotated = m * static_cast<double>(n_q + n_kv);
        q.eqops_per_layer = 3.0 * rotated;
        q.aux_bytes_per_layer =
            2.0 * e * static_cast<double>(d) * m;  // cos+sin, per position
      }
      q.matrix_bytes_per_layer = q.weight_bytes_per_pass +
                                 q.act_in_bytes_per_layer +
                                 q.act_out_bytes_per_layer;
      q.vector_bytes_per_layer = q.aux_bytes_per_layer;
      break;
    }
    case OpKind::Qk:
    case OpKind::Av:
    case OpKind::AttentionRegion: {
      // Attention family floor: per span q_i*k_i with that span's own kv
      // range; K/V bytes use the compact kv head count (GQA reuse, §5.5).
      q.kv_bytes_per_layer = 2.0 * e * static_cast<double>(kvh * d) * kv_tokens_sum;
      const double pairs = q.attention_pairs;
      const double dd = static_cast<double>(d);
      if (spec.op_kind == OpKind::Qk) {
        q.flops_per_layer = 2.0 * dd * pairs;
        q.act_in_bytes_per_layer = e * static_cast<double>(qh * d) * m;   // Q
        q.act_out_bytes_per_layer = 4.0 * pairs;  // FP32 scores
        q.matrix_bytes_per_layer = q.act_in_bytes_per_layer +
                                   q.act_out_bytes_per_layer + q.kv_bytes_per_layer;
      } else if (spec.op_kind == OpKind::Av) {
        q.flops_per_layer = 2.0 * dd * pairs;
        q.eqops_per_layer = pairs;  // FP32->BF16 probability per score (§2.2)
        q.act_in_bytes_per_layer = 4.0 * pairs;   // FP32 probabilities
        q.act_out_bytes_per_layer = e * static_cast<double>(qh * d) * m;
        q.matrix_bytes_per_layer = q.act_in_bytes_per_layer +
                                   q.act_out_bytes_per_layer + q.kv_bytes_per_layer;
      } else {
        // Fused on-tile attention (§5.5): matrix 4qkd per tile, vector base
        // 13qk, final normalization q*d per head row; the per-tile merge
        // terms q*(3d+22) are mapping-level (tile counts).  Q read once, O
        // written once, K/V once (floor).
        q.flops_per_layer = 4.0 * dd * pairs;
        q.eqops_per_layer =
            13.0 * pairs + static_cast<double>(qh * d) * m;
        q.act_in_bytes_per_layer = e * static_cast<double>(qh * d) * m;
        q.act_out_bytes_per_layer = e * static_cast<double>(qh * d) * m;
        q.matrix_bytes_per_layer = q.act_in_bytes_per_layer +
                                   q.act_out_bytes_per_layer + q.kv_bytes_per_layer;
      }
      break;
    }
    case OpKind::ScaleMask:
    case OpKind::Softmax: {
      // Unfused score chain split of the fused 12/score weight (§5.3):
      // scale+mask 2/score, softmax 10/score; FP32 score read+write per op
      // (8 bytes/score materialized traffic, ruling in the A2 report).
      const double w = spec.op_kind == OpKind::ScaleMask ? 2.0 : 10.0;
      q.eqops_per_layer = w * q.attention_pairs;
      q.vector_bytes_per_layer = 8.0 * q.attention_pairs;
      break;
    }
    case OpKind::OProj: {
      add_gemm(q, qh * d, md.hidden);
      break;
    }
    case OpKind::Residual: {
      // residual add / gate_mul: 1 eq-op per element, two reads + one write
      // (§5.3, §2.2 residual row: 6*M*H at BF16).
      q.eqops_per_layer = m * H;
      q.vector_bytes_per_layer = 3.0 * e * m * H;
      q.act_in_bytes_per_layer = 2.0 * e * m * H;
      q.act_out_bytes_per_layer = e * m * H;
      break;
    }
    case OpKind::MlpUp: {
      const std::uint64_t n =
          md.mlp_variant == MlpVariant::Swiglu ? 2 * tp.ffn_local : tp.ffn_local;
      add_gemm(q, md.hidden, n);
      break;
    }
    case OpKind::MlpActivation: {
      // SiLU 15 + gate_mul 1 = 16/elem (SwiGLU) or GELU 15/elem; the op
      // reads gate+up and writes the activation once (materialized).
      const double w = md.mlp_variant == MlpVariant::Swiglu ? 16.0 : 15.0;
      const double reads = md.mlp_variant == MlpVariant::Swiglu ? 3.0 : 2.0;
      const double f = static_cast<double>(tp.ffn_local);
      q.eqops_per_layer = w * m * f;
      q.act_in_bytes_per_layer = reads * e * m * f;
      q.act_out_bytes_per_layer = e * m * f;
      q.vector_bytes_per_layer =
          q.act_in_bytes_per_layer + q.act_out_bytes_per_layer;
      break;
    }
    case OpKind::MlpDown: {
      add_gemm(q, tp.ffn_local, md.hidden);
      break;
    }
    case OpKind::FfnRegion: {
      // Fused FFN (§5.4): SwiGLU 6MHF_local matrix + 16MF_local vector and
      // three weight slabs; GELU 4MHF_local + 15MF_local and two slabs.
      const double f = static_cast<double>(tp.ffn_local);
      const bool swiglu = md.mlp_variant == MlpVariant::Swiglu;
      q.flops_per_layer = (swiglu ? 6.0 : 4.0) * m * H * f;
      q.eqops_per_layer = (swiglu ? 16.0 : 15.0) * m * f;
      q.weight_bytes_per_pass = (swiglu ? 3.0 : 2.0) * e * H * f;
      q.act_in_bytes_per_layer = e * m * H;
      q.act_out_bytes_per_layer = e * m * H;
      q.matrix_bytes_per_layer = q.weight_bytes_per_pass +
                                 q.act_in_bytes_per_layer +
                                 q.act_out_bytes_per_layer;
      break;
    }
    case OpKind::Logits: {
      add_gemm(q, md.hidden, tp.vocab_local);
      break;
    }
    case OpKind::LocalRegion: {
      // Schema v1 carries no region_ops breakdown for a generic local
      // region, so its mathematical work is not computable here: treated as
      // a zero-work identity region (ruling recorded in the stage A2
      // report).  It still pays its macro-group launches in the mesh
      // profile.
      break;
    }
  }

  // Linear GEMM ops default their class split to matrix (set inside
  // add_gemm callers where not overridden above).
  if (q.matrix_bytes_per_layer == 0.0 && q.vector_bytes_per_layer == 0.0) {
    q.matrix_bytes_per_layer = q.weight_bytes_per_pass +
                               q.act_in_bytes_per_layer +
                               q.act_out_bytes_per_layer + q.kv_bytes_per_layer;
  }

  finalize_totals(q);
  return q;
}

}  // namespace DieModel
