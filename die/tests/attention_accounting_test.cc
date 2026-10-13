#include "test_support.hh"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

namespace coarse_die::detail {
std::vector<std::shared_ptr<coarse_die::Plan>> enumerate_mappings(
    const coarse_die::Hardware&, const nlohmann::json&);
}

using namespace coarse_die;
using namespace coarse_die_test;
using json = nlohmann::json;

namespace {

struct RefSpan {
  std::uint64_t q = 0, kv = 0, count = 1;
  std::uint64_t qs = 0, ks = 0, qstride = 0, kvstride = 0;
};

std::uint64_t ceildiv(std::uint64_t a, std::uint64_t b) {
  return a / b + (a % b != 0);
}

json normalized_task(const std::string& op, std::uint64_t q,
                     std::uint64_t kv, std::uint64_t q_heads,
                     std::uint64_t kv_heads, std::uint64_t head_dim,
                     const RefSpan& span, bool causal,
                     const std::string& dtype = "bf16",
                     const std::string& profile =
                         "int4_matrix_vec_fp32_assumed_v1",
                     const std::string& fusion = "fused") {
  json storage{{"weights", dtype}, {"activations", dtype}, {"kv", dtype},
               {"output", dtype}, {"parameters", dtype}, {"scores", "fp32"},
               {"accumulator", "fp32"}};
  return json{
      {"schema", "coarse_die_task_v2"},
      {"op_kind", op},
      {"phase", "prefill"},
      {"tensor_shapes", {{"q_heads", q_heads}, {"kv_heads", kv_heads},
                          {"head_dim", head_dim}}},
      {"spans", json::array({json{
          {"q_tokens", span.q}, {"kv_tokens", span.kv},
          {"count", span.count}, {"q_start_pos", span.qs},
          {"q_start_stride", span.qstride}, {"kv_start_pos", span.ks},
          {"kv_start_stride", 0}, {"kv_stride", span.kvstride}}})},
      {"tp_slice", {{"q_heads_local", q_heads}, {"kv_heads_local", kv_heads},
                    {"tensor_parallel", 1}, {"tensor_parallel_rank", 0}}},
      {"storage_dtype", storage}, {"execution_profile", profile},
      {"layer_instances", 1}, {"weight_passes", 1},
      {"fusion_policy", fusion}, {"causal", causal}};
}

struct RefCounts {
  std::uint64_t slots = 0;
  std::uint64_t valid = 0;
  std::uint64_t prefixes = 0;
  std::uint64_t kv_sum = 0;
  std::uint64_t kv_tiles = 0;
  std::uint64_t row_block_visits = 0;
  std::uint64_t active_rows = 0;
};

RefCounts brute_reference(const RefSpan& span, std::uint64_t tq,
                          std::uint64_t tk, bool causal, bool materialized) {
  RefCounts result;
  for (std::uint64_t r = 0; r < span.count; ++r) {
    const std::uint64_t qstart = span.qs + r * span.qstride;
    const std::uint64_t kstart = span.ks;
    const std::uint64_t kv = span.kv + r * span.kvstride;
    result.kv_sum += kv;
    result.kv_tiles += ceildiv(kv, tk);
    for (std::uint64_t qi = 0; qi < span.q; ++qi) {
      if (!causal || kstart <= qstart + qi) ++result.active_rows;
      for (std::uint64_t ki = 0; ki < kv; ++ki) {
        if (!causal || kstart + ki <= qstart + qi) ++result.valid;
      }
    }
    if (materialized || !causal) {
      result.slots += span.q * kv;
      result.prefixes += (materialized ? ceildiv(span.q, tq) : ceildiv(span.q, tq)) * kv;
      if (!materialized && !causal) {
        result.row_block_visits += span.q * ceildiv(kv, tk);
      }
      continue;
    }
    const std::uint64_t qtiles = ceildiv(span.q, tq);
    for (std::uint64_t tile = 0; tile < qtiles; ++tile) {
      const std::uint64_t begin = tile * tq;
      const std::uint64_t rows = std::min(tq, span.q - begin);
      const std::uint64_t end_abs = qstart + begin + rows - 1;
      std::uint64_t prefix = 0;
      if (end_abs >= kstart) {
        const std::uint64_t reached = end_abs - kstart + 1;
        prefix = std::min(kv, ceildiv(reached, tk) * tk);
      }
      result.slots += rows * prefix;
      result.prefixes += prefix;
      result.row_block_visits += rows * ceildiv(prefix, tk);
    }
  }
  return result;
}

const Plan& select_plan(const std::vector<std::shared_ptr<Plan>>& plans,
                        const std::string& template_id,
                        std::uint64_t tq, std::uint64_t tk) {
  for (const auto& plan : plans) {
    if (plan->summary.value("template_id", std::string()) == template_id &&
        plan->summary.value("tm", std::uint64_t{0}) == tq &&
        plan->summary.value("tk", std::uint64_t{0}) == tk) {
      return *plan;
    }
  }
  std::string seen;
  for (const auto& plan : plans) {
    seen += " [" + plan->summary.value("template_id", std::string()) + "," +
            std::to_string(plan->summary.value("tm", std::uint64_t{0})) + "," +
            std::to_string(plan->summary.value("tk", std::uint64_t{0})) + "," +
            plan->task.at("spans").dump() + "]";
  }
  throw std::runtime_error("missing attention candidate " + template_id +
                           " tq=" + std::to_string(tq) +
                           " tk=" + std::to_string(tk) + " among" + seen);
}

std::uint64_t summary_u64(const Plan& plan, const char* key) {
  require(plan.summary.contains(key), std::string("attention summary missing ") + key);
  return plan.summary.at(key).get<std::uint64_t>();
}

void require_sram_peak(const Hardware& hw, const Plan& plan,
                       std::uint64_t expected, const std::string& context) {
  const auto actual = summary_u64(plan, "sram_peak_bytes_per_core");
  require(actual == expected,
          context + " SRAM live-buffer peak mismatch: actual=" +
              std::to_string(actual) + " expected=" + std::to_string(expected));
  const auto cap = hw.derived.at("core").at("sram_bytes_per_core")
                       .get<std::uint64_t>();
  const double rho = hw.input.at("performance_profile").value("rho_sram", 0.9);
  require(static_cast<long double>(actual) <=
              static_cast<long double>(rho) * cap,
          context + " live-buffer peak exceeds the configured per-core SRAM budget");
}

void require_activity_totals(const Plan& plan) {
  long double read = 0, write = 0;
  std::uint64_t hbm_read = 0, hbm_write = 0;
  for (const auto& epoch : plan.epochs) {
    for (const auto& activity : epoch.activities) {
      read += activity.work[SRAM_READ];
      write += activity.work[SRAM_WRITE];
      for (std::size_t row = 0; row < 2; ++row) {
        hbm_read += activity.hbm_bytes[row][0];
        hbm_write += activity.hbm_bytes[row][1];
      }
    }
  }
  require(std::abs(read - summary_u64(plan, "sram_read_bytes")) < 0.01L,
          "SRAM read summary equals materialized Activity work");
  require(std::abs(write - summary_u64(plan, "sram_write_bytes")) < 0.01L,
          "SRAM write summary equals materialized Activity work");
  require(hbm_read == summary_u64(plan, "hbm_read_bytes_total"),
          "HBM read summary equals exact activity byte payload");
  require(hbm_write == summary_u64(plan, "hbm_write_bytes_total"),
          "HBM write summary equals exact activity byte payload");
}

void verify_fused_bf16_candidate(const Hardware& hw, const Plan& plan,
                                const RefSpan& span,
                                std::uint64_t tq, std::uint64_t tk,
                                std::uint64_t q_heads, std::uint64_t kv_heads,
                                std::uint64_t head_dim, bool causal) {
  const auto ref = brute_reference(span, tq, tk, causal, false);
  const std::uint64_t qtiles = ceildiv(span.q, tq);
  const std::uint64_t query_rows = span.q * span.count * q_heads;
  const std::uint64_t slots = ref.slots * q_heads;
  const std::uint64_t valid = ref.valid * q_heads;
  const std::uint64_t active_rows = ref.active_rows * q_heads;
  const std::uint64_t row_block_visits = ref.row_block_visits * q_heads;
  const std::uint64_t qbytes = query_rows * head_dim * 2;
  const std::uint64_t kvbytes = ref.prefixes * q_heads * head_dim * 2;
  const std::uint64_t outbytes = query_rows * head_dim * 2;
  require(summary_u64(plan, "executed_score_slots") == slots,
          "tile oracle matches fused executed score slots");
  require(summary_u64(plan, "valid_score_pairs") == valid,
          "absolute-position oracle matches causal source pairs");
  require(summary_u64(plan, "executed_matrix_ops") == 4 * head_dim * slots,
          "fused QK and PV matrix work follows executed slots");
  const std::uint64_t online_rows = row_block_visits > active_rows
      ? row_block_visits - active_rows : 0;
  const std::uint64_t eqops = 13 * slots + query_rows * head_dim +
      online_rows * (2 * head_dim + 4);
  const std::uint64_t quant_elements = query_rows * head_dim +
      2 * ref.prefixes * q_heads * head_dim;
  const std::uint64_t qk_acc_elements = slots * ceildiv(head_dim, 64);
  const std::uint64_t pv_acc_elements = row_block_visits * head_dim;
  const std::uint64_t accumulator_elements = qk_acc_elements + pv_acc_elements;
  const std::uint64_t accumulator_adds =
      qk_acc_elements * (ceildiv(head_dim, 64) - 1) +
      (pv_acc_elements > active_rows * head_dim
           ? pv_acc_elements - active_rows * head_dim : 0);
  const std::uint64_t expected_vec = eqops + 4 * quant_elements +
      2 * accumulator_elements + accumulator_adds + query_rows * head_dim;
  const auto actual_vec = summary_u64(plan, "vec_equivalent_ops");
  require(actual_vec == expected_vec,
          "softmax/online/quant/acc/cast Vec work mismatch: q=" +
              std::to_string(span.q) + " kv=" + std::to_string(span.kv) +
              " count=" + std::to_string(span.count) + " qstart=" +
              std::to_string(span.qs) + " kstart=" + std::to_string(span.ks) +
              " qstride=" + std::to_string(span.qstride) + " kvstride=" +
              std::to_string(span.kvstride) + " tq=" + std::to_string(tq) +
              " tk=" + std::to_string(tk) + " qh=" +
              std::to_string(q_heads) + " kvh=" + std::to_string(kv_heads) +
              " hd=" + std::to_string(head_dim) + " dtype=bf16 causal=" +
              (causal ? "true" : "false") + " slots=" +
              std::to_string(slots) + " valid=" + std::to_string(valid) +
              " row_blocks=" + std::to_string(row_block_visits) +
              " active_rows=" + std::to_string(active_rows) +
              " quant_elements=" + std::to_string(quant_elements) +
              " qk_acc=" + std::to_string(qk_acc_elements) +
              " pv_acc=" + std::to_string(pv_acc_elements) +
              " actual=" + std::to_string(actual_vec) + " expected=" +
              std::to_string(expected_vec));
  require(summary_u64(plan, "conversion_elements") ==
              quant_elements + accumulator_elements + query_rows * head_dim,
          "conversion summary preserves quant, accumulator, and output element counts");
  require(summary_u64(plan, "hbm_read_bytes_total") == qbytes + 2 * kvbytes,
          "fused BF16 physical reads match query and per-Q-head KV prefixes");
  require(summary_u64(plan, "hbm_write_bytes_total") == outbytes,
          "fused BF16 output writes preserve query/head count");
  const std::uint64_t tile_q = std::min(tq, span.q);
  const std::uint64_t max_kv = span.kv + (span.count - 1) * span.kvstride;
  const std::uint64_t tile_k = std::min(tk, max_kv);
  const std::uint64_t feature_tile = std::min<std::uint64_t>(head_dim, 64);
  auto i4_bytes = [](std::uint64_t rows, std::uint64_t cols) {
    const std::uint64_t elements = rows * cols;
    return elements / 2 + (elements % 2 != 0);
  };
  const std::uint64_t qkv_raw =
      (tile_q * head_dim + 2 * tile_k * head_dim) * 2;
  const std::uint64_t qkv_generated =
      i4_bytes(tile_q, head_dim) + i4_bytes(tile_k, head_dim) +
      i4_bytes(tile_k, feature_tile) + 12;
  const std::uint64_t score_scratch = tile_q * tile_k * 8;
  const std::uint64_t qk_partial = tile_q * tile_k * ceildiv(head_dim, 64) * 4;
  const std::uint64_t pv_partial = tile_q * feature_tile * 12;
  const std::uint64_t online_state = tile_q * (head_dim * 4 + 8);
  const std::uint64_t expected_peak = qkv_raw + qkv_generated + score_scratch +
      qk_partial + pv_partial + online_state + 64;
  require_sram_peak(hw, plan, expected_peak,
                    "fused BF16 to INT4 attention");
  require(plan.summary.at("granularity_validation") == "not_run",
          "independent attention accounting does not claim a certificate");
  require_activity_totals(plan);
  (void)qtiles;
  (void)kv_heads;
}

void exhaustive_small_causal_absolute_positions() {
  const auto hw = example_hardware();
  constexpr std::uint64_t q = 17, kv = 17, qh = 4, kvh = 2, hd = 3;
  const std::string fused = "fused_q_head_shard_v2";
  for (std::uint64_t qs = 0; qs <= 20; ++qs) {
    for (std::uint64_t ks = 0; ks <= 20; ++ks) {
      const RefSpan span{q, kv, 1, qs, ks, 0, 0};
      const auto task = normalized_task("attention", q, kv, qh, kvh, hd,
                                        span, true);
      const auto plans = detail::enumerate_mappings(hw, task);
      for (const std::uint64_t tq : {16, 17}) {
        for (const std::uint64_t tk : {16, 17}) {
          const auto& plan = select_plan(plans, fused, tq, tk);
          verify_fused_bf16_candidate(hw, plan, span, tq, tk, qh, kvh, hd, true);
        }
      }
    }
  }
}

void compact_runs_and_noncausal_tail_accounting() {
  const auto hw = example_hardware();
  constexpr std::uint64_t q = 17, qh = 4, kvh = 2, hd = 3;
  for (std::uint64_t stride = 1; stride <= 5; ++stride) {
    const RefSpan span{q, 17, 3, 0, 0, stride, stride};
    const auto task = normalized_task("attention", q, 17, qh, kvh, hd,
                                      span, true);
    const auto plans = detail::enumerate_mappings(hw, task);
    const std::uint64_t max_kv = 17 + (span.count - 1) * stride;
    for (const std::uint64_t tq : {16, 17}) {
      for (const std::uint64_t tk : {std::uint64_t{16}, max_kv}) {
        verify_fused_bf16_candidate(hw, select_plan(plans, "fused_q_head_shard_v2",
                                               tq, tk),
                                    span, tq, tk, qh, kvh, hd, true);
      }
    }
  }

  const RefSpan noncausal{17, 17, 3, 0, 0, 2, 2};
  const auto task = normalized_task("attention", 17, 17, qh, kvh, hd,
                                    noncausal, false);
  const auto plans = detail::enumerate_mappings(hw, task);
  const std::uint64_t noncausal_max_kv = 17 + (noncausal.count - 1) * noncausal.kvstride;
  for (const std::uint64_t tq : {16, 17}) {
    for (const std::uint64_t tk : {std::uint64_t{16}, noncausal_max_kv}) {
      verify_fused_bf16_candidate(hw, select_plan(plans, "fused_q_head_shard_v2",
                                             tq, tk),
                                  noncausal, tq, tk, qh, kvh, hd, false);
    }
  }
}

void int4_packing_is_per_head_and_per_tile() {
  const auto hw = example_hardware();
  const RefSpan one{3, 5, 1, 2, 0, 0, 0};
  auto task = normalized_task("attention", 3, 5, 4, 2, 3, one, true, "int4");
  const auto native_plans = detail::enumerate_mappings(hw, task);
  const auto& native = select_plan(native_plans, "fused_q_head_shard_v2", 3, 5);
  require(summary_u64(native, "hbm_read_bytes_total") == 84,
          "native INT4 HBM packs Q=5 and each per-head K/V tile=8 bytes");
  require(summary_u64(native, "hbm_write_bytes_total") == 20,
          "native INT4 output packs each odd per-head query tensor");
  require(summary_u64(native, "int4_input_bytes") == 84,
          "native INT4 compute input count preserves every per-head nibble tail");
  require(summary_u64(native, "executed_int4_ops") == 720 &&
              summary_u64(native, "vec_equivalent_ops") == 1152,
          "native INT4 matrix, accumulator, softmax, and output conversion work is closed");
  require(summary_u64(native, "logical_tensor_bytes") == 72,
          "logical INT4 Q/K/V/O tensors are packed once per source head");
  require(summary_u64(native, "sram_read_bytes") == 968 &&
              summary_u64(native, "sram_write_bytes") == 1208,
          "native INT4 bytes are consumed from their HBM landing without fake repacking");
  require_sram_peak(hw, native, 433, "native INT4 fused attention");
  require_activity_totals(native);

  task = normalized_task("attention", 3, 5, 4, 2, 3, one, true, "bf16");
  const auto quantized_plans = detail::enumerate_mappings(hw, task);
  const auto& quantized = select_plan(quantized_plans, "fused_q_head_shard_v2", 3, 5);
  require(summary_u64(quantized, "int4_input_bytes") == 84,
          "generated INT4 buffers pack independently per Q/K/V head tile");
  require(summary_u64(quantized, "executed_int4_ops") == 720 &&
              summary_u64(quantized, "vec_equivalent_ops") == 1668,
          "generated INT4 quantization and accumulator work are both charged");
  require(summary_u64(quantized, "hbm_read_bytes_total") == 312,
          "quantization does not change the BF16 source HBM payload");
  const auto quantized_sram_read = summary_u64(quantized, "sram_read_bytes");
  const auto quantized_sram_write = summary_u64(quantized, "sram_write_bytes");
  require(quantized_sram_read == 1380 && quantized_sram_write == 1620,
          "generated INT4 SRAM service mismatch: shape=q3,kv5,count1,qh4,kvh2,hd3,"
          "dtype=bf16->int4,tm3,tk5 actual_read=" +
              std::to_string(quantized_sram_read) + " expected_read=1380 actual_write=" +
              std::to_string(quantized_sram_write) + " expected_write=1620");
  require_sram_peak(hw, quantized, 523, "generated INT4 fused attention");
  require_activity_totals(quantized);

  task = normalized_task("attention", 3, 5, 4, 2, 3, one, true, "bf16",
                         "fp32_vec_matrix_assumed_v1");
  const auto strict_plans = detail::enumerate_mappings(hw, task);
  const auto& strict = select_plan(strict_plans, "fused_q_head_shard_v2", 3, 5);
  require(summary_u64(strict, "executed_int4_ops") == 0 &&
              summary_u64(strict, "vec_equivalent_ops") == 1728,
          "strict Vec work contains matrix and storage-conversion service without INT4 work");
  require(summary_u64(strict, "sram_read_bytes") == 1488 &&
              summary_u64(strict, "sram_write_bytes") == 1728,
          "strict Vec FP32 conversion buffers are written and consumed explicitly");
  require_sram_peak(hw, strict, 514, "strict Vec fused attention");
  require_activity_totals(strict);

  const RefSpan compact{3, 5, 2, 2, 0, 2, 2};
  task = normalized_task("attention", 3, 5, 4, 2, 3, compact, true, "int4");
  const auto compact_plans = detail::enumerate_mappings(hw, task);
  const auto& compact_native = select_plan(compact_plans, "fused_q_head_shard_v2", 3, 7);
  require(summary_u64(compact_native, "logical_tensor_bytes") == 156,
          "logical INT4 storage packs each compact source tensor/head once");
}

void full_and_standalone_primitives_have_independent_ledgers() {
  const auto hw = example_hardware();
  constexpr std::uint64_t q = 17, qh = 4, kvh = 2, hd = 65;
  const RefSpan span{q, 17, 3, 0, 0, 2, 2};
  const auto ref = brute_reference(span, 16, 64, true, true);
  const std::uint64_t query_rows = q * span.count * qh;
  const std::uint64_t score_slots = q * ref.kv_sum * qh;
  const std::uint64_t qtiles = ceildiv(q, 16);
  const std::uint64_t feature_tiles = ceildiv(hd, 64);
  const std::uint64_t qk_q = q * hd * 2 * qh * ref.kv_tiles;
  const std::uint64_t qk_k = ref.kv_sum * hd * 2 * qtiles * qh;
  const std::uint64_t pv_v = qk_k;
  const std::uint64_t output = query_rows * hd * 2;
  const std::uint64_t fp32_output = query_rows * hd * 4;
  const std::uint64_t score = score_slots * 4;

  const auto full_task = normalized_task("attention", q, 17, qh, kvh, hd,
                                         span, true, "bf16",
                                         "int4_matrix_vec_fp32_assumed_v1",
                                         "materialized");
  const auto full_plans = detail::enumerate_mappings(hw, full_task);
  const auto& full = select_plan(full_plans, "full_score_materialization_v2", 16, 64);
  require(summary_u64(full, "executed_score_slots") == score_slots,
          "full materialization executes the complete compact Q by KV rectangles");
  require(summary_u64(full, "valid_score_pairs") == ref.valid * qh,
          "full dense work preserves the causal source pair count");
  require(summary_u64(full, "hbm_read_bytes_total") ==
              qk_q + qk_k + score + pv_v + score * feature_tiles + fp32_output,
          "full attention HBM reads account for QK, softmax, PV probability rereads, and cast");
  require(summary_u64(full, "hbm_write_bytes_total") ==
              score + score + fp32_output + output,
          "full attention HBM writes account for dense score, probability, FP32 PV, and cast");
  require(summary_u64(full, "logical_tensor_bytes") ==
              query_rows * hd * 4 + ref.kv_sum * hd * 2 * kvh * 2,
          "full materialization workspaces do not replace source logical tensor storage");
  require_sram_peak(hw, full, 21220,
                    "full score materialization with serialized primitive buffers");
  require(full.epochs.size() >= 4 &&
              full.epochs.front().name.find("full_score_qk") == 0,
          "full materialization begins with the QK primitive");
  std::vector<std::string> stages;
  for (const auto& epoch : full.epochs) {
    for (const std::string& stage : {"full_score_qk", "full_score_softmax",
                                     "full_score_pv", "full_score_output_cast"}) {
      if (epoch.name.find(stage) == 0 &&
          (stages.empty() || stages.back() != stage)) {
        stages.push_back(stage);
      }
    }
  }
  if (stages != std::vector<std::string>({"full_score_qk", "full_score_softmax",
                                           "full_score_pv", "full_score_output_cast"})) {
    std::string actual;
    for (const auto& stage : stages) actual += " [" + stage + "]";
    throw std::runtime_error("full score plan stage order differs:" + actual);
  }
  require_activity_totals(full);

  for (const std::string op : {"qk", "av"}) {
    const auto primitive_task = normalized_task(op, q, 17, qh, kvh, hd,
        span, true, "bf16", "fp32_vec_matrix_assumed_v1");
    const auto plans = detail::enumerate_mappings(hw, primitive_task);
    const Plan* selected = nullptr;
    for (const auto& p : plans) {
      if (p->summary.value("kind", std::string()) == op) {
        selected = p.get();
        break;
      }
    }
    require(selected != nullptr, op + " standalone materialized primitive exists");
    const std::uint64_t expected_read = op == "qk"
        ? qk_q + qk_k
        : pv_v + score * feature_tiles;
    const std::uint64_t expected_write = op == "qk" ? score : output;
    require(summary_u64(*selected, "hbm_read_bytes_total") == expected_read,
            op + " primitive reads only its typed input tensors");
    require(summary_u64(*selected, "hbm_write_bytes_total") == expected_write,
            op + " primitive writes only its result tensor");
    require(summary_u64(*selected, "executed_matrix_ops") ==
                2 * hd * score_slots,
            op + " primitive has its independent dense matrix operation count");
    require(summary_u64(*selected, "valid_score_pairs") == ref.valid * qh,
            op + " primitive preserves causal source pairs beside dense execution");
    require(summary_u64(*selected, "executed_int4_ops") == 0,
            op + " strict Vec profile does not fabricate INT4 work");
    require(summary_u64(*selected, "source_matrix_ops") ==
                2 * hd * ref.valid * qh,
            op + " source matrix operations use valid causal pairs");
    require(summary_u64(*selected, "vec_equivalent_ops") >=
                2 * hd * score_slots,
            op + " strict Vec path includes the matrix-equivalent Vec service");
    const auto& selected_tile = select_plan(
        plans, op + "_attention_" + op + "_materialized_v2", 16, 64);
    require_sram_peak(hw, selected_tile, op == "qk" ? 15838 : 17728,
                      op + " standalone strict Vec primitive");
    require_activity_totals(*selected);
  }
}

void test_attention_accounting() {
  exhaustive_small_causal_absolute_positions();
  compact_runs_and_noncausal_tail_accounting();
  int4_packing_is_per_head_and_per_tile();
  full_and_standalone_primitives_have_independent_ledgers();
  std::cout << "attention_tiles=oracle_checked, int4_tail_bytes=exact, "
               "full_score_stages=4, qk_av=independent\n";
}

}  // namespace

int main() { return run_test("attention_accounting_test", test_attention_accounting); }
