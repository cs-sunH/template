/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

/*
workload_test.cc -- §4.2 workload determination (derive_workload).

Covers the §10.2 "正确的形状依赖" rows that live in the workload layer:
TP tail shards (32 heads / TP=6, empty KV shard), M=1 decode unpadded,
different KV lengths, multiple ragged spans (two independent q=1,k=1024 vs
one k=2048), weight_passes multiplying only weight traffic, GQA vs MHA head
counts, and absolute-position causal tile validity (mid-prefill chunk and a
fully-future span).

Build (standalone): cmake -S extern/die_model -B <build> && cmake --build
<build> --target workload_test.  Exit code 0 on ALL PASS.
*/

#include "test_helpers.hh"

using die_test::Checker;
using DieModel::DieOpSpec;
using DieModel::OpKind;
using DieModel::Phase;
using DieModel::WorkloadQuantities;

int main() {
  Checker t;

  // --- TP tail shards: heads=32 / TP=6 -> q extents {6,6,6,6,6,2};
  //     kv extents {2,2,2,1,1,0}.  QKV is three real GEMMs on the true
  //     local head counts (§4.2 rule 3); the empty KV shard is legal zero
  //     work for K/V (rule 5).
  {
    const std::uint64_t q_extents[6] = {6, 6, 6, 6, 6, 2};
    const std::uint64_t kv_extents[6] = {2, 2, 2, 1, 1, 0};
    double flops_sum = 0.0;
    double weight_sum = 0.0;
    for (int r = 0; r < 6; ++r) {
      const DieOpSpec s = die_test::spec_of(
          OpKind::QkvRope, Phase::Prefill, 1024, 1024, 0, 0, 1, 1,
          q_extents[r], kv_extents[r]);
      const WorkloadQuantities q = DieModel::derive_workload(s);
      const double expect_flops =
          2.0 * 1024.0 * 4096.0 *
          (static_cast<double>(q_extents[r]) * 128.0 +
           2.0 * static_cast<double>(kv_extents[r]) * 128.0);
      t.expect_near(q.flops_per_layer, expect_flops, 1e-12,
                    "qkv flops of rank " + std::to_string(r));
      const double expect_weight =
          2.0 * 4096.0 *
          (static_cast<double>(q_extents[r]) * 128.0 +
           2.0 * static_cast<double>(kv_extents[r]) * 128.0);
      t.expect_near(q.weight_bytes_per_pass, expect_weight, 1e-12,
                    "qkv weight of rank " + std::to_string(r));
      if (kv_extents[r] == 0) {
        // Empty KV shard: zero K/V work, but the Q GEMM and its traffic are
        // still real ("空分片是合法的零工作；有实际访存的零 FLOP 操作不是零工作").
        t.expect_near(q.flops_per_layer,
                      2.0 * 1024.0 * 4096.0 *
                          (static_cast<double>(q_extents[r]) * 128.0),
                      1e-12, "empty-kv rank flops are Q-only");
        t.expect(q.flops_per_layer > 0.0 && q.act_in_bytes_per_layer > 0.0,
                 "empty-kv rank is not zero work");
      }
      flops_sum += q.flops_per_layer;
      weight_sum += q.weight_bytes_per_pass;
    }
    const WorkloadQuantities full = DieModel::derive_workload(die_test::spec_of(
        OpKind::QkvRope, Phase::Prefill, 1024, 1024, 0, 0, 1, 1, 32, 8));
    t.expect_near(flops_sum, full.flops_per_layer, 1e-9,
                  "TP shards sum to the full-shard flops");
    t.expect_near(weight_sum, full.weight_bytes_per_pass, 1e-9,
                  "TP shards sum to the full-shard weight bytes");
  }

  // --- M=1 decode is never padded to 1024 (§4.2 rule 7).
  {
    const WorkloadQuantities q1 = DieModel::derive_workload(die_test::spec_of(
        OpKind::Norm, Phase::Decode, 1, 1024, 1023, 0));
    t.expect_near(q1.eqops_per_layer, 6.0 * 1.0 * 4096.0, 1e-12,
                  "decode M=1 norm eqops");
    t.expect_uint(q1.m_tokens, 1, "decode M=1 m_tokens");
    const WorkloadQuantities q2 = DieModel::derive_workload(die_test::spec_of(
        OpKind::Norm, Phase::Prefill, 1024, 1024, 0, 0));
    t.expect_near(q2.eqops_per_layer, 6.0 * 1024.0 * 4096.0, 1e-12,
                  "prefill M=1024 norm eqops");
    t.expect(q1.eqops_per_layer * 1024.0 == q2.eqops_per_layer,
             "decode/prefill ratio is exactly M");
  }

  // --- different KV lengths scale the attention floor (q at the end of
  //     its KV so the whole range is causally valid).
  {
    const WorkloadQuantities s1 = DieModel::derive_workload(die_test::spec_of(
        OpKind::Qk, Phase::Prefill, 256, 512, 512, 0));
    const WorkloadQuantities s2 = DieModel::derive_workload(die_test::spec_of(
        OpKind::Qk, Phase::Prefill, 256, 1024, 1024, 0));
    t.expect_near(s2.flops_per_layer, 2.0 * s1.flops_per_layer, 1e-12,
                  "qk flops scale with kv_tokens");
    t.expect_near(s2.kv_bytes_per_layer, 2.0 * s1.kv_bytes_per_layer, 1e-12,
                  "qk K/V bytes scale with kv_tokens");
  }

  // --- multiple ragged spans: attention per span, never sum(q)*sum(k)
  //     (§4.2 rule 2).  Two independent q=1,k=1024 decode requests are two
  //     attententions (2*1024 pairs), not one merged sequence.
  {
    DieOpSpec two = die_test::spec_of(OpKind::Qk, Phase::Decode, 1, 1024,
                                      1023, 0);
    two = die_test::add_span(two, "r1", 1, 1024, 1023, 0);
    const WorkloadQuantities w2 = DieModel::derive_workload(two);
    t.expect_near(w2.attention_pairs, 32.0 * 2048.0, 1e-12,
                  "two spans of (q=1,k=1024) give 2*1024 pairs per head");

    const WorkloadQuantities one = DieModel::derive_workload(die_test::spec_of(
        OpKind::Qk, Phase::Decode, 1, 2048, 2047, 0));
    t.expect_near(one.attention_pairs, 32.0 * 2048.0, 1e-12,
                  "one span (q=1,k=2048) gives the same 2048 pairs");

    // The merged misreading sum(q)*sum(k) = 2*2048 pairs per head must not
    // appear.
    t.expect(w2.attention_pairs != 32.0 * 4096.0,
             "two spans are not sum(q)*sum(k)");
    // But the two-request batch is still TWO tokens of linear work, and the
    // Q input of the single long request is half of it.
    t.expect_uint(w2.m_tokens, 2, "two-span M is 2");
    t.expect_uint(one.m_tokens, 1, "one-span M is 1");
    t.expect_near(w2.act_in_bytes_per_layer,
                  2.0 * one.act_in_bytes_per_layer, 1e-12,
                  "two spans read twice the Q bytes");
    // Spans with disjoint KV never merge their KV either.
    t.expect_near(w2.kv_bytes_per_layer,
                  2.0 * 2.0 * 8.0 * 128.0 * 2048.0, 1e-12,
                  "two spans read their own K/V");
  }

  // --- weight_passes multiplies ONLY weight traffic; launches are
  //     layer_count x weight_passes macro instances (§6.1 / A2 ruling R-W).
  {
    const WorkloadQuantities p1 = DieModel::derive_workload(die_test::spec_of(
        OpKind::Norm, Phase::Prefill, 1024, 1024, 0, 0, 2, 1));
    const WorkloadQuantities p3 = DieModel::derive_workload(die_test::spec_of(
        OpKind::Norm, Phase::Prefill, 1024, 1024, 0, 0, 2, 3));
    t.expect_near(p3.eqops_per_layer, p1.eqops_per_layer, 0.0,
                  "weight_passes does not change eqops");
    t.expect_near(p3.act_in_bytes_per_layer, p1.act_in_bytes_per_layer, 0.0,
                  "weight_passes does not change activation reads");
    t.expect_near(p3.act_out_bytes_per_layer, p1.act_out_bytes_per_layer, 0.0,
                  "weight_passes does not change activation writes");
    t.expect_near(p3.total_weight_bytes, 3.0 * p1.total_weight_bytes, 1e-12,
                  "weight bytes scale with weight_passes");
    t.expect_uint(p1.launch_instances, 2, "launch instances L*P (P=1)");
    t.expect_uint(p3.launch_instances, 6, "launch instances L*P (P=3)");
    t.expect_near(p3.total_eqops, p1.total_eqops, 0.0,
                  "total eqops unchanged by weight_passes");
  }

  // --- GQA (kv=8, q=32): K/V stay at the compact kv head count; every q
  //     head pays its own attention pairs (§4.2 rule 3, §5.5).
  {
    const WorkloadQuantities gqa = DieModel::derive_workload(die_test::spec_of(
        OpKind::QkvRope, Phase::Prefill, 512, 512, 0, 0, 1, 1, 32, 8));
    t.expect_near(gqa.flops_per_layer,
                  2.0 * 512.0 * 4096.0 * (32.0 * 128.0 + 2.0 * 8.0 * 128.0),
                  1e-12, "GQA qkv flops use true local head counts");
    const WorkloadQuantities attn = DieModel::derive_workload(
        die_test::spec_of(OpKind::AttentionRegion, Phase::Prefill, 512, 512,
                          512, 0, 1, 1, 32, 8));
    t.expect_near(attn.kv_bytes_per_layer,
                  2.0 * 2.0 * 8.0 * 128.0 * 512.0, 1e-12,
                  "GQA K/V bytes use the compact kv head count");
    t.expect_near(attn.attention_pairs, 32.0 * 512.0 * 512.0, 1e-12,
                  "GQA pairs per q head over the full rectangle (q_start 0)");
  }

  // --- MHA (num_kv == num_q) is the same formulas at kv=32.
  {
    const WorkloadQuantities mha = DieModel::derive_workload(die_test::spec_of(
        OpKind::QkvRope, Phase::Prefill, 512, 512, 0, 0, 1, 1, 32, 32));
    t.expect_near(mha.flops_per_layer,
                  2.0 * 512.0 * 4096.0 * (32.0 * 128.0 + 2.0 * 32.0 * 128.0),
                  1e-12, "MHA qkv flops");
    t.expect_near(mha.kv_bytes_per_layer,
                  0.0, 0.0, "qkv_rope itself carries no K/V reads");
  }

  // --- absolute-position causal validity (§4.2 rule 7): mid-prefill chunk
  //     keeps its history, only strictly-future KV is absent.
  {
    // (a) chunk q=128 starting at absolute 512 with kv [0,1152):
    //     valid = sum_{t=0..127}(513+t) = 73792 < rectangle 147456.
    const WorkloadQuantities a = DieModel::derive_workload(
        die_test::spec_of(OpKind::AttentionRegion, Phase::Prefill, 128, 1152,
                          512, 0, 1, 1, 32, 8));
    t.expect_uint(a.attention_spans[0].valid_pairs, 73792,
                  "mid-prefill valid pairs (a)");
    t.expect_uint(a.attention_spans[0].rectangle_pairs, 128 * 1152,
                  "mid-prefill rectangle (a)");
    // (b) chunk starting at 1024 with kv [0,1152): all of its KV is in the
    //     past, every pair valid.
    const WorkloadQuantities b = DieModel::derive_workload(
        die_test::spec_of(OpKind::AttentionRegion, Phase::Prefill, 128, 1152,
                          1024, 0, 1, 1, 32, 8));
    t.expect_uint(b.attention_spans[0].valid_pairs, 139328,
                  "mid-prefill valid pairs (b)");
    // (c) fully-future KV range: zero valid pairs, non-zero rectangle.
    const WorkloadQuantities c = DieModel::derive_workload(
        die_test::spec_of(OpKind::AttentionRegion, Phase::Prefill, 128, 1024,
                          0, 4096, 1, 1, 32, 8));
    t.expect_uint(c.attention_spans[0].valid_pairs, 0,
                  "fully-future span has zero valid pairs");
    t.expect_uint(c.attention_spans[0].rectangle_pairs, 128 * 1024,
                  "fully-future rectangle still non-zero");
    t.expect_near(c.attention_pairs, 0.0, 0.0, "attention floor is zero");
  }

  // --- RoPE rotates only the new Q/K, cos/sin over the actual positions
  //     (§5.3): 2 tables x elem x d x M bytes.
  {
    const WorkloadQuantities q = DieModel::derive_workload(die_test::spec_of(
        OpKind::QkvRope, Phase::Prefill, 1024, 4096, 3072, 0));
    t.expect_near(q.eqops_per_layer,
                  3.0 * 1024.0 * (32.0 * 128.0 + 8.0 * 128.0), 1e-12,
                  "rope eqops 3*M*(qh+kvh)*d");
    t.expect_near(q.aux_bytes_per_layer, 2.0 * 2.0 * 128.0 * 1024.0, 1e-12,
                  "cos/sin bytes 2*e*d*positions");
  }

  if (t.ok) {
    std::printf("[workload_test] ALL PASS\n");
    return 0;
  }
  return 1;
}
