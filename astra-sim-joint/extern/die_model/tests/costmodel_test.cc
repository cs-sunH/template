/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

/*
costmodel_test.cc -- §6.1 stage model and estimate_alone invariants
(design doc §10.2 "物理下界"/"扫参有效性"/"内存效率" rows).

  - physical lower bounds: elapsed >= total FLOP / (eta*PE die peak) and
    elapsed >= class bytes / (eta_class * HBM bandwidth);
  - no unaccounted compute: the plan's FLOP equals the op's math exactly;
  - sweep validity: adding HBM stacks (same mesh geometry) keeps the mapping
    and does not worsen the isolated cost;
  - per-core SRAM peak <= 0.90 budget on both hardware points;
  - service units = actual bytes / eta (balanced striping);
  - launch accounting = macro instances (layer_count x weight_passes);
  - NoC max_edge semantics on a hand-computed 4-stack topology.

Build (standalone): cmake -S extern/die_model -B <build> && cmake --build
<build> --target costmodel_test.  Exit code 0 on ALL PASS.
*/

#include "test_helpers.hh"

using die_test::Checker;
using DieModel::CostResult;
using DieModel::DiePlan;
using DieModel::OpKind;
using DieModel::Phase;

int main() {
  Checker t;
  const DieModel::Hardware big = die_test::hw_default();

  // --- physical lower bounds + exact FLOP accounting (o_proj, matrix only).
  {
    const DieModel::DieOpSpec s = die_test::spec_of(
        OpKind::OProj, Phase::Prefill, 1024, 1024, 0, 0, 2, 1);
    const auto plan = DieModel::compile(s, big);
    const CostResult r = DieModel::estimate_alone(*plan);
    t.expect(r.total_seconds >=
                     r.flops / (big.eta_pe * big.pe_peak_die) - 1e-18,
             "elapsed >= FLOP/(eta*PE die peak)");
    t.expect(r.total_seconds >=
                     r.matrix_actual_bytes /
                         (big.eta_matrix_mem * big.hbm_bandwidth_bytes) -
                     1e-18,
             "elapsed >= matrix bytes/(eta_m*B_hbm)");
    t.expect_near(r.flops, 2.0 * 1024.0 * 4096.0 * 4096.0 * 2.0, 1e-12,
                  "plan FLOP = 2*M*K*N*layers exactly");
    t.expect_near(r.eqops, 0.0, 0.0, "o_proj has no vector eq-ops");
  }

  // --- vector-class lower bound (norm) on the small point too.
  {
    const DieModel::Hardware small = die_test::hw_low_sram();
    const DieModel::DieOpSpec s = die_test::spec_of(
        OpKind::Norm, Phase::Prefill, 2048, 2048, 0, 0, 1, 1);
    const auto plan = DieModel::compile(s, small);
    const CostResult r = DieModel::estimate_alone(*plan);
    t.expect(r.total_seconds >=
                     r.vector_actual_bytes /
                         (small.eta_vector_mem * small.hbm_bandwidth_bytes) -
                     1e-18,
             "norm elapsed >= vector bytes/(eta_v*B_hbm)");
    t.expect_near(r.eqops, 6.0 * 2048.0 * 4096.0, 1e-12,
                  "norm eqops = 6*M*H");
    t.expect(r.sram_peak_per_core <= small.sram_usable_bytes_per_core,
             "small point peak within 0.90 budget");
  }

  // --- attention region satisfies both bounds on the default point.
  {
    const DieModel::DieOpSpec s = die_test::spec_of(
        OpKind::AttentionRegion, Phase::Prefill, 1024, 1024, 0, 0);
    const auto plan = DieModel::compile(s, big);
    const CostResult r = DieModel::estimate_alone(*plan);
    t.expect(r.total_seconds >=
                     r.flops / (big.eta_pe * big.pe_peak_die) - 1e-18,
             "attention elapsed >= FLOP bound");
    t.expect(r.total_seconds >=
                     r.matrix_actual_bytes /
                         (big.eta_matrix_mem * big.hbm_bandwidth_bytes) -
                     1e-18,
             "attention elapsed >= matrix byte bound");
    t.expect(r.sram_peak_per_core <= big.sram_usable_bytes_per_core,
             "attention peak within 0.90 budget");
  }

  // --- sweep validity (§10.2): more HBM stacks, same mesh -> same mapping,
  //     cost not worse.
  {
    const DieModel::Hardware big6 = die_test::hw_default_6hbm();
    t.expect_uint(big6.mesh_n, big.mesh_n, "same mesh_n across the sweep");
    t.expect_uint(big6.cores, big.cores, "same core count across the sweep");
    // The isolated cost must not worsen for any op; the tile mapping is
    // asserted on the GEMM path, where it is bound by the compute/weight
    // structure and not by the injection-port layout.  (Vector/attention
    // mappings may legitimately re-select contexts/groups when the extra
    // bandwidth changes which resource binds; recorded in the A2 report.)
    for (DieModel::OpKind kind :
         {OpKind::OProj, OpKind::Norm, OpKind::AttentionRegion}) {
      const DieModel::DieOpSpec s =
          die_test::spec_of(kind, Phase::Prefill, 1024, 1024, 0, 0);
      const auto p4 = DieModel::compile(s, big);
      const auto p6 = DieModel::compile(s, big6);
      const CostResult r4 = DieModel::estimate_alone(*p4);
      const CostResult r6 = DieModel::estimate_alone(*p6);
      const DieModel::DiePlanSummary& a = p4->summary();
      const DieModel::DiePlanSummary& b = p6->summary();
      if (kind == OpKind::OProj) {
        t.expect(a.m_tile == b.m_tile && a.n_tile == b.n_tile &&
                     a.k_tile == b.k_tile && a.group_rows == b.group_rows &&
                     a.group_cols == b.group_cols &&
                     a.contexts == b.contexts &&
                     a.k_parallel == b.k_parallel &&
                     a.loop_kind == b.loop_kind,
                 "GEMM mapping preserved across the HBM sweep");
      }
      t.expect(r6.total_seconds <= r4.total_seconds * (1.0 + 1e-12),
               std::string("cost not worse with more HBM stacks for ") +
                   DieModel::to_string(kind));
    }
  }

  // --- service units = actual bytes / eta under balanced striping; launch
  //     = macro instances.
  {
    // Norm, M=1, H=4096, BF16, (500 mm2, 2x2 stacks, 256 bit): every stage's
    // bytes divide evenly across the 4 stacks, so the §6.2 granularity rule
    // is exactly sum(U_j).
    const DieModel::Hardware hw4 = DieModel::make_hardware(die_test::wrapped(
        die_test::die_node(0.75, 500, 2, 2, 256, 18, 18), "mesh_die_v1"));
    t.expect_uint(hw4.hbm_count, 4, "4-stack point");
    const DieModel::DieOpSpec s = die_test::spec_of(OpKind::Norm,
                                                    Phase::Decode, 1, 1024,
                                                    1023, 0, 1, 1);
    const auto plan = DieModel::compile(s, hw4);
    const CostResult r = DieModel::estimate_alone(*plan);
    t.expect_near(r.vector_actual_bytes,
                  2.0 * 1.0 * 4096.0 /*in*/ + 2.0 * 1.0 * 4096.0 /*out*/ +
                      2.0 * 4096.0 /*gamma*/,
                  1e-12, "norm M=1 bytes");
    t.expect_near(r.vector_service_units,
                  r.vector_actual_bytes / hw4.eta_vector_mem, 1e-9,
                  "service units = actual/eta (balanced striping)");
    t.expect(r.matrix_actual_bytes == 0.0 && r.matrix_service_units == 0.0,
             "norm has no matrix-class bytes");

    // Launch: layer_count x weight_passes macro instances at 1 us each.
    const DieModel::DieOpSpec s2 = die_test::spec_of(OpKind::OProj,
                                                     Phase::Prefill, 1024,
                                                     1024, 0, 0, 3, 2);
    const auto plan2 = DieModel::compile(s2, big);
    const CostResult r2 = DieModel::estimate_alone(*plan2);
    t.expect_uint(r2.launch_instances, 6, "launch instances = L*P");
    t.expect_near(r2.launch_seconds, 6.0e-6, 1e-15, "launch seconds = 6 us");
    t.expect(r2.total_seconds >= r2.launch_seconds,
             "launch is part of the elapsed time");
  }

  // --- NoC max_edge semantics: residual M=1 on a 1x1 core group of the
  //     (500 mm2, 2x2, 256 bit) point (mesh 28x28).  The 16384 input bytes
  //     (two BF16 operand rows) stripe 4096/stack; the two bottom stacks
  //     both route through the col-0 vertical corridor, so the busiest
  //     directed edge carries 2 x 16384/4 = 8192 bytes.
  {
    const DieModel::Hardware hw4 = DieModel::make_hardware(die_test::wrapped(
        die_test::die_node(0.75, 500, 2, 2, 256, 18, 18), "mesh_die_v1"));
    t.expect_uint(hw4.mesh_n, 18, "explicit 18x18 mesh on the 4-stack fixture");
    const DieModel::DieOpSpec s = die_test::spec_of(OpKind::Residual,
                                                    Phase::Decode, 1, 1, 0,
                                                    0);
    const auto plan = DieModel::compile(s, hw4);
    const DieModel::DiePlanSummary& sum = plan->summary();
    t.expect_uint(sum.group_rows * sum.group_cols, 1,
                  "M=1 forces a 1x1 core group");
    const CostResult r = DieModel::estimate_alone(*plan);
    t.expect_near(r.noc_edge_bytes_max, 2.0 * (2.0 * 2.0 * 4096.0) / 4.0, 1e-9,
                  "max directed edge carries two stack shares (8192 B)");
    t.expect(r.noc_link_bytes_total >= r.noc_edge_bytes_max,
             "link total >= max edge");
  }

  // --- plan cache key: deterministic, distinguishes every relevant field.
  {
    const DieModel::DieOpSpec a = die_test::spec_of(OpKind::Norm,
                                                    Phase::Prefill, 1024,
                                                    1024, 0, 0);
    const std::string k1 = DieModel::plan_cache_key(a, big);
    const std::string k2 = DieModel::plan_cache_key(a, big);
    t.expect(k1 == k2, "cache key deterministic");
    t.expect(k1.size() == 16, "cache key is 16 hex digits");
    DieModel::DieOpSpec b = a;
    b.spans[0].kv_tokens = 1025;
    t.expect(DieModel::plan_cache_key(b, big) != k1,
             "cache key distinguishes kv lengths");
    b = a;
    b.layer_count = 2;
    t.expect(DieModel::plan_cache_key(b, big) != k1,
             "cache key distinguishes layer_count");
    b = a;
    b.dtype = DieModel::Dtype::Fp32;
    t.expect(DieModel::plan_cache_key(b, big) != k1,
             "cache key distinguishes dtype");
    b = a;
    b.tp_slice.q_heads_local = 16;
    t.expect(DieModel::plan_cache_key(b, big) != k1,
             "cache key distinguishes TP shards");
    t.expect(DieModel::plan_cache_key(a, die_test::hw_low_sram()) != k1,
             "cache key distinguishes hardware");

    DieModel::PlanCache cache(2);
    const auto p = DieModel::compile(a, big);
    cache.put(k1, p);
    t.expect_uint(cache.size(), 1, "cache size after one put");
    t.expect(cache.get(k1) == p, "cache returns the same plan");
    cache.put("x", p);
    cache.put("y", p);
    t.expect_uint(cache.size(), 2, "LRU eviction keeps the capacity");
    // k1 is the least recently used entry and is the one evicted.
    t.expect(cache.get(k1) == nullptr, "LRU victim (k1) is evicted");
    t.expect(cache.get("y") == p, "most recent entry survives");
    t.expect(cache.get("x") == p, "second-most-recent entry survives");
  }

  // --- estimate_alone and compile share one stage model: re-estimating a
  //     plan is stable.
  {
    const DieModel::DieOpSpec s = die_test::spec_of(OpKind::QkvRope,
                                                    Phase::Prefill, 512, 512,
                                                    0, 0);
    const auto plan = DieModel::compile(s, big);
    const CostResult r1 = DieModel::estimate_alone(*plan);
    const CostResult r2 = DieModel::estimate_alone(*plan);
    t.expect(r1.total_seconds == r2.total_seconds,
             "estimate_alone is deterministic");
    t.expect_uint(r1.total_ns_suggested,
                  r1.total_seconds > 0.0
                      ? static_cast<std::uint64_t>(
                            std::max(1.0, std::ceil(r1.total_seconds * 1e9)))
                      : 0,
                  "ns suggestion = max(1, ceil(total*1e9))");
  }

  if (t.ok) {
    std::printf("[costmodel_test] ALL PASS\n");
    return 0;
  }
  return 1;
}
