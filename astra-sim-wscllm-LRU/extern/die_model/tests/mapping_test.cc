/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

/*
mapping_test.cc -- §5 mapping behaviour through compile() (design doc §10.2
"SRAM与融合" rows).

  - the per-core SRAM budget (0.90 usable) actually shapes the legal tile
    set: the small-SRAM point picks a mapping whose active buffers would not
    fit the big point's chosen tiles and vice versa;
  - contexts=2 costs twice the per-context buffers (checked exactly on the
    row-resident vector mapping);
  - an infeasible fused FFN (row H larger than the whole small-SRAM budget)
    falls back to separate operators with the materialized intermediate
    traffic actually charged;
  - weight residency is never free together with output residency: a large-M
    GEMM pays either weight reloads (output-resident) or extra output/partial
    traffic (weight-resident), while M=1 decode pays exactly one weight pass.

Build (standalone): cmake -S extern/die_model -B <build> && cmake --build
<build> --target mapping_test.  Exit code 0 on ALL PASS.
*/

#include "test_helpers.hh"

using die_test::Checker;
using DieModel::CostResult;
using DieModel::DiePlan;
using DieModel::OpKind;
using DieModel::Phase;

// §5.2 active-buffer formula, element-size generalized (see Mapping.cc).
double sram_gemm(double e, std::uint64_t m, std::uint64_t k, std::uint64_t n,
                 bool dbl) {
  const double mk = static_cast<double>(m * k);
  const double kn = static_cast<double>(k * n);
  const double mn = static_cast<double>(m * n);
  return dbl ? (2.0 * e * mk + 2.0 * e * kn + 4.0 * mn)
             : (e * mk + e * kn + 4.0 * mn);
}

int main() {
  Checker t;
  const DieModel::Hardware big = die_test::hw_default();
  const DieModel::Hardware small = die_test::hw_small();

  // --- SRAM budget changes the legal tile set (§5.1/§5.2/§10.2).
  {
    const DieModel::DieOpSpec s = die_test::spec_of(OpKind::OProj,
                                                    Phase::Prefill, 1024,
                                                    1024, 0, 0);
    const auto plan_big = DieModel::compile(s, big);
    const auto plan_small = DieModel::compile(s, small);
    const DieModel::DiePlanSummary& sb = plan_big->summary();
    const DieModel::DiePlanSummary& ss = plan_small->summary();

    t.expect(sb.sram_peak_per_core <= big.sram_usable_bytes_per_core,
             "big point peak fits its 0.90 budget");
    t.expect(ss.sram_peak_per_core <= small.sram_usable_bytes_per_core,
             "small point peak fits its 0.90 budget");
    // The big point's chosen active buffers would NOT fit the small budget:
    // the feasible tile sets genuinely differ.
    t.expect(sb.sram_peak_per_core > small.sram_usable_bytes_per_core,
             "big point uses buffers the small point cannot hold");
    t.expect(ss.n_tile <= sb.n_tile && ss.m_tile <= sb.m_tile,
             "small point tiles are not larger");
    // Independent recomputation of the §5.2 formula from the summary.
    if (sb.loop_kind == "gemm_output_resident" || sb.loop_kind == "gemm_weight_resident") {
      t.expect_near(sb.sram_peak_per_core,
                    sram_gemm(2.0, sb.m_tile, sb.k_tile, sb.n_tile,
                              sb.double_buffered) *
                        static_cast<double>(sb.contexts),
                    1e-12, "big point peak matches S_gemm x contexts");
    }
    if (ss.loop_kind == "gemm_output_resident") {
      t.expect_near(ss.sram_peak_per_core,
                    sram_gemm(2.0, ss.m_tile, ss.k_tile, ss.n_tile,
                              ss.double_buffered) *
                        static_cast<double>(ss.contexts),
                    1e-12, "small point peak matches S_gemm x contexts");
    }
  }

  // --- contexts=2 needs 2x the per-context buffers: exact check on the
  //     row-resident vector mapping (§5.3).
  {
    const DieModel::DieOpSpec s = die_test::spec_of(OpKind::Norm,
                                                    Phase::Prefill, 1024,
                                                    1024, 0, 0);
    const auto plan = DieModel::compile(s, big);
    const DieModel::DiePlanSummary& sum = plan->summary();
    t.expect(sum.loop_kind == "vector_row_resident",
             "norm stays row-resident on the default point");
    const double e = 2.0, H = 4096.0;
    const double expected =
        (e * H /*input row*/ + e * H /*output row*/) *
            static_cast<double>(sum.contexts) +
        e * H /*gamma row*/;
    t.expect_near(sum.sram_peak_per_core, expected, 1e-12,
                  "vector peak = (in+out)*contexts + gamma row");
    t.expect(sum.sram_peak_per_core <= big.sram_usable_bytes_per_core,
             "vector peak within budget");
  }

  // --- infeasible fused FFN falls back with materialized intermediates
  //     (§5.4).  H = 384*128 = 49152 makes even the minimum fused context
  //     (input + FP32 partial) exceed the small point's 0.25 MB budget.
  {
    const DieModel::DieOpSpec s =
        die_test::spec_of(OpKind::FfnRegion, Phase::Prefill, 1024, 1024, 0, 0,
                          1, 1, 384, 8, 49152, 14336);
    const auto plan = DieModel::compile(s, small);
    const CostResult r = DieModel::estimate_alone(*plan);
    const DieModel::DiePlanSummary& sum = plan->summary();
    t.expect(sum.fallback, "huge-H FFN on small SRAM falls back");
    t.expect(!sum.fused, "fallback plan is not fused");
    t.expect(sum.loop_kind == "ffn_fallback_separate", "fallback loop kind");
    // Materialized-intermediate floor: the gate/up outputs (2*e*M*F) and
    // the activation (e*M*F) cross the external memory at least once each,
    // on top of both weight slabs and the input/output rows.
    const double e = 2.0, M = 1024.0, H = 49152.0, F = 14336.0;
    const double floor_bytes = 2.0 * e * H * F /*gate+up weights*/ +
                               e * H * F /*down weights*/ +
                               2.0 * e * M * H /*up in + down out rows*/ +
                               2.0 * e * M * F + e * M * F /*intermediates*/;
    t.expect(r.matrix_actual_bytes >= floor_bytes,
             "fallback charges the materialized intermediates");
    t.expect(r.fallback && !r.fused, "CostResult carries the fallback flags");
  }

  // --- weight residency is never free together with output residency
  //     (§5.2 "不能同时免费获得全部权重复用与全部输出驻留").
  {
    // M=1 decode GEMM: exactly one weight pass, no reload multiplier.
    const DieModel::DieOpSpec dec = die_test::spec_of(
        OpKind::OProj, Phase::Decode, 1, 1024, 1023, 0);
    const auto plan_dec = DieModel::compile(dec, big);
    const CostResult r_dec = DieModel::estimate_alone(*plan_dec);
    const double w_once = 2.0 * 4096.0 * 4096.0;  // e*K*N
    t.expect_near(plan_dec->summary().weight_hbm_bytes, w_once, 1e-9,
                  "M=1 decode reads the weights exactly once");

    // M=8192 prefill (T_m = 64 > mesh_n = 35 row blocks): output-resident
    // needs ceil(T_m/rb) >= 2 weight sweeps; weight-resident pays
    // partial/output traffic.  The chosen plan must pay at least one of the
    // two (§5.2 "不能同时免费获得全部权重复用与全部输出驻留").
    const DieModel::DieOpSpec bigm = die_test::spec_of(
        OpKind::OProj, Phase::Prefill, 8192, 8192, 0, 0);
    const auto plan_bigm = DieModel::compile(bigm, big);
    const CostResult r_bigm = DieModel::estimate_alone(*plan_bigm);
    const bool reloads_charged =
        plan_bigm->summary().weight_hbm_bytes > 1.5 * w_once;
    const double io_floor = w_once /*weights*/ +
                            2.0 * 8192.0 * 4096.0 /*A*/ +
                            2.0 * 8192.0 * 4096.0 /*Y*/;
    const bool partial_charged = r_bigm.matrix_actual_bytes > 1.05 * io_floor;
    t.expect(reloads_charged || partial_charged,
             "large-M GEMM pays weight reloads or partial/output traffic");
  }

  if (t.ok) {
    std::printf("[mapping_test] ALL PASS\n");
    return 0;
  }
  return 1;
}
