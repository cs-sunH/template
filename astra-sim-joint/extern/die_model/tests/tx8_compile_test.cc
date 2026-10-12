/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

/*
tx8_compile_test.cc -- tx8_reference_v1 profile compile (design doc §1.1/
§2.2/§2.3; contract §4).

Every op_kind that maps to one reference timed group must compile into the
aggregated-streaming envelope and estimate EXACTLY the §2.3 workbook value
(abs err < 1e-6 us).  Any non-§2.1 shape must be rejected with the "tx8
profile only reproduces the reference shapes" error.

Build (standalone): cmake -S extern/die_model -B <build> && cmake --build
<build> --target tx8_compile_test.  Exit code 0 on ALL PASS.
*/

#include "test_helpers.hh"

#include <cmath>

using die_test::Checker;
using DieModel::CostResult;
using DieModel::OpKind;
using DieModel::Phase;

int main() {
  Checker t;

  // The hardware geometry is irrelevant to the reference profile (the §2.1
  // constants are die-independent); reuse a legal wrapped node.
  const DieModel::Hardware hw = DieModel::make_hardware(die_test::wrapped(
      die_test::die_node(0.75, "large", 800, 3, 2, 512), "tx8_reference_v1"));
  const DieModel::Tx8Result ref = DieModel::tx8_reference_block();

  // op_kind -> reference group index (§2.3 order).
  struct Anchor {
    OpKind kind;
    std::size_t group;
    const char* name;
  };
  const Anchor anchors[] = {
      {OpKind::Norm, 0, "RMSNorm_1"},
      {OpKind::OProj, 1, "Q_proj"},
      {OpKind::Qk, 5, "QKT"},
      {OpKind::Av, 7, "AV/PV"},
      {OpKind::OProj, 8, "O_proj"},   // same formulas, same number
      {OpKind::Residual, 9, "res_add_1"},
      {OpKind::Residual, 12, "res_add_2"},
      {OpKind::FfnRegion, 11, "FFN"},
  };
  for (const Anchor& a : anchors) {
    const DieModel::DieOpSpec s =
        die_test::spec_of(a.kind, Phase::Prefill, 1024, 1024, 0, 0);
    const auto plan = DieModel::compile(s, hw);
    const CostResult r = DieModel::estimate_alone(*plan);
    const double got_us = r.total_seconds * 1e6;
    const double want_us = ref.groups[a.group].total_s * 1e6;
    t.expect(std::abs(got_us - want_us) < 1e-6,
             std::string(a.name) + ": estimate " + std::to_string(got_us) +
                 " us vs reference " + std::to_string(want_us) + " us");
    t.expect(plan->summary().profile == DieModel::Profile::Tx8ReferenceV1,
             std::string(a.name) + ": plan carries the tx8 profile");
    t.expect(plan->summary().loop_kind == "tx8_reference_envelope",
             std::string(a.name) + ": reference envelope loop kind");
  }

  // layer_count scales the envelope linearly and keeps launches at
  // layer_count x weight_passes (with the zero §2.1 launch cost).
  {
    const DieModel::DieOpSpec s = die_test::spec_of(OpKind::Norm,
                                                    Phase::Prefill, 1024,
                                                    1024, 0, 0, 2, 1);
    const auto plan = DieModel::compile(s, hw);
    const CostResult r = DieModel::estimate_alone(*plan);
    t.expect(std::abs(r.total_seconds - 2.0 * ref.groups[0].total_s) < 1e-15,
             "layer_count=2 doubles the reference group time");
    t.expect_uint(r.launch_instances, 2, "tx8 launch instances");
    t.expect(r.launch_seconds == 0.0, "tx8 extra_launch is zero (§2.1)");
  }

  // --- rejections: anything that is not the §2.1 reference shape.
  const auto reject = [&](const DieModel::DieOpSpec& s,
                          const std::string& what) {
    t.expect_throw([&] { DieModel::compile(s, hw); }, what,
                   "tx8_reference_v1 profile only reproduces");
  };

  reject(die_test::spec_of(OpKind::OProj, Phase::Prefill, 512, 512, 0, 0),
         "wrong Sq (512)");
  reject(die_test::spec_of(OpKind::OProj, Phase::Prefill, 1024, 2048, 0, 0),
         "wrong Sk (2048)");
  reject(die_test::spec_of(OpKind::OProj, Phase::Decode, 1, 1024, 1023, 0),
         "decode phase");
  reject(die_test::spec_of(OpKind::OProj, Phase::Prefill, 1024, 1024, 0, 0,
                           1, 2),
         "weight_passes=2");
  reject(die_test::spec_of(OpKind::OProj, Phase::Prefill, 1024, 1024, 0, 0,
                           1, 1, 16, 4),
         "non-reference head counts");
  reject(die_test::spec_of(OpKind::AttentionRegion, Phase::Prefill, 1024,
                           1024, 0, 0),
         "attention_region keeps scores on tile");
  reject(die_test::spec_of(OpKind::Logits, Phase::Prefill, 1024, 1024, 0, 0),
         "logits is not part of the reference block");
  reject(die_test::spec_of(OpKind::LocalRegion, Phase::Prefill, 1, 1, 0, 0),
         "local_region is not part of the reference block");
  {
    DieModel::DieOpSpec s = die_test::spec_of(OpKind::Norm, Phase::Prefill,
                                              1024, 1024, 0, 0);
    s.dtype = DieModel::Dtype::Fp32;
    reject(s, "fp32 dtype");
  }
  {
    DieModel::DieOpSpec two = die_test::spec_of(OpKind::Norm, Phase::Prefill,
                                                1024, 1024, 0, 0);
    two = die_test::add_span(two, "r1", 1, 1, 0, 0);
    reject(two, "two spans");
  }
  {
    // TP sharding breaks the reference identity (TP=1 extents required).
    DieModel::DieOpSpec s = die_test::spec_of(OpKind::Norm, Phase::Prefill,
                                              1024, 1024, 0, 0);
    s.tp_slice.tensor_parallel = 2;
    s.tp_slice.tensor_parallel_rank = 1;
    s.tp_slice.ffn_local = 7168;
    reject(s, "TP=2 shard");
  }

  // Both profiles share the workload semantic checks: an underived spec
  // still parses through derive_workload inside compile.
  {
    const DieModel::DieOpSpec ok = die_test::spec_of(OpKind::Qk,
                                                     Phase::Prefill, 1024,
                                                     1024, 0, 0);
    const auto plan = DieModel::compile(ok, hw);
    const CostResult r = DieModel::estimate_alone(*plan);
    t.expect_near(r.flops,
                  2.0 * 32.0 * 1024.0 * 1024.0 * 128.0, 1e-9,
                  "tx8 qk FLOP goes through the shared workload semantics");
  }

  if (t.ok) {
    std::printf("[tx8_compile_test] ALL PASS\n");
    return 0;
  }
  return 1;
}
