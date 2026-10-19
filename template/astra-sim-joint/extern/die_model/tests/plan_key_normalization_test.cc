/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

/*
plan_key_normalization_test.cc -- the key v3 span normalization (batch 2 of
the joint zero-semantics performance pass) and the cache-equivalence
classes it creates.

  - group 1: every NON-attention op_kind (10).  Two specs whose spans
    differ only in per-span split / start positions / kv content but share
    the q_tokens sum produce the SAME plan-cache key and BITWISE-IDENTICAL
    compile products (summary fields, stage fields including deps, and the
    eight scalar totals).  These ops consume the spans only through
    WorkloadQuantities::m_tokens, so one compiled plan legitimately serves
    both specs.
  - group 2: every ATTENTION op_kind (5).  The per-span v2 keying is kept:
    moved positions, and a different split with the same q_tokens sum, both
    still produce distinct keys.
  - group 3: tx8_reference_v1 keeps the v2 key and its fail-closed shape
    gate: kv != 1024 still throws out of compile, the kv=1024 and kv=2048
    keys differ, and the mesh-profile key differs from the tx8 key.

Build (standalone): cmake -S extern/die_model -B <build> && cmake --build
<build> --target plan_key_normalization_test.  Exit code 0 on ALL PASS.
*/

#include "test_helpers.hh"

#include <cstdint>
#include <cstring>
#include <string>

using die_test::Checker;
using DieModel::OpKind;
using DieModel::Phase;

namespace {

static_assert(sizeof(double) == 8, "bitwise double compare assumes IEEE754");

bool bits_equal(double x, double y) {
  std::uint64_t a = 0;
  std::uint64_t b = 0;
  std::memcpy(&a, &x, sizeof(a));
  std::memcpy(&b, &y, sizeof(b));
  return a == b;
}

bool stage_equal(const DieModel::DieStage& a, const DieModel::DieStage& b) {
  if (a.label != b.label || a.resource != b.resource ||
      a.macro_group != b.macro_group || a.wave_repeat != b.wave_repeat) {
    return false;
  }
  if (!bits_equal(a.pe_core_flop, b.pe_core_flop) ||
      !bits_equal(a.vec_core_eqop, b.vec_core_eqop) ||
      !bits_equal(a.noc_edge_bytes, b.noc_edge_bytes) ||
      !bits_equal(a.timer_seconds, b.timer_seconds)) {
    return false;
  }
  if (!bits_equal(a.mem.actual_bytes, b.mem.actual_bytes) ||
      !bits_equal(a.mem.service_units, b.mem.service_units) ||
      !bits_equal(a.mem.eta, b.mem.eta) ||
      a.mem.matrix_class != b.mem.matrix_class) {
    return false;
  }
  if (a.deps.size() != b.deps.size()) {
    return false;
  }
  for (std::size_t i = 0; i < a.deps.size(); ++i) {
    if (a.deps[i].stage != b.deps[i].stage || a.deps[i].lag != b.deps[i].lag) {
      return false;
    }
  }
  return true;
}

bool summary_equal(const DieModel::DiePlanSummary& a,
                   const DieModel::DiePlanSummary& b) {
  return a.op_kind == b.op_kind && a.profile == b.profile &&
         a.loop_kind == b.loop_kind && a.m_tile == b.m_tile &&
         a.n_tile == b.n_tile && a.k_tile == b.k_tile &&
         a.group_rows == b.group_rows && a.group_cols == b.group_cols &&
         a.contexts == b.contexts && a.k_parallel == b.k_parallel &&
         a.weight_resident == b.weight_resident &&
         a.double_buffered == b.double_buffered && a.fused == b.fused &&
         a.fallback == b.fallback &&
         a.streaming_two_pass == b.streaming_two_pass &&
         bits_equal(a.sram_peak_per_core, b.sram_peak_per_core) &&
         a.layer_count == b.layer_count &&
         a.weight_passes == b.weight_passes &&
         a.launch_instances == b.launch_instances && a.waves == b.waves &&
         bits_equal(a.weight_hbm_bytes, b.weight_hbm_bytes);
}

void expect_identical_products(Checker& t, const DieModel::DiePlan& pa,
                               const DieModel::DiePlan& pb,
                               const std::string& what) {
  t.expect(summary_equal(pa.summary(), pb.summary()),
           what + ": summary fields identical");
  const std::vector<DieModel::DieStage>& sa = pa.stages();
  const std::vector<DieModel::DieStage>& sb = pb.stages();
  t.expect(sa.size() == sb.size(), what + ": stage count identical");
  const std::size_t n = sa.size() < sb.size() ? sa.size() : sb.size();
  for (std::size_t i = 0; i < n; ++i) {
    t.expect(stage_equal(sa[i], sb[i]),
             what + ": stage " + std::to_string(i) +
                 " identical (incl. deps)");
  }
  t.expect(bits_equal(pa.flops(), pb.flops()) &&
               bits_equal(pa.eqops(), pb.eqops()) &&
               bits_equal(pa.matrix_bytes(), pb.matrix_bytes()) &&
               bits_equal(pa.vector_bytes(), pb.vector_bytes()) &&
               bits_equal(pa.matrix_service_units(),
                          pb.matrix_service_units()) &&
               bits_equal(pa.vector_service_units(),
                          pb.vector_service_units()) &&
               bits_equal(pa.noc_total_link_bytes(),
                          pb.noc_total_link_bytes()) &&
               bits_equal(pa.noc_max_edge_bytes(), pb.noc_max_edge_bytes()),
           what + ": eight scalar totals bitwise identical");
}

}  // namespace

int main() {
  Checker t;
  const DieModel::Hardware big = die_test::hw_default();

  // --- group 1: non-attention op_kinds share one key and one plan across
  //     span variations that keep the q_tokens sum.  (kv content and both
  //     start positions also differ -- these ops never read them.)
  for (OpKind kind :
       {OpKind::Norm, OpKind::QkvRope, OpKind::OProj, OpKind::Residual,
        OpKind::MlpUp, OpKind::MlpActivation, OpKind::MlpDown,
        OpKind::FfnRegion, OpKind::Logits, OpKind::LocalRegion}) {
    const DieModel::DieOpSpec one =
        die_test::spec_of(kind, Phase::Prefill, 1004, 1024, 0, 0);
    DieModel::DieOpSpec two =
        die_test::spec_of(kind, Phase::Prefill, 600, 1024, 0, 0);
    two = die_test::add_span(two, "r1", 404, 2048, 9600, 512);
    const std::string name = DieModel::to_string(kind);
    t.expect(DieModel::plan_cache_key(one, big) ==
                 DieModel::plan_cache_key(two, big),
             name + ": key collapses span variants with equal q-token sum");
    const auto plan_one = DieModel::compile(one, big);
    const auto plan_two = DieModel::compile(two, big);
    expect_identical_products(t, *plan_one, *plan_two, name);
  }

  // --- group 2: attention-family op_kinds keep per-span keying.
  for (OpKind kind :
       {OpKind::Qk, OpKind::ScaleMask, OpKind::Softmax, OpKind::Av,
        OpKind::AttentionRegion}) {
    const DieModel::DieOpSpec base =
        die_test::spec_of(kind, Phase::Prefill, 1024, 1024, 0, 0);
    const std::string k0 = DieModel::plan_cache_key(base, big);
    const std::string name = DieModel::to_string(kind);

    DieModel::DieOpSpec moved = base;
    moved.spans[0].q_start_pos = 256;
    moved.spans[0].kv_start_pos = 128;
    t.expect(DieModel::plan_cache_key(moved, big) != k0,
             name + ": moved span positions keep a distinct key");

    DieModel::DieOpSpec split =
        die_test::spec_of(kind, Phase::Prefill, 512, 512, 0, 0);
    split = die_test::add_span(split, "r1", 512, 512, 512, 512);
    t.expect(DieModel::plan_cache_key(split, big) != k0,
             name + ": equal q-token sum with a different split keeps a "
                    "distinct key");
  }

  // --- group 3: tx8_reference_v1 keeps the v2 key and the shape gate.
  const DieModel::Hardware tx8 = DieModel::make_hardware(die_test::wrapped(
      die_test::die_node(0.75, "large", 800, 3, 2, 512), "tx8_reference_v1"));
  {
    const DieModel::DieOpSpec ref = die_test::spec_of(OpKind::OProj,
                                                      Phase::Prefill, 1024,
                                                      1024, 0, 0);
    const std::string kref = DieModel::plan_cache_key(ref, tx8);
    t.expect(kref.size() == 16, "tx8 key is 16 hex digits");
    DieModel::DieOpSpec kv2048 = ref;
    kv2048.spans[0].kv_tokens = 2048;
    t.expect(DieModel::plan_cache_key(kv2048, tx8) != kref,
             "tx8 key distinguishes kv lengths");
    t.expect_throw([&] { DieModel::compile(kv2048, tx8); },
                   "tx8 kv != 1024 still rejected by compile",
                   "tx8_reference_v1 profile only reproduces");
    t.expect(DieModel::plan_cache_key(ref, big) != kref,
             "mesh key differs from tx8 key");
  }

  if (t.ok) {
    std::printf("[plan_key_normalization_test] ALL PASS\n");
    return 0;
  }
  return 1;
}
