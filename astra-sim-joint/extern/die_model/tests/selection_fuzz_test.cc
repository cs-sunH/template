/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

/*
selection_fuzz_test.cc -- stage E3 resident regression: the compile-time
pruning of detail::pick_best_candidate must never change the candidate
selection (design doc §5.1: EVERY feasible candidate is compared by the
isolated stage model; the winner is the shortest total, ties broken by
smaller SRAM, less traffic and the fixed tile order).

For every generated case -- random legal DieOpSpec x random legal hardware
points, plus a fixed set of realistic shapes -- two selections are compared:

  1. pruned:    detail::pick_best_candidate(cands, hw), the production path
                with the sound branch-and-bound lower bound, and
  2. reference: a NO-PRUNING full enumeration that evaluates EVERY
                candidate with detail::evaluate_stages and scans them in
                index order with the same Ranking/better rule.

They must select candidates with IDENTICAL Rankings.  A different index
whose Ranking is completely equal is a benign §5.1 complete tie (the rule
does not order such candidates; counted, not failed).  Anything else fails.

The test also checks the invariant that makes the prune safe: for EVERY
candidate, candidate_lower_bound(c, hw) <= the candidate's exact makespan
(soundness), plus the same soundness check on random raw stage DAGs that
stress dep shapes real mappings never produce (forward deps, aliased lag
targets, kAllDep fans, zero repeats, empty stages).

Seeds are fixed; counts are tunable so CI can run a shorter sample than the
stage verification run:
  DIE_SELECTION_FUZZ_N      random spec count (default 500; the stage E3
                            verification used 2500 x several seeds)
  DIE_SELECTION_FUZZ_SEED   base seed (default 0xE3F5ED17)
  DIE_SELECTION_FUZZ_FULL   "1" adds the heavy prefill-M512/M4096 GEMM
                            shapes to the real-shape set (stage runs)
Random specs whose full-enumeration instance count would exceed
kInstanceBudget are counted as skipped (their enumeration alone dwarfs the
whole test; the pruning soundness invariant does not depend on size).

Build (standalone): cmake -S extern/die_model -B <build> && cmake --build
<build> --target selection_fuzz_test.  Exit code 0 on ALL PASS.
*/

#include "test_helpers.hh"

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

using die_test::Checker;
using DieModel::DieOpSpec;
using DieModel::DieSpan;
using DieModel::DieStage;
using DieModel::DieStageDep;
using DieModel::Hardware;
using DieModel::OpKind;
using DieModel::Phase;
using DieModel::StageResource;
using DieModel::detail::Candidate;

namespace {

// ---------------------------------------------------------------------------
// Deterministic RNG (same shape as the stage E2 differential harness).

struct Rng {
  std::mt19937_64 e;
  explicit Rng(std::uint64_t seed) : e(seed) {}
  std::uint64_t u64(std::uint64_t lo, std::uint64_t hi) {
    return lo + (e() % (hi - lo + 1));
  }
  bool chance(double p) {
    return static_cast<double>(e() % 1000000ull) / 1000000.0 < p;
  }
};

// ---------------------------------------------------------------------------
// §5.1 tie-break replica (verbatim from CostModel.cc's Ranking/better; the
// production definitions live in an anonymous namespace there).

struct Ranking {
  double total;
  double sram;
  double bytes;
  std::uint64_t m, n, k, rows, cols, ctx, kp;
  int weight_resident;
  int fused;
  int fallback;
  std::string loop;
};

Ranking rank_of(const Candidate& c, double total) {
  Ranking r;
  r.total = total;
  r.sram = c.summary.sram_peak_per_core;
  r.bytes = c.matrix_bytes + c.vector_bytes;
  r.m = c.summary.m_tile;
  r.n = c.summary.n_tile;
  r.k = c.summary.k_tile;
  r.rows = c.summary.group_rows;
  r.cols = c.summary.group_cols;
  r.ctx = c.summary.contexts;
  r.kp = c.summary.k_parallel;
  r.weight_resident = c.summary.weight_resident ? 1 : 0;
  r.fused = c.summary.fused ? 1 : 0;
  r.fallback = c.summary.fallback ? 1 : 0;
  r.loop = c.summary.loop_kind;
  return r;
}

bool better(const Ranking& a, const Ranking& b) {
  if (a.total != b.total) return a.total < b.total;
  if (a.sram != b.sram) return a.sram < b.sram;
  if (a.bytes != b.bytes) return a.bytes < b.bytes;
  auto fields = [](const Ranking& r) {
    return std::array<std::uint64_t, 8>{r.m, r.n, r.k, r.rows, r.cols,
                                        r.ctx, r.kp,
                                        static_cast<std::uint64_t>(r.weight_resident)};
  };
  const auto fa = fields(a);
  const auto fb = fields(b);
  if (fa != fb) return fa < fb;
  if (a.fused != b.fused) return a.fused < b.fused;
  if (a.fallback != b.fallback) return a.fallback < b.fallback;
  return a.loop < b.loop;
}

bool same_ranking(const Ranking& a, const Ranking& b) {
  return !better(a, b) && !better(b, a);
}

// Tolerance of the soundness check (matching the prune slack inside
// pick_best_candidate): the bound and the evaluator accumulate the same
// positive durations in different orders (per stage vs. per pop /
// fast-forward line reconstruction, which the stage E2 differential fuzz
// pinned to <1e-12 relative), so an exact <= comparison would be ulp
// noise.  One part in 1e9 clears that by three orders of magnitude.
constexpr double kRel = 1e-9;
constexpr double kAbs = 1e-15;

// Full-enumeration instance budget per random spec (see file comment).
constexpr std::uint64_t kInstanceBudget = 20'000'000;

// ---------------------------------------------------------------------------
// Reference selection: evaluate EVERY candidate, scan in index order.

std::size_t full_pick(const std::vector<Candidate>& cands, const Hardware& hw,
                      std::vector<double>* totals) {
  std::size_t best = 0;
  Ranking best_rank;
  bool have = false;
  for (std::size_t i = 0; i < cands.size(); ++i) {
    const double t =
        DieModel::detail::evaluate_stages(cands[i].stages, hw).total_seconds;
    if (totals != nullptr) {
      totals->push_back(t);
    }
    const Ranking r = rank_of(cands[i], t);
    if (!have || better(r, best_rank)) {
      best = i;
      best_rank = r;
      have = true;
    }
  }
  return best;
}

struct Stats {
  std::uint64_t cases = 0;
  std::uint64_t skipped_big = 0;
  std::uint64_t infeasible = 0;
  std::uint64_t complete_ties = 0;
  std::uint64_t sound_bad = 0;
  std::uint64_t sel_bad = 0;
};

// One selection-equivalence + soundness case over a real candidate set.
// Returns false on failure (details printed).
bool check_candidates(Checker& t, Stats& st, const std::vector<Candidate>& cands,
                      const Hardware& hw, const std::string& label) {
  if (cands.empty()) {
    ++st.infeasible;
    return true;
  }
  std::uint64_t instances = 0;
  for (const Candidate& c : cands) {
    for (const DieStage& s : c.stages) {
      instances += s.wave_repeat;
    }
  }
  if (instances > kInstanceBudget) {
    // Too expensive to enumerate exhaustively in the test budget; the
    // pruned pick itself is exercised by the rest of the suite.
    ++st.skipped_big;
    return true;
  }
  ++st.cases;

  std::vector<double> totals;
  const std::size_t ref = full_pick(cands, hw, &totals);
  const std::size_t got = DieModel::detail::pick_best_candidate(cands, hw);

  // Soundness: every candidate's bound must sit under its exact makespan.
  for (std::size_t i = 0; i < cands.size(); ++i) {
    const double lb = DieModel::detail::candidate_lower_bound(cands[i], hw);
    if (lb > totals[i] * (1.0 + kRel) + kAbs) {
      ++st.sound_bad;
      std::printf(
          "SOUNDNESS FAIL %s cand=%zu/%zu lb=%.17g total=%.17g\n",
          label.c_str(), i, cands.size(), lb, totals[i]);
      return false;
    }
  }

  if (got == ref) {
    return true;
  }
  const Ranking ra = rank_of(cands[got], totals[got]);
  const Ranking rb = rank_of(cands[ref], totals[ref]);
  if (same_ranking(ra, rb)) {
    ++st.complete_ties;  // §5.1 does not order completely tied candidates
    return true;
  }
  ++st.sel_bad;
  std::printf(
      "SELECTION MISMATCH %s: pruned=%zu (total=%.17g) reference=%zu "
      "(total=%.17g) cands=%zu\n",
      label.c_str(), got, totals[got], ref, totals[ref], cands.size());
  return false;
}

// ---------------------------------------------------------------------------
// Random legal hardware (design doc §3.1 space: 10 layouts x 5 widths x 14
// core rows) and random legal specs (same generator shape as the stage E2
// selection fuzz, size-capped for exhaustive enumeration).

DieModel::Hardware random_hw(Rng& r) {
  struct Lay {
    double area;
    std::uint64_t per_side, sides;
  };
  static const Lay lays[] = {
      {300, 2, 1}, {300, 2, 2}, {500, 2, 1}, {500, 3, 1}, {500, 2, 2},
      {500, 3, 2}, {800, 3, 1}, {800, 4, 1}, {800, 3, 2}, {800, 4, 2}};
  static const double sram[] = {0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0};
  static const std::uint64_t widths[] = {64, 128, 256, 512, 1024};
  const Lay& l = lays[r.u64(0, sizeof(lays) / sizeof(lays[0]) - 1)];
  nlohmann::json die;
  die["sram-mb-per-core"] = sram[r.u64(0, 6)];
  die["core-type"] = r.chance(0.5) ? "small" : "large";
  die["compute-area-mm2"] = l.area;
  die["hbm-per-side"] = l.per_side;
  die["hbm-sides"] = l.sides;
  die["noc-width-bits"] = widths[r.u64(0, 4)];
  return DieModel::make_hardware(die_test::wrapped(die, "mesh_die_v1"));
}

DieOpSpec random_spec(Rng& r) {
  static const OpKind kinds[] = {
      OpKind::Norm,           OpKind::QkvRope,    OpKind::Qk,
      OpKind::ScaleMask,      OpKind::Softmax,   OpKind::Av,
      OpKind::OProj,          OpKind::Residual,  OpKind::AttentionRegion,
      OpKind::MlpUp,          OpKind::MlpActivation, OpKind::MlpDown,
      OpKind::FfnRegion,      OpKind::Logits};
  DieOpSpec spec;
  spec.op_kind = kinds[r.u64(0, sizeof(kinds) / sizeof(kinds[0]) - 1)];
  spec.phase = r.chance(0.5) ? Phase::Prefill : Phase::Decode;
  spec.dtype = DieModel::Dtype::Bf16;
  spec.model_dims = {4096, 11008, 32000,
                     r.chance(0.5) ? DieModel::MlpVariant::Swiglu
                                   : DieModel::MlpVariant::Gelu,
                     r.chance(0.5) ? DieModel::NormType::Rmsnorm
                                   : DieModel::NormType::Layernorm,
                     DieModel::PositionEncoding::Rope, 32, 8, 128};
  const std::uint64_t rank = r.u64(0, 5);
  spec.tp_slice = {6, rank, r.u64(2, 6), r.u64(1, 2), r.u64(1000, 1900),
                   r.u64(4000, 5400)};
  spec.layer_count = r.u64(1, 8);      // size-capped for full enumeration
  spec.weight_passes = r.u64(1, 4);
  spec.kv_access.identity = DieModel::KvIdentity::Local;
  spec.fusion_policy =
      r.chance(0.5) ? DieModel::FusionPolicy::Auto : DieModel::FusionPolicy::None;
  spec.config_hash = "fuzz";
  spec.provenance = "fuzz";
  const std::uint64_t nspans = r.u64(1, 3);
  for (std::uint64_t i = 0; i < nspans; ++i) {
    DieSpan sp;
    sp.request_id = "r" + std::to_string(i);
    sp.q_tokens = r.u64(1, 256);   // capped: keeps exhaustive enumeration
    sp.kv_tokens = r.u64(1, 512);  // of the candidate set affordable
    sp.q_start_pos = r.u64(0, 1024);
    sp.kv_start_pos = 0;
    spec.spans.push_back(sp);
  }
  return spec;
}

// ---------------------------------------------------------------------------
// Raw-DAG soundness fuzz: random stage lists (arbitrary backward deps,
// lags, kAllDep fans, zero repeats, every resource class).  The bound must
// stay under the exact schedule on all of them.

std::vector<DieStage> random_stages(Rng& r, std::uint64_t ns) {
  std::vector<DieStage> st(ns);
  for (std::uint64_t i = 0; i < ns; ++i) {
    DieStage& s = st[i];
    s.label = "s" + std::to_string(i);
    switch (r.u64(0, 9)) {
      case 0:
      case 1:
      case 2:
        s.resource = StageResource::PE;
        s.pe_core_flop = r.u64(0, 200000) * 1000.0;
        break;
      case 3:
      case 4:
        s.resource = StageResource::VEC;
        s.vec_core_eqop = r.u64(0, 100000) * 1000.0;
        break;
      case 5:
        s.resource = StageResource::NOC;
        s.noc_edge_bytes = r.u64(0, 400000) * 10.0;
        break;
      case 6:
      case 7:
        s.resource =
            r.chance(0.5) ? StageResource::MEM : StageResource::MEM_NOC;
        s.mem.actual_bytes = r.u64(0, 500000) * 10.0;
        s.mem.service_units = s.mem.actual_bytes / 0.7;
        s.mem.eta = 0.7;
        s.mem.matrix_class = r.chance(0.5);
        if (s.resource == StageResource::MEM_NOC) {
          s.noc_edge_bytes = r.u64(0, 300000) * 10.0;
        }
        break;
      default:
        s.resource = StageResource::TIMER;
        s.timer_seconds = r.u64(0, 3000) * 1e-9;
        break;
    }
    s.wave_repeat = r.u64(0, 40);
    if (i > 0 && r.chance(0.85)) {
      const std::uint64_t nd = r.u64(1, 2);
      for (std::uint64_t d = 0; d < nd; ++d) {
        DieStageDep dep;
        dep.stage = static_cast<std::uint32_t>(r.u64(0, i - 1));
        dep.lag = r.chance(0.25) ? DieStageDep::kAllDep : r.u64(0, 3);
        s.deps.push_back(dep);
      }
    }
  }
  return st;
}

}  // namespace

int main() {
  Checker t;
  Stats st;
  const char* env_n = std::getenv("DIE_SELECTION_FUZZ_N");
  const char* env_seed = std::getenv("DIE_SELECTION_FUZZ_SEED");
  const char* env_full = std::getenv("DIE_SELECTION_FUZZ_FULL");
  const std::uint64_t fuzz_n =
      env_n != nullptr ? std::strtoull(env_n, nullptr, 10) : 500;
  const std::uint64_t seed0 =
      env_seed != nullptr ? std::strtoull(env_seed, nullptr, 10) : 0xE3F5ED17ull;
  const bool full_shapes = env_full != nullptr && std::strcmp(env_full, "1") == 0;
  const Hardware hw_default = die_test::hw_default();

  // --- 1. Real-shape set (the smoke's shape families at enumerable sizes;
  //     the heavy prefill-M512/M4096 GEMM shapes are behind FULL).
  {
    struct Shape {
      const char* name;
      OpKind kind;
      std::uint64_t M, L, P, qh, kvh, kv, spans;
      bool heavy;
    };
    const Shape shapes[] = {
        {"decode norm M1 L32", OpKind::Norm, 1, 32, 1, 32, 8, 0, 1, false},
        {"decode residual M8", OpKind::Residual, 8, 32, 1, 32, 8, 0, 1, false},
        {"decode o_proj M8 L32", OpKind::OProj, 8, 32, 1, 32, 8, 0, 1, false},
        {"decode qkv_rope M8 kv400", OpKind::QkvRope, 8, 32, 1, 32, 8, 400, 1, false},
        {"decode attn M8 kv400", OpKind::AttentionRegion, 8, 32, 1, 32, 8, 400, 8, false},
        {"decode ffn M8 P24", OpKind::FfnRegion, 8, 32, 24, 32, 8, 0, 1, false},
        {"decode mlp_up M8", OpKind::MlpUp, 8, 32, 1, 32, 8, 0, 1, false},
        {"decode mlp_act M8", OpKind::MlpActivation, 8, 32, 1, 32, 8, 0, 1, false},
        {"decode mlp_down M8", OpKind::MlpDown, 8, 32, 1, 32, 8, 0, 1, false},
        {"decode qk M8 kv400", OpKind::Qk, 8, 2, 1, 32, 8, 400, 1, false},
        {"decode softmax M8 kv400", OpKind::Softmax, 8, 2, 1, 32, 8, 400, 1, false},
        {"decode logits M8", OpKind::Logits, 8, 2, 1, 32, 8, 0, 1, false},
        {"prefill norm M1024", OpKind::Norm, 1024, 2, 1, 32, 8, 0, 1, false},
        {"prefill residual M1024", OpKind::Residual, 1024, 2, 1, 32, 8, 0, 1, false},
        {"prefill o_proj M128", OpKind::OProj, 128, 4, 1, 32, 8, 0, 1, false},
        {"prefill qkv_rope M128 L4", OpKind::QkvRope, 128, 4, 1, 32, 8, 0, 1, false},
        {"prefill ffn M256", OpKind::FfnRegion, 256, 4, 1, 32, 8, 0, 1, false},
        {"prefill attn M256 kv256", OpKind::AttentionRegion, 256, 4, 1, 32, 8, 256, 2, false},
        {"prefill logits M256", OpKind::Logits, 256, 2, 1, 32, 8, 0, 1, false},
        {"prefill ffn M512 L32", OpKind::FfnRegion, 512, 32, 1, 32, 8, 0, 1, false},
        {"prefill attn M512 kv512", OpKind::AttentionRegion, 512, 32, 1, 32, 8, 512, 1, false},
        {"prefill qkv_rope M512 L32", OpKind::QkvRope, 512, 32, 1, 32, 8, 0, 1, true},
        {"prefill qkv_rope M512 P2", OpKind::QkvRope, 512, 32, 2, 32, 8, 0, 1, true},
        {"prefill o_proj M512 L32", OpKind::OProj, 512, 32, 1, 32, 8, 0, 1, true},
        {"prefill qkv_rope M4096", OpKind::QkvRope, 4096, 32, 1, 32, 8, 0, 1, true},
    };
    for (const Shape& sh : shapes) {
      if (sh.heavy && !full_shapes) {
        continue;
      }
      DieOpSpec spec;
      spec.op_kind = sh.kind;
      spec.phase = sh.M <= 8 ? Phase::Decode : Phase::Prefill;
      spec.dtype = DieModel::Dtype::Bf16;
      spec.model_dims = {4096, 11008, 32000, DieModel::MlpVariant::Swiglu,
                         DieModel::NormType::Rmsnorm,
                         DieModel::PositionEncoding::Rope, 32, 8, 128};
      spec.tp_slice = {1, 0, sh.qh, sh.kvh, 11008, 32000};
      spec.layer_count = sh.L;
      spec.weight_passes = sh.P;
      spec.kv_access.identity = DieModel::KvIdentity::Local;
      spec.fusion_policy = DieModel::FusionPolicy::Auto;
      spec.config_hash = "fuzz-real";
      spec.provenance = "fuzz-real";
      const std::uint64_t per = sh.M / sh.spans;
      for (std::uint64_t i = 0; i < sh.spans; ++i) {
        DieSpan sp;
        sp.request_id = "r" + std::to_string(i);
        sp.q_tokens = per ? per : 1;
        sp.kv_tokens = sh.kv;
        sp.q_start_pos = i * sp.q_tokens;
        sp.kv_start_pos = 0;
        spec.spans.push_back(sp);
      }
      try {
        const auto wl = DieModel::derive_workload(spec);
        const auto cands = DieModel::detail::enumerate_candidates(wl, hw_default);
        t.expect(check_candidates(t, st, cands, hw_default, sh.name),
                 std::string("real shape ") + sh.name);
      } catch (const std::exception& e) {
        t.expect(false, std::string("real shape ") + sh.name +
                            " threw: " + e.what());
      }
    }
  }

  // --- 2. Random spec x hardware fuzz (selection equivalence + soundness).
  for (std::uint64_t i = 0; i < fuzz_n && st.sel_bad < 5 && st.sound_bad < 5;
       ++i) {
    Rng r(seed0 + static_cast<std::uint64_t>(i) * 65537ull);
    const DieOpSpec spec = random_spec(r);
    const Hardware hw = random_hw(r);
    try {
      const auto wl = DieModel::derive_workload(spec);
      const auto cands = DieModel::detail::enumerate_candidates(wl, hw);
      if (!check_candidates(t, st, cands, hw,
                            std::string("random#") + std::to_string(i) +
                                " op=" + DieModel::to_string(spec.op_kind))) {
        t.expect(false, "random fuzz case failed");
        break;
      }
    } catch (const DieModel::DieModelError&) {
      ++st.infeasible;  // infeasible shape; not a selection case
    }
  }

  // --- 3. Raw-DAG soundness fuzz (bound <= exact schedule on structures
  //     mappings never produce; exercises the exact/backward CP switch).
  {
    std::uint64_t dag_tested = 0, dag_bad = 0, dag_thrown = 0;
    Rng r(seed0 ^ 0xD1E5EEDull);
    while (dag_tested < fuzz_n * 2 && dag_bad < 5) {
      const std::uint64_t ns = r.u64(1, 12);
      const std::vector<DieStage> stages = random_stages(r, ns);
      double total;
      try {
        total = DieModel::detail::evaluate_stages(stages, hw_default)
                    .total_seconds;
      } catch (const DieModel::DieModelError&) {
        ++dag_thrown;  // cyclic random DAG: not a schedule
        continue;
      }
      Candidate c;
      c.stages = stages;
      c.flops = 0.0;
      c.matrix_bytes = 0.0;
      c.vector_bytes = 0.0;
      for (const DieStage& s : stages) {
        c.flops += s.pe_core_flop * static_cast<double>(s.wave_repeat);
        c.matrix_bytes +=
            (s.mem.matrix_class ? s.mem.actual_bytes : 0.0) *
            static_cast<double>(s.wave_repeat);
        c.vector_bytes +=
            (s.mem.matrix_class ? 0.0 : s.mem.actual_bytes) *
            static_cast<double>(s.wave_repeat);
      }
      const double lb = DieModel::detail::candidate_lower_bound(c, hw_default);
      ++dag_tested;
      if (lb > total * (1.0 + kRel) + kAbs) {
        ++dag_bad;
        std::printf("DAG SOUNDNESS FAIL #%llu lb=%.17g total=%.17g\n",
                    (unsigned long long)dag_tested, lb, total);
      }
    }
    t.expect(dag_bad == 0, "raw-DAG soundness");
    std::printf("[dag-fuzz] tested=%llu cyclic_thrown=%llu bad=%llu\n",
                (unsigned long long)dag_tested,
                (unsigned long long)dag_thrown,
                (unsigned long long)dag_bad);
  }

  std::printf(
      "[selection-fuzz] seed=%llu n=%llu cases=%llu skipped_big=%llu "
      "infeasible=%llu complete_ties=%llu sel_bad=%llu sound_bad=%llu\n",
      (unsigned long long)seed0, (unsigned long long)fuzz_n,
      (unsigned long long)st.cases, (unsigned long long)st.skipped_big,
      (unsigned long long)st.infeasible,
      (unsigned long long)st.complete_ties,
      (unsigned long long)st.sel_bad, (unsigned long long)st.sound_bad);
  if (st.sel_bad == 0 && st.sound_bad == 0 && t.ok) {
    std::printf("[selection_fuzz_test] ALL PASS\n");
    return 0;
  }
  return 1;
}
