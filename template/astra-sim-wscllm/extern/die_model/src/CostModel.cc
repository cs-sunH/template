/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

/*
CostModel.cc -- the §6.1 resource-stage model, compile() and estimate_alone()
(contract §4), and the plan cache key (contract §5).

The isolated-run evaluator is a die-level conservative list schedule of the
repeat-compressed stage DAG on four resource slots (PE, VEC, NOC, MEM):

  - t_PE  (per wave) = max_core(F_core / (eta_pe  * P_core));   §6.1
  - t_vec (per wave) = max_core(E_core / (eta_vec * V_core));
  - t_NoC (per wave) = max_edge(D_edge / (eta_noc * link_peak));
  - t_mem (per submit) = service_units / B_memory (§6.2: service units
    already carry D/eta and the HBM granularity rule).
  - a MEM_NOC stage (HBM<->core) starts its memory and NoC legs together and
    completes at the later one, holding both slots for the whole duration
    (§6.1 "完成取两者较晚；保守占有片内 NoC 槽").
  - a TIMER stage carries the §6.1 launch overhead and occupies no slot.

Both compile()'s candidate selection and estimate_alone() call the SAME
evaluate_stages(); the same plan therefore always yields the same numbers.
*/

#include "DieModel.hh"

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <map>
#include <queue>
#include <string>
#include <utility>
#include <vector>

namespace DieModel {

namespace detail {

// Declared in DieTypes.hh; used by Mapping.cc for the FFN fallback.
//
// Stage E2 rewrite: the SAME Kahn list schedule with (stage<<32)|instance
// pop order and the SAME per-instance arithmetic (identical max/+
// operation order -- every schedule value of the exact path is bitwise
// identical to the previous CSR + binary-heap formulation), restructured
// into a per-stage head scan:
//
//   - instances of one stage pop in index order (their readiness is
//     monotone in j), so at any moment the only poppable instance of
//     stage s is its head cnt[s];
//   - the min key among ready instances therefore belongs to the LOWEST
//     stage index whose head is ready (stage is the major key), which a
//     small in-order scan finds without a heap or a CSR pass;
//   - dependency targets are recomputed on the fly from
//     DieStageDep{stage, lag} with the frozen target arithmetic.  Real
//     plans only produce lag targets inside the dep stage (equal repeats
//     along lag chains), which the in-stage fast index covers; the
//     historical cross-stage aliasing of a lag target beyond the dep
//     stage's repeat is nevertheless reproduced faithfully through the
//     flat-id round trip.
//
// On top of the exact scan sits a CERTIFIED steady-state fast-forward
// (design doc §8.5's repeat compression applied to the isolated schedule,
// which has no external events by construction): once EVERY still-active
// stage's finish sequence advances by a constant per instance (three
// equal consecutive deltas) and every never-popped stage is
// dependency-blocked, whole instance ranges are skipped in growing chunks
// along the per-stage lines; every chunk is re-verified by replaying a
// few instances exactly against the lines, and ANY mismatch restarts the
// whole evaluation in pure-exact mode (correctness first).  Skipped
// finishes stay reconstructible from the anchor + per-instance delta, so
// later dependency reads remain exact.  Equivalence was pinned by the
// stage E2 differential fuzz (>=1000 random plans per family, all five
// schedule outputs equal to the frozen implementation; see the E2
// report).

constexpr std::uint64_t kCertCheckEvery = 64;  // exact pops between attempts

ScheduleResult evaluate_stages_impl(const std::vector<DieStage>& stages,
                                    const Hardware& hw, bool allow_fast) {
  ScheduleResult out;
  const std::size_t ns = stages.size();
  if (ns == 0) {
    return out;
  }
  std::vector<std::size_t> offset(ns + 1, 0);
  std::size_t total = 0;
  for (std::size_t s = 0; s < ns; ++s) {
    total += static_cast<std::size_t>(stages[s].wave_repeat);
    offset[s + 1] = total;
  }
  // Arena reuse (stage E2): the finish array is fully written before any
  // read (dependency targets are always popped instances), so a
  // call-reused buffer avoids the multi-megabyte allocate+zero per
  // evaluation -- compile() evaluates hundreds of candidates with the
  // same instance counts back to back.  Single-threaded simulation: one
  // thread-local arena, no lock (guard with a mutex if the runtime ever
  // becomes multi-threaded).
  static thread_local std::vector<double> finish_arena;
  if (finish_arena.size() < total) {
    finish_arena.resize(total);  // grows once; then only indexed < total
  }
  std::vector<double>& finish = finish_arena;
  std::vector<double> dur(ns, 0.0);
  for (std::size_t s = 0; s < ns; ++s) {
    dur[s] = detail::stage_instance_seconds(stages[s], hw);
  }
  std::vector<std::uint64_t> cnt(ns, 0);  // popped instances per stage

  // Flat id -> (stage, instance) without a stage_of array.
  const auto flat_stage = [&](std::size_t flat, std::size_t& s,
                              std::uint64_t& j) {
    std::size_t lo = 0, hi = ns;  // offset[lo] <= flat < offset[hi]
    while (lo + 1 < hi) {
      const std::size_t mid = lo + (hi - lo) / 2;
      if (flat < offset[mid]) {
        hi = mid;
      } else {
        lo = mid;
      }
    }
    s = lo;
    j = static_cast<std::uint64_t>(flat - offset[lo]);
  };

  // Target flat id of one dep of instance (s, j); -1 when the dep imposes
  // no edge (frozen CSR construction arithmetic).
  const auto dep_target = [&](std::size_t s, std::uint64_t j,
                              const DieStageDep& dep) -> std::int64_t {
    if (dep.stage >= ns || stages[dep.stage].wave_repeat == 0) {
      return -1;
    }
    if (dep.lag == DieStageDep::kAllDep) {
      return static_cast<std::int64_t>(stages[dep.stage].wave_repeat) - 1;
    }
    if (static_cast<std::int64_t>(j) >=
        static_cast<std::int64_t>(dep.lag)) {
      return static_cast<std::int64_t>(j) -
             static_cast<std::int64_t>(dep.lag);
    }
    return -1;
  };

  // ---- steady-state fast-forward state (inert until certified) ----
  // Per-stage per-INSTANCE linearity: once every still-active stage's
  // finish sequence advances by a constant per instance (three equal
  // consecutive deltas) and every never-popped stage is dependency-blocked,
  // the isolated schedule is in a periodic max-plus regime whose pop
  // ORDER is irrelevant to the finish values (they satisfy the same
  // recurrence along the lines).  Whole instance ranges are then skipped
  // geometrically, each chunk re-verified by replaying a few instances
  // exactly; any mismatch restarts the evaluation in pure-exact mode.
  bool fast_ok = allow_fast && total >= 4096;
  bool fast_used = false;
  std::vector<std::uint64_t> anchor_cnt(ns, 0);
  std::vector<double> per_inst_delta(ns, 0.0);
  std::uint64_t pops_since_check = 0;

  const auto close = [](double a, double b) {
    const double scale = std::max(1.0, std::max(std::fabs(a), std::fabs(b)));
    return std::fabs(a - b) <= 1e-12 * scale;
  };

  // Finish of instance j of stage s (skipped instances come from the
  // anchor line).
  const auto finish_of = [&](std::size_t s, std::uint64_t j) -> double {
    if (j < cnt[s] || !fast_used || j < anchor_cnt[s]) {
      return finish[offset[s] + static_cast<std::size_t>(j)];
    }
    return finish[offset[s] + static_cast<std::size_t>(anchor_cnt[s])] +
           static_cast<double>(j - anchor_cnt[s]) * per_inst_delta[s];
  };

  // Whether every dependee instance of (s, j) has already popped (chunked
  // instances count as popped: their cnt advanced).
  const auto head_ready = [&](std::size_t s, std::uint64_t j) {
    for (const DieStageDep& dep : stages[s].deps) {
      const std::int64_t target = dep_target(s, j, dep);
      if (target < 0) {
        continue;
      }
      const std::uint64_t rep_d = stages[dep.stage].wave_repeat;
      if (static_cast<std::uint64_t>(target) < rep_d) {
        // In-stage target (every real plan; see dep_target): the instance
        // popped once the dep stage's count passed it.
        if (cnt[dep.stage] <= static_cast<std::uint64_t>(target)) {
          return false;
        }
        continue;
      }
      const std::size_t flat =
          offset[dep.stage] + static_cast<std::size_t>(target);
      std::size_t ds;
      std::uint64_t dj;
      flat_stage(flat, ds, dj);
      if (cnt[ds] <= dj) {
        return false;
      }
    }
    return true;
  };

  double pe_free = 0.0, vec_free = 0.0, noc_free = 0.0, mem_free = 0.0;
  double makespan = 0.0;
  std::size_t done = 0;

  // Pop one instance (exact arithmetic; identical operation order to the
  // frozen loop, with dependency reads routed through finish_of).
  const auto pop_one = [&](std::size_t s) {
    const std::uint64_t j = cnt[s];
    const std::size_t id = offset[s] + static_cast<std::size_t>(j);
    const DieStage& st = stages[s];
    double start = 0.0;
    for (const DieStageDep& dep : st.deps) {
      const std::int64_t target = dep_target(s, j, dep);
      if (target < 0) {
        continue;
      }
      const std::uint64_t rep_d = stages[dep.stage].wave_repeat;
      if (static_cast<std::uint64_t>(target) < rep_d) {
        // In-stage target: direct index (no flat-id round trip).
        start = std::max(
            start,
            finish_of(dep.stage, static_cast<std::uint64_t>(target)));
        continue;
      }
      const std::size_t flat =
          offset[dep.stage] + static_cast<std::size_t>(target);
      std::size_t ds;
      std::uint64_t dj;
      flat_stage(flat, ds, dj);
      start = std::max(start, finish_of(ds, dj));
    }
    switch (st.resource) {
      case StageResource::PE:
        start = std::max(start, pe_free);
        break;
      case StageResource::VEC:
        start = std::max(start, vec_free);
        break;
      case StageResource::NOC:
        start = std::max(start, noc_free);
        break;
      case StageResource::MEM:
        start = std::max(start, mem_free);
        break;
      case StageResource::MEM_NOC:
        start = std::max(start, std::max(noc_free, mem_free));
        break;
      case StageResource::TIMER:
        break;
    }
    const double d = dur[s];
    const double fin = start + d;
    finish[id] = fin;
    switch (st.resource) {
      case StageResource::PE:
        pe_free = fin;
        out.pe_seconds += d;
        break;
      case StageResource::VEC:
        vec_free = fin;
        out.vec_seconds += d;
        break;
      case StageResource::NOC:
        noc_free = fin;
        out.noc_seconds += d;
        break;
      case StageResource::MEM:
        mem_free = fin;
        out.mem_seconds += d;
        break;
      case StageResource::MEM_NOC:
        // Holds BOTH slots for max(t_mem, t_NoC) (§6.1).
        noc_free = fin;
        mem_free = fin;
        out.noc_seconds += d;
        out.mem_seconds += d;
        break;
      case StageResource::TIMER:
        break;
    }
    makespan = std::max(makespan, fin);
    ++done;
    ++cnt[s];
  };

  // The min-key ready instance: lowest stage index with a ready head.
  const auto next_stage = [&]() -> std::size_t {
    for (std::size_t s = 0; s < ns; ++s) {
      if (cnt[s] >= stages[s].wave_repeat) {
        continue;
      }
      if (head_ready(s, cnt[s])) {
        return s;
      }
    }
    return ns;
  };

  // Certification: every stage that has popped (cnt >= 4) must show three
  // equal consecutive per-instance deltas; stages that never popped must be
  // dependency-blocked (their first pop cannot fall inside a skipped
  // range); stages with 1 <= cnt < 4 or ready-but-starved heads abort the
  // certification (exact pops continue).  No stage may come within
  // kTailMargin instances of exhaustion: chunks never exhaust a stage, so
  // blocked (kAllDep) dependents only ever unblock on the exact path.
  constexpr std::uint64_t kTailMargin = 64;
  const auto try_certify = [&]() -> bool {
    for (std::size_t s = 0; s < ns; ++s) {
      const std::uint64_t rep = stages[s].wave_repeat;
      if (cnt[s] >= rep) {
        continue;  // exhausted: nothing to extrapolate
      }
      if (cnt[s] == 0) {
        if (head_ready(s, 0)) {
          return false;  // due to pop now; not absent from the near future
        }
        continue;  // blocked: stays blocked through any chunk
      }
      if (cnt[s] < 4) {
        return false;
      }
      if (cnt[s] + kTailMargin >= rep) {
        return false;
      }
      const double f1 = finish[offset[s] + static_cast<std::size_t>(cnt[s] - 1)];
      const double f0 = finish[offset[s] + static_cast<std::size_t>(cnt[s] - 2)];
      const double g0 = finish[offset[s] + static_cast<std::size_t>(cnt[s] - 3)];
      const double h0 = finish[offset[s] + static_cast<std::size_t>(cnt[s] - 4)];
      if (!close(f1 - f0, f0 - g0) || !close(f1 - f0, g0 - h0)) {
        return false;
      }
      per_inst_delta[s] = f1 - f0;
    }
    for (std::size_t s = 0; s < ns; ++s) {
      anchor_cnt[s] = cnt[s] > 0 ? cnt[s] - 1 : 0;
    }
    fast_used = true;
    return true;
  };

  // Skip n_s instances of every active stage along its line; accumulate
  // the resource sums and the makespan contribution; re-anchor each slot's
  // free time to the latest line value among its still-active stages.
  const auto advance_chunk = [&](const std::vector<std::uint64_t>& n) {
    for (std::size_t s = 0; s < ns; ++s) {
      const std::uint64_t add = n[s];
      if (add == 0) {
        continue;
      }
      const double d = dur[s] * static_cast<double>(add);
      switch (stages[s].resource) {
        case StageResource::PE:
          out.pe_seconds += d;
          break;
        case StageResource::VEC:
          out.vec_seconds += d;
          break;
        case StageResource::NOC:
          out.noc_seconds += d;
          break;
        case StageResource::MEM:
          out.mem_seconds += d;
          break;
        case StageResource::MEM_NOC:
          out.noc_seconds += d;
          out.mem_seconds += d;
          break;
        case StageResource::TIMER:
          break;
      }
      makespan = std::max(makespan, finish_of(s, cnt[s] + add - 1));
      cnt[s] += add;
      done += static_cast<std::size_t>(add);
    }
    const auto slot_follow = [&](double& free_time, int slot_res) {
      double v = free_time;
      for (std::size_t s = 0; s < ns; ++s) {
        const int r = static_cast<int>(stages[s].resource);
        const bool on_slot =
            (slot_res == 0 && r == 0) || (slot_res == 1 && r == 1) ||
            (slot_res == 2 && (r == 2 || r == 4)) ||
            (slot_res == 3 && (r == 3 || r == 4));
        if (!on_slot || cnt[s] == 0) {
          continue;
        }
        v = std::max(v, finish_of(s, cnt[s] - 1));
      }
      free_time = v;
    };
    slot_follow(pe_free, 0);
    slot_follow(vec_free, 1);
    slot_follow(noc_free, 2);
    slot_follow(mem_free, 3);
  };

  // The chunk stride: a fixed small first step that doubles on every
  // verified chunk (capped), never exceeding half of the tightest stage's
  // remaining room, so an early regime change costs at most one small
  // chunk while a long steady regime still reaches the cap quickly.
  constexpr std::uint64_t kStrideStart = 64;
  constexpr std::uint64_t kStrideCap = 4096;
  std::uint64_t stride = kStrideStart;
  const auto chunk_sizes = [&](std::vector<std::uint64_t>& n) {
    std::uint64_t min_room = UINT64_MAX;
    for (std::size_t s = 0; s < ns; ++s) {
      const std::uint64_t rep = stages[s].wave_repeat;
      if (cnt[s] >= rep || cnt[s] == 0) {
        n[s] = 0;
        continue;
      }
      const std::uint64_t room = rep - cnt[s] > kTailMargin
                                     ? rep - cnt[s] - kTailMargin
                                     : 0;
      min_room = std::min(min_room, room);
      n[s] = room;
    }
    if (min_room == UINT64_MAX) {
      return false;
    }
    std::uint64_t step = std::min(stride, min_room / 2);
    step = std::min(step, kStrideCap);
    if (step == 0) {
      return false;
    }
    for (std::size_t s = 0; s < ns; ++s) {
      n[s] = std::min(n[s], step);
    }
    return true;
  };

  bool certified = false;
  std::vector<std::uint64_t> n(ns, 0);
  while (done < total) {
    if (!certified) {
      if (fast_ok) {
        ++pops_since_check;
        if (pops_since_check >= kCertCheckEvery) {
          pops_since_check = 0;
          if (try_certify()) {
            certified = true;
          }
        }
      }
      if (!certified) {
        const std::size_t s = next_stage();
        if (s == ns) {
          throw DieModelError(
              "evaluate_stages: stage DAG has a dependency cycle");
        }
        pop_one(s);
        continue;
      }
    }
    // Certified: skip geometric chunks, verifying each chunk by replaying
    // a few instances exactly against the lines.  Any mismatch restarts
    // the whole evaluation in pure-exact mode.
    bool verified = true;
    while (verified && chunk_sizes(n)) {
      advance_chunk(n);
      stride *= 2;
      if (done >= total) {
        break;
      }
      const std::size_t verify_pops = ns * 2;
      for (std::size_t i = 0; i < verify_pops && verified; ++i) {
        const std::size_t s = next_stage();
        if (s == ns) {
          break;  // everything left is blocked: the exact tail handles it
        }
        const double predicted = finish_of(s, cnt[s]);
        pop_one(s);
        if (!close(finish_of(s, cnt[s] - 1), predicted)) {
          verified = false;
        }
      }
    }
    if (!verified) {
      return evaluate_stages_impl(stages, hw, false);
    }
    certified = false;
    // fast_used STAYS true: skipped instances remain reconstructible from
    // the anchors for every later dependency read (the exact endgame may
    // still reference them).
    pops_since_check = 0;
  }
  out.total_seconds = makespan;
  return out;
}

// Declared in DieTypes.hh; the entry point of the stage A2 report's frozen
// isolated scheduler (same Kahn pop order, same arithmetic).
ScheduleResult evaluate_stages(const std::vector<DieStage>& stages,
                               const Hardware& hw) {
  return evaluate_stages_impl(stages, hw, true);
}

}  // namespace detail

// ---------------------------------------------------------------------------
// DiePlan implementation

DiePlan::DiePlan(std::shared_ptr<const DiePlanData> data) : data_(std::move(data)) {}

const DieOpSpec& DiePlan::spec() const {
  if (!data_) {
    throw DieModelError("DiePlan::spec: empty plan");
  }
  return *data_->spec;
}

const Hardware& DiePlan::hardware() const {
  if (!data_) {
    throw DieModelError("DiePlan::hardware: empty plan");
  }
  return data_->hardware;
}

const DiePlanSummary& DiePlan::summary() const {
  if (!data_) {
    throw DieModelError("DiePlan::summary: empty plan");
  }
  return data_->summary;
}

const std::vector<DieStage>& DiePlan::stages() const {
  if (!data_) {
    throw DieModelError("DiePlan::stages: empty plan");
  }
  return data_->stages;
}

double DiePlan::flops() const { return data_ ? data_->flops : 0.0; }
double DiePlan::eqops() const { return data_ ? data_->eqops : 0.0; }
double DiePlan::matrix_bytes() const {
  return data_ ? data_->matrix_bytes : 0.0;
}
double DiePlan::vector_bytes() const {
  return data_ ? data_->vector_bytes : 0.0;
}
double DiePlan::matrix_service_units() const {
  return data_ ? data_->matrix_service_units : 0.0;
}
double DiePlan::vector_service_units() const {
  return data_ ? data_->vector_service_units : 0.0;
}
double DiePlan::noc_total_link_bytes() const {
  return data_ ? data_->noc_total_link_bytes : 0.0;
}
double DiePlan::noc_max_edge_bytes() const {
  return data_ ? data_->noc_max_edge_bytes : 0.0;
}

// ---------------------------------------------------------------------------
// Per-instance stage durations (§6.1).  In namespace detail since stage A3:
// evaluate_stages, the compile-time pruning bound and the Execution machine
// (src/Execution.cc) all consume the same duration truth.

namespace detail {

double stage_instance_seconds(const DieStage& s, const Hardware& hw) {
  if (hw.profile == Profile::Tx8ReferenceV1) {
    // Reference profile (§1.1/§2.2): the aggregated-streaming envelope has
    // no core mesh; the whole op is one budget against the §2.1 reference
    // peaks and class bandwidths.
    switch (s.resource) {
      case StageResource::PE:
        return s.pe_core_flop / hw.tx8_matrix_peak_flop_s;
      case StageResource::VEC:
        return s.vec_core_eqop / hw.tx8_vector_peak_eqop_s;
      case StageResource::MEM:
        return s.mem.service_units /
               (s.mem.matrix_class ? hw.tx8_bw_matrix_bytes_s
                                   : hw.tx8_bw_vector_bytes_s);
      case StageResource::TIMER:
        return s.timer_seconds;
      case StageResource::NOC:
      case StageResource::MEM_NOC:
        return 0.0;  // inferred_NoC_cost = 0 (§2.1)
    }
    return 0.0;
  }
  const double p_core = hw.core_row.pe_int4_gops_per_core * 1e9;
  const double v_core = hw.core_row.vec_g_eqop_per_core * 1e9;
  switch (s.resource) {
    case StageResource::PE:
      return s.pe_core_flop / (hw.eta_pe * p_core);
    case StageResource::VEC:
      return s.vec_core_eqop / (hw.eta_vec * v_core);
    case StageResource::NOC:
      return s.noc_edge_bytes / (hw.eta_noc * hw.link_peak_bytes);
    case StageResource::MEM:
      return s.mem.service_units / hw.hbm_bandwidth_bytes;
    case StageResource::MEM_NOC: {
      const double t_mem = s.mem.service_units / hw.hbm_bandwidth_bytes;
      const double t_noc = s.noc_edge_bytes / (hw.eta_noc * hw.link_peak_bytes);
      return t_mem > t_noc ? t_mem : t_noc;  // same start, later completion
    }
    case StageResource::TIMER:
      return s.timer_seconds;
  }
  return 0.0;
}

double stage_noc_leg_seconds(const DieStage& s, const Hardware& hw) {
  if (s.resource != StageResource::MEM_NOC) {
    return 0.0;
  }
  if (hw.profile == Profile::Tx8ReferenceV1) {
    return 0.0;  // inferred_NoC_cost = 0 (§2.1)
  }
  return s.noc_edge_bytes / (hw.eta_noc * hw.link_peak_bytes);
}

}  // namespace detail


// ---------------------------------------------------------------------------
// compile(): mesh profile candidate selection (design doc §5.1)

namespace {

// tx8_reference_v1 compile (defined below; §1.1 aggregated-streaming
// envelope, §2.2 reference executor).
std::shared_ptr<const DiePlan> compile_tx8_reference_impl(
    const DieOpSpec&, const Hardware&, const WorkloadQuantities&,
    std::shared_ptr<const DieOpSpec>);

// Tie-break key of §5.1: shorter time, then smaller SRAM, then less traffic,
// then the fixed tile lexicographic order (m, n, k, rows, cols, contexts,
// k-parallel, loop type, fusion).
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

Ranking rank_of(const detail::Candidate& c, double total) {
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


// ---------------------------------------------------------------------------
// Sound compile-time lower bound (stage E3).
//
// HISTORY: the stage A2 bound added a "launch" term that serialized all
// layer_count*weight_passes one-microsecond TIMER instances ahead of every
// per-stage line.  The stage E2 selection fuzz proved that term UNSOUND: a
// TIMER stage occupies no resource slot (evaluate_stages clamps no slot for
// it), so in the frozen isolated schedule every launch instance starts at
// its dependency tail and all of them complete within one instance
// duration, freely overlapped by resource work.  launch + per-stage line
// could therefore EXCEED the true makespan and prune the true optimum,
// violating the design doc §5.1 selection rule (evaluate ALL feasible
// candidates with the isolated stage model, pick the shortest, tie-break by
// smaller SRAM / less traffic / fixed tile order).  Stage E3 replaced it
// with the components below; every component is individually provably <=
// evaluate_stages(stages, hw).total_seconds for the same candidate, which
// is exactly what makes the branch-and-bound prune safe: a candidate whose
// lower bound already exceeds the incumbent's EXACT total can never win,
// not even through the §5.1 tie-break (that only applies to EQUAL totals,
// and the prune is a strict >).
//
// PROOFS.  Write makespan = evaluate_stages total, dur(s) =
// detail::stage_instance_seconds -- the very duration truth the evaluator
// itself uses, so each component is a relaxation of the exact schedule:
//
//   (1) Slot busy sums.  The evaluator serializes every instance of one
//       resource class on a single slot (start = max(dep finishes,
//       slot_free), finish = start + dur).  The busy intervals of one slot
//       are pairwise disjoint and lie inside [0, makespan], so their total
//       length -- sum over the slot's stages of dur(s)*repeat_s -- is <=
//       makespan.  A MEM_NOC instance holds BOTH the MEM and the NOC slot
//       for its whole max(t_mem, t_NoC) duration (§6.1), so its full
//       duration enters both sums; these are exactly the quantities
//       evaluate_stages accumulates into pe/vec/noc/mem_seconds.
//
//   (2) Slot prefix + busy.  Let e_q be any lower bound of the earliest
//       start of an instance on slot q; all busy of slot q then lies in
//       [e_q, makespan] and makespan >= e_q + busy_q.  We take e_q as the
//       minimum, over the slot's stages s, of the dependency-chain length
//       that must complete before instance 0 of s (the critical-path value
//       of (s, 0) minus dur(s)): every instance (s, j) starts no earlier,
//       because its dep targets are >= those of instance 0 and the
//       instance-level longest path is monotone in the instance index.
//       This is where TIMER time re-enters the bound SOUNDLY: a stage that
//       kAllDep-depends on the launch TIMER chains exactly ONE 1-us launch
//       instance ahead of its slot work instead of layer*pass of them.
//
//   (3) Critical path.  finish(s, j) >= start(s, j) + dur(s) and
//       start(s, j) >= finish of every dep target, so by induction over the
//       pop order finish(s, j) >= (sum of dur along any chain of dep edges
//       ending at (s, j)); the maximum over chains is <= makespan.  The
//       chain values come out of critical_path_bound() below in two sound
//       variants (exact small-instance mode and backward-edge-only mode);
//       see its comment.
//
//   (4) [removed in E3] The stage A2 bound also carried the whole-op
//       physical aggregates (total FLOP over the die PE peak, class byte
//       streams over the HBM bandwidth).  For every well-formed candidate
//       they are DOMINATED by (1): §6.2 service units satisfy
//       service >= bytes/eta_class per stage, so the class byte aggregate
//       is <= the MEM slot busy sum, and the FLOP aggregate divides by the
//       die peak (cores x per-core rate) while the PE slot busy already
//       serializes every instance at the per-core rate.  They were dropped
//       because their soundness additionally relies on that accounting
//       relationship holding between the candidate's byte totals and its
//       stages' service units -- a relation the structural components
//       (1)-(3) do not need; the bound is now sound for ANY stage list the
//       evaluator itself schedules.
//
// Used ONLY to prune the compile-time search; the evaluation that decides
// is always the full stage model.

// Instance budget under which critical_path_bound follows every in-range
// dep edge (the flat memo costs O(total instances); above it only
// backward edges are followed -- O(#stages + #deps) nodes).
constexpr std::uint64_t kCpExactInstanceBudget = 2048;

// Result of critical_path_bound: the critical-path value and the per-slot
// chain prefixes e_q of component (2) (-1 = the slot has no stage).
struct CriticalPathBound {
  double path = 0.0;
  double slot_prefix[4] = {-1.0, -1.0, -1.0, -1.0};
};

// Lag-folded critical path (component (3)) plus per-slot prefixes
// (component (2)).  forward_lag_edges selects between two sound variants:
//
//   true  -- EXACT mode: every in-range dep edge of the instance DAG is
//            followed, including deps on LATER stages (the real pipelines:
//            load[j] waits for pe[j-lag]) and self pipelining lags.  The
//            value equals the true instance-level longest path restricted
//            to those edges, so repeated lag cycles (single-buffered
//            pipelines) fold into their full serial depth -- the tightness
//            that separates a ctx=1 candidate from a ctx=2 one on small
//            repeat counts.  The flat memo costs O(total instances) and
//            the recursion depth is bounded by it, so the caller only
//            enables this below kCpExactInstanceBudget.
//
//   false -- BACKWARD mode: only deps on earlier stages (and strictly
//            instance-decreasing self lags) are followed, which makes the
//            recursion well-founded with O(#stages + #deps) memo nodes for
//            arbitrarily large repeats.  Large-repeat candidates are
//            dominated by their slot busy sums anyway (component (1)), so
//            the lost alternation depth costs little pruning there.
//            (Node-count note: without SELF lag deps the backward edge set
//            strictly decreases the stage index on every edge, so the memo
//            stays at O(#stages + #deps); Mapping.cc only ever emits
//            cross-stage lags, and a self lag-0 or self kAllDep dep would
//            make evaluate_stages itself throw a dependency cycle.)
//
// Edge adhesion in BOTH modes mirrors detail::evaluate_stages's dep_target
// for in-range targets (kAllDep -> repeat_d - 1 when repeat_d > 0; lag l ->
// j - l when j >= l).  Three classes of edges are DROPPED, which only
// REMOVES precedence constraints and therefore only SHORTENS the computed
// path (sound): (a) a lag target beyond the dep stage's repeat, which the
// frozen evaluator aliases through the flat-id round trip; (b) cyclic self
// shapes (self kAllDep, self lag 0) on which evaluate_stages itself throws;
// (c) in backward mode, deps on later stages.  The exact mode additionally
// carries a gray mark: should a cycle somehow be met (impossible for any
// candidate the evaluator accepts), the closing edge is dropped rather
// than looped.
CriticalPathBound critical_path_bound(const std::vector<DieStage>& stages,
                                      const std::vector<double>& dur,
                                      bool forward_lag_edges) {
  const std::size_t ns = stages.size();
  std::vector<std::uint64_t> offset(ns + 1, 0);
  for (std::size_t s = 0; s < ns; ++s) {
    offset[s + 1] = offset[s] + stages[s].wave_repeat;
  }

  struct Solver {
    const std::vector<DieStage>& stages;
    const std::vector<double>& dur;
    const std::vector<std::uint64_t>& offset;
    const bool fwd;
    std::vector<std::uint8_t> gray;                 // exact mode: 1 on stack
    std::vector<double> val;                        // exact mode
    std::vector<std::map<std::uint64_t, double>> memo;  // backward mode

    Solver(const std::vector<DieStage>& st, const std::vector<double>& d,
           const std::vector<std::uint64_t>& off, bool f)
        : stages(st), dur(d), offset(off), fwd(f) {}

    double chain(std::size_t s, std::uint64_t j) {
      if (fwd) {
        const std::size_t id = static_cast<std::size_t>(offset[s] + j);
        if (gray[id] == 2) {
          return val[id];
        }
        if (gray[id] == 1) {
          return 0.0;  // cycle: drop the closing edge (sound)
        }
        gray[id] = 1;
        const double v = dur[s] + best_dep(s, j);
        gray[id] = 2;
        val[id] = v;
        return v;
      }
      const auto it = memo[s].find(j);
      if (it != memo[s].end()) {
        return it->second;
      }
      const double v = dur[s] + best_dep(s, j);
      memo[s].emplace(j, v);
      return v;
    }

    double best_dep(std::size_t s, std::uint64_t j) {
      double best = 0.0;
      for (const DieStageDep& dep : stages[s].deps) {
        if (!fwd && dep.stage > s) {
          continue;  // later-stage dep: backward mode drops it (sound)
        }
        const std::uint64_t rep_d = stages[dep.stage].wave_repeat;
        if (rep_d == 0) {
          continue;  // no instance: vacuous edge (matches dep_target)
        }
        std::int64_t target = -1;
        if (dep.lag == DieStageDep::kAllDep) {
          if (dep.stage == s) {
            continue;  // cyclic self shape: dropped (sound)
          }
          target = static_cast<std::int64_t>(rep_d) - 1;
        } else {
          if (dep.stage == s && dep.lag == 0) {
            continue;  // zero-length self cycle: dropped (sound)
          }
          if (j >= dep.lag) {
            const std::uint64_t t = j - dep.lag;
            if (t < rep_d) {
              target = static_cast<std::int64_t>(t);
            }
            // t >= rep_d: aliased by the frozen evaluator through the
            // flat-id round trip; the bound drops the edge (sound).
          }
        }
        if (target < 0) {
          continue;
        }
        best = std::max(
            best, chain(dep.stage, static_cast<std::uint64_t>(target)));
      }
      return best;
    }
  } solver(stages, dur, offset, forward_lag_edges);

  if (forward_lag_edges) {
    solver.gray.assign(static_cast<std::size_t>(offset[ns]), 0);
    solver.val.assign(static_cast<std::size_t>(offset[ns]), 0.0);
  } else {
    solver.memo.resize(ns);
  }

  CriticalPathBound out;
  for (std::size_t s = 0; s < ns; ++s) {
    if (stages[s].wave_repeat == 0) {
      continue;
    }
    out.path = std::max(out.path, solver.chain(s, stages[s].wave_repeat - 1));
    const double prefix = solver.chain(s, 0) - dur[s];  // >= 0 by definition
    const StageResource res = stages[s].resource;
    const int slots[2] = {
        res == StageResource::PE       ? 0
        : res == StageResource::VEC    ? 1
        : res == StageResource::NOC    ? 2
        : res == StageResource::MEM    ? 3
        : res == StageResource::MEM_NOC ? 2
                                        : -1,
        res == StageResource::MEM_NOC ? 3 : -1};  // MEM_NOC holds both slots
    for (const int q : slots) {
      if (q < 0) {
        continue;  // TIMER occupies no slot
      }
      if (out.slot_prefix[q] < 0.0 || prefix < out.slot_prefix[q]) {
        out.slot_prefix[q] = prefix;
      }
    }
  }
  return out;
}

std::shared_ptr<const DiePlan> plan_from_candidate(
    const detail::Candidate& c, std::shared_ptr<const DieOpSpec> spec,
    const Hardware& hw, OpKind op_kind) {
  auto data = std::make_shared<DiePlanData>();
  data->spec = std::move(spec);
  data->hardware = hw;
  data->summary = c.summary;
  data->summary.op_kind = op_kind;
  data->summary.profile = hw.profile;
  data->stages = c.stages;
  data->flops = c.flops;
  data->eqops = c.eqops;
  data->matrix_bytes = c.matrix_bytes;
  data->vector_bytes = c.vector_bytes;
  data->matrix_service_units = c.matrix_service_units;
  data->vector_service_units = c.vector_service_units;
  data->noc_total_link_bytes = c.noc_total_link_bytes;
  data->noc_max_edge_bytes = c.noc_max_edge_bytes;
  return std::make_shared<const DiePlan>(std::move(data));
}

}  // namespace

namespace detail {

// detail::candidate_lower_bound is declared in DieTypes.hh (internal
// mapping interface) so the selection fuzz test can check soundness
// against evaluate_stages directly.
double candidate_lower_bound(const Candidate& c, const Hardware& hw) {
  const std::size_t ns = c.stages.size();
  std::vector<double> dur(ns, 0.0);
  double busy[4] = {0.0, 0.0, 0.0, 0.0};
  std::uint64_t instances = 0;
  for (std::size_t s = 0; s < ns; ++s) {
    const DieStage& st = c.stages[s];
    dur[s] = detail::stage_instance_seconds(st, hw);
    const double b = dur[s] * static_cast<double>(st.wave_repeat);
    instances += st.wave_repeat;
    switch (st.resource) {
      case StageResource::PE:
        busy[0] += b;
        break;
      case StageResource::VEC:
        busy[1] += b;
        break;
      case StageResource::NOC:
        busy[2] += b;
        break;
      case StageResource::MEM:
        busy[3] += b;
        break;
      case StageResource::MEM_NOC:
        busy[2] += b;  // holds BOTH slots for max(t_mem, t_NoC) (§6.1)
        busy[3] += b;
        break;
      case StageResource::TIMER:
        break;  // occupies no resource slot
    }
  }
  const CriticalPathBound cp = critical_path_bound(
      c.stages, dur, instances <= kCpExactInstanceBudget);
  double lb = cp.path;  // component (3)
  for (int q = 0; q < 4; ++q) {
    lb = std::max(lb, busy[q]);  // component (1)
    if (cp.slot_prefix[q] >= 0.0) {
      lb = std::max(lb, cp.slot_prefix[q] + busy[q]);  // component (2)
    }
  }
  return lb;
}

// Shared candidate selection (§5.1): full stage-model evaluation ordered by
// the SOUND lower bound (stage E3), branch-and-bound pruning, ties resolved
// by the Ranking rule.  compile() and the Mapping.cc FFN fallback both land
// here, guaranteeing one selection semantics.  The prune is exact: a
// candidate is skipped only when its lower bound strictly exceeds the
// incumbent's EXACT total, and every component of the bound is <= that
// candidate's own exact total, so the pruning can never remove a candidate
// that would have won (equal totals still fall through to the tie-break).
// Equivalence with no-pruning full enumeration is pinned by
// tests/selection_fuzz_test.cc.
std::size_t pick_best_candidate(const std::vector<Candidate>& cands,
                                const Hardware& hw) {
  // Stage E2: compute each candidate's lower bound once (the bounds are
  // pure functions of the candidate); the previous form recomputed both
  // comparands inside the sort comparator (~2*N*log N evaluations) and
  // again inside the pruning loop.  Stage E3 additionally breaks sort ties
  // by index so the scan order among equal bounds is deterministic across
  // standard-library implementations (it only matters for candidates that
  // are complete Ranking ties, where either choice satisfies §5.1).
  std::vector<double> lb(cands.size(), 0.0);
  for (std::size_t i = 0; i < cands.size(); ++i) {
    lb[i] = candidate_lower_bound(cands[i], hw);
  }
  std::vector<std::size_t> order(cands.size());
  for (std::size_t i = 0; i < order.size(); ++i) {
    order[i] = i;
  }
  std::sort(order.begin(), order.end(), [&lb](std::size_t a, std::size_t b) {
    return lb[a] != lb[b] ? lb[a] < lb[b] : a < b;
  });
  std::size_t best = order[0];
  Ranking best_rank = rank_of(
      cands[best], evaluate_stages(cands[best].stages, hw).total_seconds);
  // Prune slack: the bound's components and the evaluator accumulate the
  // same positive durations in different orders (per stage vs. per pop /
  // fast-forward line reconstruction), so a bound can sit an ulp-scale
  // amount above the exact total of its own candidate (the stage E2
  // differential fuzz pinned the fast path itself to <1e-12 relative).
  // Skipping only CLEARLY dominated candidates keeps the prune from ever
  // discarding a total-tie participant of the §5.1 tie-break; the band is
  // one part in 1e9, so the cost is negligible.
  constexpr double kPruneRel = 1e-9;
  constexpr double kPruneAbs = 1e-15;
  for (std::size_t oi = 1; oi < order.size(); ++oi) {
    const std::size_t idx = order[oi];
    if (lb[idx] > best_rank.total * (1.0 + kPruneRel) + kPruneAbs) {
      break;  // every remaining candidate is bounded above the incumbent
    }
    const double t = evaluate_stages(cands[idx].stages, hw).total_seconds;
    const Ranking r = rank_of(cands[idx], t);
    if (better(r, best_rank)) {
      best = idx;
      best_rank = r;
    }
  }
  return best;
}

}  // namespace detail

std::shared_ptr<const DiePlan> compile(const DieOpSpec& spec,
                                       const Hardware& hw) {
  // Shared workload semantic checks (both profiles, contract §4).
  const WorkloadQuantities wl = derive_workload(spec);
  auto spec_ptr = std::make_shared<const DieOpSpec>(spec);

  if (hw.profile == Profile::Tx8ReferenceV1) {
    return compile_tx8_reference_impl(spec, hw, wl, std::move(spec_ptr));
  }

  std::vector<detail::Candidate> cands = detail::enumerate_candidates(wl, hw);
  if (cands.empty()) {
    throw DieModelError("compile: no feasible mapping candidate for op_kind " +
                        std::string(to_string(spec.op_kind)));
  }
  const std::size_t best_index = detail::pick_best_candidate(cands, hw);
  return plan_from_candidate(cands[best_index], std::move(spec_ptr), hw,
                             spec.op_kind);
}

// ---------------------------------------------------------------------------
// compile(): tx8_reference_v1 profile (design doc §1.1/§2.2)
//
// Builds the aggregated-streaming reference envelope per op: the compute
// budget chain PE(Fop) -> vec(Eop) and the memory stream(s) start together
// and the op completes at the later one.  No tile mapping is involved; only
// the §2.1 reference shape is accepted.

namespace {

constexpr double kRefM = 1024.0;
constexpr double kRefH = 4096.0;
constexpr double kRefF = 14336.0;
constexpr double kRefJ = 1024.0;  // Hkv*d
constexpr double kRefA = 32.0 * 1024.0 * 1024.0;

struct RefGroup {
  double fop = 0.0;
  double eop = 0.0;
  double d_matrix = 0.0;
  double d_vector = 0.0;
};

[[noreturn]] void tx8_reject(const std::string& why) {
  throw DieModelError(
      "compile: tx8_reference_v1 profile only reproduces the design doc §2.1 "
      "reference shapes (B=1, Sq=Sk=1024, H=4096, F=14336, Hq=32, Hkv=8, "
      "d=128, TP=1, BF16, weight_passes=1, single prefill span): " + why);
}

std::shared_ptr<const DiePlan> compile_tx8_reference_impl(
    const DieOpSpec& spec, const Hardware& hw, const WorkloadQuantities& wl,
    std::shared_ptr<const DieOpSpec> spec_ptr) {
  // --- §2.1 reference shape validation ---
  if (spec.dtype != Dtype::Bf16) {
    tx8_reject("dtype must be bf16");
  }
  if (spec.phase != Phase::Prefill) {
    tx8_reject("phase must be prefill");
  }
  if (spec.spans.size() != 1 || spec.spans[0].q_tokens != 1024 ||
      spec.spans[0].kv_tokens != 1024) {
    tx8_reject("exactly one span with q_tokens=kv_tokens=1024 is required");
  }
  if (spec.weight_passes != 1) {
    tx8_reject("weight_passes must be 1");
  }
  const DieModelDims& md = spec.model_dims;
  if (md.hidden != 4096 || md.ffn != 14336 || md.num_q_heads != 32 ||
      md.num_kv_heads != 8 || md.head_dim != 128) {
    tx8_reject("model_dims must be H=4096, F=14336, Hq=32, Hkv=8, d=128");
  }
  const DieTpSlice& tp = spec.tp_slice;
  if (tp.tensor_parallel != 1 || tp.tensor_parallel_rank != 0 ||
      tp.q_heads_local != 32 || tp.kv_heads_local != 8 ||
      tp.ffn_local != 14336 || tp.vocab_local != md.vocab) {
    tx8_reject("tp_slice must be the full unsharded TP=1 extent");
  }
  switch (spec.op_kind) {
    case OpKind::AttentionRegion:
      tx8_reject("attention_region keeps scores on tile; the reference "
                 "materializes the full score rectangle (emit qk/scale_mask/"
                 "softmax/av separately)");
      break;
    case OpKind::Logits:
    case OpKind::LocalRegion:
      tx8_reject("op_kind is not part of the reference block");
      break;
    default:
      break;
  }

  const double M = kRefM;
  const double H = kRefH;
  const double F = kRefF;
  const double J = kRefJ;
  const double A = kRefA;
  const double e = 2.0;
  RefGroup g;
  bool ok = true;
  switch (spec.op_kind) {
    case OpKind::Norm:
      if (md.norm_type == NormType::Rmsnorm) {
        g.eop = 6.0 * M * H;
        g.d_vector = 4.0 * M * H + 2.0 * H;
      } else {
        g.eop = 8.0 * M * H;  // LayerNorm kept for graph compatibility (§5.3)
        g.d_vector = 4.0 * M * H + 4.0 * H;
      }
      break;
    case OpKind::QkvRope: {
      // Envelope of Q/K/V projections + RoPE (§2.2 rows 2-5); classes sum
      // on the memory side (§6.1 t_mem_alone).
      g.fop = 2.0 * M * H * H + 2.0 * (2.0 * M * H * J);
      g.d_matrix = 2.0 * (M * H + H * H + M * H) +
                   2.0 * (2.0 * (M * H + H * J + M * J));
      if (md.position_encoding == PositionEncoding::Rope) {
        g.eop = 3.0 * M * (H + J);
        g.d_vector = 4.0 * M * (H + J) + 4.0 * 1024.0 * 128.0;
      }
      break;
    }
    case OpKind::Qk:
      g.fop = 2.0 * A * 128.0;
      g.d_matrix = 2.0 * 32.0 * 1024.0 * 128.0 + 2.0 * 8.0 * 1024.0 * 128.0 + 4.0 * A;
      break;
    case OpKind::ScaleMask:  // unfused split of the 12/score chain (A2 ruling)
      g.eop = 2.0 * A;
      g.d_vector = 8.0 * A;
      break;
    case OpKind::Softmax:
      g.eop = 10.0 * A;
      g.d_vector = 8.0 * A;
      break;
    case OpKind::Av:
      g.fop = 2.0 * A * 128.0;
      g.eop = A;  // FP32->BF16 probability conversion per score (§2.2)
      g.d_matrix = 4.0 * A + 2.0 * 8.0 * 1024.0 * 128.0 + 2.0 * 32.0 * 1024.0 * 128.0;
      break;
    case OpKind::OProj:
      g.fop = 2.0 * M * H * H;
      g.d_matrix = 2.0 * (M * H + H * H + M * H);
      break;
    case OpKind::Residual:
      g.eop = M * H;
      g.d_vector = 6.0 * M * H;
      break;
    case OpKind::MlpUp:
      if (md.mlp_variant == MlpVariant::Swiglu) {
        g.fop = 2.0 * M * H * (2.0 * F);
        g.d_matrix = 2.0 * (M * H + H * 2.0 * F + M * 2.0 * F);
      } else {
        g.fop = 2.0 * M * H * F;
        g.d_matrix = 2.0 * (M * H + H * F + M * F);
      }
      break;
    case OpKind::MlpActivation:
      if (md.mlp_variant == MlpVariant::Swiglu) {
        g.eop = 16.0 * M * F;
        g.d_vector = 3.0 * e * M * F;  // gate+up read, activation write
      } else {
        g.eop = 15.0 * M * F;
        g.d_vector = 2.0 * e * M * F;
      }
      break;
    case OpKind::MlpDown:
      g.fop = 2.0 * M * F * H;
      g.d_matrix = 2.0 * (M * F + F * H + M * H);
      break;
    case OpKind::FfnRegion:
      if (md.mlp_variant == MlpVariant::Swiglu) {
        g.fop = 6.0 * M * H * F;
        g.eop = 16.0 * M * F;
        g.d_matrix = 6.0 * H * F + 4.0 * M * H;
      } else {
        g.fop = 4.0 * M * H * F;
        g.eop = 15.0 * M * F;
        g.d_matrix = 4.0 * H * F + 4.0 * M * H;
      }
      break;
    default:
      ok = false;
      break;
  }
  if (!ok) {
    tx8_reject("op_kind has no reference-group semantics");
  }
  const double L = static_cast<double>(spec.layer_count);
  g.fop *= L;
  g.eop *= L;
  g.d_matrix *= L;
  g.d_vector *= L;

  auto data = std::make_shared<DiePlanData>();
  data->spec = std::move(spec_ptr);
  data->hardware = hw;
  data->summary.op_kind = spec.op_kind;
  data->summary.profile = Profile::Tx8ReferenceV1;
  data->summary.loop_kind = "tx8_reference_envelope";
  data->summary.layer_count = spec.layer_count;
  data->summary.weight_passes = spec.weight_passes;
  data->summary.launch_instances = spec.layer_count * spec.weight_passes;

  // Launch timer (extra_launch = 0 in the reference, §2.1).
  DieStage launch;
  launch.label = "launch";
  launch.resource = StageResource::TIMER;
  launch.wave_repeat = data->summary.launch_instances;
  launch.timer_seconds = hw.tx8_extra_launch_s;
  data->stages.push_back(launch);

  // Compute budget chain PE(Fop) -> vec(Eop) (§2.2 reference executor).
  DieStage pe;
  pe.label = "pe";
  pe.resource = StageResource::PE;
  pe.wave_repeat = 1;
  pe.pe_core_flop = g.fop;
  pe.deps.push_back({0, DieStageDep::kAllDep});
  data->stages.push_back(pe);

  DieStage vec;
  vec.label = "vec";
  vec.resource = StageResource::VEC;
  vec.wave_repeat = 1;
  vec.vec_core_eqop = g.eop;
  vec.deps.push_back({1, DieStageDep::kAllDep});
  data->stages.push_back(vec);

  // Memory stream(s), started together with the compute branch; the two
  // classes sum on the service side (§6.1).
  auto push_mem = [&](const char* label, double bytes, bool matrix,
                      std::uint32_t dep) {
    if (bytes <= 0.0) {
      return;
    }
    DieStage m;
    m.label = label;
    m.resource = StageResource::MEM;
    m.wave_repeat = 1;
    m.mem.actual_bytes = bytes;
    m.mem.matrix_class = matrix;
    m.mem.eta = matrix ? hw.eta_matrix_mem : hw.eta_vector_mem;
    // Reference: service units carry the raw bytes (the class bandwidths of
    // §2.1 already include their own efficiency); no granularity rule.
    m.mem.service_units = bytes;
    m.deps.push_back({dep, DieStageDep::kAllDep});
    data->stages.push_back(m);
  };
  push_mem("mem_matrix", g.d_matrix, true, 0);
  if (g.d_matrix > 0.0 && g.d_vector > 0.0) {
    // Both classes present: they sum on the service side (§6.1 t_mem_alone),
    // so the vector stream chains after the matrix stream on the MEM slot.
    push_mem("mem_vector", g.d_vector, false,
             static_cast<std::uint32_t>(data->stages.size() - 1));
  } else {
    push_mem("mem_vector", g.d_vector, false, 0);
  }

  data->flops = g.fop;
  data->eqops = g.eop;
  data->matrix_bytes = g.d_matrix;
  data->vector_bytes = g.d_vector;
  data->matrix_service_units = g.d_matrix;
  data->vector_service_units = g.d_vector;
  return std::make_shared<const DiePlan>(std::move(data));
}

}  // namespace

// ---------------------------------------------------------------------------
// estimate_alone (contract §4)

CostResult estimate_alone(const DiePlan& plan) {
  if (!plan.valid()) {
    throw DieModelError("estimate_alone: empty plan");
  }
  const Hardware& hw = plan.hardware();
  const detail::ScheduleResult sr = detail::evaluate_stages(plan.stages(), hw);
  const DiePlanSummary& sum = plan.summary();

  CostResult r;
  r.profile = hw.profile;
  r.total_seconds = sr.total_seconds;
  r.pe_seconds = sr.pe_seconds;
  r.vec_seconds = sr.vec_seconds;
  r.noc_seconds = sr.noc_seconds;
  r.mem_seconds = sr.mem_seconds;
  r.launch_instances = sum.launch_instances;
  for (const DieStage& s : plan.stages()) {
    if (s.resource == StageResource::TIMER) {
      r.launch_seconds +=
          s.timer_seconds * static_cast<double>(s.wave_repeat);
    }
  }
  r.flops = plan.flops();
  r.eqops = plan.eqops();
  r.matrix_actual_bytes = plan.matrix_bytes();
  r.vector_actual_bytes = plan.vector_bytes();
  r.matrix_service_units = plan.matrix_service_units();
  r.vector_service_units = plan.vector_service_units();
  r.noc_link_bytes_total = plan.noc_total_link_bytes();
  r.noc_edge_bytes_max = plan.noc_max_edge_bytes();
  r.sram_peak_per_core = sum.sram_peak_per_core;
  r.fused = sum.fused;
  r.fallback = sum.fallback;
  r.total_ns_suggested =
      r.total_seconds > 0.0
          ? static_cast<std::uint64_t>(std::max(
                1.0, std::ceil(r.total_seconds * 1e9)))
          : 0;
  return r;
}

// ---------------------------------------------------------------------------
// PlanCache + frozen cache key (contract §5; FNV-1a 64, stage A2 report)

PlanCache::PlanCache(std::size_t capacity) : capacity_(capacity == 0 ? 1 : capacity) {}

std::shared_ptr<const DiePlan> PlanCache::get(const std::string& key) const {
  auto it = index_.find(key);
  if (it == index_.end()) {
    return nullptr;
  }
  lru_.splice(lru_.begin(), lru_, it->second);
  return it->second->second;
}

void PlanCache::put(const std::string& key, std::shared_ptr<const DiePlan> plan) {
  auto it = index_.find(key);
  if (it != index_.end()) {
    it->second->second = std::move(plan);
    lru_.splice(lru_.begin(), lru_, it->second);
    return;
  }
  lru_.emplace_front(key, std::move(plan));
  index_[key] = lru_.begin();
  while (lru_.size() > capacity_) {
    index_.erase(lru_.back().first);
    lru_.pop_back();
  }
}

std::size_t PlanCache::size() const { return lru_.size(); }
std::size_t PlanCache::capacity() const { return capacity_; }

std::string plan_cache_hardware_key(const Hardware& hw) {
  // The canonical hardware summary embedded by plan_cache_key (stage A2
  // report; byte-identical substring so hardware keys and plan keys can be
  // mixed by callers that share caches across model instances).
  std::string s = "hw{";
  s += to_string(hw.profile);
  s += "," + std::to_string(hw.sram_mb_per_core);
  s += "," + std::to_string(hw.compute_area_mm2) + "," +
       std::to_string(hw.hbm_per_side) + "," + std::to_string(hw.hbm_sides) +
       "," + std::to_string(hw.noc_width_bits);
  if (hw.explicit_die_dimensions) {
    s += ",shape{" + std::to_string(hw.die_width_mm) + "," +
         std::to_string(hw.die_height_mm) + "}";
  }
  s += ",d{" + std::to_string(hw.mesh_n) + "," + std::to_string(hw.cores) + "," +
       std::to_string(hw.pe_peak_die) + "," + std::to_string(hw.vec_peak_die) +
       "," + std::to_string(hw.sram_usable_bytes_per_core) + "," +
       std::to_string(hw.link_peak_bytes) + "," + std::to_string(hw.hbm_count) +
       "," + std::to_string(hw.hbm_capacity_bytes) + "," +
       std::to_string(hw.hbm_bandwidth_bytes) + "}";
  s += ",eta{" + std::to_string(hw.eta_pe) + "," + std::to_string(hw.eta_vec) +
       "," + std::to_string(hw.eta_matrix_mem) + "," +
       std::to_string(hw.eta_vector_mem) + "," + std::to_string(hw.eta_noc) +
       "," + std::to_string(hw.launch_us_per_macro_group) + "}";
  s += "}";
  return s;
}

namespace {

// Which op kinds have a compiled plan that depends on per-span shape fields
// (q_tokens, kv_tokens, q_start_pos, kv_start_pos).  Attention-family kinds
// do: their workload floor (valid pairs) and tile walks read the per-span
// causal geometry.  Every other kind is compiled from the merged token
// count M = sum(spans.q_tokens) alone -- derive_workload folds the batch
// into one physical M and no mapping path of these kinds reads
// wl.attention_spans -- so the plan-cache key collapses their spans into
// M.  No default case: adding an OpKind enumerator without deciding its
// family is a compile error under -Wswitch, and the library tests pin the
// classification by compiling shape-differing specs of every linear kind.
bool op_reads_span_shapes(OpKind kind) {
  switch (kind) {
    case OpKind::Qk:
    case OpKind::Av:
    case OpKind::AttentionRegion:
    case OpKind::ScaleMask:
    case OpKind::Softmax:
      return true;
    case OpKind::Norm:
    case OpKind::QkvRope:
    case OpKind::OProj:
    case OpKind::Residual:
    case OpKind::MlpUp:
    case OpKind::MlpActivation:
    case OpKind::MlpDown:
    case OpKind::FfnRegion:
    case OpKind::Logits:
    case OpKind::LocalRegion:
      return false;
  }
  return false;  // unreachable once the switch covers every enumerator
}

// Span-encoding policy of the plan-cache key: attention-family kinds carry
// per-span shapes; the tx8_reference_v1 profile validates the span shape
// itself (frozen reference reproduction), so under that profile every kind
// keeps the shape fields -- all legal tx8 specs share the one reference
// span list, so this costs no hits and preserves exact reject behavior.
bool spans_in_cache_key(OpKind kind, const Hardware& hw) {
  return op_reads_span_shapes(kind) || hw.profile == Profile::Tx8ReferenceV1;
}

// Bit-identical double comparison for plan verification: two compiles of
// semantically equal specs run identical operation sequences, so equal
// values carry equal bit patterns (memcmp also avoids NaN false alarms).
bool same_double(double a, double b) {
  return std::memcmp(&a, &b, sizeof(double)) == 0;
}

// Debug-only plan equality for the cache-hit verification: identical
// summary, identical stage list, bit-identical numerics.  The embedded
// DieOpSpec copies are deliberately NOT compared -- a semantic-key hit may
// legitimately carry the first spec, whose identity-only fields (request_id
// and, for linear kinds, the per-span shape fields) differ from the probe
// spec while the plan itself is unaffected (nothing reads them).
bool plans_agree(const DiePlan& hit, const DiePlan& fresh) {
  const DiePlanSummary& a = hit.summary();
  const DiePlanSummary& b = fresh.summary();
  if (a.op_kind != b.op_kind || a.profile != b.profile ||
      a.loop_kind != b.loop_kind || a.m_tile != b.m_tile ||
      a.n_tile != b.n_tile || a.k_tile != b.k_tile ||
      a.group_rows != b.group_rows || a.group_cols != b.group_cols ||
      a.contexts != b.contexts || a.k_parallel != b.k_parallel ||
      a.weight_resident != b.weight_resident ||
      a.double_buffered != b.double_buffered || a.fused != b.fused ||
      a.fallback != b.fallback ||
      a.streaming_two_pass != b.streaming_two_pass ||
      !same_double(a.sram_peak_per_core, b.sram_peak_per_core) ||
      a.layer_count != b.layer_count || a.weight_passes != b.weight_passes ||
      a.launch_instances != b.launch_instances || a.waves != b.waves ||
      !same_double(a.weight_hbm_bytes, b.weight_hbm_bytes)) {
    return false;
  }
  if (hit.stages().size() != fresh.stages().size()) {
    return false;
  }
  for (std::size_t i = 0; i < hit.stages().size(); ++i) {
    const DieStage& x = hit.stages()[i];
    const DieStage& y = fresh.stages()[i];
    if (x.label != y.label || x.resource != y.resource ||
        x.macro_group != y.macro_group || x.wave_repeat != y.wave_repeat ||
        !same_double(x.pe_core_flop, y.pe_core_flop) ||
        !same_double(x.vec_core_eqop, y.vec_core_eqop) ||
        !same_double(x.noc_edge_bytes, y.noc_edge_bytes) ||
        !same_double(x.timer_seconds, y.timer_seconds) ||
        !same_double(x.mem.actual_bytes, y.mem.actual_bytes) ||
        !same_double(x.mem.service_units, y.mem.service_units) ||
        !same_double(x.mem.eta, y.mem.eta) ||
        x.mem.matrix_class != y.mem.matrix_class ||
        x.deps.size() != y.deps.size()) {
      return false;
    }
    for (std::size_t k = 0; k < x.deps.size(); ++k) {
      if (x.deps[k].stage != y.deps[k].stage ||
          x.deps[k].lag != y.deps[k].lag) {
        return false;
      }
    }
  }
  return same_double(hit.flops(), fresh.flops()) &&
         same_double(hit.eqops(), fresh.eqops()) &&
         same_double(hit.matrix_bytes(), fresh.matrix_bytes()) &&
         same_double(hit.vector_bytes(), fresh.vector_bytes()) &&
         same_double(hit.matrix_service_units(),
                     fresh.matrix_service_units()) &&
         same_double(hit.vector_service_units(),
                     fresh.vector_service_units()) &&
         same_double(hit.noc_total_link_bytes(),
                     fresh.noc_total_link_bytes()) &&
         same_double(hit.noc_max_edge_bytes(), fresh.noc_max_edge_bytes());
}

// Key builder with a precomputed hardware summary (the summary costs ~40
// to_string calls; per-node callers precompute it once) and the explicit
// span-encoding policy of spans_in_cache_key.
std::string plan_cache_key_with_hardware(const DieOpSpec& spec,
                                         const std::string& hardware_key,
                                         bool spans_in_key) {
  // Explicitly ordered canonical string: hardware summary + coefficient
  // version + every spec field that can change the plan (contract §5/§8.5).
  std::string s = "die_model-plan-key-v1|";
  s += hardware_key;
  s += "|spec{";
  s += "v" + std::to_string(spec.schema_version);
  s += "," + std::string(to_string(spec.op_kind));
  s += "," + std::string(to_string(spec.phase));
  s += "," + std::string(to_string(spec.dtype));
  const DieModelDims& md = spec.model_dims;
  s += ",dims{" + std::to_string(md.hidden) + "," + std::to_string(md.ffn) +
       "," + std::to_string(md.vocab) + "," + to_string(md.mlp_variant) + "," +
       to_string(md.norm_type) + "," + to_string(md.position_encoding) + "," +
       std::to_string(md.num_q_heads) + "," + std::to_string(md.num_kv_heads) +
       "," + std::to_string(md.head_dim) + "}";
  const DieTpSlice& tp = spec.tp_slice;
  s += ",tp{" + std::to_string(tp.tensor_parallel) + "," +
       std::to_string(tp.tensor_parallel_rank) + "," +
       std::to_string(tp.q_heads_local) + "," + std::to_string(tp.kv_heads_local) +
       "," + std::to_string(tp.ffn_local) + "," + std::to_string(tp.vocab_local) +
       "}";
  if (spans_in_key) {
    // Attention family (and the tx8 reference profile, which validates the
    // span shape itself): the plan depends on every span's causal shape,
    // so the key carries each shape tuple in spec order (span order fixes
    // the accumulation order inside compile()).  request_id is
    // deliberately absent: it is an opaque identity that never enters any
    // plan computation, yet it made every train/rank shard a unique key.
    s += ",spans[";
    for (const DieSpan& sp : spec.spans) {
      s += std::to_string(sp.q_tokens) + "," + std::to_string(sp.kv_tokens) +
           "," + std::to_string(sp.q_start_pos) + "," +
           std::to_string(sp.kv_start_pos) + ";";
    }
    s += "]";
  } else {
    // Linear family: the plan depends on the spans only through the merged
    // token count M = sum(q_tokens) (derive_workload folds the batch into
    // one physical M; no mapping path of these kinds reads a per-span
    // field).  Encoding M instead of the span list makes shards with equal
    // M share one plan, so a decode run with a fixed batch hits the cache
    // on every step instead of recompiling per step.
    std::uint64_t m_tokens = 0;
    for (const DieSpan& sp : spec.spans) {
      m_tokens += sp.q_tokens;
    }
    s += ",M" + std::to_string(m_tokens);
  }
  s += ",L" + std::to_string(spec.layer_count);
  s += ",P" + std::to_string(spec.weight_passes);
  s += ",kv" + std::string(to_string(spec.kv_access.identity));
  s += ",fus" + std::string(to_string(spec.fusion_policy));
  s += ",hash{" + spec.config_hash + "}";
  s += ",prov{" + spec.provenance + "}";

  // FNV-1a 64-bit.
  std::uint64_t h = 14695981039346656037ULL;
  for (char c : s) {
    h ^= static_cast<std::uint64_t>(static_cast<unsigned char>(c));
    h *= 1099511628211ULL;
  }
  static const char* kHex = "0123456789abcdef";
  std::string out(16, '0');
  for (int i = 15; i >= 0; --i) {
    out[static_cast<std::size_t>(i)] = kHex[h & 0xF];
    h >>= 4;
  }
  return out;
}

}  // namespace

std::string plan_cache_key(const DieOpSpec& spec, const Hardware& hw) {
  // Semantic plan key: identical to the historical string except the span
  // section is semantic (identity-only fields out; linear-family kinds
  // collapse to the merged token count).  See spans_in_cache_key and
  // plan_cache_key_with_hardware above.
  return plan_cache_key_with_hardware(
      spec, plan_cache_hardware_key(hw), spans_in_cache_key(spec.op_kind, hw));
}

std::shared_ptr<const DiePlan> cached_compile(
    const DieOpSpec& spec, const Hardware& hw, PlanCache& cache,
    const std::string& hardware_key) {
  // Shared cache-or-compile for the C ABI model objects and the AstraSim
  // die runtime: compile() is a pure function (a mapping violation throws
  // DieModelError), so the immutable plan enters the LRU cache under the
  // semantic key and the mutable execution handle never does.
  // hardware_key is plan_cache_hardware_key(hw) computed once per model /
  // runtime -- rebuilding the hardware summary cost dozens of to_string
  // calls on every node issue.
  const std::string key = plan_cache_key_with_hardware(
      spec, hardware_key, spans_in_cache_key(spec.op_kind, hw));
  std::shared_ptr<const DiePlan> plan = cache.get(key);
  if (plan != nullptr) {
    // Semantic-key invariant: a hit must be exactly what a cold compile of
    // THIS spec produces.  The check costs one compile, so it runs only
    // where assert() is active (NDEBUG builds compile it out).
    assert(plans_agree(*plan, *compile(spec, hw)) &&
           "plan-cache key merged specs with different plans");
    return plan;
  }
  plan = compile(spec, hw);
  cache.put(key, plan);
  return plan;
}

}  // namespace DieModel
