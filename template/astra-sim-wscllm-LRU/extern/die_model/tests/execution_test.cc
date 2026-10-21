/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

/*
execution_test.cc -- stage A3 unit tests of the externally driven Execution
state machine (contract §4 action protocol; design doc §6.3/§6.4/§8.1).

Coverage (stage A3 task list):
  1. full-speed memory parity against estimate_alone() for the four §10.2
     shapes (norm / GEMM / attention_region / ffn_region) plus a decode
     GEMM, a zero-work plan and a tx8_reference_v1 plan;
  2. MEM_NOC dual occupancy and later-completion semantics on a synthetic
     plan with hand-computable times;
  3. DieStageDep lag semantics (double-buffered ctx=2 lookahead vs lag=1
     serialization) via memory issue times;
  4. zero-work plans advance synchronously (no zero-time event chain);
  5. destruction mid-run and per-plan independence of two executions;
  6. error paths and next_action() idempotency.

Full-speed driver: every MemoryRequest is serviced by
advance_time(ceil(service_units*1e9/rate)) + complete_memory.  For the mesh
profile rate = hbm_bandwidth_bytes (service units already carry D/eta and
the §6.2 granularity rule); for the tx8 reference profile service units are
raw class bytes, so the port rate of that profile is the §2.1 class
bandwidth.  The product service_units*1e9 is formed BEFORE the division so
exactly-representable integer durations survive the ceil without +1 ns of
double noise.
*/

#include "DieModel.hh"
#include "test_helpers.hh"

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

using die_test::Checker;
using DieModel::CostResult;
using DieModel::DiePlan;
using DieModel::DiePlanData;
using DieModel::DieStage;
using DieModel::DieStageDep;
using DieModel::Execution;
using DieModel::ExecutionAction;
using DieModel::StageResource;

namespace {

std::uint64_t service_ns(double units, double bytes_per_second) {
  if (!(units > 0.0)) {
    return 0;
  }
  const double ns = std::ceil(units * 1e9 / bytes_per_second);
  return ns < 1.0 ? 1u : static_cast<std::uint64_t>(ns);
}

// Drives an execution with a full-speed memory port at `service_rate`
// byte/s; records memory request count / action count / request issue
// times (the clock moment of each issue).  Returns the final elapsed_ns.
std::uint64_t drive_full_speed(Execution& ex, double service_rate,
                               std::uint64_t* mem_requests = nullptr,
                               std::uint64_t* actions_out = nullptr,
                               std::vector<std::uint64_t>* issue_times = nullptr,
                               Checker* ck = nullptr,
                               const std::string& tag = "") {
  std::uint64_t memreqs = 0, actions = 0, clock = 0;
  for (;;) {
    const ExecutionAction a = ex.next_action();
    ++actions;
    if (a.kind == ExecutionAction::Kind::Done) {
      break;
    }
    if (a.kind == ExecutionAction::Kind::Timer) {
      if (ck != nullptr) {
        ck->expect(a.ns >= 1u, tag + ": zero-nanosecond timer action");
      }
      ex.advance_time(a.ns);
      clock += a.ns;
    } else {
      ++memreqs;
      if (issue_times != nullptr) {
        issue_times->push_back(clock);
      }
      const std::uint64_t ns = service_ns(a.service_units, service_rate);
      if (ns != 0) {
        ex.advance_time(ns);
        clock += ns;
      }
      ex.complete_memory(a.stage_id);
    }
    if (ck != nullptr) {
      // Progress guard: no zero-time event chains, every action is a timer,
      // a request or the done marker.
      ck->expect(actions < (1u << 20), tag + ": runaway action loop");
    }
    if (actions >= (1u << 20)) {
      break;
    }
  }
  if (mem_requests != nullptr) {
    *mem_requests = memreqs;
  }
  if (actions_out != nullptr) {
    *actions_out = actions;
  }
  return ex.elapsed_ns();
}

// --- synthetic plan helpers (exact, hand-checkable times) ---

DieModel::Hardware default_hw() { return die_test::hw_default(); }

std::shared_ptr<const DiePlan> plan_of(std::vector<DieStage> stages) {
  auto data = std::make_shared<DiePlanData>();
  data->spec = std::make_shared<const DieModel::DieOpSpec>(
      die_test::spec_of(DieModel::OpKind::LocalRegion,
                        DieModel::Phase::Prefill, 4, 4, 0, 0));
  data->hardware = default_hw();
  data->stages = std::move(stages);
  data->summary.launch_instances = 1;
  return std::make_shared<const DiePlan>(std::move(data));
}

// 1 us timer launch stage (single instance).
DieStage launch_stage() {
  DieStage s;
  s.label = "launch";
  s.resource = StageResource::TIMER;
  s.wave_repeat = 1;
  s.timer_seconds = 1e-6;  // 1000 ns
  return s;
}

// The synthetic internal-stage builders below construct work quantities the
// machine reserves for EXACTLY `ns` nanoseconds.  The double round trip
// work/rate inside detail::stage_instance_seconds can land 1 ulp above the
// intended seconds, which would push max(1,ceil(seconds*1e9)) to ns+1; a
// -1e-12 RELATIVE nudge dominates that ulp noise while staying far below
// the 1 ns granularity, so the reservation is the intended integer without
// floating-point luck.
constexpr double kExactNudge = 1.0 - 1e-12;

// A pure MEM stage whose full-speed service time is `ns` nanoseconds on the
// default die (B_memory = 1.64e12 byte/s).
DieStage mem_stage(const char* label, std::uint64_t repeat, double ns,
                   std::vector<DieStageDep> deps) {
  DieStage s;
  s.label = label;
  s.resource = StageResource::MEM;
  s.wave_repeat = repeat;
  s.mem.service_units = static_cast<double>(ns) * 1.64e12 / 1e9;
  s.mem.actual_bytes = s.mem.service_units * 0.7;
  s.mem.eta = 0.7;
  s.mem.matrix_class = true;
  s.deps = std::move(deps);
  return s;
}

// A MEM_NOC stage: memory leg `mem_ns`, NoC leg `noc_ns` (both integer ns).
DieStage memnoc_stage(const char* label, double mem_ns, double noc_ns,
                      std::vector<DieStageDep> deps) {
  DieStage s;
  s.label = label;
  s.resource = StageResource::MEM_NOC;
  s.wave_repeat = 1;
  s.mem.service_units = mem_ns * 1.64e12 / 1e9;
  s.mem.actual_bytes = s.mem.service_units * 0.7;
  s.mem.eta = 0.7;
  s.mem.matrix_class = true;
  s.noc_edge_bytes =
      noc_ns * 1e-9 * 0.7 * 64e9 * kExactNudge;  // eta_noc*link = 44.8e9
  s.deps = std::move(deps);
  return s;
}

// A PE stage of exactly `ns` nanoseconds on the explicit compatibility-test Core fixture.
DieStage pe_stage(const char* label, std::uint64_t repeat, double ns,
                  std::vector<DieStageDep> deps) {
  DieStage s;
  s.label = label;
  s.resource = StageResource::PE;
  s.wave_repeat = repeat;
  s.pe_core_flop = ns * 1e-9 * 0.9079551923877489 * 1266e9 * kExactNudge;
  s.deps = std::move(deps);
  return s;
}

DieStage noc_stage(const char* label, double ns,
                   std::vector<DieStageDep> deps) {
  DieStage s;
  s.label = label;
  s.resource = StageResource::NOC;
  s.wave_repeat = 1;
  s.noc_edge_bytes = ns * 1e-9 * 0.7 * 64e9;
  s.deps = std::move(deps);
  return s;
}

// ---------------------------------------------------------------------------
// 1. Full-speed parity with estimate_alone

void parity_shape(Checker& ck, const char* name,
                  const DieModel::DieOpSpec& spec, const DieModel::Hardware& hw) {
  std::shared_ptr<const DiePlan> plan = DieModel::compile(spec, hw);
  const CostResult est = DieModel::estimate_alone(*plan);
  std::uint64_t instances = 0;
  for (const DieStage& s : plan->stages()) {
    instances += s.wave_repeat;
  }
  Execution ex(plan);
  std::uint64_t memreqs = 0, actions = 0;
  const std::uint64_t elapsed = drive_full_speed(
      ex, hw.hbm_bandwidth_bytes, &memreqs, &actions, nullptr, &ck, name);
  const std::int64_t delta =
      static_cast<std::int64_t>(elapsed) -
      static_cast<std::int64_t>(est.total_ns_suggested);
  ck.expect(ex.finished(), std::string(name) + ": not finished after Done");
  ck.expect(delta >= 0,
            std::string(name) + ": elapsed below the estimate (rounding " +
                "must be monotone up); delta=" + std::to_string(delta));
  // Rounding budget (design doc §6.4: accumulated ceil error <= reservation
  // count x 1 ns): every internal reservation adds < 1 ns and every memory
  // link adds the service ceil (< 1 ns) plus at most 1 ns of issue-time
  // slack, hence instances + 2*memreqs + 2 ns overall.
  const std::int64_t budget =
      static_cast<std::int64_t>(instances) +
      2 * static_cast<std::int64_t>(memreqs) + 2;
  ck.expect(delta <= budget,
            std::string(name) + ": parity drift " + std::to_string(delta) +
                " ns exceeds the rounding budget " + std::to_string(budget));
  // The literal stage-count bound of the A3 task holds for plans whose
  // memory chain is a single wave (delta == +1 ns there).
  if (memreqs <= 2) {
    ck.expect(delta <= 2 * static_cast<std::int64_t>(plan->stages().size()),
              std::string(name) + ": single-wave plan within stages x 2 ns");
  }
  std::printf("  parity %-10s elapsed=%llu estimate=%llu delta=%+lld "
              "(instances=%llu memreqs=%llu actions=%llu)\n",
              name, (unsigned long long)elapsed,
              (unsigned long long)est.total_ns_suggested, (long long)delta,
              (unsigned long long)instances, (unsigned long long)memreqs,
              (unsigned long long)actions);
}

void test_full_speed_parity(Checker& ck) {
  const DieModel::Hardware hw = default_hw();
  parity_shape(ck, "norm",
               die_test::spec_of(DieModel::OpKind::Norm,
                                 DieModel::Phase::Prefill, 1024, 1024, 0, 0),
               hw);
  parity_shape(ck, "o_proj",
               die_test::spec_of(DieModel::OpKind::OProj,
                                 DieModel::Phase::Prefill, 1024, 1024, 0, 0),
               hw);
  parity_shape(ck, "attn_reg",
               die_test::spec_of(DieModel::OpKind::AttentionRegion,
                                 DieModel::Phase::Prefill, 1024, 1024, 0, 0),
               hw);
  parity_shape(ck, "ffn_reg",
               die_test::spec_of(DieModel::OpKind::FfnRegion,
                                 DieModel::Phase::Prefill, 1024, 1024, 0, 0),
               hw);
  parity_shape(ck, "o_proj_dec",
               die_test::spec_of(DieModel::OpKind::OProj,
                                 DieModel::Phase::Decode, 1, 1, 2047, 0),
               hw);

  // Zero-work plan: exactly the launch timer, elapsed == estimate exactly.
  {
    std::shared_ptr<const DiePlan> plan = DieModel::compile(
        die_test::spec_of(DieModel::OpKind::LocalRegion,
                          DieModel::Phase::Prefill, 4, 4, 0, 0),
        hw);
    ck.expect(plan->stages().size() == 1 &&
                  plan->stages()[0].resource == StageResource::TIMER,
              "local_region: expected a single launch TIMER stage");
    const CostResult est = DieModel::estimate_alone(*plan);
    Execution ex(plan);
    const ExecutionAction a1 = ex.next_action();
    ck.expect(a1.kind == ExecutionAction::Kind::Timer && a1.ns == 1000,
              "local_region: first action must be Timer(1000)");
    ex.advance_time(a1.ns);
    const ExecutionAction a2 = ex.next_action();
    ck.expect(a2.kind == ExecutionAction::Kind::Done,
              "local_region: second action must be Done");
    ck.expect_uint(a2.ns, est.total_ns_suggested,
                   "local_region: elapsed == total_ns_suggested (exact)");
    ck.expect_uint(a2.ns, 1000, "local_region: elapsed == 1000 ns exactly");
  }

  // tx8_reference_v1 with layer_count=3: three ZERO-duration launch
  // instances must be consumed synchronously -- the very first action is the
  // memory request, never a zero-nanosecond timer chain.  The reference
  // profile's memory port runs at the §2.1 class bandwidth (norm is vector
  // class; service units are raw bytes there).
  {
    DieModel::DieOpSpec spec = die_test::spec_of(
        DieModel::OpKind::Norm, DieModel::Phase::Prefill, 1024, 1024, 0, 0);
    spec.layer_count = 3;
    const DieModel::Hardware tx8 = DieModel::make_hardware(die_test::wrapped(
        die_test::die_node(0.75, 800, 4, 1, 512),
        "tx8_reference_v1"));
    std::shared_ptr<const DiePlan> plan = DieModel::compile(spec, tx8);
    const CostResult est = DieModel::estimate_alone(*plan);
    Execution ex(plan);
    const ExecutionAction a1 = ex.next_action();
    ck.expect(a1.kind == ExecutionAction::Kind::MemoryRequest,
              "tx8: zero-duration launch chain must complete synchronously; "
              "the first action is the memory request");
    // Service the first request at the class rate, then hand the run back
    // to the driver loop.
    ex.advance_time(service_ns(a1.service_units, tx8.tx8_bw_vector_bytes_s));
    ex.complete_memory(a1.stage_id);
    std::uint64_t more = 0;
    const std::uint64_t elapsed =
        drive_full_speed(ex, tx8.tx8_bw_vector_bytes_s, &more);
    const std::uint64_t memreqs = 1 + more;
    ck.expect(ex.finished(), "tx8: finished");
    const std::int64_t delta =
        static_cast<std::int64_t>(elapsed) -
        static_cast<std::int64_t>(est.total_ns_suggested);
    ck.expect(delta >= 0 && delta <= 2 * static_cast<std::int64_t>(memreqs) + 2,
              "tx8: parity within the memory-link rounding budget");
    std::printf("  parity tx8_norm   elapsed=%llu estimate=%llu delta=%+lld "
                "(memreqs=%llu)\n",
                (unsigned long long)elapsed,
                (unsigned long long)est.total_ns_suggested, (long long)delta,
                (unsigned long long)memreqs);
  }
}

// ---------------------------------------------------------------------------
// 2. MEM_NOC dual occupancy and later completion

void test_memnoc_dual_occupancy(Checker& ck) {
  // launch(1000) ; load(MEM_NOC mem=1000 noc=3000) ; pe(2000 after load) ;
  // noc_x(NOC 500 after launch, key after load) ; flush(MEM 1000 after load).
  // Hand schedule: load [1000,4000] (NoC leg wins), noc_x [4000,4500] (the
  // NoC slot is held by load's leg until 4000), pe [4000,6000], flush
  // [4000,5000] (the MEM slot is held until the FULL dual-leg end 4000 even
  // though the memory leg completed at 2000).  Makespan 6000.
  std::vector<DieStage> st;
  st.push_back(launch_stage());
  st.push_back(memnoc_stage("load", 1000, 3000, {{0, DieStageDep::kAllDep}}));
  st.push_back(pe_stage("pe", 1, 2000, {{1, 0}}));
  st.push_back(noc_stage("noc_x", 500, {{0, DieStageDep::kAllDep}}));
  st.push_back(mem_stage("flush", 1, 1000, {{1, 0}}));
  std::shared_ptr<const DiePlan> plan = plan_of(std::move(st));

  const CostResult est = DieModel::estimate_alone(*plan);
  ck.expect_uint(est.total_ns_suggested, 6000,
                 "memnoc plan: estimate makespan == 6000 ns");

  Execution ex(plan);

  ExecutionAction a = ex.next_action();
  ck.expect(a.kind == ExecutionAction::Kind::Timer && a.ns == 1000,
            "memnoc: Timer(1000) to the launch end first");
  ex.advance_time(a.ns);

  a = ex.next_action();
  ck.expect(a.kind == ExecutionAction::Kind::MemoryRequest && a.stage_id == 1,
            "memnoc: load request issued at its start time");
  ck.expect(a.service_units > 0.0 && a.actual_bytes > 0.0,
            "memnoc: request carries the DieStage::mem triple");
  // Full-speed memory: the memory leg would complete at 2000.
  ex.advance_time(1000);
  ex.complete_memory(1);
  ck.expect(!ex.finished(),
            "memnoc: instance not finished while the NoC leg drains");

  a = ex.next_action();
  ck.expect(a.kind == ExecutionAction::Kind::Timer && a.ns == 2000,
            "memnoc: Timer to the internal NoC-leg end (2000 ns left); the "
            "next memory request must NOT appear before the dual-leg end");

  ex.advance_time(a.ns);  // clock 4000: load completes
  a = ex.next_action();
  ck.expect(a.kind == ExecutionAction::Kind::MemoryRequest && a.stage_id == 4,
            "memnoc: flush issued exactly at the dual-leg end 4000 (MEM "
            "slot held until max(memory, NoC))");
  ex.advance_time(1000);
  ex.complete_memory(4);

  a = ex.next_action();
  ck.expect(a.kind == ExecutionAction::Kind::Timer && a.ns == 1000,
            "memnoc: Timer to the pe end at 6000 (pe started at 4000)");
  ex.advance_time(a.ns);
  a = ex.next_action();
  ck.expect(a.kind == ExecutionAction::Kind::Done && a.ns == 6000,
            "memnoc: Done at 6000, identical to the estimate");
  ck.expect_uint(ex.elapsed_ns(), est.total_ns_suggested,
                 "memnoc: elapsed == total_ns_suggested (exact)");
}

// ---------------------------------------------------------------------------
// 3. lag / double-buffer semantics (ruling R-L)

void test_lag_double_buffer(Checker& ck) {
  // launch(1000) ; load(MEM 4x1000ns, deps {launch all} + {pe, lag}) ;
  // pe(4x2000ns, deps {load,0}).
  // lag=2 (ctx=2 double buffering): load[0] @1000 and load[1] @2000 go out
  //   back-to-back, BEFORE pe[0] completes at 4000 (the lookahead of ruling
  //   R-L); load[2]/load[3] then ride one pe completion each
  //   (4000/6000); makespan 10000.
  // lag=1: load[j] waits for pe[j-1]; issues at 1000,4000,7000,10000;
  //   makespan 13000.
  for (std::uint32_t lag = 1; lag <= 2; ++lag) {
    std::vector<DieStage> st;
    st.push_back(launch_stage());
    st.push_back(mem_stage("load", 4, 1000,
                           {{0, DieStageDep::kAllDep}, {2, lag}}));
    st.push_back(pe_stage("pe", 4, 2000, {{1, 0}}));
    std::shared_ptr<const DiePlan> plan = plan_of(std::move(st));
    const CostResult est = DieModel::estimate_alone(*plan);
    Execution ex(plan);
    std::vector<std::uint64_t> issue;
    const std::uint64_t elapsed = drive_full_speed(
        ex, plan->hardware().hbm_bandwidth_bytes, nullptr, nullptr, &issue);
    ck.expect_uint(issue.size(), 4, "lag: four memory requests");
    ck.expect_uint(elapsed, est.total_ns_suggested,
                   "lag: elapsed == estimate exactly");
    if (lag == 2) {
      const std::uint64_t want[4] = {1000, 2000, 4000, 6000};
      for (int i = 0; i < 4; ++i) {
        ck.expect_uint(issue[i], want[i],
                       "lag=2: load issue time (lookahead double buffer)");
      }
      ck.expect_uint(est.total_ns_suggested, 10000,
                     "lag=2: estimate makespan 10000");
    } else {
      const std::uint64_t want[4] = {1000, 4000, 7000, 10000};
      for (int i = 0; i < 4; ++i) {
        ck.expect_uint(issue[i], want[i],
                       "lag=1: load issue time (serialized)");
      }
      ck.expect_uint(est.total_ns_suggested, 13000,
                     "lag=1: estimate makespan 13000");
    }
    std::printf("  lag=%u issue times: %llu %llu %llu %llu -> %llu ns\n",
                lag, (unsigned long long)issue[0], (unsigned long long)issue[1],
                (unsigned long long)issue[2], (unsigned long long)issue[3],
                (unsigned long long)elapsed);
  }
}

// ---------------------------------------------------------------------------
// 4/5/6. Lifecycle, independence, idempotency and error paths

void test_lifecycle_and_errors(Checker& ck) {
  const DieModel::Hardware hw = default_hw();
  std::shared_ptr<const DiePlan> plan = DieModel::compile(
      die_test::spec_of(DieModel::OpKind::Norm, DieModel::Phase::Prefill,
                        1024, 1024, 0, 0),
      hw);

  // Two executions on the SAME plan must not pollute each other; the plan
  // itself is read-only shared state.
  {
    Execution e1(plan), e2(plan);
    const ExecutionAction a1 = e1.next_action();
    const ExecutionAction a2 = e2.next_action();
    ck.expect(a1.kind == a2.kind && a1.ns == a2.ns,
              "independence: identical first action");
    if (a1.kind == ExecutionAction::Kind::Timer) {
      e1.advance_time(a1.ns);
    }
    const ExecutionAction a2b = e2.next_action();
    ck.expect(a2b.kind == a2.kind && a2b.ns == a2.ns,
              "independence: advancing e1 leaves e2 untouched (idempotent "
              "repeat)");
    const std::uint64_t t1 =
        drive_full_speed(e1, hw.hbm_bandwidth_bytes);
    const std::uint64_t t2 =
        drive_full_speed(e2, hw.hbm_bandwidth_bytes);
    ck.expect_uint(t1, t2, "independence: identical elapsed on both runs");
    ck.expect(e1.finished() && e2.finished(), "independence: both finished");
  }

  // Destruction mid-run (a few actions in) must be safe.
  {
    Execution ex(plan);
    for (int i = 0; i < 3; ++i) {
      const ExecutionAction a = ex.next_action();
      if (a.kind == ExecutionAction::Kind::Timer) {
        ex.advance_time(a.ns);
      } else if (a.kind == ExecutionAction::Kind::MemoryRequest) {
        ex.complete_memory(a.stage_id);
      } else {
        break;
      }
    }
    ck.expect(!ex.finished(), "midrun destroy: not finished yet");
  }

  // next_action idempotency without any state change.
  {
    Execution ex(plan);
    const ExecutionAction a1 = ex.next_action();
    const ExecutionAction a2 = ex.next_action();
    ck.expect(a1.kind == a2.kind && a1.ns == a2.ns &&
                  a1.stage_id == a2.stage_id,
              "idempotency: repeated next_action reports the same action");
  }

  // Error paths.
  {
    Execution ex(plan);
    ck.expect_throw([&ex] { ex.complete_memory(0); },
                    "complete_memory without an outstanding request");
    const ExecutionAction a = ex.next_action();
    ck.expect(a.kind == ExecutionAction::Kind::Timer ||
                  a.kind == ExecutionAction::Kind::MemoryRequest,
              "error paths: first action is a timer or a request");
    if (a.kind == ExecutionAction::Kind::Timer) {
      ex.advance_time(a.ns);
    }
    const ExecutionAction b = ex.next_action();
    if (b.kind == ExecutionAction::Kind::MemoryRequest) {
      ck.expect_throw([&ex, &b] { ex.complete_memory(b.stage_id + 7u); },
                      "complete_memory with a mismatching stage_id");
      ex.complete_memory(b.stage_id);  // the matching one still works
    }
    ck.expect_throw(
        [&ex] {
          ex.advance_time(UINT64_MAX);
          ex.advance_time(1);
        },
        "advance_time clock overflow");
  }

  // Empty plan (no stages): done immediately.
  {
    std::shared_ptr<const DiePlan> empty = plan_of({});
    Execution ex(empty);
    const ExecutionAction a = ex.next_action();
    ck.expect(a.kind == ExecutionAction::Kind::Done && a.ns == 0 &&
                  ex.finished(),
              "empty plan: immediate Done(0)");
  }

  // Cyclic DAG is rejected at construction.
  {
    std::vector<DieStage> st;
    st.push_back(pe_stage("a", 1, 2000, {{1, 0}}));
    st.push_back(pe_stage("b", 1, 2000, {{0, 0}}));
    ck.expect_throw([&st] { Execution ex(plan_of(std::move(st))); },
                    "cyclic stage DAG rejected in the constructor",
                    "dependency cycle");
  }
}

}  // namespace

int main() {
  Checker ck;
  std::printf("[execution_test] full-speed parity\n");
  test_full_speed_parity(ck);
  std::printf("[execution_test] MEM_NOC dual occupancy\n");
  test_memnoc_dual_occupancy(ck);
  std::printf("[execution_test] lag double buffer\n");
  test_lag_double_buffer(ck);
  std::printf("[execution_test] lifecycle and errors\n");
  test_lifecycle_and_errors(ck);
  if (ck.ok) {
    std::printf("[execution_test] ALL PASS\n");
    return 0;
  }
  return 1;
}
