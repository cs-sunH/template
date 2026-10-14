/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

/*
k2_plan_negative_cache_test.cc -- v12 Phase 4 (方案 §6.2) fixture for the
process-level STRUCTURAL negative cache of query_internal_timer_window.

The cache is keyed by (plan cache key, window-approx config fingerprint)
only -- no run state -- so it may only fire where the full computation
provably returns the SAME complete rejection (reason included) in EVERY
reachable state of ANY Execution of the plan. That class is exactly
Impl::structural_negative (P_struct; see docs/PHASE4_K2_CACHE_DESIGN.md):
plans finished at construction, whose every query answer is the constant
{supported=false, reason="execution-finished"}.

Covered:
  - twin oracle: for a battery of plan families (sequential PE chain + MEM
    convergence, parallel pre-scheduled waves, pure PE chain without MEM,
    single-wave plan, lag-dependency plan, zero-duration sync-finished plan,
    empty plan) the identical drive script runs with the negative cache
    ENABLED (and pre-warmed, so entries exist) versus DISABLED; every
    per-boundary query result must be field-for-field equal, reason strings
    included, across full-speed and 2x/4x slow external-memory drives;
  - engagement: P_struct plans actually hit the negative cache (hits > 0)
    and realistic plans never write or hit it (the cache is a no-op there,
    so the v11 epoch memo behavior is untouched);
  - fire-face consistency: for every family, when P_struct holds, sampled
    drives under full/2x/4x memory service observe no accepted window and
    every reason is the constant "execution-finished"; the fixture also
    proves it reaches real accepts (sequential) and real rejections
    (parallel) on non-P_struct plans so the Sound check is not vacuous;
  - fingerprint isolation: entries under the production fingerprint are
    invisible under an injected second fingerprint (and vice versa), the
    two coexist in one cache, and answers stay identical regardless;
  - the master switch: disabled => zero writes/hits and the v11 epoch memo
    serves repeated same-state queries exactly as before;
  - argument validation (max_count outside [1,8]) still throws before any
    cache interaction.
*/

#include "DieModel.hh"

#include <cstdint>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "test_helpers.hh"

namespace DieModel {
struct ExecutionTestAccess {
  static std::uint64_t negative_hits(const Execution& execution) {
    return execution.negative_cache_hits_for_test();
  }
  static std::uint64_t negative_writes(const Execution& execution) {
    return execution.negative_cache_writes_for_test();
  }
  static bool structural_negative(const Execution& execution) {
    return execution.structural_negative_for_test();
  }
  static std::uint64_t memo_hits(const Execution& execution) {
    return execution.timer_window_memo_hits_for_test();
  }
};
}  // namespace DieModel

namespace {

int failures = 0;

void expect(bool condition, const std::string& message) {
  if (!condition) {
    std::cerr << "[k2_plan_negative_cache_test] FAIL: " << message << '\n';
    ++failures;
  }
}

// Field-for-field equality of the complete query result.
bool same_token(const DieModel::InternalServiceWindowToken& a,
                const DieModel::InternalServiceWindowToken& b) {
  return a.valid == b.valid && a.stage_id == b.stage_id &&
         a.first_wave_index == b.first_wave_index &&
         a.service_already_applied == b.service_already_applied &&
         a.completed_slices == b.completed_slices &&
         a.first_action.template_key == b.first_action.template_key &&
         a.first_action.flat_instance_id == b.first_action.flat_instance_id &&
         a.first_action.frontier_generation ==
             b.first_action.frontier_generation &&
         a.first_action.valid == b.first_action.valid &&
         a.original_service_ticks == b.original_service_ticks;
}

bool same_slice(const DieModel::InternalTemplateSliceEntry& a,
                const DieModel::InternalTemplateSliceEntry& b) {
  return a.stage_id == b.stage_id && a.wave_index == b.wave_index &&
         a.resource == b.resource && a.original_work == b.original_work &&
         a.original_service_ns == b.original_service_ns &&
         a.hardware_min_ns == b.hardware_min_ns;
}

bool same_window(const DieModel::InternalTimerWindowInfo& a,
                 const DieModel::InternalTimerWindowInfo& b) {
  if (a.supported != b.supported || a.reason != b.reason ||
      a.slices.size() != b.slices.size() || !same_token(a.token, b.token)) {
    return false;
  }
  for (std::size_t i = 0; i < a.slices.size(); ++i) {
    if (!same_slice(a.slices[i], b.slices[i])) {
      return false;
    }
  }
  return true;
}

// Deterministic cache hygiene between test sections: the singleton survives
// the whole binary, so every section starts (and the process ends) from a
// cleared cache under the production fingerprint with the switch on.
struct CacheGuard {
  DieModel::K2TimerWindowNegativeCache& cache;
  explicit CacheGuard(DieModel::K2TimerWindowNegativeCache& c) : cache(c) {
    reset();
  }
  ~CacheGuard() { reset(); }
  void reset() {
    cache.clear();
    cache.set_enabled(true);
    cache.set_config_fingerprint(
        DieModel::k2_window_approx_config::production_fingerprint());
  }
};

// ---------------------------------------------------------------------------
// Plan families.

std::shared_ptr<const DieModel::DiePlan> plan_with_spec(
    std::vector<DieModel::DieStage> stages,
    std::uint64_t q = 4, std::uint64_t kv = 4) {
  auto data = std::make_shared<DieModel::DiePlanData>();
  data->spec = std::make_shared<const DieModel::DieOpSpec>(
      die_test::spec_of(DieModel::OpKind::LocalRegion,
                        DieModel::Phase::Prefill, q, kv, 0, 0));
  data->hardware = die_test::hw_default();
  data->stages = std::move(stages);
  return std::make_shared<const DieModel::DiePlan>(std::move(data));
}

DieModel::DieStage pe_stage(const char* label, std::uint64_t repeat,
                            double work_ns) {
  const auto hw = die_test::hw_default();
  DieModel::DieStage pe;
  pe.label = label;
  pe.resource = DieModel::StageResource::PE;
  pe.wave_repeat = repeat;
  if (repeat > 1) pe.deps.emplace_back(0, 1);
  const double rate = hw.core_row.pe_gflops_per_core * 1e9 * hw.eta_pe;
  pe.pe_core_flop = work_ns * rate / 1e9 * (1.0 - 1e-12);
  return pe;
}

DieModel::DieStage mem_stage(const char* label) {
  DieModel::DieStage mem;
  mem.label = label;
  mem.resource = DieModel::StageResource::MEM;
  mem.mem.actual_bytes = 64.0;
  mem.mem.service_units = 64.0;
  mem.deps.emplace_back(0, DieModel::DieStageDep::kAllDep);
  return mem;
}

// Accepted-window family: a pure sequential PE chain converging on one MEM
// request (the k2_timer_window_memo_test accept fixture shape).
std::shared_ptr<const DieModel::DiePlan> sequential_pe_then_mem_plan() {
  return plan_with_spec({pe_stage("k2-neg-seq-pe", 4, 1000.0),
                         mem_stage("k2-neg-seq-mem")});
}

// Dominant production rejection shape: independent parallel PE waves are
// pre-scheduled, so the frontier is never a single just-started wave.
std::shared_ptr<const DieModel::DiePlan> parallel_pe_plan() {
  return plan_with_spec({pe_stage("k2-neg-par-pe", 2, 1000.0)});
}

// Pure PE chain without any MEM stage (drives straight to Done).
std::shared_ptr<const DieModel::DiePlan> pure_pe_chain_plan() {
  return plan_with_spec({pe_stage("k2-neg-chain-pe", 5, 800.0)});
}

// Single-wave plan: the window can only ever prove one slice.
std::shared_ptr<const DieModel::DiePlan> single_wave_pe_then_mem_plan() {
  return plan_with_spec({pe_stage("k2-neg-one-pe", 1, 1200.0),
                         mem_stage("k2-neg-one-mem")});
}

// Lag-dependency family: the second PE stage's wave j depends on wave j-2
// of the first stage (waves 0/1 have no dependency), mixing parallel-ready
// and lag-released frontiers.
std::shared_ptr<const DieModel::DiePlan> lag_dependent_plan() {
  DieModel::DieStage first = pe_stage("k2-neg-lag-pe0", 4, 700.0);
  DieModel::DieStage second = pe_stage("k2-neg-lag-pe1", 2, 500.0);
  second.deps.clear();
  second.deps.emplace_back(0, 2);
  return plan_with_spec({first, second});
}

// P_struct family (P_B): every instance is a zero-duration internal stage
// and no MEM/MEM_NOC instance exists, so the constructor's run() drains the
// whole DAG synchronously and every reachable state is finished.
std::shared_ptr<const DieModel::DiePlan> zero_duration_sync_plan() {
  DieModel::DieStage pe = pe_stage("k2-neg-zero-pe", 3, 0.0);
  DieModel::DieStage timer;
  timer.label = "k2-neg-zero-timer";
  timer.resource = DieModel::StageResource::TIMER;
  timer.wave_repeat = 2;
  timer.timer_seconds = 0.0;
  timer.deps.emplace_back(0, DieModel::DieStageDep::kAllDep);
  DieModel::DieStage vec;
  vec.label = "k2-neg-zero-vec";
  vec.resource = DieModel::StageResource::VEC;
  vec.wave_repeat = 2;
  vec.deps.emplace_back(2, 1);
  const auto hw = die_test::hw_default();
  const double rate = hw.core_row.vec_g_eqop_per_core * 1e9 * hw.eta_vec;
  vec.vec_core_eqop = 0.0 * rate;  // zero work => zero duration
  // Distinct q_tokens keep this family's plan cache key apart from the
  // other P_struct fixture (the empty plan below).
  return plan_with_spec({pe, timer, vec}, 8, 8);
}

// P_struct family (P_A): zero instances at all.
std::shared_ptr<const DieModel::DiePlan> empty_plan() {
  return plan_with_spec({}, 9, 9);
}

// ---------------------------------------------------------------------------
// Drive script: advance through every action boundary, sampling the window
// query before and after each step. memory_wait_ns emulates the external
// memory service time the driver inserts before complete_memory() (the
// full/2x/4x "slow memory" histories of the fire-face check).
std::vector<DieModel::InternalTimerWindowInfo> drive_and_collect(
    DieModel::Execution& execution, const std::uint32_t max_count,
    const std::uint64_t memory_wait_ns) {
  std::vector<DieModel::InternalTimerWindowInfo> samples;
  const auto sample = [&]() {
    samples.push_back(execution.query_internal_timer_window(max_count));
  };
  for (int guard = 0; guard < 4096; ++guard) {
    if (execution.finished()) break;
    const DieModel::ExecutionAction action = execution.next_action();
    if (action.kind == DieModel::ExecutionAction::Kind::Done) break;
    if (action.kind == DieModel::ExecutionAction::Kind::Timer) {
      sample();
      execution.advance_time(action.ns);
      sample();
    } else {
      sample();
      execution.advance_external_wait(memory_wait_ns, action.stage_id);
      sample();
      execution.complete_memory(action.stage_id);
      sample();
    }
  }
  sample();  // finished/idle tail state
  return samples;
}

struct Family {
  const char* name;
  std::shared_ptr<const DieModel::DiePlan> plan;
};

std::vector<Family> families() {
  return {{"sequential-pe-then-mem", sequential_pe_then_mem_plan()},
          {"parallel-pe", parallel_pe_plan()},
          {"pure-pe-chain", pure_pe_chain_plan()},
          {"single-wave-pe-then-mem", single_wave_pe_then_mem_plan()},
          {"lag-dependent", lag_dependent_plan()},
          {"zero-duration-sync", zero_duration_sync_plan()},
          {"empty", empty_plan()}};
}

void test_twin_oracle_enabled_vs_disabled() {
  auto& cache = DieModel::k2_timer_window_negative_cache();
  CacheGuard guard(cache);
  for (const Family& family : families()) {
    for (const std::uint64_t factor : {1u, 2u, 4u}) {
      for (const std::uint32_t max_count : {2u, 4u}) {
        // Warm the process cache with one full drive of this plan so the
        // enabled twin actually faces populated entries.
        DieModel::Execution warm(family.plan);
        (void)drive_and_collect(warm, max_count, 500 * factor);

        // Enabled twin FIRST (the warm entries are live), then the disabled
        // twin on an identically-driven fresh execution.
        DieModel::Execution cached(family.plan);
        const auto enabled_samples =
            drive_and_collect(cached, max_count, 500 * factor);
        cache.set_enabled(false);
        DieModel::Execution plain(family.plan);
        const auto disabled_samples =
            drive_and_collect(plain, max_count, 500 * factor);
        cache.set_enabled(true);

        expect(enabled_samples.size() == disabled_samples.size(),
               std::string(family.name) + ": twin sample counts differ");
        for (std::size_t i = 0;
             i < enabled_samples.size() && i < disabled_samples.size(); ++i) {
          expect(same_window(enabled_samples[i], disabled_samples[i]),
                 std::string(family.name) + ": twin query " +
                     std::to_string(i) + " differs (reason '" +
                     enabled_samples[i].reason + "' vs '" +
                     disabled_samples[i].reason + "')");
        }

        if (DieModel::ExecutionTestAccess::structural_negative(cached)) {
          expect(DieModel::ExecutionTestAccess::negative_hits(cached) > 0,
                 std::string(family.name) + ": P_struct plan never hit the "
                 "negative cache (fixture would pass vacuously)");
          expect(DieModel::ExecutionTestAccess::negative_writes(cached) == 0,
                 std::string(family.name) + ": enabled twin rewrote entries "
                 "that the warm drive had already stored");
        } else {
          expect(DieModel::ExecutionTestAccess::negative_hits(cached) == 0 &&
                     DieModel::ExecutionTestAccess::negative_writes(cached) ==
                         0,
                 std::string(family.name) + ": realistic plan touched the "
                 "negative cache (must be a no-op)");
        }
      }
    }
  }
}

void test_predicate_matches_sampled_drives() {
  auto& cache = DieModel::k2_timer_window_negative_cache();
  CacheGuard guard(cache);
  cache.set_enabled(false);  // sample the pure full computation
  bool saw_accept = false;
  bool saw_parallel_reject = false;
  for (const Family& family : families()) {
    const bool predicate =
        DieModel::ExecutionTestAccess::structural_negative(
            DieModel::Execution{family.plan});
    for (const std::uint64_t factor : {1u, 2u, 4u}) {
      DieModel::Execution execution(family.plan);
      const auto samples = drive_and_collect(execution, 2, 500 * factor);
      if (predicate) {
        for (std::size_t i = 0; i < samples.size(); ++i) {
          expect(!samples[i].supported,
                 std::string(family.name) + ": P_struct plan accepted a "
                 "window under the x" + std::to_string(factor) +
                 " memory drive (Sound violation)");
          expect(samples[i].reason == "execution-finished",
                 std::string(family.name) + ": P_struct reason is not the "
                 "constant 'execution-finished' but '" +
                 samples[i].reason + "'");
        }
      }
      for (const auto& sample : samples) {
        if (sample.supported) saw_accept = true;
        if (sample.reason == "other_scheduled_internal_boundary") {
          saw_parallel_reject = true;
        }
      }
    }
  }
  expect(saw_accept,
         "fire-face fixture never observed a real accepted window");
  expect(saw_parallel_reject,
         "fire-face fixture never observed the parallel-wave rejection");
}

void test_structural_plans_constant_rejection_all_max_counts() {
  auto& cache = DieModel::k2_timer_window_negative_cache();
  CacheGuard guard(cache);
  for (const auto& plan : {zero_duration_sync_plan(), empty_plan()}) {
    DieModel::Execution a(plan);
    DieModel::Execution b(plan);
    expect(DieModel::ExecutionTestAccess::structural_negative(a) &&
               DieModel::ExecutionTestAccess::structural_negative(b),
           "constant-rejection fixture: plan is not P_struct");
    for (std::uint32_t max_count = 1; max_count <= 8; ++max_count) {
      const auto qa = a.query_internal_timer_window(max_count);
      const auto qb = b.query_internal_timer_window(max_count);
      expect(!qa.supported && qa.reason == "execution-finished" &&
                 qa.slices.empty() && !qa.token.valid,
             "constant-rejection fixture: max_count " +
                 std::to_string(max_count) +
                 " did not return the finished rejection");
      expect(same_window(qa, qb),
             "constant-rejection fixture: twins disagree at max_count " +
                 std::to_string(max_count));
    }
    expect(DieModel::ExecutionTestAccess::negative_writes(a) >= 1,
           "constant-rejection fixture: rejection was never stored");
    expect(DieModel::ExecutionTestAccess::negative_hits(b) >= 1,
           "constant-rejection fixture: second execution never hit the "
           "process-level cache (cross-Execution reuse is the point)");
  }
}

void test_fingerprint_isolation() {
  auto& cache = DieModel::k2_timer_window_negative_cache();
  CacheGuard guard(cache);
  const std::string production =
      DieModel::k2_window_approx_config::production_fingerprint();

  // Warm under the production fingerprint.
  DieModel::Execution a(zero_duration_sync_plan());
  (void)a.query_internal_timer_window(4);
  expect(DieModel::ExecutionTestAccess::negative_writes(a) == 1,
         "fingerprint fixture: first query wrote exactly one entry");
  expect(cache.size() == 1, "fingerprint fixture: cache holds one entry");

  // Under an injected second fingerprint the production entry is invisible:
  // same plan key, different configuration conclusion domain.
  cache.set_config_fingerprint("test-alt-fingerprint-v1");
  DieModel::Execution b(zero_duration_sync_plan());
  const auto qb = b.query_internal_timer_window(4);
  expect(DieModel::ExecutionTestAccess::negative_hits(b) == 0,
         "fingerprint fixture: production entry leaked into the alt "
         "fingerprint probe");
  expect(DieModel::ExecutionTestAccess::negative_writes(b) == 1,
         "fingerprint fixture: the alt fingerprint stored its own entry");
  expect(cache.size() == 2,
         "fingerprint fixture: entries under both fingerprints coexist");
  expect(!qb.supported && qb.reason == "execution-finished",
         "fingerprint fixture: recomputed answer is still the constant "
         "rejection (Sound regardless of fingerprint)");

  // Switching back: the production entry survived untouched.
  cache.set_config_fingerprint(production);
  DieModel::Execution c(zero_duration_sync_plan());
  const auto qc = c.query_internal_timer_window(4);
  expect(DieModel::ExecutionTestAccess::negative_hits(c) == 1 &&
             DieModel::ExecutionTestAccess::negative_writes(c) == 0,
         "fingerprint fixture: production entry no longer serves its own "
         "fingerprint");
  expect(same_window(qb, qc),
         "fingerprint fixture: both fingerprints serve identical answers");

  // A realistic (non-P_struct) plan stores nothing under ANY fingerprint.
  cache.set_config_fingerprint("test-alt-fingerprint-v1");
  DieModel::Execution realistic(sequential_pe_then_mem_plan());
  (void)drive_and_collect(realistic, 4, 500);
  cache.set_config_fingerprint(production);
  DieModel::Execution realistic2(sequential_pe_then_mem_plan());
  (void)drive_and_collect(realistic2, 4, 1000);
  expect(DieModel::ExecutionTestAccess::negative_writes(realistic) == 0 &&
             DieModel::ExecutionTestAccess::negative_writes(realistic2) == 0,
         "fingerprint fixture: realistic plan wrote entries");
  expect(cache.size() == 2,
         "fingerprint fixture: realistic plans changed the entry count");
}

void test_disabled_switch_and_v11_memo_intact() {
  auto& cache = DieModel::k2_timer_window_negative_cache();
  CacheGuard guard(cache);
  cache.set_enabled(false);

  // P_struct plan with the cache off: no writes, no hits, and the v11 epoch
  // memo serves the repeated same-state query exactly as before.
  DieModel::Execution dead(zero_duration_sync_plan());
  const auto q1 = dead.query_internal_timer_window(4);
  expect(DieModel::ExecutionTestAccess::negative_writes(dead) == 0 &&
             DieModel::ExecutionTestAccess::negative_hits(dead) == 0,
         "switch fixture: disabled cache was still written");
  expect(DieModel::ExecutionTestAccess::memo_hits(dead) == 0,
         "switch fixture: first query must be a memo miss");
  const auto q2 = dead.query_internal_timer_window(4);
  expect(DieModel::ExecutionTestAccess::memo_hits(dead) == 1,
         "switch fixture: repeated query must hit the v11 epoch memo");
  expect(same_window(q1, q2),
         "switch fixture: memo repeat changed the answer");

  // Realistic plan with the cache off across a full drive: still nothing.
  DieModel::Execution realistic(sequential_pe_then_mem_plan());
  (void)drive_and_collect(realistic, 4, 500);
  expect(DieModel::ExecutionTestAccess::negative_writes(realistic) == 0 &&
             DieModel::ExecutionTestAccess::negative_hits(realistic) == 0,
         "switch fixture: disabled cache engaged on a realistic plan");
  expect(DieModel::ExecutionTestAccess::memo_hits(realistic) > 0,
         "switch fixture: v11 memo stopped serving repeats");
}

void test_argument_validation_precedes_cache() {
  auto& cache = DieModel::k2_timer_window_negative_cache();
  CacheGuard guard(cache);
  for (const auto& plan :
       {sequential_pe_then_mem_plan(), zero_duration_sync_plan()}) {
    DieModel::Execution execution(plan);
    for (const std::uint32_t bad : {0u, 9u}) {
      bool threw = false;
      try {
        (void)execution.query_internal_timer_window(bad);
      } catch (const DieModel::DieModelError&) {
        threw = true;
      }
      expect(threw, "validation fixture: max_count " + std::to_string(bad) +
                       " must throw before any cache interaction");
    }
    expect(DieModel::ExecutionTestAccess::negative_hits(execution) == 0 &&
               DieModel::ExecutionTestAccess::negative_writes(execution) ==
                   0,
           "validation fixture: an illegal argument mutated the cache");
  }
}

}  // namespace

int main() {
  test_twin_oracle_enabled_vs_disabled();
  test_predicate_matches_sampled_drives();
  test_structural_plans_constant_rejection_all_max_counts();
  test_fingerprint_isolation();
  test_disabled_switch_and_v11_memo_intact();
  test_argument_validation_precedes_cache();
  if (failures != 0) {
    std::cerr << "[k2_plan_negative_cache_test] " << failures
              << " failure(s)\n";
    return 1;
  }
  std::cout << "[k2_plan_negative_cache_test] PASS\n";
  return 0;
}
