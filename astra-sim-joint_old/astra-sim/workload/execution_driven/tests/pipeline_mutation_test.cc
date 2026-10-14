/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

/**
pipeline_mutation_test.cc -- D1 (exec doc §17.3/§17.4) window mutation and
resource-version ↔ window-generation linkage, M13 + the M07/M10 interplay.

Three fixture tiers, smallest that can express each assertion:

  Part 1 (pure value level, joint_pipeline_core): the §17.3 on_mutation /
  on_wakeup arithmetic on real PipelineWindowController/PipelineKernel --
  interrupt positions (front/middle/end/boundaries), the monotone
  service_already_applied prefix, remaining = original_total - target carry,
  the 8→4→2→1 refinement ladder under repeated invalidation, integer
  remainder without rounding drift, stale-generation rejection, the
  "cancel the plan, never cancel physical service" red line, and the
  frozen-snapshot exact recompute of the prediction world.

  Part 2 (hermetic Sys + EventQueue + LocalHbmBandwidthModel, same shape as
  die_execution_lifecycle_test.cc): a real DieExecution driving a
  launch-timer-only LocalRegion plan (4 uniform waves x 1000 ns) under a
  frozen K=8 contract. A test-injected COMM_READ HBM job triggers the real
  rank mutation observer mid-arm and exactly at the due tick, in BOTH
  physical insertion orders (M10): mutation before the due callback keeps
  the cookie and publishes once at the original callback position; the
  callback first leaves nothing to invalidate. Mid-arm mutation settles the
  old state to now, invalidates the generation, and never steals service
  from the exact arm (the terminal tick is unchanged vs the no-mutation
  baseline), while the injected physical HBM job keeps draining naturally
  (its completion fires once after the window was cancelled).

  Part 3 (real backends + the §17.3 host): a FluidScheduler (NoC link
  versions, M07) and an AnalyticalRemoteMemory port (remote-port versions)
  drive a MutationInvalidatingWindowHost that implements the fixed
  on_mutation flow: settle old state to now, record the partial window,
  invalidate, reject stale commits, re-plan from the authoritative
  remaining, and verify integer conservation after every step. Observers
  never re-enter the backends (§17.2 red line).

Honest scope note (reported as-is by the D1 coverage report): real die
templates are uniform per stage, so every execution-level window query is
exact and no approximate window arm ever commits from a real plan; the
approximate-arm cancel/rearm arithmetic is therefore covered at the operator
level (Part 1) while Parts 2/3 prove the invalidation linkage on real
resources with exact arms.

Build: cmake target JointPipelineMutationTest (full stack: AstraSim +
Analytical_Congestion_Aware). Exit code 0 on ALL PASS.
*/

#include "astra-sim/common/Logging.hh"
#include "astra-sim/system/Sys.hh"
#include "astra-sim/workload/HardwareResource.hh"
#include "astra-sim/workload/LocalHbmBandwidthModel.hh"
#include "astra-sim/workload/Workload.hh"
#include "astra-sim/workload/execution_driven/CompletionObserver.hh"
#include "astra-sim/workload/execution_driven/NodeStore.hh"
#include "astra-sim/workload/execution_driven/PipelineDescriptor.hh"
#include "astra-sim/workload/execution_driven/PipelineKernel.hh"
#include "DieModel.hh"
#include "congestion_aware/Link.h"
#include "congestion_aware/fluid/FluidScheduler.h"
#include <astra-network-analytical/common/EventQueue.h>
#include <remote_memory_backend/analytical/AnalyticalRemoteMemory.hh>

#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

using namespace AstraSim;
using namespace AstraSim::ExecutionDriven;
using namespace NetworkAnalytical;
using namespace NetworkAnalyticalCongestionAware;
using Analytical::AnalyticalRemoteMemory;

namespace {

bool g_ok = true;

void expect_true(const bool cond, const char* const what) {
    std::printf("%-76s -> %s\n", what, cond ? "PASS" : "FAIL");
    if (!cond) {
        g_ok = false;
    }
}

template <typename Fn>
bool throws_std_exception(Fn&& fn) {
    try {
        fn();
        return false;
    } catch (const std::exception&) {
        return true;
    }
}

nlohmann::json contract_wire(uint32_t window_limit = 8,
                             uint32_t error_ppm = 10000) {
    return nlohmann::json{
        {"projection_abi_version", 1u},
        {"descriptor_version", 1u},
        {"pipeline_protocol_version", 1u},
        {"model_version", PipelineKernelContract::kModelVersion},
        {"timer_whitelist_version",
         PipelineKernelContract::kTimerWhitelistVersion},
        {"window_tile_limit", window_limit},
        {"window_local_error_ppm", error_ppm},
        {"window_refine_margin_ppm", 20000u},
        {"prediction_config_sha256", std::string(64, 'a')}};
}

uint64_t span_of(const std::vector<uint64_t>& services) {
    uint64_t span = 0;
    for (const uint64_t service : services) {
        span += service;
    }
    return span;
}

// ---------------------------------------------------------------------------
// §17.3 mutation host: the fixed on_mutation flow over a real controller.
// ---------------------------------------------------------------------------

class MutationInvalidatingWindowHost final {
  public:
    MutationInvalidatingWindowHost(
        const PipelineKernelContract& contract, const uint32_t limit)
        : contract_(contract), controller_(limit) {}

    /// Arm one bounded window from the authoritative original services (the
    /// PE/VEC wave-frontier entry DieExecution uses).
    bool arm(const std::vector<uint64_t>& original_service_ticks) {
        plan_ = controller_.plan(contract_, controller_.current_limit(),
                                 original_service_ticks, {},
                                 span_of(original_service_ticks));
        external_start_ = 0;
        service_already_applied_ = 0;
        armed_ = plan_.accepted && !plan_.exact && plan_.selected_count > 1;
        return armed_;
    }

    /// Settle the OLD resource state to now using the frozen plan's
    /// piecewise prefix mapping (step 2 of the on_mutation flow).
    void settle_to(const uint64_t external_now) {
        if (!armed_) {
            return;
        }
        const uint64_t bounded =
            std::min<uint64_t>(external_now - external_start_,
                              external_total());
        const uint64_t target =
            PipelineKernel::original_service_at_external_tick(plan_, bounded);
        if (target < service_already_applied_) {
            throw std::runtime_error(
                "mutation host service mapping moved backwards");
        }
        service_already_applied_ = target;
    }

    /// The fixed §17.3 sequence: settle to now, account the delivered
    /// partial window, bump the generation, cancel only the queued plan.
    /// Returns the authoritative remaining original service.
    uint64_t on_mutation(const uint64_t external_now) {
        settle_to(external_now);
        if (armed_ && service_already_applied_ != 0) {
            controller_.record_partial(plan_);
        }
        controller_.invalidate();
        const uint64_t remaining =
            armed_ ? plan_.original_total_ticks - service_already_applied_ : 0;
        armed_ = false;
        ++cancelled_plans_;
        return remaining;
    }

    /// on_wakeup at the arm's endpoint: commits the window only when the
    /// full original service was delivered (stale generations were already
    /// rejected by the controller itself).
    void on_wakeup_complete() {
        if (!armed_) {
            return;
        }
        if (service_already_applied_ != plan_.original_total_ticks) {
            throw std::runtime_error(
                "mutation host wakeup reached its endpoint before the "
                "original service");
        }
        controller_.record(plan_, true);
        armed_ = false;
    }

    /// The exact K=1 fallback observation (DieExecution's exact-feedback
    /// path): valid service that never promotes the tier.
    void commit_exact_fallback() {
        if (!plan_.accepted || !plan_.exact) {
            throw std::runtime_error(
                "mutation host exact fallback requires an accepted exact "
                "plan");
        }
        controller_.record(plan_, true);
    }

    /// Committing a plan captured BEFORE an invalidation must fail (stale
    /// generation / tier); used to prove the stale-arm rejection.
    bool rejects_stale_commit(const PipelineWindowPlan& stale) {
        return throws_std_exception([&] { controller_.record(stale, true); });
    }
    bool rejects_stale_partial(const PipelineWindowPlan& stale) {
        return throws_std_exception(
            [&] { controller_.record_partial(stale); });
    }

    const PipelineWindowPlan& plan() const { return plan_; }
    PipelineWindowController& controller() { return controller_; }
    uint64_t service_already_applied() const {
        return service_already_applied_;
    }
    uint64_t cancelled_plans() const { return cancelled_plans_; }
    bool armed() const { return armed_; }

  private:
    uint64_t external_total() const {
        uint64_t total = 0;
        for (const uint64_t assigned : plan_.assigned_service_ticks) {
            total += assigned;
        }
        return total;
    }

    const PipelineKernelContract& contract_;
    PipelineWindowController controller_;
    PipelineWindowPlan plan_{};
    uint64_t external_start_ = 0;
    uint64_t service_already_applied_ = 0;
    uint64_t cancelled_plans_ = 0;
    bool armed_ = false;
};

/// Authoritative remaining original services at the wave frontier for an
/// interrupted window: fully assigned waves drop out, the straddling wave
/// keeps original - mapped_prefix, later waves are untouched.
std::vector<uint64_t> remaining_wave_services(
    const PipelineWindowPlan& plan, const uint64_t external_elapsed) {
    std::vector<uint64_t> remaining;
    uint64_t external_prefix = 0;
    for (std::size_t i = 0; i < plan.selected_count; ++i) {
        const uint64_t assigned = plan.assigned_service_ticks[i];
        const uint64_t original = plan.original_service_ticks[i];
        if (external_elapsed >= external_prefix + assigned) {
            // Wave fully covered by the published external prefix.
            external_prefix += assigned;
            continue;
        }
        if (external_elapsed > external_prefix) {
            // The straddling wave: keep its unmapped original remainder.
            const uint64_t local_external =
                external_elapsed - external_prefix;
            const uint64_t consumed =
                assigned == 0
                    ? original
                    : static_cast<uint64_t>(
                          (static_cast<unsigned __int128>(original) *
                           local_external) /
                          assigned);
            remaining.push_back(original - consumed);
        } else {
            // A future wave: fully untouched.
            remaining.push_back(original);
        }
        external_prefix += assigned;
    }
    return remaining;
}

// ---------------------------------------------------------------------------
// Part 1: value-level M13 mutation sequences.
// ---------------------------------------------------------------------------

void test_interrupt_carry_from_every_position() {
    const auto contract = PipelineKernelContract::parse(contract_wire());
    MutationInvalidatingWindowHost host(*contract, 8);
    expect_true(host.arm({9, 11, 10, 10}),
                "M13: (9,11,10,10) arms a K=4 averaged window");
    const PipelineWindowPlan& plan = host.plan();
    expect_true(plan.selected_count == 4 &&
                    plan.assigned_service_ticks ==
                        std::vector<uint64_t>({10, 10, 10, 10}) &&
                    plan.max_prefix_delta_ticks == 1,
                "M13: armed window averages to (10,10,10,10) with prefix "
                "bound 1");

    // Interrupt at every position class: first tick, mid slice, slice
    // boundaries, last tick before due, and the due tick itself.
    struct Row {
        uint64_t external;
        uint64_t expected_service;
    };
    const std::vector<Row> rows = {
        {0, 0},      {1, 0},      {5, 4},      {9, 8},       {10, 9},
        {11, 10},    {15, 14},    {20, 20},    {21, 21},     {30, 30},
        {39, 39},    {40, 40},
    };
    bool monotone_ok = true;
    bool carry_ok = true;
    bool remaining_vector_ok = true;
    uint64_t previous_service = 0;
    for (const Row& row : rows) {
        MutationInvalidatingWindowHost fresh(*contract, 8);
        fresh.arm({9, 11, 10, 10});
        fresh.settle_to(row.external);
        const uint64_t service = fresh.service_already_applied();
        if (service != row.expected_service) {
            std::printf(
                "    external=%llu service=%llu expected=%llu\n",
                static_cast<unsigned long long>(row.external),
                static_cast<unsigned long long>(service),
                static_cast<unsigned long long>(row.expected_service));
            carry_ok = false;
        }
        if (service < previous_service) {
            monotone_ok = false;
        }
        previous_service = service;
        // Remaining equals original_total - target and reconstructs exactly
        // from the wave frontier (§17.3 "剩余 = original - target 结转").
        const uint64_t remaining = plan.original_total_ticks - service;
        const std::vector<uint64_t> waves =
            remaining_wave_services(plan, row.external);
        if (span_of(waves) != remaining) {
            std::printf("    external=%llu wave-sum drift\n",
                        static_cast<unsigned long long>(row.external));
            remaining_vector_ok = false;
        }
    }
    expect_true(monotone_ok,
                "M13: service_already_applied is monotone across all "
                "interrupt positions");
    expect_true(carry_ok,
                "M13: every interrupt position maps to the exact original "
                "service prefix");
    expect_true(remaining_vector_ok,
                "M13: the carried remaining equals the wave-frontier "
                "original remainder at every position");

    // Successive interrupts on the SAME original arm keep the mapping
    // monotone (the already published external prefix is never rewritten).
    MutationInvalidatingWindowHost successive(*contract, 8);
    successive.arm({9, 11, 10, 10});
    bool successive_ok = true;
    uint64_t last = 0;
    for (const uint64_t external : {1U, 7U, 10U, 10U, 26U, 39U, 40U}) {
        successive.settle_to(external);
        if (successive.service_already_applied() < last) {
            successive_ok = false;
        }
        last = successive.service_already_applied();
    }
    expect_true(successive_ok && last == 40,
                "M13: repeated settles on one arm never move the service "
                "mapping backwards");

    // A completed wakeup commits exactly once at the full original service.
    MutationInvalidatingWindowHost completing(*contract, 8);
    completing.arm({9, 11, 10, 10});
    completing.settle_to(40);
    completing.on_wakeup_complete();
    expect_true(completing.controller().window_count() == 1,
                "M13: the completed averaged window counts exactly once");
}

void test_refinement_ladder_under_repeated_invalidation() {
    const auto contract = PipelineKernelContract::parse(contract_wire());
    MutationInvalidatingWindowHost host(*contract, 8);
    expect_true(host.controller().current_limit() == 4,
                "M13: the controller starts at min(4, configured 8)");

    // Eight stable K=4 windows promote the tier to 8.
    for (uint32_t i = 0;
         i < PipelineKernelContract::stable_window_count(); ++i) {
        host.arm({9, 11, 10, 10});
        host.settle_to(40);
        host.on_wakeup_complete();
    }
    expect_true(host.controller().current_limit() == 8,
                "M13: eight stable windows raise the tier to K=8");

    // Mutation invalidation clears the stable streak; the next arm is at
    // the fresh generation.
    host.arm({9, 11, 10, 10});
    const PipelineWindowPlan pre = host.plan();
    const uint64_t generation_before = host.controller().generation();
    const uint64_t remaining = host.on_mutation(15);
    expect_true(remaining == 40 - 14 && host.cancelled_plans() == 1 &&
                    host.controller().generation() ==
                        generation_before + 1 &&
                    host.controller().stable_windows() == 0,
                "M13: mid-window mutation carries the remaining, cancels "
                "only the plan, and bumps the generation");
    expect_true(host.rejects_stale_commit(pre) &&
                    host.rejects_stale_partial(pre),
                "M13: the pre-mutation plan is a stale generation for both "
                "commit paths");

    // 8 -> 4 -> 2 -> 1 continuous shrink: each tier's vector refines one
    // level further under a forcing floor, and every accepted level keeps
    // the integer conservation sum(d') == sum(d).
    const std::vector<std::vector<uint64_t>> ladder_vectors = {
        {9, 11, 10, 10, 9, 11, 10, 10},   // accepts at K=8
        {9, 11, 10, 10},                  // accepts at K=4
        {9, 11},                          // accepts at K=2
        {9},                              // exact at K=1
    };
    const std::vector<uint32_t> expected_tiers = {8, 4, 2, 1};
    bool ladder_ok = true;
    bool conservation_ok = true;
    for (std::size_t step = 0; step < ladder_vectors.size(); ++step) {
        host.arm(ladder_vectors[step]);
        const PipelineWindowPlan& plan = host.plan();
        if (!plan.accepted ||
            plan.selected_count != expected_tiers[step]) {
            std::printf("    ladder step %zu: selected=%u\n", step,
                        plan.selected_count);
            ladder_ok = false;
        }
        uint64_t assigned_total = 0;
        for (const uint64_t assigned : plan.assigned_service_ticks) {
            assigned_total += assigned;
        }
        if (assigned_total != plan.original_total_ticks) {
            conservation_ok = false;
        }
        if (host.armed()) {
            host.settle_to(plan.original_total_ticks);
            host.on_wakeup_complete();
        } else {
            // The final rung is the exact K=1 fallback observation.
            host.commit_exact_fallback();
        }
        if (host.controller().current_limit() != expected_tiers[step]) {
            std::printf("    ladder step %zu: tier=%u expected=%u\n", step,
                        host.controller().current_limit(),
                        expected_tiers[step]);
            ladder_ok = false;
        }
        host.controller().invalidate();  // resource change cuts the streak
    }
    expect_true(ladder_ok,
                "M13: successive windows shrink the tier 8->4->2->1");
    expect_true(conservation_ok,
                "M13: every rung of the ladder conserves sum(d')==sum(d)");

    // A hardware-floor vector refines 4 -> 2 -> 1 and fails closed when
    // even the exact slice is below its floor.
    const auto floor_refined = PipelineKernel::plan_window(
        *contract, 8, 4, {20, 5, 10, 10}, {20, 0, 0, 0}, 45);
    expect_true(floor_refined.accepted && floor_refined.exact &&
                    floor_refined.selected_count == 1 &&
                    floor_refined.refinement_count == 2,
                "M13: a forcing hardware floor refines to the exact K=1 "
                "prefix");
    const auto impossible = PipelineKernel::plan_window(
        *contract, 8, 4, {2, 18, 10, 10}, {12, 0, 0, 0}, 40);
    expect_true(!impossible.accepted && impossible.unsupported_reason ==
                    "hardware_minimum_exceeds_exact_service",
                "M13: an exact slice below its declared floor fails closed "
                "(no partial cost)");
}

void test_integer_remainder_without_rounding_drift() {
    const auto contract = PipelineKernelContract::parse(contract_wire());

    // Quotient/remainder rule: remainder ticks go to the earliest slices.
    MutationInvalidatingWindowHost qr(*contract, 8);
    expect_true(qr.arm({2, 3, 4, 6}),
                "M13: (2,3,4,6) arms with a non-divisible total");
    expect_true(qr.plan().selected_count == 2 &&
                    qr.plan().assigned_service_ticks ==
                        std::vector<uint64_t>({3, 2}),
                "M13: honest T_local refines (2,3,4,6) to the K=2 (3,2) "
                "assignment");
    bool qr_ok = true;
    uint64_t carried = 5;  // the K=2 window covers only {2,3}
    for (const uint64_t external : {1U, 3U, 4U, 5U}) {
        MutationInvalidatingWindowHost chain(*contract, 8);
        chain.arm({2, 3, 4, 6});
        chain.settle_to(external);
        const uint64_t remaining = chain.on_mutation(external);
        if (chain.service_already_applied() + remaining != 5) {
            qr_ok = false;
        }
        carried = remaining;
    }
    expect_true(qr_ok && carried == 0,
                "M13: remainder windows carry exact integers with no "
                "rounding drift (published + remaining == window total)");
    MutationInvalidatingWindowHost replan(*contract, 8);
    expect_true(replan.arm({4, 6}) &&
                    replan.plan().original_total_ticks == 10,
                "M13: the carried remaining {4,6} re-arms as its own "
                "conserving window");
    replan.settle_to(10);
    replan.on_wakeup_complete();
    expect_true(replan.controller().window_count() == 1,
                "M13: the re-armed remainder completes and counts once");

    // A remainder of exactly one tick on a larger vector.
    MutationInvalidatingWindowHost one_tick(*contract, 8);
    expect_true(one_tick.arm({10, 10, 10, 11}),
                "M13: total 41 arms with remainder 1");
    expect_true(one_tick.plan().assigned_service_ticks ==
                    std::vector<uint64_t>({11, 10, 10, 10}),
                "M13: the single remainder tick lands on the earliest "
                "slice");
    one_tick.settle_to(11);
    expect_true(one_tick.service_already_applied() == 10,
                "M13: the first slice boundary maps to exactly its "
                "original service");
    const uint64_t remaining = one_tick.on_mutation(11);
    const std::vector<uint64_t> waves =
        remaining_wave_services(one_tick.plan(), 11);
    expect_true(remaining == 31 && span_of(waves) == 31 &&
                    waves == std::vector<uint64_t>({10, 10, 11}),
                "M13: the interrupted remainder window keeps the untouched "
                "waves whole");
}

void test_cancel_plan_never_cancels_physical_service() {
    const auto contract = PipelineKernelContract::parse(contract_wire());
    MutationInvalidatingWindowHost host(*contract, 8);
    host.arm({9, 11, 10, 10});
    const PipelineWindowPlan pre = host.plan();

    // Cancel at external 15: the plan is cancelled but the already
    // published external prefix is NOT reclaimed -- the frozen mapping
    // still returns the delivered original service for the elapsed ticks.
    const uint64_t remaining = host.on_mutation(15);
    expect_true(!host.armed() && remaining == 26 &&
                    host.cancelled_plans() == 1,
                "M13: cancelling the window drops only the queued plan");
    const uint64_t published =
        PipelineKernel::original_service_at_external_tick(pre, 15);
    expect_true(published == 14 && published + remaining == 40,
                "M13: the published prefix survives cancellation and "
                "prefix + remaining == original total");

    // The stale plan cannot re-enter either record path (dual-return
    // semantics: the cancelled plan reports cancelled, never a physical
    // flow cancellation).
    expect_true(host.rejects_stale_commit(pre) &&
                    host.rejects_stale_partial(pre),
                "M13: a cancelled plan is rejected on every commit path");

    // A fresh arm at the new generation still plans and conserves.
    expect_true(host.arm({9, 11}) &&
                    host.controller().generation() >= 1,
                "M13: re-arming at the new generation plans the carried "
                "remaining waves");
}

void test_prediction_world_frozen_snapshot_recompute() {
    const auto contract = PipelineKernelContract::parse(contract_wire());
    const std::vector<uint64_t> frozen = {0, 10, 0, 10};

    // The approximation fails the prefix budget at K=4 and K=2 and must be
    // discarded -- refined all the way to the exact K=1 prefix.
    const auto approx = PipelineKernel::plan_window(*contract, 8, 4, frozen,
                                                    {}, 20);
    expect_true(approx.accepted && approx.exact &&
                    approx.selected_count == 1 &&
                    approx.refinement_count == 2,
                "M13: the failing approximation refines to the exact "
                "prefix");

    // Exact recompute from the SAME frozen snapshot: deterministic, the
    // frozen vector is untouched, and repeating the failed approximation
    // still fails identically (no state pollution between attempts).
    const auto exact_first =
        PipelineKernel::plan_window(*contract, 1, 1, frozen, {}, 20);
    const auto exact_second =
        PipelineKernel::plan_window(*contract, 1, 1, frozen, {}, 20);
    const auto approx_retry = PipelineKernel::plan_window(
        *contract, 8, 4, frozen, {}, 20);
    expect_true(exact_first.accepted && exact_first.exact &&
                    exact_first.original_service_ticks ==
                        exact_second.original_service_ticks &&
                    exact_first.original_service_ticks ==
                        std::vector<uint64_t>({0}) &&
                    approx_retry.refinement_count ==
                        approx.refinement_count,
                "M13: the frozen snapshot recomputes exactly and the "
                "discarded attempt never pollutes it");

    // A hard conservation error (exact slice below its floor) still stops
    // the run instead of refining into a partially valid cost.
    expect_true(throws_std_exception([&] {
        (void)PipelineKernel::plan_window(*contract, 3, 4,
                                          {4, 4, 4, 4}, {}, 16);
    }), "M13: an unsupported effective K is rejected before any planning");
    const auto hard_error = PipelineKernel::plan_window(
        *contract, 8, 4, {2, 18, 10, 10}, {12, 0, 0, 0}, 40);
    expect_true(!hard_error.accepted &&
                    hard_error.unsupported_reason ==
                        "hardware_minimum_exceeds_exact_service",
                "M13: a hard error fails closed with no partial cost");
}

// ---------------------------------------------------------------------------
// Part 2 fixture: hermetic Sys over the real EventQueue + HBM model.
// ---------------------------------------------------------------------------

std::map<uint64_t, uint64_t> g_terminal_fires;
std::map<uint64_t, uint64_t> g_first_tick;

void tick_hook(void* /*ctx*/, int /*rank*/, uint64_t node_id,
               const char* /*request_id*/, const char* /*stage*/,
               uint64_t /*generation*/, uint64_t tick, int status) {
    if (status != static_cast<int>(NodeTerminalStatus::Success)) {
        g_ok = false;
    }
    if (g_first_tick.find(node_id) == g_first_tick.end()) {
        g_first_tick[node_id] = tick;
    }
    ++g_terminal_fires[node_id];
}

class TestRemoteMemoryApi final : public AstraSim::AstraRemoteMemoryAPI {
  public:
    void set_sys(int, AstraSim::Sys*) override {}
    void issue(uint64_t, AstraSim::WorkloadLayerHandlerData*) override {}
};

class FakeQueueNetworkApi final : public AstraNetworkAPI {
  public:
    explicit FakeQueueNetworkApi(std::shared_ptr<EventQueue> queue)
        : AstraNetworkAPI(0), queue_(std::move(queue)) {}

    int sim_send(void*, uint64_t, int, int, int, sim_request*,
                 void (*)(void*), void*) override {
        return 0;
    }

    int sim_recv(void*, uint64_t, int, int, int, sim_request*,
                 void (*)(void*), void*) override {
        return 0;
    }

    void sim_schedule(const timespec_t delta, void (*fun_ptr)(void*),
                      void* fun_arg) override {
        const timespec_t now = sim_get_time();
        queue_->schedule_event(now.time_val + delta.time_val, fun_ptr,
                               fun_arg);
    }

    timespec_t sim_get_time() override {
        timespec_t now;
        now.time_res = NS;
        now.time_val = queue_->get_current_time();
        return now;
    }

  private:
    std::shared_ptr<EventQueue> queue_;
};

const char* kDieNodeJson = R"({
  "schema_version": 1,
  "profile": "mesh_die_v1",
  "die": {
    "sram-mb-per-core": 0.75,
    "core-type": "large",
    "compute-area-mm2": 800,
    "hbm-per-side": 4,
    "hbm-sides": 1,
    "noc-width-bits": 512
  }
})";

void write_text(const std::string& path, const std::string& content) {
    std::ofstream out(path);
    if (!out.is_open()) {
        std::fprintf(stderr,
                     "[pipeline_mutation_test] cannot write %s\n",
                     path.c_str());
        std::exit(EXIT_FAILURE);
    }
    out << content;
}

/// A LocalRegion spec is the zero-work identity region: its compiled plan
/// is exactly one launch TIMER stage with layer_count*weight_passes uniform
/// waves of launch_us_per_macro_group (1 us -> 1000 ns each). No memory
/// leg, no HBM job of its own: every HBM mutation in Part 2 comes from the
/// test injection alone.
nlohmann::json launch_region_spec_json(uint64_t layers) {
    nlohmann::json span;
    span["request_id"] = "r1";
    span["q_tokens"] = 1u;
    span["kv_tokens"] = 1u;
    span["q_start_pos"] = 0u;
    span["kv_start_pos"] = 0u;
    nlohmann::json spec;
    spec["schema_version"] = 1u;
    spec["op_kind"] = "local_region";
    spec["phase"] = "prefill";
    spec["dtype"] = "bf16";
    spec["model_dims"] = {{"hidden", 4096u},
                          {"ffn", 14336u},
                          {"vocab", 152064u},
                          {"mlp_variant", "swiglu"},
                          {"norm_type", "rmsnorm"},
                          {"position_encoding", "rope"},
                          {"num_q_heads", 32u},
                          {"num_kv_heads", 8u},
                          {"head_dim", 128u}};
    spec["tp_slice"] = {{"tensor_parallel", 1u},
                        {"tensor_parallel_rank", 0u},
                        {"q_heads_local", 32u},
                        {"kv_heads_local", 8u},
                        {"ffn_local", 14336u},
                        {"vocab_local", 152064u}};
    spec["spans"] = nlohmann::json::array({span});
    spec["layer_count"] = layers;
    spec["weight_passes"] = 1u;
    spec["kv_access"] = {{"identity", "local"}};
    spec["fusion_policy"] = "auto";
    return spec;
}

OnlineNode make_launch_region_node(uint64_t id, uint64_t layers) {
    OnlineNode node;
    node.global_id = id;
    node.rank = 0;
    node.kind = NodeKind::Compute;
    node.node_type = 4;  // ChakraProtoMsg::COMP_NODE
    node.name = "local_region";
    node.request_id = "r1";
    node.stage = "prefill";
    node.generation = 0;
    node.compute.die_spec =
        DieModel::parse_die_op_spec(launch_region_spec_json(layers));
    return node;
}

struct SysFixture {
    std::shared_ptr<EventQueue> event_queue;
    std::unique_ptr<FakeQueueNetworkApi> network_api;
    TestRemoteMemoryApi remote_memory;
    std::unique_ptr<Sys> sys;
    NodeStoreGraphSource* source = nullptr;
};

SysFixture make_sys_fixture(const std::string& config_dir,
                            const std::string& system_file,
                            const uint32_t window_limit = 8) {
    SysFixture fixture;
    fixture.event_queue = std::make_shared<EventQueue>();
    fixture.network_api =
        std::make_unique<FakeQueueNetworkApi>(fixture.event_queue);
    auto source = std::make_shared<NodeStoreGraphSource>();
    fixture.source = source.get();
    fixture.sys = std::make_unique<Sys>(
        0, config_dir + "/workload", config_dir + "/comm_group.json",
        config_dir + "/" + system_file, &fixture.remote_memory,
        fixture.network_api.get(), std::vector<int>{1}, std::vector<int>{1},
        1.0, 1.0, false, ExecutionMode::Online, source);
    fixture.sys->workload->set_pipeline_kernel_contract(
        PipelineKernelContract::parse(contract_wire(window_limit)));
    return fixture;
}

void drain_queue(const std::shared_ptr<EventQueue>& queue) {
    while (!queue->finished()) {
        queue->proceed();
    }
}

struct InjectionContext {
    Workload* workload = nullptr;
    LocalHbmBandwidthModel* model = nullptr;
    uint64_t node_id = 0;
    uint64_t bytes = 0;
    uint64_t transfers_completed = 0;
    bool has_active_at_injection = false;
    uint64_t terminals_at_injection = 0;
    uint64_t window_generation_after_mutation = 0;
    bool sampled_generation = false;
};

void inject_hbm_comm_read(void* raw) {
    std::unique_ptr<std::pair<InjectionContext*,
                              std::shared_ptr<void>>> guard(
        static_cast<std::pair<InjectionContext*, std::shared_ptr<void>>*>(
            raw));
    InjectionContext* context = guard->first;
    // Ordering discriminator, captured BEFORE the mutation pair: the die
    // execution state at the moment the same-tick injection runs.
    context->has_active_at_injection =
        context->workload->die_model_adapter()
            ->has_active_die_executions();
    context->terminals_at_injection =
        g_terminal_fires.count(context->node_id)
            ? g_terminal_fires[context->node_id]
            : 0;
    // The real rank mutation: an independent COMM_READ transfer joins the
    // HBM bus. issue_transfer publishes the before/after observer pair.
    InjectionContext* unowned = context;
    context->model->issue_transfer(
        LocalHbmBandwidthModel::JobKind::COMM_READ, context->bytes, 0,
        0x7fffffff, [unowned]() { ++unowned->transfers_completed; });
    // The window generation AFTER the mutation: the active execution's
    // controller must have been invalidated (settle + generation bump).
    const nlohmann::json quality =
        context->workload->die_model_adapter()->window_quality_json();
    if (quality.contains("active_executions") &&
        quality.at("active_executions").is_array() &&
        !quality.at("active_executions").empty()) {
        context->window_generation_after_mutation =
            quality.at("active_executions")
                .at(0)
                .at("quality")
                .at("window_generation")
                .get<uint64_t>();
        context->sampled_generation = true;
    }
}

// ---------------------------------------------------------------------------
// Part 2: DieExecution × HBM mutation.
// ---------------------------------------------------------------------------

void test_hbm_mutation_mid_arm_settles_and_keeps_physical_callback() {
    g_terminal_fires.clear();
    g_first_tick.clear();
    const std::string config_dir = "/tmp/pipeline_mutation_test_hbm";
    std::error_code ec;
    std::filesystem::remove_all(config_dir, ec);
    std::filesystem::create_directories(config_dir, ec);
    write_text(config_dir + "/comm_group.json", "{}");
    write_text(config_dir + "/resolved_die_model.json", kDieNodeJson);
    // No hbm-bandwidth-contention key: under the die backend the derived
    // local-mem-bw fill restores the contention default (an explicit key
    // with no local-mem-bw trips the auto-disable and skips the restore).
    write_text(config_dir + "/die_contention.json", R"({
  "scheduling-policy": "FIFO",
  "compute-cost-model": "die",
  "local-mem-latency": 0
})");

    constexpr uint64_t kWaves = 4;
    // launch_us_per_macro_group == 1.0 us and launch waves are slot-free:
    // every wave ends at tick 1000, so the coalesced exact arm is due at
    // 1000 regardless of the wave count.
    constexpr uint64_t kDueTick = 1000;

    // Baseline: no mutation, the coalesced exact arm completes at 4000.
    {
        SysFixture fixture =
            make_sys_fixture(config_dir, "die_contention.json");
        Workload& workload = *fixture.sys->workload;
        fixture.source->store().add_node(make_launch_region_node(31, kWaves));
        workload.issue_dep_free_nodes();
        drain_queue(fixture.event_queue);
        expect_true(g_terminal_fires.at(31) == 1 &&
                        g_first_tick.at(31) == kDueTick,
                    "HBM: baseline exact arm completes at the coalesced "
                    "stretch tick");
    }

    // Mid-arm mutation at 1500: settle + invalidate + keep the callback,
    // the terminal stays at 4000, and the injected physical job drains.
    {
        g_terminal_fires.clear();
        g_first_tick.clear();
        SysFixture fixture =
            make_sys_fixture(config_dir, "die_contention.json");
        Workload& workload = *fixture.sys->workload;
        InjectionContext context;
        context.workload = &workload;
        context.model = workload.local_hbm_bandwidth_model.get();
        context.node_id = 32;
        context.bytes = 1000;
        fixture.source->store().add_node(make_launch_region_node(32, kWaves));
        workload.issue_dep_free_nodes();
        auto* payload =
            new std::pair<InjectionContext*, std::shared_ptr<void>>(
                &context, std::shared_ptr<void>());
        fixture.event_queue->schedule_event(500, inject_hbm_comm_read,
                                            payload);
        drain_queue(fixture.event_queue);
        expect_true(g_terminal_fires.at(32) == 1 &&
                        g_first_tick.at(32) == kDueTick,
                    "HBM: mid-arm mutation never steals service from the "
                    "exact arm (terminal unchanged)");
        expect_true(context.has_active_at_injection &&
                        context.sampled_generation &&
                        context.window_generation_after_mutation >= 1,
                    "HBM: the real rank mutation invalidated the window "
                    "generation mid-arm");
        expect_true(context.transfers_completed == 1,
                    "HBM: the injected physical HBM job drained naturally "
                    "after its window was cancelled");
        const nlohmann::json quality =
            workload.die_model_adapter()->window_quality_json();
        expect_true(quality.at("completed_execution_count") == 1 &&
                        quality.at("window_count") == 0 &&
                        quality.at("fully_exact") == true,
                    "HBM: the uniform launch template honestly stays exact "
                    "(no fake averaged window)");
    }

    std::filesystem::remove_all(config_dir, ec);
}

void test_m10_due_tick_both_insertion_orders() {
    g_terminal_fires.clear();
    g_first_tick.clear();
    const std::string config_dir = "/tmp/pipeline_mutation_test_m10";
    std::error_code ec;
    std::filesystem::remove_all(config_dir, ec);
    std::filesystem::create_directories(config_dir, ec);
    write_text(config_dir + "/comm_group.json", "{}");
    write_text(config_dir + "/resolved_die_model.json", kDieNodeJson);
    // No hbm-bandwidth-contention key: under the die backend the derived
    // local-mem-bw fill restores the contention default (an explicit key
    // with no local-mem-bw trips the auto-disable and skips the restore).
    write_text(config_dir + "/die_contention.json", R"({
  "scheduling-policy": "FIFO",
  "compute-cost-model": "die",
  "local-mem-latency": 0
})");

    constexpr uint64_t kWaves = 4;
    constexpr uint64_t kDueTick = 1000;  // launch waves all end at 1 us

    // Order A (mutation physically first at the due tick): the injection
    // event is scheduled BEFORE the node issue, so it sits ahead of the
    // Sys alarm for tick 4000 in the queue's insertion order.
    {
        g_terminal_fires.clear();
        g_first_tick.clear();
        SysFixture fixture =
            make_sys_fixture(config_dir, "die_contention.json");
        Workload& workload = *fixture.sys->workload;
        InjectionContext context;
        context.workload = &workload;
        context.model = workload.local_hbm_bandwidth_model.get();
        context.node_id = 33;
        context.bytes = 1000;
        auto* payload =
            new std::pair<InjectionContext*, std::shared_ptr<void>>(
                &context, std::shared_ptr<void>());
        fixture.event_queue->schedule_event(kDueTick, inject_hbm_comm_read,
                                            payload);
        fixture.source->store().add_node(make_launch_region_node(33, kWaves));
        workload.issue_dep_free_nodes();
        drain_queue(fixture.event_queue);
        expect_true(context.has_active_at_injection &&
                        context.terminals_at_injection == 0,
                    "M10-A: the due-tick mutation ran before the physical "
                    "timer callback (execution still active, no terminal)");
        expect_true(context.sampled_generation &&
                        context.window_generation_after_mutation >= 1,
                    "M10-A: the due cookie was settled and invalidated, not "
                    "prepublished");
        expect_true(g_terminal_fires.at(33) == 1 &&
                        g_first_tick.at(33) == kDueTick,
                    "M10-A: the kept cookie published exactly once at its "
                    "original callback position");
        expect_true(context.transfers_completed == 1,
                    "M10-A: the mutation's physical job drained naturally");
    }

    // Order B (callback physically first): the injection is scheduled
    // AFTER the node issue, i.e. behind the Sys alarm for tick 4000.
    {
        g_terminal_fires.clear();
        g_first_tick.clear();
        SysFixture fixture =
            make_sys_fixture(config_dir, "die_contention.json");
        Workload& workload = *fixture.sys->workload;
        InjectionContext context;
        context.workload = &workload;
        context.model = workload.local_hbm_bandwidth_model.get();
        context.node_id = 34;
        context.bytes = 1000;
        fixture.source->store().add_node(make_launch_region_node(34, kWaves));
        workload.issue_dep_free_nodes();
        auto* payload =
            new std::pair<InjectionContext*, std::shared_ptr<void>>(
                &context, std::shared_ptr<void>());
        fixture.event_queue->schedule_event(kDueTick, inject_hbm_comm_read,
                                            payload);
        drain_queue(fixture.event_queue);
        expect_true(!context.has_active_at_injection &&
                        context.terminals_at_injection == 1,
                    "M10-B: the physical timer callback ran before the "
                    "same-tick mutation (execution retired, terminal "
                    "already fired)");
        expect_true(g_terminal_fires.at(34) == 1 &&
                        g_first_tick.at(34) == kDueTick,
                    "M10-B: the callback-first order completes exactly once "
                    "at the same tick");
        expect_true(context.transfers_completed == 1,
                    "M10-B: the later mutation still drains its physical "
                    "job naturally");
    }

    std::filesystem::remove_all(config_dir, ec);
}

// ---------------------------------------------------------------------------
// Part 3: backend resource versions driving the §17.3 host.
// ---------------------------------------------------------------------------

void no_op_callback(void*) {}

void test_noc_link_observer_drives_mutation_host() {
    const auto contract = PipelineKernelContract::parse(contract_wire());
    auto event_queue = std::make_shared<EventQueue>();
    std::vector<std::shared_ptr<const Link>> links;
    links.push_back(std::make_shared<Link>(0, 1.0, 0.0));
    links.push_back(std::make_shared<Link>(1, 0.5, 0.0));
    FluidScheduler scheduler(event_queue, std::move(links), 8, 16, 1024);

    auto shared_route = std::make_shared<FluidRoute>();
    shared_route->link_ids = {0, 1};
    shared_route->propagation_latency_ns = 7;

    MutationInvalidatingWindowHost host(*contract, 8);
    expect_true(host.arm({9, 11, 10, 10}),
                "NoC: the host arms a K=4 averaged window before any "
                "traffic");

    struct ObserverLog {
        uint64_t before_events = 0;
        uint64_t after_events = 0;
        std::vector<uint64_t> versions_at_before;
        std::vector<uint64_t> versions_at_after;
    } log;

    const uint64_t observer_id = scheduler.add_link_resource_mutation_observer(
        [&scheduler, &host, &log](const bool before_change,
                                  const EventTime tick,
                                  const std::vector<LinkId>& /*links*/) {
            if (before_change) {
                ++log.before_events;
                log.versions_at_before.push_back(
                    scheduler.link_resource_version(0));
                // §17.3 fixed flow on the OLD state: settle to now, carry
                // the remaining, invalidate. No scheduler re-entry here.
                if (host.armed()) {
                    host.on_mutation(tick);
                } else {
                    host.controller().invalidate();
                }
            } else {
                ++log.after_events;
                log.versions_at_after.push_back(
                    scheduler.link_resource_version(0));
            }
        });

    // Short background flow starts first: one membership change on each
    // traversed link -> version 1, one before/after pair.
    scheduler.start_flow(10, shared_route, no_op_callback, nullptr);
    scheduler.flush_pending_starts();
    expect_true(log.before_events == 1 && log.after_events == 1 &&
                    log.versions_at_after.back() == 1,
                "NoC: the first membership change publishes exactly one "
                "version bump");
    expect_true(!host.armed() && host.cancelled_plans() == 1,
                "NoC: the before callback cancelled the armed window plan");
    // Re-arm the full carried vector at the new generation (nothing had
    // been served at tick 0, so the authoritative remaining is the whole
    // original window).
    expect_true(host.arm({9, 11, 10, 10}) &&
                    host.controller().generation() >= 1,
                "NoC: the host re-armed the carried remaining at the new "
                "generation");

    // A second flow joins the same links mid-flight: another real mutation.
    scheduler.mark_event_loop_started();
    struct DelayedJoin {
        FluidScheduler* scheduler;
        std::shared_ptr<FluidRoute> route;
    };
    auto* join = new DelayedJoin{&scheduler, shared_route};
    event_queue->schedule_event(
        2,
        [](void* raw) {
            std::unique_ptr<DelayedJoin> delayed(
                static_cast<DelayedJoin*>(raw));
            delayed->scheduler->start_flow(6, delayed->route, no_op_callback,
                                           nullptr);
        },
        join);
    while (!event_queue->finished()) {
        event_queue->proceed();
    }

    expect_true(log.before_events >= 2 && log.after_events == log.before_events,
                "NoC: every real membership/rate change pairs one before "
                "with one after event");
    bool versions_advance = true;
    for (std::size_t i = 1; i < log.versions_at_after.size(); ++i) {
        if (log.versions_at_after[i] <= log.versions_at_after[i - 1]) {
            versions_advance = false;
        }
    }
    expect_true(versions_advance &&
                    !log.versions_at_after.empty() &&
                    log.versions_at_after.front() == 1,
                "NoC: link versions strictly advance across joins and "
                "completions");
    expect_true(scheduler.get_active_flow_count() == 0 &&
                    scheduler.get_total_completed_flows() == 2,
                "NoC: both real flows completed naturally after their "
                "windows were cancelled (no flow cancel)");
    // The physical completions (tail arrivals included) never re-entered
    // the host: cancelled plans stay cancelled and the controller keeps
    // its last generation unless a real mutation arrives.
    const uint64_t generation_after_drain = host.controller().generation();
    expect_true(generation_after_drain >= log.before_events,
                "NoC: the host generation advanced once per real mutation, "
                "never from a completion callback alone");
    scheduler.remove_link_resource_mutation_observer(observer_id);
}

void test_remote_port_observer_drives_mutation_host() {
    const std::string config_dir = "/tmp/pipeline_mutation_test_port";
    std::error_code ec;
    std::filesystem::remove_all(config_dir, ec);
    std::filesystem::create_directories(config_dir, ec);
    write_text(config_dir + "/comm_group.json", "{}");
    write_text(config_dir + "/resolved_die_model.json", kDieNodeJson);
    write_text(config_dir + "/remote_memory.json", R"({
  "memory-type": "PER_NPU_MEMORY_EXPANSION",
  "npu-ids": [0],
  "remote-mem-bw": 6.0,
  "remote-mem-latency": 100
})");
    // No hbm-bandwidth-contention key: under the die backend the derived
    // local-mem-bw fill restores the contention default (an explicit key
    // with no local-mem-bw trips the auto-disable and skips the restore).
    write_text(config_dir + "/die_contention.json", R"({
  "scheduling-policy": "FIFO",
  "compute-cost-model": "die",
  "local-mem-latency": 0
})");

    const auto contract = PipelineKernelContract::parse(contract_wire());
    AnalyticalRemoteMemory remote_memory(config_dir + "/remote_memory.json");

    g_terminal_fires.clear();
    g_first_tick.clear();
    SysFixture fixture =
        make_sys_fixture(config_dir, "die_contention.json");
    // Bind the real port backend as this Sys' remote memory (the Sys ctor
    // already called set_sys on the placeholder API; bind the real one and
    // register it for the rank explicitly).
    fixture.sys->remote_mem = &remote_memory;
    remote_memory.set_sys(0, fixture.sys.get());
    Workload& workload = *fixture.sys->workload;

    MutationInvalidatingWindowHost host(*contract, 8);
    expect_true(host.arm({9, 11, 10, 10}),
                "port: the host arms a K=4 averaged window before any "
                "port traffic");
    const PipelineWindowPlan stale = host.plan();

    struct PortLog {
        uint64_t before_events = 0;
        uint64_t after_events = 0;
        std::vector<uint64_t> versions;
    } log;
    const uint64_t observer_id =
        remote_memory.add_port_resource_mutation_observer(
            [&remote_memory, &host, &log](const bool before_change,
                                          std::size_t /*port_index*/,
                                          double /*time_ns*/,
                                          uint64_t version) {
                if (before_change) {
                    ++log.before_events;
                    // §17.3: settle the old state, cancel the queued plan.
                    if (host.armed()) {
                        host.on_mutation(0);
                    } else {
                        host.controller().invalidate();
                    }
                } else {
                    ++log.after_events;
                    log.versions.push_back(version);
                }
            });

    // Two real MEM_LOAD nodes on rank 0: their port transactions are the
    // real mutations (insert + streaming-flip + completion settlement).
    OnlineNode mem_a;
    mem_a.global_id = 41;
    mem_a.rank = 0;
    mem_a.kind = NodeKind::MemLoad;
    mem_a.node_type = 2;  // ChakraProtoMsg::MEM_LOAD_NODE
    mem_a.name = "mem_a";
    mem_a.request_id = "r1";
    mem_a.stage = "prefill";
    mem_a.mem.tensor_size = 600;
    OnlineNode mem_b = mem_a;
    mem_b.global_id = 42;
    mem_b.name = "mem_b";
    fixture.source->store().add_node(mem_a);
    fixture.source->store().add_node(mem_b);
    workload.issue_dep_free_nodes();
    drain_queue(fixture.event_queue);

    expect_true(log.before_events >= 2 &&
                    log.after_events == log.before_events &&
                    !log.versions.empty() && log.versions.front() == 1,
                "port: every real port membership change paired a "
                "before/after event with an advancing version");
    expect_true(host.rejects_stale_commit(stale) &&
                    host.rejects_stale_partial(stale),
                "port: the pre-mutation window is stale on both commit "
                "paths after the real port mutation");
    expect_true(g_terminal_fires.at(41) == 1 &&
                    g_terminal_fires.at(42) == 1,
                "port: both physical MEM transactions completed exactly "
                "once after their windows were cancelled");
    expect_true(remote_memory.is_drained(),
                "port: the backend drained fully (no fabricated "
                "cancellations)");
    remote_memory.remove_port_resource_mutation_observer(observer_id);

    std::filesystem::remove_all(config_dir, ec);
}

}  // namespace

int main() {
    setvbuf(stdout, nullptr, _IONBF, 0);
    CompletionObserver::instance().set_hook(tick_hook, nullptr);

    test_interrupt_carry_from_every_position();
    test_refinement_ladder_under_repeated_invalidation();
    test_integer_remainder_without_rounding_drift();
    test_cancel_plan_never_cancels_physical_service();
    test_prediction_world_frozen_snapshot_recompute();
    test_hbm_mutation_mid_arm_settles_and_keeps_physical_callback();
    test_m10_due_tick_both_insertion_orders();
    test_noc_link_observer_drives_mutation_host();
    test_remote_port_observer_drives_mutation_host();

    if (!g_ok) {
        std::fprintf(stderr, "[pipeline_mutation_test] FAIL (see above)\n");
        return 1;
    }
    std::printf("[pipeline_mutation_test] ALL PASS\n");
    return 0;
}
