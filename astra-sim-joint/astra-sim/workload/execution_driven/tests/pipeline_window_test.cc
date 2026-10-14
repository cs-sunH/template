#include "astra-sim/workload/execution_driven/PipelineKernel.hh"

#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using AstraSim::ExecutionDriven::PipelineKernel;
using AstraSim::ExecutionDriven::PipelineKernelContract;
using AstraSim::ExecutionDriven::PipelineWindowController;
using AstraSim::ExecutionDriven::PipelineWindowPlan;

namespace {

int failures = 0;

void expect(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "[pipeline_window_test] FAIL: " << message << '\n';
        ++failures;
    }
}

template <typename Fn>
void expect_throw(Fn&& fn, const std::string& message) {
    try {
        fn();
        expect(false, message + " (did not throw)");
    } catch (const std::exception&) {
    }
}

nlohmann::json contract_wire(uint32_t tile_limit = 8,
                            uint32_t error_ppm = 10000) {
    return nlohmann::json{
        {"projection_abi_version", 1}, {"descriptor_version", 1},
        {"pipeline_protocol_version", 1},
        {"model_version", PipelineKernelContract::kModelVersion},
        {"timer_whitelist_version", PipelineKernelContract::kTimerWhitelistVersion},
        {"window_tile_limit", tile_limit}, {"window_local_error_ppm", error_ppm},
        {"window_refine_margin_ppm", 20000},
        {"prediction_config_sha256", std::string(64, 'b')}};
}

void test_tunable_levels_and_integer_average() {
    const auto contract = PipelineKernelContract::parse(contract_wire());
    for (const uint32_t k : {1U, 2U, 4U, 8U}) {
        const std::vector<uint64_t> service(k, 10);
        const auto plan = PipelineKernel::plan_window(
            *contract, k, k, service, {}, 10 * k);
        expect(plan.accepted && plan.selected_count == k && plan.exact,
               "K=" + std::to_string(k) + " supports exact constant service");
    }

    const auto bound = PipelineKernel::plan_window(
        *contract, 4, 4, {9, 11, 10, 10}, {}, 40);
    expect(bound.accepted && bound.selected_count == 4 &&
               bound.assigned_service_ticks == std::vector<uint64_t>({10, 10, 10, 10}) &&
               bound.original_total_ticks == 40 &&
               bound.max_prefix_delta_ticks == 1,
           "(9,11,10,10) averages to 10 each with a one-tick prefix bound");
    expect(PipelineKernel::original_service_at_external_tick(bound, 5) == 4 &&
               PipelineKernel::original_service_at_external_tick(bound, 10) == 9 &&
               PipelineKernel::original_service_at_external_tick(bound, 15) == 14 &&
               PipelineKernel::original_service_at_external_tick(bound, 40) == 40,
           "external time maps piecewise to the original service prefix at tile boundaries");

    const auto nondivisible = PipelineKernel::plan_window(
        *contract, 4, 4, {2, 3, 4, 6}, {}, 300);
    expect(nondivisible.accepted && nondivisible.original_total_ticks == 15 &&
               nondivisible.assigned_service_ticks ==
                   std::vector<uint64_t>({4, 4, 4, 3}),
           "quotient/remainder gives the earliest slices the integer remainder");

    const auto tiny = PipelineKernel::plan_window(
        *contract, 4, 4, {0, 1, 0, 1}, {}, 2);
    expect(tiny.accepted && tiny.selected_count == 1 && tiny.exact &&
               tiny.assigned_service_ticks == std::vector<uint64_t>({0}),
           "zero/one tick vectors refine to an exact prefix instead of assigning positive work zero time");
}

void test_positive_original_slices_keep_an_inherent_tick_floor() {
    const auto contract = PipelineKernelContract::parse(contract_wire(2));
    // Updated for the frozen contract's kAbsolutePrefixBudgetTicks = 1
    // (exported in every run contract's kernel_contract JSON and
    // load-bearing for the "(9,11,10,10) averages to 10 each" case above,
    // whose ppm budget alone cannot admit the one-tick prefix bound).
    // Under that legalization {1,3} at K=2 accepts the averaged {2,2}:
    // the section-16.3 invariant it guards -- a positive original slice
    // is never rounded DOWN to zero external ticks -- still holds, since
    // plan_window's inherent floor keeps every positive slice at >= 1
    // external tick and the piecewise mapping returns each slice's
    // original service inside its window. The pre-budget assertion
    // (refine to the exact K=1 prefix {1}) predates that legalization
    // and contradicted it.
    const auto plan = PipelineKernel::plan_window(
        *contract, 2, 2, {1, 3}, {0, 0}, 4);
    expect(plan.accepted && plan.selected_count == 2 &&
               plan.assigned_service_ticks == std::vector<uint64_t>({2, 2}) &&
               plan.max_prefix_delta_ticks == 1,
           "a positive Timer slice keeps at least one external tick under "
           "the frozen one-tick absolute prefix budget");
    expect(PipelineKernel::original_service_at_external_tick(plan, 2) == 1 &&
               PipelineKernel::original_service_at_external_tick(plan, 4) == 4,
           "the averaged two-slice plan maps external time back onto the "
           "original service prefix at slice boundaries");
    // The inherent tick floor still forces refinement whenever integer
    // averaging WOULD assign positive work zero external ticks: {0,1} at
    // K=2 averages to {1,0} (the positive second slice gets zero), so the
    // window refines to the exact K=1 prefix.
    const auto floor = PipelineKernel::plan_window(
        *contract, 2, 2, {0, 1}, {0, 0}, 1);
    expect(floor.accepted && floor.selected_count == 1 && floor.exact &&
               floor.assigned_service_ticks == std::vector<uint64_t>({0}),
           "a would-be zero assignment for positive Timer work forces "
           "exact prefix refinement (inherent tick floor)");
}

void test_hardware_floor_and_progressive_refinement() {
    const auto contract = PipelineKernelContract::parse(contract_wire(8, 0));
    const auto floor_refined = PipelineKernel::plan_window(
        *contract, 8, 4, {20, 5, 10, 10}, {20, 0, 0, 0}, 45);
    expect(floor_refined.accepted && floor_refined.exact &&
               floor_refined.selected_count == 1 &&
               floor_refined.refinement_count == 2 &&
               floor_refined.assigned_service_ticks ==
                   std::vector<uint64_t>({20}),
           "hardware service floor forces K=4 to refine through K=2 to exact K=1");

    const auto prefix_refined = PipelineKernel::plan_window(
        *contract, 4, 4, {0, 10, 0, 10}, {}, 20);
    expect(prefix_refined.accepted && prefix_refined.exact &&
               prefix_refined.selected_count == 1 &&
               prefix_refined.refinement_count == 2,
           "failed local prefix budget shrinks each level until exact K=1");

    const auto impossible_floor = PipelineKernel::plan_window(
        *contract, 4, 4, {2, 18, 10, 10}, {12, 0, 0, 0}, 40);
    expect(!impossible_floor.accepted &&
               impossible_floor.unsupported_reason ==
                   "hardware_minimum_exceeds_exact_service",
           "invalid exact input below its declared hardware floor fails closed");

    expect_throw([&] {
        (void)PipelineKernel::plan_window(*contract, 3, 4, {4, 4, 4, 4}, {}, 16);
    }, "unsupported effective K is rejected before narrowing");
}

void test_controller_counters_and_invalidation() {
    const auto contract = PipelineKernelContract::parse(contract_wire());
    PipelineWindowController controller(8);
    expect(controller.current_limit() == 4,
           "initial active window is min(4, configured limit)");
    for (uint32_t i = 0; i < PipelineKernelContract::stable_window_count(); ++i) {
        const auto plan = controller.plan(*contract, 8, {9, 11, 10, 10}, {}, 40);
        controller.record(plan, true);
    }
    expect(controller.current_limit() == 8 && controller.stable_windows() == 0 &&
               controller.window_count() == 8 &&
               controller.local_prefix_bound_max() == 1,
           "eight committed stable windows advance tier and publish counters");

    const auto generation = controller.generation();
    controller.invalidate();
    expect(controller.generation() == generation + 1 &&
               controller.stable_windows() == 0,
           "resource invalidation advances generation and clears stable streak");

    const auto stale = controller.plan(*contract, 8, {9, 11, 10, 10}, {}, 40);
    controller.invalidate();
    expect_throw([&] { controller.record(stale, true); },
                 "a plan from a stale resource generation cannot be committed");

    const auto exact = controller.plan(*contract, 1, {9, 11, 10, 10}, {}, 40);
    controller.record(exact, true);
    expect(exact.exact && exact.selected_count == 1,
           "per-call exact arbitration tightens effective K without changing run contract");
    const auto counters = controller.counters_json(*contract, 1);
    expect(counters.at("window_tile_limit") == 8 &&
               counters.at("effective_window_tile_limit") == 1 &&
               counters.at("window_generation") == generation + 2 &&
               counters.at("window_count") == 8,
           "serializable counters preserve configured and effective precision separately");

    const auto serialized = exact.to_json(*contract, 1);
    expect(serialized.at("window_contract").at("average_remainder_rule") ==
               PipelineKernelContract::kAverageRemainderRule &&
               serialized.at("accepted") == true && serialized.at("exact") == true,
           "plan wire includes immutable algorithm contract and decision evidence");
}

void test_kernel_contract_equality_and_rejections() {
    // Migrated from the retired pipeline_descriptor_test.cc (D4: the
    // contract lives on even though the descriptor wire is gone).
    const auto a = PipelineKernelContract::parse(contract_wire(1));
    const auto b = PipelineKernelContract::parse(contract_wire(1));
    const auto c = PipelineKernelContract::parse(contract_wire(8));
    expect(*a == *b && *a != *c,
           "typed run contract equality compares all fields");
    expect(a->window_tile_limit() == 1 &&
               PipelineKernelContract::stable_window_count() == 8 &&
               PipelineKernelContract::absolute_prefix_budget_ticks() == 1,
           "parsed tunables and fixed constants exposed");
    auto bad = contract_wire(3);
    expect_throw([&] { (void)PipelineKernelContract::parse(bad); },
                 "unsupported precision limit rejected");
    bad = contract_wire(1);
    bad["extra"] = 1;
    expect_throw([&] { (void)PipelineKernelContract::parse(bad); },
                 "unknown kernel contract field rejected");
}

}  // namespace

int main() {
    test_tunable_levels_and_integer_average();
    test_positive_original_slices_keep_an_inherent_tick_floor();
    test_hardware_floor_and_progressive_refinement();
    test_controller_counters_and_invalidation();
    test_kernel_contract_equality_and_rejections();
    if (failures != 0) {
        std::cerr << "[pipeline_window_test] " << failures << " failure(s)\n";
        return 1;
    }
    std::cout << "[pipeline_window_test] PASS\n";
    return 0;
}
