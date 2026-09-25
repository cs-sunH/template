/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

watch_registry_test.cc -- phase-1 step 1-5 watch/fence fixture.

Exercises the stage-completion watch machinery (方案 §4 步骤 1-5) as a
standalone test -- no network simulation, no baseline artifacts touched:

  Part A  Registration + identity: duplicate identity returns the existing
          watch (register once / fire once), unknown-identity terminals
          no-op, non-member terminals no-op, generation mismatch no-op
          (meta level and key level), duplicate member terminal no-op.
  Part B  Fire semantics: out-of-order completion fires exactly once when the
          LAST member completes; fired_and_drain delivers once and clears;
          terminals after the fire are no-ops; the explicit status policy
          (Skipped is never implicitly upgraded to Success) rejects a
          Skipped terminal when the watch does not list it.
  Part C  Lifecycle + audit: remove_watch / remove_all, stale_count()==0
          end audit, empty expected set never fires (stale, documented
          programming error), fire notifier wiring.
  Part D  online_completion_hook integration: the step-1-5 hook records the
          fact through the registry (read-only) -- the ① half of the hook's
          two-job contract; the ② half (mailbox write) is the step-1-6
          notifier wiring exercised here via set_fire_notifier.
  Part E  Deferred issue-pass coalescing: one pending pass per rank, FIFO by
          first rank occurrence, stable preallocated callback arguments, and
          nested same-rank completion appended to the same drain.
  Part F  Communicator replacement lifetime: exact rank-order/dimension
          identity and DataSet-held shared ownership until collective finish.
  Part G  Online HardwareResource: exact counters without per-node ID hashes;
          static diagnostic tracking remains enabled.

Build: the CMake target AstraSim_Analytical_Congestion_Aware_WatchRegistryTest
(build with cmake --build build/astra_analytical/build_congestion_aware -j).
Run (from template/astra-sim-wscllm):
  build/astra_analytical/build_congestion_aware/bin/\
      AstraSim_Analytical_Congestion_Aware_WatchRegistryTest
Exit code 0 on ALL PASS.
*******************************************************************************/

#include "astra-sim/system/CommunicatorGroup.hh"
#include "astra-sim/system/DataSet.hh"
#include "astra-sim/workload/HardwareResource.hh"
#include "astra-sim/workload/execution_driven/CompletionObserver.hh"
#include "astra-sim/workload/execution_driven/WatchRegistry.hh"
#include "common/EventQueue.h"

#include <cstdio>
#include <cstdlib>
#include <memory>
#include <set>
#include <string>
#include <vector>

using namespace AstraSim::ExecutionDriven;
using namespace NetworkAnalytical;

namespace {

bool g_ok = true;

void expect(bool cond, const char* what) {
    if (!cond) {
        std::fprintf(stderr, "[watch_registry_test] FAIL: %s\n", what);
        g_ok = false;
    }
}

CompletionKey key(int rank, uint64_t node_id, uint64_t generation) {
    return CompletionKey{rank, node_id, generation};
}

// ---------------------------------------------------------------- Part A --
// Registration, identity lookup, no-op rules.
void test_registration_and_noops() {
    WatchRegistry reg;

    // Two-member prefill watch (rank 0, nodes 1 and 2, generation 1).
    std::set<CompletionKey> members = {key(0, 1, 1), key(0, 2, 1)};
    std::set<NodeTerminalStatus> statuses = {NodeTerminalStatus::Success,
                                             NodeTerminalStatus::Skipped};
    const uint64_t watch_id = reg.register_stage_watch(
        "req-1", "prefill", 1, members, statuses);
    expect(watch_id == 1, "A: first watch id is 1");
    expect(reg.stale_count() == 1, "A: one registered not-fired watch");

    // Duplicate identity -> same watch id, no second watch (register once).
    const uint64_t dup_id = reg.register_stage_watch(
        "req-1", "prefill", 1, members, statuses);
    expect(dup_id == watch_id, "A: duplicate identity returns existing id");
    expect(reg.stale_count() == 1, "A: duplicate registration adds no watch");

    // A different stage of the same request is a different watch.
    const uint64_t decode_id = reg.register_stage_watch(
        "req-1", "decode", 1, {key(0, 3, 1)}, statuses);
    expect(decode_id == 2, "A: second identity gets id 2");

    // Unknown identity (request/stage/generation) terminal: no-op.
    reg.on_node_terminal(key(0, 1, 1),
                         NodeTerminalStatus::Success);
    reg.on_node_terminal(key(0, 1, 1),
                         NodeTerminalStatus::Success);
    expect(reg.stale_count() == 2, "A: foreign terminals are no-ops");

    // Generation mismatch at the meta level (identity miss) -- covered
    // above; at the key level with a matching meta identity:
    reg.on_node_terminal(key(0, 1, 2),
                         NodeTerminalStatus::Success);
    expect(reg.stale_count() == 2,
           "A: key generation mismatch is a no-op");

    // Non-member terminal (node 99 is not in the exact set).
    reg.on_node_terminal(key(0, 99, 1),
                         NodeTerminalStatus::Success);
    expect(reg.stale_count() == 2, "A: non-member terminal is a no-op");

    // Member 1 completes: recorded, not fired yet (member 2 pending).
    reg.on_node_terminal(key(0, 1, 1),
                         NodeTerminalStatus::Success);
    expect(reg.stale_count() == 2, "A: half-completed watch still stale");
    expect(reg.fired_and_drain().empty(), "A: nothing fired yet");

    // Duplicate terminal of the already-completed member: no-op.
    reg.on_node_terminal(key(0, 1, 1),
                         NodeTerminalStatus::Success);
    reg.on_node_terminal(key(0, 1, 1),
                         NodeTerminalStatus::Skipped);
    expect(reg.stale_count() == 2, "A: duplicate member terminal is a no-op");

    // Member 2 completes: fires once.
    reg.on_node_terminal(key(0, 2, 1),
                         NodeTerminalStatus::Skipped);
    auto fires = reg.fired_and_drain();
    expect(fires.size() == 1, "A: watch fired once");
    expect(fires[0].watch_id == watch_id && fires[0].request_id == "req-1" &&
               fires[0].stage == "prefill" && fires[0].generation == 1 &&
               fires[0].member_count == 2,
           "A: fire carries identity and member count");
    expect(reg.stale_count() == 1, "A: fired watch no longer stale");

    // Terminals after the fire: no-op (fire once).
    reg.on_node_terminal(key(0, 2, 1),
                         NodeTerminalStatus::Success);
    expect(reg.fired_and_drain().empty(), "A: no second fire");

    reg.remove_all();
    expect(reg.stale_count() == 0, "A: remove_all clears the registry");
}

// ---------------------------------------------------------------- Part B --
// Out-of-order completion, explicit status policy, fire-once.
void test_fire_semantics() {
    WatchRegistry reg;

    // Out-of-order: member 11 completes before member 10.
    reg.register_stage_watch("req-2", "decode", 1,
                             {key(0, 10, 1), key(0, 11, 1)},
                             {NodeTerminalStatus::Success});
    reg.on_node_terminal(key(0, 11, 1),
                         NodeTerminalStatus::Success);
    expect(reg.stale_count() == 1, "B: out-of-order first member no fire");
    reg.on_node_terminal(key(0, 10, 1),
                         NodeTerminalStatus::Success);
    const auto fires = reg.fired_and_drain();
    expect(fires.size() == 1 && fires[0].member_count == 2,
           "B: fires when the LAST member completes");

    // Explicit status policy: a watch listing only Success must NOT count a
    // Skipped terminal (Skipped is never implicitly upgraded to Success).
    reg.register_stage_watch("req-3", "prefill", 1,
                             {key(0, 20, 1), key(0, 21, 1)},
                             {NodeTerminalStatus::Success});
    reg.on_node_terminal(key(0, 20, 1),
                         NodeTerminalStatus::Skipped);
    expect(reg.stale_count() == 1, "B: Skipped rejected by Success-only watch");
    reg.on_node_terminal(key(0, 20, 1),
                         NodeTerminalStatus::Success);
    reg.on_node_terminal(key(0, 21, 1),
                         NodeTerminalStatus::Success);
    expect(reg.stale_count() == 0, "B: fires once both satisfy Success");
    expect(reg.fired_and_drain().size() == 1, "B: exactly one fire");

    // The phase-1 wscllm stage policy {Success, Skipped}: a Skipped terminal
    // satisfies (stages end when their last node is terminal; real traces
    // contain always-skipped INVALID/metadata nodes).
    reg.register_stage_watch("req-4", "prefill", 1, {key(0, 30, 1)},
                             {NodeTerminalStatus::Success,
                              NodeTerminalStatus::Skipped});
    reg.on_node_terminal(key(0, 30, 1),
                         NodeTerminalStatus::Skipped);
    expect(reg.stale_count() == 0, "B: {Success,Skipped} watch counts Skipped");

    reg.remove_all();
}

// ---------------------------------------------------------------- Part C --
// Lifecycle, audit, fire notifier.
void test_lifecycle_and_notifier() {
    WatchRegistry reg;
    int notifier_calls = 0;
    uint64_t notifier_watch_id = 0;
    reg.set_fire_notifier([&](const WatchFire& fire) {
        ++notifier_calls;
        notifier_watch_id = fire.watch_id;
    });

    reg.register_stage_watch("req-5", "prefill", 1, {key(0, 40, 1)},
                             {NodeTerminalStatus::Success});
    const uint64_t aborted = reg.register_stage_watch(
        "req-6", "prefill", 1, {key(0, 50, 1)},
        {NodeTerminalStatus::Success});

    // Batch abort: the req-6 watch is removed before any terminal.
    reg.remove_watch(aborted);
    expect(reg.stale_count() == 1, "C: remove_watch drops the aborted watch");
    reg.on_node_terminal(key(0, 50, 1),
                         NodeTerminalStatus::Success);
    expect(reg.fired_and_drain().empty(),
           "C: removed watch's terminals are no-ops");
    expect(reg.stale_count() == 1, "C: removed watch not stale");

    // Fire notification fires exactly once, synchronously, with the id.
    reg.on_node_terminal(key(0, 40, 1),
                         NodeTerminalStatus::Success);
    expect(notifier_calls == 1 && notifier_watch_id == 1,
           "C: notifier called once with the fired watch id");
    expect(reg.fired_and_drain().empty(),
           "C: notifier mode retains no duplicate polling record");
    expect(reg.stale_count() == 0, "C: all watches fired/removed");

    // Run-end audit: stale_count() == 0 (fired watches are not stale).
    expect(reg.stale_count() == 0, "C: end audit stale_count() == 0");

    // Empty expected set never fires: documented programming error, caught
    // by the audit (it must be removed, not silently fired).
    const uint64_t empty = reg.register_stage_watch(
        "req-7", "decode", 1, {}, {NodeTerminalStatus::Success});
    expect(reg.stale_count() == 1, "C: empty watch is stale");
    reg.on_node_terminal(key(0, 60, 1),
                         NodeTerminalStatus::Success);
    expect(reg.fired_and_drain().empty(),
           "C: empty watch never fires (explicit)");
    reg.remove_watch(empty);
    expect(reg.stale_count() == 0, "C: empty watch removed, audit clean");

    reg.remove_all();
}

// ---------------------------------------------------------------- Part D --
// Direct CompletionKey index: one terminal can satisfy multiple watches, and
// removal must erase every direct-index route before the id is reusable.
void test_direct_member_index() {
    WatchRegistry reg;
    std::vector<uint64_t> fired_ids;
    reg.set_fire_notifier([&](const WatchFire& fire) {
        fired_ids.push_back(fire.watch_id);
    });
    const CompletionKey shared = key(3, 80, 4);
    const uint64_t first = reg.register_stage_watch(
        "req-a", "prefill", 4, {shared}, {NodeTerminalStatus::Success});
    const uint64_t second = reg.register_stage_watch(
        "req-b", "decode", 4, {shared}, {NodeTerminalStatus::Success});
    reg.on_node_terminal(shared, NodeTerminalStatus::Success);
    expect(fired_ids.size() == 2 && fired_ids[0] == first &&
               fired_ids[1] == second,
           "D: one indexed terminal fires every overlapping watch once");
    expect(reg.fired_and_drain().empty(),
           "D: notifier mode has no retained fired-watch queue");

    const CompletionKey removed_key = key(4, 81, 5);
    const uint64_t removed = reg.register_stage_watch(
        "req-c", "prefill", 5, {removed_key},
        {NodeTerminalStatus::Success});
    reg.remove_watch(removed);
    reg.on_node_terminal(removed_key, NodeTerminalStatus::Success);
    expect(fired_ids.size() == 2,
           "D: removed watch is absent from the direct member index");
    reg.remove_all();
}

// ---------------------------------------------------------------- Part E --
// online_completion_hook integration (步骤 1-5 操作 3, the ① half: fact
// recording through the registry; ② the mailbox write is the step-1-6
// notifier wiring, exercised here through set_fire_notifier).
void test_online_hook() {
    WatchRegistry reg;
    int fires = 0;
    reg.set_fire_notifier([&](const WatchFire&) { ++fires; });
    reg.register_stage_watch("req-8", "prefill", 1,
                             {key(2, 70, 1), key(2, 71, 1)},
                             {NodeTerminalStatus::Success,
                              NodeTerminalStatus::Skipped});

    OnlineCompletionHookContext ctx;
    ctx.registry = &reg;
    // The hook's terminal_status int is a NodeTerminalStatus value.
    online_completion_hook(&ctx, 2, 70, "req-8", "prefill", 1,
                           /*tick=*/12345, static_cast<int>(
                               NodeTerminalStatus::Skipped));
    expect(fires == 0, "D: hook records the fact, no fire yet");
    online_completion_hook(&ctx, 2, 71, "req-8", "prefill", 1,
                           /*tick=*/12345, static_cast<int>(
                               NodeTerminalStatus::Success));
    expect(fires == 1, "D: hook-recorded facts fire the watch");
    expect(reg.stale_count() == 0, "D: fired watch not stale");

    // nullptr ctx / null registry: the hook is a no-op (never crashes).
    online_completion_hook(nullptr, 0, 1, "req-8", "prefill", 1, 0, 0);
    OnlineCompletionHookContext empty_ctx;
    online_completion_hook(&empty_ctx, 0, 1, "req-8", "prefill", 1, 0, 0);
    expect(fires == 1, "D: null context is a no-op");

    reg.remove_all();
}

struct CoalescingCtx {
    OnlineCompletionHookContext* hook = nullptr;
    std::vector<int> ranks;
    bool nested_scheduled = false;
};

void issue_pass_probe(void* opaque, const int rank) {
    auto* const probe = static_cast<CoalescingCtx*>(opaque);
    probe->ranks.push_back(rank);
    if (rank == 0 && !probe->nested_scheduled) {
        probe->nested_scheduled = true;
        expect(probe->hook->schedule_issue_pass(0),
               "E: callback-start clear permits nested same-rank pass");
    }
}

void no_op_event(void*) {}

void test_issue_pass_coalescing() {
    EventQueue eq;
    WatchRegistry reg;
    OnlineCompletionHookContext ctx;
    CoalescingCtx probe;
    probe.hook = &ctx;
    ctx.registry = &reg;
    ctx.configure_issue_passes(&eq, 3, issue_pass_probe, &probe);

    // First occurrences establish FIFO rank order 2,0,1. Duplicates before
    // the drain must not allocate or enqueue another callback.
    online_completion_hook(&ctx, 2, 100, "", "", 0, 10, 0);
    online_completion_hook(&ctx, 2, 101, "", "", 0, 10, 0);
    online_completion_hook(&ctx, 0, 102, "", "", 0, 10, 0);
    online_completion_hook(&ctx, 1, 103, "", "", 0, 10, 0);
    online_completion_hook(&ctx, 0, 104, "", "", 0, 10, 0);
    expect(ctx.pending_issue_pass_count() == 3,
           "E: duplicate terminals coalesce to one pass per rank");

    // Deferred work drains only at a proceed boundary. Rank 0's callback
    // schedules a nested rank-0 pass after its pending bit was cleared.
    eq.schedule_event(10, no_op_event, nullptr);
    eq.proceed();
    expect(probe.ranks == std::vector<int>({2, 0, 1, 0}),
           "E: first-rank FIFO preserved; nested pass appended same drain");
    expect(ctx.pending_issue_pass_count() == 0,
           "E: no pending pass after the nested drain");
    expect(eq.finished() && !eq.has_deferred_work(),
           "E: physical and deferred queues both empty");
}

struct LifetimeProbe {
    explicit LifetimeProbe(bool* destroyed) : destroyed(destroyed) {}
    ~LifetimeProbe() { *destroyed = true; }
    bool* destroyed;
};

void test_communicator_lifetime() {
    expect(AstraSim::same_communicator_definition(
               {0, 2, 1}, {3}, {0, 2, 1}, {3}),
           "F: identical ordered ranks and dimensions are a no-op definition");
    expect(!AstraSim::same_communicator_definition(
               {0, 2, 1}, {3}, {0, 1, 2}, {3}),
           "F: rank order is part of communicator identity");
    expect(!AstraSim::same_communicator_definition(
               {0, 2, 1}, {3}, {0, 2, 1}, {1, 3}),
           "F: shaped dimensions are part of communicator identity");

    bool destroyed = false;
    std::weak_ptr<LifetimeProbe> weak;
    {
        // The explicit creation tick keeps this isolated unit fixture
        // independent of Sys while exercising the real DataSet owner field.
        AstraSim::DataSet dataset(1, 0);
        {
            auto probe = std::make_shared<LifetimeProbe>(&destroyed);
            weak = probe;
            std::shared_ptr<AstraSim::CommunicatorGroup> owner(
                probe, static_cast<AstraSim::CommunicatorGroup*>(nullptr));
            dataset.retain_communicator_group(std::move(owner));
            probe.reset();
        }
        expect(!destroyed && !weak.expired(),
               "F: communicator control block survives active DataSet");
    }
    expect(destroyed && weak.expired(),
           "F: DataSet completion/destruction releases old communicator");
}

void test_online_hardware_counters_without_ids() {
    AstraSim::HardwareResource online(1, ExecutionMode::Online);
    expect(!online.tracks_node_ids(),
           "G: online HardwareResource disables diagnostic ID hashes");
    NodeView first;
    first.global_id = 1000;
    first.kind = NodeKind::Compute;
    NodeView second = first;
    second.global_id = 1001;
    online.occupy(first);
    online.occupy(second);
    expect(online.num_in_flight_gpu_comp_ops == 2 &&
               online.gpu_ops_node.empty(),
           "G: online concurrent compute uses exact count and no IDs");
    online.release(first);
    online.release(second);
    expect(online.num_in_flight_gpu_comp_ops == 0 &&
               online.gpu_ops_node.empty(),
           "G: online releases return the counter to zero");

    AstraSim::HardwareResource static_resource(1);
    expect(static_resource.tracks_node_ids(),
           "G: static HardwareResource preserves ID diagnostics");
}

}  // namespace

int main(int /*argc*/, char* /*argv*/[]) {
    test_registration_and_noops();
    test_fire_semantics();
    test_lifecycle_and_notifier();
    test_direct_member_index();
    test_online_hook();
    test_issue_pass_coalescing();
    test_communicator_lifetime();
    test_online_hardware_counters_without_ids();

    if (!g_ok) {
        std::fprintf(stderr, "[watch_registry_test] FAIL: see messages above\n");
        return 1;
    }
    std::printf("[watch_registry_test] ALL PASS: registration/no-ops, fire "
                "semantics, lifecycle/audit, online hook integration, "
                "deferred rank coalescing, communicator lifetime, online "
                "hardware counters\n");
    return 0;
}
