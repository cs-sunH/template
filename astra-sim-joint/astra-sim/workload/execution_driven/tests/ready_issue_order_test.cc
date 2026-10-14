/******************************************************************************
ready_issue_order_test -- B3 ready-scan / issue-order differential
(exec doc 10.2, design doc packages/wave2-design-20261006/
B3_ready_issue_order_test_design.md).

Two layers, one binary:

Store layer (hermetic, no Sys/network): the store-side scenarios from
the frozen golden harvested out-of-repo against the A0 build
(golden/ready-issue-old-sequences.json in the B3 evidence package; lib
SHA256 52ff5216.../86a367d3..., byte-identical double run) are replayed
1:1 against the LIVE NodeStore. The expected sequences are embedded as
literals below. R3 is the single REGISTERED old/new divergence: the old
snapshot loop revisits a member taken+finished inside the same pass; the
new issue loop rejects it through ready-set membership (reason
b3_membership_dedup). The four new-only assertions (incremental ready
delta, membership dedup, batch boundary, stale-no-cut-in) run here too.

Full-stack layer (real EventQueue + FluidScheduler +
CongestionAwareNetworkApi + per-rank online Sys + WatchRegistry +
OnlineCompletionHookContext,
built with the shared fixture config writer in
plain schema-3 batches drive the real
Workload::issue_dep_free_nodes issue path (R5 rank gate, R6 watch-gated
deferred release, R7 same-tick deferred drain + t+1 bail-out, N4 rerun
convergence). Sequences are
captured through Workload::set_ledger_transition_observer ("issue") and
the Workload ledger observer ("issue" of a
endpoint), the authoritative hooks named by the design doc.

Machine-readable contract: every scenario prints its captured sequence as
  seq <case_id> <tick> <rank> <ids...>
plus a PASS/FAIL line; the run ends with "ALL PASS" on success.
******************************************************************************/


#include "astra-sim/common/Logging.hh"
#include "astra-sim/system/Sys.hh"
#include "astra-sim/workload/LocalHbmBandwidthModel.hh"
#include "astra-sim/workload/MetricCollector.hh"
#include "astra-sim/workload/Workload.hh"
#include "astra-sim/workload/execution_driven/CompletionObserver.hh"
#include "astra-sim/workload/execution_driven/DecisionMailbox.hh"
#include "astra-sim/workload/execution_driven/GraphBatchCommitter.hh"
#include "astra-sim/workload/execution_driven/NodeStore.hh"
#include "astra-sim/workload/execution_driven/WatchRegistry.hh"
#include "astra-sim/workload/execution_driven/WorkLedgerV2Publisher.hh"
#include "congestion_aware/CongestionAwareNetworkApi.hh"
#include "extern/remote_memory_backend/analytical/AnalyticalRemoteMemory.hh"
#include <astra-network-analytical/common/EventQueue.h>
#include <astra-network-analytical/common/NetworkParser.h>
#include <astra-network-analytical/congestion_aware/Helper.h>

using namespace AstraSim;
using namespace AstraSim::ExecutionDriven;
using namespace Analytical;
using namespace AstraSimAnalytical;
using namespace AstraSimAnalyticalCongestionAware;
using namespace NetworkAnalytical;
using namespace NetworkAnalyticalCongestionAware;

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <stdexcept>
#include <set>
#include <string>
#include <type_traits>
#include <utility>

namespace AstraSim {

struct WorkloadTestAccess {
    static bool snapshot_cache_valid(const Workload& workload) {
        return workload.dep_free_issue_snapshot_valid_;
    }

    static const std::vector<uint64_t>& snapshot_ids(
        const Workload& workload) {
        return workload.dep_free_issue_snapshot_ids_;
    }

    // The production guard intentionally has its existing exception behavior;
    // this reset is only for retrying the injected pre-callback test failure.
    static void reset_issue_guard(Workload& workload) {
        workload.issuing_dep_free_ = false;
        workload.issue_dep_free_rerun_needed_ = false;
    }
};

namespace ExecutionDriven {

struct NodeStoreTestAccess {
    static void set_free_id_version(NodeStore& store, const uint64_t version) {
        store.free_id_version_ = version;
    }
};

namespace {

// ---------------------------------------------------------------------------
// Golden literals (old side, frozen A0 build; see the B3 evidence package
// golden/ready-issue-old-sequences.json). Store ids: ascending auto stream
// from 1. R1 p1=1 p2=2 i0=3 c=4; R2 a=1 d=10 n_low=2 n_high=11; R3 x=1 y=2;
// R4 p=1 q=2 child=3.
// ---------------------------------------------------------------------------
const std::vector<std::vector<uint64_t>> kOldR1 = {{1, 2, 3}, {}, {4}};
const std::vector<std::vector<uint64_t>> kOldR2 = {{1, 10}, {2, 11}};
const std::vector<std::vector<uint64_t>> kOldR3 = {{1, 2}, {}};  // divergence
const std::vector<std::vector<uint64_t>> kOldR4 = {{1, 2}, {3}, {}};

int g_failures = 0;

// Rank owning the N5/N6 direct-store scenarios (was the owner rank of
// the retired remote-read fixture program).
constexpr int kOwnerRank = 6;

void b3_fail(const std::string& message) {
    std::fprintf(stderr, "[ready_issue_order_test] FAIL: %s\n",
                 message.c_str());
    std::fflush(stderr);
    ++g_failures;
    std::exit(1);
}

void b3_expect(const bool condition, const std::string& message) {
    if (!condition) {
        b3_fail(message);
    }
}

std::string join_ids(const std::vector<uint64_t>& ids) {
    std::string out;
    for (const uint64_t id : ids) {
        if (!out.empty()) {
            out += ",";
        }
        out += std::to_string(id);
    }
    return out;
}

OnlineNode b3_node(const uint64_t id, const std::string& name) {
    OnlineNode node;
    node.global_id = id;
    node.kind = NodeKind::Compute;
    node.node_type = 4;
    node.name = name;
    return node;
}

class ThrowingNodeStoreGraphSource final : public NodeStoreGraphSource {
  public:
    bool throw_on_fill = true;

    bool fill_dep_free_id_snapshot(std::vector<uint64_t>& out) override {
        if (throw_on_fill) {
            throw std::runtime_error("injected ready snapshot failure");
        }
        return NodeStoreGraphSource::fill_dep_free_id_snapshot(out);
    }
};

class FalseSnapshotNodeStoreGraphSource final : public NodeStoreGraphSource {
  public:
    bool fill_dep_free_id_snapshot(std::vector<uint64_t>&) override {
        return false;
    }
};

class FlippingNodeStoreGraphSource final : public NodeStoreGraphSource {
  public:
    bool include_all_ready_ids = false;

    bool fill_dep_free_id_snapshot(std::vector<uint64_t>& out) override {
        const bool filled =
            NodeStoreGraphSource::fill_dep_free_id_snapshot(out);
        if (filled && !include_all_ready_ids && out.size() > 1) {
            out.resize(1);
        }
        return filled;
    }
};

// The Workload issue-loop emulation at store level (mirrors the golden
// harness exactly): record, take, then the issue-callback injection.
std::vector<uint64_t> run_pass(NodeStoreGraphSource& source,
                               const std::function<void(uint64_t)>& inject =
                                   nullptr) {
    std::vector<uint64_t> visited;
    source.for_each_dep_free(
        [&source, &visited, &inject](const NodeView& node) {
            visited.push_back(node.global_id);
            source.take_node(node.global_id);
            if (inject != nullptr) {
                inject(node.global_id);
            }
        });
    return visited;
}

// The NEW-side issue loop: the Workload::issue_dep_free_nodes decision
// order (membership revalidation + lookup_ptr gate) applied to the live
// store. Used where the golden's old loop differs (R3) or where the
// scenario asserts the loop's own guards.
std::vector<uint64_t> run_new_issue_loop(NodeStoreGraphSource& source,
                                         const std::function<void(uint64_t)>&
                                             inject = nullptr) {
    std::vector<uint64_t> snapshot;
    b3_expect(source.fill_dep_free_id_snapshot(snapshot),
              "store source provides the separated ready snapshot");
    std::vector<uint64_t> issued;
    for (const uint64_t node_id : snapshot) {
        if (!source.is_dep_free_member(node_id)) {
            continue;  // membership dedup: stale captured member
        }
        const NodeView* const view = source.lookup_ptr(node_id);
        if (view == nullptr) {
            continue;  // GC-erased captured member
        }
        issued.push_back(node_id);
        source.take_node(node_id);
        if (inject != nullptr) {
            inject(node_id);
        }
    }
    return issued;
}

void print_seq(const std::string& case_id, const uint64_t tick, const int rank,
               const std::vector<uint64_t>& ids) {
    std::printf("seq %s %llu %d %s\n", case_id.c_str(),
                static_cast<unsigned long long>(tick), rank,
                join_ids(ids).c_str());
}

// ===========================================================================
// Store layer: golden replay R1/R2/R4 (identical old/new) + R3 (registered
// divergence) + the new-only faces.
// ===========================================================================

void scenario_r1_store() {
    NodeStoreGraphSource source;
    auto& store = source.store();
    const uint64_t p1 = store.add_node(b3_node(0, "p1"));
    const uint64_t p2 = store.add_node(b3_node(0, "p2"));
    const uint64_t i0 = store.add_node(b3_node(0, "i0"));
    const uint64_t c = store.add_node(b3_node(0, "c"));
    store.add_dependency(p1, c, DepKind::Data);
    store.add_dependency(p2, c, DepKind::Data);
    b3_expect(p1 == 1 && p2 == 2 && i0 == 3 && c == 4,
              "R1 ids follow the golden auto stream");
    std::vector<std::vector<uint64_t>> passes;
    passes.push_back(run_pass(source));
    print_seq("R1/pass1", 0, 0, passes.back());
    store.finish_node(p1);
    passes.push_back(run_pass(source));
    print_seq("R1/pass2", 0, 0, passes.back());
    store.finish_node(p2);
    passes.push_back(run_pass(source));
    print_seq("R1/pass3", 0, 0, passes.back());
    b3_expect(passes == kOldR1,
              "R1 new-side passes equal old golden [[p1,p2,i0],[],[c]]");
    std::printf("golden R1 match: old=%s new=%s\n",
                join_ids({1, 2, 3}).c_str(), join_ids(passes[0]).c_str());
    std::printf("PASS R1 same-tick multi-parent: child blocked until last "
                "parent, then issued exactly once\n");
}

void scenario_r2_store() {
    NodeStoreGraphSource source;
    auto& store = source.store();
    const uint64_t a = store.add_node(b3_node(0, "a"));
    const uint64_t d = store.add_node(b3_node(10, "d"));
    uint64_t n_low = 0;
    uint64_t n_high = 0;
    const std::function<void(uint64_t)> inject = [&](const uint64_t id) {
        if (id != a || n_low != 0) {
            return;
        }
        n_low = store.add_node(b3_node(a + 1, "n_low"));
        n_high = store.add_node(b3_node(0, "n_high"));
    };
    std::vector<std::vector<uint64_t>> passes;
    passes.push_back(run_pass(source, inject));
    print_seq("R2/pass1", 0, 0, passes.back());
    passes.push_back(run_pass(source));
    print_seq("R2/pass2", 0, 0, passes.back());
    b3_expect(a == 1 && d == 10 && n_low == 2 && n_high == 11,
              "R2 callback-added ids land below and above the cursor");
    b3_expect(passes == kOldR2,
              "R2 new-side passes equal old golden [[a,d],[n_low,n_high]]");
    std::printf("PASS R2 issue-callback additions: current batch unchanged, "
                "new members next pass ascending\n");
}

void scenario_r3_store() {
    NodeStoreGraphSource source;
    auto& store = source.store();
    const uint64_t x = store.add_node(b3_node(0, "x"));
    const uint64_t y = store.add_node(b3_node(0, "y"));
    b3_expect(x == 1 && y == 2, "R3 ids follow the golden auto stream");
    const std::function<void(uint64_t)> inject = [&](const uint64_t id) {
        if (id != x) {
            return;
        }
        source.take_node(y);
        source.finish_node(y);
    };
    // Old loop (snapshot walk, no membership revalidation): visits [x, y].
    const std::vector<uint64_t> old_style = run_pass(source, inject);
    print_seq("R3/old-loop", 0, 0, old_style);
    b3_expect(old_style == kOldR3[0],
              "R3 old-style loop reproduces the old golden stale revisit");

    // New loop: the same drive through the Workload issue-loop order.
    NodeStoreGraphSource source2;
    const uint64_t x2 = source2.store().add_node(b3_node(0, "x"));
    const uint64_t y2 = source2.store().add_node(b3_node(0, "y"));
    const std::function<void(uint64_t)> inject2 = [&](const uint64_t id) {
        if (id != x2) {
            return;
        }
        source2.take_node(y2);
        source2.finish_node(y2);
    };
    const std::vector<uint64_t> new_style = run_new_issue_loop(source2, inject2);
    print_seq("R3/new-loop", 0, 0, new_style);
    b3_expect(new_style == std::vector<uint64_t>({x2}),
              "R3 new issue loop skips the stale member");
    b3_expect(!source2.is_dep_free_member(y2) &&
                  source2.lookup_ptr(y2) != nullptr,
              "R3 stale member: membership rejected, record still resident");
    std::vector<uint64_t> empty_pass;
    b3_expect(!source2.fill_dep_free_id_snapshot(empty_pass) ||
                  empty_pass.empty(),
              "R3 nothing re-queued after the stale pass");
    std::printf("divergence R3 b3_membership_dedup: old issue [1,2] vs new "
                "issue [1]; stale member not re-occupied\n");
    std::printf("PASS R3 registered divergence (b3_membership_dedup)\n");
}

void scenario_r4_store() {
    NodeStoreGraphSource source;
    auto& store = source.store();
    const uint64_t p = store.add_node(b3_node(0, "p"));
    const uint64_t q = store.add_node(b3_node(0, "q"));
    const uint64_t child = store.add_node(b3_node(0, "child"));
    store.add_dependency(p, child, DepKind::Data);
    store.add_dependency(q, child, DepKind::Data);
    std::vector<std::vector<uint64_t>> passes;
    passes.push_back(run_pass(source));
    print_seq("R4/pass1", 0, 0, passes.back());
    store.finish_node(p);
    store.finish_node(p);
    store.finish_node(q);
    store.finish_node(q);
    passes.push_back(run_pass(source));
    print_seq("R4/pass2", 0, 0, passes.back());
    passes.push_back(run_pass(source));
    print_seq("R4/pass3", 0, 0, passes.back());
    b3_expect(passes == kOldR4,
              "R4 new-side passes equal old golden [[p,q],[child],[]]");
    std::printf("PASS R4 duplicate ready notifications: idempotent release, "
                "single requeue\n");
}

// N-store-1: incremental ready delta correctness. The delta mirrors every
// free-set mutation since the last snapshot consume; a repeated release
// never re-enqueues; the version is monotone.
void scenario_n_delta() {
    NodeStore store;
    const uint64_t a = store.add_node(b3_node(0, "a"));
    const uint64_t b = store.add_node(b3_node(0, "b"));
    b3_expect(store.free_delta_since_snapshot() ==
                  std::vector<uint64_t>({a, b}),
              "delta tracks both new ready ids ascending");
    std::vector<uint64_t> snapshot;
    store.fill_free_node_snapshot(snapshot);
    b3_expect(snapshot == std::vector<uint64_t>({a, b}) &&
                  store.free_delta_since_snapshot().empty(),
              "snapshot consume drains the delta");
    store.mark_issued(a);
    b3_expect(store.free_delta_since_snapshot().empty(),
              "consuming an already-snapshotted id does not re-enqueue");

    const uint64_t p = store.add_node(b3_node(0, "p"));
    const uint64_t child = store.add_node(b3_node(0, "child"));
    store.add_dependency(p, child, DepKind::Data);
    const uint64_t version_before = store.free_id_version();
    store.finish_node(p);
    const uint64_t version_after = store.free_id_version();
    b3_expect(version_after > version_before,
              "release bumps the free-set version");
    store.finish_node(p);  // idempotent duplicate
    b3_expect(store.free_id_version() == version_after,
              "duplicate finish does not bump the version");
    b3_expect(store.free_delta_since_snapshot() ==
                  std::vector<uint64_t>({child}),
              "freed child enters the delta exactly once");

    const uint64_t transient = store.add_node(b3_node(0, "transient"));
    store.mark_issued(transient);
    b3_expect(std::find(store.free_delta_since_snapshot().begin(),
                        store.free_delta_since_snapshot().end(),
                        transient) ==
                  store.free_delta_since_snapshot().end(),
              "freed-then-retaken id leaves the delta before any consume");
    std::vector<uint64_t> next_snapshot;
    store.fill_free_node_snapshot(next_snapshot);
    b3_expect(next_snapshot == std::vector<uint64_t>({b, child}),
              "next snapshot equals the live ready set ascending");
    std::printf("PASS N-delta incremental ready queue: delta drained per "
                "snapshot, no duplicate requeue, monotone version\n");
}

// N-store-scale: exercise the real NodeStore/GraphSource ready surface at
// increasing cardinalities without EventQueue, NoC, or physical runs. The
// first frontier is fully issueable; taking each root and finishing it wakes
// one previously blocked child during the captured pass. The new children
// must remain outside that caller-owned snapshot until the next pass.
void scenario_n_ready_snapshot_scale() {
    for (const size_t count : {size_t{8}, size_t{32}, size_t{128}}) {
        NodeStoreGraphSource source;
        NodeStore& store = source.store();
        std::vector<uint64_t> roots;
        std::vector<uint64_t> children;
        roots.reserve(count);
        children.reserve(count);
        for (size_t i = 0; i < count; ++i) {
            // global_id=0 is the documented auto-ID request. The first real
            // NodeStore ID is 1; zero never becomes a stored ready key.
            roots.push_back(store.add_node(b3_node(0, "ready-root")));
        }
        for (size_t i = 0; i < count; ++i) {
            children.push_back(store.add_node(b3_node(0, "blocked-child")));
        }
        b3_expect(roots.front() == 1 && roots.back() == count &&
                      children.front() == count + 1 &&
                      children.back() == count * 2,
                  "scale rows preserve the auto-ID stream from ID-zero input");
        b3_expect(!source.is_dep_free_member(0),
                  "ID zero remains the auto-ID sentinel, never a ready key");

        // Attach in reverse ID order so setup removes blocked IDs in reverse
        // order. The printed counters below are cumulative over setup and
        // both actual NodeStore issue frontiers.
        for (size_t i = count; i-- > 0;) {
            store.add_dependency(roots[i], children[i], DepKind::Data);
        }

        std::vector<uint64_t> captured_roots;
        b3_expect(source.fill_dep_free_id_snapshot(captured_roots) &&
                      captured_roots == roots,
                  "first scale frontier is the ascending ready roots only");
        for (const uint64_t id : captured_roots) {
            b3_expect(source.is_dep_free_member(id) &&
                          source.lookup_ptr(id) != nullptr,
                      "captured ready root remains a live NodeStore member");
            source.take_node(id);
        }
        // Complete in descending id order, as can happen when callbacks from
        // an earlier issue batch arrive out of order. The delta must remain a
        // set-backed sparse frontier instead of shifting a sorted vector for
        // every newly ready child.
        for (auto root_it = captured_roots.rbegin();
             root_it != captured_roots.rend(); ++root_it) {
            store.finish_node(*root_it);  // completion wakes its child
        }
        b3_expect(captured_roots == roots,
                  "callback wakeups do not rewrite the captured root batch");
        const std::vector<uint64_t>& captured_delta =
            store.free_delta_since_snapshot();
        b3_expect(captured_delta == children,
                  "descending completion callbacks materialize one sorted "
                  "child delta");

        std::vector<uint64_t> captured_children;
        b3_expect(source.fill_dep_free_id_snapshot(captured_children) &&
                      captured_children == children,
                  "next scale frontier contains the newly woken children");
        for (const uint64_t id : captured_children) {
            b3_expect(source.is_dep_free_member(id) &&
                          source.lookup_ptr(id) != nullptr,
                      "captured woken child remains a live NodeStore member");
            source.take_node(id);
        }
        b3_expect(captured_children == children,
                  "taking children does not rewrite their captured batch");

        std::vector<uint64_t> drained;
        source.fill_dep_free_id_snapshot(drained);
        b3_expect(drained.empty(),
                  "both scale frontiers drain exactly once in ID order");

        const NodeStore::ReadySetStats& stats = store.ready_set_stats();
        const uint64_t n = static_cast<uint64_t>(count);
        b3_expect(stats.snapshot_calls == 3 &&
                      stats.snapshot_ids_copied == 2 * n &&
                      stats.membership_queries == 2 * n + 1 &&
                      stats.free_insert_updates == 3 * n &&
                      stats.free_erase_updates == 4 * n &&
                      stats.delta_insertions == 3 * n &&
                      stats.delta_erasures == n &&
                      stats.delta_set_insert_calls == 3 * n &&
                      stats.delta_set_erase_calls == 4 * n &&
                      stats.delta_snapshot_reads == 1 &&
                      stats.delta_snapshot_materializations == 1 &&
                      stats.delta_snapshot_ids_copied == n,
                  "scale counters are linear in actual NodeStore snapshots, "
                  "membership/free-set updates and one ordered delta read");
        std::printf(
            "ready-scale N=%zu snapshot_calls=%llu snapshot_ids_copied=%llu "
            "membership=%llu free_insert_updates=%llu "
            "free_erase_updates=%llu delta_insertions=%llu "
            "delta_erasures=%llu delta_set_insert_calls=%llu "
            "delta_set_erase_calls=%llu delta_snapshot_reads=%llu "
            "delta_snapshot_materializations=%llu "
            "delta_snapshot_ids_copied=%llu\n",
            count,
            static_cast<unsigned long long>(stats.snapshot_calls),
            static_cast<unsigned long long>(stats.snapshot_ids_copied),
            static_cast<unsigned long long>(stats.membership_queries),
            static_cast<unsigned long long>(stats.free_insert_updates),
            static_cast<unsigned long long>(stats.free_erase_updates),
            static_cast<unsigned long long>(stats.delta_insertions),
            static_cast<unsigned long long>(stats.delta_erasures),
            static_cast<unsigned long long>(stats.delta_set_insert_calls),
            static_cast<unsigned long long>(stats.delta_set_erase_calls),
            static_cast<unsigned long long>(stats.delta_snapshot_reads),
            static_cast<unsigned long long>(
                stats.delta_snapshot_materializations),
            static_cast<unsigned long long>(stats.delta_snapshot_ids_copied));
    }
    std::printf("PASS N-store-scale: actual ordered NodeStore snapshots drain "
                "8/32/128 roots and callback-woken children without a "
                "physical event loop\n");
}

// N-store-2 + N-store-3: membership dedup across the node lifecycle and
// the captured-batch boundary (callbacks never rewrite the captured pass).
void scenario_n_membership_and_boundary() {
    NodeStoreGraphSource source;
    auto& store = source.store();
    const uint64_t x = store.add_node(b3_node(0, "x"));
    const uint64_t z = store.add_node(b3_node(0, "z"));
    std::vector<uint64_t> batch;
    b3_expect(source.fill_dep_free_id_snapshot(batch) &&
                  batch == std::vector<uint64_t>({x, z}),
              "captured batch holds the ascending ready order");
    const uint64_t w = store.add_node(b3_node(50, "w"));
    store.finish_node(x);
    b3_expect(batch == std::vector<uint64_t>({x, z}),
              "captured batch is not rewritten by later mutations");
    b3_expect(source.is_dep_free_member(z) && !source.is_dep_free_member(x) &&
                  !source.is_dep_free_member(w) ||
                  source.is_dep_free_member(w),
              "membership readable for every captured id");
    b3_expect(!source.is_dep_free_member(x),
              "finished member leaves the ready membership");
    b3_expect(source.is_dep_free_member(w),
              "newly added member is ready immediately");
    std::vector<uint64_t> next;
    source.fill_dep_free_id_snapshot(next);
    b3_expect(next == std::vector<uint64_t>({z, w}),
              "next pass sees the surviving plus the new member ascending");
    print_seq("N-boundary", 0, 0, next);

    // GC-erased member: lookup_ptr gate (the second guard of the new loop).
    store.set_gc_enabled(true);
    store.finish_node(z);
    store.collect_garbage();
    b3_expect(store.erased(z) && source.lookup_ptr(z) == nullptr,
              "collected member is gone from the record map");
    b3_expect(!source.is_dep_free_member(z),
              "collected member leaves the ready membership");
    std::printf("PASS N-membership: dedup across add/take/finish/GC; "
                "captured batch never rewritten\n");
}

// Virtual ids (hole AP) never enter any ready snapshot.
// R6 store face: register once / fire once at the last member, duplicate
// terminal is a no-op (values must equal the old golden facts).
void scenario_r6_store() {
    WatchRegistry registry;
    const CompletionKey m1{0, 101, 0};
    const CompletionKey m2{0, 102, 0};
    bool created = false;
    const uint64_t watch_id = registry.register_stage_watch(
        "b3-r6", "decode", 1, {m1, m2}, {NodeTerminalStatus::Success},
        &created);
    bool reregister_created = true;
    const uint64_t reregister_id = registry.register_stage_watch(
        "b3-r6", "decode", 1, {m1, m2}, {NodeTerminalStatus::Success},
        &reregister_created);
    int fires = 0;
    registry.set_fire_notifier([&fires](const WatchFire&) { ++fires; });
    registry.on_node_terminal(m1, NodeTerminalStatus::Success);
    const int after_first = fires;
    registry.on_node_terminal(m2, NodeTerminalStatus::Success);
    const int after_last = fires;
    registry.on_node_terminal(m2, NodeTerminalStatus::Success);
    // Old golden: watch_created=true, re_register_same_id=true,
    // re_register_created=false, count=1, fires 0/1/1, id match.
    b3_expect(created && reregister_id == watch_id && !reregister_created &&
                  registry.size() == 1 && after_first == 0 && after_last == 1 &&
                  fires == 1,
              "R6 store facts equal the old golden (register once / fire "
              "once / duplicate no-op)");
    std::printf("PASS R6-store register-once/fire-once equal old golden "
                "(fires 0->1->1)\n");
}

// R7 store face: same-tick deferred pass + t+1 bail-out with the merged
// pending bit, through the LIVE EventQueue and hook context. The drain
// pattern (empty main map + pending deferred -> wakeup at T+1 -> proceed)
// is the production main-loop bail-out mirrored by the old golden.
void noop_wakeup(void* /*arg*/) {}

void scenario_r7_store() {
    auto queue = std::make_shared<EventQueue>();
    OnlineCompletionHookContext context;
    WatchRegistry registry;
    context.registry = &registry;
    struct Recorder {
        int calls = 0;
        std::vector<int> ranks;
        uint64_t tick_at_call = UINT64_MAX;
        EventQueue* queue = nullptr;
    } recorder;
    recorder.queue = queue.get();
    context.configure_issue_passes(
        queue.get(), 2,
        [](void* opaque, int rank) {
            auto* rec = static_cast<Recorder*>(opaque);
            ++rec->calls;
            rec->ranks.push_back(rank);
            rec->tick_at_call = rec->queue->get_current_time();
        },
        &recorder);
    struct TickArg {
        OnlineCompletionHookContext* context;
    } tick_arg{&context};
    queue->schedule_event(
        1,
        [](void* arg) {
            static_cast<TickArg*>(arg)->context->schedule_issue_pass(0);
        },
        &tick_arg);
    queue->proceed();
    b3_expect(recorder.calls == 1 && recorder.tick_at_call == 1,
              "R7a same-tick: deferred pass runs at tick T after the main "
              "event");
    context.schedule_issue_pass(1);
    context.schedule_issue_pass(1);  // merged
    context.schedule_issue_pass(0);
    b3_expect(context.pending_issue_pass_count() == 2,
              "duplicate same-rank schedules merge into one pending pass");
    b3_expect(queue->finished() && queue->has_deferred_work(),
              "main map empty while deferred work pends");
    const uint64_t tick_before = queue->get_current_time();
    queue->schedule_event(tick_before + 1, noop_wakeup, nullptr);
    queue->proceed();
    b3_expect(recorder.calls == 3 &&
                  queue->get_current_time() == tick_before + 1 &&
                  queue->finished() && !queue->has_deferred_work(),
              "R7b bail-out: t+1 wakeup drains the merged passes");
    // Old golden: same_tick_issue_tick=1, tick_before_bail=1,
    // tick_after_bail=2, pass_calls_after_bail=3, ranks [0,1,0].
    std::printf("PASS R7-store same-tick pass + t+1 bail-out equal old "
                "golden (3 calls from 4 schedules)\n");
}

}  // namespace

// ===========================================================================
// Full-stack layer.
// ===========================================================================
namespace {

struct TerminalRecord {
    int rank = -1;
    uint64_t node_id = 0;
    uint64_t tick = 0;
};

struct WatchFireRecord {
    std::string request_id;
    std::string stage;
    uint64_t generation = 0;
    uint64_t tick = 0;
    uint64_t event_seq = 0;
};

std::string temporary_directory() {
    const std::string pattern = "/tmp/joint_ready_issue_order_XXXXXX";
    std::vector<char> buffer(pattern.begin(), pattern.end());
    buffer.push_back('\0');
    const char* const path = ::mkdtemp(buffer.data());
    if (path == nullptr) {
        b3_fail("mkdtemp failed");
    }
    return std::string(path);
}

void write_file(const std::string& path, const std::string& content) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) {
        b3_fail("cannot create fixture input " + path);
    }
    out << content;
    out.close();
    if (!out.good()) {
        b3_fail("failed writing fixture input " + path);
    }
}

void write_configs(const std::string& directory) {
    write_file(directory + "/system.json", R"({
  "scheduling-policy": "FIFO",
  "endpoint-delay": 1,
  "active-chunks-per-dimension": 1,
  "preferred-dataset-splits": 6,
  "all-reduce-implementation": ["ring", "ring"],
  "all-gather-implementation": ["ring", "ring"],
  "reduce-scatter-implementation": ["ring", "ring"],
  "all-to-all-implementation": ["ring", "ring"],
  "collective-optimization": "localBWAware",
  "roofline-enabled": 1,
  "replay-only": 0,
  "track-local-mem": 0,
  "trace-enabled": 0,
  "hbm-bandwidth-contention": 1,
  "peak-perf": 1000,
  "local-mem-bw": 6.0,
  "local-mem-latency": 10,
  "remote-mem-bw": 6.0,
  "remote-mem-latency": 10
})");
    write_file(directory + "/comm_group.json", "{}\n");
    write_file(directory + "/network.yml",
               "# seven-rank line: real multi-shard Sys/FluidScheduler path\n"
               "topology: [ Line, Line ]\n"
               "npus_count: [ 7, 1 ]\n"
               "bandwidth: [ 4000.0, 4000.0 ]\n"
               "latency: [ 5, 5 ]\n");
    write_file(directory + "/remote_memory.json", R"({
  "memory-type": "PER_NPU_MEMORY_EXPANSION",
  "remote-mem-bw": 6.0,
  "remote-mem-latency": 10,
  "npu-ids": [0, 1, 2, 3, 4, 5, 6]
})");
}

nlohmann::json contract_wire() {
    return nlohmann::json{
        {"projection_abi_version", 1},
        {"descriptor_version", 1},
        {"pipeline_protocol_version", 1},
        {"model_version", PipelineKernelContract::kModelVersion},
        {"timer_whitelist_version",
         PipelineKernelContract::kTimerWhitelistVersion},
        {"window_tile_limit", 1},
        {"window_local_error_ppm", 10000},
        {"window_refine_margin_ppm", 20000},
        {"prediction_config_sha256", std::string(64, 'a')}};
}

struct IssueRecord {
    uint64_t seq = 0;           // global monotone observation counter
    uint64_t event_seq = 0;     // global event counter at observation
    int rank = 0;
    uint64_t store_id = 0;
    uint64_t tick = 0;
    uint64_t pass = 0;          // per-rank issue-pass counter
};

struct FullStack {
    std::shared_ptr<EventQueue> event_queue;
    std::unique_ptr<AnalyticalRemoteMemory> memory_api;
    std::vector<std::unique_ptr<CongestionAwareNetworkApi>> network_apis;
    std::vector<std::shared_ptr<NodeStoreGraphSource>> graph_sources;
    std::vector<std::unique_ptr<Sys>> sys_owners;
    std::vector<Sys*> systems;
    std::vector<Workload*> workloads;
    std::shared_ptr<FluidScheduler> fluid_scheduler;
    WatchRegistry watch_registry;
    OnlineCompletionHookContext completion_context;
    std::unique_ptr<GraphBatchCommitter> committer;
    std::string directory;

    std::vector<IssueRecord> issues;
    std::vector<TerminalRecord> terminals;
    // Fire records carry the global event counter so ordering against
    // deferred issue passes is provable.
    struct FireRecord {
        std::string request_id;
        uint64_t event_seq = 0;
        uint64_t tick = 0;
    };
    std::vector<FireRecord> fires;
    std::vector<int> issue_rank_calls;
    uint64_t event_seq = 0;
    std::vector<uint64_t> pass_counter;
};

uint64_t next_event_seq(FullStack& stack) {
    return ++stack.event_seq;
}

void bump_pass(FullStack& stack, const int rank) {
    ++stack.pass_counter.at(static_cast<size_t>(rank));
}

// The deferred issue-pass callback: counts the pass, then runs the real
// Workload::issue_dep_free_nodes (the production wiring).
void fullstack_issue_pass(void* opaque, const int rank) {
    auto* stack = static_cast<FullStack*>(opaque);
    bump_pass(*stack, rank);
    (*stack->workloads.at(static_cast<size_t>(rank)))
        .issue_dep_free_nodes();
}

void fullstack_terminal_hook(void* opaque, const int rank,
                             const uint64_t node_id, const char* request_id,
                             const char* stage, const uint64_t generation,
                             const uint64_t tick, const int terminal_status) {
    auto* stack = static_cast<FullStack*>(opaque);
    stack->terminals.push_back(TerminalRecord{rank, node_id, tick});
    online_completion_hook(&stack->completion_context, rank, node_id,
                           request_id, stage, generation, tick,
                           terminal_status);
}

struct N6ReadyMutationHook {
    Workload* workload = nullptr;
    std::shared_ptr<NodeStoreGraphSource> replacement_source;
    std::weak_ptr<NodeStoreGraphSource> captured_source;
    int rank = -1;
    uint64_t trigger_id = 0;
    uint64_t stale_id = 0;
    bool fired = false;
    bool captured_source_alive_after_swap = false;
    bool copied_live_statistics = false;
    bool replacement_kept_generation_but_disabled_reuse = false;
    uint64_t replacement_membership_queries_at_copy = 0;
};

void n6_replace_source_and_retire_late_member(
    void* opaque, const int rank, const uint64_t node_id, const char*,
    const char*, const uint64_t, const uint64_t, const int terminal_status) {
    auto* hook = static_cast<N6ReadyMutationHook*>(opaque);
    if (hook == nullptr || hook->fired || rank != hook->rank ||
        node_id != hook->trigger_id) {
        return;
    }
    b3_expect(terminal_status == static_cast<int>(
                  NodeTerminalStatus::Skipped),
              "N6 mutation callback is the real Workload skipped terminal");
    // Workload::issue starts online statistics before it enters
    // skip_invalid. The terminal observer runs after mark_terminal_observed
    // and before skip_invalid calls finish_node and complete_online_statistics,
    // so clone the current source here to preserve that live start state.
    auto source_before_swap = hook->captured_source.lock();
    b3_expect(source_before_swap != nullptr &&
                  hook->workload->graph_source_.get() ==
                      source_before_swap.get(),
              "N6 callback still sees the exact source holding the live start");
    const NodeView* const live_trigger =
        source_before_swap->lookup_ptr(hook->trigger_id);
    b3_expect(live_trigger != nullptr &&
                  live_trigger->online_statistics.started &&
                  !live_trigger->online_statistics.completed,
              "N6 clone point is after statistics start and before completion");
    const uint64_t source_version =
        source_before_swap->store().free_id_version();
    hook->replacement_source =
        std::make_shared<NodeStoreGraphSource>(*source_before_swap);
    const NodeView* const copied_trigger =
        hook->replacement_source->lookup_ptr(hook->trigger_id);
    hook->copied_live_statistics = copied_trigger != nullptr &&
        copied_trigger->online_statistics.started &&
        !copied_trigger->online_statistics.completed;
    hook->replacement_kept_generation_but_disabled_reuse =
        hook->replacement_source->store().free_id_version() == source_version &&
        !hook->replacement_source->store().snapshot_reuse_eligible();
    hook->replacement_membership_queries_at_copy =
        hook->replacement_source->store().ready_set_stats().membership_queries;

    // Swap during the actual Workload terminal hook and retire a later ID in
    // the copied source. The next candidate must query this current source,
    // while the pass's strong owner keeps the captured source alive.
    hook->workload->graph_source_ = hook->replacement_source;
    source_before_swap.reset();
    hook->captured_source_alive_after_swap =
        !hook->captured_source.expired();
    hook->replacement_source->store().mark_issued(hook->stale_id);
    hook->replacement_source->store().finish_node(hook->stale_id);
    hook->fired = true;
}

void build_full_stack(FullStack& stack) {
    stack.directory = temporary_directory();
    std::filesystem::create_directories(stack.directory + "/workload");
    write_configs(stack.directory);

    stack.event_queue = std::make_shared<EventQueue>();
    const NetworkParser network_parser(stack.directory + "/network.yml");
    const auto topology = construct_topology(network_parser);
    CongestionAwareNetworkApi::set_event_queue(stack.event_queue);
    CongestionAwareNetworkApi::set_topology(topology);
    stack.fluid_scheduler = std::make_unique<FluidScheduler>(
        stack.event_queue, topology->get_directed_links(),
        network_parser.get_fluid_max_active_flows(),
        network_parser.get_fluid_max_route_memberships(),
        network_parser.get_progress_report_event_interval());
    stack.fluid_scheduler->set_deferred_flush_mode(true);
    CongestionAwareNetworkApi::set_fluid_scheduler(stack.fluid_scheduler);

    stack.memory_api = std::make_unique<AnalyticalRemoteMemory>(
        stack.directory + "/remote_memory.json");
    const auto npus_per_dim = network_parser.get_npus_counts_per_dim();
    std::vector<int> queues_per_dim(npus_per_dim.size(), 1);
    for (int rank = 0; rank < 7; ++rank) {
        stack.network_apis.push_back(
            std::make_unique<CongestionAwareNetworkApi>(rank));
        auto source = std::make_shared<NodeStoreGraphSource>();
        auto system = std::make_unique<Sys>(
            rank, stack.directory + "/workload",
            stack.directory + "/comm_group.json",
            stack.directory + "/system.json", stack.memory_api.get(),
            stack.network_apis.back().get(), npus_per_dim, queues_per_dim,
            1.0, 1.0, false, ExecutionMode::Online, source);
        stack.systems.push_back(system.get());
        stack.workloads.push_back(system->workload);
        stack.graph_sources.push_back(std::move(source));
        stack.sys_owners.push_back(std::move(system));
    }
    stack.pass_counter.assign(stack.workloads.size(), 0);
    stack.completion_context.registry = &stack.watch_registry;
    stack.completion_context.configure_issue_passes(
        stack.event_queue.get(), stack.workloads.size(),
        fullstack_issue_pass, &stack);

    CompletionObserver::instance().set_hook(&fullstack_terminal_hook,
                                            &stack);

    for (Workload* workload : stack.workloads) {
        workload->set_ledger_transition_observer(
            [&stack](const std::string& kind, const NodeView& node) {
                if (kind != "issue") {
                    return;
                }
                b3_expect(node.rank >= 0 &&
                              static_cast<size_t>(node.rank) <
                                  stack.workloads.size(),
                          "issue record carries a valid rank");
                IssueRecord record;
                record.seq = stack.issues.size();
                record.event_seq = next_event_seq(stack);
                record.rank = node.rank;
                record.store_id = node.global_id;
                record.tick = Sys::boostedTick();
                record.pass =
                    stack.pass_counter.at(static_cast<size_t>(node.rank));
                stack.issues.push_back(record);
            });
    }
    stack.watch_registry.set_fire_notifier(
        [&stack](const WatchFire& fire) {
            stack.fires.push_back(FullStack::FireRecord{
                fire.request_id, next_event_seq(stack),
                Sys::boostedTick()});
        });

    GraphBatchCommitter::Context ctx;
    ctx.num_ranks = 7;
    ctx.graph_sources = &stack.graph_sources;
    ctx.watch_registry = &stack.watch_registry;
    ctx.issue_rank = [&stack](const int rank) {
        stack.issue_rank_calls.push_back(rank);
        bump_pass(stack, rank);
        stack.workloads.at(static_cast<size_t>(rank))
            ->issue_dep_free_nodes();
    };
    ctx.commit_kernel_contract =
        [&stack](std::shared_ptr<const PipelineKernelContract> contract) {
            for (Workload* workload : stack.workloads) {
                workload->set_pipeline_kernel_contract(contract);
            }
        };
    stack.committer = std::make_unique<GraphBatchCommitter>(std::move(ctx));
}

void pump_until(FullStack& stack, const std::function<bool()>& done,
                const std::string& label) {
    uint64_t steps = 0;
    while (!done() && !stack.event_queue->finished()) {
        stack.event_queue->proceed();
        if (++steps > 1000000) {
            b3_fail(label + " exceeded event guard");
        }
    }
    b3_expect(done(), label + " completed before the real queue drained");
}

DecisionEvent arrival_event(const uint64_t seq, const std::string& request_id,
                            const uint64_t prefill_length = 1,
                            const uint64_t decode_length = 1) {
    DecisionEvent arrival;
    arrival.seq = seq;
    arrival.reason = DecisionReason::ARRIVAL;
    arrival.request_id = request_id;
    arrival.stage = "prefill";
    arrival.generation = 0;
    arrival.payload.session_id = "b3-ready-issue";
    arrival.payload.prefill_length = prefill_length;
    arrival.payload.decode_length = decode_length;
    return arrival;
}

// Plain schema-3 batch helpers: the same shapes the
// committer test uses, with CPU-op compute nodes so the real issue path
// schedules a replay event at +runtime_ns.
nlohmann::json plain_node(const int rank, const uint64_t id,
                          const std::string& name, const uint64_t runtime,
                          const std::string& request_id = "b3-r5") {
    return {
        {"rank", rank},       {"id", id},          {"type", 4},
        {"name", name},       {"is_cpu_op", true},
        {"is_timer_op", false}, {"inputs_values", ""},
        {"request_id", request_id}, {"stage", "prefill"}, {"generation", 0},
        {"compute", {{"num_ops", 1}, {"tensor_size", 8},
                     {"runtime_ns", runtime}}},
        {"comm",
         {{"bytes", 0}, {"src", 0}, {"dst", 0}, {"tag", 0}}},
        {"coll",
         {{"comm_type", 0},
          {"bytes", 0},
          {"priority", 0},
          {"pg_name", ""},
          {"involved_dim", nlohmann::json::array()}}},
    };
}

nlohmann::json plain_batch(const uint64_t delivery_sequence,
                           const std::vector<nlohmann::json>& nodes,
                           const std::vector<nlohmann::json>& edges = {},
                           const std::vector<nlohmann::json>& watches = {}) {
    nlohmann::json resp;
    resp["schema_version"] = 3;
    // The committer's ack rule: batch_id == source_delivery_sequence ==
    // the delivery epoch of the committing StateDelta.
    resp["batch_id"] = delivery_sequence;
    resp["source_delivery_sequence"] = delivery_sequence;
    resp["nodes"] = nodes;
    resp["parent_edges"] = edges;
    resp["watches"] = watches;
    resp["assignments"] = nlohmann::json::array();
    resp["kv_actions"] = nlohmann::json::array();
    resp["future_alarms"] = nlohmann::json::array();
    resp["kernel_contract"] = contract_wire();
    return resp;
}

std::vector<uint64_t> issue_ids(const FullStack& stack, const int rank) {
    std::vector<uint64_t> ids;
    for (const IssueRecord& record : stack.issues) {
        if (record.rank == rank) {
            ids.push_back(record.store_id);
        }
    }
    return ids;
}

size_t terminal_count(const FullStack& stack, const int rank,
                      const uint64_t node_id) {
    return static_cast<size_t>(std::count_if(
        stack.terminals.begin(), stack.terminals.end(),
        [rank, node_id](const TerminalRecord& record) {
            return record.rank == rank && record.node_id == node_id;
        }));
}

// ---------------------------------------------------------------------------
// R5 full stack: rank gate (fail-closed parse + committer preflight) and
// per-rank separation of the real issue sequences.
// ---------------------------------------------------------------------------
void scenario_r5_fullstack(FullStack& stack, const uint64_t rank0_next_json) {
    // (a) parse-level rank domain check fails closed.
    bool parse_rejected = false;
    try {
        const nlohmann::json bad =
            plain_batch(2, {plain_node(7, 0, "out_of_range", 10)});
        (void)parse_graph_batch(bad, 7);
    } catch (const ParseError&) {
        parse_rejected = true;
    }
    b3_expect(parse_rejected,
              "R5a node rank outside [0,num_ranks) fails closed at parse");

    // (b) committer preflight on a typed batch with a smuggled rank. The
    // typed batch is produced by the real parse (schema 2 mandates the
    // immutable kernel contract on every batch), then one node's rank is
    // forged outside [0, num_ranks) before the preflight.
    ParsedGraphBatch bad_batch = parse_graph_batch(
        plain_batch(2, {plain_node(0, 0, "smuggled", 10)}), 7);
    bad_batch.nodes[0].node.rank = 9;
    const std::vector<size_t> retained_before = {
        stack.graph_sources[0]->store().retained_count(),
        stack.graph_sources[1]->store().retained_count()};
    const std::vector<uint64_t> next_auto_before = {
        stack.graph_sources[0]->store().next_auto_id(),
        stack.graph_sources[1]->store().next_auto_id()};
    const auto bad_result = stack.committer->validate_and_commit(
        build_state_delta({arrival_event(2, "b3-r5")}, Sys::boostedTick(),
                          2),
        bad_batch);
    b3_expect(bad_result.error.has_value() &&
                  bad_result.error->find("rank out of range") !=
                      std::string::npos,
              "R5b committer preflight rejects the out-of-range rank: " +
                  bad_result.error.value_or(""));
    b3_expect(stack.graph_sources[0]->store().next_auto_id() ==
                  next_auto_before[0] &&
                  stack.graph_sources[1]->store().next_auto_id() ==
                      next_auto_before[1],
              "R5b rejected batch leaves every id stream untouched");
    b3_expect(stack.graph_sources[0]->store().retained_count() ==
                  retained_before[0] &&
                  stack.graph_sources[1]->store().retained_count() ==
                      retained_before[1],
              "R5b rejected batch leaves every store untouched (atomic)");

    // (c) legal two-rank batch: rank0 continues its json stream, rank1 is
    // fresh. One Data edge on rank0 chains its two nodes.
    const nlohmann::json legal = plain_batch(2,
        {plain_node(0, rank0_next_json, "r0_a", 10),
         plain_node(0, rank0_next_json + 1, "r0_b", 10),
         plain_node(1, 0, "r1_a", 10)},
        {{{"rank", 0},
          {"kind", "data"},
          {"from", rank0_next_json},
          {"to", rank0_next_json + 1}}});
    const ParsedGraphBatch batch = parse_graph_batch(legal, 7);
    const size_t issue_calls_before = stack.issue_rank_calls.size();
    const StateDelta delta = build_state_delta(
        {arrival_event(2, "b3-r5")}, Sys::boostedTick(), 2);
    const auto result = stack.committer->validate_and_commit(delta, batch);
    b3_expect(!result.error.has_value(),
              "R5c legal two-rank batch commits: " +
                  result.error.value_or(""));
    const std::vector<int> issue_rank_calls(
        stack.issue_rank_calls.begin() +
            static_cast<std::ptrdiff_t>(issue_calls_before),
        stack.issue_rank_calls.end());
    // Old golden recorded issue_rank_calls == [0, 1] (ascending touched).
    b3_expect(issue_rank_calls == std::vector<int>({0, 1}),
              "R5c commit issues the touched ranks ascending");
    const auto r0_a = stack.committer->resolve_store_id(0, rank0_next_json);
    const auto r0_b =
        stack.committer->resolve_store_id(0, rank0_next_json + 1);
    const auto r1_a = stack.committer->resolve_store_id(1, 0);
    b3_expect(r0_a.has_value() && r0_b.has_value() && r1_a.has_value(),
              "R5c committed ids resolve");
    b3_expect(issue_ids(stack, 0) == std::vector<uint64_t>({*r0_a}),
              "R5c rank0 initial sequence is exactly its first free node");
    b3_expect(issue_ids(stack, 1) == std::vector<uint64_t>({*r1_a}),
              "R5c rank1 initial sequence is exactly its own node");
    print_seq("R5/initial", Sys::boostedTick(), 0, {*r0_a});
    print_seq("R5/initial", Sys::boostedTick(), 1, {*r1_a});

    pump_until(stack, [&stack, r0_b]() {
                   return terminal_count(stack, 0, *r0_b) == 1;
               },
               "R5 two-rank chain completion");
    b3_expect(issue_ids(stack, 0) ==
                  std::vector<uint64_t>({*r0_a, *r0_b}),
              "R5c rank0 full sequence: chained node issued after its "
              "parent terminal");
    b3_expect(issue_ids(stack, 1) == std::vector<uint64_t>({*r1_a}),
              "R5c rank1 sequence unchanged");
    // (Store ids are per-rank namespaces; the old cross-rank disjointness
    // check was an artifact of the retired P0 fixture pre-advancing rank 0's
    // id stream and is not a semantic invariant.)
    print_seq("R5/final", Sys::boostedTick(), 0, {*r0_a, *r0_b});
    print_seq("R5/final", Sys::boostedTick(), 1, {*r1_a});
    b3_expect(terminal_count(stack, 0, *r0_a) == 1 &&
                  terminal_count(stack, 0, *r0_b) == 1 &&
                  terminal_count(stack, 1, *r1_a) == 1,
              "R5c every issued node reaches exactly one terminal");
    std::printf("PASS R5 rank gate: parse+preflight fail-closed, atomic; "
                "per-rank sequences disjoint and own-rank only\n");
}

// ---------------------------------------------------------------------------
// R6 + R7(a) full stack: watch fires synchronously at the last member's
// terminal inside the completion callback, does NOT issue synchronously;
// the released successor is issued by the deferred pass in the SAME tick,
// strictly after the callback returned.
// ---------------------------------------------------------------------------
struct ChainFacts {
    uint64_t a = 0;
    uint64_t b = 0;
    uint64_t c = 0;
    uint64_t a_issue_tick = 0;
    uint64_t a_terminal_tick = 0;
    uint64_t b_issue_tick = 0;
    uint64_t b_issue_seq = 0;
    uint64_t b_terminal_tick = 0;
    uint64_t c_issue_tick = 0;
    uint64_t fire_seq = 0;
    uint64_t fire_tick = 0;
};

ChainFacts scenario_r6_fullstack(FullStack& stack) {
    const nlohmann::json wire = plain_batch(3,
        {plain_node(3, 0, "a", 10, "b3-r6"),
         plain_node(3, 1, "b", 10, "b3-r6"),
         plain_node(3, 2, "c", 10, "b3-r6")},
        {{{"rank", 3}, {"kind", "data"}, {"from", 0}, {"to", 1}},
         {{"rank", 3}, {"kind", "data"}, {"from", 1}, {"to", 2}}},
        {{{"request_id", "b3-r6"},
          {"stage", "prefill"},
          {"generation", 0},
          {"members", {{"3", 0}}},
          {"statuses", nlohmann::json::array({"Success"})}}});
    const ParsedGraphBatch batch = parse_graph_batch(wire, 7);
    const StateDelta delta = build_state_delta(
        {arrival_event(3, "b3-r6")}, Sys::boostedTick(), 3);
    const auto result = stack.committer->validate_and_commit(delta, batch);
    b3_expect(!result.error.has_value(),
              "R6 chain batch commits: " + result.error.value_or(""));

    ChainFacts facts;
    facts.a = *stack.committer->resolve_store_id(3, 0);
    facts.b = *stack.committer->resolve_store_id(3, 1);
    facts.c = *stack.committer->resolve_store_id(3, 2);
    b3_expect(facts.a == 1 && facts.b == 2 && facts.c == 3,
              "R6 store ids follow the fresh rank-3 stream");
    b3_expect(issue_ids(stack, 3) == std::vector<uint64_t>({facts.a}),
              "R6 commit pass issues only the chain head");

    const size_t watch_fires_before = stack.fires.size();
    const auto a_record = std::find_if(
        stack.issues.begin(), stack.issues.end(),
        [&facts](const IssueRecord& r) {
            return r.rank == 3 && r.store_id == facts.a;
        });
    b3_expect(a_record != stack.issues.end(),
              "R6 chain head issued by the commit pass");
    facts.a_issue_tick = a_record->tick;

    pump_until(stack,
               [&stack, &facts]() {
                   return terminal_count(stack, 3, facts.c) == 1;
               },
               "R6 chain completion");
    for (const TerminalRecord& record : stack.terminals) {
        if (record.rank == 3 && record.node_id == facts.a) {
            facts.a_terminal_tick = record.tick;
        }
        if (record.rank == 3 && record.node_id == facts.b) {
            facts.b_terminal_tick = record.tick;
        }
    }
    for (const IssueRecord& record : stack.issues) {
        if (record.rank == 3 && record.store_id == facts.b) {
            facts.b_issue_tick = record.tick;
            facts.b_issue_seq = record.event_seq;
        }
        if (record.rank == 3 && record.store_id == facts.c) {
            facts.c_issue_tick = record.tick;
        }
    }
    size_t r6_fires = 0;
    for (size_t i = watch_fires_before; i < stack.fires.size(); ++i) {
        if (stack.fires.at(i).request_id == "b3-r6") {
            facts.fire_seq = stack.fires.at(i).event_seq;
            facts.fire_tick = stack.fires.at(i).tick;
            ++r6_fires;
        }
    }
    b3_expect(r6_fires == 1, "R6 watch fires exactly once");
    b3_expect(facts.a_terminal_tick == facts.a_issue_tick + 10,
              "R6 head completes after its replay runtime");
    b3_expect(facts.fire_tick == facts.a_terminal_tick,
              "R6 fire is synchronous inside the terminal callback (same "
              "tick as the head completion)");
    b3_expect(facts.b_issue_tick == facts.a_terminal_tick,
              "R7a deferred pass issues the successor at the SAME tick as "
              "the completion callback");
    b3_expect(facts.b_issue_seq > facts.fire_seq,
              "R6 successor issue happens strictly AFTER the fire notifier "
              "returned (no synchronous issue from the watch)");
    b3_expect(facts.c_issue_tick == facts.b_terminal_tick,
              "R7a second successor also issued same-tick");
    b3_expect(terminal_count(stack, 3, facts.a) == 1 &&
                  terminal_count(stack, 3, facts.b) == 1 &&
                  terminal_count(stack, 3, facts.c) == 1,
              "R6 every chain node reaches exactly one terminal");
    print_seq("R6", facts.a_issue_tick, 3, {facts.a});
    print_seq("R7a", facts.b_issue_tick, 3, {facts.b});
    print_seq("R7a", facts.c_issue_tick, 3, {facts.c});
    std::printf("PASS R6+R7a watch-gated deferred release: fire seq=%llu "
                "tick=%llu inside the terminal callback, successor issued "
                "same tick after callback return\n",
                static_cast<unsigned long long>(facts.fire_seq),
                static_cast<unsigned long long>(facts.fire_tick));
    return facts;
}

// R7(b) full stack: deferred-only residue drains via the production t+1
// bail-out; duplicate same-rank schedules merged (mirrors old golden).
void scenario_r7b_fullstack(FullStack& stack) {
    b3_expect(stack.event_queue->finished() &&
                  !stack.event_queue->has_deferred_work(),
              "R7b queue quiet before the bail-out probe");
    stack.completion_context.schedule_issue_pass(3);
    stack.completion_context.schedule_issue_pass(3);  // merged
    stack.completion_context.schedule_issue_pass(4);
    b3_expect(stack.completion_context.pending_issue_pass_count() == 2,
              "R7b three schedules merge into two pending passes");
    b3_expect(stack.event_queue->finished() &&
                  stack.event_queue->has_deferred_work(),
              "R7b main map empty while deferred work pends");
    const uint64_t tick_before = stack.event_queue->get_current_time();
    const uint64_t pass3 = stack.pass_counter[3];
    const uint64_t pass4 = stack.pass_counter[4];
    const auto timing3_before = stack.workloads[3]->dep_free_issue_timing();
    const auto timing4_before = stack.workloads[4]->dep_free_issue_timing();
    stack.event_queue->schedule_event(tick_before + 1, noop_wakeup, nullptr);
    stack.event_queue->proceed();
    b3_expect(stack.pass_counter[3] == pass3 + 1 &&
                  stack.pass_counter[4] == pass4 + 1,
              "R7b t+1 wakeup drains exactly one merged pass per rank");
    b3_expect(stack.event_queue->get_current_time() == tick_before + 1,
              "R7b wakeup advanced the clock by one tick");
    b3_expect(stack.event_queue->finished() &&
                  !stack.event_queue->has_deferred_work(),
              "R7b deferred residue fully drained");
    const auto timing3 = stack.workloads[3]->dep_free_issue_timing();
    const auto timing4 = stack.workloads[4]->dep_free_issue_timing();
    b3_expect(timing3.snapshot_passes == timing3_before.snapshot_passes + 1 &&
                  timing4.snapshot_passes == timing4_before.snapshot_passes + 1,
              "R7b drained passes each ran one snapshot");
    b3_expect(timing3.issue_calls == timing3_before.issue_calls &&
                  timing4.issue_calls == timing4_before.issue_calls,
              "R7b empty passes issue nothing");
    std::printf("PASS R7b t+1 bail-out: 3 schedules -> 2 merged passes -> "
                "drained\n");
}

// ---------------------------------------------------------------------------
// N3: the three benefit buckets (snapshot / issue / candidates+calls) are
// separated and populated on the real chain.
// ---------------------------------------------------------------------------
void scenario_n3_timing(FullStack& stack, const ChainFacts& facts,
                        const Workload::DepFreeIssueTiming& before) {
    const auto timing = stack.workloads[3]->dep_free_issue_timing();
    b3_expect(timing.snapshot_ns > before.snapshot_ns,
              "N3 snapshot bucket strictly grows (wall ns)");
    b3_expect(timing.issue_ns > before.issue_ns,
              "N3 issue bucket strictly grows (wall ns)");
    b3_expect(timing.snapshot_passes >= before.snapshot_passes + 5,
              "N3 snapshot passes cover commit + completion passes + bail");
    b3_expect(timing.issue_calls == before.issue_calls + 3,
              "N3 issue calls == the three chain nodes issued");
    b3_expect(timing.candidates == before.candidates + 3,
              "N3 candidates == the three chain nodes captured");
    b3_expect(facts.b > 0 && facts.c > 0, "N3 chain facts populated");
    std::printf("PASS N3 timing buckets: snapshot_passes +%llu, "
                "issue_calls +%llu, candidates +%llu (snapshot_ns and "
                "issue_ns both advanced)\n",
                static_cast<unsigned long long>(
                    timing.snapshot_passes - before.snapshot_passes),
                static_cast<unsigned long long>(
                    timing.issue_calls - before.issue_calls),
                static_cast<unsigned long long>(timing.candidates -
                                                before.candidates));
}

// ---------------------------------------------------------------------------
// N4: nested completion chain converges; total sequence == per-pass
// concatenation with every id exactly once.
// ---------------------------------------------------------------------------
void scenario_n4_fullstack(FullStack& stack) {
    std::vector<nlohmann::json> nodes;
    std::vector<nlohmann::json> edges;
    for (uint64_t i = 0; i < 5; ++i) {
        nodes.push_back(
            plain_node(4, i, "n" + std::to_string(i), 5, "b3-n4"));
        if (i > 0) {
            edges.push_back({{"rank", 4},
                             {"kind", "data"},
                             {"from", i - 1},
                             {"to", i}});
        }
    }
    const ParsedGraphBatch batch = parse_graph_batch(plain_batch(4, nodes, edges), 7);
    const StateDelta delta = build_state_delta(
        {arrival_event(4, "b3-n4")}, Sys::boostedTick(), 4);
    const auto result = stack.committer->validate_and_commit(delta, batch);
    b3_expect(!result.error.has_value(),
              "N4 chain batch commits: " + result.error.value_or(""));
    const uint64_t last = *stack.committer->resolve_store_id(4, 4);
    pump_until(stack,
               [&stack, last]() {
                   return terminal_count(stack, 4, last) == 1;
               },
               "N4 five-deep chain completion");
    const std::vector<uint64_t> expected{1, 2, 3, 4, 5};
    b3_expect(issue_ids(stack, 4) == expected,
              "N4 total issue sequence == ascending chain, every id once");
    print_seq("N4", Sys::boostedTick(), 4, issue_ids(stack, 4));
    for (uint64_t id = 1; id <= 5; ++id) {
        b3_expect(terminal_count(stack, 4, id) == 1,
                  "N4 chain node " + std::to_string(id) +
                      " reaches exactly one terminal");
    }
    b3_expect(stack.event_queue->finished() &&
                  !stack.event_queue->has_deferred_work(),
              "N4 chain fully converged: queue drained, no deferred left");
    std::printf("PASS N4 completion chain converges in finite passes; "
                "sequence [1,2,3,4,5], each issued and terminal once\n");
}

// N5: the serialized GPU-COMP gate keeps ready physical nodes in NodeStore.
// Three retries must still examine every candidate, but must not materialize
// NodeViews until a candidate passes the read-only resource gate.
void scenario_n5_blocked_node_view_materialization(FullStack& stack) {
    const int rank = kOwnerRank;
    Workload* const workload = stack.workloads.at(static_cast<size_t>(rank));
    NodeStore& store =
        stack.graph_sources.at(static_cast<size_t>(rank))->store();
    b3_expect(stack.event_queue->finished() &&
                  !stack.event_queue->has_deferred_work(),
              "N5 starts at a drained point in the three-rank/two-tile run");
    b3_expect(workload->hw_resource->num_in_flight_gpu_comp_ops == 0,
              "N5 begins with an idle GPU-COMP gate");
    b3_expect(store.resolve_free_nodes().empty(),
              "N5 prior pipeline nodes are all terminal before the scale rows");
    workload->issue_dep_free_nodes();

    for (const size_t count : {size_t{8}, size_t{32}, size_t{128}}) {
        OnlineNode blocker;
        blocker.rank = rank;
        blocker.kind = NodeKind::Compute;
        blocker.node_type = 4;
        workload->hw_resource->occupy(blocker);

        std::vector<uint64_t> ids;
        ids.reserve(count);
        for (size_t index = 0; index < count; ++index) {
            OnlineNode node;
            node.rank = rank;
            node.kind = NodeKind::Compute;
            node.node_type = 4;
            node.name = "layer_17/mlp_up/tile_" + std::to_string(index);
            node.is_cpu_op = false;
            node.request_id = "request-copy-gate-01";
            node.stage = "prefill";
            node.compute.num_ops = 1;
            node.compute.tensor_size = 8;
            // The blocked passes never dispatch this work. Once the resource
            // opens, the one-nanosecond fixture work drains the small case.
            node.compute.runtime_ns = 1;
            ids.push_back(store.add_node(std::move(node)));
        }

        const auto timing_before = workload->dep_free_issue_timing();
        const NodeStore::ReadySetStats ready_before = store.ready_set_stats();
        constexpr uint64_t kBlockedPasses = 3;
        for (uint64_t pass = 0; pass < kBlockedPasses; ++pass) {
            workload->issue_dep_free_nodes();
        }
        const auto timing_blocked = workload->dep_free_issue_timing();
        const NodeStore::ReadySetStats ready_blocked = store.ready_set_stats();
        const uint64_t blocked_candidates = kBlockedPasses * count;
        b3_expect(timing_blocked.snapshot_passes ==
                      timing_before.snapshot_passes + kBlockedPasses,
                  "N5 each held-GPU retry captures one issue pass");
        // v11 Phase 4: the first pass rebuilds the ready vector by SPLICING
        // the since-snapshot delta (the 128/32/8 adds) onto the cached
        // vector instead of rewalking the ordered set; the unchanged
        // generation then reuses the spliced vector for passes 2..N.

        b3_expect(timing_blocked.snapshot_full_fills ==
                          timing_before.snapshot_full_fills &&
                      timing_blocked.snapshot_splices ==
                          timing_before.snapshot_splices + 1 &&
                      timing_blocked.snapshot_reuse_hits ==
                          timing_before.snapshot_reuse_hits +
                              kBlockedPasses - 1,
                  "N5 unchanged NodeStore generation reuses the spliced ID set");
        b3_expect(timing_blocked.candidates ==
                      timing_before.candidates + blocked_candidates,
                  "N5 every held-GPU pass still examines its full ready batch");
        b3_expect(timing_blocked.issue_calls == timing_before.issue_calls &&
                      timing_blocked.node_view_materializations ==
                          timing_before.node_view_materializations,
                  "N5 blocked candidates are checked without NodeView copies");
        // F (2026-10-11, dynamic current-anchor candidate set) + v11 Phase 4:
        // the rebuild was a delta splice (no ordered-set traversal, no ID
        // copy pass), the spliced vector still holds every blocked id
        // ascending, and the proof-covered walks issue no membership probe.
        {
            const std::vector<uint64_t>& captured =
                WorkloadTestAccess::snapshot_ids(*workload);
            const bool carries_batch =
                captured.size() >= count &&
                std::vector<uint64_t>(captured.end() -
                                          static_cast<long>(count),
                                      captured.end()) == ids;
            b3_expect(carries_batch,
                      "N5 spliced vector carries the full blocked batch ascending");
        }
        b3_expect(ready_blocked.snapshot_calls ==
                      ready_before.snapshot_calls &&
                      ready_blocked.snapshot_ids_copied ==
                          ready_before.snapshot_ids_copied &&
                      ready_blocked.splice_calls ==
                          ready_before.splice_calls + 1 &&
                      ready_blocked.membership_queries ==
                          ready_before.membership_queries,
                  "N5 splices the delta and its unchanged-generation "
                  "membership proof still holds");

        workload->hw_resource->release(blocker);
        const size_t issue_record_begin = stack.issues.size();
        workload->issue_dep_free_nodes();
        const auto timing_after_first = workload->dep_free_issue_timing();
        b3_expect(timing_after_first.snapshot_passes ==
                      timing_blocked.snapshot_passes + 1 &&
                      timing_after_first.candidates ==
                          timing_blocked.candidates + count,
                  "N5 release pass still walks the complete ready batch");
        b3_expect(timing_after_first.issue_calls ==
                      timing_blocked.issue_calls + 1 &&
                      timing_after_first.node_view_materializations ==
                          timing_blocked.node_view_materializations + 1,
                  "N5 release materializes and issues only the first candidate");
        b3_expect(timing_after_first.snapshot_full_fills ==
                          timing_blocked.snapshot_full_fills &&
                      timing_after_first.snapshot_reuse_hits ==
                          timing_blocked.snapshot_reuse_hits + 1,
                  "N5 hardware release reuses IDs before the first ready-set mutation");
        b3_expect(workload->hw_resource->num_in_flight_gpu_comp_ops == 1,
                  "N5 first issue occupies the serialized GPU-COMP gate");

        std::vector<uint64_t> issued_since_release;
        for (size_t issue_index = issue_record_begin;
             issue_index < stack.issues.size(); ++issue_index) {
            const IssueRecord& record = stack.issues[issue_index];
            if (record.rank == rank) {
                issued_since_release.push_back(record.store_id);
            }
        }
        b3_expect(issued_since_release ==
                      std::vector<uint64_t>{ids.front()},
                  "N5 the lowest ready id wins the first resource slot");
        for (size_t index = 1; index < ids.size(); ++index) {
            b3_expect(store.is_free_id(ids[index]),
                      "N5 later ids remain ready behind the occupied COMP slot");
        }

        pump_until(
            stack,
            [&stack, &ids, rank]() {
                return std::all_of(ids.begin(), ids.end(),
                                   [&stack, rank](const uint64_t id) {
                                       return terminal_count(stack, rank, id) == 1;
                                   });
            },
            "N5 small GPU-COMP batch drain");
        if (stack.event_queue->has_deferred_work()) {
            const uint64_t tick = stack.event_queue->get_current_time();
            stack.event_queue->schedule_event(tick + 1, noop_wakeup, nullptr);
            stack.event_queue->proceed();
        }
        b3_expect(stack.event_queue->finished() &&
                      !stack.event_queue->has_deferred_work(),
                  "N5 callbacks and deferred issue passes are drained");
        const auto timing_drained = workload->dep_free_issue_timing();
        b3_expect(timing_drained.issue_calls ==
                      timing_before.issue_calls + count &&
                      timing_drained.node_view_materializations ==
                          timing_before.node_view_materializations + count,
                  "N5 one physical NodeView copy accompanies each real issue");

        std::vector<uint64_t> issued_all;
        for (size_t issue_index = issue_record_begin;
             issue_index < stack.issues.size(); ++issue_index) {
            const IssueRecord& record = stack.issues[issue_index];
            if (record.rank == rank) {
                issued_all.push_back(record.store_id);
            }
        }
        b3_expect(issued_all == ids,
                  "N5 deferred callbacks preserve ascending issue order");
        b3_expect(workload->hw_resource->num_in_flight_gpu_comp_ops == 0,
                  "N5 all GPU-COMP occupancy drains");
        std::printf(
            "PASS N5 blocked-copy scale N=%zu held_passes=%llu candidates=%llu "
            "full_fills=%llu splices=%llu reuse_hits=%llu "
            "actual_blocked_copies=0 release_reuse=%llu\n",
            count, static_cast<unsigned long long>(kBlockedPasses),
            static_cast<unsigned long long>(blocked_candidates),
            static_cast<unsigned long long>(
                timing_blocked.snapshot_full_fills -
                timing_before.snapshot_full_fills),
            static_cast<unsigned long long>(
                timing_blocked.snapshot_splices -
                timing_before.snapshot_splices),
            static_cast<unsigned long long>(
                timing_blocked.snapshot_reuse_hits -
                timing_before.snapshot_reuse_hits),
            static_cast<unsigned long long>(
                timing_after_first.snapshot_reuse_hits -
                timing_blocked.snapshot_reuse_hits));
    }
}

void scenario_n6_node_store_copy_move_eligibility() {
    static_assert(std::is_copy_constructible<NodeStore>::value,
                  "NodeStore remains copy constructible");
    static_assert(std::is_copy_assignable<NodeStore>::value,
                  "NodeStore remains copy assignable");
    static_assert(std::is_move_constructible<NodeStore>::value,
                  "NodeStore remains move constructible");
    static_assert(std::is_move_assignable<NodeStore>::value,
                  "NodeStore remains move assignable");

    OnlineNode node = b3_node(0, "eligibility-node");
    NodeStore original;
    original.add_node(node);
    NodeStore copy_constructed(original);
    b3_expect(original.snapshot_reuse_eligible() &&
                  !copy_constructed.snapshot_reuse_eligible(),
              "N6 copy construction disables only the copy's reuse");

    NodeStore copy_assigned;
    copy_assigned = original;
    b3_expect(original.snapshot_reuse_eligible() &&
                  !copy_assigned.snapshot_reuse_eligible(),
              "N6 copy assignment disables only the destination's reuse");

    NodeStore move_source;
    move_source.add_node(node);
    NodeStore move_constructed(std::move(move_source));
    b3_expect(!move_source.snapshot_reuse_eligible() &&
                  !move_constructed.snapshot_reuse_eligible(),
              "N6 move construction disables both stores' reuse");

    NodeStore move_assign_source;
    move_assign_source.add_node(node);
    NodeStore move_assigned;
    move_assigned = std::move(move_assign_source);
    b3_expect(!move_assign_source.snapshot_reuse_eligible() &&
                  !move_assigned.snapshot_reuse_eligible(),
              "N6 move assignment disables both stores' reuse");

    NodeStore swap_left;
    NodeStore swap_right;
    std::swap(swap_left, swap_right);
    b3_expect(!swap_left.snapshot_reuse_eligible() &&
                  !swap_right.snapshot_reuse_eligible(),
              "N6 default store swap disables both stores' reuse");
}

uint64_t n6_payload_signature(NodeStoreGraphSource& source,
                              const std::vector<uint64_t>& ids,
                              uint64_t* payload_bytes) {
    uint64_t signature = 14695981039346656037ULL;
    uint64_t bytes = 0;
    for (const uint64_t id : ids) {
        const NodeView* const node = source.lookup_ptr(id);
        b3_expect(node != nullptr,
                  "membership probe keeps every ready NodeView resident");
        signature ^= id;
        signature *= 1099511628211ULL;
        signature ^= static_cast<uint64_t>(node->kind);
        signature *= 1099511628211ULL;
        bytes += sizeof(node->global_id) + node->name.size();
        for (const char raw_ch : node->name) {
            const auto ch = static_cast<unsigned char>(raw_ch);
            signature ^= ch;
            signature *= 1099511628211ULL;
        }
    }
    *payload_bytes = bytes;
    return signature;
}

// Opt-in read-only-on-the-queue cost probe. Run this same fixture against the
// archived and new libAstraSim.a; the query count is deliberately printed,
// not asserted to a canned old/new value. All nodes remain ready behind a
// held GPU-COMP slot, so these are actual Workload passes with no event-queue
// advance or physics completion.
void scenario_membership_cost_probe(FullStack& stack) {
    const int rank = kOwnerRank;
    Workload* const workload = stack.workloads.at(static_cast<size_t>(rank));
    const auto original_source = workload->graph_source_;
    const uint64_t initial_tick = stack.event_queue->get_current_time();
    const size_t initial_issue_count = stack.issues.size();

    for (const size_t n : {size_t{8}, size_t{32}, size_t{128}}) {
        auto source = std::make_shared<NodeStoreGraphSource>();
        std::vector<uint64_t> expected_ids;
        expected_ids.reserve(n);
        uint64_t payload_bytes_before = 0;
        for (size_t index = 0; index < n; ++index) {
            OnlineNode node = b3_node(
                0, "membership-probe-" + std::to_string(index));
            node.rank = rank;
            expected_ids.push_back(source->store().add_node(std::move(node)));
        }
        const uint64_t payload_before = n6_payload_signature(
            *source, expected_ids, &payload_bytes_before);

        OnlineNode blocker = b3_node(0, "membership probe GPU-COMP hold");
        blocker.rank = rank;
        workload->hw_resource->occupy(blocker);
        workload->graph_source_ = source;
        const auto timing_before = workload->dep_free_issue_timing();
        const NodeStore::ReadySetStats ready_before =
            source->store().ready_set_stats();
        const uint64_t tick_before = stack.event_queue->get_current_time();
        const bool queue_finished_before = stack.event_queue->finished();
        const bool deferred_before = stack.event_queue->has_deferred_work();

        for (size_t pass = 0; pass < 3; ++pass) {
            workload->issue_dep_free_nodes();
        }

        const auto timing_after = workload->dep_free_issue_timing();
        const NodeStore::ReadySetStats ready_after =
            source->store().ready_set_stats();
        const uint64_t actual_queries =
            ready_after.membership_queries - ready_before.membership_queries;
        const uint64_t actual_candidates =
            timing_after.candidates - timing_before.candidates;
        // F (2026-10-11): the probe source is an exact NodeStoreGraphSource,
        // so the pass copies its ORDINARY partition (identical to the full
        // set here -- the probe rows have no anchor-role nodes).
        const uint64_t actual_snapshot_copies =
            ready_after.snapshot_ids_copied -
            ready_before.snapshot_ids_copied;
        const uint64_t actual_full_fills =
            timing_after.snapshot_full_fills -
            timing_before.snapshot_full_fills;
        const uint64_t actual_reuse_hits =
            timing_after.snapshot_reuse_hits -
            timing_before.snapshot_reuse_hits;
        uint64_t payload_bytes_after = 0;
        const uint64_t payload_after = n6_payload_signature(
            *source, expected_ids, &payload_bytes_after);
        const std::vector<uint64_t> final_ready =
            source->store().resolve_free_nodes();
        const bool ordered_snapshot_matches =
            WorkloadTestAccess::snapshot_ids(*workload) == expected_ids;
        const bool queue_unchanged =
            stack.event_queue->get_current_time() == tick_before &&
            stack.event_queue->finished() == queue_finished_before &&
            stack.event_queue->has_deferred_work() == deferred_before;

        b3_expect(actual_candidates == 3 * n &&
                      timing_after.snapshot_full_fills ==
                          timing_before.snapshot_full_fills + 1 &&
                      timing_after.snapshot_reuse_hits ==
                          timing_before.snapshot_reuse_hits + 2 &&
                      actual_snapshot_copies == n &&
                      timing_after.issue_calls == timing_before.issue_calls &&
                      final_ready == expected_ids && ordered_snapshot_matches &&
                      payload_before == payload_after &&
                      payload_bytes_before == payload_bytes_after &&
                      queue_unchanged &&
                      stack.event_queue->get_current_time() == initial_tick &&
                      stack.issues.size() == initial_issue_count &&
                      workload->hw_resource->num_in_flight_gpu_comp_ops == 1,
                  "membership cost probe preserves payload, ready order, and "
                  "no-advance state");

        std::printf(
            "membership-cost N=%zu candidates=%llu membership_queries=%llu "
            "snapshot_copied=%llu payload_bytes=%llu payload_hash=%llu "
            "fullfills=%llu reuse_hits=%llu ready_exact=1 "
            "snapshot_order_exact=1 tick_unchanged=1\n",
            n, static_cast<unsigned long long>(actual_candidates),
            static_cast<unsigned long long>(actual_queries),
            static_cast<unsigned long long>(actual_snapshot_copies),
            static_cast<unsigned long long>(payload_bytes_before),
            static_cast<unsigned long long>(payload_before),
            static_cast<unsigned long long>(actual_full_fills),
            static_cast<unsigned long long>(actual_reuse_hits));

        workload->graph_source_ = original_source;
        workload->hw_resource->release(blocker);
        b3_expect(workload->hw_resource->num_in_flight_gpu_comp_ops == 0,
                  "membership cost probe releases its held resource");
    }
}

// N6: the existing ordered ID vector is reusable only for the same source and
// the same unsaturated ready-set generation. Held-resource cases keep ready
// membership stable; one separate real Workload terminal callback replaces
// the source and makes a later captured ID stale without advancing the queue.
void scenario_n6_snapshot_generation_lifecycle(FullStack& stack) {
    scenario_n6_node_store_copy_move_eligibility();
    const int rank = kOwnerRank;
    Workload* const workload = stack.workloads.at(static_cast<size_t>(rank));
    b3_expect(stack.event_queue->finished() &&
                  !stack.event_queue->has_deferred_work() &&
                  workload->hw_resource->num_in_flight_gpu_comp_ops == 0,
              "N6 starts from a drained stack with an idle GPU-COMP gate");

    OnlineNode blocker = b3_node(0, "N6 held GPU-COMP blocker");
    blocker.rank = rank;
    workload->hw_resource->occupy(blocker);
    const auto original_source = workload->graph_source_;
    NodeStore& original_store =
        stack.graph_sources.at(static_cast<size_t>(rank))->store();
    b3_expect(original_store.resolve_free_nodes().empty(),
              "N6 original store has no ready nodes during source swaps");

    const auto make_source = [rank](const std::vector<uint64_t>& ids) {
        auto source = std::make_shared<NodeStoreGraphSource>();
        for (const uint64_t id : ids) {
            OnlineNode node = b3_node(id, "N6 blocked compute");
            node.rank = rank;
            source->store().add_node(std::move(node));
        }
        return source;
    };

    const auto timing_before = workload->dep_free_issue_timing();
    auto source_a = make_source({7001});  // generation 1
    workload->graph_source_ = source_a;
    workload->issue_dep_free_nodes();
    auto timing = workload->dep_free_issue_timing();
    // F (2026-10-11): exact NodeStore sources are snapshotted through the
    // ordinary (non-anchor) partition fill; these fixture rows contain no
    // anchors, so the ordinary counters mirror the former full-snapshot
    // counts one-for-one.
    b3_expect(timing.snapshot_full_fills ==
                      timing_before.snapshot_full_fills + 1 &&
                  timing.snapshot_reuse_hits ==
                      timing_before.snapshot_reuse_hits &&
                  source_a->store().ready_set_stats().snapshot_calls ==
                      1 &&
                  source_a->store().ready_set_stats()
                      .snapshot_ids_copied == 1 &&
                  source_a->store().ready_set_stats().membership_queries == 0,
              "N6 first source call captures one ID and trusts its exact snapshot");

    workload->issue_dep_free_nodes();
    timing = workload->dep_free_issue_timing();
    b3_expect(timing.snapshot_full_fills ==
                      timing_before.snapshot_full_fills + 1 &&
                  timing.snapshot_reuse_hits ==
                      timing_before.snapshot_reuse_hits + 1 &&
                  source_a->store().ready_set_stats().snapshot_calls ==
                      1 &&
                  source_a->store().ready_set_stats().membership_queries == 0,
              "N6 unchanged source/generation reuses IDs and membership proof");

    OnlineNode second_a = b3_node(7002, "N6 second blocked compute");
    second_a.rank = rank;
    source_a->store().add_node(std::move(second_a));  // generation 2
    workload->issue_dep_free_nodes();
    timing = workload->dep_free_issue_timing();
    // v11 Phase 4: the generation bump splices the since-snapshot delta
    // (one new id) onto the cached vector -- exactly one full fill total
    // (pass 1), one splice here, and the ordinary counters only count the
    // pass-1 fill's single ID copy.
    b3_expect(timing.snapshot_full_fills ==
                      timing_before.snapshot_full_fills + 1 &&
                  timing.snapshot_splices ==
                      timing_before.snapshot_splices + 1 &&
                  timing.snapshot_reuse_hits ==
                      timing_before.snapshot_reuse_hits + 1 &&
                  timing.candidates >= timing_before.candidates + 4 &&
                  source_a->store().ready_set_stats().snapshot_calls ==
                      1 &&
                  source_a->store().ready_set_stats()
                          .snapshot_ids_copied == 1 &&
                  source_a->store().ready_set_stats().splice_calls ==
                      1 &&
                  source_a->store().ready_set_stats().membership_queries == 0,
              "N6 a ready-set mutation splices the new generation");

    auto source_a_copy = std::make_shared<NodeStoreGraphSource>(*source_a);
    const NodeStore::ReadySetStats copied_stats_before =
        source_a_copy->store().ready_set_stats();
    b3_expect(source_a->store().snapshot_reuse_eligible() &&
                  !source_a_copy->store().snapshot_reuse_eligible() &&
                  source_a_copy->store().free_id_version() ==
                      source_a->store().free_id_version(),
              "N6 source copy preserves the original generation but disables its copy");
    workload->graph_source_ = source_a_copy;
    const auto timing_before_source_copy = workload->dep_free_issue_timing();
    workload->issue_dep_free_nodes();
    workload->issue_dep_free_nodes();
    timing = workload->dep_free_issue_timing();
    b3_expect(timing.snapshot_full_fills ==
                      timing_before_source_copy.snapshot_full_fills + 2 &&
                  timing.snapshot_reuse_hits ==
                      timing_before_source_copy.snapshot_reuse_hits &&
                  timing.candidates == timing_before_source_copy.candidates + 4 &&
                  source_a_copy->store().ready_set_stats()
                      .snapshot_calls ==
                      copied_stats_before.snapshot_calls + 2 &&
                  source_a_copy->store().ready_set_stats()
                      .snapshot_ids_copied ==
                      copied_stats_before.snapshot_ids_copied + 4 &&
                  source_a_copy->store().ready_set_stats().membership_queries ==
                      copied_stats_before.membership_queries + 4,
              "N6 copied source performs a full fill on each unchanged generation");

    workload->graph_source_ = source_a;
    const auto timing_before_source_restore = workload->dep_free_issue_timing();
    const NodeStore::ReadySetStats source_a_stats_before_restore =
        source_a->store().ready_set_stats();
    workload->issue_dep_free_nodes();
    workload->issue_dep_free_nodes();
    timing = workload->dep_free_issue_timing();
    b3_expect(timing.snapshot_full_fills ==
                      timing_before_source_restore.snapshot_full_fills + 1 &&
                  timing.snapshot_reuse_hits ==
                      timing_before_source_restore.snapshot_reuse_hits + 1 &&
                  source_a->store().snapshot_reuse_eligible() &&
                  source_a->store().ready_set_stats().snapshot_calls ==
                      source_a_stats_before_restore.snapshot_calls +
                          1 &&
                  source_a->store().ready_set_stats()
                      .snapshot_ids_copied ==
                      source_a_stats_before_restore.snapshot_ids_copied +
                          2 &&
                  source_a->store().ready_set_stats().membership_queries ==
                      source_a_stats_before_restore.membership_queries,
              "N6 the uncopied source remains eligible and reuses membership proof");

    auto assignment_source = make_source({7101, 7102});  // generation 2
    const uint64_t assignment_generation_before =
        source_a->store().free_id_version();
    b3_expect(assignment_source->store().free_id_version() ==
                  assignment_generation_before &&
                  assignment_source->store().snapshot_reuse_eligible(),
              "N6 assignment fixture has the same generation and is eligible");
    const auto timing_before_store_assignment =
        workload->dep_free_issue_timing();
    source_a->store() = assignment_source->store();
    b3_expect(!source_a->store().snapshot_reuse_eligible() &&
                  assignment_source->store().snapshot_reuse_eligible() &&
                  source_a->store().free_id_version() ==
                      assignment_generation_before,
              "N6 copy assignment retains version but disables target reuse");
    workload->issue_dep_free_nodes();
    workload->issue_dep_free_nodes();
    timing = workload->dep_free_issue_timing();
    b3_expect(timing.snapshot_full_fills ==
                      timing_before_store_assignment.snapshot_full_fills + 2 &&
                  timing.snapshot_reuse_hits ==
                      timing_before_store_assignment.snapshot_reuse_hits &&
                  timing.candidates ==
                      timing_before_store_assignment.candidates + 4 &&
                  source_a->store().ready_set_stats().snapshot_calls ==
                      2 &&
                  source_a->store().ready_set_stats()
                      .snapshot_ids_copied == 4 &&
                  source_a->store().ready_set_stats().membership_queries == 4,
              "N6 same-source/same-generation assignment refreshes both candidates");

    // Same version, different source identity and different IDs must not
    // reuse source_a's vector.
    auto source_b = make_source({8001, 8002});  // generation 2
    b3_expect(source_b->store().free_id_version() ==
                  source_a->store().free_id_version(),
              "N6 replacement sources deliberately share a generation");
    workload->graph_source_ = source_b;
    const auto timing_before_source_b = workload->dep_free_issue_timing();
    workload->issue_dep_free_nodes();
    timing = workload->dep_free_issue_timing();
    b3_expect(timing.snapshot_full_fills ==
                      timing_before_source_b.snapshot_full_fills + 1 &&
                  timing.snapshot_reuse_hits ==
                      timing_before_source_b.snapshot_reuse_hits &&
                  timing.candidates == timing_before_source_b.candidates + 2 &&
                  source_b->store().ready_set_stats().snapshot_calls ==
                      1 &&
                  source_b->store().ready_set_stats()
                      .snapshot_ids_copied == 2 &&
                  source_b->store().ready_set_stats().membership_queries == 0 &&
                  WorkloadTestAccess::snapshot_ids(*workload) ==
                      std::vector<uint64_t>({8001, 8002}),
              "N6 same-generation source replacement fills and proves its own IDs");

    // Exercise the sticky proof invalidation through the actual Workload
    // terminal callback, not the store-level run_new_issue_loop oracle. The
    // callback swaps to a same-generation NodeStore copy and retires the
    // later captured ID before the production loop reaches it.
    const auto invalid_timer_node = [rank](const std::string& name) {
        OnlineNode node;
        node.kind = NodeKind::Invalid;
        node.node_type = 0;  // Chakra INVALID_NODE
        node.is_timer_op = true;  // NoOp resource class; no occupancy leak
        node.name = name;
        node.rank = rank;
        return node;
    };
    auto callback_source = std::make_shared<NodeStoreGraphSource>();
    const uint64_t callback_trigger = callback_source->store().add_node(
        invalid_timer_node("N6 synchronous terminal trigger"));
    const uint64_t callback_stale = callback_source->store().add_node(
        invalid_timer_node("N6 callback-retired later candidate"));
    std::weak_ptr<NodeStoreGraphSource> callback_source_weak = callback_source;
    N6ReadyMutationHook mutation_hook;
    mutation_hook.workload = workload;
    mutation_hook.captured_source = callback_source_weak;
    mutation_hook.rank = rank;
    mutation_hook.trigger_id = callback_trigger;
    mutation_hook.stale_id = callback_stale;
    const auto timing_before_callback = workload->dep_free_issue_timing();
    const uint64_t tick_before_callback =
        stack.event_queue->get_current_time();
    const size_t issues_before_callback = stack.issues.size();
    workload->graph_source_ = callback_source;
    callback_source.reset();  // only the active source / pass may now own it
    CompletionObserver::instance().set_hook(
        &n6_replace_source_and_retire_late_member, &mutation_hook);
    workload->issue_dep_free_nodes();
    CompletionObserver::instance().set_hook(&fullstack_terminal_hook, &stack);
    timing = workload->dep_free_issue_timing();
    const auto callback_replacement = mutation_hook.replacement_source;
    const NodeView* const replacement_trigger =
        callback_replacement != nullptr
            ? callback_replacement->lookup_ptr(callback_trigger)
            : nullptr;
    const NodeView* const replacement_stale =
        callback_replacement != nullptr
            ? callback_replacement->lookup_ptr(callback_stale)
            : nullptr;
    const bool replacement_trigger_completed =
        replacement_trigger != nullptr &&
        replacement_trigger->online_statistics.started &&
        replacement_trigger->online_statistics.completed;
    const uint64_t replacement_queries_after =
        callback_replacement != nullptr
            ? callback_replacement->store().ready_set_stats().membership_queries
            : 0;
    b3_expect(mutation_hook.fired &&
                  mutation_hook.captured_source_alive_after_swap &&
                  mutation_hook.copied_live_statistics &&
                  mutation_hook.replacement_kept_generation_but_disabled_reuse &&
                  replacement_trigger_completed &&
                  timing.candidates == timing_before_callback.candidates + 2 &&
                  timing.issue_calls == timing_before_callback.issue_calls + 1 &&
                  replacement_queries_after ==
                      mutation_hook.replacement_membership_queries_at_copy + 1 &&
                  mutation_hook.replacement_membership_queries_at_copy == 0 &&
                  callback_replacement != nullptr &&
                  callback_replacement->store().is_finished_or_erased(
                      callback_stale) &&
                  replacement_stale != nullptr &&
                  stack.event_queue->get_current_time() ==
                      tick_before_callback &&
                  stack.issues.size() == issues_before_callback + 1 &&
                  stack.issues.back().store_id == callback_trigger,
              "N6 real Workload callback invalidates and rejects a late ID "
              "through the replacement source");

    // A derived NodeStore source can decline the ID snapshot contract. Its
    // real Workload path must stay on the existing NodeView compatibility
    // traversal and must never inherit an exact-store membership proof.
    auto false_source =
        std::make_shared<FalseSnapshotNodeStoreGraphSource>();
    for (const char* name : {"N6 false-fill candidate 1",
                             "N6 false-fill candidate 2"}) {
        OnlineNode node = b3_node(0, name);
        node.rank = rank;
        false_source->store().add_node(std::move(node));
    }
    const auto timing_before_false_fill = workload->dep_free_issue_timing();
    workload->graph_source_ = false_source;
    workload->issue_dep_free_nodes();
    timing = workload->dep_free_issue_timing();
    b3_expect(timing.snapshot_full_fills ==
                      timing_before_false_fill.snapshot_full_fills &&
                  timing.candidates == timing_before_false_fill.candidates + 2 &&
                  timing.issue_calls == timing_before_false_fill.issue_calls &&
                  false_source->store().ready_set_stats().membership_queries == 0 &&
                  false_source->store().ready_set_stats().snapshot_ids_copied == 2,
              "N6 false ID fill uses the NodeView fallback without proof reuse");

    // A derived source owns an overrideable snapshot contract. Even with an
    // unchanged backing NodeStore generation, each virtual fill must run.
    auto flipping_source =
        std::make_shared<FlippingNodeStoreGraphSource>();
    for (const uint64_t id : {8501, 8502}) {
        OnlineNode node = b3_node(id, "N6 flipping-source blocked compute");
        node.rank = rank;
        flipping_source->store().add_node(std::move(node));
    }
    b3_expect(flipping_source->store().free_id_version() ==
                  source_b->store().free_id_version(),
              "N6 derived source starts at the same ready generation");
    workload->graph_source_ = flipping_source;
    const auto timing_before_flip = workload->dep_free_issue_timing();
    workload->issue_dep_free_nodes();
    b3_expect(flipping_source->store().ready_set_stats().snapshot_calls == 1 &&
                  flipping_source->store().ready_set_stats().membership_queries == 1,
              "N6 derived source first virtual fill exposes one candidate");
    flipping_source->include_all_ready_ids = true;
    workload->issue_dep_free_nodes();
    timing = workload->dep_free_issue_timing();
    b3_expect(timing.snapshot_full_fills ==
                      timing_before_flip.snapshot_full_fills + 2 &&
                  timing.snapshot_reuse_hits ==
                      timing_before_flip.snapshot_reuse_hits &&
                  timing.candidates == timing_before_flip.candidates + 3 &&
                  flipping_source->store().ready_set_stats().snapshot_calls == 2 &&
                  flipping_source->store().ready_set_stats().snapshot_ids_copied == 4 &&
                  flipping_source->store().ready_set_stats().membership_queries == 3,
              "N6 derived override is rerun despite unchanged NodeStore version");

    // A source without the ID-snapshot fast path clears the cache tag. On
    // restoration, the same NodeStore source/generation must fill again.
    workload->graph_source_ = std::make_shared<EmptyGraphSource>();
    workload->issue_dep_free_nodes();
    const auto timing_empty = workload->dep_free_issue_timing();
    b3_expect(timing_empty.snapshot_full_fills ==
                      timing.snapshot_full_fills &&
                  timing_empty.snapshot_reuse_hits ==
                      timing.snapshot_reuse_hits &&
                  timing_empty.candidates == timing.candidates,
              "N6 unsupported snapshot source uses the empty compatibility path");
    workload->graph_source_ = source_b;
    workload->issue_dep_free_nodes();
    timing = workload->dep_free_issue_timing();
    b3_expect(timing.snapshot_full_fills ==
                      timing_empty.snapshot_full_fills + 1 &&
                  timing.snapshot_reuse_hits ==
                      timing_empty.snapshot_reuse_hits &&
                  source_b->store().ready_set_stats().snapshot_calls ==
                      2 &&
                  source_b->store().ready_set_stats()
                      .snapshot_ids_copied == 4 &&
                  source_b->store().ready_set_stats().membership_queries == 0,
              "N6 restoring a source after fallback recaptures its IDs");

    auto throwing_source =
        std::make_shared<ThrowingNodeStoreGraphSource>();
    for (const uint64_t id : {9001, 9002}) {
        OnlineNode node = b3_node(id, "N6 throw-source blocked compute");
        node.rank = rank;
        throwing_source->store().add_node(std::move(node));
    }
    b3_expect(throwing_source->store().free_id_version() ==
                  source_b->store().free_id_version(),
              "N6 throwing source also shares the replaced generation");
    workload->graph_source_ = throwing_source;
    bool caught_snapshot_failure = false;
    try {
        workload->issue_dep_free_nodes();
    } catch (const std::runtime_error& error) {
        caught_snapshot_failure =
            std::string(error.what()) == "injected ready snapshot failure";
    }
    b3_expect(caught_snapshot_failure &&
                  !WorkloadTestAccess::snapshot_cache_valid(*workload),
              "N6 failed fill propagates and leaves the prior tag invalid");
    WorkloadTestAccess::reset_issue_guard(*workload);
    throwing_source->throw_on_fill = false;
    workload->issue_dep_free_nodes();
    timing = workload->dep_free_issue_timing();
    b3_expect(timing.snapshot_full_fills ==
                      timing_empty.snapshot_full_fills + 2 &&
                  timing.snapshot_reuse_hits ==
                      timing_empty.snapshot_reuse_hits &&
                  throwing_source->store().ready_set_stats().snapshot_calls == 1 &&
                  throwing_source->store().ready_set_stats().snapshot_ids_copied == 2 &&
                  throwing_source->store().ready_set_stats().membership_queries == 2,
              "N6 retry after failed fill captures the new source completely");

    // Exercise terminal generation saturation: after UINT64_MAX, every pass
    // must fill again because no finite generation value can distinguish a
    // later ready-set mutation from the cached snapshot.
    auto max_source = make_source({10001});
    constexpr uint64_t kMaxVersion = std::numeric_limits<uint64_t>::max();
    NodeStoreTestAccess::set_free_id_version(
        max_source->store(), kMaxVersion - 1);
    workload->graph_source_ = max_source;
    const auto timing_before_max = workload->dep_free_issue_timing();
    workload->issue_dep_free_nodes();  // MAX - 1: can tag this one generation
    OnlineNode second_max = b3_node(10002, "N6 saturated compute 2");
    second_max.rank = rank;
    max_source->store().add_node(std::move(second_max));  // reaches MAX
    workload->issue_dep_free_nodes();
    workload->issue_dep_free_nodes();  // unchanged MAX must still refill
    OnlineNode third_max = b3_node(10003, "N6 saturated compute 3");
    third_max.rank = rank;
    max_source->store().add_node(std::move(third_max));  // remains MAX
    workload->issue_dep_free_nodes();
    timing = workload->dep_free_issue_timing();
    const auto& max_stats = max_source->store().ready_set_stats();
    b3_expect(max_source->store().free_id_version() == kMaxVersion &&
                  timing.snapshot_full_fills ==
                      timing_before_max.snapshot_full_fills + 4 &&
                  timing.snapshot_reuse_hits ==
                      timing_before_max.snapshot_reuse_hits &&
                  max_stats.snapshot_calls == 4 &&
                  max_stats.snapshot_ids_copied == 8 &&
                  max_stats.membership_queries == 7,
              "N6 saturated generation never wraps or enables snapshot reuse");

    workload->graph_source_ = original_source;
    workload->issue_dep_free_nodes();
    workload->hw_resource->release(blocker);
    b3_expect(workload->hw_resource->num_in_flight_gpu_comp_ops == 0,
              "N6 releases its held GPU-COMP slot");
    std::printf(
        "PASS N6 snapshot lifecycle: same_source_reuse=1 identity_refresh=1 "
        "fallback_refresh=1 throw_recovery=1 max_generation_fullfills=4\n");
}
}  // namespace
}  // namespace ExecutionDriven
}  // namespace AstraSim

int main(const int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    if (argc == 2 &&
        std::string(argv[1]) == "--membership-cost-probe") {
        AstraSim::LoggerFactory::init("empty", "off");
        MetricCollector::instance().initialize("empty", "off");
        FullStack stack;
        build_full_stack(stack);
        scenario_membership_cost_probe(stack);
        CompletionObserver::instance().set_hook(nullptr, nullptr);
        std::error_code ignored;
        std::filesystem::remove_all(stack.directory, ignored);
        if (g_failures != 0) {
            std::printf("[ready_issue_order_test] FAIL: %d failures\n",
                        g_failures);
            return 1;
        }
        std::printf("[ready_issue_order_test] MEMBERSHIP COST PROBE PASS\n");
        return 0;
    }
    if (argc != 1) {
        b3_fail("usage: ready_issue_order_test [--membership-cost-probe]");
    }
    AstraSim::LoggerFactory::init("empty", "off");
    MetricCollector::instance().initialize("empty", "off");

    // ---------------- store layer ----------------
    scenario_r1_store();
    scenario_r2_store();
    scenario_r3_store();
    scenario_r4_store();
    scenario_n_delta();
    scenario_n_ready_snapshot_scale();
    scenario_n_membership_and_boundary();
    scenario_r6_store();
    scenario_r7_store();

    // ---------------- full-stack layer ----------------
    FullStack stack;
    build_full_stack(stack);

    // R5 continues rank0's json stream after the P0 nodes; rank1 is fresh.
    // A rank's committed node count equals its next json id (json ids are
    // a dense 0-based stream; store ids start at 1).
    const uint64_t rank0_next_json =
        stack.graph_sources[0]->store().next_auto_id() - 1;
    scenario_r5_fullstack(stack, rank0_next_json);

    const auto timing_before = stack.workloads[3]->dep_free_issue_timing();
    const ChainFacts facts = scenario_r6_fullstack(stack);
    scenario_r7b_fullstack(stack);
    scenario_n3_timing(stack, facts, timing_before);
    scenario_n4_fullstack(stack);
    // Direct store construction in N5/N6 runs after every committer affine
    // check, so it cannot alter a later GraphBatch fixture's expected stream.
    scenario_n5_blocked_node_view_materialization(stack);
    scenario_n6_snapshot_generation_lifecycle(stack);

    // ---------------- wrap-up ----------------
    b3_expect(stack.event_queue->finished() &&
                  !stack.event_queue->has_deferred_work(),
              "final queue state drained");
    for (Workload* workload : stack.workloads) {
        b3_expect(workload->hw_resource->num_in_flight_gpu_comm_ops == 0 &&
                      workload->hw_resource->num_in_flight_gpu_comp_ops == 0,
                  "final in-flight occupancy zero on every rank");
    }
    for (const IssueRecord& record : stack.issues) {
        b3_expect(record.rank >= 0 && record.rank < 7,
                  "every issue record belongs to a valid rank");
    }
    std::printf("[ready_issue_order_test] issues=%zu terminals=%zu "
                "watch_fires=%zu\n",
                stack.issues.size(), stack.terminals.size(),
                stack.fires.size());

    CompletionObserver::instance().set_hook(nullptr, nullptr);
    std::error_code ignored;
    std::filesystem::remove_all(stack.directory, ignored);

    if (g_failures != 0) {
        std::printf("[ready_issue_order_test] FAIL: %d failures\n",
                    g_failures);
        return 1;
    }
    std::printf("[ready_issue_order_test] ALL PASS: store golden replay "
                "R1/R2/R4 equal, R3 divergence registered "
                "(b3_membership_dedup), R5-R7 + N3-N6 green on the real "
                "stack\n");
    return 0;
}
