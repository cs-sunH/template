/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

node_store_test.cc -- phase-1 step 1-4 NodeStore / GraphSource fixture.

Exercises the three step-1-4 mechanisms (方案 §4 步骤 1-4) as a standalone
test -- no network simulation, no baseline artifacts touched:

  Part A  NodeStore semantics: id assignment, dependency blocking/release,
          mark_issued, finish_node idempotency, dead-parent edges, reverse
          index (meta_for), pending_count.
  Part B  NodeStoreGraphSource: store-backed dep_free_nodes / lookup /
          take_node / finish_node delegation, no-auto-emit, static_all_done
          false (online authority belongs to the ServiceCoordinator).
  Part C  ETFeederGraphSource adapter: built over a real ETFeeder reading
          the step-1-3 synthetic trace (tests/make_completion_fixture_et.py,
          sh_test_mesh/generated/completion_fixture/fixture.{rank}.et);
          verifies the NodeView field mapping (kind-guarded attr reads,
          runtime_ns = micros * 1000, involved_dim bool_list, ascending
          free order, take/finish lifecycle, static_all_done).
  Part D  phase-3 sensing: per-rank injected-unfinished ledger summary
          (classification by compute ops / comm bytes / estimated remaining
          service / resource state; per (request_id, stage, generation)
          grouping; finished nodes excluded; mark_issued moves the resource
          state from free to in-flight).
  Part E  M2 node GC (2026-08-23; the --online-node-gc CLI arm was removed
          by the B.3 cleanup (2026-09-05) -- collection always on): quiescent-point
          collection of finished childless nodes, parent retention while a
          child is unfinished, dead-parent no-op onto erased ids, and the
          gc-off (default) pre-M2 never-erase behavior.

Build: the CMake target AstraSim_Analytical_Congestion_Aware_NodeStoreTest
(build with cmake --build build/astra_analytical/build_congestion_aware -j).
Run (from template/astra-sim-sh_3.0, after generating the synthetic trace):
  python3 astra-sim/workload/execution_driven/tests/make_completion_fixture_et.py
  build/astra_analytical/build_congestion_aware/bin/\
      AstraSim_Analytical_Congestion_Aware_NodeStoreTest \
      --fixture-et=sh_test_mesh/generated/completion_fixture/fixture.0.et
Exit code 0 on ALL PASS.
*******************************************************************************/

#include "astra-sim/workload/execution_driven/GraphSource.hh"
#include "astra-sim/workload/execution_driven/NodeStore.hh"
#include "extern/graph_frontend/chakra/src/feeder_v3/et_feeder.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

using namespace AstraSim::ExecutionDriven;

namespace AstraSim {
namespace ExecutionDriven {

struct NodeStoreTestAccess {
    struct StateCounts {
        uint64_t next_id = 0;
        size_t free_ids = 0;
        size_t nodes = 0;
        size_t virtual_index_blocks = 0;
        size_t virtual_progressions = 0;
    };

    struct SweepCounts {
        uint64_t envelope_candidates = 0;
        uint64_t exact_intersection_tests = 0;
    };

    struct QueryCounts {
        bool result = false;
        uint64_t reservation_binary_probes = 0;
        uint64_t reservation_blocks_examined = 0;
        uint64_t interval_tree_nodes_examined = 0;
        uint64_t envelope_candidates = 0;
        uint64_t exact_ap_tests = 0;
    };

    static StateCounts state_counts(const NodeStore& store) {
        size_t progression_count = 0;
        for (const auto& block : store.virtual_id_reservation_blocks_) {
            progression_count += block.progressions.size();
        }
        return {store.next_id_, store.free_ids_.size(), store.nodes_.size(),
                store.virtual_id_reservation_blocks_.size(),
                progression_count};
    }

    static SweepCounts reserve_counted(
        NodeStore& store, const uint64_t first_store_id, const uint64_t width,
        const std::vector<NodeStore::VirtualIdProgression>& progressions) {
        NodeStore::VirtualProgressionSweepWork work;
        store.reserve_logical_id_span_impl(first_store_id, width,
                                           progressions, &work);
        return {work.envelope_candidates, work.exact_intersection_tests};
    }

    static QueryCounts point_query(const NodeStore& store,
                                   const uint64_t node_id) {
        NodeStore::VirtualIdQueryWork work;
        const bool result = store.is_virtual_id_impl(node_id, &work);
        return {result, work.reservation_binary_probes,
                work.reservation_blocks_examined,
                work.interval_tree_nodes_examined,
                work.envelope_candidates, work.exact_ap_tests};
    }

    static QueryCounts virtual_progression_query(
        const NodeStore& store, const uint64_t first_id,
        const uint64_t count, const uint64_t stride) {
        NodeStore::VirtualIdQueryWork work;
        const bool result = store.is_virtual_id_progression_impl(
            first_id, count, stride, &work);
        return {result, work.reservation_binary_probes,
                work.reservation_blocks_examined,
                work.interval_tree_nodes_examined,
                work.envelope_candidates, work.exact_ap_tests};
    }

    static QueryCounts nonvirtual_progression_query(
        const NodeStore& store, const uint64_t first_id,
        const uint64_t count, const uint64_t stride) {
        NodeStore::VirtualIdQueryWork work;
        const bool result = store.is_allocated_nonvirtual_progression_impl(
            first_id, count, stride, &work);
        return {result, work.reservation_binary_probes,
                work.reservation_blocks_examined,
                work.interval_tree_nodes_examined,
                work.envelope_candidates, work.exact_ap_tests};
    }
};

}  // namespace ExecutionDriven
}  // namespace AstraSim

namespace {

bool g_ok = true;

void expect(bool cond, const char* what) {
    if (!cond) {
        std::fprintf(stderr, "[node_store_test] FAIL: %s\n", what);
        g_ok = false;
    }
}

bool contains(const std::vector<uint64_t>& ids, uint64_t id) {
    return std::find(ids.begin(), ids.end(), id) != ids.end();
}

// ---------------------------------------------------------------- Part A --
// NodeStore unit semantics.
void test_node_store() {
    NodeStore store;
    expect(store.empty(), "A: empty() initially true");
    expect(store.pending_count() == 0, "A: pending_count() initially 0");

    OnlineNode n;
    n.kind = NodeKind::Compute;
    n.name = "a";
    const uint64_t id1 = store.add_node(n);
    const uint64_t id2 = store.add_node(n);
    expect(id1 == 1 && id2 == 2, "A: add_node assigns 1, 2");
    expect(store.next_auto_id() == 3,
           "A: next_auto_id exposes the following automatic id");
    expect(!store.empty(), "A: non-empty after adds");
    expect(store.pending_count() == 2, "A: pending_count() == 2");
    expect(store.resolve_free_nodes() == std::vector<uint64_t>({1, 2}),
           "A: both nodes free, ascending");
    std::vector<uint64_t> free_snapshot = {99};
    store.fill_free_node_snapshot(free_snapshot);
    expect(free_snapshot == std::vector<uint64_t>({1, 2}),
           "A: caller-owned free snapshot matches resolve_free_nodes");

    OnlineNode m;
    m.global_id = 7;
    m.kind = NodeKind::Metadata;
    m.request_id = "req-7";
    m.stage = "prefill";
    m.generation = 3;
    expect(store.add_node(m) == 7, "A: explicit global_id kept");
    expect(store.next_auto_id() == 8,
           "A: explicit global_id advances the following automatic id");
    expect(store.resolve_free_nodes() == std::vector<uint64_t>({1, 2, 7}),
           "A: free set ascending incl. 7");
    const auto rec = store.node(7);
    expect(rec.has_value() && rec->kind == NodeKind::Metadata,
           "A: node() returns stored record");
    const auto meta = store.meta_for(7);
    expect(meta.has_value() && meta->request_id == "req-7" &&
               meta->stage == "prefill" && meta->generation == 3,
           "A: meta_for returns reverse index");
    expect(!store.meta_for(999).has_value(), "A: meta_for miss is nullopt");
    expect(!store.node(999).has_value(), "A: node() miss is nullopt");

    // Dependency: child added after parent -> child leaves the free set.
    store.add_dependency(1, 7, DepKind::Data);
    expect(store.resolve_free_nodes() == std::vector<uint64_t>({1, 2}),
           "A: child with recorded parent is not free");
    expect(store.node(7).has_value(), "A: blocked child still stored");

    // mark_issued consumes; idempotent; unknown ids ignored.
    store.mark_issued(1);
    expect(store.resolve_free_nodes() == std::vector<uint64_t>({2}),
           "A: mark_issued removes from free set");
    store.mark_issued(1);
    store.mark_issued(999);
    expect(store.resolve_free_nodes() == std::vector<uint64_t>({2}),
           "A: mark_issued idempotent / unknown-id no-op");

    // finish_node is the only dependency-release entry; idempotent.
    store.finish_node(1);
    expect(store.resolve_free_nodes() == std::vector<uint64_t>({2, 7}),
           "A: finish_node frees the child");
    store.finish_node(1);
    expect(store.resolve_free_nodes() == std::vector<uint64_t>({2, 7}),
           "A: finish_node idempotent (no double release)");

    // Dead-parent edge: parent already finished -> child stays free.
    const uint64_t idc = store.add_node(n);
    store.finish_node(idc);
    const uint64_t idd = store.add_node(n);
    store.add_dependency(idc, idd, DepKind::Control);
    expect(contains(store.resolve_free_nodes(), idd),
           "A: edge to finished parent does not block child");

    // DepKind is accepted and stored per edge (all resolve identically in
    // the first version; documented in NodeStore.hh).
    OnlineNode fence_a;
    fence_a.name = "fence-a";
    OnlineNode fence_b;
    fence_b.name = "fence-b";
    const uint64_t fa = store.add_node(fence_a);
    const uint64_t fb = store.add_node(fence_b);
    store.add_dependency(fa, fb, DepKind::Enabled);
    expect(!contains(store.resolve_free_nodes(), fb),
           "A: Enabled edge blocks like the others (first version)");
    store.finish_node(fa);
    expect(contains(store.resolve_free_nodes(), fb),
           "A: Enabled edge releases on finish");

    // 7 nodes stored so far (1, 2, 7, 8=idc, 9=idd, 10=fa, 11=fb --
    // the explicit global_id 7 advanced the automatic stream to 8);
    // finished so far: 1, 8, 10 -> 4 pending.
    expect(store.pending_count() == 4, "A: pending_count() == 4");
    store.finish_node(2);
    store.finish_node(7);
    store.finish_node(idd);
    store.finish_node(fb);
    expect(store.pending_count() == 0,
           "A: pending_count() 0 after all finishes");

    // Online terminal delivery is a single-fire transition.  It must reject
    // unknown and not-yet-issued ids before any Statistics/observer side
    // effect, accept exactly the first terminal callback after issue, then
    // reject the duplicate.
    NodeStore terminal_store;
    const uint64_t terminal_id = terminal_store.add_node(n);
    expect(!terminal_store.mark_terminal_observed(999),
           "A: terminal guard rejects unknown node");
    expect(!terminal_store.mark_terminal_observed(terminal_id),
           "A: terminal guard rejects unissued node");
    terminal_store.mark_issued(terminal_id);
    expect(terminal_store.mark_terminal_observed(terminal_id),
           "A: terminal guard accepts first issued callback");
    expect(!terminal_store.mark_terminal_observed(terminal_id),
           "A: terminal guard rejects duplicate callback");
}

// ---------------------------------------------------------------- Part E --
// M2 node GC (2026-08-23; B.3 cleanup 2026-09-05 removed the --online-node-gc
// CLI arm -- collection always on): finished childless nodes are
// erased at the quiescent collect_garbage() point (never inside
// finish_node); a parent is retained while any child is unfinished and is
// collected by the last child's finish; edges onto an erased (i.e. already
// finished) parent keep the dead-parent no-op; GC off (the default) keeps
// the pre-M2 never-erase behavior.
void test_node_gc() {
    // GC off (default): finishing never erases, collect is a no-op.
    NodeStore off_store;
    OnlineNode n;
    n.kind = NodeKind::Compute;
    n.name = "gc-off";
    const uint64_t off_id = off_store.add_node(n);
    off_store.finish_node(off_id);
    off_store.collect_garbage();
    expect(off_store.node(off_id).has_value(),
           "E: gc off keeps finished nodes (pre-M2 behavior)");
    expect(off_store.gc_erased_count() == 0, "E: gc off erases nothing");

    // GC on: a childless finished node is collected at the quiescent point
    // only -- finish_node itself never erases (Workload::skip_invalid looks
    // the record up again after finish_node).
    NodeStore store;
    store.set_gc_enabled(true);
    const uint64_t a = store.add_node(n);
    store.finish_node(a);
    expect(store.node(a).has_value(),
           "E: finish_node itself never erases");
    store.collect_garbage();
    expect(!store.node(a).has_value(),
           "E: childless finished node erased at collect_garbage");
    expect(store.erased(a), "E: erased() observes the collection");
    expect(store.gc_erased_count() == 1, "E: gc_erased_count tracked");
    expect(store.pending_count() == 0, "E: pending_count unaffected");

    // Chain: the parent is retained while its child is unfinished; the
    // child's finish re-enqueues it and one collect removes both.
    const uint64_t p = store.add_node(n);
    const uint64_t c = store.add_node(n);
    store.add_dependency(p, c, DepKind::Data);
    store.mark_issued(p);
    store.finish_node(p);
    expect(contains(store.resolve_free_nodes(), c),
           "E: child freed by the parent finish (unaffected by GC)");
    store.collect_garbage();
    expect(store.node(p).has_value(),
           "E: parent retained while its child is unfinished");
    store.finish_node(c);
    store.collect_garbage();
    expect(!store.node(p).has_value() && !store.node(c).has_value(),
           "E: chain collected after the last child finishes");

    // Erased => finished: the dead-parent no-op is preserved.
    const uint64_t late = store.add_node(n);
    store.add_dependency(p, late, DepKind::Data);
    expect(contains(store.resolve_free_nodes(), late),
           "E: edge onto an erased (finished) parent does not block");
    store.finish_node(p);  // finish of an erased id: still a no-op
    expect(store.retained_count() == 1, "E: only the late node remains");
    expect(store.gc_erased_count() == 3, "E: a + p + c erased in total");
}

// ---------------------------------------------------------------- Part G --
// C0 logical-id reservation: virtual ranges advance the existing affine id
// stream without manufacturing nodes or becoming completed dependency
// parents.
void test_virtual_id_ranges() {
    NodeStore store;
    OnlineNode node;
    node.kind = NodeKind::Compute;
    node.name = "before-hole";
    expect(store.add_node(node) == 1, "G: first entity keeps id 1");
    store.reserve_id_range(2, 3);
    expect(store.next_auto_id() == 5,
           "G: range reservation advances the existing id stream");
    expect(store.is_virtual_id(2) && store.is_virtual_id(3) &&
               store.is_virtual_id(4) && !store.is_virtual_id(1) &&
               !store.is_virtual_id(5),
           "G: virtual membership covers exactly the reserved interval");
    expect(!store.erased(2),
           "G: virtual ids are not treated as completed/GC-erased parents");
    expect(store.node(2) == std::nullopt && store.pending_count() == 1,
           "G: reservation creates no NodeStore records or pending nodes");

    node.name = "after-hole";
    expect(store.add_node(node) == 5,
           "G: next ordinary entity follows the virtual interval");
    bool rejected_virtual_edge = false;
    try {
        store.add_dependency(3, 5, DepKind::Data);
    } catch (const std::invalid_argument&) {
        rejected_virtual_edge = true;
    }
    expect(rejected_virtual_edge,
           "G: virtual id cannot be used as a dependency endpoint");

    bool rejected_non_frontier_range = false;
    try {
        store.reserve_id_range(7, 1);
    } catch (const std::logic_error&) {
        rejected_non_frontier_range = true;
    }
    expect(rejected_non_frontier_range,
           "G: range reservation must preserve the affine frontier");
}

void test_sparse_virtual_id_span() {
    NodeStore store;
    OnlineNode node;
    node.kind = NodeKind::Compute;
    node.name = "before-sparse-span";
    expect(store.add_node(node) == 1, "G2: first entity keeps id 1");

    // The bounding intervals overlap ([3, 7] and [4, 8]), while their AP
    // memberships are disjoint: {3, 5, 7} and {4, 8}.
    store.reserve_logical_id_span(
        2, 8, {{3, 3, 2}, {4, 2, 4}});
    expect(store.next_auto_id() == 10,
           "G2: reserving the whole span advances the affine frontier");
    expect(store.is_virtual_id(3) && store.is_virtual_id(4) &&
               store.is_virtual_id(5) && store.is_virtual_id(7) &&
               store.is_virtual_id(8) && !store.is_virtual_id(2) &&
               !store.is_virtual_id(6) && !store.is_virtual_id(9),
           "G2: sparse AP holes preserve ordinary positions inside the span");
    expect(!store.is_finished_or_erased(3) && !store.erased(3),
           "G2: a sparse virtual ID is neither a node nor a completed parent");

    for (const uint64_t id : {2, 6, 9}) {
        node.global_id = id;
        node.name = "physical-" + std::to_string(id);
        expect(store.add_node(node) == id,
               "G2: ordinary physical IDs can be inserted inside the reserved span");
    }
    expect(store.pending_count() == 4 && store.next_auto_id() == 10,
           "G2: explicit physical IDs do not perturb the reserved span frontier");
    expect(store.is_virtual_id_progression(3, 3, 2) &&
               store.is_virtual_id_progression(7, 2, 1) &&
               !store.is_virtual_id_progression(3, 4, 1),
           "G2: virtual AP query proves sparse-union membership without expansion");
    expect(store.is_allocated_nonvirtual_progression(2, 2, 4) &&
               !store.is_allocated_nonvirtual_progression(2, 3, 2),
           "G2: allocated nonvirtual AP query excludes virtual positions");
    expect(store.count_finished_or_erased_nonvirtual_progression(2, 1, 4) == 0,
           "G2: committed physical AP reports no finished members initially");
    store.set_gc_enabled(true);
    store.finish_node(2);
    store.collect_garbage();
    expect(store.is_allocated_nonvirtual_progression(2, 1, 1),
           "G2: collected physical IDs remain allocated nonvirtual history");
    expect(store.count_finished_or_erased_nonvirtual_progression(2, 1, 4) == 1,
           "G2: erased physical AP members count as finished without expansion");

    NodeStore invalid_overlap;
    bool rejected_overlap = false;
    try {
        invalid_overlap.reserve_logical_id_span(
            1, 10, {{2, 3, 4}, {6, 2, 4}});
    } catch (const std::invalid_argument&) {
        rejected_overlap = true;
    }
    expect(rejected_overlap && invalid_overlap.next_auto_id() == 1 &&
               !invalid_overlap.is_virtual_id(2),
           "G2: intersecting AP holes reject atomically before sparse reservation");

    bool rejected_out_of_span = false;
    try {
        NodeStore invalid;
        invalid.reserve_logical_id_span(1, 4, {{2, 2, 3}});
    } catch (const std::invalid_argument&) {
        rejected_out_of_span = true;
    }
    expect(rejected_out_of_span,
           "G2: sparse AP terminal ID must remain inside the reserved span");
}

void test_virtual_progression_sweep_scales() {
    for (const uint64_t progression_count : {uint64_t{8}, uint64_t{32},
                                              uint64_t{128}}) {
        NodeStore store;
        std::vector<NodeStore::VirtualIdProgression> progressions;
        progressions.reserve(static_cast<size_t>(progression_count));
        for (uint64_t i = 0; i < progression_count; i += 2) {
            const uint64_t base = 1 + (i / 2) * 8;
            progressions.push_back({base, 2, 4});
            progressions.push_back({base + 2, 2, 4});
        }
        const uint64_t span_width = progression_count * 4;
        const auto work = NodeStoreTestAccess::reserve_counted(
            store, 1, span_width, progressions);
        expect(work.envelope_candidates == progression_count / 2 &&
                   work.exact_intersection_tests == progression_count / 2 &&
                   work.envelope_candidates <
                       progression_count * (progression_count - 1) / 2 &&
                   store.next_auto_id() == 1 + span_width,
               "G2: NodeStore envelope sweep checks only active pairs at 8/32/128 ranges");
        for (uint64_t i = 0; i < progression_count; i += 2) {
            const uint64_t base = 1 + (i / 2) * 8;
            expect(store.is_virtual_id_progression(base, 2, 4) &&
                       store.is_virtual_id_progression(base + 2, 2, 4),
                   "G2: swept interleaved APs remain reserved exactly");
        }

        NodeStore broad_envelopes;
        std::vector<NodeStore::VirtualIdProgression> interleaved;
        interleaved.reserve(static_cast<size_t>(progression_count));
        for (uint64_t i = 0; i < progression_count; ++i) {
            interleaved.push_back({1 + i, 2, progression_count});
        }
        const auto broad_work = NodeStoreTestAccess::reserve_counted(
            broad_envelopes, 1, progression_count * 2, interleaved);
        const uint64_t all_pairs =
            progression_count * (progression_count - 1) / 2;
        expect(broad_work.envelope_candidates == all_pairs &&
                   broad_work.exact_intersection_tests == all_pairs,
               "G2: same-envelope stride residues preserve quadratic exact worst case");
    }
}

void test_virtual_id_query_index_scales_by_reservation() {
    std::fprintf(stderr,
                 "[node_store_test] G3 counters measure current indexed work; "
                 "old flat-history scan cost is theoretical/uninstrumented\n");
    for (const uint64_t history : {uint64_t{8}, uint64_t{32},
                                   uint64_t{128}}) {
        uint64_t binary_probe_bound = 0;
        for (uint64_t remaining = history; remaining != 0;
             remaining >>= 1) {
            ++binary_probe_bound;
        }
        for (const uint64_t query_count : {uint64_t{4}, uint64_t{128}}) {
            NodeStore store;
            uint64_t first = 1;
            for (uint64_t index = 0; index < history; ++index) {
                store.reserve_logical_id_span(
                    first, 8, {{first + 1, 2, 2}});
                first += 8;
            }
            const uint64_t latest_first = first - 8;
            const uint64_t latest_virtual_id = latest_first + 3;
            const uint64_t latest_physical_ap_first = latest_first + 2;

            NodeStoreTestAccess::QueryCounts point_totals;
            NodeStoreTestAccess::QueryCounts virtual_ap_totals;
            NodeStoreTestAccess::QueryCounts physical_ap_totals;
            for (uint64_t query = 0; query < query_count; ++query) {
                const auto point = NodeStoreTestAccess::point_query(
                    store, latest_virtual_id);
                const auto virtual_ap =
                    NodeStoreTestAccess::virtual_progression_query(
                        store, latest_first + 1, 2, 2);
                // This AP's envelope crosses the virtual AP envelope, but
                // its represented IDs occupy different stride residues.
                const auto physical_ap =
                    NodeStoreTestAccess::nonvirtual_progression_query(
                        store, latest_physical_ap_first, 2, 4);
                expect(point.result && virtual_ap.result && physical_ap.result,
                       "G3: latest-span point and AP queries preserve exact membership");
                point_totals.reservation_binary_probes +=
                    point.reservation_binary_probes;
                point_totals.reservation_blocks_examined +=
                    point.reservation_blocks_examined;
                point_totals.interval_tree_nodes_examined +=
                    point.interval_tree_nodes_examined;
                point_totals.envelope_candidates += point.envelope_candidates;
                point_totals.exact_ap_tests += point.exact_ap_tests;
                virtual_ap_totals.reservation_binary_probes +=
                    virtual_ap.reservation_binary_probes;
                virtual_ap_totals.reservation_blocks_examined +=
                    virtual_ap.reservation_blocks_examined;
                virtual_ap_totals.interval_tree_nodes_examined +=
                    virtual_ap.interval_tree_nodes_examined;
                virtual_ap_totals.envelope_candidates +=
                    virtual_ap.envelope_candidates;
                virtual_ap_totals.exact_ap_tests += virtual_ap.exact_ap_tests;
                physical_ap_totals.reservation_binary_probes +=
                    physical_ap.reservation_binary_probes;
                physical_ap_totals.reservation_blocks_examined +=
                    physical_ap.reservation_blocks_examined;
                physical_ap_totals.interval_tree_nodes_examined +=
                    physical_ap.interval_tree_nodes_examined;
                physical_ap_totals.envelope_candidates +=
                    physical_ap.envelope_candidates;
                physical_ap_totals.exact_ap_tests += physical_ap.exact_ap_tests;
            }
            const uint64_t binary_bound =
                query_count * binary_probe_bound;
            expect(point_totals.reservation_binary_probes <= binary_bound &&
                       point_totals.reservation_blocks_examined == query_count &&
                       point_totals.interval_tree_nodes_examined == query_count &&
                       point_totals.envelope_candidates == query_count &&
                       point_totals.exact_ap_tests == query_count,
                   "G3: point membership work depends on directory depth, not history length");
            expect(virtual_ap_totals.reservation_binary_probes <= binary_bound &&
                       virtual_ap_totals.reservation_blocks_examined == query_count &&
                       virtual_ap_totals.interval_tree_nodes_examined ==
                           query_count &&
                       virtual_ap_totals.envelope_candidates == query_count &&
                       virtual_ap_totals.exact_ap_tests == query_count,
                   "G3: virtual AP coverage checks only the latest span's envelope candidate");
            expect(physical_ap_totals.reservation_binary_probes <= binary_bound &&
                       physical_ap_totals.reservation_blocks_examined == query_count &&
                       physical_ap_totals.interval_tree_nodes_examined ==
                           query_count &&
                       physical_ap_totals.envelope_candidates == query_count &&
                       physical_ap_totals.exact_ap_tests == query_count,
                   "G3: nonvirtual AP checks exact residues after bounded envelope lookup");
            std::fprintf(
                stderr,
                "[node_store_test] G3 H=%llu Q=%llu P=1 "
                "point{bin=%llu blocks=%llu tree=%llu C=%llu exact=%llu} "
                "virtual{bin=%llu blocks=%llu tree=%llu C=%llu exact=%llu} "
                "nonvirtual{bin=%llu blocks=%llu tree=%llu C=%llu exact=%llu}\n",
                static_cast<unsigned long long>(history),
                static_cast<unsigned long long>(query_count),
                static_cast<unsigned long long>(
                    point_totals.reservation_binary_probes),
                static_cast<unsigned long long>(
                    point_totals.reservation_blocks_examined),
                static_cast<unsigned long long>(
                    point_totals.interval_tree_nodes_examined),
                static_cast<unsigned long long>(
                    point_totals.envelope_candidates),
                static_cast<unsigned long long>(point_totals.exact_ap_tests),
                static_cast<unsigned long long>(
                    virtual_ap_totals.reservation_binary_probes),
                static_cast<unsigned long long>(
                    virtual_ap_totals.reservation_blocks_examined),
                static_cast<unsigned long long>(
                    virtual_ap_totals.interval_tree_nodes_examined),
                static_cast<unsigned long long>(
                    virtual_ap_totals.envelope_candidates),
                static_cast<unsigned long long>(
                    virtual_ap_totals.exact_ap_tests),
                static_cast<unsigned long long>(
                    physical_ap_totals.reservation_binary_probes),
                static_cast<unsigned long long>(
                    physical_ap_totals.reservation_blocks_examined),
                static_cast<unsigned long long>(
                    physical_ap_totals.interval_tree_nodes_examined),
                static_cast<unsigned long long>(
                    physical_ap_totals.envelope_candidates),
                static_cast<unsigned long long>(
                    physical_ap_totals.exact_ap_tests));
        }
    }
}

void test_virtual_id_interval_tree_exactness() {
    for (const uint64_t progression_count : {uint64_t{8}, uint64_t{32},
                                              uint64_t{128}}) {
        NodeStore store;
        const uint64_t first = 1;
        const uint64_t stride = progression_count * 2;
        std::vector<NodeStore::VirtualIdProgression> holes;
        holes.reserve(static_cast<size_t>(progression_count));
        for (uint64_t index = 0; index < progression_count; ++index) {
            holes.push_back({first + index * 2, 3, stride});
        }
        store.reserve_logical_id_span(first, progression_count * 6, holes);

        // Every hole envelope contains first + stride although only the first
        // AP represents that exact ID. The candidate tree must return all
        // overlapping envelopes and the exact AP test must choose correctly.
        const auto point = NodeStoreTestAccess::point_query(
            store, first + stride);
        // This odd-residue point is inside every AP envelope but is not a
        // member of any AP; the unsuccessful lookup must inspect all P
        // envelope candidates before rejecting it.
        const auto nonmember_point = NodeStoreTestAccess::point_query(
            store, first + stride + 1);
        const auto virtual_ap =
            NodeStoreTestAccess::virtual_progression_query(
                store, first, 3, stride);
        // This nonvirtual AP also crosses all hole envelopes, but its residue
        // differs from every reserved AP.
        const auto physical_ap =
            NodeStoreTestAccess::nonvirtual_progression_query(
                store, first + 1, 3, stride);
        expect(point.result && point.envelope_candidates >= 1 &&
                   point.envelope_candidates <= progression_count &&
                   point.exact_ap_tests == point.envelope_candidates,
               "G3: overlapping AP envelopes do not hide the represented point");
        expect(!nonmember_point.result &&
                   nonmember_point.envelope_candidates == progression_count &&
                   nonmember_point.exact_ap_tests == progression_count,
               "G3: failed point lookup checks the full overlapping-envelope worst case");
        expect(virtual_ap.result &&
                   virtual_ap.envelope_candidates == progression_count &&
                   virtual_ap.exact_ap_tests == progression_count,
               "G3: AP-union coverage counts exact members without tile expansion");
        expect(physical_ap.result &&
                   physical_ap.envelope_candidates == progression_count &&
                   physical_ap.exact_ap_tests == progression_count,
               "G3: envelope overlap alone does not reject an allocated nonvirtual AP");
        const auto counts = NodeStoreTestAccess::state_counts(store);
        expect(counts.virtual_index_blocks == 1 &&
                   counts.virtual_progressions == progression_count,
               "G3: sparse index storage follows AP count, not represented ID count");
    }

    NodeStore nearest_first;
    nearest_first.reserve_logical_id_span(
        1, 40, {{10, 3, 10}, {15, 2, 20}});
    const auto overlapping_envelopes =
        NodeStoreTestAccess::point_query(nearest_first, 30);
    expect(overlapping_envelopes.result &&
               overlapping_envelopes.envelope_candidates == 2 &&
               overlapping_envelopes.exact_ap_tests == 2 &&
               nearest_first.is_virtual_id_progression(10, 3, 10) &&
               !nearest_first.is_allocated_nonvirtual_progression(30, 1, 1),
           "G3: nearest-first AP lookup still finds an earlier overlapping envelope");
}

void test_virtual_id_empty_blocks_and_reservation_atomicity() {
    NodeStore empty_holes;
    empty_holes.reserve_logical_id_span(1, 8, {});
    const auto empty_counts = NodeStoreTestAccess::state_counts(empty_holes);
    expect(empty_counts.virtual_index_blocks == 0 &&
               empty_counts.virtual_progressions == 0 &&
               !empty_holes.is_virtual_id(4) &&
               !empty_holes.is_virtual_id_progression(1, 8, 1) &&
               empty_holes.is_allocated_nonvirtual_progression(1, 8, 1),
           "G3: empty virtual blocks remain allocated physical spans");

    empty_holes.reserve_id_range(9, 4);
    expect(empty_holes.is_virtual_id(9) &&
               !empty_holes.is_virtual_id(8) &&
               empty_holes.is_virtual_id_progression(9, 4, 1) &&
               NodeStoreTestAccess::state_counts(empty_holes)
                       .virtual_index_blocks == 1,
           "G3: a later virtual span indexes after an unindexed physical span");

    for (const uint64_t empty_span_count : {uint64_t{8}, uint64_t{32},
                                            uint64_t{128}}) {
        NodeStore physical_only;
        uint64_t first = 1;
        for (uint64_t index = 0; index < empty_span_count; ++index) {
            physical_only.reserve_logical_id_span(first, 8, {});
            first += 8;
        }
        const auto counts = NodeStoreTestAccess::state_counts(physical_only);
        expect(counts.next_id == first && counts.virtual_index_blocks == 0 &&
                   counts.virtual_progressions == 0 &&
                   physical_only.is_allocated_nonvirtual_progression(
                       1, empty_span_count * 8, 1) &&
                   !physical_only.is_virtual_id(first - 1),
               "G3: physical-only reservation history retains no virtual index blocks");
    }

    NodeStore store;
    store.reserve_logical_id_span(1, 8, {{2, 2, 3}});
    const auto before = NodeStoreTestAccess::state_counts(store);
    bool rejected_overlap = false;
    try {
        store.reserve_logical_id_span(
            9, 12, {{10, 3, 2}, {12, 2, 2}});
    } catch (const std::invalid_argument&) {
        rejected_overlap = true;
    }
    bool rejected_out_of_span = false;
    try {
        store.reserve_logical_id_span(9, 4, {{10, 2, 3}});
    } catch (const std::invalid_argument&) {
        rejected_out_of_span = true;
    }
    const auto after = NodeStoreTestAccess::state_counts(store);
    expect(rejected_overlap && rejected_out_of_span &&
               after.next_id == before.next_id &&
               after.free_ids == before.free_ids &&
               after.nodes == before.nodes &&
               after.virtual_index_blocks == before.virtual_index_blocks &&
               after.virtual_progressions == before.virtual_progressions &&
               store.is_virtual_id(2) && !store.is_virtual_id(10),
           "G3: later reservation validation failures publish no span or index state");
}

void test_virtual_id_queries_match_small_exhaustive_oracle() {
    NodeStore store;
    uint64_t first = 1;
    for (uint64_t block = 0; block < 4; ++block) {
        store.reserve_logical_id_span(first, 12, {{first + 2, 3, 4}});
        first += 12;
    }
    const uint64_t frontier = store.next_auto_id();
    std::vector<bool> virtual_ids(static_cast<size_t>(frontier + 2), false);
    for (uint64_t block = 0; block < 4; ++block) {
        const uint64_t span_first = 1 + block * 12;
        for (uint64_t tile = 0; tile < 3; ++tile) {
            virtual_ids[static_cast<size_t>(span_first + 2 + tile * 4)] = true;
        }
    }

    for (uint64_t id = 0; id < frontier + 2; ++id) {
        const bool expected = id < frontier &&
            virtual_ids[static_cast<size_t>(id)];
        expect(store.is_virtual_id(id) == expected,
               "G3: point membership matches deterministic expanded oracle");
    }
    for (uint64_t first_id = 1; first_id <= frontier + 1; ++first_id) {
        for (uint64_t stride = 1; stride <= 5; ++stride) {
            for (uint64_t count = 1; count <= 5; ++count) {
                const unsigned __int128 last =
                    static_cast<unsigned __int128>(first_id) +
                    static_cast<unsigned __int128>(stride) * (count - 1);
                bool expected_all_virtual = last < frontier;
                bool expected_no_virtual = last < frontier;
                if (last < frontier) {
                    for (uint64_t offset = 0; offset < count; ++offset) {
                        const uint64_t id = first_id + offset * stride;
                        const bool is_virtual =
                            virtual_ids[static_cast<size_t>(id)];
                        expected_all_virtual &= is_virtual;
                        expected_no_virtual &= !is_virtual;
                    }
                }
                expect(store.is_virtual_id_progression(
                           first_id, count, stride) == expected_all_virtual,
                       "G3: virtual AP coverage matches deterministic expanded oracle");
                expect(store.is_allocated_nonvirtual_progression(
                           first_id, count, stride) == expected_no_virtual,
                       "G3: nonvirtual AP validation matches deterministic expanded oracle");
            }
        }
    }
}

void test_virtual_progression_uint64_boundary() {
    const uint64_t max_id = std::numeric_limits<uint64_t>::max();
    const uint64_t first_boundary_id = max_id - 4;
    NodeStore store;

    // One compact AP advances the allocator to the final four allocatable
    // IDs without expanding the enormous prefix into records.
    store.reserve_id_range(1, max_id - 5);
    const auto before_failures = NodeStoreTestAccess::state_counts(store);
    expect(before_failures.next_id == first_boundary_id &&
               before_failures.free_ids == 0 && before_failures.nodes == 0 &&
               store.is_virtual_id(max_id - 5),
           "G2 boundary: compact prefix reaches the final four allocatable IDs");

    const auto expect_boundary_rejection_is_atomic =
        [&store, &before_failures, max_id,
         first_boundary_id](const char* message) {
            const auto after = NodeStoreTestAccess::state_counts(store);
            expect(after.next_id == before_failures.next_id &&
                       after.free_ids == before_failures.free_ids &&
                       after.nodes == before_failures.nodes &&
                       after.virtual_index_blocks ==
                           before_failures.virtual_index_blocks &&
                       after.virtual_progressions ==
                           before_failures.virtual_progressions &&
                       store.is_virtual_id(max_id - 5) &&
                       !store.is_virtual_id(first_boundary_id),
                   message);
        };

    bool rejected_span_overflow = false;
    try {
        store.reserve_logical_id_span(first_boundary_id, 5, {});
    } catch (const std::invalid_argument&) {
        rejected_span_overflow = true;
    }
    expect(rejected_span_overflow,
           "G2 boundary: exclusive span end may not wrap past UINT64_MAX");
    expect_boundary_rejection_is_atomic(
        "G2 boundary: span-overflow rejection preserves allocator and store state");

    bool rejected_terminal_max = false;
    try {
        store.reserve_logical_id_span(
            first_boundary_id, 4, {{max_id - 3, 2, 3}});
    } catch (const std::invalid_argument&) {
        rejected_terminal_max = true;
    }
    expect(rejected_terminal_max,
           "G2 boundary: a progression ending at UINT64_MAX is outside the allocatable span");
    expect_boundary_rejection_is_atomic(
        "G2 boundary: terminal-sentinel rejection preserves allocator and store state");

    bool rejected_endpoint_touch = false;
    try {
        store.reserve_logical_id_span(
            first_boundary_id, 4,
            {{max_id - 4, 2, 3}, {max_id - 1, 1, 1}});
    } catch (const std::invalid_argument&) {
        rejected_endpoint_touch = true;
    }
    expect(rejected_endpoint_touch,
           "G2 boundary: exact overlap at the final allocatable endpoint is rejected");
    expect_boundary_rejection_is_atomic(
        "G2 boundary: endpoint-overlap rejection preserves allocator and store state");

    bool rejected_mixed_stride_collision = false;
    try {
        store.reserve_logical_id_span(
            first_boundary_id, 4,
            {{max_id - 4, 2, 3}, {max_id - 3, 2, 2}});
    } catch (const std::invalid_argument&) {
        rejected_mixed_stride_collision = true;
    }
    expect(rejected_mixed_stride_collision,
           "G2 boundary: mixed-stride APs that meet at UINT64_MAX-1 are rejected");
    expect_boundary_rejection_is_atomic(
        "G2 boundary: mixed-stride rejection preserves allocator and store state");

    const auto work = NodeStoreTestAccess::reserve_counted(
        store, first_boundary_id, 4,
        {{first_boundary_id, 2, 2}, {first_boundary_id + 1, 2, 2}});
    expect(work.envelope_candidates == 1 &&
               work.exact_intersection_tests == 1 &&
               store.next_auto_id() == max_id &&
               store.is_virtual_id_progression(first_boundary_id, 2, 2) &&
               store.is_virtual_id_progression(first_boundary_id + 1, 2, 2) &&
               store.is_virtual_id(max_id - 1),
           "G2 boundary: disjoint interleaved APs reserve through UINT64_MAX-1 exactly");
}

// ---------------------------------------------------------------- Part B --
// NodeStoreGraphSource delegation.
void test_node_store_graph_source() {
    NodeStoreGraphSource src;
    expect(src.dep_free_nodes().empty(), "B: empty source yields no nodes");

    OnlineNode x;
    x.global_id = 5;
    x.kind = NodeKind::CommSend;
    x.comm.bytes = 42;
    x.request_id = "req-5";
    src.store().add_node(x);

    auto views = src.dep_free_nodes();
    expect(views.size() == 1 && views[0].global_id == 5 &&
               views[0].comm.bytes == 42,
           "B: store-backed dep_free_nodes returns the view");
    expect(!src.dep_free_nodes().empty(),
           "B: dep_free_nodes is non-consuming");
    const auto looked = src.lookup(5);
    expect(looked.has_value() && looked->request_id == "req-5",
           "B: lookup returns the view");
    expect(!src.lookup(6).has_value(), "B: lookup miss is nullopt");

    src.take_node(5);
    expect(src.dep_free_nodes().empty(), "B: take_node consumes");
    src.take_node(5);  // idempotent
    src.finish_node(5);
    src.finish_node(5);  // idempotent
    expect(src.dep_free_nodes().empty(), "B: finished node stays consumed");

    // Child added after the parent finished: edge recorded, child free.
    OnlineNode y;
    y.global_id = 8;
    src.store().add_node(y);
    src.store().add_dependency(5, 8, DepKind::Data);
    views = src.dep_free_nodes();
    expect(views.size() == 1 && views[0].global_id == 8,
           "B: edge to finished parent does not block");

    // Online-mode invariants: no ETFeederNode handle, never static-done, and
    // finish_node releases store dependencies without auto-emitting (the
    // static auto-advance lives behind the mode gate in Workload::call).
    expect(src.et_node(8) == nullptr, "B: online source has no et node");
    expect(!src.static_all_done(), "B: online source is never static-done");
    src.finish_node(8);
    expect(src.dep_free_nodes().empty(), "B: no auto-emit after finish");

    // for_each_dep_free must retain a snapshot while its callback releases a
    // child: that child belongs to the next pass, not this traversal.
    OnlineNode parent;
    parent.global_id = 10;
    parent.kind = NodeKind::Compute;
    OnlineNode child;
    child.global_id = 11;
    child.kind = NodeKind::Compute;
    src.store().add_node(parent);
    src.store().add_node(child);
    src.store().add_dependency(10, 11, DepKind::Data);
    std::vector<uint64_t> first_pass;
    src.for_each_dep_free([&](const NodeView& view) {
        first_pass.push_back(view.global_id);
        src.finish_node(view.global_id);
    });
    expect(first_pass == std::vector<uint64_t>({10}),
           "B: released child is absent from the current snapshot pass");
    std::vector<uint64_t> second_pass;
    src.for_each_dep_free(
        [&](const NodeView& view) { second_pass.push_back(view.global_id); });
    expect(second_pass == std::vector<uint64_t>({11}),
           "B: released child is visible on the following snapshot pass");
}

// ---------------------------------------------------------------- Part F --
// B3: ready collection is a stable, ordered batch separate from actual issue.
// Completing two parents while issuing the captured batch must not splice the
// child into that same pass; its single membership appears in the next one.
void test_ready_issue_order_snapshot() {
    NodeStoreGraphSource src;
    OnlineNode node;
    node.kind = NodeKind::Compute;
    node.name = "parent-1";
    const uint64_t parent1 = src.store().add_node(node);
    node.name = "parent-2";
    const uint64_t parent2 = src.store().add_node(node);
    node.name = "child";
    const uint64_t child = src.store().add_node(node);
    node.name = "independent";
    const uint64_t independent = src.store().add_node(node);
    node.name = "stale-ready";
    const uint64_t stale_ready = src.store().add_node(node);
    src.store().add_dependency(parent1, child, DepKind::Data);
    src.store().add_dependency(parent2, child, DepKind::Data);

    std::vector<uint64_t> batch;
    expect(src.fill_dep_free_id_snapshot(batch),
           "F: NodeStore source supports a separated ready snapshot");
    expect(batch == std::vector<uint64_t>(
                        {parent1, parent2, independent, stale_ready}),
           "F: snapshot retains the ascending original ready order");

    std::vector<uint64_t> issued;
    for (const uint64_t id : batch) {
        if (!src.is_dep_free_member(id)) {
            continue;
        }
        const NodeView* const view = src.lookup_ptr(id);
        expect(view != nullptr, "F: captured ready member remains resolvable");
        if (view == nullptr) {
            continue;
        }
        const uint64_t issued_id = view->global_id;
        issued.push_back(issued_id);
        src.take_node(issued_id);
        // Model an inline completion callback, as with a zero-service node.
        src.finish_node(issued_id);
        if (issued_id == parent1) {
            // A callback may invalidate a later captured member. The record
            // intentionally remains resident, so lookup_ptr alone would not
            // be enough to reject it.
            src.take_node(stale_ready);
            src.finish_node(stale_ready);
        }
    }
    expect(issued == std::vector<uint64_t>({parent1, parent2, independent}),
           "F: actual issue follows the captured same-pass order");
    expect(batch == std::vector<uint64_t>(
                        {parent1, parent2, independent, stale_ready}),
           "F: callback-ready child does not mutate the captured batch");
    expect(src.lookup_ptr(stale_ready) != nullptr &&
               !src.is_dep_free_member(stale_ready),
           "F: snapshot membership rejects a finished but still resident id");

    std::vector<uint64_t> next_batch;
    expect(src.fill_dep_free_id_snapshot(next_batch),
           "F: next issue pass can capture a new ready batch");
    expect(next_batch == std::vector<uint64_t>({child}),
           "F: child released by multiple parents appears once next pass");
}

// ---------------------------------------------------------------- Part D --
// Phase-3 sensing: per-rank injected-unfinished ledger summary (方案 §6.2
// 操作 1 / contract ⑥: 摘要按 compute ops / comm bytes / 预计剩余服务量 /
// 资源状态分类,可回溯 request/stage/generation; finished 节点不计数).
void test_injected_unfinished_summary() {
    NodeStore store;

    // Empty store: all-zero summary with the caller-provided rank.
    auto empty = store.injected_unfinished_summary(3);
    expect(empty.rank == 3 && empty.node_count == 0 &&
               empty.compute_ops == 0 && empty.comm_bytes == 0 &&
               empty.estimated_remaining_ns == 0 &&
               empty.in_flight_node_count == 0 &&
               empty.free_node_count == 0 && empty.per_request.empty(),
           "D: empty store summary all zero");

    // One unfinished compute node (prefill, gen 0): classified and traceable.
    OnlineNode comp;
    comp.kind = NodeKind::Compute;
    comp.compute.num_ops = 1000;
    comp.compute.runtime_ns = 500;
    comp.request_id = "req-A";
    comp.stage = "prefill";
    comp.generation = 0;
    const uint64_t c1 = store.add_node(comp);

    auto s1 = store.injected_unfinished_summary(0);
    expect(s1.rank == 0 && s1.node_count == 1 && s1.compute_ops == 1000 &&
               s1.estimated_remaining_ns == 500 &&
               s1.free_node_count == 1 && s1.in_flight_node_count == 0 &&
               s1.per_request.size() == 1 &&
               s1.per_request[0].request_id == "req-A" &&
               s1.per_request[0].stage == "prefill" &&
               s1.per_request[0].generation == 0 &&
               s1.per_request[0].node_count == 1 &&
               s1.per_request[0].compute_ops == 1000 &&
               s1.per_request[0].comm_bytes == 0 &&
               s1.per_request[0].estimated_remaining_ns == 500,
           "D: unfinished compute counted with classification");

    // mark_issued -> resource state moves from free to in-flight.
    store.mark_issued(c1);
    auto s2 = store.injected_unfinished_summary(0);
    expect(s2.free_node_count == 0 && s2.in_flight_node_count == 1 &&
               s2.in_flight_gpu_ops == 1000 && s2.node_count == 1,
           "D: issued-not-finished counted as in-flight resource");

    // Comm nodes (send + collective): bytes classified; per-request grouping
    // by (request_id, stage, generation), groups sorted.
    OnlineNode send;
    send.kind = NodeKind::CommSend;
    send.comm.bytes = 2048;
    send.request_id = "req-A";
    send.stage = "prefill";
    send.generation = 0;
    const uint64_t cs = store.add_node(send);
    OnlineNode coll;
    coll.kind = NodeKind::CommCollective;
    coll.coll.bytes = 4096;
    coll.request_id = "req-B";
    coll.stage = "decode";
    coll.generation = 1;
    const uint64_t cc = store.add_node(coll);
    // Metadata: node count only, no compute/comm attribution.
    OnlineNode meta;
    meta.kind = NodeKind::Metadata;
    meta.request_id = "req-C";
    meta.stage = "prefill";
    meta.generation = 0;
    store.add_node(meta);

    auto s3 = store.injected_unfinished_summary(0);
    expect(s3.node_count == 4 && s3.compute_ops == 1000 &&
               s3.comm_bytes == 2048 + 4096 &&
               s3.estimated_remaining_ns == 500,
           "D: comm bytes classified (send + coll), metadata count-only");
    expect(s3.per_request.size() == 3,
           "D: per-request grouping by (request, stage, generation)");
    expect(s3.per_request[0].request_id == "req-A" &&
               s3.per_request[0].node_count == 2 &&
               s3.per_request[0].comm_bytes == 2048 &&
               s3.per_request[0].compute_ops == 1000,
           "D: req-A group aggregates comp + send");
    expect(s3.per_request[1].request_id == "req-B" &&
               s3.per_request[1].stage == "decode" &&
               s3.per_request[1].generation == 1 &&
               s3.per_request[1].comm_bytes == 4096 &&
               s3.per_request[1].estimated_remaining_ns == 0,
           "D: req-B decode group classified");
    expect(s3.per_request[2].request_id == "req-C" &&
               s3.per_request[2].node_count == 1 &&
               s3.per_request[2].compute_ops == 0 &&
               s3.per_request[2].comm_bytes == 0,
           "D: metadata group counts nodes only");

    // finish_node -> the node leaves the summary (terminal nodes excluded).
    store.finish_node(c1);
    auto s4 = store.injected_unfinished_summary(0);
    expect(s4.node_count == 3 && s4.compute_ops == 0 &&
               s4.comm_bytes == 2048 + 4096 &&
               s4.in_flight_node_count == 0 && s4.in_flight_gpu_ops == 0 &&
               s4.free_node_count == 3,
           "D: finished node leaves the summary");
    store.finish_node(cs);
    store.finish_node(cc);
    store.finish_node(/* metadata id */ 4);
    auto s5 = store.injected_unfinished_summary(0);
    expect(s5.node_count == 0 && s5.compute_ops == 0 &&
               s5.comm_bytes == 0 && s5.per_request.empty(),
           "D: all-finished summary back to zero");
}

// ---------------------------------------------------------------- Part C --
// ETFeederGraphSource adapter over the real synthetic trace.
void test_et_feeder_graph_source(const std::string& et_path) {
    Chakra::FeederV3::ETFeeder feeder(et_path);
    ETFeederGraphSource src(&feeder, 0);

    auto views = src.dep_free_nodes();
    // The fixture emits dependency-free nodes only: 3 core nodes (invalid /
    // comp / all_reduce) plus, on the 26 boundary-port ranks (rank 0 is
    // one), the sh_3.0 remote MEM_LOAD (node 3) and the HBM-restore
    // MEM_LOAD (node 4) -- 5 total. The lifecycle below drives whatever
    // the fixture actually emits, keeping the assertions exact for any
    // node count.
    expect(views.size() >= 3, "C: at least 3 dependency-free nodes");
    std::vector<uint64_t> free_ids;
    for (const auto& v : views) {
        free_ids.push_back(v.global_id);
    }
    expect(free_ids.size() >= 3 && free_ids[0] == 0 &&
               free_ids[1] == 1 && free_ids[2] == 2,
           "C: free ids ascending");
    if (free_ids.size() < 3) {
        return;  // failures already recorded; nothing safe to index below
    }

    // node 0: INVALID_NODE
    expect(views[0].kind == NodeKind::Invalid && views[0].node_type == 0,
           "C: node0 kind Invalid / raw type 0");
    expect(views[0].name == "fixture_invalid", "C: node0 name");
    expect(!views[0].is_cpu_op, "C: node0 is_cpu_op false");
    expect(views[0].inputs_values.empty(), "C: node0 inputs_values empty");

    // node 1: COMP_NODE (roofline attrs; no runtime attr)
    expect(views[1].kind == NodeKind::Compute &&
               views[1].node_type == static_cast<uint64_t>(
                                         ChakraProtoMsg::NodeType::COMP_NODE),
           "C: node1 kind Compute");
    expect(views[1].compute.num_ops == 10000, "C: node1 num_ops mapped");
    expect(views[1].compute.tensor_size == 4096,
           "C: node1 tensor_size mapped");
    expect(views[1].compute.runtime_ns == 0,
           "C: node1 runtime_ns 0 (no runtime attr)");
    expect(!views[1].compute.has_remote_weight_bytes,
           "C: node1 remote_weight_bytes absent");

    // node 2: COMM_COLL_NODE all_reduce with pg + involved_dim bool_list
    expect(views[2].kind == NodeKind::CommCollective &&
               views[2].node_type == static_cast<uint64_t>(ChakraProtoMsg::
                                                               NodeType::
                                                                   COMM_COLL_NODE),
           "C: node2 kind CommCollective");
    expect(views[2].coll.comm_type ==
               static_cast<uint64_t>(ChakraProtoMsg::ALL_REDUCE),
           "C: node2 comm_type ALL_REDUCE");
    expect(views[2].coll.bytes == 4096, "C: node2 comm bytes mapped");
    expect(views[2].coll.priority == 0, "C: node2 priority mapped");
    expect(views[2].coll.involved_dim.size() == 2 &&
               views[2].coll.involved_dim[0] && views[2].coll.involved_dim[1],
           "C: node2 involved_dim bool_list mapped");
    expect(!views[2].coll.pg_name.empty(), "C: node2 pg_name mapped");
    expect(views[2].compute.runtime_ns == 0,
           "C: node2 runtime_ns 0 (no runtime attr)");

    // take / finish lifecycle (all fixture nodes are dependency-free).
    src.take_node(free_ids[0]);
    views = src.dep_free_nodes();
    expect(views.size() == free_ids.size() - 1 &&
               views[0].global_id == free_ids[1],
           "C: take_node consumes from the free set");
    src.finish_node(free_ids[0]);
    expect(!src.static_all_done(), "C: not done while nodes remain");
    for (size_t i = 1; i < free_ids.size(); i++) {
        src.take_node(free_ids[i]);
        src.finish_node(free_ids[i]);
    }
    expect(src.static_all_done(),
           "C: static_all_done when free and ongoing are both empty");
    expect(src.dep_free_nodes().empty(), "C: no free nodes after all taken");

    // lookup / et_node handles after the lifecycle.
    const auto looked = src.lookup(1);
    expect(looked.has_value() && looked->compute.num_ops == 10000,
           "C: lookup after finish still returns the view");
    expect(!src.lookup(42).has_value(),
           "C: lookup miss is nullopt (legacy not-found throws as miss)");
    const auto etn = src.et_node(1);
    expect(etn != nullptr && etn->id() == 1,
           "C: et_node returns the ETFeederNode handle");
    // Legacy lookupNode shape: a wrapper is returned even for unknown ids;
    // dereferencing it for a missing node throws (as in the pre-phase-1
    // Workload.cc), so no nullptr contract is asserted here.
}

}  // namespace

int main(int argc, char* argv[]) {
    std::string et_path;
    for (int i = 1; i < argc; ++i) {
        const std::string arg(argv[i]);
        const std::string prefix = "--fixture-et=";
        if (arg.rfind(prefix, 0) == 0) {
            et_path = arg.substr(prefix.size());
        }
    }

    test_node_store();
    test_node_gc();
    test_virtual_id_ranges();
    test_sparse_virtual_id_span();
    test_virtual_progression_sweep_scales();
    test_virtual_id_query_index_scales_by_reservation();
    test_virtual_id_interval_tree_exactness();
    test_virtual_id_empty_blocks_and_reservation_atomicity();
    test_virtual_id_queries_match_small_exhaustive_oracle();
    test_virtual_progression_uint64_boundary();
    test_node_store_graph_source();
    test_ready_issue_order_snapshot();
    test_injected_unfinished_summary();
    if (et_path.empty()) {
        // R4-N1 port (2026-09-30, from the face repo): only part C needs
        // the materialized fixture ET; the hermetic parts (A/B/D) fully
        // pass without it, so the no-arg invocation is a PASS with the
        // part-C skip printed -- which is what lets this test register in
        // ctest. The manual part-C recipe stays documented in the source
        // header.
        std::fprintf(stderr,
                     "[node_store_test] SKIP part C (no --fixture-et given); "
                     "run make_completion_fixture_et.py first\n");
        if (!g_ok) {
            std::fprintf(stderr,
                         "[node_store_test] FAIL: see messages above\n");
            return 1;
        }
        std::printf("[node_store_test] ALL PASS (hermetic parts A/B/D; "
                    "part C skipped -- needs --fixture-et)\n");
        return 0;
    }
    test_et_feeder_graph_source(et_path);

    if (!g_ok) {
        std::fprintf(stderr, "[node_store_test] FAIL: see messages above\n");
        return 1;
    }
    std::printf("[node_store_test] ALL PASS: NodeStore semantics, "
                "NodeStoreGraphSource delegation, ETFeederGraphSource "
                "adapter mapping, injected-unfinished summary (part D)\n");
    return 0;
}
