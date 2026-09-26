/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

node_store_test.cc -- phase-1 step 1-4 NodeStore / GraphSource fixture.

Exercises the step-1-4 mechanisms (方案 §4 步骤 1-4) as a standalone
test -- no network simulation, no baseline artifacts touched:

  Part A  NodeStore semantics: id assignment, dependency blocking/release,
          mark_issued, finish_node idempotency, dead-parent edges, reverse
          index (meta_for), pending_count.
  Part B  NodeStoreGraphSource: store-backed for_each_dep_free / lookup_ptr /
          take_node / finish_node delegation, no-auto-emit, static_all_done
          false (online authority belongs to the ServiceCoordinator).
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

  (The former Part C, the static ETFeederGraphSource adapter over a real
  .et fixture, was removed with the offline static path on 2026-09-26.)

Build: the CMake target AstraSim_Analytical_Congestion_Aware_NodeStoreTest
(build with cmake --build build/astra_analytical/build_congestion_aware -j).
Run: build/astra_analytical/build_congestion_aware/bin/\
      AstraSim_Analytical_Congestion_Aware_NodeStoreTest
Exit code 0 on ALL PASS.
*******************************************************************************/

#include "astra-sim/workload/execution_driven/GraphSource.hh"
#include "astra-sim/workload/execution_driven/NodeStore.hh"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using namespace AstraSim::ExecutionDriven;

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

    // 7 nodes stored so far (1, 2, 7, 3=idc, 4=idd, 5=fa, 6=fb);
    // finished so far: 1, 3, 5 -> 4 pending.
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

// ---------------------------------------------------------------- Part B --
// NodeStoreGraphSource delegation.
void test_node_store_graph_source() {
    NodeStoreGraphSource src;
    // for_each_dep_free is the (non-consuming) free-set read API; collect
    // the handed-out views so the assertions below can inspect them.
    auto collect_free = [&src]() {
        std::vector<NodeView> views;
        src.for_each_dep_free(
            [&](const NodeView& view) { views.push_back(view); });
        return views;
    };
    expect(collect_free().empty(), "B: empty source yields no nodes");

    OnlineNode x;
    x.global_id = 5;
    x.kind = NodeKind::CommSend;
    x.comm.bytes = 42;
    x.request_id = "req-5";
    src.store().add_node(x);

    auto views = collect_free();
    expect(views.size() == 1 && views[0].global_id == 5 &&
               views[0].comm.bytes == 42,
           "B: store-backed for_each_dep_free returns the view");
    expect(!collect_free().empty(),
           "B: for_each_dep_free is non-consuming");
    const NodeView* looked = src.lookup_ptr(5);
    expect(looked != nullptr && looked->request_id == "req-5",
           "B: lookup_ptr returns the view");
    expect(src.lookup_ptr(6) == nullptr, "B: lookup_ptr miss is nullptr");

    src.take_node(5);
    expect(collect_free().empty(), "B: take_node consumes");
    src.take_node(5);  // idempotent
    src.finish_node(5);
    src.finish_node(5);  // idempotent
    expect(collect_free().empty(), "B: finished node stays consumed");

    // Child added after the parent finished: edge recorded, child free.
    OnlineNode y;
    y.global_id = 8;
    src.store().add_node(y);
    src.store().add_dependency(5, 8, DepKind::Data);
    views = collect_free();
    expect(views.size() == 1 && views[0].global_id == 8,
           "B: edge to finished parent does not block");

    // Online-mode invariants: no ETFeederNode handle, never static-done, and
    // finish_node releases store dependencies without auto-emitting (the
    // static auto-advance lives behind the mode gate in Workload::call).
    expect(src.et_node(8) == nullptr, "B: online source has no et node");
    expect(!src.static_all_done(), "B: online source is never static-done");
    src.finish_node(8);
    expect(collect_free().empty(), "B: no auto-emit after finish");

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

}  // namespace

int main() {
    test_node_store();
    test_node_gc();
    test_node_store_graph_source();
    test_injected_unfinished_summary();

    if (!g_ok) {
        std::fprintf(stderr, "[node_store_test] FAIL: see messages above\n");
        return 1;
    }
    std::printf("[node_store_test] ALL PASS: NodeStore semantics, "
                "NodeStoreGraphSource delegation, "
                "injected-unfinished summary (part D)\n");
    return 0;
}
