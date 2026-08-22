/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

graph_batch_committer_test.cc -- phase-5 GraphBatchCommitter fixture
(方案 §8.1/§8.2 two-phase atomic commit).

The fixture drives a real GraphBatchCommitter (bound to real NodeStore
GraphSources, a real WatchRegistry, a real RequestIngress over an EventQueue
+ DecisionMailbox + ServiceCoordinator -- the alarm path schedules a genuine
future arrival) through:

  Part A  static/pure checks: compute_touched_ranks (sorted unique, rank
          filtering, malformed-entry tolerance).
  Part B  29 deliberately illegal batches (one per Phase-A rule category:
          epoch, node structure, edge structure, cycle, watch structure /
          eligibility / identity, send-recv pairing, collective split,
          assignment, kv action, alarm, touched_ranks). EVERY case must be
          rejected by validate() AND leave the official state completely
          untouched (零节点/零 watch/零账本动作: store node counts, free
          sets, store-id map, watch registry, in-flight/prefill-drained
          tracking, counters, issue-pass calls -- all byte-identical to the
          pre-validate snapshot).
  Part C  positive commit of the baseline multi-rank batch: counters
          (graph_batch_count/total_nodes/max_nodes_per_batch/watches/
          assignments/kv_actions/future_alarms), per-rank store node counts
          and free sets (the parent edges block the children), store-id map
          persistence, watch registration, in-flight tracking, and the
          issue pass restricted to the touched ranks {0, 1, 2} exactly.
  Part D  post-commit negatives: mutations that were legal before the
          commit (alarm for a request that is in-flight NOW) must be
          rejected with the post-commit state unchanged.
  Part E  zero-node accounting batch (REQUEST_COMPLETE only): validates and
          commits with zero nodes, zero watches, zero issue calls; the
          completed request leaves the in-flight set.
  Part F  single-node batch (with a CROSS-BATCH parent edge: the parent is
          a node committed in Part C, resolved through the persistent
          (rank, json id) -> store id map): commits and increments
          single_node_bridge_count -- the official path asserts this stays
          0 (方案 §8.3), the fixture proves the counter can be exercised.
  Part I  拼 batch §3.1 前置验证, 2026-08-22 (ported from the sh_1.0
          mother's Part H): a two-request decode train (shared aggregate
          body nodes under the batch-namespace request_id
          "batch_train_0_1" -- NOT a real request -- plus one exit marker
          per member per rank and one per-train ALL_REDUCE end barrier)
          walks validate() + commit() + the full watch-fire chain: both
          member decode watches fire, the namespace body / barrier
          terminals feed nothing, in-flight/prefill-drained tracking and
          the counters stay exact. Also proves the one rule that DOES
          guard the schema boundary: a member decode watch whose prefill
          never drained (no PREFILL_DRAIN delta fact) is still rejected
          with zero side effects.
  Part J  拼 batch §3.1 前置验证, 2026-08-22 (variant, from the mother's
          Part I): one train carrying BOTH a drain marker (prefill watch,
          gen 0) and an exit marker (decode watch, gen 1) of two DIFFERENT
          requests -- the mixed prefill-chunk + decode-token folding case;
          both watches fire. Plus a sentinel-watch probe: the
          batch-namespace "batch_train_" prefill watch bypasses in-flight
          eligibility (T_max 截断列车的完成信号通道) and fires on its
          marker.

Also asserts the pre-phase-5 fixture tolerance: a batch WITHOUT the
touched_ranks field validates identically (has_touched_ranks == false).

Build: the CMake target
AstraSim_Analytical_Congestion_Aware_GraphBatchCommitterTest (build with
cmake --build build/astra_analytical/build_congestion_aware -j).
Run (from template/astra-sim-wscllm):
  build/astra_analytical/build_congestion_aware/bin/\
      AstraSim_Analytical_Congestion_Aware_GraphBatchCommitterTest
Exit code 0 on ALL PASS.
*******************************************************************************/

#include <astra-network-analytical/common/EventQueue.h>

#include <cstdio>
#include <cstdlib>
#include <map>
#include <set>
#include <string>
#include <vector>

#include <json/json.hpp>

#include "astra-sim/workload/execution_driven/DecisionMailbox.hh"
#include "astra-sim/workload/execution_driven/GraphBatchCommitter.hh"
#include "astra-sim/workload/execution_driven/NodeStore.hh"
#include "astra-sim/workload/execution_driven/RequestIngress.hh"
#include "astra-sim/workload/execution_driven/ServiceCoordinator.hh"
#include "astra-sim/workload/execution_driven/WatchRegistry.hh"

using namespace AstraSim::ExecutionDriven;
using namespace NetworkAnalytical;

namespace {

bool g_ok = true;

void expect(bool cond, const char* what) {
    if (!cond) {
        std::fprintf(stderr, "[graph_batch_committer_test] FAIL: %s\n", what);
        g_ok = false;
    }
}

// ------------------------------------------------------------ fixture ----
// The full commit environment: three real NodeStore-backed graph sources
// (the official online driver owns one per rank), a real WatchRegistry, a
// real RequestIngress bound to an EventQueue + DecisionMailbox +
// ServiceCoordinator (so future-alarm commits schedule genuine arrival
// alarms), and an issue_rank lambda recording every call.
struct Fixture {
    EventQueue eq;
    DecisionMailbox mailbox;
    ServiceCoordinator svc;
    RequestIngress ingress;
    WatchRegistry registry;
    std::vector<std::shared_ptr<NodeStoreGraphSource>> sources;
    std::vector<int> issue_calls;
    GraphBatchCommitter committer;

    Fixture()
        : ingress(4096),
          committer(GraphBatchCommitter::Context{
              3, &sources, &registry, &ingress,
              [this](int rank) { issue_calls.push_back(rank); }}) {
        ingress.bind(&eq, &mailbox, &svc);
        // One DISTINCT NodeStore per rank (a fill-constructed vector would
        // share a single store across all three "ranks").
        sources.reserve(3);
        for (int rank = 0; rank < 3; ++rank) {
            sources.push_back(std::make_shared<NodeStoreGraphSource>());
        }
    }
};

// Observable state snapshot: everything the commit can write, flattened
// for byte-level equality comparison (零副作用 assertion of Part B/D).
struct Snapshot {
    std::vector<size_t> pending;              // per-rank store node counts
    std::vector<std::vector<uint64_t>> free;  // per-rank free sets
    size_t store_ids = 0;
    size_t watches = 0;
    size_t in_flight = 0;
    size_t prefill_drained = 0;
    std::vector<int> issue_calls;
    uint64_t graph_batch_count = 0;
    uint64_t single_node_bridge_count = 0;
    uint64_t total_nodes = 0;
    uint64_t max_nodes_per_batch = 0;
    uint64_t total_watches = 0;
    uint64_t total_assignments = 0;
    uint64_t total_kv_actions = 0;
    uint64_t total_future_alarms = 0;
};

bool operator==(const Snapshot& a, const Snapshot& b) {
    return a.pending == b.pending && a.free == b.free &&
           a.store_ids == b.store_ids && a.watches == b.watches &&
           a.in_flight == b.in_flight && a.prefill_drained == b.prefill_drained &&
           a.issue_calls == b.issue_calls &&
           a.graph_batch_count == b.graph_batch_count &&
           a.single_node_bridge_count == b.single_node_bridge_count &&
           a.total_nodes == b.total_nodes &&
           a.max_nodes_per_batch == b.max_nodes_per_batch &&
           a.total_watches == b.total_watches &&
           a.total_assignments == b.total_assignments &&
           a.total_kv_actions == b.total_kv_actions &&
           a.total_future_alarms == b.total_future_alarms;
}

Snapshot snapshot_of(const Fixture& f) {
    Snapshot s;
    for (const auto& src : f.sources) {
        s.pending.push_back(src->store().pending_count());
        s.free.push_back(src->store().resolve_free_nodes());
    }
    s.store_ids = f.committer.store_ids().size();
    s.watches = f.registry.size();
    s.in_flight = f.committer.in_flight_requests().size();
    s.prefill_drained = f.committer.prefill_drained_requests().size();
    s.issue_calls = f.issue_calls;
    const auto& c = f.committer.counters();
    s.graph_batch_count = c.graph_batch_count;
    s.single_node_bridge_count = c.single_node_bridge_count;
    s.total_nodes = c.total_nodes;
    s.max_nodes_per_batch = c.max_nodes_per_batch;
    s.total_watches = c.total_watches;
    s.total_assignments = c.total_assignments;
    s.total_kv_actions = c.total_kv_actions;
    s.total_future_alarms = c.total_future_alarms;
    return s;
}

// ------------------------------------------------------- batch builders ----
// The real graph_batch_builder emits the FULL comm and coll objects on
// every node (compute/collective nodes carry the comm defaults src=0,
// dst=0, tag=0 -- verified against the real 20.csv first-30s responses).
// The fixture mirrors that shape exactly; Part G proves the same-tick
// milestone fixture's shape (compute/metadata nodes with an EMPTY comm {})
// stays legal too.
nlohmann::json comm_defaults() {
    return {{"bytes", 0}, {"src", 0}, {"dst", 0}, {"tag", 0}};
}

nlohmann::json coll_defaults() {
    return {{"comm_type", 0}, {"bytes", 0}, {"priority", 0},
            {"pg_name", ""}, {"involved_dim", nlohmann::json::array()}};
}

nlohmann::json compute_node(int rank, uint64_t id, const std::string& req,
                            const std::string& stage,
                            const std::string& name) {
    return {
        {"rank", rank}, {"id", id}, {"type", 4}, {"name", name},
        {"is_cpu_op", false}, {"is_timer_op", false}, {"inputs_values", ""},
        {"request_id", req}, {"stage", stage},
        {"generation", stage == "decode" ? 1 : 0},
        {"compute", {{"num_ops", 1000}, {"tensor_size", 4096},
                     {"runtime_ns", 10000}}},
        {"comm", comm_defaults()},
        {"coll", coll_defaults()},
    };
}

nlohmann::json comm_node(int rank, uint64_t id, int type, int src, int dst,
                         int tag) {
    return {
        {"rank", rank}, {"id", id}, {"type", type},
        {"name", type == 5 ? "send" : "recv"},
        {"is_cpu_op", false}, {"is_timer_op", false}, {"inputs_values", ""},
        {"request_id", "r1"}, {"stage", "prefill"}, {"generation", 0},
        {"compute", {{"num_ops", 1000}, {"tensor_size", 4096},
                     {"runtime_ns", 10000}}},
        {"comm", {{"bytes", 100}, {"src", src}, {"dst", dst}, {"tag", tag}}},
        {"coll", coll_defaults()},
    };
}

nlohmann::json coll_node(int rank, uint64_t id, const std::string& name) {
    return {
        {"rank", rank}, {"id", id}, {"type", 7}, {"name", name},
        {"is_cpu_op", false}, {"is_timer_op", false}, {"inputs_values", ""},
        {"request_id", "r1"}, {"stage", "prefill"}, {"generation", 0},
        {"compute", {{"num_ops", 1000}, {"tensor_size", 4096},
                     {"runtime_ns", 10000}}},
        {"comm", comm_defaults()},
        {"coll", {{"comm_type", 2}, {"bytes", 64}, {"priority", 0},
                  {"pg_name", "tp0"},
                  {"involved_dim", nlohmann::json::array({true, false})}}},
    };
}

nlohmann::json data_edge(int rank, uint64_t from, uint64_t to) {
    return {{"rank", rank}, {"kind", "data"}, {"from", from}, {"to", to}};
}

nlohmann::json prefill_watch(const std::string& request_id,
                             nlohmann::json members) {
    return {{"request_id", request_id}, {"stage", "prefill"},
            {"generation", 0}, {"members", std::move(members)},
            {"statuses", nlohmann::json::array({"Success", "Skipped"})}};
}

// The legal multi-rank baseline: 7 nodes over ranks {0, 1, 2} (compute,
// send, recv and collective types all present), complete send/recv pairing,
// one complete tp0 collective group, a (r1, prefill, 0) watch over 3
// members, one assignment, one kv action and one future alarm for r2.
GraphBatch baseline_batch() {
    GraphBatch b;
    b.batch_id = 0;
    b.source_delivery_sequence = 0;
    b.nodes = nlohmann::json::array({
        compute_node(0, 0, "r1", "prefill", "r0_prefill_comp"),
        comm_node(0, 1, 5, 0, 1, 7),
        compute_node(1, 0, "r1", "prefill", "r1_prefill_comp"),
        comm_node(1, 1, 6, 0, 1, 7),
        coll_node(1, 2, "tp0_barrier"),
        compute_node(2, 0, "r1", "prefill", "r2_prefill_comp"),
        coll_node(2, 1, "tp0_barrier"),  // same collective name on both ranks
    });
    b.parent_edges = nlohmann::json::array({
        data_edge(0, 0, 1),
        data_edge(1, 0, 1),
        data_edge(1, 1, 2),
    });
    b.watches = nlohmann::json::array({
        prefill_watch("r1", {{"0", 1}, {"1", 2}, {"2", 1}}),
    });
    b.assignments = nlohmann::json::array({
        {{"request_id", "r1"}, {"prefill_instance_index", 0},
         {"decode_instance_index", 1}},
    });
    b.kv_actions = nlohmann::json::array({
        {{"event_type", "admit"}, {"trigger_request_id", "r1"},
         {"session_id", "s1"}, {"context_tokens", 5}},
    });
    b.future_alarms = nlohmann::json::array({
        {{"arrival_world_ns", 200},
         {"envelope", {{"request_id", "r2"}, {"session_id", "s1"},
                       {"turn_index", 1}, {"prefill_length", 100},
                       {"decode_length", 10},
                       {"inter_request_interval_ns", 20000000}}}},
    });
    b.touched_ranks = nlohmann::json::array({0, 1, 2});
    b.has_touched_ranks = true;
    return b;
}

// The legal baseline delta: r1 arrives at tick 100 (delivery 0).
StateDelta baseline_delta() {
    StateDelta d;
    d.delivery_sequence = 0;
    d.delivery_epoch = 0;
    d.tick = 100;
    DecisionEvent arrival;
    arrival.reason = DecisionReason::ARRIVAL;
    arrival.request_id = "r1";
    arrival.stage = "prefill";
    arrival.generation = 0;
    arrival.payload.session_id = "s1";
    arrival.payload.prefill_length = 100;
    arrival.payload.decode_length = 10;
    d.events.push_back(arrival);
    return d;
}

// ----------------------------------------------------------- Part A ------
void test_static_checks(const Fixture& f) {
    const GraphBatch base = baseline_batch();
    expect(GraphBatchCommitter::compute_touched_ranks(base, 3) ==
               std::vector<int>({0, 1, 2}),
           "A: compute_touched_ranks covers all three ranks");
    expect(GraphBatchCommitter::compute_touched_ranks(base, 2) ==
               std::vector<int>({0, 1}),
           "A: compute_touched_ranks filters out-of-range ranks");
    GraphBatch junk = base;
    junk.nodes[0] = "junk";  // malformed entries are skipped, not fatal
    expect(GraphBatchCommitter::compute_touched_ranks(junk, 3) ==
               std::vector<int>({0, 1, 2}),
           "A: compute_touched_ranks tolerates malformed entries");
}

// ----------------------------------------------------------- Part B ------
// Every illegal batch must be rejected with the state byte-identical to the
// pre-validate snapshot (zero nodes / zero watches / zero ledger actions).
void expect_reject(Fixture& f, const StateDelta& delta, GraphBatch batch,
                   const char* what) {
    const Snapshot before = snapshot_of(f);
    const auto err = f.committer.validate(delta, batch);
    expect(err.has_value(), what);
    if (!err.has_value()) {
        std::fprintf(stderr,
                     "[graph_batch_committer_test] unexpected validation "
                     "PASS: %s\n", what);
    }
    expect(snapshot_of(f) == before, "B: validate() left zero side effects");
    if (!(snapshot_of(f) == before)) {
        std::fprintf(stderr,
                     "[graph_batch_committer_test] state mutated by %s\n",
                     what);
    }
}

void test_negative_cases(Fixture& f) {
    const StateDelta delta = baseline_delta();
    const GraphBatch base = baseline_batch();

    {
        GraphBatch b = base;
        b.error = "boom";
        expect_reject(f, delta, b, "B: batch with an error field rejected");
    }
    {
        GraphBatch b = base;
        b.batch_id = 5;
        b.source_delivery_sequence = 5;
        expect_reject(f, delta, b, "B: batch_id != delivery_sequence");
    }
    {
        GraphBatch b = base;
        b.nodes[0]["rank"] = 99;
        expect_reject(f, delta, b, "B: node rank out of range");
    }
    {
        GraphBatch b = base;
        b.nodes[1]["id"] = 0;  // duplicate id 0 on rank 0
        expect_reject(f, delta, b, "B: duplicate node id within a rank");
    }
    {
        GraphBatch b = base;
        b.nodes[0]["type"] = 9;
        expect_reject(f, delta, b, "B: node type out of range");
    }
    {
        GraphBatch b = base;
        b.nodes[0]["stage"] = "chat";
        expect_reject(f, delta, b, "B: node stage not prefill/decode");
    }
    {
        GraphBatch b = base;
        b.nodes[5]["generation"] = 1;  // prefill node with decode generation
        expect_reject(f, delta, b, "B: node generation != stage");
    }
    {
        GraphBatch b = base;
        b.nodes[5]["request_id"] = "";
        expect_reject(f, delta, b, "B: node with empty request_id");
    }
    {
        GraphBatch b = base;
        b.parent_edges[0]["kind"] = "control";
        expect_reject(f, delta, b, "B: parent edge kind != data");
    }
    {
        GraphBatch b = base;
        b.parent_edges[0]["from"] = 1;  // from == to == 1
        expect_reject(f, delta, b, "B: self-loop parent edge");
    }
    {
        GraphBatch b = base;
        b.parent_edges[0]["from"] = 999;
        expect_reject(f, delta, b, "B: unresolved parent endpoint");
    }
    {
        GraphBatch b = base;
        b.parent_edges[0]["to"] = 999;
        expect_reject(f, delta, b, "B: child endpoint not in this batch");
    }
    {
        GraphBatch b = base;
        b.parent_edges.push_back(data_edge(1, 2, 0));  // 0->1->2->0
        expect_reject(f, delta, b, "B: in-batch parent edge cycle");
    }
    {
        GraphBatch b = base;
        b.watches[0]["members"]["0"] = 99;
        expect_reject(f, delta, b, "B: watch member not a node of the batch");
    }
    {
        GraphBatch b = base;
        b.watches[0]["statuses"][1] = "Failed";
        expect_reject(f, delta, b, "B: unknown watch status");
    }
    {
        GraphBatch b = base;
        b.watches[0]["generation"] = 1;  // prefill watch with decode gen
        expect_reject(f, delta, b, "B: watch generation != stage");
    }
    {
        GraphBatch b = base;
        b.watches.push_back(
            prefill_watch("r2", {{"0", 1}}));  // r2 never arrived
        expect_reject(f, delta, b,
                      "B: prefill watch for a request not in-flight");
    }
    {
        GraphBatch b = base;
        b.watches.push_back(
            prefill_watch("r1", {{"0", 1}}));  // duplicate identity
        expect_reject(f, delta, b,
                      "B: duplicate watch identity in the batch");
    }
    {
        GraphBatch b = base;
        b.nodes[3]["comm"]["tag"] = 8;  // (0,1,7) send-only, (0,1,8) recv-only
        expect_reject(f, delta, b,
                      "B: send/recv pair incomplete within the batch");
    }
    {
        GraphBatch b = base;
        b.nodes[6]["name"] = "r2_tp0_barrier_x";  // split tp0 group
        expect_reject(f, delta, b,
                      "B: split collective group fails closed");
    }
    {
        GraphBatch b = base;
        b.assignments[0].erase("request_id");
        expect_reject(f, delta, b, "B: assignment without request_id");
    }
    {
        GraphBatch b = base;
        b.kv_actions = nlohmann::json::array({{{"foo", "bar"}}});
        expect_reject(f, delta, b,
                      "B: kv action missing event_type/trigger_request_id");
    }
    {
        GraphBatch b = base;
        b.future_alarms[0]["arrival_world_ns"] = 50;  // past the tick
        expect_reject(f, delta, b, "B: past future alarm");
    }
    {
        GraphBatch b = base;
        b.future_alarms.push_back(b.future_alarms[0]);  // r2 twice
        expect_reject(f, delta, b, "B: duplicate alarm request_id");
    }
    {
        GraphBatch b = base;
        b.future_alarms[0]["envelope"]["request_id"] = "r1";  // in-flight
        expect_reject(f, delta, b, "B: alarm for an in-flight request");
    }
    {
        GraphBatch b = base;
        b.touched_ranks = nlohmann::json::array({0, 1});
        expect_reject(f, delta, b, "B: touched_ranks != node rank set");
    }
    {
        GraphBatch b = base;
        b.touched_ranks = nlohmann::json::array({2, 0, 1});
        expect_reject(f, delta, b, "B: touched_ranks not sorted unique");
    }
    {
        GraphBatch b = base;
        b.nodes = nlohmann::json::array({"junk"});
        expect_reject(f, delta, b, "B: non-object node entry");
    }
    // decode-eligibility: the same graph as decode generation-1 watches,
    // but the delta only carries the ARRIVAL (prefill never drained).
    {
        GraphBatch b = base;
        for (auto& node : b.nodes) {
            node["stage"] = "decode";
            node["generation"] = 1;
        }
        b.watches[0]["stage"] = "decode";
        b.watches[0]["generation"] = 1;
        expect_reject(f, delta, b,
                      "B: decode watch whose prefill has not drained");
    }
}

// ----------------------------------------------------------- Part C ------
void test_positive_commit(Fixture& f) {
    const StateDelta delta = baseline_delta();
    const GraphBatch base = baseline_batch();

    // Pre-phase-5 fixture tolerance: a batch without touched_ranks still
    // validates (absent field != declared empty array).
    GraphBatch no_touched = base;
    no_touched.has_touched_ranks = false;
    no_touched.touched_ranks = nlohmann::json::array();
    expect(!f.committer.validate(delta, no_touched).has_value(),
           "C: batch without touched_ranks field validates (pre-5 fixture)");

    expect(!f.committer.validate(delta, base).has_value(),
           "C: baseline batch validates clean");
    f.committer.commit(delta, base);

    const auto& c = f.committer.counters();
    expect(c.graph_batch_count == 1, "C: graph_batch_count == 1");
    expect(c.single_node_bridge_count == 0, "C: single_node_bridge_count 0");
    expect(c.total_nodes == 7, "C: total_nodes == 7");
    expect(c.max_nodes_per_batch == 7, "C: max_nodes_per_batch == 7");
    expect(c.total_watches == 1, "C: total_watches == 1");
    expect(c.total_assignments == 1, "C: total_assignments == 1");
    expect(c.total_kv_actions == 1, "C: total_kv_actions == 1");
    expect(c.total_future_alarms == 1, "C: total_future_alarms == 1");

    expect(f.sources[0]->store().pending_count() == 2,
           "C: rank 0 store holds 2 nodes");
    expect(f.sources[1]->store().pending_count() == 3,
           "C: rank 1 store holds 3 nodes");
    expect(f.sources[2]->store().pending_count() == 2,
           "C: rank 2 store holds 2 nodes");

    // Persistent (rank, json id) -> store id map. NodeStore ids are
    // PER-RANK (each store starts at 1), so the map keys by (rank, json id).
    const auto& ids = f.committer.store_ids();
    expect(ids.size() == 7, "C: store-id map covers all 7 nodes");
    expect(ids.at(RankNodeKey{0, 0}) == 1 && ids.at(RankNodeKey{0, 1}) == 2,
           "C: rank 0 store ids 1, 2");
    expect(ids.at(RankNodeKey{1, 0}) == 1 && ids.at(RankNodeKey{1, 1}) == 2 &&
               ids.at(RankNodeKey{1, 2}) == 3,
           "C: rank 1 store ids 1, 2, 3 (per-rank id space)");
    expect(ids.at(RankNodeKey{2, 0}) == 1 && ids.at(RankNodeKey{2, 1}) == 2,
           "C: rank 2 store ids 1, 2 (per-rank id space)");

    // Parent edges block their children (free set = the batch's roots).
    expect(f.sources[0]->store().resolve_free_nodes() ==
               std::vector<uint64_t>({1}),
           "C: rank 0 free set {1} (child blocked by the parent edge)");
    expect(f.sources[1]->store().resolve_free_nodes() ==
               std::vector<uint64_t>({1}),
           "C: rank 1 free set {1} (0->1->2 chain)");
    expect(f.sources[2]->store().resolve_free_nodes() ==
               std::vector<uint64_t>({1, 2}),
           "C: rank 2 free set {1, 2} (no edges on rank 2)");

    expect(f.registry.size() == 1, "C: one watch registered");
    expect(f.committer.in_flight_requests() ==
               std::set<std::string>({"r1"}),
           "C: r1 in-flight after the commit");
    expect(f.committer.prefill_drained_requests().empty(),
           "C: nothing prefill-drained yet");
    expect(f.issue_calls == std::vector<int>({0, 1, 2}),
           "C: issue pass covered exactly the touched ranks {0, 1, 2}");
}

// ----------------------------------------------------------- Part D ------
// Post-commit negatives: rejected against the committed state -- e.g. an
// alarm for a request that is in-flight NOW (legal before the commit) --
// and still zero side effects on the committed state.
void test_post_commit_negatives(Fixture& f) {
    const StateDelta delta = baseline_delta();
    const GraphBatch base = baseline_batch();
    {
        GraphBatch b = base;
        b.future_alarms[0]["envelope"]["request_id"] = "r1";  // in-flight NOW
        expect_reject(f, delta, b,
                      "D: alarm for a now in-flight request rejected");
    }
    {
        GraphBatch b = base;
        b.nodes[1]["id"] = 0;  // in-batch duplicate id on rank 0
        expect_reject(f, delta, b,
                      "D: in-batch duplicate node id rejected, state intact");
    }
    const auto& c = f.committer.counters();
    expect(c.graph_batch_count == 1 && c.total_nodes == 7,
           "D: rejected batches never reached the counters");
}

// ----------------------------------------------------------- Part E ------
// Zero-node accounting batch: REQUEST_COMPLETE only, no nodes, no watches,
// no issue calls; the completed request leaves the in-flight set.
void test_zero_node_batch(Fixture& f) {
    StateDelta delta;
    delta.delivery_sequence = 1;
    delta.delivery_epoch = 1;
    delta.tick = 150;
    DecisionEvent done;
    done.reason = DecisionReason::REQUEST_COMPLETE;
    done.request_id = "r1";
    delta.events.push_back(done);

    GraphBatch b;
    b.batch_id = 1;
    b.source_delivery_sequence = 1;
    b.touched_ranks = nlohmann::json::array();
    b.has_touched_ranks = true;

    expect(!f.committer.validate(delta, b).has_value(),
           "E: zero-node batch validates clean");
    f.committer.commit(delta, b);

    expect(f.committer.in_flight_requests().empty(),
           "E: r1 leaves the in-flight set on REQUEST_COMPLETE");
    const auto& c = f.committer.counters();
    expect(c.graph_batch_count == 2, "E: graph_batch_count == 2");
    expect(c.total_nodes == 7, "E: zero-node batch adds no nodes");
    expect(c.max_nodes_per_batch == 7, "E: max unchanged");
    expect(c.total_watches == 1, "E: zero-node batch adds no watches");
    expect(f.issue_calls == std::vector<int>({0, 1, 2}),
           "E: zero-node batch issues no ranks");
    expect(f.registry.size() == 1,
           "E: watch removal stays the caller's job (main_online)");
}

// ----------------------------------------------------------- Part F ------
// Single-node batch with a CROSS-BATCH parent edge (the parent is rank 0's
// json id 1 -- the send node committed in Part C -- resolved through the
// persistent (rank, json id) -> store id map). Commits and increments
// single_node_bridge_count (official path asserts 0; 方案 §8.3).
void test_single_node_batch(Fixture& f) {
    StateDelta delta;
    delta.delivery_sequence = 2;
    delta.delivery_epoch = 2;
    delta.tick = 200;
    DecisionEvent arrival;
    arrival.reason = DecisionReason::ARRIVAL;
    arrival.request_id = "r3";
    arrival.stage = "prefill";
    arrival.generation = 0;
    arrival.payload.session_id = "s1";
    arrival.payload.prefill_length = 50;
    arrival.payload.decode_length = 5;
    delta.events.push_back(arrival);

    GraphBatch b;
    b.batch_id = 2;
    b.source_delivery_sequence = 2;
    b.nodes = nlohmann::json::array({
        compute_node(0, 0, "r3", "prefill", "r3_comp"),
    });
    b.parent_edges = nlohmann::json::array({data_edge(0, 1, 0)});
    b.watches = nlohmann::json::array({
        {{"request_id", "r3"}, {"stage", "prefill"}, {"generation", 0},
         {"members", {{"0", 0}}},
         {"statuses", nlohmann::json::array({"Success"})}},
    });
    b.touched_ranks = nlohmann::json::array({0});
    b.has_touched_ranks = true;

    expect(!f.committer.validate(delta, b).has_value(),
           "F: single-node batch validates clean");
    f.committer.commit(delta, b);

    const auto& c = f.committer.counters();
    expect(c.graph_batch_count == 3, "F: graph_batch_count == 3");
    expect(c.single_node_bridge_count == 1,
           "F: single_node_bridge_count == 1 (the official path asserts 0)");
    expect(c.total_nodes == 8, "F: total_nodes == 8");
    expect(c.max_nodes_per_batch == 7, "F: max unchanged at 7");
    expect(c.total_watches == 2, "F: total_watches == 2");

    expect(f.sources[0]->store().pending_count() == 3,
           "F: rank 0 store holds 3 nodes");
    expect(f.committer.store_ids().at(RankNodeKey{0, 0}) == 3,
           "F: json id 0 of batch 2 -> store id 3 on rank 0");
    // The new node's parent is the PREVIOUS batch's node (store id 2) --
    // the cross-batch edge resolved through the persistent map.
    expect(f.sources[0]->store().resolve_free_nodes() ==
               std::vector<uint64_t>({1}),
           "F: new node blocked by its cross-batch parent");
    expect(f.committer.in_flight_requests() ==
               std::set<std::string>({"r3"}),
           "F: r3 in-flight");
    expect(f.issue_calls == std::vector<int>({0, 1, 2, 0}),
           "F: issue pass touched only rank 0 for the single-node batch");
}

// ----------------------------------------------------------- Part G ------
// Fixture-shape compatibility (same-tick milestone fixture): compute/
// metadata nodes with an EMPTY comm {} and coll {} must validate (the
// comm src/dst/tag range checks are scoped to comm-typed nodes 5/6).
void test_empty_comm_compute_batch(Fixture& f) {
    StateDelta delta;
    delta.delivery_sequence = 3;
    delta.delivery_epoch = 3;
    delta.tick = 300;
    DecisionEvent arrival;
    arrival.reason = DecisionReason::ARRIVAL;
    arrival.request_id = "r9";
    arrival.stage = "prefill";
    arrival.payload.session_id = "s1";
    arrival.payload.prefill_length = 10;
    arrival.payload.decode_length = 1;
    delta.events.push_back(arrival);

    GraphBatch b;
    b.batch_id = 3;
    b.source_delivery_sequence = 3;
    b.nodes = nlohmann::json::array({
        {{"id", 0}, {"rank", 0}, {"type", 1}, {"name", "stm_prefill_0"},
         {"request_id", "r9"}, {"stage", "prefill"}, {"generation", 0},
         {"is_cpu_op", false}, {"is_timer_op", false},
         {"inputs_values", ""},
         {"compute", {{"num_ops", 0}, {"tensor_size", 0},
                      {"runtime_ns", 0}}},
         {"comm", nlohmann::json::object()},
         {"coll", nlohmann::json::object()}},
        {{"id", 1}, {"rank", 0}, {"type", 4}, {"name", "stm_prefill_1"},
         {"request_id", "r9"}, {"stage", "prefill"}, {"generation", 0},
         {"is_cpu_op", false}, {"is_timer_op", false},
         {"inputs_values", ""},
         {"compute", {{"num_ops", 1}, {"tensor_size", 1},
                      {"runtime_ns", 1}}},
         {"comm", nlohmann::json::object()},
         {"coll", nlohmann::json::object()}},
    });
    b.parent_edges = nlohmann::json::array({data_edge(0, 0, 1)});
    b.watches = nlohmann::json::array({
        {{"request_id", "r9"}, {"stage", "prefill"}, {"generation", 0},
         {"members", {{"0", 1}}},
         {"statuses", nlohmann::json::array({"Skipped"})}},
    });
    b.touched_ranks = nlohmann::json::array({0});
    b.has_touched_ranks = true;

    expect(!f.committer.validate(delta, b).has_value(),
           "G: compute/metadata nodes with empty comm/coll validate");
    f.committer.commit(delta, b);
    expect(f.committer.counters().graph_batch_count == 4,
           "G: graph_batch_count == 4");
}

// ----------------------------------------------------------- Part H ------
// Online JSON "hbm_charge" parsing nail (低-2 online key unification): the
// snake_case comm-section key is parsed into NodeView comm.hbm_charge; an
// absent key defaults to true; the legacy kebab spelling comm["hbm-charge"]
// is no longer read (reverse nail pinning the new spelling).
void test_hbm_charge_key_parsing(Fixture& f) {
    StateDelta delta;
    delta.delivery_sequence = 4;
    delta.delivery_epoch = 4;
    delta.tick = 400;
    DecisionEvent arrival;
    arrival.reason = DecisionReason::ARRIVAL;
    arrival.request_id = "r10";
    arrival.stage = "prefill";
    arrival.generation = 0;
    arrival.payload.session_id = "s1";
    arrival.payload.prefill_length = 10;
    arrival.payload.decode_length = 1;
    delta.events.push_back(arrival);

    // comm_node() hardcodes request_id "r1" -- retarget the pair to this
    // batch's request so node/watch (request, stage) coverage matches.
    auto send = comm_node(0, 1, 5, 0, 1, 7);
    send["request_id"] = "r10";
    send["comm"]["hbm_charge"] = false;

    auto recv = comm_node(1, 0, 6, 0, 1, 7);
    recv["request_id"] = "r10";
    recv["comm"]["hbm-charge"] = false;  // legacy kebab spelling: inert

    GraphBatch b;
    b.batch_id = 4;
    b.source_delivery_sequence = 4;
    b.nodes = nlohmann::json::array({
        compute_node(0, 0, "r10", "prefill", "r10_comp"),  // key absent
        send,
        recv,
    });
    b.watches = nlohmann::json::array({
        prefill_watch("r10", {{"0", 1}, {"1", 0}}),
    });
    b.touched_ranks = nlohmann::json::array({0, 1});
    b.has_touched_ranks = true;

    expect(!f.committer.validate(delta, b).has_value(),
           "H: hbm_charge batch validates clean");
    if (const auto err = f.committer.validate(delta, b)) {
        std::fprintf(stderr,
                     "[graph_batch_committer_test] H validate error: %s\n",
                     err->c_str());
    }
    f.committer.commit(delta, b);

    const auto& ids = f.committer.store_ids();
    const auto comp_view = f.sources[0]->lookup(ids.at(RankNodeKey{0, 0}));
    expect(comp_view.has_value() && comp_view->comm.hbm_charge,
           "H: absent comm.hbm_charge defaults to true");
    const auto send_view = f.sources[0]->lookup(ids.at(RankNodeKey{0, 1}));
    expect(send_view.has_value() && !send_view->comm.hbm_charge,
           "H: comm.hbm_charge=false parsed into the NodeView");
    const auto recv_view = f.sources[1]->lookup(ids.at(RankNodeKey{1, 0}));
    expect(recv_view.has_value() && recv_view->comm.hbm_charge,
           "H: legacy comm hbm-charge spelling no longer read (default true)");
}

// ------------------------------------- 拼 batch §3.1 前置验证, 2026-08-22 ----
// Frozen batch-train schema (拼 batch 改造; sh_1.0 母本同构):
//   - shared aggregate body node: request_id is the batch-namespace string
//     ("batch_train_0_1" -- NOT a real request), stage "decode", gen 1,
//     COMP node, any positive compute;
//   - exit marker: one per exiting member per rank, real member id,
//     stage "decode", gen 1, COMP, num_ops = tensor_size = 1; drain marker
//     is the same shape with stage "prefill", gen 0;
//   - one end barrier per train: COMM_COLL_NODE, coll.comm_type =
//     ALL_REDUCE (1), bytes = iteration count, pg_name shared with the
//     body collectives;
//   - per-rank data edges body -> markers -> barrier.
nlohmann::json train_body_node(int rank, uint64_t id,
                               const std::string& train_ns) {
    return {
        {"rank", rank}, {"id", id}, {"type", 4}, {"name", "train_body"},
        {"is_cpu_op", false}, {"is_timer_op", false}, {"inputs_values", ""},
        {"request_id", train_ns}, {"stage", "decode"}, {"generation", 1},
        {"compute", {{"num_ops", 4096}, {"tensor_size", 8192},
                     {"runtime_ns", 20000}}},
        {"comm", comm_defaults()},
        {"coll", coll_defaults()},
    };
}

nlohmann::json train_marker_node(int rank, uint64_t id,
                                 const std::string& req,
                                 const std::string& stage) {
    return {
        {"rank", rank}, {"id", id},
        {"type", 4}, {"name", stage == "decode" ? "train_exit_marker"
                                                : "train_drain_marker"},
        {"is_cpu_op", false}, {"is_timer_op", false}, {"inputs_values", ""},
        {"request_id", req}, {"stage", stage},
        {"generation", stage == "decode" ? 1 : 0},
        {"compute", {{"num_ops", 1}, {"tensor_size", 1}, {"runtime_ns", 1}}},
        {"comm", comm_defaults()},
        {"coll", coll_defaults()},
    };
}

nlohmann::json train_barrier_node(int rank, uint64_t id,
                                  const std::string& train_ns,
                                  uint64_t iterations) {
    return {
        {"rank", rank}, {"id", id}, {"type", 7},
        {"name", "train_end_barrier"},
        {"is_cpu_op", false}, {"is_timer_op", false}, {"inputs_values", ""},
        {"request_id", train_ns}, {"stage", "decode"}, {"generation", 1},
        {"compute", {{"num_ops", 1}, {"tensor_size", 1}, {"runtime_ns", 1}}},
        {"comm", comm_defaults()},
        {"coll", {{"comm_type", 1}, {"bytes", iterations}, {"priority", 0},
                  {"pg_name", "train_pg"},
                  {"involved_dim", nlohmann::json::array({true, false})}}},
    };
}

// One train batch over ranks {0, 1, 2}: per rank a body node (json id 0,
// batch namespace), one marker per member (json ids 1..n -- decode markers
// for exiting members, prefill markers for drain members), the end barrier
// (json id n + 1, ALL_REDUCE, bytes = iterations) and the data edges
// body -> every marker -> barrier (a per-rank diamond, acyclic). One watch
// per member over that member's markers on all three ranks.
struct TrainMember {
    std::string request_id;
    std::string stage;  // "decode" (exit) or "prefill" (drain)
};

GraphBatch train_batch(uint64_t batch_id, const std::string& train_ns,
                       const std::vector<TrainMember>& members,
                       uint64_t iterations) {
    GraphBatch b;
    b.batch_id = batch_id;
    b.source_delivery_sequence = batch_id;
    const uint64_t barrier_id = members.size() + 1;
    for (int rank = 0; rank < 3; ++rank) {
        b.nodes.push_back(train_body_node(rank, 0, train_ns));
        for (size_t m = 0; m < members.size(); ++m) {
            b.nodes.push_back(train_marker_node(rank, m + 1,
                                                members[m].request_id,
                                                members[m].stage));
            b.parent_edges.push_back(data_edge(rank, 0, m + 1));
            b.parent_edges.push_back(data_edge(rank, m + 1, barrier_id));
        }
        b.nodes.push_back(
            train_barrier_node(rank, barrier_id, train_ns, iterations));
    }
    for (size_t m = 0; m < members.size(); ++m) {
        nlohmann::json member_ids;
        for (int rank = 0; rank < 3; ++rank) {
            member_ids[std::to_string(rank)] = m + 1;
        }
        b.watches.push_back(
            {{"request_id", members[m].request_id},
             {"stage", members[m].stage},
             {"generation", members[m].stage == "decode" ? 1 : 0},
             {"members", std::move(member_ids)},
             {"statuses", nlohmann::json::array({"Success", "Skipped"})}});
    }
    b.touched_ranks = nlohmann::json::array({0, 1, 2});
    return b;
}

DecisionEvent arrival_event(const std::string& req) {
    DecisionEvent ev;
    ev.reason = DecisionReason::ARRIVAL;
    ev.request_id = req;
    ev.stage = "prefill";
    ev.generation = 0;
    ev.payload.session_id = "s1";
    ev.payload.prefill_length = 32;
    ev.payload.decode_length = 8;
    return ev;
}

DecisionEvent drain_event(const std::string& req) {
    DecisionEvent ev;
    ev.reason = DecisionReason::PREFILL_DRAIN;
    ev.request_id = req;
    ev.stage = "prefill";
    ev.generation = 0;
    ev.payload.watch_member_count = 3;
    return ev;
}

// Drive one committed node to terminal exactly like the online path would:
// the NodeStore dependency release (Workload::call owns it in the real
// system) plus the completion fact into the WatchRegistry
// (online_completion_hook's job) with the store's OWN meta for the node.
void drive_terminal(Fixture& f, int rank, uint64_t store_id,
                    NodeTerminalStatus status) {
    f.sources[rank]->store().finish_node(store_id);
    const auto meta = f.sources[rank]->store().meta_for(store_id);
    expect(meta.has_value(), "train: meta_for the driven node");
    if (meta.has_value()) {
        f.registry.on_node_terminal(
            CompletionKey{rank, store_id, meta->generation}, *meta, status);
    }
}

// ----------------------------------------------------------- Part I ------
// The two-request decode train (from the sh_1.0 mother's Part H): epoch 1
// lands the arrivals + req_A's prefill drain (a zero-node accounting
// batch), epoch 2 commits the train (req_B's prefill drain is a delta
// fact of the SAME epoch -- delta facts first). The negative probe proves
// the schema boundary that still guards: without req_B's drain fact, its
// decode watch is rejected (eligibility). Counter/pending expectations are
// captured as DELTAS against the pre-Part-I snapshot (face's Part H
// hbm-charge probe shifted the absolutes vs the mother).
void test_train_batch_two_members(Fixture& f) {
    const Snapshot before = snapshot_of(f);

    // Epoch i1: arrivals + req_A drain, zero-node batch (Part E style).
    StateDelta i1;
    i1.delivery_sequence = 5;
    i1.delivery_epoch = 5;
    i1.tick = 500;
    i1.events.push_back(arrival_event("req_A"));
    i1.events.push_back(arrival_event("req_B"));
    i1.events.push_back(drain_event("req_A"));
    GraphBatch setup;
    setup.batch_id = 5;
    setup.source_delivery_sequence = 5;
    setup.touched_ranks = nlohmann::json::array();
    setup.has_touched_ranks = true;
    expect(!f.committer.validate(i1, setup).has_value(),
           "I: epoch-1 zero-node setup batch validates clean");
    f.committer.commit(i1, setup);
    expect(f.committer.in_flight_requests() ==
               std::set<std::string>({"r3", "r9", "r10", "req_A", "req_B"}),
           "I: both train members in-flight after epoch 1");
    expect(f.committer.prefill_drained_requests() ==
               std::set<std::string>({"req_A"}),
           "I: req_A prefill-drained after epoch 1");

    // The train: 12 nodes (3 ranks x [body, exit marker req_A, exit marker
    // req_B, end barrier]), 2 member decode watches.
    const GraphBatch train =
        train_batch(6, "batch_train_0_1",
                    {{"req_A", "decode"}, {"req_B", "decode"}}, 8);

    // Negative probe: the same train validated against an epoch whose delta
    // carries NO req_B prefill drain -- the member decode watch eligibility
    // rule must block it with zero side effects (this is the one rule the
    // batch schema genuinely leans on; it is NOT relaxed by 拼 batch).
    {
        StateDelta no_drain_b;
        no_drain_b.delivery_sequence = 6;
        no_drain_b.delivery_epoch = 6;
        no_drain_b.tick = 550;
        const Snapshot probe_before = snapshot_of(f);
        const auto err = f.committer.validate(no_drain_b, train);
        expect(err.has_value(),
               "I: decode watch of a never-drained member is blocked");
        if (err.has_value()) {
            expect(err->find("whose prefill has not drained") !=
                       std::string::npos,
                   "I: the block is the decode-watch eligibility rule");
        }
        expect(snapshot_of(f) == probe_before,
               "I: blocked train left zero side effects");
    }

    // Epoch i2: req_B's prefill drain is a delta fact of the train epoch
    // itself (delta facts first -- the realistic first-decode-train tick).
    StateDelta i2;
    i2.delivery_sequence = 6;
    i2.delivery_epoch = 6;
    i2.tick = 550;
    i2.events.push_back(drain_event("req_B"));
    expect(!f.committer.validate(i2, train).has_value(),
           "I: two-request train batch validates clean (zero relaxation)");
    f.committer.commit(i2, train);

    const auto& c = f.committer.counters();
    expect(c.graph_batch_count == before.graph_batch_count + 2,
           "I: graph_batch_count +2 (setup + train)");
    expect(c.total_nodes == before.total_nodes + 12,
           "I: total_nodes +12 (the 12-node train)");
    expect(c.total_watches == before.total_watches + 2,
           "I: total_watches +2 (the member watches)");
    expect(c.max_nodes_per_batch == 12, "I: max_nodes_per_batch == 12");
    expect(f.committer.in_flight_requests() ==
               std::set<std::string>({"r3", "r9", "r10", "req_A", "req_B"}),
           "I: train commit adds no arrivals");
    expect(f.committer.prefill_drained_requests() ==
               std::set<std::string>({"req_A", "req_B"}),
           "I: req_B drained by the train epoch's own delta fact");
    expect(f.registry.size() == before.watches + 2,
           "I: registry holds the 2 member watches");
    expect(f.issue_calls.size() == before.issue_calls.size() + 3,
           "I: the train epoch issued exactly the 3 touched ranks");
    for (int rank = 0; rank < 3; ++rank) {
        expect(f.issue_calls[before.issue_calls.size() + rank] == rank,
               "I: issue pass covered exactly the touched ranks {0, 1, 2}");
        expect(f.sources[rank]->store().pending_count() ==
                   before.pending[rank] + 4,
               "I: per-rank store gained the 4 train nodes");
    }

    // Store-id translation: json ids 0..3 on each rank resolved through the
    // persistent (rank, json id) -> store id map.
    const auto& ids = f.committer.store_ids();
    std::vector<uint64_t> body_ids;
    std::vector<uint64_t> barrier_ids;
    std::vector<std::vector<uint64_t>> marker_ids(2);  // [member][rank]
    for (int rank = 0; rank < 3; ++rank) {
        body_ids.push_back(ids.at(RankNodeKey{rank, 0}));
        marker_ids[0].push_back(ids.at(RankNodeKey{rank, 1}));
        marker_ids[1].push_back(ids.at(RankNodeKey{rank, 2}));
        barrier_ids.push_back(ids.at(RankNodeKey{rank, 3}));
    }

    // Namespace terminals feed nothing: the body / barrier nodes carry the
    // batch-namespace identity ("batch_train_0_1", decode, 1), which has no
    // registered watch -- on_node_terminal is a no-op, no fire.
    for (int rank = 0; rank < 3; ++rank) {
        drive_terminal(f, rank, body_ids[rank], NodeTerminalStatus::Success);
        drive_terminal(f, rank, barrier_ids[rank],
                       NodeTerminalStatus::Success);
    }
    expect(f.registry.fired_and_drain().empty(),
           "I: namespace body/barrier terminals fire no watch");

    // Member markers feed the member watches (one Skipped terminal on the
    // req_B train proves the {Success, Skipped} policy holds for markers).
    for (int rank = 0; rank < 3; ++rank) {
        drive_terminal(f, rank, marker_ids[0][rank],
                       NodeTerminalStatus::Success);
    }
    for (int rank = 0; rank < 3; ++rank) {
        drive_terminal(f, rank, marker_ids[1][rank],
                       rank == 1 ? NodeTerminalStatus::Skipped
                                 : NodeTerminalStatus::Success);
    }
    const std::vector<WatchFire> fires = f.registry.fired_and_drain();
    expect(fires.size() == 2, "I: both member decode watches fired");
    std::set<std::string> fired_members;
    for (const auto& fire : fires) {
        expect(fire.stage == "decode" && fire.generation == 1,
               "I: fire identity is the member decode stage");
        expect(fire.member_count == 3,
               "I: fire covers the member's markers on all 3 ranks");
        expect(fire.member_ranks == std::vector<int>({0, 1, 2}),
               "I: fire member_ranks == {0, 1, 2}");
        fired_members.insert(fire.request_id);
    }
    expect(fired_members == std::set<std::string>({"req_A", "req_B"}),
           "I: exactly req_A and req_B fired");

    // The dependency chain closed: with body + markers finished on every
    // rank, each rank's end barrier entered its free set.
    for (int rank = 0; rank < 3; ++rank) {
        const auto free = f.sources[rank]->store().resolve_free_nodes();
        const std::set<uint64_t> free_set(free.begin(), free.end());
        expect(free_set.count(barrier_ids[rank]) == 1,
               "I: end barrier free after its train finished");
    }
}

// ----------------------------------------------------------- Part J ------
// Variant (from the mother's Part I): ONE train folding a prefill chunk of
// req_C (drain marker, prefill watch, gen 0) and a decode token of req_D
// (exit marker, decode watch, gen 1) -- the mixed train. Both watches fire.
// Then the sentinel probe: the batch-namespace prefill watch
// ("batch_train_9_9") of a T_max-truncated train validates WITHOUT any
// in-flight membership (拼 batch eligibility bypass) and fires.
void test_train_batch_drain_and_exit_markers(Fixture& f) {
    const Snapshot before = snapshot_of(f);
    StateDelta j1;
    j1.delivery_sequence = 7;
    j1.delivery_epoch = 7;
    j1.tick = 600;
    j1.events.push_back(arrival_event("req_C"));
    j1.events.push_back(arrival_event("req_D"));
    j1.events.push_back(drain_event("req_D"));

    const GraphBatch train =
        train_batch(7, "batch_train_2_3",
                    {{"req_C", "prefill"}, {"req_D", "decode"}}, 4);
    expect(!f.committer.validate(j1, train).has_value(),
           "J: mixed drain+exit train validates clean (zero relaxation)");
    f.committer.commit(j1, train);

    const auto& c = f.committer.counters();
    expect(c.graph_batch_count == before.graph_batch_count + 1,
           "J: graph_batch_count +1");
    expect(c.total_nodes == before.total_nodes + 12,
           "J: total_nodes +12 (the 12-node mixed train)");
    expect(c.total_watches == before.total_watches + 2,
           "J: total_watches +2");
    expect(f.committer.in_flight_requests() ==
               std::set<std::string>(
                   {"r3", "r9", "r10", "req_A", "req_B", "req_C", "req_D"}),
           "J: mixed-train members in-flight");
    expect(f.committer.prefill_drained_requests() ==
               std::set<std::string>({"req_A", "req_B", "req_D"}),
           "J: req_D drained by the train epoch's own delta fact");
    expect(f.registry.size() == before.watches + 2,
           "J: registry holds both member watches");

    const auto& ids = f.committer.store_ids();
    for (int rank = 0; rank < 3; ++rank) {
        drive_terminal(f, rank, ids.at(RankNodeKey{rank, 0}),
                       NodeTerminalStatus::Success);  // body (namespace)
        drive_terminal(f, rank, ids.at(RankNodeKey{rank, 1}),
                       NodeTerminalStatus::Success);  // req_C drain marker
        drive_terminal(f, rank, ids.at(RankNodeKey{rank, 2}),
                       NodeTerminalStatus::Success);  // req_D exit marker
    }
    const std::vector<WatchFire> fires = f.registry.fired_and_drain();
    expect(fires.size() == 2, "J: drain watch and exit watch both fired");
    std::map<std::string, std::pair<std::string, uint64_t>> fired;
    for (const auto& fire : fires) {
        fired[fire.request_id] = {fire.stage, fire.generation};
    }
    expect(fired.count("req_C") == 1 &&
               fired.at("req_C") ==
                   std::make_pair(std::string("prefill"), uint64_t{0}),
           "J: req_C fired its prefill drain watch (gen 0)");
    expect(fired.count("req_D") == 1 &&
               fired.at("req_D") ==
                   std::make_pair(std::string("decode"), uint64_t{1}),
           "J: req_D fired its decode exit watch (gen 1)");

    // Sentinel probe: a T_max-truncated train's completion signal -- the
    // watch's request_id IS the batch namespace, stage fixed prefill/gen 0,
    // and the C++ eligibility check must bypass in-flight membership (the
    // signal is train-scoped, not request-scoped).
    StateDelta j2;
    j2.delivery_sequence = 8;
    j2.delivery_epoch = 8;
    j2.tick = 700;
    const GraphBatch sentinel =
        train_batch(8, "batch_train_9_9",
                    {{"batch_train_9_9", "prefill"}}, 4);
    expect(!f.committer.validate(j2, sentinel).has_value(),
           "J: batch-namespace sentinel watch bypasses in-flight "
           "eligibility");
    f.committer.commit(j2, sentinel);
    const auto& sids = f.committer.store_ids();
    for (int rank = 0; rank < 3; ++rank) {
        drive_terminal(f, rank, sids.at(RankNodeKey{rank, 1}),
                       NodeTerminalStatus::Success);  // sentinel marker
    }
    const std::vector<WatchFire> sentinel_fires =
        f.registry.fired_and_drain();
    expect(sentinel_fires.size() == 1,
           "J: sentinel watch fired exactly once");
    if (!sentinel_fires.empty()) {
        expect(sentinel_fires[0].request_id == "batch_train_9_9" &&
                   sentinel_fires[0].stage == "prefill",
               "J: sentinel fire carries the train identity");
    }
}

}  // namespace

int main() {
    Fixture f;
    test_static_checks(f);
    test_negative_cases(f);
    test_positive_commit(f);
    test_post_commit_negatives(f);
    test_zero_node_batch(f);
    test_single_node_batch(f);
    test_empty_comm_compute_batch(f);
    test_hbm_charge_key_parsing(f);
    test_train_batch_two_members(f);
    test_train_batch_drain_and_exit_markers(f);
    if (g_ok) {
        std::printf("[graph_batch_committer_test] ALL PASS\n");
        return 0;
    }
    std::fprintf(stderr, "[graph_batch_committer_test] FAILURES\n");
    return 1;
}
