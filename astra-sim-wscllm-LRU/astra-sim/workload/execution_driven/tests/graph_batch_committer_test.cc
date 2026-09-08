/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in this source tree.

graph_batch_committer_test.cc -- phase-5 GraphBatchCommitter fixture
(方案 §8.1/§8.2 two-phase atomic commit).

The fixture drives a real GraphBatchCommitter (bound to real NodeStore
GraphSources, a real WatchRegistry, a real RequestIngress over an EventQueue
+ DecisionMailbox + ServiceCoordinator -- the alarm path schedules a genuine
future arrival) through:

  Part A  static/pure checks: compute_touched_ranks (sorted unique, rank
          filtering).
  Part B  deliberately illegal batches (one per Phase-A rule category:
          epoch, node structure, edge structure, cycle, watch structure /
          eligibility / identity, send-recv pairing, collective split,
          assignment, kv action, alarm, touched_ranks). EVERY case must be
          rejected by validate() AND leave the official state completely
          untouched (零节点/零 watch/零账本动作: store node counts, free
          sets, store-id map, watch registry, in-flight/prefill-drained
          tracking, counters, issue-pass calls -- all byte-identical to the
          pre-validate snapshot).
          [C1 note] the pure JSON-SHAPE violations (non-object entries,
          unknown keys, kind != "data", unknown watch status, missing
          assignment/kv fields, float fields) moved to the parse layer and
          are covered by parsed_graph_batch_test.cc; the cases below keep
          every committer-STATE rule plus the typed-field domain rules.
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
  Part G  metadata/compute nodes carrying the comm DEFAULTS validate (the
          comm src/dst/tag range checks are scoped to comm-typed nodes 5/6;
          pre-C1 this case used an EMPTY comm {}, which the C1 parse-layer
          key-set rule now rejects -- the same-tick milestone verify service
          was updated to emit the defaults with it).
  Part H  Online JSON "hbm_charge" parsing nail (wscllm; 低-2 online key
          unification): the snake_case comm-section key parses into
          comm.hbm_charge, an absent key defaults to true, and the legacy
          kebab spelling comm["hbm-charge"] is now REJECTED by the C1
          parse-layer key set (fail-closed) -- a reverse nail pinning the
          new spelling.
  Part I  拼 batch §3.1 前置验证,2026-08-22 (sh_1.0 母本 Part H 对应): a two-request decode train
          (shared aggregate body nodes under the batch-namespace
          request_id "batch_train_0_1" -- NOT a real request -- plus one
          exit marker per member per rank and one per-train ALL_REDUCE
          end barrier) walks validate() + commit() + the full watch-fire
          chain: both member decode watches fire, the namespace body /
          barrier terminals feed nothing, in-flight/prefill-drained
          tracking and the counters stay exact. Also proves the one rule
          that DOES guard the schema boundary: a member decode watch whose
          prefill never drained (no PREFILL_DRAIN delta fact) is still
          rejected with zero side effects.
  Part J  拼 batch §3.1 前置验证,2026-08-22 (variant; sh_1.0 母本 Part I
          对应): one train carrying
          BOTH a drain marker (prefill watch, gen 0) and an exit marker
          (decode watch, gen 1) of two DIFFERENT requests -- the mixed
          prefill-chunk + decode-token folding case; both watches fire.
  Part P  C1 (2026-08-29) parse/atomicity: a malformed response throws
          ParseError with the committer state byte-identical (parse is a
          pure local construction), a post-parse TYPED mutation is still
          rejected fail-closed with zero side effects, and a watch-member
          mutation is blocked by the preflight BEFORE any Phase-B assembly
          runs (two-phase atomicity under the typed batch).

Also asserts the pre-phase-5 fixture tolerance: a batch WITHOUT the
touched_ranks field validates identically (has_touched_ranks == false).

Batches are BUILT as response JSON documents and converted through the
production parse_graph_batch (num_ranks = 3), so every fixture batch
exercises the real C1 single-parse path; typed mutations model what a
post-parse corruption of the CommitArg could do (defense in depth: the
committer's own domain rules must still reject them).

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
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <vector>

#include <json/json.hpp>

#include "astra-sim/workload/execution_driven/DecisionMailbox.hh"
#include "astra-sim/workload/execution_driven/GraphBatchCommitter.hh"
#include "astra-sim/workload/execution_driven/NodeStore.hh"
#include "astra-sim/workload/execution_driven/ParsedGraphBatch.hh"
#include "astra-sim/workload/execution_driven/RequestIngress.hh"
#include "astra-sim/workload/execution_driven/ServiceCoordinator.hh"
#include "astra-sim/workload/execution_driven/WatchRegistry.hh"

using namespace AstraSim::ExecutionDriven;
using namespace NetworkAnalytical;

namespace AstraSim {
namespace ExecutionDriven {

// Test-only access to the fixed, rank-bounded preflight scratch.  Production
// code exposes no mutable diagnostics for these implementation details.
struct GraphBatchCommitterTestAccess {
    static void seed_json_id_stream_stamp(GraphBatchCommitter& committer,
                                          uint64_t stamp) {
        committer.json_id_stream_stamp_ = stamp;
    }

    static size_t commit_touched_capacity(const GraphBatchCommitter& committer) {
        return committer.commit_touched_ranks_.capacity();
    }

    static size_t discovered_capacity(const GraphBatchCommitter& committer) {
        return committer.json_id_discovered_touched_ranks_.capacity();
    }
};

}  // namespace ExecutionDriven
}  // namespace AstraSim

namespace {

bool g_ok = true;

void expect(bool cond, const char* what) {
    if (!cond) {
        std::fprintf(stderr, "[graph_batch_committer_test] FAIL: %s\n", what);
        g_ok = false;
    }
}

uint64_t resolved_store_id(const GraphBatchCommitter& committer, int rank,
                          uint64_t json_id) {
    const auto store_id = committer.resolve_store_id(rank, json_id);
    return store_id.value_or(0);  // NodeStore never assigns automatic id 0.
}

bool same_counters(const GraphBatchCommitter::Counters& a,
                   const GraphBatchCommitter::Counters& b) {
    return a.graph_batch_count == b.graph_batch_count &&
           a.single_node_bridge_count == b.single_node_bridge_count &&
           a.total_nodes == b.total_nodes &&
           a.max_nodes_per_batch == b.max_nodes_per_batch &&
           a.total_watches == b.total_watches &&
           a.total_assignments == b.total_assignments &&
           a.total_kv_actions == b.total_kv_actions &&
           a.total_future_alarms == b.total_future_alarms;
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

    static GraphBatchCommitter::Context make_context(Fixture* fixture) {
        GraphBatchCommitter::Context ctx;
        ctx.num_ranks = 3;
        ctx.graph_sources = &fixture->sources;
        ctx.watch_registry = &fixture->registry;
        ctx.ingress = &fixture->ingress;
        ctx.issue_rank = [fixture](int rank) {
            fixture->issue_calls.push_back(rank);
        };
        ctx.communicator_members_for_pg =
            [](const std::string& pg_name,
               const std::vector<int>& /* participant_ranks */)
            -> std::optional<std::vector<int>> {
            if (pg_name == "tp0") {
                return std::vector<int>({1, 2});
            }
            if (pg_name == "train_pg") {
                return std::vector<int>({0, 1, 2});
            }
            return std::nullopt;
        };
        return ctx;
    }

    Fixture()
        : ingress(4096), committer(make_context(this)) {
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
// for byte-level equality comparison (零副作用 assertion of Part B/D/P).
struct Snapshot {
    std::vector<size_t> pending;              // per-rank store node counts
    std::vector<std::vector<uint64_t>> free;  // per-rank free sets
    size_t affine_ranks = 0;
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
           a.affine_ranks == b.affine_ranks && a.watches == b.watches &&
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
    s.affine_ranks = f.committer.rank_affine_count();
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
// C1 (2026-08-29): batches are assembled as response documents and go
// through the production parse (num_ranks = 3). The node/watch/edge JSON
// builders below keep the pre-C1 shapes (the real graph_batch_builder
// emits the FULL comm and coll objects on every node; compute/collective
// nodes carry the comm defaults src=0, dst=0, tag=0 -- verified against
// the real 20.csv first-30s responses).

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

// One batch node, parsed through the production path in isolation.
ParsedNode parse_single_node(const nlohmann::json& node_json, int num_ranks) {
    nlohmann::json resp;
    resp["schema_version"] = 1;
    resp["source_delivery_sequence"] = 0;
    resp["nodes"] = nlohmann::json::array({node_json});
    return parse_graph_batch(resp, num_ranks).nodes.at(0);
}

GraphBatch parse_batch(const nlohmann::json& resp, int num_ranks = 3) {
    return parse_graph_batch(resp, num_ranks);
}

// A typed prefill/decode watch over explicit (rank, json id) members.
ParsedWatch typed_watch(const std::string& request_id, const std::string& stage,
                        std::vector<ParsedWatchMember> members,
                        std::vector<NodeTerminalStatus> statuses) {
    ParsedWatch watch;
    watch.request_id = request_id;
    watch.stage = stage;
    watch.generation = stage == "decode" ? 1 : 0;
    watch.members = std::move(members);
    watch.statuses = std::move(statuses);
    return watch;
}

// The legal multi-rank baseline: 7 nodes over ranks {0, 1, 2} (compute,
// send, recv and collective types all present), complete send/recv pairing,
// one complete tp0 collective group, a (r1, prefill, 0) watch over 3
// members, one assignment, one kv action and one future alarm for r2.
nlohmann::json baseline_response() {
    nlohmann::json resp;
    resp["schema_version"] = 1;
    resp["batch_id"] = 0;
    resp["source_delivery_sequence"] = 0;
    resp["nodes"] = nlohmann::json::array({
        compute_node(0, 0, "r1", "prefill", "r0_prefill_comp"),
        comm_node(0, 1, 5, 0, 1, 7),
        compute_node(1, 0, "r1", "prefill", "r1_prefill_comp"),
        comm_node(1, 1, 6, 0, 1, 7),
        coll_node(1, 2, "tp0_barrier"),
        compute_node(2, 0, "r1", "prefill", "r2_prefill_comp"),
        coll_node(2, 1, "tp0_barrier"),  // same collective name on both ranks
    });
    resp["parent_edges"] = nlohmann::json::array({
        data_edge(0, 0, 1),
        data_edge(1, 0, 1),
        data_edge(1, 1, 2),
    });
    resp["watches"] = nlohmann::json::array({
        prefill_watch("r1", {{"0", 1}, {"1", 2}, {"2", 1}}),
    });
    resp["assignments"] = nlohmann::json::array({
        {{"request_id", "r1"}, {"prefill_instance_index", 0},
         {"decode_instance_index", 1}},
    });
    resp["kv_actions"] = nlohmann::json::array({
        {{"event_type", "admit"}, {"trigger_request_id", "r1"},
         {"session_id", "s1"}, {"context_tokens", 5}},
    });
    resp["future_alarms"] = nlohmann::json::array({
        {{"arrival_world_ns", 200},
         {"envelope", {{"request_id", "r2"}, {"session_id", "s1"},
                       {"turn_index", 1}, {"prefill_length", 100},
                       {"decode_length", 10},
                       {"inter_request_interval_ns", 20000000}}}},
    });
    resp["touched_ranks"] = nlohmann::json::array({0, 1, 2});
    return resp;
}

GraphBatch baseline_batch() {
    return parse_batch(baseline_response());
}

// A response skeleton with only the header (every array absent == empty).
nlohmann::json empty_response(uint64_t sequence) {
    nlohmann::json resp;
    resp["schema_version"] = 1;
    resp["batch_id"] = sequence;
    resp["source_delivery_sequence"] = sequence;
    resp["touched_ranks"] = nlohmann::json::array();
    return resp;
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
    (void)f;
    const GraphBatch base = baseline_batch();
    expect(GraphBatchCommitter::compute_touched_ranks(base, 3) ==
               std::vector<int>({0, 1, 2}),
           "A: compute_touched_ranks covers all three ranks");
    expect(GraphBatchCommitter::compute_touched_ranks(base, 2) ==
               std::vector<int>({0, 1}),
           "A: compute_touched_ranks filters out-of-range ranks");
    // C1: non-object entries can no longer reach this pure helper (the
    // parse layer rejects them); the rank filter above keeps the tolerance
    // contract for out-of-range typed values.
}

// ----------------------------------------------------------- Part B ------
// Every illegal batch must be rejected with the state byte-identical to the
// pre-validate snapshot (zero nodes / zero watches / zero ledger actions).
void expect_reject(Fixture& f, const StateDelta& delta, GraphBatch batch,
                   const char* what, const char* expected_error = nullptr) {
    const Snapshot before = snapshot_of(f);
    const auto err = f.committer.validate(delta, batch);
    expect(err.has_value(), what);
    if (!err.has_value()) {
        std::fprintf(stderr,
                     "[graph_batch_committer_test] unexpected validation "
                     "PASS: %s\n", what);
    }
    if (err.has_value() && expected_error != nullptr) {
        expect(*err == expected_error,
               "B: validation reports the exact fail-closed reason");
    }
    expect(snapshot_of(f) == before, "B: validate() left zero side effects");
    if (!(snapshot_of(f) == before)) {
        std::fprintf(stderr,
                     "[graph_batch_committer_test] state mutated by %s\n",
                     what);
    }
}

void expect_direct_commit_reject(Fixture& f, const StateDelta& delta,
                                 const GraphBatch& batch, const char* what) {
    const Snapshot before = snapshot_of(f);
    bool threw = false;
    try {
        f.committer.commit(delta, batch);
    } catch (const std::runtime_error&) {
        threw = true;
    }
    expect(threw, what);
    expect(snapshot_of(f) == before,
           "B: validate-off direct commit rejection has zero side effects");
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
        b.nodes[0].node.rank = 99;
        expect_reject(f, delta, b, "B: node rank out of range");
    }
    {
        GraphBatch b = base;
        b.nodes[1].json_id = 0;  // duplicate id 0 on rank 0
        expect_reject(f, delta, b, "B: duplicate node id within a rank");
    }
    {
        GraphBatch b = base;
        b.nodes[0].node.node_type = 9;
        expect_reject(f, delta, b, "B: node type out of range");
    }
    {
        GraphBatch b = base;
        b.nodes[0].node.stage = "chat";
        expect_reject(f, delta, b, "B: node stage not prefill/decode");
    }
    {
        GraphBatch b = base;
        b.nodes[5].node.generation = 1;  // prefill node with decode generation
        expect_reject(f, delta, b, "B: node generation != stage");
    }
    {
        GraphBatch b = base;
        b.nodes[5].node.request_id = "";
        expect_reject(f, delta, b, "B: node with empty request_id");
    }
    {
        GraphBatch b = base;
        b.parent_edges[0].from_json = 1;  // from == to == 1
        expect_reject(f, delta, b, "B: self-loop parent edge");
    }
    {
        GraphBatch b = base;
        b.parent_edges[0].from_json = 999;
        expect_reject(f, delta, b, "B: unresolved parent endpoint");
    }
    {
        GraphBatch b = base;
        b.parent_edges[0].to_json = 999;
        expect_reject(f, delta, b, "B: child endpoint not in this batch");
    }
    {
        GraphBatch b = base;
        b.parent_edges.push_back(ParsedEdge{1, 2, 0});  // 0->1->2->0
        expect_reject(f, delta, b, "B: in-batch parent edge cycle");
    }
    {
        GraphBatch b = base;
        b.watches[0].members[0].json_id = 99;  // rank 0's member
        expect_reject(f, delta, b, "B: watch member not a node of the batch");
    }
    {
        GraphBatch b = base;
        b.watches[0].generation = 1;  // prefill watch with decode gen
        expect_reject(f, delta, b, "B: watch generation != stage");
    }
    {
        GraphBatch b = base;
        b.watches.push_back(typed_watch(
            "r2", "prefill", {ParsedWatchMember{0, 1}},
            {NodeTerminalStatus::Success}));  // r2 never arrived
        expect_reject(f, delta, b,
                      "B: prefill watch for a request not in-flight");
    }
    {
        GraphBatch b = base;
        b.watches.push_back(typed_watch(
            "r1", "prefill", {ParsedWatchMember{0, 1}},
            {NodeTerminalStatus::Success}));  // duplicate identity
        expect_reject(f, delta, b,
                      "B: duplicate watch identity in the batch");
    }
    {
        GraphBatch b = base;
        b.nodes[3].node.comm.tag = 8;  // (0,1,7) send-only, (0,1,8) recv-only
        expect_reject(f, delta, b,
                      "B: send/recv pair incomplete within the batch");
    }
    {
        GraphBatch b = base;
        b.nodes[6].node.name = "r2_tp0_barrier_x";  // split tp0 group
        expect_reject(f, delta, b,
                      "B: split collective group fails closed");
    }
    {
        GraphBatch b = base;
        b.nodes[4].node.coll.bytes = 0;
        b.nodes[6].node.coll.bytes = 0;
        expect_reject(f, delta, b, "B: zero-byte collective rejected",
                      "node[4] collective bytes must be positive");
        Fixture direct;
        expect_direct_commit_reject(
            direct, delta, b,
            "B: validate-off direct commit rejects zero-byte collective");
    }
    {
        GraphBatch b = base;
        b.assignments[0].request_id.clear();
        expect_reject(f, delta, b, "B: assignment without request_id");
    }
    {
        GraphBatch b = base;
        b.kv_actions[0].trigger_request_id.clear();
        expect_reject(f, delta, b,
                      "B: kv action missing event_type/trigger_request_id");
    }
    {
        GraphBatch b = base;
        b.future_alarms[0].arrival_world_ns = 50;  // past the tick
        expect_reject(f, delta, b, "B: past future alarm");
    }
    {
        GraphBatch b = base;
        b.future_alarms.push_back(b.future_alarms[0]);  // r2 twice
        expect_reject(f, delta, b, "B: duplicate alarm request_id");
    }
    {
        GraphBatch b = base;
        b.future_alarms[0].envelope.request_id = "r1";  // in-flight
        expect_reject(f, delta, b, "B: alarm for an in-flight request");
    }
    {
        GraphBatch b = base;
        b.touched_ranks = std::vector<int>({0, 1});
        expect_reject(f, delta, b, "B: touched_ranks != node rank set");
    }
    {
        GraphBatch b = base;
        b.touched_ranks = std::vector<int>({2, 0, 1});
        expect_reject(f, delta, b, "B: touched_ranks not sorted unique");
    }
    // decode-eligibility: the same graph as decode generation-1 watches,
    // but the delta only carries the ARRIVAL (prefill never drained).
    {
        GraphBatch b = base;
        for (auto& node : b.nodes) {
            node.node.stage = "decode";
            node.node.generation = 1;
        }
        b.watches[0].stage = "decode";
        b.watches[0].generation = 1;
        expect_reject(f, delta, b,
                      "B: decode watch whose prefill has not drained");
    }
}

// ----------------------------------------------------------- Part C ------
// The online full-validation path must not hand a mutable validation result
// back to its caller.  validate_and_commit() keeps Phase A and Phase B in one
// call, returns a rich validation error with zero writes on failure, and
// commits exactly once on success.
void test_validate_and_commit_atomic() {
    {
        Fixture f;
        const StateDelta delta = baseline_delta();
        const GraphBatch batch = baseline_batch();
        const auto result = f.committer.validate_and_commit(delta, batch);
        expect(!result.error.has_value(),
               "C0: validate_and_commit accepts the legal baseline");
        expect(f.committer.counters().graph_batch_count == 1 &&
                   f.committer.counters().total_nodes == 7 &&
                   f.issue_calls == std::vector<int>({0, 1, 2}),
               "C0: successful validate_and_commit reaches Phase B exactly once");
        // A zero duration is legal on an exotic coarse steady clock; the
        // by-value result removes any output-pointer aliasing window.
        (void)result.validation_ns;
    }
    {
        Fixture f;
        const StateDelta delta = baseline_delta();
        GraphBatch invalid = baseline_batch();
        invalid.nodes[0].node.name.clear();
        const Snapshot before = snapshot_of(f);
        const auto result = f.committer.validate_and_commit(delta, invalid);
        expect(result.error.has_value(),
               "C0: validate_and_commit returns the full validation error");
        expect(snapshot_of(f) == before,
               "C0: failed validate_and_commit enters no Phase-B state");
        (void)result.validation_ns;
    }
}

// The per-rank stamp avoids O(ranks) clearing on the hot path.  Force the
// otherwise unreachable wrap boundary, reject one malformed stream, then
// prove the next batch recovers without allocating either touched-rank vector.
void test_json_id_stamp_wrap_and_recovery() {
    Fixture f;
    const StateDelta delta = baseline_delta();
    const GraphBatch base = baseline_batch();
    const size_t commit_capacity =
        GraphBatchCommitterTestAccess::commit_touched_capacity(f.committer);
    const size_t discovered_capacity =
        GraphBatchCommitterTestAccess::discovered_capacity(f.committer);

    GraphBatchCommitterTestAccess::seed_json_id_stream_stamp(
        f.committer, uint64_t(-1));
    GraphBatch invalid = base;
    invalid.nodes[1].json_id = 2;  // rank 0 expects id 1 after its first node
    const Snapshot before_rejection = snapshot_of(f);
    const auto rejection = f.committer.validate(delta, invalid);
    expect(rejection.has_value(),
           "C1: wrapped json-id stamp still rejects a broken stream");
    expect(snapshot_of(f) == before_rejection,
           "C1: wrapped-stamp rejected validation leaves state unchanged");

    const auto result = f.committer.validate_and_commit(delta, base);
    expect(!result.error.has_value(),
           "C1: json-id preflight recovers after wrapped-stamp rejection");
    expect(f.committer.counters().graph_batch_count == 1 &&
               f.issue_calls == std::vector<int>({0, 1, 2}),
           "C1: recovered validation commits exactly once on sorted ranks");
    expect(GraphBatchCommitterTestAccess::commit_touched_capacity(f.committer) ==
               commit_capacity &&
               GraphBatchCommitterTestAccess::discovered_capacity(f.committer) ==
                   discovered_capacity,
           "C1: json-id recovery reuses fixed touched-rank scratch capacity");
}

void test_positive_commit(Fixture& f) {
    const StateDelta delta = baseline_delta();
    const GraphBatch base = baseline_batch();

    // Pre-phase-5 fixture tolerance: a batch without touched_ranks still
    // validates (absent field != declared empty array).
    GraphBatch no_touched = base;
    no_touched.has_touched_ranks = false;
    no_touched.touched_ranks.clear();
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

    // Exact per-rank affine translation. NodeStore ids are PER-RANK (each
    // store starts at 1 here), while the metadata itself stays O(ranks).
    expect(f.committer.rank_affine_count() == 3,
           "C: affine metadata has one record per rank");
    expect(resolved_store_id(f.committer, 0, 0) == 1 &&
               resolved_store_id(f.committer, 0, 1) == 2,
           "C: rank 0 store ids 1, 2");
    expect(resolved_store_id(f.committer, 1, 0) == 1 &&
               resolved_store_id(f.committer, 1, 1) == 2 &&
               resolved_store_id(f.committer, 1, 2) == 3,
           "C: rank 1 store ids 1, 2, 3 (per-rank id space)");
    expect(resolved_store_id(f.committer, 2, 0) == 1 &&
               resolved_store_id(f.committer, 2, 1) == 2,
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
        b.future_alarms[0].envelope.request_id = "r1";  // in-flight NOW
        expect_reject(f, delta, b,
                      "D: alarm for a now in-flight request rejected");
    }
    {
        GraphBatch b = base;
        b.nodes[1].json_id = 0;  // in-batch duplicate id on rank 0
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

    GraphBatch b = parse_batch(empty_response(1));

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

    nlohmann::json resp = empty_response(2);
    resp["nodes"] = nlohmann::json::array({
        compute_node(0, 2, "r3", "prefill", "r3_comp"),
    });
    resp["parent_edges"] = nlohmann::json::array({
        data_edge(0, 1, 2),
    });
    resp["watches"] = nlohmann::json::array({
        {{"request_id", "r3"}, {"stage", "prefill"}, {"generation", 0},
         {"members", {{"0", 2}}},
         {"statuses", nlohmann::json::array({"Success"})}},
    });
    resp["touched_ranks"] = nlohmann::json::array({0});
    GraphBatch b = parse_batch(resp);

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
    expect(resolved_store_id(f.committer, 0, 2) == 3,
           "F: json id 2 of batch 2 -> store id 3 on rank 0");
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
// Fixture-shape compatibility (same-tick milestone fixture): metadata and
// compute nodes carrying the comm/coll DEFAULTS validate (the comm
// src/dst/tag range checks are scoped to comm-typed nodes 5/6). Pre-C1 the
// milestone service emitted an EMPTY comm {}/coll {}; the C1 parse-layer
// key-set rule requires the four comm keys, and the verify service was
// updated to emit the defaults with it (same-tick milestone fixture
// service, 2026-08-29).
void test_default_comm_compute_batch(Fixture& f) {
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

    nlohmann::json resp = empty_response(3);
    resp["nodes"] = nlohmann::json::array({
        {{"id", 3}, {"rank", 0}, {"type", 1}, {"name", "stm_prefill_0"},
         {"request_id", "r9"}, {"stage", "prefill"}, {"generation", 0},
         {"is_cpu_op", false}, {"is_timer_op", false},
         {"inputs_values", ""},
         {"compute", {{"num_ops", 0}, {"tensor_size", 0},
                      {"runtime_ns", 0}}},
         {"comm", comm_defaults()},
         {"coll", coll_defaults()}},
        {{"id", 4}, {"rank", 0}, {"type", 4}, {"name", "stm_prefill_1"},
         {"request_id", "r9"}, {"stage", "prefill"}, {"generation", 0},
         {"is_cpu_op", false}, {"is_timer_op", false},
         {"inputs_values", ""},
         {"compute", {{"num_ops", 1}, {"tensor_size", 1},
                      {"runtime_ns", 1}}},
         {"comm", comm_defaults()},
         {"coll", coll_defaults()}},
    });
    resp["parent_edges"] = nlohmann::json::array({data_edge(0, 3, 4)});
    resp["watches"] = nlohmann::json::array({
        {{"request_id", "r9"}, {"stage", "prefill"}, {"generation", 0},
         {"members", {{"0", 4}}},
         {"statuses", nlohmann::json::array({"Skipped"})}},
    });
    resp["touched_ranks"] = nlohmann::json::array({0});
    GraphBatch b = parse_batch(resp);

    expect(!f.committer.validate(delta, b).has_value(),
           "G: metadata/compute nodes with default comm/coll validate");
    f.committer.commit(delta, b);
    expect(f.committer.counters().graph_batch_count == 4,
           "G: graph_batch_count == 4");
}

// ----------------------------------------------------------- Part H ------
// Online JSON "hbm_charge" parsing nail (wscllm; 低-2 online key
// unification, typed C1 form): the snake_case comm-section key parses into
// comm.hbm_charge via the production parse; an absent key defaults to
// true; the legacy kebab spelling comm["hbm-charge"] is now REJECTED by
// the parse-layer comm key set (fail-closed) -- a reverse nail pinning
// the new spelling.
void test_hbm_charge_key_parsing(Fixture& f) {
    StateDelta delta;
    delta.delivery_sequence = 7;
    delta.delivery_epoch = 7;
    delta.tick = 700;
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
    nlohmann::json send = comm_node(0, 14, 5, 0, 1, 7);
    send["request_id"] = "r10";
    send["comm"]["hbm_charge"] = false;
    nlohmann::json recv = comm_node(1, 11, 6, 0, 1, 7);
    recv["request_id"] = "r10";  // key absent on the recv side

    nlohmann::json resp = empty_response(7);
    resp["nodes"] = nlohmann::json::array({
        compute_node(0, 13, "r10", "prefill", "r10_comp"),  // key absent
        send,
        recv,
    });
    resp["watches"] = nlohmann::json::array({
        prefill_watch("r10", {{"0", 14}}),
    });
    resp["touched_ranks"] = nlohmann::json::array({0, 1});
    GraphBatch b = parse_batch(resp);

    expect(!f.committer.validate(delta, b).has_value(),
           "H: hbm_charge batch validates clean");
    if (const auto err = f.committer.validate(delta, b)) {
        std::fprintf(stderr,
                     "[graph_batch_committer_test] H validate error: %s\n",
                     err->c_str());
    }
    f.committer.commit(delta, b);

    const auto comp_view =
        f.sources[0]->lookup(resolved_store_id(f.committer, 0, 13));
    expect(comp_view.has_value() && comp_view->comm.hbm_charge,
           "H: absent comm.hbm_charge defaults to true");
    const auto send_view =
        f.sources[0]->lookup(resolved_store_id(f.committer, 0, 14));
    expect(send_view.has_value() && !send_view->comm.hbm_charge,
           "H: comm.hbm_charge=false parsed into the NodeView");

    // The legacy kebab spelling is a parse-layer unknown key now: the
    // same response with only that spelling must fail closed.
    nlohmann::json kebab_resp = resp;
    nlohmann::json kebab_recv = recv;
    kebab_recv["comm"]["hbm-charge"] = false;  // legacy kebab spelling
    kebab_resp["nodes"][2] = kebab_recv;
    bool threw = false;
    try {
        (void)parse_batch(kebab_resp);
    } catch (const ParseError&) {
        threw = true;
    }
    expect(threw,
           "H: legacy comm hbm-charge spelling rejected by the parse layer");
}

// ------------------------------------- 拼 batch §3.1 前置验证, 2026-08-22 ----
// Frozen batch-train schema (拼 batch 改造):
//   - shared aggregate body node: request_id is the batch-namespace string
//     ("batch_train_0_1" -- NOT a real request), stage "decode", gen 1,
//     COMP node, any positive compute;
//   - exit marker: one per exiting member per rank, real member id,
//     stage "decode", gen 1, COMP, num_ops = tensor_size = 1; drain marker
//     is the same shape with stage "prefill", gen 0;
//   - one end barrier per train: COMM_COLL_NODE, coll.comm_type =
//     ALL_REDUCE (0), bytes = iteration count, pg_name shared with the
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
        {"coll", {{"comm_type", 0}, {"bytes", iterations}, {"priority", 0},
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
                       uint64_t iterations,
                       const std::vector<uint64_t>& first_json_ids) {
    nlohmann::json resp = empty_response(batch_id);
    nlohmann::json nodes = nlohmann::json::array();
    nlohmann::json edges = nlohmann::json::array();
    nlohmann::json watches = nlohmann::json::array();
    const uint64_t barrier_id = members.size() + 1;
    for (int rank = 0; rank < 3; ++rank) {
        const uint64_t first_json_id = first_json_ids.at(rank);
        nodes.push_back(train_body_node(rank, first_json_id, train_ns));
        for (size_t m = 0; m < members.size(); ++m) {
            nodes.push_back(train_marker_node(rank, first_json_id + m + 1,
                                              members[m].request_id,
                                              members[m].stage));
            edges.push_back(
                data_edge(rank, first_json_id, first_json_id + m + 1));
            edges.push_back(data_edge(rank, first_json_id + m + 1,
                                               first_json_id + barrier_id));
        }
        nodes.push_back(
            train_barrier_node(rank, first_json_id + barrier_id, train_ns,
                               iterations));
    }
    for (size_t m = 0; m < members.size(); ++m) {
        nlohmann::json member_ids;
        for (int rank = 0; rank < 3; ++rank) {
            member_ids[std::to_string(rank)] =
                first_json_ids.at(rank) + m + 1;
        }
        watches.push_back(
            {{"request_id", members[m].request_id},
             {"stage", members[m].stage},
             {"generation", members[m].stage == "decode" ? 1 : 0},
             {"members", std::move(member_ids)},
             {"statuses", nlohmann::json::array({"Success", "Skipped"})}});
    }
    resp["nodes"] = std::move(nodes);
    resp["parent_edges"] = std::move(edges);
    resp["watches"] = std::move(watches);
    resp["touched_ranks"] = nlohmann::json::array({0, 1, 2});
    return parse_batch(resp);
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
            CompletionKey{rank, store_id, meta->generation}, status);
    }
}

// ----------------------------------------------------------- Part H ------
// The two-request decode train: epoch 1 lands the arrivals + req_A's
// prefill drain (a zero-node accounting batch), epoch 2 commits the train
// (req_B's prefill drain is a delta fact of the SAME epoch -- delta facts
// first). The negative probe proves the schema boundary that still guards:
// without req_B's drain fact, its decode watch is rejected (eligibility).
void test_train_batch_two_members(Fixture& f) {
    // Epoch h1: arrivals + req_A drain, zero-node batch (Part E style).
    StateDelta h1;
    h1.delivery_sequence = 4;
    h1.delivery_epoch = 4;
    h1.tick = 400;
    h1.events.push_back(arrival_event("req_A"));
    h1.events.push_back(arrival_event("req_B"));
    h1.events.push_back(drain_event("req_A"));
    GraphBatch setup = parse_batch(empty_response(4));
    expect(!f.committer.validate(h1, setup).has_value(),
           "I: epoch-1 zero-node setup batch validates clean");
    f.committer.commit(h1, setup);
    expect(f.committer.in_flight_requests() ==
               std::set<std::string>({"r3", "r9", "req_A", "req_B"}),
           "I: both train members in-flight after epoch 1");
    expect(f.committer.prefill_drained_requests() ==
               std::set<std::string>({"req_A"}),
           "I: req_A prefill-drained after epoch 1");
    expect(f.committer.counters().graph_batch_count == 5,
           "I: graph_batch_count == 5 after the setup epoch");

    // The train: 12 nodes (3 ranks x [body, exit marker req_A, exit marker
    // req_B, end barrier]), 2 member decode watches.
    const std::vector<uint64_t> train_first_json_ids{5, 3, 2};
    const GraphBatch train = train_batch(
        5, "batch_train_0_1", {{"req_A", "decode"}, {"req_B", "decode"}},
        8, train_first_json_ids);

    // Negative probe: the same train validated against an epoch whose delta
    // carries NO req_B prefill drain -- the member decode watch eligibility
    // rule must block it with zero side effects (this is the one rule the
    // batch schema genuinely leans on; it is NOT relaxed by 拼 batch).
    {
        StateDelta no_drain_b;
        no_drain_b.delivery_sequence = 5;
        no_drain_b.delivery_epoch = 5;
        no_drain_b.tick = 450;
        const Snapshot before = snapshot_of(f);
        const auto err = f.committer.validate(no_drain_b, train);
        expect(err.has_value(),
               "I: decode watch of a never-drained member is blocked");
        if (err.has_value()) {
            expect(err->find("whose prefill has not drained") !=
                       std::string::npos,
                   "I: the block is the decode-watch eligibility rule");
        }
        expect(snapshot_of(f) == before,
               "I: blocked train left zero side effects");
    }

    // Epoch h2: req_B's prefill drain is a delta fact of the train epoch
    // itself (delta facts first -- the realistic first-decode-train tick).
    StateDelta h2;
    h2.delivery_sequence = 5;
    h2.delivery_epoch = 5;
    h2.tick = 450;
    h2.events.push_back(drain_event("req_B"));
    expect(!f.committer.validate(h2, train).has_value(),
           "I: two-request train batch validates clean (zero relaxation)");
    f.committer.commit(h2, train);

    const auto& c = f.committer.counters();
    expect(c.graph_batch_count == 6, "I: graph_batch_count == 6");
    expect(c.single_node_bridge_count == 1,
           "I: single_node_bridge_count unchanged (12-node train)");
    expect(c.total_nodes == 22, "I: total_nodes == 22 (10 + 12)");
    expect(c.max_nodes_per_batch == 12, "I: max_nodes_per_batch == 12");
    expect(c.total_watches == 5, "I: total_watches == 5 (3 + 2 members)");
    expect(f.committer.in_flight_requests() ==
               std::set<std::string>({"r3", "r9", "req_A", "req_B"}),
           "I: train commit adds no arrivals");
    expect(f.committer.prefill_drained_requests() ==
               std::set<std::string>({"req_A", "req_B"}),
           "I: req_B drained by the train epoch's own delta fact");
    expect(f.registry.size() == 5,
           "I: registry holds the 2 member watches (+3 legacy)");
    expect(f.issue_calls ==
               std::vector<int>({0, 1, 2, 0, 0, 0, 1, 2}),
           "I: issue pass covered exactly the touched ranks {0, 1, 2}");
    // Per-rank pending nodes after Part G: {5, 3, 2}; the train adds 4.
    const size_t pending_after_g[3] = {5, 3, 2};
    for (int rank = 0; rank < 3; ++rank) {
        expect(f.sources[rank]->store().pending_count() ==
                   pending_after_g[rank] + 4,
               "I: per-rank store gained the 4 train nodes");
    }

    // Store-id translation: rank-local contiguous train ids resolve through
    // the persistent per-rank affine mapping.
    std::vector<uint64_t> body_ids;
    std::vector<uint64_t> barrier_ids;
    std::vector<std::vector<uint64_t>> marker_ids(2);  // [member][rank]
    for (int rank = 0; rank < 3; ++rank) {
        const uint64_t first_json_id = train_first_json_ids.at(rank);
        body_ids.push_back(resolved_store_id(f.committer, rank, first_json_id));
        marker_ids[0].push_back(
            resolved_store_id(f.committer, rank, first_json_id + 1));
        marker_ids[1].push_back(
            resolved_store_id(f.committer, rank, first_json_id + 2));
        barrier_ids.push_back(
            resolved_store_id(f.committer, rank, first_json_id + 3));
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

// ----------------------------------------------------------- Part I ------
// Variant: ONE train folding a prefill chunk of req_C (drain marker,
// prefill watch, gen 0) and a decode token of req_D (exit marker, decode
// watch, gen 1) -- the mixed train. Both watches fire.
void test_train_batch_drain_and_exit_markers(Fixture& f) {
    StateDelta i1;
    i1.delivery_sequence = 6;
    i1.delivery_epoch = 6;
    i1.tick = 500;
    i1.events.push_back(arrival_event("req_C"));
    i1.events.push_back(arrival_event("req_D"));
    i1.events.push_back(drain_event("req_D"));

    const std::vector<uint64_t> mixed_first_json_ids{9, 7, 6};
    const GraphBatch train = train_batch(
        6, "batch_train_2_3", {{"req_C", "prefill"}, {"req_D", "decode"}},
        4, mixed_first_json_ids);
    expect(!f.committer.validate(i1, train).has_value(),
           "J: mixed drain+exit train validates clean (zero relaxation)");
    f.committer.commit(i1, train);

    const auto& c = f.committer.counters();
    expect(c.graph_batch_count == 7, "J: graph_batch_count == 7");
    expect(c.total_nodes == 34, "J: total_nodes == 34 (22 + 12)");
    expect(c.total_watches == 7, "J: total_watches == 7 (5 + 2)");
    expect(f.committer.in_flight_requests() ==
               std::set<std::string>(
                   {"r3", "r9", "req_A", "req_B", "req_C", "req_D"}),
           "J: mixed-train members in-flight");
    expect(f.committer.prefill_drained_requests() ==
               std::set<std::string>({"req_A", "req_B", "req_D"}),
           "J: req_D drained by the train epoch's own delta fact");
    expect(f.registry.size() == 7, "J: registry holds both member watches");

    for (int rank = 0; rank < 3; ++rank) {
        const uint64_t first_json_id = mixed_first_json_ids.at(rank);
        drive_terminal(f, rank,
                       resolved_store_id(f.committer, rank, first_json_id),
                       NodeTerminalStatus::Success);  // body (namespace)
        drive_terminal(f, rank,
                       resolved_store_id(f.committer, rank, first_json_id + 1),
                       NodeTerminalStatus::Success);  // req_C drain marker
        drive_terminal(f, rank,
                       resolved_store_id(f.committer, rank, first_json_id + 2),
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
}

// ----------------------------------------------------------- Part P ------
// C1 (2026-08-29) parse/atomicity: the single structural parse is a pure
// local construction, a malformed response aborts BEFORE any committer
// state exists to mutate, and post-parse typed mutations of the CommitArg
// batch remain fail-closed through the committer's own domain rules with
// zero side effects (two-phase atomicity under the typed batch).
void test_parse_failure_atomicity(Fixture& f) {
    const StateDelta delta = baseline_delta();

    // (1) A malformed response never becomes a batch: ParseError, and the
    //     committer state is byte-identical around the failed exchange.
    {
        const Snapshot before = snapshot_of(f);
        nlohmann::json resp = baseline_response();
        resp["nodes"][2].erase("stage");  // missing required key
        bool threw = false;
        try {
            (void)parse_graph_batch(resp, 3);
        } catch (const ParseError&) {
            threw = true;
        }
        expect(threw, "P: malformed response throws ParseError");
        expect(snapshot_of(f) == before,
               "P: parse failures leave the committer state untouched");
    }
    // (2) Post-parse typed corruption is still rejected fail-closed (the
    //     CommitArg carries the typed batch across the deferred-event
    //     boundary; the committer's own domain rules are the backstop).
    {
        GraphBatch b = baseline_batch();
        b.nodes[0].node.rank = 99;
        const Snapshot before = snapshot_of(f);
        expect(f.committer.validate(delta, b).has_value(),
               "P: post-parse rank corruption rejected by validate");
        expect(snapshot_of(f) == before,
               "P: rejection of the corrupted batch has zero side effects");
    }
    // (3) A corrupted watch member is blocked by the MANDATORY preflight
    //     (member-not-a-batch-node) BEFORE any Phase-B assembly can run --
    //     the direct validate-off commit path stays atomic.
    {
        GraphBatch b = baseline_batch();
        b.watches[0].members[0].json_id = 7777;
        const Snapshot before = snapshot_of(f);
        bool threw = false;
        try {
            f.committer.commit(delta, b);
        } catch (const std::runtime_error&) {
            threw = true;
        }
        expect(threw,
               "P: corrupted watch member blocked before Phase B");
        expect(snapshot_of(f) == before,
               "P: blocked assembly left zero side effects");
    }
}


// ------------------------------------------------------- GC regressions ----
// Build this fixture in the only order that actually arms NodeStore GC:
// sources first, then the committer Context with node_gc=true.
struct GcFixture {
    RequestIngress ingress;
    WatchRegistry registry;
    std::vector<std::shared_ptr<NodeStoreGraphSource>> sources;
    std::unique_ptr<GraphBatchCommitter> committer;

    explicit GcFixture(size_t preseed_nodes = 0) : ingress(4096) {
        sources.push_back(std::make_shared<NodeStoreGraphSource>());
        OnlineNode preseed;
        preseed.kind = NodeKind::Compute;
        preseed.name = "gc-preseed";
        for (size_t i = 0; i < preseed_nodes; ++i) {
            sources[0]->store().add_node(preseed);
        }
        GraphBatchCommitter::Context ctx;
        ctx.num_ranks = 1;
        ctx.graph_sources = &sources;
        ctx.watch_registry = &registry;
        ctx.ingress = &ingress;
        ctx.issue_rank = [](int) {};
        ctx.node_gc = true;
        committer = std::make_unique<GraphBatchCommitter>(std::move(ctx));
    }
};

StateDelta gc_delta(uint64_t sequence) {
    StateDelta delta;
    delta.delivery_sequence = sequence;
    delta.delivery_epoch = sequence;
    delta.tick = sequence + 1;
    return delta;
}

GraphBatch gc_compute_batch(uint64_t sequence, uint64_t first_json_id,
                            uint64_t count, const std::string& request_id,
                            uint64_t parent_json_id = uint64_t(-1)) {
    nlohmann::json resp = empty_response(sequence);
    nlohmann::json nodes = nlohmann::json::array();
    for (uint64_t offset = 0; offset < count; ++offset) {
        nodes.push_back(compute_node(
            0, first_json_id + offset, request_id, "prefill", "gc_compute"));
    }
    resp["nodes"] = std::move(nodes);
    if (count == 1 && parent_json_id != uint64_t(-1)) {
        resp["parent_edges"] = nlohmann::json::array(
            {data_edge(0, parent_json_id, first_json_id)});
    }
    resp["touched_ranks"] = nlohmann::json::array({0});
    return parse_batch(resp, 1);
}

GraphBatch gc_empty_batch(uint64_t sequence) {
    return parse_batch(empty_response(sequence), 1);
}

bool commit_gc_batch(GcFixture& fixture, const GraphBatch& batch,
                     const char* what) {
    StateDelta delta = gc_delta(batch.batch_id);
    std::set<std::string> arrivals;
    for (const auto& node : batch.nodes) {
        const std::string& request_id = node.node.request_id;
        if (!request_id.empty() &&
            fixture.committer->in_flight_requests().count(request_id) == 0) {
            arrivals.insert(request_id);
        }
    }
    for (const auto& request_id : arrivals) {
        DecisionEvent arrival;
        arrival.reason = DecisionReason::ARRIVAL;
        arrival.request_id = request_id;
        arrival.stage = "prefill";
        arrival.generation = 0;
        delta.events.push_back(std::move(arrival));
    }
    const auto error = fixture.committer->validate(delta, batch);
    expect(!error.has_value(), what);
    if (error.has_value()) {
        std::fprintf(stderr, "[graph_batch_committer_test] GC validate: %s\n",
                     error->c_str());
        return false;
    }
    fixture.committer->commit(delta, batch);
    return true;
}

void finish_gc_json_id(GcFixture& fixture, uint64_t json_id,
                       const char* what) {
    const auto store_id = fixture.committer->resolve_store_id(0, json_id);
    expect(store_id.has_value(), what);
    if (store_id.has_value()) {
        fixture.sources[0]->store().finish_node(*store_id);
    }
}

void test_gc_bounded_id_history_and_pruning() {
    GcFixture fixture;

    // A non-zero first id is represented exactly as [7,next), not as an
    // implicit prefix from zero.
    const GraphBatch first = gc_compute_batch(0, 7, 1, "gc-first");
    if (!commit_gc_batch(fixture, first, "K: first non-zero id validates")) {
        return;
    }
    const auto first_store_id = fixture.committer->resolve_store_id(0, 7);
    expect(first_store_id.has_value() && *first_store_id == 1,
           "K: first non-zero json id captures the actual store id");
    finish_gc_json_id(fixture, 7, "K: first id has a live mapping");
    fixture.committer->finalize_node_garbage();
    expect(first_store_id.has_value() &&
               fixture.sources[0]->store().erased(*first_store_id) &&
               fixture.committer->rank_affine_count() == 1,
           "K: finalized node is erased but its affine translation persists");

    // The producer contract is one contiguous per-rank stream. A gap must be
    // rejected even when expensive semantic validation is skipped, and the
    // direct commit preflight must leave all state untouched.
    const GraphBatch gap = gc_compute_batch(1, 9, 1, "gc-gap");
    const auto gap_error = fixture.committer->validate(gc_delta(1), gap);
    expect(gap_error.has_value(), "K: sparse id gap fails validation");
    const auto counters_before_gap = fixture.committer->counters();
    bool gap_commit_threw = false;
    try {
        fixture.committer->commit(gc_delta(1), gap);
    } catch (const std::runtime_error&) {
        gap_commit_threw = true;
    }
    expect(gap_commit_threw,
           "K: validate-off direct commit rejects sparse id gap");
    expect(first_store_id.has_value() &&
               fixture.committer->resolve_store_id(0, 7) == first_store_id &&
               fixture.committer->rank_affine_count() == 1 &&
               fixture.sources[0]->store().retained_count() == 0 &&
               fixture.committer->counters().graph_batch_count ==
                   counters_before_gap.graph_batch_count,
           "K: rejected direct commit has zero graph/counter side effects");

    // The next contiguous id may depend on collected id 7: the exact bounded
    // history recognizes the dead parent and commit turns its edge into a
    // no-op.
    const GraphBatch after_gc =
        gc_compute_batch(1, 8, 1, "gc-after", 7);
    if (!commit_gc_batch(fixture, after_gc,
                         "K: collected committed parent resolves")) {
        return;
    }
    const auto after_store_id = fixture.committer->resolve_store_id(0, 8);
    expect(after_store_id.has_value() &&
               fixture.sources[0]->store().resolve_free_nodes() ==
                   std::vector<uint64_t>({*after_store_id}),
           "K: edge from collected parent is non-blocking");
    finish_gc_json_id(fixture, 8, "K: child after collected parent maps");
    fixture.committer->finalize_node_garbage();

    const GraphBatch next = gc_compute_batch(2, 9, 1, "gc-next");
    if (!commit_gc_batch(fixture, next,
                         "K: next contiguous id validates")) {
        return;
    }
    finish_gc_json_id(fixture, 9, "K: next contiguous id maps");
    fixture.committer->finalize_node_garbage();
    const auto next_store_id = fixture.committer->resolve_store_id(0, 9);
    expect(first_store_id.has_value() && after_store_id.has_value() &&
               next_store_id.has_value() &&
               fixture.sources[0]->store().erased(*first_store_id) &&
               fixture.sources[0]->store().erased(*after_store_id) &&
               fixture.sources[0]->store().erased(*next_store_id) &&
               fixture.committer->rank_affine_count() == 1,
           "K: all collected ids remain exactly resolvable in O(ranks) state");

    const auto duplicate_seven =
        fixture.committer->validate(gc_delta(3),
                                    gc_compute_batch(3, 7, 1, "gc-dup-7"));
    expect(duplicate_seven.has_value(),
           "K: duplicate collected first id fails closed");
    const auto duplicate_eight =
        fixture.committer->validate(gc_delta(4),
                                    gc_compute_batch(4, 8, 1, "gc-dup-8"));
    expect(duplicate_eight.has_value(),
           "K: duplicate committed range id fails closed");
    const auto unknown_parent = fixture.committer->validate(
        gc_delta(5), gc_compute_batch(5, 10, 1, "gc-unknown", 6));
    expect(unknown_parent.has_value(),
           "K: never-committed low-id parent fails closed");
}

void test_gc_amortized_tracking_bound() {
    GcFixture fixture;
    constexpr uint64_t kLongId = 0;
    const uint64_t threshold =
        static_cast<uint64_t>(GraphBatchCommitter::kGcAmortizeThreshold);

    if (!commit_gc_batch(fixture,
                         gc_compute_batch(0, kLongId, 1, "gc-long"),
                         "L: long-lived root validates")) {
        return;
    }
    // Leave id 0 unfinished while later nodes are collected. The affine
    // metadata must stay one fixed rank record, not grow with either set.

    if (!commit_gc_batch(fixture,
                         gc_compute_batch(1, 1, threshold - 1, "gc-bulk-a"),
                         "L: threshold-minus-one batch validates")) {
        return;
    }
    for (uint64_t id = 1; id < threshold; ++id) {
        finish_gc_json_id(fixture, id, "L: bulk-a id has a live mapping");
    }
    if (!commit_gc_batch(fixture, gc_empty_batch(2),
                         "L: below-threshold tail validates")) {
        return;
    }
    expect(fixture.sources[0]->store().pending_gc_count() == threshold - 1,
           "L: 4095 candidates remain below the amortized threshold");

    if (!commit_gc_batch(fixture,
                         gc_compute_batch(3, threshold, 1, "gc-threshold"),
                         "L: threshold node validates")) {
        return;
    }
    finish_gc_json_id(fixture, threshold,
                      "L: threshold node has a live mapping");
    if (!commit_gc_batch(fixture, gc_empty_batch(4),
                         "L: threshold collection tail validates")) {
        return;
    }
    const auto root_store_id = fixture.committer->resolve_store_id(0, kLongId);
    const auto threshold_store_id =
        fixture.committer->resolve_store_id(0, threshold);
    expect(root_store_id.has_value() && threshold_store_id.has_value() &&
               !fixture.sources[0]->store().erased(*root_store_id) &&
               fixture.sources[0]->store().erased(*threshold_store_id) &&
               fixture.committer->rank_affine_count() == 1 &&
               fixture.sources[0]->store().retained_count() == 1,
           "L: collection keeps one live record and fixed affine metadata");

    const uint64_t second_first = threshold + 1;
    const uint64_t second_count = threshold + 1;
    if (!commit_gc_batch(
            fixture,
            gc_compute_batch(5, second_first, second_count, "gc-bulk-b"),
            "L: threshold-plus-one batch validates")) {
        return;
    }
    for (uint64_t id = second_first; id < second_first + second_count; ++id) {
        finish_gc_json_id(fixture, id, "L: bulk-b id has a live mapping");
    }
    if (!commit_gc_batch(fixture, gc_empty_batch(6),
                         "L: second collection tail validates")) {
        return;
    }
    const uint64_t second_last = second_first + second_count - 1;
    const auto second_last_store_id =
        fixture.committer->resolve_store_id(0, second_last);
    expect(second_last_store_id.has_value() &&
               fixture.sources[0]->store().erased(*second_last_store_id) &&
               fixture.committer->rank_affine_count() == 1,
           "L: later collected ids retain translation without per-node state");

    const uint64_t final_short = second_first + second_count;
    if (!commit_gc_batch(fixture,
                         gc_compute_batch(7, final_short, 1, "gc-final"),
                         "L: final short node validates")) {
        return;
    }
    finish_gc_json_id(fixture, final_short,
                      "L: final short node has a live mapping");
    fixture.committer->finalize_node_garbage();
    const auto final_store_id =
        fixture.committer->resolve_store_id(0, final_short);
    expect(root_store_id.has_value() && final_store_id.has_value() &&
               !fixture.sources[0]->store().erased(*root_store_id) &&
               fixture.sources[0]->store().erased(*final_store_id) &&
               fixture.committer->rank_affine_count() == 1 &&
               fixture.sources[0]->store().retained_count() == 1,
           "L: final drain retains only the live NodeStore record");
}

void test_preseeded_store_id_is_captured() {
    GcFixture fixture(2);
    expect(fixture.sources[0]->store().next_auto_id() == 3,
           "M: preseeded store advertises its real next automatic id");

    const GraphBatch first = gc_compute_batch(0, 73, 1, "gc-preseeded");
    if (!commit_gc_batch(fixture, first,
                         "M: arbitrary first json id validates after preseed")) {
        return;
    }
    const auto store_id = fixture.committer->resolve_store_id(0, 73);
    expect(store_id.has_value() && *store_id == 3 &&
               fixture.sources[0]->store().next_auto_id() == 4 &&
               !fixture.committer->resolve_store_id(0, 72).has_value() &&
               fixture.committer->rank_affine_count() == 1,
           "M: first affine mapping uses add_node's actual returned id");
}

void test_affine_drift_fails_closed() {
    GcFixture fixture;
    if (!commit_gc_batch(fixture, gc_compute_batch(0, 41, 1, "gc-drift"),
                         "N: drift fixture's first batch validates")) {
        return;
    }
    const auto first_store_id = fixture.committer->resolve_store_id(0, 41);
    expect(first_store_id.has_value(), "N: first committed id resolves");

    OnlineNode external;
    external.kind = NodeKind::Compute;
    external.name = "external-drift";
    const uint64_t external_store_id =
        fixture.sources[0]->store().add_node(external);
    expect(external_store_id == 2,
           "N: external insert advances the NodeStore automatic stream");

    StateDelta drift_delta = gc_delta(1);
    DecisionEvent arrival;
    arrival.reason = DecisionReason::ARRIVAL;
    arrival.request_id = "must-not-arrive";
    arrival.stage = "prefill";
    arrival.generation = 0;
    drift_delta.events.push_back(arrival);
    const GraphBatch next = gc_compute_batch(1, 42, 1, "gc-drift-next");

    const size_t pending_before = fixture.sources[0]->store().pending_count();
    const size_t retained_before = fixture.sources[0]->store().retained_count();
    const uint64_t next_auto_before =
        fixture.sources[0]->store().next_auto_id();
    const size_t affine_before = fixture.committer->rank_affine_count();
    const auto counters_before = fixture.committer->counters();
    const auto in_flight_before = fixture.committer->in_flight_requests();

    const auto validation_error = fixture.committer->validate(drift_delta, next);
    expect(validation_error.has_value() &&
               validation_error->find("drift") != std::string::npos,
           "N: validate fails closed on external NodeStore drift");
    expect(fixture.sources[0]->store().pending_count() == pending_before &&
               fixture.sources[0]->store().retained_count() == retained_before &&
               fixture.sources[0]->store().next_auto_id() == next_auto_before &&
               fixture.committer->rank_affine_count() == affine_before &&
               fixture.committer->resolve_store_id(0, 41) == first_store_id &&
               !fixture.committer->resolve_store_id(0, 42).has_value() &&
               fixture.committer->in_flight_requests() == in_flight_before &&
               same_counters(fixture.committer->counters(), counters_before),
           "N: validate drift failure leaves delta/store/affine/counters intact");

    bool threw = false;
    try {
        fixture.committer->commit(drift_delta, next);
    } catch (const std::runtime_error&) {
        threw = true;
    }
    expect(threw, "N: direct commit fails closed on external NodeStore drift");
    expect(fixture.sources[0]->store().pending_count() == pending_before &&
               fixture.sources[0]->store().retained_count() == retained_before &&
               fixture.sources[0]->store().next_auto_id() == next_auto_before &&
               fixture.committer->rank_affine_count() == affine_before &&
               fixture.committer->resolve_store_id(0, 41) == first_store_id &&
               !fixture.committer->resolve_store_id(0, 42).has_value() &&
               fixture.committer->in_flight_requests() == in_flight_before &&
               same_counters(fixture.committer->counters(), counters_before),
           "N: direct drift failure leaves delta/store/affine/counters intact");
}

void test_affine_metadata_under_million_node_pressure() {
    GcFixture fixture;
    constexpr uint64_t kTotalNodes = 1000000;
    constexpr uint64_t kBatchNodes =
        static_cast<uint64_t>(GraphBatchCommitter::kGcAmortizeThreshold);
    constexpr uint64_t kFirstJsonId = 1000000;
    uint64_t next_json_id = kFirstJsonId;
    uint64_t sequence = 0;
    while (next_json_id < kFirstJsonId + kTotalNodes) {
        const uint64_t remaining =
            kFirstJsonId + kTotalNodes - next_json_id;
        const uint64_t count = remaining < kBatchNodes ? remaining : kBatchNodes;
        if (!commit_gc_batch(
                fixture,
                gc_compute_batch(sequence, next_json_id, count, "gc-pressure"),
                "O: million-node pressure batch validates")) {
            return;
        }
        for (uint64_t offset = 0; offset < count; ++offset) {
            const auto store_id =
                fixture.committer->resolve_store_id(0, next_json_id + offset);
            if (!store_id.has_value()) {
                expect(false, "O: every committed pressure id resolves");
                return;
            }
            fixture.sources[0]->store().finish_node(*store_id);
        }
        next_json_id += count;
        ++sequence;
    }
    fixture.committer->finalize_node_garbage();

    const auto first_store_id =
        fixture.committer->resolve_store_id(0, kFirstJsonId);
    const auto last_store_id = fixture.committer->resolve_store_id(
        0, kFirstJsonId + kTotalNodes - 1);
    expect(first_store_id.has_value() && last_store_id.has_value() &&
               *first_store_id == 1 && *last_store_id == kTotalNodes &&
               fixture.sources[0]->store().erased(*first_store_id) &&
               fixture.sources[0]->store().erased(*last_store_id) &&
               fixture.sources[0]->store().retained_count() == 0 &&
               fixture.committer->rank_affine_count() == 1,
           "O: one million committed ids retain O(ranks) affine metadata");
}

}  // namespace
int main() {
    Fixture f;
    test_static_checks(f);
    test_negative_cases(f);
    test_validate_and_commit_atomic();
    test_json_id_stamp_wrap_and_recovery();
    test_positive_commit(f);
    test_post_commit_negatives(f);
    test_zero_node_batch(f);
    test_single_node_batch(f);
    test_default_comm_compute_batch(f);
    test_parse_failure_atomicity(f);
    test_train_batch_two_members(f);
    test_train_batch_drain_and_exit_markers(f);
    test_hbm_charge_key_parsing(f);
    test_gc_bounded_id_history_and_pruning();
    test_gc_amortized_tracking_bound();
    test_preseeded_store_id_is_captured();
    test_affine_drift_fails_closed();
    test_affine_metadata_under_million_node_pressure();
    if (g_ok) {
        std::printf("[graph_batch_committer_test] ALL PASS\n");
        return 0;
    }
    std::fprintf(stderr, "[graph_batch_committer_test] FAILURES\n");
    return 1;
}
