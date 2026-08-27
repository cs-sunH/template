/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

GraphBatchCommitter -- execution-driven mechanism layer (wscllm phase 5).
Atomic GraphBatch commit (方案 §8.1/§8.2).

The tick-end commit is a two-phase protocol:

  Phase A -- validate(delta, batch): the FULL pre-commit checklist (方案 §8.1
  Phase A). PURE: no NodeStore node, no watch registration, no ingress
  alarm, no issue pass, no counter and no tracking-set mutation happens
  while validating, and NONE happens on a validation failure. The official
  path (ed_commit_cb in main_online.cc) treats a validation failure as a
  fail-closed abort; the phase-5 fixture (tests/graph_batch_committer_test.cc)
  asserts the zero side-effect property directly.

  Phase B -- commit(delta, batch): applies the delta facts to the in-flight
  request tracking, then adds the batch's nodes (persistent (rank, json id)
  -> store id map, cross-batch), parent edges, watches, future arrival
  alarms, runs the per-rank issue pass over the batch's TOUCHED RANKS ONLY,
  and updates the counters. The caller (main_online) performs the
  ServiceCoordinator REQUEST_COMPLETE accounting, the completed requests'
  watch removal and the commit ack AFTER commit() returns.

Validation rules (every rule empirically verified against the real 20.csv
first-30s runs: strategy 3531 batches + replay 3491 batches, zero
violations):
  [epoch]   batch.batch_id == batch.source_delivery_sequence ==
            delta.delivery_sequence; batch.error empty.
  [node]    rank in [0, num_ranks); id >= 0, per-rank unique in batch; type
            in 1..7 (NodeKind); name/inputs_values strings, is_cpu_op /
            is_timer_op booleans; request_id non-empty; stage in
            {prefill, decode}; generation == stage (prefill 0 / decode 1);
            compute/comm/coll well-typed; comm src/dst/tag range checks are
            scoped to the comm-typed nodes (5/6) -- send node rank ==
            comm.src, recv node rank == comm.dst (non-comm nodes carry the
            comm defaults in real data and an empty comm in fixtures).
  [edge]    kind == "data"; from != to; from resolves (this batch, an
            earlier batch's store id, or -- M2 node GC, 2026-08-23 -- a
            pruned store id below the per-rank prune watermark; cross-batch
            parents are legal, e.g. the interval gate chaining the previous
            request's completion barrier); to is a node of THIS batch.
  [cycle]   no cycle among the in-batch edges of one rank.
  [watch]   request_id non-empty; stage in {prefill, decode}; members
            non-empty and all in THIS batch; statuses subset of
            {Success, Skipped}; identity (request_id, stage, generation)
            unique in batch; (request_id, stage) coverage: the node-stage
            set EQUALS the watch-stage set; eligibility (delta facts first):
            prefill -> the request is in-flight (arrived at this or an
            earlier epoch, not yet completed), decode -> the request's
            prefill has drained at this or an earlier epoch.
  [assign]  entries are objects with request_id / prefill_instance_index /
            decode_instance_index (opaque to C++; Python-authoritative).
  [kv]      entries are objects with event_type / trigger_request_id
            (opaque to C++; Python-authoritative ledger).
  [alarm]   arrival_world_ns >= delta.tick (no past alarm); alarm request_id
            unique in batch and NOT in-flight.
  [comm]    send/recv (src, dst, tag) pairs complete within the batch;
            collective groups of one pg_name agree on the same rank set
            (a split collective fails closed).
  [touched] when the batch carries touched_ranks (phase 5; Python-computed),
            it must be sorted unique and equal the batch's node rank set.

Counters (run-end audit): graph_batch_count (committed batches, zero-node
included), single_node_bridge_count (committed batches with exactly one
node -- the official path asserts 0, 方案 §8.3), total_nodes,
max_nodes_per_batch (avg = total / count), total_watches / total_assignments
/ total_kv_actions / total_future_alarms.
*******************************************************************************/

#ifndef EXECUTION_DRIVEN_GRAPHBATCHCOMMITTER_HH
#define EXECUTION_DRIVEN_GRAPHBATCHCOMMITTER_HH

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

#include "astra-sim/workload/execution_driven/DecisionBridge.hh"
#include "astra-sim/workload/execution_driven/DecisionMailbox.hh"

namespace AstraSim {
namespace ExecutionDriven {

class NodeStoreGraphSource;
class WatchRegistry;
class RequestIngress;

/// Persistent (rank, graph-batch json node id) -> store id key. Nodes are
/// added to the per-rank NodeStore across batches; edges and watch members
/// reference the graph-batch ids, which must be translated to the store ids
/// at commit time.
struct RankNodeKey {
    int rank = 0;
    uint64_t json_id = 0;

    bool operator==(const RankNodeKey& other) const {
        return rank == other.rank && json_id == other.json_id;
    }
};

struct RankNodeKeyHash {
    size_t operator()(const RankNodeKey& key) const {
        return std::hash<int>()(key.rank) ^
               (std::hash<uint64_t>()(key.json_id) << 1);
    }
};

/// Atomic GraphBatch committer (phase 5). One instance per online run.
class GraphBatchCommitter {
  public:
    /// Services the commit writes through. issue_rank drains one rank's free
    /// set (the official path binds systems[rank]->workload->
    /// issue_dep_free_nodes(); fixtures may pass a no-op). The commit's
    /// issue pass covers ONLY the batch's touched ranks.
    struct Context {
        int num_ranks = 0;
        std::vector<std::shared_ptr<NodeStoreGraphSource>>* graph_sources =
            nullptr;
        WatchRegistry* watch_registry = nullptr;
        RequestIngress* ingress = nullptr;
        std::function<void(int rank)> issue_rank;
        // Phase-7 §10.3: optional side-band metrics anchor registration. When
        // set, commit() invokes it right after Phase B-1 (nodes), BEFORE the
        // issue pass, with the batch and the (rank, json id) -> store id map
        // so the anchors bind to the STORE ids (the ids on_node_issue /
        // on_node_complete observe). Registering before commit with the json
        // ids would be off by one: NodeStore assigns store ids starting at 1
        // while the online graph's json ids start at 0, so a json-id anchor
        // never matches its own node (the id-0 start anchors never fire at
        // all). The watch registry translates through the same map -- the
        // anchors must too.
        std::function<void(const GraphBatch& batch,
                           const std::unordered_map<RankNodeKey, uint64_t,
                                                    RankNodeKeyHash>& store_ids)>
            metrics_anchor_hook;
        // M2 node GC (2026-08-23, --online-node-gc on the official path):
        // when set, the constructor enables collection on every per-rank
        // store and commit() collects finished childless nodes at its tail,
        // pruning store_ids_ at the same watermark (see collect_node_garbage
        // / pruned_json_watermark_). Internal default OFF: the phase fixtures
        // construct their Context explicitly and keep the pre-M2 behavior
        // (including the memory profile) -- main_online passes the CLI value
        // (frozen default 0, the 2026-08-23 ruling flip).
        bool node_gc = false;
    };

    /// Phase-5 run-end counters (方案 §8.3: single_node_bridge_count == 0 on
    /// the official path).
    struct Counters {
        uint64_t graph_batch_count = 0;         // committed batches
        uint64_t single_node_bridge_count = 0;  // committed batches with 1 node
        uint64_t total_nodes = 0;
        uint64_t max_nodes_per_batch = 0;
        uint64_t total_watches = 0;
        uint64_t total_assignments = 0;
        uint64_t total_kv_actions = 0;
        uint64_t total_future_alarms = 0;
    };

    explicit GraphBatchCommitter(Context ctx) : ctx_(std::move(ctx)) {
        // M2 (2026-08-23): propagate the node-GC switch to every per-rank
        // store once, here (fixtures leave node_gc off and keep the pre-M2
        // no-collection behavior byte-for-byte).
        if (ctx_.node_gc && ctx_.graph_sources != nullptr) {
            for (auto& source : *ctx_.graph_sources) {
                source->store().set_gc_enabled(true);
            }
        }
    }

    /// Phase A (pure). Returns an error description on ANY violation; the
    /// committer state (stores, watches, counters, tracking) is untouched.
    /// Throws only on a malformed batch entry (type errors are converted to
    /// the returned error instead; the throw path is defensive only).
    std::optional<std::string> validate(const StateDelta& delta,
                                        const GraphBatch& batch) const;

    /// Phase B. Precondition: validate(delta, batch) returned nullopt (the
    /// caller MUST gate on it). Applies the delta facts to the in-flight
    /// tracking, adds the batch's nodes/edges/watches/alarms, runs the
    /// per-rank issue pass over the touched ranks ONLY, updates the
    /// counters. Throws std::runtime_error on internal inconsistency
    /// (unreachable on validated input).
    void commit(const StateDelta& delta, const GraphBatch& batch);

    /// The persistent (rank, json id) -> store id map (cross-batch edges and
    /// watch members resolve through it; read-only outside commit()). M2
    /// (--online-node-gc): entries whose node was collected are pruned at
    /// the per-rank dense prefix watermark at commit tails; pruned parents
    /// stay resolvable in validate() and their edges become no-ops (the
    /// collected node was finished -- NodeStore's dead-parent rule).
    const std::unordered_map<RankNodeKey, uint64_t, RankNodeKeyHash>&
    store_ids() const {
        return store_ids_;
    }

    /// Phase-5 in-flight request tracking (queries for fixtures/diagnostics).
    const std::set<std::string>& in_flight_requests() const {
        return in_flight_;
    }
    const std::set<std::string>& prefill_drained_requests() const {
        return prefill_drained_;
    }

    const Counters& counters() const { return counters_; }

    /// One-line run-end counter report ("[online] phase-5 commit counters:").
    std::string counters_report() const;

    /// The sorted unique rank set owning the batch's nodes (pure).
    static std::vector<int> compute_touched_ranks(const GraphBatch& batch,
                                                  int num_ranks);

  private:
    /// Delta facts first: arrivals -> in-flight; PREFILL_DRAIN ->
    /// prefill-drained; REQUEST_COMPLETE -> removed from both. Used on local
    /// copies by validate() and on the real sets by commit().
    static void apply_delta_facts(const StateDelta& delta,
                                  std::set<std::string>& in_flight,
                                  std::set<std::string>& prefill_drained);

    /// M2 (2026-08-23) store_ids_ prune state, per rank. entries is the
    /// commit-order FIFO of (json id, store id) pairs appended at Phase B-1;
    /// per-rank json ids are allocated densely and ascending across batches
    /// (graph_batch_builder next_id counter), so the queue is ascending per
    /// rank and pruned_json_watermark_ is the strict dense prefix of json
    /// ids [0, watermark) already erased from store_ids_ (their nodes were
    /// GC'd). validate()'s cross-batch parent resolution accepts exactly
    /// in-batch | still-mapped | below-the-prune-watermark, so it never
    /// weakens: a never-emitted id stays "unresolved" and fail-closes.
    struct RankPruneQueue {
        std::vector<std::pair<uint64_t, uint64_t>> entries;
        size_t head = 0;
    };

    /// M2: collect finished childless nodes in every per-rank store and
    /// prune store_ids_ at the matching watermark. Called at the very end of
    /// commit() (a quiescent point) only when ctx_.node_gc is set.
    void collect_node_garbage();

    Context ctx_;
    std::unordered_map<RankNodeKey, uint64_t, RankNodeKeyHash> store_ids_;
    std::set<std::string> in_flight_;           // arrived, not completed
    std::set<std::string> prefill_drained_;     // prefill watch fired
    Counters counters_;
    // M2 node GC prune state (sized lazily at the first commit).
    std::vector<RankPruneQueue> prune_queues_;
    std::vector<uint64_t> pruned_json_watermark_;
};

}  // namespace ExecutionDriven
}  // namespace AstraSim

#endif  // EXECUTION_DRIVEN_GRAPHBATCHCOMMITTER_HH
