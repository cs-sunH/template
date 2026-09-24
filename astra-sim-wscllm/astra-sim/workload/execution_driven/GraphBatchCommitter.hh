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
  request tracking, then adds the batch's nodes (persistent per-rank affine
  (json id) -> store-id translation, cross-batch), parent edges, watches,
  future arrival
  alarms, runs the per-rank issue pass over the batch's TOUCHED RANKS ONLY,
  and updates the counters. The caller (main_online) performs the
  ServiceCoordinator REQUEST_COMPLETE accounting, the completed requests'
  watch removal and the commit ack AFTER commit() returns.

Validation rules (every rule empirically verified against the real 20.csv
first-30s runs: strategy 3531 batches + replay 3491 batches, zero
violations):
  [epoch]   batch.batch_id == batch.source_delivery_sequence ==
            delta.delivery_sequence; batch.error empty.
  [node]    rank in [0, num_ranks); per-rank ids form one strictly contiguous
            stream (the first id may be non-zero), so committed history is
            represented by one exact bounded range; type
            in 1..7 (NodeKind); name/inputs_values strings, is_cpu_op /
            is_timer_op booleans; request_id non-empty; stage in
            {prefill, decode}; generation == stage (prefill 0 / decode 1);
            compute/comm/coll well-typed; type-7 collective bytes > 0 (also a
            mandatory cheap commit preflight when full validation is off);
            comm src/dst/tag range checks are
            scoped to the comm-typed nodes (5/6) -- send node rank ==
            comm.src, recv node rank == comm.dst (non-comm nodes carry the
            comm defaults in real data and an empty comm in fixtures).
  [edge]    kind == "data"; from != to; from resolves (this batch, an
            earlier batch's store id, or -- M2 node GC, 2026-08-23 -- an id
            in the rank's exact committed range; cross-batch
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
#include <vector>

#include "astra-sim/workload/execution_driven/DecisionBridge.hh"
#include "astra-sim/workload/execution_driven/DecisionMailbox.hh"

namespace AstraSim {
namespace ExecutionDriven {

class NodeStoreGraphSource;
class WatchRegistry;
class RequestIngress;
struct GraphBatchCommitterTestAccess;

/// Atomic GraphBatch committer (phase 5). One instance per online run.
/// Simulation-thread-only: its fixed JSON-id stamp scratch is deliberately
/// mutable to avoid per-batch allocation and is not externally synchronized.
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
        // issue pass, with the batch and this committer's O(1) affine
        // resolver so the anchors bind to the STORE ids (the ids on_node_issue /
        // on_node_complete observe). Registering before commit with the json
        // ids would be off by one: NodeStore assigns store ids starting at 1
        // while the online graph's json ids start at 0, so a json-id anchor
        // never matches its own node (the id-0 start anchors never fire at
        // all). The watch registry translates through the same map -- the
        // anchors must too. The hook must resolve only the ids it needs; it
        // must not recreate a per-batch id map.
        std::function<void(const GraphBatch& batch,
                           const GraphBatchCommitter& committer)>
            metrics_anchor_hook;
        // Mandatory collective-liveness preflight. The committer owns only
        // NodeStore-backed graph sources, while the declared process-group
        // membership lives in the per-rank Workload::comm_groups map. The
        // online entry supplies this read-only resolver. It receives the
        // candidate collective's participating ranks, must verify that every
        // one of those Workloads exposes the same ordered members AND
        // dimension_sizes declaration for pg_name, and returns the members.
        // Returning nullopt means missing/inconsistent metadata and fails a
        // collective batch closed before any commit state changes. Batches
        // without collective nodes do not require the callback.
        std::function<std::optional<std::vector<int>>(
            const std::string& pg_name,
            const std::vector<int>& participant_ranks)>
            communicator_members_for_pg;
        // M2 node GC (2026-08-23; the --online-node-gc CLI arm was removed
        // by the B.3 cleanup (2026-09-05), so the official path always sets
        // this switch; A1 amortization 2026-08-28): when set, the
        // constructor enables collection on every per-rank store and the
        // commit tail counts
        // pending GC candidates, draining NodeStore records only once
        // >= kGcAmortizeThreshold have
        // accumulated; finalize_node_garbage() forces one final drain before
        // the run-end diagnostics. Internal default OFF: the phase fixtures
        // construct their Context explicitly and keep the pre-M2 behavior
        // (including the memory profile) -- main_online passes true
        // unconditionally (the B.3 cleanup (2026-09-05) removed the
        // --online-node-gc CLI arm).
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

    explicit GraphBatchCommitter(Context ctx)
        : ctx_(std::move(ctx)),
          rank_affines_(ctx_.num_ranks > 0
                            ? static_cast<size_t>(ctx_.num_ranks)
                            : size_t{0}) {
        const size_t rank_count = rank_affines_.size();
        json_id_expected_by_rank_.assign(rank_count, 0);
        json_id_seen_stamp_by_rank_.assign(rank_count, 0);
        json_id_discovered_touched_ranks_.reserve(rank_count);
        commit_touched_ranks_.reserve(rank_count);
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

    struct ValidateAndCommitResult {
        std::optional<std::string> error;
        uint64_t validation_ns = 0;
    };

    /// Run the complete validator and, only on success, immediately apply
    /// Phase B in the same call stack.  There is deliberately no externally
    /// storable "validated" capability: callers cannot mutate a batch or the
    /// committer between Phase A and Phase B.  Its by-value result exposes
    /// only validate_impl() wall time (not Phase B) after the operation has
    /// returned, so a timing output cannot alias and mutate the inputs.  A
    /// result error means Phase B was not entered; commit-time internal
    /// failures still throw like commit().
    ValidateAndCommitResult validate_and_commit(const StateDelta& delta,
                                                const GraphBatch& batch);

    /// Phase B. Full semantic validation is caller-selectable; commit always
    /// runs the mandatory affine/id/liveness safety preflight (including
    /// cycles, satisfiable watches, exact p2p multiplicities, exact declared
    /// collective membership, future-alarm accounting, and variant-specific
    /// resource-liveness checks) before it changes any state.
    /// Applies the delta facts to the in-flight
    /// tracking, adds the batch's nodes/edges/watches/alarms, runs the
    /// per-rank issue pass over the touched ranks ONLY, updates the
    /// counters. Throws std::runtime_error on internal inconsistency
    /// (unreachable on validated input).
    void commit(const StateDelta& delta, const GraphBatch& batch);

    /// Translate a committed (rank, json id) to its NodeStore id without
    /// allocating. The affine range remains valid after NodeStore GC; callers
    /// that need liveness must query NodeStore::erased() separately.
    std::optional<uint64_t> resolve_store_id(int rank,
                                             uint64_t json_id) const;

    /// Test/diagnostic view of the fixed metadata footprint: exactly one
    /// RankAffine slot per configured rank, independent of node count.
    size_t rank_affine_count() const { return rank_affines_.size(); }

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

    /// A1 (2026-08-28): amortization threshold -- the commit tail drains the
    /// per-rank GC candidate FIFOs only once this many finished nodes have
    /// accumulated since the last collection. 4096 keeps the retained
    /// window bounded while making the per-commit cost an O(#ranks) counter
    /// sum instead of a full drain (the light-load wall regression that
    /// forced the old default-off ruling came from draining every commit).
    static constexpr size_t kGcAmortizeThreshold = 4096;

    /// A1 (2026-08-28): run-end forced collection -- drains every candidate
    /// FIFO regardless of the amortization counter (no-op when node_gc is
    /// off), so the run-end
    /// diagnostics report the true final retained window. Call after the
    /// event loop ends (all anchors/metrics already emitted).
    void finalize_node_garbage();

  private:
    friend struct GraphBatchCommitterTestAccess;
    /// Delta facts first: arrivals -> in-flight; PREFILL_DRAIN ->
    /// prefill-drained; REQUEST_COMPLETE -> removed from both. Used on local
    /// copies by validate() and on the real sets by commit().
    static void apply_delta_facts(const StateDelta& delta,
                                  std::set<std::string>& in_flight,
                                  std::set<std::string>& prefill_drained);

    /// Exact O(1) translation for one rank's strict contiguous json-id stream.
    /// [json_first, json_next) is the committed json-id range and maps to
    /// [store_first, store_next) in that rank's monotonic NodeStore id space.
    /// The first returned store id is captured from add_node(), so preseeded
    /// stores and arbitrary json starts are represented exactly.
    struct RankAffine {
        bool initialized = false;
        uint64_t json_first = 0;
        uint64_t json_next = 0;
        uint64_t store_first = 0;
        uint64_t store_next = 0;
    };

    /// M2: collect finished childless nodes in every per-rank store. The
    /// affine translations survive collection, allowing an erased committed
    /// parent to remain an exact finished-parent no-op. Called at the very end
    /// of commit() (a quiescent point, A1-amortized: only once the pending
    /// candidate count reaches kGcAmortizeThreshold) and from
    /// finalize_node_garbage(), only when ctx_.node_gc is set.
    void collect_node_garbage();
    std::optional<std::string> validate_impl(
        const StateDelta& delta, const GraphBatch& batch,
        std::vector<int>* json_id_touched_ranks) const;
    std::optional<std::string> validate_affine_drift() const;
    std::optional<std::string> validate_json_id_stream(
        const GraphBatch& batch,
        std::vector<int>* touched_ranks = nullptr) const;
    /// Production-safe subset of validation. Unlike validate(), this checks
    /// only invariants whose violation can permanently strand a node, watch,
    /// callback, collective, or request-accounting record. It is pure and
    /// runs unconditionally at the start of commit().
    std::optional<std::string> mandatory_liveness_preflight(
        const StateDelta& delta, const GraphBatch& batch) const;
    void record_affine_node(int rank, uint64_t json_id, uint64_t store_id);
    void commit_after_preflight(const StateDelta& delta, const GraphBatch& batch,
                                const std::vector<int>& touched_ranks);

    Context ctx_;
    std::set<std::string> in_flight_;           // arrived, not completed
    std::set<std::string> prefill_drained_;     // prefill watch fired
    Counters counters_;
    // Fixed O(num_ranks) committed-id history and json->store translation.
    std::vector<RankAffine> rank_affines_;
    // Commit-time json-id preflight scratch.  A rank stamp replaces the old
    // per-batch zeroed vectors: one allocation per committer, O(nodes) work
    // per batch, and sorted touched ranks only over ranks actually present.
    mutable std::vector<uint64_t> json_id_expected_by_rank_;
    mutable std::vector<uint64_t> json_id_seen_stamp_by_rank_;
    mutable uint64_t json_id_stream_stamp_ = 0;
    mutable std::vector<int> json_id_discovered_touched_ranks_;
    std::vector<int> commit_touched_ranks_;
    // validate_and_commit() owns commit_touched_ranks_ across its whole
    // validation/commit sequence.  Context callbacks attempting a nested
    // mutating commit fail closed instead of corrupting that scratch.
    bool validation_commit_in_progress_ = false;
};

}  // namespace ExecutionDriven
}  // namespace AstraSim

#endif  // EXECUTION_DRIVEN_GRAPHBATCHCOMMITTER_HH
