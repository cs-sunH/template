/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

NodeStore -- execution-driven mechanism layer (wscllm phase 1).

Minimal dynamic node store (方案 §4 步骤 1-4). First version supports only
"the current request's current stage": a GraphBatch adds a batch of nodes,
finish_node releases their successors, and the batch is dropped once done.
No general dynamic-graph platform is built (仿真加速分析.md §5.5).

Interfaces implemented here:
  - NodeStore: the store itself (add_node / add_dependency / resolve_free_nodes
    / finish_node / meta_for / pending_count / empty, plus mark_issued and
    node() as GraphSource::take_node / lookup backing -- documented additions
    to the plan's skeleton).
  - NodeStoreGraphSource: online-mode GraphSource over a NodeStore.
  - ETFeederGraphSource: static-mode GraphSource over ETFeeder +
    DependancyResolver; the ONLY place Workload.cc still sees
    getDependancyResolver (step-1-4 acceptance grep).

Dependency semantics: add_dependency(parent, child, kind) records a
precedence edge; a child becomes free when every recorded parent has
finished. DepKind (Data/Control/Enabled) is stored for the later fence
upgrade; in this first version all kinds resolve identically (documented;
the plan's own §1-5 note forbids building the general fence early).
finish_node is idempotent -- the sole dependency-release owner.
*******************************************************************************/

#ifndef EXECUTION_DRIVEN_NODESTORE_HH
#define EXECUTION_DRIVEN_NODESTORE_HH

#include <cstdint>
#include <cstddef>
#include <utility>
#include <optional>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

#include "astra-sim/workload/execution_driven/GraphSource.hh"

namespace Chakra {
namespace FeederV3 {
class ETFeeder;
}  // namespace FeederV3
}  // namespace Chakra

namespace AstraSim {
namespace ExecutionDriven {

enum class DepKind : int { Data = 0, Control = 1, Enabled = 2 };

/// Reverse-index payload for watch/fence (步骤 1-5).
struct NodeStoreMeta {
    std::string request_id;
    std::string stage;
    uint64_t generation = 0;
};

struct NodeStoreTestAccess;

// ---------------------------------------------------------------------------
// Phase-3 sensing: per-rank injected-unfinished ledger summary (方案 §6.2
// 操作 1 / contract ⑥ "决策摘要不得只给节点数"). One (request_id, stage,
// generation) group of committed-but-not-yet-terminal nodes.
// ---------------------------------------------------------------------------

/// One committed-but-unfinished group, traceable per contract ⑥.
struct InjectedUnfinishedEntry {
    std::string request_id;
    std::string stage;
    uint64_t generation = 0;
    uint64_t node_count = 0;                  // unfinished nodes of the group
    uint64_t compute_ops = 0;                 // sum of num_ops (compute only)
    uint64_t comm_bytes = 0;                  // sum of bytes (send/recv/coll)
    uint64_t estimated_remaining_ns = 0;      // sum of runtime_ns (compute
                                              // service estimate; comm has no
                                              // duration model in wscllm --
                                              // counted in comm_bytes only)
};

/// Per-rank injected-unfinished summary at a delivery epoch (phase 3 感知,
/// --sensing-enabled). Classification per contract ⑥: compute ops / comm
/// bytes / estimated remaining service / resource state, all traceable to
/// request/stage/generation. Query/audit data only -- the wscllm strategy's
/// red-line decision inputs (Python queue ledger + KV ledger + static route)
/// never consume it.
struct RankInjectedSummary {
    int rank = 0;
    uint64_t node_count = 0;              // unfinished nodes (issued or not)
    uint64_t compute_ops = 0;             // classification by compute ops
    uint64_t comm_bytes = 0;              // classification by comm bytes
    uint64_t estimated_remaining_ns = 0;  // 预计剩余服务量 (compute runtime
                                          // sum; timers count 0)
    uint64_t in_flight_node_count = 0;    // resource state: issued, unfinished
    uint64_t in_flight_gpu_ops = 0;       // resource state: issued compute ops
    uint64_t free_node_count = 0;         // resource state: free, not issued
    std::vector<InjectedUnfinishedEntry> per_request;  // traceable breakdown
};

class NodeStore {
  public:
    /// Cumulative work counters for the ready-set surface. These are small
    /// monotone counters used by focused issue-path diagnostics; they do not
    /// own or duplicate any ready IDs.
    struct ReadySetStats {
        uint64_t snapshot_calls = 0;
        uint64_t snapshot_ids_copied = 0;
        // v11 Phase 4 (issue-pass increment): the snapshot rebuilt by
        // splicing the since-snapshot delta onto a previous exact vector
        // instead of re-walking the whole ordered set. splice_ids_written is
        // the sequential element writes of the rebuild (the cost that
        // replaced the O(R) set traversal); splice_delta_ids is |insert
        // delta| + |erase delta| consumed by that splice.
        uint64_t splice_calls = 0;
        uint64_t splice_ids_written = 0;
        uint64_t splice_delta_ids = 0;
        uint64_t membership_queries = 0;
        uint64_t free_insert_updates = 0;
        uint64_t free_erase_updates = 0;
        uint64_t delta_insertions = 0;
        uint64_t delta_erasures = 0;
        uint64_t delta_set_insert_calls = 0;
        uint64_t delta_set_erase_calls = 0;
        uint64_t delta_snapshot_reads = 0;
        uint64_t delta_snapshot_materializations = 0;
        uint64_t delta_snapshot_ids_copied = 0;
    };

    /// Insert a node. If global_id == 0, a fresh unique id is assigned.
    /// Nodes with no dependencies yet are free immediately.
    uint64_t add_node(OnlineNode node);

    /// Read-only diagnostic: the id a subsequent add_node() with global_id 0
    /// must receive. GraphBatchCommitter uses this to fail closed on external
    /// NodeStore id-stream drift before it mutates a batch.
    uint64_t next_auto_id() const { return next_id_; }

    /// True for a previously allocated physical ID whose node has finished,
    /// whether the record is still resident or already GC'd. Unknown/future
    /// ids are false. (Store-level generic query; survived the retirement of
    /// the descriptor virtual holes, which used to be probed here.)
    bool is_finished_or_erased(uint64_t node_id) const;

    /// Precedence edge: parent must finish before child becomes free.
    /// Safe when the parent was added after the child (child is removed from
    /// the free set when the edge is recorded).
    void add_dependency(uint64_t parent, uint64_t child, DepKind kind);

    /// Free (all parents finished) and not yet issued nodes, ascending id.
    /// Non-consuming; mark_issued consumes.
    std::vector<uint64_t> resolve_free_nodes() const;

    /// Current ready-set membership check for a captured id. This is an
    /// ordered-set lookup, not a scan; issue snapshots use it to skip a node
    /// that was taken or finished by an earlier synchronous callback.
    bool is_free_id(uint64_t node_id) const;

    /// Fill a caller-owned snapshot of the current free ids, ascending.
    /// Reusing the caller's vector avoids allocating a fresh snapshot during
    /// every issue pass. The snapshot is non-consuming and remains stable if
    /// a callback mutates the store's free set while it is being traversed.
    /// The ordered free_ids_ set is copied directly; no second mutable ready
    /// vector is maintained on each insert/erase.
    void fill_free_node_snapshot(std::vector<uint64_t>& out) const;

    /// v11 Phase 4 (issue-pass increment, 方案 §6.1): rebuild the caller's
    /// previous snapshot in place by splicing the since-snapshot delta onto
    /// it, avoiding the O(R) ordered-set traversal of
    /// fill_free_node_snapshot. EQUIVALENCE PROOF (must stay in sync with
    /// the implementation):
    ///   Let V = `inout` content right after the caller's last successful
    ///   fill/splice of this store at generation G0 (the caller-side tags
    ///   guarantee this; see free_delta_window_base_version()). Since G0
    ///   every free-set mutation went through record_free_id_insert /
    ///   record_free_id_erase, and those record exactly:
    ///     - D_ins  = ids inserted since G0 and not erased again
    ///                (the existing free_delta_ids_),
    ///     - D_era  = ids erased since G0 that were NOT a D_ins member at
    ///                their erase time (the erase-delta set below; an
    ///                inserted-then-erased id leaves no trace in either,
    ///                which is correct: it never entered V).
    ///   For x in D_era that was re-inserted afterwards, the re-insert put
    ///   it back into D_ins (erase removes from D_ins, insert re-adds), so
    ///   "x in D_ins" always wins over "x in D_era" and reflects current
    ///   live membership. Hence the ready set NOW equals
    ///        merge( V \ D_era , D_ins )
    ///   element-for-element, ascending, deduplicated -- exactly what
    ///   fill_free_node_snapshot would assign. The splice consumes BOTH
    ///   deltas and re-arms the window at the current generation, exactly
    ///   like a full fill.
    /// Precondition (caller-enforced, see Workload.cc): free_id_version()
    /// differs from the caller's recorded snapshot generation (otherwise
    /// the caller should reuse, not splice) and
    /// free_delta_window_base_version() equals that recorded generation
    /// (the delta window still starts exactly at the caller's vector; an
    /// intervening fill by anyone else would have consumed the window).
    /// Returns true on success; inout is untouched (strong guarantee) if
    /// the splice throws, so caller-side validity tags may describe the
    /// vector across an exception.
    bool splice_free_node_snapshot(std::vector<uint64_t>& inout) const;

    /// The free_id_version at which the current since-snapshot delta window
    /// was armed (the last fill or splice). The v11 Phase 4 splice
    /// preconditions compare it against the caller's recorded snapshot
    /// generation.
    uint64_t free_delta_window_base_version() const {
        return free_delta_window_base_version_;
    }

    /// B3 incremental ready queue: ids that entered the free set since the
    /// last fill_free_node_snapshot consume (ascending, deduplicated --
    /// a repeated dependency release never re-enqueues an id). Read-only
    /// inspection; the sorted vector is materialized from the delta set on
    /// demand and drained by the next snapshot fill. Reacquire the reference
    /// after a ready-set mutation to observe the current delta.
    const std::vector<uint64_t>& free_delta_since_snapshot() const;
    /// Saturating monotone free-set mutation version (B3 bucket bookkeeping).
    /// UINT64_MAX is terminal: equality-based snapshot reuse must stay disabled.
    uint64_t free_id_version() const { return free_id_version_; }
    /// Copying from an original store leaves it eligible. Copy destinations
    /// and both sides of a move use full snapshots; assignment invalidates its
    /// destination before container changes. Payload copy/move stays normal.
    bool snapshot_reuse_eligible() const {
        return snapshot_reuse_eligibility_.enabled;
    }
    /// Counters for the production NodeStore ready-set operations.
    const ReadySetStats& ready_set_stats() const { return ready_set_stats_; }

    /// Mark a node as issued (removes it from the free set). No-op on
    /// unknown / already issued nodes.
    void mark_issued(uint64_t node_id);

    /// The ONLY dependency release entry (idempotent): finishes the node and
    /// frees every child whose parents are all finished.
    void finish_node(uint64_t node_id);

    /// Full record (GraphSource::lookup backing).
    std::optional<OnlineNode> node(uint64_t node_id) const;

    /// Zero-copy full record (GraphSource::lookup_ptr / for_each_dep_free
    /// backing). Lifetime contract (M2 node GC, 2026-08-23): collection
    /// happens ONLY at the committer's end-of-commit quiescent point
    /// (collect_garbage); finish_node never erases synchronously, so a
    /// pointer obtained inside a Workload callback stays valid until that
    /// callback returns (every current holder uses it within one callback:
    /// issue paths touch free i.e. unfinished nodes; the terminal paths run
    /// before finish_node, except Workload::skip_invalid which re-looks-up
    /// within the same synchronous flow). Never hold the pointer across a
    /// commit boundary.
    const OnlineNode* node_ptr(uint64_t node_id) const;

    // Mutable, short-lived online statistics payload for an issued node.
    // It belongs to the NodeStore record and disappears with the record at
    // the normal GC quiescent point; Statistics never owns a second per-node
    // map in compact service mode.
    OnlineStatisticsState* mutable_online_statistics(uint64_t node_id);

    // R2 (2026-08-29) MetricCollector anchor fast path: OR the sparse
    // routing flags into the stored record. Called ONLY from the dynamic
    // anchor registration hook (GraphBatchCommitter Phase B-1.5, AFTER
    // add_node assigned the store id and BEFORE the issue pass), because
    // add_node itself cannot know yet whether the node will be anchored.
    // OR semantics: separate registrations each know only their own edge.
    // Unknown node_id: no-op (that anchor can never fire either; matches
    // online_register_ranks silently ignoring unknown requests).
    void set_metric_anchor_flags(uint64_t node_id, bool issue, bool complete);

    // Claim the sole terminal-observer transition.  Unknown, unissued,
    // already-finished, or duplicate terminal callbacks return false so
    // Workload can fail closed before recording statistics or observer facts.
    bool mark_terminal_observed(uint64_t node_id);

    /// Reverse index (watch/fence).
    std::optional<NodeStoreMeta> meta_for(uint64_t node_id) const;

    size_t pending_count() const;
    bool empty() const;

    // -------------------------------------------------------------------------
    // M2 node GC (2026-08-23; the --online-node-gc CLI arm was removed by
    // the B.3 cleanup (2026-09-05) -- collection is always on on the
    // official path). Finished nodes with no
    // unfinished children are erased at the committer's end-of-commit
    // quiescent point, keeping the store at the in-flight window instead of
    // the whole-run cumulative graph. Amortized O(1) per node (candidate
    // FIFO: every node enters at most once -- see finish_node).
    // -------------------------------------------------------------------------

    /// Enable/disable collection (default OFF: fixtures keep the pre-M2
    /// no-collection behavior, including the memory profile; the
    /// --online-node-gc CLI arm was removed by the B.3 cleanup (2026-09-05),
    /// so the official path always enables collection). With GC off,
    /// finish_node does not even enqueue.
    void set_gc_enabled(bool enabled);

    /// Drain the candidate FIFO and erase every eligible node. MUST only be
    /// called from a quiescent point (no Workload callback in flight); the
    /// official caller is GraphBatchCommitter::commit's tail.
    void collect_garbage();

    /// True when the node id is not in the store. For a committed store id
    /// this means "collected" (finished + all children finished); used by
    /// the committer to recognize an affine-resolved finished parent.
    bool erased(uint64_t node_id) const;

    /// A1 amortization (2026-08-28): finished candidates currently waiting
    /// in the FIFO (drained by the next collect_garbage). The committer's
    /// commit tail sums this across stores to decide whether the amortized
    /// collection threshold has been reached.
    size_t pending_gc_count() const {
        return gc_fifo_.size() - gc_fifo_head_;
    }

    /// Diagnostics: nodes erased by collect_garbage so far.
    size_t gc_erased_count() const { return gc_erased_count_; }

    /// Diagnostics: records currently retained (unfinished + pinned-by-
    /// children finished nodes + not-yet-collected candidates).
    size_t retained_count() const { return nodes_.size(); }

    /// Phase-3 sensing: per-rank injected-unfinished summary (committed but
    /// not yet terminal), classified by compute ops / comm bytes / estimated
    /// remaining service / resource state and grouped per
    /// (request_id, stage, generation). Pure query -- no state is touched;
    /// the cost is paid only when --sensing-enabled turns it on.
    RankInjectedSummary injected_unfinished_summary(int rank) const;

    // -------------------------------------------------------------------------
    // W1 (joint work ledger, 2026-09-29): per-node const view for the causal
    // work ledger. Unlike injected_unfinished_summary (attribute sums over
    // unfinished nodes), this exports PER-NODE facts -- issued/finished
    // lifecycle, blocking parents, and the raw workload attributes -- so the
    // Python-side ledger can key works by (rank, node_id), reconstruct
    // dependencies, and diff consecutive frames into the six lifecycle event
    // kinds (commit/issue/progress/completion/ack/release). The query runs at
    // the tick-end static point (same argument as the sensing summary: single
    // threaded event loop; M2 GC only fires at the committer's end-of-commit
    // tail, so a finished-but-retained node -- one with unfinished children,
    // i.e. a live release obligation -- is still visible here).
    // -------------------------------------------------------------------------

    /// One node's work-ledger facts (see work_ledger_view).
    struct NodeWorkView {
        uint64_t node_id = 0;
        std::string request_id;
        std::string stage;
        uint64_t generation = 0;
        int kind = 0;                    // NodeKind raw value
        /// Stable kind name (Python keys on this; immune to enum renumber).
        std::string kind_name;
        bool issued = false;
        bool finished = false;
        uint32_t unresolved_parents = 0;
        /// Parents still resident AND unfinished (= who blocks this node).
        std::vector<uint64_t> blocking_parents;
        uint64_t num_ops = 0;
        uint64_t bytes = 0;              // comm bytes (send/recv/collective)
        uint64_t runtime_ns = 0;         // compute/replay static estimate
        uint64_t tensor_size = 0;        // MEM_LOAD/MEM_STORE payload
    };

    /// Per-store node work view, ascending node id (deterministic diffing).
    /// Pure query; includes finished-but-uncollected nodes.
    std::vector<NodeWorkView> work_ledger_view() const;

  private:
    friend struct NodeStoreTestAccess;

    // Called after free_ids_ has been changed. This preserves the public
    // delta/version semantics without maintaining a second sorted snapshot.
    void record_free_id_insert(uint64_t node_id);
    void record_free_id_erase(uint64_t node_id);

    struct NodeRecord {
        OnlineNode node;
        std::vector<uint64_t> parents;
        std::vector<uint64_t> children;
        uint32_t unresolved_parents = 0;
        // M2 node GC: live-child counter of the node (incremented when an
        // edge is recorded onto an unfinished parent -- validate enforces
        // that every edge child is a fresh batch node; decremented when that
        // child finishes). Zero + finished = GC-eligible.
        uint32_t unfinished_children = 0;
        bool issued = false;
        bool finished = false;
        bool terminal_observed = false;
    };

    // The first NodeStore data member so defaulted NodeStore copy/move
    // assignment invalidates reuse eligibility before touching any container
    // that may throw. The wrapper deliberately does not alter payload copies:
    // copies disable only the destination; moves disable source and target.
    struct SnapshotReuseEligibility {
        bool enabled = true;

        SnapshotReuseEligibility() noexcept = default;
        SnapshotReuseEligibility(const SnapshotReuseEligibility&) noexcept
            : enabled(false) {}
        SnapshotReuseEligibility(SnapshotReuseEligibility&& other) noexcept
            : enabled(false) {
            other.enabled = false;
        }
        SnapshotReuseEligibility& operator=(
            const SnapshotReuseEligibility&) noexcept {
            enabled = false;
            return *this;
        }
        SnapshotReuseEligibility& operator=(
            SnapshotReuseEligibility&& other) noexcept {
            enabled = false;
            other.enabled = false;
            return *this;
        }
    };

    SnapshotReuseEligibility snapshot_reuse_eligibility_;
    uint64_t next_id_ = 1;
    std::unordered_map<uint64_t, NodeRecord> nodes_;
    std::set<uint64_t> free_ids_;
    // The ordered free_ids_ set is both the ready-set authority and the source
    // for callback-safe caller-owned snapshots. The since-snapshot delta uses
    // a set so arbitrary callback order cannot cause suffix shifts. Its vector
    // API is materialized lazily for the diagnostic/test consumer only.
    mutable std::set<uint64_t> free_delta_ids_;
    // v11 Phase 4 (issue-pass increment): ids erased from the free set
    // since the last delta consume that were NOT simultaneously pending
    // insert-delta members (an insert-then-erase pair vanishes from both --
    // it never reached any snapshot). Splice-maintained and consumed by
    // the same two consume points as free_delta_ids_ (both the fill and
    // the splice). Entry is harmless if the id was never in a snapshot
    // (a no-op erase still bumps the version and lands here).
    mutable std::set<uint64_t> free_erase_delta_ids_;
    mutable std::vector<uint64_t> free_delta_snapshot_cache_;
    mutable bool free_delta_snapshot_cache_valid_ = true;
    // The generation at which the current delta window was armed; see
    // free_delta_window_base_version(). Bumped at every consume point
    // alongside the delta clears. Payload copy/move: follows the container
    // contents; reuse eligibility is what actually gates splice/reuse, and
    // the SnapshotReuseEligibility wrapper disables it on copy/move.
    mutable uint64_t free_delta_window_base_version_ = 0;
    uint64_t free_id_version_ = 0;
    mutable ReadySetStats ready_set_stats_;

    // M2 node GC state. gc_fifo_ holds node ids in finish order; every node
    // enters at most once (either childless at its own finish, or via the
    // last child's finish re-enqueue), so the total FIFO traffic is O(nodes)
    // and collect_garbage is amortized O(1) per node.
    bool gc_enabled_ = false;
    std::vector<uint64_t> gc_fifo_;
    size_t gc_fifo_head_ = 0;
    size_t gc_erased_count_ = 0;
};

/// Online-mode GraphSource over a NodeStore. Yields nodes only after a
/// GraphBatch populated the store; the post-commit deferred path decides
/// when (步骤 1-6/1-11). Never auto-emits successors.
class NodeStoreGraphSource : public GraphSource {
  public:
    NodeStore& store() { return store_; }
    const NodeStore& store() const { return store_; }

    std::vector<NodeView> dep_free_nodes() override;
    bool fill_dep_free_id_snapshot(std::vector<uint64_t>& out) override;
    bool is_dep_free_member(uint64_t node_id) override;
    void for_each_dep_free(
        const std::function<void(const NodeView&)>& consume) override;
    void finish_node(uint64_t node_id) override;
    std::optional<NodeView> lookup(uint64_t node_id) override;
    const NodeView* lookup_ptr(uint64_t node_id) override;
    OnlineStatisticsState* mutable_online_statistics(
        uint64_t node_id) override;
    bool mark_terminal_observed(uint64_t node_id) override;
    void take_node(uint64_t node_id) override;
    std::shared_ptr<Chakra::FeederV3::ETFeederNode> et_node(
        uint64_t) override {
        return nullptr;
    }
    bool static_all_done() override { return false; }

  private:
    NodeStore store_;
    // Reused by for_each_dep_free(): it must remain a snapshot because a
    // callback can finish a node and release further children. Those children
    // become visible on the next issue pass, never the current one.
    std::vector<uint64_t> dep_free_scratch_ids_;
};

/// Static-mode GraphSource over the ETFeeder + DependancyResolver. NodeViews
/// are built in place from the ETFeederNodes, keeping the static path
/// byte-for-byte equivalent. Non-owning ETFeeder pointer (Workload owns and
/// destroys it).
class ETFeederGraphSource : public GraphSource {
  public:
    ETFeederGraphSource(Chakra::FeederV3::ETFeeder* et_feeder, int rank);

    std::vector<NodeView> dep_free_nodes() override;
    void finish_node(uint64_t node_id) override;
    std::optional<NodeView> lookup(uint64_t node_id) override;
    void take_node(uint64_t node_id) override;
    std::shared_ptr<Chakra::FeederV3::ETFeederNode> et_node(
        uint64_t node_id) override;
    bool static_all_done() override;

  private:
    NodeView view_of(uint64_t node_id) const;

    Chakra::FeederV3::ETFeeder* et_feeder_;
    int rank_;
};

}  // namespace ExecutionDriven
}  // namespace AstraSim

#endif  // EXECUTION_DRIVEN_NODESTORE_HH
