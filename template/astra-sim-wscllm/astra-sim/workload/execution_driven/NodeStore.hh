/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

NodeStore -- execution-driven mechanism layer (wscllm phase 1).

Minimal dynamic node store (方案 §4 步骤 1-4). Historical note: the
phase-1 skeleton served only "the current request's current stage" --
one GraphBatch in flight, dropped once done. Two deliberate departures
since, both documented below: M2 node GC (2026-08-23; always on since
the 2026-09-05 B.3 cleanup) erases finished nodes at the committer's
commit-tail quiescent point, and committed store ids are never reused,
so a past batch's finished parent stays recognizable to the committer's
cross-batch affine resolution (erased()). Still no general
dynamic-graph platform is built (仿真加速分析.md §5.5).

Interfaces implemented here:
  - NodeStore: the store itself (add_node / add_dependency /
    fill_free_node_snapshot / finish_node / node_ptr, plus mark_issued --
    documented additions to the plan's skeleton).
  - NodeStoreGraphSource: online-mode GraphSource over a NodeStore.

Dependency semantics: add_dependency(parent, child, kind) records a
precedence edge; a child becomes free when every recorded parent has
finished. The DepKind parameter exists for interface stability but is
deliberately NOT stored: in this first version every kind resolves as
plain precedence (documented; the plan's own §1-5 note forbids building
the general fence early, so only DepKind::Data is constructed in
production).
finish_node is idempotent -- the sole dependency-release owner.
*******************************************************************************/

#ifndef EXECUTION_DRIVEN_NODESTORE_HH
#define EXECUTION_DRIVEN_NODESTORE_HH

#include <cstdint>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

#include "astra-sim/workload/execution_driven/GraphSource.hh"

namespace AstraSim {
namespace ExecutionDriven {

enum class DepKind : int { Data = 0, Control = 1, Enabled = 2 };

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
    /// Insert a node. If global_id == 0, a fresh unique id is assigned.
    /// Nodes with no dependencies yet are free immediately.
    uint64_t add_node(OnlineNode node);

    /// Read-only diagnostic: the id a subsequent add_node() with global_id 0
    /// must receive. GraphBatchCommitter uses this to fail closed on external
    /// NodeStore id-stream drift before it mutates a batch.
    uint64_t next_auto_id() const { return next_id_; }

    /// Precedence edge: parent must finish before child becomes free.
    /// Safe when the parent was added after the child (child is removed from
    /// the free set when the edge is recorded). Fail-closed on a late edge
    /// onto an already-issued/finished child or a self-loop (S5 guard,
    /// NodeStore.cc add_dependency) -- free-set accounting would corrupt
    /// otherwise.
    void add_dependency(uint64_t parent, uint64_t child, DepKind kind);

    /// Fill a caller-owned snapshot of the current free ids, ascending.
    /// Reusing the caller's vector avoids allocating a fresh snapshot during
    /// every issue pass. The snapshot is non-consuming and remains stable if
    /// a callback mutates the store's free set while it is being traversed.
    void fill_free_node_snapshot(std::vector<uint64_t>& out) const;
    /// 2.5-3: view-pointer variant used by for_each_dep_free (same ordering
    /// and snapshot contract as the id variant).
    void fill_free_view_snapshot(std::vector<const OnlineNode*>& out) const;

    /// Mark a node as issued (removes it from the free set). No-op on
    /// unknown / already issued nodes.
    void mark_issued(uint64_t node_id);

    /// The ONLY dependency release entry (idempotent): finishes the node and
    /// frees every child whose parents are all finished.
    void finish_node(uint64_t node_id);

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

    // 2.5-1: one-lookup record API for the online terminal/issue paths.
    // find_record is the ONLY nodes_ hash find of a terminal sequence;
    // claim_terminal_observed / finish_node_record / take_node_record are
    // behavior-identical to mark_terminal_observed / finish_node /
    // mark_issued + mutable_online_statistics on the same id -- they just
    // reuse the minted record (see OnlineRecordRef in GraphSource.hh).
    OnlineRecordRef find_record(uint64_t node_id);
    bool claim_terminal_observed(const OnlineRecordRef& ref);
    void finish_node_record(const OnlineRecordRef& ref);
    /// mark_issued + the mutable statistics state in one find (issue path).
    OnlineRecordRef take_node_record(uint64_t node_id);

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

  private:
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

    uint64_t next_id_ = 1;
    std::unordered_map<uint64_t, NodeRecord> nodes_;
    /// 2.5-3: free entries carry the record's node address so an issue
    /// pass consumes views without re-finding each id. Ordered by id
    /// alone; the pointer is pure payload (never compared; ids are never
    /// reused), so key-equivalence semantics match the old id-only set.
    struct FreeEntry {
        uint64_t id;
        const OnlineNode* node;
    };
    struct FreeEntryLess {
        bool operator()(const FreeEntry& a, const FreeEntry& b) const {
            return a.id < b.id;
        }
    };
    std::set<FreeEntry, FreeEntryLess> free_ids_;

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
    // 2.5-1: tag override -- Workload routes the terminal/issue paths
    // through the one-lookup record API when this returns non-null.
    NodeStoreGraphSource* as_node_store_source() override { return this; }
    // Non-const only: every caller (committer, main_online, tests) holds a
    // mutable source and mutates or reads through it; the former const
    // overload had zero references (dead accessor, C2 2026-09-28).
    NodeStore& store() { return store_; }

    void for_each_dep_free(
        const std::function<void(const NodeView&)>& consume) override;
    void finish_node(uint64_t node_id) override;
    const NodeView* lookup_ptr(uint64_t node_id) override;
    OnlineStatisticsState* mutable_online_statistics(
        uint64_t node_id) override;
    bool mark_terminal_observed(uint64_t node_id) override;
    void take_node(uint64_t node_id) override;

  private:
    NodeStore store_;
    // Reused by for_each_dep_free(): it must remain a snapshot because a
    // callback can finish a node and release further children. Those children
    // become visible on the next issue pass, never the current one.
    std::vector<const OnlineNode*> dep_free_scratch_nodes_;
};

}  // namespace ExecutionDriven
}  // namespace AstraSim

#endif  // EXECUTION_DRIVEN_NODESTORE_HH
