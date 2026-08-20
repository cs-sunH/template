/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

NodeStore -- execution-driven mechanism layer (sh_1.0 port; blueprint wscllm phase 1).

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
                                              // duration model (n/a in both repos) --
                                              // counted in comm_bytes only)
};

/// Per-rank injected-unfinished summary at a delivery epoch (phase 3 感知,
/// --sensing-enabled). Classification per contract ⑥: compute ops / comm
/// bytes / estimated remaining service / resource state, all traceable to
/// request/stage/generation. Query/audit data only -- the sh_1.0 strategy's
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

    /// Precedence edge: parent must finish before child becomes free.
    /// Safe when the parent was added after the child (child is removed from
    /// the free set when the edge is recorded).
    void add_dependency(uint64_t parent, uint64_t child, DepKind kind);

    /// Free (all parents finished) and not yet issued nodes, ascending id.
    /// Non-consuming; mark_issued consumes.
    std::vector<uint64_t> resolve_free_nodes() const;

    /// Mark a node as issued (removes it from the free set). No-op on
    /// unknown / already issued nodes.
    void mark_issued(uint64_t node_id);

    /// The ONLY dependency release entry (idempotent): finishes the node and
    /// frees every child whose parents are all finished.
    void finish_node(uint64_t node_id);

    /// Full record (GraphSource::lookup backing).
    std::optional<OnlineNode> node(uint64_t node_id) const;

    /// Reverse index (watch/fence).
    std::optional<NodeStoreMeta> meta_for(uint64_t node_id) const;

    size_t pending_count() const;
    bool empty() const;

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
        bool issued = false;
        bool finished = false;
    };

    uint64_t next_id_ = 1;
    std::unordered_map<uint64_t, NodeRecord> nodes_;
    std::set<uint64_t> free_ids_;
};

/// Online-mode GraphSource over a NodeStore. Yields nodes only after a
/// GraphBatch populated the store; the post-commit deferred path decides
/// when (步骤 1-6/1-11). Never auto-emits successors.
class NodeStoreGraphSource : public GraphSource {
  public:
    NodeStore& store() { return store_; }
    const NodeStore& store() const { return store_; }

    std::vector<NodeView> dep_free_nodes() override;
    void finish_node(uint64_t node_id) override;
    std::optional<NodeView> lookup(uint64_t node_id) override;
    void take_node(uint64_t node_id) override;
    std::shared_ptr<Chakra::FeederV3::ETFeederNode> et_node(
        uint64_t) override {
        return nullptr;
    }
    bool static_all_done() override { return false; }

  private:
    NodeStore store_;
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
