/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

GraphSource -- execution-driven mechanism layer (wscllm phase 1).

Dynamic graph source abstraction (方案 §4 步骤 1-2 操作 4 / 步骤 1-4 操作 4;
动态图边界 contract ①). The online-mode source (NodeStoreGraphSource) is
injected at Sys creation time so the ETFeeder is never constructed and no
.et file is required.

**依赖状态唯一所有者 = GraphSource**: Workload calls GraphSource::finish_node
exactly once per node; CompletionObserver/watch only record facts and never
release dependencies a second time (步骤 1-5 操作 3).

**在线模式不得自动发射后继**: for_each_dep_free() in online mode yields nodes
only when the post-commit deferred path drains them into the source; the
static auto-advance (issue_dep_free_nodes after every completion) must not
leak into the online mode -- the online mode gate lives in Workload::call.

NodeView / OnlineNode share one POD: add_node stores it, for_each_dep_free()
/ lookup_ptr() hand it out as the read view. Fields are the full transitive
consumer inventory of a node from issue to terminal (step-1-4 operation 1):
HardwareResource (is_timer_op / is_cpu_op / kind), Statistics (id / type /
is_cpu_op), MetricCollector (rank + node_id only), comm
src/dst/tag/size, collective comm_type/size/priority/pg_name/involved_dim,
metadata inputs_values, plus the online-only request_id/stage/generation
reverse index (步骤 1-5).
*******************************************************************************/

#ifndef EXECUTION_DRIVEN_GRAPHSOURCE_HH
#define EXECUTION_DRIVEN_GRAPHSOURCE_HH

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "DieTypes.hh"

namespace AstraSim {
namespace ExecutionDriven {

/// Semantic node classification (mirrors ChakraProtoMsg::NodeType; the raw
/// numeric value is kept alongside in OnlineNode::node_type for byte-exact
/// logs and statistics mapping).
enum class NodeKind : int {
    Invalid = 0,
    Metadata = 1,
    MemLoad = 2,
    MemStore = 3,
    Compute = 4,
    CommSend = 5,
    CommRecv = 6,
    CommCollective = 7,
};

/// Compute-node role (die backend; contract §2): "compute" is a real
/// compute op, "control_marker" a micro/sentinel node (first_token, drain,
/// exit, sentinel, join, pstart, wakeup, ... -- the comp(name,1,1) class)
/// that must keep the legacy micro-job completion semantics instead of a
/// die plan. The JSON spellings are exactly "compute" | "control_marker".
enum class OpRole { Compute, ControlMarker };

/// Compute-node attributes (issue_comp / issue_replay).
struct ComputeAttrs {
    uint64_t num_ops = 0;
    uint64_t tensor_size = 0;
    uint64_t runtime_ns = 0;  // online GraphBatch sends ns directly (no
                              // unit conversion anywhere on the C++ side)
    // Retained cross-repo guard shape (2026-09-30 note): production parse
    // paths reject the remote_weight_bytes key outright (ParsedGraphBatch
    // fail-closes on it), so this flag stays false on every production
    // node; it survives for direct-construction fixtures and the
    // Workload-level defensive throw that mirrors it.
    bool has_remote_weight_bytes = false;
    uint64_t remote_weight_bytes = 0;
    // Die backend: the node's dispatch role and, when present, the
    // strictly parsed immutable DieOpSpec. die_op_spec present implies
    // runtime_ns == 0 (parser-enforced, contract §2); a real
    // op_role=Compute node WITHOUT a die_spec fails closed in the die
    // backend dispatch (never silently falls back to roofline). The
    // shared_ptr keeps one immutable spec per node without copying the
    // JSON payload into every short-lived view (design doc §8.3).
    OpRole op_role = OpRole::Compute;
    std::shared_ptr<const DieModel::DieOpSpec> die_spec;
    // Preserve the complete source descriptor alongside the legacy parsed
    // view. coarse_die_v2 compiles the original JSON so spans, TP ownership,
    // storage precision, readiness and future schema additions are never
    // reconstructed from num_ops/tensor_size or a lossy typed projection.
    std::shared_ptr<const nlohmann::json> die_task_json;
};

/// Point-to-point comm attributes (issue_send_comm / issue_recv_comm).
struct CommAttrs {
    uint64_t bytes = 0;
    int src = 0;
    int dst = 0;
    uint32_t tag = 0;
    // GraphBatch comm attribute "hbm_charge" (snake_case, bool, default
    // true): false = this endpoint creates no local-HBM job under
    // hbm-bandwidth-contention (e.g. a pure control/signal transfer). All
    // wscllm comm nodes are real data endpoints, so the generator never
    // writes false; the field exists so the C++ side can honor an explicit
    // opt-out. The kebab-case spelling "hbm-charge" is NOT legal input --
    // the parser's N13 exact-key-set rule fail-closed rejects it.
    bool hbm_charge = true;
};

/// Collective comm attributes (issue_coll_comm).
struct CollAttrs {
    uint64_t comm_type = 0;  // ChakraProtoMsg::CollectiveCommType value
    uint64_t bytes = 0;
    uint32_t priority = 0;
    std::string pg_name;
    std::vector<bool> involved_dim;  // parsed bool_list; absent = stays
                                     // empty and the commit preflight
                                     // fails closed (must be non-empty
                                     // and match the declared dimension
                                     // count -- the retired ET adapter's
                                     // 4x-true default no longer exists)
};

// Online-only, short-lived statistics state.  It lives in the NodeStore
// record rather than Statistics::operator_statistics, so service execution
// can retire terminal nodes without a global per-node statistics hash-table
// operation.  Static ET adapters leave this POD at its defaults.
struct OnlineStatisticsState {
    static constexpr uint64_t kInvalidTick = UINT64_MAX;

    uint64_t start_time = kInvalidTick;
    bool started = false;
    bool completed = false;
    bool is_gpu = false;

    // Short-lived per-node roofline statistics state, filled by the
    // online Workload while the node executes and kept until terminal
    // completion.  Only compute/memory utilization is compacted globally.
    // vector_utilization is the die-backend addition (contract §7.6):
    // eq-ops / (vector peak * elapsed); unset for the legacy
    // roofline/replay paths.
    std::optional<double> memory_utilization;
    std::optional<double> compute_utilization;
    std::optional<double> vector_utilization;
};

/// The node record / read view. The NodeStore fills every field from
/// GraphBatch data, including the reverse index request_id / stage /
/// generation.
struct OnlineNode {
    uint64_t global_id = 0;
    int rank = 0;
    NodeKind kind = NodeKind::Invalid;
    uint64_t node_type = 0;  // raw ChakraProtoMsg::NodeType value
    std::string name;
    bool is_cpu_op = false;
    bool is_timer_op = false;
    // R2 MetricCollector anchor fast-path cache (sparse): set by
    // NodeStore::set_metric_anchor_flags when the dynamic anchor
    // registration path (GraphBatchCommitter Phase B-1.5) registers an
    // anchor for this (rank, node); false = no routing entry exists on
    // that edge, so Workload skips the two-level MetricCollector hash
    // lookup entirely. Placed right after is_timer_op to reuse the 6-byte
    // tail padding of the bool group (is_timer_op ends at offset 57, the
    // next member is 8-byte aligned), keeping sizeof(OnlineNode) unchanged
    // (locked by the R2 regression's static_asserts).
    bool metric_issue_anchor = false;
    bool metric_complete_anchor = false;
    std::string inputs_values;  // metadata pg info (issue_pytorch_pg_metadata)
    // Reverse index (online mode; static mode leaves them empty/0):
    std::string request_id;
    std::string stage;
    uint64_t generation = 0;

    ComputeAttrs compute;
    CommAttrs comm;
    CollAttrs coll;
    OnlineStatisticsState online_statistics;
};

/// Read-side view; same POD as OnlineNode (add_node stores it, for_each_
/// dep_free / lookup_ptr hand it out).
using NodeView = OnlineNode;

class GraphSource {
  public:
    virtual ~GraphSource() = default;

    /// The ONLY dependency-release entry. Releasing a node frees its
    /// children. Idempotent.
    virtual void finish_node(uint64_t node_id) = 0;

    /// Consume a node from the free set at issue time (required so a later
    /// issue pass cannot re-issue it).
    virtual void take_node(uint64_t node_id) = 0;

    /// Zero-copy issue pass over the currently free (dependency-satisfied,
    /// not yet issued) nodes, ascending global_id for deterministic issue
    /// order. Non-consuming: take_node marks a node as issued. The
    /// NodeStore-backed source iterates its stable storage (online nodes are
    /// collected only at quiescent commit boundaries -- never inside a
    /// callback -- so the references outlive the consume callback).
    virtual void for_each_dep_free(
        const std::function<void(const NodeView&)>& consume) = 0;

    /// Zero-copy lookup for the online terminal paths. Returns nullptr for
    /// an unknown node. Pure virtual: NodeStoreGraphSource (the only
    /// GraphSource subclass since the static ET route was removed) always
    /// overrides it, so the old "return nullptr" default body was dead.
    virtual const NodeView* lookup_ptr(uint64_t node_id) = 0;

    // Online-only mutable short-lived statistics state and terminal
    // single-fire guard. Pure virtual for the same reason: the sole
    // subclass overrides both (the former null/false defaults existed only
    // for the removed static ET adapters).
    virtual OnlineStatisticsState* mutable_online_statistics(
        uint64_t node_id) = 0;
    virtual bool mark_terminal_observed(uint64_t node_id) = 0;
};

}  // namespace ExecutionDriven
}  // namespace AstraSim

#endif  // EXECUTION_DRIVEN_GRAPHSOURCE_HH
