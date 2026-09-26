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
is_cpu_op), MetricCollector (rank + node_id only), local_mem tracker
(ETFeederNode-bound in static mode, via GraphSource::et_node), comm
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

namespace Chakra {
namespace FeederV3 {
class ETFeederNode;
}  // namespace FeederV3
}  // namespace Chakra

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

/// Compute-node attributes (issue_comp / issue_replay).
struct ComputeAttrs {
    uint64_t num_ops = 0;
    uint64_t tensor_size = 0;
    uint64_t runtime_ns = 0;  // replay runtime, duration_micros * 1000
    bool has_remote_weight_bytes = false;
    uint64_t remote_weight_bytes = 0;
};

/// Point-to-point comm attributes (issue_send_comm / issue_recv_comm).
struct CommAttrs {
    uint64_t bytes = 0;
    int src = 0;
    int dst = 0;
    uint32_t tag = 0;
    // ET attribute "hbm-charge" (bool, default true): false = this endpoint
    // creates no local-HBM job under hbm-bandwidth-contention (e.g. a pure
    // control/signal transfer). All wscllm comm nodes are real data
    // endpoints, so the generator never writes false; the field exists so
    // the C++ side can honor an explicit opt-out.
    bool hbm_charge = true;
};

/// Collective comm attributes (issue_coll_comm).
struct CollAttrs {
    uint64_t comm_type = 0;  // ChakraProtoMsg::CollectiveCommType value
    uint64_t bytes = 0;
    uint32_t priority = 0;
    std::string pg_name;
    std::vector<bool> involved_dim;  // parsed bool_list; adapter fills the
                                     // legacy 4x-true default when absent
};

// Online-only, short-lived statistics state.  It lives in the NodeStore
// record rather than Statistics::operator_statistics, so service execution
// can retire terminal nodes without a global per-node statistics hash-table
// operation.  Static ET adapters leave this POD at its defaults.
struct OnlineStatisticsState {
    static constexpr uint64_t kInvalidTick = UINT64_MAX;

    uint64_t start_time = kInvalidTick;
    uint64_t end_time = kInvalidTick;
    bool started = false;
    bool completed = false;
    bool is_gpu = false;

    // Short-lived per-node roofline/comm statistics state, filled by the
    // online Workload while the node executes and kept until terminal
    // completion.  Only compute/memory utilization is compacted globally.
    std::optional<double> memory_utilization;
    std::optional<double> compute_utilization;
    std::optional<uint64_t> comm_size;
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

    /// Static-compatibility handle for the ETFeederNode-bound consumers
    /// (HardwareResource / Statistics / local_mem tracker). nullptr in
    /// online mode; those consumers gain NodeView overloads in later steps
    /// when the online path actually executes nodes.
    virtual std::shared_ptr<Chakra::FeederV3::ETFeederNode> et_node(
        uint64_t node_id) = 0;

    /// Static finish check: dependency resolver empty AND no ongoing nodes.
    /// False in online mode (the end authority belongs exclusively to the
    /// ServiceCoordinator).
    virtual bool static_all_done() = 0;

    /// Zero-copy issue pass over the currently free (dependency-satisfied,
    /// not yet issued) nodes, ascending global_id for deterministic issue
    /// order. Non-consuming: take_node marks a node as issued. The
    /// NodeStore-backed source iterates its stable storage (online nodes are
    /// collected only at quiescent commit boundaries -- never inside a
    /// callback -- so the references outlive the consume callback).
    virtual void for_each_dep_free(
        const std::function<void(const NodeView&)>& consume) = 0;

    /// Zero-copy lookup for the online terminal paths. Returns nullptr for
    /// an unknown node. Default returns nullptr; the NodeStore-backed
    /// source overrides.
    virtual const NodeView* lookup_ptr(uint64_t /*node_id*/) { return nullptr; }

    // Online-only mutable short-lived statistics state and terminal
    // single-fire guard.  Static sources deliberately use the default
    // unsupported implementations, preserving their ET behavior.
    virtual OnlineStatisticsState* mutable_online_statistics(
        uint64_t /*node_id*/) {
        return nullptr;
    }
    virtual bool mark_terminal_observed(uint64_t /*node_id*/) { return false; }
};

}  // namespace ExecutionDriven
}  // namespace AstraSim

#endif  // EXECUTION_DRIVEN_GRAPHSOURCE_HH
