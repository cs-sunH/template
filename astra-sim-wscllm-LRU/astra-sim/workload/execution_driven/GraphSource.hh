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

/// Compute-node role (die backend, stage B; contract §2): "compute" is a
/// real compute op, "control_marker" a micro/sentinel node (first_token,
/// drain, exit, sentinel, join, pstart, ... -- the comp(name,1,1) class)
/// that must keep the legacy micro-job completion semantics instead of a
/// die plan. The JSON spellings are exactly "compute" | "control_marker".
enum class OpRole { Compute, ControlMarker };

/// Compute-node attributes (issue_comp / issue_replay / issue_remote_mem).
struct ComputeAttrs {
    uint64_t num_ops = 0;
    uint64_t tensor_size = 0;
    uint64_t runtime_ns = 0;  // replay runtime, duration_micros * 1000
    bool has_remote_weight_bytes = false;
    uint64_t remote_weight_bytes = 0;
    // Stage B (die backend): the node's dispatch role and, when present,
    // the strictly parsed immutable DieOpSpec. die_op_spec present implies
    // runtime_ns == 0 (parser-enforced, contract §2); a real
    // op_role=Compute node WITHOUT a die_spec fails closed in the die
    // backend dispatch (never silently falls back to roofline). The
    // shared_ptr keeps one immutable spec per node without copying the
    // JSON payload into every short-lived view (design doc §8.3).
    OpRole op_role = OpRole::Compute;
    std::shared_ptr<const DieModel::DieOpSpec> die_spec;
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
    // parsed bool_list; may be empty -- the adapter no longer fills any
    // default (the legacy 4x-true default fill moved to
    // Sys::generate_collective, which pads a short vector with true up to
    // the topology's dimension count, N14 parser contract; the commit
    // preflight additionally requires the exact declared dimension count).
    std::vector<bool> involved_dim;
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

    // Retained until terminal completion; only compute/memory utilization
    // is compacted globally.  (The former terminal-communication fact field
    // comm_size was removed: unlike face-LRU, no production writer ever
    // existed in this repository, so the field was write-only test
    // self-reference.)
    // vector_utilization is the die-backend addition (stage B, contract
    // §7.6): eq-ops / (vector peak * elapsed); unset for the legacy
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
    // sh_2.0/sh_3.0 specific (§4.3.3): MEM_LOAD/MEM_STORE nodes carrying this
    // attribute dispatch to issue_local_hbm_kv_restore (LocalHbmBandwidthModel
    // restore DMA, HardwareResource hbm_dma class) instead of the remote-mem
    // FIFO. The online GraphBatch carries the attribute explicitly.
    bool is_local_hbm_kv_restore = false;
    // sh_2.0 N-way HBM contention: MEM_LOAD/MEM_STORE endpoint charging mode
    // (attr "hbm-access-mode", 0/absent = none, 1 = local HBM read, 2 = local
    // HBM write, bytes = tensor_size). The port transaction and the HBM job
    // complete as a join (Workload hbm_endpoint_joins_).
    int hbm_access_mode = 0;
    // R2 MetricCollector anchor fast-path cache (sparse): set by
    // NodeStore::set_metric_anchor_flags when the dynamic anchor
    // registration path (GraphBatchCommitter Phase B-1.5) registers an
    // anchor for this (rank, node); false = no routing entry exists on
    // that edge, so Workload skips the two-level MetricCollector hash
    // lookup entirely.
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
    /// an unknown node. Default returns nullptr; the NodeStore-backed
    /// source overrides.
    virtual const NodeView* lookup_ptr(uint64_t /*node_id*/) { return nullptr; }

    // Online-only mutable short-lived statistics state and terminal
    // single-fire guard. The default implementations are the unsupported
    // "no online statistics" fallbacks (return nullptr / false); the
    // NodeStore-backed source overrides both. (2026-09-29 note: no static
    // source exists in the production tree any more -- the defaults stay
    // because the virtual interface is deliberately kept as the seam for
    // future source implementations and test doubles, not because a static
    // adapter consumes them.)
    virtual OnlineStatisticsState* mutable_online_statistics(
        uint64_t /*node_id*/) {
        return nullptr;
    }
    virtual bool mark_terminal_observed(uint64_t /*node_id*/) { return false; }
};

}  // namespace ExecutionDriven
}  // namespace AstraSim

#endif  // EXECUTION_DRIVEN_GRAPHSOURCE_HH
