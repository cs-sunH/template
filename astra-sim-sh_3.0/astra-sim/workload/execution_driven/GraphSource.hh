/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

GraphSource -- execution-driven mechanism layer (sh30 (wscllm-blueprint) phase 1).

Dynamic graph source abstraction (方案 §4 步骤 1-2 操作 4 / 步骤 1-4 操作 4;
动态图边界 contract ①). In online mode the source is injected at Sys
creation time so the ETFeeder is never constructed and no .et file is
required; in static mode Workload constructs the ETFeederGraphSource
wrapping the ETFeeder + DependancyResolver.

**依赖状态唯一所有者 = GraphSource**: Workload calls GraphSource::finish_node
exactly once per node; CompletionObserver/watch only record facts and never
release dependencies a second time (步骤 1-5 操作 3).

**在线模式不得自动发射后继**: dep_free_nodes() in online mode yields nodes
only when the post-commit deferred path drains them into the source; the
static auto-advance (issue_dep_free_nodes after every completion) must not
leak into the online mode -- the online mode gate lives in Workload::call.

NodeView / OnlineNode share one POD: add_node stores it, dep_free_nodes()
/ lookup() return it as the read view. Fields are the full transitive
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

/// Compute-node attributes (issue_comp / issue_replay / issue_remote_mem).
struct ComputeAttrs {
    uint64_t num_ops = 0;
    uint64_t tensor_size = 0;
    uint64_t runtime_ns = 0;  // replay runtime, duration_micros * 1000
    bool has_remote_weight_bytes = false;
    uint64_t remote_weight_bytes = 0;
};

/// Memory-node attributes (issue_remote_mem / issue_local_hbm_kv_restore).
/// sh_3.0 (wscllm-blueprint had no remote memory): is_local_hbm_kv_restore
/// routes a MEM_LOAD node to the LocalHbmBandwidthModel restore path and the
/// HardwareResource hbm_dma slot instead of the remote-memory FIFO.
struct MemAttrs {
    uint64_t tensor_size = 0;
    bool is_local_hbm_kv_restore = false;
};

/// Point-to-point comm attributes (issue_send_comm / issue_recv_comm).
struct CommAttrs {
    uint64_t bytes = 0;
    int src = 0;
    int dst = 0;
    uint32_t tag = 0;
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

/// The node record / read view. In static mode the ETFeederGraphSource
/// fills every field from the ETFeederNode; in online mode the NodeStore
/// fills them from GraphBatch data, including the reverse index
/// request_id / stage / generation.
struct OnlineNode {
    uint64_t global_id = 0;
    int rank = 0;
    NodeKind kind = NodeKind::Invalid;
    uint64_t node_type = 0;  // raw ChakraProtoMsg::NodeType value
    std::string name;
    bool is_cpu_op = false;
    bool is_timer_op = false;
    std::string inputs_values;  // metadata pg info (issue_pytorch_pg_metadata)
    // Reverse index (online mode; static mode leaves them empty/0):
    std::string request_id;
    std::string stage;
    uint64_t generation = 0;

    ComputeAttrs compute;
    MemAttrs mem;
    CommAttrs comm;
    CollAttrs coll;
};

/// Read-side view; same POD as OnlineNode (add_node stores it, dep_free_nodes
/// / lookup return it).
using NodeView = OnlineNode;

class GraphSource {
  public:
    virtual ~GraphSource() = default;

    /// Currently free (dependency-satisfied, not yet issued) nodes, sorted by
    /// global_id for deterministic issue order. NOT consuming: take_node
    /// marks a node as issued.
    virtual std::vector<NodeView> dep_free_nodes() = 0;

    /// The ONLY dependency-release entry. Releasing a node frees its
    /// children. Idempotent.
    virtual void finish_node(uint64_t node_id) = 0;

    /// Stable async-callback handle; the NodeView must stay valid for the
    /// consumers until the terminal record is done.
    virtual std::optional<NodeView> lookup(uint64_t node_id) = 0;

    /// Consume a node from the free set at issue time (static resolver
    /// semantic preserved; required so dep_free_nodes cannot re-issue).
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
};

/// Empty source: yields no nodes, finishes nothing. Used as the step-1-2
/// placeholder before the NodeStore-backed source landed (step 1-4).
class EmptyGraphSource : public GraphSource {
  public:
    std::vector<NodeView> dep_free_nodes() override { return {}; }
    void finish_node(uint64_t) override {}
    std::optional<NodeView> lookup(uint64_t) override { return std::nullopt; }
    void take_node(uint64_t) override {}
    std::shared_ptr<Chakra::FeederV3::ETFeederNode> et_node(
        uint64_t) override {
        return nullptr;
    }
    bool static_all_done() override { return false; }
};

}  // namespace ExecutionDriven
}  // namespace AstraSim

#endif  // EXECUTION_DRIVEN_GRAPHSOURCE_HH
