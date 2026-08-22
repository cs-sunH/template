/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#ifndef __WORKLOAD_HH__
#define __WORKLOAD_HH__

#include <memory>
#include <string>
#include <unordered_map>

#include "astra-sim/system/Callable.hh"
#include "astra-sim/system/CommunicatorGroup.hh"
#include "astra-sim/workload/execution_driven/ExecutionMode.hh"
#include "astra-sim/workload/execution_driven/GraphSource.hh"
#include "astra-sim/workload/HardwareResource.hh"
#include "astra-sim/workload/Statistics.hh"
#include "astra-sim/workload/LocalMemUsageTracker.hh"
#include "extern/graph_frontend/chakra/src/feeder_v3/et_feeder.h"

namespace spdlog {
class logger;
}  // namespace spdlog

namespace AstraSim {

class Sys;
class DataSet;
class LocalHbmBandwidthModel;
class WorkloadLayerHandlerData;

class Workload : public Callable {
  public:
    // execution_mode/graph_source: step-1-2 execution-mode factory
    // (ExecutionMode.hh). Static default keeps the legacy call sites and the
    // byte-for-byte static behavior unchanged.
    Workload(Sys* sys,
             std::string et_filename,
             std::string comm_group_filename,
             ExecutionDriven::ExecutionMode execution_mode =
                 ExecutionDriven::ExecutionMode::Static,
             std::shared_ptr<ExecutionDriven::GraphSource> graph_source =
                 nullptr);
    ~Workload();

    // communicator groups
    // Parse the user provided 'comm_group_filename' and extract the list of
    // communicator groups. Refer to the wiki for the format.
    void initialize_comm_groups(std::string comm_group_filename);
    void issue_pytorch_pg_metadata(const ExecutionDriven::NodeView& node);

    // event-based simulation. Step 1-4 (方案 §4 步骤 1-4 操作 4): every issue_*
    // consumes the NodeView read view from the GraphSource (依赖状态唯一所有
    // 者); the ETFeederNode handle is fetched through GraphSource::et_node
    // only where the static-path consumers (HardwareResource / Statistics /
    // local_mem tracker) still need it.
    void issue_dep_free_nodes();
    void issue(const ExecutionDriven::NodeView& node);
    void issue_metadata(const ExecutionDriven::NodeView& node);
    void issue_replay(const ExecutionDriven::NodeView& node);
    void issue_remote_mem(const ExecutionDriven::NodeView& node);
    // sh_3.0 fourth-slot MEM dispatch (HBM restore model; preserved object
    // #18): routed from issue() when node.mem.is_local_hbm_kv_restore.
    void issue_local_hbm_kv_restore(const ExecutionDriven::NodeView& node);
    void issue_comp(const ExecutionDriven::NodeView& node);
    void issue_comm(const ExecutionDriven::NodeView& node);
    void issue_coll_comm(const ExecutionDriven::NodeView& node);
    void issue_send_comm(const ExecutionDriven::NodeView& node);
    void issue_recv_comm(const ExecutionDriven::NodeView& node);
    void skip_invalid(const ExecutionDriven::NodeView& node);
    void call(EventType event, CallData* data);
    void fire();

    // stats
    void report();

    Chakra::ETFeeder* et_feeder;
    std::unordered_map<int, CommunicatorGroup*> comm_groups;
    HardwareResource* hw_resource;
    Sys* sys;
    Statistics* stats;
    std::unique_ptr<LocalMemUsageTracker> local_mem_usage_tracker;
    std::unique_ptr<LocalHbmBandwidthModel> local_hbm_bandwidth_model;
    std::unordered_map<int, uint64_t> collective_comm_node_id_map;
    std::unordered_map<int, DataSet*> collective_comm_wrapper_map;
    bool is_finished;

    // step-1-2 execution-mode factory state: online mode never constructs the
    // ETFeeder and never requires .et files; the dynamic GraphSource is
    // injected at Sys creation (NodeStore-backed implementation in step 1-4).
    ExecutionDriven::ExecutionMode execution_mode_;
    std::shared_ptr<ExecutionDriven::GraphSource> graph_source_;
    // Path-2 removal (2026-08-18): the replay-clock scope flag was deleted
    // with the replay route; strategy mode always keeps real physics.

  private:
    // R4-14: cached "workload" logger -- fetched once in the constructor;
    // the registry returns the same logger object per name for the process
    // lifetime, so member reuse is behavior-equivalent (and skips the
    // per-call registry mutex + map lookup on the per-node hot paths).
    std::shared_ptr<spdlog::logger> workload_logger_;

    // From the node view, find out the corresponding communicator group, and
    // return the pointer. If no communicator group is specified for this
    // node, return nullptr.
    CommunicatorGroup* extract_comm_group(const ExecutionDriven::NodeView& node);

    // ---------------------------------------------------------------
    // N-way HBM contention endpoint join (system key
    // "hbm-bandwidth-contention"). A charged p2p comm node completes on the
    // join of (network-side completion, local-HBM COMM_READ/COMM_WRITE job
    // completion); a charged pool MEM node completes on the join of
    // (remote-memory port transaction, local-HBM POOL_READ/POOL_WRITE job).
    // Implemented entirely inside Workload: each side carries its own
    // WorkloadLayerHandlerData; the join state stores the local-HBM-side
    // cookie so a delivery is side-classified by comparing against a live
    // join (R4-13: no wlhd-pointer map keys -- a side that never fires
    // must not leave a dangling key behind); the node terminal path runs
    // exactly once, after both sides fired (idempotent by node-id map
    // erasure). The network / remote-memory APIs are untouched.
    // ---------------------------------------------------------------
    enum class HbmJoinSide { NetworkPort, LocalHbm };
    struct HbmJoinState {
        uint64_t node_id = 0;
        EventType terminal_event = EventType::General;
        bool network_done = false;
        bool local_hbm_done = false;
        // Cookie of the local-HBM side (valid for the join's lifetime, from
        // begin_hbm_join until the double-arrival erases the pending entry).
        // Any other cookie reaching consume_hbm_join_event for this node id
        // is the network/port side.
        WorkloadLayerHandlerData* hbm_side_wlhd = nullptr;
    };
    // True when this rank must create endpoint HBM jobs for comm / pool
    // nodes (flag on, model alive; per-node opt-outs like hbm-charge=false
    // or zero bytes are checked by the callers).
    bool hbm_endpoint_charge_active() const;
    // Creates a fresh local-HBM-side wlhd for a joined node (returned; the
    // caller feeds it to the LocalHbmBandwidthModel issue_* call) and
    // registers the pending join state keyed by node id.
    WorkloadLayerHandlerData* begin_hbm_join(
        uint64_t node_id, EventType terminal_event);
    // Workload::call entry for wlhd-carrying events: returns true when the
    // event was one side of a pending join (already accounted; possibly the
    // terminal completion ran). The caller must then skip the normal
    // terminal path.
    bool consume_hbm_join_event(WorkloadLayerHandlerData* wlhd);
    // The shared wlhd-branch terminal body (release / stats / metrics /
    // node-terminal record / dependency release / static auto-advance).
    void finish_general_node(uint64_t node_id, EventType event);

    std::unordered_map<uint64_t, HbmJoinState> hbm_join_pending_;
    // Fail-closed fire-after guard (R8-7): identity of the most recently
    // fired join's local-HBM side (cookie + node id). The pending entry is
    // erased at fire so the map stays bounded by in-flight joins, so this
    // single-slot record is what an already-fired HBM-side cookie is
    // checked against (constant memory; a re-delivery arriving only after
    // further joins fired falls back to the ordinary terminal path as
    // before -- the same-side live-join guard below catches the
    // double-fire while the join is still pending).
    WorkloadLayerHandlerData* last_fired_hbm_side_wlhd_ = nullptr;
    uint64_t last_fired_hbm_join_node_ = 0;
};

}  // namespace AstraSim

#endif /* __WORKLOAD_HH__ */
