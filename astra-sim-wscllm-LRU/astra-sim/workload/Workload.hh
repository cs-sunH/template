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
#include "astra-sim/workload/DieModelAdapter.hh"
#include "astra-sim/workload/HardwareResource.hh"
#include "astra-sim/workload/LocalHbmBandwidthModel.hh"
#include "astra-sim/workload/Statistics.hh"

namespace spdlog {
class logger;
}  // namespace spdlog

namespace AstraSim {

class Sys;
class DataSet;
class WorkloadLayerHandlerData;

class Workload : public Callable {
  public:
    // execution_mode/graph_source: step-1-2 execution-mode factory
    // (ExecutionMode.hh). The offline static path is retired in this
    // repository (the ETFeederGraphSource adapter was removed with it);
    // Online is the only constructible mode.
    Workload(Sys* sys,
             std::string et_filename,
             std::string comm_group_filename,
             ExecutionDriven::ExecutionMode execution_mode =
                 ExecutionDriven::ExecutionMode::Online,
             std::shared_ptr<ExecutionDriven::GraphSource> graph_source =
                 nullptr);
    ~Workload();

    // communicator groups
    // Parse the user provided 'comm_group_filename' and extract the list of
    // communicator groups. Refer to the wiki for the format.
    void initialize_comm_groups(std::string comm_group_filename);
    void issue_pytorch_pg_metadata(const ExecutionDriven::NodeView& node);

    // Teardown-only（2026-10-01 对齐）：丢弃该节点挂起的 Hbm 端点 join
    // 表项——不投递任何完成、不触发任何指标锚点（进程已在
    // fatal/teardown 路径上）。由 ~LocalHbmBandwidthModel 的残余作业
    // drain 调用。
    void abandon_hbm_endpoint_join(uint64_t node_id);

    // event-based simulation. Step 1-4 (方案 §4 步骤 1-4 操作 4): every issue_*
    // consumes the NodeView read view from the GraphSource (依赖状态唯一所有
    // 者).
    void issue_dep_free_nodes();
    void issue(const ExecutionDriven::NodeView& node);
    void issue_metadata(const ExecutionDriven::NodeView& node);
    void issue_replay(const ExecutionDriven::NodeView& node);
    void issue_remote_mem(const ExecutionDriven::NodeView& node);
    void issue_local_hbm_kv_restore(const ExecutionDriven::NodeView& node);
    void issue_comp(const ExecutionDriven::NodeView& node);
    void issue_comm(const ExecutionDriven::NodeView& node);
    void issue_coll_comm(const ExecutionDriven::NodeView& node);
    void issue_send_comm(const ExecutionDriven::NodeView& node);
    void issue_recv_comm(const ExecutionDriven::NodeView& node);
    void skip_invalid(const ExecutionDriven::NodeView& node);
    void call(EventType event, CallData* data);

    // Body shared by every generic (wlhd) node completion: node release /
    // stats / metrics / terminal record / dependency release.  Extracted
    // from Workload::call's generic branch so the LocalHbmBandwidthModel
    // callback can complete COMP nodes through the exact same path.  Public
    // since stage B: the DieExecution of a die-backend COMP node finishes
    // through this same single exit (contract §7.3; the HBM-delivered
    // completions already ran it from Workload::call's join tail).
    void finish_generic_node(uint64_t node_id);

    // Die backend (stage B, contract §7.6): fill the utilization fields of
    // the node's statistics state ahead of finish_generic_node -- compact
    // OnlineStatisticsState in online service mode (including the
    // die-only vector_utilization), the legacy OperatorStatistics record
    // otherwise. Must run BEFORE the generic completion consumes them.
    void record_die_utilizations(uint64_t node_id,
                                 double compute_utilization,
                                 double memory_utilization,
                                 double vector_utilization);

    // Die backend adapter accessor (stage B; null off the die backend).
    DieModelAdapter* die_model_adapter() {
        return die_adapter_.get();
    }

    // DeadEnd-census diagnostic only (2026-10-11): number of pending
    // remote-MEM endpoint join latches (each latch's two completion legs
    // are still in flight). Read-only; never called on the hot path.
    size_t hbm_endpoint_joins_count() const {
        return hbm_endpoint_joins_.size();
    }
    // ... and the sum of their remaining legs (2 -> both pending, 1 -> one
    // leg arrived, 0 would mean erased -- reported for completeness).
    size_t hbm_endpoint_joins_pending_legs() const {
        size_t pending = 0;
        for (const auto& entry : hbm_endpoint_joins_) {
            pending += entry.second.pending_completions;
        }
        return pending;
    }

    std::unordered_map<int, std::shared_ptr<CommunicatorGroup>> comm_groups;
    HardwareResource* hw_resource;
    Sys* sys;
    Statistics* stats;
    // Multi-user local-HBM bandwidth contention (hbm-bandwidth-contention):
    // one fluid model per rank; null in the legacy closed-form mode.
    std::unique_ptr<LocalHbmBandwidthModel> local_hbm_bandwidth_model;
    // Die backend (stage B): one adapter per rank, constructed only under
    // compute-cost-model=die (its constructor fail-closes without the Sys
    // die runtime). DECLARED AFTER local_hbm_bandwidth_model on purpose:
    // members destruct in reverse order, so the adapter (and its live
    // DieExecutions, which must drop their pending HBM die jobs) unwinds
    // while the fluid model is still alive.
    std::unique_ptr<DieModelAdapter> die_adapter_;
    std::unordered_map<int, uint64_t> collective_comm_node_id_map;
    std::unordered_map<int, DataSet*> collective_comm_wrapper_map;

    // step-1-2 execution-mode factory state: Workload never constructs an
    // ETFeeder (the offline static path is retired) and never requires .et
    // files; the dynamic GraphSource is injected at Sys creation
    // (NodeStore-backed implementation in step 1-4). The execution mode is
    // validated in the constructor (only Online is constructible) and not
    // retained -- the former execution_mode_ member was write-only.
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
    // return its shared owner. If no communicator group is specified for this
    // node, return nullptr. A collective DataSet retains this owner so an
    // in-flight definition survives later metadata replacement.
    std::shared_ptr<CommunicatorGroup> extract_comm_group(
        const ExecutionDriven::NodeView& node);

    // Compact online Statistics keeps its per-node transient data in the
    // NodeStore record.  These helpers centralize the no-operator-map service
    // path and the terminal single-fire guard; static ET callers never enter
    // them.
    ExecutionDriven::OnlineStatisticsState&
    online_statistics_state_or_fail(uint64_t node_id);
    void start_online_statistics(const ExecutionDriven::NodeView& node,
                                 Tick start_time);
    void complete_online_statistics(const ExecutionDriven::NodeView& node,
                                    Tick end_time);
    void mark_online_terminal_or_fail(uint64_t node_id);
    // record_network_bandwidth 删除：comm-size → 带宽整条链无生产读者，
    // 各调用点算出的耗时全部被空实现丢弃（对齐 face-LRU 姊妹仓）。

    // Die-backend COMP dispatch (contract §7.3 frozen order): timer -> cpu
    // -> control_marker legacy micro-job -> die_spec via DieModelAdapter ->
    // fail-closed. Reached only under
    // sys->compute_cost_model == ComputeCostModel::Die.
    void issue_die_comp(const ExecutionDriven::NodeView& node);

    // sh_2.0 N-way HBM contention: endpoint join state for nodes whose
    // completion is the JOIN of two independent async completions -- the
    // network/port side (fluid scheduler packet event for comm send/recv,
    // AnalyticalRemoteMemory fluid-port transition event for pool MEM
    // nodes) AND the local HBM job in LocalHbmBandwidthModel. Cookie
    // ownership differs by join kind (方案 §3.4): a p2p comm join shares
    // ONE wlhd across both legs and the latch-opening arrival deletes it,
    // while a pool MEM join gives each leg its OWN cookie -- each arrival
    // delivers Workload::call(General, wlhd) with its own leg's cookie and
    // deletes it on that own arrival. Either way the latch counts down
    // (2 -> 1 -> 0, idempotent per arrival) and only the arrival that
    // reaches 0 runs the terminal handling (exactly once). Keyed by node
    // id (unique per rank workload).
    struct HbmEndpointJoinState {
        unsigned int pending_completions = 2;
        // 网络侧完成事件（face hbm_comm_join_.completion_event 同款）：
        // 闩锁开闸后的终态带宽门用此存储事件，后到者（HBM 侧 General）
        // 不再使 p2p 带宽统计静默缺失（中-4①，2026-08-20）。
        EventType completion_event = EventType::General;
        // Fail-closed double-fire flags (R8-7): each side must arrive
        // exactly once; a same-side second arrival is a model-layer
        // mechanism violation and exits instead of opening the latch early
        // (2 -> 1 -> 0 while the other side is still pending). Side
        // attribution happens in Workload::call (by event type).
        bool network_done = false;
        bool hbm_done = false;
    };
    std::unordered_map<uint64_t, HbmEndpointJoinState> hbm_endpoint_joins_;
};

}  // namespace AstraSim

#endif /* __WORKLOAD_HH__ */
