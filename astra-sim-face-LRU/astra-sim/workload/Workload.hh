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

    // 方案 §3.4/阶段2（并发化改造）normal-end 审计：存在未开的 HBM
    // endpoint join（端口/网络腿与本地 HBM 腿未全部完成）时，online main
    // 必须报错而非静默清理。只读；join cookie 始终归 Workload 所有，与
    // remote backend 的 wlhd 是两个东西。
    [[nodiscard]] bool has_unfinished_hbm_endpoint_joins() const {
        return !hbm_endpoint_joins_.empty();
    }

    Chakra::ETFeeder* et_feeder;
    std::unordered_map<int, std::shared_ptr<CommunicatorGroup>> comm_groups;
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

    // sh_2.0 N-way HBM contention: endpoint join state for nodes whose
    // completion is the JOIN of two independent async completions -- the
    // network/port side (fluid scheduler packet event for comm send/recv,
    // AnalyticalRemoteMemory FIFO event for pool MEM nodes) AND the local
    // HBM job in LocalHbmBandwidthModel. Both sides deliver
    // Workload::call(General|PacketSent|PacketReceived, wlhd) with the SAME
    // wlhd; the latch counts down (2 -> 1 -> 0, idempotent per arrival) and
    // only the arrival that reaches 0 runs the terminal handling (exactly
    // once). Keyed by node id (unique per rank workload).
    struct HbmEndpointJoinState {
        unsigned int pending_completions = 2;
        // 网络侧完成事件（face hbm_comm_join_.completion_event 同款）：
        // 记录 comm endpoint 的网络腿事件类型，供 call() 的双发侧归属
        // 判定使用（completion_event != General 的节点上，General 到达即
        // 为本地 HBM 腿）。其原有的"终态 p2p 带宽统计"消费链已随
        // record_network_bandwidth 删除（整条链无生产读者）。
        EventType completion_event = EventType::General;
        // Fail-closed double-fire flags (R8-7): comm send/recv endpoints
        // (and pool MEM nodes) each get one guarded arrival per side -- a
        // duplicate packet-side or packet-joined HBM-side arrival exits
        // instead of opening the latch early. CAVEAT: a pool MEM node's two
        // legs BOTH deliver General and are indistinguishable by event type,
        // so for them only the latch count is guarded -- a same-side second
        // General arrival decrements the count and may open the latch before
        // both legs are done, with no exit (Workload::call side-attribution
        // carve-out).
        bool network_done = false;
        bool hbm_done = false;
    };
    std::unordered_map<uint64_t, HbmEndpointJoinState> hbm_endpoint_joins_;
};

}  // namespace AstraSim

#endif /* __WORKLOAD_HH__ */
