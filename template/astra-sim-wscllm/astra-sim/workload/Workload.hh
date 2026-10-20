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

    // event-based simulation. Step 1-4 (方案 §4 步骤 1-4 操作 4): every issue_*
    // consumes the NodeView read view from the GraphSource (依赖状态唯一所有
    // 者).
    void issue_dep_free_nodes();
    void issue(const ExecutionDriven::NodeView& node);
    /// Coarse v2 only: issue an already prepared Plan after its rank-local
    /// workspace lease has been acquired before take_node/occupy.
    void issue_prepared_coarse(
        const ExecutionDriven::NodeView& node,
        DieModelAdapter::PreparedCoarsePlan prepared,
        WorkspaceAdmission::Lease lease);
    void issue_metadata(const ExecutionDriven::NodeView& node);
    void issue_replay(const ExecutionDriven::NodeView& node);
    void issue_comp(const ExecutionDriven::NodeView& node);
    void issue_comm(const ExecutionDriven::NodeView& node);
    void issue_coll_comm(const ExecutionDriven::NodeView& node);
    void issue_send_comm(const ExecutionDriven::NodeView& node);
    void issue_recv_comm(const ExecutionDriven::NodeView& node);
    void skip_invalid(const ExecutionDriven::NodeView& node);
    void call(EventType event, CallData* data);

    std::unordered_map<int, std::shared_ptr<CommunicatorGroup>> comm_groups;
    HardwareResource* hw_resource;
    Sys* sys;
    Statistics* stats;
    // Multi-user local-HBM bandwidth contention (hbm-bandwidth-contention):
    // one fluid model per rank; null in the legacy closed-form mode.
    std::unique_ptr<LocalHbmBandwidthModel> local_hbm_bandwidth_model;
    // Die backend: one adapter per rank, constructed only under
    // compute-cost-model=die (its constructor fail-closes without the Sys
    // die runtime). DECLARED AFTER local_hbm_bandwidth_model on purpose:
    // members destruct in reverse order, so the adapter (and its live
    // DieExecutions, which must drop their pending HBM die jobs) unwinds
    // while the fluid model is still alive; ~Workload additionally shuts
    // it down FIRST, before the model reset, for the same reason.
    std::unique_ptr<DieModelAdapter> die_adapter_;
    std::unordered_map<int, uint64_t> collective_comm_node_id_map;
    std::unordered_map<int, DataSet*> collective_comm_wrapper_map;

    // step-1-2 execution-mode factory state: Workload never constructs an
    // ETFeeder (the offline static path is retired) and never requires .et
    // files; the dynamic GraphSource is injected at Sys creation
    // (NodeStore-backed implementation in step 1-4). The execution mode
    // itself is constructor-local: Online is the only accepted value and no
    // code reads it afterwards.
    std::shared_ptr<ExecutionDriven::GraphSource> graph_source_;
    // 2.5-1 fast path: set when the injected online source is
    // NodeStore-backed (GraphSource::as_node_store_source); the terminal
    // and issue paths then reuse ONE NodeStore find per node. Non-owning;
    // null in static mode / for any other source -> by-id virtuals.
    ExecutionDriven::NodeStoreGraphSource* node_store_source_ = nullptr;
    // Path-2 removal (2026-08-18): the replay-clock scope flag was deleted
    // with the replay-only dispatch route (--online-mode replay; distinct
    // from the issue_replay operator issue path, which remains); strategy
    // mode always keeps real physics.

    // LocalHbmBandwidthModel completion callback (re-entrant, fired from the
    // model's transition event): COMP jobs take the plain generic completion
    // path; COMM_READ/COMM_WRITE jobs mark the HBM side of the joined comm
    // node and complete it only when the network side already arrived.  Takes
    // ownership of wlhd.
    void on_local_hbm_job_complete(WorkloadLayerHandlerData* wlhd,
                                   LocalHbmBandwidthModel::JobKind kind);

    /// Teardown-only companion of on_local_hbm_job_complete: when the
    /// LocalHbmBandwidthModel is destroyed with jobs still in flight (fatal
    /// exit path / test harness dismantling Sys mid-run), the destructor
    /// deletes the residual payloads itself and drops the pending
    /// HbmCommJoin entry of the joined comm node -- that node can never
    /// complete through a join whose endpoint job will never fire. Absent
    /// entry (COMP jobs, or the network side already consumed the join) is
    /// a silent no-op.
    void abandon_local_hbm_join_entry(uint64_t node_id);

    /// Body shared by every generic (wlhd) node completion: node release /
    /// stats / metrics / terminal record / dependency release. Extracted
    /// from Workload::call's generic branch so the LocalHbmBandwidthModel
    /// callback can complete COMP nodes through the exact same path.
    /// Public since the die backend: the DieExecution of a die-backend
    /// COMP node finishes through this same single exit (contract §7.3;
    /// the HBM-delivered completions already ran it from
    /// on_local_hbm_job_complete).
    void finish_generic_node(uint64_t node_id);
    /// Terminal path for an explicitly cancelled live coarse COMP. It
    /// releases the occupied slot and dependencies once with Skipped status.
    void finish_cancelled_coarse_node(uint64_t node_id);
    /// Count live coarse COMP cancellations for rank/service metric sideband.
    /// Shutdown cancellation is intentionally excluded.
    uint64_t die_cancelled_comp_count() const {
        return die_cancelled_comp_count_;
    }

    /// Die backend (contract §7.6): fill the utilization fields of the
    /// node's statistics state ahead of finish_generic_node -- compact
    /// OnlineStatisticsState in online service mode (including the
    /// die-only vector_utilization), the legacy OperatorStatistics record
    /// otherwise. Must run BEFORE the generic completion consumes them.
    void record_die_utilizations(uint64_t node_id,
                                 double compute_utilization,
                                 double memory_utilization,
                                 double vector_utilization);

    /// Die backend adapter accessor (null off the die backend).
    DieModelAdapter* die_model_adapter() {
        return die_adapter_.get();
    }

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
    // 2.5-1: pre_resolved (default null) carries the terminal/issue-path
    // handle's statistics slot / record; null = resolve by id as before.
    void start_online_statistics(
        const ExecutionDriven::NodeView& node, Tick start_time,
        ExecutionDriven::OnlineStatisticsState* pre_resolved = nullptr);
    void complete_online_statistics(
        const ExecutionDriven::NodeView& node, Tick end_time,
        ExecutionDriven::OnlineStatisticsState* pre_resolved = nullptr);
    void mark_online_terminal_or_fail(
        uint64_t node_id,
        const ExecutionDriven::OnlineRecordRef* pre_resolved = nullptr);

    // Die-backend COMP dispatch (contract §7.3 frozen order): timer -> cpu
    // -> control_marker legacy micro-job -> die_spec via DieModelAdapter ->
    // fail-closed throw. Reached only under
    // sys->compute_cost_model == ComputeCostModel::Die.
    void issue_die_comp(
        const ExecutionDriven::NodeView& node,
        DieModelAdapter::PreparedCoarsePlan* prepared = nullptr,
        WorkspaceAdmission::Lease* lease = nullptr);
    void issue_impl(
        const ExecutionDriven::NodeView& node,
        DieModelAdapter::PreparedCoarsePlan* prepared,
        WorkspaceAdmission::Lease* lease);

    // Two-event join for p2p comm nodes with an HBM endpoint job: the node
    // completes exactly once, when both the network-side callback and the
    // endpoint HBM job have fired (two pending flags; the entry is erased at
    // the single fire, which also guards against re-entry).
    struct HbmCommJoin {
        bool network_done = false;
        bool hbm_done = false;
    };
    std::unordered_map<uint64_t, HbmCommJoin> hbm_comm_join_;
    uint64_t die_cancelled_comp_count_ = 0;
    void maybe_complete_hbm_joined_comm(uint64_t node_id);
};

}  // namespace AstraSim

#endif /* __WORKLOAD_HH__ */
