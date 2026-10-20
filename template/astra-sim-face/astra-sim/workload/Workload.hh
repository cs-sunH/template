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
#include "astra-sim/workload/LocalMemUsageTracker.hh"
#include "extern/graph_frontend/chakra/src/feeder_v3/et_feeder.h"

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
    void issue_comp(const ExecutionDriven::NodeView& node);
    void issue_comm(const ExecutionDriven::NodeView& node);
    void issue_coll_comm(const ExecutionDriven::NodeView& node);
    void issue_send_comm(const ExecutionDriven::NodeView& node);
    void issue_recv_comm(const ExecutionDriven::NodeView& node);
    void skip_invalid(const ExecutionDriven::NodeView& node);
    void call(EventType event, CallData* data);

    // stats
    void report();

    Chakra::ETFeeder* et_feeder;
    std::unordered_map<int, std::shared_ptr<CommunicatorGroup>> comm_groups;
    HardwareResource* hw_resource;
    Sys* sys;
    Statistics* stats;
    std::unique_ptr<LocalMemUsageTracker> local_mem_usage_tracker;
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
    bool is_finished;

    // step-1-2 execution-mode factory state: online mode never constructs the
    // ETFeeder and never requires .et files; the dynamic GraphSource is
    // injected at Sys creation (NodeStore-backed implementation in step 1-4).
    ExecutionDriven::ExecutionMode execution_mode_;
    std::shared_ptr<ExecutionDriven::GraphSource> graph_source_;
    // 2.5-1 fast path: set when the injected online source is
    // NodeStore-backed (GraphSource::as_node_store_source); the terminal
    // and issue paths then reuse ONE NodeStore find per node. Non-owning;
    // null in static mode / for any other source -> by-id virtuals.
    ExecutionDriven::NodeStoreGraphSource* node_store_source_ = nullptr;

    // LocalHbmBandwidthModel completion callback (re-entrant, fired from the
    // model's transition event): COMP jobs take the plain generic completion
    // path; COMM_READ/COMM_WRITE jobs mark the HBM side of the joined comm
    // node and complete it only when the network side already arrived.  Takes
    // ownership of wlhd.
    void on_local_hbm_job_complete(WorkloadLayerHandlerData* wlhd,
                                   LocalHbmBandwidthModel::JobKind kind);

    // Body shared by every generic (wlhd) node completion: node release /
    // stats / metrics / terminal record / dependency release / static-mode
    // auto-advance.  Extracted from Workload::call's generic branch so the
    // LocalHbmBandwidthModel callback can complete COMP nodes through the
    // exact same path.  (The completion-event parameter was removed: the
    // network-bandwidth statistics it used to feed were deleted as
    // write-only dead state.)  Public since stage B: the DieExecution of a
    // die-backend COMP node finishes through this same single exit (contract
    // §7.3; the HBM-delivered completions already ran it from
    // on_local_hbm_job_complete).
    void finish_generic_node(uint64_t node_id);
    void finish_cancelled_coarse_node(uint64_t node_id);
    uint64_t die_cancelled_comp_count() const {
        return die_cancelled_comp_count_;
    }

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

    // Prepared coarse COMP seam. The caller prepares before acquiring a
    // rank-local workspace lease; this entry validates ownership before the
    // node can be taken or its GPU slot occupied.
    void issue_prepared_coarse(
        const ExecutionDriven::NodeView& node,
        DieModelAdapter::PreparedCoarsePlan prepared,
        WorkspaceAdmission::Lease lease);

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

    // Static-run final end authority (report / sim_notify_finished /
    // is_finished), extracted from Workload::call's tail so the
    // LocalHbmBandwidthModel-delivered completions (which finish nodes
    // through finish_generic_node without re-entering Workload::call) also
    // finalize the run. Idempotent via the is_finished guard; a no-op in
    // online mode (the ServiceCoordinator owns the online end authority).
    void maybe_finalize_static_run();

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

    // Static auto-advance re-entrancy guard: issue() can synchronously
    // complete nodes (zero-byte collectives complete inside set_notifier;
    // the HBM model harvests finished jobs inside issue_compute), and every
    // completion re-enters issue_dep_free_nodes from inside its own
    // for_each_dep_free snapshot iteration. The nested call must not take
    // nodes out of the stale outer snapshot (the outer loop would revisit
    // them and double-take); it only flags a rerun, and the outermost call
    // re-snapshots until a full pass completes without synchronous
    // completions. Online mode never nests (its completion paths are
    // Static-gated), so the guard is idle there.
    bool issuing_dep_free_ = false;
    bool issue_dep_free_rerun_needed_ = false;
};

}  // namespace AstraSim

#endif /* __WORKLOAD_HH__ */
