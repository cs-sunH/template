/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#ifndef __WORKLOAD_HH__
#define __WORKLOAD_HH__

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "astra-sim/system/Callable.hh"
#include "astra-sim/system/CommunicatorGroup.hh"
#include "astra-sim/workload/execution_driven/ExecutionMode.hh"
#include "astra-sim/workload/execution_driven/GraphSource.hh"
#include "astra-sim/workload/DieModelAdapter.hh"
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
namespace ExecutionDriven {
class PipelineKernelContract;
}

class Workload : public Callable {
  public:
    struct DepFreeIssueTiming {
        uint64_t snapshot_ns = 0;
        uint64_t issue_ns = 0;
        uint64_t snapshot_passes = 0;
        uint64_t snapshot_full_fills = 0;
        uint64_t snapshot_reuse_hits = 0;
        // v11 Phase 4 (issue-pass increment, 方案 §6.1): snapshots rebuilt
        // by splicing the NodeStore's since-snapshot delta onto the cached
        // vector instead of rewalking the whole ordered set. Same result,
        // lower cost; reported separately so full fills stay honest.
        uint64_t snapshot_splices = 0;
        uint64_t candidates = 0;
        uint64_t issue_calls = 0;
        uint64_t node_view_materializations = 0;  // ID-snapshot value copies
    };

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
    void set_pipeline_kernel_contract(
        std::shared_ptr<const ExecutionDriven::PipelineKernelContract> contract);
    [[nodiscard]] std::shared_ptr<const ExecutionDriven::PipelineKernelContract>
    pipeline_kernel_contract() const;
    using LedgerTransitionObserver = std::function<void(
        const std::string& kind, const ExecutionDriven::NodeView& node)>;
    void set_ledger_transition_observer(LedgerTransitionObserver observer) {
        ledger_transition_observer_ = std::move(observer);
    }
    [[nodiscard]] bool requires_pipeline_kernel_contract() const {
        return execution_mode_ == ExecutionDriven::ExecutionMode::Online;
    }
    [[nodiscard]] const DepFreeIssueTiming& dep_free_issue_timing() const {
        return dep_free_issue_timing_;
    }
    void issue(const ExecutionDriven::NodeView& node);
    /// Issue coarse v2 after the same Plan passed rank-local workspace
    /// admission while the node was still free.
    void issue_prepared_coarse(
        const ExecutionDriven::NodeView& node,
        DieModelAdapter::PreparedCoarsePlan prepared,
        WorkspaceAdmission::Lease lease);
    void issue_metadata(const ExecutionDriven::NodeView& node);
    void issue_replay(const ExecutionDriven::NodeView& node);
    void issue_remote_mem(const ExecutionDriven::NodeView& node);
    // sh_3.0 fourth-slot MEM dispatch (HBM restore model; preserved object
    // #18): routed from issue() when node.mem.is_local_hbm_kv_restore.
    void issue_local_hbm_kv_restore(const ExecutionDriven::NodeView& node);
    void issue_comp(const ExecutionDriven::NodeView& node);
    // Die-backend COMP dispatch (contract §7.3 frozen order): timer -> cpu
    // -> control_marker legacy micro-job -> die_spec via DieModelAdapter ->
    // fail-closed. Reached only under
    // sys->compute_cost_model == ComputeCostModel::Die.
    void issue_die_comp(
        const ExecutionDriven::NodeView& node,
        DieModelAdapter::PreparedCoarsePlan* prepared = nullptr,
        WorkspaceAdmission::Lease* lease = nullptr);
    void issue_comm(const ExecutionDriven::NodeView& node);
    void issue_coll_comm(const ExecutionDriven::NodeView& node);
    void issue_send_comm(const ExecutionDriven::NodeView& node);
    void issue_recv_comm(const ExecutionDriven::NodeView& node);
    void skip_invalid(const ExecutionDriven::NodeView& node);
    void call(EventType event, CallData* data);

    // Diagnostic-only dispatch label (v7 event attribution); see Callable.
    [[nodiscard]] const char* profile_dispatch_label() const noexcept override {
        return "workload_call";
    }

    void fire();

    // stats
    void report();

    // The shared wlhd-branch terminal body (release / stats / metrics /
    // node-terminal record / dependency release / static auto-advance).
    // Public since the die backend rollout (stage 1): the DieExecution of a
    // die-backend COMP node finishes through this same single exit (contract
    // §7.3) -- this repository's shared body takes the triggering event
    // alongside the node id (the LocalHbmBandwidthModel completions reach it
    // through Workload::call / the N-way join with EventType::General).
    void finish_general_node(uint64_t node_id, EventType event);

    // Die backend (joint rollout stage 1, contract §7.6): fill the
    // utilization fields of the node's statistics state ahead of
    // finish_general_node -- compact OnlineStatisticsState in online
    // service mode (including the die-only vector_utilization), the legacy
    // OperatorStatistics record otherwise. Must run BEFORE the generic
    // completion consumes them.
    void record_die_utilizations(uint64_t node_id,
                                 double compute_utilization,
                                 double memory_utilization,
                                 double vector_utilization);

    // Die backend adapter accessor (rollout stage 1; null off the die
    // backend).
    DieModelAdapter* die_model_adapter() {
        return die_adapter_.get();
    }

    // 方案 §3.4 normal-end audit: number of nodes with an armed (half-fired)
    // HBM endpoint join. Zero at a normal end; > 0 means a node whose port /
    // network leg or local-HBM leg never fired -- the online main gate must
    // report it instead of cleaning up silently. (The HBM-side cookie inside
    // each pending join is owned by Workload, never by the remote backend.)
    [[nodiscard]] uint64_t hbm_join_pending_count() const {
        return hbm_join_pending_.size();
    }
    /// Teardown-only: forget an endpoint join whose Runtime job was cancelled
    /// without notification while Workload is being dismantled.
    void abandon_hbm_endpoint_join(uint64_t node_id);

    Chakra::ETFeeder* et_feeder;
    std::unordered_map<int, std::shared_ptr<CommunicatorGroup>> comm_groups;
    HardwareResource* hw_resource;
    Sys* sys;
    Statistics* stats;
    std::unique_ptr<LocalMemUsageTracker> local_mem_usage_tracker;
    std::unique_ptr<LocalHbmBandwidthModel> local_hbm_bandwidth_model;
    // Die backend (rollout stage 1): one adapter per rank, constructed only
    // under compute-cost-model=die (its constructor fail-closes without the
    // Sys die runtime). DECLARED AFTER local_hbm_bandwidth_model on purpose:
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
    std::shared_ptr<const ExecutionDriven::PipelineKernelContract>
        pipeline_kernel_contract_;
    LedgerTransitionObserver ledger_transition_observer_;
    // Path-2 removal (2026-08-18): the replay-clock scope flag was deleted
    // with the replay route; strategy mode always keeps real physics.

  private:
    friend struct WorkloadTestAccess;

    void issue_impl(
        const ExecutionDriven::NodeView& node,
        DieModelAdapter::PreparedCoarsePlan* prepared,
        WorkspaceAdmission::Lease* lease);

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
    void publish_ledger_transition(
        const char* kind, const ExecutionDriven::NodeView& node) const;

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

    std::unordered_map<uint64_t, HbmJoinState> hbm_join_pending_;
    // Static auto-advance re-entrancy guard (H1/R1-C1 port, 2026-09-30):
    // issue() can synchronously complete nodes (zero-byte collectives fire
    // set_notifier inline; the HBM model harvests finished jobs inside its
    // issue path), and every completion re-enters issue_dep_free_nodes from
    // inside its own for_each_dep_free snapshot iteration. The nested call
    // must not take nodes out of the stale outer snapshot (the outer loop
    // would revisit them and double-take); it only flags a rerun, and the
    // outermost call re-snapshots until a full pass completes without
    // synchronous completions. Online mode never nests (its completion
    // paths are Static-gated), so the guard is idle there.
    bool issuing_dep_free_ = false;
    bool issue_dep_free_rerun_needed_ = false;
    // Reused online ready-batch snapshot. Its complete ascending contents are
    // captured before issue() can run a synchronous completion callback; such
    // callbacks may add ready ids, which belong to the next pass. Across
    // separate calls the vector is reused only while its NodeStore source and
    // ready-set generation still match.
    std::vector<uint64_t> dep_free_issue_snapshot_ids_;
    std::vector<ExecutionDriven::NodeView> dep_free_issue_snapshot_views_;
    // The ID vector remains a snapshot of the NodeStore's ordered free set.
    // Reuse it only while both its source identity and ready-set generation
    // still match; UINT64_MAX deliberately disables reuse after generation
    // saturation.
    std::weak_ptr<ExecutionDriven::GraphSource>
        dep_free_issue_snapshot_source_;
    uint64_t dep_free_issue_snapshot_version_ = 0;
    bool dep_free_issue_snapshot_valid_ = false;
    DepFreeIssueTiming dep_free_issue_timing_;
    // Fail-closed fire-after guard (R8-7): identity of the most recently
    // fired join's local-HBM side (cookie value + node id). The pending entry
    // is erased at fire so the map stays bounded by in-flight joins, so this
    // single-slot record is what an already-fired HBM-side cookie is
    // checked against (constant memory; a re-delivery arriving only after
    // further joins fired falls back to the ordinary terminal path as
    // before -- the same-side live-join guard below catches the
    // double-fire while the join is still pending).
    // Store the callback cookie as an integer after the HBM-side object is
    // delivered and deleted. Retaining a pointer to that object would leave
    // a dangling pointer in the fire-after guard; a later allocation could
    // reuse the address and turn an unrelated callback into a false
    // double-fire match.
    std::uintptr_t last_fired_hbm_side_cookie_ = 0;
    uint64_t last_fired_hbm_join_node_ = 0;
};

}  // namespace AstraSim

#endif /* __WORKLOAD_HH__ */
