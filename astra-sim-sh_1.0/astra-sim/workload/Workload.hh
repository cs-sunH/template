/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#ifndef __WORKLOAD_HH__
#define __WORKLOAD_HH__

#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "astra-sim/system/Callable.hh"
#include "astra-sim/system/CommunicatorGroup.hh"
#include "astra-sim/workload/execution_driven/ExecutionMode.hh"
#include "astra-sim/workload/execution_driven/GraphSource.hh"
#include "astra-sim/workload/HardwareResource.hh"
#include "astra-sim/workload/Statistics.hh"
#include "astra-sim/workload/LocalMemUsageTracker.hh"
#include "astra-sim/workload/LocalHbmBandwidthModel.hh"
#include "extern/graph_frontend/chakra/src/feeder_v3/et_feeder.h"

namespace AstraSim {

class Sys;
class DataSet;

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
    // Multi-user local-HBM contention model (sys->hbm_bandwidth_contention);
    // null when the flag is off -> every issue path below keeps the legacy
    // single-owner timing byte-for-byte.
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
    // From the node view, find out the corresponding communicator group, and
    // return the pointer. If no communicator group is specified for this
    // node, return nullptr.
    CommunicatorGroup* extract_comm_group(const ExecutionDriven::NodeView& node);

    // -----------------------------------------------------------------
    // Local-HBM contention node-completion join (hbm-bandwidth-contention).
    //
    // A COMM_SEND/COMM_RECV endpoint with HBM charge and a MEM_LOAD/
    // MEM_STORE node with hbm-access-mode completes only when BOTH its
    // network-side event (PacketSent/PacketReceived / the remote-port
    // transaction callback) AND its local-HBM endpoint job have fired. The
    // join is implemented inside Workload (two pending flags + idempotent
    // fire); no network API signature changes.
    struct HbmNodeJoin {
        bool network_done = false;
        bool hbm_done = false;
        // Whichever side arrived first holds the terminal handler data
        // alive until the join fires; the terminal sequence runs with it.
        WorkloadLayerHandlerData* held_wlhd = nullptr;
        EventType held_event = EventType::General;
        // The HBM-side handler data (is_local_hbm_job), awaiting deletion
        // at join fire.
        WorkloadLayerHandlerData* hbm_wlhd = nullptr;
    };
    std::unordered_map<uint64_t, HbmNodeJoin> pending_hbm_joins_;

    /// Register the two-sided completion for a node; called at issue time,
    /// before the network/port side can possibly fire.
    void arm_hbm_join(uint64_t node_id);
    /// Allocate the HBM endpoint job's handler data (is_local_hbm_job).
    WorkloadLayerHandlerData* make_hbm_job_wlhd(uint64_t node_id);
    /// Generic wlhd terminal sequence (release / record / finish_node /
    /// auto-advance); does NOT delete the wlhd -- the caller owns it.
    void run_wlhd_terminal(WorkloadLayerHandlerData* wlhd, EventType event);
    /// Join bookkeeping for one wlhd arrival (network/port or HBM side).
    /// Returns true when the arrival was consumed by a pending join (the
    /// caller must not run the terminal sequence itself).
    bool try_join_hbm_node(WorkloadLayerHandlerData* wlhd, EventType event);
};

namespace ExecutionDriven {

// ---------------------------------------------------------------------------
// Phase-4 sensing (方案 §6.2 操作 2 / contract ⑥): remote-memory FIFO
// 实账本层 -- sh_1.0's distinguishing ledger layer (the blueprint repos hold
// an "n/a" placeholder here; this repo executes MEM_LOAD/MEM_STORE traffic
// through the real AnalyticalRemoteMemory 26-port FIFO).
//
// Observation without touching the red-line AnalyticalRemoteMemory files
// (实录: 两文件零改动): every real-FIFO request enters through
// Workload::issue_remote_mem -> issue() and completes through the generic
// wlhd terminal branch of Workload::call (the port-FIFO completion callback
// registers the workload's own event back). Per port (= per npu-id under
// PER_NPU_MEMORY_EXPANSION, the configured architecture here), the counters
// give the exact FIFO state because the port FIFO is a strict single server:
//   in_flight  = issued - completed   (== active + pending)
//   active     = (in_flight > 0) ? 1 : 0
//   pending    = in_flight - active
// (server is busy iff any started request is unfinished; every issue either
// starts or enqueues, every completion either starts the next or idles the
// server -- AnalyticalRemoteMemory::issue/call, read-only analysis).
// Query/audit data only; no simulation semantics are touched anywhere.
// Single-threaded event loop -> no locking.
// ---------------------------------------------------------------------------
class RemoteFifoLedger {
  public:
    struct PortCounters {
        uint64_t issued_count = 0;
        uint64_t issued_bytes = 0;
        uint64_t completed_count = 0;
        uint64_t completed_bytes = 0;
        uint64_t peak_in_flight_count = 0;   // max active+pending requests
        uint64_t peak_in_flight_bytes = 0;   // max active+pending bytes
    };

    static RemoteFifoLedger& instance();

    void record_issue(int sys_id, uint64_t tensor_size);
    void record_completion(int sys_id, uint64_t tensor_size);

    /// Ports with any activity, sorted by sys_id (deterministic dump order).
    std::vector<int> active_ports() const;
    const PortCounters* port(int sys_id) const;
    uint64_t total_issued_count() const;
    uint64_t total_issued_bytes() const;
    uint64_t total_completed_count() const;
    uint64_t total_completed_bytes() const;
    /// True iff every port drained (issued == completed per port). A run that
    /// ends undrained lost a completion -- the caller fails closed on it.
    bool drained() const;

  private:
    RemoteFifoLedger() = default;
    std::map<int, PortCounters> ports_;
};

}  // namespace ExecutionDriven

}  // namespace AstraSim

#endif /* __WORKLOAD_HH__ */
