/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

// TODO: HardwareResource.hh should be moved to the system layer.

#ifndef __HARDWARE_RESOURCE_HH__
#define __HARDWARE_RESOURCE_HH__

#include "astra-sim/common/Logging.hh"
#include <cstdint>

#include "extern/graph_frontend/chakra/src/feeder_v3/et_feeder.h"
#include "astra-sim/workload/execution_driven/ExecutionMode.hh"
#include "astra-sim/workload/execution_driven/GraphSource.hh"

namespace AstraSim {

class HardwareResource {
  public:
    HardwareResource(
        int sys_id = -1,
        ExecutionDriven::ExecutionMode execution_mode =
            ExecutionDriven::ExecutionMode::Static);
    ~HardwareResource() {
        auto logger = LoggerFactory::get_logger("HardwareResource");
        if (this->num_in_flight_cpu_ops != 0 ||
            this->num_in_flight_gpu_comm_ops != 0 ||
            this->num_in_flight_gpu_comp_ops != 0 ||
            this->num_in_flight_hbm_dma_ops != 0 ||
            this->num_in_flight_remote_mem_ops != 0) {
            logger->critical(
                "!!!Hardware Resource sys.id={} has unreleased nodes!!!",
                this->sys_id);
        }
        for (auto node_id : cpu_ops_node) {
            logger->critical("CPU node id: {}", node_id);
        }
        for (auto node_id : gpu_ops_node) {
            logger->critical("GPU comp node id: {}", node_id);
        }
        for (auto node_id : gpu_comms_node) {
            logger->critical("GPU comm node id: {}", node_id);
        }
        for (auto node_id : hbm_dma_ops_node) {
            logger->critical("HBM DMA node id: {}", node_id);
        }
        for (auto node_id : remote_mem_ops_node) {
            logger->critical("Remote MEM node id: {}", node_id);
        }
    }
    void occupy(const std::shared_ptr<Chakra::FeederV3::ETFeederNode> node);
    void release(const std::shared_ptr<Chakra::FeederV3::ETFeederNode> node);
    bool is_available(
        const std::shared_ptr<Chakra::FeederV3::ETFeederNode> node) const;
    // Both overload sets (static ETFeederNode and online NodeView) share ONE
    // semantic classification, centralized in HardwareResource.cc's
    // classify_hw_resource() (SerDes 片外链路并发化改造执行方案 §4 发射门控改造;
    // PARTIAL 跨实例 copy 流水化 2026-09-25 放宽 hbm_dma/comm 两类):
    //   timer -> no-op;
    //   local HBM KV restore -> hbm_dma class (sh_3.0 fourth resource
    //     class): a counted, UNLIMITED node-occupancy slot (same contract
    //     as remote_mem) -- is_available never blocks on its in-flight
    //     count, so multiple KV restores may be in flight at once; HBM
    //     bandwidth arbitration belongs to LocalHbmBandwidthModel's N-way
    //     equal split over all active jobs, never to a slot;
    //   every other MEM_LOAD/MEM_STORE remote node -> remote_mem class: an
    //     independent, counted, UNLIMITED node-occupancy slot; is_available
    //     never blocks on its in-flight count; nodes carrying
    //     "hbm-access-mode" release only when the whole node terminates
    //     after both the port and local-HBM legs (Workload hbm_endpoint_
    //     joins_ latch releases exactly once);
    //   then is_cpu_op -> CPU; Compute -> GPU comp; CommRecv -> no-op;
    //   everything else (COMM_SEND/COMM_COLL and the legacy metadata/
    //     invalid fall-through) -> the gpu_comm class: also a counted,
    //     UNLIMITED node-occupancy slot (2026-09-25) -- multiple sends,
    //     acks and collectives may be in flight at once; link arbitration
    //     belongs to the FluidScheduler's per-link N-way equal split.
    // Remote MEM never occupies the comm slot. The counters below exist
    // only for drain detection (Workload static-finish gate), the
    // release-precondition checks, and destructor diagnostics -- they are
    // node-occupancy counts, never bandwidth denominators. COMP keeps its
    // is_available single-slot gate (unchanged scope): concurrent online
    // COMP occupancy remains serialized only through the is_available
    // gate in Workload::issue_dep_free_nodes.
    void occupy(const ExecutionDriven::NodeView& node);
    void release(const ExecutionDriven::NodeView& node);
    bool is_available(const ExecutionDriven::NodeView& node) const;
    [[nodiscard]] bool tracks_node_ids() const {
        return retain_node_ids_;
    }

    std::unordered_set<uint64_t> cpu_ops_node;
    std::unordered_set<uint64_t> gpu_ops_node;
    std::unordered_set<uint64_t> gpu_comms_node;
    std::unordered_set<uint64_t> hbm_dma_ops_node;
    std::unordered_set<uint64_t> remote_mem_ops_node;

    const int sys_id;

    // Static mode retains exact IDs for legacy destructor diagnostics. Online
    // mode relies on NodeStore's fail-closed lifecycle and keeps only exact
    // per-resource counters here, including sh_3.0's HBM-DMA class and the
    // remote-MEM class.
    const bool retain_node_ids_;
    uint32_t num_in_flight_cpu_ops;
    uint32_t num_in_flight_gpu_comp_ops;
    // PARTIAL 跨实例 copy 流水化（2026-09-25）: comm and hbm_dma are
    // counted, UNLIMITED node-occupancy slots (same contract as the
    // remote-MEM class below). The counts feed the static-finish gate,
    // the release-precondition checks, and destructor diagnostics only.
    uint32_t num_in_flight_gpu_comm_ops;
    uint32_t num_in_flight_hbm_dma_ops;
    // 方案 §4: in-flight remote-MEM NODE occupancy count (remote
    // MEM_LOAD/MEM_STORE; "hbm-access-mode" endpoint nodes release only at
    // whole-node terminal). This is a node count, NOT a SerDes stream count:
    // it must never be used as a port concurrency denominator, busy_ns, or
    // pool-bandwidth estimate. The slot is unlimited and never blocks issue.
    uint32_t num_in_flight_remote_mem_ops;

    // Busy-time accumulator read by Workload::report (exposed-communication
    // metric); the former per-class tics/num counters were write-only
    // (HardwareResource::report was dead) and were removed. Deliberately NO
    // remote-MEM tics: any port service metric must come from the backend
    // fluid/ledger state (方案 §4).
    uint64_t tics_gpu_ops;
};

}  // namespace AstraSim

#endif /* __HARDWARE_RESOURCE_HH__ */
