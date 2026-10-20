/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/


#ifndef __HARDWARE_RESOURCE_HH__
#define __HARDWARE_RESOURCE_HH__

#include "astra-sim/common/Logging.hh"
#include <cstdint>

#include "astra-sim/workload/execution_driven/GraphSource.hh"

namespace AstraSim {

class HardwareResource {
  public:
    HardwareResource(int sys_id = -1);
    // Destructor diagnostics: an unreleased node at teardown is a lifecycle
    // violation, so report it fail-closed. (The former per-node ID dump loops
    // died with the diagnostic ID retention they served -- the retired static
    // ET path was the only retainer.)
    ~HardwareResource() {
        auto logger = LoggerFactory::get_logger("HardwareResource");
        if (this->num_in_flight_cpu_ops != 0 ||
            this->num_in_flight_gpu_comm_ops != 0 ||
            this->num_in_flight_gpu_comm_coll_ops != 0 ||
            this->num_in_flight_gpu_comp_ops != 0 ||
            this->num_in_flight_hbm_dma_ops != 0 ||
            this->num_in_flight_remote_mem_ops != 0) {
            logger->critical(
                "!!!Hardware Resource sys.id={} has unreleased nodes!!!",
                this->sys_id);
        }
    }
    // NodeView classification (方案 §4 发射门控改造), centralized in
    // HardwareResource.cc's classify_hw_resource():
    //   timer -> no-op;
    //   local HBM KV restore -> hbm_dma class (sh_2.0 fourth resource class,
    //     single slot, serialized through the is_available gate --
    //     Workload::issue_dep_free_nodes checks it before every issue; the
    //     former replay-only bypass was removed with the Path-2 replay
    //     route);
    //   every other MEM_LOAD/MEM_STORE remote node -> remote_mem class: an
    //     independent, counted, UNLIMITED node-occupancy slot; is_available
    //     never blocks on its in-flight count; nodes carrying
    //     "hbm-access-mode" release only when the whole node terminates
    //     after both the port and local-HBM legs (Workload hbm_endpoint_
    //     joins_ latch releases exactly once);
    //   then is_cpu_op -> CPU; Compute -> GPU comp; CommRecv -> no-op;
    //   everything else (COMM_SEND/COMM_COLL and the legacy metadata/
    //     invalid fall-through) -> the existing GPU comm slot with its
    //     unchanged single-slot occupancy semantics.
    // Remote MEM never occupies the comm slot, so MEM traffic does not
    // retain the former comm single-slot hidden gate. COMP stays a plain
    // count (concurrent COMP occupancy is a supported state, serialized in
    // production only through the is_available gate in
    // Workload::issue_dep_free_nodes).
    void occupy(const ExecutionDriven::NodeView& node);
    void release(const ExecutionDriven::NodeView& node);
    bool is_available(const ExecutionDriven::NodeView& node) const;

    const int sys_id;

    uint32_t num_in_flight_cpu_ops;
    uint32_t num_in_flight_gpu_comp_ops;
    uint32_t num_in_flight_gpu_comm_ops;
    // 死锁真修 ③-lite（2026-10-11）：集合通信在役计数（不占 gpu_comm
    // 单槽；观测/审计用，无容量语义）。
    uint32_t num_in_flight_gpu_comm_coll_ops;
    uint32_t num_in_flight_hbm_dma_ops;
    // 方案 §4: in-flight remote-MEM NODE occupancy count (remote
    // MEM_LOAD/MEM_STORE; "hbm-access-mode" endpoint nodes release only at
    // whole-node terminal). This is a node count, NOT a SerDes stream count:
    // it must never be used as a port concurrency denominator, busy_ns, or
    // pool-bandwidth estimate. The slot is unlimited and never blocks issue.
    uint32_t num_in_flight_remote_mem_ops;
};

}  // namespace AstraSim

#endif /* __HARDWARE_RESOURCE_HH__ */
