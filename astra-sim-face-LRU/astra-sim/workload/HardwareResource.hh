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
            this->num_in_flight_hbm_dma_ops != 0) {
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
    }
    void occupy(const std::shared_ptr<Chakra::FeederV3::ETFeederNode> node);
    void release(const std::shared_ptr<Chakra::FeederV3::ETFeederNode> node);
    bool is_available(
        const std::shared_ptr<Chakra::FeederV3::ETFeederNode> node) const;
    // Online-mode overloads keyed by NodeView (§4.3.3). The static
    // ETFeederNode path above is untouched (byte-exact baseline); the online
    // path (GraphSource::et_node == nullptr) dispatches on the NodeView
    // fields: is_timer_op -> no-op, MEM_LOAD/MEM_STORE with
    // is_local_hbm_kv_restore -> hbm_dma class (sh_2.0 fourth resource
    // class, single slot; count-based occupy/release like Compute -- the
    // replay bypass in Workload::issue_dep_free_nodes can issue several
    // concurrently), is_cpu_op -> CPU, CommRecv -> no-op, Compute -> GPU
    // comp, else -> GPU comm.
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

    const int sys_id;

    // Static mode retains exact IDs for legacy destructor diagnostics. Online
    // mode relies on NodeStore's fail-closed lifecycle and keeps only exact
    // per-resource counters here, including sh_2.0's HBM-DMA class.
    const bool retain_node_ids_;
    uint32_t num_in_flight_cpu_ops;
    uint32_t num_in_flight_gpu_comp_ops;
    uint32_t num_in_flight_gpu_comm_ops;
    uint32_t num_in_flight_hbm_dma_ops;

    uint64_t num_cpu_ops;
    uint64_t num_gpu_ops;
    uint64_t num_gpu_comms;
    uint64_t num_hbm_dma_ops;

    uint64_t tics_cpu_ops;
    uint64_t tics_gpu_ops;
    uint64_t tics_gpu_comms;
    uint64_t tics_hbm_dma_ops;
};

}  // namespace AstraSim

#endif /* __HARDWARE_RESOURCE_HH__ */
