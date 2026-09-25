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
            this->num_in_flight_gpu_comp_ops != 0) {
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
    }
    void occupy(const std::shared_ptr<Chakra::FeederV3::ETFeederNode> node);
    void release(const std::shared_ptr<Chakra::FeederV3::ETFeederNode> node);
    bool is_available(
        const std::shared_ptr<Chakra::FeederV3::ETFeederNode> node) const;
    // Step 1-8: online-mode overloads keyed by NodeView (global_id). The
    // static ETFeederNode path above is untouched (byte-exact baseline);
    // the online path (GraphSource::et_node == nullptr) dispatches on the
    // NodeView fields: is_timer_op -> no-op, is_cpu_op -> CPU, kind==Compute
    // -> GPU comp, kind==CommRecv -> no-op, else -> GPU comm.
    void occupy(const ExecutionDriven::NodeView& node);
    void release(const ExecutionDriven::NodeView& node);
    bool is_available(const ExecutionDriven::NodeView& node) const;
    [[nodiscard]] bool tracks_node_ids() const {
        return retain_node_ids_;
    }

    std::unordered_set<uint64_t> cpu_ops_node;
    std::unordered_set<uint64_t> gpu_ops_node;
    std::unordered_set<uint64_t> gpu_comms_node;

    const int sys_id;

    // Static mode retains exact IDs for legacy destructor diagnostics. Online
    // mode already has NodeStore's fail-closed lifecycle and keeps only exact
    // in-flight counters here, avoiding hash work on the per-node hot path.
    const bool retain_node_ids_;
    uint32_t num_in_flight_cpu_ops;
    uint32_t num_in_flight_gpu_comp_ops;
    uint32_t num_in_flight_gpu_comm_ops;

    // Live accumulator: tics_gpu_ops feeds Workload::report's exposed-
    // communication line.  The write-only num_*/tics_cpu_ops/tics_gpu_comms
    // debug counters (and HardwareResource::report) were removed as dead,
    // and the write-only tics_hbm_dma_ops local-HBM occupancy counter joined
    // them (2026-09-25, workload-F4: zero readers repo-wide; the local-HBM
    // bandwidth model no longer accumulates it).
    uint64_t tics_gpu_ops;
};

}  // namespace AstraSim

#endif /* __HARDWARE_RESOURCE_HH__ */
