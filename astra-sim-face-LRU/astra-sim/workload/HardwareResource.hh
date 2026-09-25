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
            this->num_in_flight_gpu_comp_ops != 0 ||
            this->num_in_flight_gpu_comm_ops != 0 ||
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
            logger->critical("remote MEM node id: {}", node_id);
        }
    }
    void occupy(const std::shared_ptr<Chakra::FeederV3::ETFeederNode> node);
    void release(const std::shared_ptr<Chakra::FeederV3::ETFeederNode> node);
    bool is_available(
        const std::shared_ptr<Chakra::FeederV3::ETFeederNode> node) const;
    // Online-mode overloads keyed by NodeView. Both overloads dispatch
    // through ONE shared semantic classification (SerDes 片外链路并发化改造
    // 执行方案 §4):
    //   is_timer_op                       -> no-op (no slot, always
    //                                        available);
    //   is_local_hbm_kv_restore           -> HBM-DMA dedicated slot
    //                                        (single slot, unchanged
    //                                        sh_2.0 semantics);
    //   MEM_LOAD / MEM_STORE (the rest)   -> remote MEM slot: independent,
    //                                        count-based, UNBOUNDED. It is
    //                                        a NODE-occupancy count, never
    //                                        a SerDes stream count: it must
    //                                        not feed any port concurrency
    //                                        denominator, busy_ns or
    //                                        bandwidth estimate, and
    //                                        is_available() never blocks on
    //                                        the in-flight count (issue
    //                                        gating for remote MEM lives
    //                                        nowhere -- every dep-free node
    //                                        issues immediately). A MEM
    //                                        node carrying
    //                                        "hbm-access-mode" releases
    //                                        only when the port leg AND
    //                                        the local HBM leg have both
    //                                        completed (Workload
    //                                        hbm_endpoint_joins_ latch),
    //                                        i.e. at whole-node
    //                                        termination; the release call
    //                                        arrives exactly once from
    //                                        Workload::call, so the count
    //                                        brackets the full node
    //                                        lifetime by construction.
    //                                        Remote MEM nodes occupy NO
    //                                        comm slot -- the old
    //                                        comm-single-slot hidden gate
    //                                        is gone from both paths.
    //   CPU / COMP / COMM_RECV / COMM     -> legacy slots, semantics
    //                                        unchanged (CPU and COMM stay
    //                                        single-slot; COMP is gated by
    //                                        is_available's in-flight==0
    //                                        check on BOTH paths, so
    //                                        independent COMP nodes issue
    //                                        serially online and static
    //                                        alike -- the counter is what
    //                                        that gate reads, not a
    //                                        concurrency switch; COMM_RECV
    //                                        is a no-op).
    //
    // Failure policy: release() pre-decrements checks (counter non-zero;
    // when node-id retention is enabled the node must be in its set) and
    // occupy() duplicate-id checks are std::abort() fatals -- deliberately
    // NOT assert(), so they survive NDEBUG builds. No remote MEM "tics"
    // accumulator exists on purpose: any future node-time metric must be an
    // end-to-end node time (including the HBM join leg), never a remote-port
    // service time.
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
    // per-resource counters here, including sh_2.0's HBM-DMA class and the
    // remote MEM node count.
    const bool retain_node_ids_;
    uint32_t num_in_flight_cpu_ops;
    uint32_t num_in_flight_gpu_comp_ops;
    uint32_t num_in_flight_gpu_comm_ops;
    uint32_t num_in_flight_hbm_dma_ops;
    uint32_t num_in_flight_remote_mem_ops;

    // Only counters with a live reader are kept: num_gpu_comms /
    // num_remote_mem_ops are consumed by remote_port_static_gate_test (in
    // build), tics_gpu_ops feeds the exposed-communication figure in
    // Workload::report. The former write-only totals (num_cpu_ops /
    // num_gpu_ops / num_hbm_dma_ops, tics_cpu_ops / tics_gpu_comms /
    // tics_hbm_dma_ops) had no reader after HardwareResource::report() was
    // removed and are deleted.
    uint64_t num_gpu_comms;
    uint64_t num_remote_mem_ops;

    uint64_t tics_gpu_ops;
};

}  // namespace AstraSim

#endif /* __HARDWARE_RESOURCE_HH__ */
