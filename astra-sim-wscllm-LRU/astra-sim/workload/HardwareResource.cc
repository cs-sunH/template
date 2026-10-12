/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/


#include "astra-sim/workload/HardwareResource.hh"

#include <cstdlib>

#include "extern/graph_frontend/chakra/src/feeder_v3/et_feeder.h"

using namespace std;
using namespace AstraSim;

typedef ChakraProtoMsg::NodeType ChakraNodeType;

namespace {

// ---------------------------------------------------------------------------
// 方案 §4 发射门控改造: ONE semantic classification for the NodeView
// overloads.
//
// kind_value is the node kind in the shared 0..7 value space:
// static_cast<uint64_t>(node.kind) (ExecutionDriven::NodeKind,
// GraphSource.hh), which mirrors the raw ChakraProtoMsg::NodeType values the
// committed GraphBatch carries.
enum class HwResClass {
    NoOp,       // timer nodes; COMM_RECV endpoints (existing semantics)
    HbmDma,     // local HBM KV restore DMA (sh_2.0 fourth class, single slot)
    RemoteMem,  // remote MEM_LOAD/MEM_STORE: counted, UNLIMITED node slot
    Cpu,
    GpuComp,
    GpuComm,  // COMM_SEND + legacy metadata/invalid fall-through
    GpuCommColl,  // COMM_COLL（all_reduce/屏障）: 死锁真修 ③-lite
                  // (2026-10-11)——集合通信不再占 gpu_comm 单槽等成员：
                  // 等成员/交换期间不阻塞同 rank 的 p2p（四方环闭合成分
                  // 之三拆除）。计数观测、无限额、available 恒真；同实例
                  // 集合间仍由 per-rank 图 frontier 串行（体节点链位）。
};

HwResClass classify_hw_resource(const uint64_t kind_value,
                                const bool is_cpu_op,
                                const bool is_timer_op,
                                const bool is_local_hbm_kv_restore) {
    if (is_timer_op) {
        return HwResClass::NoOp;
    }
    if (is_local_hbm_kv_restore) {
        return HwResClass::HbmDma;
    }
    // Every remaining MEM_LOAD/MEM_STORE node is remote-port traffic
    // (optionally joined with a local-HBM endpoint job via hbm-access-mode).
    // It gets the dedicated remote-MEM slot instead of the comm slot: the
    // former fall-through that parked these nodes on the single comm slot
    // was the comm single-slot hidden gate and is gone from both paths.
    if (kind_value ==
            static_cast<uint64_t>(ChakraNodeType::MEM_LOAD_NODE) ||
        kind_value ==
            static_cast<uint64_t>(ChakraNodeType::MEM_STORE_NODE)) {
        return HwResClass::RemoteMem;
    }
    if (is_cpu_op) {
        return HwResClass::Cpu;
    }
    if (kind_value == static_cast<uint64_t>(ChakraNodeType::COMP_NODE)) {
        return HwResClass::GpuComp;
    }
    if (kind_value ==
        static_cast<uint64_t>(ChakraNodeType::COMM_RECV_NODE)) {
        return HwResClass::NoOp;
    }
    // 死锁真修 ③-lite（2026-10-11）：COMM_COLL 独立成类、不占 gpu_comm
    // 单槽（见 HwResClass::GpuCommColl 注记）。须在 GpuComm 兜底前。
    if (kind_value ==
        static_cast<uint64_t>(ChakraNodeType::COMM_COLL_NODE)) {
        return HwResClass::GpuCommColl;
    }
    return HwResClass::GpuComm;
}

const char* hw_res_class_label(const HwResClass cls) {
    switch (cls) {
        case HwResClass::NoOp:
            return "noop";
        case HwResClass::HbmDma:
            return "hbm_dma";
        case HwResClass::RemoteMem:
            return "remote_mem";
        case HwResClass::Cpu:
            return "cpu";
        case HwResClass::GpuComp:
            return "gpu_comp";
        case HwResClass::GpuComm:
            return "gpu_comm";
        case HwResClass::GpuCommColl:
            return "gpu_comm_coll";
    }
    return "unknown";
}

// release-side pre-decrement check (方案 §4, NDEBUG-independent): the class
// counter must be non-zero -- a release without a matching occupy is a
// lifecycle violation.
void check_release_preconditions(const uint32_t counter,
                                 const HwResClass cls,
                                 const int sys_id,
                                 const uint64_t node_id) {
    if (counter == 0) {
        LoggerFactory::get_logger("HardwareResource")
            ->critical("HardwareResource release underflow (class={}): "
                       "sys.id={} node={}",
                       hw_res_class_label(cls), sys_id, node_id);
        std::abort();
    }
}

// occupy-side single-slot pre-increment check (NDEBUG-independent, same
// discipline as check_release_preconditions below): a second concurrent
// occupation of a single-slot class means the is_available gate in
// Workload::issue_dep_free_nodes was bypassed -- a lifecycle violation that
// must abort, not assert (Sys::call_events swallows std::exception, so the
// accounting corruption has to fail closed here).
[[noreturn]] void occupy_class_fatal(const char* what,
                                     const int sys_id,
                                     const uint64_t node_id) {
    LoggerFactory::get_logger("HardwareResource")
        ->critical("HardwareResource fatal: {} (sys.id={} node={})", what,
                   sys_id, node_id);
    std::abort();
}

void occupy_class(HardwareResource& hr,
                  const HwResClass cls,
                  const uint64_t node_id) {
    switch (cls) {
        case HwResClass::NoOp:
            return;
        case HwResClass::HbmDma:
            if (hr.num_in_flight_hbm_dma_ops != 0) {
                occupy_class_fatal("HBM-DMA single slot busy on occupy",
                                   hr.sys_id, node_id);
            }
            ++hr.num_in_flight_hbm_dma_ops;
            return;
        case HwResClass::RemoteMem:
            // 方案 §4: independent counted slot -- no capacity limit and no
            // single-slot assert; arbitrarily many remote MEM nodes may be
            // in flight at once.
            ++hr.num_in_flight_remote_mem_ops;
            return;
        case HwResClass::Cpu:
            if (hr.num_in_flight_cpu_ops != 0) {
                occupy_class_fatal("CPU single slot busy on occupy", hr.sys_id,
                                   node_id);
            }
            ++hr.num_in_flight_cpu_ops;
            return;
        case HwResClass::GpuComp:
            // Plain count (two concurrently occupied COMP nodes are a
            // supported state, serialized in production only by the
            // is_available gate in Workload::issue_dep_free_nodes).
            ++hr.num_in_flight_gpu_comp_ops;
            return;
        case HwResClass::GpuCommColl:
            // 死锁真修 ③-lite（2026-10-11）：集合不占 gpu_comm 单槽等
            // 成员——计数观测（无单槽断言、无容量门），available 恒真。
            ++hr.num_in_flight_gpu_comm_coll_ops;
            return;
        case HwResClass::GpuComm:
            // Existing comm single-slot occupancy semantics, unchanged.
            if (hr.num_in_flight_gpu_comm_ops != 0) {
                occupy_class_fatal("GPU comm single slot busy on occupy",
                                   hr.sys_id, node_id);
            }
            ++hr.num_in_flight_gpu_comm_ops;
            return;
    }
}

void release_class(HardwareResource& hr,
                   const HwResClass cls,
                   const uint64_t node_id) {
    switch (cls) {
        case HwResClass::NoOp:
            return;
        case HwResClass::HbmDma:
            check_release_preconditions(hr.num_in_flight_hbm_dma_ops, cls,
                                        hr.sys_id, node_id);
            --hr.num_in_flight_hbm_dma_ops;
            assert(hr.num_in_flight_hbm_dma_ops == 0);
            return;
        case HwResClass::RemoteMem:
            check_release_preconditions(hr.num_in_flight_remote_mem_ops, cls,
                                        hr.sys_id, node_id);
            --hr.num_in_flight_remote_mem_ops;
            return;
        case HwResClass::Cpu:
            check_release_preconditions(hr.num_in_flight_cpu_ops, cls,
                                        hr.sys_id, node_id);
            --hr.num_in_flight_cpu_ops;
            assert(hr.num_in_flight_cpu_ops == 0);
            return;
        case HwResClass::GpuComp:
            check_release_preconditions(hr.num_in_flight_gpu_comp_ops, cls,
                                        hr.sys_id, node_id);
            --hr.num_in_flight_gpu_comp_ops;
            return;
        case HwResClass::GpuCommColl:
            check_release_preconditions(
                hr.num_in_flight_gpu_comm_coll_ops, cls, hr.sys_id, node_id);
            --hr.num_in_flight_gpu_comm_coll_ops;
            return;
        case HwResClass::GpuComm:
            check_release_preconditions(hr.num_in_flight_gpu_comm_ops, cls,
                                        hr.sys_id, node_id);
            --hr.num_in_flight_gpu_comm_ops;
            assert(hr.num_in_flight_gpu_comm_ops == 0);
            return;
    }
}

bool available_class(const HardwareResource& hr, const HwResClass cls) {
    switch (cls) {
        case HwResClass::NoOp:
            return true;
        case HwResClass::RemoteMem:
            // 方案 §4: the remote-MEM slot never blocks issue on its current
            // in-flight count (unlimited capacity). This gate must not
            // become a hidden serialization point for remote traffic.
            return true;
        case HwResClass::HbmDma:
            return hr.num_in_flight_hbm_dma_ops == 0;
        case HwResClass::Cpu:
            return hr.num_in_flight_cpu_ops == 0;
        case HwResClass::GpuComp:
            return hr.num_in_flight_gpu_comp_ops == 0;
        case HwResClass::GpuCommColl:
            // ③-lite：集合永不因槽位阻塞 issue（等成员不占槽）。
            return true;
        case HwResClass::GpuComm:
            return hr.num_in_flight_gpu_comm_ops == 0;
    }
    return true;  // unreachable; every enumerator handled above
}

}  // namespace

HardwareResource::HardwareResource(int sys_id)
    : sys_id(sys_id),
      num_in_flight_cpu_ops(0),
      num_in_flight_gpu_comp_ops(0),
      num_in_flight_gpu_comm_ops(0),
      num_in_flight_gpu_comm_coll_ops(0),
      num_in_flight_hbm_dma_ops(0),
      num_in_flight_remote_mem_ops(0) {}

// ---------------------------------------------------------- NodeView (§4.3.3)

void HardwareResource::occupy(const ExecutionDriven::NodeView& node) {
    occupy_class(*this,
                 classify_hw_resource(static_cast<uint64_t>(node.kind),
                                      node.is_cpu_op,
                                      node.is_timer_op,
                                      node.is_local_hbm_kv_restore),
                 node.global_id);
}

void HardwareResource::release(const ExecutionDriven::NodeView& node) {
    release_class(*this,
                  classify_hw_resource(static_cast<uint64_t>(node.kind),
                                       node.is_cpu_op,
                                       node.is_timer_op,
                                       node.is_local_hbm_kv_restore),
                  node.global_id);
}

bool HardwareResource::is_available(
    const ExecutionDriven::NodeView& node) const {
    return available_class(
        *this,
        classify_hw_resource(static_cast<uint64_t>(node.kind),
                             node.is_cpu_op,
                             node.is_timer_op,
                             node.is_local_hbm_kv_restore));
}
