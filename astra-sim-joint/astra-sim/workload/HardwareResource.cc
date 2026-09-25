/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

// TODO: HardwareResource.cc should be moved to the system layer.

#include "astra-sim/workload/HardwareResource.hh"

#include <cstdlib>

using namespace std;
using namespace AstraSim;
using namespace Chakra;

typedef ChakraProtoMsg::NodeType ChakraNodeType;

namespace {

// sh_3.0: static ET nodes carry the KV-restore flag as an ET attribute; the
// GraphSource adapters copy the same value into the NodeView field
// (NodeStore.cc / ParsedGraphBatch.cc -> OnlineNode::mem), so both overload
// sets classify on identical inputs.
bool is_local_hbm_kv_restore(
    const shared_ptr<Chakra::FeederV3::ETFeederNode>& node) {
    return node->get_attr<bool>("is_local_hbm_kv_restore", false);
}

// ---------------------------------------------------------------------------
// 方案 §4 发射门控改造: ONE semantic classification shared verbatim by both
// overload sets (static ETFeederNode and online NodeView).
//
// kind_value is the node kind in the shared 0..7 value space: the static
// overloads pass static_cast<uint64_t>(node->type()) (ChakraProtoMsg::
// NodeType, et_def.proto) and the NodeView overloads pass
// static_cast<uint64_t>(node.kind) (ExecutionDriven::NodeKind,
// GraphSource.hh). The two enums mirror each other value-for-value and the
// GraphSource adapters derive kind from the raw type, so this single
// function classifies both paths identically. MEM is checked before
// is_cpu_op/COMP/COMM, mirroring Workload::issue's dispatch (a MEM node goes
// to the remote/HBM issue leg regardless of is_cpu_op).
enum class HwResClass {
    NoOp,       // timer nodes; COMM_RECV endpoints (existing semantics)
    HbmDma,     // local HBM KV restore DMA (counted, UNLIMITED node slot)
    RemoteMem,  // remote MEM_LOAD/MEM_STORE: counted, UNLIMITED node slot
    Cpu,
    GpuComp,
    GpuComm,  // COMM_SEND/COMM_COLL (counted, UNLIMITED node slot) + legacy
              //   metadata/invalid fall-through
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
    }
    return "unknown";
}

// occupy-side check: with node-id retention enabled, an id must never be
// inserted twice (a second insert would pair with only one release).
void ensure_unique_node_id(std::unordered_set<uint64_t>& node_ids,
                           const HwResClass cls,
                           const int sys_id,
                           const uint64_t node_id) {
    if (!node_ids.emplace(node_id).second) {
        LoggerFactory::get_logger("HardwareResource")
            ->critical("HardwareResource duplicate node occupy (class={}): "
                       "sys.id={} node={}",
                       hw_res_class_label(cls), sys_id, node_id);
        std::abort();
    }
}

// release-side pre-decrement checks (方案 §4, NDEBUG-independent): the class
// counter is non-zero, and with node-id retention enabled the node is
// actually tracked in the set. node_ids == nullptr models retention off.
// Fail-closed via abort rather than throw: Sys::call_events() swallows
// std::exception, so a broken resource-accounting invariant must terminate.
void check_release_preconditions(
    const uint32_t counter,
    const std::unordered_set<uint64_t>* node_ids,
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
    if (node_ids != nullptr && node_ids->count(node_id) == 0) {
        LoggerFactory::get_logger("HardwareResource")
            ->critical("HardwareResource release of untracked node id "
                       "(class={}): sys.id={} node={}",
                       hw_res_class_label(cls), sys_id, node_id);
        std::abort();
    }
}

void occupy_class(HardwareResource& hr,
                  const HwResClass cls,
                  const uint64_t node_id,
                  const bool static_et_path) {
    switch (cls) {
        case HwResClass::NoOp:
            return;
        case HwResClass::HbmDma:
            // PARTIAL 跨实例 copy 流水化（2026-09-25）：counted, UNLIMITED
            // node-occupancy slot（与 remote_mem 同款）—— arbitrarily many
            // KV restores may be in flight at once; the former single-slot
            // debug assert is gone (bandwidth arbitration belongs to
            // LocalHbmBandwidthModel's N-way equal split, not to a slot).
            ++hr.num_in_flight_hbm_dma_ops;
            if (hr.retain_node_ids_) {
                ensure_unique_node_id(hr.hbm_dma_ops_node, cls, hr.sys_id,
                                      node_id);
            }
            return;
        case HwResClass::RemoteMem:
            // 方案 §4: independent counted slot -- no capacity limit and no
            // single-slot assert; arbitrarily many remote MEM nodes may be
            // in flight at once. It is a NODE-occupancy count, never a
            // SerDes stream count: it must not feed any port concurrency
            // denominator, busy_ns or bandwidth estimate.
            ++hr.num_in_flight_remote_mem_ops;
            if (hr.retain_node_ids_) {
                ensure_unique_node_id(hr.remote_mem_ops_node, cls, hr.sys_id,
                                      node_id);
            }
            return;
        case HwResClass::Cpu:
            assert(hr.num_in_flight_cpu_ops == 0);
            ++hr.num_in_flight_cpu_ops;
            if (hr.retain_node_ids_) {
                ensure_unique_node_id(hr.cpu_ops_node, cls, hr.sys_id,
                                      node_id);
            }
            return;
        case HwResClass::GpuComp:
            // The static ET path keeps its legacy single-slot debug assert;
            // the online NodeView path stays a plain count (Step 1-8
            // root-cause #3: concurrent online COMP occupancy is a
            // supported state, serialized in production only by the
            // is_available gate in Workload::issue_dep_free_nodes).
            if (static_et_path) {
                assert(hr.num_in_flight_gpu_comp_ops == 0);
            }
            ++hr.num_in_flight_gpu_comp_ops;
            if (hr.retain_node_ids_) {
                ensure_unique_node_id(hr.gpu_ops_node, cls, hr.sys_id,
                                      node_id);
            }
            return;
        case HwResClass::GpuComm:
            // PARTIAL 跨实例 copy 流水化（2026-09-25）：counted, UNLIMITED
            // node-occupancy slot（与 remote_mem 同款）—— multiple
            // COMM_SEND/COMM_COLL nodes may be in flight at once; the
            // former single-slot debug assert is gone (link arbitration
            // belongs to the FluidScheduler per-link N-way equal split).
            ++hr.num_in_flight_gpu_comm_ops;
            if (hr.retain_node_ids_) {
                ensure_unique_node_id(hr.gpu_comms_node, cls, hr.sys_id,
                                      node_id);
            }
            return;
    }
}

void release_class(HardwareResource& hr,
                   const HwResClass cls,
                   const uint64_t node_id,
                   const bool static_et_path) {
    switch (cls) {
        case HwResClass::NoOp:
            return;
        case HwResClass::HbmDma:
            check_release_preconditions(
                hr.num_in_flight_hbm_dma_ops,
                hr.retain_node_ids_ ? &hr.hbm_dma_ops_node : nullptr, cls,
                hr.sys_id, node_id);
            --hr.num_in_flight_hbm_dma_ops;
            if (hr.retain_node_ids_) {
                hr.hbm_dma_ops_node.erase(node_id);
            }
            return;
        case HwResClass::RemoteMem:
            check_release_preconditions(
                hr.num_in_flight_remote_mem_ops,
                hr.retain_node_ids_ ? &hr.remote_mem_ops_node : nullptr, cls,
                hr.sys_id, node_id);
            --hr.num_in_flight_remote_mem_ops;
            if (hr.retain_node_ids_) {
                hr.remote_mem_ops_node.erase(node_id);
            }
            return;
        case HwResClass::Cpu:
            check_release_preconditions(
                hr.num_in_flight_cpu_ops,
                hr.retain_node_ids_ ? &hr.cpu_ops_node : nullptr, cls,
                hr.sys_id, node_id);
            --hr.num_in_flight_cpu_ops;
            assert(hr.num_in_flight_cpu_ops == 0);
            if (hr.retain_node_ids_) {
                hr.cpu_ops_node.erase(node_id);
            }
            return;
        case HwResClass::GpuComp:
            check_release_preconditions(
                hr.num_in_flight_gpu_comp_ops,
                hr.retain_node_ids_ ? &hr.gpu_ops_node : nullptr, cls,
                hr.sys_id, node_id);
            --hr.num_in_flight_gpu_comp_ops;
            if (static_et_path) {
                assert(hr.num_in_flight_gpu_comp_ops == 0);
            }
            if (hr.retain_node_ids_) {
                hr.gpu_ops_node.erase(node_id);
            }
            return;
        case HwResClass::GpuComm:
            check_release_preconditions(
                hr.num_in_flight_gpu_comm_ops,
                hr.retain_node_ids_ ? &hr.gpu_comms_node : nullptr, cls,
                hr.sys_id, node_id);
            --hr.num_in_flight_gpu_comm_ops;
            if (hr.retain_node_ids_) {
                hr.gpu_comms_node.erase(node_id);
            }
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
            // PARTIAL 跨实例 copy 流水化（2026-09-25）: unlimited counted
            // slot -- never blocks issue on its in-flight count. The gate
            // must not become a hidden serialization point for KV restore;
            // bandwidth arbitration belongs to LocalHbmBandwidthModel's
            // N-way equal split over all active jobs.
            return true;
        case HwResClass::Cpu:
            return hr.num_in_flight_cpu_ops == 0;
        case HwResClass::GpuComp:
            return hr.num_in_flight_gpu_comp_ops == 0;
        case HwResClass::GpuComm:
            // PARTIAL 跨实例 copy 流水化（2026-09-25）: unlimited counted
            // slot -- multiple D2D sends/acks/coll may be in flight at
            // once; link arbitration belongs to the FluidScheduler's
            // per-link N-way equal split over active flows.
            return true;
    }
    return true;  // unreachable; every enumerator handled above
}

}  // namespace

HardwareResource::HardwareResource(
    int sys_id, const ExecutionDriven::ExecutionMode execution_mode)
    : sys_id(sys_id),
      retain_node_ids_(execution_mode ==
                       ExecutionDriven::ExecutionMode::Static),
      num_in_flight_cpu_ops(0),
      num_in_flight_gpu_comp_ops(0),
      num_in_flight_gpu_comm_ops(0),
      num_in_flight_hbm_dma_ops(0),
      num_in_flight_remote_mem_ops(0) {

    tics_gpu_ops = 0;

    // cpu_ops_node = NULL;
    // gpu_ops_node = NULL;
    // gpu_comms_node = NULL;
}

void HardwareResource::occupy(
    const shared_ptr<Chakra::FeederV3::ETFeederNode> node) {
    occupy_class(*this,
                 classify_hw_resource(static_cast<uint64_t>(node->type()),
                                      node->is_cpu_op(),
                                      node->get_attr<bool>("is_timer_op",
                                                           false),
                                      is_local_hbm_kv_restore(node)),
                 node->id(),
                 /*static_et_path=*/true);
}

void HardwareResource::release(
    const shared_ptr<Chakra::FeederV3::ETFeederNode> node) {
    release_class(*this,
                  classify_hw_resource(static_cast<uint64_t>(node->type()),
                                       node->is_cpu_op(),
                                       node->get_attr<bool>("is_timer_op",
                                                            false),
                                       is_local_hbm_kv_restore(node)),
                  node->id(),
                  /*static_et_path=*/true);
}

bool HardwareResource::is_available(
    const shared_ptr<Chakra::FeederV3::ETFeederNode> node) const {
    return available_class(
        *this,
        classify_hw_resource(static_cast<uint64_t>(node->type()),
                             node->is_cpu_op(),
                             node->get_attr<bool>("is_timer_op", false),
                             is_local_hbm_kv_restore(node)));
}

// ------------------------- NodeView overloads (方案 §4 shared classification)

void HardwareResource::occupy(const ExecutionDriven::NodeView& node) {
    occupy_class(*this,
                 classify_hw_resource(static_cast<uint64_t>(node.kind),
                                      node.is_cpu_op,
                                      node.is_timer_op,
                                      node.mem.is_local_hbm_kv_restore),
                 node.global_id,
                 /*static_et_path=*/false);
}

void HardwareResource::release(const ExecutionDriven::NodeView& node) {
    release_class(*this,
                  classify_hw_resource(static_cast<uint64_t>(node.kind),
                                       node.is_cpu_op,
                                       node.is_timer_op,
                                       node.mem.is_local_hbm_kv_restore),
                  node.global_id,
                  /*static_et_path=*/false);
}

bool HardwareResource::is_available(const ExecutionDriven::NodeView& node) const {
    return available_class(
        *this,
        classify_hw_resource(static_cast<uint64_t>(node.kind),
                             node.is_cpu_op,
                             node.is_timer_op,
                             node.mem.is_local_hbm_kv_restore));
}
