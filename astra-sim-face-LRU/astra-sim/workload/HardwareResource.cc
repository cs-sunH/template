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
bool is_local_hbm_kv_restore(
    const shared_ptr<Chakra::FeederV3::ETFeederNode>& node) {
    return node->get_attr<bool>("is_local_hbm_kv_restore", false);
}

// 方案 §4：两套 overload（ETFeederNode 与 NodeView）共用一个语义分类。
// 顺序即优先级：timer no-op -> local HBM KV restore -> HBM 专用槽；
// 其余 MEM_LOAD/MEM_STORE -> remote MEM 槽（MEM 先于 CPU/COMP/COMM，与
// Workload::issue 的节点分发一致，MEM 节点无论 is_cpu_op 都走远端/HBM
// 发射腿）；再分类 CPU、COMP、COMM_RECV no-op、COMM。
enum class ResourceClass {
    Timer,
    HbmRestore,
    RemoteMem,
    Cpu,
    Comp,
    CommRecv,
    Comm,
};

ResourceClass classify(const bool is_timer, const bool is_local_hbm_restore,
                       const bool is_cpu, const bool is_mem,
                       const bool is_comp, const bool is_comm_recv) {
    if (is_timer) {
        return ResourceClass::Timer;
    }
    if (is_local_hbm_restore) {
        return ResourceClass::HbmRestore;
    }
    if (is_mem) {
        return ResourceClass::RemoteMem;
    }
    if (is_cpu) {
        return ResourceClass::Cpu;
    }
    if (is_comp) {
        return ResourceClass::Comp;
    }
    if (is_comm_recv) {
        return ResourceClass::CommRecv;
    }
    return ResourceClass::Comm;
}

// NDEBUG 无关的 fail-closed：Sys::call_events() 会吞 std::exception，
// 资源记账不变量被破坏时必须 abort 而不是 throw（方案 §3.3/§4）。
[[noreturn]] void resource_fatal(const char* what, const int sys_id,
                                 const uint64_t node_id) {
    LoggerFactory::get_logger("HardwareResource")
        ->critical("HardwareResource fatal: {} (sys.id={} node={})", what,
                   sys_id, node_id);
    std::abort();
}

// occupy：单槽类（HBM restore/CPU/COMM）容量违规、node-id 保留模式下的
// 重复占用都在递增前 fatal；remote MEM 与 COMP 是计数制，无容量上限。
void occupy_class(HardwareResource& hr, const ResourceClass cls,
                  const uint64_t node_id) {
    switch (cls) {
        case ResourceClass::Timer:
        case ResourceClass::CommRecv:
            return;
        case ResourceClass::HbmRestore:
            if (hr.num_in_flight_hbm_dma_ops != 0) {
                resource_fatal("HBM-DMA single slot busy on occupy", hr.sys_id,
                               node_id);
            }
            ++hr.num_in_flight_hbm_dma_ops;
            break;
        case ResourceClass::RemoteMem:
            // 计数制、无上限：这是节点占用数，不是 SerDes 流数；不得用作
            // 端口并发分母 / busy_ns / 带宽估算（方案 §4）。
            ++hr.num_in_flight_remote_mem_ops;
            ++hr.num_remote_mem_ops;
            break;
        case ResourceClass::Cpu:
            if (hr.num_in_flight_cpu_ops != 0) {
                resource_fatal("CPU single slot busy on occupy", hr.sys_id,
                               node_id);
            }
            ++hr.num_in_flight_cpu_ops;
            break;
        case ResourceClass::Comp:
            // 计数制：在途数本身无单槽容量上限，但 is_available_class 以
            // num_in_flight_gpu_comp_ops==0 作为发射门，独立 COMP 节点在
            // online 与 static 两条路径上一致串行（path-2 removal 后无
            // bypass，见 Workload::issue_dep_free_nodes 注释）。
            ++hr.num_in_flight_gpu_comp_ops;
            break;
        case ResourceClass::Comm:
            if (hr.num_in_flight_gpu_comm_ops != 0) {
                resource_fatal("GPU comm single slot busy on occupy",
                               hr.sys_id, node_id);
            }
            ++hr.num_in_flight_gpu_comm_ops;
            ++hr.num_gpu_comms;
            break;
    }
    if (!hr.tracks_node_ids()) {
        return;
    }
    std::unordered_set<uint64_t>* node_set = nullptr;
    switch (cls) {
        case ResourceClass::HbmRestore:
            node_set = &hr.hbm_dma_ops_node;
            break;
        case ResourceClass::RemoteMem:
            node_set = &hr.remote_mem_ops_node;
            break;
        case ResourceClass::Cpu:
            node_set = &hr.cpu_ops_node;
            break;
        case ResourceClass::Comp:
            node_set = &hr.gpu_ops_node;
            break;
        case ResourceClass::Comm:
            node_set = &hr.gpu_comms_node;
            break;
        default:
            return;  // Timer / CommRecv 已在上方返回
    }
    // occupy 查重：同一节点重复占用是机制错误，不受 NDEBUG 影响。
    if (!node_set->insert(node_id).second) {
        resource_fatal("duplicate node occupy", hr.sys_id, node_id);
    }
}

// release：递减前做 NDEBUG 无关的前置 fatal 检查——在途计数非零，且启用
// node-id 保留时该节点确实在集合中。旧版递减后的 assert(==0) 由 occupy
// 的容量 fatal 取代（单槽在途数不可能超过 1，能到 release 的必为 1->0）。
void release_class(HardwareResource& hr, const ResourceClass cls,
                   const uint64_t node_id) {
    switch (cls) {
        case ResourceClass::Timer:
        case ResourceClass::CommRecv:
            return;
        case ResourceClass::HbmRestore:
            if (hr.num_in_flight_hbm_dma_ops == 0) {
                resource_fatal("HBM-DMA release underflow", hr.sys_id,
                               node_id);
            }
            if (hr.tracks_node_ids() &&
                hr.hbm_dma_ops_node.count(node_id) == 0) {
                resource_fatal("HBM-DMA release of unoccupied node id",
                               hr.sys_id, node_id);
            }
            --hr.num_in_flight_hbm_dma_ops;
            if (hr.tracks_node_ids()) {
                hr.hbm_dma_ops_node.erase(node_id);
            }
            return;
        case ResourceClass::RemoteMem:
            if (hr.num_in_flight_remote_mem_ops == 0) {
                resource_fatal("remote MEM release underflow", hr.sys_id,
                               node_id);
            }
            if (hr.tracks_node_ids() &&
                hr.remote_mem_ops_node.count(node_id) == 0) {
                resource_fatal("remote MEM release of unoccupied node id",
                               hr.sys_id, node_id);
            }
            --hr.num_in_flight_remote_mem_ops;
            if (hr.tracks_node_ids()) {
                hr.remote_mem_ops_node.erase(node_id);
            }
            return;
        case ResourceClass::Cpu:
            if (hr.num_in_flight_cpu_ops == 0) {
                resource_fatal("CPU release underflow", hr.sys_id, node_id);
            }
            if (hr.tracks_node_ids() && hr.cpu_ops_node.count(node_id) == 0) {
                resource_fatal("CPU release of unoccupied node id", hr.sys_id,
                               node_id);
            }
            --hr.num_in_flight_cpu_ops;
            if (hr.tracks_node_ids()) {
                hr.cpu_ops_node.erase(node_id);
            }
            return;
        case ResourceClass::Comp:
            if (hr.num_in_flight_gpu_comp_ops == 0) {
                resource_fatal("GPU comp release underflow", hr.sys_id,
                               node_id);
            }
            if (hr.tracks_node_ids() &&
                hr.gpu_ops_node.count(node_id) == 0) {
                resource_fatal("GPU comp release of unoccupied node id",
                               hr.sys_id, node_id);
            }
            --hr.num_in_flight_gpu_comp_ops;
            if (hr.tracks_node_ids()) {
                hr.gpu_ops_node.erase(node_id);
            }
            return;
        case ResourceClass::Comm:
            if (hr.num_in_flight_gpu_comm_ops == 0) {
                resource_fatal("GPU comm release underflow", hr.sys_id,
                               node_id);
            }
            if (hr.tracks_node_ids() &&
                hr.gpu_comms_node.count(node_id) == 0) {
                resource_fatal("GPU comm release of unoccupied node id",
                               hr.sys_id, node_id);
            }
            --hr.num_in_flight_gpu_comm_ops;
            if (hr.tracks_node_ids()) {
                hr.gpu_comms_node.erase(node_id);
            }
            return;
    }
}

bool is_available_class(const HardwareResource& hr, const ResourceClass cls) {
    switch (cls) {
        case ResourceClass::Timer:
        case ResourceClass::CommRecv:
            return true;
        case ResourceClass::HbmRestore:
            return hr.num_in_flight_hbm_dma_ops == 0;
        case ResourceClass::RemoteMem:
            // 计数制、无上限：is_available 不因在途数阻塞发射（方案 §4）。
            return true;
        case ResourceClass::Cpu:
            return hr.num_in_flight_cpu_ops == 0;
        case ResourceClass::Comp:
            return hr.num_in_flight_gpu_comp_ops == 0;
        case ResourceClass::Comm:
            return hr.num_in_flight_gpu_comm_ops == 0;
    }
    return true;
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

    num_gpu_comms = 0;
    num_remote_mem_ops = 0;

    tics_gpu_ops = 0;
}

void HardwareResource::occupy(
    const shared_ptr<Chakra::FeederV3::ETFeederNode> node) {
    occupy_class(*this,
                 classify(node->get_attr<bool>("is_timer_op", false),
                          is_local_hbm_kv_restore(node), node->is_cpu_op(),
                          node->type() == ChakraNodeType::MEM_LOAD_NODE ||
                              node->type() == ChakraNodeType::MEM_STORE_NODE,
                          node->type() == ChakraNodeType::COMP_NODE,
                          node->type() == ChakraNodeType::COMM_RECV_NODE),
                 node->id());
}

void HardwareResource::release(
    const shared_ptr<Chakra::FeederV3::ETFeederNode> node) {
    release_class(*this,
                  classify(node->get_attr<bool>("is_timer_op", false),
                           is_local_hbm_kv_restore(node), node->is_cpu_op(),
                           node->type() == ChakraNodeType::MEM_LOAD_NODE ||
                               node->type() == ChakraNodeType::MEM_STORE_NODE,
                           node->type() == ChakraNodeType::COMP_NODE,
                           node->type() == ChakraNodeType::COMM_RECV_NODE),
                  node->id());
}

bool HardwareResource::is_available(
    const shared_ptr<Chakra::FeederV3::ETFeederNode> node) const {
    return is_available_class(
        *this,
        classify(node->get_attr<bool>("is_timer_op", false),
                 is_local_hbm_kv_restore(node), node->is_cpu_op(),
                 node->type() == ChakraNodeType::MEM_LOAD_NODE ||
                     node->type() == ChakraNodeType::MEM_STORE_NODE,
                 node->type() == ChakraNodeType::COMP_NODE,
                 node->type() == ChakraNodeType::COMM_RECV_NODE));
}


// ---------------------------------------------------------- NodeView (§4.3.3)

void HardwareResource::occupy(const ExecutionDriven::NodeView& node) {
    occupy_class(*this,
                 classify(node.is_timer_op, node.is_local_hbm_kv_restore,
                          node.is_cpu_op,
                          node.kind == ExecutionDriven::NodeKind::MemLoad ||
                              node.kind == ExecutionDriven::NodeKind::MemStore,
                          node.kind == ExecutionDriven::NodeKind::Compute,
                          node.kind == ExecutionDriven::NodeKind::CommRecv),
                 node.global_id);
}

void HardwareResource::release(const ExecutionDriven::NodeView& node) {
    release_class(*this,
                  classify(node.is_timer_op, node.is_local_hbm_kv_restore,
                           node.is_cpu_op,
                           node.kind == ExecutionDriven::NodeKind::MemLoad ||
                               node.kind == ExecutionDriven::NodeKind::MemStore,
                           node.kind == ExecutionDriven::NodeKind::Compute,
                           node.kind == ExecutionDriven::NodeKind::CommRecv),
                  node.global_id);
}

bool HardwareResource::is_available(
    const ExecutionDriven::NodeView& node) const {
    return is_available_class(
        *this,
        classify(node.is_timer_op, node.is_local_hbm_kv_restore, node.is_cpu_op,
                 node.kind == ExecutionDriven::NodeKind::MemLoad ||
                     node.kind == ExecutionDriven::NodeKind::MemStore,
                 node.kind == ExecutionDriven::NodeKind::Compute,
                 node.kind == ExecutionDriven::NodeKind::CommRecv));
}
