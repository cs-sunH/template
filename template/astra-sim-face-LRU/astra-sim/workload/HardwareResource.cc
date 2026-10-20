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
// 发射腿）；再分类 CPU、COMP、COMM_RECV no-op、COMM_COLL（③-lite 集合
// 独立成类，不占 gpu_comm 单槽）、COMM。
enum class ResourceClass {
    Timer,
    HbmRestore,
    RemoteMem,
    Cpu,
    Comp,
    CommRecv,
    // 死锁真修 ③-lite（2026-10-11，自 WL 同构移植）：COMM_COLL（all_reduce
    // /屏障）独立成类——集合通信不再占 gpu_comm 单槽等成员：等成员/交换
    // 期间不阻塞同 rank 的 p2p（四方环闭合成分之三拆除）。计数观测、无
    // 限额、available 恒真；同实例集合间仍由 per-rank 图 frontier 串行
    // （体节点链位）。
    CommColl,
    Comm,
    // (2026-10-01) METADATA / INVALID nodes (and the zero-tensor COMP that
    // skip_invalid finishes synchronously) used to fall through to Comm --
    // inflating num_gpu_comms AND gating same-rank nodes behind the Comm
    // single-slot availability during the transient occupancy (a timing
    // side effect, not just a counting one). Give them a no-op class so
    // the terminal counters stay per-semantic and the emit gate ignores
    // them. (Zero-tensor COMP still classifies as Comp/Comm per shape:
    // reading tensor_size here would duplicate Workload's skip decision on
    // attribute surfaces that may be absent on metadata nodes.)
    Other,
};

ResourceClass classify(const bool is_timer, const bool is_local_hbm_restore,
                       const bool is_cpu, const bool is_mem,
                       const bool is_comp, const bool is_comm_recv,
                       const bool is_comm_coll) {
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
    // ③-lite：COMM_COLL 独立成类、不占 gpu_comm 单槽（见 ResourceClass::
    // CommColl 注记）。须在 Comm 兜底前。
    if (is_comm_coll) {
        return ResourceClass::CommColl;
    }
    return ResourceClass::Comm;
}

// (2026-10-01) Terminal-status-aware classification wrapper: terminal
// Skipped nodes (metadata / INVALID via Workload::skip_invalid; zero-tensor
// COMP stays shape-classified, see the Other enum note) classify as Other.
// All three public entries (occupy / release / is_available) must classify
// the SAME node the SAME way -- the occupy/release pairing is symmetric.
ResourceClass classify_with_status(
    const bool is_timer, const bool is_local_hbm_restore,
    const bool is_cpu, const bool is_mem, const bool is_comp,
    const bool is_comm_recv, const bool is_comm_coll, const bool skipped) {
    if (skipped) {
        return ResourceClass::Other;
    }
    return classify(is_timer, is_local_hbm_restore, is_cpu, is_mem, is_comp,
                    is_comm_recv, is_comm_coll);
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
        case ResourceClass::Other:
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
        case ResourceClass::CommColl:
            // 死锁真修 ③-lite（2026-10-11，自 WL 同构移植）：集合不占
            // gpu_comm 单槽等成员——计数观测（无单槽断言、无容量门），
            // available 恒真；node-id 保留不适用（同 Timer/CommRecv）。
            ++hr.num_in_flight_gpu_comm_coll_ops;
            return;
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
        case ResourceClass::Other:
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
        case ResourceClass::CommColl:
            if (hr.num_in_flight_gpu_comm_coll_ops == 0) {
                resource_fatal("GPU comm-coll release underflow", hr.sys_id,
                               node_id);
            }
            --hr.num_in_flight_gpu_comm_coll_ops;
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
        case ResourceClass::Other:
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
        case ResourceClass::CommColl:
            // ③-lite（2026-10-11，自 WL 同构移植）：集合永不因槽位阻塞
            // issue（等成员不占槽）。
            return true;
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
      num_in_flight_gpu_comm_coll_ops(0),
      num_in_flight_hbm_dma_ops(0),
      num_in_flight_remote_mem_ops(0) {

    num_gpu_comms = 0;
    num_remote_mem_ops = 0;

    tics_gpu_ops = 0;
}

void HardwareResource::occupy(
    const shared_ptr<Chakra::FeederV3::ETFeederNode> node) {
    occupy_class(*this,
                 classify_with_status(
                     node->get_attr<bool>("is_timer_op", false),
                     is_local_hbm_kv_restore(node), node->is_cpu_op(),
                     node->type() == ChakraNodeType::MEM_LOAD_NODE ||
                         node->type() == ChakraNodeType::MEM_STORE_NODE,
                     node->type() == ChakraNodeType::COMP_NODE,
                     node->type() == ChakraNodeType::COMM_RECV_NODE,
                     node->type() == ChakraNodeType::COMM_COLL_NODE,
                     node->type() == ChakraNodeType::INVALID_NODE ||
                         node->type() == ChakraNodeType::METADATA_NODE),
                 node->id());
}

void HardwareResource::release(
    const shared_ptr<Chakra::FeederV3::ETFeederNode> node) {
    release_class(*this,
                  classify_with_status(
                      node->get_attr<bool>("is_timer_op", false),
                      is_local_hbm_kv_restore(node), node->is_cpu_op(),
                      node->type() == ChakraNodeType::MEM_LOAD_NODE ||
                          node->type() == ChakraNodeType::MEM_STORE_NODE,
                      node->type() == ChakraNodeType::COMP_NODE,
                      node->type() == ChakraNodeType::COMM_RECV_NODE,
                      node->type() == ChakraNodeType::COMM_COLL_NODE,
                      node->type() == ChakraNodeType::INVALID_NODE ||
                          node->type() == ChakraNodeType::METADATA_NODE),
                  node->id());
}

bool HardwareResource::is_available(
    const shared_ptr<Chakra::FeederV3::ETFeederNode> node) const {
    return is_available_class(
        *this,
        classify_with_status(
            node->get_attr<bool>("is_timer_op", false),
            is_local_hbm_kv_restore(node), node->is_cpu_op(),
            node->type() == ChakraNodeType::MEM_LOAD_NODE ||
                node->type() == ChakraNodeType::MEM_STORE_NODE,
            node->type() == ChakraNodeType::COMP_NODE,
            node->type() == ChakraNodeType::COMM_RECV_NODE,
            node->type() == ChakraNodeType::COMM_COLL_NODE,
            node->type() == ChakraNodeType::INVALID_NODE ||
                node->type() == ChakraNodeType::METADATA_NODE));
}


// ---------------------------------------------------------- NodeView (§4.3.3)

void HardwareResource::occupy(const ExecutionDriven::NodeView& node) {
    occupy_class(*this,
                 classify_with_status(
                     node.is_timer_op, node.is_local_hbm_kv_restore,
                     node.is_cpu_op,
                     node.kind == ExecutionDriven::NodeKind::MemLoad ||
                         node.kind == ExecutionDriven::NodeKind::MemStore,
                     node.kind == ExecutionDriven::NodeKind::Compute,
                     node.kind == ExecutionDriven::NodeKind::CommRecv,
                     node.kind == ExecutionDriven::NodeKind::CommCollective,
                     node.kind == ExecutionDriven::NodeKind::Invalid ||
                         node.kind == ExecutionDriven::NodeKind::Metadata),
                 node.global_id);
}

void HardwareResource::release(const ExecutionDriven::NodeView& node) {
    release_class(*this,
                  classify_with_status(
                      node.is_timer_op, node.is_local_hbm_kv_restore,
                      node.is_cpu_op,
                      node.kind == ExecutionDriven::NodeKind::MemLoad ||
                          node.kind == ExecutionDriven::NodeKind::MemStore,
                      node.kind == ExecutionDriven::NodeKind::Compute,
                      node.kind == ExecutionDriven::NodeKind::CommRecv,
                      node.kind == ExecutionDriven::NodeKind::CommCollective,
                      node.kind == ExecutionDriven::NodeKind::Invalid ||
                          node.kind == ExecutionDriven::NodeKind::Metadata),
                  node.global_id);
}

bool HardwareResource::is_available(
    const ExecutionDriven::NodeView& node) const {
    return is_available_class(
        *this,
        classify_with_status(
            node.is_timer_op, node.is_local_hbm_kv_restore, node.is_cpu_op,
            node.kind == ExecutionDriven::NodeKind::MemLoad ||
                node.kind == ExecutionDriven::NodeKind::MemStore,
            node.kind == ExecutionDriven::NodeKind::Compute,
            node.kind == ExecutionDriven::NodeKind::CommRecv,
            node.kind == ExecutionDriven::NodeKind::CommCollective,
            node.kind == ExecutionDriven::NodeKind::Invalid ||
                node.kind == ExecutionDriven::NodeKind::Metadata));
}
