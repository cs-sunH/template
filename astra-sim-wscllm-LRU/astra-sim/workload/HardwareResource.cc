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
}  // namespace

HardwareResource::HardwareResource(
    int sys_id, const ExecutionDriven::ExecutionMode execution_mode)
    : sys_id(sys_id),
      retain_node_ids_(execution_mode ==
                       ExecutionDriven::ExecutionMode::Static),
      num_in_flight_cpu_ops(0),
      num_in_flight_gpu_comp_ops(0),
      num_in_flight_gpu_comm_ops(0),
      num_in_flight_hbm_dma_ops(0) {

    tics_gpu_ops = 0;
}

void HardwareResource::occupy(
    const shared_ptr<Chakra::FeederV3::ETFeederNode> node) {
    if (node->get_attr<bool>("is_timer_op", false)) {
        return;
    }
    if (is_local_hbm_kv_restore(node)) {
        assert(num_in_flight_hbm_dma_ops == 0);
        ++num_in_flight_hbm_dma_ops;
        hbm_dma_ops_node.emplace(node->id());
        return;
    }
    if (node->is_cpu_op()) {
        assert(num_in_flight_cpu_ops == 0);
        ++num_in_flight_cpu_ops;
        cpu_ops_node.emplace(node->id());
    } else {
        if (node->type() == ChakraNodeType::COMP_NODE) {
            assert(num_in_flight_gpu_comp_ops == 0);
            ++num_in_flight_gpu_comp_ops;
            // gpu_ops_node = node;
            gpu_ops_node.emplace(node->id());
        } else {
            if (node->type() == ChakraNodeType::COMM_RECV_NODE) {
                return;
            }
            assert(num_in_flight_gpu_comm_ops == 0);
            ++num_in_flight_gpu_comm_ops;
            // gpu_comms_node = node;
            gpu_comms_node.emplace(node->id());
        }
    }
}

void HardwareResource::release(
    const shared_ptr<Chakra::FeederV3::ETFeederNode> node) {
    if (node->get_attr<bool>("is_timer_op", false)) {
        return;
    }
    if (is_local_hbm_kv_restore(node)) {
        --num_in_flight_hbm_dma_ops;
        assert(num_in_flight_hbm_dma_ops == 0);
        hbm_dma_ops_node.erase(node->id());
        return;
    }
    if (node->is_cpu_op()) {
        --num_in_flight_cpu_ops;
        assert(num_in_flight_cpu_ops == 0);
        this->cpu_ops_node.erase(node->id());
    } else {
        if (node->type() == ChakraNodeType::COMP_NODE) {
            --num_in_flight_gpu_comp_ops;
            assert(num_in_flight_gpu_comp_ops == 0);
            this->gpu_ops_node.erase(node->id());
        } else {
            if (node->type() == ChakraNodeType::COMM_RECV_NODE) {
                return;
            }
            --num_in_flight_gpu_comm_ops;
            assert(num_in_flight_gpu_comm_ops == 0);
            this->gpu_comms_node.erase(node->id());
        }
    }
}

bool HardwareResource::is_available(
    const shared_ptr<Chakra::FeederV3::ETFeederNode> node) const {
    if (node->get_attr<bool>("is_timer_op", false)) {
        return true;
    }
    if (is_local_hbm_kv_restore(node)) {
        return num_in_flight_hbm_dma_ops == 0;
    }
    if (node->is_cpu_op()) {
        if (num_in_flight_cpu_ops == 0) {
            return true;
        } else {
            return false;
        }
    } else {
        if (node->type() == ChakraNodeType::COMP_NODE) {
            if (num_in_flight_gpu_comp_ops == 0) {
                return true;
            } else {
                return false;
            }
        } else {
            if (num_in_flight_gpu_comm_ops == 0) {
                return true;
            } else {
                if (node->type() == ChakraNodeType::COMM_RECV_NODE) {
                    return true;
                }
                return false;
            }
        }
    }
}

// ---------------------------------------------------------- NodeView (§4.3.3)

void HardwareResource::occupy(const ExecutionDriven::NodeView& node) {
    if (node.is_timer_op) {
        return;
    }
    if (node.is_local_hbm_kv_restore) {
        // sh_2.0 fourth resource class (hbm_dma restore DMA): single slot,
        // serialized through the is_available check below (static mode never
        // runs concurrent DMA).
        assert(num_in_flight_hbm_dma_ops == 0);
        ++num_in_flight_hbm_dma_ops;
        if (retain_node_ids_) {
            hbm_dma_ops_node.emplace(node.global_id);
        }
        return;
    }
    if (node.is_cpu_op) {
        assert(num_in_flight_cpu_ops == 0);
        ++num_in_flight_cpu_ops;
        if (retain_node_ids_) {
            cpu_ops_node.emplace(node.global_id);
        }
    } else {
        if (node.kind == ExecutionDriven::NodeKind::Compute) {
            // Online COMP occupancy is serialized in production through the
            // is_available gate in Workload::issue_dep_free_nodes (the
            // former calibrated-chain gate bypass was removed with the
            // Path-2 replay route), so the counter only ever toggles 0/1;
            // it stays a plain count rather than the static path's
            // single-slot assert. The static ETFeederNode path above keeps
            // its single-slot assert.
            ++num_in_flight_gpu_comp_ops;
            if (retain_node_ids_) {
                gpu_ops_node.emplace(node.global_id);
            }
        } else {
            if (node.kind == ExecutionDriven::NodeKind::CommRecv) {
                return;
            }
            assert(num_in_flight_gpu_comm_ops == 0);
            ++num_in_flight_gpu_comm_ops;
            if (retain_node_ids_) {
                gpu_comms_node.emplace(node.global_id);
            }
        }
    }
}

void HardwareResource::release(const ExecutionDriven::NodeView& node) {
    if (node.is_timer_op) {
        return;
    }
    if (node.is_local_hbm_kv_restore) {
        if (num_in_flight_hbm_dma_ops == 0) {
            LoggerFactory::get_logger("HardwareResource")
                ->critical("online HBM-DMA release underflow: sys.id={} node={}",
                           sys_id, node.global_id);
            std::abort();
        }
        --num_in_flight_hbm_dma_ops;
        assert(num_in_flight_hbm_dma_ops == 0);
        if (retain_node_ids_) {
            hbm_dma_ops_node.erase(node.global_id);
        }
        return;
    }
    if (node.is_cpu_op) {
        if (num_in_flight_cpu_ops == 0) {
            LoggerFactory::get_logger("HardwareResource")
                ->critical("online CPU release underflow: sys.id={} node={}",
                           sys_id, node.global_id);
            std::abort();
        }
        --num_in_flight_cpu_ops;
        assert(num_in_flight_cpu_ops == 0);
        if (retain_node_ids_) {
            this->cpu_ops_node.erase(node.global_id);
        }
    } else {
        if (node.kind == ExecutionDriven::NodeKind::Compute) {
            if (num_in_flight_gpu_comp_ops == 0) {
                LoggerFactory::get_logger("HardwareResource")
                    ->critical(
                        "online GPU-comp release underflow: sys.id={} node={}",
                        sys_id, node.global_id);
                std::abort();
            }
            --num_in_flight_gpu_comp_ops;
            if (retain_node_ids_) {
                this->gpu_ops_node.erase(node.global_id);
            }
        } else {
            if (node.kind == ExecutionDriven::NodeKind::CommRecv) {
                return;
            }
            if (num_in_flight_gpu_comm_ops == 0) {
                LoggerFactory::get_logger("HardwareResource")
                    ->critical(
                        "online GPU-comm release underflow: sys.id={} node={}",
                        sys_id, node.global_id);
                std::abort();
            }
            --num_in_flight_gpu_comm_ops;
            assert(num_in_flight_gpu_comm_ops == 0);
            if (retain_node_ids_) {
                this->gpu_comms_node.erase(node.global_id);
            }
        }
    }
}

bool HardwareResource::is_available(const ExecutionDriven::NodeView& node) const {
    if (node.is_timer_op) {
        return true;
    }
    if (node.is_local_hbm_kv_restore) {
        return num_in_flight_hbm_dma_ops == 0;
    }
    if (node.is_cpu_op) {
        return num_in_flight_cpu_ops == 0;
    }
    if (node.kind == ExecutionDriven::NodeKind::Compute) {
        return num_in_flight_gpu_comp_ops == 0;
    }
    if (node.kind == ExecutionDriven::NodeKind::CommRecv) {
        return true;
    }
    return num_in_flight_gpu_comm_ops == 0;
}
