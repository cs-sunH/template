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

HardwareResource::HardwareResource(
    int sys_id,
    const ExecutionDriven::ExecutionMode execution_mode)
    : sys_id(sys_id),
      retain_node_ids_(execution_mode ==
                       ExecutionDriven::ExecutionMode::Static),
      num_in_flight_cpu_ops(0),
      num_in_flight_gpu_comm_ops(0),
      num_in_flight_gpu_comp_ops(0) {

    tics_gpu_ops = 0;
}

void HardwareResource::occupy(
    const shared_ptr<Chakra::FeederV3::ETFeederNode> node) {
    if (node->get_attr<bool>("is_timer_op", false)) {
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
            gpu_ops_node.emplace(node->id());
        } else {
            if (node->type() == ChakraNodeType::COMM_RECV_NODE) {
                return;
            }
            assert(num_in_flight_gpu_comm_ops == 0);
            ++num_in_flight_gpu_comm_ops;
            gpu_comms_node.emplace(node->id());
        }
    }
}

void HardwareResource::release(
    const shared_ptr<Chakra::FeederV3::ETFeederNode> node) {
    if (node->get_attr<bool>("is_timer_op", false)) {
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

// ---------------------------------------------------------- NodeView (step 1-8)

void HardwareResource::occupy(const ExecutionDriven::NodeView& node) {
    if (node.is_timer_op) {
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
        // Step 1-8 (root-cause #3): count-based, not a single slot.
        // Correction (2026-09-25, workload-F2): the old note claimed the
        // gate was bypassed for calibrated COMP chains in
        // Workload::issue_dep_free_nodes -- that replay-only concurrent
        // calibrated-COMP bypass was deleted with the replay route
        // (2026-08-18; see the comment there), the is_available single-slot
        // gate still applies, so the production online count stays 0/1.
        // The count semantics are kept because watch_registry_test.cc
        // occupies two Compute nodes directly and asserts count==2. The
        // static ETFeederNode path above keeps its single-slot assert
        // (static mode never runs concurrent COMPs).
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
            // Step 1-8 (root-cause #3): count-based, see occupy() -- the
            // single-slot COMP gate still holds in production (workload-F2),
            // so this counter stays 0/1 there; watch_registry_test.cc is the
            // only direct count==2 user.
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
    if (node.is_cpu_op) {
        return num_in_flight_cpu_ops == 0;
    }
    if (node.kind == ExecutionDriven::NodeKind::Compute) {
        return num_in_flight_gpu_comp_ops == 0;
    }
    // COMM_RECV never occupies the hardware (static semantic preserved:
    // it is always available, comm-in-flight or not).
    if (node.kind == ExecutionDriven::NodeKind::CommRecv) {
        return true;
    }
    return num_in_flight_gpu_comm_ops == 0;
}
