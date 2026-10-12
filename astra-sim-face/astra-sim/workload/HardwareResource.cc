/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

// Status note (2026-09-26 review, replaces the bare upstream TODO): this
// file conceptually belongs to the system layer, but it stays in
// workload/ on purpose so the five sibling repositories remain
// diff-comparable; move it only as a coordinated cross-repo change.

#include "astra-sim/workload/HardwareResource.hh"

#include <cstdlib>

using namespace std;
using namespace AstraSim;
using namespace Chakra;

typedef ChakraProtoMsg::NodeType ChakraNodeType;

HardwareResource::HardwareResource(
    int sys_id, const ExecutionDriven::ExecutionMode execution_mode)
    : sys_id(sys_id),
      retain_node_ids_(execution_mode ==
                       ExecutionDriven::ExecutionMode::Static),
      num_in_flight_cpu_ops(0),
      num_in_flight_gpu_comm_ops(0),
      num_in_flight_gpu_comp_ops(0),
      tics_gpu_ops(0) {
}

void HardwareResource::occupy(
    const shared_ptr<Chakra::FeederV3::ETFeederNode> node) {
    if (node->get_attr<bool>("is_timer_op", false)) {
        return;
    }
    // (2026-10-01) the occupy-side single-slot asserts are Release-
    // compiled-out, asymmetric with the R1-C4 release-side guards below: a
    // regression that defeats the is_available() gate would silently stack
    // two nodes into one slot under NDEBUG. Mirror the same critical+abort
    // shape (zero reachability today -- both call sites are gated).
    if (node->is_cpu_op()) {
        if (num_in_flight_cpu_ops != 0) {
            LoggerFactory::get_logger("HardwareResource")
                ->critical("static CPU occupy double-occupancy: sys.id={} "
                           "node={}", sys_id, node->id());
            std::abort();
        }
        ++num_in_flight_cpu_ops;
        cpu_ops_node.emplace(node->id());
    } else {
        if (node->type() == ChakraNodeType::COMP_NODE) {
            if (num_in_flight_gpu_comp_ops != 0) {
                LoggerFactory::get_logger("HardwareResource")
                    ->critical("static GPU-comp occupy double-occupancy: "
                               "sys.id={} node={}", sys_id, node->id());
                std::abort();
            }
            ++num_in_flight_gpu_comp_ops;
            gpu_ops_node.emplace(node->id());
        } else {
            if (node->type() == ChakraNodeType::COMM_RECV_NODE) {
                return;
            }
            if (num_in_flight_gpu_comm_ops != 0) {
                LoggerFactory::get_logger("HardwareResource")
                    ->critical("static GPU-comm occupy double-occupancy: "
                               "sys.id={} node={}", sys_id, node->id());
                std::abort();
            }
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
    // R1-C4 fix (2026-09-29): the single-slot asserts below are compiled
    // out under NDEBUG, so a release without a matching occupy wrapped the
    // counters to ~1.8e19. Mirror the online NodeView overload's explicit
    // underflow guards.
    if (node->is_cpu_op()) {
        if (num_in_flight_cpu_ops == 0) {
            LoggerFactory::get_logger("HardwareResource")
                ->critical("static CPU release underflow: sys.id={} node={}",
                           sys_id, node->id());
            std::abort();
        }
        --num_in_flight_cpu_ops;
        assert(num_in_flight_cpu_ops == 0);
        this->cpu_ops_node.erase(node->id());
    } else {
        if (node->type() == ChakraNodeType::COMP_NODE) {
            if (num_in_flight_gpu_comp_ops == 0) {
                LoggerFactory::get_logger("HardwareResource")
                    ->critical(
                        "static GPU-comp release underflow: sys.id={} "
                        "node={}",
                        sys_id, node->id());
                std::abort();
            }
            --num_in_flight_gpu_comp_ops;
            assert(num_in_flight_gpu_comp_ops == 0);
            this->gpu_ops_node.erase(node->id());
        } else {
            if (node->type() == ChakraNodeType::COMM_RECV_NODE) {
                return;
            }
            if (num_in_flight_gpu_comm_ops == 0) {
                LoggerFactory::get_logger("HardwareResource")
                    ->critical(
                        "static GPU-comm release underflow: sys.id={} "
                        "node={}",
                        sys_id, node->id());
                std::abort();
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
        // Step 1-8 (root-cause #3): unlike the single-slot asserts on the
        // other resource classes, the COMP in-flight tracker is kept as a
        // count without an assert. Historically the online replay route ran
        // calibrated COMP chains concurrently with this gate bypassed
        // ("path-2"); that bypass was deleted on 2026-08-18 and every
        // online issue is now serialized through Workload::issue_dep_free_nodes'
        // is_available() check (Compute requires an empty COMP slot), so the
        // count cannot exceed 1 on any live path today. The counter form
        // remains as defensive bookkeeping only.
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
            // Step 1-8 (root-cause #3): count-based, see occupy().
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
