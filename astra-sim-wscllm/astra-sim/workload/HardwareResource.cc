/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#include "astra-sim/workload/HardwareResource.hh"

#include <cstdlib>

using namespace std;
using namespace AstraSim;


HardwareResource::HardwareResource(int sys_id)
    : sys_id(sys_id),
      num_in_flight_cpu_ops(0),
      num_in_flight_gpu_comm_ops(0),
      num_in_flight_gpu_comp_ops(0) {}

// ---------------------------------------------------------- NodeView (step 1-8)

void HardwareResource::occupy(const ExecutionDriven::NodeView& node) {
    if (node.is_timer_op) {
        return;
    }
    if (node.is_cpu_op) {
        // C1-5 fail-closed: a second overlapping CPU op is an invariant
        // violation and must never be silently absorbed under NDEBUG.
        if (num_in_flight_cpu_ops != 0) {
            LoggerFactory::get_logger("HardwareResource")
                ->critical("online CPU double-occupy: sys.id={} node={}",
                           sys_id, node.global_id);
            std::abort();
        }
        ++num_in_flight_cpu_ops;
    } else {
        if (node.kind == ExecutionDriven::NodeKind::Compute) {
        // Step 1-8 (root-cause #3): count-based, not a single slot.
        // Correction (2026-09-25, workload-F2): the is_available single-slot
        // gate still applies in production, so the production online count
        // stays 0/1. The count semantics are kept because
        // watch_registry_test.cc occupies two Compute nodes directly and
        // asserts count==2.
        ++num_in_flight_gpu_comp_ops;
        } else {
            if (node.kind == ExecutionDriven::NodeKind::CommRecv) {
                return;
            }
            if (num_in_flight_gpu_comm_ops != 0) {
                LoggerFactory::get_logger("HardwareResource")
                    ->critical("online GPU-comm double-occupy: sys.id={} node={}",
                               sys_id, node.global_id);
                std::abort();
            }
            ++num_in_flight_gpu_comm_ops;
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
        if (num_in_flight_cpu_ops != 0) {
            LoggerFactory::get_logger("HardwareResource")
                ->critical("online CPU release residual: sys.id={} node={}",
                           sys_id, node.global_id);
            std::abort();
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
            if (num_in_flight_gpu_comm_ops != 0) {
                LoggerFactory::get_logger("HardwareResource")
                    ->critical(
                        "online GPU-comm release residual: sys.id={} node={}",
                        sys_id, node.global_id);
                std::abort();
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
