/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

// TODO: HardwareResource.hh should be moved to the system layer.

#ifndef __HARDWARE_RESOURCE_HH__
#define __HARDWARE_RESOURCE_HH__

#include "astra-sim/common/Logging.hh"
#include <cstdint>

#include "astra-sim/workload/execution_driven/GraphSource.hh"

namespace AstraSim {

class HardwareResource {
  public:
    explicit HardwareResource(int sys_id = -1);
    ~HardwareResource() {
        auto logger = LoggerFactory::get_logger("HardwareResource");
        if (this->num_in_flight_cpu_ops != 0 ||
            this->num_in_flight_gpu_comm_ops != 0 ||
            this->num_in_flight_gpu_comp_ops != 0) {
            logger->critical(
                "!!!Hardware Resource sys.id={} has unreleased nodes!!!",
                this->sys_id);
        }
    }
    // Domain assumption (deliberate asymmetry, registered wl-F7): occupy/
    // release abort on every reachable invariant violation (double-occupy,
    // underflow, residual at the release of the last node), but the
    // destructor only logs a critical line for residual in-flight counters.
    // Rationale: the destructor runs inside process teardown -- a fatal
    // exit path or a test harness dismantling Sys mid-run -- where an
    // abort() could mask the ORIGINAL fatal cause and break harness exit
    // semantics; the residual counters there are a symptom, never the
    // first failure. Keep the critical line so the imbalance stays
    // visible in logs; promote it to abort only if a teardown-order audit
    // ever proves it can be the first failure.
    // NodeView-keyed accessors (step 1-8). Dispatch on the NodeView
    // fields: is_timer_op -> no-op, is_cpu_op -> CPU, kind==Compute
    // -> GPU comp, kind==CommRecv -> no-op, else -> GPU comm.
    void occupy(const ExecutionDriven::NodeView& node);
    void release(const ExecutionDriven::NodeView& node);
    bool is_available(const ExecutionDriven::NodeView& node) const;

    const int sys_id;

    // NodeStore owns the fail-closed node lifecycle; only the exact
    // in-flight counters live here, avoiding hash work on the per-node hot
    // path (the static-mode per-node ID hash sets retired with the static
    // ET path, 2026-09-27).
    uint32_t num_in_flight_cpu_ops;
    uint32_t num_in_flight_gpu_comp_ops;
    uint32_t num_in_flight_gpu_comm_ops;

    // The write-only tics_gpu_ops accumulator (and the earlier num_*/
    // tics_cpu_ops/tics_gpu_comms debug counters plus HardwareResource::
    // report) were removed as dead on 2026-09-27: its only reader,
    // Workload::report's exposed-communication line, had zero callers
    // repo-wide (workload-F4 follow-up).
};

}  // namespace AstraSim

#endif /* __HARDWARE_RESOURCE_HH__ */
