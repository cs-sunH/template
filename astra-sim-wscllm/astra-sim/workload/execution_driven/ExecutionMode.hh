/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

ExecutionMode -- execution-driven mechanism layer (wscllm phase 1).

Execution mode of the simulation binary, introduced by the step-1-2
execution-mode factory (方案 §4 步骤 1-2 操作 4). Only replacing the main()
function is not enough: the mode must flow as an explicit construction
parameter from the entry point into Sys and Workload (the execution-mode
factory branch in Sys::Sys; the Workload constructor takes the mode plus the
injected GraphSource), so that:

  - Static: the legacy ETFeeder path. RETIRED in this repository (the
    offline static arm was removed with ETFeederGraphSource and the
    Workload::fire() auto-advance entry with it); Online is the only
    constructible mode and the enum member survives as the parameter's
    domain value.
  - Online: a dynamic GraphSource is injected at Sys creation time; the
    ETFeeder is never constructed and no .et file is required. Final end
    authority belongs to ServiceCoordinator, never to Workload.
*******************************************************************************/

#ifndef EXECUTION_DRIVEN_EXECUTIONMODE_HH
#define EXECUTION_DRIVEN_EXECUTIONMODE_HH

namespace AstraSim {
namespace ExecutionDriven {

enum class ExecutionMode { Static, Online };

}  // namespace ExecutionDriven
}  // namespace AstraSim

#endif  // EXECUTION_DRIVEN_EXECUTIONMODE_HH
