/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

ExecutionMode -- execution-driven mechanism layer (wscllm phase 1).

Execution mode of the simulation binary, introduced by the step-1-2
execution-mode factory (方案 §4 步骤 1-2 操作 4). Only replacing the main()
function is not enough: the static Sys construction chain unconditionally
constructs Workload, whose constructor immediately checks the .et file and
creates the ETFeeder (Sys.cc:259-260, Workload.cc:30-35). Online mode must
therefore be an explicit construction-mode parameter that flows from the
online entry point into Sys and Workload, so that:

  - Static: legacy ETFeeder path, byte-for-byte the pre-phase-1 behavior
    (default; the static main.cc passes no extra arguments).
  - Online: a dynamic GraphSource is injected at Sys creation time; the
    ETFeeder is never constructed and no .et file is required. Final end
    authority belongs to ServiceCoordinator, never to Workload::call's
    sim_notify_finished()/is_finished side effect (Workload.cc:625-634),
    which is disabled for online mode.
*******************************************************************************/

#ifndef EXECUTION_DRIVEN_EXECUTIONMODE_HH
#define EXECUTION_DRIVEN_EXECUTIONMODE_HH

namespace AstraSim {
namespace ExecutionDriven {

enum class ExecutionMode { Static, Online };

}  // namespace ExecutionDriven
}  // namespace AstraSim

#endif  // EXECUTION_DRIVEN_EXECUTIONMODE_HH
