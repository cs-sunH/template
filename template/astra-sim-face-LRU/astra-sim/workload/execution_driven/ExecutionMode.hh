/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

ExecutionMode -- execution-driven mechanism layer (wscllm phase 1).

Execution mode of the simulation binary, introduced by the step-1-2
execution-mode factory (方案 §4 步骤 1-2 操作 4). Only replacing the main()
function is not enough: the static Sys construction chain unconditionally
constructs Workload, whose constructor immediately checks the .et file and
creates the ETFeeder (see Sys.cc's execution-mode factory branch and the
Workload constructor's Online/Static fork -- anchors intentionally given by
symbol, not line number: those sites have drifted repeatedly). Online mode
must therefore be an explicit construction-mode parameter that flows from
the online entry point into Sys and Workload, so that:

  - Static: legacy ETFeeder path, byte-for-byte the pre-phase-1 behavior
    (the library-level default -- Sys/Workload construct it when no online
    entry passes an explicit mode. Comment fix 2026-09-30: the former
    parenthetical "the static main.cc passes no extra arguments" referenced
    a static main entry that no longer exists; the online entry
    (congestion_aware/main_online.cc) is the only main. Static is reachable
    today only through the in-library defaults and test fixtures.)
  - Online: a dynamic GraphSource is injected at Sys creation time; the
    ETFeeder is never constructed and no .et file is required. Final end
    authority belongs to ServiceCoordinator, never to Workload::call's
    sim_notify_finished()/is_finished side effect (the "Final end authority"
    block of Workload::call, guarded by execution_mode_ != Online), which is
    disabled for online mode.
*******************************************************************************/

#ifndef EXECUTION_DRIVEN_EXECUTIONMODE_HH
#define EXECUTION_DRIVEN_EXECUTIONMODE_HH

namespace AstraSim {
namespace ExecutionDriven {

enum class ExecutionMode { Static, Online };

}  // namespace ExecutionDriven
}  // namespace AstraSim

#endif  // EXECUTION_DRIVEN_EXECUTIONMODE_HH
