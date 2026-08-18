/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

CompletionObserver -- execution-driven mechanism layer (sh_1.0 port; blueprint wscllm phase 1).
Implementation (方案 §4 步骤 1-3).
*******************************************************************************/

#include "astra-sim/workload/execution_driven/CompletionObserver.hh"

namespace AstraSim {
namespace ExecutionDriven {

CompletionObserver& CompletionObserver::instance() {
    static CompletionObserver observer;
    return observer;
}

void CompletionObserver::set_hook(CompletionHook hook, void* ctx) {
    hook_ = hook;
    hook_ctx_ = ctx;
}

void CompletionObserver::record_node_terminal(
    int rank, uint64_t node_id, const char* request_id, const char* stage,
    uint64_t generation, uint64_t tick, NodeTerminalStatus status) {
    if (hook_ == nullptr) {
        return;  // zero overhead when unset (static path unchanged)
    }
    hook_(hook_ctx_, rank, node_id, request_id, stage, generation, tick,
          static_cast<int>(status));
}

}  // namespace ExecutionDriven
}  // namespace AstraSim
