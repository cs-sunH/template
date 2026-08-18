/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

CompletionObserver -- execution-driven mechanism layer (sh30 (wscllm-blueprint) phase 1).

Node-terminal recording independent of the metrics switch (总体方案 §5.1
清单第 1 条; 仿真加速分析.md §5.1). Mounted unconditionally on the three
terminal paths of Workload::call / skip_invalid (step 1-3); the terminal
facts feed the step-1-5 watch/fence machinery.

Zero-cost contract: with no hook installed the record path returns
immediately, so the static path keeps its pre-phase-1 behavior byte-for-byte.

Terminal status semantics (explicit per path, never implied):
  Success  -- the node's execution completed (collective branch and the
              generic WorkloadLayerHandlerData branch of Workload::call).
  Skipped  -- the node never executed: INVALID_NODE, metadata nodes, or a
              roofline COMP node with tensor_size == 0 (Workload::skip_invalid).
There is no cancellation path in this codebase (grep "cancel" over
Workload.cc / the chakra feeder finds none), so a CANCELLED status is NOT
defined here -- per the step-1-3 acceptance rule, unconstructible statuses
are deleted from the enum instead of being declared. Whether Skipped
satisfies a watch is decided explicitly by the watch type (step 1-5), never
by defaulting Skipped to Success.

request_id/stage/generation are filled by the online mode via the NodeStore
reverse index (steps 1-4/1-5); the static path passes nullptr/0.
*******************************************************************************/

#ifndef EXECUTION_DRIVEN_COMPLETION_OBSERVER_HH
#define EXECUTION_DRIVEN_COMPLETION_OBSERVER_HH

#include <cstdint>

namespace AstraSim {
namespace ExecutionDriven {

/// Terminal status of an executed node. CANCELLED deliberately absent:
/// unconstructible in this codebase (see file header).
enum class NodeTerminalStatus : int {
    Success = 0,
    Skipped = 1,
};

/// Hook signature (fixable at set_hook time; the terminal_status int is a
/// NodeTerminalStatus value, passed as int for ABI simplicity).
using CompletionHook = void (*)(void* ctx, int rank, uint64_t node_id,
                                const char* request_id, const char* stage,
                                uint64_t generation, uint64_t tick,
                                int terminal_status);

/// Process-wide observer. Singleton state is a function-local static (C++11
/// thread-safe init); only the simulation thread calls record_node_terminal.
class CompletionObserver {
  public:
    static CompletionObserver& instance();

    /// Install the terminal-recording hook; nullptr (the default) = zero
    /// overhead: record_node_terminal returns immediately.
    void set_hook(CompletionHook hook, void* ctx);

    /// Record a node terminal fact. No-op when no hook is installed.
    void record_node_terminal(int rank, uint64_t node_id,
                              const char* request_id, const char* stage,
                              uint64_t generation, uint64_t tick,
                              NodeTerminalStatus status);

  private:
    CompletionObserver() = default;

    CompletionHook hook_ = nullptr;
    void* hook_ctx_ = nullptr;
};

}  // namespace ExecutionDriven
}  // namespace AstraSim

#endif  // EXECUTION_DRIVEN_COMPLETION_OBSERVER_HH
