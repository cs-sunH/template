/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

RequestIngress -- execution-driven mechanism layer (sh30 (wscllm-blueprint) phase 1).

Request-neutral ingress (方案 §4 步骤 1-2 操作 2). Thread contract: external
producers (injection thread, later the step-1-7 bridge reader) only write the
thread-safe bounded command queue (submit/close/EOF/error); the simulation
thread alone drains commands and schedules arrival alarms. External threads
must never call EventQueue or schedule arrivals directly.

Arrival alarms use schedule_event() with a strictly future tick (legal under
the EventQueue strict-increase rule; a late arrival is clamped to
current+1). The arrival callback writes the ServiceCoordinator accounting
(step 1-2) and the DecisionMailbox(ARRIVAL, payload) deposit (step 1-6) --
the only decision event produced by the ingress; a submit is a decision
boundary, not an execution event.

The command queue and the step-1-7 C++<->Python decision bridge are separate
channels by design; together they solve the IDLE-time mutual blocking
problem (总体方案 §5.4).
*******************************************************************************/

#ifndef EXECUTION_DRIVEN_REQUESTINGRESS_HH
#define EXECUTION_DRIVEN_REQUESTINGRESS_HH

#include <condition_variable>
#include <cstdint>
#include <deque>
#include <functional>
#include <mutex>
#include <string>
#include <unordered_map>

#include "astra-sim/workload/execution_driven/DecisionMailbox.hh"
#include "astra-sim/workload/execution_driven/ServiceCoordinator.hh"
#include "common/EventQueue.h"

namespace AstraSim {
namespace ExecutionDriven {

struct RequestEnvelope {
    uint64_t ingress_seq = 0;
    std::string session_id;
    int turn_index = 0;
    std::string request_id;
    uint64_t prefill_length = 0;
    uint64_t decode_length = 0;
    uint64_t arrival_world_ns = 0;
    uint64_t inter_request_interval_ns = 0;
    // Phase 4 (schema v1): frozen queue index (CSV data-row order, 0-based;
    // -1 = unknown, defensive only). The CSV loader registers every data
    // row (turn-0 and turn>0) through register_queue_index; future-arrival
    // scheduling looks it up from the map.
    int64_t queue_index = -1;
};

enum class IngressCommandKind { Submit, CloseInput, EndOfFile, Error };

struct IngressCommand {
    IngressCommandKind kind = IngressCommandKind::Submit;
    RequestEnvelope envelope;
};

class RequestIngress {
  public:
    explicit RequestIngress(size_t capacity = 4096) noexcept;

    void bind(NetworkAnalytical::EventQueue* eq, DecisionMailbox* mailbox,
              ServiceCoordinator* svc) noexcept;

    /// Thread-safe, bounded, with backpressure: returns false when the queue
    /// is full (producer must retry) or after the input was closed.
    bool enqueue_command(IngressCommand cmd);

    /// Simulation thread only: pop all pending commands, schedule arrival
    /// alarms, drive IDLE -> ACTIVE accounting.
    void drain_commands();

    /// Step 1-8: schedule a future arrival alarm from inside a commit (the
    /// GraphBatch future_alarms[] contract). Simulation thread only. Direct
    /// schedule_event with the strict-future clamp; accounts the alarm
    /// (on_alarm_scheduled) but not a command acceptance (no
    /// on_command_accepted: this arrival was never a queued command). The
    /// alarm fires through the same arrival_cb as command-driven arrivals
    /// (on_request_arrived + ARRIVAL mailbox deposit).
    void schedule_future_arrival(const RequestEnvelope& envelope);

    /// Thread-safe: close the input (direct ServiceCoordinator call; the
    /// CloseInput command kind is kept for the step-1-7 bridge command stream).
    void mark_input_closed();

    /// Phase 4 (schema v1): register the frozen queue index of one request.
    /// The CSV loader calls this for EVERY data row (turn-0 and the skipped
    /// turn>0 rows alike) so future-arrival scheduling can fill the envelope
    /// from the map. Thread-safe (called from the loader on the main thread).
    void register_queue_index(const std::string& request_id,
                              int64_t queue_index);

    /// Arrival alarm handler (static; event-loop context).
    static void arrival_cb(void* arg);

    /// Fixture/injection hook: invoked by arrival_cb right after the
    /// ServiceCoordinator accounting, on the simulation thread. The IDLE
    /// fixture uses it to complete no-node requests immediately; the
    /// production completion path lands with CompletionObserver (step 1-3).
    void set_arrival_hook(std::function<void(const RequestEnvelope&)> hook);

    [[nodiscard]] size_t pending_command_count() const;

    /// Phase 7 §10.4: how many arrival alarms were clamped to current+1
    /// because their target tick was already in the past (late arrivals).
    /// Observation only -- the clamping itself is the frozen late-arrival
    /// policy, unchanged. Simulation-thread report point.
    size_t late_arrival_count() const { return late_arrival_count_; }

    /// Phase 7 §10.7: overflow audit -- how many enqueue attempts were
    /// rejected because the bounded command queue was full (capacity_).
    /// Thread-safe (matching enqueue_command's lock). 0 on every official
    /// run (the windowed reader tops up at most high_water=128 un-consumed
    /// rows << capacity 4096); the counter exists so a producer that grows
    /// without bound is caught by the run-end audit instead of silently
    /// backpressuring forever.
    size_t overflow_count() const;
    /// Phase 7 §10.7: peak command-queue occupancy (un-consumed commands)
    /// observed since construction. Thread-safe.
    size_t peak_command_occupancy() const;

  private:
    struct ArrivalAlarmArg {
        RequestIngress* ingress;
        RequestEnvelope envelope;
    };

    NetworkAnalytical::EventQueue* eq_ = nullptr;
    DecisionMailbox* mailbox_ = nullptr;
    ServiceCoordinator* svc_ = nullptr;
    std::function<void(const RequestEnvelope&)> arrival_hook_;
    mutable std::mutex mtx_;
    std::deque<IngressCommand> queue_;
    size_t capacity_;
    uint64_t ingress_seq_ = 0;
    // Phase 4 (schema v1): request_id -> frozen queue index (registered for
    // every CSV data row by the loader). Guarded by mtx_.
    std::unordered_map<std::string, int64_t> queue_index_map_;
    // Phase 7 §10.4: late-arrival clamp counter (observation only).
    size_t late_arrival_count_ = 0;
    // Phase 7 §10.7: overflow audit counters (guarded by mtx_).
    size_t overflow_count_ = 0;
    size_t peak_command_occupancy_ = 0;
    bool closed_ = false;
};

}  // namespace ExecutionDriven
}  // namespace AstraSim

#endif  // EXECUTION_DRIVEN_REQUESTINGRESS_HH
