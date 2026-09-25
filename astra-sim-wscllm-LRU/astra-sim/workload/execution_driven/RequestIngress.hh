/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

RequestIngress -- execution-driven mechanism layer (wscllm phase 1).

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

#include <cassert>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <functional>
#include <map>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "astra-sim/workload/execution_driven/DecisionMailbox.hh"
#include "astra-sim/workload/execution_driven/ServiceCoordinator.hh"
#include "common/EventQueue.h"

namespace AstraSim {
namespace ExecutionDriven {

/// P0 turn-0 late-discovery fix (2026-08-30): the provenance of an arrival --
/// which producer path scheduled it. The late-arrival clamp counters are split
/// by this enum (see RequestIngress): a finite static-CSV run must show
/// late_static_submit == 0 at run end (fail-closed gate), so clamps coming
/// from the windowed/calendar reader must never be conflated with external
/// stream lateness or future-alarm sub-tick rounding.
enum class RequestSource {
    /// Turn-0 Submit produced by WindowedTraceReader (static CSV input).
    StaticCsv,
    /// Arrival scheduled by a REQUEST_COMPLETE commit (schedule_future_arrival).
    FutureAlarm,
    /// Any other producer: the --command-fifo injection thread, fixtures.
    ExternalStream,
};

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
    // -1 = unknown, defensive only). Turn-0 rows carry this value directly in
    // their Submit envelope.  Only turn>0 rows need the one-shot lookup used
    // by future-arrival scheduling.
    int64_t queue_index = -1;
    // P0 turn-0 late-discovery fix (2026-08-30): producer path of this
    // arrival. Default ExternalStream (fixtures / command FIFO); the
    // WindowedTraceReader marks its turn-0 Submits StaticCsv;
    // schedule_future_arrival marks its envelopes FutureAlarm.
    RequestSource source = RequestSource::ExternalStream;
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

    /// Read-only, thread-safe preflight for one commit's future alarms. It
    /// rejects duplicate/empty request ids, ids already reserved by an
    /// outstanding future alarm, and (when strict queue-index tracking is
    /// enabled) missing or mismatched one-shot queue-index registrations.
    /// The simulation thread is the sole future-alarm scheduler, so a
    /// successful preflight immediately followed by schedule_future_arrival()
    /// cannot race another commit; schedule_future_arrival() repeats the
    /// checks defensively.
    [[nodiscard]] std::optional<std::string> validate_future_arrivals(
        const std::vector<RequestEnvelope>& envelopes) const;

    /// Thread-safe: atomically close the producer gate, then close the
    /// ServiceCoordinator. Commands linearized before this call remain queued
    /// for the drain-first loop; later commands are rejected.
    void mark_input_closed();

    /// Enable strict one-shot queue-index resolution for a production CSV
    /// input. Once enabled, every future arrival must consume a registered
    /// turn>0 entry, even before the first such row has been read.
    void enable_queue_index_tracking();

    /// Phase 4 (schema v1): register the frozen queue index of one turn>0
    /// request.  Future-arrival scheduling consumes this entry exactly once;
    /// duplicate registration or a missing lookup fails closed. Thread-safe
    /// (called from the loader on the main thread).
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

    /// Phase 7 §10.4 / P0 fix (2026-08-30): how many arrival alarms were
    /// clamped to current+1 because their target tick was already in the past
    /// (late arrivals). TOTAL across the split counters below -- kept for
    /// report compatibility. Observation only -- the clamping itself is the
    /// frozen late-arrival policy, unchanged. Simulation-thread report point.
    size_t late_arrival_count() const {
        return late_static_submit_ + late_external_stream_ +
               late_future_alarm_rounding_ + t0_boundary_clamp_;
    }
    /// P0 fix (2026-08-30): late-arrival clamp counters SPLIT by producer
    /// path. late_static_submit_ counts Submit-path clamps of
    /// RequestSource::StaticCsv envelopes (turn-0 rows read from the request
    /// queue CSV) EXCLUDING the t=0 boundary case (see t0_boundary_clamp_):
    /// a static CSV is finite and fully indexed before the simulation starts,
    /// so ANY such clamp means the reader discovered the row too late -- the
    /// defect this fix eradicates -- and the run-end gate fail-closes on it.
    size_t late_static_submit_count() const { return late_static_submit_; }
    /// Submit-path clamps of RequestSource::ExternalStream envelopes (the
    /// command-FIFO producer / fixtures may legitimately run late; no gate).
    size_t late_external_stream_count() const { return late_external_stream_; }
    /// Clamps inside schedule_future_arrival (RequestSource::FutureAlarm):
    /// sub-tick rounding of relative-interval scheduling, not a reader defect.
    size_t late_future_alarm_rounding_count() const {
        return late_future_alarm_rounding_;
    }
    /// The t=0 boundary exemption (design ruling 2026-08-30, REPORT item): a
    /// turn-0 row declared arrival==0, discovered at simulation time 0, gets
    /// its alarm clamped from 0 to current+1 == 1 purely because EventQueue
    /// requires strictly future ticks and the source CSV legitimately contains
    /// arrival==0. Counted separately, NOT in late_static_submit_, and exempt
    /// from the run-end static-arrival gate.
    size_t t0_boundary_clamp_count() const { return t0_boundary_clamp_; }

    /// P0 fix (2026-08-30): simulation clock of the bound EventQueue
    /// (asserts bind() happened). Lets the reader stamp each turn-0 Submit
    /// with its discovery tick without reaching into the queue itself.
    NetworkAnalytical::EventTime current_time() const {
        assert(eq_ != nullptr);
        return eq_->get_current_time();
    }

    /// P0 fix (2026-08-30): per-turn-0 arrival timing record for the run-end
    /// arrival audit. declared = envelope.arrival_world_ns at Submit drain;
    /// effective = the actually scheduled alarm tick (max(declared,
    /// current+1); effective >= declared always). Keyed by the frozen queue
    /// index; only RequestSource::StaticCsv Submits are recorded, so the map
    /// is bounded by the turn-0 row count, not by total rows. Guarded by
    /// mtx_; returns a copy.
    struct StaticCsvArrivalRecord {
        uint64_t declared_arrival_ns = 0;
        uint64_t effective_arrival_ns = 0;
    };
    std::map<int64_t, StaticCsvArrivalRecord> static_csv_arrivals() const;

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
        bool is_future_alarm = false;
    };

    /// mtx_ must be held. Shared by the public pure preflight and the
    /// defensive recheck in schedule_future_arrival().
    [[nodiscard]] std::optional<std::string>
    validate_future_arrivals_locked(
        const std::vector<RequestEnvelope>& envelopes) const;

    NetworkAnalytical::EventQueue* eq_ = nullptr;
    DecisionMailbox* mailbox_ = nullptr;
    ServiceCoordinator* svc_ = nullptr;
    std::function<void(const RequestEnvelope&)> arrival_hook_;
    mutable std::mutex mtx_;
    std::deque<IngressCommand> queue_;
    size_t capacity_;
    uint64_t ingress_seq_ = 0;
    // Phase 4 (schema v1): turn>0 request_id -> frozen queue index. Entries
    // are consumed and erased by schedule_future_arrival. Guarded by mtx_.
    std::unordered_map<std::string, int64_t> queue_index_map_;
    // Request ids for successfully scheduled future alarms which have not
    // fired yet. Guarded by mtx_; arrival_cb removes the entry before it
    // publishes ARRIVAL. This closes cross-batch duplicate alarms without
    // retaining historical request ids after arrival.
    std::unordered_set<std::string> scheduled_future_request_ids_;
    // Fixtures that do not use a CSV reader retain the legacy unknown-index
    // behavior. A non-empty WindowedTraceReader enables strict mode before it
    // reads any rows; register_queue_index also enables it defensively.
    bool queue_index_tracking_enabled_ = false;
    // P0 fix (2026-08-30): late-arrival clamp counters split by producer
    // path (see the public accessors above for the exact semantics).
    size_t late_static_submit_ = 0;
    size_t late_external_stream_ = 0;
    size_t late_future_alarm_rounding_ = 0;
    size_t t0_boundary_clamp_ = 0;
    // P0 fix (2026-08-30): per-turn-0 arrival timing records (queue_index ->
    // declared/effective), for the reader's run-end arrival audit. Guarded by
    // mtx_; written by the simulation thread at Submit drain time.
    std::map<int64_t, StaticCsvArrivalRecord> static_csv_arrivals_;
    // Phase 7 §10.7: overflow audit counters (guarded by mtx_).
    size_t overflow_count_ = 0;
    size_t peak_command_occupancy_ = 0;
    // Producer terminal gate, guarded by mtx_. Set by direct close or when a
    // CloseInput/EndOfFile/Error command is accepted into queue_.
    bool closed_ = false;
};

}  // namespace ExecutionDriven
}  // namespace AstraSim

#endif  // EXECUTION_DRIVEN_REQUESTINGRESS_HH
