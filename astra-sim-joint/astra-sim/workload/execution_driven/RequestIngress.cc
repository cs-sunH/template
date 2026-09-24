/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

RequestIngress -- execution-driven mechanism layer (wscllm phase 1).
Implementation (方案 §4 步骤 1-2 操作 2).
*******************************************************************************/

#include "astra-sim/workload/execution_driven/RequestIngress.hh"

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <stdexcept>

using namespace NetworkAnalytical;

namespace AstraSim {
namespace ExecutionDriven {

RequestIngress::RequestIngress(const size_t capacity) noexcept
    : capacity_(capacity) {}

void RequestIngress::bind(EventQueue* const eq, DecisionMailbox* const mailbox,
                          ServiceCoordinator* const svc) noexcept {
    assert(eq != nullptr);
    assert(mailbox != nullptr);
    assert(svc != nullptr);
    eq_ = eq;
    mailbox_ = mailbox;
    svc_ = svc;
}

bool RequestIngress::enqueue_command(IngressCommand cmd) {
    std::lock_guard<std::mutex> lock(mtx_);
    if (closed_) {
        return false;
    }
    if (queue_.size() >= capacity_) {
        // Phase-7 §10.7: overflow audit. The rejection itself is unchanged
        // (bounded queue + backpressure); the counter records HOW OFTEN a
        // producer hit the bound, so a growing producer is visible in the
        // run-end audit instead of only as silent backpressure latency.
        ++overflow_count_;
        return false;  // bounded; backpressure: producer must retry
    }
    const bool terminal = cmd.kind != IngressCommandKind::Submit;
    cmd.envelope.ingress_seq = ingress_seq_++;
    queue_.push_back(std::move(cmd));
    // Linearize terminal input with producers under the same lock. Commands
    // accepted before the terminal remain ahead of it in queue_; every later
    // enqueue observes closed_ and is rejected.
    if (terminal) {
        closed_ = true;
    }
    if (queue_.size() > peak_command_occupancy_) {
        peak_command_occupancy_ = queue_.size();
    }
    if (svc_ != nullptr) {
        svc_->signal_work();
    }
    return true;
}

void RequestIngress::drain_commands() {
    std::deque<IngressCommand> batch;
    {
        std::lock_guard<std::mutex> lock(mtx_);
        batch.swap(queue_);
    }
    for (auto& cmd : batch) {
        switch (cmd.kind) {
            case IngressCommandKind::Submit: {
                // arrival alarms must be strictly future (EventQueue :33);
                // a late request is clamped to the next tick (arrives now)
                const auto current = eq_->get_current_time();
                const auto arrival = cmd.envelope.arrival_world_ns;
                const EventTime alarm_time =
                    (arrival > current) ? arrival : current + 1;
                // P0 turn-0 late-discovery fix (2026-08-30): count the clamps
                // SPLIT by producer path (observation only; the clamp policy
                // itself is frozen and unchanged). A StaticCsv clamp is a
                // reader defect unless it is the pure t=0 boundary case
                // (declared 0, drained at current 0, alarm forced to 1 by the
                // strict-future rule; the source CSV legitimately contains
                // arrival==0) -- that one is counted separately and exempt
                // from the run-end gate. discovered <= current always (a
                // Submit is enqueued at the queue's then-current time), so
                // current==0 also implies the reader discovered it at t=0.
                if (arrival <= current) {
                    if (cmd.envelope.source == RequestSource::StaticCsv) {
                        if (arrival == 0 && current == 0) {
                            ++t0_boundary_clamp_;
                        } else {
                            ++late_static_submit_;
                        }
                    } else if (cmd.envelope.source ==
                               RequestSource::ExternalStream) {
                        ++late_external_stream_;
                    } else {
                        ++late_future_alarm_rounding_;
                    }
                }
                if (cmd.envelope.source == RequestSource::StaticCsv) {
                    // Run-end arrival audit record (bounded by turn-0 rows).
                    std::lock_guard<std::mutex> lock(mtx_);
                    static_csv_arrivals_[cmd.envelope.queue_index] =
                        StaticCsvArrivalRecord{arrival,
                                               static_cast<uint64_t>(
                                                   alarm_time)};
                }
                svc_->on_command_accepted();
                svc_->on_alarm_scheduled();
                auto* arg = new ArrivalAlarmArg{
                    this, std::move(cmd.envelope), false};
                eq_->schedule_event(alarm_time, RequestIngress::arrival_cb,
                                    arg);
                break;
            }
            case IngressCommandKind::CloseInput:
                // explicit close (合同②: submit/close/EOF/error 三态区分;
                // phase-7 §10.7 completes the skeleton -- the terminal
                // command kinds now carry their source into the lifecycle
                // audit instead of collapsing into one close path).
                svc_->mark_input_closed(InputCloseReason::ExplicitClose);
                break;
            case IngressCommandKind::EndOfFile:
                // EOF: the producer declares the input naturally exhausted
                // (distinct from an explicit close -- the run-end audit
                // distinguishes EOF from close per 合同②).
                svc_->mark_input_closed(InputCloseReason::EndOfFile);
                break;
            case IngressCommandKind::Error:
                // 合同②: 异常/错误 → fail-closed abort (进程退出非 0,不做
                // 静默降级). Phase-7 §10.7: previously this kind collapsed
                // into mark_input_closed() (silent degradation -- the exact
                // behavior the contract forbids); no producer in the
                // official path ever sends Error, so this is a skeleton
                // completion, not a behavior change on any official run.
                std::fprintf(stderr,
                             "[Error] (execution_driven/ingress) Error "
                             "command drained: aborting (fail-closed)\n");
                std::abort();
                break;
        }
    }
}

void RequestIngress::enable_queue_index_tracking() {
    std::lock_guard<std::mutex> lock(mtx_);
    queue_index_tracking_enabled_ = true;
}

void RequestIngress::register_queue_index(const std::string& request_id,
                                          const int64_t queue_index) {
    std::lock_guard<std::mutex> lock(mtx_);
    if (request_id.empty() || queue_index < 0 ||
        scheduled_future_request_ids_.count(request_id) != 0 ||
        !queue_index_map_.emplace(request_id, queue_index).second) {
        std::fprintf(stderr,
                     "[Error] (execution_driven/ingress) duplicate or invalid "
                     "future queue index: request_id=%s queue_index=%lld\n",
                     request_id.c_str(), static_cast<long long>(queue_index));
        std::abort();
    }
    queue_index_tracking_enabled_ = true;
}

std::optional<std::string> RequestIngress::validate_future_arrivals_locked(
    const std::vector<RequestEnvelope>& envelopes) const {
    std::unordered_set<std::string> batch_request_ids;
    batch_request_ids.reserve(envelopes.size());
    for (const RequestEnvelope& envelope : envelopes) {
        if (envelope.request_id.empty()) {
            return "future arrival has empty request_id";
        }
        if (envelope.queue_index < -1) {
            return "future arrival request_id " + envelope.request_id +
                   " has invalid queue_index " +
                   std::to_string(envelope.queue_index);
        }
        if (!batch_request_ids.insert(envelope.request_id).second) {
            return "future arrival request_id " + envelope.request_id +
                   " appears more than once in the batch";
        }
        if (scheduled_future_request_ids_.count(envelope.request_id) != 0) {
            return "future arrival request_id " + envelope.request_id +
                   " is already scheduled and awaiting arrival";
        }
        if (!queue_index_tracking_enabled_) {
            continue;
        }
        const auto index_it = queue_index_map_.find(envelope.request_id);
        if (index_it == queue_index_map_.end()) {
            return "future arrival request_id " + envelope.request_id +
                   " has no registered queue index";
        }
        if (envelope.queue_index >= 0 &&
            envelope.queue_index != index_it->second) {
            return "future arrival request_id " + envelope.request_id +
                   " queue_index " + std::to_string(envelope.queue_index) +
                   " does not match registered queue index " +
                   std::to_string(index_it->second);
        }
    }
    return std::nullopt;
}

std::optional<std::string> RequestIngress::validate_future_arrivals(
    const std::vector<RequestEnvelope>& envelopes) const {
    std::lock_guard<std::mutex> lock(mtx_);
    return validate_future_arrivals_locked(envelopes);
}

void RequestIngress::schedule_future_arrival(const RequestEnvelope& envelope) {
    // Allocate and copy the callback payload before reserving the request id,
    // consuming its one-shot queue index, or advancing the ingress serial.
    // A bad_alloc therefore leaves every observable ingress field unchanged.
    std::unique_ptr<ArrivalAlarmArg> arg(
        new ArrivalAlarmArg{this, envelope, true});
    RequestEnvelope& filled = arg->envelope;
    const std::vector<RequestEnvelope> one_envelope{envelope};
    std::optional<std::string> validation_error;
    // Phase 4: resolve the one-shot turn>0 queue index and reserve the id
    // before changing arrival accounting. A successful GraphBatch preflight
    // makes this repetition defensive in the single simulation-thread commit
    // path; keeping it here prevents an unsafe direct caller from consuming a
    // map entry or scheduling a duplicate future arrival.
    {
        std::lock_guard<std::mutex> lock(mtx_);
        validation_error = validate_future_arrivals_locked(one_envelope);
        if (!validation_error.has_value()) {
            const auto inserted =
                scheduled_future_request_ids_.insert(envelope.request_id);
            if (!inserted.second) {
                validation_error =
                    "future arrival request_id " + envelope.request_id +
                    " is already scheduled and awaiting arrival";
            }
        }
        if (!validation_error.has_value() && queue_index_tracking_enabled_) {
            const auto it = queue_index_map_.find(envelope.request_id);
            assert(it != queue_index_map_.end());
            filled.queue_index = it->second;
            queue_index_map_.erase(it);
        }
        if (!validation_error.has_value()) {
            // enqueue_command assigns the same serial while holding mtx_. Keep
            // future scheduling under that lock too: external command producers
            // may run concurrently with the simulation thread.
            filled.ingress_seq = ingress_seq_++;
        }
    }
    if (validation_error.has_value()) {
        throw std::runtime_error("future-arrival scheduling rejected: " +
                                 *validation_error);
    }
    // arrival alarms must be strictly future (EventQueue :33); a late
    // arrival is clamped to the next tick (arrives now)
    const auto current = eq_->get_current_time();
    const auto arrival = envelope.arrival_world_ns;
    // P0 fix (2026-08-30): this path only ever schedules FutureAlarm
    // envelopes; its clamps are relative-interval sub-tick rounding, never a
    // static-CSV reader defect, and are never gated.
    filled.source = RequestSource::FutureAlarm;
    if (arrival <= current) {
        ++late_future_alarm_rounding_;
    }
    const EventTime alarm_time =
        (arrival > current) ? arrival : current + 1;
    // Phase 4 (schema v1): every arrival gets a fresh, globally monotonic
    // ingress serial -- future-arrival scheduling assigns it here (the
    // command path assigns it in enqueue_command). The frozen queue index
    // comes from the loader's per-request map (the future_alarm envelope
    // from Python carries no queue_index).
    eq_->schedule_event(alarm_time, RequestIngress::arrival_cb, arg.get());
    (void)arg.release();  // EventQueue callback now owns and deletes it.
    svc_->on_alarm_scheduled();
}

void RequestIngress::mark_input_closed() {
    {
        std::lock_guard<std::mutex> lock(mtx_);
        if (closed_) {
            return;
        }
        // Set the ingress gate before closing the coordinator. A producer is
        // therefore either ordered before this close (and left in queue_ for
        // the drain-first loop) or rejected after it; there is no gap in which
        // svc.finished() can become true while a new command is still accepted.
        closed_ = true;
    }
    if (svc_ != nullptr) {
        svc_->mark_input_closed();
    }
}

void RequestIngress::arrival_cb(void* const arg) {
    auto* const data = static_cast<ArrivalAlarmArg*>(arg);
    auto* const ingress = data->ingress;
    if (data->is_future_alarm) {
        std::lock_guard<std::mutex> lock(ingress->mtx_);
        ingress->scheduled_future_request_ids_.erase(data->envelope.request_id);
    }
    ingress->svc_->on_request_arrived();
    // Step 1-6: the DecisionMailbox(ARRIVAL, payload) deposit -- the only
    // decision event the ingress produces (a submit is a decision boundary,
    // not an execution event). Same-tick same-identity dedup is the
    // mailbox's own contract.
    DecisionEvent ev;
    ev.reason = DecisionReason::ARRIVAL;
    ev.request_id = data->envelope.request_id;
    ev.stage.clear();  // arrival is request-level, not stage-level
    ev.generation = 0;
    ev.payload.session_id = data->envelope.session_id;
    ev.payload.turn_index = data->envelope.turn_index;
    ev.payload.prefill_length = data->envelope.prefill_length;
    ev.payload.decode_length = data->envelope.decode_length;
    ev.payload.inter_request_interval_ns =
        data->envelope.inter_request_interval_ns;
    ev.payload.arrival_world_ns = data->envelope.arrival_world_ns;
    // Phase 4 (schema v1): the arrival carries its global ingress serial
    // and the frozen queue index (set by the loader / future-arrival
    // scheduling on the envelope).
    ev.payload.ingress_seq = data->envelope.ingress_seq;
    ev.payload.queue_index = data->envelope.queue_index;
    ingress->mailbox_->push(std::move(ev));
    if (ingress->arrival_hook_) {
        ingress->arrival_hook_(data->envelope);
    }
    delete data;
}

void RequestIngress::set_arrival_hook(
    std::function<void(const RequestEnvelope&)> hook) {
    arrival_hook_ = std::move(hook);
}

size_t RequestIngress::pending_command_count() const {
    std::lock_guard<std::mutex> lock(mtx_);
    return queue_.size();
}

size_t RequestIngress::overflow_count() const {
    std::lock_guard<std::mutex> lock(mtx_);
    return overflow_count_;
}

size_t RequestIngress::peak_command_occupancy() const {
    std::lock_guard<std::mutex> lock(mtx_);
    return peak_command_occupancy_;
}

size_t RequestIngress::pending_queue_index_count() const {
    std::lock_guard<std::mutex> lock(mtx_);
    return queue_index_map_.size();
}

std::map<int64_t, RequestIngress::StaticCsvArrivalRecord>
RequestIngress::static_csv_arrivals() const {
    std::lock_guard<std::mutex> lock(mtx_);
    return static_csv_arrivals_;
}

}  // namespace ExecutionDriven
}  // namespace AstraSim
