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
    cmd.envelope.ingress_seq = ingress_seq_++;
    queue_.push_back(std::move(cmd));
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
                // Phase 7 §10.4: count the clamps (observation only; the
                // clamp policy itself is frozen and unchanged).
                if (arrival <= current) {
                    ++late_arrival_count_;
                }
                const EventTime alarm_time =
                    (arrival > current) ? arrival : current + 1;
                svc_->on_command_accepted();
                svc_->on_alarm_scheduled();
                auto* arg = new ArrivalAlarmArg{this, std::move(cmd.envelope)};
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

void RequestIngress::register_queue_index(const std::string& request_id,
                                          const int64_t queue_index) {
    std::lock_guard<std::mutex> lock(mtx_);
    queue_index_map_[request_id] = queue_index;
}

void RequestIngress::schedule_future_arrival(const RequestEnvelope& envelope) {
    // arrival alarms must be strictly future (EventQueue :33); a late
    // arrival is clamped to the next tick (arrives now)
    const auto current = eq_->get_current_time();
    const auto arrival = envelope.arrival_world_ns;
    // Phase 7 §10.4: count the clamps (observation only; the clamp policy
    // itself is frozen and unchanged).
    if (arrival <= current) {
        ++late_arrival_count_;
    }
    const EventTime alarm_time =
        (arrival > current) ? arrival : current + 1;
    svc_->on_alarm_scheduled();
    RequestEnvelope filled = envelope;
    // Phase 4 (schema v1): every arrival gets a fresh, globally monotonic
    // ingress serial -- future-arrival scheduling assigns it here (the
    // command path assigns it in enqueue_command). The frozen queue index
    // comes from the loader's per-request map (the future_alarm envelope
    // from Python carries no queue_index).
    filled.ingress_seq = ingress_seq_++;
    {
        std::lock_guard<std::mutex> lock(mtx_);
        const auto it = queue_index_map_.find(envelope.request_id);
        if (it != queue_index_map_.end()) {
            filled.queue_index = it->second;
        }
    }
    auto* arg = new ArrivalAlarmArg{this, std::move(filled)};
    eq_->schedule_event(alarm_time, RequestIngress::arrival_cb, arg);
}

void RequestIngress::mark_input_closed() {
    if (svc_ != nullptr) {
        svc_->mark_input_closed();
    }
}

void RequestIngress::arrival_cb(void* const arg) {
    auto* const data = static_cast<ArrivalAlarmArg*>(arg);
    auto* const ingress = data->ingress;
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

}  // namespace ExecutionDriven
}  // namespace AstraSim
