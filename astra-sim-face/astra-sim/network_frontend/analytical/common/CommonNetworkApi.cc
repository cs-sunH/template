/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#include "common/CommonNetworkApi.hh"
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <string>

using namespace AstraSim;
using namespace AstraSimAnalytical;
using namespace NetworkAnalytical;

std::shared_ptr<EventQueue> CommonNetworkApi::event_queue = nullptr;

ChunkIdGenerator CommonNetworkApi::chunk_id_generator = {};

CallbackTracker CommonNetworkApi::callback_tracker = {};

int CommonNetworkApi::dims_count = -1;

std::vector<Bandwidth> CommonNetworkApi::bandwidth_per_dim = {};

void CommonNetworkApi::set_event_queue(
    std::shared_ptr<EventQueue> event_queue_ptr) noexcept {
    assert(event_queue_ptr != nullptr);

    CommonNetworkApi::event_queue = std::move(event_queue_ptr);
}

void CommonNetworkApi::audit_run_end_residuals() {
    // Run-end residual audit (warn-only, never aborts): a quiescent run end
    // should have recycled every ChunkIdGenerator key (each key is erased
    // once its completion gate proves totality) and every CallbackTracker
    // entry (each completion pops it). recv-first keys intentionally stay
    // resident until their sends arrive -- so a resident key at a quiescent
    // run end means a completion never happened. This can be a real defect
    // OR a legal truncated-window shape, hence warn (with counts and key
    // samples) instead of the abort used for mid-run invariant breaks.
    static constexpr size_t max_key_samples = 5;
    const auto chunk_residuals =
        chunk_id_generator.resident_key_samples(max_key_samples);
    const auto tracker_residuals =
        callback_tracker.resident_key_samples(max_key_samples);
    if (chunk_residuals.empty() && tracker_residuals.empty()) {
        return;
    }
    const auto format_chunk_key = [](const auto& key) {
        return "tag=" + std::to_string(std::get<0>(key)) +
               " src=" + std::to_string(std::get<1>(key)) +
               " dest=" + std::to_string(std::get<2>(key)) +
               " size=" + std::to_string(std::get<3>(key));
    };
    const auto format_tracker_key = [](const auto& key) {
        return "tag=" + std::to_string(std::get<0>(key)) +
               " src=" + std::to_string(std::get<1>(key)) +
               " dest=" + std::to_string(std::get<2>(key)) +
               " size=" + std::to_string(std::get<3>(key)) +
               " chunk=" + std::to_string(std::get<4>(key));
    };
    if (!chunk_residuals.empty()) {
        std::fprintf(stderr,
                     "[Warn] (network/analytical) run-end residual audit: "
                     "%zu ChunkIdGenerator key(s) still resident "
                     "(%zu sampled):",
                     chunk_id_generator.size(), chunk_residuals.size());
        for (const auto& key : chunk_residuals) {
            std::fprintf(stderr, " [%s]", format_chunk_key(key).c_str());
        }
        std::fprintf(stderr, "\n");
    }
    if (!tracker_residuals.empty()) {
        std::fprintf(stderr,
                     "[Warn] (network/analytical) run-end residual audit: "
                     "%zu CallbackTracker entry(s) still resident "
                     "(%zu sampled):",
                     callback_tracker.size(), tracker_residuals.size());
        for (const auto& key : tracker_residuals) {
            std::fprintf(stderr, " [%s]", format_tracker_key(key).c_str());
        }
        std::fprintf(stderr, "\n");
    }
}

void CommonNetworkApi::process_chunk_arrival(void* args) noexcept {
    assert(args != nullptr);

    // parse chunk data
    auto* const data =
        static_cast<std::tuple<int, int, int, uint64_t, int>*>(args);
    const auto [tag, src, dest, count, chunk_id] = *data;
    delete data;

    // search tracker
    auto& tracker = callback_tracker;
    const auto entry = tracker.search_entry(tag, src, dest, count, chunk_id);
    if (!entry.has_value()) {
        std::fprintf(stderr,
                     "[Error] (network/analytical) chunk arrival without "
                     "a callback tracker entry (tag=%d src=%d dest=%d "
                     "size=%llu chunk=%d)\n",
                     tag, src, dest,
                     static_cast<unsigned long long>(count), chunk_id);
        std::abort();
    }

    // if both callbacks are registered, invoke both callbacks
    if (entry.value()->both_callbacks_registered()) {
        entry.value()->invoke_send_handler();
        entry.value()->invoke_recv_handler();

        // remove entry
        tracker.pop_entry(tag, src, dest, count, chunk_id);
        CommonNetworkApi::chunk_id_generator.complete(tag, src, dest, count,
                                                       chunk_id);
    } else {
        // run only send callback, as recv is not ready yet.
        entry.value()->invoke_send_handler();

        // mark the transmission as finished
        // so that recv callback will be invoked immediately
        // when sim_recv() is called
        entry.value()->set_transmission_finished();
    }
}

CommonNetworkApi::CommonNetworkApi(const int rank) noexcept
    : AstraNetworkAPI(rank) {
    assert(rank >= 0);
}

CommonNetworkApi::~CommonNetworkApi() {
    // A normally delivered wrapper removes itself before it invokes the target
    // callback.  This loop is only a defensive teardown path for pending
    // opt-in alarms.  cancel_event() synchronously erases the registry entry,
    // so never retain an iterator across the call.
    while (!cancellable_events_.empty()) {
        const uint64_t token = cancellable_events_.begin()->first;
        auto handle = make_cancellable_schedule_handle(token);
        const bool cancelled = sim_cancel_event(handle);
        if (!cancelled) {
            const auto event_it = cancellable_events_.find(token);
            if (event_it != cancellable_events_.end()) {
                cancellable_events_.erase(event_it);
            }
        }
    }
}

timespec_t CommonNetworkApi::sim_get_time() {
    // get current time from event queue
    const auto current_time = event_queue->get_current_time();

    // Keep the uint64 event time exact.  Converting through double loses
    // integer nanoseconds above 2^53; a later completion can then be scheduled
    // under a Sys event-map key that differs by one nanosecond from the network
    // callback time, leaving the hardware resource permanently occupied.
    const auto astra_sim_time = static_cast<long double>(current_time);
    return {NS, astra_sim_time};
}

bool CommonNetworkApi::backend_in_invoke_context() const {
    return event_queue == nullptr || event_queue->in_invoke_context();
}

void CommonNetworkApi::sim_schedule(const timespec_t delta,
                                    void (*fun_ptr)(void*),
                                    void* const fun_arg) {
    // (2026-09-30) The time_res == NS assert is gone: time_type_e has a
    // single enumerator, so the branch could never fire. The time-regression
    // check below was assert-only and compiled out of Release builds -- a
    // negative delta would then insert into a past EventQueue bucket that
    // never fires (a silent hang instead of a named error), so it is now a
    // hard abort (fail-closed in every build type).
    assert(fun_ptr != nullptr);

    // calculate absolute event time
    const auto current_time = sim_get_time();
    const auto event_time = current_time.time_val + delta.time_val;
    // (2026-10-01) range guard before the long double -> uint64
    // conversion: a sum past UINT64_MAX is UB (x86-64 yields the 0x8000...
    // "integer indefinite" sentinel, which then sorts INTO the queue).
    // Unreachable with realistic deltas (< 1e12 ns), but this is the one
    // unguarded conversion left on the README §7 precision chain (same
    // discipline as FluidScheduler's checked_add_time).
    if (!(event_time >= 0.0L) ||
        event_time > static_cast<long double>(
                         std::numeric_limits<EventTime>::max())) {
        std::fprintf(stderr,
                     "[Error] (analytical/common) sim_schedule overflow: "
                     "event_time is outside the uint64 ns range\n");
        std::abort();
    }
    const auto event_time_ns = static_cast<EventTime>(event_time);

    // schedule the event to the event queue
    if (event_time_ns < event_queue->get_current_time()) {
        std::fprintf(stderr,
                     "[Error] (analytical/common) sim_schedule regression: "
                     "event_time=%llu < current_time=%llu (negative delta?)\n",
                     static_cast<unsigned long long>(event_time_ns),
                     static_cast<unsigned long long>(
                         event_queue->get_current_time()));
        std::abort();
    }
    event_queue->schedule_event(event_time_ns, fun_ptr, fun_arg);
}

AstraNetworkAPI::CancellableScheduleHandle
CommonNetworkApi::sim_schedule_cancellable(
    const timespec_t delta,
    void (*fun_ptr)(void*),
    void* const fun_arg,
    const AstraNetworkAPI::ScheduleCancellationCallback cancellation_cleanup) {
    assert(fun_ptr != nullptr);
    if (next_cancellable_schedule_token_ == 0) {
        std::abort();
    }

    // Keep sim_schedule() above on its original fast path.  Only callers that
    // explicitly ask for cancellation pay for a wrapper, registry entry, and
    // EventQueue control block.
    const auto current_time = sim_get_time();
    const auto event_time = current_time.time_val + delta.time_val;
    // (2026-10-01) same pre-conversion range guard as sim_schedule above.
    if (!(event_time >= 0.0L) ||
        event_time > static_cast<long double>(
                         std::numeric_limits<EventTime>::max())) {
        std::fprintf(stderr,
                     "[Error] (analytical/common) sim_schedule_cancellable "
                     "overflow: event_time is outside the uint64 ns range\n");
        std::abort();
    }
    const auto event_time_ns = static_cast<EventTime>(event_time);
    // Release-active regression guard, same rationale as sim_schedule.
    if (event_time_ns < event_queue->get_current_time()) {
        std::fprintf(stderr,
                     "[Error] (analytical/common) sim_schedule_cancellable "
                     "regression: event_time=%llu < current_time=%llu\n",
                     static_cast<unsigned long long>(event_time_ns),
                     static_cast<unsigned long long>(
                         event_queue->get_current_time()));
        std::abort();
    }

    const uint64_t token = next_cancellable_schedule_token_++;
    auto* const context = new CancellableScheduleContext{
        this, token, fun_ptr, fun_arg, cancellation_cleanup};
    auto event_handle = event_queue->schedule_event_cancellable(
        event_time_ns, &CommonNetworkApi::invoke_cancellable_schedule, context,
        &CommonNetworkApi::cancel_cancellable_schedule);
    cancellable_events_.emplace(token, std::move(event_handle));
    return make_cancellable_schedule_handle(token);
}

bool CommonNetworkApi::sim_cancel_event(
    AstraNetworkAPI::CancellableScheduleHandle& handle) {
    if (!owns_cancellable_schedule_handle(handle) || event_queue == nullptr) {
        handle.reset();
        return false;
    }

    const uint64_t token = cancellable_schedule_token(handle);
    const auto event_it = cancellable_events_.find(token);
    if (event_it == cancellable_events_.end()) {
        handle.reset();
        return false;
    }

    // Pass a local copy: the EventQueue cancellation callback erases the map
    // entry synchronously, which would invalidate a reference to its handle.
    auto event_handle = event_it->second;
    const bool cancelled = event_queue->cancel_event(event_handle);
    handle.reset();
    if (!cancelled) {
        const auto stale_event_it = cancellable_events_.find(token);
        if (stale_event_it != cancellable_events_.end()) {
            cancellable_events_.erase(stale_event_it);
        }
    }
    return cancelled;
}

void CommonNetworkApi::invoke_cancellable_schedule(void* const arg) noexcept {
    assert(arg != nullptr);
    auto* const context = static_cast<CancellableScheduleContext*>(arg);
    CommonNetworkApi* const owner = context->owner;
    const uint64_t token = context->token;
    const auto callback = context->callback;
    void* const callback_arg = context->callback_arg;

    // Remove the weak identity before handing control to the target.  The
    // target may re-enter scheduling or destroy its owning Sys/API.
    if (owner != nullptr) {
        owner->cancellable_events_.erase(token);
    }
    delete context;
    callback(callback_arg);
}

void CommonNetworkApi::cancel_cancellable_schedule(void* const arg) noexcept {
    assert(arg != nullptr);
    auto* const context = static_cast<CancellableScheduleContext*>(arg);
    CommonNetworkApi* const owner = context->owner;
    const uint64_t token = context->token;
    const auto cancellation_cleanup = context->cancellation_cleanup;
    void* const callback_arg = context->callback_arg;

    if (owner != nullptr) {
        owner->cancellable_events_.erase(token);
    }
    delete context;
    if (cancellation_cleanup != nullptr) {
        cancellation_cleanup(callback_arg);
    }
}

int CommonNetworkApi::sim_recv(void* const buffer,
                               const uint64_t count,
                               const int type,
                               const int src,
                               const int tag,
                               sim_request* const request,
                               void (*msg_handler)(void*),
                               void* const fun_arg) {
    // Fail closed before any bookkeeping: a null handler would otherwise be
    // registered in the CallbackTracker and later invoked (or scheduled) as
    // a null call. No chunk id is consumed on this path.
    if (msg_handler == nullptr) {
        std::fprintf(stderr,
                     "[Error] (network/analytical) sim_recv requires a "
                     "non-null msg_handler (tag=%d src=%d count=%llu)\n",
                     tag, src, static_cast<unsigned long long>(count));
        std::abort();
    }

    // query chunk id
    const auto dst = sim_comm_get_rank();
    const auto chunk_id =
        CommonNetworkApi::chunk_id_generator.create_recv_chunk_id(tag, src, dst,
                                                                  count);

    // search tracker
    auto entry = callback_tracker.search_entry(tag, src, dst, count, chunk_id);
    if (entry.has_value()) {
        // send() already invoked
        // behavior is decided whether the transmission is already finished or
        // not
        if (entry.value()->is_transmission_finished()) {
            // transmission already finished, run callback immediately

            // pop entry
            callback_tracker.pop_entry(tag, src, dst, count, chunk_id);
            CommonNetworkApi::chunk_id_generator.complete(tag, src, dst,
                                                           count, chunk_id);

            // run recv callback immediately. In the invoke context (legacy
            // arm: backend/ED tests driving the backend directly) the
            // current_time alarm merges into the EventList currently
            // being invoked and executes within the same pass -- pre-extension
            // behavior, byte-for-byte preserved. From a tick-end/deferred
            // context (online post-commit issue pass) a schedule_event at
            // current_time would trip the strict-increase assert on the next
            // proceed(), so the callback goes through the same-tick deferred
            // drain instead (EventQueue hard rule).
            if (event_queue->in_invoke_context()) {
                const auto delta = timespec_t{NS, 0};
                sim_schedule(delta, msg_handler, fun_arg);
            } else {
                event_queue->schedule_event_deferred(msg_handler, fun_arg);
            }
        } else {
            // transmission not finished yet, just register callback
            entry.value()->register_recv_callback(msg_handler, fun_arg);
        }
    } else {
        // send() not yet called
        // create new entry and insert callback
        auto* const new_entry =
            callback_tracker.create_new_entry(tag, src, dst, count, chunk_id);
        new_entry->register_recv_callback(msg_handler, fun_arg);
    }

    // return
    return 0;
}

double CommonNetworkApi::get_BW_at_dimension(const int dim) {
    assert(0 <= dim && dim < dims_count);

    // return bandwidth of the requested dimension
    return bandwidth_per_dim[dim];
}
