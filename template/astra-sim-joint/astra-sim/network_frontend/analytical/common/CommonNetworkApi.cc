/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#include "common/CommonNetworkApi.hh"
#include <cassert>
#include <cstdio>
#include <cstdlib>

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

CallbackTracker& CommonNetworkApi::get_callback_tracker() noexcept {
    return callback_tracker;
}

void CommonNetworkApi::process_chunk_arrival(void* args) noexcept {
    assert(args != nullptr);

    // parse chunk data
    auto* const data =
        static_cast<std::tuple<int, int, int, uint64_t, int>*>(args);
    const auto [tag, src, dest, count, chunk_id] = *data;
    delete data;

    // search tracker
    auto& tracker = CommonNetworkApi::get_callback_tracker();
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
    // The token registry is gone (batch 3, plan 4.1): cancellable payloads
    // now live directly in the backend QueuedEvent, whose own teardown
    // safety net (~QueuedEvent runs the cancellation cleanup exactly once)
    // covers any alarm still pending when the event queue is destroyed.
}

timespec_t CommonNetworkApi::sim_get_time() {
    // get current time from event queue
    const auto current_time = event_queue->get_current_time();

    // Keep the uint64 event time exact.  Converting through double loses
    // integer nanoseconds above 2^53; a later completion can then be scheduled
    // under a Sys event-map key that differs by one nanosecond from the network
    // callback time, leaving the hardware resource permanently occupied.
    // Platform note: this exactness argument relies on long double carrying
    // the full 64-bit mantissa (x86-64's 80-bit type). On ABIs where long
    // double is a 64-bit double (MSVC) the precision is lost again above
    // 2^53 -- porting to those targets needs a fixed-width integer time path.
    const auto astra_sim_time = static_cast<long double>(current_time);
    return {NS, astra_sim_time};
}

bool CommonNetworkApi::backend_in_invoke_context() const {
    return event_queue == nullptr || event_queue->in_invoke_context();
}

void CommonNetworkApi::sim_schedule(const timespec_t delta,
                                    void (*fun_ptr)(void*),
                                    void* const fun_arg) {
    assert(delta.time_res == NS);
    assert(fun_ptr != nullptr);

    // calculate absolute event time
    const auto current_time = sim_get_time();
    const auto event_time = current_time.time_val + delta.time_val;
    const auto event_time_ns = static_cast<EventTime>(event_time);

    // schedule the event to the event queue
    assert(event_time_ns >= event_queue->get_current_time());
    event_queue->schedule_event(event_time_ns, fun_ptr, fun_arg);
}

AstraNetworkAPI::CancellableScheduleHandle
CommonNetworkApi::sim_schedule_cancellable(
    const timespec_t delta,
    void (*fun_ptr)(void*),
    void* const fun_arg,
    const AstraNetworkAPI::ScheduleCancellationCallback cancellation_cleanup) {
    assert(delta.time_res == NS);
    assert(fun_ptr != nullptr);

    // The user callback/argument/cleanup go straight into the backend
    // QueuedEvent: its cancellation contract is what the deleted token-registry
    // wrapper used to re-implement on top of it.
    const auto current_time = sim_get_time();
    const auto event_time = current_time.time_val + delta.time_val;
    const auto event_time_ns = static_cast<EventTime>(event_time);
    assert(event_time_ns >= event_queue->get_current_time());
    auto event_handle = event_queue->schedule_event_cancellable(
        event_time_ns, fun_ptr, fun_arg, cancellation_cleanup);
    return make_cancellable_schedule_handle(std::move(event_handle));
}

bool CommonNetworkApi::sim_cancel_event(
    AstraNetworkAPI::CancellableScheduleHandle& handle) {
    if (!owns_cancellable_schedule_handle(handle) || event_queue == nullptr) {
        handle.reset();
        return false;
    }
    const bool cancelled =
        event_queue->cancel_event(cancellable_schedule_event_handle(handle));
    handle.reset();
    return cancelled;
}

int CommonNetworkApi::sim_recv(void* const buffer,
                               const uint64_t count,
                               const int type,
                               const int src,
                               const int tag,
                               sim_request* const request,
                               void (*msg_handler)(void*),
                               void* const fun_arg) {
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

            // run recv callback immediately. In the invoke context (static
            // path) the current_time alarm merges into the EventList currently
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
        // Count-mismatch diagnostic (2026-09-30, 自 wscllm 母仓 C3 同款移植):
        // the review's "zero diagnostics" hang. If a same-(tag,src,dst)
        // entry with a DIFFERENT count is outstanding, this registration is
        // the wrong half of a count-mismatched pair: both entries now wait
        // on each other forever (the send's chunk arrival only completes its
        // own count key) and the run used to hang silently. Collective tags
        // are per-stream (stream_id), so a same-key different-count entry
        // cannot belong to a legitimate concurrent transfer. Diagnostic
        // only: warn once per logical transfer (chunk 0), then register
        // normally -- the run's outcome is unchanged, the mismatch is just
        // visible.
        if (chunk_id == 0 && CommonNetworkApi::callback_tracker
                                 .has_entry_with_mismatched_count(tag, src,
                                                                  dst,
                                                                  count)) {
            std::fprintf(stderr,
                         "[Warning] (network/analytical) send/recv count "
                         "mismatch suspected: sim_recv(tag=%d src=%d "
                         "dst=%d count=%llu) does not match an "
                         "outstanding entry with the same key but a "
                         "different count -- the pair will wait on each "
                         "other forever (silent-hang signature)\n",
                         tag, src, dst,
                         static_cast<unsigned long long>(count));
        }
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
