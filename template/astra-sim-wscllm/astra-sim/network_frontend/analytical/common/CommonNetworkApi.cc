/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#include "common/CommonNetworkApi.hh"
#include <cassert>
#include <cstdio>
#include <limits>
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
    // Perf item 4.4 (handle search): one find returns the entry and an
    // O(1)-pop iterator handle; the pop below reuses the handle instead of
    // re-walking the tree by key. The map node is removed at exactly the
    // same point in the sequence (after both callbacks returned), so
    // re-entrant lookups during the callbacks observe the same state as
    // before.
    const auto found =
        tracker.find_entry_with_handle(tag, src, dest, count, chunk_id);
    if (found.entry == nullptr) {
        std::fprintf(stderr,
                     "[Error] (network/analytical) chunk arrival without "
                     "a callback tracker entry (tag=%d src=%d dest=%d "
                     "size=%llu chunk=%d)\n",
                     tag, src, dest,
                     static_cast<unsigned long long>(count), chunk_id);
        std::abort();
    }

    // if both callbacks are registered, invoke both callbacks
    if (found.entry->both_callbacks_registered()) {
        found.entry->invoke_send_handler();
        found.entry->invoke_recv_handler();

        // remove entry
        tracker.pop_entry(found.handle);
        CommonNetworkApi::chunk_id_generator.complete(tag, src, dest, count,
                                                       chunk_id);
    } else {
        // run only send callback, as recv is not ready yet.
        found.entry->invoke_send_handler();

        // mark the transmission as finished
        // so that recv callback will be invoked immediately
        // when sim_recv() is called
        found.entry->set_transmission_finished();
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

void CommonNetworkApi::sim_schedule(const timespec_t delta,
                                    void (*fun_ptr)(void*),
                                    void* const fun_arg) {
    assert(delta.time_res == NS);
    assert(fun_ptr != nullptr);

    // P2-10 fix (2026-09-28): same-tick dual-route, aligned with sim_recv
    // below. From a tick-end/deferred context (online post-commit issue
    // pass, tick-end observer) a delta==0 schedule_event would insert a
    // current_time EventList into the main map and trip the
    // strict-increase abort on the next proceed() (EventQueue hard rule;
    // Release-always fail-closed there), so the callback goes through the
    // same-tick deferred drain instead. In the invoke context (static path)
    // the current_time schedule merges into the EventList currently being
    // invoked and executes within the same pass -- pre-extension behavior,
    // byte-for-byte preserved.
    if (delta.time_val == 0 && !event_queue->in_invoke_context()) {
        event_queue->schedule_event_deferred(fun_ptr, fun_arg);
        return;
    }

    // calculate absolute event time
    const auto current_time = sim_get_time();
    const auto event_time = current_time.time_val + delta.time_val;

    // (2026-10-01, D1 review / face parity) Range guard before the long
    // double -> uint64 conversion: a sum past UINT64_MAX (or a NaN delta)
    // is UB in the conversion itself. NaN also hides from every ordered
    // comparison below, so it is caught here via the negated form.
    if (!(event_time >= 0.0L) ||
        event_time > static_cast<long double>(
                         std::numeric_limits<EventTime>::max())) {
        std::fprintf(stderr,
                     "[Error] (analytical/common) sim_schedule overflow: "
                     "event_time is outside the uint64 ns range\n");
        std::abort();
    }

    // schedule the event to the event queue
    // Release-always guard (2026-09-30 face parity): the old assert-only
    // check vanished under NDEBUG, and a negative delta then landed in a
    // PAST map bucket -- the failure surfaced only at the next proceed()
    // as the EventQueue's "same-tick events must go through
    // schedule_event_deferred" abort, misattributing the root cause.
    // (2026-10-01 refinement, B1 review) The comparison happens in the
    // long double domain BEFORE the EventTime cast: a strongly negative
    // delta makes the float->unsigned conversion itself UB (wrapping to a
    // huge value on typical targets) and would slip past a post-cast
    // comparison.
    if (event_time < static_cast<long double>(event_queue->get_current_time())) {
        std::fprintf(stderr,
                     "[Error] (analytical/common) sim_schedule regression: "
                     "event_time=%Lf < current_time=%llu (negative delta?)\n",
                     static_cast<long double>(event_time),
                     static_cast<unsigned long long>(
                         event_queue->get_current_time()));
        std::abort();
    }
    const auto event_time_ns = static_cast<EventTime>(event_time);
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
    if (next_cancellable_schedule_token_ == 0) {
        std::abort();
    }

    // Keep sim_schedule() above on its original fast path.  Only callers that
    // explicitly ask for cancellation pay for a wrapper, registry entry, and
    // EventQueue control block.
    const auto current_time = sim_get_time();
    auto event_time = current_time.time_val + delta.time_val;
    // P2-10 fix (2026-09-28): same-tick companion of the sim_schedule
    // dual-route above. A cancellable event cannot take the deferred drain
    // (its handle must stay resident in -- and cancellable from -- the main
    // map), so from a tick-end/deferred context a delta==0 request takes
    // the +1 ns strict-future lower bound instead (the EventQueue
    // strict-increase rule; the same clamp family as the ingress
    // late-arrival rule). The invoke-context arm keeps the exact
    // current_time merge semantics.
    if (delta.time_val == 0 && !event_queue->in_invoke_context()) {
        event_time = current_time.time_val + 1;
    }
    // (2026-10-01, D1 review / face parity) Same pre-cast range guard as
    // sim_schedule above (NaN / > UINT64_MAX are UB in the conversion).
    if (!(event_time >= 0.0L) ||
        event_time > static_cast<long double>(
                         std::numeric_limits<EventTime>::max())) {
        std::fprintf(stderr,
                     "[Error] (analytical/common) sim_schedule_cancellable "
                     "overflow: event_time is outside the uint64 ns "
                     "range\n");
        std::abort();
    }
    // Release-always guard, same refined form as sim_schedule above
    // (2026-10-01): compare in the long double domain BEFORE the
    // EventTime cast, so a strongly negative delta cannot wrap through
    // the UB float->unsigned conversion and slip past.
    if (event_time < static_cast<long double>(event_queue->get_current_time())) {
        std::fprintf(stderr,
                     "[Error] (analytical/common) sim_schedule_cancellable "
                     "regression: event_time=%Lf < current_time=%llu "
                     "(negative delta?)\n",
                     static_cast<long double>(event_time),
                     static_cast<unsigned long long>(
                         event_queue->get_current_time()));
        std::abort();
    }
    const auto event_time_ns = static_cast<EventTime>(event_time);

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
    // Fail closed before any bookkeeping (2026-09-30 face parity): a null
    // handler would otherwise be registered in the CallbackTracker and
    // later invoked (or scheduled) as a call through nullptr. No chunk id
    // is consumed on this path.
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
    // Perf item 4.4 (handle search): one find returns the entry and an
    // O(1)-pop iterator handle; the pop on the finished path below reuses
    // the handle instead of re-walking the tree by key.
    const auto found =
        callback_tracker.find_entry_with_handle(tag, src, dst, count, chunk_id);
    if (found.entry != nullptr) {
        // send() already invoked
        // behavior is decided whether the transmission is already finished or
        // not
        if (found.entry->is_transmission_finished()) {
            // transmission already finished, run callback immediately

            // pop entry (O(1) erase through the handle found above)
            callback_tracker.pop_entry(found.handle);
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
            found.entry->register_recv_callback(msg_handler, fun_arg);
        }
    } else {
        // send() not yet called
        // create new entry and insert callback
        // C3 fix (2026-09-28): the review's "zero diagnostics" hang. If a
        // same-(tag,src,dst) entry with a DIFFERENT count is outstanding,
        // this registration is the wrong half of a count-mismatched pair:
        // both entries now wait on each other forever (the send's chunk
        // arrival only completes its own count key) and the run used to
        // hang silently. Collective tags are per-stream (stream_id), so a
        // same-key different-count entry cannot belong to a legitimate
        // concurrent transfer. Diagnostic only: warn once per logical
        // transfer (chunk 0), then register normally -- the run's outcome
        // is unchanged, the mismatch is just visible.
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
