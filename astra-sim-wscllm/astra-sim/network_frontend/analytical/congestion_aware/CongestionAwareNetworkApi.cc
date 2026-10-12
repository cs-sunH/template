/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#include "congestion_aware/CongestionAwareNetworkApi.hh"
#include <cassert>
#include <cstdio>

using namespace AstraSim;
using namespace AstraSimAnalyticalCongestionAware;
using namespace NetworkAnalytical;
using namespace NetworkAnalyticalCongestionAware;

std::shared_ptr<Topology> CongestionAwareNetworkApi::topology;
std::shared_ptr<FluidScheduler> CongestionAwareNetworkApi::fluid_scheduler;

void CongestionAwareNetworkApi::set_topology(
    std::shared_ptr<Topology> topology_ptr) noexcept {
    assert(topology_ptr != nullptr);

    CongestionAwareNetworkApi::topology = std::move(topology_ptr);

    CongestionAwareNetworkApi::dims_count =
        CongestionAwareNetworkApi::topology->get_dims_count();
    CongestionAwareNetworkApi::bandwidth_per_dim =
        CongestionAwareNetworkApi::topology->get_bandwidth_per_dim();
}

void CongestionAwareNetworkApi::set_fluid_scheduler(
    std::shared_ptr<FluidScheduler> scheduler) noexcept {
    assert(scheduler != nullptr);
    CongestionAwareNetworkApi::fluid_scheduler = std::move(scheduler);
}

CongestionAwareNetworkApi::CongestionAwareNetworkApi(const int rank) noexcept
    : CommonNetworkApi(rank) {
    assert(rank >= 0);
}

int CongestionAwareNetworkApi::sim_send(void* const buffer,
                                        const uint64_t count,
                                        const int type,
                                        const int dst,
                                        const int tag,
                                        sim_request* const request,
                                        void (*msg_handler)(void*),
                                        void* const fun_arg) {
    // Fail closed before any bookkeeping (2026-09-30 face parity): a null
    // handler would otherwise be registered in the CallbackTracker and
    // later invoked as a call through nullptr. sim_send's only defense used
    // to be Event's assert, which Release builds compile out. No chunk id
    // is consumed on this path.
    if (msg_handler == nullptr) {
        std::fprintf(stderr,
                     "[Error] (analytical/congestion_aware) sim_send called "
                     "with a null msg_handler (tag=%d dst=%d count=%llu); "
                     "the completion callback would invoke nullptr\n",
                     tag, dst, static_cast<unsigned long long>(count));
        std::abort();
    }
    // query chunk id
    const auto src = sim_comm_get_rank();
    const auto chunk_id =
        CongestionAwareNetworkApi::chunk_id_generator.create_send_chunk_id(
            tag, src, dst, count);

    // search tracker
    const auto entry =
        callback_tracker.search_entry(tag, src, dst, count, chunk_id);
    if (entry.has_value()) {
        // recv operation already issued.
        // register send callback
        entry.value()->register_send_callback(msg_handler, fun_arg);
    } else {
        // recv operation not issued yet
        // create new entry and insert callback
        // C3 fix (2026-09-28): send-side companion of the count-mismatch
        // diagnostic in CommonNetworkApi::sim_recv -- the mismatch is
        // visible from whichever half registers second. Warn once per
        // logical transfer (chunk 0); simulation behavior unchanged.
        if (chunk_id == 0 && callback_tracker.has_entry_with_mismatched_count(
                                 tag, src, dst, count)) {
            std::fprintf(stderr,
                         "[Warning] (network/analytical) send/recv count "
                         "mismatch suspected: sim_send(tag=%d src=%d "
                         "dst=%d count=%llu) does not match an "
                         "outstanding entry with the same key but a "
                         "different count -- the pair will wait on each "
                         "other forever (silent-hang signature)\n",
                         tag, src, dst,
                         static_cast<unsigned long long>(count));
        }
        auto* const new_entry =
            callback_tracker.create_new_entry(tag, src, dst, count, chunk_id);
        new_entry->register_send_callback(msg_handler, fun_arg);
    }

    // Keep one tracker entry and one completion argument per logical send.
    // Fluid-internal rate changes never enter CallbackTracker.
    auto chunk_arrival_arg = std::tuple(tag, src, dst, count, chunk_id);
    auto arg = std::make_unique<decltype(chunk_arrival_arg)>(chunk_arrival_arg);
    const auto arg_ptr = static_cast<void*>(arg.release());

    assert(fluid_scheduler != nullptr);
    fluid_scheduler->start_flow(
        count, topology->fluid_route(src, dst),
        CongestionAwareNetworkApi::process_chunk_arrival, arg_ptr);

    return 0;
}
