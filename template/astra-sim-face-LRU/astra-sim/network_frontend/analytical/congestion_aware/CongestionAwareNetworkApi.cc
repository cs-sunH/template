/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

// <cstdio>/<cstdlib> restored (2026-10-01): the sim_send null-handler hard
// abort below reports through stdio and exits via std::abort.
#include "congestion_aware/CongestionAwareNetworkApi.hh"
#include <cassert>
#include <cstdio>
#include <cstdlib>

using namespace AstraSim;
using namespace AstraSimAnalyticalCongestionAware;
using namespace NetworkAnalytical;
using namespace NetworkAnalyticalCongestionAware;

std::shared_ptr<Topology> CongestionAwareNetworkApi::topology;
std::shared_ptr<FluidScheduler> CongestionAwareNetworkApi::fluid_scheduler;

void CongestionAwareNetworkApi::set_topology(
    std::shared_ptr<Topology> topology_ptr) noexcept {
    assert(topology_ptr != nullptr);

    // move topology
    CongestionAwareNetworkApi::topology = std::move(topology_ptr);
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
    // Null-handler hard abort (fix (2026-10-01), ported from the face repo):
    // sim_recv rejects a null handler with std::abort (a registered null
    // callback would fire as a call through nullptr); sim_send's only
    // defense used to be Event's assert, which Release builds compile out.
    // Same contract, same failure mode, same channel.
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

    // search tracker / create entry (perf item 4.4): both previous branches
    // ended in register_send_callback() on the same entry, so one
    // lower_bound + hinted-emplace walk replaces the find plus the miss
    // branch's emplace. The map contents and their order are unchanged.
    bool entry_created = false;
    auto* const entry = callback_tracker.search_or_create_entry(
        tag, src, dst, count, chunk_id, entry_created);
    entry->register_send_callback(msg_handler, fun_arg);

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
