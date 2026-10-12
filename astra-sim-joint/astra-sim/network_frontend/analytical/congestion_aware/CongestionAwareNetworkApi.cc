/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#include "congestion_aware/CongestionAwareNetworkApi.hh"
#include <cassert>
#include <chrono>

using namespace AstraSim;
using namespace AstraSimAnalyticalCongestionAware;
using namespace NetworkAnalytical;
using namespace NetworkAnalyticalCongestionAware;

std::shared_ptr<Topology> CongestionAwareNetworkApi::topology;
std::shared_ptr<FluidScheduler> CongestionAwareNetworkApi::fluid_scheduler;
CongestionAwareNetworkApi::FlowBoundCallback
    CongestionAwareNetworkApi::flow_bound_callback_;
CongestionAwareNetworkApi::SendWallStats CongestionAwareNetworkApi::send_wall_stats_;

void CongestionAwareNetworkApi::set_topology(
    std::shared_ptr<Topology> topology_ptr) noexcept {
    assert(topology_ptr != nullptr);

    // move topology
    CongestionAwareNetworkApi::topology = std::move(topology_ptr);

    // set topology-related values
    CongestionAwareNetworkApi::dims_count =
        CongestionAwareNetworkApi::topology->get_dims_count();
    CongestionAwareNetworkApi::bandwidth_per_dim =
        CongestionAwareNetworkApi::topology->get_bandwidth_per_dim();
}

void CongestionAwareNetworkApi::set_fluid_scheduler(
    std::shared_ptr<FluidScheduler> scheduler) noexcept {
    assert(scheduler != nullptr);
    CongestionAwareNetworkApi::fluid_scheduler = std::move(scheduler);
    install_flow_bound_callback();
}

void CongestionAwareNetworkApi::set_flow_bound_callback(
    FlowBoundCallback callback) {
    flow_bound_callback_ = std::move(callback);
    install_flow_bound_callback();
}

void CongestionAwareNetworkApi::install_flow_bound_callback() {
    if (fluid_scheduler == nullptr) {
        return;
    }
    fluid_scheduler->set_on_flow_bound(
        [](const NetworkAnalyticalCongestionAware::LogicalFlowKey& key,
           const NetworkAnalyticalCongestionAware::FlowId flow_id) {
            if (!flow_bound_callback_) {
                return;
            }
            AstraNetworkAPI::LogicalFlowKey exported;
            exported.descriptor_identity = key.descriptor_identity;
            exported.generation = key.generation;
            exported.tile = key.tile;
            exported.rank = key.rank;
            exported.shard = key.shard;
            exported.leg = key.leg;
            exported.tag = key.tag;
            exported.chunk_id = key.chunk_id;
            flow_bound_callback_(exported, flow_id);
        });
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
    return sim_send_impl(buffer, count, type, dst, tag, request, msg_handler,
                         fun_arg, nullptr);
}

int CongestionAwareNetworkApi::sim_send_correlated(
    void* const buffer,
    const uint64_t count,
    const int type,
    const int dst,
    const int tag,
    sim_request* const request,
    void (*msg_handler)(void*),
    void* const fun_arg,
    const AstraNetworkAPI::LogicalFlowKey& key) {
    return sim_send_impl(buffer, count, type, dst, tag, request, msg_handler,
                         fun_arg, &key);
}

int CongestionAwareNetworkApi::sim_send_impl(
    void* const buffer,
    const uint64_t count,
    const int type,
    const int dst,
    const int tag,
    sim_request* const request,
    void (*msg_handler)(void*),
    void* const fun_arg,
    const AstraNetworkAPI::LogicalFlowKey* const key) {
    // instrumentation brackets (measurement only): presend covers chunk-id +
    // CallbackTracker + route lookup; start_flow covers the scheduler call.
    const auto presend_t0 = std::chrono::steady_clock::now();

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
        auto* const new_entry =
            callback_tracker.create_new_entry(tag, src, dst, count, chunk_id);
        new_entry->register_send_callback(msg_handler, fun_arg);
    }

    // Keep one tracker entry and one completion argument per logical send.
    // Fluid-internal rate changes never enter CallbackTracker.
    auto chunk_arrival_arg = std::tuple(tag, src, dst, count, chunk_id);
    auto arg = std::make_unique<decltype(chunk_arrival_arg)>(chunk_arrival_arg);
    const auto arg_ptr = static_cast<void*>(arg.release());

    // Route lookup hoisted out of the start_flow call expression so the
    // presend/start_flow wall split lands exactly on the requested boundary
    // (evaluation itself unchanged).
    auto route = topology->fluid_route(src, dst);

    const auto presend_t1 = std::chrono::steady_clock::now();
    send_wall_stats_.presend.ns += static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(presend_t1 - presend_t0)
            .count());
    ++send_wall_stats_.presend.calls;

    assert(fluid_scheduler != nullptr);
    const auto start_flow_t0 = std::chrono::steady_clock::now();
    std::optional<NetworkAnalyticalCongestionAware::LogicalFlowKey>
        logical_key;
    if (key != nullptr) {
        NetworkAnalyticalCongestionAware::LogicalFlowKey bound;
        bound.descriptor_identity = key->descriptor_identity;
        bound.generation = key->generation;
        bound.tile = key->tile;
        bound.rank = key->rank >= 0 ? key->rank : src;
        bound.shard = key->shard;
        bound.leg = key->leg;
        bound.tag = tag;
        bound.chunk_id = chunk_id;
        logical_key = std::move(bound);
    }
    fluid_scheduler->start_flow(
        count, std::move(route),
        CongestionAwareNetworkApi::process_chunk_arrival, arg_ptr, src, dst,
        std::move(logical_key));
    send_wall_stats_.start_flow.ns += static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - start_flow_t0)
            .count());
    ++send_wall_stats_.start_flow.calls;

    return 0;
}
