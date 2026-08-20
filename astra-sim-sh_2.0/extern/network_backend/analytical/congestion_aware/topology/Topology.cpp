/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#include "congestion_aware/Topology.h"
#include "congestion_aware/Link.h"
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>

using namespace NetworkAnalyticalCongestionAware;

Topology::Topology() noexcept : npus_count(-1), devices_count(-1), dims_count(-1) {
    npus_count_per_dim = {};
}

std::shared_ptr<const FluidRoute> Topology::fluid_route(const DeviceId src,
                                                        const DeviceId dest) const noexcept {
    const auto key = (static_cast<uint64_t>(static_cast<uint32_t>(src)) << 32U) |
                     static_cast<uint32_t>(dest);
    const auto cached = fluid_route_cache.find(key);
    if (cached != fluid_route_cache.end()) {
        return cached->second;
    }

    const auto device_route = route(src, dest);
    if (device_route.size() < 2) {
        std::cerr << "[Error] (network/analytical/congestion_aware) fluid route must contain at least one link"
                  << std::endl;
        std::exit(-1);
    }

    auto fluid = std::make_shared<FluidRoute>();
    long double total_latency = 0.0L;
    auto current = device_route.begin();
    auto next = std::next(current);
    while (next != device_route.end()) {
        const auto link = (*current)->get_link((*next)->get_id());
        if (link == nullptr || link->get_bandwidth_Bpns() <= 0) {
            std::cerr << "[Error] (network/analytical/congestion_aware) invalid link in fluid route" << std::endl;
            std::exit(-1);
        }
        fluid->link_ids.push_back(link->get_id());
        total_latency += static_cast<long double>(link->get_latency());
        ++current;
        ++next;
    }

    if (!std::isfinite(total_latency) ||
        total_latency > static_cast<long double>(std::numeric_limits<EventTime>::max())) {
        std::cerr << "[Error] (network/analytical/congestion_aware) fluid route latency overflows EventTime"
                  << std::endl;
        std::exit(-1);
    }
    fluid->propagation_latency_ns = static_cast<EventTime>(std::ceil(total_latency));
    fluid_route_cache.emplace(key, fluid);
    return fluid;
}

const std::vector<std::shared_ptr<const Link>>& Topology::get_directed_links() const noexcept {
    return directed_links;
}

int Topology::get_devices_count() const noexcept {
    assert(devices_count > 0);
    assert(npus_count > 0);
    assert(devices_count >= npus_count);

    return devices_count;
}

int Topology::get_npus_count() const noexcept {
    assert(devices_count > 0);
    assert(npus_count > 0);
    assert(devices_count >= npus_count);

    return npus_count;
}

int Topology::get_links_count() const noexcept {
    assert(devices_count > 0);
    assert(devices.size() == static_cast<size_t>(devices_count));

    auto links_count = 0;
    for (const auto& device : devices) {
        links_count += device->get_links_count();
    }

    return links_count;
}

int Topology::get_dims_count() const noexcept {
    assert(dims_count > 0);

    return dims_count;
}

std::vector<int> Topology::get_npus_count_per_dim() const noexcept {
    assert(npus_count_per_dim.size() == dims_count);

    return npus_count_per_dim;
}

std::vector<Bandwidth> Topology::get_bandwidth_per_dim() const noexcept {
    assert(bandwidth_per_dim.size() == dims_count);

    return bandwidth_per_dim;
}

void Topology::connect(const DeviceId src,
                       const DeviceId dest,
                       const Bandwidth bandwidth,
                       const Latency latency,
                       const bool bidirectional) noexcept {
    // assert the src and dest are valid
    assert(0 <= src && src < devices_count);
    assert(0 <= dest && dest < devices_count);

    // assert bandwidth and latency are valid
    assert(bandwidth > 0);
    assert(latency >= 0);

    const auto links_to_add = bidirectional ? 2U : 1U;
    if (directed_links.size() >
        static_cast<size_t>(std::numeric_limits<LinkId>::max()) + 1U - links_to_add) {
        std::cerr << "[Error] (network/analytical/congestion_aware) directed LinkId space exhausted"
                  << std::endl;
        std::exit(-1);
    }

    // connect src -> dest
    auto link = devices[src]->connect(
        dest, static_cast<LinkId>(directed_links.size()), bandwidth, latency);
    directed_links.push_back(std::move(link));

    // if bidirectional, connect dest -> src
    if (bidirectional) {
        auto reverse_link = devices[dest]->connect(
            src, static_cast<LinkId>(directed_links.size()), bandwidth, latency);
        directed_links.push_back(std::move(reverse_link));
    }
}

void Topology::instantiate_devices() noexcept {
    // instantiate all devices
    for (auto i = 0; i < devices_count; i++) {
        devices.push_back(std::make_shared<Device>(i));
    }
}
