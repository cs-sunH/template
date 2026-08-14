/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#include "congestion_aware/MultiDimTopology.h"
#include <cassert>
#include <cstdlib>
#include <iostream>
#include <utility>

using namespace NetworkAnalytical;
using namespace NetworkAnalyticalCongestionAware;

MultiDimTopology::MultiDimTopology(std::vector<TopologyBuildingBlock> topology_per_dim,
                                   std::vector<int> npus_count_per_dim,
                                   std::vector<Bandwidth> bandwidth_per_dim,
                                   std::vector<Latency> latency_per_dim) noexcept
    : topology_per_dim(std::move(topology_per_dim)),
      latency_per_dim(std::move(latency_per_dim)),
      strides() {
    dims_count = static_cast<int>(this->topology_per_dim.size());
    assert(dims_count > 0);
    assert(npus_count_per_dim.size() == static_cast<size_t>(dims_count));
    assert(bandwidth_per_dim.size() == static_cast<size_t>(dims_count));
    assert(this->latency_per_dim.size() == static_cast<size_t>(dims_count));

    for (const auto topology_type : this->topology_per_dim) {
        if (topology_type == TopologyBuildingBlock::Mesh || topology_type == TopologyBuildingBlock::Ring ||
            topology_type == TopologyBuildingBlock::FullyConnected) {
            continue;
        }

        std::cerr << "[Error] (network/analytical/congestion_aware) "
                  << "multi-dimensional topology supports Line/Mesh, Ring, and FullyConnected dimensions"
                  << std::endl;
        std::exit(-1);
    }

    this->npus_count_per_dim = std::move(npus_count_per_dim);
    this->bandwidth_per_dim = std::move(bandwidth_per_dim);

    npus_count = 1;
    strides.clear();
    for (auto dim = 0; dim < dims_count; dim++) {
        const auto dim_size = this->npus_count_per_dim[dim];
        assert(dim_size >= 1);

        strides.push_back(npus_count);
        npus_count *= dim_size;
    }

    devices_count = npus_count;
    instantiate_devices();

    for (auto dim = 0; dim < dims_count; dim++) {
        connect_dimension(dim);
    }
}

Route MultiDimTopology::route(const DeviceId src, const DeviceId dest) const noexcept {
    assert(0 <= src && src < npus_count);
    assert(0 <= dest && dest < npus_count);

    auto route = Route();
    auto current_address = translate_address(src);
    const auto dest_address = translate_address(dest);

    route.push_back(devices[src]);

    for (auto dim = 0; dim < dims_count; dim++) {
        if (current_address[dim] == dest_address[dim]) {
            continue;
        }

        const auto topology_type = topology_per_dim[dim];
        const auto dim_size = npus_count_per_dim[dim];

        if (topology_type == TopologyBuildingBlock::Mesh) {
            const auto step = (current_address[dim] < dest_address[dim]) ? 1 : -1;
            while (current_address[dim] != dest_address[dim]) {
                current_address[dim] += step;
                route.push_back(devices[translate_address(current_address)]);
            }
            continue;
        }

        if (topology_type == TopologyBuildingBlock::Ring) {
            auto clockwise_dist = dest_address[dim] - current_address[dim];
            if (clockwise_dist < 0) {
                clockwise_dist += dim_size;
            }
            const auto anticlockwise_dist = dim_size - clockwise_dist;
            const auto step = (anticlockwise_dist < clockwise_dist) ? -1 : 1;

            while (current_address[dim] != dest_address[dim]) {
                current_address[dim] += step;
                if (current_address[dim] < 0) {
                    current_address[dim] += dim_size;
                } else if (current_address[dim] >= dim_size) {
                    current_address[dim] -= dim_size;
                }
                route.push_back(devices[translate_address(current_address)]);
            }
            continue;
        }

        if (topology_type == TopologyBuildingBlock::FullyConnected) {
            current_address[dim] = dest_address[dim];
            route.push_back(devices[translate_address(current_address)]);
            continue;
        }

        std::cerr << "[Error] (network/analytical/congestion_aware) "
                  << "unsupported topology in multi-dimensional route" << std::endl;
        std::exit(-1);
    }

    return route;
}

MultiDimTopology::MultiDimAddress MultiDimTopology::translate_address(const DeviceId npu_id) const noexcept {
    assert(0 <= npu_id && npu_id < npus_count);

    auto address = MultiDimAddress(dims_count, 0);
    auto leftover = npu_id;
    auto denominator = npus_count;

    for (auto dim = dims_count - 1; dim >= 0; dim--) {
        denominator /= npus_count_per_dim[dim];

        const auto quotient = leftover / denominator;
        leftover %= denominator;
        address[dim] = quotient;
    }

    return address;
}

DeviceId MultiDimTopology::translate_address(const MultiDimAddress& address) const noexcept {
    assert(address.size() == static_cast<size_t>(dims_count));

    auto npu_id = 0;
    for (auto dim = 0; dim < dims_count; dim++) {
        assert(0 <= address[dim]);
        assert(address[dim] < npus_count_per_dim[dim]);

        npu_id += address[dim] * strides[dim];
    }

    assert(0 <= npu_id && npu_id < npus_count);
    return npu_id;
}

void MultiDimTopology::connect_dimension(const int dim) noexcept {
    assert(0 <= dim && dim < dims_count);

    const auto topology_type = topology_per_dim[dim];
    switch (topology_type) {
    case TopologyBuildingBlock::Mesh:
        connect_mesh_dimension(dim);
        return;
    case TopologyBuildingBlock::Ring:
        connect_ring_dimension(dim);
        return;
    case TopologyBuildingBlock::FullyConnected:
        connect_fully_connected_dimension(dim);
        return;
    default:
        std::cerr << "[Error] (network/analytical/congestion_aware) "
                  << "multi-dimensional topology supports Line/Mesh, Ring, and FullyConnected dimensions"
                  << std::endl;
        std::exit(-1);
    }
}

void MultiDimTopology::connect_mesh_dimension(const int dim) noexcept {
    assert(0 <= dim && dim < dims_count);

    for (auto src = 0; src < npus_count; src++) {
        auto address = translate_address(src);
        if (address[dim] + 1 >= npus_count_per_dim[dim]) {
            continue;
        }

        address[dim]++;
        const auto dest = translate_address(address);
        connect(src, dest, bandwidth_per_dim[dim], latency_per_dim[dim], true);
    }
}

void MultiDimTopology::connect_ring_dimension(const int dim) noexcept {
    assert(0 <= dim && dim < dims_count);

    if (npus_count_per_dim[dim] == 2) {
        connect_mesh_dimension(dim);
        return;
    }

    for (auto src = 0; src < npus_count; src++) {
        auto address = translate_address(src);
        address[dim] = (address[dim] + 1) % npus_count_per_dim[dim];

        const auto dest = translate_address(address);
        connect(src, dest, bandwidth_per_dim[dim], latency_per_dim[dim], true);
    }
}

void MultiDimTopology::connect_fully_connected_dimension(const int dim) noexcept {
    assert(0 <= dim && dim < dims_count);

    for (auto src = 0; src < npus_count; src++) {
        auto address = translate_address(src);
        const auto src_dim_address = address[dim];

        for (auto dim_address = src_dim_address + 1; dim_address < npus_count_per_dim[dim];
             dim_address++) {
            address[dim] = dim_address;

            const auto dest = translate_address(address);
            connect(src, dest, bandwidth_per_dim[dim], latency_per_dim[dim], true);
        }
    }
}
