/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#include "congestion_aware/Mesh.h"
#include <cassert>

using namespace NetworkAnalyticalCongestionAware;

Mesh::Mesh(const int npus_count, const Bandwidth bandwidth, const Latency latency) noexcept
    : BasicTopology(npus_count, npus_count, bandwidth, latency) {
    assert(npus_count > 0);
    assert(bandwidth > 0);
    assert(latency >= 0);

    basic_topology_type = TopologyBuildingBlock::Mesh;

    for (auto i = 0; i < npus_count - 1; i++) {
        connect(i, i + 1, bandwidth, latency, true);
    }
}

Route Mesh::route(const DeviceId src, const DeviceId dest) const noexcept {
    assert(0 <= src && src < npus_count);
    assert(0 <= dest && dest < npus_count);

    auto route = Route();
    const auto step = (src <= dest) ? 1 : -1;

    auto current = src;
    while (current != dest) {
        route.push_back(devices[current]);
        current += step;
    }
    route.push_back(devices[dest]);

    return route;
}
