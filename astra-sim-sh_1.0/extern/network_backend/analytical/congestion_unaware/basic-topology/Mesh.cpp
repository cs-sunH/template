/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#include "congestion_unaware/Mesh.h"
#include <cassert>
#include <cstdlib>

using namespace NetworkAnalytical;
using namespace NetworkAnalyticalCongestionUnaware;

Mesh::Mesh(const int npus_count, const Bandwidth bandwidth, const Latency latency) noexcept
    : BasicTopology(npus_count, bandwidth, latency) {
    assert(npus_count > 0);
    assert(bandwidth > 0);
    assert(latency >= 0);

    basic_topology_type = TopologyBuildingBlock::Mesh;
}

int Mesh::compute_hops_count(const DeviceId src, const DeviceId dest) const noexcept {
    assert(0 <= src && src < npus_count);
    assert(0 <= dest && dest < npus_count);
    assert(src != dest);

    const auto distance = dest - src;
    return (distance >= 0) ? distance : -distance;
}
