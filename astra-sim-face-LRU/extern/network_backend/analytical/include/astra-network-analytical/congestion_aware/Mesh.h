/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#pragma once

#include "common/Type.h"
#include "congestion_aware/BasicTopology.h"

using namespace NetworkAnalytical;

namespace NetworkAnalyticalCongestionAware {

/**
 * Implements a 1D mesh topology.
 *
 * Mesh(4) example:
 * 0 - 1 - 2 - 3
 *
 * Unlike Ring, Mesh does not wrap the last NPU back to the first NPU.
 */
class Mesh final : public BasicTopology {
  public:
    /**
     * Constructor.
     *
     * @param npus_count number of npus in the mesh line
     * @param bandwidth bandwidth of each link
     * @param latency latency of each link
     */
    Mesh(int npus_count, Bandwidth bandwidth, Latency latency) noexcept;

    /**
     * Implementation of route function in Topology.
     */
    [[nodiscard]] Route route(DeviceId src, DeviceId dest) const noexcept override;
};

}  // namespace NetworkAnalyticalCongestionAware
