/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#pragma once

#include "common/Type.h"
#include "congestion_unaware/BasicTopology.h"

using namespace NetworkAnalytical;

namespace NetworkAnalyticalCongestionUnaware {

/**
 * Implements a 1D mesh/line topology.
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
     * @param npus_count number of NPUs in the mesh line
     * @param bandwidth bandwidth of each link
     * @param latency latency of each link
     */
    Mesh(int npus_count, Bandwidth bandwidth, Latency latency) noexcept;

  private:
    /**
     * Implements the compute_hops_count method of BasicTopology.
     */
    [[nodiscard]] int compute_hops_count(DeviceId src, DeviceId dest) const noexcept override;
};

}  // namespace NetworkAnalyticalCongestionUnaware
