/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#pragma once

#include "common/Type.h"
#include "congestion_aware/Topology.h"
#include <vector>

using namespace NetworkAnalytical;

namespace NetworkAnalyticalCongestionAware {

/**
 * Implements multi-dimensional congestion-aware topologies over one global
 * device graph.
 *
 * Line/Mesh dimensions connect nearest neighbors without wraparound. Ring
 * dimensions connect nearest neighbors with wraparound. FullyConnected
 * dimensions connect every NPU pair that differs only in that dimension.
 */
class MultiDimTopology final : public Topology {
  public:
    /**
     * Constructor.
     *
     * @param topology_per_dim topology type per dimension
     * @param npus_count_per_dim number of NPUs per dimension
     * @param bandwidth_per_dim bandwidth per dimension
     * @param latency_per_dim latency per dimension
     */
    MultiDimTopology(std::vector<TopologyBuildingBlock> topology_per_dim,
                     std::vector<int> npus_count_per_dim,
                     std::vector<Bandwidth> bandwidth_per_dim,
                     std::vector<Latency> latency_per_dim) noexcept;

    /**
     * Implementation of route function in Topology.
     */
    [[nodiscard]] Route route(DeviceId src, DeviceId dest) const noexcept override;

  private:
    /// Each NPU ID can be broken down into multiple dimensions.
    using MultiDimAddress = std::vector<DeviceId>;

    /// topology type per dimension
    std::vector<TopologyBuildingBlock> topology_per_dim;

    /// latency per dimension
    std::vector<Latency> latency_per_dim;

    /// address stride per dimension
    std::vector<int> strides;

    /**
     * Translate the NPU ID into a multi-dimensional address.
     */
    [[nodiscard]] MultiDimAddress translate_address(DeviceId npu_id) const noexcept;

    /**
     * Translate a multi-dimensional address into an NPU ID.
     */
    [[nodiscard]] DeviceId translate_address(const MultiDimAddress& address) const noexcept;

    /**
     * Connect all links for a dimension.
     */
    void connect_dimension(int dim) noexcept;

    /**
     * Connect a mesh dimension.
     */
    void connect_mesh_dimension(int dim) noexcept;

    /**
     * Connect a ring dimension.
     */
    void connect_ring_dimension(int dim) noexcept;

    /**
     * Connect a fully-connected dimension.
     */
    void connect_fully_connected_dimension(int dim) noexcept;
};

}  // namespace NetworkAnalyticalCongestionAware
