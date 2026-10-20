/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#pragma once

#include "common/Type.h"
#include "congestion_aware/Topology.h"

using namespace NetworkAnalytical;

namespace NetworkAnalyticalCongestionAware {

/**
 * BasicTopology defines 1D topology
 * such as Ring, FullyConnected, and Switch topology,
 * which can be used to construct multi-dimensional topology.
 */
class BasicTopology : public Topology {
  public:
    /**
     * Constructor.
     *
     * @param npus_count number of NPUs in the topology
     * @param devices_count number of devices in the topology
     * @param bandwidth bandwidth of each link
     * @param latency latency of each link
     */
    BasicTopology(int npus_count, int devices_count, Bandwidth bandwidth, Latency latency) noexcept;

    /**
     * Destructor.
     */
    virtual ~BasicTopology() noexcept;

    /**
     * Return the type of the basic topology
     * as a TopologyBuildingBlock enum class element.
     *
     * @return type of the basic topology
     *
     * Dead accessor (2026-09 deep-dive): zero callers in this repo, leaving
     * basic_topology_type below write-only. Retained by repo decision
     * (2026-09-27, infra-D2: annotation route -- the three sister repos
     * deleted this cluster, so deletion here would be its own port; note
     * that Ring/Switch never assign basic_topology_type, so a future
     * caller of this accessor on them would assert).
     */
    [[nodiscard]] TopologyBuildingBlock get_basic_topology_type() const noexcept;

  protected:
    /// bandwidth of each link (write-only: assigned in the constructors, no
    /// reader in this repo -- registered 2026-09 deep-dive, retained by the
    /// 2026-09-27 infra-D2 annotation-route decision)
    Bandwidth bandwidth;

    /// latency of each link (write-only: assigned in the constructors, no
    /// reader in this repo -- registered 2026-09 deep-dive, retained by the
    /// 2026-09-27 infra-D2 annotation-route decision)
    Latency latency;

    /// basic topology type (write-only: set by the concrete topologies, only
    /// read by the dead accessor above)
    TopologyBuildingBlock basic_topology_type;
};

}  // namespace NetworkAnalyticalCongestionAware
