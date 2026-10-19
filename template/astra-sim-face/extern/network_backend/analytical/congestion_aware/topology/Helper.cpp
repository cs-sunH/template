/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#include "congestion_aware/Helper.h"
#include "congestion_aware/FullyConnected.h"
#include "congestion_aware/Mesh.h"
#include "congestion_aware/MultiDimTopology.h"
#include "congestion_aware/Ring.h"
#include "congestion_aware/Switch.h"
#include <cstdlib>
#include <iostream>

using namespace NetworkAnalytical;
using namespace NetworkAnalyticalCongestionAware;

std::shared_ptr<Topology> NetworkAnalyticalCongestionAware::construct_topology(
    const NetworkParser& network_parser) noexcept {
    // get network_parser info
    const auto dims_count = network_parser.get_dims_count();
    const auto topologies_per_dim = network_parser.get_topologies_per_dim();
    const auto npus_counts_per_dim = network_parser.get_npus_counts_per_dim();
    const auto bandwidths_per_dim = network_parser.get_bandwidths_per_dim();
    const auto latencies_per_dim = network_parser.get_latencies_per_dim();

    // if dims_count is 1, just create basic topology
    if (dims_count == 1) {
        // retrieve basic basic-topology info
        const auto topology_type = topologies_per_dim[0];
        const auto npus_count = npus_counts_per_dim[0];
        const auto bandwidth = bandwidths_per_dim[0];
        const auto latency = latencies_per_dim[0];

        switch (topology_type) {
        case TopologyBuildingBlock::Ring:
            return std::make_shared<Ring>(npus_count, bandwidth, latency);
        case TopologyBuildingBlock::Switch:
            return std::make_shared<Switch>(npus_count, bandwidth, latency);
        case TopologyBuildingBlock::FullyConnected:
            return std::make_shared<FullyConnected>(npus_count, bandwidth, latency);
        case TopologyBuildingBlock::Mesh:
            return std::make_shared<Mesh>(npus_count, bandwidth, latency);
        default:
            // shouldn't reach here
            std::cerr << "[Error] (network/analytical/congestion_aware) " << "not supported basic-topology"
                      << std::endl;
            std::exit(-1);
        }
    }

    return std::make_shared<MultiDimTopology>(
        topologies_per_dim, npus_counts_per_dim, bandwidths_per_dim, latencies_per_dim);
}
