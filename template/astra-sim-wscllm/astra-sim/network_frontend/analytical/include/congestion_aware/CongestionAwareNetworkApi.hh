/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#pragma once

#include "common/CommonNetworkApi.hh"
#include <astra-network-analytical/congestion_aware/fluid/FluidScheduler.h>
#include <astra-network-analytical/congestion_aware/Topology.h>

// The using-directives live INSIDE this header's own namespace (network-F10
// fix, former file-top globals): a header-scope `using namespace` leaked
// four namespaces into every includer's global scope. Scoped here, the
// names resolve for this header's own declarations exactly as before, but
// the pollution no longer spreads; every .cc that includes this header
// already declares its own file-scope using-directives.
namespace AstraSimAnalyticalCongestionAware {

using namespace AstraSim;
using namespace AstraSimAnalytical;
using namespace NetworkAnalytical;
using namespace NetworkAnalyticalCongestionAware;

class CongestionAwareNetworkApi final : public CommonNetworkApi {
  public:
    /**
     * Set the topology to be used.
     *
     * @param topology_ptr pointer to the topology
     */
    static void set_topology(std::shared_ptr<Topology> topology_ptr) noexcept;

    static void set_fluid_scheduler(std::shared_ptr<FluidScheduler> scheduler) noexcept;

    /**
     * Constructor.
     *
     * @param rank id of the API
     */
    explicit CongestionAwareNetworkApi(int rank) noexcept;

    /**
     * Implement sim_send of AstraNetworkAPI.
     */
    int sim_send(void* buffer,
                 uint64_t count,
                 int type,
                 int dst,
                 int tag,
                 sim_request* request,
                 void (*msg_handler)(void* fun_arg),
                 void* fun_arg) override;

  private:
    /// topology
    static std::shared_ptr<Topology> topology;

    static std::shared_ptr<FluidScheduler> fluid_scheduler;
};

}  // namespace AstraSimAnalyticalCongestionAware
