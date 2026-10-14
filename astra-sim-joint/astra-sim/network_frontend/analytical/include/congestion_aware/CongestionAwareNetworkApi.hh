/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#pragma once

#include "common/CommonNetworkApi.hh"
#include <astra-network-analytical/congestion_aware/fluid/FluidScheduler.h>
#include <astra-network-analytical/congestion_aware/Topology.h>

using namespace AstraSim;
using namespace AstraSimAnalytical;
using namespace NetworkAnalytical;
using namespace NetworkAnalyticalCongestionAware;

namespace AstraSimAnalyticalCongestionAware {

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

    /// Additive wall-clock instrumentation for sim_send (measurement only):
    /// presend covers chunk-id generation + CallbackTracker lookup/insert +
    /// route lookup; start_flow covers the FluidScheduler::start_flow call.
    /// Static: sim_send is per-rank, the report is per-run.
    struct SendWallStats {
        NetworkAnalytical::WallClockBucket presend;
        NetworkAnalytical::WallClockBucket start_flow;
    };

    [[nodiscard]] static const SendWallStats& send_wall_stats() noexcept {
        return send_wall_stats_;
    }

  private:
    int sim_send_impl(void* buffer,
                      uint64_t count,
                      int type,
                      int dst,
                      int tag,
                      sim_request* request,
                      void (*msg_handler)(void* fun_arg),
                      void* fun_arg);
    /// topology
    static std::shared_ptr<Topology> topology;

    static std::shared_ptr<FluidScheduler> fluid_scheduler;

    /// instrumentation counters (see send_wall_stats())
    static SendWallStats send_wall_stats_;
};

}  // namespace AstraSimAnalyticalCongestionAware
