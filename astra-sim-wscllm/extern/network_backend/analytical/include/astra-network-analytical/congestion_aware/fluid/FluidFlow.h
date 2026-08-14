/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#pragma once

#include "common/Type.h"
#include "congestion_aware/Type.h"
#include <memory>
#include <vector>

namespace NetworkAnalyticalCongestionAware {

struct FluidRoute {
    std::vector<LinkId> link_ids;
    NetworkAnalytical::EventTime propagation_latency_ns;
};

enum class FluidFlowState { Active, PropagatingTail, Completed };

struct LinkMembership {
    LinkId link_id;
    uint32_t index_in_link_active_flows;
};

struct FluidFlow {
    FlowId flow_id;
    NetworkAnalytical::ChunkSize total_bytes;
    long double remaining_bytes;

    NetworkAnalytical::Bandwidth current_rate_Bpns;
    NetworkAnalytical::EventTime last_rate_update_time;
    uint64_t rate_version;
    uint64_t dirty_epoch;

    std::shared_ptr<const FluidRoute> route;
    std::vector<LinkMembership> memberships;
    FluidFlowState state;

    NetworkAnalytical::Callback completion_callback;
    NetworkAnalytical::CallbackArg completion_arg;
};

}  // namespace NetworkAnalyticalCongestionAware
