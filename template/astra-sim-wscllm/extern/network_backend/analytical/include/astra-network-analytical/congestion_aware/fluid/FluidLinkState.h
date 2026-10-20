/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#pragma once

#include "common/Type.h"
#include "congestion_aware/Type.h"
#include <vector>

namespace NetworkAnalyticalCongestionAware {

struct ActiveFlowRef {
    FlowId flow_id;
    uint32_t membership_index_in_flow;
};

struct FluidLinkState {
    // (The former `LinkId link_id` field was write-only: the vector index in
    // FluidScheduler::link_states is the link identity everywhere.)
    NetworkAnalytical::Bandwidth capacity_Bpns;
    std::vector<ActiveFlowRef> active_flows;
};

}  // namespace NetworkAnalyticalCongestionAware
