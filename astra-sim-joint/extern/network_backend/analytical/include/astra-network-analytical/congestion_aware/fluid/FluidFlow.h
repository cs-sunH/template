/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#pragma once

#include "common/Type.h"
#include "congestion_aware/Type.h"
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace NetworkAnalyticalCongestionAware {

class FluidScheduler;

struct FluidRoute {
    std::vector<LinkId> link_ids;
    NetworkAnalytical::EventTime propagation_latency_ns;
};

enum class FluidFlowState { Active, PropagatingTail, Completed };

/// Side-band owner identity. This is not part of the physical packet-match
/// key and never affects flow allocation or arbitration.
struct LogicalFlowKey {
    std::string descriptor_identity;
    uint64_t generation = 0;
    uint64_t tile = 0;
    int rank = -1;
    uint64_t shard = 0;
    std::string leg;
    int tag = 0;
    int chunk_id = -1;
};

struct LinkMembership {
    LinkId link_id;
    uint32_t index_in_link_active_flows;
};

struct FluidFlow {
    FlowId flow_id;
    // Endpoint identity is carried with the flow so a work-ledger snapshot
    // can form a stable owner key.  Direct scheduler fixtures may leave the
    // values at -1; the online network API supplies both ranks.
    int src_rank = -1;
    int dst_rank = -1;
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
    // A tail event borrows this stable unordered_map element as its callback
    // context.  It is set exactly while PropagatingTail and cleared immediately
    // before the element is erased by FluidScheduler.
    FluidScheduler* tail_scheduler = nullptr;
    std::optional<LogicalFlowKey> logical_key;
};

}  // namespace NetworkAnalyticalCongestionAware
