// TypedRows.cc -- name tables of the typed row enums (TypedRows.hh).
#include "joint_ledger/TypedRows.hh"

#include "joint_ledger/LedgerTypes.hh"

namespace jointledger {

// ------------------------------------------------- query-boundary JSON ----
// Serialization of the typed rows back to the historical wire/query shape.
// Only the query surfaces (pricing snapshot, offline exports) call these;
// the per-frame receive/apply path never round-trips through JSON.

namespace {

Json num_to_json(const WireNum& value) {
    switch (value.kind) {
        case WireNum::Kind::Null: return Json();
        case WireNum::Kind::Int:
            return Json(value.integer);
        case WireNum::Kind::Float: default:
            return Json(value.real);
    }
}

Json owner_to_json(const SlotOwner& owner) {
    Json out = Json::object();
    out["segment_id"] = owner.segment_id;
    out["pipeline_generation"] = owner.pipeline_generation;
    out["stage_epoch"] = owner.stage_epoch;
    out["staging_allocation_id"] = owner.staging_allocation_id;
    out["group_ordinal"] = owner.group_ordinal;
    out["tile_ordinal"] = owner.tile_ordinal;
    return out;
}

}  // namespace

Json amount_to_json(const WireAmount& amount) {
    Json out = Json::object();
    out["ops"] = num_to_json(amount.ops);
    out["bytes"] = num_to_json(amount.bytes);
    out["latency_ns"] = num_to_json(amount.latency_ns);
    return out;
}

Json slot_row_to_json(const SlotRow& row) {
    Json out = Json::object();
    out["buffer_domain_id"] = row.domain;
    out["role"] = row.role_source ? "source" : "target";
    out["rank"] = row.rank;
    out["shard_ordinal"] = row.has_shard ? Json(row.shard) : Json();
    out["slot_index"] = row.slot_index;
    out["occupied"] = row.occupied;
    out["owner"] = row.owner ? owner_to_json(*row.owner) : Json();
    out["consumed"] = row.consumed;
    Json targets = Json::array();
    for (std::uint64_t shard : row.target_shards_to_consume) {
        targets.push_back(shard);
    }
    out["target_shards_to_consume"] = std::move(targets);
    Json consumed = Json::array();
    for (std::uint64_t shard : row.consumed_target_shards) {
        consumed.push_back(shard);
    }
    out["consumed_target_shards"] = std::move(consumed);
    return out;
}

Json credit_row_to_json(const CreditRow& row) {
    Json out = Json::object();
    out["buffer_domain_id"] = row.domain;
    out["source_rank"] = row.source_rank;
    out["shard_ordinal"] = row.shard;
    out["capacity"] = row.capacity;
    out["available"] = row.available;
    out["outstanding_ack_count"] = row.outstanding_ack_count;
    return out;
}

Json obligation_row_to_json(const ObligationRow& row) {
    Json out = Json::object();
    out["obligation_id"] = row.obligation_id;
    out["segment_id"] = row.segment_id;
    out["rank"] = row.rank;
    out["first_store_id"] = row.first_store_id;
    out["remaining_endpoint_count"] = row.remaining_endpoint_count;
    out["state"] = row.state == ObligationState::Active ? "active"
                                                         : "pending";
    return out;
}

Json cursor_run_to_json(const CursorRun& run) {
    Json out = Json::object();
    out["rank"] = run.rank;
    out["group_ordinal"] = run.group_ordinal;
    out["first_tile"] = run.first_tile;
    out["tile_count"] = run.tile_count;
    out["completed_endpoint_count"] = run.completed_endpoint_count;
    out["cancelled_endpoint_count"] = run.cancelled_endpoint_count;
    out["remaining_endpoint_count"] = run.remaining_endpoint_count;
    out["in_flight"] = run.in_flight;
    out["next_store_id"] = run.has_next_store ? Json(run.next_store_id)
                                              : Json();
    out["tile_offset"] = run.tile_offset;
    out["shard_index"] = run.shard_index;
    out["leg_index"] = run.leg_index;
    out["initial_parent_store_id"] = run.has_initial_parent
        ? Json(run.initial_parent_store_id) : Json();
    Json ooo = Json::array();
    for (const auto& range : run.completed_out_of_order) {
        Json item = Json::object();
        item["first_ordinal"] = range.first;
        item["end_ordinal_exclusive"] = range.second;
        ooo.push_back(std::move(item));
    }
    out["completed_out_of_order"] = std::move(ooo);
    Json endpoints = Json::array();
    for (const ActiveEndpoint& endpoint : run.active_endpoints) {
        Json item = Json::object();
        item["store_id"] = endpoint.store_id;
        item["tile_ordinal"] = endpoint.tile_ordinal;
        item["shard_ordinal"] = endpoint.shard_ordinal;
        item["endpoint"] = endpoint.endpoint;
        item["transfer_id"] = endpoint.transfer_id;
        item["flow_id"] = endpoint.flow_bound ? Json(endpoint.flow_id)
                                              : Json();
        item["network_done"] = endpoint.network_done;
        item["hbm_done"] = endpoint.hbm_done;
        item["owner"] = owner_to_json(endpoint.owner);
        endpoints.push_back(std::move(item));
    }
    out["active_endpoints"] = std::move(endpoints);
    return out;
}

Json resource_row_to_json(const ResourceRow& row) {
    Json out = Json::object();
    out["resource_type"] = resource_kind_name(row.kind);
    out["work_id"] = row.work_id;
    out["request_id"] = row.request_id;
    out["stage"] = row.stage;
    out["generation"] = row.generation;
    out["rank"] = row.rank;
    out["node_id"] = row.node_id;
    out["operation"] = row.operation;
    out["leg"] = row.leg;
    out["state"] = row.state;
    out["total"] = amount_to_json(row.total);
    out["remaining"] = amount_to_json(row.remaining);
    Json route = Json::array();
    for (std::uint64_t hop : row.route) route.push_back(hop);
    out["route"] = std::move(route);
    out["as_of_tick"] = row.as_of_tick;
    out["issued_tick"] = row.has_issued_tick ? Json(row.issued_tick) : Json();
    out["progress_source"] = "authoritative";
    out["service_unit"] = "bytes";
    out["descriptor_segment_id"] = row.bound
        ? Json(row.descriptor_segment_id) : Json();
    if (row.has_endpoint_selector) {
        Json selector = Json::object();
        selector["group_ordinal"] = row.endpoint_selector.group_ordinal;
        selector["tile_ordinal"] = row.endpoint_selector.tile_ordinal;
        selector["shard_ordinal"] = row.endpoint_selector.shard_ordinal;
        selector["endpoint"] = row.endpoint_selector.endpoint;
        out["endpoint_selector"] = std::move(selector);
    } else {
        out["endpoint_selector"] = Json();
    }
    switch (row.kind) {
        case ResourceKind::NocFlow:
            out["flow_id"] = row.flow_id;
            out["current_rate_Bpns"] = num_to_json(row.current_rate_Bpns);
            out["last_rate_update_time"] = num_to_json(
                row.last_rate_update_time);
            break;
        case ResourceKind::HbmJob:
            out["kind"] = row.job_kind;
            out["start_tick"] = row.start_tick;
            break;
        case ResourceKind::PoolJob:
            out["port_index"] = row.port_index;
            out["issue_seq"] = row.issue_seq;
            out["latency_ready_ns"] = num_to_json(row.latency_ready_ns);
            break;
    }
    return out;
}


const char* transition_kind_name(TransitionKind kind) {
    switch (kind) {
        case TransitionKind::DescriptorCommit: return "descriptor_commit";
        case TransitionKind::TileIssue: return "tile_issue";
        case TransitionKind::DataReady: return "data_ready";
        case TransitionKind::EndpointDone: return "endpoint_done";
        case TransitionKind::FillAckProgress: return "fill_ack_progress";
        case TransitionKind::FillAckDone: return "fill_ack_done";
        case TransitionKind::ConsumeStart: return "consume_start";
        case TransitionKind::ConsumeDone: return "consume_done";
        case TransitionKind::ReleaseAckProgress: return "release_ack_progress";
        case TransitionKind::ReleaseAckDone: return "release_ack_done";
        case TransitionKind::CreditReturn: return "credit_return";
        case TransitionKind::ServiceDone: return "service_done";
        case TransitionKind::RequestClose: return "request_close";
        case TransitionKind::SettleDone: return "settle_done";
        case TransitionKind::DescriptorRetire: return "descriptor_retire";
    }
    throw LedgerError("unknown transition kind");
}

TransitionKind transition_kind_from_name(const std::string& name) {
    for (std::uint8_t value = 0;
         value <= static_cast<std::uint8_t>(TransitionKind::DescriptorRetire);
         ++value) {
        const TransitionKind kind = static_cast<TransitionKind>(value);
        if (name == transition_kind_name(kind)) return kind;
    }
    throw LedgerError("unknown transition kind '" + name + "'");
}

const char* resource_leg_name(ResourceLeg leg) {
    switch (leg) {
        case ResourceLeg::Endpoint: return "endpoint";
        case ResourceLeg::BodyMarker: return "body_marker";
        case ResourceLeg::Noc: return "noc";
        case ResourceLeg::HbmCommRead: return "hbm:COMM_READ";
        case ResourceLeg::HbmCommWrite: return "hbm:COMM_WRITE";
        case ResourceLeg::None: break;
    }
    throw LedgerError("resource leg is absent");
}

const char* resource_kind_name(ResourceKind kind) {
    switch (kind) {
        case ResourceKind::NocFlow: return "noc_flow";
        case ResourceKind::HbmJob: return "hbm_job";
        case ResourceKind::PoolJob: return "pool_job";
    }
    throw LedgerError("unknown resource kind");
}

const char* node_kind_name(NodeKind kind) {
    switch (kind) {
        case NodeKind::Commit: return "commit";
        case NodeKind::Issue: return "issue";
        case NodeKind::Progress: return "progress";
        case NodeKind::Completion: return "completion";
        case NodeKind::Ack: return "ack";
        case NodeKind::Release: return "release";
    }
    throw LedgerError("unknown node kind");
}

NodeKind node_kind_from_name(const std::string& name) {
    for (std::uint8_t value = 0;
         value <= static_cast<std::uint8_t>(NodeKind::Release); ++value) {
        const NodeKind kind = static_cast<NodeKind>(value);
        if (name == node_kind_name(kind)) return kind;
    }
    throw LedgerError("unknown node kind '" + name + "'");
}

namespace {

const char* const kNocStates[] = {"Active", "PropagatingTail", "Completed"};
const char* const kHbmStates[] = {"issued", "in_service"};
const char* const kPoolStates[] = {"LatencyWaiting", "Streaming",
                                   "FluidCompleteAwaitingCallback"};

}  // namespace

const char* ResourceRow::state_name(ResourceKind kind, std::uint8_t index) {
    switch (kind) {
        case ResourceKind::NocFlow:
            if (index < 3) return kNocStates[index];
            break;
        case ResourceKind::HbmJob:
            if (index < 2) return kHbmStates[index];
            break;
        case ResourceKind::PoolJob:
            if (index < 3) return kPoolStates[index];
            break;
    }
    throw LedgerError("resource state index is invalid for its kind");
}

std::uint8_t ResourceRow::state_index() const {
    const auto find = [&](const char* const* table,
                          std::size_t count) -> std::uint8_t {
        for (std::size_t i = 0; i < count; ++i) {
            if (state == table[i]) return static_cast<std::uint8_t>(i);
        }
        throw LedgerError("resource state '" + state
                          + "' is invalid for its kind");
    };
    switch (kind) {
        case ResourceKind::NocFlow:
            return find(kNocStates, 3);
        case ResourceKind::HbmJob:
            return find(kHbmStates, 2);
        case ResourceKind::PoolJob:
            return find(kPoolStates, 3);
    }
    throw LedgerError("unknown resource kind");
}

}  // namespace jointledger
