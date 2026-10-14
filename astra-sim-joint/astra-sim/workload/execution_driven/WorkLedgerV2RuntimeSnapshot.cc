#include "astra-sim/workload/execution_driven/WorkLedgerV2RuntimeSnapshot.hh"

#include "astra-sim/system/Sys.hh"
#include "astra-sim/workload/LocalHbmBandwidthModel.hh"
#include "astra-sim/workload/Workload.hh"
#include "astra-sim/workload/execution_driven/NodeStore.hh"
#include "congestion_aware/fluid/FluidScheduler.h"
#include "../../../extern/remote_memory_backend/analytical/AnalyticalRemoteMemory.hh"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <tuple>

namespace AstraSim {
namespace ExecutionDriven {

namespace {

NodeStoreMeta node_meta(
    const std::vector<std::shared_ptr<NodeStoreGraphSource>>& graph_sources,
    const int rank, const uint64_t node_id) {
    if (rank < 0 || rank >= static_cast<int>(graph_sources.size()) ||
        graph_sources[static_cast<size_t>(rank)] == nullptr) {
        return {};
    }
    const auto meta =
        graph_sources[static_cast<size_t>(rank)]->store().meta_for(node_id);
    return meta.value_or(NodeStoreMeta{});
}

const char* hbm_kind_name(const LocalHbmBandwidthModel::JobKind kind) {
    switch (kind) {
        case LocalHbmBandwidthModel::JobKind::COMPUTE: return "COMPUTE";
        case LocalHbmBandwidthModel::JobKind::RESTORE: return "RESTORE";
        case LocalHbmBandwidthModel::JobKind::COMM_READ: return "COMM_READ";
        case LocalHbmBandwidthModel::JobKind::COMM_WRITE: return "COMM_WRITE";
        case LocalHbmBandwidthModel::JobKind::POOL_READ: return "POOL_READ";
        case LocalHbmBandwidthModel::JobKind::POOL_WRITE: return "POOL_WRITE";
        case LocalHbmBandwidthModel::JobKind::KIND_COUNT: break;
    }
    return "UNKNOWN";
}

}  // namespace

std::string canonical_work_ledger_id(const WorkLedgerV2Identity& identity) {
    return identity.request_id + "#" + identity.stage + ":" +
        std::to_string(identity.generation) + ":r" +
        std::to_string(identity.rank) + "n" +
        std::to_string(identity.node_id);
}

const char* work_ledger_node_operation(const OnlineNode& node) {
    switch (node.kind) {
        case NodeKind::Metadata: return "metadata";
        case NodeKind::MemLoad: return "mem_load";
        case NodeKind::MemStore: return "mem_store";
        case NodeKind::Compute: return "compute";
        case NodeKind::CommSend: return "comm_send";
        case NodeKind::CommRecv: return "comm_recv";
        case NodeKind::CommCollective: return "comm_collective";
        case NodeKind::Invalid: break;
    }
    return "invalid";
}

WorkLedgerV2Identity resolve_work_ledger_identity(
    const std::vector<std::shared_ptr<NodeStoreGraphSource>>& graph_sources,
    const int rank, const uint64_t node_id) {
    const NodeStoreMeta meta = node_meta(graph_sources, rank, node_id);
    if (meta.request_id.empty() || meta.stage.empty()) {
        throw std::runtime_error(
            "active work-ledger identity lacks request/stage metadata");
    }
    return WorkLedgerV2Identity{meta.request_id, meta.stage, meta.generation,
                                rank, node_id};
}

std::string work_ledger_resource_operation(
    const std::vector<std::shared_ptr<NodeStoreGraphSource>>& graph_sources,
    const int rank, const uint64_t node_id, const std::string& fallback) {
    if (rank < 0 || rank >= static_cast<int>(graph_sources.size()) ||
        graph_sources[static_cast<size_t>(rank)] == nullptr) {
        throw std::runtime_error("resource rank is outside NodeStore array");
    }
    const OnlineNode* node =
        graph_sources[static_cast<size_t>(rank)]->store().node_ptr(node_id);
    return node == nullptr ? fallback : work_ledger_node_operation(*node);
}

void append_work_ledger_v2_node_transition(
    WorkLedgerV2Publisher& publisher,
    const std::vector<std::shared_ptr<NodeStoreGraphSource>>& graph_sources,
    const std::string& kind, const int rank,
    const uint64_t node_id, OnlineNode node,
    const std::vector<uint64_t>& parent_store_ids, const uint64_t tick) {
    if (kind != "commit" && kind != "issue" && kind != "progress" &&
        kind != "completion" && kind != "ack" && kind != "release") {
        throw std::invalid_argument("unknown protocol-v3 node transition kind");
    }
    if (rank < 0 || rank >= static_cast<int>(graph_sources.size())) {
        throw std::invalid_argument("node lifecycle transition rank is invalid");
    }
    node.global_id = node_id;
    const WorkLedgerV2Identity identity = resolve_work_ledger_identity(
        graph_sources, rank, node_id);
    if (identity.request_id.empty() || identity.stage.empty()) {
        throw std::runtime_error(
            "node lifecycle transition lacks request/stage context");
    }

    const std::string operation = work_ledger_node_operation(node);
    nlohmann::json payload{
        {"request_id", identity.request_id}, {"stage", identity.stage},
        {"generation", identity.generation}, {"rank", rank},
        {"node_id", node_id}, {"operation", operation}};
    if (kind == "commit") {
        nlohmann::json parents = nlohmann::json::array();
        nlohmann::json dependencies = nlohmann::json::array();
        for (const uint64_t parent_id : parent_store_ids) {
            const WorkLedgerV2Identity parent = resolve_work_ledger_identity(
                graph_sources, rank, parent_id);
            parents.push_back(parent_id);
            dependencies.push_back(canonical_work_ledger_id(parent));
        }
        if (!parents.empty()) {
            payload["parents"] = std::move(parents);
            payload["dependencies"] = std::move(dependencies);
        }
        uint64_t ops = 0;
        uint64_t bytes = 0;
        uint64_t runtime_ns = 0;
        switch (node.kind) {
            case NodeKind::Compute:
                ops = node.compute.num_ops;
                runtime_ns = node.compute.runtime_ns;
                break;
            case NodeKind::CommSend:
            case NodeKind::CommRecv:
                bytes = node.comm.bytes;
                break;
            case NodeKind::CommCollective:
                bytes = node.coll.bytes;
                runtime_ns = node.compute.runtime_ns;
                break;
            default:
                break;
        }
        payload["num_ops"] = ops;
        payload["bytes"] = bytes;
        payload["runtime_ns"] = runtime_ns;
        if (node.kind == NodeKind::Compute) {
            payload["tensor_size"] = node.compute.tensor_size;
        } else if (node.kind == NodeKind::MemLoad ||
                   node.kind == NodeKind::MemStore) {
            payload["tensor_size"] = node.mem.tensor_size;
        }
        payload["total"] = {
            {"ops", ops}, {"bytes", bytes}, {"latency_ns", runtime_ns}};
        payload["service_unit"] =
            node.kind == NodeKind::Compute ? "ops" : "bytes";
        payload["state"] = "committed";
    }

    publisher.append_node_transition(
        kind, tick, canonical_work_ledger_id(identity), std::move(payload));
}

std::vector<jointledger::ResourceRow>
collect_work_ledger_v2_active_resources(
    const std::vector<std::shared_ptr<NodeStoreGraphSource>>& graph_sources,
    const std::vector<Sys*>& systems,
    Analytical::AnalyticalRemoteMemory* memory_api,
    const std::shared_ptr<NetworkAnalyticalCongestionAware::FluidScheduler>&
        fluid_scheduler,
    const uint64_t tick) {
    if (graph_sources.empty() || graph_sources.size() != systems.size() ||
        memory_api == nullptr || fluid_scheduler == nullptr) {
        throw std::invalid_argument(
            "protocol-v3 runtime resource view is not fully wired");
    }
    std::vector<jointledger::ResourceRow> rows;
    const auto common_row = [](
            jointledger::ResourceKind kind,
            const WorkLedgerV2Identity& identity,
            const std::string& operation, const std::string& leg,
            const std::string& state, jointledger::WireAmount total,
            jointledger::WireAmount remaining,
            std::vector<uint64_t> route, const uint64_t as_of_tick,
            const bool has_issued_tick, const uint64_t issued_tick) {
        jointledger::ResourceRow row;
        row.kind = kind;
        row.work_id = canonical_work_ledger_id(identity) + "@" + leg;
        row.request_id = identity.request_id;
        row.stage = identity.stage;
        row.generation = identity.generation;
        row.rank = static_cast<uint64_t>(identity.rank);
        row.node_id = identity.node_id;
        row.operation = operation;
        row.leg = leg;
        row.state = state;
        row.total = std::move(total);
        row.remaining = std::move(remaining);
        row.route = std::move(route);
        row.as_of_tick = as_of_tick;
        row.has_issued_tick = has_issued_tick;
        row.issued_tick = issued_tick;
        return row;
    };

    const auto flow_snapshot = fluid_scheduler->flow_snapshot();
    const uint64_t flow_as_of = fluid_scheduler->flow_snapshot_as_of_tick();
    if (flow_as_of != tick) {
        throw std::runtime_error(
            "NoC frame tick differs from work-ledger boundary");
    }
    for (const auto& flow : flow_snapshot) {
        // No executor/transport endpoint binding anymore: an active NoC flow
        // exports under the generic "noc" identity (numeric accounting --
        // total/remaining/rate/route -- is unchanged).
        const int rank = flow.src_rank >= 0 ? flow.src_rank : 0;
        const WorkLedgerV2Identity identity{"noc", "", 0, rank, flow.flow_id};
        const std::string leg = "noc:" + std::to_string(flow.flow_id);
        std::vector<uint64_t> route;
        for (const auto link : flow.route) route.push_back(link);
        jointledger::WireAmount total;
        total.bytes = jointledger::WireNum::of_uint(flow.total_bytes);
        jointledger::WireAmount remaining;
        remaining.bytes = jointledger::WireNum::of_double(
            static_cast<double>(flow.remaining_bytes));
        auto row = common_row(
            jointledger::ResourceKind::NocFlow, identity, "comm_send", leg,
            flow.state == 0 ? "Active" :
                flow.state == 1 ? "PropagatingTail" : "Completed",
            std::move(total), std::move(remaining), std::move(route),
            flow_as_of, false, 0);
        row.flow_id = flow.flow_id;
        row.current_rate_Bpns = jointledger::WireNum::of_double(
            static_cast<double>(flow.current_rate_Bpns));
        row.last_rate_update_time = jointledger::WireNum::of_uint(
            static_cast<uint64_t>(flow.last_rate_update_time));
        rows.push_back(std::move(row));
    }

    for (size_t rank = 0; rank < systems.size(); ++rank) {
        if (systems[rank] == nullptr || systems[rank]->workload == nullptr) {
            throw std::runtime_error("protocol-v3 HBM system is incomplete");
        }
        auto* model = systems[rank]->workload->local_hbm_bandwidth_model.get();
        if (model == nullptr) continue;
        const auto jobs = model->job_snapshot();
        const uint64_t as_of = model->jobs_as_of_tick();
        if (as_of != tick) {
            throw std::runtime_error(
                "HBM frame tick differs from work-ledger boundary");
        }
        for (const auto& job : jobs) {
            const int job_rank = job.sys_id < 0
                ? static_cast<int>(rank) : job.sys_id;
            const auto kind = static_cast<LocalHbmBandwidthModel::JobKind>(job.kind);
            const std::string kind_name = hbm_kind_name(kind);
            const std::string leg = "hbm:" + kind_name;
            WorkLedgerV2Identity identity;
            std::string hbm_leg;
            try {
                identity = resolve_work_ledger_identity(
                    graph_sources, job_rank, job.node_id);
            } catch (const std::runtime_error& error) {
                // Exogenous background HBM job (execdoc 14.3, 2026-10-07
                // ruling): a transfer job injected via the 19.1A
                // external_resource(hbm_rank join) path owns no NodeStore
                // node. It exports under the explicitly named background
                // identity -- the ONE other allowed empty-stage form,
                // mirroring the unassociated ordinary noc_flow precedent.
                // Any other identity failure stays fail-closed.
                const bool background_eligible =
                    job.transfer_sink &&
                    (kind == LocalHbmBandwidthModel::JobKind::COMM_READ ||
                     kind == LocalHbmBandwidthModel::JobKind::COMM_WRITE);
                if (std::string(error.what()).find(
                        "active work-ledger identity lacks request/stage "
                        "metadata") == std::string::npos ||
                    !background_eligible) {
                    throw;
                }
                identity = WorkLedgerV2Identity{
                    "external_background", "", 0, job_rank, job.job_id};
                hbm_leg = "hbm_external:" + std::to_string(job_rank) +
                    "#" + std::to_string(job.job_id);
            }
            const std::string fallback =
                kind == LocalHbmBandwidthModel::JobKind::COMM_READ ? "comm_send" :
                kind == LocalHbmBandwidthModel::JobKind::COMM_WRITE ? "comm_recv" :
                kind == LocalHbmBandwidthModel::JobKind::COMPUTE ? "compute" :
                kind == LocalHbmBandwidthModel::JobKind::POOL_WRITE ? "mem_store" :
                                                                      "mem_load";
            const std::string operation = hbm_leg.empty()
                ? fallback
                : work_ledger_resource_operation(
                      graph_sources, job_rank, job.node_id, fallback);
            jointledger::WireAmount total;
            total.ops = jointledger::WireNum::of_double(job.total_ops);
            total.bytes = jointledger::WireNum::of_double(job.total_bytes);
            total.latency_ns =
                jointledger::WireNum::of_double(job.total_latency_ns);
            jointledger::WireAmount remaining;
            remaining.ops =
                jointledger::WireNum::of_double(job.remaining_ops);
            remaining.bytes =
                jointledger::WireNum::of_double(job.remaining_bytes);
            remaining.latency_ns =
                jointledger::WireNum::of_double(job.memory_latency_ns);
            auto row = common_row(
                jointledger::ResourceKind::HbmJob, identity, operation,
                hbm_leg.empty() ? leg : hbm_leg,
                job.memory_latency_ns > 1e-9 ? "issued" : "in_service",
                std::move(total), std::move(remaining), {}, as_of, true,
                static_cast<uint64_t>(job.start_tick));
            row.job_kind = kind_name;
            row.start_tick = static_cast<uint64_t>(job.start_tick);
            rows.push_back(std::move(row));
        }
    }

    for (const auto& port : memory_api->port_snapshot()) {
        if (!std::isfinite(port.as_of_ns) ||
            std::abs(port.as_of_ns - static_cast<double>(tick)) > 1e-6) {
            throw std::runtime_error(
                "remote-memory frame is not integrated to boundary tick");
        }
        for (const auto& job : port.jobs) {
            const WorkLedgerV2Identity identity = resolve_work_ledger_identity(
                graph_sources, job.sys_id, job.node_id);
            const std::string leg = "pool:" + std::to_string(port.port_index) +
                                    ":" + std::to_string(job.issue_seq);
            jointledger::WireAmount total;
            total.bytes = jointledger::WireNum::of_uint(job.bytes);
            jointledger::WireAmount remaining;
            remaining.bytes = jointledger::WireNum::of_double(
                job.remaining_bytes);
            auto row = common_row(
                jointledger::ResourceKind::PoolJob, identity,
                work_ledger_resource_operation(graph_sources, job.sys_id,
                                               job.node_id, "mem_load"),
                leg, job.state == 0 ? "LatencyWaiting" :
                     job.state == 1 ? "Streaming" :
                                      "FluidCompleteAwaitingCallback",
                std::move(total), std::move(remaining), {}, tick, false, 0);
            row.port_index = port.port_index;
            row.issue_seq = job.issue_seq;
            row.latency_ready_ns =
                jointledger::WireNum::of_double(job.latency_ready_ns);
            rows.push_back(std::move(row));
        }
    }

    std::sort(rows.begin(), rows.end(),
              [](const jointledger::ResourceRow& left,
                 const jointledger::ResourceRow& right) {
                  return std::make_tuple(
                             left.rank, left.node_id, left.leg,
                             std::string(jointledger::resource_kind_name(
                                 left.kind))) <
                         std::make_tuple(
                             right.rank, right.node_id, right.leg,
                             std::string(jointledger::resource_kind_name(
                                 right.kind)));
              });
    return rows;
}

}  // namespace ExecutionDriven
}  // namespace AstraSim
