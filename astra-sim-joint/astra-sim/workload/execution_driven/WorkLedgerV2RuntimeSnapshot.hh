#ifndef ASTRASIM_WORKLOAD_EXECUTION_DRIVEN_WORK_LEDGER_V2_RUNTIME_SNAPSHOT_HH
#define ASTRASIM_WORKLOAD_EXECUTION_DRIVEN_WORK_LEDGER_V2_RUNTIME_SNAPSHOT_HH

#include <json/json.hpp>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "astra-sim/workload/execution_driven/WorkLedgerV2Publisher.hh"

namespace Analytical {
class AnalyticalRemoteMemory;
}

namespace NetworkAnalyticalCongestionAware {
class FluidScheduler;
}

namespace AstraSim {
class Sys;
namespace ExecutionDriven {
class NodeStoreGraphSource;
class RemoteReadExecutor;
class TileTransportAdapter;
struct OnlineNode;

struct WorkLedgerV2Identity {
    std::string request_id;
    std::string stage;
    uint64_t generation = 0;
    int rank = 0;
    uint64_t node_id = 0;
    nlohmann::json descriptor_segment_id = nullptr;
    nlohmann::json endpoint_selector = nullptr;
};

std::string canonical_work_ledger_id(const WorkLedgerV2Identity& identity);
const char* work_ledger_node_operation(const OnlineNode& node);
WorkLedgerV2Identity resolve_work_ledger_identity(
    const std::vector<std::shared_ptr<NodeStoreGraphSource>>& graph_sources,
    const RemoteReadExecutor* executor, int rank, uint64_t node_id);
std::string work_ledger_resource_operation(
    const std::vector<std::shared_ptr<NodeStoreGraphSource>>& graph_sources,
    int rank, uint64_t node_id, const WorkLedgerV2Identity& identity,
    const std::string& fallback);
/// Convert and append one actual lifecycle callback using the same canonical
/// node identity/payload path for Online and the executable fixture. Pipeline
/// anchors are structural proxies and are deliberately suppressed.
void append_work_ledger_v2_node_transition(
    WorkLedgerV2Publisher& publisher,
    const std::vector<std::shared_ptr<NodeStoreGraphSource>>& graph_sources,
    const RemoteReadExecutor* executor, const std::string& kind, int rank,
    uint64_t node_id, OnlineNode node,
    const std::vector<uint64_t>& parent_store_ids, uint64_t tick);

/// Read the actual active NoC, HBM and remote-memory legs at one integrated
/// backend tick. This function is shared by Online and the execution fixture.
nlohmann::json collect_work_ledger_v2_active_resources(
    const std::vector<std::shared_ptr<NodeStoreGraphSource>>& graph_sources,
    const std::vector<Sys*>& systems,
    Analytical::AnalyticalRemoteMemory* memory_api,
    const std::shared_ptr<NetworkAnalyticalCongestionAware::FluidScheduler>&
        fluid_scheduler,
    const RemoteReadExecutor* executor,
    const TileTransportAdapter* transport, uint64_t tick);

}  // namespace ExecutionDriven
}  // namespace AstraSim

#endif  // ASTRASIM_WORKLOAD_EXECUTION_DRIVEN_WORK_LEDGER_V2_RUNTIME_SNAPSHOT_HH
