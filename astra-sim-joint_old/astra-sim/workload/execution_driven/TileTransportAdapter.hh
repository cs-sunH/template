/******************************************************************************
Descriptor-driven p2p transfers through the ordinary Sys/network/HBM path.
******************************************************************************/

#ifndef EXECUTION_DRIVEN_TILE_TRANSPORT_ADAPTER_HH
#define EXECUTION_DRIVEN_TILE_TRANSPORT_ADAPTER_HH

#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "astra-sim/common/AstraNetworkAPI.hh"
#include "astra-sim/system/Callable.hh"
#include "astra-sim/workload/execution_driven/GraphSource.hh"

namespace AstraSim {

class Sys;
class Workload;

namespace ExecutionDriven {

/// Owns only live endpoint work. Each communication endpoint uses the same
/// Sys frontend, fluid matcher, and local-HBM latency/share path as a normal
/// COMM node; descriptor identity is side-band only.
class TileTransportAdapter final : public Callable {
  public:
    struct TransferSpec {
        int rank = -1;
        int src = -1;
        int dst = -1;
        uint64_t bytes = 0;
        uint32_t tag = 0;
        bool hbm_charge = true;
        bool control_marker = false;
        AstraNetworkAPI::LogicalFlowKey logical_key;
    };

    struct ActiveTransfer {
        uint64_t transfer_id = 0;
        uint64_t submit_tick = 0;
        int rank = -1;
        uint64_t logical_node_id = 0;
        int src = -1;
        int dst = -1;
        uint64_t total_bytes = 0;
        uint32_t tag = 0;
        bool control_marker = false;
        bool network_done = false;
        bool hbm_done = false;
        bool flow_bound = false;
        uint64_t flow_id = 0;
        AstraNetworkAPI::LogicalFlowKey logical_key;
    };

    using Completion = std::function<void()>;

    TileTransportAdapter(std::vector<Sys*> systems,
                         std::vector<Workload*> workloads);
    ~TileTransportAdapter() override;

    /// Submit an already resource-gated operation. This method occupies the
    /// exact COMM_SEND/COMM_RECV/COMP gate represented by resource_node and
    /// releases it only when all required network/HBM legs have completed.
    uint64_t submit(const TransferSpec& spec,
                    const NodeView& resource_node,
                    Completion completion);

    void call(EventType type, CallData* data) override;

    /// Bind the side-band key to the scheduler's original physical FlowId at
    /// the existing pending-start flush point.
    void on_flow_bound(const AstraNetworkAPI::LogicalFlowKey& key,
                       uint64_t flow_id);

    /// Administratively complete every stranded transfer: an active,
    /// non-marker leg whose HBM part finished but whose network part is
    /// still pending AND was never bound to a scheduler flow. Call this
    /// ONLY at a provably-no-future-events moment (the event queue is
    /// finished with no deferred work left): at that point no flow can
    /// still bind (a flush needs a proceed), no packet can arrive (an
    /// arrival needs a scheduled event), so every such leg's completion
    /// source is gone by construction -- typically a receive whose paired
    /// data send retired unissued with the request lease (the 2026-10-06
    /// close ruling's "never-issued tail endpoints retire with the lease"
    /// vs "issued legs keep draining naturally" interaction). The
    /// completions run through the ordinary per-leg callbacks.
    void retire_stranded_transfers();

    [[nodiscard]] std::vector<ActiveTransfer> active_snapshot() const;
    [[nodiscard]] bool find_active_by_node(int rank, uint64_t node_id,
                                           ActiveTransfer* out) const;
    [[nodiscard]] bool find_active_by_flow(uint64_t flow_id,
                                           ActiveTransfer* out) const;
    [[nodiscard]] uint64_t active_count() const { return active_.size(); }

  private:
    struct Progress;
    struct CallbackCookie;

    static void hbm_transfer_complete(std::shared_ptr<Progress> progress);
    static ActiveTransfer snapshot(const Progress& progress);
    void mark_network_done(const std::shared_ptr<Progress>& progress);
    void mark_marker_done(const std::shared_ptr<Progress>& progress);
    void finish_if_ready(const std::shared_ptr<Progress>& progress);
    [[noreturn]] static void fail_closed(const std::string& message);

    std::vector<Sys*> systems_;
    std::vector<Workload*> workloads_;
    std::unordered_map<uint64_t, std::shared_ptr<Progress>> active_;
    std::unordered_map<std::string, uint64_t> pending_flow_bindings_;
    std::map<std::pair<int, uint64_t>, uint64_t> active_by_node_;
    std::unordered_map<uint64_t, uint64_t> active_by_flow_;
    uint64_t next_transfer_id_ = 1;
};

}  // namespace ExecutionDriven
}  // namespace AstraSim

#endif  // EXECUTION_DRIVEN_TILE_TRANSPORT_ADAPTER_HH
