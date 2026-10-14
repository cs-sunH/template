/******************************************************************************
Descriptor-driven p2p transfers through the ordinary Sys/network/HBM path.
*******************************************************************************/

#include "astra-sim/workload/execution_driven/TileTransportAdapter.hh"

#include <cstdio>
#include <cstdlib>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <utility>

#include "astra-sim/system/RecvPacketEventHandlerData.hh"
#include "astra-sim/system/SendPacketEventHandlerData.hh"
#include "astra-sim/system/Sys.hh"
#include "astra-sim/system/WorkloadLayerHandlerData.hh"
#include "astra-sim/workload/LocalHbmBandwidthModel.hh"
#include "astra-sim/workload/Workload.hh"

namespace AstraSim {
namespace ExecutionDriven {

struct TileTransportAdapter::Progress {
    TileTransportAdapter* owner = nullptr;
    uint64_t transfer_id = 0;
    uint64_t submit_tick = 0;
    TransferSpec spec;
    NodeView resource_node;
    Completion completion;
    bool network_done = false;
    bool hbm_done = false;
    bool finished = false;
    bool flow_bound = false;
    uint64_t flow_id = 0;
    std::string flow_key_base;
};

namespace {

std::string flow_key_base(const AstraNetworkAPI::LogicalFlowKey& key) {
    std::ostringstream out;
    out << key.descriptor_identity << '\x1f' << key.generation << '\x1f'
        << key.tile << '\x1f' << key.rank << '\x1f' << key.shard << '\x1f'
        << key.leg << '\x1f' << key.tag;
    return out.str();
}

}  // namespace

struct TileTransportAdapter::CallbackCookie : WorkloadLayerHandlerData {
    enum class Side { Network, Marker };

    std::weak_ptr<Progress> progress;
    Side side = Side::Network;
};

TileTransportAdapter::TileTransportAdapter(std::vector<Sys*> systems,
                                           std::vector<Workload*> workloads)
    : systems_(std::move(systems)), workloads_(std::move(workloads)) {
    if (systems_.empty() || systems_.size() != workloads_.size()) {
        throw std::invalid_argument(
            "TileTransportAdapter requires equally sized non-empty Sys and "
            "Workload arrays");
    }
    for (size_t rank = 0; rank < systems_.size(); ++rank) {
        if (systems_[rank] == nullptr || workloads_[rank] == nullptr ||
            systems_[rank]->workload != workloads_[rank] ||
            systems_[rank]->id != static_cast<int>(rank)) {
            throw std::invalid_argument(
                "TileTransportAdapter rank arrays are inconsistent");
        }
    }
}

TileTransportAdapter::~TileTransportAdapter() {
    // Network flows have no cancellation API. Production keeps this adapter
    // alive until physical quiescence; on exceptional teardown, make any HBM
    // closures harmless and let their model-owned captures drop naturally.
    for (auto& entry : active_) {
        entry.second->owner = nullptr;
        entry.second->completion = nullptr;
    }
    active_.clear();
}

[[noreturn]] void TileTransportAdapter::fail_closed(
    const std::string& message) {
    std::fprintf(stderr,
                 "[Error] (execution_driven/tile_transport) %s\n",
                 message.c_str());
    std::abort();
}

uint64_t TileTransportAdapter::submit(const TransferSpec& spec,
                                      const NodeView& resource_node,
                                      Completion completion) {
    if (spec.rank < 0 || static_cast<size_t>(spec.rank) >= systems_.size() ||
        resource_node.rank != spec.rank || completion == nullptr) {
        fail_closed("invalid rank, resource view, or completion callback");
    }
    const bool is_send = resource_node.kind == NodeKind::CommSend;
    const bool is_recv = resource_node.kind == NodeKind::CommRecv;
    const bool is_marker = spec.control_marker;
    if (spec.src < 0 || spec.dst < 0 ||
        spec.src >= static_cast<int>(systems_.size()) ||
        spec.dst >= static_cast<int>(systems_.size()) ||
        (!is_marker && spec.src == spec.dst)) {
        fail_closed("invalid p2p source/destination");
    }
    if (spec.tag > static_cast<uint32_t>(std::numeric_limits<int>::max())) {
        fail_closed("tag does not fit Sys frontend int");
    }
    if ((!is_marker && !(is_send || is_recv)) ||
        (is_marker && resource_node.kind != NodeKind::Compute)) {
        fail_closed("resource view kind disagrees with transfer kind");
    }
    if (is_send && spec.src != spec.rank) {
        fail_closed("send endpoint rank does not match source rank");
    }
    if (is_recv && spec.dst != spec.rank) {
        fail_closed("receive endpoint rank does not match destination rank");
    }
    if (is_marker && (spec.bytes != 0 || spec.src != spec.rank ||
                      spec.dst != spec.rank)) {
        fail_closed("control marker must not carry a network transfer");
    }
    if (!is_marker && spec.bytes == 0) {
        fail_closed("zero-byte p2p endpoint must not be submitted");
    }

    Sys* const sys = systems_[spec.rank];
    Workload* const workload = workloads_[spec.rank];
    if (next_transfer_id_ == 0 ||
        next_transfer_id_ == std::numeric_limits<uint64_t>::max()) {
        fail_closed("transfer identity exhausted");
    }

    auto progress = std::make_shared<Progress>();
    progress->owner = this;
    progress->transfer_id = next_transfer_id_++;
    progress->submit_tick = Sys::boostedTick();
    progress->spec = spec;
    progress->resource_node = resource_node;
    progress->completion = std::move(completion);
    progress->network_done = is_marker;
    progress->hbm_done = !is_marker;
    if (!is_marker && resource_node.kind == NodeKind::CommSend) {
        progress->flow_key_base = flow_key_base(spec.logical_key);
    }

    // The adapter is called only after the combined-ready issuer checked
    // this exact synthetic COMM/COMP view. Occupy it before either physical
    // leg starts, preserving the ordinary Workload gate lifetime.
    workload->hw_resource->occupy(resource_node);
    if (!active_.emplace(progress->transfer_id, progress).second ||
        !active_by_node_.emplace(
             std::make_pair(spec.rank, resource_node.global_id),
             progress->transfer_id).second) {
        fail_closed("duplicate active transfer identity");
    }

    if (is_marker) {
        auto* cookie = new CallbackCookie;
        cookie->sys_id = spec.rank;
        cookie->workload = workload;
        cookie->node_id = resource_node.global_id;
        cookie->pipeline_sink = this;
        cookie->progress = progress;
        cookie->side = CallbackCookie::Side::Marker;
        workload->issue_pipeline_control_marker(resource_node, cookie);
        return progress->transfer_id;
    }

    auto* cookie = new CallbackCookie;
    cookie->sys_id = spec.rank;
    cookie->workload = workload;
    cookie->node_id = resource_node.global_id;
    cookie->progress = progress;
    cookie->side = CallbackCookie::Side::Network;

    const bool charge_hbm = spec.hbm_charge && spec.bytes > 0 &&
        sys->hbm_bandwidth_contention;
    if (charge_hbm) {
        if (workload->local_hbm_bandwidth_model == nullptr) {
            delete cookie;
            fail_closed("charged endpoint has no local HBM model");
        }
        const auto kind = is_send ? LocalHbmBandwidthModel::JobKind::COMM_READ
                                  : LocalHbmBandwidthModel::JobKind::COMM_WRITE;
        const std::weak_ptr<Progress> weak_progress = progress;
        progress->hbm_done = false;
        workload->local_hbm_bandwidth_model->issue_transfer(
            kind, spec.bytes, spec.rank, resource_node.global_id,
            [weak_progress]() {
                const auto locked = weak_progress.lock();
                if (locked != nullptr) {
                    TileTransportAdapter::hbm_transfer_complete(locked);
                }
            });
    }

    sim_request request{};
    if (is_send) {
        if (progress->flow_key_base.empty() ||
            !pending_flow_bindings_
                 .emplace(progress->flow_key_base, progress->transfer_id)
                 .second) {
            delete cookie;
            fail_closed("duplicate logical flow correlation key");
        }
        auto* send_data = new SendPacketEventHandlerData;
        send_data->callable = this;
        send_data->wlhd = cookie;
        send_data->event = EventType::PacketSent;
        sys->front_end_sim_send_correlated(
            0, Sys::dummy_data, spec.bytes, UINT8, spec.dst,
            static_cast<int>(spec.tag), &request,
            Sys::FrontEndSendRecvType::NATIVE, &Sys::handleEvent, send_data,
            spec.logical_key);
    } else {
        auto* recv_data = new RecvPacketEventHandlerData;
        recv_data->workload = nullptr;
        recv_data->callable = this;
        recv_data->wlhd = cookie;
        recv_data->event = EventType::PacketReceived;
        sys->front_end_sim_recv(
            0, Sys::dummy_data, spec.bytes, UINT8, spec.src,
            static_cast<int>(spec.tag), &request,
            Sys::FrontEndSendRecvType::NATIVE, &Sys::handleEvent, recv_data);
    }
    return progress->transfer_id;
}

void TileTransportAdapter::call(const EventType type, CallData* data) {
    auto* const base = static_cast<WorkloadLayerHandlerData*>(data);
    auto* const cookie = static_cast<CallbackCookie*>(base);
    if (cookie == nullptr) {
        fail_closed("callback arrived without a transport cookie");
    }
    const auto progress = cookie->progress.lock();
    const auto side = cookie->side;
    delete cookie;
    if (progress == nullptr) {
        // This is possible only after exceptional teardown. There is no live
        // executor to advance, and the physical cookie has been reclaimed.
        return;
    }
    if (progress->owner != this || progress->finished) {
        fail_closed("callback has stale or mismatched adapter owner");
    }
    if (side == CallbackCookie::Side::Marker) {
        if (type != EventType::General) {
            fail_closed("control marker completed with a non-General event");
        }
        mark_marker_done(progress);
        return;
    }
    const bool is_send = progress->resource_node.kind == NodeKind::CommSend;
    const EventType expected = is_send ? EventType::PacketSent
                                       : EventType::PacketReceived;
    if (type != expected) {
        fail_closed("network endpoint callback type mismatch");
    }
    mark_network_done(progress);
}

void TileTransportAdapter::on_flow_bound(
    const AstraNetworkAPI::LogicalFlowKey& key, const uint64_t flow_id) {
    const std::string base = flow_key_base(key);
    const auto pending = pending_flow_bindings_.find(base);
    if (pending == pending_flow_bindings_.end()) {
        fail_closed("scheduler bound an unknown or already-bound logical flow");
    }
    const auto active = active_.find(pending->second);
    if (active == active_.end() || active->second->flow_bound ||
        active->second->flow_key_base != base) {
        fail_closed("logical flow correlation points to a stale transfer");
    }
    active->second->flow_bound = true;
    active->second->flow_id = flow_id;
    if (!active_by_flow_.emplace(flow_id, active->second->transfer_id).second) {
        fail_closed("scheduler reused an active flow identity");
    }
    active->second->spec.logical_key.chunk_id = key.chunk_id;
    active->second->spec.logical_key.tag = key.tag;
    pending_flow_bindings_.erase(pending);
}

void TileTransportAdapter::hbm_transfer_complete(
    std::shared_ptr<Progress> progress) {
    if (progress == nullptr || progress->finished || progress->owner == nullptr) {
        return;
    }
    TileTransportAdapter* const owner = progress->owner;
    progress->hbm_done = true;
    owner->finish_if_ready(progress);
}

void TileTransportAdapter::retire_stranded_transfers() {
    // See the header contract: only legal at a provably-no-future-events
    // moment. finish_if_ready() erases from active_ and runs the leg's
    // ordinary completion (which may schedule follow-up work), so the
    // scan restarts after every retirement -- the map is small (bounded
    // by the in-flight leg count).
    for (auto it = active_.begin(); it != active_.end();) {
        const std::shared_ptr<Progress>& progress = it->second;
        if (!progress->finished && !progress->spec.control_marker &&
            !progress->network_done && !progress->flow_bound &&
            progress->hbm_done) {
            progress->network_done = true;
            finish_if_ready(progress);
            it = active_.begin();
        } else {
            ++it;
        }
    }
}

void TileTransportAdapter::mark_network_done(
    const std::shared_ptr<Progress>& progress) {
    if (progress->network_done) {
        fail_closed("duplicate network completion");
    }
    progress->network_done = true;
    finish_if_ready(progress);
}

void TileTransportAdapter::mark_marker_done(
    const std::shared_ptr<Progress>& progress) {
    if (!progress->spec.control_marker || progress->hbm_done) {
        fail_closed("duplicate or non-marker completion");
    }
    progress->hbm_done = true;
    finish_if_ready(progress);
}

void TileTransportAdapter::finish_if_ready(
    const std::shared_ptr<Progress>& progress) {
    if (progress->finished || !progress->network_done || !progress->hbm_done) {
        return;
    }
    const auto it = active_.find(progress->transfer_id);
    if (it == active_.end() || it->second.get() != progress.get()) {
        fail_closed("active transfer disappeared before terminal completion");
    }
    progress->finished = true;
    progress->owner = nullptr;
    const Completion completion = std::move(progress->completion);
    if (!progress->flow_key_base.empty()) {
        const auto pending = pending_flow_bindings_.find(progress->flow_key_base);
        if (pending != pending_flow_bindings_.end() &&
            pending->second == progress->transfer_id) {
            pending_flow_bindings_.erase(pending);
        }
    }
    const auto node_index = active_by_node_.find(
        {progress->spec.rank, progress->resource_node.global_id});
    if (node_index == active_by_node_.end() ||
        node_index->second != progress->transfer_id) {
        fail_closed("active transfer node index is inconsistent");
    }
    active_by_node_.erase(node_index);
    if (progress->flow_bound) {
        const auto flow_index = active_by_flow_.find(progress->flow_id);
        if (flow_index == active_by_flow_.end() ||
            flow_index->second != progress->transfer_id) {
            fail_closed("active transfer flow index is inconsistent");
        }
        active_by_flow_.erase(flow_index);
    }
    active_.erase(it);
    workloads_[progress->spec.rank]->hw_resource->release(
        progress->resource_node);
    if (completion != nullptr) {
        completion();
    }
}

std::vector<TileTransportAdapter::ActiveTransfer>
TileTransportAdapter::active_snapshot() const {
    std::vector<ActiveTransfer> result;
    result.reserve(active_.size());
    for (const auto& entry : active_) {
        result.push_back(snapshot(*entry.second));
    }
    return result;
}

TileTransportAdapter::ActiveTransfer
TileTransportAdapter::snapshot(const Progress& progress) {
    ActiveTransfer result;
    result.transfer_id = progress.transfer_id;
    result.submit_tick = progress.submit_tick;
    result.rank = progress.spec.rank;
    result.logical_node_id = progress.resource_node.global_id;
    result.src = progress.spec.src;
    result.dst = progress.spec.dst;
    result.total_bytes = progress.spec.bytes;
    result.tag = progress.spec.tag;
    result.control_marker = progress.spec.control_marker;
    result.network_done = progress.network_done;
    result.hbm_done = progress.hbm_done;
    result.flow_bound = progress.flow_bound;
    result.flow_id = progress.flow_id;
    result.logical_key = progress.spec.logical_key;
    return result;
}

bool TileTransportAdapter::find_active_by_node(
    const int rank, const uint64_t node_id, ActiveTransfer* out) const {
    if (out == nullptr) return false;
    const auto indexed = active_by_node_.find({rank, node_id});
    if (indexed == active_by_node_.end()) return false;
    const auto found = active_.find(indexed->second);
    if (found == active_.end() || found->second == nullptr ||
        found->second->spec.rank != rank ||
        found->second->resource_node.global_id != node_id) {
        fail_closed("active transfer node index points to stale state");
    }
    *out = snapshot(*found->second);
    return true;
}

bool TileTransportAdapter::find_active_by_flow(
    const uint64_t flow_id, ActiveTransfer* out) const {
    if (out == nullptr) return false;
    const auto indexed = active_by_flow_.find(flow_id);
    if (indexed == active_by_flow_.end()) return false;
    const auto found = active_.find(indexed->second);
    if (found == active_.end() || found->second == nullptr ||
        !found->second->flow_bound || found->second->flow_id != flow_id) {
        fail_closed("active transfer flow index points to stale state");
    }
    *out = snapshot(*found->second);
    return true;
}

}  // namespace ExecutionDriven
}  // namespace AstraSim
