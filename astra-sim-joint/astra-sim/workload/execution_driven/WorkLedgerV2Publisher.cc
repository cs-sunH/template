#include "astra-sim/workload/execution_driven/WorkLedgerV2Publisher.hh"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <utility>

namespace AstraSim {
namespace ExecutionDriven {

WorkLedgerV2Publisher::WorkLedgerV2Publisher(std::string epoch)
    : epoch_(std::move(epoch)) {
    if (epoch_.empty()) {
        throw std::invalid_argument("protocol-v4 epoch must be nonempty");
    }
}

void WorkLedgerV2Publisher::require_nonempty(const std::string& what) {
    if (what.empty()) {
        throw std::invalid_argument("protocol-v4 field must be nonempty");
    }
}

uint64_t WorkLedgerV2Publisher::next_sequence_candidate() const {
    if (next_logical_seq_ == 0 ||
        next_logical_seq_ == std::numeric_limits<uint64_t>::max()) {
        throw std::overflow_error("protocol-v4 logical sequence exhausted");
    }
    return next_logical_seq_;
}

void WorkLedgerV2Publisher::append_node_transition(
    const std::string& kind, const uint64_t tick, std::string work_id,
    nlohmann::json payload) {
    require_nonempty(kind);
    require_nonempty(work_id);
    if (!payload.is_object()) {
        throw std::invalid_argument(
            "protocol-v4 node transition requires work_id and object payload");
    }
    const uint64_t sequence = next_sequence_candidate();
    jointledger::CompactNodeRecord record;
    record.kind = jointledger::node_kind_from_name(kind);
    record.logical_seq = sequence;
    record.tick = tick;
    record.work_id = std::move(work_id);
    record.payload = std::move(payload);
    node_transitions_.push_back(std::move(record));
    next_logical_seq_ = sequence + 1;
}

WorkLedgerWire WorkLedgerV2Publisher::publish(
    const uint64_t delivery_sequence, const uint64_t tick,
    std::vector<jointledger::ResourceRow> active_resources) {
    if (!has_previous_frame_) {
        if (delivery_sequence != 0) {
            throw std::invalid_argument(
                "initial protocol-v4 full frame must use delivery_sequence 0");
        }
    } else {
        if (last_delivery_sequence_ == std::numeric_limits<uint64_t>::max() ||
            delivery_sequence != last_delivery_sequence_ + 1) {
            throw std::invalid_argument(
                "protocol-v4 delivery_sequence is not consecutive");
        }
        if (tick < last_tick_) {
            throw std::invalid_argument("protocol-v4 frame tick regressed");
        }
    }

    const uint64_t sequence_end = next_logical_seq_ - 1;
    if (sequence_end < delivered_watermark_) {
        throw std::logic_error("protocol-v4 watermark exceeds callback sequence");
    }
    const uint64_t event_count = sequence_end - delivered_watermark_;
    if (event_count != node_transitions_.size()) {
        throw std::logic_error(
            "protocol-v4 callback buffers do not span one global sequence");
    }
    // The node callback buffer is already monotone by the single allocator.
    uint64_t expected_seq = delivered_watermark_ + 1;
    uint64_t previous_tick = has_previous_frame_ ? last_tick_ : 0;
    for (const auto& record : node_transitions_) {
        if (record.logical_seq != expected_seq++ ||
            record.tick < previous_tick || record.tick > tick) {
            throw std::logic_error(
                "protocol-v4 transitions are not a monotone callback prefix");
        }
        previous_tick = record.tick;
    }

    bool resources_equal =
        previous_resources_.size() == active_resources.size();
    if (resources_equal) {
        for (size_t i = 0; i < active_resources.size(); ++i) {
            if (!previous_resources_[i].typed_eq(active_resources[i])) {
                resources_equal = false;
                break;
            }
        }
    }
    const bool changed = event_count != 0 || !resources_equal;
    if (changed && state_version_ == std::numeric_limits<uint64_t>::max()) {
        throw std::overflow_error("protocol-v4 state version exhausted");
    }
    const uint64_t base_version = state_version_;
    const uint64_t result_version = base_version + (changed ? 1 : 0);
    const uint64_t logical_seq_start = delivered_watermark_ + 1;
    const uint64_t logical_seq_end = event_count == 0
        ? delivered_watermark_ : sequence_end;
    const bool full = !has_previous_frame_;

    jointledger::CompactFrameInput frame;
    frame.full = full;
    frame.epoch = epoch_;
    frame.delivery_sequence = delivery_sequence;
    frame.tick = tick;
    frame.base_watermark = delivered_watermark_;
    frame.seq_start = logical_seq_start;
    frame.seq_end = logical_seq_end;
    frame.event_count = event_count;
    frame.base_state_version = base_version;
    frame.result_state_version = result_version;
    frame.nodes = std::move(node_transitions_);
    frame.resources = active_resources;

    WorkLedgerWire wire;
    jointledger::encode_compact_frame(frame, wire.bytes);
    wire.full = full;
    wire.delivery_sequence = delivery_sequence;
    wire.tick = tick;
    wire.state_version = result_version;
    wire.logical_seq_start = logical_seq_start;
    wire.logical_seq_end = logical_seq_end;
    wire.logical_event_count = event_count;

    has_previous_frame_ = true;
    last_delivery_sequence_ = delivery_sequence;
    last_tick_ = tick;
    delivered_watermark_ = sequence_end;
    state_version_ = result_version;
    previous_resources_ = std::move(active_resources);
    return wire;
}

}  // namespace ExecutionDriven
}  // namespace AstraSim
