#include "astra-sim/workload/execution_driven/WorkLedgerV2Publisher.hh"

#include <algorithm>
#include <limits>
#include <set>
#include <stdexcept>
#include <utility>

namespace AstraSim {
namespace ExecutionDriven {

namespace {

const std::vector<std::string>& node_kinds() {
    static const std::vector<std::string> kinds{
        "commit", "issue", "progress", "completion", "ack", "release"};
    return kinds;
}

}  // namespace

WorkLedgerV2Publisher::WorkLedgerV2Publisher(std::string epoch)
    : epoch_(std::move(epoch)) {
    if (epoch_.empty()) {
        throw std::invalid_argument("protocol-v4 epoch must be nonempty");
    }
}

void WorkLedgerV2Publisher::require_array(const nlohmann::json& value,
                                          const char* field) {
    if (!value.is_array()) {
        throw std::invalid_argument(std::string(field) + " must be an array");
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
    if (std::find(node_kinds().begin(), node_kinds().end(), kind) ==
        node_kinds().end()) {
        throw std::invalid_argument("unknown protocol-v4 node transition kind");
    }
    if (work_id.empty() || !payload.is_object()) {
        throw std::invalid_argument(
            "protocol-v4 node transition requires work_id and object payload");
    }
    const uint64_t sequence = next_sequence_candidate();
    node_transitions_.push_back({
        {"logical_seq", sequence}, {"tick", tick},
        {"kind", kind}, {"work_id", std::move(work_id)},
        {"payload", std::move(payload)},
    });
    next_logical_seq_ = sequence + 1;
}

void WorkLedgerV2Publisher::append_pipeline_transition(
    nlohmann::json transition) {
    if (!transition.is_object() || transition.contains("first_logical_seq")) {
        throw std::invalid_argument(
            "pipeline transition must be an unsequenced object");
    }
    const uint64_t sequence = next_sequence_candidate();
    std::vector<TargetSlotKey> target_keys;
    std::vector<TargetSlotKey> new_target_keys;
    std::vector<TargetSlotKey> inserted_placeholders;
    std::set<TargetSlotKey> keys_in_transition;
    const auto slot_patches = transition.find("slot_patches");
    if (slot_patches != transition.end()) {
        if (!slot_patches->is_array()) {
            throw std::invalid_argument(
                "pipeline slot_patches must be an array");
        }
        target_keys.reserve(slot_patches->size());
        new_target_keys.reserve(slot_patches->size());
        for (auto& patch : *slot_patches) {
            if (!patch.is_object() || !patch.contains("form") ||
                !patch.at("form").is_string() ||
                patch.at("form").get_ref<const nlohmann::json::string_t&>() !=
                    "target_delta") {
                continue;
            }
            if (patch.contains("predecessor_logical_seq")) {
                throw std::invalid_argument(
                    "target delta predecessor is Publisher-owned");
            }
            if (!patch.contains("key") || !patch.at("key").is_object()) {
                throw std::invalid_argument("target delta key must be an object");
            }
            const nlohmann::json& key_row = patch.at("key");
            if (!key_row.contains("buffer_domain_id") ||
                !key_row.at("buffer_domain_id").is_string() ||
                !key_row.contains("role") ||
                !key_row.at("role").is_string() ||
                key_row.at("role").get_ref<const nlohmann::json::string_t&>() !=
                    "target" ||
                !key_row.contains("rank") ||
                !key_row.at("rank").is_number_integer() ||
                !key_row.contains("shard_ordinal") ||
                !key_row.at("shard_ordinal").is_null() ||
                !key_row.contains("slot_index") ||
                !key_row.at("slot_index").is_number_unsigned()) {
                throw std::invalid_argument(
                    "target delta key has invalid stable coordinates");
            }
            const TargetSlotKey key{
                key_row.at("buffer_domain_id").get<std::string>(),
                key_row.at("role").get<std::string>(),
                key_row.at("rank").get<int>(),
                std::nullopt,
                key_row.at("slot_index").get<size_t>()};
            if (std::get<0>(key).empty() || std::get<2>(key) < 0 ||
                std::get<4>(key) >= 2) {
                throw std::invalid_argument(
                    "target delta key is outside its stable key domain");
            }
            if (!keys_in_transition.insert(key).second) {
                throw std::invalid_argument(
                    "a pipeline transition may contain one target delta per key");
            }
            const auto previous = target_delta_predecessor_seq_.find(key);
            const uint64_t predecessor =
                previous == target_delta_predecessor_seq_.end()
                    ? 0 : previous->second;
            patch["predecessor_logical_seq"] = predecessor;
            target_keys.push_back(key);
            if (previous == target_delta_predecessor_seq_.end()) {
                new_target_keys.push_back(key);
            }
        }
    }
    inserted_placeholders = new_target_keys;
    try {
        for (const TargetSlotKey& key : new_target_keys) {
            const auto inserted = target_delta_predecessor_seq_.try_emplace(
                key, uint64_t{0});
            if (!inserted.second) {
                throw std::logic_error(
                    "target delta lineage changed during append preparation");
            }
        }
        transition["first_logical_seq"] = sequence;
        pipeline_transition_runs_.push_back(std::move(transition));
    } catch (...) {
        for (const TargetSlotKey& key : inserted_placeholders) {
            target_delta_predecessor_seq_.erase(key);
        }
        throw;
    }
    for (const TargetSlotKey& key : target_keys) {
        const auto found = target_delta_predecessor_seq_.find(key);
        found->second = sequence;
    }
    next_logical_seq_ = sequence + 1;
}

void WorkLedgerV2Publisher::append_descriptor_commit(
    nlohmann::json declaration) {
    if (!declaration.is_object()) {
        throw std::invalid_argument(
            "descriptor declaration must be an object");
    }
    descriptor_commits_.push_back(std::move(declaration));
}

nlohmann::json WorkLedgerV2Publisher::publish(
    const uint64_t delivery_sequence, const uint64_t tick,
    nlohmann::json cursor_patches,
    nlohmann::json active_resources) {
    require_array(cursor_patches, "cursor_patches");
    require_array(active_resources, "active_resources");

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
    if (event_count != node_transitions_.size() +
                           pipeline_transition_runs_.size()) {
        throw std::logic_error(
            "protocol-v4 callback buffers do not span one global sequence");
    }
    // Each callback buffer is already monotone by the single allocator.
    // Merge the two streams directly instead of making/sorting a T-sized
    // duplicate of the transition history at every delivery.
    size_t node_index = 0;
    size_t pipeline_index = 0;
    uint64_t expected_seq = delivered_watermark_ + 1;
    uint64_t previous_tick = has_previous_frame_ ? last_tick_ : 0;
    while (node_index < node_transitions_.size() ||
           pipeline_index < pipeline_transition_runs_.size()) {
        const bool take_node = pipeline_index ==
                pipeline_transition_runs_.size() ||
            (node_index < node_transitions_.size() &&
             node_transitions_[node_index].at("logical_seq").get<uint64_t>() <
                 pipeline_transition_runs_[pipeline_index]
                     .at("first_logical_seq").get<uint64_t>());
        const uint64_t sequence = take_node
            ? node_transitions_[node_index].at("logical_seq").get<uint64_t>()
            : pipeline_transition_runs_[pipeline_index]
                  .at("first_logical_seq").get<uint64_t>();
        const uint64_t transition_tick = take_node
            ? node_transitions_[node_index++].at("tick").get<uint64_t>()
            : pipeline_transition_runs_[pipeline_index++]
                  .at("tick_first").get<uint64_t>();
        if (sequence != expected_seq++ || transition_tick < previous_tick ||
            transition_tick > tick) {
            throw std::logic_error(
                "protocol-v4 transitions are not a monotone callback prefix");
        }
        previous_tick = transition_tick;
    }

    const bool changed = event_count != 0 || !descriptor_commits_.empty() ||
        !cursor_patches.empty() || active_resources != previous_resources_;
    if (changed && state_version_ == std::numeric_limits<uint64_t>::max()) {
        throw std::overflow_error("protocol-v4 state version exhausted");
    }
    const uint64_t base_version = state_version_;
    const uint64_t result_version = base_version + (changed ? 1 : 0);
    const uint64_t logical_seq_start = delivered_watermark_ + 1;
    const uint64_t logical_seq_end = event_count == 0
        ? delivered_watermark_ : sequence_end;
    const char* const snapshot_kind = has_previous_frame_ ? "delta" : "full";

    // Retain just one previous O(A) frame. Build that copy before consuming
    // callback buffers so validation/allocation failures leave the publisher
    // retryable with the same buffered transition prefix.
    nlohmann::json next_previous_resources = active_resources;

    nlohmann::json envelope{
        {"producer", "cpp_online"}, {"schema_version", 4},
        {"protocol_version", 4},
        {"bridge_protocol", "bridge-v4-20261011"},
        {"work_ledger_schema", "workledger-v4-20261011"},
        {"epoch", epoch_}, {"delivery_sequence", delivery_sequence},
        {"base_watermark", delivered_watermark_},
        {"logical_seq_start", logical_seq_start},
        {"logical_seq_end", logical_seq_end},
        {"logical_event_count", event_count},
        {"replay_token", "cpp_online/" + epoch_ + "/" +
                              std::to_string(delivery_sequence)},
        {"descriptor_version", 1}, {"base_state_version", base_version},
        {"result_state_version", result_version},
        {"snapshot_kind", snapshot_kind}, {"tick", tick},
        {"descriptor_commits", nlohmann::json::array()},
        {"node_transitions", nlohmann::json::array()},
        {"pipeline_transition_runs", nlohmann::json::array()},
        {"cursor_patches", std::move(cursor_patches)},
        {"active_resources", std::move(active_resources)},
    };

    // The arrays already exist in the envelope; swapping transfers their
    // backing vectors without copying the T-sized transition history.
    envelope["descriptor_commits"].swap(descriptor_commits_);
    envelope["node_transitions"].swap(node_transitions_);
    envelope["pipeline_transition_runs"].swap(pipeline_transition_runs_);

    has_previous_frame_ = true;
    last_delivery_sequence_ = delivery_sequence;
    last_tick_ = tick;
    delivered_watermark_ = sequence_end;
    state_version_ = result_version;
    previous_resources_.swap(next_previous_resources);
    return envelope;
}

}  // namespace ExecutionDriven
}  // namespace AstraSim
