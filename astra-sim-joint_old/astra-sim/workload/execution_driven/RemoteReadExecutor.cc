/******************************************************************************
Bounded, descriptor-driven remote-read execution over real Sys endpoints.
*******************************************************************************/

#include "astra-sim/workload/execution_driven/RemoteReadExecutor.hh"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <iterator>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <tuple>
#include <utility>

#include "astra-sim/system/Sys.hh"
#include "astra-sim/workload/Workload.hh"
#include "astra-sim/workload/execution_driven/PipelineKernel.hh"

namespace AstraSim {
namespace ExecutionDriven {
namespace {

using SignedWide = __int128_t;
using UnsignedWide = __uint128_t;
using SlotPatchStableKey =
    std::tuple<std::string, std::string, int,
               std::optional<uint64_t>, size_t>;
using TargetSlotResourceCoordinates = std::tuple<std::string, size_t>;

std::optional<TargetSlotResourceCoordinates>
target_slot_resource_coordinates(const std::string& stable_key) {
    if (stable_key.compare(0, 5, "slot:") != 0) return std::nullopt;
    const size_t target_mark = stable_key.rfind(":target:");
    const size_t source_mark = stable_key.rfind(":source:");
    if (target_mark == std::string::npos ||
        (source_mark != std::string::npos && source_mark > target_mark)) {
        return std::nullopt;
    }
    const std::string domain_id = stable_key.substr(5, target_mark - 5);
    const std::string suffix = stable_key.substr(
        target_mark + sizeof(":target:") - 1);
    if (domain_id.empty() || suffix.empty() ||
        suffix.find(':') != std::string::npos) {
        throw std::invalid_argument("pipeline target slot key is malformed");
    }
    size_t slot_parsed = 0;
    const uint64_t slot64 = std::stoull(suffix, &slot_parsed);
    if (slot_parsed != suffix.size() || slot64 >= 2) {
        throw std::invalid_argument("pipeline target slot key has invalid coordinates");
    }
    return TargetSlotResourceCoordinates{domain_id,
                                         static_cast<size_t>(slot64)};
}

SlotPatchStableKey slot_patch_stable_key(const nlohmann::json& row) {
    const std::string role = row.at("role").get<std::string>();
    const nlohmann::json& shard_value = row.at("shard_ordinal");
    if ((role == "source") == shard_value.is_null() ||
        (role != "source" && role != "target")) {
        throw std::invalid_argument("slot patch has invalid role/shard key");
    }
    const std::optional<uint64_t> shard = shard_value.is_null()
        ? std::nullopt
        : std::optional<uint64_t>(shard_value.get<uint64_t>());
    return {row.at("buffer_domain_id").get<std::string>(), role,
            row.at("rank").get<int>(), shard,
            row.at("slot_index").get<size_t>()};
}

class DiagnosticRowCaptureScope final {
  public:
    explicit DiagnosticRowCaptureScope(bool& active)
        : active_(active), previous_(active) {
        active_ = true;
    }
    ~DiagnosticRowCaptureScope() { active_ = previous_; }

  private:
    bool& active_;
    bool previous_;
};

struct IdProgression {
    uint64_t first = 0;
    uint64_t count = 0;
    uint64_t stride = 1;
};

uint64_t modular_inverse(uint64_t value, uint64_t modulus) {
    SignedWide old_r = static_cast<SignedWide>(value);
    SignedWide r = static_cast<SignedWide>(modulus);
    SignedWide old_s = 1;
    SignedWide s = 0;
    while (r != 0) {
        const SignedWide quotient = old_r / r;
        const SignedWide next_r = old_r - quotient * r;
        old_r = r;
        r = next_r;
        const SignedWide next_s = old_s - quotient * s;
        old_s = s;
        s = next_s;
    }
    if (old_r != 1) {
        throw std::logic_error("non-coprime modular inverse");
    }
    old_s %= static_cast<SignedWide>(modulus);
    if (old_s < 0) old_s += static_cast<SignedWide>(modulus);
    return static_cast<uint64_t>(old_s);
}

bool progression_last(const IdProgression& p, uint64_t* last) {
    if (last == nullptr || p.count == 0) return false;
    const uint64_t stride = p.count == 1 ? 1 : p.stride;
    if (stride == 0 ||
        static_cast<UnsignedWide>(p.first) +
                static_cast<UnsignedWide>(p.count - 1) * stride >
            std::numeric_limits<uint64_t>::max()) {
        return false;
    }
    *last = p.first + (p.count - 1) * stride;
    return true;
}

uint64_t progression_intersection_count(IdProgression left,
                                        IdProgression right) {
    if (left.count == 0 || right.count == 0) return 0;
    if (left.count > 1 && left.stride == 0) return 0;
    if (right.count > 1 && right.stride == 0) return 0;
    if (left.count == 1) left.stride = 1;
    if (right.count == 1) right.stride = 1;
    uint64_t left_last = 0;
    uint64_t right_last = 0;
    if (!progression_last(left, &left_last) ||
        !progression_last(right, &right_last)) {
        throw std::overflow_error("progression intersection overflows uint64");
    }
    const uint64_t low = std::max(left.first, right.first);
    const uint64_t high = std::min(left_last, right_last);
    if (low > high) return 0;
    const uint64_t divisor = std::gcd(left.stride, right.stride);
    const SignedWide difference = static_cast<SignedWide>(right.first) -
        static_cast<SignedWide>(left.first);
    if (difference % static_cast<SignedWide>(divisor) != 0) return 0;
    const uint64_t right_modulus = right.stride / divisor;
    uint64_t multiplier = 0;
    if (right_modulus != 1) {
        const uint64_t left_reduced = (left.stride / divisor) % right_modulus;
        SignedWide reduced = difference / static_cast<SignedWide>(divisor);
        reduced %= static_cast<SignedWide>(right_modulus);
        if (reduced < 0) reduced += static_cast<SignedWide>(right_modulus);
        multiplier = static_cast<uint64_t>(
            (static_cast<UnsignedWide>(reduced) *
             modular_inverse(left_reduced, right_modulus)) % right_modulus);
    }
    const UnsignedWide period =
        static_cast<UnsignedWide>(left.stride) * right_modulus;
    const UnsignedWide product =
        static_cast<UnsignedWide>(left.stride) * multiplier;
    const UnsignedWide product_mod = product % period;
    const UnsignedWide base_mod = left.first % period;
    const UnsignedWide residue = product_mod >= period - base_mod
        ? product_mod - (period - base_mod)
        : product_mod + base_mod;
    if (residue > high) return 0;
    UnsignedWide first = residue;
    if (first < low) {
        const UnsignedWide delta = static_cast<UnsignedWide>(low) - first;
        const UnsignedWide steps = delta / period + (delta % period != 0);
        if (steps != 0 && period >
                (static_cast<UnsignedWide>(high) - first) / steps) {
            return 0;
        }
        first += period * steps;
    }
    if (first > high) return 0;
    const UnsignedWide hits =
        (static_cast<UnsignedWide>(high) - first) / period + 1;
    if (hits > std::numeric_limits<uint64_t>::max()) {
        throw std::overflow_error("progression intersection count overflows");
    }
    return static_cast<uint64_t>(hits);
}

uint64_t add_checked(const uint64_t left, const uint64_t right,
                     const char* label) {
    if (right > std::numeric_limits<uint64_t>::max() - left) {
        throw std::overflow_error(std::string("remote-read ") + label +
                                  " overflows uint64");
    }
    return left + right;
}

uint64_t mul_checked(const uint64_t left, const uint64_t right,
                     const char* label) {
    if (left != 0 && right > std::numeric_limits<uint64_t>::max() / left) {
        throw std::overflow_error(std::string("remote-read ") + label +
                                  " overflows uint64");
    }
    return left * right;
}

const PipelineShardTile& shard_for(const PipelineTile& tile,
                                   const uint64_t shard_ordinal) {
    const auto it = std::find_if(
        tile.shards.begin(), tile.shards.end(),
        [shard_ordinal](const PipelineShardTile& shard) {
            return shard.shard_ordinal == shard_ordinal;
        });
    if (it == tile.shards.end()) {
        throw std::invalid_argument("endpoint run references missing shard");
    }
    return *it;
}

bool is_send_leg(const PipelineEndpointLeg leg) {
    return leg == PipelineEndpointLeg::DataSend ||
        leg == PipelineEndpointLeg::FillAckSend ||
        leg == PipelineEndpointLeg::ReleaseAckSend;
}

bool is_ack_leg(const PipelineEndpointLeg leg) {
    return leg == PipelineEndpointLeg::FillAckSend ||
        leg == PipelineEndpointLeg::FillAckRecv ||
        leg == PipelineEndpointLeg::ReleaseAckSend ||
        leg == PipelineEndpointLeg::ReleaseAckRecv;
}

const char* leg_name(const PipelineEndpointLeg leg) {
    switch (leg) {
        case PipelineEndpointLeg::DataSend: return "data_send";
        case PipelineEndpointLeg::DataRecv: return "data_recv";
        case PipelineEndpointLeg::FillAckSend: return "fill_ack_send";
        case PipelineEndpointLeg::FillAckRecv: return "fill_ack_recv";
        case PipelineEndpointLeg::ReleaseAckSend: return "release_ack_send";
        case PipelineEndpointLeg::ReleaseAckRecv: return "release_ack_recv";
        case PipelineEndpointLeg::BodyMarker: return "body_marker";
    }
    return "invalid";
}

GraphAnchorRole graph_anchor_role(const PipelineAnchorRole role) {
    switch (role) {
        case PipelineAnchorRole::Start: return GraphAnchorRole::Start;
        case PipelineAnchorRole::Frontier: return GraphAnchorRole::Frontier;
        case PipelineAnchorRole::Service: return GraphAnchorRole::Service;
        case PipelineAnchorRole::Settle: return GraphAnchorRole::Settle;
    }
    return GraphAnchorRole::Start;
}

nlohmann::json identity_json(const PipelineIdentity& identity) {
    return nlohmann::json{
        {"request_id", identity.request_id},
        {"stage", identity.stage == PipelineStage::Prefill ? "prefill" : "decode"},
        {"generation", identity.generation},
        {"stage_epoch", identity.stage_epoch},
        {"segment_ordinal", identity.segment_ordinal},
        {"rank", identity.rank},
        {"staging_allocation_id", identity.staging_allocation_id},
        {"segment_id", identity.segment_id},
        {"buffer_domain_id", identity.buffer_domain_id},
    };
}

}  // namespace

struct RemoteReadExecutor::PreparedPlan {
    std::vector<std::unique_ptr<RuntimeDescriptor>> descriptors;
    std::vector<std::unordered_map<uint64_t, AnchorBinding>>
        anchors_by_rank;
    std::vector<std::vector<TerminalWatcher>> terminal_watchers_by_rank;
};

RemoteReadExecutor::RemoteReadExecutor(
    std::vector<std::shared_ptr<NodeStoreGraphSource>> graph_sources,
    std::vector<Workload*> workloads,
    TileTransportAdapter* transport,
    ScheduleIssuePass schedule_issue_pass)
    : graph_sources_(std::move(graph_sources)),
      workloads_(std::move(workloads)),
      transport_(transport),
      schedule_issue_pass_(std::move(schedule_issue_pass)) {
    if (graph_sources_.empty() || graph_sources_.size() != workloads_.size() ||
        transport_ == nullptr || schedule_issue_pass_ == nullptr) {
        throw std::invalid_argument(
            "RemoteReadExecutor requires graph sources, workloads, transport, "
            "and deferred issue scheduling");
    }
    physical_anchors_by_rank_.resize(graph_sources_.size());
    terminal_watchers_by_rank_.resize(graph_sources_.size());
    terminal_watcher_indexes_by_rank_.resize(graph_sources_.size());
    for (size_t rank = 0; rank < graph_sources_.size(); ++rank) {
        if (graph_sources_[rank] == nullptr || workloads_[rank] == nullptr ||
            workloads_[rank]->sys == nullptr ||
            workloads_[rank]->sys->id != static_cast<int>(rank)) {
            throw std::invalid_argument(
                "RemoteReadExecutor rank arrays are inconsistent");
        }
    }
}

void RemoteReadExecutor::set_ledger_publishers(
    LedgerNodeTransitionPublisher node_publisher,
    PipelineTransitionPublisher pipeline_publisher,
    DescriptorCommitPublisher descriptor_publisher) {
    node_transition_publisher_ = std::move(node_publisher);
    pipeline_transition_publisher_ = std::move(pipeline_publisher);
    descriptor_commit_publisher_ = std::move(descriptor_publisher);
}

void RemoteReadExecutor::set_watch_reference_counter(
    WatchReferenceCounter counter) {
    watch_reference_counter_ = std::move(counter);
}

void RemoteReadExecutor::notify_watch_registry_changed() {
    std::vector<RuntimeDescriptor*> pending;
    pending.swap(pending_watch_retirements_);
    for (RuntimeDescriptor* owner : pending) {
        if (owner == nullptr || owner->retired) continue;
        owner->queued_for_watch_retirement = false;
        try_retire_state(*owner);
    }
}

RemoteReadExecutor::ResourceMutationObserverToken
RemoteReadExecutor::add_resource_mutation_observer(
    std::weak_ptr<void> lifetime, ResourceMutationObserver observer) {
    if (lifetime.expired() || !observer ||
        next_resource_observer_token_ == 0 ||
        next_resource_observer_token_ ==
            std::numeric_limits<ResourceMutationObserverToken>::max()) {
        throw std::invalid_argument(
            "resource mutation observer requires a live lifetime and callback");
    }
    const ResourceMutationObserverToken token = next_resource_observer_token_++;
    resource_mutation_observers_.emplace(
        token, ResourceObserverEntry{std::move(lifetime), std::move(observer)});
    return token;
}

void RemoteReadExecutor::remove_resource_mutation_observer(
    const ResourceMutationObserverToken token) {
    resource_mutation_observers_.erase(token);
}

std::vector<std::pair<std::string, uint64_t>>
RemoteReadExecutor::resource_version_vector() const {
    std::vector<std::pair<std::string, uint64_t>> result;
    result.reserve(resource_versions_.size());
    for (const auto& entry : resource_versions_) {
        result.emplace_back(entry.first.first + ":" + entry.first.second,
                            entry.second);
    }
    return result;
}

uint64_t RemoteReadExecutor::resource_mutation_before(
    const std::string& resource_kind, const std::string& stable_key,
    const int target_rank_hint) {
    const auto key = std::make_pair(resource_kind, stable_key);
    const auto found = resource_versions_.find(key);
    const uint64_t version = found == resource_versions_.end()
        ? 0 : found->second;
    if (pipeline_transition_publisher_ != nullptr &&
        resource_kind == "pipeline_slot") {
        const auto target_coordinates =
            target_slot_resource_coordinates(stable_key);
        if (target_coordinates.has_value()) {
            const std::string& domain_id = std::get<0>(*target_coordinates);
            int target_rank = target_rank_hint;
            if (target_rank < 0) {
                const auto ranked = buffer_domain_target_ranks_.find(domain_id);
                if (ranked != buffer_domain_target_ranks_.end()) {
                    target_rank = ranked->second;
                } else {
                    const auto provenance =
                        buffer_domain_provenance_.find(domain_id);
                    if (provenance != buffer_domain_provenance_.end()) {
                        target_rank = provenance->second.target_rank;
                    }
                }
            }
            if (target_rank < 0) {
                fail_closed(
                    "new target slot initialization requires its immutable rank");
            }
            const TargetSlotStableKey target_key{
                domain_id, "target", target_rank, std::nullopt,
                std::get<1>(*target_coordinates)};
            if (pending_target_slot_deltas_.find(target_key) ==
                pending_target_slot_deltas_.end()) {
                PendingTargetSlotDelta pending;
                const auto domain = buffer_domains_.find(domain_id);
                if (domain != buffer_domains_.end()) {
                    const auto target_rank = buffer_domain_target_ranks_.find(
                        domain_id);
                    if (target_rank == buffer_domain_target_ranks_.end() ||
                        target_rank->second != std::get<2>(target_key)) {
                        fail_closed(
                            "target slot mutation rank disagrees with its domain");
                    }
                    const SlotState& slot = domain->second.target_slots[
                        std::get<4>(target_key)];
                    pending.before_slot_generation = slot.generation;
                    pending.predecessor_consumed_count =
                        static_cast<uint64_t>(
                            slot.consumed_target_shards.size());
                    pending.before_present = true;
                } else {
                    const auto provenance =
                        buffer_domain_provenance_.find(domain_id);
                    if (provenance != buffer_domain_provenance_.end()) {
                        if (provenance->second.target_rank !=
                            std::get<2>(target_key)) {
                            fail_closed(
                                "target slot provenance rank disagrees with its domain");
                        }
                        pending.before_slot_generation =
                            provenance->second.target_slot_generations[
                                std::get<4>(target_key)];
                    }
                    pending.before_present = false;
                }
                pending_target_slot_deltas_.emplace(
                    target_key, std::move(pending));
            }
        } else {
            const auto capture_t0 = std::chrono::steady_clock::now();
            resource_mutation_before_rows_[key] =
                resource_state_row_json(resource_kind, stable_key);
            producer_metrics_.resource_row_capture_ns +=
                static_cast<uint64_t>(std::chrono::duration_cast<
                    std::chrono::nanoseconds>(std::chrono::steady_clock::now() -
                                              capture_t0).count());
            ++producer_metrics_.resource_row_capture_calls;
        }
    } else if (pipeline_transition_publisher_ != nullptr &&
               resource_kind == "pipeline_credit") {
        capture_credit_row_before(stable_key);
    }
    notify_resource_mutation(ResourceMutation{
        true, resource_kind, stable_key, version, Sys::boostedTick()});
    return version;
}

void RemoteReadExecutor::resource_mutation_after(
    const std::string& resource_kind, const std::string& stable_key,
    const uint64_t previous_version, const bool allow_target_noop) {
    const auto key = std::make_pair(resource_kind, stable_key);
    const auto found = resource_versions_.find(key);
    const uint64_t current = found == resource_versions_.end()
        ? 0 : found->second;
    if (current != previous_version || current ==
            std::numeric_limits<uint64_t>::max()) {
        fail_closed("executor resource version changed during mutation");
    }
    const uint64_t next = current + 1;
    resource_versions_[key] = next;
    if (pipeline_transition_publisher_ != nullptr &&
        resource_kind == "pipeline_slot") {
        const auto target_coordinates =
            target_slot_resource_coordinates(stable_key);
        if (target_coordinates.has_value()) {
            const std::string& domain_id = std::get<0>(*target_coordinates);
            int target_rank = -1;
            const auto ranked = buffer_domain_target_ranks_.find(domain_id);
            if (ranked != buffer_domain_target_ranks_.end()) {
                target_rank = ranked->second;
            } else {
                const auto provenance = buffer_domain_provenance_.find(domain_id);
                if (provenance != buffer_domain_provenance_.end()) {
                    target_rank = provenance->second.target_rank;
                }
            }
            if (target_rank < 0) {
                fail_closed("target slot mutation lost its domain rank");
            }
            const TargetSlotStableKey target_key{
                domain_id, "target", target_rank, std::nullopt,
                std::get<1>(*target_coordinates)};
            const auto pending = pending_target_slot_deltas_.find(target_key);
            if (pending == pending_target_slot_deltas_.end()) {
                if (!allow_target_noop) {
                    fail_closed(
                        "target slot mutation lacks scalar before capture");
                }
            }
            if (pending != pending_target_slot_deltas_.end() &&
                pending->second.operations.empty()) {
                if (!pending->second.before_present) {
                    const auto domain = buffer_domains_.find(domain_id);
                    if (domain == buffer_domains_.end()) {
                        fail_closed(
                            "target slot initialize lacks its new buffer domain");
                    }
                    const SlotState& slot = domain->second.target_slots[
                        std::get<4>(target_key)];
                    append_target_slot_operation(
                        stable_key,
                        nlohmann::json{{"op", "initialize"},
                                       {"slot_generation", slot.generation}});
                } else if (allow_target_noop) {
                    pending_target_slot_deltas_.erase(pending);
                } else {
                    fail_closed(
                        "target slot mutation did not record its operation");
                }
            }
        } else {
            const auto before = resource_mutation_before_rows_.find(key);
            if (before == resource_mutation_before_rows_.end()) {
                fail_closed("resource mutation lacks its sparse before row");
            }
            const auto capture_t0 = std::chrono::steady_clock::now();
            record_resource_row_patch(
                resource_kind, before->second,
                resource_state_row_json(resource_kind, stable_key));
            producer_metrics_.resource_row_capture_ns +=
                static_cast<uint64_t>(std::chrono::duration_cast<
                    std::chrono::nanoseconds>(std::chrono::steady_clock::now() -
                                              capture_t0).count());
            ++producer_metrics_.resource_row_capture_calls;
            resource_mutation_before_rows_.erase(before);
        }
    }
    notify_resource_mutation(ResourceMutation{
        false, resource_kind, stable_key, next, Sys::boostedTick()});
}

void RemoteReadExecutor::notify_resource_mutation(
    const ResourceMutation& mutation) {
    std::vector<std::pair<std::shared_ptr<void>, ResourceMutationObserver>>
        callbacks;
    for (auto it = resource_mutation_observers_.begin();
         it != resource_mutation_observers_.end();) {
        std::shared_ptr<void> keep_alive = it->second.lifetime.lock();
        if (keep_alive == nullptr) {
            it = resource_mutation_observers_.erase(it);
            continue;
        }
        callbacks.emplace_back(std::move(keep_alive), it->second.observer);
        ++it;
    }
    for (const auto& callback : callbacks) {
        try {
            callback.second(mutation);
        } catch (const std::exception& error) {
            fail_closed(std::string("resource mutation observer threw: ") +
                        error.what());
        } catch (...) {
            fail_closed("resource mutation observer threw an unknown error");
        }
    }
}

std::string RemoteReadExecutor::slot_resource_key(
    const std::string& domain_id, const std::string& role, const int rank,
    const uint64_t shard_ordinal, const size_t slot_index) const {
    if (role == "target") {
        return "slot:" + domain_id + ":target:" +
            std::to_string(slot_index);
    }
    if (role != "source") {
        fail_closed("unknown executor slot resource role");
    }
    return "slot:" + domain_id + ":source:" + std::to_string(rank) + ":" +
        std::to_string(shard_ordinal) + ":" + std::to_string(slot_index);
}

std::string RemoteReadExecutor::credit_resource_key(
    const std::string& domain_id, const int rank,
    const uint64_t shard_ordinal) const {
    return "credit:" + domain_id + ":source:" + std::to_string(rank) + ":" +
        std::to_string(shard_ordinal);
}

void RemoteReadExecutor::record_resource_row_patch(
    const std::string& resource_kind, nlohmann::json before,
    nlohmann::json after) {
    if (before == after) return;
    if (resource_kind == "pipeline_slot") {
        const nlohmann::json& identity = before.is_null() ? after : before;
        if (!identity.is_null() &&
            identity.value("role", std::string()) == "target") {
            fail_closed("target slot patches must use v4 target deltas");
        }
    }
    nlohmann::json patch{{"before", std::move(before)},
                         {"after", std::move(after)}};
    if (resource_kind == "pipeline_slot") {
        pending_slot_patches_.push_back(std::move(patch));
    } else if (resource_kind == "pipeline_credit") {
        pending_credit_patches_.push_back(std::move(patch));
    }
}

void RemoteReadExecutor::append_target_slot_operation(
    const std::string& stable_key, nlohmann::json operation) {
    if (!pipeline_transition_publisher_) return;
    const auto coordinates = target_slot_resource_coordinates(stable_key);
    if (!coordinates.has_value()) {
        fail_closed("target slot operation has a non-target resource key");
    }
    const std::string& domain_id = std::get<0>(*coordinates);
    int target_rank = -1;
    const auto ranked = buffer_domain_target_ranks_.find(domain_id);
    if (ranked != buffer_domain_target_ranks_.end()) {
        target_rank = ranked->second;
    } else {
        const auto provenance = buffer_domain_provenance_.find(domain_id);
        if (provenance != buffer_domain_provenance_.end()) {
            target_rank = provenance->second.target_rank;
        }
    }
    if (target_rank < 0) {
        fail_closed("target slot operation lost its domain rank");
    }
    const TargetSlotStableKey key{
        domain_id, "target", target_rank, std::nullopt,
        std::get<1>(*coordinates)};
    const auto pending = pending_target_slot_deltas_.find(key);
    if (pending == pending_target_slot_deltas_.end()) {
        fail_closed("target slot operation lacks a scalar before capture");
    }
    if (!operation.is_object() || !operation.contains("op") ||
        !operation.contains("slot_generation")) {
        fail_closed("target slot operation is missing its tag or generation");
    }
    if (operation.at("op") == "claim") {
        producer_metrics_.target_claim_shard_elements +=
            operation.at("target_shards").size();
    } else if (operation.at("op") == "consume") {
        producer_metrics_.target_consume_added_shards +=
            operation.at("add_shards").size();
    }
    pending->second.operations.push_back(std::move(operation));
    ++producer_metrics_.target_delta_operations;
}

void RemoteReadExecutor::record_target_slot_claim(
    const std::string& stable_key, const SlotState& slot) {
    if (!pipeline_transition_publisher_) return;
    nlohmann::json target_shards = nlohmann::json::array();
    for (const uint64_t shard : slot.target_shards_to_consume) {
        target_shards.push_back(shard);
    }
    nlohmann::json owner{
        {"segment_id", slot.owner_segment_id},
        {"pipeline_generation", slot.owner_pipeline_generation},
        {"stage_epoch", slot.owner_stage_epoch},
        {"staging_allocation_id", slot.owner_staging_allocation_id},
        {"group_ordinal", slot.owner_group_ordinal},
        {"tile_ordinal", slot.owner_local_tile},
    };
    append_target_slot_operation(
        stable_key,
        nlohmann::json{{"op", "claim"},
                       {"slot_generation", slot.generation},
                       {"owner", std::move(owner)},
                       {"target_shards", std::move(target_shards)}});
}

void RemoteReadExecutor::record_target_slot_consume(
    const std::string& stable_key, const uint64_t slot_generation,
    const int terminal_rank, const uint64_t terminal_store_id,
    const size_t consume_run_index,
    const uint64_t* const added_shards,
    const size_t added_shard_count, const bool cleared) {
    if (!pipeline_transition_publisher_) return;
    if (added_shards == nullptr || added_shard_count == 0) {
        fail_closed("target consume additions must be nonempty sorted unique IDs");
    }
    nlohmann::json additions = nlohmann::json::array();
    for (size_t index = 0; index < added_shard_count; ++index) {
        if (index != 0 && added_shards[index - 1] >= added_shards[index]) {
            fail_closed(
                "target consume additions must be nonempty sorted unique IDs");
        }
        additions.push_back(added_shards[index]);
    }
    append_target_slot_operation(
        stable_key,
        nlohmann::json{{"op", "consume"},
                       {"slot_generation", slot_generation},
                       {"cause", {
                           {"terminal_rank", terminal_rank},
                           {"terminal_store_id", terminal_store_id},
                           {"consume_run_index", consume_run_index},
                       }},
                       {"add_shards", std::move(additions)},
                       {"cleared", cleared}});
}

void RemoteReadExecutor::record_target_slot_request_close_clear(
    const std::string& stable_key, const uint64_t slot_generation) {
    append_target_slot_operation(
        stable_key,
        nlohmann::json{{"op", "request_close_clear"},
                       {"slot_generation", slot_generation}});
}

void RemoteReadExecutor::record_target_slot_erase(
    const std::string& stable_key, const uint64_t slot_generation) {
    append_target_slot_operation(
        stable_key,
        nlohmann::json{{"op", "erase"},
                       {"slot_generation", slot_generation}});
}

void RemoteReadExecutor::capture_credit_row_before(
    const std::string& stable_key) {
    if (!pipeline_transition_publisher_) return;
    if (stable_key.compare(0, 7, "credit:") != 0) {
        fail_closed("pipeline credit key lacks its stable prefix");
    }
    const size_t source_mark = stable_key.rfind(":source:");
    if (source_mark == std::string::npos) {
        fail_closed("pipeline credit key lacks its source coordinates");
    }
    const std::string domain_id = stable_key.substr(7, source_mark - 7);
    const std::string suffix = stable_key.substr(source_mark + 8);
    const size_t separator = suffix.find(':');
    if (separator == std::string::npos) {
        fail_closed("pipeline credit key has malformed coordinates");
    }
    size_t rank_parsed = 0;
    size_t shard_parsed = 0;
    const int rank = std::stoi(suffix.substr(0, separator), &rank_parsed);
    const std::string rank_text = suffix.substr(0, separator);
    const uint64_t shard = std::stoull(suffix.substr(separator + 1),
                                       &shard_parsed);
    if (rank_parsed != rank_text.size() ||
        shard_parsed != suffix.size() - separator - 1) {
        fail_closed("pipeline credit key has malformed numeric coordinates");
    }
    const auto key = std::make_tuple(domain_id, rank, shard);
    if (pending_credit_row_before_.find(key) !=
        pending_credit_row_before_.end()) {
        return;
    }
    const auto capture_t0 = std::chrono::steady_clock::now();
    nlohmann::json before = resource_state_row_json(
        "pipeline_credit", stable_key);
    producer_metrics_.resource_row_capture_ns +=
        static_cast<uint64_t>(std::chrono::duration_cast<
            std::chrono::nanoseconds>(std::chrono::steady_clock::now() -
                                      capture_t0).count());
    ++producer_metrics_.resource_row_capture_calls;
    pending_credit_row_before_.emplace(
        key, PendingCreditRowCapture{stable_key, std::move(before)});
}

void RemoteReadExecutor::capture_source_credit_row_before(
    const std::string& domain_id, const int rank,
    const uint64_t shard_ordinal) {
    capture_credit_row_before(
        credit_resource_key(domain_id, rank, shard_ordinal));
}

void RemoteReadExecutor::flush_pending_credit_row_patches() {
    for (auto& entry : pending_credit_row_before_) {
        const auto capture_t0 = std::chrono::steady_clock::now();
        nlohmann::json after = resource_state_row_json(
            "pipeline_credit", entry.second.stable_key);
        producer_metrics_.resource_row_capture_ns +=
            static_cast<uint64_t>(std::chrono::duration_cast<
                std::chrono::nanoseconds>(std::chrono::steady_clock::now() -
                                          capture_t0).count());
        ++producer_metrics_.resource_row_capture_calls;
        const size_t old_size = pending_credit_patches_.size();
        record_resource_row_patch("pipeline_credit",
                                  std::move(entry.second.before),
                                  std::move(after));
        producer_metrics_.credit_patch_rows +=
            pending_credit_patches_.size() - old_size;
    }
    pending_credit_row_before_.clear();
}

void RemoteReadExecutor::flush_pending_slot_row_patches() {
    std::map<SlotPatchStableKey, nlohmann::json> coalesced;
    for (auto& patch : pending_slot_patches_) {
        const nlohmann::json& before = patch.at("before");
        const nlohmann::json& after = patch.at("after");
        if (before.is_null() && after.is_null()) {
            fail_closed("slot row patch has no stable identity");
        }
        const nlohmann::json& identity = before.is_null() ? after : before;
        const SlotPatchStableKey key = slot_patch_stable_key(identity);
        if (std::get<1>(key) != "source") {
            fail_closed("legacy full-row patches are restricted to source slots");
        }
        if (!before.is_null() && !after.is_null() &&
            slot_patch_stable_key(after) != key) {
            fail_closed("slot row patch changes its stable key");
        }
        auto found = coalesced.find(key);
        if (found == coalesced.end()) {
            coalesced.emplace(key, std::move(patch));
            continue;
        }
        if (found->second.at("after") != patch.at("before")) {
            fail_closed("same-transition slot row before chain is discontinuous");
        }
        found->second["after"] = std::move(patch["after"]);
    }

    uint64_t target_patch_rows = 0;
    for (auto& entry : pending_target_slot_deltas_) {
        if (entry.second.operations.empty()) {
            fail_closed("target slot delta has no recorded operation");
        }
        const SlotPatchStableKey key = entry.first;
        nlohmann::json operations = nlohmann::json::array();
        for (auto& operation : entry.second.operations) {
            operations.push_back(std::move(operation));
        }
        nlohmann::json delta{
            {"form", "target_delta"},
            {"key", {
                {"buffer_domain_id", std::get<0>(entry.first)},
                {"role", std::get<1>(entry.first)},
                {"rank", std::get<2>(entry.first)},
                {"shard_ordinal", nullptr},
                {"slot_index", std::get<4>(entry.first)},
            }},
            {"before_slot_generation",
             entry.second.before_slot_generation},
            {"predecessor_consumed_count",
             entry.second.predecessor_consumed_count},
            {"operations", std::move(operations)},
        };
        if (!coalesced.emplace(key, std::move(delta)).second) {
            fail_closed("target and source slot patches share a stable key");
        }
        ++target_patch_rows;
    }
    pending_slot_patches_.clear();
    std::vector<nlohmann::json> canonical;
    canonical.reserve(coalesced.size());
    for (auto& entry : coalesced) {
        if (entry.second.contains("form")) {
            canonical.push_back(std::move(entry.second));
            continue;
        }
        if (entry.second.at("before") == entry.second.at("after")) continue;
        canonical.push_back(std::move(entry.second));
    }
    pending_target_slot_deltas_.clear();
    producer_metrics_.slot_patch_rows += canonical.size();
    producer_metrics_.target_delta_patch_rows += target_patch_rows;
    pending_slot_patches_ = std::move(canonical);
}

void RemoteReadExecutor::record_obligation_row_patch(
    nlohmann::json before, nlohmann::json after) {
    if (before == after) return;
    pending_obligation_patches_.push_back({
        {"before", std::move(before)}, {"after", std::move(after)}});
}

void RemoteReadExecutor::flush_pending_obligation_row_patches() {
    if (pending_obligation_patches_.empty()) return;
    std::map<std::string, nlohmann::json> coalesced;
    const auto stable_key = [](const nlohmann::json& row) {
        if (!row.is_object() || !row.contains("obligation_id") ||
            !row.at("obligation_id").is_string()) {
            RemoteReadExecutor::fail_closed(
                "obligation row patch lacks a string stable identity");
        }
        return row.at("obligation_id").get<std::string>();
    };
    for (auto& patch : pending_obligation_patches_) {
        const nlohmann::json& before = patch.at("before");
        const nlohmann::json& after = patch.at("after");
        if (before.is_null() && after.is_null()) {
            fail_closed("obligation row patch has no stable identity");
        }
        const std::string key = stable_key(before.is_null() ? after : before);
        if (!before.is_null() && !after.is_null() &&
            stable_key(after) != key) {
            fail_closed("obligation row patch changes its stable key");
        }
        auto found = coalesced.find(key);
        if (found == coalesced.end()) {
            coalesced.emplace(key, std::move(patch));
            continue;
        }
        if (found->second.at("after") != patch.at("before")) {
            fail_closed(
                "same-transition obligation row before chain is discontinuous");
        }
        found->second["after"] = std::move(patch["after"]);
    }
    pending_obligation_patches_.clear();
    std::vector<nlohmann::json> canonical;
    canonical.reserve(coalesced.size());
    for (auto& entry : coalesced) {
        if (entry.second.at("before") == entry.second.at("after")) continue;
        canonical.push_back(std::move(entry.second));
    }
    producer_metrics_.obligation_patch_rows += canonical.size();
    pending_obligation_patches_ = std::move(canonical);
}

void RemoteReadExecutor::count_row_helper_visit() const {
    if (diagnostic_row_capture_active_) {
        ++producer_metrics_.diagnostic_row_helper_visits;
    } else {
        ++producer_metrics_.row_helper_visits;
    }
}

nlohmann::json RemoteReadExecutor::resource_state_row_json(
    const std::string& resource_kind, const std::string& stable_key) const {
    const auto parse_uint = [](const std::string& value) -> uint64_t {
        size_t parsed = 0;
        const uint64_t result = std::stoull(value, &parsed);
        if (parsed != value.size()) {
            throw std::invalid_argument("invalid stable resource key number");
        }
        return result;
    };
    if (resource_kind == "pipeline_slot") {
        if (stable_key.compare(0, 5, "slot:") != 0) {
            fail_closed("pipeline slot key lacks its stable prefix");
        }
        const size_t target_mark = stable_key.rfind(":target:");
        const size_t source_mark = stable_key.rfind(":source:");
        const bool target = target_mark != std::string::npos &&
            (source_mark == std::string::npos || target_mark > source_mark);
        const size_t mark = target ? target_mark : source_mark;
        if (mark == std::string::npos) {
            fail_closed("pipeline slot key lacks its role");
        }
        const std::string domain_id = stable_key.substr(5, mark - 5);
        const auto domain = buffer_domains_.find(domain_id);
        if (domain == buffer_domains_.end()) return nullptr;
        const std::string suffix = stable_key.substr(
            mark + (target ? sizeof(":target:") - 1
                           : sizeof(":source:") - 1));
        if (target) {
            const size_t slot_index = static_cast<size_t>(parse_uint(suffix));
            if (slot_index >= domain->second.target_slots.size()) {
                fail_closed("pipeline target slot key index is outside domain");
            }
            const auto target_rank = buffer_domain_target_ranks_.find(domain_id);
            const int rank = target_rank == buffer_domain_target_ranks_.end()
                ? -1 : target_rank->second;
            return slot_row_json(domain_id, "target", rank, 0, slot_index,
                                 domain->second.target_slots[slot_index]);
        }
        const size_t first = suffix.find(':');
        const size_t second = first == std::string::npos
            ? std::string::npos : suffix.find(':', first + 1);
        if (first == std::string::npos || second == std::string::npos) {
            fail_closed("pipeline source slot key has malformed coordinates");
        }
        const int rank = std::stoi(suffix.substr(0, first));
        const uint64_t shard = parse_uint(
            suffix.substr(first + 1, second - first - 1));
        const size_t slot_index = static_cast<size_t>(
            parse_uint(suffix.substr(second + 1)));
        const auto source = domain->second.source_slots.find({rank, shard});
        if (source == domain->second.source_slots.end()) return nullptr;
        if (slot_index >= source->second.size()) {
            fail_closed("pipeline source slot key index is outside domain");
        }
        return slot_row_json(domain_id, "source", rank, shard, slot_index,
                             source->second[slot_index]);
    }
    if (resource_kind == "pipeline_credit") {
        if (stable_key.compare(0, 7, "credit:") != 0) {
            fail_closed("pipeline credit key lacks its stable prefix");
        }
        const size_t source_mark = stable_key.rfind(":source:");
        if (source_mark == std::string::npos) {
            fail_closed("pipeline credit key lacks its source role");
        }
        const std::string domain_id = stable_key.substr(7, source_mark - 7);
        const std::string suffix = stable_key.substr(source_mark + 8);
        const size_t separator = suffix.find(':');
        if (separator == std::string::npos) {
            fail_closed("pipeline credit key has malformed coordinates");
        }
        const int rank = std::stoi(suffix.substr(0, separator));
        const uint64_t shard = parse_uint(suffix.substr(separator + 1));
        return credit_row_json(domain_id, rank, shard);
    }
    return nullptr;
}

RemoteReadExecutor::~RemoteReadExecutor() {
    // The online main retains executor/transport until physical quiescence.
    // A remaining callback at teardown would otherwise point into a dead
    // descriptor registry; fail closed instead of silently dropping a leg.
    if (transport_ != nullptr && transport_->active_count() != 0) {
        std::fprintf(stderr,
                     "[Error] (execution_driven/remote_read) executor "
                     "destroyed with active physical transfers=%llu\n",
                     static_cast<unsigned long long>(transport_->active_count()));
        std::abort();
    }
}

[[noreturn]] void RemoteReadExecutor::fail_closed(
    const std::string& message) {
    std::fprintf(stderr,
                 "[Error] (execution_driven/remote_read) %s\n",
                 message.c_str());
    std::abort();
}

uint64_t RemoteReadExecutor::count_ordinary_terminal_candidate_matches(
    const std::unordered_map<uint64_t, const ParsedNode*>& nodes_by_id,
    const uint64_t first_json_id, const uint64_t count,
    const uint64_t stride, uint64_t* const membership_probes) {
    if (count == 0) return 0;
    // A zero tile stride names one unique parent regardless of the declared
    // operation count. prepare_batch currently normalizes this before calling
    // us; keep the helper's point semantics exact as well.
    const uint64_t unique_count = stride == 0 ? uint64_t{1} : count;
    const uint64_t effective_stride = unique_count == 1 ? uint64_t{1} : stride;
    // A progression cannot match more distinct physical IDs than the batch
    // index contains. This also prevents a malformed huge compact run from
    // being expanded during a failed preflight.
    if (unique_count > static_cast<uint64_t>(nodes_by_id.size())) return 0;

    uint64_t matches = 0;
    for (uint64_t offset = 0; offset < unique_count; ++offset) {
        const uint64_t json_id = add_checked(
            first_json_id,
            mul_checked(offset, effective_stride,
                        "ordinary terminal candidate offset"),
            "ordinary terminal candidate ID");
        if (membership_probes != nullptr) ++*membership_probes;
        if (nodes_by_id.find(json_id) != nodes_by_id.end()) ++matches;
    }
    return matches;
}

std::shared_ptr<void> RemoteReadExecutor::prepare_batch(
    const GraphBatch& batch,
    const GraphBatchCommitter& committer) const {
    const auto candidate_id_preview = committer.candidate_id_preview(batch);
    auto prepared = std::make_shared<PreparedPlan>();
    prepared->descriptors.reserve(batch.pipeline_descriptors.size());
    prepared->anchors_by_rank.resize(graph_sources_.size());
    prepared->terminal_watchers_by_rank.resize(graph_sources_.size());
    for (auto& anchors : prepared->anchors_by_rank) {
        anchors.reserve(batch.pipeline_descriptors.size() * 2);
    }
    std::vector<std::unordered_map<uint64_t, const ParsedNode*>>
        nodes_by_rank(graph_sources_.size());
    for (const ParsedNode& parsed : batch.nodes) {
        if (parsed.node.rank < 0 ||
            static_cast<size_t>(parsed.node.rank) >= graph_sources_.size()) {
            throw std::invalid_argument("pipeline batch node rank is out of range");
        }
        nodes_by_rank[static_cast<size_t>(parsed.node.rank)].emplace(
            parsed.json_id, &parsed);
    }
    std::unordered_map<std::string, bool> batch_identities;
    batch_identities.reserve(batch.pipeline_descriptors.size());

    const auto unique_tile_count = [](const uint64_t count,
                                      const uint64_t stride) {
        return stride == 0 ? uint64_t{1} : count;
    };
    const auto completed_old_terminal_count = [&](
        const int rank, const uint64_t first_store,
        const uint64_t unique_count, const uint64_t tile_stride,
        const uint64_t obligations_per_id,
        const PipelineEndpointDependencyKind kind) -> uint64_t {
        if (unique_count == 0) return 0;
        const uint64_t normalized_stride =
            unique_count == 1 ? uint64_t{1} : tile_stride;
        if (kind == PipelineEndpointDependencyKind::OrdinaryNodeTerminal) {
            const uint64_t finished = graph_sources_[rank]->store()
                .count_finished_or_erased_nonvirtual_progression(
                    first_store, unique_count, normalized_stride);
            return mul_checked(finished, obligations_per_id,
                               "already-completed consumer obligations");
        }
        // Virtual consume terminals are required by the descriptor parser to
        // resolve to this descriptor's body_marker run, so none can belong to
        // a previously committed span. Virtual terminals from an earlier
        // descriptor are legal endpoint *parents*, but are not body-consume
        // authorities for this descriptor.
        return 0;
    };

    const auto historical_virtual_endpoint_count = [&](
        const int rank, const uint64_t first_store,
        const uint64_t count, const uint64_t stride) -> uint64_t {
        const IdProgression referenced{first_store, count,
                                       count == 1 ? uint64_t{1} : stride};
        uint64_t mapped_count = 0;
        const uint64_t last_store = add_checked(
            first_store,
            mul_checked(count - 1, referenced.stride,
                        "historical virtual endpoint query span"),
            "historical virtual endpoint query span");
        for (const EndpointProgressionIndexEntry* entry :
             overlapping_endpoint_progressions(rank, first_store,
                                                last_store)) {
            mapped_count = add_checked(
                mapped_count,
                progression_intersection_count(
                    referenced,
                    IdProgression{entry->first_store_id, entry->tile_count,
                                  entry->tile_stride}),
                "historical virtual endpoint mapping count");
        }
        return mapped_count;
    };

    const auto validate_terminal_mapping = [&](
        const int rank, const uint64_t first_json,
        const uint64_t tile_count, const uint64_t tile_stride,
        const std::vector<uint64_t>& shard_offsets,
        const PipelineEndpointDependencyKind kind,
        const std::string& context) -> uint64_t {
        if (rank < 0 || static_cast<size_t>(rank) >= graph_sources_.size() ||
            tile_count == 0 || shard_offsets.empty()) {
            throw std::invalid_argument(context + ": invalid terminal span");
        }
        const uint64_t unique_count = unique_tile_count(tile_count, tile_stride);
        const uint64_t effective_stride = unique_count == 1 ? 1 : tile_stride;
        uint64_t precompleted_obligations = 0;
        for (const uint64_t shard_offset : shard_offsets) {
            const uint64_t json_first = add_checked(
                first_json, shard_offset, "terminal shard JSON offset");

            // Committed IDs form a prefix in the affine stream. Binary search
            // that prefix so a huge compact run is split without expanding
            // per-tile IDs.
            uint64_t old_count = 0;
            if (committer.resolve_store_id(rank, json_first).has_value()) {
                uint64_t low = 1;
                uint64_t high = unique_count;
                old_count = 1;
                while (low <= high) {
                    const uint64_t mid = low + (high - low) / 2;
                    const uint64_t probe_json = add_checked(
                        json_first,
                        mul_checked(mid - 1, effective_stride,
                                    "terminal committed-prefix probe"),
                        "terminal committed-prefix probe");
                    if (committer.resolve_store_id(rank, probe_json)) {
                        old_count = mid;
                        low = mid + 1;
                    } else {
                        if (mid == 0) break;
                        high = mid - 1;
                    }
                }
            }
            if (old_count != 0) {
                const auto old_first = committer.resolve_store_id(rank, json_first);
                if (!old_first.has_value()) {
                    throw std::logic_error("resolved terminal prefix lost its first ID");
                }
                const uint64_t old_stride = old_count == 1 ? 1 : effective_stride;
                const NodeStore& store = graph_sources_[rank]->store();
                const bool old_kind_valid =
                    kind == PipelineEndpointDependencyKind::OrdinaryNodeTerminal
                    ? store.is_allocated_nonvirtual_progression(
                          *old_first, old_count, old_stride)
                    : store.is_virtual_id_progression(
                          *old_first, old_count, old_stride);
                if (!old_kind_valid) {
                    throw std::invalid_argument(
                        context + ": committed terminal IDs have the wrong kind "
                        "or include an unknown/virtual hole");
                }
                if (kind == PipelineEndpointDependencyKind::VirtualEndpointTerminal &&
                    historical_virtual_endpoint_count(
                        rank, *old_first, old_count, old_stride) != old_count) {
                    throw std::invalid_argument(
                        context + ": historical virtual terminal IDs are holes "
                        "without a retained descriptor endpoint mapping");
                }
                if (kind == PipelineEndpointDependencyKind::OrdinaryNodeTerminal) {
                    const uint64_t obligations_per_id =
                        tile_stride == 0 ? tile_count : uint64_t{1};
                    precompleted_obligations = add_checked(
                        precompleted_obligations,
                        completed_old_terminal_count(
                            rank, *old_first, old_count, old_stride,
                            obligations_per_id, kind),
                        "precompleted terminal obligations");
                }
            }

            const uint64_t new_count = unique_count - old_count;
            if (new_count == 0) continue;
            const uint64_t new_json_first = add_checked(
                json_first,
                mul_checked(old_count, effective_stride,
                            "terminal new-span offset"),
                "terminal new-span JSON ID");
            if (!candidate_id_preview.resolve(rank, new_json_first)) {
                throw std::invalid_argument(
                    context + ": terminal ID is outside committed and candidate spans");
            }
            const IdProgression candidate{new_json_first, new_count,
                                          effective_stride};
            uint64_t candidate_matches = 0;
            if (kind == PipelineEndpointDependencyKind::OrdinaryNodeTerminal) {
                candidate_matches = count_ordinary_terminal_candidate_matches(
                    nodes_by_rank[rank], candidate.first, candidate.count,
                    candidate.stride, nullptr);
            } else {
                for (const auto& descriptor : batch.pipeline_descriptors) {
                    if (descriptor == nullptr) continue;
                    for (const auto& hole : descriptor->logical_id_ranges()) {
                        if (hole.rank != rank) continue;
                        candidate_matches = add_checked(
                            candidate_matches,
                            progression_intersection_count(
                                candidate,
                                IdProgression{hole.first_json_id,
                                              hole.count, hole.stride}),
                            "candidate virtual terminal membership");
                    }
                }
            }
            if (candidate_matches != new_count) {
                throw std::invalid_argument(
                    context + (kind ==
                                       PipelineEndpointDependencyKind::OrdinaryNodeTerminal
                                   ? ": ordinary terminal does not resolve to every "
                                     "candidate-batch node"
                                   : ": virtual terminal does not resolve to every "
                                     "candidate descriptor hole"));
            }
            if (kind == PipelineEndpointDependencyKind::VirtualEndpointTerminal) {
                // A typed hole is not itself evidence that an endpoint will
                // ever complete. Require every candidate virtual parent to
                // intersect an actual endpoint-run progression in this
                // batch. This closes the gap where an arbitrary reserved
                // hole passed Phase A and left a permanently blocked cursor.
                uint64_t mapped_endpoint_count = 0;
                for (const auto& descriptor : batch.pipeline_descriptors) {
                    if (descriptor == nullptr) continue;
                    for (const PipelineEndpointRun& endpoint_run :
                         descriptor->endpoint_runs()) {
                        if (endpoint_run.rank != rank) continue;
                        for (const PipelineEndpointLegRun& endpoint_leg :
                             endpoint_run.legs) {
                            for (const uint64_t endpoint_shard :
                                 endpoint_run.shard_ordinals) {
                                const uint64_t endpoint_first_json =
                                    descriptor->endpoint_json_id(
                                        endpoint_run, endpoint_leg,
                                        endpoint_run.first_tile,
                                        endpoint_shard);
                                if (!candidate_id_preview.resolve(
                                         rank, endpoint_first_json)
                                         .has_value()) {
                                    throw std::invalid_argument(
                                        context + ": candidate endpoint has no "
                                        "affine store mapping");
                                }
                                mapped_endpoint_count = add_checked(
                                    mapped_endpoint_count,
                                    progression_intersection_count(
                                        candidate,
                                        IdProgression{
                                            endpoint_first_json,
                                            endpoint_run.tile_count,
                                            endpoint_run.tile_stride}),
                                    "candidate virtual endpoint mapping count");
                            }
                        }
                    }
                }
                if (mapped_endpoint_count != new_count) {
                    throw std::invalid_argument(
                        context + ": candidate virtual terminal does not "
                        "resolve to every declared endpoint run mapping");
                }
            }
        }
        return precompleted_obligations;
    };

    for (const auto& descriptor : batch.pipeline_descriptors) {
        if (descriptor == nullptr) {
            throw std::invalid_argument("null pipeline descriptor");
        }
        if (descriptor->endpoint_runs().empty() ||
            descriptor->anchors().empty()) {
            throw std::invalid_argument(
                "prediction-only pipeline descriptor cannot be installed for "
                "execution");
        }
        const std::string& segment_id = descriptor->identity().segment_id;
        const bool inserted_in_batch =
            batch_identities.emplace(segment_id, true).second;
        if (segment_id.empty() || !inserted_in_batch ||
            descriptor_by_identity_.find(segment_id) !=
                descriptor_by_identity_.end()) {
            throw std::invalid_argument(
                "duplicate pipeline descriptor identity " + segment_id);
        }

        auto runtime = std::make_unique<RuntimeDescriptor>();
        runtime->descriptor = descriptor;
        const auto add_watch_progression = [&runtime](
            const int rank, const uint64_t first_store_id,
            const uint64_t count, const uint64_t stride) {
            if (rank < 0 || first_store_id == 0 || count == 0 || stride == 0) {
                throw std::invalid_argument(
                    "invalid compact WatchRegistry pin progression");
            }
            if (count > 1 &&
                static_cast<UnsignedWide>(first_store_id) +
                        static_cast<UnsignedWide>(count - 1) * stride >
                    std::numeric_limits<uint64_t>::max()) {
                throw std::overflow_error(
                    "WatchRegistry pin progression overflows uint64");
            }
            runtime->watch_member_ranges.push_back(
                WatchMemberProgression{rank, first_store_id, count, stride});
        };
        runtime->runs.reserve(descriptor->endpoint_runs().size());
        for (const PipelineEndpointRun& run : descriptor->endpoint_runs()) {
            if (run.rank < 0 ||
                static_cast<size_t>(run.rank) >= graph_sources_.size()) {
                throw std::invalid_argument("endpoint run rank is out of range");
            }
            const auto first_store_id = candidate_id_preview.resolve(
                run.rank, run.first_json_id);
            if (!first_store_id.has_value()) {
                throw std::invalid_argument(
                    "endpoint run's first JSON ID has no affine store mapping");
            }
            RunCursor cursor;
            cursor.descriptor = descriptor.get();
            cursor.run = &run;
            cursor.first_store_id = *first_store_id;
            if (run.has_initial_parent_json_id) {
                const auto parent_store_id = candidate_id_preview.resolve(
                    run.rank, run.initial_parent_json_id);
                if (!parent_store_id.has_value()) {
                    throw std::invalid_argument(
                        "endpoint run initial parent is unresolved");
                }
                cursor.has_initial_parent = true;
                cursor.initial_parent_store_id = *parent_store_id;
            }
            // A shared stream-bytes axis may outlive an individual shard's
            // byte range. Advance through those absent operations here, so
            // neither readiness nor endpoint construction can synthesize
            // post-cap COMM/ACK work. The cursor still uses the group's
            // common global tile axis for slots and parity.
            while (!cursor.exhausted && !current_position_active(cursor)) {
                mark_cursor_position_completed(cursor);
                advance(cursor);
            }
            runtime->runs.push_back(cursor);
            const uint64_t events_per_tile = mul_checked(
                static_cast<uint64_t>(run.shard_ordinals.size()),
                static_cast<uint64_t>(run.legs.size()), "run endpoint count");
            runtime->total_endpoint_count = add_checked(
                runtime->total_endpoint_count,
                mul_checked(run.tile_count, events_per_tile,
                            "run endpoint count"),
                "descriptor endpoint count");
        }

        runtime->dependency_runs.reserve(
            descriptor->endpoint_dependency_runs().size());
        for (const auto& dependency : descriptor->endpoint_dependency_runs()) {
            std::vector<uint64_t> shard_offsets;
            shard_offsets.reserve(dependency.shard_ordinals.size());
            for (size_t shard_index = 0;
                 shard_index < dependency.shard_ordinals.size(); ++shard_index) {
                shard_offsets.push_back(mul_checked(
                    static_cast<uint64_t>(shard_index),
                    dependency.dependency_shard_stride,
                    "dependency shard offset"));
            }
            const auto first_store = candidate_id_preview.resolve(
                dependency.rank,
                dependency.first_dependency_json_id);
            if (!first_store.has_value()) {
                // Store IDs start at 1; JSON ID 0 remains legal and maps here
                // to a real nonzero NodeStore ID.
                throw std::invalid_argument(
                    "dependency first ID has no affine store mapping");
            }
            static_cast<void>(validate_terminal_mapping(
                dependency.rank, dependency.first_dependency_json_id,
                dependency.tile_count, dependency.dependency_tile_stride,
                shard_offsets, dependency.dependency_kind,
                "endpoint dependency"));
            runtime->dependency_runs.push_back(
                ResolvedDependencyRun{&dependency, *first_store});
            if (dependency.dependency_kind ==
                PipelineEndpointDependencyKind::OrdinaryNodeTerminal) {
                const uint64_t unique_count = unique_tile_count(
                    dependency.tile_count, dependency.dependency_tile_stride);
                const uint64_t normalized_stride = unique_count == 1
                    ? uint64_t{1} : dependency.dependency_tile_stride;
                for (const uint64_t shard_offset : shard_offsets) {
                    add_watch_progression(
                        dependency.rank,
                        add_checked(*first_store, shard_offset,
                                    "watch dependency shard first"),
                        unique_count, normalized_stride);
                }
            }
            TerminalWatcher watcher;
            watcher.owner = runtime.get();
            watcher.wake_rank = dependency.rank;
            watcher.group_ordinal = dependency.group_ordinal;
            watcher.kind = dependency.dependency_kind;
            watcher.first_store_id = *first_store;
            watcher.tile_count = dependency.tile_count;
            watcher.tile_stride = dependency.dependency_tile_stride;
            watcher.shard_offsets = std::move(shard_offsets);
            prepared->terminal_watchers_by_rank[dependency.rank]
                .push_back(std::move(watcher));
        }
        build_dependency_interval_indexes(
            runtime->dependency_runs,
            &runtime->dependency_interval_indexes);

        runtime->consume_runs.reserve(
            descriptor->consume_completion_runs().size());
        const auto& consume_completions =
            descriptor->consume_completion_runs();
        for (size_t completion_index = 0;
             completion_index < consume_completions.size();
             ++completion_index) {
            const PipelineConsumeCompletionRun& completion =
                consume_completions[completion_index];
            std::vector<uint64_t> shard_offsets;
            shard_offsets.reserve(completion.shard_ordinals.size());
            for (size_t shard_index = 0;
                 shard_index < completion.shard_ordinals.size(); ++shard_index) {
                shard_offsets.push_back(mul_checked(
                    static_cast<uint64_t>(shard_index),
                    completion.dependency_shard_stride,
                    "consume completion shard offset"));
            }
            const auto first_store = candidate_id_preview.resolve(
                completion.rank,
                completion.first_dependency_json_id);
            if (!first_store.has_value()) {
                throw std::invalid_argument(
                    "consume completion first ID has no affine store mapping");
            }
            const uint64_t completed = validate_terminal_mapping(
                completion.rank, completion.first_dependency_json_id,
                completion.tile_count, completion.dependency_tile_stride,
                shard_offsets, completion.dependency_kind,
                "consume completion");
            const uint64_t obligations = mul_checked(
                completion.tile_count,
                static_cast<uint64_t>(completion.shard_ordinals.size()),
                "consume completion obligation count");
            if (completed > obligations) {
                throw std::invalid_argument(
                    "precompleted consume terminals exceed mapped obligations");
            }
            runtime->consume_pending_count = add_checked(
                runtime->consume_pending_count, obligations - completed,
                "pending consume completion count");
            runtime->consume_runs.push_back(
                ResolvedConsumeRun{&completion, *first_store});
            if (completion.dependency_kind ==
                PipelineEndpointDependencyKind::OrdinaryNodeTerminal) {
                const uint64_t unique_count = unique_tile_count(
                    completion.tile_count,
                    completion.dependency_tile_stride);
                const uint64_t normalized_stride = unique_count == 1
                    ? uint64_t{1} : completion.dependency_tile_stride;
                for (const uint64_t shard_offset : shard_offsets) {
                    add_watch_progression(
                        completion.rank,
                        add_checked(*first_store, shard_offset,
                                    "watch consume shard first"),
                        unique_count, normalized_stride);
                }
            }
            TerminalWatcher watcher;
            watcher.owner = runtime.get();
            watcher.consume_run = &completion;
            watcher.consume_run_index = completion_index;
            watcher.wake_rank = completion.rank;
            watcher.group_ordinal = completion.group_ordinal;
            watcher.kind = completion.dependency_kind;
            watcher.is_consume_completion = true;
            watcher.first_store_id = *first_store;
            watcher.tile_count = completion.tile_count;
            watcher.tile_stride = completion.dependency_tile_stride;
            watcher.shard_offsets = std::move(shard_offsets);
            watcher.total_obligations = obligations;
            watcher.preaccounted = completed;
            watcher.delivered = 0;
            prepared->terminal_watchers_by_rank[completion.rank]
                .push_back(std::move(watcher));
        }

        RuntimeDescriptor* const runtime_ptr = runtime.get();
        const auto anchor_position = [&](const PipelineAnchor& anchor)
            -> std::optional<EndpointPosition> {
            if (anchor.binding != PipelineAnchorBinding::PhysicalEndpointSlot ||
                !anchor.has_node_id) {
                return std::nullopt;
            }
            std::optional<EndpointPosition> found;
            for (size_t run_index = 0; run_index < descriptor->endpoint_runs().size();
                 ++run_index) {
                const auto& run = descriptor->endpoint_runs()[run_index];
                if (run.rank != anchor.rank ||
                    run.group_ordinal != anchor.selector.group_ordinal ||
                    anchor.selector.tile_ordinal < run.first_tile ||
                    anchor.selector.tile_ordinal - run.first_tile >= run.tile_count) {
                    continue;
                }
                for (size_t shard_index = 0;
                     shard_index < run.shard_ordinals.size(); ++shard_index) {
                    if (run.shard_ordinals[shard_index] !=
                        anchor.selector.shard_ordinal) {
                        continue;
                    }
                for (size_t leg_index = 0; leg_index < run.legs.size();
                     ++leg_index) {
                    const auto& leg = run.legs[leg_index];
                    const bool internal_marker = anchor.selector.kind ==
                        PipelineSelectorKind::InternalMarkerDone;
                    if (leg.endpoint != anchor.selector.endpoint ||
                        internal_marker !=
                            (leg.endpoint == PipelineEndpointLeg::BodyMarker)) {
                        continue;
                    }
                        const uint64_t json_id = descriptor->endpoint_json_id(
                            run, leg, anchor.selector.tile_ordinal,
                            anchor.selector.shard_ordinal);
                        if (json_id != anchor.node_id) {
                            continue;
                        }
                        if (found.has_value()) {
                            throw std::invalid_argument(
                                "physical anchor resolves ambiguously to endpoint runs");
                        }
                        EndpointPosition position;
                        position.owner = runtime_ptr;
                        position.run_index = run_index;
                        position.tile_offset =
                            anchor.selector.tile_ordinal - run.first_tile;
                        position.shard_index = shard_index;
                        position.leg_index = leg_index;
                        found = position;
                    }
                }
            }
            if (!found.has_value()) {
                throw std::invalid_argument(
                    "physical endpoint anchor has no endpoint-run mapping");
            }
            return found;
        };

        for (const PipelineAnchor& anchor : descriptor->anchors()) {
            if (anchor.binding != PipelineAnchorBinding::PhysicalEndpointSlot) {
                if (anchor.role == PipelineAnchorRole::Start ||
                    anchor.role == PipelineAnchorRole::Frontier) {
                    throw std::invalid_argument(
                        "start/frontier anchors must bind physical endpoint slots");
                }
                continue;
            }
            if (anchor.role != PipelineAnchorRole::Start &&
                anchor.role != PipelineAnchorRole::Frontier) {
                throw std::invalid_argument(
                    "service/settle anchors cannot bind physical node slots");
            }
            const auto position = anchor_position(anchor);
            if (!position.has_value()) {
                throw std::invalid_argument("invalid physical anchor selector");
            }
            const auto store_id = candidate_id_preview.resolve(
                anchor.rank, anchor.node_id);
            if (!store_id.has_value()) {
                throw std::invalid_argument("physical anchor has no store ID mapping");
            }
            AnchorBinding binding;
            binding.owner = runtime_ptr;
            binding.run_index = position->run_index;
            binding.tile_ordinal =
                descriptor->endpoint_runs()[position->run_index].first_tile +
                position->tile_offset;
            binding.shard_index = position->shard_index;
            binding.leg_index = position->leg_index;
            binding.role = anchor.role;
            if (!prepared->anchors_by_rank[anchor.rank]
                     .emplace(*store_id, binding)
                     .second) {
                throw std::invalid_argument(
                    "multiple physical anchors bind one store ID");
            }
            add_watch_progression(anchor.rank, *store_id, 1, 1);
        }

        prepared->descriptors.push_back(std::move(runtime));
    }
    return prepared;
}

void RemoteReadExecutor::commit_batch(std::shared_ptr<void> erased_prepared,
                                      const GraphBatch& batch,
                                      const GraphBatchCommitter& /*committer*/) {
    auto prepared = std::static_pointer_cast<PreparedPlan>(erased_prepared);
    if (prepared == nullptr ||
        prepared->descriptors.size() != batch.pipeline_descriptors.size()) {
        fail_closed("pipeline prepared plan does not match committed batch");
    }
    std::vector<RuntimeDescriptor*> newly_committed;
    newly_committed.reserve(prepared->descriptors.size());
    for (size_t index = 0; index < prepared->descriptors.size(); ++index) {
        auto& runtime = prepared->descriptors[index];
        if (runtime == nullptr || runtime->descriptor == nullptr) {
            fail_closed("prepared descriptor is incomplete");
        }
        if (runtime->descriptor.get() != batch.pipeline_descriptors[index].get()) {
            fail_closed("pipeline prepared plan no longer matches batch wire");
        }
        const std::string& segment_id = runtime->descriptor->identity().segment_id;
        if (!descriptor_by_identity_.emplace(segment_id, runtime.get()).second) {
            fail_closed("descriptor identity became duplicate before commit");
        }
        const std::string& buffer_domain_id =
            runtime->descriptor->identity().buffer_domain_id;
        RuntimeDescriptor* const runtime_raw = runtime.get();
        newly_committed.push_back(runtime_raw);
        const bool new_domain =
            buffer_domains_.find(buffer_domain_id) == buffer_domains_.end();
        std::array<uint64_t, 2> target_slot_before_versions{};
        if (new_domain) {
            for (size_t slot_index = 0; slot_index < 2; ++slot_index) {
                target_slot_before_versions[slot_index] = resource_mutation_before(
                    "pipeline_slot",
                    slot_resource_key(buffer_domain_id, "target",
                                      runtime->descriptor->identity().rank, 0,
                                      slot_index),
                    runtime->descriptor->identity().rank);
            }
        }
        auto domain_insert = buffer_domains_.try_emplace(buffer_domain_id);
        BufferDomainState& domain = domain_insert.first->second;
        if (new_domain) {
            restore_buffer_domain_provenance(
                buffer_domain_id, domain,
                runtime->descriptor->identity().rank);
        }
        const auto target_rank_insert = buffer_domain_target_ranks_.emplace(
            buffer_domain_id, runtime->descriptor->identity().rank);
        if (!target_rank_insert.second &&
            target_rank_insert.first->second !=
                runtime->descriptor->identity().rank) {
            fail_closed("buffer domain has inconsistent target ranks");
        }
        if (new_domain) {
            for (size_t slot_index = 0; slot_index < 2; ++slot_index) {
                resource_mutation_after(
                    "pipeline_slot",
                    slot_resource_key(buffer_domain_id, "target",
                                      runtime->descriptor->identity().rank, 0,
                                      slot_index),
                    target_slot_before_versions[slot_index]);
            }
        }
        for (const auto& group : runtime->descriptor->groups()) {
            for (const auto& shard : group.shards) {
                const auto source_key =
                    std::make_pair(shard.source_rank, shard.shard_ordinal);
                if (domain.source_slots.find(source_key) ==
                    domain.source_slots.end()) {
                    std::array<uint64_t, 2> source_slot_before_versions{};
                    for (size_t slot_index = 0; slot_index < 2; ++slot_index) {
                        source_slot_before_versions[slot_index] =
                            resource_mutation_before(
                                "pipeline_slot",
                                slot_resource_key(buffer_domain_id, "source",
                                                  shard.source_rank,
                                                  shard.shard_ordinal,
                                                  slot_index));
                    }
                    const uint64_t credit_before_version =
                        resource_mutation_before(
                            "pipeline_credit",
                            credit_resource_key(buffer_domain_id,
                                                shard.source_rank,
                                                shard.shard_ordinal));
                    auto source_insert =
                        domain.source_slots.try_emplace(source_key);
                    if (source_insert.second) {
                        restore_source_slot_provenance(
                            buffer_domain_id, source_key,
                            source_insert.first->second);
                    }
                    domain.outstanding_release_acks.try_emplace(source_key, 0);
                    for (size_t slot_index = 0; slot_index < 2; ++slot_index) {
                        resource_mutation_after(
                            "pipeline_slot",
                            slot_resource_key(buffer_domain_id, "source",
                                              shard.source_rank,
                                              shard.shard_ordinal, slot_index),
                            source_slot_before_versions[slot_index]);
                    }
                    resource_mutation_after(
                        "pipeline_credit",
                        credit_resource_key(buffer_domain_id,
                                            shard.source_rank,
                                            shard.shard_ordinal),
                        credit_before_version);
                } else {
                    domain.outstanding_release_acks.try_emplace(source_key, 0);
                }
            }
        }
        runtime->commit_tick = Sys::boostedTick();
        if (descriptor_commit_publisher_) {
            descriptor_commit_publisher_(nlohmann::json{
                {"identity", identity_json(runtime->descriptor->identity())},
                {"descriptor", runtime->descriptor->wire()},
                {"commit_tick", runtime->commit_tick},
                {"cursor_version", runtime->cursor_version},
                {"service_done", runtime->service_done},
                {"settle_done", runtime->settle_done},
                {"retired", runtime->retired},
            });
        }
        publish_pipeline_transition(
            *runtime, "descriptor_commit", nullptr, 0, "",
            nlohmann::json{{"ops", nullptr}, {"bytes", nullptr},
                           {"latency_ns", nullptr}},
            runtime->cursor_version);
        descriptors_.push_back(std::move(runtime));
        if (std::find(active_descriptors_.begin(), active_descriptors_.end(),
                      runtime_raw) != active_descriptors_.end()) {
            fail_closed("committed descriptor already exists in active index");
        }
        active_descriptors_.push_back(runtime_raw);
        const std::string& domain_id =
            runtime_raw->descriptor->identity().buffer_domain_id;
        ++active_owner_count_by_domain_[domain_id];
        for (size_t run_index = 0;
             run_index < runtime_raw->runs.size(); ++run_index) {
            set_current_cursor(*runtime_raw, run_index);
        }
    }
    index_batch_descriptors(newly_committed);
    for (size_t rank = 0; rank < prepared->anchors_by_rank.size(); ++rank) {
        for (const auto& anchor : prepared->anchors_by_rank[rank]) {
            if (!physical_anchors_by_rank_[rank]
                     .emplace(anchor.first, anchor.second)
                     .second) {
                fail_closed(
                    "physical anchor binding became duplicate before commit");
            }
            if (anchor.second.owner == nullptr) {
                fail_closed("physical anchor lacks its descriptor owner");
            }
            anchor.second.owner->physical_anchor_store_ids.emplace_back(
                static_cast<int>(rank), anchor.first);
            // F (2026-10-11): run the per-query fail-closed binding chain
            // once at registration so an illegally described FUTURE anchor
            // (which the issue pass no longer enumerates) is still rejected
            // here, and admit the id into the current-anchor set when its
            // run cursor already points at it (the set_current_cursor calls
            // above ran before this loop, so a start anchor whose commit
            // cursor lands on it is admitted here, not there).
            validate_anchor_binding_static(static_cast<int>(rank),
                                           anchor.first, anchor.second);
            uint64_t cursor_current = 0;
            if (current_store_id(
                    anchor.second.owner->runs[anchor.second.run_index],
                    &cursor_current) &&
                cursor_current == anchor.first) {
                note_current_cursor_anchor(static_cast<int>(rank),
                                           anchor.first);
            }
        }
        auto& incoming = prepared->terminal_watchers_by_rank[rank];
        for (TerminalWatcher& watcher : incoming) {
            install_terminal_watcher(static_cast<int>(rank),
                                    std::move(watcher));
        }
    }
}

bool RemoteReadExecutor::current_store_id(const RunCursor& cursor,
                                          uint64_t* store_id) const {
    if (cursor.run == nullptr || cursor.exhausted || store_id == nullptr ||
        cursor.tile_offset >= cursor.run->tile_count ||
        cursor.shard_index >= cursor.run->shard_ordinals.size() ||
        cursor.leg_index >= cursor.run->legs.size()) {
        return false;
    }
    if (!current_position_active(cursor)) return false;
    const uint64_t tile_delta =
        mul_checked(cursor.tile_offset, cursor.run->tile_stride,
                    "current tile ID offset");
    const uint64_t shard_delta =
        mul_checked(static_cast<uint64_t>(cursor.shard_index),
                    cursor.run->shard_stride, "current shard ID offset");
    const uint64_t leg_offset =
        cursor.run->legs[cursor.leg_index].json_offset;
    *store_id = add_checked(
        cursor.first_store_id,
        add_checked(add_checked(tile_delta, shard_delta, "current endpoint ID"),
                    leg_offset, "current endpoint ID"),
        "current endpoint store ID");
    return true;
}

bool RemoteReadExecutor::current_position_active(const RunCursor& cursor) {
    if (cursor.descriptor == nullptr || cursor.run == nullptr ||
        cursor.exhausted || cursor.tile_offset >= cursor.run->tile_count ||
        cursor.shard_index >= cursor.run->shard_ordinals.size() ||
        cursor.leg_index >= cursor.run->legs.size()) {
        return false;
    }
    const PipelineTile tile = PipelineKernel::tile(
        *cursor.descriptor, cursor.run->group_ordinal,
        cursor.run->first_tile + cursor.tile_offset - 1);
    const PipelineShardTile& shard = shard_for(
        tile, cursor.run->shard_ordinals[cursor.shard_index]);
    // The original control marker has its own one-op/one-byte COMP duty and
    // remains legal even when a shorter shard has no data tile at this axis
    // position. Every network/ACK leg requires a real active shard tile.
    return cursor.run->legs[cursor.leg_index].endpoint ==
               PipelineEndpointLeg::BodyMarker ||
           (shard.emit_endpoint && shard.data_bytes != 0);
}

void RemoteReadExecutor::mark_cursor_position_completed(RunCursor& cursor) {
    const uint64_t ordinal = position_ordinal(
        cursor, cursor.tile_offset, cursor.shard_index, cursor.leg_index);
    mark_cursor_range_completed(cursor, ordinal, 1);
}

void RemoteReadExecutor::mark_cursor_range_completed(
    RunCursor& cursor, const uint64_t first_ordinal, const uint64_t count) {
    if (count == 0 || first_ordinal < cursor.completed_prefix ||
        count > std::numeric_limits<uint64_t>::max() - first_ordinal) {
        fail_closed("invalid or duplicate completed endpoint cursor range");
    }
    // Validate the complete insertion/merge window before mutating the map.
    // Reject overlap and count overflow before publishing the union node;
    // both the cached count and intervals remain unchanged on rejection.
    uint64_t validated_first = first_ordinal;
    uint64_t validated_end = first_ordinal + count;
    bool previous_is_adjacent = false;
    auto validation = cursor.completed_out_of_order.lower_bound(
        validated_first);
    if (validation != cursor.completed_out_of_order.begin()) {
        const auto previous = std::prev(validation);
        if (previous->second > validated_first) {
            fail_closed("completed endpoint cursor range overlaps prior work");
        }
        if (previous->second == validated_first) {
            validated_first = previous->first;
            previous_is_adjacent = true;
        }
    }
    validation = cursor.completed_out_of_order.lower_bound(validated_first);
    if (previous_is_adjacent) ++validation;
    while (validation != cursor.completed_out_of_order.end() &&
           validation->first <= validated_end) {
        if (validation->first < validated_end) {
            fail_closed("completed endpoint cursor range overlaps prior work");
        }
        validated_end = std::max(validated_end, validation->second);
        ++validation;
    }
    if (cursor.completed_count < cursor.completed_prefix) {
        fail_closed("completed endpoint count is below its contiguous prefix");
    }
    const uint64_t next_completed_count = add_checked(
        cursor.completed_count, count, "completed endpoint count");

    // Publish the final union node before erasing any adjacent old intervals.
    // If allocation fails, the original map and cached count are still intact.
    // A predecessor-adjacent union already has the required start key, so
    // extending its scalar end requires no allocation.
    auto retained = cursor.completed_out_of_order.end();
    if (validated_first != cursor.completed_prefix) {
        retained = cursor.completed_out_of_order.find(validated_first);
        if (retained != cursor.completed_out_of_order.end()) {
            retained->second = validated_end;
        } else {
            const auto insertion = cursor.completed_out_of_order.emplace(
                validated_first, validated_end);
            if (!insertion.second) {
                fail_closed(
                    "completed endpoint cursor union start already exists");
            }
            retained = insertion.first;
        }
    }

    auto it = cursor.completed_out_of_order.lower_bound(validated_first);
    if (retained != cursor.completed_out_of_order.end() && it == retained) {
        ++it;
    }
    while (it != cursor.completed_out_of_order.end() &&
           it->first <= validated_end) {
        it = cursor.completed_out_of_order.erase(it);
    }
    if (validated_first == cursor.completed_prefix) {
        cursor.completed_prefix = validated_end;
    }
    while (!cursor.completed_out_of_order.empty() &&
           cursor.completed_out_of_order.begin()->first <=
               cursor.completed_prefix) {
        auto front = cursor.completed_out_of_order.begin();
        if (front->first < cursor.completed_prefix) {
            if (front->second <= cursor.completed_prefix) {
                cursor.completed_out_of_order.erase(front);
                continue;
            }
            cursor.completed_prefix = front->second;
            cursor.completed_out_of_order.erase(front);
        } else {
            cursor.completed_prefix = front->second;
            cursor.completed_out_of_order.erase(front);
        }
    }
    cursor.completed_count = next_completed_count;
}

uint64_t RemoteReadExecutor::position_ordinal(
    const RunCursor& cursor, const uint64_t tile_offset,
    const size_t shard_index, const size_t leg_index) {
    if (cursor.run == nullptr || tile_offset >= cursor.run->tile_count ||
        shard_index >= cursor.run->shard_ordinals.size() ||
        leg_index >= cursor.run->legs.size()) {
        throw std::invalid_argument("endpoint position is outside its run");
    }
    const uint64_t per_tile = mul_checked(
        static_cast<uint64_t>(cursor.run->shard_ordinals.size()),
        static_cast<uint64_t>(cursor.run->legs.size()),
        "endpoint position ordinal");
    return add_checked(
        mul_checked(tile_offset, per_tile, "endpoint position ordinal"),
        add_checked(mul_checked(static_cast<uint64_t>(shard_index),
                                 static_cast<uint64_t>(cursor.run->legs.size()),
                                 "endpoint position ordinal"),
                    static_cast<uint64_t>(leg_index),
                    "endpoint position ordinal"),
        "endpoint position ordinal");
}

void RemoteReadExecutor::build_dependency_interval_indexes(
    const std::vector<ResolvedDependencyRun>& dependency_runs,
    DependencyIntervalIndexes* const indexes) {
    if (indexes == nullptr) {
        throw std::invalid_argument(
            "endpoint dependency interval index output is null");
    }

    // Build into a local map and publish only after every bucket and tree is
    // complete. The owning RuntimeDescriptor is still in prepare_batch's
    // unpublished plan at this point, so allocation/validation exceptions
    // cannot expose a partially built index.
    std::map<DependencyIntervalKey, std::vector<DependencyIntervalEntry>>
        grouped_entries;
    for (size_t dependency_index = 0;
         dependency_index < dependency_runs.size(); ++dependency_index) {
        const ResolvedDependencyRun& resolved =
            dependency_runs[dependency_index];
        if (resolved.run == nullptr) {
            throw std::invalid_argument(
                "endpoint dependency interval index has a null run");
        }
        const PipelineEndpointDependencyRun& dependency = *resolved.run;
        if (dependency.tile_count == 0) {
            // Invalid in a parsed descriptor, and it could never match the
            // old subtraction-based query if a test constructs one directly.
            continue;
        }
        const UnsignedWide last_tile_wide =
            static_cast<UnsignedWide>(dependency.first_tile) +
            static_cast<UnsignedWide>(dependency.tile_count) - 1;
        const uint64_t last_tile = last_tile_wide >
                std::numeric_limits<uint64_t>::max()
            ? std::numeric_limits<uint64_t>::max()
            : static_cast<uint64_t>(last_tile_wide);

        // The parser rejects duplicate shard ordinals. Retaining only the
        // first occurrence here also exactly matches the former std::find
        // behavior for a privately constructed malformed test row.
        std::set<uint64_t> indexed_shards;
        for (size_t local_shard = 0;
             local_shard < dependency.shard_ordinals.size(); ++local_shard) {
            const uint64_t shard_ordinal =
                dependency.shard_ordinals[local_shard];
            if (!indexed_shards.insert(shard_ordinal).second) continue;
            const DependencyIntervalKey key{
                dependency.rank, dependency.group_ordinal,
                dependency.endpoint, shard_ordinal};
            grouped_entries[key].push_back(DependencyIntervalEntry{
                dependency.first_tile, last_tile, dependency_index,
                local_shard});
        }
    }

    DependencyIntervalIndexes built;
    for (auto& grouped : grouped_entries) {
        std::vector<DependencyIntervalEntry>& entries = grouped.second;
        std::sort(entries.begin(), entries.end(),
            [](const DependencyIntervalEntry& left,
               const DependencyIntervalEntry& right) {
                return std::tie(left.first_tile,
                                left.dependency_run_index,
                                left.local_shard_index) <
                    std::tie(right.first_tile,
                             right.dependency_run_index,
                             right.local_shard_index);
            });

        DependencyIntervalIndex index;
        index.entries = std::move(entries);
        index.leaf_base = 1;
        while (index.leaf_base < index.entries.size()) {
            if (index.leaf_base >
                std::numeric_limits<size_t>::max() / 2) {
                throw std::length_error(
                    "endpoint dependency interval index is too large");
            }
            index.leaf_base *= 2;
        }
        if (index.leaf_base >
            std::numeric_limits<size_t>::max() / 2) {
            throw std::length_error(
                "endpoint dependency interval tree is too large");
        }
        index.subtree_max_last_tile.assign(index.leaf_base * 2, 0);
        for (size_t entry = 0; entry < index.entries.size(); ++entry) {
            index.subtree_max_last_tile[index.leaf_base + entry] =
                index.entries[entry].last_tile;
        }
        for (size_t node = index.leaf_base; node > 1; --node) {
            const size_t parent = node - 1;
            index.subtree_max_last_tile[parent] = std::max(
                index.subtree_max_last_tile[parent * 2],
                index.subtree_max_last_tile[parent * 2 + 1]);
        }
        built.emplace(grouped.first, std::move(index));
    }
    indexes->swap(built);
}

bool RemoteReadExecutor::query_dependency_interval_node(
    const DependencyIntervalIndex& index, const size_t tree_node,
    const size_t first_entry, const size_t end_entry,
    const uint64_t tile_ordinal,
    std::vector<DependencyIndexMatch>* const matches,
    uint64_t* const visited_nodes) {
    if (first_entry >= index.entries.size()) return false;
    if (visited_nodes != nullptr) ++*visited_nodes;
    // Entries in each tree range are sorted by first_tile, so the range's
    // first entry is its minimum start. Its stored maximum end gives the
    // other half of the point-in-interval prune.
    if (index.subtree_max_last_tile[tree_node] < tile_ordinal ||
        index.entries[first_entry].first_tile > tile_ordinal) {
        return false;
    }
    if (end_entry - first_entry == 1) {
        const DependencyIntervalEntry& entry = index.entries[first_entry];
        if (tile_ordinal > entry.last_tile) return false;
        if (matches == nullptr) return true;
        matches->push_back(DependencyIndexMatch{
            entry.dependency_run_index, entry.local_shard_index});
        return true;
    }

    const size_t middle = first_entry + (end_entry - first_entry) / 2;
    const bool found_left = query_dependency_interval_node(
        index, tree_node * 2, first_entry, middle, tile_ordinal,
        matches, visited_nodes);
    if (matches == nullptr && found_left) return true;
    const bool found_right = query_dependency_interval_node(
        index, tree_node * 2 + 1, middle, end_entry, tile_ordinal,
        matches, visited_nodes);
    return found_left || found_right;
}

std::vector<RemoteReadExecutor::DependencyIndexMatch>
RemoteReadExecutor::matching_dependency_runs(
    const RuntimeDescriptor& owner, const int rank,
    const uint64_t group_ordinal, const PipelineEndpointLeg endpoint,
    const uint64_t tile_ordinal, const uint64_t shard_ordinal,
    uint64_t* const visited_nodes) {
    const DependencyIntervalKey key{
        rank, group_ordinal, endpoint, shard_ordinal};
    const auto found = owner.dependency_interval_indexes.find(key);
    if (found == owner.dependency_interval_indexes.end() ||
        found->second.entries.empty()) {
        return {};
    }
    const DependencyIntervalIndex& index = found->second;
    std::vector<DependencyIndexMatch> matches;
    static_cast<void>(query_dependency_interval_node(
        index, 1, 0, index.leaf_base, tile_ordinal, &matches,
        visited_nodes));
    std::sort(matches.begin(), matches.end(),
        [](const DependencyIndexMatch& left,
           const DependencyIndexMatch& right) {
            return std::tie(left.dependency_run_index,
                            left.local_shard_index) <
                std::tie(right.dependency_run_index,
                         right.local_shard_index);
        });
    return matches;
}

bool RemoteReadExecutor::has_matching_dependency_run(
    const RuntimeDescriptor& owner, const int rank,
    const uint64_t group_ordinal, const PipelineEndpointLeg endpoint,
    const uint64_t tile_ordinal, const uint64_t shard_ordinal,
    uint64_t* const visited_nodes) {
    const DependencyIntervalKey key{
        rank, group_ordinal, endpoint, shard_ordinal};
    const auto found = owner.dependency_interval_indexes.find(key);
    if (found == owner.dependency_interval_indexes.end() ||
        found->second.entries.empty()) {
        return false;
    }
    const DependencyIntervalIndex& index = found->second;
    return query_dependency_interval_node(
        index, 1, 0, index.leaf_base, tile_ordinal, nullptr,
        visited_nodes);
}

bool RemoteReadExecutor::sort_and_detect_duplicate_dependency_parents(
    std::vector<uint64_t>* const parent_store_ids) {
    if (parent_store_ids == nullptr) {
        throw std::invalid_argument(
            "endpoint parent duplicate check input is null");
    }
    std::sort(parent_store_ids->begin(), parent_store_ids->end());
    return std::adjacent_find(parent_store_ids->begin(),
                              parent_store_ids->end()) !=
        parent_store_ids->end();
}

bool RemoteReadExecutor::endpoint_dependencies_ready(
    const RuntimeDescriptor& owner, const RunCursor& cursor) const {
    if (cursor.run == nullptr || cursor.exhausted) return false;
    const PipelineEndpointRun& endpoint_run = *cursor.run;
    const uint64_t tile_ordinal = endpoint_run.first_tile + cursor.tile_offset;
    const PipelineEndpointLeg endpoint =
        endpoint_run.legs[cursor.leg_index].endpoint;
    const uint64_t shard_ordinal =
        endpoint_run.shard_ordinals[cursor.shard_index];
    const auto matches = matching_dependency_runs(
        owner, endpoint_run.rank, endpoint_run.group_ordinal, endpoint,
        tile_ordinal, shard_ordinal);
    for (const DependencyIndexMatch& match : matches) {
        const ResolvedDependencyRun& resolved =
            owner.dependency_runs[match.dependency_run_index];
        const PipelineEndpointDependencyRun& dependency = *resolved.run;
        const uint64_t local_shard =
            static_cast<uint64_t>(match.local_shard_index);
        const uint64_t store_id = add_checked(
            resolved.first_store_id,
            add_checked(
                mul_checked(tile_ordinal - dependency.first_tile,
                            dependency.dependency_tile_stride,
                            "endpoint dependency tile offset"),
                mul_checked(local_shard,
                            dependency.dependency_shard_stride,
                            "endpoint dependency shard offset"),
                "endpoint dependency offset"),
            "endpoint dependency store ID");
        if (!terminal_completed(dependency.rank, store_id,
                                dependency.dependency_kind)) {
            return false;
        }
    }
    // An omitted dependency run means this endpoint is an explicit root. No
    // graph reachability heuristic is permitted here: every authored parent
    // must arrive through the compact dependency table.
    return true;
}

bool RemoteReadExecutor::terminal_completed(
    const int rank, const uint64_t store_id,
    const PipelineEndpointDependencyKind kind) const {
    if (rank < 0 || static_cast<size_t>(rank) >= graph_sources_.size()) {
        return false;
    }
    const NodeStore& store = graph_sources_[rank]->store();
    if (kind == PipelineEndpointDependencyKind::OrdinaryNodeTerminal) {
        return !store.is_virtual_id(store_id) &&
            graph_sources_[rank]->is_finished_or_erased(store_id);
    }
    if (!store.is_virtual_id(store_id)) return false;
    const auto position = locate_endpoint(rank, store_id);
    return position.has_value() && endpoint_position_completed(*position);
}

size_t RemoteReadExecutor::terminal_watcher_kind_index(
    const PipelineEndpointDependencyKind kind) {
    switch (kind) {
        case PipelineEndpointDependencyKind::OrdinaryNodeTerminal:
            return 0;
        case PipelineEndpointDependencyKind::VirtualEndpointTerminal:
            return 1;
    }
    throw std::invalid_argument("invalid terminal watcher kind");
}

int RemoteReadExecutor::terminal_watcher_node_height(
    const std::unique_ptr<TerminalWatcherIntervalNode>& node) {
    return node == nullptr ? 0 : node->height;
}

void RemoteReadExecutor::update_terminal_watcher_node(
    TerminalWatcherIntervalNode& node) {
    node.height = 1 + std::max(
        terminal_watcher_node_height(node.left),
        terminal_watcher_node_height(node.right));
    node.subtree_min_first_store_id = node.interval.first_store_id;
    node.subtree_max_last_store_id = node.interval.last_store_id;
    if (node.left != nullptr) {
        node.subtree_min_first_store_id = std::min(
            node.subtree_min_first_store_id,
            node.left->subtree_min_first_store_id);
        node.subtree_max_last_store_id = std::max(
            node.subtree_max_last_store_id,
            node.left->subtree_max_last_store_id);
    }
    if (node.right != nullptr) {
        node.subtree_min_first_store_id = std::min(
            node.subtree_min_first_store_id,
            node.right->subtree_min_first_store_id);
        node.subtree_max_last_store_id = std::max(
            node.subtree_max_last_store_id,
            node.right->subtree_max_last_store_id);
    }
}

void RemoteReadExecutor::rotate_terminal_watcher_left(
    std::unique_ptr<TerminalWatcherIntervalNode>& root) {
    if (root == nullptr || root->right == nullptr) {
        throw std::logic_error("invalid terminal watcher AVL left rotation");
    }
    std::unique_ptr<TerminalWatcherIntervalNode> new_root =
        std::move(root->right);
    root->right = std::move(new_root->left);
    update_terminal_watcher_node(*root);
    new_root->left = std::move(root);
    update_terminal_watcher_node(*new_root);
    root = std::move(new_root);
}

void RemoteReadExecutor::rotate_terminal_watcher_right(
    std::unique_ptr<TerminalWatcherIntervalNode>& root) {
    if (root == nullptr || root->left == nullptr) {
        throw std::logic_error("invalid terminal watcher AVL right rotation");
    }
    std::unique_ptr<TerminalWatcherIntervalNode> new_root =
        std::move(root->left);
    root->left = std::move(new_root->right);
    update_terminal_watcher_node(*root);
    new_root->right = std::move(root);
    update_terminal_watcher_node(*new_root);
    root = std::move(new_root);
}

void RemoteReadExecutor::insert_terminal_watcher_interval(
    std::unique_ptr<TerminalWatcherIntervalNode>& root,
    TerminalWatcherInterval interval, uint64_t* insert_steps) {
    if (insert_steps == nullptr) {
        throw std::invalid_argument(
            "terminal watcher insertion lacks its operation counter");
    }
    if (root == nullptr) {
        root = std::make_unique<TerminalWatcherIntervalNode>(interval);
        return;
    }
    ++*insert_steps;
    const auto key = std::tie(interval.first_store_id,
                              interval.watcher_index,
                              interval.shard_index);
    const auto root_key = std::tie(root->interval.first_store_id,
                                   root->interval.watcher_index,
                                   root->interval.shard_index);
    if (key < root_key) {
        insert_terminal_watcher_interval(root->left, interval, insert_steps);
    } else if (root_key < key) {
        insert_terminal_watcher_interval(root->right, interval, insert_steps);
    } else {
        throw std::logic_error("duplicate terminal watcher interval key");
    }
    update_terminal_watcher_node(*root);
    const int balance = terminal_watcher_node_height(root->left) -
        terminal_watcher_node_height(root->right);
    if (balance > 1) {
        const auto child_left_key = std::tie(
            root->left->interval.first_store_id,
            root->left->interval.watcher_index,
            root->left->interval.shard_index);
        if (key > child_left_key) {
            rotate_terminal_watcher_left(root->left);
        }
        rotate_terminal_watcher_right(root);
    } else if (balance < -1) {
        const auto child_right_key = std::tie(
            root->right->interval.first_store_id,
            root->right->interval.watcher_index,
            root->right->interval.shard_index);
        if (key < child_right_key) {
            rotate_terminal_watcher_right(root->right);
        }
        rotate_terminal_watcher_left(root);
    }
}

void RemoteReadExecutor::query_terminal_watcher_intervals(
    const TerminalWatcherIntervalNode* root, const uint64_t store_id,
    std::vector<TerminalWatcherInterval>* intervals,
    uint64_t* node_visits) {
    if (root == nullptr || intervals == nullptr || node_visits == nullptr ||
        root->subtree_min_first_store_id > store_id ||
        root->subtree_max_last_store_id < store_id) {
        return;
    }
    ++*node_visits;
    query_terminal_watcher_intervals(root->left.get(), store_id, intervals,
                                     node_visits);
    if (root->interval.first_store_id <= store_id &&
        store_id <= root->interval.last_store_id) {
        intervals->push_back(root->interval);
    }
    query_terminal_watcher_intervals(root->right.get(), store_id, intervals,
                                     node_visits);
}

void RemoteReadExecutor::install_terminal_watcher(
    const int rank, TerminalWatcher watcher) {
    if (rank < 0 || static_cast<size_t>(rank) >=
            terminal_watchers_by_rank_.size() ||
        static_cast<size_t>(rank) >=
            terminal_watcher_indexes_by_rank_.size()) {
        fail_closed("terminal watcher registration has an invalid rank");
    }
    const size_t kind_index = terminal_watcher_kind_index(watcher.kind);
    std::vector<TerminalWatcherInterval> intervals;
    intervals.reserve(watcher.shard_offsets.size());
    const size_t watcher_index = terminal_watchers_by_rank_[rank].size();
    for (size_t shard_index = 0;
         shard_index < watcher.shard_offsets.size(); ++shard_index) {
        const uint64_t first_store_id = add_checked(
            watcher.first_store_id, watcher.shard_offsets[shard_index],
            "terminal watcher index shard offset");
        if (watcher.tile_stride != 0 && watcher.tile_count == 0) {
            continue;
        }
        const uint64_t last_store_id = watcher.tile_stride == 0
            ? first_store_id
            : add_checked(
                first_store_id,
                mul_checked(watcher.tile_count - 1, watcher.tile_stride,
                            "terminal watcher index tile span"),
                "terminal watcher index last store ID");
        intervals.push_back(TerminalWatcherInterval{
            first_store_id, last_store_id, watcher_index, shard_index});
    }
    terminal_watchers_by_rank_[rank].push_back(std::move(watcher));
    TerminalWatcherProgressionIndex& index =
        terminal_watcher_indexes_by_rank_[rank][kind_index];
    for (const TerminalWatcherInterval& interval : intervals) {
        insert_terminal_watcher_interval(
            index.root, interval,
            &producer_metrics_.terminal_watcher_index_insert_steps);
        ++producer_metrics_.terminal_watcher_index_entries_built;
    }
}

std::vector<RemoteReadExecutor::TerminalWatcherMatch>
RemoteReadExecutor::collect_terminal_watcher_matches(
    const int rank, const uint64_t store_id,
    const PipelineEndpointDependencyKind kind) {
    if (rank < 0 || static_cast<size_t>(rank) >=
            terminal_watchers_by_rank_.size() ||
        static_cast<size_t>(rank) >=
            terminal_watcher_indexes_by_rank_.size()) {
        fail_closed("terminal watcher lookup has an invalid rank");
    }
    std::vector<TerminalWatcherInterval> candidates;
    uint64_t node_visits = 0;
    const TerminalWatcherProgressionIndex& index =
        terminal_watcher_indexes_by_rank_[rank][
            terminal_watcher_kind_index(kind)];
    query_terminal_watcher_intervals(index.root.get(), store_id,
                                     &candidates, &node_visits);
    producer_metrics_.terminal_watcher_index_query_node_visits += node_visits;
    std::sort(candidates.begin(), candidates.end(),
              [](const TerminalWatcherInterval& left,
                 const TerminalWatcherInterval& right) {
                  return std::tie(left.watcher_index, left.shard_index) <
                      std::tie(right.watcher_index, right.shard_index);
              });

    std::vector<TerminalWatcherMatch> matches;
    const auto& watchers = terminal_watchers_by_rank_[rank];
    for (const TerminalWatcherInterval& candidate : candidates) {
        ++producer_metrics_.terminal_watcher_index_candidate_examinations;
        if (candidate.watcher_index >= watchers.size()) {
            fail_closed("terminal watcher index points past registered rows");
        }
        const TerminalWatcher& watcher = watchers[candidate.watcher_index];
        if (watcher.kind != kind ||
            candidate.shard_index >= watcher.shard_offsets.size()) {
            fail_closed("terminal watcher index entry has stale coordinates");
        }
        const uint64_t first_store_id = add_checked(
            watcher.first_store_id, watcher.shard_offsets[candidate.shard_index],
            "terminal watcher lookup shard offset");
        if (first_store_id != candidate.first_store_id ||
            store_id < first_store_id) {
            fail_closed("terminal watcher interval index has stale bounds");
        }
        const uint64_t delta = store_id - first_store_id;
        uint64_t completed_obligations = 0;
        if (watcher.tile_stride == 0) {
            if (delta == 0) {
                completed_obligations = watcher.is_consume_completion
                    ? watcher.tile_count : uint64_t{1};
            }
        } else if (delta % watcher.tile_stride == 0 &&
                   delta / watcher.tile_stride < watcher.tile_count) {
            completed_obligations = 1;
        }
        if (completed_obligations == 0) continue;
        if (!matches.empty() &&
            matches.back().watcher_index == candidate.watcher_index) {
            matches.back().completed_obligations = add_checked(
                matches.back().completed_obligations, completed_obligations,
                "terminal watcher matched obligations");
        } else {
            matches.push_back(TerminalWatcherMatch{
                candidate.watcher_index, completed_obligations});
        }
    }
    return matches;
}

bool RemoteReadExecutor::slot_matches(const SlotState& slot,
                                      const RuntimeDescriptor& owner,
                                      const RunCursor& cursor,
                                      const uint64_t global_tile) {
    if (!slot.occupied || cursor.run == nullptr) return false;
    const PipelineIdentity& identity = owner.descriptor->identity();
    return slot.owner_tile == global_tile &&
        slot.owner_segment_id == identity.segment_id &&
        slot.owner_pipeline_generation == identity.generation &&
        slot.owner_stage_epoch == identity.stage_epoch &&
        slot.owner_staging_allocation_id == identity.staging_allocation_id &&
        slot.owner_group_ordinal == cursor.run->group_ordinal &&
        slot.owner_local_tile == cursor.run->first_tile + cursor.tile_offset;
}

bool RemoteReadExecutor::foreign_segment_drained(const SlotState& slot) const {
    if (!slot.occupied) return false;
    const auto it = descriptor_by_identity_.find(slot.owner_segment_id);
    if (it == descriptor_by_identity_.end() || it->second == nullptr) {
        return false;
    }
    // Settled means every physical leg already drained (outstanding_count was
    // zero and every cursor exhausted at the settle gate), so the occupancy is
    // a bookkeeping residual the segment can no longer need; retired is the
    // same fact plus retirement. In-flight foreign owners are NOT drained and
    // must keep blocking a foreign claim until their own lag-2 release.
    return it->second->settle_done || it->second->retired;
}

void RemoteReadExecutor::retain_buffer_domain_provenance(
    const std::string& domain_id, const BufferDomainState& domain,
    const int target_rank) {
    BufferDomainProvenance& provenance =
        buffer_domain_provenance_[domain_id];
    if (provenance.target_rank != -1 &&
        provenance.target_rank != target_rank) {
        fail_closed("buffer-domain provenance has inconsistent target ranks");
    }
    provenance.target_rank = target_rank;
    for (size_t slot_index = 0; slot_index < 2; ++slot_index) {
        provenance.target_slot_generations[slot_index] =
            domain.target_slots[slot_index].generation;
    }
    for (const auto& source : domain.source_slots) {
        auto& generations =
            provenance.source_slot_generations[source.first];
        for (size_t slot_index = 0; slot_index < 2; ++slot_index) {
            generations[slot_index] = source.second[slot_index].generation;
        }
    }
    for (const auto& watermark : domain.source_slot_released_through) {
        provenance.source_slot_released_through.insert_or_assign(
            watermark.first, watermark.second);
    }
}

void RemoteReadExecutor::restore_buffer_domain_provenance(
    const std::string& domain_id, BufferDomainState& domain,
    const int target_rank) {
    const auto found = buffer_domain_provenance_.find(domain_id);
    if (found == buffer_domain_provenance_.end()) return;
    const BufferDomainProvenance& provenance = found->second;
    if (provenance.target_rank != target_rank) {
        fail_closed("recreated buffer domain changed target rank");
    }
    for (size_t slot_index = 0; slot_index < 2; ++slot_index) {
        domain.target_slots[slot_index].generation =
            provenance.target_slot_generations[slot_index];
    }
    for (const auto& watermark : provenance.source_slot_released_through) {
        domain.source_slot_released_through.insert_or_assign(
            watermark.first, watermark.second);
    }
}

void RemoteReadExecutor::restore_source_slot_provenance(
    const std::string& domain_id,
    const std::pair<int, uint64_t>& source_key,
    std::array<SlotState, 2>& slots) {
    const auto domain = buffer_domain_provenance_.find(domain_id);
    if (domain == buffer_domain_provenance_.end()) return;
    const auto generations =
        domain->second.source_slot_generations.find(source_key);
    if (generations == domain->second.source_slot_generations.end()) return;
    for (size_t slot_index = 0; slot_index < 2; ++slot_index) {
        slots[slot_index].generation = generations->second[slot_index];
    }
}

void RemoteReadExecutor::claim_slot(SlotState& slot,
                                    RuntimeDescriptor& owner,
                                    const RunCursor& cursor,
                                    const uint64_t global_tile) {
    if (slot.occupied || cursor.run == nullptr) {
        fail_closed("attempted to claim an occupied or unbound buffer slot");
    }
    if (slot.generation == std::numeric_limits<uint64_t>::max()) {
        fail_closed("buffer slot generation exhausted");
    }
    ++slot.generation;
    const PipelineIdentity& identity = owner.descriptor->identity();
    slot.occupied = true;
    slot.owner_tile = global_tile;
    slot.consumed = false;
    slot.owner_segment_id = identity.segment_id;
    slot.owner_pipeline_generation = identity.generation;
    slot.owner_stage_epoch = identity.stage_epoch;
    slot.owner_staging_allocation_id = identity.staging_allocation_id;
    slot.owner_group_ordinal = cursor.run->group_ordinal;
    slot.owner_local_tile = cursor.run->first_tile + cursor.tile_offset;
    slot.target_shards_to_consume.clear();
    slot.consumed_target_shards.clear();
    if (cursor.run->legs[cursor.leg_index].endpoint ==
        PipelineEndpointLeg::DataRecv) {
        const PipelineTile tile = PipelineKernel::tile(
            *owner.descriptor, cursor.run->group_ordinal,
            cursor.run->first_tile + cursor.tile_offset - 1);
        for (const PipelineShardTile& shard : tile.shards) {
            if (shard.emit_endpoint && shard.target_rank == cursor.run->rank) {
                slot.target_shards_to_consume.insert(shard.shard_ordinal);
            }
        }
        if (slot.target_shards_to_consume.empty()) {
            fail_closed("target data slot has no local shard consumption authority");
        }
        slot.remaining_target_count = static_cast<uint64_t>(
            slot.target_shards_to_consume.size());
    }
}

void RemoteReadExecutor::clear_slot(SlotState& slot) {
    slot.occupied = false;
    slot.owner_tile = 0;
    slot.consumed = false;
    slot.owner_segment_id.clear();
    slot.owner_pipeline_generation = 0;
    slot.owner_stage_epoch = 0;
    slot.owner_staging_allocation_id = 0;
    slot.owner_group_ordinal = 0;
    slot.owner_local_tile = 0;
    slot.remaining_target_count = 0;
    slot.target_shards_to_consume.clear();
    slot.consumed_target_shards.clear();
}

void RemoteReadExecutor::release_target_slot_for_consume(
    const TerminalWatcher& watcher, const int terminal_rank,
    const uint64_t terminal_store_id) {
    if (watcher.owner == nullptr || watcher.consume_run == nullptr) return;
    const PipelineConsumeCompletionRun& run = *watcher.consume_run;
    const PipelineIdentity& identity = watcher.owner->descriptor->identity();
    if (terminal_rank != run.rank || terminal_rank != watcher.wake_rank) {
        fail_closed("consume slot terminal rank disagrees with its registered run");
    }
    const auto domain = buffer_domains_.find(identity.buffer_domain_id);
    if (domain == buffer_domains_.end()) return;

    // A shared ordinary body-tail may consume a compact interval with a
    // zero tile stride. Inspect the at-most-two live slots and prove that
    // this terminal is the mapped authority for their exact owner identity;
    // never expand the interval or release a different descriptor's tile
    // merely because its numeric tile ordinal aliases.
    for (size_t slot_index = 0;
         slot_index < domain->second.target_slots.size(); ++slot_index) {
        SlotState& slot = domain->second.target_slots[slot_index];
        if (!slot.occupied || slot.owner_segment_id != identity.segment_id ||
            slot.owner_pipeline_generation != identity.generation ||
            slot.owner_stage_epoch != identity.stage_epoch ||
            slot.owner_staging_allocation_id != identity.staging_allocation_id ||
            slot.owner_group_ordinal != run.group_ordinal ||
            slot.owner_local_tile < run.first_tile ||
            slot.owner_local_tile - run.first_tile >= run.tile_count) {
            continue;
        }
        const uint64_t tile_delta = slot.owner_local_tile - run.first_tile;
        const std::string resource_key = slot_resource_key(
            identity.buffer_domain_id, "target", identity.rank, 0,
            slot_index);
        const uint64_t tile_offset = mul_checked(
            tile_delta, run.dependency_tile_stride,
            "consume slot terminal tile offset");
        const uint64_t mapped_base = add_checked(
            watcher.first_store_id, tile_offset,
            "consume slot terminal base store ID");
        std::array<uint64_t, 1> single_added_shard{};
        std::vector<uint64_t> shared_added_shards;
        bool single_added_present = false;
        uint64_t selected_shard = 0;
        bool changed = false;
        uint64_t old_version = 0;
        const uint64_t slot_generation = slot.generation;
        const auto insert_member = [&](const uint64_t shard_ordinal) {
            if (slot.consumed_target_shards.find(shard_ordinal) ==
                slot.consumed_target_shards.end()) {
                if (!changed) {
                    old_version = resource_mutation_before(
                        "pipeline_slot", resource_key);
                    changed = true;
                }
                const auto inserted =
                    slot.consumed_target_shards.insert(shard_ordinal);
                if (inserted.second) {
                    if (run.dependency_shard_stride == 0) {
                        shared_added_shards.push_back(shard_ordinal);
                    } else {
                        single_added_shard[0] = shard_ordinal;
                        single_added_present = true;
                    }
                    if (slot.target_shards_to_consume.find(shard_ordinal) !=
                        slot.target_shards_to_consume.end()) {
                        if (slot.remaining_target_count == 0) {
                            fail_closed(
                                "target slot remaining membership underflow");
                        }
                        --slot.remaining_target_count;
                    }
                }
                slot.consumed = true;
            }
        };
        if (run.dependency_shard_stride == 0) {
            if (terminal_store_id != mapped_base) continue;
            for (const uint64_t shard_ordinal : run.shard_ordinals) {
                insert_member(shard_ordinal);
            }
        } else {
            if (terminal_store_id < mapped_base) continue;
            const uint64_t difference = terminal_store_id - mapped_base;
            if (difference % run.dependency_shard_stride != 0) continue;
            const uint64_t shard_index =
                difference / run.dependency_shard_stride;
            if (shard_index >= run.shard_ordinals.size()) continue;
            selected_shard =
                run.shard_ordinals[static_cast<size_t>(shard_index)];
            insert_member(selected_shard);
        }
        const uint64_t* added_shards = run.dependency_shard_stride == 0
            ? shared_added_shards.data() : single_added_shard.data();
        const size_t added_shard_count = run.dependency_shard_stride == 0
            ? shared_added_shards.size()
            : (single_added_present ? size_t{1} : size_t{0});
        const bool cleared =
            !slot.target_shards_to_consume.empty() &&
            slot.remaining_target_count == 0;
        if (!changed && !cleared) continue;
        if (!changed && cleared) {
            clear_slot(slot);
            continue;
        }
        if (cleared && added_shard_count == 0) {
            fail_closed("target clear has no new terminal-mapped members");
        }
        if (cleared) {
            clear_slot(slot);
        }
        if (added_shard_count != 0) {
            record_target_slot_consume(
                resource_key, slot_generation, terminal_rank, terminal_store_id,
                watcher.consume_run_index, added_shards, added_shard_count,
                cleared);
        }
        resource_mutation_after("pipeline_slot", resource_key, old_version,
                                added_shard_count == 0);
    }
}

uint64_t RemoteReadExecutor::source_released_through(
    const BufferDomainState& domain, const int rank,
    const uint64_t shard_ordinal, const size_t slot_index) const {
    const auto it = domain.source_slot_released_through.find(
        {rank, shard_ordinal});
    if (it == domain.source_slot_released_through.end()) return 0;
    return it->second[slot_index];
}

bool RemoteReadExecutor::protocol_ready(
    const RuntimeDescriptor& owner, const RunCursor& cursor) const {
    if (cursor.run == nullptr || cursor.exhausted) {
        return false;
    }
    const PipelineEndpointRun& run = *cursor.run;
    const PipelineEndpointLegRun& endpoint = run.legs[cursor.leg_index];
    const uint64_t tile_ordinal = run.first_tile + cursor.tile_offset;
    const PipelineTile tile = PipelineKernel::tile(
        *owner.descriptor, run.group_ordinal, tile_ordinal - 1);
    const PipelineShardTile& shard = shard_for(
        tile, run.shard_ordinals[cursor.shard_index]);
    const uint64_t global_tile = tile.global_tile_ordinal;
    const size_t slot_index = static_cast<size_t>((global_tile - 1) % 2);
    const auto domain_it = buffer_domains_.find(
        owner.descriptor->identity().buffer_domain_id);
    if (domain_it == buffer_domains_.end()) {
        return false;
    }
    const BufferDomainState& domain = domain_it->second;
    switch (endpoint.endpoint) {
        case PipelineEndpointLeg::DataSend: {
            const auto source_it = domain.source_slots.find(
                {run.rank, shard.shard_ordinal});
            if (source_it == domain.source_slots.end()) {
                return true;  // the default two startup credits are free
            }
            const SlotState& slot = source_it->second[slot_index];
            if (!slot.occupied ||
                slot_matches(slot, owner, cursor, global_tile)) {
                return true;
            }
            // A residual owner at or below the release watermark was already
            // authorized for reuse by a completed lag-2 release ACK (the ACK
            // may legitimately complete before this owner's claim, so its
            // reserve-time capture missed it). A residual owner from a
            // DIFFERENT segment of the lease is displaced only once that
            // foreign segment's own obligations settled or retired: the
            // cross-segment sequencing itself is expressed by the wire's
            // declared graph dependencies, but an IN-FLIGHT foreign owner's
            // legs may still legitimately need this slot's contents, so it
            // blocks the claim until its own lag-2 release (unconditional
            // displacement let later segments churn in-flight earlier owners
            // into a livelock once dozens of segments share one domain).
            if (slot.owner_segment_id !=
                    owner.descriptor->identity().segment_id &&
                foreign_segment_drained(slot)) {
                return true;
            }
            const uint64_t released = source_released_through(
                domain, run.rank, shard.shard_ordinal, slot_index);
            return slot.owner_tile <= released;
        }
        case PipelineEndpointLeg::DataRecv: {
            const SlotState& slot = domain.target_slots[slot_index];
            return !slot.occupied ||
                slot_matches(slot, owner, cursor, global_tile);
        }
        case PipelineEndpointLeg::FillAckSend: {
            const SlotState& slot = domain.target_slots[slot_index];
            return slot_matches(slot, owner, cursor, global_tile);
        }
        case PipelineEndpointLeg::FillAckRecv: {
            const auto source_it = domain.source_slots.find(
                {run.rank, shard.shard_ordinal});
            if (source_it == domain.source_slots.end()) return false;
            const SlotState& slot = source_it->second[slot_index];
            if (slot_matches(slot, owner, cursor, global_tile)) {
                return true;
            }
            if (slot.occupied && slot.owner_segment_id !=
                    owner.descriptor->identity().segment_id &&
                foreign_segment_drained(slot)) {
                return true;  // drained continuation segment; see DataSend note
            }
            const uint64_t released = source_released_through(
                domain, run.rank, shard.shard_ordinal, slot_index);
            return slot.occupied && slot.owner_tile <= released;
        }
        case PipelineEndpointLeg::ReleaseAckSend: {
            // Release ACK i belongs to the current network endpoint i, but
            // authorizes reuse of the lag-2 data slot. Its old consumer is
            // an explicit endpoint dependency; target contents are retired
            // by the consume-terminal watcher, not by this ACK callback.
            // Keeping the control transfer independent also lets its HBM/
            // network cookie drain after a later generation owns the slot.
            return has_matching_dependency_run(
                owner, run.rank, run.group_ordinal, endpoint.endpoint,
                tile_ordinal, shard.shard_ordinal);
        }
        case PipelineEndpointLeg::ReleaseAckRecv: {
            // A source-side release ACK is a real receive even when the
            // current source rank did not own the lag-2 tile (source/pass
            // boundary). The actual ACK edge/network pair is authoritative;
            // slot retirement is an optional exact-owner side effect.
            return true;
        }
        case PipelineEndpointLeg::BodyMarker: {
            const SlotState& slot = domain.target_slots[slot_index];
            return slot_matches(slot, owner, cursor, global_tile);
        }
    }
    return false;
}

bool RemoteReadExecutor::reserve_slot_for_issue(
    RuntimeDescriptor& owner, const RunCursor& cursor,
    EndpointPosition* position, bool* newly_reserved) {
    if (position == nullptr || newly_reserved == nullptr ||
        cursor.run == nullptr ||
        cursor.exhausted) {
        return false;
    }
    position->slot_generation = 0;
    position->releases_source_slot = false;
    *newly_reserved = false;
    const PipelineEndpointRun& run = *cursor.run;
    const PipelineEndpointLegRun& endpoint = run.legs[cursor.leg_index];
    const uint64_t tile_ordinal = run.first_tile + cursor.tile_offset;
    const PipelineTile tile = PipelineKernel::tile(
        *owner.descriptor, run.group_ordinal, tile_ordinal - 1);
    const uint64_t global_tile = tile.global_tile_ordinal;
    const size_t slot_index = static_cast<size_t>((global_tile - 1) % 2);
    BufferDomainState& domain = buffer_domains_[
        owner.descriptor->identity().buffer_domain_id];
    const PipelineShardTile& shard = shard_for(
        tile, run.shard_ordinals[cursor.shard_index]);
    switch (endpoint.endpoint) {
        case PipelineEndpointLeg::DataSend: {
            SlotState& slot = domain.source_slots[
                {run.rank, shard.shard_ordinal}][slot_index];
            const std::string credit_key = credit_resource_key(
                owner.descriptor->identity().buffer_domain_id,
                run.rank, shard.shard_ordinal);
            if (slot.occupied &&
                !slot_matches(slot, owner, cursor, global_tile)) {
                const bool cross_segment = slot.owner_segment_id !=
                    owner.descriptor->identity().segment_id;
                if (slot.owner_tile > source_released_through(
                        domain, run.rank, shard.shard_ordinal, slot_index) &&
                    !(cross_segment && foreign_segment_drained(slot))) {
                    return false;
                }
                // Stale residual whose reuse a completed lag-2 release ACK
                // already authorized (the watermark above), or whose foreign
                // segment settled/retired: retire it and let the new claim
                // proceed.
                const std::string stale_key = slot_resource_key(
                    owner.descriptor->identity().buffer_domain_id, "source",
                    run.rank, shard.shard_ordinal, slot_index);
                capture_credit_row_before(credit_key);
                const uint64_t stale_version = resource_mutation_before(
                    "pipeline_slot", stale_key);
                clear_slot(slot);
                resource_mutation_after("pipeline_slot", stale_key,
                                        stale_version);
            }
            if (!slot.occupied) {
                const std::string resource_key = slot_resource_key(
                    owner.descriptor->identity().buffer_domain_id, "source",
                    run.rank, shard.shard_ordinal, slot_index);
                capture_credit_row_before(credit_key);
                const uint64_t old_version = resource_mutation_before(
                    "pipeline_slot", resource_key);
                claim_slot(slot, owner, cursor, global_tile);
                resource_mutation_after("pipeline_slot", resource_key,
                                        old_version);
                *newly_reserved = true;
            }
            position->slot_generation = slot.generation;
            return true;
        }
        case PipelineEndpointLeg::DataRecv: {
            SlotState& slot = domain.target_slots[slot_index];
            if (slot.occupied &&
                !slot_matches(slot, owner, cursor, global_tile)) return false;
            if (!slot.occupied) {
                const std::string resource_key = slot_resource_key(
                    owner.descriptor->identity().buffer_domain_id, "target",
                    owner.descriptor->identity().rank, 0, slot_index);
                const uint64_t old_version = resource_mutation_before(
                    "pipeline_slot", resource_key);
                claim_slot(slot, owner, cursor, global_tile);
                record_target_slot_claim(resource_key, slot);
                resource_mutation_after("pipeline_slot", resource_key,
                                        old_version);
                *newly_reserved = true;
            }
            position->slot_generation = slot.generation;
            return true;
        }
        case PipelineEndpointLeg::FillAckSend: {
            const SlotState& slot = domain.target_slots[slot_index];
            if (!slot_matches(slot, owner, cursor, global_tile)) return false;
            position->slot_generation = slot.generation;
            return true;
        }
        case PipelineEndpointLeg::FillAckRecv: {
            const auto source = domain.source_slots.find(
                {run.rank, shard.shard_ordinal});
            if (source == domain.source_slots.end()) return false;
            const SlotState& slot = source->second[slot_index];
            const bool cross_segment = slot.occupied &&
                slot.owner_segment_id !=
                    owner.descriptor->identity().segment_id;
            if (!slot_matches(slot, owner, cursor, global_tile) &&
                !(cross_segment && foreign_segment_drained(slot)) &&
                !(slot.occupied && slot.owner_tile <= source_released_through(
                     domain, run.rank, shard.shard_ordinal, slot_index))) {
                return false;
            }
            position->slot_generation = slot.generation;
            return true;
        }
        case PipelineEndpointLeg::ReleaseAckSend:
            // The consumed target slot was retired at its authoritative
            // terminal; the ACK is an independent control obligation.
            return true;
        case PipelineEndpointLeg::ReleaseAckRecv: {
            const uint64_t lag = owner.descriptor->protocol().release_lag_tiles;
            if (global_tile <= lag) return true;
            const uint64_t old_tile = global_tile - lag;
            const size_t old_slot_index = static_cast<size_t>((old_tile - 1) % 2);
            auto source = domain.source_slots.find(
                {run.rank, shard.shard_ordinal});
            if (source == domain.source_slots.end()) return true;
            const SlotState& old_slot = source->second[old_slot_index];
            if (!old_slot.occupied || old_slot.owner_tile != old_tile) {
                return true;  // no old source content at this boundary
            }
            // Capture the exact old owner. Completion is allowed to drain
            // after the runtime's current stage/generation changes, and may
            // only retire this slot if no legal reuse replaced that owner.
            position->releases_source_slot = true;
            position->released_slot_tile = old_slot.owner_tile;
            position->released_slot_generation = old_slot.generation;
            position->released_slot_segment_id = old_slot.owner_segment_id;
            position->released_slot_pipeline_generation =
                old_slot.owner_pipeline_generation;
            position->released_slot_stage_epoch = old_slot.owner_stage_epoch;
            position->released_slot_staging_allocation_id =
                old_slot.owner_staging_allocation_id;
            return true;
        }
        case PipelineEndpointLeg::BodyMarker: {
            const SlotState& slot = domain.target_slots[slot_index];
            if (!slot_matches(slot, owner, cursor, global_tile)) return false;
            position->slot_generation = slot.generation;
            return true;
        }
    }
    return false;
}

void RemoteReadExecutor::complete_protocol_leg(
    RuntimeDescriptor& owner, const EndpointPosition& position) {
    if (position.run_index >= owner.runs.size()) {
        fail_closed("protocol completion references an unknown run");
    }
    const RunCursor& cursor = owner.runs[position.run_index];
    const PipelineEndpointRun& run = *cursor.run;
    const PipelineEndpointLegRun& endpoint = run.legs[position.leg_index];
    const uint64_t tile_ordinal = run.first_tile + position.tile_offset;
    const PipelineTile tile = PipelineKernel::tile(
        *owner.descriptor, run.group_ordinal, tile_ordinal - 1);
    BufferDomainState& domain = buffer_domains_[
        owner.descriptor->identity().buffer_domain_id];
    switch (endpoint.endpoint) {
        case PipelineEndpointLeg::ReleaseAckRecv: {
            const PipelineShardTile& shard = shard_for(
                tile, run.shard_ordinals[position.shard_index]);
            // Advance the lag-2 reuse watermark regardless of whether the
            // reserve-time capture found the old owner: the published
            // protocol legitimately emits release ACKs as independent
            // control obligations that may complete before the old tile's
            // data leg has claimed its slot, so the capture alone would
            // leak that occupancy (decode-tail stall, case-021).
            {
                const uint64_t lag =
                    owner.descriptor->protocol().release_lag_tiles;
                if (tile.global_tile_ordinal > lag) {
                    const uint64_t old_tile =
                        tile.global_tile_ordinal - lag;
                    auto& watermark = domain.source_slot_released_through[
                        {run.rank, shard.shard_ordinal}];
                    const size_t old_slot_index = static_cast<size_t>(
                        (old_tile - 1) % 2);
                    if (watermark[old_slot_index] < old_tile) {
                        watermark[old_slot_index] = old_tile;
                    }
                }
            }
            if (!position.releases_source_slot) break;
            const size_t old_slot_index = static_cast<size_t>(
                (position.released_slot_tile - 1) % 2);
            const auto source = domain.source_slots.find(
                {run.rank, shard.shard_ordinal});
            if (source == domain.source_slots.end()) break;
            SlotState& slot = source->second[old_slot_index];
            if (slot.occupied &&
                slot.owner_tile == position.released_slot_tile &&
                slot.generation == position.released_slot_generation &&
                slot.owner_segment_id == position.released_slot_segment_id &&
                slot.owner_pipeline_generation ==
                    position.released_slot_pipeline_generation &&
                slot.owner_stage_epoch == position.released_slot_stage_epoch &&
                slot.owner_staging_allocation_id ==
                    position.released_slot_staging_allocation_id) {
                capture_source_credit_row_before(
                    owner.descriptor->identity().buffer_domain_id,
                    run.rank, shard.shard_ordinal);
                const std::string resource_key = slot_resource_key(
                    owner.descriptor->identity().buffer_domain_id, "source",
                    run.rank, shard.shard_ordinal, old_slot_index);
                const uint64_t old_version = resource_mutation_before(
                    "pipeline_slot", resource_key);
                clear_slot(slot);
                resource_mutation_after("pipeline_slot", resource_key,
                                        old_version);
            }
            break;
        }
        default:
            break;
    }
}

void RemoteReadExecutor::notify_node_completed(const int rank,
                                               const uint64_t store_id) {
    if (rank < 0 || static_cast<size_t>(rank) >= graph_sources_.size()) {
        fail_closed("ordinary terminal notification has an invalid rank");
    }
    notify_terminal_completed(
        rank, store_id,
        PipelineEndpointDependencyKind::OrdinaryNodeTerminal);
}

void RemoteReadExecutor::notify_terminal_completed(
    const int rank, const uint64_t store_id,
    const PipelineEndpointDependencyKind kind) {
    if (rank < 0 || static_cast<size_t>(rank) >=
            terminal_watchers_by_rank_.size()) {
        fail_closed("terminal notification has an invalid rank");
    }
    auto& watchers = terminal_watchers_by_rank_[rank];
    const std::vector<TerminalWatcherMatch> matched_watchers =
        collect_terminal_watcher_matches(rank, store_id, kind);
    std::set<int> wake_ranks;
    std::set<RuntimeDescriptor*> changed_owners;
    std::vector<ConsumeMutationSnapshot> consume_mutations;
    const auto capture_consume_state = [](RuntimeDescriptor* owner,
                                          std::vector<ConsumeMutationSnapshot>* rows) {
        const auto found = std::find_if(
            rows->begin(), rows->end(),
            [owner](const ConsumeMutationSnapshot& row) {
                return row.owner == owner;
            });
        if (found != rows->end()) return;
        rows->push_back(ConsumeMutationSnapshot{
            owner, owner->cursor_version});
    };
    for (const TerminalWatcherMatch& match : matched_watchers) {
        if (match.watcher_index >= watchers.size()) {
            fail_closed("terminal watcher match points past registered rows");
        }
        TerminalWatcher& watcher = watchers[match.watcher_index];
        if (watcher.owner == nullptr) continue;
        uint64_t completed_obligations = match.completed_obligations;
        RuntimeDescriptor* const owner = watcher.owner;
        if (watcher.is_consume_completion) {
            // Signature 6 ledger: a delivery first consumes this
            // watcher's preaccounted budget (terminals already finished at
            // commit; the owner-global pending excludes them), then the
            // owner pending; a true over-delivery (delivered + this >
            // registered total) still fails closed.
            if (watcher.delivered + completed_obligations >
                watcher.total_obligations) {
                fail_closed("consume terminal delivered more mapped "
                            "obligations than were registered");
            }
            uint64_t from_preaccounted = 0;
            if (watcher.preaccounted > watcher.delivered) {
                from_preaccounted = std::min(
                    completed_obligations,
                    watcher.preaccounted - watcher.delivered);
            }
            const uint64_t from_pending =
                completed_obligations - from_preaccounted;
            if (owner->consume_pending_count < from_pending) {
                fail_closed("consume terminal delivered more mapped "
                            "obligations than remain pending");
            }
            watcher.delivered += completed_obligations;
            completed_obligations = from_pending;

            capture_consume_state(
                owner, defer_terminal_followups_
                    ? &deferred_consume_mutations_ : &consume_mutations);
            release_target_slot_for_consume(watcher, rank, store_id);
            const std::string consume_key =
                "consume:" + owner->descriptor->identity().segment_id + ":" +
                std::to_string(watcher.consume_run->group_ordinal) + ":" +
                std::to_string(watcher.wake_rank);
            const uint64_t old_version = resource_mutation_before(
                "pipeline_consume", consume_key);
            owner->consume_pending_count -= completed_obligations;
            resource_mutation_after("pipeline_consume", consume_key,
                                    old_version);
        }
        const std::string dependency_key =
            "dependency:" + owner->descriptor->identity().segment_id + ":" +
            std::to_string(watcher.group_ordinal) + ":" +
            std::to_string(watcher.wake_rank);
        const uint64_t dependency_version = resource_mutation_before(
            "pipeline_dependency", dependency_key);
        resource_mutation_after("pipeline_dependency", dependency_key,
                                dependency_version);
        wake_ranks.insert(watcher.wake_rank);
        changed_owners.insert(owner);
    }
    if (defer_terminal_followups_) {
        deferred_terminal_settle_owners_.insert(changed_owners.begin(),
                                                 changed_owners.end());
        deferred_terminal_wake_ranks_.insert(wake_ranks.begin(),
                                             wake_ranks.end());
    } else {
        for (const ConsumeMutationSnapshot& snapshot : consume_mutations) {
            ++snapshot.owner->cursor_version;
            publish_consume_done_transition(snapshot, nullptr, 0);
        }
        for (RuntimeDescriptor* owner : changed_owners) {
            update_settle_state(*owner);
        }
        for (const int wake_rank : wake_ranks) {
            schedule_issue_pass_(wake_rank);
        }
    }
}

void RemoteReadExecutor::publish_consume_done_transition(
    const ConsumeMutationSnapshot& snapshot,
    const EndpointPosition* endpoint_position, const uint64_t store_id) {
    if (snapshot.owner == nullptr) {
        fail_closed("consume completion snapshot lacks its descriptor owner");
    }
    const nlohmann::json null_amount{
        {"ops", nullptr}, {"bytes", nullptr}, {"latency_ns", nullptr}};
    publish_pipeline_transition(
        *snapshot.owner, "consume_done", endpoint_position, store_id, "",
        null_amount, snapshot.cursor_version_start);
}

void RemoteReadExecutor::flush_deferred_terminal_followups() {
    if (defer_terminal_followups_) {
        fail_closed("terminal followups flushed while deferral is active");
    }
    const std::set<RuntimeDescriptor*> owners =
        std::move(deferred_terminal_settle_owners_);
    deferred_terminal_settle_owners_.clear();
    const std::set<int> wake_ranks =
        std::move(deferred_terminal_wake_ranks_);
    deferred_terminal_wake_ranks_.clear();
    for (RuntimeDescriptor* owner : owners) {
        if (owner != nullptr) update_settle_state(*owner);
    }
    for (const int wake_rank : wake_ranks) {
        schedule_issue_pass_(wake_rank);
    }
}

std::optional<RemoteReadExecutor::EndpointPosition>
RemoteReadExecutor::locate_endpoint(const int rank,
                                    const uint64_t store_id) const {
    const auto candidates = overlapping_endpoint_progressions(
        rank, store_id, store_id);
    const EndpointProgressionIndexEntry* selected = nullptr;
    uint64_t selected_tile_offset = 0;
    for (const EndpointProgressionIndexEntry* entry : candidates) {
        const uint64_t delta = store_id - entry->first_store_id;
        if (delta % entry->tile_stride != 0) continue;
        const uint64_t tile_offset = delta / entry->tile_stride;
        if (tile_offset >= entry->tile_count) continue;
        if (selected == nullptr ||
            entry->insertion_order < selected->insertion_order) {
            selected = entry;
            selected_tile_offset = tile_offset;
        }
    }
    if (selected != nullptr) {
        EndpointPosition position;
        position.owner = selected->owner;
        position.run_index = selected->run_index;
        position.tile_offset = selected_tile_offset;
        position.shard_index = selected->shard_index;
        position.leg_index = selected->leg_index;
        return position;
    }
    return std::nullopt;
}

void RemoteReadExecutor::index_batch_descriptors(
    const std::vector<RuntimeDescriptor*>& owners) {
    std::map<int, std::vector<EndpointProgressionIndexEntry>> new_entries;
    for (RuntimeDescriptor* owner : owners) {
        if (owner == nullptr) continue;
        for (size_t run_index = 0; run_index < owner->runs.size(); ++run_index) {
            const RunCursor& cursor = owner->runs[run_index];
            const PipelineEndpointRun& run = *cursor.run;
            for (size_t shard_index = 0;
                 shard_index < run.shard_ordinals.size(); ++shard_index) {
                for (size_t leg_index = 0; leg_index < run.legs.size();
                     ++leg_index) {
                    if (next_endpoint_index_order_ ==
                        std::numeric_limits<uint64_t>::max()) {
                        fail_closed(
                            "endpoint provenance index order exhausted");
                    }
                    const uint64_t first = add_checked(
                        cursor.first_store_id,
                        add_checked(
                            mul_checked(static_cast<uint64_t>(shard_index),
                                        run.shard_stride,
                                        "endpoint index shard offset"),
                            run.legs[leg_index].json_offset,
                            "endpoint index leg offset"),
                        "endpoint index first store ID");
                    const uint64_t interval_count = run.tile_count;
                    const uint64_t interval_stride = run.tile_stride;
                    const uint64_t last = add_checked(
                        first,
                        mul_checked(interval_count - 1, interval_stride,
                                    "endpoint index progression extent"),
                        "endpoint index last store ID");
                    EndpointProgressionIndexEntry entry;
                    entry.owner = owner;
                    entry.run_index = run_index;
                    entry.shard_index = shard_index;
                    entry.leg_index = leg_index;
                    entry.first_store_id = first;
                    entry.last_store_id = last;
                    entry.tile_stride = interval_stride;
                    entry.tile_count = interval_count;
                    entry.insertion_order = next_endpoint_index_order_++;
                    new_entries[run.rank].push_back(std::move(entry));
                    ++producer_metrics_.endpoint_index_entries_built;
                }
            }
        }
    }
    for (auto& rank_entries : new_entries) {
        const int rank = rank_entries.first;
        auto& entries = rank_entries.second;
        std::sort(entries.begin(), entries.end(),
                  [this](const EndpointProgressionIndexEntry& left,
                         const EndpointProgressionIndexEntry& right) {
            ++producer_metrics_.endpoint_index_sort_comparisons;
            return std::tie(left.first_store_id, left.insertion_order) <
                   std::tie(right.first_store_id, right.insertion_order);
        });
        EndpointProgressionIndexBlock block;
        block.first_store_id = entries.front().first_store_id;
        block.last_store_id = entries.front().last_store_id;
        for (const EndpointProgressionIndexEntry& entry : entries) {
            block.first_store_id = std::min(block.first_store_id,
                                            entry.first_store_id);
            block.last_store_id = std::max(block.last_store_id,
                                           entry.last_store_id);
        }
        size_t base = 1;
        while (base < entries.size()) base *= 2;
        block.leaf_base = base;
        block.entries = std::move(entries);
        block.max_last_store_id_tree.assign(base * 2, 0);
        for (size_t entry_index = 0;
             entry_index < block.entries.size(); ++entry_index) {
            block.max_last_store_id_tree[base + entry_index] =
                block.entries[entry_index].last_store_id;
            ++producer_metrics_.endpoint_index_tree_build_operations;
        }
        for (size_t node = base; node-- > 1;) {
            block.max_last_store_id_tree[node] = std::max(
                block.max_last_store_id_tree[node * 2],
                block.max_last_store_id_tree[node * 2 + 1]);
            ++producer_metrics_.endpoint_index_tree_build_operations;
        }
        EndpointProgressionIndex& rank_index =
            endpoint_progressions_by_rank_[rank];
        if (!rank_index.blocks.empty() &&
            block.first_store_id <= rank_index.blocks.back().last_store_id) {
            fail_closed("accepted batch endpoint spans overlap or regress");
        }
        rank_index.blocks.push_back(std::move(block));
    }
}

std::vector<const RemoteReadExecutor::EndpointProgressionIndexEntry*>
RemoteReadExecutor::overlapping_endpoint_progressions(
    const int rank, const uint64_t first_store_id,
    const uint64_t last_store_id) const {
    std::vector<const EndpointProgressionIndexEntry*> result;
    if (first_store_id > last_store_id) return result;
    const auto found = endpoint_progressions_by_rank_.find(rank);
    if (found == endpoint_progressions_by_rank_.end() ||
        found->second.blocks.empty()) {
        return result;
    }
    const auto& blocks = found->second.blocks;
    size_t block_lo = 0;
    size_t block_hi = blocks.size();
    while (block_lo < block_hi) {
        const size_t middle = block_lo + (block_hi - block_lo) / 2;
        ++producer_metrics_.endpoint_index_block_search_steps;
        if (blocks[middle].last_store_id < first_store_id) {
            block_lo = middle + 1;
        } else {
            block_hi = middle;
        }
    }
    auto block_it = blocks.begin() + static_cast<std::ptrdiff_t>(block_lo);
    for (; block_it != blocks.end() &&
           block_it->first_store_id <= last_store_id; ++block_it) {
        ++producer_metrics_.endpoint_index_blocks_examined;
        const EndpointProgressionIndexBlock& block = *block_it;
        size_t entry_lo = 0;
        size_t entry_hi = block.entries.size();
        while (entry_lo < entry_hi) {
            const size_t middle = entry_lo + (entry_hi - entry_lo) / 2;
            ++producer_metrics_.endpoint_index_block_search_steps;
            if (block.entries[middle].first_store_id <= last_store_id) {
                entry_lo = middle + 1;
            } else {
                entry_hi = middle;
            }
        }
        const size_t limit = entry_lo;
        const auto visit = [&](const auto& self, const size_t node,
                               const size_t left, const size_t right) -> void {
            if (left >= limit ||
                block.max_last_store_id_tree[node] < first_store_id) {
                return;
            }
            if (right - left == 1) {
                if (left < block.entries.size()) {
                    ++producer_metrics_.endpoint_index_candidate_examinations;
                    const auto& entry = block.entries[left];
                    if (entry.last_store_id >= first_store_id &&
                        entry.first_store_id <= last_store_id) {
                        result.push_back(&entry);
                    }
                }
                return;
            }
            const size_t middle = left + (right - left) / 2;
            self(self, node * 2, left, middle);
            self(self, node * 2 + 1, middle, right);
        };
        visit(visit, 1, 0, block.leaf_base);
    }
    return result;
}

void RemoteReadExecutor::set_current_cursor(RuntimeDescriptor& owner,
                                            const size_t run_index) {
    if (run_index >= owner.runs.size()) {
        fail_closed("current cursor index references an unknown run");
    }
    uint64_t store_id = 0;
    const RunCursor& cursor = owner.runs[run_index];
    if (!current_store_id(cursor, &store_id)) return;
    const auto key = std::make_pair(cursor.run->rank, store_id);
    const auto inserted = current_cursors_by_store_id_.emplace(
        key, CurrentCursorBinding{&owner, run_index});
    if (!inserted.second &&
        (inserted.first->second.owner != &owner ||
         inserted.first->second.run_index != run_index)) {
        fail_closed("two descriptor cursors expose the same current endpoint");
    }
    // F (2026-10-11): the cursor now points at store_id; when a physical
    // anchor is registered there (a same-descriptor chain anchor whose run
    // just advanced onto it, or an earlier-batch anchor whose run got here),
    // it becomes a current anchor candidate. Pure id bookkeeping.
    if (!physical_anchors_by_rank_[cursor.run->rank].empty() &&
        physical_anchors_by_rank_[cursor.run->rank].find(store_id) !=
            physical_anchors_by_rank_[cursor.run->rank].end()) {
        note_current_cursor_anchor(cursor.run->rank, store_id);
    }
}

void RemoteReadExecutor::erase_current_cursor(RuntimeDescriptor& owner,
                                              const size_t run_index) {
    if (run_index >= owner.runs.size()) {
        fail_closed("current cursor erase references an unknown run");
    }
    uint64_t store_id = 0;
    const RunCursor& cursor = owner.runs[run_index];
    if (!current_store_id(cursor, &store_id)) return;
    const auto found = current_cursors_by_store_id_.find(
        std::make_pair(cursor.run->rank, store_id));
    if (found == current_cursors_by_store_id_.end() ||
        found->second.owner != &owner ||
        found->second.run_index != run_index) {
        fail_closed("current cursor is missing from its endpoint index");
    }
    current_cursors_by_store_id_.erase(found);
    // F: the cursor left store_id; its anchor (if any) stops being current.
    // Idempotent, pointer-free.
    drop_current_cursor_anchor(cursor.run->rank, store_id);
}

void RemoteReadExecutor::note_current_cursor_anchor(const int rank,
                                                    const uint64_t store_id) {
    current_anchor_ids_by_rank_[static_cast<uint64_t>(rank)].insert(store_id);
}

void RemoteReadExecutor::drop_current_cursor_anchor(const int rank,
                                                    const uint64_t store_id) {
    const auto found =
        current_anchor_ids_by_rank_.find(static_cast<uint64_t>(rank));
    if (found != current_anchor_ids_by_rank_.end()) {
        found->second.erase(store_id);
    }
}

bool RemoteReadExecutor::next_current_anchor_id(const uint64_t rank,
                                                const uint64_t after_id,
                                                uint64_t* out) const {
    if (out == nullptr || after_id == std::numeric_limits<uint64_t>::max()) {
        return false;
    }
    const auto found = current_anchor_ids_by_rank_.find(rank);
    if (found == current_anchor_ids_by_rank_.end()) {
        return false;
    }
    const auto it = found->second.upper_bound(after_id);
    if (it == found->second.end()) {
        return false;
    }
    *out = *it;
    return true;
}

size_t RemoteReadExecutor::current_anchor_count(const uint64_t rank) const {
    const auto found = current_anchor_ids_by_rank_.find(rank);
    return found == current_anchor_ids_by_rank_.end() ? 0 : found->second.size();
}

bool RemoteReadExecutor::is_current_anchor_id(const uint64_t rank,
                                              const uint64_t id) const {
    const auto found = current_anchor_ids_by_rank_.find(rank);
    return found != current_anchor_ids_by_rank_.end() &&
        found->second.find(id) != found->second.end();
}

void RemoteReadExecutor::validate_anchor_binding_static(
    const int rank, const uint64_t store_id,
    const AnchorBinding& binding) const {
    // The fail-closed prefix of resource_view_for_anchor (everything up to
    // the dynamic coordinate/preflight checks), executed once at
    // registration. Requires the committed NodeView for the segment/role
    // agreement; a missing record means the anchor can never fire (it is
    // not in the free set), matching the per-query early false.
    if (rank < 0 ||
        static_cast<size_t>(rank) >= physical_anchors_by_rank_.size()) {
        fail_closed("registered physical anchor rank is invalid");
    }
    if (graph_sources_.empty() ||
        static_cast<size_t>(rank) >= graph_sources_.size() ||
        graph_sources_[static_cast<size_t>(rank)] == nullptr) {
        fail_closed("registered physical anchor rank has no graph source");
    }
    const NodeView* node = graph_sources_[static_cast<size_t>(rank)]
                               ->store()
                               .node_ptr(store_id);
    if (node == nullptr) {
        return;  // never free -> never enumerated -> no query to fail
    }
    if (node->compute.exec_role != ComputeExecRole::RemoteReadPipelineAnchor ||
        !node->compute.pipeline_anchor.present) {
        fail_closed("registered anchor node lacks its pipeline-anchor role");
    }
    if (binding.owner == nullptr ||
        binding.run_index >= binding.owner->runs.size()) {
        fail_closed("physical anchor disagrees with committed descriptor binding");
    }
    if (binding.owner->descriptor->identity().segment_id !=
            node->compute.pipeline_anchor.segment_id ||
        graph_anchor_role(binding.role) != node->compute.pipeline_anchor.role) {
        fail_closed("physical anchor disagrees with committed descriptor binding");
    }
    const RunCursor& cursor = binding.owner->runs[binding.run_index];
    if (cursor.run == nullptr || cursor.exhausted) {
        // Match the per-query readiness helpers' exhausted/null behavior.
        return;
    }
    const auto& descriptor_runs = binding.owner->descriptor->endpoint_runs();
    if (binding.run_index >= descriptor_runs.size() ||
        cursor.run != &descriptor_runs[binding.run_index]) {
        fail_closed("physical anchor binding disagrees with its immutable run");
    }
    const PipelineEndpointRun& run = *cursor.run;
    if (binding.tile_ordinal < run.first_tile ||
        binding.tile_ordinal - run.first_tile >= run.tile_count ||
        binding.shard_index >= run.shard_ordinals.size() ||
        binding.leg_index >= run.legs.size()) {
        fail_closed("physical anchor binding position is outside its run");
    }
    if (cursor.tile_offset >= run.tile_count ||
        cursor.shard_index >= run.shard_ordinals.size() ||
        cursor.leg_index >= run.legs.size()) {
        fail_closed("non-exhausted physical anchor cursor position is invalid");
    }
}

bool RemoteReadExecutor::endpoint_position_completed(
    const EndpointPosition& position) const {
    if (position.owner == nullptr || position.owner->descriptor == nullptr ||
        position.run_index >=
            position.owner->descriptor->endpoint_runs().size()) {
        return false;
    }
    const PipelineEndpointRun& run =
        position.owner->descriptor->endpoint_runs()[position.run_index];
    if (position.tile_offset >= run.tile_count ||
        position.shard_index >= run.shard_ordinals.size() ||
        position.leg_index >= run.legs.size()) {
        return false;
    }
    const uint64_t per_tile = mul_checked(
        static_cast<uint64_t>(run.shard_ordinals.size()),
        static_cast<uint64_t>(run.legs.size()),
        "historical endpoint ordinal width");
    const uint64_t ordinal = add_checked(
        mul_checked(position.tile_offset, per_tile,
                    "historical endpoint ordinal"),
        add_checked(
            mul_checked(static_cast<uint64_t>(position.shard_index),
                        static_cast<uint64_t>(run.legs.size()),
                        "historical endpoint ordinal"),
            static_cast<uint64_t>(position.leg_index),
            "historical endpoint ordinal"),
        "historical endpoint ordinal");
    if (position.owner->retired) {
        if (position.run_index >= position.owner->historical_runs.size()) {
            return false;
        }
        // Retirement is legal only after every actually issued endpoint has
        // drained. The retained prefix is therefore the complete set of real
        // callbacks; the administrative canceled suffix remains false.
        return ordinal <
            position.owner->historical_runs[position.run_index].completed_prefix;
    }
    if (position.run_index >= position.owner->runs.size()) return false;
    const RunCursor& cursor = position.owner->runs[position.run_index];
    if (ordinal < cursor.completed_prefix) return true;
    const auto it = cursor.completed_out_of_order.upper_bound(ordinal);
    if (it == cursor.completed_out_of_order.begin()) return false;
    const auto previous = std::prev(it);
    return ordinal < previous->second;
}

NodeView RemoteReadExecutor::make_resource_view(
    const RuntimeDescriptor& owner, const RunCursor& cursor,
    const uint64_t store_id) const {
    if (cursor.run == nullptr || cursor.tile_offset >= cursor.run->tile_count ||
        cursor.shard_index >= cursor.run->shard_ordinals.size() ||
        cursor.leg_index >= cursor.run->legs.size()) {
        throw std::invalid_argument("cannot make view for exhausted endpoint cursor");
    }
    const PipelineDescriptor& descriptor = *owner.descriptor;
    const PipelineEndpointRun& run = *cursor.run;
    const PipelineEndpointLegRun& leg = run.legs[cursor.leg_index];
    const uint64_t tile_ordinal = run.first_tile + cursor.tile_offset;
    const uint64_t shard_ordinal = run.shard_ordinals[cursor.shard_index];
    const PipelineTile tile = PipelineKernel::tile(
        descriptor, run.group_ordinal, tile_ordinal - 1);
    const PipelineShardTile& shard = shard_for(tile, shard_ordinal);

    NodeView view;
    view.global_id = store_id;
    view.rank = run.rank;
    view.request_id = run.legacy_context.request_id;
    view.stage = run.legacy_context.stage;
    view.generation = run.legacy_context.generation;
    view.name = std::string("pipeline/") + descriptor.identity().segment_id +
        "/tile/" + std::to_string(tile_ordinal) + "/shard/" +
        std::to_string(shard_ordinal) + "/" + leg_name(leg.endpoint);

    if (leg.endpoint == PipelineEndpointLeg::BodyMarker) {
        view.kind = NodeKind::Compute;
        view.node_type = 4;
        view.compute.num_ops = 1;
        view.compute.tensor_size = 1;
        view.compute.op_role = OpRole::ControlMarker;
        return view;
    }

    const bool send = is_send_leg(leg.endpoint);
    const bool ack = is_ack_leg(leg.endpoint);
    const int source = ack ? shard.target_rank : shard.source_rank;
    const int target = ack ? shard.source_rank : shard.target_rank;
    view.kind = send ? NodeKind::CommSend : NodeKind::CommRecv;
    view.node_type = send ? 5 : 6;
    view.comm.src = source;
    view.comm.dst = target;
    view.comm.tag = descriptor.endpoint_tag(run, leg, tile_ordinal,
                                            shard_ordinal);
    if (leg.endpoint == PipelineEndpointLeg::DataSend ||
        leg.endpoint == PipelineEndpointLeg::DataRecv) {
        if (!shard.emit_endpoint || shard.data_bytes == 0) {
            throw std::invalid_argument(
                "data endpoint maps to a zero-byte shard tile");
        }
        view.comm.bytes = shard.data_bytes;
    } else if (leg.endpoint == PipelineEndpointLeg::FillAckSend ||
               leg.endpoint == PipelineEndpointLeg::FillAckRecv) {
        view.comm.bytes = descriptor.protocol().fill_ack_bytes;
    } else {
        view.comm.bytes = descriptor.protocol().release_ack_bytes;
    }
    view.comm.hbm_charge = descriptor.protocol().hbm_charge;
    return view;
}

std::vector<uint64_t> RemoteReadExecutor::endpoint_parent_store_ids(
    const RuntimeDescriptor& owner,
    const EndpointPosition& position) const {
    if (position.run_index >= owner.runs.size()) {
        fail_closed("endpoint parent lookup references an unknown run");
    }
    const RunCursor& cursor = owner.runs[position.run_index];
    if (position.shard_index >= cursor.run->shard_ordinals.size() ||
        position.leg_index >= cursor.run->legs.size()) {
        fail_closed("endpoint parent lookup position is outside its run");
    }
    const uint64_t tile_ordinal =
        cursor.run->first_tile + position.tile_offset;
    const uint64_t shard_ordinal =
        cursor.run->shard_ordinals[position.shard_index];
    const PipelineEndpointLeg endpoint =
        cursor.run->legs[position.leg_index].endpoint;
    std::vector<uint64_t> parents;
    const auto matches = matching_dependency_runs(
        owner, cursor.run->rank, cursor.run->group_ordinal, endpoint,
        tile_ordinal, shard_ordinal);
    for (const DependencyIndexMatch& match : matches) {
        const ResolvedDependencyRun& resolved =
            owner.dependency_runs[match.dependency_run_index];
        const PipelineEndpointDependencyRun& dependency = *resolved.run;
        const uint64_t local_shard =
            static_cast<uint64_t>(match.local_shard_index);
        const uint64_t parent_id = add_checked(
            resolved.first_store_id,
            add_checked(
                mul_checked(tile_ordinal - dependency.first_tile,
                            dependency.dependency_tile_stride,
                            "endpoint parent tile offset"),
                mul_checked(local_shard,
                            dependency.dependency_shard_stride,
                            "endpoint parent shard offset"),
                "endpoint parent offset"),
            "endpoint parent store ID");
        parents.push_back(parent_id);
    }
    if (sort_and_detect_duplicate_dependency_parents(&parents)) {
        fail_closed("endpoint maps to a duplicate dependency parent");
    }
    return parents;
}

TileTransportAdapter::TransferSpec RemoteReadExecutor::make_transfer_spec(
    const RuntimeDescriptor& owner, const RunCursor& cursor,
    const uint64_t store_id, const NodeView& resource_view) const {
    const PipelineEndpointRun& run = *cursor.run;
    const PipelineEndpointLegRun& leg = run.legs[cursor.leg_index];
    const uint64_t tile_ordinal = run.first_tile + cursor.tile_offset;
    const uint64_t shard_ordinal = run.shard_ordinals[cursor.shard_index];
    TileTransportAdapter::TransferSpec spec;
    spec.rank = run.rank;
    spec.src = resource_view.kind == NodeKind::Compute
        ? run.rank : resource_view.comm.src;
    spec.dst = resource_view.kind == NodeKind::Compute
        ? run.rank : resource_view.comm.dst;
    spec.bytes = resource_view.comm.bytes;
    spec.tag = resource_view.comm.tag;
    spec.hbm_charge = resource_view.comm.hbm_charge;
    spec.control_marker = leg.endpoint == PipelineEndpointLeg::BodyMarker;
    const PipelineTile tile = PipelineKernel::tile(
        *owner.descriptor, run.group_ordinal, tile_ordinal - 1);
    spec.logical_key.descriptor_identity =
        owner.descriptor->identity().segment_id;
    spec.logical_key.generation = owner.descriptor->identity().generation;
    spec.logical_key.tile = tile.global_tile_ordinal;
    spec.logical_key.rank = run.rank;
    spec.logical_key.shard = shard_ordinal;
    spec.logical_key.leg = leg_name(leg.endpoint);
    spec.logical_key.tag = static_cast<int>(spec.tag);
    spec.logical_key.chunk_id = -1;
    static_cast<void>(store_id);
    return spec;
}

std::vector<RemoteReadExecutor::ReadyEndpoint>
RemoteReadExecutor::ready_virtual_endpoints(const int rank) const {
    std::vector<ReadyEndpoint> result;
    if (rank < 0 || static_cast<size_t>(rank) >= graph_sources_.size()) {
        return result;
    }
    auto current = current_cursors_by_store_id_.lower_bound(
        std::make_pair(rank, uint64_t{0}));
    for (; current != current_cursors_by_store_id_.end() &&
           current->first.first == rank; ++current) {
        ++producer_metrics_.current_cursor_candidates_examined;
        const CurrentCursorBinding& binding = current->second;
        if (binding.owner == nullptr || binding.owner->retired ||
            binding.run_index >= binding.owner->runs.size()) {
            fail_closed("current endpoint index references a retired cursor");
        }
        const RuntimeDescriptor& owner = *binding.owner;
        const RunCursor& cursor = owner.runs[binding.run_index];
        const uint64_t store_id = current->first.second;
        if (cursor.run->rank != rank || cursor.exhausted) {
            continue;
        }
        if (!graph_sources_[rank]->store().is_virtual_id(store_id)) {
            ++producer_metrics_.virtual_physical_cursor_skips;
            continue;  // physical anchors use the NodeStore issue snapshot
        }
        if (!endpoint_dependencies_ready(owner, cursor) ||
            !protocol_ready(owner, cursor)) {
            continue;
        }
        ReadyEndpoint candidate;
        candidate.store_id = store_id;
        candidate.resource_node = make_resource_view(owner, cursor, store_id);
        result.push_back(std::move(candidate));
    }
    std::sort(result.begin(), result.end(),
              [](const ReadyEndpoint& left, const ReadyEndpoint& right) {
                  return left.store_id < right.store_id;
              });
    for (size_t i = 1; i < result.size(); ++i) {
        if (result[i - 1].store_id == result[i].store_id) {
            fail_closed("two descriptor cursors expose the same virtual endpoint");
        }
    }
    return result;
}

std::optional<RemoteReadExecutor::EndpointPosition>
RemoteReadExecutor::locate_anchor_endpoint(
    const PipelineDescriptor& descriptor, const PipelineAnchor& anchor) const {
    for (size_t run_index = 0; run_index < descriptor.endpoint_runs().size();
         ++run_index) {
        const auto& run = descriptor.endpoint_runs()[run_index];
        if (run.rank != anchor.rank ||
            run.group_ordinal != anchor.selector.group_ordinal ||
            anchor.selector.tile_ordinal < run.first_tile ||
            anchor.selector.tile_ordinal - run.first_tile >= run.tile_count) {
            continue;
        }
        for (size_t shard_index = 0; shard_index < run.shard_ordinals.size();
             ++shard_index) {
            if (run.shard_ordinals[shard_index] != anchor.selector.shard_ordinal) {
                continue;
            }
            for (size_t leg_index = 0; leg_index < run.legs.size(); ++leg_index) {
                const auto& leg = run.legs[leg_index];
                if (leg.endpoint != anchor.selector.endpoint ||
                    descriptor.endpoint_json_id(run, leg,
                                                anchor.selector.tile_ordinal,
                                                anchor.selector.shard_ordinal) !=
                        anchor.node_id) {
                    continue;
                }
                EndpointPosition position;
                position.run_index = run_index;
                position.tile_offset = anchor.selector.tile_ordinal - run.first_tile;
                position.shard_index = shard_index;
                position.leg_index = leg_index;
                return position;
            }
        }
    }
    return std::nullopt;
}

bool RemoteReadExecutor::resource_view_for_anchor(
    const NodeView& anchor, NodeView* resource_view) const {
    ++producer_metrics_.anchor_resource_queries;
    if (resource_view == nullptr ||
        anchor.compute.exec_role != ComputeExecRole::RemoteReadPipelineAnchor ||
        !anchor.compute.pipeline_anchor.present) {
        return false;
    }
    if (anchor.rank < 0 ||
        static_cast<size_t>(anchor.rank) >= physical_anchors_by_rank_.size()) {
        return false;
    }
    const auto binding_it =
        physical_anchors_by_rank_[anchor.rank].find(anchor.global_id);
    if (binding_it == physical_anchors_by_rank_[anchor.rank].end()) {
        return false;
    }
    const AnchorBinding& binding = binding_it->second;
    if (binding.owner == nullptr || binding.run_index >= binding.owner->runs.size() ||
        binding.owner->descriptor->identity().segment_id !=
            anchor.compute.pipeline_anchor.segment_id ||
        graph_anchor_role(binding.role) != anchor.compute.pipeline_anchor.role) {
        fail_closed("physical anchor disagrees with committed descriptor binding");
    }
    const RunCursor& cursor = binding.owner->runs[binding.run_index];
    if (cursor.run == nullptr || cursor.exhausted) {
        // Match the existing readiness helpers' exhausted/null behavior.
        return false;
    }
    const auto& descriptor_runs = binding.owner->descriptor->endpoint_runs();
    if (binding.run_index >= descriptor_runs.size() ||
        cursor.run != &descriptor_runs[binding.run_index]) {
        fail_closed("physical anchor binding disagrees with its immutable run");
    }
    const PipelineEndpointRun& run = *cursor.run;
    if (binding.tile_ordinal < run.first_tile ||
        binding.tile_ordinal - run.first_tile >= run.tile_count ||
        binding.shard_index >= run.shard_ordinals.size() ||
        binding.leg_index >= run.legs.size()) {
        fail_closed("physical anchor binding position is outside its run");
    }
    if (cursor.tile_offset >= run.tile_count ||
        cursor.shard_index >= run.shard_ordinals.size() ||
        cursor.leg_index >= run.legs.size()) {
        fail_closed("non-exhausted physical anchor cursor position is invalid");
    }
    const uint64_t binding_tile_offset =
        binding.tile_ordinal - run.first_tile;
    if (cursor.tile_offset != binding_tile_offset ||
        cursor.shard_index != binding.shard_index ||
        cursor.leg_index != binding.leg_index) {
        ++producer_metrics_.anchor_coordinate_rejections;
        return false;
    }

    uint64_t current = 0;
    ++producer_metrics_.anchor_dynamic_preflights;
    if (!endpoint_dependencies_ready(*binding.owner, cursor) ||
        !protocol_ready(*binding.owner, cursor) ||
        !current_store_id(cursor, &current) || current != anchor.global_id) {
        return false;
    }
    *resource_view = make_resource_view(*binding.owner, cursor, current);
    return true;
}

bool RemoteReadExecutor::dispatch_anchor(const NodeView& anchor) {
    NodeView resource_view;
    if (!resource_view_for_anchor(anchor, &resource_view)) {
        return false;
    }
    const AnchorBinding& binding =
        physical_anchors_by_rank_[anchor.rank].at(anchor.global_id);
    return begin_transfer(binding.owner, binding.run_index, anchor.global_id,
                          resource_view, true);
}

bool RemoteReadExecutor::dispatch_virtual_endpoint(const int rank,
                                                   const uint64_t store_id) {
    if (rank < 0 || static_cast<size_t>(rank) >= graph_sources_.size() ||
        !graph_sources_[rank]->store().is_virtual_id(store_id)) {
        return false;
    }
    const auto found = current_cursors_by_store_id_.find(
        std::make_pair(rank, store_id));
    if (found == current_cursors_by_store_id_.end()) return false;
    RuntimeDescriptor* owner = found->second.owner;
    const size_t run_index = found->second.run_index;
    if (owner == nullptr || owner->retired || run_index >= owner->runs.size()) {
        fail_closed("dispatch index references a retired endpoint cursor");
    }
    RunCursor& cursor = owner->runs[run_index];
    uint64_t current = 0;
    if (cursor.run->rank != rank || cursor.exhausted ||
        !endpoint_dependencies_ready(*owner, cursor) ||
        !protocol_ready(*owner, cursor) ||
        !current_store_id(cursor, &current) || current != store_id) {
        return false;
    }
    const NodeView resource_view = make_resource_view(*owner, cursor, store_id);
    return begin_transfer(owner, run_index, store_id, resource_view, false);
}

void RemoteReadExecutor::notify_service_gate(const std::string& request_id,
                                             const std::string& stage,
                                             const uint64_t generation) {
    for (RuntimeDescriptor* owner_pointer : active_descriptors_) {
        if (owner_pointer == nullptr) continue;
        RuntimeDescriptor& owner = *owner_pointer;
        const PipelineIdentity& identity = owner.descriptor->identity();
        const std::string descriptor_stage =
            identity.stage == PipelineStage::Prefill ? "prefill" : "decode";
        // Two legal bindings for the cursor-only existing_service_gate:
        // (a) the descriptor's own identity (request/stage/generation) --
        //     the prefill drain and the seven-rank fixture convention; and
        // (b) the owning TRAIN's sentinel watch context: production graphs
        //     author the train drain watch under the train id with the
        //     admission context (stage=prefill, generation=0) while the
        //     descriptor carries the decode request identity. The train id
        //     is recovered from the endpoint runs' legacy contexts (execdoc
        //     7.2: body-emitted endpoints keep the train NodeView/watch
        //     context), so the sentinel fire opens this descriptor's
        //     service gate exactly at the train drain.
        bool identity_match = identity.request_id == request_id &&
            descriptor_stage == stage && identity.generation == generation;
        bool train_match = false;
        if (!identity_match) {
            for (const PipelineEndpointRun& run :
                 owner.descriptor->endpoint_runs()) {
                if (run.legacy_context.request_id == request_id &&
                    run.legacy_context.request_id != identity.request_id) {
                    train_match = true;
                    break;
                }
            }
        }
        if ((!identity_match && !train_match) || owner.service_done) {
            continue;
        }
        const auto service = std::find_if(
            owner.descriptor->anchors().begin(), owner.descriptor->anchors().end(),
            [](const PipelineAnchor& anchor) {
                return anchor.role == PipelineAnchorRole::Service &&
                    anchor.binding == PipelineAnchorBinding::DescriptorCursorSelector &&
                    anchor.selector.kind == PipelineSelectorKind::ExistingServiceGate;
            });
        if (service == owner.descriptor->anchors().end()) {
            fail_closed("descriptor lacks its cursor-only service selector");
        }
        const std::string gate_key =
            "gate:" + identity.segment_id + ":service";
        const uint64_t cursor_version_start = owner.cursor_version;
        const uint64_t old_version = resource_mutation_before(
            "pipeline_gate", gate_key);
        owner.service_done = true;
        ++owner.cursor_version;
        resource_mutation_after("pipeline_gate", gate_key, old_version);
        publish_pipeline_transition(
            owner, "service_done", nullptr, 0, "",
            nlohmann::json{{"ops", nullptr}, {"bytes", nullptr},
                           {"latency_ns", nullptr}},
            cursor_version_start);
        update_settle_state(owner);
    }
}

bool RemoteReadExecutor::begin_transfer(RuntimeDescriptor* owner,
                                        const size_t run_index,
                                        const uint64_t store_id,
                                        const NodeView& resource_view,
                                        const bool physical_anchor) {
    if (owner == nullptr || run_index >= owner->runs.size()) {
        fail_closed("transfer references an unknown cursor run");
    }
    RunCursor& cursor = owner->runs[run_index];
    uint64_t current = 0;
    if (cursor.exhausted ||
        !endpoint_dependencies_ready(*owner, cursor) ||
        !protocol_ready(*owner, cursor) ||
        !current_store_id(cursor, &current) || current != store_id) {
        return false;
    }
    if (physical_anchor !=
        (physical_anchors_by_rank_[cursor.run->rank].find(store_id) !=
         physical_anchors_by_rank_[cursor.run->rank].end())) {
        fail_closed("physical/virtual endpoint classification mismatch");
    }
    const std::string domain_id =
        owner->descriptor->identity().buffer_domain_id;
    const nlohmann::json obligation_before =
        obligation_row_json(*owner, run_index);
    const uint64_t cursor_version_start = owner->cursor_version;
    TileTransportAdapter::TransferSpec spec =
        make_transfer_spec(*owner, cursor, store_id, resource_view);
    const uint64_t ordinal = position_ordinal(
        cursor, cursor.tile_offset, cursor.shard_index, cursor.leg_index);
    EndpointPosition position;
    position.owner = owner;
    position.run_index = run_index;
    position.tile_offset = cursor.tile_offset;
    position.shard_index = cursor.shard_index;
    position.leg_index = cursor.leg_index;
    bool newly_reserved = false;
    if (!reserve_slot_for_issue(*owner, cursor, &position,
                                &newly_reserved)) {
        return false;
    }
    ++cursor.in_flight_count;
    if (!cursor.active_store_ids.emplace(ordinal, store_id).second) {
        fail_closed("endpoint cursor attempted to issue one position twice");
    }
    ++owner->outstanding_count;
    if (cursor.run->legs[cursor.leg_index].endpoint ==
        PipelineEndpointLeg::ReleaseAckSend) {
        const PipelineTile tile = PipelineKernel::tile(
            *owner->descriptor, cursor.run->group_ordinal,
            cursor.run->first_tile + cursor.tile_offset - 1);
        const PipelineShardTile& shard = shard_for(
            tile, cursor.run->shard_ordinals[cursor.shard_index]);
        const std::string credit_key = credit_resource_key(
            domain_id, shard.source_rank, shard.shard_ordinal);
        const uint64_t old_version = resource_mutation_before(
            "pipeline_credit", credit_key);
        ++buffer_domains_[domain_id].outstanding_release_acks[
            {shard.source_rank, shard.shard_ordinal}];
        resource_mutation_after("pipeline_credit", credit_key, old_version);
    }
    ++owner->cursor_version;
    if (node_transition_publisher_) {
        const std::vector<uint64_t> parents =
            endpoint_parent_store_ids(*owner, position);
        node_transition_publisher_("commit", cursor.run->rank, store_id,
                                   resource_view, parents);
        node_transition_publisher_("issue", cursor.run->rank, store_id,
                                   resource_view, {});
    }
    const PipelineEndpointLeg issued_leg =
        cursor.run->legs[cursor.leg_index].endpoint;
    const char* transition_kind = issued_leg == PipelineEndpointLeg::BodyMarker
        ? "consume_start"
        : issued_leg == PipelineEndpointLeg::FillAckSend ||
          issued_leg == PipelineEndpointLeg::FillAckRecv
            ? "fill_ack_progress"
            : issued_leg == PipelineEndpointLeg::ReleaseAckSend ||
              issued_leg == PipelineEndpointLeg::ReleaseAckRecv
                ? "release_ack_progress" : "tile_issue";
    const nlohmann::json amount = issued_leg == PipelineEndpointLeg::BodyMarker
        ? nlohmann::json{{"ops", 1}, {"bytes", 1}, {"latency_ns", nullptr}}
        : nlohmann::json{{"ops", nullptr}, {"bytes", spec.bytes},
                         {"latency_ns", nullptr}};
    record_obligation_row_patch(
        obligation_before, obligation_row_json(*owner, run_index));
    publish_pipeline_transition(
        *owner, transition_kind, &position, store_id,
        issued_leg == PipelineEndpointLeg::BodyMarker ? "body_marker"
                                                       : "endpoint",
        amount, cursor_version_start);
    const auto issue_t0 = std::chrono::steady_clock::now();
    try {
        transport_->submit(
            spec, resource_view,
            [this, owner, run_index, position, store_id, physical_anchor,
             resource_view]() mutable {
                complete_transfer(owner, run_index, position, store_id,
                                  physical_anchor, std::move(resource_view));
            });
        producer_metrics_.actual_issue_ns += static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - issue_t0).count());
        ++producer_metrics_.actual_issue_calls;
    } catch (...) {
        --cursor.in_flight_count;
        cursor.active_store_ids.erase(ordinal);
        --owner->outstanding_count;
        if (newly_reserved) {
            const PipelineEndpointLeg endpoint =
                cursor.run->legs[cursor.leg_index].endpoint;
            const uint64_t tile_ordinal =
                cursor.run->first_tile + cursor.tile_offset;
            const PipelineTile tile = PipelineKernel::tile(
                *owner->descriptor, cursor.run->group_ordinal,
                tile_ordinal - 1);
            const size_t slot_index = static_cast<size_t>(
                (tile.global_tile_ordinal - 1) % 2);
            BufferDomainState& domain = buffer_domains_[
                owner->descriptor->identity().buffer_domain_id];
            SlotState* slot = nullptr;
            if (endpoint == PipelineEndpointLeg::DataSend) {
                const PipelineShardTile& shard = shard_for(
                    tile, cursor.run->shard_ordinals[cursor.shard_index]);
                capture_source_credit_row_before(
                    owner->descriptor->identity().buffer_domain_id,
                    cursor.run->rank, shard.shard_ordinal);
                slot = &domain.source_slots[
                    {cursor.run->rank, shard.shard_ordinal}][slot_index];
            } else {
                slot = &domain.target_slots[slot_index];
            }
            if (!slot->occupied ||
                !slot_matches(*slot, *owner, cursor,
                              tile.global_tile_ordinal) ||
                slot->generation != position.slot_generation) {
                fail_closed("failed to roll back a newly reserved stale slot");
            }
            clear_slot(*slot);
        }
        throw;
    }
    erase_current_cursor(*owner, run_index);
    advance(cursor);
    set_current_cursor(*owner, run_index);
    schedule_issue_pass_(cursor.run->rank);
    return true;
}

void RemoteReadExecutor::complete_transfer(RuntimeDescriptor* owner,
                                           const size_t run_index,
                                           EndpointPosition position,
                                           const uint64_t store_id,
                                           const bool physical_anchor,
                                           NodeView resource_view) {
    if (owner == nullptr || run_index >= owner->runs.size()) {
        fail_closed("completion references an unknown descriptor cursor");
    }
    RunCursor& cursor = owner->runs[run_index];
    const std::string domain_id =
        owner->descriptor->identity().buffer_domain_id;
    const nlohmann::json obligation_before =
        obligation_row_json(*owner, run_index);
    const uint64_t cursor_version_start = owner->cursor_version;
    const uint64_t ordinal = position_ordinal(
        cursor, position.tile_offset, position.shard_index, position.leg_index);
    const auto active = cursor.active_store_ids.find(ordinal);
    if (active == cursor.active_store_ids.end() || active->second != store_id ||
        cursor.in_flight_count == 0 || owner->outstanding_count == 0) {
        fail_closed("duplicate or out-of-order virtual endpoint completion");
    }
    cursor.active_store_ids.erase(active);
    --cursor.in_flight_count;
    --owner->outstanding_count;
    mark_cursor_range_completed(cursor, ordinal, 1);
    complete_protocol_leg(*owner, position);
    ++owner->cursor_version;
    const PipelineEndpointLeg completed_leg =
        cursor.run->legs[position.leg_index].endpoint;
    if (completed_leg == PipelineEndpointLeg::ReleaseAckRecv) {
        const PipelineTile tile = PipelineKernel::tile(
            *owner->descriptor, cursor.run->group_ordinal,
            cursor.run->first_tile + position.tile_offset - 1);
        const PipelineShardTile& shard = shard_for(
            tile, cursor.run->shard_ordinals[position.shard_index]);
        auto& outstanding = buffer_domains_[domain_id].outstanding_release_acks[
            {shard.source_rank, shard.shard_ordinal}];
        if (outstanding == 0) {
            // Round-23 close reconciliation: the pairing ReleaseAckSend
            // registers the obligation at ITS submit; a send that retired
            // unissued with the request lease (never-issued tail
            // endpoints retire with it, per the 2026-10-06 ruling)
            // registered none, so its already-submitted partner receive
            // completes administratively without one. Anything else is a
            // genuine accounting fault.
            if (!owner->request_closed) {
                fail_closed("release ACK receive completed without a pending obligation");
            }
        } else {
            const std::string credit_key = credit_resource_key(
                domain_id, shard.source_rank, shard.shard_ordinal);
            const uint64_t old_version = resource_mutation_before(
                "pipeline_credit", credit_key);
            --outstanding;
            resource_mutation_after("pipeline_credit", credit_key, old_version);
        }
    }
    const char* semantic_transition_kind = completed_leg == PipelineEndpointLeg::BodyMarker
        ? "consume_done"
        : completed_leg == PipelineEndpointLeg::DataRecv
            ? "data_ready"
            : completed_leg == PipelineEndpointLeg::FillAckSend ||
              completed_leg == PipelineEndpointLeg::FillAckRecv
                ? "fill_ack_done"
            : completed_leg == PipelineEndpointLeg::ReleaseAckSend ||
              completed_leg == PipelineEndpointLeg::ReleaseAckRecv
                    ? "release_ack_done" : "tile_issue";
    const nlohmann::json amount = completed_leg == PipelineEndpointLeg::BodyMarker
        ? nlohmann::json{{"ops", 1}, {"bytes", 1}, {"latency_ns", nullptr}}
        : nlohmann::json{{"ops", nullptr},
                         {"bytes", resource_view.comm.bytes},
                         {"latency_ns", nullptr}};
    record_obligation_row_patch(
        obligation_before, obligation_row_json(*owner, run_index));
    publish_pipeline_transition(
        *owner, "endpoint_done", &position, store_id,
        completed_leg == PipelineEndpointLeg::BodyMarker ? "body_marker"
                                                          : "endpoint",
        amount, cursor_version_start);
    const ConsumeMutationSnapshot marker_consume_start{
        owner, owner->cursor_version};
    if (defer_terminal_followups_) {
        fail_closed("nested terminal callback deferral");
    }
    defer_terminal_followups_ = true;
    if (node_transition_publisher_) {
        node_transition_publisher_("completion", cursor.run->rank, store_id,
                                   resource_view, {});
    }
    if (physical_anchor) {
        workloads_[cursor.run->rank]->finish_general_node(store_id,
                                                          EventType::General);
        if (node_transition_publisher_) {
            node_transition_publisher_("release", cursor.run->rank, store_id,
                                       resource_view, {});
        }
    } else {
        // A virtual endpoint has no NodeStore record/watch entry. Its only
        // graph effect is to satisfy typed virtual-terminal subscriptions and
        // expose the next bounded frontier on the regular deferred path.
        notify_terminal_completed(
            cursor.run->rank, store_id,
            PipelineEndpointDependencyKind::VirtualEndpointTerminal);
        if (node_transition_publisher_) {
            node_transition_publisher_("release", cursor.run->rank, store_id,
                                       resource_view, {});
        }
    }
    defer_terminal_followups_ = false;

    bool marker_consume_published = false;
    for (const ConsumeMutationSnapshot& snapshot : deferred_consume_mutations_) {
        if (completed_leg == PipelineEndpointLeg::BodyMarker &&
            snapshot.owner == owner) {
            publish_consume_done_transition(snapshot, &position, store_id);
            marker_consume_published = true;
        } else {
            ++snapshot.owner->cursor_version;
            publish_consume_done_transition(snapshot, nullptr, 0);
        }
    }
    deferred_consume_mutations_.clear();
    if (completed_leg == PipelineEndpointLeg::BodyMarker &&
        !marker_consume_published) {
        publish_consume_done_transition(marker_consume_start, &position, store_id);
    }
    if (semantic_transition_kind != nullptr &&
        std::string(semantic_transition_kind) != "tile_issue" &&
        std::string(semantic_transition_kind) != "consume_done") {
        const nlohmann::json null_amount{
            {"ops", nullptr}, {"bytes", nullptr}, {"latency_ns", nullptr}};
        publish_pipeline_transition(
            *owner, semantic_transition_kind, &position, store_id, "",
            null_amount, owner->cursor_version);
    }
    deferred_terminal_settle_owners_.insert(owner);
    deferred_terminal_wake_ranks_.insert(cursor.run->rank);
    flush_deferred_terminal_followups();
}

void RemoteReadExecutor::notify_request_closed(const std::string& request_id) {
    // Iterate a SNAPSHOT of the active index: the close body can retire a
    // fully-drained descriptor inline (update_settle_state ->
    // try_retire_state erases it from active_descriptors_), which shifts
    // the vector under a range-for and silently SKIPS the next descriptor
    // -- a lease that then never receives its close (observed as the sole
    // unclosed descriptor stranding M13-seed002/M02-seed001; the r1 call
    // is the LAST of the fixture's alphabetically-ordered close set, so a
    // skip there is never repaired by a later call). Descriptor pointers
    // stay valid: descriptors_ owns them and nothing is destroyed here.
    const std::vector<RuntimeDescriptor*> active_snapshot = active_descriptors_;
    for (RuntimeDescriptor* owner_pointer : active_snapshot) {
        if (owner_pointer == nullptr || owner_pointer->retired) continue;
        RuntimeDescriptor& owner = *owner_pointer;
        if (owner.request_closed || owner.retired) continue;
        const PipelineIdentity& identity = owner.descriptor->identity();
        bool owns_request = identity.request_id == request_id;
        if (!owns_request) {
            for (const PipelineEndpointRun& run :
                 owner.descriptor->endpoint_runs()) {
                if (run.legacy_context.request_id == request_id) {
                    owns_request = true;
                    break;
                }
            }
        }
        if (!owns_request) continue;
        const uint64_t cursor_version_start = owner.cursor_version;
        const std::string close_key =
            "gate:" + identity.segment_id + ":request_close";
        const uint64_t close_old_version = resource_mutation_before(
            "pipeline_gate", close_key);
        std::vector<nlohmann::json> obligations_before;
        obligations_before.reserve(owner.runs.size());
        for (size_t index = 0; index < owner.runs.size(); ++index) {
            obligations_before.push_back(obligation_row_json(owner, index));
            erase_current_cursor(owner, index);
        }
        owner.request_closed = true;
        // Retire the unconsumed staging slots with the lease (both target
        // staging and source credits this descriptor still holds).
        BufferDomainState& domain =
            buffer_domains_[owner.descriptor->identity().buffer_domain_id];
        const auto retire_slot = [&](SlotState& slot, const char* role,
                                       const int rank, const uint64_t shard,
                                       const size_t slot_index) {
            if (!slot.occupied) return;
            const std::string key = slot_resource_key(
                owner.descriptor->identity().buffer_domain_id, role, rank,
                shard, slot_index);
            if (std::string(role) == "source") {
                capture_source_credit_row_before(
                    owner.descriptor->identity().buffer_domain_id, rank, shard);
            }
            const uint64_t version =
                resource_mutation_before("pipeline_slot", key);
            const uint64_t generation = slot.generation;
            clear_slot(slot);
            if (std::string(role) == "target") {
                record_target_slot_request_close_clear(key, generation);
            }
            resource_mutation_after("pipeline_slot", key, version);
        };
        for (size_t index = 0; index < domain.target_slots.size(); ++index) {
            retire_slot(domain.target_slots[index], "target",
                        owner.descriptor->identity().rank, 0, index);
        }
        for (auto& entry : domain.source_slots) {
            for (size_t index = 0; index < entry.second.size(); ++index) {
                retire_slot(entry.second[index], "source", entry.first.first,
                            entry.first.second, index);
            }
        }
        // Never-issued tail endpoints retire with the lease: their cursors
        // close administratively. Already-submitted physical legs keep
        // draining (the transport/cookie audit still gates executor drain).
        for (RunCursor& cursor : owner.runs) {
            const uint64_t total = mul_checked(
                cursor.run->tile_count,
                mul_checked(
                    static_cast<uint64_t>(cursor.run->shard_ordinals.size()),
                    static_cast<uint64_t>(cursor.run->legs.size()),
                    "request-close endpoint count"),
                "request-close endpoint count");
            uint64_t completed = cursor.completed_prefix;
            for (const auto& interval : cursor.completed_out_of_order) {
                completed = add_checked(
                    completed, interval.second - interval.first,
                    "request-close completed endpoint count");
            }
            if (completed != cursor.completed_count) {
                fail_closed(
                    "request-close completed count disagrees with cursor intervals");
            }
            if (completed > total ||
                cursor.cancelled_endpoint_count > total - completed ||
                cursor.in_flight_count > total - completed -
                    cursor.cancelled_endpoint_count ||
                cursor.active_store_ids.size() != cursor.in_flight_count) {
                fail_closed("request-close cursor accounting is inconsistent");
            }
            const uint64_t unissued_tail = total - completed -
                cursor.cancelled_endpoint_count - cursor.in_flight_count;
            cursor.cancelled_endpoint_count = add_checked(
                cursor.cancelled_endpoint_count, unissued_tail,
                "request-close cancelled endpoint count");
            cursor.exhausted = true;
        }
        for (size_t index = 0; index < owner.runs.size(); ++index) {
            record_obligation_row_patch(
                std::move(obligations_before[index]),
                obligation_row_json(owner, index));
        }
        // outstanding_count is NOT zeroed: already-submitted physical legs
        // keep draining and decrement it naturally; settle stays gated on
        // it exactly per the ruling ("data/ACK/HBM physical legs must
        // drain before settle"). Zeroing it here made the first post-close
        // completion trip the duplicate-completion guard (M15 signature 2).
        // consume_pending_count is NOT zeroed: deferred consume
        // deliveries may still be pending replay when the close runs, and
        // the settle gate already ignores pending after request_closed
        // (settle bypass), so zeroing is redundant -- and it made the
        // replayed deliveries trip the pending guard (M10/M03/M07
        // signature 6).
        if (!owner.service_done) {
            owner.service_done = true;
            // A closed request has no live stage gate to wait on; record
            // the administrative service fact without fabricating a watch.
        }
        ++owner.cursor_version;
        resource_mutation_after("pipeline_gate", close_key,
                                close_old_version);
        publish_pipeline_transition(
            owner, "request_close", nullptr, 0, "",
            nlohmann::json{{"ops", nullptr}, {"bytes", nullptr},
                           {"latency_ns", nullptr}},
            cursor_version_start);
        update_settle_state(owner);
    }
    flush_deferred_terminal_followups();
}

void RemoteReadExecutor::retire_dead_consume_obligations() {
    // Round-24 ruling (rounds 5+9 synthesis). This is the settle/retire
    // TERMINAL gate, never close processing: notify_request_closed above
    // leaves consume_pending_count untouched on purpose, and only the
    // four-condition proof below may retire the remainder. Conditions
    // (ii)/(iv) (event queue finished, no deferred work) are guaranteed by
    // the caller's provably-no-future-events contract; (i)/(iii) are
    // re-checked here per descriptor.
    if (transport_ == nullptr || transport_->active_count() != 0) {
        return;  // condition (iii): a live transport leg can still pay
    }
    std::vector<RuntimeDescriptor*> affected;
    std::set<int> wake_ranks;
    for (RuntimeDescriptor* owner_pointer : active_descriptors_) {
        if (owner_pointer == nullptr) continue;
        RuntimeDescriptor& owner = *owner_pointer;
        if (!owner.request_closed || owner.retired ||
            owner.consume_pending_count == 0) {
            continue;  // condition (i): only ended leases retire here
        }
        owner.administratively_retired_consume_obligations = add_checked(
            owner.administratively_retired_consume_obligations,
            owner.consume_pending_count,
            "administratively retired consume obligations");
        owner.consume_pending_count = 0;
        affected.push_back(&owner);
        for (const PipelineEndpointRun& run :
             owner.descriptor->endpoint_runs()) {
            wake_ranks.insert(run.rank);
        }
    }
    if (affected.empty()) {
        return;
    }
    // Re-run the ordinary settle/retire transitions for the retired
    // balances (closed+settled descriptors go straight to the retire
    // gate; any not-yet-settled closed descriptor settles first).
    for (RuntimeDescriptor* owner : affected) {
        update_settle_state(*owner);
    }
    // The retirements can unblock continuation claims on every rank the
    // affected descriptors live on (coalesced no-op when nothing is newly
    // ready -- the round-18 settle-transition missed-wake precedent).
    for (const int wake_rank : wake_ranks) {
        schedule_issue_pass_(wake_rank);
    }
    flush_deferred_terminal_followups();
}

nlohmann::json RemoteReadExecutor::administratively_retired_consume_summary()
    const {
    nlohmann::json rows = nlohmann::json::value_t::discarded;
    for (const auto& owner_holder : descriptors_) {
        if (owner_holder == nullptr) continue;
        const RuntimeDescriptor& owner = *owner_holder;
        if (owner.administratively_retired_consume_obligations == 0) {
            continue;
        }
        if (rows.is_discarded()) {
            rows = nlohmann::json::array();
        }
        rows.push_back({
            {"segment_id", owner.descriptor->identity().segment_id},
            {"retired_count",
             owner.administratively_retired_consume_obligations},
        });
    }
    return rows.is_discarded() ? nlohmann::json() : rows;
}

void RemoteReadExecutor::update_settle_state(RuntimeDescriptor& owner) {
    if (owner.settle_done) {
        try_retire_state(owner);
        return;
    }
    if (owner.settle_done || !owner.service_done ||
        owner.outstanding_count != 0 ||
        (!owner.request_closed && owner.consume_pending_count != 0)) {
        return;
    }
    if (!owner.request_closed) {
        for (const RunCursor& cursor : owner.runs) {
            if (!cursor.exhausted) {
                return;
            }
        }
    }
    const auto settle = std::find_if(
        owner.descriptor->anchors().begin(), owner.descriptor->anchors().end(),
        [](const PipelineAnchor& anchor) {
            return anchor.role == PipelineAnchorRole::Settle &&
                anchor.binding == PipelineAnchorBinding::DescriptorCursorSelector &&
                anchor.selector.kind == PipelineSelectorKind::AllObligationsSettled;
        });
    if (settle == owner.descriptor->anchors().end()) {
        fail_closed("descriptor lacks its cursor-only settle selector");
    }
    const std::string gate_key =
        "gate:" + owner.descriptor->identity().segment_id + ":settle";
    const uint64_t cursor_version_start = owner.cursor_version;
    const uint64_t old_version = resource_mutation_before(
        "pipeline_gate", gate_key);
    owner.settle_done = true;
    ++owner.cursor_version;
    resource_mutation_after("pipeline_gate", gate_key, old_version);
    publish_pipeline_transition(
        owner, "settle_done", nullptr, 0, "",
        nlohmann::json{{"ops", nullptr}, {"bytes", nullptr},
                       {"latency_ns", nullptr}},
        cursor_version_start);

    // The settle makes this descriptor's residual slot owners displaceable
    // (foreign_segment_drained): every rank its runs live on may host a
    // continuation-segment claim that was protocol-blocked on those
    // residuals and has no other wake source once the descriptor's last leg
    // completed on a different rank. A pass with nothing newly ready is a
    // coalesced no-op.
    for (const PipelineEndpointRun& settled_run :
         owner.descriptor->endpoint_runs()) {
        schedule_issue_pass_(settled_run.rank);
    }

    try_retire_state(owner);
}

void RemoteReadExecutor::try_retire_state(RuntimeDescriptor& owner) {
    if (owner.retired || !owner.service_done || !owner.settle_done ||
        owner.outstanding_count != 0 || owner.consume_pending_count != 0) {
        return;
    }
    for (const RunCursor& cursor : owner.runs) {
        if (!cursor.exhausted || cursor.in_flight_count != 0 ||
            !cursor.active_store_ids.empty()) {
            return;
        }
    }
    const std::string domain = owner.descriptor->identity().buffer_domain_id;
    const auto domain_it = buffer_domains_.find(domain);
    if (domain_it != buffer_domains_.end()) {
        const auto owned_slot = [&](const SlotState& slot) {
            return slot.occupied &&
                slot.owner_segment_id == owner.descriptor->identity().segment_id;
        };
        for (const SlotState& slot : domain_it->second.target_slots) {
            if (owned_slot(slot)) return;
        }
        for (const auto& source : domain_it->second.source_slots) {
            for (const SlotState& slot : source.second) {
                if (owned_slot(slot)) return;
            }
        }
    }
    if (!watch_reference_counter_) {
        fail_closed("cannot retire a pipeline descriptor without an "
                    "authoritative WatchRegistry pin query");
    }
    const uint64_t watch_pins =
        watch_reference_counter_(owner.watch_member_ranges);
    if (watch_pins != 0) {
        if (!owner.queued_for_watch_retirement) {
            owner.queued_for_watch_retirement = true;
            pending_watch_retirements_.push_back(&owner);
        }
        return;
    }
    owner.queued_for_watch_retirement = false;

    owner.historical_runs.clear();
    owner.historical_runs.reserve(owner.runs.size());
    for (const RunCursor& cursor : owner.runs) {
        const uint64_t total = mul_checked(
            cursor.run->tile_count,
            mul_checked(static_cast<uint64_t>(cursor.run->shard_ordinals.size()),
                        static_cast<uint64_t>(cursor.run->legs.size()),
                        "retired endpoint count"),
            "retired endpoint count");
        uint64_t completed = cursor.completed_prefix;
        for (const auto& interval : cursor.completed_out_of_order) {
            completed = add_checked(completed, interval.second - interval.first,
                                    "retired completed endpoint count");
        }
        if (completed != cursor.completed_count) {
            fail_closed(
                "retired completed count disagrees with cursor intervals");
        }
        if (cursor.in_flight_count != 0 ||
            !cursor.active_store_ids.empty() ||
            !cursor.completed_out_of_order.empty() ||
            completed > total ||
            cursor.cancelled_endpoint_count != total - completed) {
            fail_closed("retired cursor is not a compact completed prefix");
        }
        owner.historical_runs.push_back(HistoricalRunProgress{
            cursor.run, cursor.first_store_id, cursor.completed_prefix,
            cursor.cancelled_endpoint_count});
    }

    std::vector<nlohmann::json> obligations_before;
    obligations_before.reserve(owner.runs.size());
    for (size_t index = 0; index < owner.runs.size(); ++index) {
        obligations_before.push_back(obligation_row_json(owner, index));
    }

    // Retirement is a separate observable transition. Immutable descriptor
    // provenance remains addressable for legal historical virtual parents;
    // only active owners disappear from cursor/readiness/resource scans.
    const uint64_t retire_cursor_version_start = owner.cursor_version;
    const std::string retire_key =
        "gate:" + owner.descriptor->identity().segment_id + ":retire";
    const uint64_t retire_old_version = resource_mutation_before(
        "pipeline_gate", retire_key);
    owner.retired = true;
    ++owner.cursor_version;
    resource_mutation_after("pipeline_gate", retire_key, retire_old_version);

    // The cursor tombstone is the legal release point for heavy mutable
    // descriptor state. Keep only descriptor-backed run provenance and the
    // completed prefix needed to answer later virtual-terminal lookups.
    for (const auto& anchor : owner.physical_anchor_store_ids) {
        if (anchor.first < 0 ||
            static_cast<size_t>(anchor.first) >= physical_anchors_by_rank_.size()) {
            fail_closed("retired physical anchor rank is invalid");
        }
        auto& by_id = physical_anchors_by_rank_[anchor.first];
        const auto found = by_id.find(anchor.second);
        if (found == by_id.end() || found->second.owner != &owner) {
            fail_closed("retired physical anchor is missing from its index");
        }
        by_id.erase(found);
        // F (2026-10-11): unregistered anchors leave the current-anchor set
        // as well (the id simply becomes unfindable for the issue walk).
        drop_current_cursor_anchor(anchor.first, anchor.second);
    }
    owner.physical_anchor_store_ids.clear();

    // When this is the last active descriptor in a quiescent domain, row
    // removal itself is the actual U. Capture only those rows being erased,
    // then emit their tombstones on this descriptor_retire transition.
    const std::string& domain_id =
        owner.descriptor->identity().buffer_domain_id;
    const auto active_count = active_owner_count_by_domain_.find(domain_id);
    if (active_count != active_owner_count_by_domain_.end() &&
        active_count->second == 1) {
        const auto domain = buffer_domains_.find(domain_id);
        bool quiescent = domain != buffer_domains_.end();
        if (quiescent) {
            for (const SlotState& slot : domain->second.target_slots) {
                if (slot.occupied) quiescent = false;
            }
            for (const auto& source : domain->second.source_slots) {
                for (const SlotState& slot : source.second) {
                    if (slot.occupied) quiescent = false;
                }
            }
            for (const auto& ack : domain->second.outstanding_release_acks) {
                if (ack.second != 0) quiescent = false;
            }
        }
        if (quiescent) {
            std::vector<std::tuple<std::string, std::string, uint64_t>> rows;
            rows.reserve(2 + domain->second.source_slots.size() * 3);
            const int target_rank =
                buffer_domain_target_ranks_.at(domain_id);
            for (size_t slot_index = 0;
                 slot_index < domain->second.target_slots.size(); ++slot_index) {
                const std::string key = slot_resource_key(
                    domain_id, "target", target_rank, 0, slot_index);
                const uint64_t version = resource_mutation_before(
                    "pipeline_slot", key);
                record_target_slot_erase(
                    key, domain->second.target_slots[slot_index].generation);
                rows.emplace_back("pipeline_slot", key, version);
            }
            std::set<std::pair<int, uint64_t>> source_keys;
            for (const auto& source : domain->second.source_slots) {
                source_keys.insert(source.first);
            }
            for (const auto& ack : domain->second.outstanding_release_acks) {
                source_keys.insert(ack.first);
            }
            for (const auto& source : source_keys) {
                for (size_t slot_index = 0; slot_index < 2; ++slot_index) {
                    const std::string key = slot_resource_key(
                        domain_id, "source", source.first, source.second,
                        slot_index);
                    rows.emplace_back("pipeline_slot", key,
                        resource_mutation_before("pipeline_slot", key));
                }
                const std::string key = credit_resource_key(
                    domain_id, source.first, source.second);
                rows.emplace_back("pipeline_credit", key,
                    resource_mutation_before("pipeline_credit", key));
            }
            retain_buffer_domain_provenance(
                domain_id, domain->second, target_rank);
            buffer_domains_.erase(domain);
            buffer_domain_target_ranks_.erase(domain_id);
            for (const auto& row : rows) {
                resource_mutation_after(std::get<0>(row), std::get<1>(row),
                                        std::get<2>(row));
            }
        }
    }
    for (size_t index = 0; index < obligations_before.size(); ++index) {
        record_obligation_row_patch(std::move(obligations_before[index]),
                                    obligation_row_json(owner, index));
    }
    publish_pipeline_transition(
        owner, "descriptor_retire", nullptr, 0, "",
        nlohmann::json{{"ops", nullptr}, {"bytes", nullptr},
                       {"latency_ns", nullptr}},
        retire_cursor_version_start);
    const auto active_position = std::find(active_descriptors_.begin(),
                                           active_descriptors_.end(), &owner);
    if (active_position == active_descriptors_.end()) {
        fail_closed("retired descriptor was missing from active index");
    }
    active_descriptors_.erase(active_position);
    const auto domain_count = active_owner_count_by_domain_.find(domain_id);
    if (domain_count == active_owner_count_by_domain_.end() ||
        domain_count->second == 0) {
        fail_closed("active buffer-domain owner count underflow");
    }
    if (--domain_count->second == 0) {
        active_owner_count_by_domain_.erase(domain_count);
    }
    owner.runs.clear();
    owner.dependency_runs.clear();
    owner.dependency_interval_indexes.clear();
    owner.consume_runs.clear();
    owner.watch_member_ranges.clear();
}

void RemoteReadExecutor::advance(RunCursor& cursor) {
    if (cursor.run == nullptr || cursor.exhausted) {
        fail_closed("advance of an exhausted pipeline cursor");
    }
    const auto advance_one = [&cursor]() {
        ++cursor.leg_index;
        if (cursor.leg_index == cursor.run->legs.size()) {
            cursor.leg_index = 0;
            ++cursor.shard_index;
            if (cursor.shard_index == cursor.run->shard_ordinals.size()) {
                cursor.shard_index = 0;
                ++cursor.tile_offset;
                if (cursor.tile_offset == cursor.run->tile_count) {
                    cursor.exhausted = true;
                    return;
                }
            }
        }
    };
    advance_one();
    while (!cursor.exhausted && !current_position_active(cursor)) {
        const uint64_t first_skipped = position_ordinal(
            cursor, cursor.tile_offset, cursor.shard_index, cursor.leg_index);
        const uint64_t remaining_legs =
            static_cast<uint64_t>(cursor.run->legs.size() - cursor.leg_index);
        mark_cursor_range_completed(cursor, first_skipped, remaining_legs);
        cursor.leg_index = 0;
        ++cursor.shard_index;
        if (cursor.shard_index == cursor.run->shard_ordinals.size()) {
            cursor.shard_index = 0;
            ++cursor.tile_offset;
            if (cursor.tile_offset == cursor.run->tile_count) {
                cursor.exhausted = true;
                break;
            }

            // Per-shard active tiles are prefixes on the shared group axis.
            // If no shard or retained marker has work at this boundary,
            // the entire remaining suffix is absent and can be retired with
            // one compact interval instead of a tile-by-tile walk.
            bool any_active_on_tile = false;
            const PipelineTile tile = PipelineKernel::tile(
                *cursor.descriptor, cursor.run->group_ordinal,
                cursor.run->first_tile + cursor.tile_offset - 1);
            for (size_t shard_index = 0;
                 shard_index < cursor.run->shard_ordinals.size(); ++shard_index) {
                const PipelineShardTile& shard = shard_for(
                    tile, cursor.run->shard_ordinals[shard_index]);
                for (const PipelineEndpointLegRun& leg : cursor.run->legs) {
                    if (leg.endpoint == PipelineEndpointLeg::BodyMarker ||
                        (shard.emit_endpoint && shard.data_bytes != 0)) {
                        any_active_on_tile = true;
                        break;
                    }
                }
                if (any_active_on_tile) break;
            }
            if (!any_active_on_tile) {
                const uint64_t first_suffix = position_ordinal(
                    cursor, cursor.tile_offset, 0, 0);
                const uint64_t shard_legs = mul_checked(
                    static_cast<uint64_t>(cursor.run->shard_ordinals.size()),
                    static_cast<uint64_t>(cursor.run->legs.size()),
                    "inactive endpoint suffix width");
                const uint64_t tiles_left =
                    cursor.run->tile_count - cursor.tile_offset;
                mark_cursor_range_completed(
                    cursor, first_suffix,
                    mul_checked(tiles_left, shard_legs,
                                "inactive endpoint suffix count"));
                cursor.exhausted = true;
                break;
            }
        }
    }
}

uint64_t RemoteReadExecutor::active_cursor_count() const {
    uint64_t count = 0;
    for (const RuntimeDescriptor* descriptor : active_descriptors_) {
        if (descriptor == nullptr) continue;
        for (const RunCursor& cursor : descriptor->runs) {
            if (!cursor.exhausted) {
                ++count;
            }
        }
    }
    return count;
}

std::optional<nlohmann::json> RemoteReadExecutor::ledger_endpoint_binding(
    const int rank, const uint64_t store_id) const {
    const auto position = locate_endpoint(rank, store_id);
    if (!position.has_value() || position->owner == nullptr ||
        position->run_index >=
            position->owner->descriptor->endpoint_runs().size()) {
        return std::nullopt;
    }
    const RuntimeDescriptor& owner = *position->owner;
    const PipelineEndpointRun& run =
        owner.descriptor->endpoint_runs()[position->run_index];
    if (position->shard_index >= run.shard_ordinals.size() ||
        position->leg_index >= run.legs.size()) {
        fail_closed("ledger endpoint binding resolved an invalid cursor position");
    }
    const uint64_t tile = run.first_tile + position->tile_offset;
    const uint64_t shard = run.shard_ordinals[position->shard_index];
    const PipelineEndpointLeg leg =
        run.legs[position->leg_index].endpoint;
    const PipelineLegacyContext& context = run.legacy_context;
    return nlohmann::json{
        {"descriptor_segment_id", owner.descriptor->identity().segment_id},
        {"endpoint_selector", {
            {"group_ordinal", run.group_ordinal},
            {"tile_ordinal", tile}, {"shard_ordinal", shard},
            {"endpoint", leg_name(leg)},
        }},
        {"request_id", context.request_id},
        {"stage", context.stage}, {"generation", context.generation},
        {"rank", rank}, {"node_id", store_id},
    };
}

bool RemoteReadExecutor::is_drained() const {
    if (transport_ == nullptr || transport_->active_count() != 0) {
        return false;
    }
    for (const RuntimeDescriptor* descriptor : active_descriptors_) {
        if (descriptor == nullptr || descriptor->outstanding_count != 0 ||
            !descriptor->service_done || !descriptor->settle_done) {
            return false;
        }
        for (const RunCursor& cursor : descriptor->runs) {
            if (!cursor.exhausted || cursor.in_flight_count != 0) {
                return false;
            }
        }
    }
    return true;
}

nlohmann::json RemoteReadExecutor::slot_state_json(
    const std::string& domain_id) const {
    const auto found = buffer_domains_.find(domain_id);
    if (found == buffer_domains_.end()) return nlohmann::json::array();
    const auto target_rank = buffer_domain_target_ranks_.find(domain_id);
    const int rank = target_rank == buffer_domain_target_ranks_.end()
        ? -1 : target_rank->second;
    nlohmann::json out = nlohmann::json::array();
    for (size_t index = 0; index < found->second.target_slots.size(); ++index) {
        out.push_back(slot_row_json(domain_id, "target", rank, 0, index,
                                    found->second.target_slots[index]));
    }
    for (const auto& source : found->second.source_slots) {
        for (size_t index = 0; index < source.second.size(); ++index) {
            out.push_back(slot_row_json(domain_id, "source",
                                        source.first.first,
                                        source.first.second, index,
                                        source.second[index]));
        }
    }
    const auto shard_key = [](const nlohmann::json& row) {
        return row.at("shard_ordinal").is_null()
            ? int64_t{-1}
            : static_cast<int64_t>(
                  row.at("shard_ordinal").get<uint64_t>());
    };
    std::sort(out.begin(), out.end(),
              [&](const nlohmann::json& left,
                  const nlohmann::json& right) {
        return std::make_tuple(left.at("buffer_domain_id").get<std::string>(),
                               left.at("role").get<std::string>(),
                               left.at("rank").get<int>(), shard_key(left),
                               left.at("slot_index").get<size_t>()) <
               std::make_tuple(right.at("buffer_domain_id").get<std::string>(),
                               right.at("role").get<std::string>(),
                               right.at("rank").get<int>(), shard_key(right),
                               right.at("slot_index").get<size_t>());
    });
    return out;
}

nlohmann::json RemoteReadExecutor::slot_row_json(
    const std::string& domain_id, const std::string& role, const int rank,
    const uint64_t shard_ordinal, const size_t slot_index,
    const SlotState& slot) const {
    count_row_helper_visit();
    if (role == "target") ++producer_metrics_.target_full_row_json_builds;
    const auto owner_json = [](const SlotState& value) -> nlohmann::json {
        if (!value.occupied) return nullptr;
        return nlohmann::json{
            {"segment_id", value.owner_segment_id},
            {"pipeline_generation", value.owner_pipeline_generation},
            {"stage_epoch", value.owner_stage_epoch},
            {"staging_allocation_id", value.owner_staging_allocation_id},
            {"group_ordinal", value.owner_group_ordinal},
            {"tile_ordinal", value.owner_local_tile},
        };
    };
    const auto uint_set_json = [](const std::set<uint64_t>& values) {
        nlohmann::json out = nlohmann::json::array();
        for (const uint64_t value : values) out.push_back(value);
        return out;
    };
    return nlohmann::json{
        {"buffer_domain_id", domain_id}, {"role", role},
        {"rank", rank},
        {"shard_ordinal", role == "target"
             ? nlohmann::json(nullptr) : nlohmann::json(shard_ordinal)},
        {"slot_index", slot_index}, {"occupied", slot.occupied},
        {"owner", owner_json(slot)}, {"consumed", slot.consumed},
        {"target_shards_to_consume",
         uint_set_json(slot.target_shards_to_consume)},
        {"consumed_target_shards", uint_set_json(slot.consumed_target_shards)},
    };
}

nlohmann::json RemoteReadExecutor::credit_state_json(
    const std::string& domain_id) const {
    const auto found = buffer_domains_.find(domain_id);
    nlohmann::json out = nlohmann::json::array();
    if (found == buffer_domains_.end()) return out;
    for (const auto& source : found->second.source_slots) {
        out.push_back(credit_row_json(domain_id, source.first.first,
                                      source.first.second));
    }
    return out;
}

nlohmann::json RemoteReadExecutor::credit_row_json(
    const std::string& domain_id, const int rank,
    const uint64_t shard_ordinal) const {
    count_row_helper_visit();
    const auto found = buffer_domains_.find(domain_id);
    if (found == buffer_domains_.end()) return nullptr;
    const auto source = found->second.source_slots.find({rank, shard_ordinal});
    if (source == found->second.source_slots.end()) return nullptr;
    uint64_t occupied = 0;
    for (const SlotState& slot : source->second) {
        occupied += slot.occupied ? 1 : 0;
    }
    if (occupied > 2) fail_closed("source slot occupancy exceeds credit capacity");
    const auto outstanding = found->second.outstanding_release_acks.find(
        {rank, shard_ordinal});
    return nlohmann::json{
        {"buffer_domain_id", domain_id}, {"source_rank", rank},
        {"shard_ordinal", shard_ordinal}, {"capacity", 2},
        {"available", 2 - occupied},
        {"outstanding_ack_count",
         outstanding == found->second.outstanding_release_acks.end()
             ? uint64_t{0} : outstanding->second},
    };
}

nlohmann::json RemoteReadExecutor::obligation_state_json(
    const RuntimeDescriptor& owner) const {
    return obligation_state_json_for_domain(
        owner.descriptor->identity().buffer_domain_id, &owner, nullptr);
}

nlohmann::json RemoteReadExecutor::obligation_state_json_for_domain(
    const std::string& domain_id,
    const RuntimeDescriptor* additional_owner,
    const RuntimeDescriptor* excluded_owner) const {
    nlohmann::json out = nlohmann::json::array();
    std::vector<const RuntimeDescriptor*> owners;
    owners.reserve(active_descriptors_.size() +
                   (additional_owner == nullptr ? size_t{0} : size_t{1}));
    for (const RuntimeDescriptor* candidate : active_descriptors_) {
        if (candidate == nullptr || candidate == excluded_owner ||
            candidate->retired ||
            candidate->descriptor->identity().buffer_domain_id != domain_id) {
            continue;
        }
        owners.push_back(candidate);
    }
    if (additional_owner != nullptr && additional_owner != excluded_owner &&
        !additional_owner->retired &&
        additional_owner->descriptor->identity().buffer_domain_id == domain_id &&
        std::find(owners.begin(), owners.end(), additional_owner) == owners.end()) {
        owners.push_back(additional_owner);
    }
    std::sort(owners.begin(), owners.end(), [](const auto* left,
                                                const auto* right) {
        return left->descriptor->identity().segment_id <
               right->descriptor->identity().segment_id;
    });
    for (const RuntimeDescriptor* domain_owner : owners) {
      const RuntimeDescriptor& owner = *domain_owner;
      for (size_t index = 0; index < owner.runs.size(); ++index) {
        nlohmann::json row = obligation_row_json(owner, index);
        if (!row.is_null()) out.push_back(std::move(row));
      }
    }
    std::sort(out.begin(), out.end(),
              [](const nlohmann::json& left,
                 const nlohmann::json& right) {
        return std::make_tuple(left.at("segment_id").get<std::string>(),
                               left.at("rank").get<int>(),
                               left.at("first_store_id").get<uint64_t>(),
                               left.at("obligation_id").get<std::string>()) <
               std::make_tuple(right.at("segment_id").get<std::string>(),
                               right.at("rank").get<int>(),
                               right.at("first_store_id").get<uint64_t>(),
                               right.at("obligation_id").get<std::string>());
    });
    return out;
}

nlohmann::json RemoteReadExecutor::obligation_row_json(
    const RuntimeDescriptor& owner, const size_t run_index) const {
    count_row_helper_visit();
    if (owner.retired || run_index >= owner.runs.size()) return nullptr;
    const RunCursor& cursor = owner.runs[run_index];
    const uint64_t total = mul_checked(
        cursor.run->tile_count,
        mul_checked(static_cast<uint64_t>(cursor.run->shard_ordinals.size()),
                    static_cast<uint64_t>(cursor.run->legs.size()),
                    "obligation endpoint count"),
        "obligation endpoint count");
    // This scalar read replaces summing the out-of-order interval map here;
    // cursor serialization, close, and retirement still verify the cache
    // against the exact intervals where those rows are already traversed.
    const uint64_t completed = cursor.completed_count;
    ++producer_metrics_.obligation_completed_count_reads;
    if (completed < cursor.completed_prefix) {
        fail_closed("cached completed count is below its cursor prefix");
    }
    if (completed > total) fail_closed("cursor completion exceeds endpoint run");
    if (cursor.cancelled_endpoint_count > total - completed) {
        fail_closed("cancelled endpoint count exceeds unresolved run");
    }
    if (!owner.request_closed && cursor.cancelled_endpoint_count != 0) {
        fail_closed("cursor has cancelled endpoints before request close");
    }
    const uint64_t remaining = total - completed -
        cursor.cancelled_endpoint_count;
    if (remaining == 0) return nullptr;
    uint64_t first_store_id = 0;
    if (owner.request_closed) {
        if (cursor.active_store_ids.empty() ||
            cursor.active_store_ids.size() != cursor.in_flight_count ||
            remaining != cursor.in_flight_count) {
            fail_closed("closed obligation does not match issued endpoints");
        }
        first_store_id = cursor.active_store_ids.begin()->second;
    } else {
        if (cursor.completed_prefix >= total) {
            fail_closed("unfinished endpoint obligation has no unresolved prefix");
        }
        const uint64_t legs_per_tile =
            static_cast<uint64_t>(cursor.run->legs.size());
        const uint64_t shards_per_tile =
            static_cast<uint64_t>(cursor.run->shard_ordinals.size());
        const uint64_t endpoints_per_tile = mul_checked(
            shards_per_tile, legs_per_tile,
            "obligation endpoint position width");
        const uint64_t first_unresolved = cursor.completed_prefix;
        const uint64_t tile_offset = first_unresolved / endpoints_per_tile;
        const uint64_t within_tile = first_unresolved % endpoints_per_tile;
        const uint64_t shard_offset = within_tile / legs_per_tile;
        const uint64_t leg_offset = within_tile % legs_per_tile;
        first_store_id = add_checked(
            cursor.first_store_id,
            add_checked(
                add_checked(
                    mul_checked(tile_offset, cursor.run->tile_stride,
                                "obligation tile store offset"),
                    mul_checked(shard_offset, cursor.run->shard_stride,
                                "obligation shard store offset"),
                    "obligation endpoint store offset"),
                cursor.run->legs[static_cast<size_t>(leg_offset)].json_offset,
                "obligation endpoint store offset"),
            "obligation first unresolved store ID");
    }
    return nlohmann::json{
        {"obligation_id", owner.descriptor->identity().segment_id + "/run/" +
                              std::to_string(run_index)},
        {"segment_id", owner.descriptor->identity().segment_id},
        {"rank", cursor.run->rank}, {"first_store_id", first_store_id},
        {"remaining_endpoint_count", remaining},
        {"state", cursor.in_flight_count != 0 ? "active" : "pending"},
    };
}

nlohmann::json RemoteReadExecutor::cursor_run_json(
    const RuntimeDescriptor& owner, const size_t run_index) const {
    count_row_helper_visit();
    ++producer_metrics_.cursor_run_captures;
    if (run_index >= owner.runs.size()) {
        fail_closed("cursor patch references an unknown run");
    }
    const RunCursor& cursor = owner.runs[run_index];
    uint64_t next_store_id = 0;
    const bool has_next = current_store_id(cursor, &next_store_id);
    const uint64_t total_count = mul_checked(
        cursor.run->tile_count,
        mul_checked(static_cast<uint64_t>(cursor.run->shard_ordinals.size()),
                    static_cast<uint64_t>(cursor.run->legs.size()),
                    "run endpoint count"),
        "run endpoint count");
    const uint64_t completed_prefix = cursor.completed_prefix;
    uint64_t out_of_order_count = 0;
    nlohmann::json out_of_order = nlohmann::json::array();
    for (const auto& interval : cursor.completed_out_of_order) {
        if (interval.first < completed_prefix ||
            interval.second <= interval.first ||
            interval.second > total_count) {
            fail_closed("cursor has an invalid out-of-order completion interval");
        }
        out_of_order_count = add_checked(
            out_of_order_count, interval.second - interval.first,
            "cursor out-of-order completion count");
        out_of_order.push_back({
            {"first_ordinal", interval.first},
            {"end_ordinal_exclusive", interval.second},
        });
    }
    if (completed_prefix > total_count ||
        out_of_order_count > total_count - completed_prefix) {
        fail_closed("cursor completed count exceeds its endpoint run");
    }
    const uint64_t completed_count = completed_prefix + out_of_order_count;
    if (completed_count != cursor.completed_count) {
        fail_closed("cursor cached completed count disagrees with intervals");
    }
    if (cursor.cancelled_endpoint_count > total_count - completed_count ||
        (!owner.request_closed && cursor.cancelled_endpoint_count != 0)) {
        fail_closed("cursor cancelled endpoint count is inconsistent");
    }
    const uint64_t remaining_count =
        total_count - completed_count - cursor.cancelled_endpoint_count;
    uint64_t active_count = 0;
    nlohmann::json active_endpoints = nlohmann::json::array();
    const uint64_t endpoints_per_tile = mul_checked(
        static_cast<uint64_t>(cursor.run->shard_ordinals.size()),
        static_cast<uint64_t>(cursor.run->legs.size()),
        "active endpoint position width");
    const PipelineIdentity& identity = owner.descriptor->identity();
    for (const auto& active : cursor.active_store_ids) {
        const uint64_t ordinal = active.first;
        if (ordinal >= total_count || endpoints_per_tile == 0) {
            fail_closed("active endpoint ordinal is outside its run");
        }
        const uint64_t tile_offset = ordinal / endpoints_per_tile;
        const uint64_t within_tile = ordinal % endpoints_per_tile;
        const size_t shard_index = static_cast<size_t>(
            within_tile / cursor.run->legs.size());
        const size_t leg_index = static_cast<size_t>(
            within_tile % cursor.run->legs.size());
        const uint64_t tile_ordinal = cursor.run->first_tile + tile_offset;
        const uint64_t shard_ordinal = cursor.run->shard_ordinals[shard_index];
        TileTransportAdapter::ActiveTransfer transfer;
        if (transport_ == nullptr || !transport_->find_active_by_node(
                cursor.run->rank, active.second, &transfer)) {
            fail_closed("active cursor endpoint lacks its transport identity");
        }
        active_endpoints.push_back({
            {"store_id", active.second},
            {"tile_ordinal", tile_ordinal},
            {"shard_ordinal", shard_ordinal},
            {"endpoint", leg_name(cursor.run->legs[leg_index].endpoint)},
            {"transfer_id", transfer.transfer_id},
            {"flow_id", transfer.flow_bound
                 ? nlohmann::json(transfer.flow_id) : nlohmann::json(nullptr)},
            {"network_done", transfer.network_done},
            {"hbm_done", transfer.hbm_done},
            {"owner", {
                {"segment_id", identity.segment_id},
                {"pipeline_generation", identity.generation},
                {"stage_epoch", identity.stage_epoch},
                {"staging_allocation_id", identity.staging_allocation_id},
                {"group_ordinal", cursor.run->group_ordinal},
                {"tile_ordinal", tile_ordinal},
            }},
        });
        ++active_count;
    }
    if (active_count != cursor.active_store_ids.size() ||
        active_count != cursor.in_flight_count) {
        fail_closed("cursor active endpoint count is inconsistent");
    }
    if (owner.request_closed &&
        (has_next || remaining_count != active_count)) {
        fail_closed("closed cursor remaining count differs from active work");
    }
    return nlohmann::json{
        {"rank", cursor.run->rank},
        {"group_ordinal", cursor.run->group_ordinal},
        {"first_tile", cursor.run->first_tile},
        {"tile_count", cursor.run->tile_count},
        {"completed_endpoint_count", completed_prefix},
        {"cancelled_endpoint_count", cursor.cancelled_endpoint_count},
        {"remaining_endpoint_count", remaining_count},
        {"in_flight", cursor.in_flight_count != 0},
        {"next_store_id", has_next ? nlohmann::json(next_store_id)
                                      : nlohmann::json(nullptr)},
        {"tile_offset", cursor.tile_offset},
        {"shard_index", cursor.shard_index},
        {"leg_index", cursor.leg_index},
        {"initial_parent_store_id", cursor.has_initial_parent
             ? nlohmann::json(cursor.initial_parent_store_id)
             : nlohmann::json(nullptr)},
        {"completed_out_of_order", std::move(out_of_order)},
        {"active_endpoints", std::move(active_endpoints)},
    };
}

void RemoteReadExecutor::mark_cursor_dirty(
    RuntimeDescriptor& owner, const std::optional<size_t> run_index) {
    dirty_cursor_segments_.insert(&owner);
    if (run_index.has_value()) {
        if (*run_index >= owner.runs.size()) {
            fail_closed("cursor dirty mark references an unknown run");
        }
        dirty_cursor_runs_[&owner].insert(*run_index);
    }
}

nlohmann::json RemoteReadExecutor::take_cursor_patches(
    const uint64_t now_tick) {
    const auto capture_t0 = std::chrono::steady_clock::now();
    static_cast<void>(now_tick);  // the envelope tick is the cursor as-of tick.
    std::vector<RuntimeDescriptor*> owners(dirty_cursor_segments_.begin(),
                                           dirty_cursor_segments_.end());
    std::sort(owners.begin(), owners.end(), [](const auto* left,
                                               const auto* right) {
        return left->descriptor->identity().segment_id <
               right->descriptor->identity().segment_id;
    });
    nlohmann::json patches = nlohmann::json::array();
    for (RuntimeDescriptor* owner : owners) {
        if (owner == nullptr) continue;
        nlohmann::json runs = nlohmann::json::array();
        if (owner->retired || retired_cursor_segments_.find(owner) !=
                                  retired_cursor_segments_.end()) {
            for (size_t index = 0;
                 index < owner->descriptor->endpoint_runs().size(); ++index) {
                runs.push_back({{"run_index", index}, {"value", nullptr}});
                ++producer_metrics_.cursor_run_patch_rows;
            }
        } else {
            const auto dirty = dirty_cursor_runs_.find(owner);
            if (dirty != dirty_cursor_runs_.end()) {
                for (const size_t index : dirty->second) {
                    runs.push_back({{"run_index", index},
                                    {"value", cursor_run_json(*owner, index)}});
                    ++producer_metrics_.cursor_run_patch_rows;
                }
            }
        }
        patches.push_back({
            {"segment_id", owner->descriptor->identity().segment_id},
            {"cursor_version", owner->cursor_version},
            {"retired", owner->retired || retired_cursor_segments_.find(owner) !=
                                          retired_cursor_segments_.end()},
            {"runs", std::move(runs)},
        });
    }
    dirty_cursor_runs_.clear();
    dirty_cursor_segments_.clear();
    retired_cursor_segments_.clear();
    producer_metrics_.cursor_patch_capture_ns +=
        static_cast<uint64_t>(std::chrono::duration_cast<
            std::chrono::nanoseconds>(std::chrono::steady_clock::now() -
                                      capture_t0).count());
    ++producer_metrics_.cursor_patch_capture_calls;
    return patches;
}

void RemoteReadExecutor::publish_pipeline_transition(
    RuntimeDescriptor& owner, const std::string& transition_kind,
    const EndpointPosition* position, const uint64_t store_id,
    const std::string& resource_leg, const nlohmann::json& amount,
    const uint64_t cursor_version_start) {
    if (!pipeline_transition_publisher_) return;
    const auto publish_t0 = std::chrono::steady_clock::now();
    const PipelineIdentity& identity = owner.descriptor->identity();
    flush_pending_slot_row_patches();
    flush_pending_credit_row_patches();
    if (transition_kind == "descriptor_commit") {
        mark_cursor_dirty(owner);
        for (size_t index = 0; index < owner.runs.size(); ++index) {
            dirty_cursor_runs_[&owner].insert(index);
            record_obligation_row_patch(nullptr,
                                        obligation_row_json(owner, index));
        }
    } else if (transition_kind == "descriptor_retire") {
        mark_cursor_dirty(owner);
        retired_cursor_segments_.insert(&owner);
    } else if (transition_kind == "request_close") {
        mark_cursor_dirty(owner);
        for (size_t index = 0; index < owner.runs.size(); ++index) {
            dirty_cursor_runs_[&owner].insert(index);
        }
    } else {
        mark_cursor_dirty(owner, position == nullptr
            ? std::nullopt : std::optional<size_t>(position->run_index));
    }
    flush_pending_obligation_row_patches();
    nlohmann::json run;
    run["pattern_id"] = identity.segment_id + "/" + transition_kind + "/" +
        (position == nullptr ? std::string("descriptor")
                             : std::to_string(store_id));
    run["repeat_count"] = 1;
    run["transition_kind"] = transition_kind;
    run["descriptor_segment_id"] = identity.segment_id;
    run["resource_leg"] = resource_leg.empty()
        ? nlohmann::json(nullptr) : nlohmann::json(resource_leg);
    run["resource_quantity_prefix"] = nlohmann::json::array();
    nlohmann::json zero;
    for (const char* key : {"ops", "bytes", "latency_ns"}) {
        if (!amount.contains(key) || amount.at(key).is_null()) {
            zero[key] = nullptr;
        } else {
            zero[key] = 0;
        }
    }
    run["resource_quantity_prefix"].push_back(std::move(zero));
    run["resource_quantity_prefix"].push_back(amount);
    run["cursor_version_start"] = cursor_version_start;
    run["cursor_version_end"] = owner.cursor_version;
    run["slot_patches"] = std::move(pending_slot_patches_);
    run["credit_patches"] = std::move(pending_credit_patches_);
    run["obligation_patches"] = std::move(pending_obligation_patches_);
    pending_slot_patches_.clear();
    pending_credit_patches_.clear();
    pending_obligation_patches_.clear();
    if (position == nullptr) {
        run["tile_first"] = nullptr;
        run["tile_stride"] = nullptr;
        run["rank_first"] = nullptr;
        run["rank_stride"] = nullptr;
        run["shard_first"] = nullptr;
        run["shard_stride"] = nullptr;
        run["endpoint_selector"] = nullptr;
        run["store_id_first"] = nullptr;
        run["store_id_stride"] = nullptr;
        run["legacy_context"] = nullptr;
        run["tick_first"] = Sys::boostedTick();
        run["tick_stride"] = 0;
    } else {
        if (position->run_index >= owner.runs.size()) {
            fail_closed("pipeline transition references an unknown run");
        }
        const RunCursor& cursor = owner.runs[position->run_index];
        const PipelineEndpointRun& endpoint_run = *cursor.run;
        if (position->shard_index >= endpoint_run.shard_ordinals.size() ||
            position->leg_index >= endpoint_run.legs.size()) {
            fail_closed("pipeline transition position is outside its run");
        }
        const uint64_t tile = endpoint_run.first_tile + position->tile_offset;
        const uint64_t shard = endpoint_run.shard_ordinals[position->shard_index];
        const auto& leg = endpoint_run.legs[position->leg_index];
        run["tile_first"] = tile;
        run["tile_stride"] = 0;
        run["rank_first"] = endpoint_run.rank;
        run["rank_stride"] = 0;
        run["shard_first"] = shard;
        run["shard_stride"] = 0;
        run["endpoint_selector"] = nlohmann::json{
            {"group_ordinal", endpoint_run.group_ordinal},
            {"tile_ordinal", tile}, {"shard_ordinal", shard},
            {"endpoint", leg_name(leg.endpoint)},
        };
        run["store_id_first"] = store_id;
        run["store_id_stride"] = 0;
        run["legacy_context"] = nlohmann::json{
            {"request_id", endpoint_run.legacy_context.request_id},
            {"stage", endpoint_run.legacy_context.stage},
            {"generation", endpoint_run.legacy_context.generation},
        };
        run["tick_first"] = Sys::boostedTick();
        run["tick_stride"] = 0;
    }
    pipeline_transition_publisher_(std::move(run));
    producer_metrics_.transition_publish_ns +=
        static_cast<uint64_t>(std::chrono::duration_cast<
            std::chrono::nanoseconds>(std::chrono::steady_clock::now() -
                                      publish_t0).count());
    ++producer_metrics_.transition_publish_calls;
}

nlohmann::json RemoteReadExecutor::blocked_endpoints_diagnostics() const {
    ++producer_metrics_.blocked_diagnostics_calls;
    const DiagnosticRowCaptureScope diagnostic_scope(
        diagnostic_row_capture_active_);
    nlohmann::json rows = nlohmann::json::array();
    // Transport-side in-flight view: names the physical transfers that are
    // still active at capture time (id, endpoints, per-phase completion), so
    // executor outstanding counters can be reconciled against the transport.
    if (transport_ != nullptr) {
        for (const TileTransportAdapter::ActiveTransfer& transfer :
             transport_->active_snapshot()) {
            rows.push_back({
                {"row_kind", "active_transfer"},
                {"transfer_id", transfer.transfer_id},
                {"submit_tick", transfer.submit_tick},
                {"rank", transfer.rank},
                {"logical_node_id", transfer.logical_node_id},
                {"src", transfer.src},
                {"dst", transfer.dst},
                {"total_bytes", transfer.total_bytes},
                {"tag", transfer.tag},
                {"control_marker", transfer.control_marker},
                {"network_done", transfer.network_done},
                {"hbm_done", transfer.hbm_done},
                {"flow_bound", transfer.flow_bound},
            });
        }
    }
    // Slot-occupancy view per buffer domain: who owns every occupied slot at
    // capture time (adjudicates cross-segment displacement decisions).
    for (const auto& domain_entry : buffer_domains_) {
        for (const nlohmann::json& slot_row :
             slot_state_json(domain_entry.first)) {
            nlohmann::json row = slot_row;
            row["row_kind"] = "slot_state";
            rows.push_back(std::move(row));
        }
    }
    // Iterate ALL descriptors (not just the active index): after an
    // administrative close the cursors are marked exhausted but their
    // dependency blockers remain computable and are exactly the
    // adjudication facts we need.
    for (const auto& owner_holder : descriptors_) {
        if (owner_holder == nullptr) continue;
        const RuntimeDescriptor& owner = *owner_holder;
        // Settle-gate summary (one row per descriptor): adjudicates WHY a
        // descriptor that stopped making progress never reached its settle
        // transition -- which of the settle gates (service watch, physical
        // legs outstanding, consume deliveries pending, cursor exhaustion)
        // was still holding it.
        size_t exhausted_cursors = 0;
        uint64_t in_flight_total = 0;
        for (const RunCursor& cursor : owner.runs) {
            if (cursor.run == nullptr) continue;
            if (cursor.exhausted) ++exhausted_cursors;
            in_flight_total += cursor.in_flight_count;
        }
        rows.push_back({
            {"row_kind", "settle_gate"},
            {"segment_id", owner.descriptor->identity().segment_id},
            {"service_done", owner.service_done},
            {"settle_done", owner.settle_done},
            {"request_closed", owner.request_closed},
            {"retired", owner.retired},
            {"outstanding_count", owner.outstanding_count},
            {"consume_pending_count", owner.consume_pending_count},
            {"administratively_retired_consume_obligations",
             owner.administratively_retired_consume_obligations},
            {"runs", owner.runs.size()},
            {"exhausted_cursors", exhausted_cursors},
            {"in_flight_total", in_flight_total},
        });
        for (size_t run_index = 0; run_index < owner.runs.size();
             ++run_index) {
            const RunCursor& cursor = owner.runs[run_index];
            if (cursor.run == nullptr) continue;
            uint64_t current = 0;
            if (cursor.exhausted) {
                // Exhausted (administratively closed) cursors have no
                // current position; report the run's remaining obligation
                // count as a summary row instead.
                rows.push_back({
                    {"segment_id",
                     owner.descriptor->identity().segment_id},
                    {"rank", cursor.run->rank},
                    {"run_index", run_index},
                    {"store_id", nullptr},
                    {"endpoint",
                     leg_name(cursor.run->legs[cursor.leg_index].endpoint)},
                    {"tile_offset", cursor.tile_offset},
                    {"descriptor_retired", owner.retired},
                    {"cursor_exhausted", true},
                    {"remaining_endpoints",
                     cursor.run->legs.size() *
                         cursor.run->shard_ordinals.size()},
                    {"in_flight", cursor.in_flight_count}});
                continue;
            }
            if (!current_store_id(cursor, &current)) continue;
            nlohmann::json row{
                {"segment_id", owner.descriptor->identity().segment_id},
                {"rank", cursor.run->rank},
                {"run_index", run_index},
                {"store_id", current},
                {"endpoint",
                 leg_name(cursor.run->legs[cursor.leg_index]
                                       .endpoint)},
                {"tile_offset", cursor.tile_offset},
                {"descriptor_retired", owner.retired},
                {"cursor_exhausted", cursor.exhausted},
                {"deps_ready",
                 endpoint_dependencies_ready(owner, cursor)},
                {"protocol_ready", protocol_ready(owner, cursor)}};
            if (!row["deps_ready"].get<bool>()) {
                const uint64_t tile_ordinal =
                    cursor.run->first_tile + cursor.tile_offset;
                for (const ResolvedDependencyRun& resolved :
                     owner.dependency_runs) {
                    const auto& dependency = *resolved.run;
                    if (dependency.rank != cursor.run->rank ||
                        dependency.group_ordinal !=
                            cursor.run->group_ordinal ||
                        tile_ordinal < dependency.first_tile ||
                        tile_ordinal - dependency.first_tile >=
                            dependency.tile_count) {
                        continue;
                    }
                    const auto shard = std::find(
                        dependency.shard_ordinals.begin(),
                        dependency.shard_ordinals.end(),
                        cursor.run->shard_ordinals[cursor.shard_index]);
                    if (shard == dependency.shard_ordinals.end()) continue;
                    const uint64_t local_shard = static_cast<uint64_t>(
                        std::distance(dependency.shard_ordinals.begin(),
                                      shard));
                    const uint64_t dep_store = add_checked(
                        resolved.first_store_id,
                        add_checked(
                            mul_checked(
                                tile_ordinal - dependency.first_tile,
                                dependency.dependency_tile_stride,
                                "diag tile"),
                            mul_checked(
                                local_shard,
                                dependency.dependency_shard_stride,
                                "diag shard"),
                            "diag off"),
                        "diag store");
                    const int dep_rank = dependency.rank;
                    row["first_unresolved_dep"] = {
                        {"store_id", dep_store},
                        {"rank", dep_rank},
                        {"name",
                         (dep_rank >= 0 &&
                          static_cast<size_t>(dep_rank) <
                              graph_sources_.size() &&
                          graph_sources_[dep_rank]->lookup_ptr(dep_store) !=
                              nullptr)
                             ? graph_sources_[dep_rank]
                                   ->lookup_ptr(dep_store)
                                   ->name
                             : std::string()},
                        {"kind", dependency.dependency_kind ==
                                     PipelineEndpointDependencyKind::
                                         OrdinaryNodeTerminal
                                     ? "ordinary"
                                     : "virtual"},
                        {"virtual_id",
                         dep_rank >= 0 &&
                                 static_cast<size_t>(dep_rank) <
                                     graph_sources_.size()
                             ? graph_sources_[dep_rank]
                                   ->store()
                                   .is_virtual_id(dep_store)
                             : false},
                        {"finished",
                         dep_rank >= 0 &&
                                 static_cast<size_t>(dep_rank) <
                                     graph_sources_.size() &&
                                 graph_sources_[dep_rank]
                                     ->is_finished_or_erased(dep_store)}};
                    break;
                }
            }
            rows.push_back(std::move(row));
        }
    }
    return rows;
}

nlohmann::json RemoteReadExecutor::authoritative_snapshot(
    const uint64_t now_tick) const {
    ++producer_metrics_.authoritative_snapshot_calls;
    const DiagnosticRowCaptureScope diagnostic_scope(
        diagnostic_row_capture_active_);
    nlohmann::json descriptors = nlohmann::json::array();
    nlohmann::json cursors = nlohmann::json::array();
    nlohmann::json obligations = nlohmann::json::array();
    std::vector<const RuntimeDescriptor*> active_owners;
    active_owners.reserve(active_descriptors_.size());
    for (const RuntimeDescriptor* owner : active_descriptors_) {
        if (owner != nullptr && !owner->retired) active_owners.push_back(owner);
    }
    std::sort(active_owners.begin(), active_owners.end(),
              [](const RuntimeDescriptor* left,
                 const RuntimeDescriptor* right) {
        return left->descriptor->identity().segment_id <
               right->descriptor->identity().segment_id;
    });
    std::set<std::string> obligation_domains;
    std::set<std::string> live_domains;
    for (const RuntimeDescriptor* owner_pointer : active_owners) {
        const RuntimeDescriptor& owner = *owner_pointer;
        const PipelineDescriptor& descriptor = *owner.descriptor;
        const std::string& domain_id = descriptor.identity().buffer_domain_id;
        live_domains.insert(domain_id);
        nlohmann::json descriptor_entry;
        descriptor_entry["identity"] = identity_json(descriptor.identity());
        descriptor_entry["descriptor"] = descriptor.wire();
        descriptor_entry["commit_tick"] = owner.commit_tick;
        descriptor_entry["cursor_version"] = owner.cursor_version;
        descriptor_entry["service_done"] = owner.service_done;
        descriptor_entry["settle_done"] = owner.settle_done;
        descriptor_entry["retired"] = owner.retired;
        descriptors.push_back(std::move(descriptor_entry));

        nlohmann::json cursor_entry;
        cursor_entry["segment_id"] = descriptor.identity().segment_id;
        cursor_entry["buffer_domain_id"] = descriptor.identity().buffer_domain_id;
        cursor_entry["generation"] = descriptor.identity().generation;
        cursor_entry["cursor_version"] = owner.cursor_version;
        cursor_entry["as_of_tick"] = now_tick;
        cursor_entry["runs"] = nlohmann::json::array();
        for (size_t run_index = 0; run_index < owner.runs.size(); ++run_index) {
            const RunCursor& cursor = owner.runs[run_index];
            uint64_t next_store_id = 0;
            const bool has_next = current_store_id(cursor, &next_store_id);
            uint64_t total_count = mul_checked(
                cursor.run->tile_count,
                mul_checked(static_cast<uint64_t>(cursor.run->shard_ordinals.size()),
                            static_cast<uint64_t>(cursor.run->legs.size()),
                            "run endpoint count"),
                "run endpoint count");
            const uint64_t completed_prefix = cursor.completed_prefix;
            uint64_t out_of_order_count = 0;
            nlohmann::json out_of_order = nlohmann::json::array();
            for (const auto& interval : cursor.completed_out_of_order) {
                if (interval.first < completed_prefix ||
                    interval.second <= interval.first ||
                    interval.second > total_count) {
                    fail_closed("cursor has an invalid out-of-order completion interval");
                }
                out_of_order_count = add_checked(
                    out_of_order_count, interval.second - interval.first,
                    "cursor out-of-order completion count");
                out_of_order.push_back({
                    {"first_ordinal", interval.first},
                    {"end_ordinal_exclusive", interval.second},
                });
            }
            if (completed_prefix > total_count ||
                out_of_order_count > total_count - completed_prefix) {
                fail_closed("cursor completed count exceeds its endpoint run");
            }
            const uint64_t completed_count =
                completed_prefix + out_of_order_count;
            if (completed_count != cursor.completed_count) {
                fail_closed(
                    "snapshot cached completed count disagrees with intervals");
            }
            if (cursor.cancelled_endpoint_count >
                    total_count - completed_count ||
                (!owner.request_closed &&
                 cursor.cancelled_endpoint_count != 0)) {
                fail_closed("cursor cancelled endpoint count is inconsistent");
            }
            const uint64_t remaining_count = total_count - completed_count -
                cursor.cancelled_endpoint_count;
            uint64_t active_count = 0;
            nlohmann::json active_endpoints = nlohmann::json::array();
            const uint64_t endpoints_per_tile = mul_checked(
                static_cast<uint64_t>(cursor.run->shard_ordinals.size()),
                static_cast<uint64_t>(cursor.run->legs.size()),
                "active endpoint position width");
            const PipelineIdentity& owner_identity = descriptor.identity();
            for (const auto& active : cursor.active_store_ids) {
                const uint64_t ordinal = active.first;
                if (ordinal >= total_count || endpoints_per_tile == 0) {
                    fail_closed("active endpoint ordinal is outside its run");
                }
                const uint64_t tile_offset = ordinal / endpoints_per_tile;
                const uint64_t within_tile = ordinal % endpoints_per_tile;
                const size_t shard_index = static_cast<size_t>(
                    within_tile / cursor.run->legs.size());
                const size_t leg_index = static_cast<size_t>(
                    within_tile % cursor.run->legs.size());
                const uint64_t tile_ordinal =
                    cursor.run->first_tile + tile_offset;
                const uint64_t shard_ordinal =
                    cursor.run->shard_ordinals[shard_index];
                TileTransportAdapter::ActiveTransfer transfer;
                if (transport_ == nullptr || !transport_->find_active_by_node(
                        cursor.run->rank, active.second, &transfer)) {
                    fail_closed("active cursor endpoint lacks its transport identity");
                }
                active_endpoints.push_back({
                    {"store_id", active.second},
                    {"tile_ordinal", tile_ordinal},
                    {"shard_ordinal", shard_ordinal},
                    {"endpoint", leg_name(
                         cursor.run->legs[leg_index].endpoint)},
                    {"transfer_id", transfer.transfer_id},
                    {"flow_id", transfer.flow_bound
                         ? nlohmann::json(transfer.flow_id)
                         : nlohmann::json(nullptr)},
                    {"network_done", transfer.network_done},
                    {"hbm_done", transfer.hbm_done},
                    {"owner", {
                        {"segment_id", owner_identity.segment_id},
                        {"pipeline_generation", owner_identity.generation},
                        {"stage_epoch", owner_identity.stage_epoch},
                        {"staging_allocation_id",
                         owner_identity.staging_allocation_id},
                        {"group_ordinal", cursor.run->group_ordinal},
                        {"tile_ordinal", tile_ordinal},
                    }},
                });
                ++active_count;
            }
            if (active_count != cursor.active_store_ids.size() ||
                active_count != cursor.in_flight_count) {
                fail_closed("cursor active endpoint count is inconsistent");
            }
            if (owner.request_closed &&
                (has_next || remaining_count != active_count)) {
                fail_closed("closed cursor remaining count differs from active work");
            }
            nlohmann::json run_entry{
                {"rank", cursor.run->rank},
                {"group_ordinal", cursor.run->group_ordinal},
                {"first_tile", cursor.run->first_tile},
                {"tile_count", cursor.run->tile_count},
                {"completed_endpoint_count", completed_prefix},
                {"cancelled_endpoint_count", cursor.cancelled_endpoint_count},
                {"remaining_endpoint_count", remaining_count},
                {"in_flight", cursor.in_flight_count != 0},
                {"next_store_id", has_next ? nlohmann::json(next_store_id)
                                             : nlohmann::json(nullptr)},
                {"tile_offset", cursor.tile_offset},
                {"shard_index", cursor.shard_index},
                {"leg_index", cursor.leg_index},
                {"initial_parent_store_id",
                 cursor.has_initial_parent
                     ? nlohmann::json(cursor.initial_parent_store_id)
                     : nlohmann::json(nullptr)},
                {"completed_out_of_order", std::move(out_of_order)},
                {"active_endpoints", std::move(active_endpoints)},
            };
            cursor_entry["runs"].push_back(run_entry);
        }
        if (obligation_domains.insert(domain_id).second) {
            const nlohmann::json domain_obligations =
                obligation_state_json_for_domain(domain_id);
            obligations.insert(obligations.end(), domain_obligations.begin(),
                               domain_obligations.end());
        }
        cursors.push_back(std::move(cursor_entry));
    }
    const auto segment_order = [](const nlohmann::json& left,
                                  const nlohmann::json& right) {
        return left.at("identity").at("segment_id").get<std::string>() <
               right.at("identity").at("segment_id").get<std::string>();
    };
    std::sort(descriptors.begin(), descriptors.end(), segment_order);
    std::sort(cursors.begin(), cursors.end(),
              [](const nlohmann::json& left, const nlohmann::json& right) {
        return left.at("segment_id").get<std::string>() <
               right.at("segment_id").get<std::string>();
    });
    std::sort(obligations.begin(), obligations.end(),
              [](const nlohmann::json& left,
                 const nlohmann::json& right) {
        return std::make_tuple(left.at("segment_id").get<std::string>(),
                               left.at("rank").get<int>(),
                               left.at("first_store_id").get<uint64_t>(),
                               left.at("obligation_id").get<std::string>()) <
               std::make_tuple(right.at("segment_id").get<std::string>(),
                               right.at("rank").get<int>(),
                               right.at("first_store_id").get<uint64_t>(),
                               right.at("obligation_id").get<std::string>());
    });
    nlohmann::json slots = nlohmann::json::array();
    nlohmann::json credits = nlohmann::json::array();
    for (const std::string& domain : live_domains) {
        const nlohmann::json domain_slots = slot_state_json(domain);
        const nlohmann::json domain_credits = credit_state_json(domain);
        slots.insert(slots.end(), domain_slots.begin(), domain_slots.end());
        credits.insert(credits.end(), domain_credits.begin(), domain_credits.end());
    }
    std::sort(slots.begin(), slots.end(),
              [](const nlohmann::json& left,
                 const nlohmann::json& right) {
        const auto shard_key = [](const nlohmann::json& row) {
            return row.at("shard_ordinal").is_null()
                ? int64_t{-1}
                : static_cast<int64_t>(row.at("shard_ordinal").get<uint64_t>());
        };
        return std::make_tuple(left.at("buffer_domain_id").get<std::string>(),
                               left.at("role").get<std::string>(),
                               left.at("rank").get<int>(),
                               shard_key(left),
                               left.at("slot_index").get<size_t>()) <
               std::make_tuple(right.at("buffer_domain_id").get<std::string>(),
                               right.at("role").get<std::string>(),
                               right.at("rank").get<int>(),
                               shard_key(right),
                               right.at("slot_index").get<size_t>());
    });
    std::sort(credits.begin(), credits.end(),
              [](const nlohmann::json& left,
                 const nlohmann::json& right) {
        return std::make_tuple(left.at("buffer_domain_id").get<std::string>(),
                               left.at("source_rank").get<int>(),
                               left.at("shard_ordinal").get<uint64_t>()) <
               std::make_tuple(right.at("buffer_domain_id").get<std::string>(),
                               right.at("source_rank").get<int>(),
                               right.at("shard_ordinal").get<uint64_t>());
    });
    return nlohmann::json{
        {"as_of_tick", now_tick},
        {"descriptor_commits", std::move(descriptors)},
        {"authoritative_cursors", std::move(cursors)},
        {"obligations", std::move(obligations)},
        {"slots", std::move(slots)},
        {"credits", std::move(credits)},
    };
}

}  // namespace ExecutionDriven
}  // namespace AstraSim
