/******************************************************************************
Real two-rank remote-read executor fixture.

This is deliberately an online GraphBatch fixture rather than a transport
mock: it commits schema-2 descriptor data through GraphBatchCommitter, then
drains real Sys send/receive, FluidScheduler and local-HBM jobs. It records
the logical send bindings, physical endpoint issue/completion order, original
marker cost, service-watch fire and later settle frontier.

The program/wire builder lives in remote_read_executor_program.hh (shared
with the cold execution-fixture case generator); this file keeps the run
loop and the physical assertions.
******************************************************************************/

#include "remote_read_executor_program.hh"
#include "astra-sim/workload/execution_driven/PipelineKernel.hh"

#include <algorithm>
#include <array>
#include <csignal>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <map>
#include <memory>
#include <new>
#include <set>
#include <stdexcept>
#include <string>
#include <sys/resource.h>
#include <sys/wait.h>
#include <unistd.h>
#include <utility>
#include <vector>

namespace AstraSim {
namespace ExecutionDriven {

struct WorkLedgerV2PublisherTestAccess {
    static WorkLedgerV2Publisher* allocation_probe_publisher;
    static bool saw_lineage_during_allocation_failure;

    static size_t lineage_size(const WorkLedgerV2Publisher& publisher) {
        return publisher.target_delta_predecessor_seq_.size();
    }

    static uint64_t lineage_sequence(const WorkLedgerV2Publisher& publisher,
                                     const std::string& domain,
                                     const int rank,
                                     const size_t slot_index) {
        const WorkLedgerV2Publisher::TargetSlotKey key{
            domain, "target", rank, std::nullopt, slot_index};
        const auto found = publisher.target_delta_predecessor_seq_.find(key);
        return found == publisher.target_delta_predecessor_seq_.end()
            ? 0 : found->second;
    }

    static void note_allocation_failure() {
        if (allocation_probe_publisher != nullptr &&
            !allocation_probe_publisher->target_delta_predecessor_seq_.empty()) {
            saw_lineage_during_allocation_failure = true;
        }
    }
};

WorkLedgerV2Publisher*
    WorkLedgerV2PublisherTestAccess::allocation_probe_publisher = nullptr;
bool WorkLedgerV2PublisherTestAccess::saw_lineage_during_allocation_failure =
    false;

}  // namespace ExecutionDriven
}  // namespace AstraSim

namespace {

// Inject only into the calling test thread; async logging must keep allocating.
thread_local int64_t g_remote_test_fail_new_after = -1;

void* remote_test_allocate(const std::size_t size) {
    const int64_t remaining = g_remote_test_fail_new_after;
    if (remaining >= 0) {
        if (remaining == 0) {
            g_remote_test_fail_new_after = -1;
            AstraSim::ExecutionDriven::WorkLedgerV2PublisherTestAccess::
                note_allocation_failure();
            throw std::bad_alloc();
        }
        g_remote_test_fail_new_after = remaining - 1;
    }
    void* const memory = std::malloc(size == 0 ? 1 : size);
    if (memory == nullptr) throw std::bad_alloc();
    return memory;
}

}  // namespace

void* operator new(const std::size_t size) {
    return remote_test_allocate(size);
}

void* operator new[](const std::size_t size) {
    return remote_test_allocate(size);
}

void operator delete(void* memory) noexcept { std::free(memory); }
void operator delete[](void* memory) noexcept { std::free(memory); }
void operator delete(void* memory, const std::size_t) noexcept {
    std::free(memory);
}
void operator delete[](void* memory, const std::size_t) noexcept {
    std::free(memory);
}

void set_remote_test_alloc_fail_after(const int64_t countdown) {
    g_remote_test_fail_new_after = countdown;
}

namespace AstraSim::ExecutionDriven {

/// Exercises the production interval-index builder and lookup code with
/// immutable synthetic run programs. No graph commit, EventQueue advancement,
/// NoC/Die request, or physical endpoint loop is involved.
class RemoteReadExecutorTestProbe final {
  public:
    static void run(RemoteReadExecutor& executor,
                    const uint64_t starting_tick,
                    const BuiltFixture& fixture) {
        run_ordinary_terminal_candidate_membership();
        run_dependency_interval_index(executor, starting_tick);
        const std::vector<size_t> scales{8, 32, 128};
        for (const size_t run_count : scales) {
            clear(executor);
            std::vector<std::unique_ptr<RemoteReadExecutor::RuntimeDescriptor>>
                owners;
            std::vector<std::unique_ptr<PipelineEndpointRun>> runs;
            owners.reserve(run_count);
            runs.reserve(run_count);
            std::vector<RemoteReadExecutor::RuntimeDescriptor*> batch;
            batch.reserve(run_count);
            for (size_t index = 0; index < run_count; ++index) {
                auto owner =
                    std::make_unique<RemoteReadExecutor::RuntimeDescriptor>();
                auto run = std::make_unique<PipelineEndpointRun>();
                run->rank = 0;
                run->tile_count = 8;
                run->tile_stride = 4;
                run->shard_stride = 32;
                run->shard_ordinals = {0};
                run->legs.emplace_back();
                RemoteReadExecutor::RunCursor cursor;
                cursor.run = run.get();
                cursor.first_store_id = 1 + index * 64;
                owner->runs.push_back(cursor);
                batch.push_back(owner.get());
                owners.push_back(std::move(owner));
                runs.push_back(std::move(run));
            }
            executor.index_batch_descriptors(batch);
            owners.front()->retired = true;
            verify_queries(executor, owners, run_count, "runs");
            print_scale(executor, run_count, "runs");
            expect(Sys::boostedTick() == starting_tick,
                   "pure run-scale index checks do not advance simulation time");
        }

        // Keep T=4 scalar lookups constant while H, the accepted descriptor /
        // batch history axis, grows. Each index insertion is a separate
        // immutable batch block; adding it never revisits prior blocks.
        clear(executor);
        std::vector<std::unique_ptr<RemoteReadExecutor::RuntimeDescriptor>>
            owners;
        std::vector<std::unique_ptr<PipelineEndpointRun>> runs;
        owners.reserve(128);
        runs.reserve(128);
        size_t inserted = 0;
        for (const size_t history_count : scales) {
            for (; inserted < history_count; ++inserted) {
                auto owner =
                    std::make_unique<RemoteReadExecutor::RuntimeDescriptor>();
                auto run = std::make_unique<PipelineEndpointRun>();
                run->rank = 0;
                run->tile_count = 8;
                run->tile_stride = 4;
                run->shard_stride = 32;
                run->shard_ordinals = {0};
                run->legs.emplace_back();
                RemoteReadExecutor::RunCursor cursor;
                cursor.run = run.get();
                cursor.first_store_id = 1 + inserted * 64;
                owner->runs.push_back(cursor);
                executor.index_batch_descriptors({owner.get()});
                owners.push_back(std::move(owner));
                runs.push_back(std::move(run));
            }
            owners.front()->retired = true;
            verify_queries(executor, owners, history_count, "history");
            print_scale(executor, history_count, "history");
            expect(Sys::boostedTick() == starting_tick,
                   "pure history-scale index checks do not advance simulation time");
        }
        clear(executor);
        verify_domain_provenance(executor, starting_tick);
        run_terminal_watcher_index(executor, starting_tick);
        run_completed_count_scaling(executor, fixture);
        run_obligation_patch_coalescing(executor);
    }

    static void run_anchor_coordinate_prefilter(
        RemoteReadExecutor& executor,
        const std::vector<std::shared_ptr<NodeStoreGraphSource>>& graph_sources,
        const std::vector<Workload*>& workloads,
        const std::shared_ptr<EventQueue>& event_queue,
        const std::string& decode_segment) {
        struct AnchorCandidate {
            RemoteReadExecutor::RuntimeDescriptor* owner = nullptr;
            size_t run_index = 0;
            uint64_t store_id = 0;
            uint64_t tile_ordinal = 0;
            size_t shard_index = 0;
            size_t leg_index = 0;
            NodeView node;
            bool current = false;
        };

        std::vector<AnchorCandidate> anchors;
        for (size_t rank = 0;
             rank < executor.physical_anchors_by_rank_.size(); ++rank) {
            if (rank >= graph_sources.size() || graph_sources[rank] == nullptr) {
                continue;
            }
            const NodeStore& store = graph_sources[rank]->store();
            for (const auto& entry : executor.physical_anchors_by_rank_[rank]) {
                if (!store.is_free_id(entry.first)) continue;
                const auto& binding = entry.second;
                if (binding.owner == nullptr ||
                    binding.owner->retired ||
                    binding.owner->descriptor == nullptr ||
                    binding.run_index >= binding.owner->runs.size() ||
                    binding.owner->descriptor->identity().segment_id !=
                        decode_segment) {
                    continue;
                }
                const RemoteReadExecutor::RunCursor& cursor =
                    binding.owner->runs[binding.run_index];
                if (cursor.run == nullptr || cursor.exhausted ||
                    cursor.run->tile_count < 2 ||
                    cursor.tile_offset >= cursor.run->tile_count ||
                    cursor.shard_index >= cursor.run->shard_ordinals.size() ||
                    cursor.leg_index >= cursor.run->legs.size() ||
                    binding.tile_ordinal < cursor.run->first_tile ||
                    binding.tile_ordinal - cursor.run->first_tile >=
                        cursor.run->tile_count ||
                    binding.shard_index >=
                        cursor.run->shard_ordinals.size() ||
                    binding.leg_index >= cursor.run->legs.size()) {
                    continue;
                }
                const NodeView* const node =
                    graph_sources[rank]->lookup_ptr(entry.first);
                if (node == nullptr ||
                    node->compute.exec_role !=
                        ComputeExecRole::RemoteReadPipelineAnchor) {
                    continue;
                }
                const bool current =
                    cursor.tile_offset ==
                        binding.tile_ordinal - cursor.run->first_tile &&
                    cursor.shard_index == binding.shard_index &&
                    cursor.leg_index == binding.leg_index;
                anchors.push_back(AnchorCandidate{
                    binding.owner, binding.run_index, entry.first,
                    binding.tile_ordinal, binding.shard_index,
                    binding.leg_index, *node, current});
            }
        }

        const AnchorCandidate* current = nullptr;
        const AnchorCandidate* future = nullptr;
        for (const AnchorCandidate& candidate : anchors) {
            if (!candidate.current) continue;
            for (const AnchorCandidate& later : anchors) {
                if (later.current || later.owner != candidate.owner ||
                    later.run_index != candidate.run_index ||
                    later.tile_ordinal == candidate.tile_ordinal) {
                    continue;
                }
                current = &candidate;
                future = &later;
                break;
            }
            if (current != nullptr) break;
        }
        expect(current != nullptr && future != nullptr,
               "wide D1 fixture has ready current/future anchors in one run");
        expect(current->owner->runs[current->run_index]
                       .run->legs[current->leg_index].endpoint ==
                   PipelineEndpointLeg::DataRecv &&
                   future->owner->runs[future->run_index]
                       .run->legs[future->leg_index].endpoint ==
                   PipelineEndpointLeg::DataRecv,
               "wide D1 probe uses real target COMM receive anchors");
        expect(current->owner->runs[current->run_index].tile_offset ==
                   current->tile_ordinal -
                       current->owner->runs[current->run_index].run->first_tile,
               "wide D1 current anchor agrees with the committed cursor");
        expect(current->owner->runs[current->run_index].tile_offset !=
                   future->tile_ordinal -
                       future->owner->runs[future->run_index].run->first_tile,
               "wide D1 future anchor disagrees with the unchanged cursor");

        const uint64_t tick_before = Sys::boostedTick();
        const uint64_t cursors_before = executor.active_cursor_count();
        const uint64_t issues_before =
            executor.producer_metrics().actual_issue_calls;
        const bool deferred_before = event_queue->has_deferred_work();
        NodeView resource_view;
        expect(!executor.resource_view_for_anchor(current->node,
                                                   &resource_view),
               "current D1 DataRecv anchor is blocked on its unissued ACK parent");
        expect(!executor.resource_view_for_anchor(future->node,
                                                   &resource_view),
               "future D1 physical anchor returns false at its static coordinates");
        expect(Sys::boostedTick() == tick_before &&
                   executor.active_cursor_count() == cursors_before &&
                   executor.producer_metrics().actual_issue_calls == issues_before &&
                   event_queue->has_deferred_work() == deferred_before,
               "direct anchor probes do not issue or advance the event queue");

        const auto before = executor.producer_metrics();
        constexpr uint64_t kProbePasses = 3;
        for (uint64_t pass = 0; pass < kProbePasses; ++pass) {
            expect(!executor.resource_view_for_anchor(current->node,
                                                       &resource_view),
                   "current D1 anchor remains dependency-blocked across probes");
            expect(!executor.resource_view_for_anchor(future->node,
                                                       &resource_view),
                   "future D1 anchor remains a coordinate rejection");
            static_cast<void>(executor.ready_virtual_endpoints(
                current->node.rank));
        }
        const auto& after = executor.producer_metrics();
        const uint64_t query_delta =
            after.anchor_resource_queries - before.anchor_resource_queries;
        const uint64_t preflight_delta =
            after.anchor_dynamic_preflights -
                before.anchor_dynamic_preflights;
        const uint64_t reject_delta =
            after.anchor_coordinate_rejections -
                before.anchor_coordinate_rejections;
        const uint64_t physical_skip_delta =
            after.virtual_physical_cursor_skips -
                before.virtual_physical_cursor_skips;
        expect(query_delta == 2 * kProbePasses &&
                   preflight_delta == kProbePasses &&
                   reject_delta == kProbePasses &&
                   physical_skip_delta >= kProbePasses,
               "wide D1 counters separate one current preflight from one future rejection per pass");
        expect(Sys::boostedTick() == tick_before &&
                   executor.active_cursor_count() == cursors_before &&
                   executor.producer_metrics().actual_issue_calls == issues_before &&
                   event_queue->has_deferred_work() == deferred_before,
               "three direct preflight rounds preserve tick, cursor and callback state");
        std::printf(
            "[anchor-coordinate-prefilter] passes=%llu queries=%llu "
            "current_preflights=%llu coordinate_rejections=%llu "
            "virtual_physical_skips=%llu event_tick_unchanged=true\n",
            static_cast<unsigned long long>(kProbePasses),
            static_cast<unsigned long long>(query_delta),
            static_cast<unsigned long long>(preflight_delta),
            static_cast<unsigned long long>(reject_delta),
            static_cast<unsigned long long>(physical_skip_delta));
    }

    static void run_target_slot_delta_cost_probe(
        RemoteReadExecutor& executor, const BuiltFixture& fixture,
        const std::vector<std::shared_ptr<NodeStoreGraphSource>>& graph_sources,
        const std::shared_ptr<EventQueue>& event_queue,
        const uint64_t starting_tick) {
        expect(graph_sources.size() > 6 && graph_sources[6] != nullptr &&
                   event_queue != nullptr,
               "target-slot probe uses the existing rank-6 owner and event queue");
        run_publisher_failure_atomicity();
        const uint64_t queue_tick_before = event_queue->get_current_time();
        const uint64_t queue_proceeds_before =
            EventQueue::queue_wall_stats().proceed.calls;
        uint64_t id_namespace_offset = 0;
        for (const size_t shard_count : {size_t{8}, size_t{32}, size_t{128}}) {
            const nlohmann::json wire = target_slot_delta_descriptor_wire(
                fixture.programs.at(0).wire, shard_count,
                "direct-S" + std::to_string(shard_count),
                id_namespace_offset,
                /*inactive_tail=*/false, /*zero_stride_shared=*/false,
                /*split_shared_runs=*/false);
            const auto descriptor = PipelineDescriptor::parse(wire);
            WorkLedgerV2Publisher publisher(
                "target-slot-direct-S" + std::to_string(shard_count));
            uint64_t delivery_sequence = 0;
            run_direct_target_slot_case(executor, graph_sources, event_queue,
                                        descriptor, publisher,
                                        &delivery_sequence, shard_count,
                                        /*split_shared_runs=*/false,
                                        starting_tick);
            id_namespace_offset += 100000000;
        }

        // Two immutable completion runs share one terminal ID and split the
        // active target shards. Both watcher records point directly into the
        // validated descriptor vector; sequential real helper calls model
        // their same-callback fanout and expose the old intermediate row.
        constexpr size_t kSharedShardCount = 8;
        const auto shared_wire = target_slot_delta_descriptor_wire(
            fixture.programs.at(0).wire, kSharedShardCount,
            "zero-stride-split-runs", id_namespace_offset,
            /*inactive_tail=*/false,
            /*zero_stride_shared=*/true, /*split_shared_runs=*/true);
        const auto shared_descriptor = PipelineDescriptor::parse(shared_wire);
        WorkLedgerV2Publisher shared_publisher("target-slot-shared-zero-stride");
        uint64_t shared_delivery_sequence = 0;
        run_direct_target_slot_case(executor, graph_sources, event_queue,
                                    shared_descriptor, shared_publisher,
                                    &shared_delivery_sequence, kSharedShardCount,
                                    /*split_shared_runs=*/true,
                                    starting_tick);
        id_namespace_offset += 100000000;
        run_target_slot_reinitialize_versions(executor, 6);

        // A zero-byte shard is an inactive group member. The parser refuses
        // to let a consume-completion run cover it, so a validated descriptor
        // cannot create an occupied target row with this inactive addition.
        const auto inactive_wire = target_slot_delta_descriptor_wire(
            fixture.programs.at(0).wire, size_t{3},
            "inactive-zero-stride", id_namespace_offset,
            /*inactive_tail=*/true,
            /*zero_stride_shared=*/true, /*split_shared_runs=*/false);
        bool rejected_inactive_mapping = false;
        std::string inactive_rejection;
        try {
            static_cast<void>(PipelineDescriptor::parse(inactive_wire));
        } catch (const std::invalid_argument& error) {
            inactive_rejection = error.what();
            rejected_inactive_mapping =
                inactive_rejection.find(
                    "completion run extends beyond the shard's active tile count") !=
                std::string::npos;
        }
        expect(rejected_inactive_mapping,
               "validated descriptor rejects zero-stride consume mapping of an inactive tail shard");
        std::printf("[target-slot-inactive-boundary] validated_descriptor_rejects="
                    "true reason=%s occupied_row=unreachable\n",
                    inactive_rejection.c_str());

        expect(Sys::boostedTick() == starting_tick &&
                   event_queue->get_current_time() == queue_tick_before &&
                   EventQueue::queue_wall_stats().proceed.calls ==
                       queue_proceeds_before &&
                   executor.producer_metrics().actual_issue_calls == 0 &&
                   executor.active_cursor_count() == 0,
               "target-slot direct callbacks preserve tick and do not issue transport work");
        std::printf(
            "[target-slot-delta-cost] sizes=8,32,128 same_owner_rank=6 "
            "same_local_tile=1 zero_stride_same_terminal_runs=2 "
            "event_queue_advances=0 noc_die_issues=0 epoch_and_generation="
            "preserved\n");
    }

  private:
    static void run_target_slot_reinitialize_versions(
        RemoteReadExecutor& executor, const int target_rank) {
        constexpr const char* kDomain = "target-slot-version-reinitialize-v4";
        const std::string domain_id = kDomain;
        executor.set_ledger_publishers(
            LedgerNodeTransitionPublisher{}, [](nlohmann::json) {},
            RemoteReadExecutor::DescriptorCommitPublisher{});
        expect(executor.buffer_domains_.find(domain_id) ==
                       executor.buffer_domains_.end() &&
                   executor.buffer_domain_target_ranks_.find(domain_id) ==
                       executor.buffer_domain_target_ranks_.end(),
               "reinitialize probe starts without live rows for its domain");

        RemoteReadExecutor::BufferDomainState retired_domain;
        retired_domain.target_slots[0].generation = 7;
        retired_domain.target_slots[1].generation = 9;
        auto& retired_source = retired_domain.source_slots[{0, 0}];
        retired_source[0].generation = 3;
        retired_source[1].generation = 5;
        retired_domain.source_slot_released_through[{0, 0}] = {11, 13};
        executor.retain_buffer_domain_provenance(
            domain_id, retired_domain, target_rank);

        const std::array<std::string, 2> target_keys{{
            executor.slot_resource_key(domain_id, "target", target_rank, 0, 0),
            executor.slot_resource_key(domain_id, "target", target_rank, 0, 1),
        }};
        const std::array<std::string, 2> source_keys{{
            executor.slot_resource_key(domain_id, "source", 0, 0, 0),
            executor.slot_resource_key(domain_id, "source", 0, 0, 1),
        }};
        const std::string credit_key =
            executor.credit_resource_key(domain_id, 0, 0);
        executor.resource_versions_[{"pipeline_slot", target_keys[0]}] = 17;
        executor.resource_versions_[{"pipeline_slot", target_keys[1]}] = 19;
        executor.resource_versions_[{"pipeline_slot", source_keys[0]}] = 23;
        executor.resource_versions_[{"pipeline_slot", source_keys[1]}] = 29;
        executor.resource_versions_[{"pipeline_credit", credit_key}] = 31;

        std::array<uint64_t, 2> target_before_versions{};
        for (size_t index = 0; index < target_keys.size(); ++index) {
            target_before_versions[index] = executor.resource_mutation_before(
                "pipeline_slot", target_keys[index], target_rank);
        }
        auto insertion = executor.buffer_domains_.try_emplace(domain_id);
        RemoteReadExecutor::BufferDomainState& domain = insertion.first->second;
        executor.restore_buffer_domain_provenance(
            domain_id, domain, target_rank);
        executor.buffer_domain_target_ranks_[domain_id] = target_rank;
        for (size_t index = 0; index < target_keys.size(); ++index) {
            executor.resource_mutation_after(
                "pipeline_slot", target_keys[index],
                target_before_versions[index]);
        }

        std::array<uint64_t, 2> source_before_versions{};
        for (size_t index = 0; index < source_keys.size(); ++index) {
            source_before_versions[index] = executor.resource_mutation_before(
                "pipeline_slot", source_keys[index]);
        }
        const uint64_t credit_before_version = executor.resource_mutation_before(
            "pipeline_credit", credit_key);
        auto source_insertion = domain.source_slots.try_emplace({0, 0});
        executor.restore_source_slot_provenance(
            domain_id, {0, 0}, source_insertion.first->second);
        domain.outstanding_release_acks.try_emplace({0, 0}, 0);
        for (size_t index = 0; index < source_keys.size(); ++index) {
            executor.resource_mutation_after(
                "pipeline_slot", source_keys[index],
                source_before_versions[index]);
        }
        executor.resource_mutation_after(
            "pipeline_credit", credit_key, credit_before_version);

        executor.flush_pending_slot_row_patches();
        executor.flush_pending_credit_row_patches();
        uint64_t target_initializations = 0;
        uint64_t source_full_rows = 0;
        for (const auto& patch : executor.pending_slot_patches_) {
            if (patch.contains("form")) {
                const size_t slot_index = patch.at("key")
                                              .at("slot_index").get<size_t>();
                const uint64_t expected_generation =
                    slot_index == 0 ? uint64_t{7} : uint64_t{9};
                expect(patch.at("form") == "target_delta" &&
                           patch.at("before_slot_generation") ==
                               expected_generation &&
                           patch.at("predecessor_consumed_count") == 0 &&
                           patch.at("operations").size() == 1 &&
                           patch.at("operations").front().at("op") ==
                               "initialize" &&
                           patch.at("operations").front()
                                   .at("slot_generation") ==
                               expected_generation,
                       "recreated target rows use initialize operations");
                ++target_initializations;
            } else {
                const auto& row = patch.at("before").is_null()
                    ? patch.at("after") : patch.at("before");
                expect(row.at("role") == "source",
                       "recreated source rows retain full-row form");
                ++source_full_rows;
            }
        }
        expect(target_before_versions == std::array<uint64_t, 2>{17, 19} &&
                   executor.resource_versions_.at(
                       {"pipeline_slot", target_keys[0]}) == 18 &&
                   executor.resource_versions_.at(
                       {"pipeline_slot", target_keys[1]}) == 20 &&
                   executor.resource_versions_.at(
                       {"pipeline_slot", source_keys[0]}) == 24 &&
                   executor.resource_versions_.at(
                       {"pipeline_slot", source_keys[1]}) == 30 &&
                   executor.resource_versions_.at(
                       {"pipeline_credit", credit_key}) == 32 &&
                   domain.target_slots[0].generation == 7 &&
                   domain.target_slots[1].generation == 9 &&
                   domain.source_slots.at({0, 0})[0].generation == 3 &&
                   domain.source_slots.at({0, 0})[1].generation == 5 &&
                   domain.source_slot_released_through.at({0, 0}) ==
                       std::array<uint64_t, 2>{11, 13} &&
                   target_initializations == 2 && source_full_rows == 2 &&
                   executor.pending_credit_patches_.size() == 1 &&
                   executor.pending_target_slot_deltas_.empty(),
               "domain reinitialization forwards nonzero resource versions and restores provenance");
        std::printf(
            "[target-slot-version-reinitialize] before=(17,19,23,29,31) "
            "after=(18,20,24,30,32) generations=(7,9,3,5) "
            "target_initializations=2 source_full_rows=2 credit_rows=1\n");

        executor.pending_slot_patches_.clear();
        executor.pending_credit_patches_.clear();
        executor.pending_target_slot_deltas_.clear();
        executor.buffer_domains_.erase(domain_id);
        executor.buffer_domain_target_ranks_.erase(domain_id);
        executor.buffer_domain_provenance_.erase(domain_id);
        for (const std::string& key : target_keys) {
            executor.resource_versions_.erase({"pipeline_slot", key});
        }
        for (const std::string& key : source_keys) {
            executor.resource_versions_.erase({"pipeline_slot", key});
        }
        executor.resource_versions_.erase({"pipeline_credit", credit_key});
        executor.set_ledger_publishers(
            LedgerNodeTransitionPublisher{},
            RemoteReadExecutor::PipelineTransitionPublisher{},
            RemoteReadExecutor::DescriptorCommitPublisher{});
    }

    static nlohmann::json publisher_test_target_delta(
        const std::string& domain_id, const size_t slot_index,
        const bool include_publisher_owned_predecessor = false) {
        nlohmann::json delta{
            {"form", "target_delta"},
            {"key", {
                {"buffer_domain_id", domain_id}, {"role", "target"},
                {"rank", 6}, {"shard_ordinal", nullptr},
                {"slot_index", slot_index},
            }},
            {"before_slot_generation", 0},
            {"predecessor_consumed_count", 0},
            {"operations", nlohmann::json::array({
                {{"op", "initialize"}, {"slot_generation", 0}}})},
        };
        if (include_publisher_owned_predecessor) {
            delta["predecessor_logical_seq"] = 99;
        }
        return delta;
    }

    static void run_publisher_failure_atomicity() {
        constexpr const char* kDomain = "publisher-atomicity-target-domain";
        const auto make_transition = [](nlohmann::json slots) {
            return nlohmann::json{
                {"transition_kind", "descriptor_commit"},
                {"descriptor_segment_id", "publisher-atomicity-segment"},
                {"tick_first", 0}, {"slot_patches", std::move(slots)}};
        };

        WorkLedgerV2Publisher duplicate_publisher("duplicate-key-atomicity");
        bool duplicate_rejected = false;
        try {
            duplicate_publisher.append_pipeline_transition(make_transition(
                nlohmann::json::array({
                    publisher_test_target_delta(kDomain, 0),
                    publisher_test_target_delta(kDomain, 0)})));
        } catch (const std::invalid_argument&) {
            duplicate_rejected = true;
        }
        expect(duplicate_rejected &&
                   duplicate_publisher.next_logical_sequence() == 1 &&
                   WorkLedgerV2PublisherTestAccess::lineage_size(
                       duplicate_publisher) == 0,
               "duplicate target key rejection leaves sequence and lineage unchanged");

        bool external_predecessor_rejected = false;
        try {
            duplicate_publisher.append_pipeline_transition(make_transition(
                nlohmann::json::array({publisher_test_target_delta(
                    kDomain, 0, /*include_publisher_owned_predecessor=*/true)})));
        } catch (const std::invalid_argument&) {
            external_predecessor_rejected = true;
        }
        expect(external_predecessor_rejected &&
                   duplicate_publisher.next_logical_sequence() == 1 &&
                   WorkLedgerV2PublisherTestAccess::lineage_size(
                       duplicate_publisher) == 0,
               "caller-supplied predecessor rejection is atomic");

        duplicate_publisher.append_pipeline_transition(make_transition(
            nlohmann::json::array({publisher_test_target_delta(kDomain, 0)})));
        expect(duplicate_publisher.next_logical_sequence() == 2 &&
                   WorkLedgerV2PublisherTestAccess::lineage_sequence(
                       duplicate_publisher, kDomain, 6, 0) == 1,
               "first valid target delta commits sequence and exact-key lineage");
        const auto initial_frame = duplicate_publisher.publish(
            0, 0, nlohmann::json::array(), nlohmann::json::array());
        const auto& initial_delta = initial_frame.at("pipeline_transition_runs")
                                       .front().at("slot_patches").front();
        expect(initial_delta.at("predecessor_logical_seq") == 0 &&
                   initial_frame.at("schema_version").get<uint64_t>() == 4 &&
                   initial_frame.at("protocol_version").get<uint64_t>() == 4 &&
                   initial_frame.at("bridge_protocol") ==
                       "bridge-v4-20261011" &&
                   initial_frame.at("work_ledger_schema") ==
                       "workledger-v4-20261011",
               "fresh target lineage serializes predecessor zero in a V4 frame");
        duplicate_publisher.append_pipeline_transition(make_transition(
            nlohmann::json::array({publisher_test_target_delta(kDomain, 0)})));
        const auto next_frame = duplicate_publisher.publish(
            1, 0, nlohmann::json::array(), nlohmann::json::array());
        const auto& next_delta = next_frame.at("pipeline_transition_runs")
                                     .front().at("slot_patches").front();
        expect(next_delta.at("predecessor_logical_seq") == 1 &&
                   WorkLedgerV2PublisherTestAccess::lineage_sequence(
                       duplicate_publisher, kDomain, 6, 0) == 2,
               "successive target deltas serialize exact predecessor sequence");

        bool observed_failure_after_placeholder = false;
        for (int64_t fail_after = 0; fail_after < 256 &&
             !observed_failure_after_placeholder; ++fail_after) {
            WorkLedgerV2Publisher publisher(
                "allocation-failure-" + std::to_string(fail_after));
            nlohmann::json transition = make_transition(
                nlohmann::json::array({
                    publisher_test_target_delta("publisher-allocation-domain", 1)}));
            WorkLedgerV2PublisherTestAccess::allocation_probe_publisher =
                &publisher;
            WorkLedgerV2PublisherTestAccess::
                saw_lineage_during_allocation_failure = false;
            set_remote_test_alloc_fail_after(fail_after);
            bool allocation_failed = false;
            try {
                publisher.append_pipeline_transition(std::move(transition));
            } catch (const std::bad_alloc&) {
                allocation_failed = true;
            }
            set_remote_test_alloc_fail_after(-1);
            WorkLedgerV2PublisherTestAccess::allocation_probe_publisher = nullptr;
            if (!allocation_failed) break;
            expect(publisher.next_logical_sequence() == 1 &&
                       WorkLedgerV2PublisherTestAccess::lineage_size(publisher) == 0,
                   "allocation failure leaves no sequence gap or target lineage key");
            if (!WorkLedgerV2PublisherTestAccess::
                    saw_lineage_during_allocation_failure) {
                continue;
            }
            observed_failure_after_placeholder = true;
            publisher.append_pipeline_transition(make_transition(
                nlohmann::json::array({publisher_test_target_delta(
                    "publisher-allocation-domain", 1)})));
            const auto recovered = publisher.publish(
                0, 0, nlohmann::json::array(), nlohmann::json::array());
            const auto& recovered_delta = recovered
                .at("pipeline_transition_runs").front()
                .at("slot_patches").front();
            expect(publisher.next_logical_sequence() == 2 &&
                       recovered_delta.at("predecessor_logical_seq") == 0 &&
                       WorkLedgerV2PublisherTestAccess::lineage_sequence(
                           publisher, "publisher-allocation-domain", 6, 1) == 1,
                   "Publisher retries cleanly after allocation fails with a prepared lineage placeholder");
        }
        expect(observed_failure_after_placeholder,
               "test-only allocation probe reaches Publisher's post-placeholder append failure");
        std::printf("[target-slot-publisher-atomicity] duplicate_and_external_predecessor=atomic "
                    "allocation_after_lineage_placeholder=rollback_and_retry\n");
    }

    static nlohmann::json target_slot_delta_descriptor_wire(
        const nlohmann::json& base_wire, const size_t shard_count,
        const std::string& suffix, const uint64_t id_namespace_offset,
        const bool inactive_tail,
        const bool zero_stride_shared, const bool split_shared_runs) {
        expect(shard_count >= 2 && (!inactive_tail || shard_count >= 3) &&
                   (!split_shared_runs || shard_count >= 2),
               "target-slot descriptor scale is large enough for its layout");
        nlohmann::json wire = base_wire;
        auto& identity = wire["identity"];
        identity["segment_id"] = "target-slot-delta-v4/" + suffix;
        identity["buffer_domain_id"] = "target-slot-domain-v4/" + suffix;

        auto& group = wire["groups"][0];
        const uint64_t tile_count = group["tile_axis"]["tile_count"]
                                        .get<uint64_t>();
        expect(tile_count == 2,
               "target-slot cost descriptor retains the real two-tile axis");
        const uint64_t total_units = group["tile_axis"]["total_units"]
                                         .get<uint64_t>();
        const size_t active_shard_count =
            shard_count - (inactive_tail ? size_t{1} : size_t{0});
        const nlohmann::json shard_template = group["shards"][0];
        nlohmann::json shards = nlohmann::json::array();
        std::vector<uint64_t> all_shards;
        std::vector<uint64_t> active_shards;
        all_shards.reserve(shard_count);
        active_shards.reserve(active_shard_count);
        for (size_t index = 0; index < shard_count; ++index) {
            nlohmann::json shard = shard_template;
            shard["shard_ordinal"] = index;
            shard["source_rank"] = 0;
            shard["target_rank"] = 6;
            shard["total_data_bytes"] =
                inactive_tail && index + 1 == shard_count ? 0 : total_units;
            shard["bytes_per_unit"] = 1;
            shard["noc_path"] = nlohmann::json::array({0, 6});
            shards.push_back(std::move(shard));
            all_shards.push_back(static_cast<uint64_t>(index));
            if (index < active_shard_count) {
                active_shards.push_back(static_cast<uint64_t>(index));
            }
        }
        group["shards"] = std::move(shards);

        const std::array<std::pair<const char*, int>, 6> endpoint_definitions{{
            {"data_send", 0}, {"data_recv", 6},
            {"fill_ack_send", 6}, {"fill_ack_recv", 0},
            {"release_ack_send", 6}, {"release_ack_recv", 0},
        }};
        const std::array<uint64_t, 6> endpoint_id_bases{{
            1000000 + id_namespace_offset,
            2000000 + id_namespace_offset,
            3000000 + id_namespace_offset,
            4000000 + id_namespace_offset,
            5000000 + id_namespace_offset,
            6000000 + id_namespace_offset,
        }};
        const uint64_t start_anchor_node_id = endpoint_id_bases[0];
        const uint64_t frontier_anchor_node_id =
            endpoint_id_bases[1] + active_shard_count;
        nlohmann::json tag_runs = nlohmann::json::array();
        nlohmann::json endpoint_runs = nlohmann::json::array();
        uint64_t next_tag = 1;
        const uint64_t tags_per_endpoint =
            static_cast<uint64_t>(active_shard_count) * tile_count;
        for (size_t endpoint_index = 0;
             endpoint_index < endpoint_definitions.size(); ++endpoint_index) {
            const uint64_t emission = endpoint_index;
            tag_runs.push_back({
                {"allocator_scope_id", "target-slot-cost-tags/" + suffix},
                {"emission_ordinal", emission}, {"first_tag", next_tag},
                {"count", tags_per_endpoint}, {"group_ordinal", 0},
                {"first_tile", 1}, {"tile_count", tile_count},
                {"first_shard", 0},
                {"shard_count", active_shard_count},
            });
            const auto& definition = endpoint_definitions[endpoint_index];
            endpoint_runs.push_back({
                {"rank", definition.second}, {"group_ordinal", 0},
                {"first_tile", 1}, {"tile_count", tile_count},
                {"shard_ordinals", active_shards},
                {"first_json_id", endpoint_id_bases[endpoint_index]},
                {"tile_stride", active_shard_count}, {"shard_stride", 1},
                {"initial_parent_json_id", nullptr},
                {"legacy_context", {
                    {"request_id", identity["request_id"]},
                    {"stage", "prefill"}, {"generation", 0}}},
                {"legs", nlohmann::json::array({
                    {{"endpoint", definition.first}, {"json_offset", 0},
                     {"tag_emission_ordinal", emission},
                     {"tag_offset", 0},
                     {"tag_tile_stride", active_shard_count},
                     {"tag_shard_stride", 1}}})},
            });
            next_tag += tags_per_endpoint;
        }
        wire["tag_allocation_runs"] = std::move(tag_runs);
        wire["endpoint_runs"] = std::move(endpoint_runs);

        // Every endpoint ID must be classified exactly once as either a
        // virtual AP hole or one of the two physical anchor nodes below.
        // Endpoint IDs are contiguous across shards within each tile, so a
        // small interval list represents the full S=8/32/128 matrix without
        // expanding it into one logical range per node.
        std::vector<std::tuple<int, uint64_t, uint64_t>> virtual_intervals;
        const auto add_virtual_interval = [&](const int rank,
                                              const uint64_t first,
                                              const uint64_t count) {
            if (count != 0) {
                virtual_intervals.emplace_back(rank, first, count);
            }
        };
        for (size_t endpoint_index = 0;
             endpoint_index < endpoint_definitions.size(); ++endpoint_index) {
            const int rank = endpoint_definitions[endpoint_index].second;
            const uint64_t base = endpoint_id_bases[endpoint_index];
            for (uint64_t tile_index = 0; tile_index < tile_count;
                 ++tile_index) {
                const uint64_t tile_first =
                    base + tile_index * active_shard_count;
                const uint64_t tile_end = tile_first + active_shard_count;
                uint64_t physical_anchor_id = 0;
                if (endpoint_index == 0 && tile_index == 0) {
                    physical_anchor_id = start_anchor_node_id;
                } else if (endpoint_index == 1 && tile_index == 1) {
                    physical_anchor_id = frontier_anchor_node_id;
                }
                if (physical_anchor_id == 0) {
                    add_virtual_interval(rank, tile_first,
                                         active_shard_count);
                    continue;
                }
                add_virtual_interval(rank, tile_first,
                                     physical_anchor_id - tile_first);
                add_virtual_interval(rank, physical_anchor_id + 1,
                                     tile_end - (physical_anchor_id + 1));
            }
        }
        std::sort(virtual_intervals.begin(), virtual_intervals.end(),
                  [](const auto& left, const auto& right) {
                      return std::tie(std::get<0>(left), std::get<1>(left)) <
                             std::tie(std::get<0>(right), std::get<1>(right));
                  });
        nlohmann::json logical_ranges = nlohmann::json::array();
        for (const auto& interval : virtual_intervals) {
            logical_ranges.push_back({
                {"rank", std::get<0>(interval)},
                {"first_json_id", std::get<1>(interval)},
                {"count", std::get<2>(interval)}, {"stride", 1},
                {"kind", "virtual_endpoint_hole"}});
        }
        wire["logical_id_ranges"] = std::move(logical_ranges);

        // The data-receive runs have an explicit full affine ordinary-parent
        // map, while all six communication leg families are declared above.
        const uint64_t dependency_base =
            10000000 + id_namespace_offset +
            static_cast<uint64_t>(shard_count) * 1000;
        wire["endpoint_dependency_runs"] = nlohmann::json::array({
            {{"rank", 6}, {"group_ordinal", 0}, {"first_tile", 1},
             {"tile_count", tile_count},
             {"shard_ordinals", active_shards}, {"endpoint", "data_recv"},
             {"dependency_kind", "ordinary_node_terminal"},
             {"first_dependency_json_id", dependency_base},
             {"dependency_tile_stride",
              static_cast<uint64_t>(active_shard_count) * 2},
             {"dependency_shard_stride", 1}}
        });

        const uint64_t consume_base =
            20000000 + id_namespace_offset +
            static_cast<uint64_t>(shard_count) * 10000 +
            (zero_stride_shared ? 5000000 : 0) +
            (inactive_tail ? 7000000 : 0);
        nlohmann::json consume_runs = nlohmann::json::array();
        if (split_shared_runs) {
            expect(zero_stride_shared && !inactive_tail,
                   "split completion runs are the active shared-terminal case");
            const size_t midpoint = shard_count / 2;
            for (size_t partition = 0; partition < 2; ++partition) {
                const size_t first = partition == 0 ? 0 : midpoint;
                const size_t last = partition == 0 ? midpoint : shard_count;
                std::vector<uint64_t> members;
                for (size_t index = first; index < last; ++index) {
                    members.push_back(static_cast<uint64_t>(index));
                }
                consume_runs.push_back({
                    {"rank", 6}, {"group_ordinal", 0}, {"first_tile", 1},
                    {"tile_count", tile_count},
                    {"shard_ordinals", std::move(members)},
                    {"dependency_kind", "ordinary_node_terminal"},
                    {"first_dependency_json_id", consume_base},
                    {"dependency_tile_stride", 0},
                    {"dependency_shard_stride", 0},
                });
            }
        } else {
            consume_runs.push_back({
                {"rank", 6}, {"group_ordinal", 0}, {"first_tile", 1},
                {"tile_count", tile_count},
                {"shard_ordinals", all_shards},
                {"dependency_kind", "ordinary_node_terminal"},
                {"first_dependency_json_id", consume_base},
                {"dependency_tile_stride",
                 zero_stride_shared
                     ? uint64_t{0}
                     : static_cast<uint64_t>(shard_count) * 2},
                {"dependency_shard_stride",
                 zero_stride_shared ? uint64_t{0} : uint64_t{1}},
            });
        }
        wire["consume_completion_runs"] = std::move(consume_runs);

        const auto physical_anchor = [&](const char* role, const int rank,
                                         const uint64_t node_id,
                                         const uint64_t tile,
                                         const char* endpoint,
                                         const uint64_t emission,
                                         const uint64_t tag_offset) {
            return nlohmann::json{
                {"role", role}, {"rank", rank},
                {"binding", "physical_endpoint_slot"}, {"node_id", node_id},
                {"selector", {
                    {"kind", "tile_endpoint_leg"}, {"group_ordinal", 0},
                    {"tile_ordinal", tile}, {"shard_ordinal", 0},
                    {"endpoint", endpoint},
                    {"tag_allocation_emission_ordinal", emission},
                    {"tag_offset", tag_offset}}},
            };
        };
        wire["anchors"] = nlohmann::json::array({
            physical_anchor("start", 0, start_anchor_node_id, 1,
                            "data_send", 0, 0),
            physical_anchor("frontier", 6, frontier_anchor_node_id, 2,
                            "data_recv", 1, active_shard_count),
            {{"role", "service"}, {"rank", 6},
             {"binding", "descriptor_cursor_selector"}, {"node_id", nullptr},
             {"selector", {{"kind", "existing_service_gate"},
                            {"existing_gate", "prefill_service_gate"}}}},
            {{"role", "settle"}, {"rank", 6},
             {"binding", "descriptor_cursor_selector"}, {"node_id", nullptr},
             {"selector", {{"kind", "all_obligations_settled"}}}},
        });
        return wire;
    }

    static nlohmann::json publish_direct_pipeline_transition(
        RemoteReadExecutor& executor, WorkLedgerV2Publisher& publisher,
        uint64_t* delivery_sequence,
        RemoteReadExecutor::RuntimeDescriptor& owner,
        const std::string& transition_kind,
        const RemoteReadExecutor::EndpointPosition* position,
        const uint64_t store_id,
        const uint64_t cursor_version_start) {
        expect(delivery_sequence != nullptr,
               "direct target-slot publisher has a delivery counter");
        const uint64_t tick = Sys::boostedTick();
        const nlohmann::json amount{
            {"ops", nullptr}, {"bytes", nullptr}, {"latency_ns", nullptr}};
        executor.publish_pipeline_transition(
            owner, transition_kind, position, store_id, "", amount,
            cursor_version_start);
        return publisher.publish(
            (*delivery_sequence)++, tick,
            executor.take_cursor_patches(tick), nlohmann::json::array());
    }

    static const nlohmann::json& only_pipeline_transition(
        const nlohmann::json& frame) {
        const auto& transitions = frame.at("pipeline_transition_runs");
        expect(transitions.size() == 1,
               "direct target-slot publication has one pipeline transition");
        return transitions.front();
    }

    static const nlohmann::json& only_target_delta(
        const nlohmann::json& transition, const std::string& domain_id,
        const size_t slot_index) {
        const auto& patches = transition.at("slot_patches");
        const nlohmann::json* selected = nullptr;
        for (const auto& patch : patches) {
            if (!patch.is_object() ||
                patch.value("form", std::string()) != "target_delta") {
                continue;
            }
            const auto& key = patch.at("key");
            if (key.at("buffer_domain_id").get<std::string>() == domain_id &&
                key.at("slot_index").get<size_t>() == slot_index) {
                expect(selected == nullptr,
                       "one target delta exists per stable target slot key");
                selected = &patch;
            }
        }
        expect(selected != nullptr,
               "published transition contains its target slot delta");
        return *selected;
    }

    static void run_direct_target_slot_case(
        RemoteReadExecutor& executor,
        const std::vector<std::shared_ptr<NodeStoreGraphSource>>& graph_sources,
        const std::shared_ptr<EventQueue>& event_queue,
        const std::shared_ptr<const PipelineDescriptor>& descriptor,
        WorkLedgerV2Publisher& publisher,
        uint64_t* delivery_sequence,
        const size_t shard_count, const bool split_shared_runs,
        const uint64_t starting_tick) {
        expect(descriptor != nullptr && graph_sources.size() > 6 &&
                   graph_sources[6] != nullptr,
               "direct target-slot fixture owns a validated descriptor and target source");
        const int target_rank = descriptor->identity().rank;
        const std::string& domain_id =
            descriptor->identity().buffer_domain_id;
        expect(target_rank == 6 && shard_count >= 2,
               "target slot has one shared rank owner and multiple shards");

        executor.buffer_domains_.clear();
        executor.buffer_domain_target_ranks_.clear();
        executor.buffer_domain_provenance_.clear();
        executor.resource_versions_.clear();
        executor.resource_mutation_before_rows_.clear();
        executor.pending_target_slot_deltas_.clear();
        executor.pending_slot_patches_.clear();
        executor.pending_credit_patches_.clear();
        executor.pending_obligation_patches_.clear();
        executor.producer_metrics_ = {};
        executor.set_ledger_publishers(
            LedgerNodeTransitionPublisher{},
            [&publisher](nlohmann::json transition) {
                publisher.append_pipeline_transition(std::move(transition));
            },
            RemoteReadExecutor::DescriptorCommitPublisher{});

        auto owner =
            std::make_unique<RemoteReadExecutor::RuntimeDescriptor>();
        owner->descriptor = descriptor;
        const auto target_run = std::find_if(
            descriptor->endpoint_runs().begin(),
            descriptor->endpoint_runs().end(),
            [](const PipelineEndpointRun& run) {
                return run.rank == 6 && run.group_ordinal == 0 &&
                    std::any_of(run.legs.begin(), run.legs.end(),
                        [](const PipelineEndpointLegRun& leg) {
                            return leg.endpoint == PipelineEndpointLeg::DataRecv;
                        });
            });
        expect(target_run != descriptor->endpoint_runs().end(),
               "validated descriptor declares target DataRecv endpoint runs");
        const auto target_leg = std::find_if(
            target_run->legs.begin(), target_run->legs.end(),
            [](const PipelineEndpointLegRun& leg) {
                return leg.endpoint == PipelineEndpointLeg::DataRecv;
            });
        expect(target_leg != target_run->legs.end(),
               "target endpoint run contains its DataRecv leg");
        RemoteReadExecutor::RunCursor cursor;
        cursor.run = &*target_run;
        cursor.first_store_id = descriptor->endpoint_json_id(
            *target_run, *target_leg, target_run->first_tile,
            target_run->shard_ordinals.front());
        cursor.tile_offset = 0;
        cursor.shard_index = 0;
        cursor.leg_index = static_cast<size_t>(
            std::distance(target_run->legs.begin(), target_leg));
        owner->runs.push_back(cursor);
        const PipelineTile tile = PipelineKernel::tile(
            *descriptor, target_run->group_ordinal,
            target_run->first_tile - 1);
        const size_t slot_index = static_cast<size_t>(
            (tile.global_tile_ordinal - 1) % 2);
        expect(slot_index == 0,
               "cost fixture's same local tile uses target slot zero");

        const std::string stable_slot_key = executor.slot_resource_key(
            domain_id, "target", target_rank, 0, slot_index);
        std::array<uint64_t, 2> initialize_versions{};
        std::array<std::string, 2> target_slot_keys;
        for (size_t index = 0; index < target_slot_keys.size(); ++index) {
            target_slot_keys[index] = executor.slot_resource_key(
                domain_id, "target", target_rank, 0, index);
            initialize_versions[index] = executor.resource_mutation_before(
                "pipeline_slot", target_slot_keys[index], target_rank);
        }
        auto domain_insert = executor.buffer_domains_.try_emplace(domain_id);
        RemoteReadExecutor::BufferDomainState& domain =
            domain_insert.first->second;
        executor.restore_buffer_domain_provenance(
            domain_id, domain, target_rank);
        executor.buffer_domain_target_ranks_[domain_id] = target_rank;
        for (size_t index = 0; index < target_slot_keys.size(); ++index) {
            executor.resource_mutation_after(
                "pipeline_slot", target_slot_keys[index],
                initialize_versions[index]);
        }
        const uint64_t event_tick_before = event_queue->get_current_time();
        const uint64_t epoch_before = descriptor->identity().stage_epoch;
        publisher.append_descriptor_commit({
            {"identity", descriptor->wire().at("identity")},
            {"descriptor", descriptor->wire()},
            {"commit_tick", Sys::boostedTick()},
            {"cursor_version", owner->cursor_version},
            {"service_done", owner->service_done},
            {"settle_done", owner->settle_done},
            {"retired", owner->retired},
        });
        const nlohmann::json initialize_frame =
            publish_direct_pipeline_transition(
                executor, publisher, delivery_sequence, *owner,
                "descriptor_commit", nullptr, 0, owner->cursor_version);
        expect(initialize_frame.at("schema_version").get<uint64_t>() == 4 &&
                   initialize_frame.at("protocol_version").get<uint64_t>() == 4 &&
                   only_target_delta(only_pipeline_transition(initialize_frame),
                                     domain_id, 0)
                           .at("predecessor_logical_seq") == 0 &&
                   only_target_delta(only_pipeline_transition(initialize_frame),
                                     domain_id, 1)
                           .at("predecessor_logical_seq") == 0 &&
                   only_target_delta(only_pipeline_transition(initialize_frame),
                                     domain_id, 0)
                           .at("operations").at(0).at("op") == "initialize",
               "V4 descriptor commit initializes both target slot lineages");

        const uint64_t claim_before = executor.resource_mutation_before(
            "pipeline_slot", stable_slot_key);
        executor.claim_slot(domain.target_slots[slot_index], *owner, cursor,
                            tile.global_tile_ordinal);
        executor.record_target_slot_claim(stable_slot_key,
                                          domain.target_slots[slot_index]);
        executor.resource_mutation_after("pipeline_slot", stable_slot_key,
                                         claim_before);
        const auto claim_resource_version_entry =
            executor.resource_versions_.find(
                std::make_pair(std::string("pipeline_slot"), stable_slot_key));
        expect(claim_resource_version_entry !=
                   executor.resource_versions_.end(),
               "real slot claim records its resource version");
        const uint64_t resource_version_after_claim =
            claim_resource_version_entry->second;
        RemoteReadExecutor::EndpointPosition claim_position;
        claim_position.owner = owner.get();
        claim_position.run_index = 0;
        claim_position.tile_offset = cursor.tile_offset;
        claim_position.shard_index = cursor.shard_index;
        claim_position.leg_index = cursor.leg_index;
        claim_position.slot_generation =
            domain.target_slots[slot_index].generation;
        const uint64_t claim_cursor_version_start = owner->cursor_version;
        ++owner->cursor_version;
        const nlohmann::json claim_frame =
            publish_direct_pipeline_transition(
                executor, publisher, delivery_sequence, *owner,
                "tile_issue", &claim_position, cursor.first_store_id,
                claim_cursor_version_start);
        const nlohmann::json& claim_delta = only_target_delta(
            only_pipeline_transition(claim_frame), domain_id, slot_index);
        uint64_t claim_target_elements = 0;
        for (const auto& operation : claim_delta.at("operations")) {
            if (operation.at("op") == "claim") {
                claim_target_elements +=
                    operation.at("target_shards").size();
            }
        }
        expect(claim_delta.at("before_slot_generation") == 0 &&
                   claim_delta.at("predecessor_logical_seq") == 1 &&
                   claim_delta.at("predecessor_consumed_count") == 0 &&
                   claim_target_elements == shard_count,
               "V4 claim declares full target membership once after initialize");
        executor.producer_metrics_ = {};
        const nlohmann::json public_before = executor.slot_state_json(domain_id);
        expect(public_before.size() == 2 &&
                   public_before.at(slot_index).at("occupied") == true &&
                   public_before.at(slot_index).at("owner").at("segment_id") ==
                       descriptor->identity().segment_id &&
                   public_before.at(slot_index).at("owner").at("stage_epoch") ==
                       descriptor->identity().stage_epoch &&
                   public_before.at(slot_index).at("owner").at("tile_ordinal") ==
                       target_run->first_tile &&
                   public_before.at(slot_index).at("target_shards_to_consume").size() ==
                       shard_count &&
                   public_before.at(slot_index).at("consumed_target_shards").empty(),
               "real claim_slot builds the complete occupied target row");
        const uint64_t slot_generation_before =
            domain.target_slots[slot_index].generation;
        expect(slot_generation_before == 1 &&
                   domain.target_slots[slot_index].target_shards_to_consume.size() ==
                       shard_count,
               "target owner row is claimed once with all active declared shards");

        // Give each descriptor-declared ordinary terminal a real NodeStore
        // record at the identity-mapped store ID used by this direct probe.
        std::set<uint64_t> ordinary_store_ids;
        for (const auto& dependency : descriptor->endpoint_dependency_runs()) {
            for (uint64_t tile_offset = 0;
                 tile_offset < dependency.tile_count; ++tile_offset) {
                for (size_t shard_index = 0;
                     shard_index < dependency.shard_ordinals.size();
                     ++shard_index) {
                    ordinary_store_ids.insert(
                        dependency.first_dependency_json_id +
                        tile_offset * dependency.dependency_tile_stride +
                        static_cast<uint64_t>(shard_index) *
                            dependency.dependency_shard_stride);
                }
            }
        }
        for (const auto& consume : descriptor->consume_completion_runs()) {
            for (uint64_t tile_offset = 0; tile_offset < consume.tile_count;
                 ++tile_offset) {
                for (size_t shard_index = 0;
                     shard_index < consume.shard_ordinals.size();
                     ++shard_index) {
                    ordinary_store_ids.insert(
                        consume.first_dependency_json_id +
                        tile_offset * consume.dependency_tile_stride +
                        static_cast<uint64_t>(shard_index) *
                            consume.dependency_shard_stride);
                }
            }
        }
        for (const uint64_t store_id : ordinary_store_ids) {
            OnlineNode node;
            node.global_id = store_id;
            node.rank = target_rank;
            node.kind = NodeKind::Metadata;
            node.node_type = 1;
            node.name = "target-slot-probe-terminal-" +
                std::to_string(store_id);
            expect(graph_sources[target_rank]->store().add_node(
                       std::move(node)) == store_id,
                   "descriptor ordinary terminal maps to its real NodeStore ID");
        }

        std::vector<nlohmann::json> intermediate_rows;
        std::vector<nlohmann::json> consume_frames;
        std::map<size_t, uint64_t> consume_run_index_calls;
        uint64_t direct_release_calls = 0;
        if (!split_shared_runs) {
            expect(descriptor->consume_completion_runs().size() == 1,
                   "scale fixture has one immutable consume completion run");
            const PipelineConsumeCompletionRun& consume =
                descriptor->consume_completion_runs().front();
            expect(consume.first_tile == target_run->first_tile &&
                       consume.dependency_shard_stride == 1 &&
                       consume.dependency_kind ==
                           PipelineEndpointDependencyKind::OrdinaryNodeTerminal,
                   "scale fixture maps one ordinary terminal per target shard");
            RemoteReadExecutor::TerminalWatcher watcher;
            watcher.owner = owner.get();
            watcher.consume_run = &consume;
            watcher.consume_run_index = 0;
            watcher.wake_rank = consume.rank;
            watcher.group_ordinal = consume.group_ordinal;
            watcher.kind = consume.dependency_kind;
            watcher.is_consume_completion = true;
            watcher.first_store_id = consume.first_dependency_json_id;
            watcher.tile_count = consume.tile_count;
            watcher.tile_stride = consume.dependency_tile_stride;
            for (size_t index = 0; index < consume.shard_ordinals.size(); ++index) {
                watcher.shard_offsets.push_back(
                    static_cast<uint64_t>(index) *
                    consume.dependency_shard_stride);
            }
            for (size_t shard_index = 0;
                 shard_index < consume.shard_ordinals.size(); ++shard_index) {
                const uint64_t terminal_store_id =
                    consume.first_dependency_json_id +
                    static_cast<uint64_t>(shard_index) *
                        consume.dependency_shard_stride;
                graph_sources[target_rank]->take_node(terminal_store_id);
                expect(graph_sources[target_rank]->mark_terminal_observed(
                           terminal_store_id),
                       "direct per-shard cause is a real issued terminal");
                graph_sources[target_rank]->finish_node(terminal_store_id);
                executor.release_target_slot_for_consume(
                    watcher, target_rank, terminal_store_id);
                ++consume_run_index_calls[0];
                ++direct_release_calls;
                const uint64_t cursor_version_start = owner->cursor_version;
                ++owner->cursor_version;
                consume_frames.push_back(
                    publish_direct_pipeline_transition(
                        executor, publisher, delivery_sequence, *owner,
                        "consume_done", nullptr, 0,
                        cursor_version_start));
            }
        } else {
            expect(descriptor->consume_completion_runs().size() == 2,
                   "shared terminal fixture has two immutable consume runs");
            const uint64_t shared_store_id =
                descriptor->consume_completion_runs().front()
                    .first_dependency_json_id;
            graph_sources[target_rank]->take_node(shared_store_id);
            expect(graph_sources[target_rank]->mark_terminal_observed(
                       shared_store_id),
                   "shared zero-stride cause is one real issued terminal");
            graph_sources[target_rank]->finish_node(shared_store_id);
            for (size_t run_index = 0;
                 run_index < descriptor->consume_completion_runs().size();
                 ++run_index) {
                const auto& consume =
                    descriptor->consume_completion_runs()[run_index];
                expect(consume.first_dependency_json_id == shared_store_id &&
                           consume.dependency_shard_stride == 0 &&
                           consume.dependency_tile_stride == 0,
                       "shared terminal runs retain exact zero-stride mapping");
                RemoteReadExecutor::TerminalWatcher watcher;
                watcher.owner = owner.get();
                watcher.consume_run = &consume;
                watcher.consume_run_index = run_index;
                watcher.wake_rank = consume.rank;
                watcher.group_ordinal = consume.group_ordinal;
                watcher.kind = consume.dependency_kind;
                watcher.is_consume_completion = true;
                watcher.first_store_id = consume.first_dependency_json_id;
                watcher.tile_count = consume.tile_count;
                watcher.tile_stride = consume.dependency_tile_stride;
                watcher.shard_offsets.assign(consume.shard_ordinals.size(), 0);
                executor.release_target_slot_for_consume(
                    watcher, target_rank, shared_store_id);
                ++consume_run_index_calls[run_index];
                ++direct_release_calls;
                if (run_index == 0) {
                    intermediate_rows.push_back(
                        executor.slot_state_json(domain_id));
                }
            }
            const uint64_t cursor_version_start = owner->cursor_version;
            ++owner->cursor_version;
            consume_frames.push_back(
                publish_direct_pipeline_transition(
                    executor, publisher, delivery_sequence, *owner,
                    "consume_done", nullptr, 0,
                    cursor_version_start));
        }

        const auto full_public_after = executor.slot_state_json(domain_id);
        const RemoteReadExecutor::ProducerMetrics measured =
            executor.producer_metrics();
        const auto count_public_arrays = [](
            const nlohmann::json& rows) {
            std::pair<uint64_t, uint64_t> counts{0, 0};
            for (const auto& row : rows) {
                counts.first += row.at("target_shards_to_consume").size();
                counts.second += row.at("consumed_target_shards").size();
            }
            return counts;
        };
        const auto before_public_counts = count_public_arrays(public_before);
        const auto after_public_counts = count_public_arrays(full_public_after);
        uint64_t consume_delta_rows = 0;
        uint64_t consume_operation_count = 0;
        uint64_t consume_addition_count = 0;
        uint64_t target_full_array_patch_elements = 0;
        const uint64_t claim_logical_seq =
            only_pipeline_transition(claim_frame)
                .at("first_logical_seq").get<uint64_t>();
        for (size_t frame_index = 0; frame_index < consume_frames.size();
             ++frame_index) {
            const auto& transition =
                only_pipeline_transition(consume_frames[frame_index]);
            const auto& patches = transition.at("slot_patches");
            expect(patches.size() == 1 &&
                       patches.front().at("form") == "target_delta",
                   "each real terminal callback emits one compact target delta");
            const auto& delta = only_target_delta(
                transition, domain_id, slot_index);
            ++consume_delta_rows;
            expect(delta.at("before_slot_generation") ==
                           slot_generation_before &&
                       delta.at("predecessor_logical_seq") ==
                           claim_logical_seq + frame_index &&
                       delta.at("predecessor_consumed_count") == frame_index,
                   "each consume delta preserves its predecessor sequence and count");
            for (const auto& operation : delta.at("operations")) {
                expect(operation.at("op") == "consume",
                       "terminal transition preserves ordered consume operations");
                ++consume_operation_count;
                consume_addition_count +=
                    operation.at("add_shards").size();
                expect(operation.at("cause").at("terminal_rank") ==
                           target_rank &&
                           operation.at("cause").at("consume_run_index") <
                               descriptor->consume_completion_runs().size(),
                       "consume cause names the actual terminal rank and descriptor run index");
                const auto& cause_run = descriptor->consume_completion_runs().at(
                    operation.at("cause").at("consume_run_index")
                        .get<size_t>());
                const uint64_t expected_terminal_id = split_shared_runs
                    ? cause_run.first_dependency_json_id
                    : cause_run.first_dependency_json_id +
                          frame_index * cause_run.dependency_shard_stride;
                expect(operation.at("cause").at("terminal_store_id") ==
                           expected_terminal_id,
                       "consume cause preserves the exact terminal store ID bytes");
                if (!split_shared_runs) {
                    expect(operation.at("cause").at("consume_run_index") == 0 &&
                               operation.at("add_shards").size() == 1 &&
                               operation.at("cleared").get<bool>() ==
                                   (frame_index + 1 == shard_count),
                           "direct per-shard callbacks retain one exact ordered member delta");
                }
            }
            expect(delta.contains("operations"),
                   "V4 target patch has its ordered operation list");
        }
        if (split_shared_runs) {
            const auto& transition =
                only_pipeline_transition(consume_frames.front());
            const auto& delta = only_target_delta(
                transition, domain_id, slot_index);
            expect(consume_frames.size() == 1 &&
                       delta.at("operations").size() == 2 &&
                       delta.at("operations").at(0).at("cause")
                               .at("consume_run_index") == 0 &&
                       delta.at("operations").at(0).at("cleared") == false &&
                       delta.at("operations").at(1).at("cause")
                               .at("consume_run_index") == 1 &&
                       delta.at("operations").at(1).at("cleared") == true,
                   "same-terminal run operations retain their order in one transition");
        }
        for (const auto& frame : consume_frames) {
            for (const auto& transition : frame.at("pipeline_transition_runs")) {
                for (const auto& patch : transition.at("slot_patches")) {
                    if (patch.contains("form")) continue;
                    for (const char* side : {"before", "after"}) {
                        if (patch.at(side).is_null()) continue;
                        target_full_array_patch_elements +=
                            patch.at(side).at("target_shards_to_consume").size() +
                            patch.at(side).at("consumed_target_shards").size();
                    }
                }
            }
        }

        const auto target_delta_dump_bytes = [](
            const nlohmann::json& frame) {
            uint64_t bytes = 0;
            for (const auto& transition :
                 frame.at("pipeline_transition_runs")) {
                for (const auto& patch : transition.at("slot_patches")) {
                    if (patch.is_object() && patch.contains("form") &&
                        patch.at("form") == "target_delta") {
                        bytes += patch.dump().size();
                    }
                }
            }
            return bytes;
        };
        uint64_t consume_frame_dump_bytes = 0;
        uint64_t consume_target_delta_dump_bytes = 0;
        for (const auto& frame : consume_frames) {
            consume_frame_dump_bytes += frame.dump().size();
            consume_target_delta_dump_bytes +=
                target_delta_dump_bytes(frame);
        }
        const size_t initialize_frame_dump_bytes =
            initialize_frame.dump().size();
        const uint64_t initialize_target_delta_dump_bytes =
            target_delta_dump_bytes(initialize_frame);
        const size_t claim_frame_dump_bytes = claim_frame.dump().size();
        const uint64_t claim_target_delta_dump_bytes =
            target_delta_dump_bytes(claim_frame);

        const uint64_t terminal_base = descriptor->consume_completion_runs()
                                           .front()
                                           .first_dependency_json_id;
        const uint64_t terminal_last = split_shared_runs
            ? terminal_base
            : terminal_base +
                  static_cast<uint64_t>(shard_count - 1) *
                      descriptor->consume_completion_runs().front()
                          .dependency_shard_stride;
        const auto& final_slot = domain.target_slots[slot_index];
        const auto& final_resource_versions = executor.resource_versions_;
        const auto final_resource_version = final_resource_versions.find(
            {"pipeline_slot", stable_slot_key});
        const uint64_t expected_resource_version =
            resource_version_after_claim + direct_release_calls;
        std::string consume_run_indices;
        for (const auto& entry : consume_run_index_calls) {
            if (!consume_run_indices.empty()) consume_run_indices += ",";
            consume_run_indices += std::to_string(entry.first) + ":" +
                std::to_string(entry.second);
        }
        expect(!final_slot.occupied && final_slot.generation ==
                   slot_generation_before &&
                   final_slot.consumed_target_shards.empty() &&
                   final_slot.target_shards_to_consume.empty() &&
                   final_resource_version != final_resource_versions.end() &&
                   final_resource_version->second == expected_resource_version &&
                   consume_delta_rows ==
                       (split_shared_runs ? uint64_t{1} : shard_count) &&
                   consume_operation_count == direct_release_calls &&
                   consume_addition_count == shard_count &&
                   measured.target_delta_patch_rows == consume_delta_rows &&
                   measured.target_delta_operations == consume_operation_count &&
                   measured.target_consume_added_shards == shard_count &&
                   measured.resource_row_capture_calls == 0 &&
                   measured.target_full_row_json_builds ==
                       4 + (split_shared_runs ? 2 : 0) &&
                   measured.row_helper_visits ==
                       measured.target_full_row_json_builds &&
                   measured.actual_issue_calls == 0 &&
                   executor.resource_mutation_before_rows_.empty() &&
                   executor.pending_target_slot_deltas_.empty(),
               "real consume callbacks emit exact deltas without full target-row capture");
        expect(full_public_after.size() == 2 &&
                   !full_public_after.at(slot_index).at("occupied").get<bool>() &&
                   full_public_after.at(slot_index).at("owner").is_null() &&
                   full_public_after.at(slot_index).at("target_shards_to_consume").empty() &&
                   full_public_after.at(slot_index).at("consumed_target_shards").empty(),
               "real final consume clears the complete public slot row");
        if (split_shared_runs) {
            expect(intermediate_rows.size() == 1 &&
                       intermediate_rows.front().size() == 2 &&
                       intermediate_rows.front().at(slot_index)
                               .at("occupied").get<bool>() &&
                       intermediate_rows.front().at(slot_index)
                               .at("target_shards_to_consume").size() ==
                           shard_count &&
                       intermediate_rows.front().at(slot_index)
                               .at("consumed_target_shards").size() ==
                           shard_count / 2,
                   "first same-terminal zero-stride run leaves the exact intermediate target subset");
        }
        expect(Sys::boostedTick() == starting_tick &&
                   event_queue->get_current_time() == event_tick_before &&
                   descriptor->identity().stage_epoch == epoch_before &&
                   executor.active_cursor_count() == 0,
               "direct target consume does not advance simulation state");

        std::printf(
            "[target-slot-direct] S=%zu cause_rank=%d tile=%llu "
            "consume_run_index_calls=%s "
            "terminal_ids=%llu..%llu direct_release_calls=%llu "
            "claim_target_array_elements=%llu consume_add_shard_elements=%llu "
            "consume_delta_rows=%llu consume_operations=%llu "
            "legacy_target_full_array_patch_elements=%llu "
            "public_before_arrays=(%llu,%llu) public_after_arrays=(%llu,%llu) "
            "row_capture_calls=%llu target_full_row_json_builds=%llu "
            "intermediate_snapshot_row_visits=%llu "
            "slot_generation=%llu->%llu resource_version=%llu->%llu "
            "descriptor_epoch=%llu tick=%llu event_tick=%llu\n",
            shard_count, target_rank,
            static_cast<unsigned long long>(target_run->first_tile),
            consume_run_indices.c_str(),
            static_cast<unsigned long long>(terminal_base),
            static_cast<unsigned long long>(terminal_last),
            static_cast<unsigned long long>(direct_release_calls),
            static_cast<unsigned long long>(claim_target_elements),
            static_cast<unsigned long long>(consume_addition_count),
            static_cast<unsigned long long>(consume_delta_rows),
            static_cast<unsigned long long>(consume_operation_count),
            static_cast<unsigned long long>(target_full_array_patch_elements),
            static_cast<unsigned long long>(before_public_counts.first),
            static_cast<unsigned long long>(before_public_counts.second),
            static_cast<unsigned long long>(after_public_counts.first),
            static_cast<unsigned long long>(after_public_counts.second),
            static_cast<unsigned long long>(measured.resource_row_capture_calls),
            static_cast<unsigned long long>(measured.target_full_row_json_builds),
            static_cast<unsigned long long>(split_shared_runs ? 2 : 0),
            static_cast<unsigned long long>(slot_generation_before),
            static_cast<unsigned long long>(final_slot.generation),
            static_cast<unsigned long long>(resource_version_after_claim),
            static_cast<unsigned long long>(expected_resource_version),
            static_cast<unsigned long long>(descriptor->identity().stage_epoch),
            static_cast<unsigned long long>(Sys::boostedTick()),
            static_cast<unsigned long long>(event_tick_before));
        std::printf("[target-slot-public-before] %s\n",
                    public_before.dump().c_str());
        if (!intermediate_rows.empty()) {
            std::printf("[target-slot-public-middle-after-run-0] %s\n",
                        intermediate_rows.front().dump().c_str());
        }
        std::printf("[target-slot-public-after] %s\n",
                    full_public_after.dump().c_str());
        std::printf(
            "[target-slot-wire-dump-bytes] initialize_frame=%zu "
            "initialize_target_delta=%llu claim_frame=%zu "
            "claim_target_delta=%llu consume_frames=%zu "
            "consume_callbacks=%llu consume_target_delta_rows=%llu "
            "consume_frame_sum=%llu consume_target_delta_sum=%llu\n",
            initialize_frame_dump_bytes,
            static_cast<unsigned long long>(
                initialize_target_delta_dump_bytes),
            claim_frame_dump_bytes,
            static_cast<unsigned long long>(claim_target_delta_dump_bytes),
            consume_frames.size(),
            static_cast<unsigned long long>(direct_release_calls),
            static_cast<unsigned long long>(consume_delta_rows),
            static_cast<unsigned long long>(consume_frame_dump_bytes),
            static_cast<unsigned long long>(consume_target_delta_dump_bytes));
        executor.set_ledger_publishers(
            LedgerNodeTransitionPublisher{},
            RemoteReadExecutor::PipelineTransitionPublisher{},
            RemoteReadExecutor::DescriptorCommitPublisher{});
    }

    static bool completed_range_aborts(
        const RemoteReadExecutor::RunCursor& cursor,
        const uint64_t first, const uint64_t count) {
        const pid_t child = fork();
        if (child < 0) return false;
        if (child == 0) {
            const struct rlimit no_core{0, 0};
            static_cast<void>(setrlimit(RLIMIT_CORE, &no_core));
            static_cast<void>(std::freopen("/dev/null", "w", stderr));
            RemoteReadExecutor::RunCursor child_cursor = cursor;
            RemoteReadExecutor::mark_cursor_range_completed(
                child_cursor, first, count);
            _exit(0);
        }
        int status = 0;
        if (waitpid(child, &status, 0) != child) return false;
        return WIFSIGNALED(status) && WTERMSIG(status) == SIGABRT;
    }

    static void run_completed_count_scaling(
        RemoteReadExecutor& executor, const BuiltFixture& fixture) {
        const auto descriptor = PipelineDescriptor::parse(
            fixture.programs.at(0).wire);
        const std::vector<uint64_t> interval_counts{8, 32, 128};
        for (const uint64_t interval_count : interval_counts) {
            PipelineEndpointRun run;
            run.rank = 0;
            run.group_ordinal = 0;
            run.first_tile = 1;
            run.tile_count = interval_count * 2 + 2;
            run.shard_ordinals = {0};
            run.first_json_id = 0;
            run.tile_stride = 1;
            run.shard_stride = 1;
            run.legs.push_back(
                PipelineEndpointLegRun{PipelineEndpointLeg::DataSend, 0});

            RemoteReadExecutor::RunCursor cursor;
            cursor.run = &run;
            cursor.first_store_id = 100;
            cursor.completed_prefix = 1;
            cursor.completed_count = 1;
            for (uint64_t index = 0; index < interval_count; ++index) {
                RemoteReadExecutor::mark_cursor_range_completed(
                    cursor, 2 + index * 2, 1);
            }
            uint64_t interval_oracle = cursor.completed_prefix;
            uint64_t interval_oracle_visits = 0;
            for (const auto& interval : cursor.completed_out_of_order) {
                ++interval_oracle_visits;
                interval_oracle += interval.second - interval.first;
            }
            expect(cursor.completed_count == interval_oracle &&
                       cursor.completed_out_of_order.size() == interval_count &&
                       interval_oracle_visits == interval_count,
                   "cached completed count matches independent interval sum");

            RemoteReadExecutor::RuntimeDescriptor owner;
            owner.descriptor = descriptor;
            owner.runs.push_back(cursor);
            executor.producer_metrics_ = {};
            const uint64_t count_reads_before = executor.producer_metrics_
                .obligation_completed_count_reads;
            const nlohmann::json obligation =
                executor.obligation_row_json(owner, 0);
            const uint64_t count_read_delta = executor.producer_metrics_
                .obligation_completed_count_reads - count_reads_before;
            const uint64_t total = run.tile_count;
            expect(obligation.at("remaining_endpoint_count") ==
                       total - interval_oracle &&
                       executor.producer_metrics_.row_helper_visits == 1 &&
                       count_read_delta == 1,
                   "obligation row reads one cached count independent of H");

            std::printf(
                "[cursor-completed-count] T=1 H=%llu completed=%llu "
                "oracle=%llu oracle_interval_visits=%llu "
                "completed_count_reads=%llu\n",
                static_cast<unsigned long long>(interval_count),
                static_cast<unsigned long long>(cursor.completed_count),
                static_cast<unsigned long long>(interval_oracle),
                static_cast<unsigned long long>(interval_oracle_visits),
                static_cast<unsigned long long>(count_read_delta));
        }

        RemoteReadExecutor::RunCursor late_cursor;
        RemoteReadExecutor::mark_cursor_range_completed(late_cursor, 4, 1);
        RemoteReadExecutor::mark_cursor_range_completed(late_cursor, 2, 1);
        RemoteReadExecutor::mark_cursor_range_completed(late_cursor, 1, 1);
        RemoteReadExecutor::mark_cursor_range_completed(late_cursor, 3, 1);
        expect(late_cursor.completed_prefix == 0 &&
                   late_cursor.completed_out_of_order.size() == 1 &&
                   late_cursor.completed_out_of_order.begin()->first == 1 &&
                   late_cursor.completed_out_of_order.begin()->second == 5 &&
                   late_cursor.completed_count == 4,
               "late completions and adjacent ranges preserve exact count");
        RemoteReadExecutor::mark_cursor_range_completed(late_cursor, 0, 1);
        expect(late_cursor.completed_prefix == 5 &&
                   late_cursor.completed_out_of_order.empty() &&
                   late_cursor.completed_count == 5,
               "completion of the first gap absorbs adjacent ranges into the prefix");
        RemoteReadExecutor::mark_cursor_range_completed(late_cursor, 8, 2);
        RemoteReadExecutor::mark_cursor_range_completed(late_cursor, 6, 2);
        expect(late_cursor.completed_prefix == 5 &&
                   late_cursor.completed_out_of_order.size() == 1 &&
                   late_cursor.completed_out_of_order.begin()->first == 6 &&
                   late_cursor.completed_out_of_order.begin()->second == 10 &&
                   late_cursor.completed_count == 9,
               "adjacent ranges coalesce without changing completed cardinality");
        RemoteReadExecutor::mark_cursor_range_completed(late_cursor, 5, 1);
        expect(late_cursor.completed_prefix == 10 &&
                   late_cursor.completed_out_of_order.empty() &&
                   late_cursor.completed_count == 10,
               "gap completion absorbs adjacent ranges into the prefix");

        RemoteReadExecutor::RunCursor duplicate_cursor;
        RemoteReadExecutor::mark_cursor_range_completed(
            duplicate_cursor, 2, 1);
        expect(completed_range_aborts(duplicate_cursor, 2, 1),
               "duplicate endpoint completion remains fail-closed");
        expect(completed_range_aborts(duplicate_cursor, 2, 2),
               "overlapping endpoint completion remains fail-closed");
        expect(completed_range_aborts(
                   RemoteReadExecutor::RunCursor{},
                   std::numeric_limits<uint64_t>::max() - 1, 2),
               "endpoint range end overflow remains fail-closed");

        RemoteReadExecutor::RunCursor overflow_cursor;
        overflow_cursor.completed_count =
            std::numeric_limits<uint64_t>::max();
        bool count_overflow = false;
        try {
            RemoteReadExecutor::mark_cursor_range_completed(
                overflow_cursor, 0, 1);
        } catch (const std::overflow_error&) {
            count_overflow = true;
        }
        expect(count_overflow && overflow_cursor.completed_prefix == 0 &&
                   overflow_cursor.completed_out_of_order.empty() &&
                   overflow_cursor.completed_count ==
                       std::numeric_limits<uint64_t>::max(),
               "completed-count overflow is detected before interval mutation");

        PipelineEndpointRun cancelled_run;
        cancelled_run.rank = 0;
        cancelled_run.group_ordinal = 0;
        cancelled_run.first_tile = 1;
        cancelled_run.tile_count = 4;
        cancelled_run.shard_ordinals = {0};
        cancelled_run.legs.push_back(
            PipelineEndpointLegRun{PipelineEndpointLeg::DataSend, 0});
        RemoteReadExecutor::RunCursor cancelled_cursor;
        cancelled_cursor.run = &cancelled_run;
        cancelled_cursor.completed_prefix = 1;
        cancelled_cursor.completed_count = 1;
        cancelled_cursor.cancelled_endpoint_count = 2;
        cancelled_cursor.in_flight_count = 1;
        cancelled_cursor.active_store_ids.emplace(2, 302);
        RemoteReadExecutor::RuntimeDescriptor closed_owner;
        closed_owner.descriptor = descriptor;
        closed_owner.request_closed = true;
        closed_owner.runs.push_back(cancelled_cursor);
        const nlohmann::json closed_obligation =
            executor.obligation_row_json(closed_owner, 0);
        expect(closed_obligation.at("remaining_endpoint_count") == 1 &&
                   closed_owner.runs[0].completed_count == 1 &&
                   closed_owner.runs[0].cancelled_endpoint_count == 2,
               "request-close cancellations remain separate from completion count");
        std::printf(
            "[cursor-completed-count-validation] late=exact gap=exact "
            "duplicate=reject overlap=reject range_overflow=reject "
            "count_overflow_pre_mutation=true cancellation_separate=true\n");
    }

    static void run_ordinary_terminal_candidate_membership() {
        constexpr size_t kFixedPointQueries = 4;
        for (const size_t node_count : {size_t{8}, size_t{32}, size_t{128}}) {
            std::vector<ParsedNode> parsed_nodes(node_count);
            std::unordered_map<uint64_t, const ParsedNode*> nodes_by_id;
            nodes_by_id.reserve(node_count);
            for (size_t index = 0; index < node_count; ++index) {
                parsed_nodes[index].json_id = 100 + index * 2;
                nodes_by_id.emplace(parsed_nodes[index].json_id,
                                    &parsed_nodes[index]);
            }

            uint64_t point_probes = 0;
            for (size_t query = 0; query < kFixedPointQueries; ++query) {
                const size_t node_index =
                    query * node_count / kFixedPointQueries;
                const uint64_t matches =
                    RemoteReadExecutor::count_ordinary_terminal_candidate_matches(
                        nodes_by_id, parsed_nodes[node_index].json_id,
                        /*count=*/1, /*stride=*/1, &point_probes);
                expect(matches == 1,
                       "ordinary point terminal resolves by candidate ID");
            }
            expect(point_probes == kFixedPointQueries,
                   "fixed point-query count probes independently of batch V");

            uint64_t ap_probes = 0;
            const uint64_t ap_count = node_count / 2;
            expect(RemoteReadExecutor::count_ordinary_terminal_candidate_matches(
                       nodes_by_id, 100, ap_count, /*stride=*/4,
                       &ap_probes) == ap_count &&
                       ap_probes == ap_count,
                   "sparse ordinary AP preserves exact candidate count and stride");

            uint64_t zero_stride_probes = 0;
            expect(RemoteReadExecutor::count_ordinary_terminal_candidate_matches(
                       nodes_by_id, 100, node_count, /*stride=*/0,
                       &zero_stride_probes) == 1 &&
                       zero_stride_probes == 1,
                   "zero stride retains one unique terminal and one lookup");

            uint64_t missing_point_probes = 0;
            expect(RemoteReadExecutor::count_ordinary_terminal_candidate_matches(
                       nodes_by_id, 100 + node_count * 2 + 7,
                       /*count=*/1, /*stride=*/0,
                       &missing_point_probes) == 0 &&
                       missing_point_probes == 1,
                   "missing stride-zero terminal remains one exact point lookup");
            uint64_t oversized_run_probes = 0;
            expect(RemoteReadExecutor::count_ordinary_terminal_candidate_matches(
                       nodes_by_id, 100, node_count + 1, /*stride=*/2,
                       &oversized_run_probes) == 0 &&
                       oversized_run_probes == 0,
                   "impossible AP cardinality fails without expanding its IDs");

            std::printf(
                "[ordinary-terminal-candidate-scale] V=%zu Q=%zu "
                "point_membership_probes=%llu sparse_ap_probes=%llu "
                "zero_stride_probes=%llu missing_point_probes=%llu "
                "oversized_ap_probes=%llu\n",
                node_count, kFixedPointQueries,
                static_cast<unsigned long long>(point_probes),
                static_cast<unsigned long long>(ap_probes),
                static_cast<unsigned long long>(zero_stride_probes),
                static_cast<unsigned long long>(missing_point_probes),
                static_cast<unsigned long long>(oversized_run_probes));
        }
    }

    static void run_dependency_interval_index(
        RemoteReadExecutor& executor, const uint64_t starting_tick) {
        using Match = RemoteReadExecutor::DependencyIndexMatch;
        const auto same_matches = [](const std::vector<Match>& left,
                                     const std::vector<Match>& right) {
            if (left.size() != right.size()) return false;
            for (size_t index = 0; index < left.size(); ++index) {
                if (left[index].dependency_run_index !=
                        right[index].dependency_run_index ||
                    left[index].local_shard_index !=
                        right[index].local_shard_index) {
                    return false;
                }
            }
            return true;
        };

        for (const size_t unrelated_count :
             {size_t{8}, size_t{32}, size_t{128}}) {
            RemoteReadExecutor::RuntimeDescriptor owner;
            std::vector<PipelineEndpointDependencyRun> dependencies;
            dependencies.reserve(unrelated_count + 6);
            const auto add_shards = [&owner, &dependencies](
                const int rank, const uint64_t group,
                const PipelineEndpointLeg endpoint,
                const std::vector<uint64_t>& shards,
                const uint64_t first_tile, const uint64_t tile_count,
                const uint64_t first_store_id,
                const uint64_t tile_stride,
                const uint64_t shard_stride) {
                PipelineEndpointDependencyRun dependency;
                dependency.rank = rank;
                dependency.group_ordinal = group;
                dependency.first_tile = first_tile;
                dependency.tile_count = tile_count;
                dependency.shard_ordinals = shards;
                dependency.endpoint = endpoint;
                dependency.dependency_kind =
                    PipelineEndpointDependencyKind::OrdinaryNodeTerminal;
                dependency.first_dependency_json_id = first_store_id;
                dependency.dependency_tile_stride = tile_stride;
                dependency.dependency_shard_stride = shard_stride;
                dependencies.push_back(std::move(dependency));
                owner.dependency_runs.push_back(
                    RemoteReadExecutor::ResolvedDependencyRun{
                        &dependencies.back(), first_store_id});
            };
            const auto add = [&add_shards](
                const int rank, const uint64_t group,
                const PipelineEndpointLeg endpoint,
                const uint64_t shard, const uint64_t first_tile,
                const uint64_t tile_count, const uint64_t first_store_id,
                const uint64_t tile_stride) {
                add_shards(rank, group, endpoint,
                           std::vector<uint64_t>{shard}, first_tile,
                           tile_count, first_store_id, tile_stride, 0);
            };

            // Declaration order intentionally differs from interval start
            // order. Both rows match tile 129 and must reach terminal checks
            // in declaration order, even though the tree visits [127,130]
            // before [129,130].
            add_shards(0, 7, PipelineEndpointLeg::DataSend, {2, 3},
                       129, 2, 2200, 0, 10);
            add_shards(0, 7, PipelineEndpointLeg::DataSend, {3, 4},
                       127, 4, 1100, 5, 0);
            for (size_t index = 0; index < unrelated_count; ++index) {
                const uint64_t first_tile = index < unrelated_count / 2
                    ? static_cast<uint64_t>(index + 1)
                    : static_cast<uint64_t>(130 +
                        (index - unrelated_count / 2));
                add(0, 7, PipelineEndpointLeg::DataSend, 3,
                    first_tile, 1, 5000 + index, 0);
            }
            // Neighboring static keys must not leak matches across rank,
            // group-local tile domain, endpoint leg, or shard identity.
            add(1, 7, PipelineEndpointLeg::DataSend, 3,
                129, 1, 8001, 0);
            add(0, 8, PipelineEndpointLeg::DataSend, 3,
                129, 1, 8002, 0);
            add(0, 7, PipelineEndpointLeg::ReleaseAckSend, 3,
                129, 1, 8003, 0);
            add(0, 7, PipelineEndpointLeg::DataSend, 4,
                129, 1, 8004, 0);

            RemoteReadExecutor::build_dependency_interval_indexes(
                owner.dependency_runs, &owner.dependency_interval_indexes);

            const auto old_scan = [&owner](
                const int rank, const uint64_t group,
                const PipelineEndpointLeg endpoint,
                const uint64_t tile_ordinal, const uint64_t shard_ordinal,
                uint64_t* rows_scanned) {
                std::vector<Match> matches;
                for (size_t dependency_index = 0;
                     dependency_index < owner.dependency_runs.size();
                     ++dependency_index) {
                    if (rows_scanned != nullptr) ++*rows_scanned;
                    const auto& dependency =
                        *owner.dependency_runs[dependency_index].run;
                    if (dependency.rank != rank ||
                        dependency.group_ordinal != group ||
                        dependency.endpoint != endpoint ||
                        tile_ordinal < dependency.first_tile ||
                        tile_ordinal - dependency.first_tile >=
                            dependency.tile_count) {
                        continue;
                    }
                    const auto shard = std::find(
                        dependency.shard_ordinals.begin(),
                        dependency.shard_ordinals.end(), shard_ordinal);
                    if (shard == dependency.shard_ordinals.end()) continue;
                    matches.push_back(Match{
                        dependency_index,
                        static_cast<size_t>(std::distance(
                            dependency.shard_ordinals.begin(), shard))});
                }
                return matches;
            };

            // Compare the real indexed query with the former row-by-row
            // predicate over several keys and group-local tile points. This
            // includes overlapping intervals, gaps, both ends of the sparse
            // ranges, and empty keys.
            for (const int rank : {0, 1}) {
                for (const uint64_t group : {uint64_t{7}, uint64_t{8}}) {
                    for (const PipelineEndpointLeg endpoint : {
                             PipelineEndpointLeg::DataSend,
                             PipelineEndpointLeg::ReleaseAckSend}) {
                        for (const uint64_t shard : {
                                 uint64_t{2}, uint64_t{3}, uint64_t{4}}) {
                            for (const uint64_t tile : {
                                     uint64_t{1}, uint64_t{64},
                                     uint64_t{127}, uint64_t{129},
                                     uint64_t{130}, uint64_t{193},
                                     uint64_t{194}}) {
                                const auto expected = old_scan(
                                    rank, group, endpoint, tile, shard,
                                    nullptr);
                                const auto actual =
                                    RemoteReadExecutor::matching_dependency_runs(
                                        owner, rank, group, endpoint, tile,
                                        shard);
                                expect(same_matches(actual, expected),
                                       "dependency interval query matches old ordered scan");
                                expect(RemoteReadExecutor::has_matching_dependency_run(
                                           owner, rank, group, endpoint, tile,
                                           shard) == !expected.empty(),
                                       "ReleaseAck parent-existence query matches old scan");
                            }
                        }
                    }
                }
            }

            uint64_t tree_nodes = 0;
            const auto indexed_matches =
                RemoteReadExecutor::matching_dependency_runs(
                    owner, 0, 7, PipelineEndpointLeg::DataSend, 129, 3,
                    &tree_nodes);
            uint64_t oracle_rows = 0;
            const auto scanned_matches = old_scan(
                0, 7, PipelineEndpointLeg::DataSend, 129, 3, &oracle_rows);
            expect(same_matches(indexed_matches, scanned_matches) &&
                       indexed_matches.size() == 2 &&
                       indexed_matches[0].dependency_run_index == 0 &&
                       indexed_matches[1].dependency_run_index == 1,
                   "overlapping matches preserve dependency declaration order");
            expect(tree_nodes < oracle_rows &&
                       oracle_rows == owner.dependency_runs.size(),
                   "indexed query visits fewer nodes than the old full scan");
            uint64_t indexed_memberships = 0;
            for (const auto& keyed_index :
                 owner.dependency_interval_indexes) {
                indexed_memberships += keyed_index.second.entries.size();
            }
            expect(indexed_memberships == owner.dependency_runs.size() + 2,
                   "dependency index stores shard memberships, not expanded tiles");

            PipelineEndpointRun endpoint_run;
            endpoint_run.rank = 0;
            endpoint_run.group_ordinal = 7;
            endpoint_run.first_tile = 129;
            endpoint_run.tile_count = 1;
            endpoint_run.shard_ordinals = {3};
            endpoint_run.legs.push_back(
                PipelineEndpointLegRun{PipelineEndpointLeg::DataSend, 0});
            RemoteReadExecutor::RunCursor cursor;
            cursor.run = &endpoint_run;
            owner.runs.push_back(cursor);
            RemoteReadExecutor::EndpointPosition position;
            position.run_index = 0;
            position.tile_offset = 0;
            position.shard_index = 0;
            position.leg_index = 0;
            const std::vector<uint64_t> parents =
                executor.endpoint_parent_store_ids(owner, position);
            expect(parents == std::vector<uint64_t>({1110, 2210}),
                   "parent IDs retain checked mapping and final numeric sort");
            expect(Sys::boostedTick() == starting_tick,
                   "dependency index scales and lookups do not advance simulated time");

            std::printf(
                "[endpoint-dependency-index-scale] unrelated=%zu rows=%zu "
                "shard_memberships=%llu fixed_matches=%zu "
                "linear_oracle_rows=%llu "
                "actual_tree_nodes=%llu tile_expansion=0\n",
                unrelated_count, owner.dependency_runs.size(),
                static_cast<unsigned long long>(indexed_memberships),
                indexed_matches.size(),
                static_cast<unsigned long long>(oracle_rows),
                static_cast<unsigned long long>(tree_nodes));
        }

        // A malformed-but-queryable overlap can map two different run rows
        // to the same parent ID. The index must report both rows in their
        // original order so endpoint_parent_store_ids' existing sorted
        // adjacent-duplicate fail-closed guard still observes the bad input.
        RemoteReadExecutor::RuntimeDescriptor duplicate_owner;
        std::vector<PipelineEndpointDependencyRun> duplicate_dependencies;
        duplicate_dependencies.reserve(2);
        const auto add_duplicate = [&duplicate_owner,
                                    &duplicate_dependencies](
            const uint64_t first_tile) {
            PipelineEndpointDependencyRun dependency;
            dependency.rank = 0;
            dependency.group_ordinal = 7;
            dependency.first_tile = first_tile;
            dependency.tile_count = 3;
            dependency.shard_ordinals = {3};
            dependency.endpoint = PipelineEndpointLeg::DataSend;
            dependency.first_dependency_json_id = 333;
            duplicate_dependencies.push_back(std::move(dependency));
            duplicate_owner.dependency_runs.push_back(
                RemoteReadExecutor::ResolvedDependencyRun{
                    &duplicate_dependencies.back(), 333});
        };
        add_duplicate(129);
        add_duplicate(127);
        duplicate_dependencies.back().dependency_kind =
            PipelineEndpointDependencyKind::VirtualEndpointTerminal;
        RemoteReadExecutor::build_dependency_interval_indexes(
            duplicate_owner.dependency_runs,
            &duplicate_owner.dependency_interval_indexes);
        const auto duplicate_matches =
            RemoteReadExecutor::matching_dependency_runs(
                duplicate_owner, 0, 7, PipelineEndpointLeg::DataSend,
                129, 3);
        std::vector<uint64_t> duplicate_parent_ids;
        for (const Match& match : duplicate_matches) {
            duplicate_parent_ids.push_back(
                duplicate_owner.dependency_runs[
                    match.dependency_run_index].first_store_id);
        }
        const bool has_duplicate_parent =
            RemoteReadExecutor::sort_and_detect_duplicate_dependency_parents(
                &duplicate_parent_ids);
        expect(duplicate_matches.size() == 2 &&
                   duplicate_matches[0].dependency_run_index == 0 &&
                   duplicate_matches[1].dependency_run_index == 1 &&
                   has_duplicate_parent &&
                   duplicate_parent_ids == std::vector<uint64_t>({333, 333}),
               "duplicate parent mappings remain visible to the existing fail-closed guard");
        expect(Sys::boostedTick() == starting_tick,
               "duplicate dependency mapping probe does not advance time");
    }

    static void clear_terminal_watchers(RemoteReadExecutor& executor) {
        for (size_t rank = 0;
             rank < executor.terminal_watchers_by_rank_.size(); ++rank) {
            executor.terminal_watchers_by_rank_[rank].clear();
            executor.terminal_watcher_indexes_by_rank_[rank] = {};
        }
        executor.producer_metrics_ = {};
    }

    static void run_terminal_watcher_index(
        RemoteReadExecutor& executor, const uint64_t starting_tick) {
        const std::vector<size_t> scales{8, 32, 128};
        for (const size_t watcher_count : scales) {
            clear_terminal_watchers(executor);
            std::vector<std::unique_ptr<RemoteReadExecutor::RuntimeDescriptor>>
                owners;
            owners.reserve(watcher_count);
            for (size_t index = 0; index < watcher_count; ++index) {
                auto owner = std::make_unique<
                    RemoteReadExecutor::RuntimeDescriptor>();
                owner->retired = true;
                RemoteReadExecutor::TerminalWatcher watcher;
                watcher.owner = owner.get();
                watcher.wake_rank = 0;
                watcher.kind =
                    PipelineEndpointDependencyKind::OrdinaryNodeTerminal;
                watcher.first_store_id = 2 + index * 10000001;
                watcher.tile_count = 1000000;
                watcher.tile_stride = 5;
                watcher.shard_offsets = {0};
                executor.install_terminal_watcher(0, std::move(watcher));
                owners.push_back(std::move(owner));
            }

            // Exercise the exact callback entry point with one terminal and
            // only unrelated retired watcher intervals at larger IDs.
            executor.notify_terminal_completed(
                0, 1, PipelineEndpointDependencyKind::OrdinaryNodeTerminal);
            const auto& metrics = executor.producer_metrics_;
            expect(metrics.terminal_watcher_index_entries_built ==
                       watcher_count &&
                   metrics.terminal_watcher_index_candidate_examinations == 0 &&
                   metrics.terminal_watcher_index_query_node_visits == 0,
                   "one unmatched terminal avoids unrelated watcher history");
            expect(metrics.terminal_watcher_index_insert_steps <
                       watcher_count * 16,
                   "terminal watcher registration is incremental AVL work");
            std::printf(
                "[terminal-watcher-scale] H=%zu entries_built=%llu "
                "insert_steps=%llu fixed_callbacks=1 query_node_visits=%llu "
                "candidate_examinations=%llu expanded_tiles=0\n",
                watcher_count,
                static_cast<unsigned long long>(
                    metrics.terminal_watcher_index_entries_built),
                static_cast<unsigned long long>(
                    metrics.terminal_watcher_index_insert_steps),
                static_cast<unsigned long long>(
                    metrics.terminal_watcher_index_query_node_visits),
                static_cast<unsigned long long>(
                    metrics.terminal_watcher_index_candidate_examinations));
            expect(Sys::boostedTick() == starting_tick,
                   "terminal watcher history scale does not advance time");
        }

        clear_terminal_watchers(executor);
        std::vector<std::unique_ptr<RemoteReadExecutor::RuntimeDescriptor>>
            owners;
        const auto add_watcher = [&executor, &owners](
            const int rank, const PipelineEndpointDependencyKind kind,
            const uint64_t first, const uint64_t tile_count,
            const uint64_t tile_stride, std::vector<uint64_t> shard_offsets,
            const bool is_consume) {
            auto owner = std::make_unique<
                RemoteReadExecutor::RuntimeDescriptor>();
            owner->retired = true;
            RemoteReadExecutor::TerminalWatcher watcher;
            watcher.owner = owner.get();
            watcher.wake_rank = rank;
            watcher.kind = kind;
            watcher.first_store_id = first;
            watcher.tile_count = tile_count;
            watcher.tile_stride = tile_stride;
            watcher.shard_offsets = std::move(shard_offsets);
            watcher.is_consume_completion = is_consume;
            executor.install_terminal_watcher(rank, std::move(watcher));
            owners.push_back(std::move(owner));
        };

        // Repeated shard offsets and duplicate mappings must remain separate
        // registration identities, with same-watcher obligations aggregated
        // in original shard order after the interval query.
        add_watcher(0,
                    PipelineEndpointDependencyKind::OrdinaryNodeTerminal,
                    100, 4, 0, {0, 0}, false);
        add_watcher(0,
                    PipelineEndpointDependencyKind::OrdinaryNodeTerminal,
                    100, 3, 4, {0, 8}, true);
        add_watcher(0,
                    PipelineEndpointDependencyKind::VirtualEndpointTerminal,
                    108, 3, 0, {0, 0}, true);
        add_watcher(0,
                    PipelineEndpointDependencyKind::OrdinaryNodeTerminal,
                    108, 2, 0, {0, 0}, true);
        add_watcher(0,
                    PipelineEndpointDependencyKind::OrdinaryNodeTerminal,
                    100, 1, 0, {0}, false);
        add_watcher(1,
                    PipelineEndpointDependencyKind::OrdinaryNodeTerminal,
                    0, 2, 0, {0, 0}, true);

        const auto ordinary_100 = executor.collect_terminal_watcher_matches(
            0, 100, PipelineEndpointDependencyKind::OrdinaryNodeTerminal);
        expect(ordinary_100.size() == 3 &&
                   ordinary_100[0].watcher_index == 0 &&
                   ordinary_100[0].completed_obligations == 2 &&
                   ordinary_100[1].watcher_index == 1 &&
                   ordinary_100[1].completed_obligations == 1 &&
                   ordinary_100[2].watcher_index == 4 &&
                   ordinary_100[2].completed_obligations == 1,
               "ordinary match order, duplicate watcher identity, and stride-zero shard multiplicity");
        const auto ordinary_108 = executor.collect_terminal_watcher_matches(
            0, 108, PipelineEndpointDependencyKind::OrdinaryNodeTerminal);
        expect(ordinary_108.size() == 2 &&
                   ordinary_108[0].watcher_index == 1 &&
                   ordinary_108[0].completed_obligations == 2 &&
                   ordinary_108[1].watcher_index == 3 &&
                   ordinary_108[1].completed_obligations == 4,
               "ordinary affine and stride-zero consume obligations preserve callback order");
        const auto virtual_108 = executor.collect_terminal_watcher_matches(
            0, 108,
            PipelineEndpointDependencyKind::VirtualEndpointTerminal);
        expect(virtual_108.size() == 1 &&
                   virtual_108[0].watcher_index == 2 &&
                   virtual_108[0].completed_obligations == 6,
               "virtual kind retains stride-zero tile and shard multiplicity");
        const auto rank_one_zero = executor.collect_terminal_watcher_matches(
            1, 0, PipelineEndpointDependencyKind::OrdinaryNodeTerminal);
        const auto rank_zero_zero = executor.collect_terminal_watcher_matches(
            0, 0, PipelineEndpointDependencyKind::OrdinaryNodeTerminal);
        expect(rank_one_zero.size() == 1 &&
                   rank_one_zero[0].watcher_index == 0 &&
                   rank_one_zero[0].completed_obligations == 4 &&
                   rank_zero_zero.empty(),
               "rank partition and zero-valued NodeStore ID queries are exact");
        std::printf(
            "[terminal-watcher-semantics] ordinary_100=3 ordered_hits "
            "ordinary_108=2 virtual_108=1 rank1_id0=1 stride0_multiplicity=exact\n");
        expect(Sys::boostedTick() == starting_tick,
               "terminal watcher semantic checks do not advance time");
        clear_terminal_watchers(executor);
    }

    static void run_obligation_patch_coalescing(
        RemoteReadExecutor& executor) {
        executor.pending_obligation_patches_.clear();
        executor.producer_metrics_ = {};
        const auto make_row = [](const size_t run_index,
                                 const std::string& state,
                                 const uint64_t remaining) {
            const std::string obligation_id =
                "synthetic-segment/run/" + std::to_string(run_index);
            return nlohmann::json{
                {"obligation_id", obligation_id},
                {"segment_id", "synthetic-segment"},
                {"rank", 0},
                {"first_store_id", 100 + run_index},
                {"remaining_endpoint_count", remaining},
                {"state", state},
            };
        };
        const std::vector<std::string> lexical_ids{
            "synthetic-segment/run/0", "synthetic-segment/run/1",
            "synthetic-segment/run/10", "synthetic-segment/run/2",
            "synthetic-segment/run/3", "synthetic-segment/run/4",
            "synthetic-segment/run/5", "synthetic-segment/run/6",
            "synthetic-segment/run/7", "synthetic-segment/run/8",
            "synthetic-segment/run/9"};

        // New descriptor declaration patches arrive in numeric run order,
        // while canonical order is the consumer's string obligation ID.
        for (size_t run_index = 0; run_index < 11; ++run_index) {
            executor.record_obligation_row_patch(
                nullptr, make_row(run_index, "pending", 7));
        }
        executor.flush_pending_obligation_row_patches();
        expect(executor.pending_obligation_patches_.size() == 11 &&
                   executor.producer_metrics_.obligation_patch_rows == 11,
               "descriptor declaration keeps all eleven initial obligation rows");
        for (size_t index = 0; index < lexical_ids.size(); ++index) {
            const auto& patch = executor.pending_obligation_patches_[index];
            expect(patch.at("before").is_null() &&
                       patch.at("after").at("obligation_id") ==
                           lexical_ids[index],
                   "declaration obligation patches sort by string ID");
        }

        // Close/tombstone rows can be recorded in run order but must be
        // emitted with the same lexical order as declaration and callbacks.
        executor.pending_obligation_patches_.clear();
        for (size_t run_index = 11; run_index-- > 0;) {
            executor.record_obligation_row_patch(
                make_row(run_index, "active", 1), nullptr);
        }
        executor.flush_pending_obligation_row_patches();
        expect(executor.pending_obligation_patches_.size() == 11 &&
                   executor.producer_metrics_.obligation_patch_rows == 22,
               "close tombstones preserve all eleven obligation identities");
        for (size_t index = 0; index < lexical_ids.size(); ++index) {
            const auto& patch = executor.pending_obligation_patches_[index];
            expect(patch.at("before").at("obligation_id") ==
                       lexical_ids[index] && patch.at("after").is_null(),
                   "close obligation patches sort by string ID");
        }

        // Within one logical transition, repeated touches retain the earliest
        // before and final after. A change that returns to its original row is
        // omitted as a no-op; an absent-to-present-to-absent row is too.
        executor.pending_obligation_patches_.clear();
        const nlohmann::json initial = make_row(2, "pending", 7);
        const nlohmann::json intermediate = make_row(2, "active", 4);
        const nlohmann::json final = make_row(2, "draining", 1);
        const nlohmann::json unchanged = make_row(3, "pending", 7);
        executor.record_obligation_row_patch(initial, intermediate);
        executor.record_obligation_row_patch(intermediate, final);
        executor.record_obligation_row_patch(
            unchanged, make_row(3, "active", 3));
        executor.record_obligation_row_patch(
            make_row(3, "active", 3), unchanged);
        executor.record_obligation_row_patch(
            nullptr, make_row(11, "pending", 7));
        executor.record_obligation_row_patch(
            make_row(11, "pending", 7), nullptr);
        executor.flush_pending_obligation_row_patches();
        expect(executor.pending_obligation_patches_.size() == 1 &&
                   executor.pending_obligation_patches_[0].at("before") ==
                       initial &&
                   executor.pending_obligation_patches_[0].at("after") ==
                       final &&
                   executor.producer_metrics_.obligation_patch_rows == 23,
               "transition obligation patches coalesce chains and omit no-ops");
        std::printf(
            "[obligation-patch-coalescing] declaration_rows=11 "
            "close_tombstones=11 lexical_run10=true repeated_key=1 "
            "noops_omitted=2 before_chain=valid\n");
        executor.pending_obligation_patches_.clear();
        executor.producer_metrics_ = {};
    }

    static void verify_domain_provenance(RemoteReadExecutor& executor,
                                         const uint64_t starting_tick) {
        const std::string domain_id = "test-open-request-reuse-domain";
        const int target_rank = 3;
        const std::pair<int, uint64_t> source_key{1, 7};
        auto& retired = executor.buffer_domains_[domain_id];
        retired.target_slots[0].generation = 17;
        retired.target_slots[1].generation = 23;
        auto& source_slots = retired.source_slots[source_key];
        source_slots[0].generation = 31;
        source_slots[1].generation = 37;
        retired.source_slot_released_through[source_key] = {41, 43};
        executor.buffer_domain_target_ranks_[domain_id] = target_rank;

        // Model quiescent retirement while the request remains open: the
        // large occupancy state goes away, compact source-axis provenance
        // stays, and a later descriptor recreates empty slot rows.
        executor.retain_buffer_domain_provenance(domain_id, retired,
                                                 target_rank);
        executor.buffer_domains_.erase(domain_id);
        executor.buffer_domain_target_ranks_.erase(domain_id);
        expect(executor.buffer_domains_.find(domain_id) ==
                   executor.buffer_domains_.end() &&
                   executor.buffer_domain_provenance_.at(domain_id)
                           .source_slot_generations.size() == 1,
               "quiescent reuse keeps only compact slot provenance");

        auto& recreated = executor.buffer_domains_[domain_id];
        executor.restore_buffer_domain_provenance(domain_id, recreated,
                                                  target_rank);
        executor.buffer_domain_target_ranks_[domain_id] = target_rank;
        auto& restored_source = recreated.source_slots[source_key];
        executor.restore_source_slot_provenance(domain_id, source_key,
                                                restored_source);
        const auto watermark =
            recreated.source_slot_released_through.find(source_key);
        expect(recreated.target_slots[0].generation == 17 &&
                   recreated.target_slots[1].generation == 23 &&
                   restored_source[0].generation == 31 &&
                   restored_source[1].generation == 37 &&
                   watermark != recreated.source_slot_released_through.end() &&
                   watermark->second[0] == 41 && watermark->second[1] == 43,
               "same-domain recreation restores slot generations and release watermarks");
        executor.buffer_domains_.erase(domain_id);
        executor.buffer_domain_target_ranks_.erase(domain_id);
        executor.buffer_domain_provenance_.erase(domain_id);
        std::printf("[buffer-domain-provenance] source_axes=1 target_generations=2 "
                    "source_slot_generations=2 watermarks=2 restored=exact\n");
        expect(Sys::boostedTick() == starting_tick,
               "buffer-domain provenance regression does not advance event time");
    }

    static void clear(RemoteReadExecutor& executor) {
        executor.endpoint_progressions_by_rank_.clear();
        executor.next_endpoint_index_order_ = 0;
        executor.producer_metrics_ = {};
    }

    static void verify_queries(
        RemoteReadExecutor& executor,
        const std::vector<
            std::unique_ptr<RemoteReadExecutor::RuntimeDescriptor>>& owners,
        const size_t count, const char* axis) {
        const uint64_t oldest = owners.front()->runs.front().first_store_id;
        const uint64_t middle =
            owners.at(count / 2)->runs.front().first_store_id + 3 * 4;
        const uint64_t newest =
            owners.back()->runs.front().first_store_id + 7 * 4;
        const auto retired_result = executor.locate_endpoint(0, oldest);
        const auto ordinary_result = executor.locate_endpoint(0, newest - 1);
        const auto middle_result = executor.locate_endpoint(0, middle);
        const auto future_result = executor.locate_endpoint(
            0, owners.back()->runs.front().first_store_id);
        expect(retired_result.has_value() &&
                   retired_result->owner == owners.front().get() &&
                   owners.front()->retired,
               std::string("retired endpoint provenance remains queryable on ") +
                   axis);
        expect(!ordinary_result.has_value(),
               std::string("ordinary/non-endpoint ID is not indexed on ") + axis);
        expect(middle_result.has_value() &&
                   middle_result->owner == owners.at(count / 2).get() &&
                   middle_result->tile_offset == 3,
               std::string("middle affine endpoint resolves on ") + axis);
        expect(future_result.has_value() &&
                   future_result->owner == owners.back().get() &&
                   future_result->tile_offset == 0 &&
                   !owners.back()->retired,
               std::string("future endpoint resolves on ") + axis);
    }

    static void print_scale(const RemoteReadExecutor& executor,
                            const size_t count, const char* axis) {
        const auto& m = executor.producer_metrics_;
        const size_t blocks = executor.endpoint_progressions_by_rank_.at(0)
                                  .blocks.size();
        std::printf(
            "[endpoint-index-scale] axis=%s H=%zu blocks=%zu fixed_queries=4 "
            "entries_built=%llu sort_comparisons=%llu tree_build_ops=%llu "
            "block_search_steps=%llu blocks_examined=%llu "
            "candidate_entries=%llu\n",
            axis, count, blocks,
            static_cast<unsigned long long>(m.endpoint_index_entries_built),
            static_cast<unsigned long long>(m.endpoint_index_sort_comparisons),
            static_cast<unsigned long long>(m.endpoint_index_tree_build_operations),
            static_cast<unsigned long long>(m.endpoint_index_block_search_steps),
            static_cast<unsigned long long>(m.endpoint_index_blocks_examined),
            static_cast<unsigned long long>(
                m.endpoint_index_candidate_examinations));
    }
};

}  // namespace AstraSim::ExecutionDriven

namespace {

using SourceCreditKey = std::tuple<std::string, int, uint64_t>;
using SourceSlotKey = std::tuple<std::string, int, uint64_t, size_t>;
using SlotPatchReplayKey =
    std::tuple<std::string, std::string, int,
               std::optional<uint64_t>, size_t>;

SlotPatchReplayKey slot_patch_key_from_row(const nlohmann::json& row) {
    const std::string role = row.at("role").get<std::string>();
    const nlohmann::json& shard_value = row.at("shard_ordinal");
    const std::optional<uint64_t> shard = shard_value.is_null()
        ? std::nullopt
        : std::optional<uint64_t>(shard_value.get<uint64_t>());
    return {row.at("buffer_domain_id").get<std::string>(), role,
            row.at("rank").get<int>(), shard,
            row.at("slot_index").get<size_t>()};
}

SourceCreditKey source_credit_key_from_row(const nlohmann::json& row) {
    return {row.at("buffer_domain_id").get<std::string>(),
            row.at("source_rank").get<int>(),
            row.at("shard_ordinal").get<uint64_t>()};
}

uint64_t source_occupancy(
    const std::map<SourceSlotKey, nlohmann::json>& slots,
    const SourceCreditKey& credit_key) {
    uint64_t occupied = 0;
    for (size_t slot_index = 0; slot_index < 2; ++slot_index) {
        const SourceSlotKey slot_key{
            std::get<0>(credit_key), std::get<1>(credit_key),
            std::get<2>(credit_key), slot_index};
        const auto found = slots.find(slot_key);
        if (found != slots.end() &&
            found->second.at("occupied").get<bool>()) {
            ++occupied;
        }
    }
    return occupied;
}

void verify_credit_patch_replay(const nlohmann::json& frames) {
    std::map<SlotPatchReplayKey, nlohmann::json> all_slots;
    std::map<SourceSlotKey, nlohmann::json> source_slots;
    std::map<SourceCreditKey, nlohmann::json> credits;
    std::map<SlotPatchReplayKey, std::tuple<uint64_t, uint64_t, uint64_t>>
        target_lineages;
    size_t transition_count = 0;
    size_t slot_patch_count = 0;
    size_t credit_patch_count = 0;
    size_t availability_change_count = 0;
    size_t coalesced_ack_and_occupancy_count = 0;
    for (const auto& frame : frames) {
        for (const auto& transition : frame.at("pipeline_transition_runs")) {
            ++transition_count;
            std::map<SourceCreditKey, uint64_t> occupancy_before;
            std::set<SourceCreditKey> changed_source_credits;
            std::set<SlotPatchReplayKey> patched_slot_keys;
            std::optional<SlotPatchReplayKey> previous_slot_key;
            for (const auto& patch : transition.at("slot_patches")) {
                if (patch.is_object() && patch.contains("form") &&
                    patch.at("form") == "target_delta") {
                    const auto& key_row = patch.at("key");
                    const SlotPatchReplayKey slot_key{
                        key_row.at("buffer_domain_id").get<std::string>(),
                        "target", key_row.at("rank").get<int>(),
                        std::nullopt,
                        key_row.at("slot_index").get<size_t>()};
                    expect(key_row.at("role") == "target" &&
                               key_row.at("shard_ordinal").is_null(),
                           "target delta has the exact typed target key");
                    expect(!previous_slot_key.has_value() ||
                               *previous_slot_key < slot_key,
                           "target deltas share stable slot-patch ordering");
                    previous_slot_key = slot_key;
                    expect(patched_slot_keys.insert(slot_key).second,
                           "one target delta is emitted per key and transition");
                    const auto lineage = target_lineages.find(slot_key);
                    const uint64_t known_generation =
                        lineage == target_lineages.end()
                            ? 0 : std::get<0>(lineage->second);
                    const uint64_t known_sequence =
                        lineage == target_lineages.end()
                            ? 0 : std::get<1>(lineage->second);
                    const uint64_t known_count =
                        lineage == target_lineages.end()
                            ? 0 : std::get<2>(lineage->second);
                    expect(patch.at("before_slot_generation") ==
                               known_generation &&
                               patch.at("predecessor_logical_seq") ==
                                   known_sequence &&
                               patch.at("predecessor_consumed_count") ==
                                   known_count,
                           "target delta predecessor generation/sequence/count replay exactly");
                    const auto slot_it = all_slots.find(slot_key);
                    std::optional<nlohmann::json> row =
                        slot_it == all_slots.end()
                            ? std::nullopt
                            : std::optional<nlohmann::json>(slot_it->second);
                    uint64_t generation = known_generation;
                    for (const auto& operation : patch.at("operations")) {
                        const std::string& op =
                            operation.at("op").get_ref<const std::string&>();
                        const uint64_t op_generation =
                            operation.at("slot_generation").get<uint64_t>();
                        if (op == "initialize") {
                            expect(!row.has_value() &&
                                       op_generation == generation,
                                   "target initialize preserves absent-row generation");
                            row = nlohmann::json{
                                {"buffer_domain_id", std::get<0>(slot_key)},
                                {"role", "target"},
                                {"rank", std::get<2>(slot_key)},
                                {"shard_ordinal", nullptr},
                                {"slot_index", std::get<4>(slot_key)},
                                {"occupied", false}, {"owner", nullptr},
                                {"consumed", false},
                                {"target_shards_to_consume",
                                 nlohmann::json::array()},
                                {"consumed_target_shards",
                                 nlohmann::json::array()},
                            };
                        } else if (op == "claim") {
                            expect(row.has_value() &&
                                       !row->at("occupied").get<bool>() &&
                                       generation !=
                                           std::numeric_limits<uint64_t>::max() &&
                                       op_generation == generation + 1,
                                   "target claim advances one empty slot incarnation");
                            generation = op_generation;
                            (*row)["occupied"] = true;
                            (*row)["owner"] = operation.at("owner");
                            (*row)["consumed"] = false;
                            (*row)["target_shards_to_consume"] =
                                operation.at("target_shards");
                            (*row)["consumed_target_shards"] =
                                nlohmann::json::array();
                        } else if (op == "consume") {
                            expect(row.has_value() &&
                                       row->at("occupied").get<bool>() &&
                                       op_generation == generation,
                                   "target consume retains the occupied slot generation");
                            std::set<uint64_t> consumed;
                            for (const auto& member :
                                 row->at("consumed_target_shards")) {
                                consumed.insert(member.get<uint64_t>());
                            }
                            for (const auto& member : operation.at("add_shards")) {
                                expect(consumed.insert(
                                           member.get<uint64_t>()).second,
                                       "consume delta contains only newly inserted raw members");
                            }
                            std::set<uint64_t> targets;
                            for (const auto& member :
                                 row->at("target_shards_to_consume")) {
                                targets.insert(member.get<uint64_t>());
                            }
                            uint64_t remaining = 0;
                            for (const uint64_t target : targets) {
                                if (consumed.find(target) == consumed.end()) {
                                    ++remaining;
                                }
                            }
                            const bool cleared = remaining == 0;
                            expect(operation.at("cleared").get<bool>() ==
                                       cleared,
                                   "consume cleared bit matches exact target membership");
                            if (cleared) {
                                row = nlohmann::json{
                                    {"buffer_domain_id", std::get<0>(slot_key)},
                                    {"role", "target"},
                                    {"rank", std::get<2>(slot_key)},
                                    {"shard_ordinal", nullptr},
                                    {"slot_index", std::get<4>(slot_key)},
                                    {"occupied", false}, {"owner", nullptr},
                                    {"consumed", false},
                                    {"target_shards_to_consume",
                                     nlohmann::json::array()},
                                    {"consumed_target_shards",
                                     nlohmann::json::array()},
                                };
                            } else {
                                expect(std::includes(
                                           targets.begin(), targets.end(),
                                           consumed.begin(), consumed.end()),
                                       "coalesced occupied row retains the legacy consumed subset");
                                (*row)["consumed"] = !consumed.empty();
                                (*row)["consumed_target_shards"] =
                                    nlohmann::json::array();
                                for (const uint64_t member : consumed) {
                                    (*row)["consumed_target_shards"]
                                        .push_back(member);
                                }
                            }
                        } else if (op == "request_close_clear") {
                            expect(row.has_value() &&
                                       row->at("occupied").get<bool>() &&
                                       op_generation == generation,
                                   "request close clears a live target incarnation");
                            row = nlohmann::json{
                                {"buffer_domain_id", std::get<0>(slot_key)},
                                {"role", "target"},
                                {"rank", std::get<2>(slot_key)},
                                {"shard_ordinal", nullptr},
                                {"slot_index", std::get<4>(slot_key)},
                                {"occupied", false}, {"owner", nullptr},
                                {"consumed", false},
                                {"target_shards_to_consume",
                                 nlohmann::json::array()},
                                {"consumed_target_shards",
                                 nlohmann::json::array()},
                            };
                        } else if (op == "erase") {
                            expect(row.has_value() &&
                                       !row->at("occupied").get<bool>() &&
                                       op_generation == generation &&
                                       transition.at("transition_kind") ==
                                           "descriptor_retire",
                                   "target row erase is limited to empty retired rows");
                            row.reset();
                        } else {
                            fail("unknown target delta operation in CPP replay");
                        }
                    }
                    if (row.has_value()) {
                        all_slots[slot_key] = *row;
                    } else {
                        all_slots.erase(slot_key);
                    }
                    const uint64_t consumed_count =
                        row.has_value() && row->at("occupied").get<bool>()
                            ? row->at("consumed_target_shards").size() : 0;
                    target_lineages[slot_key] = {
                        generation,
                        transition.at("first_logical_seq").get<uint64_t>(),
                        consumed_count};
                    ++slot_patch_count;
                    continue;
                }
                const nlohmann::json& before = patch.at("before");
                const nlohmann::json& after = patch.at("after");
                const nlohmann::json& identity =
                    before.is_null() ? after : before;
                expect(!identity.is_null() && before != after,
                       "slot patch has identity and changes its row");
                const SlotPatchReplayKey slot_key =
                    slot_patch_key_from_row(identity);
                expect(!previous_slot_key.has_value() ||
                           *previous_slot_key < slot_key,
                       "slot patches are sorted and unique by typed stable key");
                previous_slot_key = slot_key;
                expect(patched_slot_keys.insert(slot_key).second,
                       "one coalesced slot patch is emitted per key and transition");
                if (!before.is_null() && !after.is_null()) {
                    expect(slot_patch_key_from_row(after) == slot_key,
                           "slot patch preserves its stable row key");
                }
                const auto prior_slot = all_slots.find(slot_key);
                expect(before.is_null()
                           ? prior_slot == all_slots.end()
                           : prior_slot != all_slots.end() &&
                                 prior_slot->second == before,
                       "slot patch before row matches sparse replay state");
                if (after.is_null()) {
                    all_slots.erase(slot_key);
                } else {
                    all_slots[slot_key] = after;
                }
                ++slot_patch_count;
                if (identity.at("role").get<std::string>() != "source") continue;
                const SourceCreditKey credit_key{
                    identity.at("buffer_domain_id").get<std::string>(),
                    identity.at("rank").get<int>(),
                    identity.at("shard_ordinal").get<uint64_t>()};
                occupancy_before.try_emplace(
                    credit_key, source_occupancy(source_slots, credit_key));
                changed_source_credits.insert(credit_key);
                const SourceSlotKey source_slot_key{
                    std::get<0>(credit_key), std::get<1>(credit_key),
                    std::get<2>(credit_key),
                    identity.at("slot_index").get<size_t>()};
                const auto current = source_slots.find(source_slot_key);
                expect(before.is_null()
                           ? current == source_slots.end()
                           : current != source_slots.end() &&
                                 current->second == before,
                       "source slot patch before row matches sparse replay state");
                if (after.is_null()) {
                    source_slots.erase(source_slot_key);
                } else {
                    source_slots[source_slot_key] = after;
                }
            }

            std::set<SourceCreditKey> patched_credit_keys;
            std::optional<SourceCreditKey> previous_key;
            for (const auto& patch : transition.at("credit_patches")) {
                const nlohmann::json& before = patch.at("before");
                const nlohmann::json& after = patch.at("after");
                expect(!before.is_null() || !after.is_null(),
                       "credit patch has a stable row identity");
                const SourceCreditKey key = source_credit_key_from_row(
                    before.is_null() ? after : before);
                expect(!previous_key.has_value() || *previous_key < key,
                       "credit patches are unique and sorted by stable key");
                previous_key = key;
                expect(patched_credit_keys.insert(key).second,
                       "one coalesced credit patch is emitted per key and transition");
                const auto current = credits.find(key);
                expect(before.is_null()
                           ? current == credits.end()
                           : current != credits.end() &&
                                 current->second == before,
                       "credit patch before row matches sparse replay state");
                const std::string& transition_kind =
                    transition.at("transition_kind").get_ref<const std::string&>();
                if (after.is_null()) {
                    expect(source_occupancy(source_slots, key) == 0 &&
                               before.at("outstanding_ack_count").get<uint64_t>() == 0,
                           "credit tombstone has no occupied slot or outstanding ACK");
                    credits.erase(key);
                } else {
                    const uint64_t capacity =
                        after.at("capacity").get<uint64_t>();
                    const uint64_t occupied =
                        source_occupancy(source_slots, key);
                    expect(occupied <= capacity &&
                               after.at("available").get<uint64_t>() ==
                                   capacity - occupied,
                           "credit available equals capacity minus source occupancy");
                    if (!before.is_null() &&
                        before.at("available").get<uint64_t>() !=
                            after.at("available").get<uint64_t>() &&
                        before.at("outstanding_ack_count").get<uint64_t>() !=
                            after.at("outstanding_ack_count").get<uint64_t>()) {
                        ++coalesced_ack_and_occupancy_count;
                    }
                    if (!before.is_null()) {
                        const uint64_t old_acks =
                            before.at("outstanding_ack_count").get<uint64_t>();
                        const uint64_t new_acks =
                            after.at("outstanding_ack_count").get<uint64_t>();
                        if (new_acks == old_acks + 1) {
                            expect(transition_kind == "release_ack_progress",
                                   "credit ACK increment belongs to release_ack_send");
                        } else if (old_acks == new_acks + 1) {
                            expect(transition_kind == "endpoint_done",
                                   "credit ACK decrement belongs to release_ack_recv completion");
                        } else {
                            expect(new_acks == old_acks,
                                   "one endpoint transition changes at most one ACK credit");
                        }
                    }
                    credits[key] = after;
                }
                ++credit_patch_count;
            }
            for (const auto& entry : occupancy_before) {
                const uint64_t after_occupied =
                    source_occupancy(source_slots, entry.first);
                if (entry.second == after_occupied) continue;
                ++availability_change_count;
                expect(patched_credit_keys.find(entry.first) !=
                           patched_credit_keys.end(),
                       "every source occupancy change carries its credit row patch");
                const auto current = credits.find(entry.first);
                expect(current != credits.end() &&
                           current->second.at("available").get<uint64_t>() ==
                               2 - after_occupied,
                       "replayed credit availability reflects final transition occupancy");
            }
        }
    }
    std::printf("[slot-credit-patch-replay] transitions=%zu slot_patches=%zu "
                "credit_patches=%zu "
                "source_occupancy_changes=%zu coalesced_ack_occupancy=%zu\n",
                transition_count, slot_patch_count, credit_patch_count,
                availability_change_count,
                coalesced_ack_and_occupancy_count);
    expect(transition_count > 0 && slot_patch_count > 0 &&
               credit_patch_count > 0 &&
               availability_change_count > 0 &&
               coalesced_ack_and_occupancy_count > 0,
           "real multi-tile fixture exercises sparse credit availability patches");
}

// ---------------------------------------------------------------------------
// Wide-run coalescing mode (--wide-runs).
//
// The default fixture emits one endpoint run and one count-1 tag allocation
// per (tile, shard, flow family), mirroring the legacy per-tile emission
// shape.  The online producer now coalesces deferred per-tile extents into
// wide tile ranges (joint/compact_pipeline.py::_coalesce_affine_runs), so the
// executor must consume runs whose tile_count spans a whole release chain.
// This test-only transform rebuilds the same fixture wires in that wide
// shape: every (rank, group, shard, endpoint) chain whose per-tile IDs form
// one increasing arithmetic progression merges into one run, and the
// per-family tag entries are renumbered into disjoint per-chain spans (the
// fixture's original count-1 entries interleave families, so they cannot be
// absorbed in place the way the producer's contiguous per-emission spans
// are; tag VALUES change but stay unique and consistently paired because
// both legs of a transfer family only merge across a tile step together,
// resolving through the same merged entry and affine formula).
//
// Physical equivalence contract of the mode: every behavioral assertion of
// the default run (transfer schedule, flow count, ACK-before-data ordering,
// delayed-consumer ordering, marker cost, peaks, drain, settle) must pass
// unchanged, and service/settle/final ticks must match the fragmented run.
// ---------------------------------------------------------------------------
struct WideChainKey {
    int rank;
    uint64_t group_ordinal;
    uint64_t shard_ordinal;
    std::string endpoint;
    bool operator<(const WideChainKey& other) const {
        return std::tie(rank, group_ordinal, shard_ordinal, endpoint) <
               std::tie(other.rank, other.group_ordinal, other.shard_ordinal,
                        other.endpoint);
    }
};

nlohmann::json* tag_entry_by_ordinal(nlohmann::json& tag_runs,
                                     const uint64_t emission_ordinal) {
    for (auto& entry : tag_runs) {
        if (entry.at("emission_ordinal").get<uint64_t>() == emission_ordinal) {
            return &entry;
        }
    }
    return nullptr;
}

void coalesce_wide_descriptor(nlohmann::json& wire) {
    nlohmann::json& runs = wire.at("endpoint_runs");
    nlohmann::json& tag_runs = wire.at("tag_allocation_runs");
    std::map<WideChainKey, std::vector<size_t>> chains;
    for (size_t index = 0; index < runs.size(); ++index) {
        nlohmann::json& run = runs.at(index);
        if (run.at("legs").size() != 1) {
            fail("wide-runs fixture expects single-leg endpoint runs");
        }
        WideChainKey key{
            run.at("rank").get<int>(),
            run.at("group_ordinal").get<uint64_t>(),
            run.at("shard_ordinals").at(0).get<uint64_t>(),
            run.at("legs").at(0).at("endpoint").get<std::string>()};
        chains[key].push_back(index);
    }
    // Per chain: how many leading tile steps are individually mergeable
    // (tile-adjacent, positive constant endpoint-ID stride, positive
    // constant tag stride through the shared family entries).  The fixture
    // assigns endpoint IDs sequentially per event, so the per-tile spacing
    // of one chain is the emitted cycle width, not the declared per-run
    // stride; a chain merges only when its IDs form one arithmetic
    // progression across the adjacent tiles.
    struct ChainPlan {
        std::vector<size_t> members;
        std::vector<bool> step_ok;
        uint64_t id_stride = 0;
        uint64_t tag_stride = 0;
        bool tagged = false;
        uint64_t head_ordinal = 0;
    };
    std::vector<ChainPlan> plans;
    for (auto& [key, members] : chains) {
        std::sort(members.begin(), members.end(),
                  [&runs](const size_t left, const size_t right) {
                      return runs.at(left).at("first_tile").get<uint64_t>() <
                             runs.at(right).at("first_tile").get<uint64_t>();
                  });
        ChainPlan plan;
        plan.members = members;
        const nlohmann::json& head_leg = runs.at(members.front()).at("legs").at(0);
        plan.tagged = !head_leg.at("tag_emission_ordinal").is_null();
        if (plan.tagged) {
            plan.head_ordinal =
                head_leg.at("tag_emission_ordinal").get<uint64_t>();
        }
        bool have_strides = false;
        for (size_t position = 1; position < members.size(); ++position) {
            const nlohmann::json& survivor = runs.at(members.at(position - 1));
            const nlohmann::json& candidate = runs.at(members.at(position));
            const bool tile_adjacent =
                candidate.at("first_tile").get<uint64_t>() ==
                survivor.at("first_tile").get<uint64_t>() +
                    survivor.at("tile_count").get<uint64_t>();
            const int64_t id_delta_signed =
                static_cast<int64_t>(
                    candidate.at("first_json_id").get<uint64_t>()) -
                static_cast<int64_t>(
                    survivor.at("first_json_id").get<uint64_t>());
            const uint64_t id_delta = id_delta_signed > 0
                ? static_cast<uint64_t>(id_delta_signed) : 0;
            bool ok = tile_adjacent && id_delta > 0 &&
                (!have_strides || id_delta == plan.id_stride);
            if (ok && plan.tagged) {
                const nlohmann::json& survivor_leg = survivor.at("legs").at(0);
                const nlohmann::json& candidate_leg =
                    candidate.at("legs").at(0);
                const nlohmann::json* survivor_entry = tag_entry_by_ordinal(
                    tag_runs,
                    survivor_leg.at("tag_emission_ordinal").get<uint64_t>());
                const nlohmann::json* candidate_entry = tag_entry_by_ordinal(
                    tag_runs,
                    candidate_leg.at("tag_emission_ordinal").get<uint64_t>());
                const int64_t tag_delta_signed =
                    survivor_entry == nullptr || candidate_entry == nullptr
                        ? 0
                        : static_cast<int64_t>(
                              candidate_entry->at("first_tag").get<uint32_t>())
                          - static_cast<int64_t>(
                              survivor_entry->at("first_tag").get<uint32_t>());
                const uint64_t tag_delta = tag_delta_signed > 0
                    ? static_cast<uint64_t>(tag_delta_signed) : 0;
                ok = tag_delta > 0 &&
                    (!have_strides || tag_delta == plan.tag_stride) &&
                    candidate_leg.at("tag_offset").get<uint64_t>() ==
                        survivor_leg.at("tag_offset").get<uint64_t>();
                if (ok) {
                    plan.tag_stride = tag_delta;
                }
            }
            plan.step_ok.push_back(ok);
            if (ok) {
                plan.id_stride = id_delta;
                have_strides = true;
            } else {
                break;
            }
        }
        plans.push_back(plan);
    }
    // Tag entries are shared by the send and receive legs of one transfer
    // family, so a tile step may only be absorbed when every chain of that
    // family merges across it; otherwise the surviving partner leg would
    // reference a deleted allocation with a diverging tag value.
    struct FamilyKey {
        uint64_t group_ordinal;
        uint64_t shard_ordinal;
        std::string family;
        bool operator<(const FamilyKey& other) const {
            return std::tie(group_ordinal, shard_ordinal, family) <
                   std::tie(other.group_ordinal, other.shard_ordinal,
                            other.family);
        }
    };
    std::map<FamilyKey, uint64_t> family_allowed_steps;
    for (const ChainPlan& plan : plans) {
        if (!plan.tagged) {
            continue;
        }
        const nlohmann::json& head = runs.at(plan.members.front());
        FamilyKey family{
            head.at("group_ordinal").get<uint64_t>(),
            head.at("shard_ordinals").at(0).get<uint64_t>(),
            flow_family(head.at("legs").at(0).at("endpoint")
                            .get<std::string>())};
        uint64_t allowed = 0;
        while (allowed < plan.step_ok.size() && plan.step_ok.at(allowed)) {
            ++allowed;
        }
        const auto prior = family_allowed_steps.find(family);
        if (prior == family_allowed_steps.end()) {
            family_allowed_steps.emplace(family, allowed);
        } else {
            prior->second = std::min(prior->second, allowed);
        }
    }
    std::set<size_t> absorbed_runs;
    std::set<uint64_t> absorbed_ordinals;
    std::map<uint64_t, uint64_t> head_tiles;   // head ordinal -> merged tiles
    std::map<uint64_t, uint64_t> head_tag_stride;
    for (const ChainPlan& plan : plans) {
        uint64_t allowed = 0;
        if (plan.tagged) {
            const nlohmann::json& head = runs.at(plan.members.front());
            const FamilyKey family{
                head.at("group_ordinal").get<uint64_t>(),
                head.at("shard_ordinals").at(0).get<uint64_t>(),
                flow_family(head.at("legs").at(0).at("endpoint")
                                .get<std::string>())};
            allowed = family_allowed_steps.at(family);
        } else {
            while (allowed < plan.step_ok.size() && plan.step_ok.at(allowed)) {
                ++allowed;
            }
        }
        if (allowed == 0) {
            continue;
        }
        nlohmann::json& head = runs.at(plan.members.front());
        uint64_t chain_tiles =
            head.at("tile_count").get<uint64_t>();
        for (size_t position = 1; position <= allowed; ++position) {
            const nlohmann::json& candidate = runs.at(plan.members.at(position));
            chain_tiles += candidate.at("tile_count").get<uint64_t>();
            absorbed_runs.insert(plan.members.at(position));
            const nlohmann::json& candidate_leg =
                candidate.at("legs").at(0);
            if (!candidate_leg.at("tag_emission_ordinal").is_null()) {
                absorbed_ordinals.insert(
                    candidate_leg.at("tag_emission_ordinal").get<uint64_t>());
            }
        }
        head["tile_count"] = chain_tiles;
        head["tile_stride"] = plan.id_stride;
        if (plan.tagged) {
            head_tiles.emplace(plan.head_ordinal, chain_tiles);
            head_tag_stride.emplace(plan.head_ordinal, plan.tag_stride);
        }
    }
    // Renumber the kept tag entries into disjoint spans in allocator order.
    uint32_t next_tag = 1;
    for (auto& entry : tag_runs) {
        const uint64_t ordinal = entry.at("emission_ordinal").get<uint64_t>();
        if (absorbed_ordinals.count(ordinal) != 0) {
            continue;
        }
        const auto tiles = head_tiles.find(ordinal);
        if (tiles != head_tiles.end()) {
            const uint64_t stride = head_tag_stride.at(ordinal);
            const uint64_t count = (tiles->second - 1) * stride + 1;
            entry["first_tag"] = next_tag;
            entry["count"] = count;
            entry["tile_count"] = tiles->second;
            next_tag += static_cast<uint32_t>(count);
        } else {
            const uint32_t count = entry.at("count").get<uint32_t>();
            entry["first_tag"] = next_tag;
            next_tag += count;
        }
    }
    nlohmann::json kept_tag_runs = nlohmann::json::array();
    for (auto& entry : tag_runs) {
        if (absorbed_ordinals.count(
                entry.at("emission_ordinal").get<uint64_t>()) != 0) {
            continue;
        }
        kept_tag_runs.push_back(entry);
    }
    tag_runs = kept_tag_runs;
    // Drop absorbed runs and give every surviving tagged leg its chain's
    // wide affine tag mapping.
    nlohmann::json kept_runs = nlohmann::json::array();
    for (size_t index = 0; index < runs.size(); ++index) {
        if (absorbed_runs.count(index) != 0) {
            continue;
        }
        kept_runs.push_back(runs.at(index));
    }
    runs = kept_runs;
    for (auto& run : runs) {
        nlohmann::json& leg = run.at("legs").at(0);
        if (leg.at("tag_emission_ordinal").is_null()) {
            continue;
        }
        const auto stride = head_tag_stride.find(
            leg.at("tag_emission_ordinal").get<uint64_t>());
        if (stride != head_tag_stride.end()) {
            leg["tag_offset"] = 0;
            leg["tag_tile_stride"] = stride->second;
        }
        leg["tag_shard_stride"] = 1;
    }
    // Recompute physical anchor tag selectors from the merged legs.
    for (auto& anchor : wire.at("anchors")) {
        nlohmann::json& selector = anchor.at("selector");
        const std::string kind = selector.at("kind").get<std::string>();
        if (kind != "tile_endpoint_leg" && kind != "tile_endpoint_terminal") {
            continue;
        }
        const uint64_t tile = selector.at("tile_ordinal").get<uint64_t>();
        const uint64_t shard = selector.at("shard_ordinal").get<uint64_t>();
        const std::string endpoint = selector.at("endpoint").get<std::string>();
        bool resolved = false;
        for (const auto& run : runs) {
            if (run.at("group_ordinal").get<uint64_t>() !=
                    selector.at("group_ordinal").get<uint64_t>() ||
                tile < run.at("first_tile").get<uint64_t>() ||
                tile >= run.at("first_tile").get<uint64_t>() +
                            run.at("tile_count").get<uint64_t>() ||
                run.at("shard_ordinals").at(0).get<uint64_t>() != shard ||
                run.at("legs").at(0).at("endpoint").get<std::string>() !=
                    endpoint) {
                continue;
            }
            const nlohmann::json& leg = run.at("legs").at(0);
            if (!leg.at("tag_emission_ordinal").is_null()) {
                selector["tag_allocation_emission_ordinal"] =
                    leg.at("tag_emission_ordinal");
                selector["tag_offset"] =
                    leg.at("tag_offset").get<uint64_t>() +
                    (tile - run.at("first_tile").get<uint64_t>()) *
                        leg.at("tag_tile_stride").get<uint64_t>();
            }
            resolved = true;
            break;
        }
        expect(resolved, "wide-runs anchor selector lost its endpoint run");
    }
}

uint32_t planned_tag(const nlohmann::json& wire, const int rank,
                     const std::string& endpoint, const uint64_t tile,
                     const uint64_t shard) {
    std::map<uint64_t, std::pair<uint32_t, uint64_t>> entries;
    for (const auto& entry : wire.at("tag_allocation_runs")) {
        entries.emplace(entry.at("emission_ordinal").get<uint64_t>(),
                        std::make_pair(
                            entry.at("first_tag").get<uint32_t>(),
                            entry.at("count").get<uint64_t>()));
    }
    for (const auto& run : wire.at("endpoint_runs")) {
        if (run.at("rank").get<int>() != rank ||
            run.at("shard_ordinals").at(0).get<uint64_t>() != shard ||
            run.at("legs").at(0).at("endpoint").get<std::string>() != endpoint) {
            continue;
        }
        if (tile < run.at("first_tile").get<uint64_t>() ||
            tile >= run.at("first_tile").get<uint64_t>() +
                        run.at("tile_count").get<uint64_t>()) {
            continue;
        }
        const nlohmann::json& leg = run.at("legs").at(0);
        expect(!leg.at("tag_emission_ordinal").is_null(),
               "planned tag lookup hit a marker leg");
        const auto found = entries.find(
            leg.at("tag_emission_ordinal").get<uint64_t>());
        expect(found != entries.end(), "planned tag entry is missing");
        const uint64_t offset =
            leg.at("tag_offset").get<uint64_t>() +
            (tile - run.at("first_tile").get<uint64_t>()) *
                leg.at("tag_tile_stride").get<uint64_t>();
        return found->second.first + static_cast<uint32_t>(offset);
    }
    fail("planned tag lookup found no endpoint run");
}

}  // namespace

int main(const int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    AstraSim::LoggerFactory::init("empty", "off");
    MetricCollector::instance().initialize("empty", "off");

    std::string work_ledger_output;
    bool wide_runs = false;
    bool target_slot_delta_cost = false;
    if (argc == 3 && std::string(argv[1]) == "--work-ledger-out") {
        work_ledger_output = argv[2];
    } else if (argc == 2 && std::string(argv[1]) == "--wide-runs") {
        wide_runs = true;
    } else if (argc == 2 &&
               std::string(argv[1]) == "--target-slot-delta-cost") {
        target_slot_delta_cost = true;
    } else if (argc != 1) {
        fail("usage: remote_read_executor_test [--work-ledger-out PATH] "
             "[--wide-runs] [--target-slot-delta-cost]");
    }

    const BuiltFixture fixture = build_fixture();
    nlohmann::json prefill_wire = stage_batch_wire(fixture, 0);
    nlohmann::json decode_wire = stage_batch_wire(fixture, 1);
    if (wide_runs) {
        for (auto& wire : {&prefill_wire, &decode_wire}) {
            const size_t before = (*wire)["pipeline_descriptors"][0]
                                       ["endpoint_runs"].size();
            coalesce_wide_descriptor((*wire)["pipeline_descriptors"][0]);
            const size_t after = (*wire)["pipeline_descriptors"][0]
                                     ["endpoint_runs"].size();
            std::printf("[wide-runs] %s endpoint runs %zu -> %zu\n",
                        (*wire)["pipeline_descriptors"][0]["identity"]
                               ["segment_id"].get<std::string>().c_str(),
                        before, after);
            expect(after < before,
                   "wide-runs coalescing merged nothing (fixture regression)");
        }
    }
    const std::string directory = temporary_directory();
    std::filesystem::create_directories(directory + "/workload");
    write_configs(directory);

    auto event_queue = std::make_shared<EventQueue>();
    const NetworkParser network_parser(directory + "/network.yml");
    const auto topology = construct_topology(network_parser);
    CongestionAwareNetworkApi::set_event_queue(event_queue);
    CongestionAwareNetworkApi::set_topology(topology);
    auto fluid_scheduler = std::make_shared<FluidScheduler>(
        event_queue, topology->get_directed_links(),
        network_parser.get_fluid_max_active_flows(),
        network_parser.get_fluid_max_route_memberships(),
        network_parser.get_progress_report_event_interval());
    fluid_scheduler->set_deferred_flush_mode(true);
    CongestionAwareNetworkApi::set_fluid_scheduler(fluid_scheduler);

    // The real seven-rank Sys topology includes two source shards, a target,
    // an unrelated source rank, and the old rank-5 source-boundary ACK path.
    auto memory_api = std::make_unique<AnalyticalRemoteMemory>(
        directory + "/remote_memory.json");
    std::vector<std::unique_ptr<CongestionAwareNetworkApi>> network_apis;
    std::vector<std::shared_ptr<NodeStoreGraphSource>> graph_sources;
    std::vector<std::unique_ptr<Sys>> sys_owners;
    std::vector<Sys*> systems;
    std::vector<Workload*> workloads;
    const auto npus_per_dim = network_parser.get_npus_counts_per_dim();
    std::vector<int> queues_per_dim(npus_per_dim.size(), 1);
    for (int rank = 0; rank < 7; ++rank) {
        network_apis.push_back(
            std::make_unique<CongestionAwareNetworkApi>(rank));
        auto source = std::make_shared<NodeStoreGraphSource>();
        auto system = std::make_unique<Sys>(
            rank, directory + "/workload", directory + "/comm_group.json",
            directory + "/system.json", memory_api.get(),
            network_apis.back().get(), npus_per_dim, queues_per_dim, 1.0, 1.0,
            false, ExecutionMode::Online, source);
        systems.push_back(system.get());
        workloads.push_back(system->workload);
        graph_sources.push_back(std::move(source));
        sys_owners.push_back(std::move(system));
    }

    std::vector<TerminalRecord> terminals;
    std::vector<FlowRecord> flow_bindings;
    std::vector<WatchFireRecord> watch_fires;
    WatchRegistry watch_registry;
    OnlineCompletionHookContext completion_context;
    completion_context.registry = &watch_registry;
    IssuePassState issue_pass_state{&workloads};
    completion_context.configure_issue_passes(
        event_queue.get(), workloads.size(), &issue_pass_callback,
        &issue_pass_state);
    TerminalHookContext terminal_context{&terminals, &completion_context};
    CompletionObserver::instance().set_hook(&terminal_hook, &terminal_context);

    TileTransportAdapter transport(systems, workloads);
    RemoteReadExecutor executor(
        graph_sources, workloads, &transport,
        [&completion_context](const int rank) {
            completion_context.schedule_issue_pass(rank);
        });
    if (target_slot_delta_cost) {
        const uint64_t direct_probe_start_tick = Sys::boostedTick();
        const uint64_t direct_probe_queue_tick =
            event_queue->get_current_time();
        RemoteReadExecutorTestProbe::run_target_slot_delta_cost_probe(
            executor, fixture, graph_sources, event_queue,
            direct_probe_start_tick);
        expect(Sys::boostedTick() == direct_probe_start_tick &&
                   event_queue->get_current_time() == direct_probe_queue_tick,
               "target-slot cost mode leaves the simulation clock untouched");
        CompletionObserver::instance().set_hook(nullptr, nullptr);
        std::filesystem::remove_all(directory);
        return 0;
    }
    const uint64_t index_test_start_tick = Sys::boostedTick();
    RemoteReadExecutorTestProbe::run(
        executor, index_test_start_tick, fixture);
    expect(Sys::boostedTick() == index_test_start_tick &&
               executor.producer_metrics().endpoint_index_entries_built == 0,
           "synthetic index scales leave no test state in the physical fixture");
    WorkLedgerV2Publisher v2_publisher("remote-read-fixture-v2");
    nlohmann::json v2_frames = nlohmann::json::array();
    const auto publish_v2_frame = [&](const uint64_t delivery_sequence) {
        const uint64_t tick = Sys::boostedTick();
        nlohmann::json cursor_patches =
            executor.take_cursor_patches(tick);
        nlohmann::json frame = v2_publisher.publish(
            delivery_sequence, tick, std::move(cursor_patches),
            collect_work_ledger_v2_active_resources(
                graph_sources, systems, memory_api.get(), fluid_scheduler,
                &executor, &transport, tick));
        expect(frame.at("schema_version").get<uint64_t>() == 4 &&
                   frame.at("protocol_version").get<uint64_t>() == 4 &&
                   frame.at("bridge_protocol").get<std::string>() ==
                       "bridge-v4-20261011" &&
                   frame.at("work_ledger_schema").get<std::string>() ==
                       "workledger-v4-20261011" &&
                   frame.contains("cursor_patches") &&
                   frame.contains("active_resources") &&
                   !frame.contains("authoritative_cursors") &&
                   !frame.contains("obligations"),
               "publisher emits the sparse protocol-v4 top-level frame");
        for (const auto& transition : frame.at("pipeline_transition_runs")) {
            expect(transition.contains("slot_patches") &&
                       transition.contains("credit_patches") &&
                       transition.contains("obligation_patches") &&
                       !transition.contains("slot_state_start") &&
                       !transition.contains("slot_state_end") &&
                       !transition.contains("credit_state_start") &&
                       !transition.contains("credit_state_end") &&
                       !transition.contains("obligation_state_start") &&
                       !transition.contains("obligation_state_end"),
                   "pipeline transitions carry row patches without full arrays");
            for (const auto& slot_patch : transition.at("slot_patches")) {
                if (slot_patch.contains("form")) {
                    expect(slot_patch.at("form") == "target_delta" &&
                               slot_patch.at("key").at("role") == "target" &&
                               slot_patch.at("key").at("shard_ordinal").is_null() &&
                               slot_patch.contains("predecessor_logical_seq") &&
                               slot_patch.contains("operations"),
                           "V4 target rows use sequenced target-specific deltas");
                } else {
                    const nlohmann::json& source_row =
                        slot_patch.at("before").is_null()
                            ? slot_patch.at("after")
                            : slot_patch.at("before");
                    expect(source_row.is_null() ||
                               source_row.at("role") == "source",
                           "V4 keeps only source slot rows in the legacy full-row form");
                }
            }
        }
        v2_frames.push_back(std::move(frame));
    };
    executor.set_ledger_publishers(
        [&v2_publisher, &graph_sources, &executor](
            const std::string& kind, const int rank, const uint64_t node_id,
            const OnlineNode& node, const std::vector<uint64_t>& parents) {
            append_work_ledger_v2_node_transition(
                v2_publisher, graph_sources, &executor, kind, rank, node_id,
                node, parents, Sys::boostedTick());
        },
        [&v2_publisher](nlohmann::json transition) {
            v2_publisher.append_pipeline_transition(std::move(transition));
        },
        [&v2_publisher](nlohmann::json declaration) {
            v2_publisher.append_descriptor_commit(std::move(declaration));
        });
    for (Workload* workload : workloads) {
        workload->set_ledger_transition_observer(
            [&v2_publisher, &graph_sources, &executor](
                const std::string& kind, const NodeView& node) {
                append_work_ledger_v2_node_transition(
                    v2_publisher, graph_sources, &executor, kind, node.rank,
                    node.global_id, node, {}, Sys::boostedTick());
            });
    }
    publish_v2_frame(0);
    executor.set_watch_reference_counter(
        [&watch_registry](const auto& progressions) {
            std::vector<WatchRegistry::MemberProgression> ranges;
            ranges.reserve(progressions.size());
            for (const auto& progression : progressions) {
                ranges.push_back({progression.rank,
                                  progression.first_store_id,
                                  progression.count,
                                  progression.stride});
            }
            return watch_registry.member_reference_count(ranges);
        });
    for (Workload* workload : workloads) {
        workload->set_remote_read_executor(&executor);
    }
    CongestionAwareNetworkApi::set_flow_bound_callback(
        [&transport, &flow_bindings](const AstraNetworkAPI::LogicalFlowKey& key,
                                    const FlowId flow_id) {
            flow_bindings.push_back(FlowRecord{
                key.descriptor_identity, key.generation, key.leg, key.tile,
                key.shard, key.rank, static_cast<uint32_t>(key.tag), flow_id,
                Sys::boostedTick()});
            transport.on_flow_bound(key, flow_id);
        });
    struct FlowCallbackReset {
        ~FlowCallbackReset() {
            CongestionAwareNetworkApi::set_flow_bound_callback({});
        }
    } flow_callback_reset;
    watch_registry.set_fire_notifier(
        [&executor, &watch_fires](const WatchFire& fire) {
            watch_fires.push_back(WatchFireRecord{
                fire.request_id, fire.stage, fire.generation, Sys::boostedTick()});
            executor.notify_service_gate(fire.request_id, fire.stage,
                                         fire.generation);
            executor.notify_watch_registry_changed();
        });

    GraphBatchCommitter::Context committer_context;
    bool suppress_committer_issue_passes = false;
    committer_context.num_ranks = 7;
    committer_context.graph_sources = &graph_sources;
    committer_context.watch_registry = &watch_registry;
    committer_context.publish_node_transition =
        [&v2_publisher, &graph_sources, &executor](
            const std::string& kind, const int rank, const uint64_t node_id,
            const OnlineNode& node, const std::vector<uint64_t>& parents) {
            append_work_ledger_v2_node_transition(
                v2_publisher, graph_sources, &executor, kind, rank, node_id,
                node, parents, Sys::boostedTick());
        };
    committer_context.issue_rank = [&workloads,
                                    &suppress_committer_issue_passes](
                                       const int rank) {
        if (suppress_committer_issue_passes) return;
        workloads.at(static_cast<size_t>(rank))->issue_dep_free_nodes();
    };
    committer_context.prepare_pipeline_batch =
        [&executor](const GraphBatch& batch,
                    const GraphBatchCommitter& committer) {
            return executor.prepare_batch(batch, committer);
        };
    committer_context.commit_pipeline_batch =
        [&executor](std::shared_ptr<void> plan, const GraphBatch& batch,
                    const GraphBatchCommitter& committer) {
            executor.commit_batch(std::move(plan), batch, committer);
        };
    committer_context.commit_kernel_contract =
        [&workloads](std::shared_ptr<const PipelineKernelContract> contract) {
            for (Workload* workload : workloads) {
                workload->set_pipeline_kernel_contract(contract);
            }
        };
    GraphBatchCommitter committer(std::move(committer_context));

    DecisionEvent arrival;
    arrival.seq = 1;
    arrival.reason = DecisionReason::ARRIVAL;
    arrival.request_id = kRequestId;
    arrival.stage = "prefill";
    arrival.generation = 0;
    arrival.payload.session_id = "fixture-session";
    arrival.payload.prefill_length = 1;
    arrival.payload.decode_length = 1;
    const StateDelta arrival_delta = build_state_delta({arrival}, 0, 1);

    // A malformed dependency is rejected before any affine frontier, watch,
    // or executor cursor mutates. The valid input then reuses those exact IDs.
    const uint64_t next_before = graph_sources[0]->store().next_auto_id();
    nlohmann::json invalid_prefill = prefill_wire;
    invalid_prefill["pipeline_descriptors"][0]
                   ["endpoint_dependency_runs"][0]
                   ["first_dependency_json_id"] = 999999;
    bool invalid_rejected = false;
    try {
        const ParsedGraphBatch invalid = parse_graph_batch(invalid_prefill, 7);
        const auto result = committer.validate_and_commit(arrival_delta, invalid);
        invalid_rejected = result.error.has_value();
    } catch (const std::exception&) {
        invalid_rejected = true;
    }
    expect(invalid_rejected &&
               graph_sources[0]->store().next_auto_id() == next_before &&
               watch_registry.size() == 0 &&
               executor.active_descriptor_count() == 0,
           "invalid dependency has no graph/watch/cursor side effects");

    // Keep the ID inside the candidate logical stream, but point an ordinary
    // terminal at a descriptor-owned virtual hole. This reaches the exact
    // candidate-membership check (rather than failing the affine-span bound)
    // and must still leave every commit surface untouched.
    nlohmann::json invalid_ordinary_hole = prefill_wire;
    auto& invalid_descriptor =
        invalid_ordinary_hole["pipeline_descriptors"][0];
    auto& invalid_dependency =
        invalid_descriptor["endpoint_dependency_runs"][0];
    const int dependency_rank = invalid_dependency.at("rank").get<int>();
    uint64_t virtual_hole_json_id = 0;
    bool found_virtual_hole = false;
    for (const auto& range : invalid_descriptor["logical_id_ranges"]) {
        if (range.at("rank").get<int>() == dependency_rank) {
            virtual_hole_json_id =
                range.at("first_json_id").get<uint64_t>();
            found_virtual_hole = true;
            break;
        }
    }
    expect(found_virtual_hole,
           "fixture has a virtual progression on the ordinary terminal rank");
    const uint64_t old_dependency_json_id = invalid_dependency
        .at("first_dependency_json_id").get<uint64_t>();
    invalid_dependency["dependency_kind"] = "ordinary_node_terminal";
    invalid_dependency["first_dependency_json_id"] = virtual_hole_json_id;
    for (auto& endpoint_run : invalid_descriptor["endpoint_runs"]) {
        if (endpoint_run.at("rank").get<int>() == dependency_rank &&
            endpoint_run.contains("initial_parent_json_id") &&
            endpoint_run["initial_parent_json_id"].is_number_integer() &&
            endpoint_run["initial_parent_json_id"].get<uint64_t>() ==
                old_dependency_json_id) {
            endpoint_run["initial_parent_json_id"] = virtual_hole_json_id;
        }
    }
    bool ordinary_hole_rejected = false;
    std::string ordinary_hole_error;
    try {
        const ParsedGraphBatch invalid =
            parse_graph_batch(invalid_ordinary_hole, 7);
        const auto result = committer.validate_and_commit(arrival_delta, invalid);
        if (result.error.has_value()) {
            ordinary_hole_error = *result.error;
            ordinary_hole_rejected =
                ordinary_hole_error.find(
                    "ordinary terminal does not resolve to every ") !=
                std::string::npos;
        }
    } catch (const std::exception& exc) {
        ordinary_hole_error = exc.what();
    }
    expect(ordinary_hole_rejected &&
               graph_sources[0]->store().next_auto_id() == next_before &&
               watch_registry.size() == 0 &&
               executor.active_descriptor_count() == 0,
           "ordinary terminal on a candidate virtual hole fails closed before mutation: " +
               ordinary_hole_error);

    const ParsedGraphBatch prefill = parse_graph_batch(prefill_wire, 7);
    const auto prefill_commit =
        committer.validate_and_commit(arrival_delta, prefill);
    if (prefill_commit.error.has_value()) {
        fail("valid P0 batch rejected: " + *prefill_commit.error);
    }
    expect(executor.active_descriptor_count() == 1,
           "P0 descriptor committed before D1 continuation");
    expect(std::all_of(workloads.begin(), workloads.end(),
                       [&prefill](const Workload* workload) {
                           return workload->pipeline_kernel_contract() != nullptr &&
                               *workload->pipeline_kernel_contract() ==
                                   *prefill.kernel_contract;
                       }),
           "one accepted immutable kernel contract reaches every rank");

    std::map<uint64_t, TransferObservation> transfer_observations;
    size_t peak_active_per_rank = 0;
    size_t peak_target_data_tiles = 0;
    size_t peak_source_shard_data_tiles = 0;
    const auto sample = [&]() {
        sample_transfers(transport, Sys::boostedTick(), transfer_observations,
                         &peak_active_per_rank, &peak_target_data_tiles,
                         &peak_source_shard_data_tiles);
    };
    sample();
    fluid_scheduler->flush_pending_starts();
    fluid_scheduler->mark_event_loop_started();

    const auto pump_until = [&](const std::function<bool()>& done,
                                const std::string& label) {
        uint64_t steps = 0;
        while (!done() && !event_queue->finished()) {
            event_queue->proceed();
            sample();
            if (++steps > 1000000) fail(label + " exceeded event guard");
        }
        expect(done(), label + " did not complete before the real queue drained");
    };
    const auto snapshot_now = [&]() {
        return executor.authoritative_snapshot(Sys::boostedTick());
    };
    const std::string prefill_segment =
        fixture.programs[0].wire["identity"]["segment_id"].get<std::string>();
    const auto prefill_done = [&]() {
        const auto snapshot = snapshot_now();
        const auto descriptor = descriptor_for(snapshot, prefill_segment);
        return descriptor.at("service_done").get<bool>() &&
               descriptor.at("settle_done").get<bool>();
    };
    pump_until(prefill_done, "P0 service and settle");
    const uint64_t prefill_settle_tick = Sys::boostedTick();
    publish_v2_frame(1);
    expect(transfer_observations.size() == 18,
           "P0 emits 16 real COMM endpoints and two charged body markers");

    DecisionEvent prefill_drain;
    prefill_drain.seq = 2;
    prefill_drain.reason = DecisionReason::PREFILL_DRAIN;
    prefill_drain.request_id = kRequestId;
    prefill_drain.stage = "prefill";
    prefill_drain.generation = 0;
    prefill_drain.payload.watch_member_count = 1;
    const StateDelta decode_delta = build_state_delta(
        {prefill_drain}, Sys::boostedTick(), 2);

    // Malformed physical-anchor selectors still fail during descriptor
    // parsing, before any committed stream or executor state can change.
    std::vector<uint64_t> next_ids_before_bad_anchors;
    next_ids_before_bad_anchors.reserve(graph_sources.size());
    for (const auto& source : graph_sources) {
        next_ids_before_bad_anchors.push_back(
            source->store().next_auto_id());
    }
    const size_t active_descriptors_before_bad_anchors =
        executor.active_descriptor_count();
    const size_t watches_before_bad_anchors = watch_registry.size();
    const auto malformed_anchor_rejected = [](const nlohmann::json& wire) {
        try {
            static_cast<void>(parse_graph_batch(wire, 7));
        } catch (const std::exception&) {
            return true;
        }
        return false;
    };
    nlohmann::json bad_anchor_role = decode_wire;
    bool changed_anchor_role = false;
    for (auto& anchor : bad_anchor_role["pipeline_descriptors"][0]["anchors"]) {
        if (anchor.at("binding").get<std::string>() ==
            "physical_endpoint_slot") {
            anchor["role"] = "service";
            changed_anchor_role = true;
            break;
        }
    }
    expect(changed_anchor_role && malformed_anchor_rejected(bad_anchor_role),
           "malformed physical anchor role remains parser-rejected");
    nlohmann::json bad_anchor_selector = decode_wire;
    bool changed_anchor_selector = false;
    for (auto& anchor : bad_anchor_selector["pipeline_descriptors"][0]["anchors"]) {
        if (anchor.at("binding").get<std::string>() ==
                "physical_endpoint_slot" &&
            anchor.at("selector").contains("tile_ordinal")) {
            anchor["selector"]["tile_ordinal"] = 0;
            changed_anchor_selector = true;
            break;
        }
    }
    expect(changed_anchor_selector &&
               malformed_anchor_rejected(bad_anchor_selector),
           "malformed physical anchor selector remains parser-rejected");
    for (size_t rank = 0; rank < graph_sources.size(); ++rank) {
        expect(graph_sources[rank]->store().next_auto_id() ==
                   next_ids_before_bad_anchors[rank],
               "malformed anchor parse leaves every NodeStore ID stream unchanged");
    }
    expect(executor.active_descriptor_count() ==
               active_descriptors_before_bad_anchors &&
               watch_registry.size() == watches_before_bad_anchors,
           "malformed anchor parse leaves descriptors and watches unchanged");

    const ParsedGraphBatch decode = parse_graph_batch(decode_wire, 7);
    suppress_committer_issue_passes = wide_runs;
    const auto decode_commit = committer.validate_and_commit(decode_delta, decode);
    if (decode_commit.error.has_value()) {
        fail("valid D1 continuation rejected: " + *decode_commit.error);
    }
    if (wide_runs) {
        const std::string decode_segment =
            fixture.programs[1].wire["identity"]["segment_id"]
                .get<std::string>();
        RemoteReadExecutorTestProbe::run_anchor_coordinate_prefilter(
            executor, graph_sources, workloads, event_queue, decode_segment);
        suppress_committer_issue_passes = false;
        // Replay the original committer issue-rank order at the same tick;
        // all extra preflight rounds above were read-only.
        for (const int rank : {0, 5, 6}) {
            workloads[static_cast<size_t>(rank)]->issue_dep_free_nodes();
        }
    }
    expect(executor.active_descriptor_count() == 2,
           "D1 descriptor commits while reusing P0 buffer-domain identity");
    const auto decode_domain =
        fixture.programs[1].wire["identity"]["buffer_domain_id"];
    expect(fixture.programs[0].wire["identity"]["buffer_domain_id"] ==
               decode_domain,
           "P0 and D1 share the real two-slot buffer domain");
    sample();
    fluid_scheduler->flush_pending_starts();
    publish_v2_frame(2);

    const std::string decode_segment =
        fixture.programs[1].wire["identity"]["segment_id"].get<std::string>();
    const auto both_done = [&]() {
        const auto snapshot = snapshot_now();
        const auto descriptor = descriptor_for(snapshot, decode_segment);
        return descriptor.at("service_done").get<bool>() &&
               descriptor.at("settle_done").get<bool>();
    };
    pump_until(both_done, "D1 service and settle");
    const uint64_t final_tick = Sys::boostedTick();
    sample();
    publish_v2_frame(3);
    verify_credit_patch_replay(v2_frames);

    expect(executor.is_drained(), "executor and physical cookies fully drained");
    expect(transport.active_count() == 0, "transport adapter is drained");
    expect(fluid_scheduler->get_active_flow_count() == 0,
           "real FluidScheduler flows are drained");
    expect(memory_api->is_drained(), "remote-memory backend is drained");
    expect(transfer_observations.size() == 54,
           "P0 plus D1 emitted the complete bounded real endpoint schedule");
    expect(flow_bindings.size() == 26,
           "all real data/ACK sends bind to physical FlowIds");
    expect(peak_active_per_rank >= 2,
           "same-rank endpoints overlap under real COMM/HBM resource gates");
    expect(peak_target_data_tiles <= 2 && peak_source_shard_data_tiles <= 2,
           "shared target and each source-shard content slots stay bounded at two");

    const auto find_transfer = [&transfer_observations](
        const uint64_t generation, const std::string& leg,
        const uint64_t global_tile, const int rank, const uint64_t shard)
        -> const TransferObservation* {
        for (const auto& entry : transfer_observations) {
            const auto& item = entry.second;
            const auto& key = item.transfer.logical_key;
            if (key.generation == generation && key.leg == leg &&
                key.tile == global_tile && key.rank == rank &&
                key.shard == shard) return &item;
        }
        return nullptr;
    };
    const auto* const boundary_ack_recv =
        find_transfer(1, "release_ack_recv", 3, 5, 0);
    const auto* const boundary_data_send =
        find_transfer(1, "data_send", 3, 5, 0);
    // Global decode tile 3 is the group-local tile 1 (global_tile_first=3).
    // The planned tags reproduce 12/13 in the fragmented default and the
    // renumbered wide-run allocation in --wide-runs mode.
    const nlohmann::json& decode_descriptor = decode_wire.at("pipeline_descriptors").at(0);
    const uint32_t planned_ack_tag =
        planned_tag(decode_descriptor, 5, "release_ack_recv", 1, 0);
    const uint32_t planned_data_tag =
        planned_tag(decode_descriptor, 5, "data_send", 1, 0);
    expect(boundary_ack_recv != nullptr && boundary_data_send != nullptr &&
               boundary_ack_recv->transfer.tag == planned_ack_tag &&
               boundary_data_send->transfer.tag == planned_data_tag &&
               boundary_ack_recv->completed &&
               boundary_ack_recv->completion_tick <=
                   boundary_data_send->issue_tick,
           "rank-5 source-boundary ACK tag completes before its data tag");
    const auto* const consumer_a =
        find_transfer(1, "data_recv", 3, kOwnerRank, 0);
    const auto* const consumer_b =
        find_transfer(1, "data_recv", 5, kOwnerRank, 0);
    const auto* const marker_p0 =
        find_transfer(0, "body_marker", 1, kOwnerRank, 0);
    const auto* const marker_p1 =
        find_transfer(0, "body_marker", 2, kOwnerRank, 0);
    expect(consumer_a != nullptr && consumer_b != nullptr &&
               marker_p0 != nullptr && marker_p1 != nullptr,
           "two body consumers and one canonical shared marker per P tile execute");
    if (consumer_a != nullptr && consumer_b != nullptr) {
        expect(consumer_b->issue_tick > consumer_a->completion_tick,
               "second decode consumer cannot pass the delayed body-A terminal");
    }
    for (const auto* marker : {marker_p0, marker_p1}) {
        if (marker == nullptr) continue;
        expect(marker->transfer.control_marker &&
                   marker->transfer.total_bytes == 0 && marker->completed &&
                   marker->completion_tick - marker->issue_tick >= 10,
               "canonical marker retains its original 1-op/1-byte COMP/HBM cost");
    }
    for (const auto& entry : transfer_observations) {
        expect(entry.second.completed,
               "every real endpoint/marker callback drains: transfer=" +
                   std::to_string(entry.first));
    }

    const auto require_event_terminal = [&](const DescriptorProgram& program,
                                            const EventKey& event_key) {
        const EndpointEvent& event = program.events.at(event_key);
        if (!event.physical_anchor) return;
        const auto store_id = committer.resolve_store_id(event.rank, event.json_id);
        expect(store_id.has_value(), "physical endpoint anchor maps to NodeStore");
        require_one_terminal(terminals, event.rank, *store_id);
        expect(graph_sources[event.rank]->store().is_finished_or_erased(*store_id),
               "physical anchor finishes through the shared terminal path");
    };
    for (const DescriptorProgram& program : fixture.programs) {
        for (const EventKey& event_key : program.emission_order) {
            require_event_terminal(program, event_key);
        }
    }
    for (const auto& entry : fixture.body_json_ids) {
        const auto store_id = committer.resolve_store_id(kOwnerRank, entry.second);
        expect(store_id.has_value(), "body consumer maps to NodeStore");
        require_one_terminal(terminals, kOwnerRank, *store_id);
        expect(graph_sources[kOwnerRank]->store().is_finished_or_erased(*store_id),
               "body consumer reaches one ordinary terminal");
    }
    for (Workload* workload : workloads) {
        expect(workload->hw_resource->num_in_flight_gpu_comm_ops == 0 &&
                   workload->hw_resource->num_in_flight_gpu_comp_ops == 0,
               "all synthetic COMM/COMP occupancy releases on every rank");
        expect(!workload->local_hbm_bandwidth_model->has_active_jobs(),
               "all COMM and original marker HBM jobs drain on every rank");
        expect(workload->pipeline_kernel_contract() != nullptr &&
                   *workload->pipeline_kernel_contract() == *prefill.kernel_contract,
               "run-frozen kernel contract remains identical after continuation");
    }
    expect(watch_fires.size() == 2 &&
               watch_fires[0].stage == "prefill" &&
               watch_fires[1].stage == "decode" &&
               watch_fires[0].tick < watch_fires[1].tick &&
               prefill_settle_tick >= watch_fires[0].tick,
           "real P0 watch/settle precedes the independent D1 watch");
    watch_registry.remove_watches_for_request(kRequestId);
    executor.notify_watch_registry_changed();
    expect(watch_registry.empty() && watch_registry.stale_count() == 0,
           "watch registry closes after both stage fires");

    std::sort(flow_bindings.begin(), flow_bindings.end(),
              [](const FlowRecord& a, const FlowRecord& b) {
                  return a.flow_id < b.flow_id;
              });
    std::printf("[remote_read_executor_test] service_ticks=%llu,%llu "
                "settle_tick=%llu flows=%zu endpoints=%zu peak_per_rank=%zu "
                "target_active=%zu source_active=%zu final_tick=%llu\n",
                static_cast<unsigned long long>(watch_fires[0].tick),
                static_cast<unsigned long long>(watch_fires[1].tick),
                static_cast<unsigned long long>(final_tick), flow_bindings.size(),
                transfer_observations.size(), peak_active_per_rank,
                peak_target_data_tiles, peak_source_shard_data_tiles,
                static_cast<unsigned long long>(final_tick));
    const auto& producer_metrics = executor.producer_metrics();
    expect(producer_metrics.authoritative_snapshot_calls > 0 &&
               producer_metrics.diagnostic_row_helper_visits > 0,
           "fixture diagnostics are reported separately from producer row work");
    const std::string work_ledger_dump = v2_frames.dump(2);
    size_t pipeline_transition_count = 0;
    size_t cursor_run_patch_count = 0;
    size_t target_delta_patch_count = 0;
    size_t source_full_row_patch_count = 0;
    for (const auto& frame : v2_frames) {
        pipeline_transition_count +=
            frame.at("pipeline_transition_runs").size();
        for (const auto& segment : frame.at("cursor_patches")) {
            cursor_run_patch_count += segment.at("runs").size();
        }
        for (const auto& transition : frame.at("pipeline_transition_runs")) {
            for (const auto& patch : transition.at("slot_patches")) {
                if (patch.contains("form")) {
                    ++target_delta_patch_count;
                    expect(patch.at("form") == "target_delta" &&
                               patch.at("key").at("role") == "target",
                           "online execution emits target-only V4 slot deltas");
                } else {
                    const auto& row = patch.at("before").is_null()
                        ? patch.at("after") : patch.at("before");
                    expect(row.is_null() || row.at("role") == "source",
                           "online execution keeps full rows only for source slots");
                    ++source_full_row_patch_count;
                }
            }
        }
    }
    expect(target_delta_patch_count > 0 && source_full_row_patch_count > 0,
           "physical fixture emits both target deltas and compatible source rows");
    std::printf("[workledger-v4] deliveries=%zu pipeline_transitions=%zu "
                "target_delta_patches=%zu source_full_row_patches=%zu "
                "cursor_run_patches=%zu serialized_bytes=%zu "
                "row_helper_visits=%llu diagnostic_row_helper_visits=%llu "
                "authoritative_snapshot_calls=%llu blocked_diagnostic_calls=%llu "
                "row_capture_ns=%llu "
                "cursor_capture_ns=%llu publish_ns=%llu issue_ns=%llu "
                "issue_calls=%llu index_entries=%llu index_sort_cmp=%llu "
                "index_tree_ops=%llu index_search_steps=%llu "
                "index_blocks=%llu index_candidates=%llu "
                "current_cursor_candidates=%llu\n",
                v2_frames.size(), pipeline_transition_count,
                target_delta_patch_count, source_full_row_patch_count,
                cursor_run_patch_count, work_ledger_dump.size(),
                static_cast<unsigned long long>(producer_metrics.row_helper_visits),
                static_cast<unsigned long long>(producer_metrics.diagnostic_row_helper_visits),
                static_cast<unsigned long long>(producer_metrics.authoritative_snapshot_calls),
                static_cast<unsigned long long>(producer_metrics.blocked_diagnostics_calls),
                static_cast<unsigned long long>(producer_metrics.resource_row_capture_ns),
                static_cast<unsigned long long>(producer_metrics.cursor_patch_capture_ns),
                static_cast<unsigned long long>(producer_metrics.transition_publish_ns),
                static_cast<unsigned long long>(producer_metrics.actual_issue_ns),
                static_cast<unsigned long long>(producer_metrics.actual_issue_calls),
                static_cast<unsigned long long>(producer_metrics.endpoint_index_entries_built),
                static_cast<unsigned long long>(producer_metrics.endpoint_index_sort_comparisons),
                static_cast<unsigned long long>(producer_metrics.endpoint_index_tree_build_operations),
                static_cast<unsigned long long>(producer_metrics.endpoint_index_block_search_steps),
                static_cast<unsigned long long>(producer_metrics.endpoint_index_blocks_examined),
                static_cast<unsigned long long>(producer_metrics.endpoint_index_candidate_examinations),
                static_cast<unsigned long long>(producer_metrics.current_cursor_candidates_examined));
    for (const auto& flow : flow_bindings) {
        std::printf("[flow] id=%llu descriptor=%s generation=%llu leg=%s "
                    "tile=%llu rank=%d shard=%llu tag=%u bound_tick=%llu\n",
                    static_cast<unsigned long long>(flow.flow_id),
                    flow.descriptor_identity.c_str(),
                    static_cast<unsigned long long>(flow.generation),
                    flow.leg.c_str(),
                    static_cast<unsigned long long>(flow.tile), flow.rank,
                    static_cast<unsigned long long>(flow.shard), flow.tag,
                    static_cast<unsigned long long>(flow.bound_tick));
    }
    std::printf("[remote_read_executor_test] ALL PASS: P0->D1 two-slot "
                "continuation, real multi-rank COMM/HBM, source-boundary ACK, "
                "shared marker, delayed consumers, service and settle\n");

    if (!work_ledger_output.empty()) {
        std::ofstream out(work_ledger_output, std::ios::binary | std::ios::trunc);
        if (!out) fail("cannot create protocol-v4 frame output");
        out << work_ledger_dump << '\n';
        out.close();
        if (!out.good()) fail("failed writing protocol-v4 frame output");
    }

    CompletionObserver::instance().set_hook(nullptr, nullptr);
    CongestionAwareNetworkApi::set_flow_bound_callback({});
    std::error_code ignored;
    std::filesystem::remove_all(directory, ignored);
    return 0;
}
