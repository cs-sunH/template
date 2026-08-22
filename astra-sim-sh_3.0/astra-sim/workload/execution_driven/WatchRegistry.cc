/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

WatchRegistry -- execution-driven mechanism layer (wscllm phase 1).
Implementation (方案 §4 步骤 1-5).
*******************************************************************************/

#include "astra-sim/workload/execution_driven/WatchRegistry.hh"

#include "astra-sim/workload/Workload.hh"
#include "common/EventQueue.h"

#include <algorithm>
#include <utility>

namespace AstraSim {
namespace ExecutionDriven {

uint64_t WatchRegistry::register_stage_watch(
    const std::string& request_id, const std::string& stage,
    uint64_t generation, std::set<CompletionKey> expected_members,
    std::set<NodeTerminalStatus> satisfying_statuses) {
    const Identity identity{request_id, stage, generation};
    const auto existing = identity_to_watch_.find(identity);
    if (existing != identity_to_watch_.end()) {
        // Register once, fire once: a duplicate registration (e.g. a double
        // commit of the same stage) must not create a second watch.
        return existing->second;
    }
    const uint64_t watch_id = next_watch_id_++;
    StageWatch watch;
    watch.watch_id = watch_id;
    watch.request_id = request_id;
    watch.stage = stage;
    watch.generation = generation;
    watch.expected_members = std::move(expected_members);
    watch.satisfying_statuses = std::move(satisfying_statuses);
    watches_.emplace(watch_id, std::move(watch));
    identity_to_watch_.emplace(std::move(identity), watch_id);
    // Phase 4: per-request watch-id index (REQUEST_COMPLETE-commit removal).
    request_to_watch_ids_[request_id].push_back(watch_id);
    return watch_id;
}

void WatchRegistry::on_node_terminal(const CompletionKey& key,
                                     const NodeStoreMeta& meta,
                                     NodeTerminalStatus status) {
    // Unknown identity (request_id, stage, generation): no watch registered
    // for this logical stage -- no-op (also covers generation mismatch at
    // the meta level).
    const auto identity_it = identity_to_watch_.find(
        Identity{meta.request_id, meta.stage, meta.generation});
    if (identity_it == identity_to_watch_.end()) {
        return;
    }
    const auto watch_it = watches_.find(identity_it->second);
    if (watch_it == watches_.end() || watch_it->second.fired) {
        return;  // already fired: every later terminal is a no-op
    }
    auto& watch = watch_it->second;
    // Generation mismatch at the key level (belt and suspenders: the
    // identity index already carries the generation, but the key must agree
    // too -- a node of another generation is a different logical node).
    if (key.generation != watch.generation) {
        return;
    }
    // The watch's own explicit policy: statuses not listed never satisfy
    // (Skipped is never implicitly upgraded to Success).
    if (!watch.satisfying_statuses.count(status)) {
        return;
    }
    // Exact member set: foreign terminals are no-ops.
    if (watch.expected_members.count(key) == 0) {
        return;
    }
    // Only the FIRST terminal of an exact member may enter.
    if (!watch.completed_members.insert(key).second) {
        return;
    }
    if (watch.completed_members.size() != watch.expected_members.size()) {
        return;  // still waiting for the remaining members
    }
    // Every expected member has a satisfying terminal: fire once.
    watch.fired = true;
    WatchFire fire;
    fire.watch_id = watch.watch_id;
    fire.request_id = watch.request_id;
    fire.stage = watch.stage;
    fire.generation = watch.generation;
    fire.member_count = watch.completed_members.size();
    // Phase 4 (schema v1): the frozen member RANKS (distinct, ascending) --
    // feeds StateDelta.affected_ranks.
    for (const auto& key : watch.expected_members) {
        if (fire.member_ranks.empty() || fire.member_ranks.back() != key.rank) {
            fire.member_ranks.push_back(key.rank);  // set order = ascending
        }
    }
    fired_.push_back(fire);
    if (notifier_) {
        notifier_(fire);
    }
}

std::vector<WatchFire> WatchRegistry::fired_and_drain() {
    std::vector<WatchFire> result = std::move(fired_);
    fired_.clear();
    return result;
}

size_t WatchRegistry::stale_count() const {
    size_t count = 0;
    for (const auto& entry : watches_) {
        if (!entry.second.fired) {
            ++count;
        }
    }
    return count;
}

void WatchRegistry::remove_watch(uint64_t watch_id) {
    const auto it = watches_.find(watch_id);
    if (it == watches_.end()) {
        return;
    }
    identity_to_watch_.erase(Identity{it->second.request_id, it->second.stage,
                                      it->second.generation});
    // Phase 4: per-request index maintenance (erase one watch id).
    auto request_it = request_to_watch_ids_.find(it->second.request_id);
    if (request_it != request_to_watch_ids_.end()) {
        auto& ids = request_it->second;
        ids.erase(std::remove(ids.begin(), ids.end(), watch_id), ids.end());
        if (ids.empty()) {
            request_to_watch_ids_.erase(request_it);
        }
    }
    watches_.erase(it);
}

void WatchRegistry::remove_watches_for_request(const std::string& request_id) {
    // Phase 4: O(watches of the request) via the per-request index -- no
    // full-registry scan. Fired or not, every watch of the request is
    // removed (REQUEST_COMPLETE commit; both stage watches have fired by
    // then). Absent from the index = no registered watch: no-op.
    const auto request_it = request_to_watch_ids_.find(request_id);
    if (request_it == request_to_watch_ids_.end()) {
        return;
    }
    // Copy the id list: remove_watch mutates the index we iterate.
    const std::vector<uint64_t> ids = request_it->second;
    for (const uint64_t watch_id : ids) {
        remove_watch(watch_id);
    }
}

void WatchRegistry::remove_all() {
    watches_.clear();
    identity_to_watch_.clear();
    request_to_watch_ids_.clear();
}

void WatchRegistry::set_fire_notifier(WatchFireNotifier notifier) {
    notifier_ = std::move(notifier);
}

namespace {

// Step 1-8: one per-rank post-commit deferred issue pass (the plan's
// "ready 集合 -> post-commit deferred 后继发射" drain, 方案 §4 步骤 1-4
// 操作 4 / 仿真加速分析.md §3.3). The pass runs in the same tick's
// deferred drain -- i.e. after the completion callback has fully executed,
// including its graph_source_->finish_node dependency release -- so the
// freed children are in the store's free set when it scans. The pipeline
// therefore advances continuously (a freed child is issued in the same tick
// it was freed), while Workload::call itself never calls
// issue_dep_free_nodes in online mode (no static auto-advance).
struct OnlineIssuePassArg {
    AstraSim::Workload* workload = nullptr;
    int rank = -1;
};

void online_issue_pass_cb(void* arg) {
    auto* pass = static_cast<OnlineIssuePassArg*>(arg);
    pass->workload->issue_dep_free_nodes();
    delete pass;
}

}  // namespace

void online_completion_hook(void* ctx, int rank, uint64_t node_id,
                            const char* request_id, const char* stage,
                            uint64_t generation, uint64_t tick,
                            int terminal_status) {
    auto* context = static_cast<OnlineCompletionHookContext*>(ctx);
    if (context == nullptr || context->registry == nullptr) {
        return;
    }
    // (1) Record the completion fact -- read-only with respect to the graph.
    const NodeStoreMeta meta{request_id ? request_id : "",
                             stage ? stage : "", generation};
    context->registry->on_node_terminal(
        CompletionKey{rank, node_id, generation}, meta,
        static_cast<NodeTerminalStatus>(terminal_status));
    // (1b) Phase 4 (schema v1): buffer the per-node terminal fact for
    //     StateDelta.completed_nodes (one fact per node terminal; the
    //     tick-end gate moves the buffer into the delivery and clears it).
    //     Audit/reconciliation input only -- never a strategy input.
    if (context->completed_facts != nullptr) {
        CompletedNodeFact fact;
        fact.rank = rank;
        fact.node_id = node_id;
        fact.request_id = request_id ? request_id : "";
        fact.stage = stage ? stage : "";
        fact.generation = generation;
        fact.tick = tick;
        fact.terminal_status = terminal_status;
        context->completed_facts->push_back(std::move(fact));
    }
    // (2) The DecisionMailbox write arrives through the registry's fire
    //     notifier (step 1-6 wiring). NEVER finish_node here.
    // (3) Step 1-8: schedule the per-rank post-commit deferred issue pass.
    //     The EventQueue hard rule (step 1-1) requires same-tick events
    //     from inside an event handler to go through
    //     schedule_event_deferred -- this hook runs inside Workload::call
    //     (an event-handler context). Unwired (fixture mode) => facts only.
    if (context->event_queue != nullptr && context->workloads != nullptr &&
        rank >= 0 && rank < static_cast<int>(context->workloads->size())) {
        auto* arg = new OnlineIssuePassArg;
        arg->workload = (*context->workloads)[rank];
        arg->rank = rank;
        context->event_queue->schedule_event_deferred(online_issue_pass_cb,
                                                      arg);
    }
}

}  // namespace ExecutionDriven
}  // namespace AstraSim
