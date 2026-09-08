/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

WatchRegistry -- execution-driven mechanism layer (wscllm phase 1).
Implementation (方案 §4 步骤 1-5).
*******************************************************************************/

#include "astra-sim/workload/execution_driven/WatchRegistry.hh"

#include "common/EventQueue.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
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
    const auto inserted = watches_.emplace(watch_id, std::move(watch));
    for (const auto& member : inserted.first->second.expected_members) {
        member_to_watch_ids_[member].push_back(watch_id);
    }
    identity_to_watch_.emplace(std::move(identity), watch_id);
    // Phase 4: per-request watch-id index (REQUEST_COMPLETE-commit removal).
    request_to_watch_ids_[request_id].push_back(watch_id);
    // 拼 batch 适配(2026-08-22):列车哨兵入独立索引。
    if (request_id.rfind("batch_train_", 0) == 0) {
        sentinel_watch_ids_.push_back(watch_id);
    }
    return watch_id;
}

void WatchRegistry::on_node_terminal(const CompletionKey& key,
                                     const NodeTerminalStatus status) {
    // The overwhelmingly common terminal is not a watch member. Route it by
    // numeric key before touching request/stage metadata or any tree index.
    const auto members_it = member_to_watch_ids_.find(key);
    if (members_it == member_to_watch_ids_.end()) {
        return;
    }
    // A physical node can intentionally belong to more than one watch. The
    // id vector remains stable in this call: watches are only removed by the
    // post-delivery lifecycle, never by a synchronous fire notifier.
    for (const uint64_t watch_id : members_it->second) {
        const auto watch_it = watches_.find(watch_id);
        if (watch_it == watches_.end() || watch_it->second.fired) {
            continue;
        }
        auto& watch = watch_it->second;
        // The watch's own explicit policy: statuses not listed never satisfy
        // (Skipped is never implicitly upgraded to Success).
        if (!watch.satisfying_statuses.count(status)) {
            continue;
        }
        // Only the FIRST terminal of an exact member may enter. The direct
        // index proves membership; retain the set insert for duplicate/firing
        // semantics and defensive consistency.
        if (!watch.completed_members.insert(key).second) {
            continue;
        }
        if (watch.completed_members.size() != watch.expected_members.size()) {
            continue;  // still waiting for the remaining members
        }
        // Every expected member has a satisfying terminal: fire once.
        watch.fired = true;
        WatchFire fire;
        fire.watch_id = watch.watch_id;
        fire.request_id = watch.request_id;
        fire.stage = watch.stage;
        fire.generation = watch.generation;
        fire.member_count = watch.completed_members.size();
        // Phase 4 (schema v1): the frozen member RANKS (distinct, ascending)
        // feeds StateDelta.affected_ranks.
        for (const auto& member : watch.expected_members) {
            if (fire.member_ranks.empty() ||
                fire.member_ranks.back() != member.rank) {
                fire.member_ranks.push_back(member.rank);
            }
        }
        // Production installs a synchronous notifier, so retaining a second
        // copy for a polling consumer would grow linearly with fired watches.
        // Queue only the fixture/polling mode that has no notifier.
        if (notifier_) {
            notifier_(fire);
        } else {
            fired_.push_back(std::move(fire));
        }
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
    for (const auto& member : it->second.expected_members) {
        const auto members_it = member_to_watch_ids_.find(member);
        if (members_it == member_to_watch_ids_.end()) {
            continue;
        }
        auto& ids = members_it->second;
        ids.erase(std::remove(ids.begin(), ids.end(), watch_id), ids.end());
        if (ids.empty()) {
            member_to_watch_ids_.erase(members_it);
        }
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

void WatchRegistry::drain_fired_sentinels() {
    // 拼 batch 适配(2026-08-22):已 fire 的哨兵逐个移除;未 fire 者保留。
    size_t kept = 0;
    for (size_t i = 0; i < sentinel_watch_ids_.size(); ++i) {
        const uint64_t watch_id = sentinel_watch_ids_[i];
        const auto it = watches_.find(watch_id);
        if (it != watches_.end() && it->second.fired) {
            remove_watch(watch_id);
            continue;
        }
        sentinel_watch_ids_[kept++] = watch_id;
    }
    sentinel_watch_ids_.resize(kept);
}

void WatchRegistry::remove_all() {
    watches_.clear();
    identity_to_watch_.clear();
    member_to_watch_ids_.clear();
    request_to_watch_ids_.clear();
    sentinel_watch_ids_.clear();
}

void WatchRegistry::set_fire_notifier(WatchFireNotifier notifier) {
    notifier_ = std::move(notifier);
}

void OnlineCompletionHookContext::configure_issue_passes(
    NetworkAnalytical::EventQueue* event_queue, const size_t rank_count,
    const IssuePassCallback callback, void* const callback_context) {
    if (event_queue == nullptr || callback == nullptr ||
        pending_issue_pass_count() != 0) {
        std::fprintf(stderr,
                     "[Error] (execution_driven/completion_hook) invalid "
                     "issue-pass configuration\n");
        std::abort();
    }
    event_queue_ = event_queue;
    issue_pass_callback_ = callback;
    issue_pass_callback_context_ = callback_context;
    issue_pass_pending_.assign(rank_count, 0);
    issue_pass_args_.resize(rank_count);
    for (size_t rank = 0; rank < rank_count; ++rank) {
        issue_pass_args_[rank] =
            IssuePassArg{this, static_cast<int>(rank)};
    }
}

bool OnlineCompletionHookContext::schedule_issue_pass(const int rank) {
    // An unconfigured context is the documented facts-only fixture mode.
    if (event_queue_ == nullptr) {
        return false;
    }
    if (rank < 0 || rank >= static_cast<int>(issue_pass_pending_.size())) {
        std::fprintf(stderr,
                     "[Error] (execution_driven/completion_hook) invalid "
                     "issue-pass rank=%d rank_count=%zu\n",
                     rank, issue_pass_pending_.size());
        std::abort();
    }
    if (issue_pass_pending_[rank] != 0) {
        return false;
    }
    issue_pass_pending_[rank] = 1;
    event_queue_->schedule_event_deferred(
        OnlineCompletionHookContext::issue_pass_trampoline,
        &issue_pass_args_[rank]);
    return true;
}

size_t OnlineCompletionHookContext::pending_issue_pass_count() const {
    return static_cast<size_t>(std::count(
        issue_pass_pending_.begin(), issue_pass_pending_.end(), uint8_t{1}));
}

void OnlineCompletionHookContext::issue_pass_trampoline(void* arg) {
    auto* const pass = static_cast<IssuePassArg*>(arg);
    auto* const owner = pass == nullptr ? nullptr : pass->owner;
    if (owner == nullptr || pass->rank < 0 ||
        pass->rank >= static_cast<int>(owner->issue_pass_pending_.size()) ||
        owner->issue_pass_pending_[pass->rank] == 0 ||
        owner->issue_pass_callback_ == nullptr) {
        std::fprintf(stderr,
                     "[Error] (execution_driven/completion_hook) corrupt "
                     "deferred issue-pass state\n");
        std::abort();
    }
    // Clear before the callback. If issuing newly-ready nodes completes one
    // synchronously, that nested completion may append another pass for this
    // rank to the same deferred FIFO.
    owner->issue_pass_pending_[pass->rank] = 0;
    owner->issue_pass_callback_(owner->issue_pass_callback_context_,
                                pass->rank);
}

void online_completion_hook(void* ctx, int rank, uint64_t node_id,
                            const char* request_id, const char* stage,
                            uint64_t generation, uint64_t tick,
                            int terminal_status) {
    auto* context = static_cast<OnlineCompletionHookContext*>(ctx);
    if (context == nullptr || context->registry == nullptr) {
        return;
    }
    // (1) Route the terminal by its numeric member key -- read-only with
    // respect to the graph. The direct index deliberately avoids constructing
    // NodeStoreMeta(request_id, stage) for every non-watch terminal.
    context->registry->on_node_terminal(
        CompletionKey{rank, node_id, generation},
        static_cast<NodeTerminalStatus>(terminal_status));
    // (1b) Phase 4: buffer the terminal fact for StateDelta.completed_nodes.
    // Default mode aggregates count/rank evidence before delivery; exact mode
    // is opt-in for deep audit. Audit/reconciliation input only -- never a
    // strategy input.
    if (context->completed_facts != nullptr) {
        context->completed_facts->record(
            rank, node_id, request_id, stage, generation, tick,
            terminal_status);
    }
    // (2) The DecisionMailbox write arrives through the registry's fire
    //     notifier (step 1-6 wiring). NEVER finish_node here.
    // (3) Step 1-8: schedule the per-rank post-commit deferred issue pass.
    //     The EventQueue hard rule (step 1-1) requires same-tick events
    //     from inside an event handler to go through
    //     schedule_event_deferred -- this hook runs inside Workload::call
    //     (an event-handler context). Unwired (fixture mode) => facts only.
    context->schedule_issue_pass(rank);
}

}  // namespace ExecutionDriven
}  // namespace AstraSim
