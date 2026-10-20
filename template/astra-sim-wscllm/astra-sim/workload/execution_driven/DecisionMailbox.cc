/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

DecisionMailbox -- execution-driven mechanism layer (wscllm phase 1).
Implementation (方案 §4 步骤 1-6).
*******************************************************************************/

#include "astra-sim/workload/execution_driven/DecisionMailbox.hh"
#include "astra-sim/workload/execution_driven/CompletionObserver.hh"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <utility>

namespace AstraSim {
namespace ExecutionDriven {

namespace {

bool completed_nodes_exact_from_env() {
    const char* const mode = std::getenv("ASTRA_SIM_ONLINE_COMPLETED_NODES");
    if (mode == nullptr) {
        return false;
    }
    if (std::strcmp(mode, "exact") == 0) {
        return true;
    }
    // Fail-closed with the module's standard one-line [Error] + non-zero
    // exit (2026-10-01 fix): this fires from the CompletedFactAccumulator
    // constructor, whose only construction site is a stack object in
    // main_online.cc with no try/catch -- the former std::runtime_error
    // terminated the process via std::terminate (still non-zero, but with
    // an uncaught-exception trace instead of the disciplined diagnostic).
    std::fprintf(stderr,
                 "[Error] (execution_driven/mailbox) ASTRA_SIM_ONLINE_"
                 "COMPLETED_NODES must be unset (default) or 'exact', got: "
                 "%s\n",
                 mode);
    std::exit(EXIT_FAILURE);
}

}  // namespace

CompletedFactAccumulator::CompletedFactAccumulator()
    : exact_mode_(completed_nodes_exact_from_env()) {}

void CompletedFactAccumulator::record(
    const int rank, const uint64_t node_id, const char* request_id,
    const char* stage, const uint64_t generation, const uint64_t tick,
    const int terminal_status) {
    ++counters_.total;
    switch (terminal_status) {
        case static_cast<int>(NodeTerminalStatus::Success):
            ++counters_.success;
            break;
        case static_cast<int>(NodeTerminalStatus::Skipped):
            ++counters_.skipped;
            break;
        default:
            ++counters_.other;
            break;
    }
    if (!exact_mode_) {
        return;
    }
    CompletedNodeFact fact;
    fact.rank = rank;
    fact.node_id = node_id;
    fact.request_id = request_id == nullptr ? "" : request_id;
    fact.stage = stage == nullptr ? "" : stage;
    fact.generation = generation;
    fact.tick = tick;
    fact.terminal_status = terminal_status;
    exact_facts_.push_back(std::move(fact));
}

std::vector<CompletedNodeFact> CompletedFactAccumulator::drain() {
    std::vector<CompletedNodeFact> result = std::move(exact_facts_);
    exact_facts_.clear();
    return result;
}

size_t CompletedFactAccumulator::size() const {
    return exact_facts_.size();
}

StateDelta build_state_delta_v1(
    std::vector<DecisionEvent> events, const uint64_t tick,
    const uint64_t delivery_sequence, const uint64_t deferred_from_tick,
    std::vector<CompletedNodeFact> completed_nodes,
    std::vector<int> affected_ranks,
    std::vector<RankInjectedSummary> injected_unfinished) {
    StateDelta delta;
    delta.delivery_sequence = delivery_sequence;
    // v1: the epoch counter is ALWAYS the delivery sequence (the Python
    // validator asserts equality; divergence is reserved for the future
    // deferred-epoch interleave and fails closed today).
    delta.delivery_epoch = delivery_sequence;
    delta.tick = tick;
    delta.deferred_from_tick = deferred_from_tick;
    delta.events = std::move(events);
    delta.completed_nodes = std::move(completed_nodes);
    delta.affected_ranks = std::move(affected_ranks);
    // v1 placeholder + the frozen expiry rule (valid only in the delivery
    // epoch and tick it was created in): the self-consistent handle.
    delta.snapshot_handle.epoch = delivery_sequence;
    delta.snapshot_handle.tick = tick;
    delta.snapshot_handle.kind.clear();
    delta.injected_unfinished = std::move(injected_unfinished);
    return delta;
}

StateDelta build_state_delta_v2(
    std::vector<DecisionEvent> events, const uint64_t tick,
    const uint64_t delivery_sequence, const uint64_t deferred_from_tick,
    std::vector<CompletedNodeFact> completed_nodes,
    std::vector<int> affected_ranks,
    std::vector<WorkspaceRankState> workspace_state,
    std::vector<RankInjectedSummary> injected_unfinished) {
    StateDelta delta = build_state_delta_v1(
        std::move(events), tick, delivery_sequence, deferred_from_tick,
        std::move(completed_nodes), std::move(affected_ranks),
        std::move(injected_unfinished));
    if (workspace_state.empty()) {
        throw std::invalid_argument(
            "v2 workspace_state must cover at least rank zero");
    }
    int64_t expected_rank = 0;
    for (const auto& rank_state : workspace_state) {
        if (rank_state.rank < 0 || rank_state.rank != expected_rank) {
            throw std::invalid_argument(
                "v2 workspace_state must cover contiguous ranks starting at "
                "zero");
        }
        ++expected_rank;
    }
    delta.schema_version = 2;
    delta.workspace_state = std::move(workspace_state);
    return delta;
}

void DecisionMailbox::push(DecisionEvent e, const uint64_t formation_tick) {
    push_common(e, true, formation_tick);
}

void DecisionMailbox::push(DecisionEvent e) {
    // Fixture convenience: no formation tick, excluded from the minimum.
    push_common(e, false, 0);
}

void DecisionMailbox::push_common(DecisionEvent& e, const bool has_tick,
                                  const uint64_t formation_tick) {
    const Identity identity{static_cast<int>(e.reason), e.request_id, e.stage,
                            e.generation};
    if (!pending_identities_.insert(identity).second) {
        return;  // same-epoch duplicate: no-op (the original event's
                 // formation tick already stands)
    }
    // Step 1-11: track the pending set's earliest formation tick (the
    // delivery gate reads it for StateDelta.deferred_from_tick).
    if (has_tick && (!pending_formation_valid_ ||
                     formation_tick < min_pending_formation_tick_)) {
        min_pending_formation_tick_ = formation_tick;
        pending_formation_valid_ = true;
    }
    e.seq = next_seq_++;
    events_.push_back(std::move(e));
    ++event_count_;
}

bool DecisionMailbox::has_decision_work() const {
    return !events_.empty();
}

std::vector<DecisionEvent> DecisionMailbox::drain() {
    std::vector<DecisionEvent> result = std::move(events_);
    events_.clear();
    pending_identities_.clear();  // a new dedup epoch starts
    pending_formation_valid_ = false;  // the formation record ends with it
    if (!result.empty()) {
        ++delivery_count_;  // delivery epochs, not events
    }
    return result;
}

}  // namespace ExecutionDriven
}  // namespace AstraSim
