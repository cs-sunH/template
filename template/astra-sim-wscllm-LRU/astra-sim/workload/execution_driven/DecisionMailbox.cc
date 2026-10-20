/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

DecisionMailbox -- execution-driven mechanism layer (wscllm phase 1).
Implementation (方案 §4 步骤 1-6).
*******************************************************************************/

#include "astra-sim/workload/execution_driven/DecisionMailbox.hh"
#include "astra-sim/workload/execution_driven/CompletionObserver.hh"

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <utility>

namespace AstraSim {
namespace ExecutionDriven {

bool parse_completed_nodes_exact_env(bool& exact_mode, std::string& error) {
    exact_mode = false;
    error.clear();
    const char* const mode = std::getenv("ASTRA_SIM_ONLINE_COMPLETED_NODES");
    if (mode == nullptr) {
        return true;
    }
    if (std::strcmp(mode, "exact") == 0) {
        exact_mode = true;
        return true;
    }
    error = "ASTRA_SIM_ONLINE_COMPLETED_NODES must be unset (default) or "
            "'exact', got: " +
            std::string(mode);
    return false;
}

CompletedFactAccumulator::CompletedFactAccumulator() {
    // Backstop only: main_online.cc runs the same no-throw parse BEFORE the
    // driver context (and this accumulator) exists and exits through the
    // standard [Error] + exit(1) path on an invalid value, so this arm is
    // unreachable on the production path. It stays fail-closed for any
    // embedding that skips that pre-parse: the accumulator's constructor
    // used to throw here, which escaped as a bare std::terminate (the stack
    // construction sits outside every try block) and lost the message.
    std::string error;
    if (!parse_completed_nodes_exact_env(exact_mode_, error)) {
        std::cerr << "[Error] (execution_driven/decision_mailbox) " << error
                  << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

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

void CompletedFactAccumulator::set_exact_mode(const bool exact_mode) {
    if (counters_.total != 0) {
        throw std::logic_error(
            "CompletedFactAccumulator mode cannot change after a terminal");
    }
    exact_mode_ = exact_mode;
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

void DecisionMailbox::push(DecisionEvent e) {
    const Identity identity{static_cast<int>(e.reason), e.request_id, e.stage,
                            e.generation};
    if (!pending_identities_.insert(identity).second) {
        return;  // same-epoch duplicate: no-op
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
    if (!result.empty()) {
        ++delivery_count_;  // delivery epochs, not events
    }
    return result;
}

}  // namespace ExecutionDriven
}  // namespace AstraSim
