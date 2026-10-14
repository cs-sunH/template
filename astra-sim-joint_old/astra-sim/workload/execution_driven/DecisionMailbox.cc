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
#include <limits>
#include <stdexcept>
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
    // L9 diagnostic (2026-09-29): the name itself is consistent across the
    // repo (DecisionMailbox.hh -- verified), so the failure mode is a bad
    // VALUE, and the throw used to escape the DecisionMailbox member
    // construction in main_online as a bare terminate. Print an
    // attributable line before rethrowing so the invalid value and its
    // remedy are in the log even if the terminate handler's what() output
    // is lost.
    std::fprintf(stderr,
                 "[Error] (execution_driven/mailbox) env "
                 "ASTRA_SIM_ONLINE_COMPLETED_NODES=%s is invalid (must be "
                 "unset or 'exact'); completed-nodes exact audit stays off "
                 "only if unset\n",
                 mode);
    // R1-B-L1 port (2026-09-30): this runs from the
    // CompletedFactAccumulator constructor deep inside main's stack; a
    // thrown runtime_error had no catcher and terminated the process
    // through std::terminate, bypassing the binary's uniform [Error]
    // channel. Fail through the channel instead.
    std::fprintf(stderr,
                 "[Error] (execution_driven/online) "
                 "ASTRA_SIM_ONLINE_COMPLETED_NODES must be unset (default) "
                 "or 'exact', got: %s\n",
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

void CompletedFactAccumulator::set_exact_mode(const bool exact_mode) {
    if (counters_.total != 0) {
        throw std::logic_error(
            "CompletedFactAccumulator mode cannot change after a terminal");
    }
    exact_mode_ = exact_mode;
}

StateDelta build_state_delta(std::vector<DecisionEvent> events,
                             const uint64_t tick,
                             const uint64_t delivery_sequence,
                             const uint64_t deferred_from_tick,
                             std::vector<RankInjectedSummary>
                                 injected_unfinished) {
    // Phase 4 (v1): the compatibility form fills the v1 defaults --
    // delivery_epoch == delivery_sequence, empty completed_nodes /
    // affected_ranks, and the self-consistent snapshot handle. Existing
    // call sites (fixtures) compile unchanged and stay v1-consistent.
    return build_state_delta_v1(std::move(events), tick, delivery_sequence,
                                deferred_from_tick, {}, {},
                                std::move(injected_unfinished));
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

void DecisionMailbox::push(DecisionEvent e) {
    const Identity identity{static_cast<int>(e.reason), e.request_id, e.stage,
                            e.generation};
    if (pending_identities_.find(identity) != pending_identities_.end()) {
        return;  // same-epoch duplicate: no-op
    }
    if (next_seq_ == std::numeric_limits<uint64_t>::max() ||
        event_count_ == std::numeric_limits<uint64_t>::max()) {
        std::fprintf(stderr,
                     "[Error] (execution_driven/mailbox) decision event "
                     "sequence/counter exhausted (fail-closed)\n");
        std::exit(EXIT_FAILURE);
    }
    const auto inserted = pending_identities_.emplace(identity);
    if (!inserted.second) {
        return;
    }
    e.seq = next_seq_;
    try {
        events_.push_back(std::move(e));
    } catch (...) {
        pending_identities_.erase(inserted.first);
        throw;
    }
    ++next_seq_;
    ++event_count_;
}

bool DecisionMailbox::has_decision_work() const {
    return !events_.empty() || finalize_pending_;
}

std::vector<DecisionEvent> DecisionMailbox::drain() {
    if (!events_.empty() &&
        delivery_count_ == std::numeric_limits<uint64_t>::max()) {
        std::fprintf(stderr,
                     "[Error] (execution_driven/mailbox) delivery "
                     "counter exhausted (fail-closed)\n");
        std::exit(EXIT_FAILURE);
    }
    std::vector<DecisionEvent> result = std::move(events_);
    events_.clear();
    pending_identities_.clear();  // a new dedup epoch starts
    // The drain delivers ALL pending work: events and/or the finalize flag
    // (a finalize-only epoch delivers an empty delta).
    finalize_pending_ = false;
    if (!result.empty()) {
        ++delivery_count_;  // delivery epochs, not events
    }
    return result;
}

}  // namespace ExecutionDriven
}  // namespace AstraSim
