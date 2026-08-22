/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

DecisionMailbox -- execution-driven mechanism layer (wscllm phase 1).
Implementation (方案 §4 步骤 1-6).
*******************************************************************************/

#include "astra-sim/workload/execution_driven/DecisionMailbox.hh"

#include <utility>

namespace AstraSim {
namespace ExecutionDriven {

StateDelta build_state_delta(std::vector<DecisionEvent> events,
                             const uint64_t tick,
                             const uint64_t delivery_sequence,
                             const uint64_t deferred_from_tick,
                             std::vector<RankInjectedSummary>
                                 injected_unfinished) {
    // Phase 4 (v1): the compatibility form fills the v1 defaults --
    // delivery_epoch == delivery_sequence, empty completed_nodes /
    // retry_items / affected_ranks, and the self-consistent snapshot
    // handle. Existing call sites (fixtures) compile unchanged and stay
    // v1-consistent.
    return build_state_delta_v1(std::move(events), tick, delivery_sequence,
                                deferred_from_tick, {}, {}, {},
                                std::move(injected_unfinished));
}

StateDelta build_state_delta_v1(
    std::vector<DecisionEvent> events, const uint64_t tick,
    const uint64_t delivery_sequence, const uint64_t deferred_from_tick,
    std::vector<CompletedNodeFact> completed_nodes,
    std::vector<int64_t> retry_items, std::vector<int> affected_ranks,
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
    delta.retry_items = std::move(retry_items);
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
    if (!pending_identities_.insert(identity).second) {
        return;  // same-epoch duplicate: no-op
    }
    e.seq = next_seq_++;
    events_.push_back(std::move(e));
    ++event_count_;
}

bool DecisionMailbox::has_decision_work() const {
    return !events_.empty() || finalize_pending_;
}

std::vector<DecisionEvent> DecisionMailbox::drain() {
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
