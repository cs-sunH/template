/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

DecisionMailbox -- execution-driven mechanism layer (wscllm phase 1).
Implementation (方案 §4 步骤 1-6).

Aggregates only scheduler-visible events. The tick-end activity gate
(ed_driver_tick_end in main_online.cc) drains the mailbox at most once per
tick; when the mailbox has no decision work, the gate returns WITHOUT any
Python round-trip (仿真加速分析.md §3.3 门控条件).

Reason closed set (wscllm, phase-1 minimal; written into contract ④):
  ARRIVAL            -- a request arrived (ingress arrival alarm deposit).
  PREFILL_DRAIN      -- the request's prefill stage is fully terminal
                        (watch fire, stage "prefill").
  DECODE_COMPLETION  -- the request's decode stage is fully terminal
                        (watch fire, stage "decode"; request-aggregated
                        graph granularity per the step 0-4 contract).
  REQUEST_COMPLETE   -- the request's stages are all done (wrap-up / ledger
                        close-out / next-session-arrival scheduling edge).
There is deliberately NO DECODE_ITERATION_COMPLETE: one decode iteration
advancing all active requests by one token is ledger/queueing logic inside
the OnlineScheduler, not a graph-structure event (构图粒度见步骤 1-9).
Ordinary node completions, ordinary rank idles and metrics-only events NEVER
set a decision flag -- they aggregate only through the watch/fence
machinery into the stage milestones above.

Same-epoch dedup: within one delivery epoch (the interval between two
drains, i.e. one tick at the gate), a push with an identity that already
exists in the pending set is a no-op ("同 tick 同 identity 去重"). After a
drain the pending set is empty, so the same identity may legitimately
appear in a later epoch.

Counters (report at run end):
  event_count                        -- events ACCEPTED by push (post-dedup).
  delivery_count                     -- delivery EPOCHS (drain calls that
                                       delivered at least one event).
  coalescing_ratio                   -- average events per delivery epoch
                                       (delivery_count > 0 ? event_count /
                                       delivery_count : 0); >1.0 shows the
                                       tick-end gate batched several events
                                       into one Python round-trip.
  tick_end_without_decision_count    -- every no-work tick-end gate call
                                       (normal, allowed nonzero).
*******************************************************************************/

#ifndef EXECUTION_DRIVEN_DECISIONMAILBOX_HH
#define EXECUTION_DRIVEN_DECISIONMAILBOX_HH

#include <cstdint>
#include <functional>
#include <set>
#include <string>
#include <tuple>
#include <vector>

#include "astra-sim/workload/execution_driven/NodeStore.hh"

namespace AstraSim {
namespace ExecutionDriven {

/// Decision reasons (closed set, phase-1 minimal).
enum class DecisionReason : int {
    ARRIVAL = 0,
    PREFILL_DRAIN = 1,
    DECODE_COMPLETION = 2,
    REQUEST_COMPLETE = 3,
};

/// Per-reason payload (phase-1 minimal; fields documented per reason).
struct DecisionPayload {
    // ARRIVAL: request envelope facts (mirror RequestEnvelope; the
    // inter_request_interval_ns / arrival_world_ns support the next-arrival
    // scheduling edge of REQUEST_COMPLETE processing in step 1-8/1-9).
    std::string session_id;
    int turn_index = 0;
    uint64_t prefill_length = 0;
    uint64_t decode_length = 0;
    uint64_t inter_request_interval_ns = 0;
    uint64_t arrival_world_ns = 0;
    // Phase 4 (schema v1): the request's ingress serial number (globally
    // monotonic, assigned by RequestIngress) and the frozen queue index
    // (CSV data-row order, 0-based; -1 = unknown, defensive only -- the
    // loader registers every data row, turn-0 and turn>0 alike).
    uint64_t ingress_seq = 0;
    int64_t queue_index = -1;
    // PREFILL_DRAIN / DECODE_COMPLETION: watch fire facts.
    uint64_t watch_member_count = 0;
};

/// One scheduler-visible event. seq is assigned by the mailbox at push
/// (accepted events only). stage is opaque ("prefill"/"decode" in wscllm
/// phase 1); generation disambiguates same-request repeated stages.
struct DecisionEvent {
    uint64_t seq = 0;
    DecisionReason reason = DecisionReason::ARRIVAL;
    std::string request_id;
    std::string stage;
    uint64_t generation = 0;
    DecisionPayload payload;
};

/// One legacy per-node terminal-fact wire record. Production deliberately
/// does not retain or serialize these facts: completed_nodes stays present as
/// an empty schema field. ASTRA_SIM_ONLINE_COMPLETED_NODES=exact enables this
/// record for deep audit/replay only. terminal_status follows
/// NodeTerminalStatus (0 = Success, 1 = Skipped); it is never a strategy
/// decision input (红线 §0.4). See 原契约 §5.2 (文档已删除).
struct CompletedNodeFact {
    int rank = 0;
    uint64_t node_id = 0;
    std::string request_id;
    std::string stage;
    uint64_t generation = 0;
    uint64_t tick = 0;
    int terminal_status = 0;
};

/// Run-lifetime terminal accounting. These counters include terminal nodes
/// that complete after the final bridge delivery (for example an end-barrier
/// control tail), so the online run-end audit can fail closed against the
/// GraphBatchCommitter's total committed-node count without keeping node
/// metadata or JSON facts resident. `other` is an invalid observer status.
struct CompletedFactCounters {
    uint64_t total = 0;
    uint64_t success = 0;
    uint64_t skipped = 0;
    uint64_t other = 0;
};

/// Per-delivery terminal-fact collector. Default production mode updates only
/// CompletedFactCounters and drains an empty completed_nodes array. The
/// explicit ASTRA_SIM_ONLINE_COMPLETED_NODES=exact mode retains the frozen
/// legacy per-node records for deep audit/replay. The collector is
/// simulation-thread-only, matching CompletionObserver's contract.
class CompletedFactAccumulator {
  public:
    CompletedFactAccumulator();

    void record(int rank, uint64_t node_id, const char* request_id,
                const char* stage, uint64_t generation, uint64_t tick,
                int terminal_status);

    /// Transfer this delivery's wire records and reset the accumulator.
    std::vector<CompletedNodeFact> drain();
    [[nodiscard]] size_t size() const;
    [[nodiscard]] const CompletedFactCounters& counters() const {
        return counters_;
    }

    /// Test/embedding override. It is only legal before the first record of
    /// an epoch, preventing one StateDelta from mixing exact and summary
    /// records.
    void set_exact_mode(bool exact_mode);

  private:
    bool exact_mode_ = false;
    std::vector<CompletedNodeFact> exact_facts_;
    CompletedFactCounters counters_;
};

/// Snapshot handle (schema v1, phase 4): a placeholder until phase 7 wires
/// the congestion snapshot. v1 self-consistent value: {"epoch": delivery_
/// sequence, "tick": tick, "kind": ""}. Expiry rule (frozen now): the
/// handle is valid ONLY in the same delivery epoch and the same tick it was
/// created in; any cross-tick/cross-epoch use fails closed. The Python
/// validator asserts epoch == delivery_sequence && tick == tick.
/// See 原契约 §4 (文档已删除).
struct SnapshotHandle {
    uint64_t epoch = 0;
    uint64_t tick = 0;
    std::string kind;
};

/// StateDelta: the C++->Python delivery payload of one delivery epoch
/// (built from a mailbox drain by the tick-end gate). GraphBatch (the
/// response type) lands with the step-1-7 bridge header. Schema v1 fields
/// (phase 4) were documented in a deleted contract doc; this struct is
/// the authority.
struct StateDelta {
    uint64_t delivery_sequence = 0;  // monotonically increasing epoch number
    // Phase 4 (v1): the delivery-epoch counter. In v1 it is ALWAYS equal to
    // delivery_sequence (the Python validator asserts equality); the field
    // exists separately for the future deferred-epoch interleave (one epoch
    // built from multiple drains) -- v1 forbids divergence.
    uint64_t delivery_epoch = 0;
    uint64_t tick = 0;               // EventQueue global time of the epoch
    // Step 1-11: when the epoch's events were formed at a tick EARLIER than
    // the delivery tick (the explicit T+1 next-decision-boundary wakeup --
    // 仿真加速分析.md §4.3), records the formation tick; 0 = the delivery
    // happened in the same tick the events formed (the ordinary case). The
    // "T -> T+1 延后" must be explicit, never silent (方案 step 1-11).
    uint64_t deferred_from_tick = 0;
    std::vector<DecisionEvent> events;
    // Phase 4 (v1): per-node terminal facts since the previous delivery
    // (hook facts buffer; drained into the delivery at the tick-end gate).
    std::vector<CompletedNodeFact> completed_nodes;
    // Phase 4 (v1): affected rank set = union of the member ranks of this
    // epoch's completed_groups (WatchFire::member_ranks), sorted unique.
    std::vector<int> affected_ranks;
    // Phase 4 (v1): placeholder handle (see SnapshotHandle; expiry rule
    // frozen now).
    SnapshotHandle snapshot_handle;
    // Phase-3 sensing (方案 §6.2 操作 1): per-rank injected-unfinished ledger
    // summary at this delivery epoch. Empty when --sensing-enabled is off
    // (perception feature flag; default off until phase 6). Query/audit data
    // only -- the strategy's red-line decision inputs never consume it.
    std::vector<RankInjectedSummary> injected_unfinished;
};

/// Assemble a StateDelta from a drained event list (step-1-6 tick-end gate).
/// deferred_from_tick: 0 = same-tick delivery, nonzero = the explicit
/// T+1 wakeup delivered events formed at that tick (step 1-11).
/// injected_unfinished: phase-3 sensing summary (empty when sensing is off).
/// Phase-4 v1 defaults: delivery_epoch == delivery_sequence; completed_nodes
/// / affected_ranks empty; snapshot_handle self-consistent
/// {epoch: delivery_sequence, tick: tick, kind: ""} -- existing 4/5-arg call
/// sites (fixtures) compile unchanged and stay v1-consistent.
StateDelta build_state_delta(
    std::vector<DecisionEvent> events, uint64_t tick,
    uint64_t delivery_sequence, uint64_t deferred_from_tick = 0,
    std::vector<RankInjectedSummary> injected_unfinished = {});
/// v1: explicit form -- callers with completed-facts / affected-ranks
/// payloads use the full signature (the tick-end gate in main_online.cc).
/// injected_unfinished (phase-3 sensing) defaults to empty.
StateDelta build_state_delta_v1(
    std::vector<DecisionEvent> events, uint64_t tick,
    uint64_t delivery_sequence, uint64_t deferred_from_tick,
    std::vector<CompletedNodeFact> completed_nodes,
    std::vector<int> affected_ranks,
    std::vector<RankInjectedSummary> injected_unfinished = {});

class DecisionMailbox {
  public:
    /// Push one event; same-epoch same-identity duplicates are no-ops
    /// ("同 tick 同 identity 去重"; identity = reason/request_id/stage/
    /// generation). Accepted events get consecutive seqs.
    void push(DecisionEvent e);

    /// True iff a delivery epoch has work: pending events.
    [[nodiscard]] bool has_decision_work() const;

    /// Deliver all pending events in insertion order and clear the pending
    /// set (a new dedup epoch starts after the call).
    std::vector<DecisionEvent> drain();

    // --- counters (report at run end) ---
    [[nodiscard]] uint64_t event_count() const { return event_count_; }
    [[nodiscard]] uint64_t delivery_count() const { return delivery_count_; }
    [[nodiscard]] double coalescing_ratio() const {
        return delivery_count_ > 0
                   ? static_cast<double>(event_count_) / delivery_count_
                   : 0.0;
    }
    [[nodiscard]] uint64_t tick_end_without_decision_count() const {
        return tick_end_without_decision_count_;
    }

    /// Tick-end gate no-work path (normal, allowed nonzero).
    void count_tick_end_without_decision() {
        ++tick_end_without_decision_count_;
    }

  private:
    using Identity = std::tuple<int, std::string, std::string, uint64_t>;

    uint64_t next_seq_ = 1;
    std::vector<DecisionEvent> events_;
    std::set<Identity> pending_identities_;

    uint64_t event_count_ = 0;
    uint64_t delivery_count_ = 0;
    uint64_t tick_end_without_decision_count_ = 0;
};

}  // namespace ExecutionDriven
}  // namespace AstraSim

#endif  // EXECUTION_DRIVEN_DECISIONMAILBOX_HH
