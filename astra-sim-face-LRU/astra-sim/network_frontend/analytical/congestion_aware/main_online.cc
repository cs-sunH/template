/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

main_online.cc -- online (execution-driven) entry point (wscllm phase 1).

Step-1-2 implementation (方案 §4 步骤 1-2): initialization mirrors the static
main.cc (MetricCollector init, topology, FluidScheduler, Sys) but

  - parses the online CLI family explicitly (OnlineCli). Step 1-8/1-9:
    --online-mode takes the mode token replay|strategy; replay serves the
    offline decision log (LUT clock, step 1-8), strategy runs the real policy
    scheduler with real physics (step 1-9);
  - constructs Sys in ExecutionMode::Online with an injected GraphSource
    (NodeStore-backed), so no ETFeeder is built and no .et file is required;
  - runs the request-neutral main loop, draining ingress commands before
    every state decision (drain-before-finished per 主控裁决 2026-08-15:
    commands submitted before Close are never dropped).  svc.finished()
    denotes logical request-lifecycle completion only; it is not an immediate
    process-exit condition.  The loop must continue to drain the EventQueue,
    its deferred issue pass, and the DecisionMailbox, and exits only after all
    four are physically quiescent.  Conversely, an empty EventQueue never
    ends a run while the service is still logically active.

Step 1-6: the DecisionMailbox aggregates scheduler-visible events (ARRIVAL
from the ingress alarm, PREFILL_DRAIN / DECODE_COMPLETION / REQUEST_COMPLETE
from watch fires) and the tick-end activity gate (ed_driver_tick_end) drains
it at most once per tick, delivering one StateDelta epoch to Python.

Step 1-8: the decision loop is fully wired (决策边界驱动的 Execution-Driven
闭环):
  - ed_driver_tick_end drains the mailbox into one StateDelta, delivers it
    through the FileDecisionBridge (blocking round-trip; backpressure: one
    in-flight delivery at most), and schedules the four-phase commit as a
    same-tick deferred event (hard rule: from inside the tick-end callback,
    same-tick events MUST go through schedule_event_deferred);
  - ed_commit_cb executes the four-phase commit:
      1. nodes  -> NodeStore::add_node, extending the persistent per-rank
         affine graph-batch-id -> store-id translation (跨批次);
      2. parent_edges -> add_dependency (Data kind; the phase-1 DepKind
         resolves all kinds identically);
      3. watches -> WatchRegistry::register_stage_watch with the
         (rank, json member id) -> store id translation and the explicit
         satisfying status set;
      4. future_alarms -> RequestIngress::schedule_future_arrival (the
         next-turn arrival alarm; replay authority = the offline prefill
         record tick);
      then the per-rank issue pass (workload->issue_dep_free_nodes()),
      the REQUEST_COMPLETE ServiceCoordinator accounting (after the alarms:
      the last alarm of the batch keeps the service ACTIVE until the next
      arrival), and the commit ack (send_commit_ack; Python's provisional
      ledger finalizes on it).
  - watch fire -> DecisionMailbox: the decode watch fire is the
    request-completion point, so it pushes BOTH DECODE_COMPLETION and
    REQUEST_COMPLETE (dedup identity = reason/request_id/stage/generation --
    the reasons differ, so both enter the same epoch safely).
  - the CSV loader submits only the rows with an explicit
    session_arrival_time_ns (turn-0 rows; the turn>0 arrivals are
    future_alarm-driven -- submitting them here would double-arrive them)
    and counts ALL data rows for the run-end completion assertion
    (expected: 1177 for the 20.csv first-30-seconds input).
  - run end: svc.finished() marks logical completion; process exit additionally
    requires the EventQueue, deferred issue pass, and DecisionMailbox to be
    drained.  Assertions completed == CSV data rows
    (acceptance: 1177/1177 replay).

Step 1-10 (runners + IDLE fixture):
  - --request-queue-csv is optional (合同② request-neutral default): absent
    -> expected_requests = 0 and the service starts IDLE; no preset/stub
    queue is ever created. The run-end delivery gate is relaxed accordingly
    (delivery_count == 0 is legal for a zero-request run).
  - lifecycle transition log: a ServiceCoordinator transition hook prints
    every IDLE/ACTIVE/DRAINING/FINISHED transition with elapsed + wall
    timestamps ("[online] lifecycle: ...").
  - --command-fifo <path>: a detached external-producer thread reads JSON
    Submit/CloseInput/EndOfFile/Error lines from the FIFO and writes the
    thread-safe bounded ingress command queue (合同② 线程合同; the decision
    bridge stays the decision channel only). This is the IDLE fixture's
    injection channel -- the official runners never use it. Phase-7 §10.7:
    the terminal command kinds (CloseInput/EndOfFile/Error) are
    distinguished in the drain path and the run-end lifecycle audit
    (close_source=explicit|eof|error), per 合同② EOF vs 显式 close vs
    异常退出 三态区分; Error is fail-closed abort (never a silent close).
  - ServiceCoordinator::maybe_finish returns a fully drained ACTIVE to IDLE
    while the input stays open (五态迁移 IDLE->ACTIVE->IDLE->DRAINING->
    FINISHED); the official runners close the input at startup, so their
    behavior is unchanged.
*******************************************************************************/

#include "astra-sim/common/Logging.hh"
#include "astra-sim/system/Sys.hh"
#include "astra-sim/workload/MetricCollector.hh"
#include "astra-sim/workload/Workload.hh"
#include "astra-sim/workload/execution_driven/DecisionBridge.hh"
#include "astra-sim/workload/execution_driven/ExecutionMode.hh"
#include "astra-sim/workload/execution_driven/GraphBatchCommitter.hh"
#include "astra-sim/workload/execution_driven/NodeStore.hh"
#include "astra-sim/workload/execution_driven/OnlineCli.hh"
#include "astra-sim/workload/execution_driven/OnlineStatsCounters.hh"
#include "astra-sim/workload/execution_driven/RequestIngress.hh"
#include "astra-sim/workload/execution_driven/ServiceCoordinator.hh"
#include "astra-sim/workload/execution_driven/WindowedTraceReader.hh"
#include "astra-sim/workload/execution_driven/WatchRegistry.hh"
#include "common/CmdLineParser.hh"
#include "congestion_aware/CongestionAwareNetworkApi.hh"
#include <astra-network-analytical/common/EventQueue.h>
#include <astra-network-analytical/common/NetworkParser.h>
#include <astra-network-analytical/congestion_aware/Helper.h>
#include <json/json.hpp>
#include <remote_memory_backend/analytical/AnalyticalRemoteMemory.hh>

#include <sys/resource.h>

#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <fstream>
#include <iostream>
#include <map>
#include <optional>
#include <set>
#include <sstream>
#include <thread>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

using namespace AstraSim;
using namespace Analytical;
using namespace AstraSimAnalytical;
using namespace AstraSimAnalyticalCongestionAware;
using namespace AstraSim::ExecutionDriven;
using namespace NetworkAnalytical;
using namespace NetworkAnalyticalCongestionAware;

namespace {

// ---------------------------------------------------------------------------
// Step-1-8 decision-loop context
// ---------------------------------------------------------------------------

// NOTE (phase 5): GraphBatchCommitter owns the persistent per-rank affine
// json-id -> store-id translation and the in-flight request tracking.

struct OnlineDriverContext {
    DecisionMailbox* mailbox = nullptr;
    NetworkAnalytical::EventQueue* event_queue = nullptr;
    FileDecisionBridge* bridge = nullptr;
    ServiceCoordinator* svc = nullptr;
    WatchRegistry* watch_registry = nullptr;
    // The per-rank NodeStore-backed sources are kept alive here (Sys stores
    // its own shared_ptr; this vector is the commit's access path).
    std::vector<std::shared_ptr<NodeStoreGraphSource>>* graph_sources =
        nullptr;
    // Phase 5 (方案 §8): the atomic GraphBatch committer -- Phase A
    // validate (pure) then Phase B commit; owns the persistent store-id
    // map, the in-flight request tracking and the phase-5 counters.
    GraphBatchCommitter* committer = nullptr;
    uint64_t delivery_seq = 0;
    // Step 1-11: pending T->T+1 deferral record. Set by the main loop when
    // it schedules the explicit next-decision-boundary wakeup; consumed (and
    // reset) by the next ed_driver_tick_end delivery, which serializes it as
    // StateDelta.deferred_from_tick. 0 = no pending deferral.
    uint64_t deferred_from_tick = 0;
    // Phase-3 perception feature flag (方案 §6.2 操作 1; default off until
    // phase 6): when set, the tick-end gate computes the per-rank
    // injected-unfinished ledger summary into StateDelta.injected_unfinished.
    bool sensing_enabled = false;
    // Phase 4 (schema v1): terminal accounting. Production updates only
    // O(1) counters and preserves completed_nodes as []; the explicit exact
    // audit mode retains legacy per-node facts until the next delivery.
    CompletedFactAccumulator completed_facts;
    // Phase 4 (schema v1): per-epoch affected-rank accumulator
    // (StateDelta.affected_ranks). The watch-fire notifier (main scope)
    // appends fire.member_ranks; the tick-end gate sorts/dedups into the
    // delivery and clears it.
    std::vector<int>* affected_ranks_accumulator = nullptr;
    // Phase 6 (方案 §9.1): per-run mechanism counters (C++ side; the bridge
    // round-trip/channel counters live in FileDecisionBridge::Stats and are
    // printed through bridge->stats_report() at run end). See
    // OnlineStatsCounters.hh for the field semantics.
    OnlineStatsCounters stats;
    // C1 validate switch (2026-08-28, --online-validate), B.2 cleanup
    // (2026-09-05): strict <0|1> enum -- 1 = validate every batch (pre-C1
    // behavior, the CLI default), 0 = production skip. Copied from the
    // parsed CLI here so ed_commit_cb can gate Phase A without reaching
    // back into main's scope.
    int online_validate = 1;
};

// The commit payload: the StateDelta + GraphBatch response of one delivery
// epoch plus the request ids whose REQUEST_COMPLETE event was in that epoch
// (the commit performs the svc accounting AFTER the future alarms are
// scheduled). Phase 5: the delta travels with the batch because the
// committer validates the batch against the delta facts (epoch match,
// arrivals/completions eligibility, alarm recency).
struct CommitArg {
    OnlineDriverContext* driver = nullptr;
    StateDelta delta;
    GraphBatch batch;
    std::vector<std::string> completed_requests;
};

[[noreturn]] void online_fatal(const std::string& what) {
    std::fprintf(stderr, "[Error] (execution_driven/online) %s\n",
                 what.c_str());
    std::abort();
}

// ---------------------------------------------------------------------------
// Two-phase atomic commit (step 1-8 + phase 5 方案 §8)
// ---------------------------------------------------------------------------

// Phase-7 §10.3: online dynamic anchor registration from one committed
// GraphBatch (plan step 1-8 contract ⑤/⑥, implemented at phase 7 -- the
// static manifest sparse events only match the offline per-rank node-id
// sequence; the online dynamic graph interleaves requests across stages so
// most boundary nodes would not match, measured 708/1177 completion
// anchors). The batch's nodes carry request_id/stage (graph_batch_builder
// contract); per (rank, request_id, stage) the FIRST node of the group is
// the start anchor (issue event) and the LAST is the end anchor (complete
// event). Transfer nodes of the KV routes (node names contain "kv" --
// history_kv / prefill_to_decode_kv) register the memory anchor (code 7)
// on the last transfer node of each (request_id, rank): the offline
// doc-sec.4.3 semantics (transfer-arrival memory anchor = last transfer
// node on each target rank). Watch members supply the prefill/decode rank
// sets.
//
// Installed as GraphBatchCommitter::Context::metrics_anchor_hook and
// invoked from inside commit() (Phase B-1.5) -- see the "PHASE-7 §10.3
// ROOT-CAUSE FIX" note at the definition.
//
// PHASE-7 §10.3 ROOT-CAUSE FIX: the anchors are registered from inside
// GraphBatchCommitter::commit (via the Context::metrics_anchor_hook,
// Phase B-1.5, BEFORE the issue pass) because the anchor node ids must be
// the STORE ids, not the graph-batch json ids. NodeStore assigns store ids
// starting at 1 per rank while the online graph's json ids start at 0, so
// registering with the json ids off-by-ones every anchor: the id-0 start
// anchors (arrival_timer_gate / prefill_to_decode_kv_recv) would NEVER
// fire (no store id 0 exists) and every other anchor would bind to the
// PREVIOUS node on the rank (measured: 103c diagnostic run -- only
// late-request issue anchors ever matched, 20/1177 issues, and those were
// shifted by one). The watch registry and these anchors instead translate
// the committed ids through GraphBatchCommitter's per-rank affine resolver.
// It is exact for every committed JSON id (including a preseeded NodeStore),
// does not allocate, and avoids rebuilding a whole-batch id hash.
static void register_online_metrics_anchors(
    const GraphBatch& batch,
    const GraphBatchCommitter& committer,
    std::vector<std::shared_ptr<NodeStoreGraphSource>>& graph_sources) {
    struct NodeRef {
        int rank = -1;
        uint64_t node_id = 0;
    };
    // R2 (2026-08-29) anchor fast path: registration returns true exactly
    // when a routing entry exists on the registered edge, so the flag is
    // set HERE (Phase B-1.5: after add_node assigned the store ids, before
    // the issue pass) -- add_node itself cannot know yet whether the node
    // will be anchored. OR semantics per edge; a defensive rank-bounds
    // skip mirrors the collector's unknown-request no-op.
    const auto set_anchor_flags = [&graph_sources](int rank, uint64_t node_id,
                                                   bool issue, bool complete) {
        if (rank < 0 ||
            rank >= static_cast<int>(graph_sources.size())) {
            return;  // defensive: bounds mirror the collector's no-ops
        }
        graph_sources[rank]->store().set_metric_anchor_flags(node_id, issue,
                                                             complete);
    };
    // (request_id, rank, stage) -> first node of the group (start anchor).
    std::map<std::tuple<std::string, int, std::string>, NodeRef> first_of;
    // (request_id, rank) -> last KV-route transfer node.
    std::map<std::pair<std::string, int>, NodeRef> last_transfer;
    // WP9 (WP9_CONTRACT §1): every first-token marker node of the batch,
    // keyed (request_id, rank) -- each occurrence registers its own anchor
    // (apply_event keeps the min tick per subject, so duplicates are
    // harmless and cross-TP ranks each contribute).
    std::vector<std::pair<std::pair<std::string, int>, NodeRef>>
        first_token_nodes;
    // C1 (2026-08-29): typed walk (this was DOM walk #4) -- request_id /
    // stage / rank / json id / name are direct field reads off the parsed
    // batch; the vector preserves the emission order, so the first/last
    // group selections see the exact same sequence the DOM walk saw.
    for (const auto& parsed : batch.nodes) {
        const std::string& request_id = parsed.node.request_id;
        const std::string& stage = parsed.node.stage;
        const int rank = parsed.node.rank;
        const uint64_t json_id = parsed.json_id;
        const auto store_id = committer.resolve_store_id(rank, json_id);
        if (!store_id.has_value()) {
            continue;  // defensive: B-1 just inserted every batch node
        }
        const std::string& name = parsed.node.name;
        if (name.find("first_token") != std::string::npos) {
            // WP9 observation-only marker: never feeds the first_of
            // start-anchor or last_transfer groups below.
            first_token_nodes.push_back(
                {{request_id, rank}, NodeRef{rank, *store_id}});
            continue;
        }
        const auto group = std::make_tuple(request_id, rank, stage);
        auto& first = first_of[group];
        if (first.rank < 0) {
            first = {rank, *store_id};
        }
        if (name.find("kv") != std::string::npos) {
            last_transfer[{request_id, rank}] = {rank, *store_id};
        }
    }
    // Start anchors: the FIRST node of each (request, rank, stage) group
    // (issue event; min-tick semantics in apply_event, so re-registration
    // across batches is harmless).
    for (const auto& [group, ref] : first_of) {
        const auto& [request_id, rank, stage] = group;
        (void)rank;
        const std::string kind =
            stage == "prefill" ? "prefill_start" : "decode_start";
        if (MetricCollector::instance().online_register_node_anchor(
                ref.rank, ref.node_id, request_id, kind, false)) {
            set_anchor_flags(ref.rank, ref.node_id, /*issue=*/true,
                             /*complete=*/false);
        }
    }
    // End anchors come from the WATCH MEMBERS, not from the last batch node:
    // the watch member value is the stage's real last node per rank (the
    // node whose completion satisfies the watch), while the batch's last
    // node is often a 31-byte end-barrier control tail that finishes after
    // the watch fired (known R5/R7 window) -- anchoring to it would push
    // prefill_end/completion past decode_start/sim_end and trip ordering
    // violations. Same source as the watch registry = same semantics. The
    // member values are json ids and must be translated to store ids the
    // same way.
    for (const auto& watch : batch.watches) {
        const std::string& request_id = watch.request_id;
        const std::string& stage = watch.stage;
        const bool is_prefill = stage == "prefill";
        std::vector<int> ranks;
        // C1: typed members in JSON key order (rank-string ascending --
        // the same iteration order the DOM walk saw; the stoi and the
        // defensive try/catch around it are gone: the parser validated
        // every member key/range once).
        for (const auto& member : watch.members) {
            const int rank = member.rank;
            ranks.push_back(rank);
            const uint64_t json_id = member.json_id;
            const auto store_id = committer.resolve_store_id(rank, json_id);
            if (!store_id.has_value()) {
                continue;  // defensive: every watch member is a batch node
            }
            if (MetricCollector::instance().online_register_node_anchor(
                    rank, *store_id, request_id,
                    is_prefill ? "prefill_end" : "completion", false)) {
                set_anchor_flags(rank, *store_id, /*issue=*/false,
                                 /*complete=*/true);
            }
        }
        MetricCollector::instance().online_register_ranks(
            request_id, is_prefill, ranks);
    }
    // Transfer anchors (doc sec.4.3): the last KV-route node per
    // (request_id, rank) -- history_kv / prefill_to_decode_kv routes.
    for (const auto& [key, ref] : last_transfer) {
        if (MetricCollector::instance().online_register_node_anchor(
                ref.rank, ref.node_id, key.first, "", true)) {
            set_anchor_flags(ref.rank, ref.node_id, /*issue=*/false,
                             /*complete=*/true);
        }
    }
    // WP9 first-token anchors (WP9_CONTRACT §1): kind "first_token" -> event
    // code 8 (complete edge, subject=request, min tick per subject). Every
    // occurrence registers -- cross-TP ranks each contribute their own
    // completion and the collector takes the min.
    for (const auto& [key, ref] : first_token_nodes) {
        if (MetricCollector::instance().online_register_node_anchor(
                ref.rank, ref.node_id, key.first, "first_token", false)) {
            set_anchor_flags(ref.rank, ref.node_id, /*issue=*/false,
                             /*complete=*/true);
        }
    }
}

// Phase A (validate) then Phase B (commit) through GraphBatchCommitter. The
// phase-1 inline four-phase commit (nodes added as parsed, later failures
// leaving earlier nodes committed) is REPLACED: a validation failure aborts
// the run BEFORE a single node/watch/alarm/ledger action enters the official
// state (fail-closed, zero side effects; 方案 §8.2).
void ed_commit_cb(void* arg) {
    auto* commit = static_cast<CommitArg*>(arg);
    auto* driver = commit->driver;
    const GraphBatch& batch = commit->batch;

    // ---- Phase A: full pre-commit validation (pure, zero state mutation) ----
    // C1 (2026-08-28, --online-validate), B.2 cleanup (2026-09-05): 1 = every
    // batch (pre-C1 behavior, the CLI default -- fail-closed for bare
    // invocations); 0 = production skip. The former N >= 2 sample-every-Nth
    // tier is removed (the parser only accepts <0|1>).
    // Full-validation batches use one atomic validate_and_commit()
    // call; validation-off batches retain commit()'s mandatory preflight.
    // Phase 6 (方案 §9.1): graph_validate_ns is the API's validate_impl-only
    // duration; the public validate() itself remains pure for fixtures.
    const bool should_validate = driver->online_validate == 1;
    if (should_validate) {
        GraphBatchCommitter::ValidateAndCommitResult validation_result;
        try {
            validation_result =
                driver->committer->validate_and_commit(commit->delta, batch);
            if (validation_result.error.has_value()) {
                online_fatal(
                    "GraphBatch validation failed (delivery_sequence=" +
                    std::to_string(commit->delta.delivery_sequence) + "): " +
                    *validation_result.error);
            }
        } catch (const std::exception& exc) {
            online_fatal("GraphBatch commit threw (delivery_sequence=" +
                         std::to_string(commit->delta.delivery_sequence) +
                         "): " + exc.what());
        }
        driver->stats.graph_validate_count += 1;
        driver->stats.graph_validate_ns += validation_result.validation_ns;
    }

    // ---- Phase-7 §10.3: dynamic metrics anchor registration runs INSIDE
    //      commit() (Phase B-1.5, via the Context::metrics_anchor_hook),
    //      AFTER validation (fail-closed: a rejected batch registers
    //      nothing) and AFTER the nodes are in the NodeStore (the anchors
    //      must bind to the STORE ids, not the json ids -- NodeStore assigns
    //      store ids starting at 1 per rank; see the definition's
    //      ROOT-CAUSE FIX note). The hook fires BEFORE the commit's issue
    //      pass, so the anchors always exist when on_node_issue fires.
    //      No-op when metrics are disabled (hook not installed). ----

    // ---- Phase B: full-validation batches already committed inside
    //      validate_and_commit(), so no caller-visible gap exists between the
    //      full validation and mutation.  Validation-off batches retain the
    //      independent mandatory affine/id/liveness preflight in commit(). ----
    if (!should_validate) {
        try {
            driver->committer->commit(commit->delta, batch);
        } catch (const std::exception& exc) {
            online_fatal("GraphBatch commit threw (delivery_sequence=" +
                         std::to_string(commit->delta.delivery_sequence) +
                         "): " + exc.what());
        }
    }

    // ---- REQUEST_COMPLETE accounting (after the alarms: the batch's last
    //      future alarm keeps the service ACTIVE until the next arrival) ----
    for (size_t i = 0; i < commit->completed_requests.size(); ++i) {
        driver->svc->on_request_completed();
    }

    // ---- Phase 4 (schema v1): remove every watch of the completed
    //      requests (per-request index; by now both stage watches fired).
    //      This keeps the phase-4 run-end end-audit empty-registry
    //      invariant (mailbox/watch/heap/ready set all empty) -- fired
    //      watches stay registered until the request completes.
    //      A2 (2026-08-28): the completed request's metric anchors are dead
    //      weight by now (every anchored node of the request has fired both
    //      its issue and complete hooks before the decode watch could fire,
    //      and the REQUEST_COMPLETE delta fact arrives at a strictly later
    //      epoch) -- release the routing entries through the same loop. ----
    for (const auto& request_id : commit->completed_requests) {
        driver->watch_registry->remove_watches_for_request(request_id);
        MetricCollector::instance().online_release_request_anchors(
            request_id);
    }

    // ---- 拼 batch 适配(2026-08-22):回收已 fire 的列车哨兵 watch ----
    driver->watch_registry->drain_fired_sentinels();

    // ---- Commit ack: Python's provisional ledger finalizes ----
    driver->bridge->send_commit_ack(batch.batch_id,
                                    batch.source_delivery_sequence, true);

    delete commit;
}

// Step 1-11 (CORE): the explicit next-decision-boundary wakeup event. A
// milestone can form in the deferred drain of a tick whose main queue is
// then empty -- e.g. a real control node (METADATA_NODE, the synchronously
// completing issue path) issued by the commit's own issue pass completes in
// the same tick and fires its watch. Without a wakeup the main loop's
// wait_for_work() would block forever and the milestone would never be
// delivered (仿真加速分析.md §4.3 / 方案 step 1-11). The main loop schedules
// this event at T+1 whenever the mailbox holds work and the queue is empty;
// the event body is a no-op -- the delivery itself happens at the T+1
// tick-end gate, which records the T -> T+1 deferral in
// StateDelta.deferred_from_tick (explicit, never silent).
void ed_delivery_wakeup(void*) {}

// ---------------------------------------------------------------------------
// Tick-end activity gate (step 1-6/1-8)
// ---------------------------------------------------------------------------

// Invoked by EventQueue once per proceed() after the tick's physical events,
// before the same-tick deferred drain. No decision work -> count and return
// (NO Python entry). Decision work -> drain into one StateDelta, deliver it
// through the bridge (blocking round-trip; one in-flight delivery at most),
// and schedule the four-phase commit as a same-tick deferred event.
void ed_driver_tick_end(void* ctx) {
    auto* driver = static_cast<OnlineDriverContext*>(ctx);
    if (!driver->mailbox->has_decision_work()) {
        // Step 1-11: defensive reset -- a pending deferral must never leak
        // into a later, unrelated delivery (there is no code path that
        // clears the mailbox without delivering, so this is unreachable;
        // belt and suspenders).
        driver->deferred_from_tick = 0;
        driver->mailbox->count_tick_end_without_decision();
        return;  // no Python
    }
    // Step 1-11: consume the pending deferral record. The wakeup event fired
    // this tick (main loop scheduled it at T+1 with the mailbox non-empty),
    // so this epoch's events were formed at deferred_from_tick < tick -- the
    // delivery is explicitly recorded as the T->T+1 deferral (the plan's
    // "必须显式记录,不得静默改变顺序"; ordinary epochs carry 0).
    const uint64_t deferred_from = driver->deferred_from_tick;
    driver->deferred_from_tick = 0;
    // Phase-3 sensing (方案 §6.2 操作 1): per-rank injected-unfinished ledger
    // summary at this epoch. Pure query over the committed graphs -- no state
    // is touched and no simulation semantics change, so the phase-2 decision
    // sequence is byte-identical whether sensing is on or off (the summary is
    // query/audit data only; the strategy's red-line inputs never consume it).
    // Only computed when --sensing-enabled turns it on (default off).
    // Phase 6 (方案 §9.1): snapshot_ns timed here -- one computation per
    // decision-work tick-end gate (snapshot_count), ~0 with sensing off.
    std::vector<RankInjectedSummary> injected_unfinished;
    const auto snapshot_t0 = std::chrono::steady_clock::now();
    if (driver->sensing_enabled) {
        injected_unfinished.reserve(driver->graph_sources->size());
        for (int rank = 0;
             rank < static_cast<int>(driver->graph_sources->size()); ++rank) {
            injected_unfinished.push_back(
                (*driver->graph_sources)[rank]->store()
                    .injected_unfinished_summary(rank));
        }
    }
    driver->stats.snapshot_count += 1;
    driver->stats.snapshot_ns +=
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - snapshot_t0)
            .count();
    // Phase 4 (schema v1): drain the per-epoch affected-rank accumulator
    // (union of the fired watches' member ranks, sorted unique) and, only in
    // explicit exact-audit mode, terminal facts into the delivery. Production
    // keeps completed_nodes as [] and retains only O(1) terminal counters.
    std::vector<int> affected_ranks;
    if (driver->affected_ranks_accumulator != nullptr &&
        !driver->affected_ranks_accumulator->empty()) {
        affected_ranks = std::move(*driver->affected_ranks_accumulator);
        driver->affected_ranks_accumulator->clear();
        std::sort(affected_ranks.begin(), affected_ranks.end());
        affected_ranks.erase(std::unique(affected_ranks.begin(),
                                         affected_ranks.end()),
                             affected_ranks.end());
    }
    StateDelta delta = build_state_delta_v1(
        driver->mailbox->drain(),
        driver->event_queue->get_current_time(),
        driver->delivery_seq++, deferred_from, driver->completed_facts.drain(),
        std::move(affected_ranks),
        std::move(injected_unfinished));
    auto* commit = new CommitArg;
    commit->driver = driver;
    for (const auto& ev : delta.events) {
        if (ev.reason == DecisionReason::REQUEST_COMPLETE) {
            commit->completed_requests.push_back(ev.request_id);
        }
    }
    // Bridge round-trip (blocks until the Python scheduler replies), then the
    // commit runs in the same-tick deferred drain (hard rule: from inside the
    // tick-end callback, same-tick events MUST go through
    // schedule_event_deferred). Phase 5: the delta travels with the batch
    // (the committer validates the batch against the delta facts).
    commit->batch = driver->bridge->deliver_and_receive(delta);
    commit->delta = std::move(delta);
    driver->event_queue->schedule_event_deferred(ed_commit_cb, commit);
}

// ---------------------------------------------------------------------------
// Step 1-10: lifecycle transition log + external-producer command FIFO
// ---------------------------------------------------------------------------

const char* service_state_name(const ServiceState state) {
    switch (state) {
        case ServiceState::IDLE:
            return "IDLE";
        case ServiceState::ACTIVE:
            return "ACTIVE";
        case ServiceState::DRAINING:
            return "DRAINING";
        case ServiceState::FINISHED:
            return "FINISHED";
    }
    return "UNKNOWN";
}

// External producer thread for the IDLE fixture (合同② 线程合同: external
// producers only write the thread-safe bounded ingress command queue; the
// decision bridge stays the decision channel, never a request injection
// channel). Reads JSON lines from --command-fifo:
//   {"kind":"Submit", "session_id":..., "turn_index":..., "request_id":...,
//    "prefill_length":..., "decode_length":..., "arrival_world_ns":...,
//    "inter_request_interval_ns":...}
//   {"kind":"CloseInput"}
// The FIFO open blocks until a writer opens the read end (the fixture script
// holds one writer open for the whole injection session). EOF (writer
// closed) exits the thread; the input stays open. The thread is detached:
// it may be parked in the blocking FIFO open/read at run end and dies with
// the process.
void command_fifo_reader(const std::string& path, RequestIngress& ingress,
                         ServiceCoordinator& svc) {
    std::ifstream fifo(path);
    if (!fifo.is_open()) {
        std::cerr << "[online] command-fifo: cannot open " << path << ": "
                  << std::strerror(errno) << std::endl;
        return;
    }
    std::string line;
    while (std::getline(fifo, line)) {
        if (line.empty()) {
            continue;
        }
        nlohmann::json cmd;
        try {
            cmd = nlohmann::json::parse(line);
        } catch (const std::exception&) {
            std::cerr << "[online] command-fifo: skipping malformed line: "
                      << line << std::endl;
            continue;
        }
        const std::string kind = cmd.value("kind", std::string());
        IngressCommand ic;
        if (kind == "CloseInput") {
            // Phase-7 §10.7: explicit close (合同② close 命令). The
            // lifecycle audit records close_source=explicit.
            ic.kind = IngressCommandKind::CloseInput;
        } else if (kind == "EndOfFile") {
            // Phase-7 §10.7: EOF terminal command (合同② EOF 命令) -- the
            // producer declares the input naturally exhausted; distinct
            // from an explicit close in the run-end lifecycle audit.
            ic.kind = IngressCommandKind::EndOfFile;
        } else if (kind == "Error") {
            // Phase-7 §10.7: error terminal command (合同② error 命令) --
            // fail-closed abort (进程退出非 0,不做静默降级); the drain
            // path aborts on it.
            ic.kind = IngressCommandKind::Error;
        } else if (kind == "Submit") {
            RequestEnvelope env;
            env.session_id = cmd.value("session_id", std::string());
            env.turn_index = cmd.value("turn_index", 0);
            env.request_id = cmd.value("request_id", std::string());
            env.prefill_length = cmd.value("prefill_length", uint64_t{0});
            env.decode_length = cmd.value("decode_length", uint64_t{0});
            env.arrival_world_ns = cmd.value("arrival_world_ns", uint64_t{0});
            env.inter_request_interval_ns =
                cmd.value("inter_request_interval_ns", uint64_t{0});
            ic.kind = IngressCommandKind::Submit;
            ic.envelope = std::move(env);
        } else {
            std::cerr << "[online] command-fifo: unknown command kind: "
                      << kind << std::endl;
            continue;
        }
        // Bounded queue with backpressure: retry until accepted or the run
        // has finished (input closed and drained); a command rejected after
        // close is dropped once the run ends.
        while (!ingress.enqueue_command(ic)) {
            if (svc.finished()) {
                std::cerr << "[online] command-fifo: dropping command after "
                             "input closed: "
                          << line << std::endl;
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }
    std::cout << "[online] command-fifo: EOF (writer closed); producer "
                 "thread exits, input stays open"
              << std::endl;
}

// ---------------------------------------------------------------------------
// WP6 NoC link observer wiring (SLO pipeline B2, /tmp/slo_wps/plans/
// CPP_SPEC.md §D).
// ---------------------------------------------------------------------------

/// Mesh-boundary ("edge") directed-link set, computed by replicating
/// MultiDimTopology's deterministic connect order (dims ascending; every
/// connect() appends src->dest then dest->src). A directed link of a
/// Line/Mesh dimension is a boundary link when its hop sits at either end
/// of that dimension (coordinate 0 or dim_size-1). Ring/FullyConnected
/// dimensions have no boundary (their links consume ids but never
/// qualify). This backend has no remote-memory port topology inside the
/// fluid link table (remote memory is a separate API), so the mesh
/// boundary IS the edge set; when a dimension block is unsupported the
/// set is marked not derived and the records fall back to -1 + note.
struct EdgeLinkSet {
    std::set<LinkId> links;
    bool derived = false;
    std::string note;
};

static EdgeLinkSet compute_mesh_edge_links(const NetworkParser& parser) {
    EdgeLinkSet result;
    const auto blocks = parser.get_topologies_per_dim();
    const auto sizes = parser.get_npus_counts_per_dim();
    const int dims = parser.get_dims_count();
    if (static_cast<int>(blocks.size()) != dims ||
        static_cast<int>(sizes.size()) != dims || dims <= 0) {
        result.note = "network parser dimension mismatch; edge set unavailable";
        return result;
    }
    // Row-major strides exactly as MultiDimTopology builds them.
    std::vector<int64_t> stride(dims, 1);
    int64_t npus = 1;
    for (int d = 0; d < dims; ++d) {
        stride[d] = npus;
        npus *= sizes[d];
    }
    const auto address_of = [&](int64_t id) {
        std::vector<int64_t> address(dims, 0);
        for (int d = dims - 1; d >= 0; --d) {
            address[d] = id / stride[d];
            id %= stride[d];
        }
        return address;
    };

    LinkId next_id = 0;
    bool saw_non_mesh_dim = false;
    for (int d = 0; d < dims; ++d) {
        switch (blocks[d]) {
        case TopologyBuildingBlock::Mesh: {
            for (int64_t src = 0; src < npus; ++src) {
                const auto address = address_of(src);
                if (address[d] + 1 >= sizes[d]) {
                    continue;
                }
                const bool boundary =
                    address[d] == 0 || address[d] + 1 == sizes[d] - 1;
                if (boundary) {
                    result.links.insert(next_id);
                    result.links.insert(next_id + 1);
                }
                next_id += 2;
            }
            break;
        }
        case TopologyBuildingBlock::Ring: {
            // Ring: one bidirectional connect per src; no boundary concept.
            // A width==2 Ring degenerates to the mesh fallback
            // (MultiDimTopology::connect_ring_dimension), connecting only
            // the npus/2 address-0 srcs -- npus link ids, not 2*npus.
            saw_non_mesh_dim = true;
            if (sizes[d] == 2) {
                next_id += static_cast<LinkId>(npus);
            } else {
                next_id += 2 * static_cast<LinkId>(npus);
            }
            break;
        }
        case TopologyBuildingBlock::FullyConnected: {
            saw_non_mesh_dim = true;
            for (int64_t src = 0; src < npus; ++src) {
                const auto address = address_of(src);
                for (int64_t further = address[d] + 1; further < sizes[d];
                     ++further) {
                    (void)further;
                    next_id += 2;
                }
            }
            break;
        }
        default: {
            result.note = "unsupported topology block; edge set unavailable";
            return result;
        }
        }
    }
    result.derived = true;
    result.note = saw_non_mesh_dim
        ? "mesh boundary of Line/Mesh dims (Ring/FullyConnected dims have "
          "no boundary); no remote-memory port topology exists in this "
          "backend's fluid link table"
        : "mesh boundary of the Line/Mesh dimensions; no remote-memory "
          "port topology exists in this backend's fluid link table";
    return result;
}

/// Emit the link_bucket / link_total [METRIC] records from the observer's
/// integrated series (summary/full runs only; the caller gates). Closed
/// bucket rows are replayed in bucket/link order from the scheduler's
/// anonymous spool, so only this one aggregate stays resident. The documented
/// first/last-window anchors are emitted without filling empty gaps.
static void emit_link_observer_records(
    const std::shared_ptr<FluidScheduler>& scheduler,
    const EdgeLinkSet& edge_links) {
    const uint64_t bucket_ns = scheduler->link_observer_bucket_ns();
    const uint64_t window_ns = scheduler->link_observer_window_ns();
    const auto& collector = MetricCollector::instance();

    struct BucketAggregate {
        uint64_t total_bytes = 0;
        uint64_t max_bytes = 0;
        LinkId max_link = 0;
        uint64_t edge_max_bytes = 0;
        LinkId edge_max_link = 0;
    };
    struct BucketEmitter {
        const EdgeLinkSet& edge_links;
        const MetricCollector& collector;
        uint64_t bucket_ns;
        BucketAggregate aggregate{};
        uint64_t current_bucket = 0;
        uint64_t last_emitted_bucket = 0;
        bool have_current = false;
        bool emitted_any = false;

        void emit(const uint64_t bucket, const BucketAggregate& values) {
            json record;
            record["schema"] = 1;
            record["type"] = "link_bucket";
            record["source"] = "simulator";
            record["repo_variant"] = collector.metric_repo_variant();
            record["run_id"] = collector.metric_run_id();
            record["bucket_start_ns"] = bucket * bucket_ns;
            record["max_link"] = values.max_link;
            record["max_bytes"] = values.max_bytes;
            record["total_bytes"] = values.total_bytes;
            if (edge_links.derived) {
                record["edge_max_link"] = values.edge_max_link;
                record["edge_max_bytes"] = values.edge_max_bytes;
                record["edge_note"] = edge_links.note;
            } else {
                record["edge_max_link"] = -1;
                record["edge_max_bytes"] = -1;
                record["edge_note"] =
                    "edge link set unavailable: " + edge_links.note;
            }
            record["link_bucket_ns"] = bucket_ns;
            record["provisional"] = collector.slo_sampling_provisional();
            collector.emit_observer_record(record.dump());
            last_emitted_bucket = bucket;
            emitted_any = true;
        }

        void begin(const uint64_t bucket) {
            current_bucket = bucket;
            aggregate = BucketAggregate{};
            have_current = true;
        }

        void flush() {
            emit(current_bucket, aggregate);
            have_current = false;
        }

        bool consume(const uint64_t bucket, const LinkId link_id, const uint64_t bytes) {
            if (bytes == 0) {
                return false;
            }
            if (!have_current) {
                if (bucket != 0) {
                    emit(0, BucketAggregate{});
                }
                begin(bucket);
            } else if (bucket != current_bucket) {
                if (bucket < current_bucket) {
                    return false;
                }
                flush();
                begin(bucket);
            }

            aggregate.total_bytes += bytes;
            // The scheduler's spool order is ascending LinkId. Strict > keeps
            // the legacy lowest-link result when equal byte counts tie.
            if (bytes > aggregate.max_bytes) {
                aggregate.max_bytes = bytes;
                aggregate.max_link = link_id;
            }
            if (edge_links.derived && edge_links.links.count(link_id) > 0 &&
                bytes > aggregate.edge_max_bytes) {
                aggregate.edge_max_bytes = bytes;
                aggregate.edge_max_link = link_id;
            }
            return true;
        }
    };

    BucketEmitter emitter{edge_links, collector, bucket_ns};
    const auto visitor = [](void* const context, const uint64_t bucket,
                            const LinkId link_id, const uint64_t bytes) -> bool {
        return static_cast<BucketEmitter*>(context)->consume(bucket, link_id, bytes);
    };
    scheduler->link_observer_visit_buckets(visitor, &emitter);

    // Coordinator ruling 2026-08-26: emit first and last window buckets even
    // when empty, without filling intervening zero buckets.
    if (emitter.have_current) {
        emitter.flush();
    }
    if (!emitter.emitted_any && window_ns > 0) {
        emitter.emit(0, BucketAggregate{});
    }
    if (bucket_ns > 0 && window_ns > 0) {
        const auto last_bucket = (window_ns - 1) / bucket_ns;
        if (last_bucket > emitter.last_emitted_bucket) {
            emitter.emit(last_bucket, BucketAggregate{});
        }
    }

    const auto& totals = scheduler->link_observer_totals();
    for (size_t link = 0; link < totals.size(); ++link) {
        json record;
        record["schema"] = 1;
        record["type"] = "link_total";
        record["source"] = "simulator";
        record["repo_variant"] = collector.metric_repo_variant();
        record["run_id"] = collector.metric_run_id();
        record["link_id"] = static_cast<LinkId>(link);
        record["total_bytes"] = totals[link].total_bytes;
        record["active_ns"] = totals[link].active_ns;
        record["window_ns"] = window_ns;
        record["link_count"] = totals.size();
        record["link_bucket_ns"] = bucket_ns;
        record["provisional"] = collector.slo_sampling_provisional();
        collector.emit_observer_record(record.dump());
    }
}

}  // namespace

int main(int argc, char* argv[]) {
    // Phase 7 (方案 §10.3): forced-flush governance. Make stdout fully
    // unbuffered so every log line lands in cpp.log immediately -- a crash
    // or kill never loses the buffered tail, and std::cout lines interleave
    // with the [METRIC] lines (which already write via ::write) in true
    // order. Online binary only: the offline static ET binary is untouched.
    ::setvbuf(stdout, nullptr, _IONBF, 0);

    // Total wall-clock timer from main() entry; logged at run end on both
    // exit paths (gate failure and normal return) so every ended run
    // reports how long the simulation took. Logged through the "main"
    // logger: the console sink prints it and the logging-folder file sink
    // persists it to log.log (console-only when --logging-folder=off).
    const auto sim_wall_start = std::chrono::steady_clock::now();
    const auto print_total_wall_time = [&sim_wall_start]() {
        const auto total_ms =
            std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - sim_wall_start)
                .count();
        LoggerFactory::get_logger("main")->info(
            "[online] total simulation wall time: {} ms ({:.3f} s)",
            total_ms, static_cast<double>(total_ms) / 1000.0);
    };

    // Online CLI family contract (step 1-2; hard errors on violation).
    OnlineCliOptions online_cli;
    std::string cli_error;
    if (!parse_online_cli(argc, argv, online_cli, cli_error)) {
        std::cerr << "[Error] (execution_driven/online) " << cli_error
                  << std::endl;
        return EXIT_FAILURE;
    }
    // Step 1-8/1-9 mode gates: --online-mode strategy is the only mode
    // (path-2 removal 2026-08-18 deleted the replay token with the replay
    // route); strategy runs real physics and requires the decision bridge.
    // --request-queue-csv is optional (合同② request-neutral default, step
    // 1-10): when absent the service starts IDLE and reads no pre-loaded
    // queue -- it never falls back to a preset/stub queue (fail-closed).
    if (online_cli.bridge_dir.empty()) {
        std::cerr << "[Error] (execution_driven/online) --bridge-dir is "
                     "required in online mode (the decision bridge)"
                  << std::endl;
        return EXIT_FAILURE;
    }

    // Shared static options (same parser and extraction as the static main).
    auto cmd_line_parser = CmdLineParser(argv[0]);
    cmd_line_parser.parse(argc, argv);

    const auto workload_configuration =
        cmd_line_parser.get<std::string>("workload-configuration");
    const auto comm_group_configuration =
        cmd_line_parser.get<std::string>("comm-group-configuration");
    const auto system_configuration =
        cmd_line_parser.get<std::string>("system-configuration");
    const auto remote_memory_configuration =
        cmd_line_parser.get<std::string>("remote-memory-configuration");
    const auto network_configuration =
        cmd_line_parser.get<std::string>("network-configuration");
    const auto logging_configuration =
        cmd_line_parser.get<std::string>("logging-configuration");
    const auto logging_folder =
        cmd_line_parser.get<std::string>("logging-folder");
    const auto num_queues_per_dim =
        cmd_line_parser.get<int>("num-queues-per-dim");
    const auto comm_scale = cmd_line_parser.get<double>("comm-scale");
    const auto injection_scale = cmd_line_parser.get<double>("injection-scale");
    const auto rendezvous_protocol =
        cmd_line_parser.get<bool>("rendezvous-protocol");
    const auto metrics_configuration =
        cmd_line_parser.get<std::string>("metrics-configuration");
    const auto metrics_detail =
        cmd_line_parser.get<std::string>("metrics-detail");

    AstraSim::LoggerFactory::init(logging_configuration, logging_folder);

    // Side-band metrics collection, same as the static main.
    MetricCollector::instance().initialize(metrics_configuration,
                                           metrics_detail);
    // Phase-7 §10.3: the manifest's static node-event table is keyed by the
    // OFFLINE ET's node-id space, which overlaps the online per-rank id
    // space (online ids 0..7733 vs static ids 1..8000+): static events would
    // fire on unrelated online nodes and corrupt every request state. The
    // online run therefore drops the static event tables here and lets the
    // dynamic anchors (registered per committed batch, before each issue
    // pass) fully own the event tables. No-op when metrics are disabled.
    MetricCollector::instance().clear_static_node_events();

    // Instantiate event queue
    const auto event_queue = std::make_shared<EventQueue>();

    // Generate topology
    const auto network_parser = NetworkParser(network_configuration);
    const auto topology = construct_topology(network_parser);

    // Get topology information
    const auto npus_count = topology->get_npus_count();
    const auto npus_count_per_dim = topology->get_npus_count_per_dim();
    const auto dims_count = topology->get_dims_count();

    // Set up Network API
    CongestionAwareNetworkApi::set_event_queue(event_queue);
    CongestionAwareNetworkApi::set_topology(topology);
    const auto fluid_scheduler = std::make_shared<FluidScheduler>(
        event_queue, topology->get_directed_links(),
        network_parser.get_fluid_max_active_flows(),
        network_parser.get_fluid_max_route_memberships(),
        network_parser.get_progress_report_event_interval());
    CongestionAwareNetworkApi::set_fluid_scheduler(fluid_scheduler);
    // Step 1-8: the online loop commits/emits comm from tick-end and deferred
    // contexts (post-commit issue pass). A start_flow flush scheduled at
    // current_time via schedule_event would insert a current_time EventList
    // into the main queue and trip the strict-increase assert on the next
    // proceed() (EventQueue.cpp:33). Deferred mode routes the flush through
    // schedule_event_deferred (same-tick drain). The static main never
    // enables it -- pre-extension behavior is byte-for-byte preserved.
    fluid_scheduler->set_deferred_flush_mode(true);

    // WP6 NoC link observer gating (CPP_SPEC §D): metrics != off AND env
    // ASTRA_LINK_OBSERVER != 0 (unset == enabled; "0" disables). When the
    // gate is off the scheduler never even checks the observer flag -- zero
    // work, zero records, byte-identical [METRIC] stream modulo nothing.
    // The bucket length comes from the manifest slo_sampling node (the
    // collector falls back to the documented provisional anchor).
    bool fluid_link_observer_on = false;
    EdgeLinkSet edge_links;
    if (MetricCollector::instance().enabled()) {
        const char* link_observer_env = std::getenv("ASTRA_LINK_OBSERVER");
        const bool disabled_by_env =
            link_observer_env != nullptr &&
            std::string(link_observer_env) == "0";
        if (!disabled_by_env) {
            fluid_scheduler->enable_link_observer(
                MetricCollector::instance().slo_link_bucket_ns());
            fluid_link_observer_on = true;
            edge_links = compute_mesh_edge_links(network_parser);
            std::cout << "[online] link observer: enabled (bucket_ns="
                      << MetricCollector::instance().slo_link_bucket_ns()
                      << ", provisional="
                      << (MetricCollector::instance().slo_sampling_provisional()
                              ? "true"
                              : "false")
                      << ", edge_links=" << edge_links.links.size() << ")"
                      << std::endl;
        } else {
            std::cout << "[online] link observer: disabled "
                         "(ASTRA_LINK_OBSERVER=0)" << std::endl;
        }
    }

    // Create ASTRA-sim related resources
    auto network_apis =
        std::vector<std::unique_ptr<CongestionAwareNetworkApi>>();
    const auto memory_api =
        std::make_unique<AnalyticalRemoteMemory>(remote_memory_configuration);
    auto systems = std::vector<Sys*>();

    auto queues_per_dim = std::vector<int>();
    for (auto i = 0; i < dims_count; i++) {
        queues_per_dim.push_back(num_queues_per_dim);
    }

    // Execution-mode factory (step 1-2): online Sys never constructs an
    // ETFeeder and never requires .et files; the dynamic GraphSource is
    // injected at Sys creation. Each rank owns its NodeStore-backed source;
    // the vector below is the commit's access path (kept alive here).
    auto graph_sources = std::vector<std::shared_ptr<NodeStoreGraphSource>>();
    for (int i = 0; i < npus_count; i++) {
        auto graph_source = std::make_shared<NodeStoreGraphSource>();
        graph_sources.push_back(graph_source);
        // create network and system
        auto network_api = std::make_unique<CongestionAwareNetworkApi>(i);
        // Step 1-8 (main ruling 2026-08-15): the replay-clock scope flag
        // (concurrent calibrated COMP + instant comms) was bound to the
        // --online-mode replay token; path-2 removal (2026-08-18) deleted
        // that flag with the replay route -- strategy (the only mode) always
        // runs real physics.
        auto* const system = new Sys(
            i, workload_configuration, comm_group_configuration,
            system_configuration, memory_api.get(), network_api.get(),
            npus_count_per_dim, queues_per_dim, injection_scale, comm_scale,
            rendezvous_protocol, ExecutionMode::Online, graph_source);

        // push back network and system
        network_apis.push_back(std::move(network_api));
        systems.push_back(system);
    }

    // Online service: lifecycle authority + request-neutral ingress. No
    // systems[i]->workload->fire(): in online mode the graph is fed by the
    // dynamic source, not by the ETFeeder.
    ServiceCoordinator svc;
    DecisionMailbox mailbox;
    RequestIngress ingress;
    ingress.bind(event_queue.get(), &mailbox, &svc);
    // Step 1-10: --request-queue-csv is optional (request-neutral default).
    // Absent -> expected_requests = 0 and the service stays IDLE; the CSV
    // file itself, when given, still fails closed if unreadable (OnlineCli
    // already rejects unreadable paths; the reader re-checks for the file
    // disappearing in between). No preset/stub queue is ever created.
    // P0 turn-0 late-discovery fix (2026-08-30): the CSV is read through the
    // index-pass + turn-0-calendar WindowedTraceReader. The first pump
    // streams the whole file once (structure validation fail-closed,
    // provenance sidecar gate, per-row metrics registration and turn>0
    // queue-index pre-registration in row order), stable-sorts the turn-0
    // calendar by (arrival, queue_index) and queues every turn-0 Submit in
    // ARRIVAL order -- the submission order is fully decoupled from the file
    // position (the old row-order window made late-file early arrivals fire
    // after their declared time; 491 clamped Submits on the full TraceLab
    // run). expected_requests is
    // filled from reader.data_rows() once the calendar is fully drained
    // (guaranteed before finished() because pump() runs before every
    // finished() check and completion requires every arrival alarm to have
    // fired).
    uint64_t expected_requests = 0;
    WindowedTraceReader windowed(
        online_cli.request_queue_csv.empty() ? "" :
            online_cli.request_queue_csv,
        ingress, online_cli.request_max_arrival_ns);
    if (online_cli.request_queue_csv.empty()) {
        std::cout << "[online] request queue: none (--request-queue-csv "
                     "absent); request-neutral IDLE start, no preset queue"
                  << std::endl;
    } else {
        windowed.pump();
        std::cout << "[online] request queue: "
                  << online_cli.request_queue_csv
                  << " (calendar reader, max_arrival_ns="
                  << online_cli.request_max_arrival_ns
                  << (online_cli.request_max_arrival_ns == 0 ?
                          " (unbounded default; backport fix "
                          "2026-08-16, sh_2.0测试 §5.1)" :
                          "")
                  << ")" << std::endl;
        // The arrival hook (simulation thread, from arrival_cb) retires the
        // outstanding turn-0 entry: a submitted turn-0 row leaves the
        // outstanding set when its arrival alarm fires.
        ingress.set_arrival_hook(
            [&windowed](const RequestEnvelope& env) {
                windowed.notify_consumed(env.queue_index);
            });
    }

    // Step 1-10: lifecycle transition log (合同② 目标 5: 五态迁移日志与时间).
    // Installed BEFORE --close-input so the close-at-startup transitions
    // (IDLE -> DRAINING -> FINISHED) are logged too. The hook prints every
    // IDLE/ACTIVE/DRAINING/FINISHED transition with elapsed- and wall-clock
    // timestamps; the initial state is printed explicitly (IDLE is not a
    // transition-log entry).
    const auto online_start = std::chrono::steady_clock::now();
    svc.set_transition_hook([&online_start](const ServiceState prev,
                                            const ServiceState next) {
        const auto now = std::chrono::steady_clock::now();
        const auto elapsed_ms =
            std::chrono::duration_cast<std::chrono::milliseconds>(
                now - online_start)
                .count();
        const auto wall_ms =
            std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::system_clock::now().time_since_epoch())
                .count();
        std::cout << "[online] lifecycle: " << service_state_name(prev)
                  << " -> " << service_state_name(next)
                  << " (t=" << elapsed_ms << "ms wall_ms=" << wall_ms << ")"
                  << std::endl;
    });
    const auto wall_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch())
            .count();
    std::cout << "[online] lifecycle: start " << service_state_name(svc.state())
              << " (t=0ms wall_ms=" << wall_ms << ")" << std::endl;

    // Step 1-10: external-producer command FIFO (IDLE fixture injection
    // channel; the fixture script must mkfifo it before the binary starts).
    if (!online_cli.command_fifo.empty()) {
        std::thread(command_fifo_reader, online_cli.command_fifo,
                    std::ref(ingress), std::ref(svc))
            .detach();
    }

    // --close-input closes external production only after every CSV row has
    // entered the bounded reader. Closing at startup made later window pumps
    // enqueue into an already-finished service and defeated ingress close
    // linearization. No-CSV and already-at-EOF inputs still close immediately.
    if (online_cli.close_input &&
        (online_cli.request_queue_csv.empty() || windowed.eof())) {
        ingress.mark_input_closed();
    }

    // Step 1-7/1-8: the decision bridge. ensure_bridge_dir runs BEFORE the
    // Python side starts (FIFO open order rule); the bridge's req_notify
    // write end is held open for the run lifetime, so Python sees a clean
    // EOF at run end and finalizes its ledger. open_notify() right here is
    // step 1-10: the write end pairs with Python's blocking read-end open
    // even when the run delivers nothing (IDLE fixture / zero-request runs),
    // so a zero-delivery run still terminates Python via EOF at exit.
    FileDecisionBridge::ensure_bridge_dir(online_cli.bridge_dir);
    // C1 (2026-08-29): pass the NPU count so parse_graph_batch can enforce
    // the rank domains (node/edge/watch-member/touched-rank ranges) at the
    // single structural parse, before any consumer sees the batch.
    FileDecisionBridge bridge(online_cli.bridge_dir,
                              online_cli.bridge_timeout_ms,
                              static_cast<int>(npus_count));
    // timeout 0 = wait forever (frozen default; --bridge-timeout-ms opt-in).
    bridge.open_notify();

    // Step 1-5: the online CompletionObserver hook. It records completion
    // facts into the WatchRegistry (read-only with respect to the graph) --
    // the ONLY thing the hook does in phase 1 besides the step-1-6 mailbox
    // write (the registry fire notifier is the wiring point). It never calls
    // GraphSource::finish_node: dependency release stays exclusively in
    // Workload::call. The static binary never installs a hook (zero
    // overhead contract).
    WatchRegistry watch_registry;
    // Step 1-8: the hook additionally schedules the per-rank post-commit
    // deferred issue pass (ready-set drain) into the same tick's deferred
    // queue -- the pipeline advances continuously between decisions without
    // Workload auto-advancing (方案 §4 步骤 1-4 操作 4 / 仿真加速分析.md
    // §3.3; the E2E proved commit-only issue stalls: prefill chains only
    // advanced at arrival epochs and the replay desynced).
    std::vector<Workload*> workloads;
    for (auto* system : systems) {
        workloads.push_back(system->workload);
    }
    OnlineCompletionHookContext online_hook_ctx;
    online_hook_ctx.registry = &watch_registry;
    online_hook_ctx.configure_issue_passes(
        event_queue.get(), workloads.size(),
        [](void* opaque, const int rank) {
            auto* const online_workloads =
                static_cast<std::vector<Workload*>*>(opaque);
            (*online_workloads)[rank]->issue_dep_free_nodes();
        },
        &workloads);
    CompletionObserver::instance().set_hook(online_completion_hook,
                                            &online_hook_ctx);

    // Step 1-6/1-8: watch fire -> DecisionMailbox push (the step-1-5 hook
    // contract ②). The stage->reason mapping is the wscllm boundary
    // (步骤 1-5 操作 1): prefill -> PREFILL_DRAIN, decode ->
    // DECODE_COMPLETION + REQUEST_COMPLETE (the decode watch fire is the
    // request-completion point: the last decode node is terminal). Any other
    // stage is a programming error (fail closed). The two decode-stage
    // events share one fire but have distinct dedup identities (reason
    // differs), so both enter the same epoch safely.
    // Phase 4 (schema v1): the fire's member ranks accumulate into the
    // per-epoch affected-rank set (StateDelta.affected_ranks). Watch fires
    // happen on the simulation thread only (Workload::call), so no locking.
    std::vector<int> epoch_affected_ranks;
    watch_registry.set_fire_notifier(
        [&mailbox, &epoch_affected_ranks](const WatchFire& fire) {
            epoch_affected_ranks.insert(epoch_affected_ranks.end(),
                                        fire.member_ranks.begin(),
                                        fire.member_ranks.end());
        if (fire.stage == "prefill") {
            DecisionEvent ev;
            ev.reason = DecisionReason::PREFILL_DRAIN;
            ev.request_id = fire.request_id;
            ev.stage = fire.stage;
            ev.generation = fire.generation;
            ev.payload.watch_member_count = fire.member_count;
            mailbox.push(std::move(ev));
        } else if (fire.stage == "decode") {
            DecisionEvent decode_ev;
            decode_ev.reason = DecisionReason::DECODE_COMPLETION;
            decode_ev.request_id = fire.request_id;
            decode_ev.stage = fire.stage;
            decode_ev.generation = fire.generation;
            decode_ev.payload.watch_member_count = fire.member_count;
            mailbox.push(std::move(decode_ev));
            DecisionEvent complete_ev;
            complete_ev.reason = DecisionReason::REQUEST_COMPLETE;
            complete_ev.request_id = fire.request_id;
            complete_ev.stage.clear();  // request-level, not stage-level
            complete_ev.generation = fire.generation;
            complete_ev.payload.watch_member_count = fire.member_count;
            mailbox.push(std::move(complete_ev));
        } else {
            std::cerr << "[Error] (execution_driven/online) watch fired with "
                         "unmapped stage: "
                      << fire.stage << std::endl;
            std::abort();
        }
    });

    // Step 1-6/1-8: the tick-end activity gate (once per tick, after the
    // tick's physical events, before the same-tick deferred drain); the
    // bridge delivers one epoch and the commit runs as a deferred event.
    OnlineDriverContext driver_ctx;
    driver_ctx.mailbox = &mailbox;
    driver_ctx.event_queue = event_queue.get();
    driver_ctx.bridge = &bridge;
    driver_ctx.svc = &svc;
    driver_ctx.watch_registry = &watch_registry;
    driver_ctx.graph_sources = &graph_sources;
    // Phase-3 perception feature flag (default off until phase 6; the
    // --sensing-enabled token gates the injected-unfinished summary delivery
    // only -- query/audit data, never a strategy decision input).
    driver_ctx.sensing_enabled = online_cli.sensing_enabled;
    // Phase 4 (schema v1): wire the per-epoch accumulators. The hook context
    // was installed BEFORE driver_ctx existed (the hook itself only fires
    // during event processing, i.e. after the main loop starts), so the
    // completed-facts buffer pointer is filled here.
    online_hook_ctx.completed_facts = &driver_ctx.completed_facts;
    driver_ctx.affected_ranks_accumulator = &epoch_affected_ranks;
    // Phase 5 (方案 §8): the atomic GraphBatch committer. Owns the
    // persistent per-rank affine json-id -> store-id translation, the in-flight
    // request
    // tracking and the phase-5 counters; ed_commit_cb runs its
    // validate-then-commit two-phase protocol. The issue pass covers the
    // batch's touched ranks ONLY (ranks without new nodes cannot have new
    // free nodes -- the completion hook's deferred per-rank passes drain
    // every other rank).
    GraphBatchCommitter::Context committer_ctx;
    committer_ctx.num_ranks = static_cast<int>(npus_count);
    committer_ctx.graph_sources = &graph_sources;
    committer_ctx.watch_registry = &watch_registry;
    committer_ctx.ingress = &ingress;
    committer_ctx.issue_rank = [&systems](int rank) {
        systems[rank]->workload->issue_dep_free_nodes();
    };
    // The mandatory GraphBatch liveness preflight cannot infer a collective's
    // true process-group membership from the batch itself: that declaration is
    // owned by each participating Workload.  Resolve it read-only here, across
    // precisely the participating ranks, and fail closed when a pg is missing
    // or those Workloads disagree.  Preserve and compare BOTH the declared
    // rank order and dimension_sizes: rank order selects algorithm positions,
    // while dimension_sizes select topology/phase semantics.  The preflight
    // compares the returned membership set to the batch participants
    // separately.
    committer_ctx.communicator_members_for_pg =
        [&systems](const std::string& pg_name,
                   const std::vector<int>& participant_ranks)
        -> std::optional<std::vector<int>> {
        int pg_id = 0;
        size_t parsed = 0;
        try {
            pg_id = std::stoi(pg_name, &parsed);
        } catch (const std::exception&) {
            return std::nullopt;
        }
        if (parsed != pg_name.size()) {
            return std::nullopt;
        }

        std::optional<std::vector<int>> declared_members;
        std::optional<std::vector<int>> declared_dimension_sizes;
        for (const int rank : participant_ranks) {
            if (rank < 0 || rank >= static_cast<int>(systems.size()) ||
                systems[rank] == nullptr || systems[rank]->workload == nullptr) {
                return std::nullopt;
            }
            const auto group_it =
                systems[rank]->workload->comm_groups.find(pg_id);
            if (group_it == systems[rank]->workload->comm_groups.end() ||
                !group_it->second) {
                return std::nullopt;
            }
            const std::vector<int>& members = group_it->second->involved_NPUs;
            const std::vector<int>& dimension_sizes =
                group_it->second->get_dimension_sizes();
            if (!declared_members.has_value()) {
                declared_members = members;
                declared_dimension_sizes = dimension_sizes;
            } else if (*declared_members != members ||
                       *declared_dimension_sizes != dimension_sizes) {
                return std::nullopt;
            }
        }
        return declared_members;
    };
    // Phase-7 §10.3: metrics anchors register from inside commit() so they
    // bind to the STORE ids (B-1 establishes the affine translation just
    // before the hook runs). Installed ONLY when metrics are enabled:
    // with metrics off the hook stays empty and the commit pays zero extra
    // cost per batch (the enabled() gate is checked once here, not per
    // commit).
    if (MetricCollector::instance().enabled()) {
        // R2 (2026-08-29): the hook gains the per-rank sources so anchor
        // registration can set the OnlineNode fast-path flags in the same
        // Phase B-1.5 step (add_node has already stored the nodes; the
        // issue pass runs later in the same commit). graph_sources is this
        // scope's vector and outlives the committer (same function), so the
        // reference capture is safe.
        committer_ctx.metrics_anchor_hook =
            [&graph_sources](const GraphBatch& batch,
                             const GraphBatchCommitter& committer) {
                register_online_metrics_anchors(batch, committer,
                                                graph_sources);
            };
    }
    // M2 node GC (2026-08-23; A1 amortization 2026-08-28). B.3 cleanup
    // (2026-09-05): the --online-node-gc CLI arm was removed -- production
    // always collects (amortized): the constructor propagates the switch to
    // every per-rank store and the commit tail drains once >=
    // kGcAmortizeThreshold candidates accumulated; the run end forces one
    // final drain. Fixtures keep the internal Context switch off.
    committer_ctx.node_gc = true;
    GraphBatchCommitter committer(committer_ctx);
    driver_ctx.committer = &committer;
    driver_ctx.online_validate = online_cli.online_validate;
    if (online_cli.sensing_enabled) {
        std::cout << "[online] sensing: enabled (--sensing-enabled; "
                     "injected-unfinished ledger summary delivered per "
                     "epoch)" << std::endl;
    }
    // M2 node GC (2026-08-23; A1 2026-08-28): evidence line (cpp.log is an
    // allowed-diff log). Collection is always on since the B.3 cleanup
    // (2026-09-05) removed the --online-node-gc CLI arm.
    std::cout << "[online] node gc: "
              << "enabled (amortized: commit tails drain once >= "
              << GraphBatchCommitter::kGcAmortizeThreshold
              << " finished nodes accumulate, run end forces a final drain)"
              << std::endl;
    // C1 (2026-08-28): evidence line for the validate mode (cpp.log is an
    // allowed-diff log; the counters themselves are whitelisted diffs).
    std::cout << "[online] graph validate: "
              << (online_cli.online_validate == 0 ? "off" : "full")
              << " (--online-validate " << online_cli.online_validate
              << "; production runners default 0, smoke/fixture/verify runs "
                 "pass 1)"
              << std::endl;
    event_queue->set_tick_end_callback(ed_driver_tick_end, &driver_ctx);

    fluid_scheduler->flush_pending_starts();
    fluid_scheduler->mark_event_loop_started();

    // Online main loop: empty queue => block for work (no busy wait, no
    // exit); svc.finished() remains the logical request-lifecycle authority.
    // After that logical end, however, already-issued graph tail events must
    // still reach physical quiescence before process exit: a decode watch can
    // complete a request before its train's end-barrier collective terminals.
    // wait_for_work returns only when a command was enqueued or the input
    // closed; the loop then drains (alarms land in the queue) or re-checks
    // finished(), so it never spins forever.
    // Drain-first (主控裁决 2026-08-15, steps-1-6 偏差①): finished() is
    // checked AFTER drain_commands(), so commands submitted before a Close
    // (e.g. --close-input at startup with a preloaded CSV) are never
    // silently dropped -- close + empty queue exits immediately, close +
    // queued commands run them to completion (DRAINING semantics per
    // 合同②/总体方案 §5.5), and the negative CSV test proves the queue is
    // drained even when finished() was already true at loop entry.
    while (true) {
        ingress.drain_commands();
        // Phase 7 §10.4: top the window up right after the drain. pump()
        // only reads disk and queues Submit commands (processed on the next
        // drain); it never schedules or decides anything. The window can
        // only advance when rows were consumed (arrival alarms fired), so
        // every turn>0 row is registered before its future alarm can fire.
        windowed.pump();
        if (online_cli.close_input && !svc.input_closed() && windowed.eof()) {
            // Every turn-0 Submit is now either already drained or queued, and
            // every turn>0 row has registered its future-arrival identity.
            // Linearize the producer close only at this exact finite-input
            // boundary so no later CSV Submit is rejected.
            ingress.mark_input_closed();
        }
        // Defect fix A (2026-08-16, face主动测试错误分析.md 缺陷 A): pump()
        // QUEUES Submit commands into the bounded ingress queue; they only
        // register as pending alarms (on_alarm_scheduled) on the NEXT
        // drain_commands(). Between this pump() and that drain the service
        // counters pass through a vacuum -- active==0 && pending_alarm==0
        // while queued-but-undrained commands exist -- and the finished()
        // check below could end the run with thousands of CSV rows still
        // undelivered (F2/F3/F4: ack_count != delivery_count, 恒差 1; the
        // window had topped up un-consumed rows right as the last pending
        // alarm of the previous batch fired). Drain whatever pump() just
        // queued BEFORE consulting finished(), so the check can never
        // observe the vacuum. The extra drain is a no-op when pump()
        // queued nothing (the common case).
        if (ingress.pending_command_count() > 0) {
            ingress.drain_commands();
        }
        if (expected_requests == 0 &&
            !online_cli.request_queue_csv.empty() && windowed.eof()) {
            // EOF reached: the window read the whole file; data_rows() is
            // now the run-end completion assertion target (1177 for the
            // 20.csv first-30-seconds input).
            expected_requests = windowed.data_rows();
        }
        const bool service_finished = svc.finished();
        if (service_finished) {
            // Defect fix A (belt-and-braces audit): with the drain above,
            // finished() while the window has NOT reached EOF is impossible
            // (unread rows => queued Submits => drained => pending_alarm>0
            // until every alarm fired; a row leaves the window only when
            // its alarm fired). If it ever happens anyway, fail closed with
            // the full counters instead of silently dropping the file tail.
            if (!online_cli.request_queue_csv.empty() && !windowed.eof()) {
                online_fatal(
                    "service finished before the request CSV reached EOF "
                    "(undelivered tail: active=" +
                    std::to_string(svc.active_request_count()) +
                    " pending_alarm=" +
                    std::to_string(svc.pending_alarm_count()) +
                    " queued_commands=" +
                    std::to_string(ingress.pending_command_count()) + ")");
            }
        }
        if (event_queue->finished()) {
            // P0-2 (2026-08-31, 总文档 §4 P0-2.2/§4 P0-2.3): shared
            // parking-point diagnostics for the fail-loud guard below
            // (wall-clock idle watchdog). Every counter the 形态判据
            // reasons over is printed: tick, service counters
            // (active/pending alarm/fence), reader window state
            // (occupancy/rows_read/data_rows/EOF), the input-close knob,
            // and the four emptiness witnesses (mailbox/deferred/commands
            // + the event queue itself, which finished() already proved).
            // wscllm (sync-A16 批次4, 2026-09-01, 合同 §2.1/P1): the
            // input-open dead-end BRANCH is intentionally NOT ported -- the
            // calendar reader's machine states make that shape unreachable
            // (cursor 停驻形态/泵送-二次 drain 次序/Error 终止与 calendar
            // 不变量互相闭合; 重构打破不变量必须重做可达性分析). The
            // 11-field report itself is kept for the A3 watchdog so any
            // silent-stall family stays attributable. window_occupancy in
            // the calendar reader counts committed-but-untriggered turn-0
            // entries.
            const auto parking_diagnostics = [&]() {
                return "tick=" +
                       std::to_string(event_queue->get_current_time()) +
                       " active=" +
                       std::to_string(svc.active_request_count()) +
                       " pending_alarm=" +
                       std::to_string(svc.pending_alarm_count()) +
                       " window_occupancy=" +
                       std::to_string(
                           windowed.current_window_occupancy()) +
                       " rows_read=" +
                       std::to_string(windowed.rows_read()) +
                       " data_rows=" +
                       std::to_string(windowed.data_rows()) +
                       " reader_eof=" + (windowed.eof() ? "yes" : "no") +
                       " close_input=" +
                       (online_cli.close_input ? "1" : "0") +
                       " mailbox_work=" +
                       (mailbox.has_decision_work() ? "yes" : "no") +
                       " deferred_work=" +
                       (event_queue->has_deferred_work() ? "yes" : "no") +
                       " pending_commands=" +
                       std::to_string(ingress.pending_command_count());
            };
            if (ingress.pending_command_count() > 0) {
                // A producer command may have linearized just before a
                // concurrent close but after the top-of-loop drain. Go around
                // once more; closed_ now makes this count stable at zero before
                // the service_finished exit can be taken.
                continue;
            }
            if (mailbox.has_decision_work()) {
                // Step 1-11 (CORE): post-commit same-tick milestone. The
                // tick that just ended committed a graph whose dep-free
                // nodes completed synchronously inside its deferred drain
                // (real control nodes / the normal issue path), so the
                // mailbox holds a decision epoch formed at tick T while the
                // main queue is empty. wait_for_work() would block forever
                // and the milestone would never be delivered -- schedule the
                // explicit next decision boundary at T+1 (方案 step 1-11:
                // "schedule_event(T+1, delivery_cb) 或等价机制") and record
                // the T -> T+1 deferral; the T+1 tick-end gate delivers the
                // epoch with tick=T+1, deferred_from_tick=T. Each wakeup
                // strictly advances the clock and each commit strictly
                // advances the graphs, so this cannot loop forever.
                driver_ctx.deferred_from_tick =
                    event_queue->get_current_time();
                // Phase 6 (方案 §9.1): global_wakeup_count -- the explicit
                // T->T+1 wakeups only; reported separately, never merged into
                // the callback total (仿真加速分析.md §9.4/§15).
                driver_ctx.stats.global_wakeup_count += 1;
                event_queue->schedule_event(
                    event_queue->get_current_time() + 1,
                    ed_delivery_wakeup, &driver_ctx);
            } else if (event_queue->has_deferred_work()) {
                // Defect fix C (2026-08-16, face主动测试错误分析.md 缺陷 C):
                // the main list is empty but the same-tick deferred queue
                // still holds unexecuted events (scheduled outside a proceed
                // context -- e.g. a deferred issue pass appended after the
                // drain pass ended). finished() is blind to them and no
                // proceed() would ever run to drain them: force the explicit
                // next decision boundary (same mechanism as the milestone
                // branch above) so a proceed() executes and drains the
                // deferred queue. NOT a delivery deferral (no mailbox work):
                // deferred_from_tick stays 0.
                driver_ctx.stats.global_wakeup_count += 1;
                event_queue->schedule_event(
                    event_queue->get_current_time() + 1,
                    ed_delivery_wakeup, &driver_ctx);
            } else if (service_finished) {
                // The service has no more logical requests, and all physical
                // events, deferred issue passes, and scheduler-visible work
                // are now drained.  Do not move this branch above the queue
                // checks: the final decode watch fires before the last shared
                // end-barrier collective, whose terminal callbacks are part
                // of the committed-node audit.
                break;
            } else if (svc.input_closed()) {
                // Unified dead-end guard (收敛 2026-08-16): sh_2.0 的
                // livelock guard (实录 bug#11) 与 face 缺陷 C 修复的
                // lost-wakeup dead end fail-closed (face 0049ef5 /
                // face-defectfix2-done) 属同族机制层缺陷——input 已关 +
                // 主队列空 + mailbox 空 + deferred 空 而 svc.finished()
                // 为假,是协议死端(无事件可再触发、官方路径无人
                // signal_work;wait_for_work 在 !input_open_ 谓词下立即
                // 返回,旧代码 busy-spin/静默挂死)。收敛为单一 guard:
                // 保留 sh_2.0 的 per-rank waits-for 证据转储(strategy
                // 死锁诊断价值),终判消息采用 face 的可归因
                // "lost-wakeup dead end" 语义并携带全计数诊断。IDLE
                // fixture 合同不变: input 仍开时走下方分支阻塞等待外部
                // 注入。
                std::cerr << "[Error] (execution_driven/online) run-end "
                             "livelock: queue empty, mailbox empty, input "
                             "closed, but svc not finished: accepted="
                          << svc.accepted_request_count() << " completed="
                          << svc.completed_request_count() << " active="
                          << svc.active_request_count() << " pending_alarm="
                          << svc.pending_alarm_count() << " tick="
                          << event_queue->get_current_time() << std::endl;
                // Waits-for evidence dump (strategy deadlock diagnosis):
                // per-rank unfinished/free node heads + resource-slot
                // holders (comm/comp/hbm). Free-but-unissued interaction
                // nodes + occupied peer slots identify the circular wait.
                for (size_t r = 0; r < graph_sources.size(); ++r) {
                    const auto& store = graph_sources[r]->store();
                    const auto free = store.resolve_free_nodes();
                    const auto* slots = systems[r]->workload->hw_resource;
                    if (store.pending_count() == 0 && free.empty()
                        && slots->gpu_comms_node.empty()
                        && slots->gpu_ops_node.empty()
                        && slots->hbm_dma_ops_node.empty()) {
                        continue;
                    }
                    std::cerr << "[livelock] rank=" << r
                              << " pending=" << store.pending_count()
                              << " free=" << free.size() << " slot_comm=[";
                    for (const auto id : slots->gpu_comms_node) {
                        std::cerr << id << " ";
                    }
                    std::cerr << "] slot_comp=[";
                    for (const auto id : slots->gpu_ops_node) {
                        std::cerr << id << " ";
                    }
                    std::cerr << "] free_head=[";
                    size_t shown = 0;
                    for (const auto id : free) {
                        const auto nv = graph_sources[r]->lookup(id);
                        if (nv.has_value()) {
                            std::cerr << id << ":" << nv->name << " ";
                        }
                        if (++shown >= 3) { break; }
                    }
                    std::cerr << "]" << std::endl;
                }
                // Defect fix C (face parity): the attributable dead-end
                // verdict -- same message family as face's fix, full
                // counters included (deferred_work is necessarily "no"
                // here: the has_deferred_work branch above precedes).
                online_fatal(
                    "lost-wakeup dead end: queue+mailbox empty but service "
                    "not finished (active=" +
                    std::to_string(svc.active_request_count()) +
                    " pending_alarm=" +
                    std::to_string(svc.pending_alarm_count()) +
                    " deferred_work=" +
                    (event_queue->has_deferred_work() ? "yes" : "no") +
                    "; a strategy-deferred request or an alarm-less pending "
                    "count has no delivery path -- this is a protocol dead "
                    "end, not a wait state)");
            } else {
                // IDLE fixture contract (step 1-10): input still open --
                // block for the external producer's signal_work().
                // P0-2 (2026-08-31, 总文档 §4 P0-2.3): --idle-watchdog-s > 0
                // arms a WALL-CLOCK deadline (steady_clock, never the
                // simulation clock -- real-time traces space turn arrivals
                // hours apart) on the parking point; a timeout is a
                // fail-closed abort with the parking diagnostics above.
                // Default 1.0 = armed backstop (2026-09-05; healthy
                // official runs pre-schedule every calendar arrival as a
                // queue event, so they never park). The explicit
                // --idle-watchdog-s 0 escape restores the original
                // unbounded wait_for_work() contract (fixtures, IDLE
                // runs).
                // wscllm (sync-A16 批次4, 2026-09-01, 合同 §2.1/P1): with
                // no input-open dead-end branch (that shape is unreachable
                // under the calendar invariants), this watchdog is the
                // sole backstop for every silent-stall family -- including
                // any future refactor that breaks the pump/drain order the
                // unreachability proof leans on.
                // FP1 (2026-09-01, sync-A16 batch P; contract §2.4/E9):
                // exactly one now() and one deadline computation per
                // arming -- checked_wait_deadline does the tick-domain and
                // pre-addition bound checks, then wait_for_work_until
                // waits on the absolute deadline and does no arithmetic of
                // its own. E26: with the watchdog OFF (0) the loop MUST
                // stay on the original blocking wait_for_work() -- "off"
                // closes the watchdog only, never the wait itself.
                if (online_cli.idle_watchdog_s <= 0.0) {
                    svc.wait_for_work();
                } else {
                    const auto now = std::chrono::steady_clock::now();
                    std::chrono::steady_clock::time_point deadline;
                    if (!ServiceCoordinator::checked_wait_deadline(
                            online_cli.idle_watchdog_s, now, deadline) ||
                        !svc.wait_for_work_until(deadline)) {
                        online_fatal(
                            "idle watchdog: no work within " +
                            std::to_string(online_cli.idle_watchdog_s) +
                            "s of wall clock at the event-loop parking "
                            "point (" +
                            parking_diagnostics() +
                            "); every producer channel is silent -- "
                            "fail-closed abort instead of an unbounded "
                            "wait (raise --idle-watchdog-s above the "
                            "largest legal idle gap if this run is "
                            "legitimately long-idle)");
                    }
                }
            }
        } else {
            event_queue->proceed();
        }
    }

    // Emit all metric records after the event loop, while Statistics are
    // still alive; no-op when metrics are disabled (doc sec.5.6).
    MetricCollector::instance().finalize(systems, Sys::boostedTick());

    // WP6 link observer records (CPP_SPEC §D): only reachable when metrics
    // are on and the observer was enabled. Emitted through the collector's
    // single-write channel; the integration arrays are freed right after
    // (RSS discipline).
    if (fluid_link_observer_on) {
        emit_link_observer_records(fluid_scheduler, edge_links);
        fluid_scheduler->link_observer_release();
        // C5 (2026-08-28): the observer records went through the buffered
        // [METRIC] channel -- drain it so the block is complete before the
        // run-end log lines below.
        MetricCollector::instance().flush_emit_buffer();
    }

    // Step-1-6/1-8 gate counters and run-end assertions. Phase-1 acceptance:
    // completed_request_count == CSV data rows (1177 for the 20.csv
    // first-30-seconds input). tick_end_without_decision_count is a normal
    // allowed-nonzero counter, reported separately.
    std::cout << "[online] gate counters: event_count=" << mailbox.event_count()
              << " delivery_count=" << mailbox.delivery_count()
              << " coalescing_ratio=" << mailbox.coalescing_ratio()
              << " tick_end_without_decision_count="
              << mailbox.tick_end_without_decision_count() << std::endl;
    std::cout << "[online] service counters: accepted="
              << svc.accepted_request_count()
              << " completed=" << svc.completed_request_count()
              << " active=" << svc.active_request_count()
              << " pending_alarm=" << svc.pending_alarm_count() << std::endl;

    // Phase 7 §10.7: lifecycle full-semantics audit (合同② 三态区分:
    // EOF vs 显式 close vs 异常退出 + 迟到/拒绝/溢出审计). The close
    // source is audited explicitly (explicit close command / CLI / EOF
    // terminal command / error); the CSV window EOF is reported separately
    // because it is NOT a close (the input stays open for external
    // injection -- the IDLE fixture contract). The ingress overflow audit
    // counts bounded-queue rejections (0 on every official run: high_water
    // 128 << capacity 4096; a growing producer would surface here).
    const char* close_source = "not_closed";
    if (svc.input_closed()) {
        switch (svc.input_close_reason()) {
            case InputCloseReason::ExplicitClose:
                close_source = "explicit";
                break;
            case InputCloseReason::EndOfFile:
                close_source = "eof";
                break;
            case InputCloseReason::Error:
                close_source = "error";
                break;
        }
    }
    std::cout << "[online] lifecycle audit: close_source=" << close_source
              << " csv_eof="
              << (windowed.eof() ? "true" : "false")
              << " ingress_overflow=" << ingress.overflow_count()
              << " ingress_peak_commands=" << ingress.peak_command_occupancy()
              << std::endl;

    // Phase 4 (schema v1): run-end end-audit -- the watch registry must be
    // EMPTY (every watch of every completed request was removed at its
    // REQUEST_COMPLETE commit) and stale_count() must be 0 (no registered
    // watch ever left unfired). The per-epoch affected-rank accumulator must
    // be drained (non-empty = a watch fired after the final delivery, i.e.
    // mailbox work without a delivery -- impossible under the main loop's
    // wakeup rule). Final end-barrier control nodes can complete after the
    // last delivery (phase-3 R5), so terminal counters are run-lifetime and
    // are audited below against every committed node.
    const auto& terminal_counts = driver_ctx.completed_facts.counters();
    std::cout << "[online] phase-4 end audit: watch_registry_size="
              << watch_registry.size()
              << " watch_stale=" << watch_registry.stale_count()
              << " completed_facts_residual="
              << driver_ctx.completed_facts.size()
              << " affected_ranks_residual=" << epoch_affected_ranks.size()
              << " issue_pass_pending="
              << online_hook_ctx.pending_issue_pass_count()
              << " terminal_total=" << terminal_counts.total
              << " terminal_success=" << terminal_counts.success
              << " terminal_skipped=" << terminal_counts.skipped
              << " terminal_other=" << terminal_counts.other
              << std::endl;

    // Phase 5 (方案 §8.3): the atomic-commit counters. Every delivery epoch
    // produced exactly one GraphBatch (graph_batch_count == delivery_count,
    // zero-node accounting epochs included) and NO batch ever went through a
    // single-node bridge (single_node_bridge_count == 0 -- the per-node
    // commit path is gone; the counter exists as a gate on the official
    // path). avg_nodes_per_batch is reported for the phase-5 acceptance
    // record.
    std::cout << "[online] phase-5 commit counters: "
              << committer.counters_report() << std::endl;

    // M2 node GC (2026-08-23; A1 2026-08-28): run-end evidence (measurement
    // only; cpp.log is an allowed-diff log). erased = nodes collected over
    // the run; retained = records still in the stores at run end (the
    // in-flight window). A1: the forced final drain right above leaves only
    // genuinely unfinished / pinned nodes behind (the pre-A1 tail -- nodes
    // finishing after the last commit -- is collected too).
    committer.finalize_node_garbage();
    {
        uint64_t node_gc_erased = 0;
        uint64_t node_gc_retained = 0;
        for (const auto& source : graph_sources) {
            node_gc_erased += source->store().gc_erased_count();
            node_gc_retained += source->store().retained_count();
        }
        std::cout << "[online] node gc: erased=" << node_gc_erased
                  << " retained=" << node_gc_retained << std::endl;
    }

    // Phase 6 (方案 §9.1): per-run mechanism counters (measurement only; the
    // decision gate is evaluated from the collected benchmark matrix, not
    // from hard C++ assertions -- see 方案 §9.3).
    std::cout << "[online] phase-6 stats counters: "
              << driver_ctx.stats.report() << " " << bridge.stats_report()
              << std::endl;

    // Phase 7 §10.4 / P0 fix (2026-08-30): calendar-reader report (rows /
    // sessions / calendar stats / io time / throughput), the late-arrival
    // clamp counters SPLIT by producer path (static-CSV Submit / external
    // stream / future-alarm rounding / t=0 boundary), the out-of-range
    // rejection counter (frozen rule: turn-0 arrival beyond
    // --request-max-arrival-ns is rejected; 0 on the allowed 20.csv input),
    // and the peak RSS (getrusage ru_maxrss, KiB).
    windowed.report(std::cout);
    std::cout << "[online] ingress late arrivals: total="
              << ingress.late_arrival_count()
              << " late_static_submit="
              << ingress.late_static_submit_count()
              << " late_external_stream="
              << ingress.late_external_stream_count()
              << " late_future_alarm_rounding="
              << ingress.late_future_alarm_rounding_count()
              << " t0_boundary_clamp=" << ingress.t0_boundary_clamp_count()
              << std::endl;

    // P0 fix: per-turn-0 arrival audit + the formal static-arrival gate
    // (finite static CSV runs only; external stream and future alarms are
    // never gated). Gate: late_static_submit == 0 and every turn-0
    // ingress_delay == 0 except the t=0 boundary rows (declared 0,
    // discovered 0, effective 1 -- the EventQueue strict-future rule's
    // inherent boundary; REPORT design ruling). Metric origins stay
    // DECLARED arrivals -- the gate exists so a reader defect can never be
    // masked by rebasing metrics onto effective arrivals.
    const auto arrival_summary = windowed.audit_static_arrivals();
    std::cout << "[online] arrival audit: turn0_submitted="
              << arrival_summary.turn0_submitted
              << " late_static_submit="
              << arrival_summary.late_static_submit
              << " late_external_stream="
              << arrival_summary.late_external_stream
              << " late_future_alarm_rounding="
              << arrival_summary.late_future_alarm_rounding
              << " t0_boundary_clamp=" << arrival_summary.t0_boundary_clamp
              << " ingress_delay_ns p50=" << arrival_summary.delay_p50_ns
              << " p99=" << arrival_summary.delay_p99_ns
              << " max=" << arrival_summary.delay_max_ns
              << " nonzero_delay_rows="
              << arrival_summary.nonzero_delay_rows
              << " gate=" << (arrival_summary.gate_ok ?
                                  "ok" : "FAIL")
              << std::endl;

    struct rusage rusage_usage {};
    if (getrusage(RUSAGE_SELF, &rusage_usage) == 0) {
        std::cout << "[online] peak rss kib: " << rusage_usage.ru_maxrss
                  << std::endl;
    }

    bool gate_ok = true;
    // P0 turn-0 late-discovery fix (2026-08-30): the FORMAL static-arrival
    // gate, evaluated with the completion-audit family. Scoped exactly like
    // the completed==expected audit below: finite static CSV runs only
    // (expected_requests > 0); external stream (--command-fifo) and
    // future-alarm clamps are never gated. late_static_submit must be 0 and
    // every turn-0 ingress_delay must be 0 except the t=0 boundary rows
    // (see the arrival audit line above for the numbers).
    if (expected_requests > 0 && !arrival_summary.gate_ok) {
        std::cerr << "[Error] (execution_driven/online) static-arrival "
                     "gate FAIL: "
                  << arrival_summary.gate_why
                  << " (finite static CSV input must submit every turn-0 "
                     "before its declared arrival; see the arrival audit "
                     "above)"
                  << std::endl;
        gate_ok = false;
    }
    // Step 1-10: the completed==expected audit only applies to CSV-driven
    // runs (the CSV is the source of truth). Injection-only runs
    // (--command-fifo, no --request-queue-csv) complete requests that the
    // CSV never saw -- the IDLE fixture's scenario 2 -- so expected_requests
    // == 0 skips the audit; the fixture script asserts the counts instead.
    // Backport fix (2026-08-16, sh_2.0测试 §5.1; unified 2026-08-20 中-3):
    // the audit is the fail-closed audit_completion() -- denominator = the
    // CSV's TOTAL data rows (a rejected row is consumed at reject time, so
    // the window always flows to EOF and total_data_rows() at run end IS
    // the whole-file count), any explicit-window drop is itself a failure
    // with its count, and the accepted+dropped accounting must balance.
    // The old completed==rows-the-window-read form silently PASSED
    // dropping runs.
    if (expected_requests > 0) {
        const CompletionAuditCounts audit_counts{
            windowed.total_data_rows(),
            windowed.turn0_data_rows(),
            svc.accepted_request_count(),
            svc.completed_request_count(),
            windowed.rejected_out_of_range()};
        const CompletionAuditVerdict audit_verdict =
            audit_completion(audit_counts);
        if (audit_verdict != CompletionAuditVerdict::Ok) {
            std::cerr << "[Error] (execution_driven/online) completion "
                         "audit FAIL ("
                      << completion_audit_why(audit_verdict)
                      << "): total_rows=" << audit_counts.total_rows
                      << " turn0_rows=" << audit_counts.turn0_rows
                      << " accepted=" << audit_counts.accepted
                      << " dropped_out_of_range=" << audit_counts.dropped
                      << " completed=" << audit_counts.completed << std::endl;
            gate_ok = false;
        }
    }
    if (mailbox.delivery_count() == 0 && expected_requests > 0) {
        std::cerr << "[Error] (execution_driven/online) delivery_count == 0: "
                     "the decision bridge never served an epoch"
                  << std::endl;
        gate_ok = false;
    }
    if (!watch_registry.empty() || watch_registry.stale_count() != 0 ||
        !epoch_affected_ranks.empty() ||
        online_hook_ctx.pending_issue_pass_count() != 0) {
        std::cerr << "[Error] (execution_driven/online) phase-4 end audit: "
                     "watch registry not empty (size="
                  << watch_registry.size() << " stale="
                  << watch_registry.stale_count() << ") or affected-ranks "
                     "accumulator not drained (residual="
                  << epoch_affected_ranks.size()
                  << ") or deferred issue passes remain (pending="
                  << online_hook_ctx.pending_issue_pass_count() << ")"
                  << std::endl;
        gate_ok = false;
    }
    if (terminal_counts.total != committer.counters().total_nodes ||
        terminal_counts.other != 0) {
        std::cerr << "[Error] (execution_driven/online) terminal audit: "
                  << "total=" << terminal_counts.total
                  << " committed_nodes=" << committer.counters().total_nodes
                  << " success=" << terminal_counts.success
                  << " skipped=" << terminal_counts.skipped
                  << " other=" << terminal_counts.other
                  << std::endl;
        gate_ok = false;
    }
    // Phase 5 (方案 §8.3) gates: every delivery produced exactly one
    // committed batch (all runs), and NO single-node batch ever bridged on
    // the OFFICIAL path (CSV-driven runs, expected_requests > 0). The
    // single_node_bridge_count == 0 assertion is scoped to the official path
    // exactly like the completed-requests audit above: injection-only
    // fixture runs (--command-fifo, no --request-queue-csv -- the step-1-11
    // same-tick milestone fixture's batches ARE single-node by construction,
    // that is the fixture's purpose) legitimately count > 0 and assert their
    // own semantics in the fixture script.
    if (expected_requests > 0 &&
        committer.counters().single_node_bridge_count != 0) {
        std::cerr << "[Error] (execution_driven/online) "
                     "single_node_bridge_count="
                  << committer.counters().single_node_bridge_count
                  << " != 0 (the per-node commit bridge must never be used)"
                  << std::endl;
        gate_ok = false;
    }
    if (mailbox.delivery_count() != committer.counters().graph_batch_count) {
        std::cerr << "[Error] (execution_driven/online) graph_batch_count="
                  << committer.counters().graph_batch_count
                  << " != delivery_count=" << mailbox.delivery_count()
                  << " (every delivery must commit exactly one GraphBatch)"
                  << std::endl;
        gate_ok = false;
    }
    if (!gate_ok) {
        print_total_wall_time();
        // The "main" logger is async: shutdown drains its queue so the
        // wall-time line lands before this failure exit terminates the
        // process (the normal path relies on the shutdown below).
        AstraSim::LoggerFactory::shutdown();
        return EXIT_FAILURE;
    }

    for (auto it : systems) {
        delete it;
    }
    systems.clear();

    print_total_wall_time();

    // terminate simulation (the bridge destructor then closes req_notify ->
    // Python sees EOF and finalizes; the run script waits for Python).
    AstraSim::LoggerFactory::shutdown();
    return 0;
}
