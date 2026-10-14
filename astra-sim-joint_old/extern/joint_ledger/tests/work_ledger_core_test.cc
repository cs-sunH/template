// work_ledger_core_test.cc -- Python-parity tests for the native WorkLedger
// application core.  Cases are ported from sh_test_mesh/workload/
// llama2_7b_inference/joint/test_work_ledger.py onto the coordinated
// protocol-v3 transaction surface (begin_protocol_v3_transaction /
// tx_stage_event / tx_finish / tx_commit); the legacy apply_payload decoder
// and parse_work_ledger_state full-snapshot loader are Python-only surfaces
// and are not exercised here.
#include <cstdio>
#include <limits>
#include <optional>
#include <string>
#include <vector>

#include "joint_ledger/LedgerTypes.hh"
#include "joint_ledger/WorkLedgerCore.hh"

using jointledger::CommittedWork;
using jointledger::Json;
using jointledger::LedgerError;
using jointledger::NodeEvent;
using Transaction = jointledger::WorkLedgerCore::Transaction;
using jointledger::WorkAmount;
using jointledger::WorkLedgerCore;
using jointledger::WorkState;
using jointledger::canonical_work_id;
using jointledger::leg_work_id;
using jointledger::work_state_name;

static int g_checks = 0;
static int g_failures = 0;

#define CHECK(cond)                                                     \
    do {                                                                \
        ++g_checks;                                                     \
        if (!(cond)) {                                                  \
            ++g_failures;                                               \
            std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
        }                                                               \
    } while (0)

#define CHECK_THROW(expr)                                               \
    do {                                                                \
        ++g_checks;                                                     \
        bool threw_ = false;                                            \
        try {                                                           \
            (void)(expr);                                               \
        } catch (const LedgerError&) {                                  \
            threw_ = true;                                              \
        } catch (...) {                                                 \
        }                                                               \
        if (!threw_) {                                                  \
            ++g_failures;                                               \
            std::printf("FAIL %s:%d: expected LedgerError: %s\n",       \
                        __FILE__, __LINE__, #expr);                     \
        }                                                               \
    } while (0)

#define CHECK_NO_THROW(expr)                                            \
    do {                                                                \
        ++g_checks;                                                     \
        try {                                                           \
            (void)(expr);                                               \
        } catch (...) {                                                 \
            ++g_failures;                                               \
            std::printf("FAIL %s:%d: unexpected throw: %s\n",           \
                        __FILE__, __LINE__, #expr);                     \
        }                                                               \
    } while (0)

// ------------------------------------------------------------- fixtures --

static NodeEvent make_event(const std::string& kind, const std::string& work_id,
                            std::uint64_t local_seq, std::uint64_t tick,
                            std::uint64_t delivery, Json payload,
                            std::optional<WorkAmount> delta = std::nullopt) {
    NodeEvent event;
    event.kind = kind;
    event.work_id = work_id;
    event.tick = tick;
    event.producer = "cpp_online";
    event.delivery_sequence = delivery;
    event.local_seq = local_seq;
    event.delta_served = delta;
    event.payload = std::move(payload);
    return event;
}

static WorkAmount amount(std::optional<double> ops, std::optional<double> bytes,
                         std::optional<double> latency_ns) {
    return WorkAmount{ops, bytes, latency_ns};
}

// v4 commit payload shaped like test_work_ledger._v4_envelope / _node_entry:
// identity + total + parents flat at the payload top level.
static Json node_commit_payload(std::uint64_t node_id,
                                std::uint64_t num_ops = 100,
                                std::uint64_t bytes = 0,
                                std::uint64_t runtime_ns = 50,
                                Json parents = Json::array()) {
    Json payload = Json::object();
    payload["request_id"] = "req-1";
    payload["stage"] = "prefill";
    payload["generation"] = 0;
    payload["rank"] = 0;
    payload["node_id"] = node_id;
    payload["operation"] = "compute";
    payload["total"] =
        Json{{"ops", num_ops}, {"bytes", bytes}, {"latency_ns", runtime_ns}};
    payload["parents"] = std::move(parents);
    return payload;
}

// One coordinated delivery: begin -> stage -> finish -> commit.
static void apply_delivery(WorkLedgerCore& ledger, std::uint64_t delivery,
                           std::uint64_t tick, const std::string& kind,
                           const std::vector<NodeEvent>& events,
                           std::uint64_t seq_end) {
    Transaction tx =
        ledger.begin_protocol_v3_transaction(delivery, tick, kind);
    for (const NodeEvent& event : events) {
        ledger.tx_stage_event(tx, event.local_seq, event);
    }
    ledger.tx_finish(tx, seq_end);
    ledger.tx_commit(tx);
}

static std::uint64_t seq_end_of(const std::vector<NodeEvent>& events) {
    return events.empty() ? 0 : events.back().local_seq;
}

static void apply_delivery(WorkLedgerCore& ledger, std::uint64_t delivery,
                           std::uint64_t tick, const std::string& kind,
                           const std::vector<NodeEvent>& events) {
    apply_delivery(ledger, delivery, tick, kind, events, seq_end_of(events));
}

// Ported test_work_ledger.TestLifecycle._ledger_with_node: full frame with
// one node commit.
static WorkLedgerCore ledger_with_node(const std::string& node_id) {
    WorkLedgerCore ledger;
    apply_delivery(ledger, 0, 10, "full",
                   {make_event("commit", node_id, 1, 10, 0,
                               node_commit_payload(1))});
    return ledger;
}

static const std::string kNode =
    canonical_work_id("req-1", "prefill", 0, 0, 1);
static const std::string kLeg =
    leg_work_id(canonical_work_id("req-1", "prefill", 0, 0, 1), "hbm:COMPUTE");

// ---------------------------------------------------------------- tests --

// test_work_ledger.TestWorkIdentity.test_canonical_work_id /
// test_leg_work_id_distinct_from_node
static void test_canonical_work_id() {
    CHECK(canonical_work_id("r1", "prefill", 2, 3, 7) == "r1#prefill:2:r3n7");
    CHECK(leg_work_id(kNode, "hbm:COMPUTE") == kNode + "@hbm:COMPUTE");
    CHECK(leg_work_id(kNode, "hbm:COMPUTE") != kNode);
}

// test_full_load_states: a commit event builds the initial row.
static void test_full_load_states() {
    WorkLedgerCore ledger = ledger_with_node(kNode);
    const CommittedWork* work = ledger.work(kNode);
    CHECK(work != nullptr);
    CHECK(work_state_name(work->state) == std::string("committed"));
    CHECK(work->total.ops == std::optional<double>(100.0));
    CHECK(work->total.bytes == std::optional<double>(0.0));
    CHECK(work->remaining.ops == std::optional<double>(100.0));
    CHECK(!work->served.ops.has_value());  // served starts all-None
    CHECK(work->committed_tick == std::optional<std::uint64_t>(10));
    CHECK(work->progress_source == "stale");  // upper-bound placeholder
    CHECK(work->service_unit == "job");
    CHECK(work->identity.operation == "compute");
    CHECK(work->dependencies.empty());
    CHECK(ledger.active_work_count() == 1);
    CHECK(ledger.version() == 1);
    CHECK(ledger.full_load_count() == 1);
    CHECK(ledger.delta_load_count() == 0);
    CHECK(ledger.applied_event_count() == 1);
    CHECK(ledger.replayed_event_count() == 0);
    CHECK(ledger.event_watermark() == 1);
    CHECK(ledger.last_delivery_sequence() == 0);
    CHECK(ledger.last_tick() == 10);
}

// test_issue_progress_completion_ack_release
static void test_issue_progress_completion_release() {
    WorkLedgerCore ledger = ledger_with_node(kNode);
    apply_delivery(ledger, 1, 40, "delta",
                   {make_event("issue", kNode, 2, 20, 1, Json::object()),
                    make_event("progress", kNode, 3, 30, 1, Json::object(),
                               amount(40, std::nullopt, std::nullopt)),
                    make_event("completion", kNode, 4, 40, 1, Json::object()),
                    make_event("ack", kNode, 5, 40, 1, Json::object()),
                    make_event("release", kNode, 6, 40, 1, Json::object())});
    CHECK(ledger.work(kNode) == nullptr);  // released = retired
    CHECK(ledger.active_work_count() == 0);
    CHECK(ledger.applied_event_count() == 6);
    CHECK(ledger.event_watermark() == 6);
    CHECK(ledger.version() == 2);
    CHECK(ledger.delta_load_count() == 1);
    Json report = ledger.conservation_json();
    CHECK(report["conserved"] == Json(true));
    CHECK(report["active_works"] == Json(std::uint64_t{0}));
    CHECK(report["violations"].empty());
}

// test_progress_marks_inferred_and_conserves
static void test_progress_marks_inferred_and_conserves() {
    WorkLedgerCore ledger = ledger_with_node(kNode);
    apply_delivery(ledger, 1, 30, "delta",
                   {make_event("progress", kNode, 2, 30, 1, Json::object(),
                               amount(40, std::nullopt, std::nullopt))});
    const CommittedWork* work = ledger.work(kNode);
    CHECK(work->progress_source == "inferred");
    CHECK(work->served.ops == std::optional<double>(40.0));
    CHECK(work->remaining.ops == std::optional<double>(60.0));
    CHECK(work_state_name(work->state) == std::string("in_service"));
    CHECK(work->issued_tick == std::optional<std::uint64_t>(30));
    Json report = ledger.conservation_json();
    CHECK(report["conserved"] == Json(true));
    CHECK(report["inferred_progress_works"] == Json(std::uint64_t{1}));
    CHECK(report["stale_amount_works"] == Json(std::uint64_t{0}));
}

// test_progress_remaining_field_reconciles_backend_tick_end
static void test_progress_remaining_field_reconciles() {
    WorkLedgerCore ledger = ledger_with_node(kNode);
    apply_delivery(ledger, 1, 30, "delta",
                   {make_event("progress", kNode, 2, 30, 1,
                               Json{{"remaining_ops", 60}},
                               amount(40, std::nullopt, std::nullopt))});
    const CommittedWork* work = ledger.work(kNode);
    CHECK(work->remaining.ops == std::optional<double>(60.0));
    CHECK(work->served.ops == std::optional<double>(40.0));  // total - residual

    // A wire residual disagreeing with the delta fails closed and leaves the
    // accepted state untouched.
    CHECK_THROW(apply_delivery(
        ledger, 2, 40, "delta",
        {make_event("progress", kNode, 3, 40, 2, Json{{"remaining_ops", 1}},
                    amount(10, std::nullopt, std::nullopt))}));
    CHECK(ledger.work(kNode)->remaining.ops == std::optional<double>(60.0));
    CHECK(ledger.work(kNode)->served.ops == std::optional<double>(40.0));
    CHECK(ledger.applied_event_count() == 2);
    // A matching residual continues the exact conservation line.
    apply_delivery(ledger, 2, 40, "delta",
                   {make_event("progress", kNode, 3, 40, 2,
                               Json{{"remaining_ops", 50}},
                               amount(10, std::nullopt, std::nullopt))});
    CHECK(ledger.work(kNode)->remaining.ops == std::optional<double>(50.0));
    CHECK(ledger.work(kNode)->served.ops == std::optional<double>(50.0));
    CHECK(ledger.conservation_json()["conserved"] == Json(true));
}

// test_illegal_transition_fail_closed (ack on committed)
static void test_illegal_transition_fail_closed() {
    WorkLedgerCore ledger = ledger_with_node(kNode);
    CHECK_THROW(apply_delivery(ledger, 1, 20, "delta",
                               {make_event("ack", kNode, 2, 20, 1,
                                           Json::object())}));
    CHECK(ledger.work(kNode) != nullptr);
    CHECK(work_state_name(ledger.work(kNode)->state)
          == std::string("committed"));
    CHECK(ledger.applied_event_count() == 1);
    CHECK(ledger.active_work_count() == 1);
}

// test_over_serving_fail_closed
static void test_over_serving_fail_closed() {
    WorkLedgerCore ledger = ledger_with_node(kNode);
    CHECK_THROW(apply_delivery(
        ledger, 1, 30, "delta",
        {make_event("progress", kNode, 2, 30, 1, Json::object(),
                    amount(1000, std::nullopt, std::nullopt))}));
    CHECK(ledger.work(kNode)->remaining.ops == std::optional<double>(100.0));
    CHECK(!ledger.work(kNode)->served.ops.has_value());  // untouched served
}

// commit for already-known work / lifecycle events for unknown work
static void test_commit_known_and_unknown_work() {
    WorkLedgerCore ledger = ledger_with_node(kNode);
    CHECK_THROW(apply_delivery(ledger, 1, 20, "delta",
                               {make_event("commit", kNode, 2, 20, 1,
                                           node_commit_payload(1))}));
    CHECK_THROW(apply_delivery(ledger, 1, 20, "delta",
                               {make_event("progress", "ghost", 2, 20, 1,
                                           Json::object())}));
    CHECK_THROW(apply_delivery(ledger, 1, 20, "delta",
                               {make_event("issue", "ghost", 2, 20, 1,
                                           Json::object())}));
    CHECK_THROW(apply_delivery(ledger, 1, 20, "delta",
                               {make_event("completion", "ghost", 2, 20, 1,
                                           Json::object())}));
    CHECK_THROW(apply_delivery(ledger, 1, 20, "delta",
                               {make_event("ack", "ghost", 2, 20, 1,
                                           Json::object())}));
    CHECK_THROW(apply_delivery(ledger, 1, 20, "delta",
                               {make_event("release", "ghost", 2, 20, 1,
                                           Json::object())}));
    CHECK(ledger.active_work_count() == 1);
    CHECK(ledger.applied_event_count() == 1);
}

// _KIND_SOURCE_STATES domains (work_ledger.py 86-96).
static void test_kind_source_state_domains() {
    // issue is legal on committed/ready only; a repeated issue fails the
    // domain check even though transition() itself is idempotent.
    {
        WorkLedgerCore ledger = ledger_with_node(kNode);
        apply_delivery(ledger, 1, 20, "delta",
                       {make_event("issue", kNode, 2, 20, 1, Json::object())});
        CHECK(work_state_name(ledger.work(kNode)->state)
              == std::string("issued"));
        CHECK(ledger.work(kNode)->issued_tick
              == std::optional<std::uint64_t>(20));
        CHECK_THROW(apply_delivery(ledger, 2, 30, "delta",
                                   {make_event("issue", kNode, 3, 30, 2,
                                               Json::object())}));
    }
    // progress is illegal on finished works.
    {
        WorkLedgerCore ledger = ledger_with_node(kNode);
        apply_delivery(ledger, 1, 30, "delta",
                       {make_event("completion", kNode, 2, 30, 1,
                                   Json::object())});
        CHECK(work_state_name(ledger.work(kNode)->state)
              == std::string("completed"));
        CHECK(!ledger.work(kNode)->remaining.ops.has_value());  // finish clears
        CHECK_THROW(apply_delivery(
            ledger, 2, 40, "delta",
            {make_event("progress", kNode, 3, 40, 2, Json::object(),
                        amount(1, std::nullopt, std::nullopt))}));
    }
    // completion reaffirms completed but not acknowledged.
    {
        WorkLedgerCore ledger = ledger_with_node(kNode);
        apply_delivery(ledger, 1, 30, "delta",
                       {make_event("completion", kNode, 2, 30, 1,
                                   Json::object()),
                        make_event("completion", kNode, 3, 30, 1,
                                   Json::object())});
        CHECK(work_state_name(ledger.work(kNode)->state)
              == std::string("completed"));
        apply_delivery(ledger, 2, 40, "delta",
                       {make_event("ack", kNode, 4, 40, 2, Json::object())});
        CHECK(work_state_name(ledger.work(kNode)->state)
              == std::string("acknowledged"));
        CHECK(ledger.work(kNode)->acked_tick
              == std::optional<std::uint64_t>(40));
        CHECK_THROW(apply_delivery(ledger, 3, 50, "delta",
                                   {make_event("completion", kNode, 5, 50, 3,
                                               Json::object())}));
    }
    // ack requires completed; release requires completed/acknowledged.
    {
        WorkLedgerCore ledger = ledger_with_node(kNode);
        apply_delivery(ledger, 1, 30, "delta",
                       {make_event("issue", kNode, 2, 30, 1, Json::object())});
        CHECK_THROW(apply_delivery(ledger, 2, 40, "delta",
                                   {make_event("ack", kNode, 3, 40, 2,
                                               Json::object())}));
        CHECK_THROW(apply_delivery(ledger, 2, 40, "delta",
                                   {make_event("release", kNode, 3, 40, 2,
                                               Json::object())}));
        apply_delivery(ledger, 2, 40, "delta",
                       {make_event("completion", kNode, 3, 40, 2,
                                   Json::object()),
                        make_event("release", kNode, 4, 40, 2,
                                   Json::object())});  // completed -> released
        CHECK(ledger.work(kNode) == nullptr);
    }
}

// tx gates: strictly increasing logical seq (duplicate / decrease / zero).
static void test_seq_gates() {
    WorkLedgerCore ledger = ledger_with_node(kNode);
    const Json none = Json::object();
    {
        Transaction tx = ledger.begin_protocol_v3_transaction(1, 40, "delta");
        ledger.tx_stage_event(tx, 2,
                              make_event("issue", kNode, 2, 40, 1, none));
        CHECK_THROW(ledger.tx_stage_event(
            tx, 2, make_event("issue", kNode, 2, 40, 1, none)));  // duplicate
        CHECK_THROW(ledger.tx_stage_event(
            tx, 1, make_event("issue", kNode, 1, 40, 1, none)));  // decrease
        CHECK(tx.event_count == 1);
        CHECK(tx.previous_seq == 2);
    }
    {
        Transaction tx = ledger.begin_protocol_v3_transaction(2, 40, "delta");
        CHECK_THROW(ledger.tx_stage_event(
            tx, 0, make_event("issue", kNode, 0, 40, 2, none)));  // zero
        CHECK(tx.event_count == 0);
    }
}

// tx gates: event identity must agree with the envelope.
static void test_event_identity_gates() {
    WorkLedgerCore ledger = ledger_with_node(kNode);
    Transaction tx = ledger.begin_protocol_v3_transaction(1, 40, "delta");
    CHECK_THROW(ledger.tx_stage_event(
        tx, 5, make_event("issue", kNode, 4, 40, 1, Json::object())));
    CHECK_THROW(ledger.tx_stage_event(
        tx, 5, make_event("issue", kNode, 5, 40, 99, Json::object())));
    CHECK(tx.event_count == 0);
}

// tx gates: monotonic envelope time window [previous_tick, envelope tick].
static void test_tick_window_gates() {
    WorkLedgerCore ledger = ledger_with_node(kNode);  // last_tick 10
    const Json none = Json::object();
    {
        Transaction tx = ledger.begin_protocol_v3_transaction(1, 40, "delta");
        CHECK_THROW(ledger.tx_stage_event(
            tx, 2, make_event("issue", kNode, 2, 41, 1, none)));  // > envelope
        CHECK_THROW(ledger.tx_stage_event(
            tx, 2,
            make_event("issue", kNode, 2, 9, 1, none)));  // < ledger last_tick
        ledger.tx_stage_event(tx, 2,
                              make_event("issue", kNode, 2, 40, 1, none));
        ledger.tx_stage_event(tx, 3,
                              make_event("completion", kNode, 3, 40, 1, none));
        CHECK_THROW(ledger.tx_stage_event(
            tx, 4, make_event("ack", kNode, 4, 39, 1, none)));  // regression
        CHECK(tx.event_count == 2);
        CHECK(tx.previous_tick == 40);
    }
    // begin gates: envelope tick regression and delivery monotonicity.
    CHECK_THROW(ledger.begin_protocol_v3_transaction(2, 5, "delta"));
    CHECK_THROW(ledger.begin_protocol_v3_transaction(0, 40, "delta"));
    CHECK_NO_THROW(ledger.begin_protocol_v3_transaction(2, 40, "delta"));
}

// begin gates: snapshot_kind domain and the pristine-full rule.
static void test_begin_gates() {
    WorkLedgerCore ledger;
    CHECK_THROW(ledger.begin_protocol_v3_transaction(0, 0, "partial"));
    apply_delivery(ledger, 0, 10, "full",
                   {make_event("commit", kNode, 1, 10, 0,
                               node_commit_payload(1))});
    CHECK_THROW(ledger.begin_protocol_v3_transaction(1, 20, "full"));
    // Even an empty (all-retired) ledger with load history stays non-full.
    apply_delivery(ledger, 1, 40, "delta",
                   {make_event("completion", kNode, 2, 40, 1, Json::object()),
                    make_event("ack", kNode, 3, 40, 1, Json::object()),
                    make_event("release", kNode, 4, 40, 1, Json::object())});
    CHECK(ledger.active_work_count() == 0);
    CHECK_THROW(ledger.begin_protocol_v3_transaction(2, 50, "full"));
    // Delta on a loaded history is fine.
    Transaction tx = ledger.begin_protocol_v3_transaction(2, 50, "delta");
    ledger.tx_finish(tx, 4);
    ledger.tx_commit(tx);
    CHECK(ledger.delta_load_count() == 2);
    // A delta-first ledger is also non-pristine for full.
    WorkLedgerCore delta_first;
    apply_delivery(delta_first, 0, 5, "delta", {});
    CHECK_THROW(delta_first.begin_protocol_v3_transaction(1, 6, "full"));
}

// Staging after finish and double commit are rejected.
static void test_transaction_lifecycle_gates() {
    WorkLedgerCore ledger;
    Transaction tx = ledger.begin_protocol_v3_transaction(0, 10, "full");
    ledger.tx_stage_event(tx, 1, make_event("commit", kNode, 1, 10, 0,
                                            node_commit_payload(1)));
    ledger.tx_finish(tx, 1);
    CHECK(tx.finish_called);
    CHECK_THROW(ledger.tx_stage_event(
        tx, 2, make_event("issue", kNode, 2, 10, 0, Json::object())));
    ledger.tx_commit(tx);
    CHECK_THROW(ledger.tx_commit(tx));
    // finish below the staged seq is rejected.
    Transaction tx2 = ledger.begin_protocol_v3_transaction(1, 20, "delta");
    ledger.tx_stage_event(tx2, 5,
                          make_event("issue", kNode, 5, 20, 1, Json::object()));
    CHECK_THROW(ledger.tx_finish(tx2, 4));
    ledger.tx_finish(tx2, 5);
    ledger.tx_commit(tx2);
    CHECK(ledger.event_watermark() == 5);
}

// Commit counters for full and delta deliveries, empty-frame watermark, and
// the last_tick max across interleaved transactions.
static void test_commit_counters() {
    WorkLedgerCore ledger;
    // Empty full: load counted, version untouched (event_count == 0).
    apply_delivery(ledger, 0, 0, "full", {}, 0);
    CHECK(ledger.version() == 0);
    CHECK(ledger.full_load_count() == 1);
    CHECK(ledger.applied_event_count() == 0);
    CHECK(ledger.event_watermark() == 0);
    // Empty delta: watermark moves to the frame's logical end.
    apply_delivery(ledger, 1, 10, "delta", {}, 7);
    CHECK(ledger.delta_load_count() == 1);
    CHECK(ledger.event_watermark() == 7);
    CHECK(ledger.version() == 0);
    CHECK(ledger.last_delivery_sequence() == 1);
    CHECK(ledger.last_tick() == 10);
    // Non-empty delivery bumps version once and advances every counter.
    apply_delivery(ledger, 2, 40, "delta",
                   {make_event("commit", kNode, 8, 30, 2,
                               node_commit_payload(1)),
                    make_event("issue", kNode, 9, 40, 2, Json::object())});
    CHECK(ledger.version() == 1);
    CHECK(ledger.applied_event_count() == 2);
    CHECK(ledger.event_watermark() == 9);
    CHECK(ledger.last_delivery_sequence() == 2);
    CHECK(ledger.last_tick() == 40);
    // Interleaved transactions: an older envelope tick cannot regress
    // last_tick (commit keeps the max like work_ledger.py 1925).
    Transaction early = ledger.begin_protocol_v3_transaction(3, 45, "delta");
    Transaction late = ledger.begin_protocol_v3_transaction(4, 60, "delta");
    ledger.tx_finish(late, 9);
    ledger.tx_commit(late);
    CHECK(ledger.last_tick() == 60);
    CHECK(ledger.last_delivery_sequence() == 4);
    ledger.tx_finish(early, 9);
    ledger.tx_commit(early);
    CHECK(ledger.last_tick() == 60);
    CHECK(ledger.last_delivery_sequence() == 3);
}

// COW isolation: staged rows are invisible until commit; tx_shadow_work sees
// the staged view (work_ledger shadow._works lookups).
static void test_shadow_isolation() {
    WorkLedgerCore ledger;
    Transaction tx = ledger.begin_protocol_v3_transaction(0, 10, "full");
    ledger.tx_stage_event(tx, 1, make_event("commit", kNode, 1, 10, 0,
                                            node_commit_payload(1)));
    CHECK(ledger.work(kNode) == nullptr);
    CHECK(ledger.active_work_count() == 0);
    CHECK(ledger.works_export_json()["nodes"].empty());
    CHECK(ledger.pricing_json()["pricing_rows"].empty());
    CHECK(ledger.tx_shadow_work(tx, kNode) != nullptr);
    CHECK(ledger.tx_shadow_work(tx, kNode)->total.ops
          == std::optional<double>(100.0));
    CHECK(tx.touched_set.count(kNode) == 1);
    CHECK(tx.touched_work_ids.size() == 1);
    CHECK(tx.event_count == 1);
    ledger.tx_finish(tx, 1);
    ledger.tx_commit(tx);
    CHECK(ledger.work(kNode) != nullptr);
    // A release staged over a base row hides it in the shadow view only.
    Transaction tx3 = ledger.begin_protocol_v3_transaction(1, 30, "delta");
    ledger.tx_stage_event(tx3, 2,
                          make_event("completion", kNode, 2, 30, 1,
                                     Json::object()));
    ledger.tx_stage_event(tx3, 3, make_event("ack", kNode, 3, 30, 1,
                                             Json::object()));
    CHECK(ledger.tx_shadow_work(tx3, kNode) != nullptr);  // overlay clone
    ledger.tx_stage_event(tx3, 4, make_event("release", kNode, 4, 30, 1,
                                             Json::object()));
    CHECK(ledger.tx_shadow_work(tx3, kNode) == nullptr);
    CHECK(ledger.work(kNode) != nullptr);  // base untouched pre-commit
    CHECK(tx3.work_deleted.count(kNode) == 1);
    CHECK(tx3.work_writes.count(kNode) == 0);
    ledger.tx_commit(tx3);
    CHECK(ledger.work(kNode) == nullptr);
}

// A failed staged event leaves zero residue; later deliveries apply cleanly.
static void test_atomicity_on_failed_delivery() {
    WorkLedgerCore ledger;
    Transaction tx = ledger.begin_protocol_v3_transaction(0, 10, "full");
    ledger.tx_stage_event(tx, 1, make_event("commit", kNode, 1, 10, 0,
                                            node_commit_payload(1)));
    CHECK_THROW(ledger.tx_stage_event(
        tx, 2, make_event("commit", kNode, 2, 10, 0, node_commit_payload(2))));
    CHECK(ledger.work(kNode) == nullptr);
    CHECK(ledger.active_work_count() == 0);
    CHECK(ledger.version() == 0);
    CHECK(ledger.full_load_count() == 0);
    CHECK(ledger.event_watermark() == 0);
    CHECK((ledger.works_export_json() ==
           Json{{"nodes", Json::array()},
                {"hbm", Json::array()},
                {"pool", Json::array()},
                {"noc", {{"as_of_tick", nullptr},
                         {"flows", Json::array()},
                         {"links", Json::array()}}},
                {"slots", Json::array()}}));
    // The ledger is still usable and pristine for a full frame.
    apply_delivery(ledger, 0, 10, "full",
                   {make_event("commit", kNode, 1, 10, 0,
                               node_commit_payload(1))});
    CHECK(ledger.active_work_count() == 1);
    CHECK(ledger.full_load_count() == 1);
}

// test_compact_dependencies_canonicalized (parents -> canonical work ids)
// plus the explicit-dependencies precedence rule.
static void test_parents_canonicalized() {
    const std::string node2 = canonical_work_id("req-1", "prefill", 0, 0, 2);
    const std::string node1 = canonical_work_id("req-1", "prefill", 0, 0, 1);
    WorkLedgerCore ledger;
    Transaction tx = ledger.begin_protocol_v3_transaction(0, 10, "full");
    ledger.tx_stage_event(tx, 1,
                          make_event("commit", node2, 1, 10, 0,
                                     node_commit_payload(2, 100, 0, 50, {1})));
    const CommittedWork* work = ledger.tx_shadow_work(tx, node2);
    CHECK(work != nullptr);
    CHECK(work->dependencies == std::vector<std::string>({node1}));
    // The dependents index registers the child under the parent id.
    const auto dependents = tx.dependents_writes.find(node1);
    CHECK(dependents != tx.dependents_writes.end());
    CHECK(dependents->second.count(node2) == 1);
    // Explicit canonical dependencies take precedence over parents.
    Json payload = node_commit_payload(2, 100, 0, 50, {1});
    payload["dependencies"] = Json::array({"other#stage:0:r0n9"});
    ledger.tx_stage_event(tx, 2, make_event("commit", "w-explicit", 2, 10, 0,
                                            payload));
    CHECK(ledger.tx_shadow_work(tx, "w-explicit")->dependencies
          == std::vector<std::string>({"other#stage:0:r0n9"}));
    // An empty parents list canonicalizes to no dependencies.
    ledger.tx_stage_event(tx, 3,
                          make_event("commit", "w-empty", 3, 10, 0,
                                     node_commit_payload(3, 100, 0, 50,
                                                         Json::array())));
    CHECK(ledger.tx_shadow_work(tx, "w-empty")->dependencies.empty());
    // Present-null parents skip canonicalization entirely (Python treats a
    // falsy parents value as absent for the parents branch).
    CHECK_NO_THROW(ledger.tx_stage_event(
        tx, 4,
        make_event("commit", "w-null-parents", 4, 10, 0,
                   Json{{"request_id", "req-1"},
                        {"stage", "prefill"},
                        {"generation", 0},
                        {"rank", 0},
                        {"node_id", 3},
                        {"parents", nullptr}})));
    CHECK(ledger.tx_shadow_work(tx, "w-null-parents")->dependencies.empty());
}

// test_release_condition_retained_while_depended.
static void test_release_retains_obligations() {
    const std::string parent = canonical_work_id("req-1", "prefill", 0, 0, 1);
    const std::string child = canonical_work_id("req-1", "prefill", 0, 0, 2);
    WorkLedgerCore ledger;
    Json parent_payload = node_commit_payload(1);
    parent_payload["release_conditions"] = Json::array({"watch#1"});
    Json child_payload = node_commit_payload(2);
    child_payload["dependencies"] = Json::array({parent});
    apply_delivery(ledger, 0, 10, "full",
                   {make_event("commit", parent, 1, 10, 0, parent_payload),
                    make_event("commit", child, 2, 10, 0, child_payload)});
    // Parent completes and retires while the child is still in flight.
    Transaction tx = ledger.begin_protocol_v3_transaction(1, 40, "delta");
    ledger.tx_stage_event(tx, 3, make_event("completion", parent, 3, 40, 1,
                                            Json::object()));
    ledger.tx_stage_event(tx, 4, make_event("ack", parent, 4, 40, 1,
                                            Json::object()));
    ledger.tx_stage_event(tx, 5, make_event("release", parent, 5, 40, 1,
                                            Json::object()));
    const auto retained = tx.obligations_writes.find(parent);
    CHECK(retained != tx.obligations_writes.end());
    CHECK(retained->second == std::vector<std::string>({"watch#1"}));
    CHECK(tx.obligations_deleted.empty());
    CHECK(tx.work_deleted.count(parent) == 1);
    ledger.tx_commit(tx);
    CHECK(ledger.work(parent) == nullptr);
    CHECK(ledger.work(child) != nullptr);
    // Child retires next: the dependency closure disappears and the retained
    // obligation dies with the last reference.
    Transaction tx2 = ledger.begin_protocol_v3_transaction(2, 60, "delta");
    ledger.tx_stage_event(tx2, 6,
                          make_event("completion", child, 6, 60, 2,
                                     Json::object()));
    ledger.tx_stage_event(tx2, 7, make_event("ack", child, 7, 60, 2,
                                             Json::object()));
    ledger.tx_stage_event(tx2, 8, make_event("release", child, 8, 60, 2,
                                             Json::object()));
    CHECK(tx2.dependents_deleted.count(parent) == 1);
    CHECK(tx2.obligations_deleted.count(parent) == 1);
    CHECK(tx2.obligations_writes.empty());
    ledger.tx_commit(tx2);
    CHECK(ledger.active_work_count() == 0);
}

// Releasing a work removes it from its dependencies' dependent sets; a
// dependency without retained obligations drops nothing.
static void test_dependents_cleanup_on_release() {
    const std::string dep_a =
        canonical_work_id("req-1", "prefill", 0, 0, 11);
    const std::string dependent =
        canonical_work_id("req-1", "prefill", 0, 0, 12);
    WorkLedgerCore ledger;
    Json payload = node_commit_payload(12);
    payload["dependencies"] = Json::array({dep_a});
    apply_delivery(ledger, 0, 10, "full",
                   {make_event("commit", dep_a, 1, 10, 0,
                               node_commit_payload(11)),
                    make_event("commit", dependent, 2, 10, 0, payload)});
    Transaction tx = ledger.begin_protocol_v3_transaction(1, 40, "delta");
    ledger.tx_stage_event(tx, 3,
                          make_event("completion", dependent, 3, 40, 1,
                                     Json::object()));
    ledger.tx_stage_event(tx, 4, make_event("ack", dependent, 4, 40, 1,
                                            Json::object()));
    ledger.tx_stage_event(tx, 5, make_event("release", dependent, 5, 40, 1,
                                            Json::object()));
    // dependents[dep_a] became empty -> tombstoned in the overlay.
    CHECK(tx.dependents_deleted.count(dep_a) == 1);
    CHECK(tx.obligations_writes.empty());
    CHECK(tx.obligations_deleted.empty());
    ledger.tx_commit(tx);
    CHECK(ledger.active_work_count() == 1);  // dep_a itself still active
}

// test_full_vs_incremental_midflight_first_seen: an authoritative remaining
// on a mid-flight first-seen leg stays authoritative through later progress.
static void test_commit_authoritative_remaining() {
    WorkLedgerCore ledger;
    apply_delivery(ledger, 0, 0, "full", {}, 0);
    apply_delivery(ledger, 1, 100, "delta",
                   {make_event("commit", kLeg, 1, 100, 1,
                               Json{{"leg", "hbm:COMPUTE"},
                                    {"request_id", "req-1"},
                                    {"stage", "prefill"},
                                    {"generation", 0},
                                    {"rank", 0},
                                    {"node_id", 1},
                                    {"service_unit", "bytes"},
                                    {"total", Json{{"ops", 100},
                                                   {"bytes", 200},
                                                   {"latency_ns", 80}}},
                                    {"remaining_ops", 40},
                                    {"remaining_bytes", 80},
                                    {"remaining_latency_ns", 0}})});
    const CommittedWork* work = ledger.work(kLeg);
    CHECK(work->progress_source == "authoritative");
    CHECK(work->remaining.ops == std::optional<double>(40.0));
    CHECK(work->served.ops == std::optional<double>(60.0));
    CHECK(work->remaining.bytes == std::optional<double>(80.0));
    CHECK(work->served.bytes == std::optional<double>(120.0));
    CHECK(work->served.latency_ns == std::optional<double>(80.0));
    CHECK(work->service_unit == "bytes");
    CHECK(work->leg == std::optional<std::string>("hbm:COMPUTE"));
    // Progress continues along the same conservation line.
    apply_delivery(ledger, 2, 200, "delta",
                   {make_event("progress", kLeg, 2, 200, 2,
                               Json{{"remaining_ops", 10},
                                    {"remaining_bytes", 20}},
                               amount(30, 60, std::nullopt))});
    CHECK(work->remaining.ops == std::optional<double>(10.0));
    CHECK(work->served.ops == std::optional<double>(90.0));
    CHECK(work->remaining.bytes == std::optional<double>(20.0));
    CHECK(work->served.bytes == std::optional<double>(180.0));
    CHECK(work->progress_source == "authoritative");
    Json report = ledger.conservation_json();
    CHECK(report["conserved"] == Json(true));
    CHECK(report["stale_amount_works"] == Json(std::uint64_t{0}));
    CHECK(report["active_works"] == Json(std::uint64_t{1}));
}

// test_delta_commit_without_remaining_is_stale_upper_bound.
static void test_commit_without_remaining_is_stale() {
    WorkLedgerCore ledger;
    apply_delivery(ledger, 0, 0, "full", {}, 0);
    apply_delivery(ledger, 1, 100, "delta",
                   {make_event("commit", kLeg, 1, 100, 1,
                               Json{{"leg", "hbm:COMPUTE"},
                                    {"request_id", "req-1"},
                                    {"stage", "prefill"},
                                    {"generation", 0},
                                    {"rank", 0},
                                    {"node_id", 1},
                                    {"service_unit", "bytes"},
                                    {"total", Json{{"bytes", 100}}}})});
    const CommittedWork* work = ledger.work(kLeg);
    CHECK(work->remaining.bytes == std::optional<double>(100.0));
    CHECK(!work->served.bytes.has_value());
    CHECK(work->progress_source == "stale");
    Json report = ledger.conservation_json();
    CHECK(report["conserved"] == Json(true));
    CHECK(report["stale_amount_works"] == Json(std::uint64_t{1}));
}

// remaining > total is physically illegal and fails closed (served < 0).
static void test_commit_remaining_exceeds_total() {
    WorkLedgerCore ledger;
    Transaction tx = ledger.begin_protocol_v3_transaction(0, 0, "full");
    CHECK_THROW(ledger.tx_stage_event(
        tx, 1,
        make_event("commit", kLeg, 1, 0, 0,
                   Json{{"leg", "hbm:COMPUTE"},
                        {"total", Json{{"ops", 100}}},
                        {"remaining_ops", 150}})));
    CHECK(tx.event_count == 0);
    // A residual inside the epsilon band of zero total is accepted.
    CHECK_NO_THROW(ledger.tx_stage_event(
        tx, 2, make_event("commit", "w-tiny", 2, 0, 0,
                          Json{{"leg", "hbm:COMPUTE"},
                               {"total", Json{{"ops", 0}}},
                               {"remaining_ops", 1e-9}})));
    CHECK(ledger.tx_shadow_work(tx, "w-tiny") != nullptr);
    // Negative residuals beyond the epsilon fail the amount validation.
    CHECK_THROW(ledger.tx_stage_event(
        tx, 3, make_event("commit", "w-neg", 3, 0, 0,
                          Json{{"leg", "hbm:COMPUTE"},
                               {"total", Json{{"ops", 100}}},
                               {"remaining_ops", -5}})));
}

// Terminal first sight: completed commits replay with served = total.
static void test_terminal_first_seen_completed() {
    WorkLedgerCore ledger;
    apply_delivery(ledger, 0, 0, "full", {}, 0);
    apply_delivery(ledger, 1, 100, "delta",
                   {make_event("commit", "gone#stage:0:r0n7", 1, 100, 1,
                               Json{{"request_id", "gone"},
                                    {"stage", "stage"},
                                    {"generation", 0},
                                    {"rank", 0},
                                    {"node_id", 7},
                                    {"total", Json{{"ops", 50},
                                                   {"bytes", 70}}},
                                    {"state", "completed"}})});
    const CommittedWork* work = ledger.work("gone#stage:0:r0n7");
    CHECK(work_state_name(work->state) == std::string("completed"));
    CHECK(work->served == work->total);
    CHECK(work->remaining.is_empty());
    CHECK(work->completed_tick == std::optional<std::uint64_t>(100));
    CHECK(work->issued_tick == std::optional<std::uint64_t>(100));
    CHECK(work->progress_source == "stale");
    Json report = ledger.conservation_json();
    CHECK(report["conserved"] == Json(true));
}

// test_delta_commit_pool_state_string_mapping + _LEG_COMMIT_STATE_NAMES.
static void test_leg_commit_state_names() {
    WorkLedgerCore ledger;
    apply_delivery(ledger, 0, 0, "full", {}, 0);
    const std::string base = canonical_work_id("req-1", "prefill", 0, 0, 3);
    auto commit_leg = [&](const std::string& id, std::uint64_t seq,
                          const std::string& state) {
        Json payload = Json{{"leg", "pool:0:17"},
                            {"request_id", "req-1"},
                            {"stage", "prefill"},
                            {"generation", 0},
                            {"rank", 0},
                            {"node_id", 3},
                            {"total", Json{{"bytes", 500}}},
                            {"remaining_bytes", 200}};
        if (!state.empty()) payload["state"] = state;
        return make_event("commit", id, seq, 100, 1, payload);
    };
    std::vector<NodeEvent> events;
    std::vector<std::string> ids;
    std::uint64_t seq = 1;
    for (const std::string state :
         {"Streaming", "LatencyWaiting", "active", "PropagatingTail",
          "FluidCompleteAwaitingCallback", "streaming", "issued", ""}) {
        const std::string id =
            leg_work_id(base, "pool:0:1" + std::to_string(seq));
        ids.push_back(id);
        events.push_back(commit_leg(id, seq, state));
        ++seq;
    }
    apply_delivery(ledger, 1, 100, "delta", events);
    CHECK(work_state_name(ledger.work(ids[0])->state) == std::string("in_service"));
    CHECK(work_state_name(ledger.work(ids[1])->state) == std::string("issued"));
    CHECK(work_state_name(ledger.work(ids[2])->state) == std::string("in_service"));
    CHECK(work_state_name(ledger.work(ids[3])->state) == std::string("completed"));
    CHECK(work_state_name(ledger.work(ids[4])->state) == std::string("completed"));
    CHECK(work_state_name(ledger.work(ids[5])->state) == std::string("in_service"));
    CHECK(work_state_name(ledger.work(ids[6])->state) == std::string("issued"));
    CHECK(work_state_name(ledger.work(ids[7])->state) == std::string("committed"));
    CHECK(ledger.work(ids[0])->progress_source == "authoritative");
    // Unknown, non-string and wrongly-cased state names fail closed.
    Transaction tx = ledger.begin_protocol_v3_transaction(2, 110, "delta");
    CHECK_THROW(ledger.tx_stage_event(
        tx, seq,
        make_event("commit", leg_work_id(base, "pool:0:99"), seq, 110, 2,
                   Json{{"leg", "pool:0:17"}, {"state", "Hyperspace"}})));
    CHECK_THROW(ledger.tx_stage_event(
        tx, seq,
        make_event("commit", leg_work_id(base, "pool:0:98"), seq, 110, 2,
                   Json{{"leg", "pool:0:17"}, {"state", 5}})));
    CHECK_THROW(ledger.tx_stage_event(
        tx, seq,
        make_event("commit", leg_work_id(base, "pool:0:97"), seq, 110, 2,
                   Json{{"leg", "pool:0:17"}, {"state", "Committed"}})));
}

// Direct state-machine names on commit: issued/in_service replay; names with
// no event path (ready/acknowledged/released) settle at committed.
static void test_commit_state_machine_names() {
    WorkLedgerCore ledger;
    apply_delivery(ledger, 0, 0, "full", {}, 0);
    std::uint64_t delivery = 1;
    auto commit_state = [&](const std::string& tag,
                            const std::string& state) {
        const std::string id = "req#s:0:r0n" + tag;
        apply_delivery(ledger, delivery, 50, "delta",
                       {make_event("commit", id, delivery, 50, delivery,
                                   Json{{"request_id", "req"},
                                        {"stage", "s"},
                                        {"state", state}})},
                       delivery);
        ++delivery;
        return id;
    };
    CHECK(work_state_name(ledger.work(commit_state("1", "issued"))->state)
          == std::string("issued"));
    const std::string in_service = commit_state("2", "in_service");
    CHECK(work_state_name(ledger.work(in_service)->state) == std::string("in_service"));
    CHECK(ledger.work(in_service)->progress_source == "inferred");
    CHECK(ledger.work(in_service)->issued_tick
          == std::optional<std::uint64_t>(50));
    CHECK(work_state_name(ledger.work(commit_state("3", "ready"))->state)
          == std::string("committed"));
    CHECK(work_state_name(ledger.work(commit_state("4", "acknowledged"))->state)
          == std::string("committed"));
    CHECK(work_state_name(ledger.work(commit_state("5", "released"))->state)
          == std::string("committed"));
}

// progress serving an unavailable dimension / residual without a delta.
static void test_progress_dimension_gates() {
    WorkLedgerCore ledger;
    apply_delivery(ledger, 0, 0, "full", {}, 0);
    const std::string id = "req#s:0:r0n1";
    apply_delivery(ledger, 1, 10, "delta",
                   {make_event("commit", id, 1, 10, 1,
                               Json{{"request_id", "req"},
                                    {"stage", "s"},
                                    {"total", Json{{"bytes", 100}}}})});
    Transaction tx = ledger.begin_protocol_v3_transaction(2, 20, "delta");
    // delta.ops against a work without an ops dimension.
    CHECK_THROW(ledger.tx_stage_event(
        tx, 2, make_event("progress", id, 2, 20, 2, Json::object(),
                          amount(5, std::nullopt, std::nullopt))));
    // remaining_ops without a matching delta/remaining dimension.
    CHECK_THROW(ledger.tx_stage_event(
        tx, 2, make_event("progress", id, 2, 20, 2,
                          Json{{"remaining_ops", 3}}, std::nullopt)));
    CHECK(tx.event_count == 0);
    // The bytes dimension works and keeps conservation exact.
    ledger.tx_stage_event(tx, 2,
                          make_event("progress", id, 2, 20, 2,
                                     Json{{"remaining_bytes", 40}},
                                     amount(std::nullopt, 60, std::nullopt)));
    CHECK(ledger.tx_shadow_work(tx, id)->remaining.bytes
          == std::optional<double>(40.0));
    CHECK(ledger.tx_shadow_work(tx, id)->served.bytes
          == std::optional<double>(60.0));
    ledger.tx_finish(tx, 2);
    ledger.tx_commit(tx);
    CHECK(ledger.conservation_json()["conserved"] == Json(true));
}

// test_amount_math_and_validation: bool/negative/string/non-finite amounts
// and int() coercions on identity fields.
static void test_bad_amounts() {
    WorkLedgerCore ledger;
    Transaction tx = ledger.begin_protocol_v3_transaction(0, 0, "full");
    const auto commit_total = [&](const Json& total) {
        return make_event("commit", "probe#s:0:r0n1", 1, 0, 0,
                          Json{{"request_id", "probe"},
                               {"stage", "s"},
                               {"total", total}});
    };
    CHECK_THROW(ledger.tx_stage_event(
        tx, 1,
        commit_total(Json{{"ops", true},
                          {"bytes", nullptr},
                          {"latency_ns", nullptr}})));
    CHECK_THROW(ledger.tx_stage_event(
        tx, 1,
        commit_total(Json{{"ops", -5}, {"bytes", nullptr},
                          {"latency_ns", nullptr}})));
    CHECK_THROW(ledger.tx_stage_event(
        tx, 1,
        commit_total(Json{{"ops", "10"},
                          {"bytes", nullptr},
                          {"latency_ns", nullptr}})));
    CHECK_THROW(ledger.tx_stage_event(
        tx, 1,
        commit_total(Json{{"ops", nullptr},
                          {"bytes", nullptr},
                          {"latency_ns", nullptr},
                          {"surprise", 1}})));
    // Python json.loads admits NaN/Infinity literals; strict nlohmann parsing
    // does not, so the non-finite gate is exercised through a directly
    // constructed double (the C++ equivalent of float("inf") on the wire).
    const Json inf_total =
        Json{{"ops", std::numeric_limits<double>::infinity()},
             {"bytes", nullptr},
             {"latency_ns", nullptr}};
    CHECK_THROW(ledger.tx_stage_event(tx, 1, commit_total(inf_total)));
    CHECK(tx.event_count == 0);
    // Python int() coercions: bool -> 0/1, float -> truncation.
    CHECK_NO_THROW(ledger.tx_stage_event(
        tx, 1,
        make_event("commit", "probe#s:1:r0n1", 1, 0, 0,
                   Json{{"request_id", "probe"}, {"stage", "s"},
                        {"generation", true}})));
    CHECK(ledger.tx_shadow_work(tx, "probe#s:1:r0n1")->identity.generation
          == 1);
    CHECK_NO_THROW(ledger.tx_stage_event(
        tx, 2,
        make_event("commit", "probe#s:0:r0n2", 2, 0, 0,
                   Json{{"request_id", "probe"}, {"stage", "s"},
                        {"generation", 1.5}})));
    CHECK(ledger.tx_shadow_work(tx, "probe#s:0:r0n2")->identity.generation
          == 1);  // int(1.5) == 1
    // Integers beyond the native int64 range fail closed.
    const Json huge = Json::parse("{\"generation\": 10000000000000000000}");
    CHECK_THROW(ledger.tx_stage_event(
        tx, 3,
        make_event("commit", "probe#s:0:r0n3", 3, 0, 0,
                   Json{{"request_id", "probe"}, {"stage", "s"},
                        {"generation", huge["generation"]}})));
}

// export_state: nodes grouped by rank with static totals and state booleans.
static void test_export_nodes() {
    WorkLedgerCore ledger;
    apply_delivery(ledger, 0, 10, "full",
                   {make_event("commit", kNode, 1, 10, 0,
                               node_commit_payload(1)),
                    make_event("commit",
                               canonical_work_id("req-1", "decode", 0, 2, 5),
                               2, 10, 0,
                               [] {
                                   Json p = node_commit_payload(5, 7, 8, 9);
                                   p["stage"] = "decode";
                                   p["rank"] = 2;
                                   return p;
                               }())});
    apply_delivery(ledger, 1, 20, "delta",
                   {make_event("issue", kNode, 3, 20, 1, Json::object())});
    Json exported = ledger.works_export_json();
    CHECK(exported["nodes"].size() == 2);
    CHECK(exported["nodes"][0]["rank"] == Json(std::int64_t{0}));
    CHECK(exported["nodes"][0]["nodes"].size() == 1);
    Json row = exported["nodes"][0]["nodes"][0];
    CHECK(row["work_id"] == Json(kNode));
    CHECK(row["request_id"] == Json("req-1"));
    CHECK(row["stage"] == Json("prefill"));
    CHECK(row["kind"] == Json("compute"));
    CHECK(row["issued"] == Json(true));
    CHECK(row["finished"] == Json(false));
    CHECK(row["num_ops"] == Json(100.0));
    CHECK(row["bytes"] == Json(0.0));
    CHECK(row["runtime_ns"] == Json(50.0));
    CHECK(row["dependencies"] == Json::array());
    CHECK(row["progress_source"] == Json("stale"));
    CHECK(row["generation"] == Json(std::uint64_t{0}));
    CHECK(row["node_id"] == Json(std::uint64_t{1}));
    Json rank2 = exported["nodes"][1]["nodes"][0];
    CHECK(exported["nodes"][1]["rank"] == Json(std::int64_t{2}));
    CHECK(rank2["issued"] == Json(false));
    CHECK(rank2["num_ops"] == Json(7.0));
    CHECK(exported["hbm"].empty());
    CHECK(exported["pool"].empty());
    CHECK(exported["slots"].empty());
    // completed marks finished.
    apply_delivery(ledger, 2, 30, "delta",
                   {make_event("completion", kNode, 4, 30, 2,
                               Json::object())});
    CHECK(ledger.works_export_json()["nodes"][0]["nodes"][0]["finished"]
          == Json(true));
    // operation defaults to "invalid" when the payload omits it.
    WorkLedgerCore bare;
    apply_delivery(bare, 0, 5, "full",
                   {make_event("commit", "r#s:0:r0n1", 1, 5, 0,
                               Json{{"request_id", "r"},
                                    {"stage", "s"}})});
    CHECK(bare.works_export_json()["nodes"][0]["nodes"][0]["kind"]
          == Json("invalid"));
}

// export_state: hbm legs by rank with nullable authoritative residuals.
static void test_export_hbm() {
    WorkLedgerCore ledger;
    apply_delivery(ledger, 0, 0, "full", {}, 0);
    apply_delivery(ledger, 1, 100, "delta",
                   {make_event("commit", kLeg, 1, 100, 1,
                               Json{{"leg", "hbm:COMM_READ"},
                                    {"request_id", "req-1"},
                                    {"stage", "prefill"},
                                    {"generation", 0},
                                    {"rank", 0},
                                    {"node_id", 1},
                                    {"state", "Streaming"},
                                    {"total", Json{{"ops", 100},
                                                   {"bytes", 200}}},
                                    {"remaining_bytes", 80}}),
                    make_event("commit",
                               leg_work_id(
                                   canonical_work_id("req-2", "prefill", 0, 3,
                                                     4),
                                   "hbm:COMPUTE"),
                               2, 100, 1,
                               Json{{"leg", "hbm:COMPUTE"},
                                    {"request_id", "req-2"},
                                    {"stage", "prefill"},
                                    {"generation", 0},
                                    {"rank", 3},
                                    {"node_id", 4},
                                    {"total", Json{{"ops", 10}}}})});
    Json exported = ledger.works_export_json();
    CHECK(exported["hbm"].size() == 2);
    Json block0 = exported["hbm"][0];
    CHECK(block0["rank"] == Json(std::int64_t{0}));
    CHECK(block0["as_of_tick"].is_null());  // delta path never discloses it
    CHECK(block0["jobs"].size() == 1);
    Json job = block0["jobs"][0];
    // The export emits the identity-derived work id, which carries the wire
    // leg ("hbm:COMM_READ"), not the dict key (kLeg uses hbm:COMPUTE).
    CHECK(job["work_id"] == Json(kNode + "@hbm:COMM_READ"));
    CHECK(job["node_id"] == Json(std::uint64_t{1}));
    CHECK(job["kind"] == Json("COMM_READ"));
    CHECK(job["num_ops"] == Json(100.0));
    CHECK(job["bytes"] == Json(200.0));
    CHECK(job["total_latency_ns"] == Json(0));  // missing dimension -> 0
    CHECK(job["remaining_ops"].is_null());      // raw residuals stay null
    CHECK(job["remaining_bytes"] == Json(80.0));
    CHECK(job["remaining_latency_ns"].is_null());
    CHECK(job["state"] == Json("in_service"));
    CHECK(job["progress_source"] == Json("authoritative"));
    CHECK(exported["hbm"][1]["rank"] == Json(std::int64_t{3}));
    CHECK(exported["hbm"][1]["jobs"][0]["kind"] == Json("COMPUTE"));
    CHECK(exported["nodes"].empty());
}

// export_state: pool ports parse port:issue_seq and map lifecycle strings.
static void test_export_pool() {
    WorkLedgerCore ledger;
    apply_delivery(ledger, 0, 0, "full", {}, 0);
    const std::string streaming =
        leg_work_id(canonical_work_id("req-1", "prefill", 0, 0, 3), "pool:0:17");
    const std::string waiting =
        leg_work_id(canonical_work_id("req-1", "prefill", 0, 0, 4), "pool:2:18");
    apply_delivery(ledger, 1, 100, "delta",
                   {make_event("commit", streaming, 1, 100, 1,
                               Json{{"leg", "pool:0:17"},
                                    {"request_id", "req-1"},
                                    {"stage", "prefill"},
                                    {"generation", 0},
                                    {"rank", 5},
                                    {"node_id", 3},
                                    {"total", Json{{"bytes", 500}}},
                                    {"remaining_bytes", 200},
                                    {"state", "Streaming"}}),
                    make_event("commit", waiting, 2, 100, 1,
                               Json{{"leg", "pool:2:18"},
                                    {"request_id", "req-1"},
                                    {"stage", "prefill"},
                                    {"generation", 0},
                                    {"rank", 6},
                                    {"node_id", 4},
                                    {"total", Json{{"bytes", 500}}},
                                    {"state", "LatencyWaiting"}})});
    Json exported = ledger.works_export_json();
    CHECK(exported["pool"].size() == 2);
    CHECK(exported["pool"][0]["port_index"] == Json(std::int64_t{0}));
    CHECK(exported["pool"][0]["as_of_ns"].is_null());
    Json job = exported["pool"][0]["jobs"][0];
    // Identity-derived id: the payload carried rank 5 (sys_id), so the
    // derived id differs from the wire id used as the ledger key.
    CHECK(job["work_id"] == Json("req-1#prefill:0:r5n3@pool:0:17"));
    CHECK(job["issue_seq"] == Json(std::int64_t{17}));
    CHECK(job["sys_id"] == Json(std::uint64_t{5}));
    CHECK(job["node_id"] == Json(std::uint64_t{3}));
    CHECK(job["bytes"] == Json(500.0));
    CHECK(job["remaining_bytes"] == Json(200.0));
    CHECK(job["state"] == Json("Streaming"));
    CHECK(job["progress_source"] == Json("authoritative"));
    CHECK(exported["pool"][1]["port_index"] == Json(std::int64_t{2}));
    CHECK(exported["pool"][1]["jobs"][0]["state"] == Json("LatencyWaiting"));
    // Without an authoritative residual the upper-bound placeholder exports.
    CHECK(exported["pool"][1]["jobs"][0]["remaining_bytes"]
          == Json(500.0));
    // Malformed pool legs fail closed at export.
    WorkLedgerCore bad;
    apply_delivery(bad, 0, 0, "full", {}, 0);
    apply_delivery(bad, 1, 10, "delta",
                   {make_event("commit", "r#s:0:r0n1@pool:alpha:1", 1, 10, 1,
                               Json{{"leg", "pool:alpha:1"},
                                    {"request_id", "r"},
                                    {"stage", "s"},
                                    {"state", "Streaming"}})});
    CHECK_THROW(bad.works_export_json());
}

// export_state: noc flows aggregate active_flow_count per distinct link;
// unknown leg prefixes are skipped.
static void test_export_noc() {
    WorkLedgerCore ledger;
    apply_delivery(ledger, 0, 0, "full", {}, 0);
    auto noc_payload = [&](std::uint64_t flow_id, Json route) {
        return Json{{"leg", "noc:" + std::to_string(flow_id)},
                    {"request_id", "req-1"},
                    {"stage", "prefill"},
                    {"generation", 0},
                    {"rank", 2},
                    {"node_id", flow_id},
                    {"state", "active"},
                    {"total", Json{{"bytes", 400}}},
                    {"remaining_bytes", 250},
                    {"route", std::move(route)}};
    };
    apply_delivery(ledger, 1, 100, "delta",
                   {make_event("commit",
                               leg_work_id(canonical_work_id("req-1",
                                                             "prefill", 0, 2,
                                                             7),
                                           "noc:7"),
                               1, 100, 1,
                               noc_payload(7, Json::array({8, 7}))),
                    make_event("commit",
                               leg_work_id(canonical_work_id("req-1",
                                                             "prefill", 0, 2,
                                                             9),
                                           "noc:9"),
                               2, 100, 1, noc_payload(9, Json::array({8, 9})))});
    Json exported = ledger.works_export_json();
    CHECK(exported["noc"]["as_of_tick"].is_null());
    CHECK(exported["noc"]["flows"].size() == 2);
    Json flow = exported["noc"]["flows"][0];
    CHECK(flow["work_id"] == Json(leg_work_id(
                                canonical_work_id("req-1", "prefill", 0, 2, 7),
                                "noc:7")));
    CHECK(flow["flow_id"] == Json(std::uint64_t{7}));
    CHECK(flow["total_bytes"] == Json(400.0));
    CHECK(flow["remaining_bytes"] == Json(250.0));
    CHECK(flow["route"] == Json::array({8, 7}));  // route order preserved
    CHECK(flow["state"] == Json("in_service"));
    CHECK(flow["request_id"] == Json("req-1"));
    Json links = exported["noc"]["links"];
    CHECK(links.size() == 3);
    CHECK(links[0] == Json({{"link_id", 7}, {"active_flow_count", 1}}));
    CHECK(links[1] == Json({{"link_id", 8}, {"active_flow_count", 2}}));
    CHECK(links[2] == Json({{"link_id", 9}, {"active_flow_count", 1}}));
    // Pricing rows follow the works insertion order across legs.
    Json pricing_rows = ledger.pricing_json()["pricing_rows"];
    CHECK(pricing_rows.size() == 2);
    CHECK(pricing_rows[0]["work_id"]
          == Json(leg_work_id(canonical_work_id("req-1", "prefill", 0, 2, 7),
                              "noc:7")));
    CHECK(pricing_rows[1]["work_id"]
          == Json(leg_work_id(canonical_work_id("req-1", "prefill", 0, 2, 9),
                              "noc:9")));
    // Unknown leg prefixes are skipped entirely (no fourth family).
    WorkLedgerCore odd;
    apply_delivery(odd, 0, 0, "full", {}, 0);
    apply_delivery(odd, 1, 10, "delta",
                   {make_event("commit", "r#s:0:r0n1@gpu:0", 1, 10, 1,
                               Json{{"leg", "gpu:0"},
                                    {"request_id", "r"},
                                    {"stage", "s"},
                                    {"state", "issued"}})});
    Json odd_export = odd.works_export_json();
    CHECK(odd_export["nodes"].empty());
    CHECK(odd_export["hbm"].empty());
    CHECK(odd_export["pool"].empty());
    CHECK(odd_export["noc"]["flows"].empty());
    CHECK(odd.active_work_count() == 1);  // active, just not exported
    CHECK(odd.pricing_json()["pricing_rows"].empty());
}

// pricing_json: legs past the issue boundary, insertion order, committed
// counter, full row shape.
static void test_pricing_json() {
    WorkLedgerCore ledger;
    apply_delivery(ledger, 0, 10, "full",
                   {make_event("commit", kNode, 1, 10, 0,
                               node_commit_payload(1)),
                    make_event("commit", kLeg, 2, 10, 0,
                               Json{{"leg", "hbm:COMPUTE"},
                                    {"request_id", "req-1"},
                                    {"stage", "prefill"},
                                    {"generation", 0},
                                    {"rank", 0},
                                    {"node_id", 1},
                                    {"total", Json{{"ops", 100},
                                                   {"bytes", 200}}},
                                    {"remaining_ops", 40},
                                    {"remaining_bytes", 80}})});
    Json pricing = ledger.pricing_json();
    CHECK(pricing["pricing_rows"].empty());  // committed legs are not priced
    CHECK(pricing["committed_not_issued"] == Json(std::uint64_t{2}));
    apply_delivery(ledger, 1, 30, "delta",
                   {make_event("issue", kLeg, 3, 30, 1, Json::object()),
                    make_event("progress", kLeg, 4, 30, 1,
                               Json{{"remaining_ops", 30}},
                               amount(10, std::nullopt, std::nullopt))});
    pricing = ledger.pricing_json();
    CHECK(pricing["committed_not_issued"] == Json(std::uint64_t{1}));
    CHECK(pricing["pricing_rows"].size() == 1);
    Json row = pricing["pricing_rows"][0];
    CHECK(row["work_id"] == Json(kLeg));
    CHECK(row["state"] == Json("in_service"));
    CHECK(row["identity"]["request_id"] == Json("req-1"));
    CHECK(row["identity"]["stage"] == Json("prefill"));
    CHECK(row["identity"]["generation"] == Json(std::uint64_t{0}));
    CHECK(row["identity"]["rank"] == Json(std::uint64_t{0}));
    CHECK(row["identity"]["node_id"] == Json(std::uint64_t{1}));
    CHECK(row["identity"]["operation"] == Json(""));  // absent on this payload
    CHECK(row["total"] == Json({{"ops", 100.0},
                                {"bytes", 200.0},
                                {"latency_ns", nullptr}}));
    CHECK(row["served"] == Json({{"ops", 70.0},
                                 {"bytes", 120.0},
                                 {"latency_ns", nullptr}}));
    CHECK(row["remaining"] == Json({{"ops", 30.0},
                                    {"bytes", 80.0},
                                    {"latency_ns", nullptr}}));
    CHECK(row["progress_source"] == Json("authoritative"));
    CHECK(row["service_unit"] == Json("job"));
    CHECK(row["dependencies"] == Json::array());
    CHECK(row["release_conditions"] == Json::array());
    CHECK(row["route"].is_null());
    CHECK(row["weight"] == Json(0));
    CHECK(row["leg"] == Json("hbm:COMPUTE"));
    CHECK(row["committed_tick"] == Json(std::uint64_t{10}));
    CHECK(row["issued_tick"] == Json(std::uint64_t{30}));
    CHECK(row["completed_tick"].is_null());
    CHECK(row["acked_tick"].is_null());
    CHECK(row["released_tick"].is_null());
    CHECK(row["as_of_tick"].is_null());
}

// Insertion order is the Python dict order: re-inserted works append last.
static void test_insertion_order_semantics() {
    const std::string node_a = canonical_work_id("req-1", "prefill", 0, 0, 1);
    const std::string node_b = canonical_work_id("req-1", "prefill", 0, 0, 2);
    WorkLedgerCore ledger;
    apply_delivery(ledger, 0, 10, "full",
                   {make_event("commit", node_a, 1, 10, 0,
                               node_commit_payload(1)),
                    make_event("commit", node_b, 2, 10, 0,
                               node_commit_payload(2))});
    // Retire A, then re-commit it: B keeps slot 0, A re-appends at the end.
    apply_delivery(ledger, 1, 40, "delta",
                   {make_event("completion", node_a, 3, 40, 1,
                               Json::object()),
                    make_event("ack", node_a, 4, 40, 1, Json::object()),
                    make_event("release", node_a, 5, 40, 1,
                               Json::object())});
    CHECK(ledger.active_work_count() == 1);
    apply_delivery(ledger, 2, 50, "delta",
                   {make_event("commit", node_a, 6, 50, 2,
                               node_commit_payload(1))});
    Json nodes = ledger.works_export_json()["nodes"][0]["nodes"];
    CHECK(nodes.size() == 2);
    CHECK(nodes[0]["work_id"] == Json(node_b));
    CHECK(nodes[1]["work_id"] == Json(node_a));
    // Later commits keep appending in staging order.
    const std::string node_c = canonical_work_id("req-1", "prefill", 0, 0, 3);
    apply_delivery(ledger, 3, 60, "delta",
                   {make_event("commit", node_c, 7, 60, 3,
                               node_commit_payload(3))});
    nodes = ledger.works_export_json()["nodes"][0]["nodes"];
    CHECK(nodes.size() == 3);
    CHECK(nodes[2]["work_id"] == Json(node_c));
}

// work_id fallback helpers (_split_work_id family) drive identity defaults.
static void test_work_id_fallbacks() {
    WorkLedgerCore ledger;
    Transaction tx = ledger.begin_protocol_v3_transaction(0, 0, "full");
    ledger.tx_stage_event(tx, 1,
                          make_event("commit", "req#s:2:r3n7@hbm:X", 1, 0, 0,
                                     Json{{"leg", "hbm:X"}}));
    const CommittedWork* parsed =
        ledger.tx_shadow_work(tx, "req#s:2:r3n7@hbm:X");
    CHECK(parsed->identity.request_id == "req");
    CHECK(parsed->identity.stage == "s");
    CHECK(parsed->identity.generation == 2);
    CHECK(parsed->identity.rank == 3);
    CHECK(parsed->identity.node_id == 7);
    CHECK(parsed->work_id() == "req#s:2:r3n7@hbm:X");
    // Non-canonical NoC-style ids fall back to zeros like the Python
    // IndexError/ValueError handlers.
    ledger.tx_stage_event(tx, 2,
                          make_event("commit", "noc#42@noc:42", 2, 0, 0,
                                     Json{{"leg", "noc:42"}}));
    const CommittedWork* noc = ledger.tx_shadow_work(tx, "noc#42@noc:42");
    CHECK(noc->identity.request_id == "");
    CHECK(noc->identity.generation == 0);
    CHECK(noc->identity.rank == 0);
    CHECK(noc->identity.node_id == 0);
    CHECK(noc->work_id() == "#:0:r0n0@noc:42");  // derived id, not the wire id
    // node_id falls back to flow_id when node_id is absent.
    ledger.tx_stage_event(tx, 3,
                          make_event("commit", "req#s:0:r0n0", 3, 0, 0,
                                     Json{{"request_id", "req"},
                                          {"stage", "s"},
                                          {"flow_id", 42}}));
    CHECK(ledger.tx_shadow_work(tx, "req#s:0:r0n0")->identity.node_id == 42);
    CHECK(ledger.tx_shadow_work(tx, "req#s:0:r0n0")->work_id()
          == "req#s:0:r0n42");
}

// Same-tick staging stays inside the window across a multi-event delivery.
static void test_tick_windows_and_multi_event_deliveries() {
    WorkLedgerCore ledger = ledger_with_node(kNode);  // last_tick 10
    apply_delivery(ledger, 1, 40, "delta",
                   {make_event("issue", kNode, 2, 20, 1, Json::object()),
                    make_event("progress", kNode, 3, 20, 1, Json::object(),
                               amount(10, std::nullopt, std::nullopt)),
                    make_event("completion", kNode, 4, 40, 1,
                               Json::object())});
    CHECK(work_state_name(ledger.work(kNode)->state)
          == std::string("completed"));
    CHECK(ledger.work(kNode)->issued_tick == std::optional<std::uint64_t>(20));
    CHECK(ledger.work(kNode)->completed_tick
          == std::optional<std::uint64_t>(40));
    CHECK(ledger.work(kNode)->served.ops == std::optional<double>(10.0));
    CHECK(!ledger.work(kNode)->remaining.ops.has_value());
}

int main() {
    test_canonical_work_id();
    test_full_load_states();
    test_issue_progress_completion_release();
    test_progress_marks_inferred_and_conserves();
    test_progress_remaining_field_reconciles();
    test_illegal_transition_fail_closed();
    test_over_serving_fail_closed();
    test_commit_known_and_unknown_work();
    test_kind_source_state_domains();
    test_seq_gates();
    test_event_identity_gates();
    test_tick_window_gates();
    test_begin_gates();
    test_transaction_lifecycle_gates();
    test_commit_counters();
    test_shadow_isolation();
    test_atomicity_on_failed_delivery();
    test_parents_canonicalized();
    test_release_retains_obligations();
    test_dependents_cleanup_on_release();
    test_commit_authoritative_remaining();
    test_commit_without_remaining_is_stale();
    test_commit_remaining_exceeds_total();
    test_terminal_first_seen_completed();
    test_leg_commit_state_names();
    test_commit_state_machine_names();
    test_progress_dimension_gates();
    test_bad_amounts();
    test_export_nodes();
    test_export_hbm();
    test_export_pool();
    test_export_noc();
    test_pricing_json();
    test_insertion_order_semantics();
    test_work_id_fallbacks();
    test_tick_windows_and_multi_event_deliveries();
    std::printf("work_ledger_core_test: %d checks, %d failures\n", g_checks,
                g_failures);
    return g_failures == 0 ? 0 : 1;
}
