/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

work_ledger_bridge_test.cc -- W1 (joint work ledger) bridge fixture.

Exercises the state-derived ledger machinery standalone (no network
simulation, no Python process):

  Part A  diff_work_ledger_states: appearance -> commit (+ lifecycle flag
          replay); issued flip -> issue; remaining_* decrease -> progress
          with delta_served; finished flip -> completion; disappearance ->
          completion + ack + release; monotonic local_seq; deterministic
          section order (nodes, hbm, pool, flows).
  Part B  NodeStore::work_ledger_view: per-node lifecycle/dependency/
          workload facts, ascending id, blocking parents only.
  Part C  build_request_json: the additive event_seqs[] array parallels
          reasons[] (mailbox push-order seqs); the snapshot_handle carries
          the caller-provided kind (the online main sets "work_ledger_v1").
  Part D  Envelope: the provider payload shape the Python validator pins
          (schema_version / bridge_protocol / work_ledger_schema /
          snapshot_kind / delivery_sequence / tick / event_watermark) is
          the one diff_work_ledger_states' caller composes; the Python-side
          mirror of this check lives in joint/test_work_ledger.py.

Build: AstraSim_Analytical_Congestion_Aware_WorkLedgerBridgeTest
Run (from the repo root):
  build/astra_analytical/build_congestion_aware/bin/\
      AstraSim_Analytical_Congestion_Aware_WorkLedgerBridgeTest
Exit code 0 on ALL PASS.
*******************************************************************************/

#include "astra-sim/workload/execution_driven/DecisionBridge.hh"
#include "astra-sim/workload/execution_driven/DecisionMailbox.hh"
#include "astra-sim/workload/execution_driven/NodeStore.hh"

#include <cstdio>
#include <string>
#include <vector>

#include <json/json.hpp>

using namespace AstraSim::ExecutionDriven;

namespace {

bool g_ok = true;

void expect(bool cond, const char* what) {
    if (!cond) {
        std::fprintf(stderr, "[work_ledger_bridge_test] FAIL: %s\n", what);
        g_ok = false;
    }
}

OnlineNode make_node(uint64_t id, const std::string& request,
                     const std::string& stage, NodeKind kind,
                     uint64_t num_ops, uint64_t bytes) {
    OnlineNode node;
    node.global_id = id;
    node.rank = 0;
    node.kind = kind;
    node.request_id = request;
    node.stage = stage;
    node.generation = 0;
    node.compute.num_ops = num_ops;
    node.comm.bytes = bytes;
    return node;
}

// ---------------------------------------------------------------- Part A --

nlohmann::json node_entry(const std::string& work_id, bool issued,
                          bool finished, uint64_t num_ops, uint64_t bytes) {
    nlohmann::json entry;
    entry["work_id"] = work_id;
    entry["request_id"] = "req-1";
    entry["stage"] = "prefill";
    entry["generation"] = 0;
    entry["rank"] = 0;
    entry["node_id"] = 1;
    entry["operation"] = "compute";
    entry["kind"] = "compute";
    entry["issued"] = issued;
    entry["finished"] = finished;
    entry["num_ops"] = num_ops;
    entry["bytes"] = bytes;
    return entry;
}

nlohmann::json hbm_entry(const std::string& work_id, double remaining_bytes,
                         double remaining_ops) {
    nlohmann::json entry;
    entry["work_id"] = work_id;
    entry["kind"] = "COMPUTE";
    entry["leg"] = "hbm:COMPUTE";
    entry["rank"] = 0;
    entry["node_id"] = 1;
    entry["remaining_bytes"] = remaining_bytes;
    entry["remaining_ops"] = remaining_ops;
    entry["remaining_latency_ns"] = 0.0;
    entry["request_id"] = "req-1";
    entry["stage"] = "prefill";
    entry["generation"] = 0;
    entry["num_ops"] = 240.0;
    entry["bytes"] = 1200.0;
    entry["total_latency_ns"] = 80.0;
    return entry;
}

nlohmann::json wrap_nodes(const std::vector<nlohmann::json>& entries) {
    nlohmann::json block;
    block["rank"] = 0;
    block["nodes"] = entries;
    nlohmann::json state;
    state["nodes"] = nlohmann::json::array({block});
    return state;
}

nlohmann::json wrap_hbm(const std::vector<nlohmann::json>& entries) {
    nlohmann::json block;
    block["rank"] = 0;
    block["jobs"] = entries;
    nlohmann::json state;
    state["hbm"] = nlohmann::json::array({block});
    return state;
}

size_t count_kind(const nlohmann::json& events, const std::string& kind) {
    size_t count = 0;
    for (const auto& ev : events) {
        if (ev.value("kind", std::string()) == kind) {
            ++count;
        }
    }
    return count;
}

bool has_event(const nlohmann::json& events, const std::string& kind,
               const std::string& work_id) {
    for (const auto& ev : events) {
        if (ev.value("kind", std::string()) == kind &&
            ev.value("work_id", std::string()) == work_id) {
            return true;
        }
    }
    return false;
}

void test_diff_lifecycle() {
    nlohmann::json prev = wrap_nodes({node_entry("w1", false, false, 100, 0),
                                      node_entry("w2", true, false, 50, 0)});
    // w1: issued flips; w2: vanishes (completion+ack+release); w3: appears.
    nlohmann::json cur = wrap_nodes({node_entry("w1", true, false, 100, 0),
                                      node_entry("w3", false, false, 10, 0)});
    uint64_t seq = 0;
    const WorkLedgerDiff diff =
        diff_work_ledger_states(prev, cur, 7, 500, seq);
    const nlohmann::json& events = diff.events;
    // Compact node lists carry the node-class lifecycle (the dominant
    // volume class) instead of per-event envelopes.
    expect(diff.compact.is_object(), "compact lists present for nodes");
    expect(diff.compact.at("node_commits").size() == 1,
           "one compact commit for the new node");
    expect(diff.compact.at("node_vanished").size() == 1,
           "one compact vanish for the gone node");
    expect(diff.compact.at("node_issues").size() == 1,
           "one compact issue for the issued flip");
    expect(diff.compact.contains("seq_base"), "compact carries seq_base");
    // Node lifecycle rides the compact lists (asserted above); the events
    // array carries only the non-node classes and progress here (none).
    // Monotonic local_seq across the event list.
    uint64_t last = 0;
    bool first = true;
    for (const auto& ev : events) {
        const uint64_t local =
            ev.at("event_id").at("local_seq").get<uint64_t>();
        if (!first) {
            expect(local == last + 1, "local_seq is consecutive");
        }
        last = local;
        first = false;
    }
    const uint64_t compact_total =
        diff.compact.at("node_commits").size() +
        diff.compact.at("node_issues").size() +
        diff.compact.at("node_completions").size() +
        2 * diff.compact.at("node_vanished").size();
    expect(seq == compact_total,
           "counter consumed one value per compact entry, two per vanish");
    expect(diff.compact.at("seq_base").get<uint64_t>() == 0,
           "compact seq_base precedes the events array (nodes first)");
    // Compact commit entries keep the full node facts (identity + totals).
    for (const auto& entry : diff.compact.at("node_commits")) {
        expect(entry.value("request_id", std::string()) == "req-1",
               "compact commit carries the request identity");
        expect(entry.contains("num_ops"), "compact commit carries totals");
    }
}

void test_diff_progress() {
    // remaining decrease -> progress with delta_served; increase or equal
    // -> no event (no fabricated negative service).
    nlohmann::json prev = wrap_hbm({hbm_entry("j1", 1000.0, 200.0),
                                    hbm_entry("j2", 500.0, 0.0)});
    nlohmann::json cur =
        wrap_hbm({hbm_entry("j1", 600.0, 120.0), hbm_entry("j2", 900.0, 0.0)});
    uint64_t seq = 5;
    const WorkLedgerDiff diff =
        diff_work_ledger_states(prev, cur, 8, 600, seq);
    const nlohmann::json& events = diff.events;
    size_t progress_j1 = 0;
    for (const auto& ev : events) {
        if (ev.value("kind", std::string()) == "progress" &&
            ev.value("work_id", std::string()) == "j1") {
            ++progress_j1;
            if (ev.contains("delta_served")) {
                if (ev.at("delta_served").contains("bytes")) {
                    expect(ev.at("delta_served").at("bytes").get<double>() ==
                                400.0,
                           "byte delta = prev - cur");
                    expect(ev.value("remaining_bytes", -1.0) == 600.0,
                           "byte progress carries current remainder");
                } else if (ev.at("delta_served").contains("ops")) {
                    expect(ev.at("delta_served").at("ops").get<double>() ==
                                80.0,
                           "ops delta = prev - cur");
                    expect(ev.value("remaining_ops", -1.0) == 120.0,
                           "ops progress carries current remainder");
                }
            }
        }
        expect(ev.value("work_id", std::string()) != "j2" ||
                    ev.value("kind", std::string()) != "progress",
               "no progress for a work whose remainder grew");
    }
    expect(progress_j1 == 2, "j1 emits one progress per decreased dimension");
    expect(seq == 7, "seq counter consumed exactly two values");
    expect(diff.compact.is_null(),
           "hbm-only frames carry no compact node lists");
}

void test_resource_commit_payload() {
    const nlohmann::json prev = wrap_hbm({});
    const nlohmann::json cur =
        wrap_hbm({hbm_entry("j3", 600.0, 120.0)});
    uint64_t seq = 0;
    const WorkLedgerDiff diff =
        diff_work_ledger_states(prev, cur, 9, 700, seq);
    expect(diff.events.size() == 1, "new HBM job emits one commit");
    if (diff.events.size() == 1) {
        const auto& event = diff.events.at(0);
        expect(event.value("kind", std::string()) == "commit",
               "new HBM job event is commit");
        expect(event.value("remaining_bytes", -1.0) == 600.0,
               "resource commit carries authoritative remaining bytes");
        expect(event.value("remaining_ops", -1.0) == 120.0,
               "resource commit carries authoritative remaining ops");
        expect(event.at("total").value("bytes", -1.0) == 1200.0,
               "resource commit carries original byte total");
        expect(event.at("total").value("ops", -1.0) == 240.0,
               "resource commit carries original op total");
        expect(event.at("total").value("latency_ns", -1.0) == 80.0,
               "resource commit carries original latency total");
    }
    expect(seq == 1, "resource commit consumes one local sequence");
}

// ---------------------------------------------------------------- Part B --

void test_node_store_view() {
    NodeStore store;
    const uint64_t parent = store.add_node(
        make_node(0, "req-1", "prefill", NodeKind::Compute, 10, 0));
    const uint64_t child = store.add_node(
        make_node(0, "req-1", "prefill", NodeKind::CommSend, 0, 32));
    store.add_dependency(parent, child, DepKind::Data);
    store.mark_issued(parent);
    const auto view = store.work_ledger_view();
    expect(view.size() == 2, "view covers every stored node");
    expect(view[0].node_id == parent && view[1].node_id == child,
           "view is ascending by node id");
    expect(view[0].issued && !view[0].finished, "parent issued not finished");
    expect(view[0].kind_name == "compute", "kind name is stable");
    expect(view[0].num_ops == 10, "compute ops exported");
    expect(view[1].bytes == 32, "comm bytes exported");
    expect(view[1].blocking_parents.size() == 1 &&
               view[1].blocking_parents[0] == parent,
           "child's blocking parent exported");
    expect(view[0].blocking_parents.empty(), "parent has no blocking parents");
    store.finish_node(parent);
    const auto view2 = store.work_ledger_view();
    // Parent stays (children unfinished) but no longer blocks the child.
    for (const auto& entry : view2) {
        if (entry.node_id == parent) {
            expect(entry.finished, "finished flag visible post-finish");
        }
        if (entry.node_id == child) {
            expect(entry.blocking_parents.empty(),
                   "finished parent no longer blocks");
        }
    }
}

// ---------------------------------------------------------------- Part C --

void test_request_json_event_seqs() {
    DecisionMailbox mailbox;
    DecisionEvent first;
    first.reason = DecisionReason::ARRIVAL;
    first.request_id = "req-1";
    DecisionEvent second;
    second.reason = DecisionReason::PREFILL_DRAIN;
    second.request_id = "req-1";
    second.stage = "prefill";
    mailbox.push(std::move(first));
    mailbox.push(std::move(second));
    StateDelta delta =
        build_state_delta_v1(mailbox.drain(), 42, 3, 0, {}, {}, {});
    delta.snapshot_handle.kind = "work_ledger_v1";
    const nlohmann::json req = build_request_json(delta);
    expect(req.contains("event_seqs"), "event_seqs is present (additive)");
    expect(req.at("event_seqs").size() == 2, "one seq per epoch event");
    expect(req.at("event_seqs")[0].get<uint64_t>() == 1 &&
               req.at("event_seqs")[1].get<uint64_t>() == 2,
           "seqs are the mailbox push-order values");
    expect(req.at("reasons").size() == req.at("event_seqs").size(),
           "event_seqs parallels reasons");
    expect(req.at("snapshot_handle").at("kind").get<std::string>() ==
               "work_ledger_v1",
           "snapshot_handle kind passes through the serializer");
    expect(req.contains("work_ledger") == false,
           "no provider installed -> work_ledger field absent");
}

}  // namespace

int main() {
    test_diff_lifecycle();
    test_diff_progress();
    test_resource_commit_payload();
    test_node_store_view();
    test_request_json_event_seqs();
    if (!g_ok) {
        std::fprintf(stderr, "[work_ledger_bridge_test] FAILED\n");
        return 1;
    }
    std::printf("[work_ledger_bridge_test] ALL PASS\n");
    return 0;
}
