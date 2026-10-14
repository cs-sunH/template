/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

parsed_graph_batch_test.cc -- C1 (2026-08-29) ParsedGraphBatch fixture.

Covers the one-shot structural parse of one GraphBatch response
(parse_graph_batch, ParsedGraphBatch.hh) across the full malformed-protocol
matrix of the frozen rule families:

  T1-T8   top level (schema_version, source_delivery_sequence, batch_id,
          error, the six optional arrays, unknown keys, array typing,
          touched_ranks);
  N1-N15  node rules (exact 13-key set, field typing/domains, compute/comm/
          coll sub-object key sets, integer overflow, negatives,
          tag/priority <= INT_MAX, hbm_charge/involved_dim shapes);
  E1-E4   parent-edge rules (exact key set, rank domain, kind == "data",
          no self-loop);
  W1-W4   watch rules (exact key set, identity fields, decimal member rank
          keys with integer ids, Success/Skipped statuses);
  A1-A3   alarm rules (exact key set, envelope key set, non-empty ids,
          non-negative numerics, queue_index >= -1);
  S1/S2   half-opaque assignments/kv_actions (required structural fields,
          unknown sibling keys tolerated);
  O1      array-order preservation (nodes/watches/alarms keep the emission
          order; watch members keep the rank-string key order; duplicate
          statuses preserved in raw order);
  O3      unicode passthrough (no validation, byte-exact std::string);
  O4      the frozen error skeleton parses cleanly and carries the error;
  O5      no size cap; >UINT64 integer literals fail closed via the
          typed getters (nlohmann stores them as double -> not
          is_number_integer).

Positive controls: a max-shape response (all 7 node types, every optional
field, a cross-batch parent edge, a full watch, an alarm with queue_index,
touched_ranks) parses with every typed field asserted equal; num_ranks = -1
disables the rank-domain checks (legacy-fixture contract).

Build: the CMake target AstraSim_Analytical_Congestion_Aware_ParsedBatchTest.
Run: build/astra_analytical/build_congestion_aware/bin/\
     AstraSim_Analytical_Congestion_Aware_ParsedBatchTest
Exit code 0 on ALL PASS.
*******************************************************************************/

#include <cstdio>
#include <limits>
#include <string>

#include <json/json.hpp>

#include "DieTypes.hh"
#include "astra-sim/workload/execution_driven/ParsedGraphBatch.hh"

using namespace AstraSim::ExecutionDriven;
using AstraSim::WorkspaceRankState;

namespace {

bool g_ok = true;

void expect(bool cond, const char* what) {
    if (!cond) {
        std::fprintf(stderr, "[parsed_graph_batch_test] FAIL: %s\n", what);
        g_ok = false;
    }
}

// Every malformed response must throw ParseError whose message localizes
// the violation (entry index + field name are part of the contract).
void expect_parse_error(const nlohmann::json& resp, const char* what,
                        const char* needle = nullptr) {
    try {
        parse_graph_batch(resp, 3);
        std::fprintf(stderr,
                     "[parsed_graph_batch_test] unexpected parse PASS: %s\n",
                     what);
        g_ok = false;
    } catch (const ParseError& exc) {
        if (needle != nullptr &&
            std::string(exc.what()).find(needle) == std::string::npos) {
            std::fprintf(stderr,
                         "[parsed_graph_batch_test] message mismatch for "
                         "%s:\n  got: %s\n  want substring: %s\n",
                         what, exc.what(), needle);
            g_ok = false;
        }
    } catch (const std::exception& exc) {
        std::fprintf(stderr,
                     "[parsed_graph_batch_test] wrong exception type for %s: "
                     "%s\n",
                     what, exc.what());
        g_ok = false;
    }
}

void expect_v2_parse_error(
    const nlohmann::json& resp,
    const std::vector<WorkspaceRankState>& workspace_state,
    const char* what, const char* needle = nullptr) {
    try {
        (void)parse_graph_batch(resp, 2, 2, workspace_state);
        std::fprintf(stderr,
                     "[parsed_graph_batch_test] unexpected v2 parse PASS: "
                     "%s\n",
                     what);
        g_ok = false;
    } catch (const ParseError& exc) {
        if (needle != nullptr &&
            std::string(exc.what()).find(needle) == std::string::npos) {
            std::fprintf(stderr,
                         "[parsed_graph_batch_test] v2 message mismatch for "
                         "%s:\n  got: %s\n  want substring: %s\n",
                         what, exc.what(), needle);
            g_ok = false;
        }
    } catch (const std::exception& exc) {
        std::fprintf(stderr,
                     "[parsed_graph_batch_test] wrong v2 exception type for "
                     "%s: %s\n",
                     what, exc.what());
        g_ok = false;
    }
}

ParsedGraphBatch parse_ok(const nlohmann::json& resp, const char* what,
                          int num_ranks = 3) {
    try {
        return parse_graph_batch(resp, num_ranks);
    } catch (const std::exception& exc) {
        std::fprintf(stderr,
                     "[parsed_graph_batch_test] unexpected parse error for "
                     "%s: %s\n",
                     what, exc.what());
        g_ok = false;
        return ParsedGraphBatch();
    }
}

// ---------------------------------------------------------------- builders --

nlohmann::json comm_full() {
    return {{"bytes", 128}, {"src", 0}, {"dst", 1}, {"tag", 7},
            {"hbm_charge", false}};
}

nlohmann::json coll_full() {
    return {{"comm_type", 2}, {"bytes", 64}, {"priority", 3},
            {"pg_name", "tp0"},
            {"involved_dim", nlohmann::json::array({true, false})}};
}

nlohmann::json compute_full() {
    return {{"num_ops", 5000000000ULL},      // > 2^32: uint64 must survive
            {"tensor_size", 6000000000ULL},  // (the int-default truncation
            {"runtime_ns", 12345},           //  NOTE of the old DOM path)
            {"remote_weight_bytes", 777},
            {"hbm_access_mode", 1}};
}


// Stage B (die backend, contract §2): a legal die_op_spec object. Unsigned
// literals on purpose (the strict spec parser requires is_number_unsigned,
// which nlohmann text parsing always provides; in-test construction with
// plain ints would store signed and be rejected).
nlohmann::json valid_die_op_spec() {
    nlohmann::json span;
    span["request_id"] = "r1";
    span["q_tokens"] = 8u;
    span["kv_tokens"] = 128u;
    span["q_start_pos"] = 0u;
    span["kv_start_pos"] = 0u;
    nlohmann::json spec;
    spec["schema_version"] = 1u;
    spec["op_kind"] = "norm";
    spec["phase"] = "prefill";
    spec["dtype"] = "bf16";
    spec["model_dims"] = {{"hidden", 4096u},
                          {"ffn", 14336u},
                          {"vocab", 152064u},
                          {"mlp_variant", "swiglu"},
                          {"norm_type", "rmsnorm"},
                          {"position_encoding", "rope"},
                          {"num_q_heads", 32u},
                          {"num_kv_heads", 8u},
                          {"head_dim", 128u}};
    spec["tp_slice"] = {{"tensor_parallel", 1u},
                        {"tensor_parallel_rank", 0u},
                        {"q_heads_local", 32u},
                        {"kv_heads_local", 8u},
                        {"ffn_local", 14336u},
                        {"vocab_local", 152064u}};
    spec["spans"] = nlohmann::json::array({span});
    spec["layer_count"] = 1u;
    spec["weight_passes"] = 1u;
    spec["kv_access"] = {{"identity", "local"}};
    spec["fusion_policy"] = "auto";
    return spec;
}
nlohmann::json node_json(int rank, uint64_t id, int type,
                         const std::string& name) {
    return {{"rank", rank},
            {"id", id},
            {"type", type},
            {"name", name},
            {"is_cpu_op", false},
            {"is_timer_op", false},
            {"inputs_values", ""},
            {"request_id", "r1"},
            {"stage", "prefill"},
            {"generation", 0},
            {"compute", compute_full()},
            {"comm", comm_full()},
            {"coll", coll_full()}};
}

// The max-shape legal response: every node type, every optional field, a
// cross-batch parent edge (from references nothing in this batch), a full
// watch, an alarm carrying queue_index, touched_ranks.
nlohmann::json max_shape_response() {
    nlohmann::json resp;
    resp["schema_version"] = 1;
    resp["batch_id"] = 42;
    resp["source_delivery_sequence"] = 42;
    nlohmann::json nodes = nlohmann::json::array();
    for (int type = 1; type <= 7; ++type) {
        nlohmann::json node = node_json(0, static_cast<uint64_t>(type - 1),
                                        type, "n" + std::to_string(type));
        if (type == 5) {
            node["name"] = "send";
        } else if (type == 6) {
            node["name"] = "recv";
        }
        if (type == 5 || type == 6) {
            // src/dst 0 == this node's rank: endpoint ownership (src == rank
            // for 5 / dst == rank for 6) is a PREFLIGHT rule; parse must not
            // duplicate it.
            node["comm"] = {{"bytes", 100},
                            {"src", 0},
                            {"dst", 0},
                            {"tag", 9},
                            {"hbm_charge", false}};
        }
        // sh_2.0-specific optional node key: exercise the true branch on the
        // type-2 MEM_LOAD node (the only producer that emits it).
        if (type == 2) {
            node["is_local_hbm_kv_restore"] = true;
        }
        nodes.push_back(std::move(node));
    }
    // type-7 collective: bytes > 0 is a preflight rule, not parse's.
    resp["nodes"] = std::move(nodes);
    resp["parent_edges"] = nlohmann::json::array(
        {{{"rank", 0}, {"kind", "data"}, {"from", 999}, {"to", 6}}});
    resp["watches"] = nlohmann::json::array(
        {{{"request_id", "r1"},
          {"stage", "prefill"},
          {"generation", 0},
          {"members", {{"0", 6}}},
          {"statuses", nlohmann::json::array({"Success", "Skipped",
                                              "Success"})}}});
    resp["assignments"] = nlohmann::json::array(
        {{{"request_id", "r1"},
          {"prefill_instance_index", 0},
          {"decode_instance_index", 1},
          {"prefill_assignment_key", nlohmann::json::array({"i0", "i1"})}}});
    resp["kv_actions"] = nlohmann::json::array(
        {{{"event_type", "admit"},
          {"trigger_request_id", "r1"},
          {"session_id", "s1"},
          {"context_tokens", 5}}});
    resp["future_alarms"] = nlohmann::json::array(
        {{{"arrival_world_ns", 200000},
          {"envelope",
           {{"request_id", "r2"},
            {"session_id", "s1"},
            {"turn_index", 1},
            {"prefill_length", 100},
            {"decode_length", 10},
            {"inter_request_interval_ns", 20000000},
            {"queue_index", 3}}}}});
    resp["touched_ranks"] = nlohmann::json::array({0});
    return resp;
}

// --------------------------------------------------------------- Part T ----
void test_top_level_rules() {
    const nlohmann::json base = max_shape_response();

    {  // T1: missing schema_version
        nlohmann::json resp = base;
        resp.erase("schema_version");
        expect_parse_error(resp, "T1: missing schema_version",
                           "schema_version");
    }
    {  // T1: wrong value
        nlohmann::json resp = base;
        resp["schema_version"] = 2;
        expect_parse_error(resp, "T1: schema_version != 1", "schema_version");
    }
    {  // T1: wrong type
        nlohmann::json resp = base;
        resp["schema_version"] = "1";
        expect_parse_error(resp, "T1: schema_version not integer",
                           "schema_version");
    }
    {  // T2: missing source_delivery_sequence
        nlohmann::json resp = base;
        resp.erase("source_delivery_sequence");
        expect_parse_error(resp, "T2: missing source_delivery_sequence",
                           "source_delivery_sequence");
    }
    {  // T4: batch_id is optional (default 0) and must be a uint when set
        nlohmann::json resp = base;
        resp.erase("batch_id");
        const ParsedGraphBatch parsed = parse_ok(resp, "T4: batch_id absent");
        expect(parsed.batch_id == 0, "T4: absent batch_id defaults to 0");
        resp["batch_id"] = -1;
        expect_parse_error(resp, "T4: negative batch_id", "batch_id");
        resp["batch_id"] = 1.5;
        expect_parse_error(resp, "T4: float batch_id", "batch_id");
    }
    {  // T5: all six arrays optional == empty
        nlohmann::json resp;
        resp["schema_version"] = 1;
        resp["source_delivery_sequence"] = 7;
        const ParsedGraphBatch parsed =
            parse_ok(resp, "T5: bare-bones response");
        expect(parsed.nodes.empty() && parsed.parent_edges.empty() &&
                   parsed.watches.empty() && parsed.assignments.empty() &&
                   parsed.kv_actions.empty() && parsed.future_alarms.empty(),
               "T5: absent arrays parse as empty");
        expect(!parsed.has_touched_ranks && parsed.touched_ranks.empty(),
               "T5: absent touched_ranks -> has_touched_ranks false");
        expect(parsed.source_delivery_sequence == 7,
               "T5: source_delivery_sequence parsed");
    }
    {  // T6: unknown top-level key (the pre-C1 echo fixtures used one)
        nlohmann::json resp = base;
        resp["echo_of_request"] = nlohmann::json::object();
        expect_parse_error(resp, "T6: unknown top-level key",
                           "unknown field \"echo_of_request\"");
    }
    {  // T7: present arrays must be arrays
        nlohmann::json resp = base;
        resp["nodes"] = nlohmann::json::object();
        expect_parse_error(resp, "T7: nodes not an array", "not an array");
        resp = base;
        resp["watches"] = "junk";
        expect_parse_error(resp, "T7: watches not an array", "not an array");
    }
    {  // T0: the document itself must be an object
        expect_parse_error(nlohmann::json::array(), "T0: array document",
                           "not an object");
        expect_parse_error(nlohmann::json(7), "T0: scalar document",
                           "not an object");
    }
    {  // T8: touched_ranks typing and domain
        nlohmann::json resp = base;
        resp["touched_ranks"] = nlohmann::json::object();
        expect_parse_error(resp, "T8: touched_ranks not an array",
                           "touched_ranks");
        resp["touched_ranks"] = nlohmann::json::array({0, 1.5});
        expect_parse_error(resp, "T8: touched_ranks float entry",
                           "touched_ranks");
        resp["touched_ranks"] = nlohmann::json::array({0, 99});
        expect_parse_error(resp, "T8: touched_ranks rank out of range",
                           "out of range");
    }
    {  // T8/positive: sorted-unique/equality is validate's rule, not parse's
        nlohmann::json resp = base;
        resp["touched_ranks"] = nlohmann::json::array({2, 0, 2});
        const ParsedGraphBatch parsed =
            parse_ok(resp, "T8: unsorted duplicates reach validate");
        expect(parsed.touched_ranks == std::vector<int>({2, 0, 2}) &&
                   parsed.has_touched_ranks,
               "T8: touched_ranks parsed verbatim (order preserved)");
    }
}

void test_workspace_v2_snapshot_rules() {
    const std::vector<WorkspaceRankState> request_state{
        {0, 3, 128}, {1, 8, 0}};
    nlohmann::json response = {
        {"schema_version", 2},
        {"batch_id", 42},
        {"source_delivery_sequence", 42},
        {"workspace_capacity_snapshot",
         {{"protocol_version", 1},
          {"ranks",
           nlohmann::json::array(
               {{{"rank", 0},
                 {"workspace_revision", 3},
                 {"capacity_bytes", 1024},
                 {"model_weight_bytes", 128},
                 {"resident_kv_bytes", 64},
                 {"reserved_request_bytes", 32}},
                {{"rank", 1},
                 {"workspace_revision", 8},
                 {"capacity_bytes", 2048},
                 {"model_weight_bytes", 256},
                 {"resident_kv_bytes", 512},
                 {"reserved_request_bytes", 128}}})}}}};

    ParsedGraphBatch parsed;
    try {
        parsed = parse_graph_batch(response, 2, 2, request_state);
    } catch (const std::exception& exc) {
        std::fprintf(stderr,
                     "[parsed_graph_batch_test] valid v2 snapshot rejected: "
                     "%s\n",
                     exc.what());
        g_ok = false;
    }
    expect(parsed.schema_version == 2 &&
               parsed.source_delivery_sequence == 42 &&
               parsed.has_workspace_capacity_snapshot &&
               parsed.workspace_capacity_snapshot.size() == 2 &&
               parsed.workspace_capacity_snapshot[0].rank == 0 &&
               parsed.workspace_capacity_snapshot[0].workspace_revision == 3 &&
               parsed.workspace_capacity_snapshot[0].capacity_bytes == 1024 &&
               parsed.workspace_capacity_snapshot[1].rank == 1 &&
               parsed.workspace_capacity_snapshot[1].resident_kv_bytes == 512,
           "v2 snapshot parses the full rank/revision/capacity table");

    auto malformed = response;
    malformed.erase("workspace_capacity_snapshot");
    expect_v2_parse_error(malformed, request_state,
                          "missing v2 workspace snapshot",
                          "workspace_capacity_snapshot");

    malformed = response;
    malformed["delivery_sequence"] = 42;
    expect_v2_parse_error(malformed, request_state,
                          "response must not repeat delivery sequence",
                          "unknown field \"delivery_sequence\"");

    malformed = response;
    malformed["workspace_capacity_snapshot"]["ranks"][0]
             ["workspace_revision"] = 4;
    expect_v2_parse_error(malformed, request_state,
                          "snapshot revision echoes the request view",
                          "does not echo request view");

    malformed = response;
    malformed["workspace_capacity_snapshot"]["ranks"][0]["rank"] = 1;
    expect_v2_parse_error(malformed, request_state,
                          "snapshot rank table is complete and ascending",
                          "complete, unique, and ascending");

    malformed = response;
    malformed["workspace_capacity_snapshot"]["ranks"][0]
             ["capacity_bytes"] = 1024.0;
    expect_v2_parse_error(malformed, request_state,
                          "snapshot byte fields reject floating point",
                          "not an integer");

    malformed = response;
    malformed["workspace_capacity_snapshot"]["ranks"][1]
             ["workspace_revision"] = true;
    expect_v2_parse_error(malformed, request_state,
                          "snapshot revision rejects bool",
                          "not an integer");

    malformed = response;
    malformed["workspace_capacity_snapshot"]["ranks"][0]
             ["model_weight_bytes"] = std::numeric_limits<uint64_t>::max();
    malformed["workspace_capacity_snapshot"]["ranks"][0]
             ["resident_kv_bytes"] = 1;
    expect_v2_parse_error(malformed, request_state,
                          "snapshot base sum must not overflow",
                          "base-used byte sum overflows");
}

// --------------------------------------------------------------- Part N ----
void test_node_rules() {
    const nlohmann::json base = max_shape_response();

    {  // N1: non-object node
        nlohmann::json resp = base;
        resp["nodes"][0] = "junk";
        expect_parse_error(resp, "N1: node not an object", "node[0]");
    }
    {  // N2: unknown node key
        nlohmann::json resp = base;
        resp["nodes"][0]["mem"] = nlohmann::json::object();
        expect_parse_error(resp, "N2: unknown node key",
                           "unknown field \"mem\"");
    }
    {  // N2: missing node key
        nlohmann::json resp = base;
        resp["nodes"][0].erase("inputs_values");
        expect_parse_error(resp, "N2: missing node key",
                           "missing field \"inputs_values\"");
    }
    {  // N3: rank domain
        nlohmann::json resp = base;
        resp["nodes"][0]["rank"] = 99;
        expect_parse_error(resp, "N3: rank out of range", "out of range");
        resp["nodes"][0]["rank"] = -1;
        expect_parse_error(resp, "N3: negative rank rejected by the domain",
                           "out of range");
        resp["nodes"][0]["rank"] = 0.5;
        expect_parse_error(resp, "N3: float rank", "not an integer");
    }
    {  // N3/positive: num_ranks = -1 disables the domain checks
        nlohmann::json resp = base;
        resp["nodes"][0]["rank"] = 99;
        const ParsedGraphBatch parsed =
            parse_ok(resp, "N3: num_ranks -1 disables domain", -1);
        expect(parsed.nodes[0].node.rank == 99,
               "N3: num_ranks -1 tolerates rank 99");
    }
    {  // N4: id domain
        nlohmann::json resp = base;
        resp["nodes"][0]["id"] = -1;
        expect_parse_error(resp, "N4: negative id", "negative");
        // A near-max id (18446744073709551515 == UINT64_MAX-100) is a legal
        // PARSE value (the committer's missing/invalid sentinel downstream
        // is UINT64_MAX itself, which this value deliberately avoids --
        // unchanged semantics).
        resp["nodes"][0]["id"] = 18446744073709551515ULL;
        const ParsedGraphBatch parsed =
            parse_ok(resp, "N4: near-max id (UINT64_MAX-100) parses");
        expect(parsed.nodes[0].json_id == 18446744073709551515ULL,
               "N4: near-max id (UINT64_MAX-100) round-trips");
    }
    {  // N5: type domain
        nlohmann::json resp = base;
        resp["nodes"][0]["type"] = 9;
        expect_parse_error(resp, "N5: type out of range", "out of range");
        resp["nodes"][0]["type"] = 0;
        expect_parse_error(resp, "N5: type 0 out of range", "out of range");
        resp["nodes"][0]["type"] = 4.0;
        expect_parse_error(resp, "N5: float type", "not an integer");
    }
    {  // N6: name
        nlohmann::json resp = base;
        resp["nodes"][0]["name"] = "";
        expect_parse_error(resp, "N6: empty name", "empty name");
        resp["nodes"][0]["name"] = 7;
        expect_parse_error(resp, "N6: name not a string", "not a string");
    }
    {  // N7: flags must be booleans
        nlohmann::json resp = base;
        resp["nodes"][0]["is_cpu_op"] = 0;
        expect_parse_error(resp, "N7: is_cpu_op int", "not a boolean");
        resp["nodes"][0]["is_cpu_op"] = false;
        resp["nodes"][0]["is_timer_op"] = "false";
        expect_parse_error(resp, "N7: is_timer_op string", "not a boolean");
    }
    {  // N8: inputs_values must be a string
        nlohmann::json resp = base;
        resp["nodes"][0]["inputs_values"] = 3;
        expect_parse_error(resp, "N8: inputs_values int", "not a string");
    }
    {  // N9: request_id non-empty
        nlohmann::json resp = base;
        resp["nodes"][0]["request_id"] = "";
        expect_parse_error(resp, "N9: empty request_id", "request_id");
    }
    {  // N10: stage domain
        nlohmann::json resp = base;
        resp["nodes"][0]["stage"] = "chat";
        expect_parse_error(resp, "N10: invalid stage", "invalid stage");
    }
    {  // N11: generation == stage
        nlohmann::json resp = base;
        resp["nodes"][0]["generation"] = 1;  // prefill must carry 0
        expect_parse_error(resp, "N11: generation mismatch",
                           "does not match stage");
    }
    {  // N12: compute key set / domains
        nlohmann::json resp = base;
        resp["nodes"][0]["compute"].erase("num_ops");
        expect_parse_error(resp, "N12: compute missing num_ops", "num_ops");
        resp = base;
        resp["nodes"][0]["compute"]["flops"] = 1;
        expect_parse_error(resp, "N12: compute unknown key", "flops");
        resp = base;
        resp["nodes"][0]["compute"]["tensor_size"] = -1;
        expect_parse_error(resp, "N12: negative tensor_size", "negative");
        resp = base;
        // sh_2.0 posture: hbm_access_mode is in the allowed compute key set
        // (an integer, parsed into the TOP-LEVEL OnlineNode field); only a
        // malformed value (non-integer) fails closed here.
        resp["nodes"][0]["compute"]["hbm_access_mode"] = "read";
        expect_parse_error(resp, "N12: hbm_access_mode string",
                           "hbm_access_mode");
    }
    {  // N12/die (stage B, contract §2): op_role + die_op_spec extensions.
        using OpRole = AstraSim::ExecutionDriven::OpRole;
        // Accept: op_role=control_marker parses to the enum.
        {
            nlohmann::json resp = base;
            resp["nodes"][0]["compute"]["op_role"] = "control_marker";
            const ParsedGraphBatch parsed =
                parse_ok(resp, "N12: op_role control_marker accepted");
            expect(parsed.nodes[0].node.compute.op_role ==
                       OpRole::ControlMarker,
                   "N12: op_role control_marker -> enum");
            expect(parsed.nodes[0].node.compute.die_spec == nullptr,
                   "N12: op_role alone leaves die_spec null");
        }
        // Accept: op_role=compute is the explicit default spelling.
        {
            nlohmann::json resp = base;
            resp["nodes"][0]["compute"]["op_role"] = "compute";
            const ParsedGraphBatch parsed =
                parse_ok(resp, "N12: op_role compute accepted");
            expect(parsed.nodes[0].node.compute.op_role == OpRole::Compute,
                   "N12: op_role compute -> enum");
        }
        // Accept: die_op_spec with runtime_ns == 0 parses the immutable
        // spec (DieModel::parse_die_op_spec ran without error).
        {
            nlohmann::json resp = base;
            resp["nodes"][0]["compute"]["runtime_ns"] = 0;
            resp["nodes"][0]["compute"]["die_op_spec"] = valid_die_op_spec();
            const ParsedGraphBatch parsed =
                parse_ok(resp, "N12: die_op_spec accepted");
            expect(parsed.nodes[0].node.compute.die_spec != nullptr,
                   "N12: die_op_spec -> shared spec");
            expect(parsed.nodes[0].node.compute.die_spec->op_kind ==
                       DieModel::OpKind::Norm,
                   "N12: parsed spec carries op_kind norm");
        }
        // Reject: unknown op_role spelling.
        {
            nlohmann::json resp = base;
            resp["nodes"][0]["compute"]["op_role"] = "marker";
            expect_parse_error(resp, "N12: op_role unknown spelling",
                               "op_role");
        }
        // Reject: op_role must be a string.
        {
            nlohmann::json resp = base;
            resp["nodes"][0]["compute"]["op_role"] = 1;
            expect_parse_error(resp, "N12: op_role int", "not a string");
        }
        // Reject: die_op_spec + runtime_ns != 0 (contract §2 conflict rule:
        // a calibrated runtime must never silently bypass the die plan).
        {
            nlohmann::json resp = base;  // compute_full carries runtime_ns
            resp["nodes"][0]["compute"]["die_op_spec"] = valid_die_op_spec();
            expect_parse_error(resp, "N12: die_op_spec with runtime_ns",
                               "runtime_ns == 0");
        }
        // Reject: die_op_spec must be an object.
        {
            nlohmann::json resp = base;
            resp["nodes"][0]["compute"]["runtime_ns"] = 0;
            resp["nodes"][0]["compute"]["die_op_spec"] = "norm";
            expect_parse_error(resp, "N12: die_op_spec string",
                               "die_op_spec is not an object");
        }
        // Reject: an inner spec violation surfaces as a ParseError carrying
        // the original library message (here: an unknown key inside the
        // spec object).
        {
            nlohmann::json resp = base;
            resp["nodes"][0]["compute"]["runtime_ns"] = 0;
            nlohmann::json spec = valid_die_op_spec();
            spec["surprise"] = 1;
            resp["nodes"][0]["compute"]["die_op_spec"] = spec;
            expect_parse_error(resp, "N12: die_op_spec unknown inner key",
                               "die_op_spec rejected");
        }
        // Reject: a missing required key inside the spec object (spans).
        {
            nlohmann::json resp = base;
            resp["nodes"][0]["compute"]["runtime_ns"] = 0;
            nlohmann::json spec = valid_die_op_spec();
            spec.erase("spans");
            resp["nodes"][0]["compute"]["die_op_spec"] = spec;
            expect_parse_error(resp, "N12: die_op_spec missing spans",
                               "die_op_spec rejected");
        }
        // Reject: an enum misspelling inside the spec object.
        {
            nlohmann::json resp = base;
            resp["nodes"][0]["compute"]["runtime_ns"] = 0;
            nlohmann::json spec = valid_die_op_spec();
            spec["op_kind"] = "normalize";
            resp["nodes"][0]["compute"]["die_op_spec"] = spec;
            expect_parse_error(resp, "N12: die_op_spec bad op_kind",
                               "die_op_spec rejected");
        }
        // Reject: a signed integer inside the spec object (the strict
        // spec parser requires unsigned storage).
        {
            nlohmann::json resp = base;
            resp["nodes"][0]["compute"]["runtime_ns"] = 0;
            nlohmann::json spec = valid_die_op_spec();
            spec["layer_count"] = -1;
            resp["nodes"][0]["compute"]["die_op_spec"] = spec;
            expect_parse_error(resp, "N12: die_op_spec negative layer_count",
                               "die_op_spec rejected");
        }
    }
    {  // N13: comm key set / domains
        nlohmann::json resp = base;
        resp["nodes"][0]["comm"].erase("tag");
        expect_parse_error(resp, "N13: comm missing tag", "tag");
        resp = base;
        resp["nodes"][0]["comm"]["extra"] = 1;
        expect_parse_error(resp, "N13: comm unknown key", "extra");
        resp = base;
        resp["nodes"][0]["comm"]["tag"] = 4294967296ULL;  // 2^32
        expect_parse_error(resp, "N13: tag > INT_MAX", "tag out of range");
        resp = base;
        // 2^31 is inside the old UINT32_MAX bound but truncates to a
        // negative int at the Sys::front_end_sim_send/recv `int` parameter;
        // the tightened parse gate must reject it too.
        resp["nodes"][0]["comm"]["tag"] = 2147483648ULL;  // 2^31
        expect_parse_error(resp, "N13: tag > INT_MAX", "tag out of range");
        resp = base;
        // [5e8, 2^31) is representable as int, but Sys folds every NATIVE
        // tag modulo 5e8 -- such a tag would silently alias another
        // stream's tag space; the parse-boundary ceiling gate must reject
        // it (boundary-E1's parse-layer landing point, restored 2026-10-01).
        resp["nodes"][0]["comm"]["tag"] = 500000000ULL;
        expect_parse_error(resp, "N13: tag >= 5e8 NATIVE ceiling",
                           "window ceiling");
        resp = base;
        // 5e8-1 is the last legal allocator-segment tag and must parse.
        resp["nodes"][0]["comm"]["tag"] = 499999999ULL;
        try {
            parse_graph_batch(resp, 3);
        } catch (const std::exception& exc) {
            std::fprintf(stderr,
                         "[parsed_graph_batch_test] tag 499999999 must "
                         "parse, got: %s\n", exc.what());
            g_ok = false;
        }
        resp = base;
        resp["nodes"][0]["comm"]["hbm_charge"] = 0;
        expect_parse_error(resp, "N13: hbm_charge int", "hbm_charge");
    }
    {  // N14: coll key set / domains
        nlohmann::json resp = base;
        resp["nodes"][0]["coll"].erase("pg_name");
        expect_parse_error(resp, "N14: coll missing pg_name", "pg_name");
        resp = base;
        resp["nodes"][0]["coll"]["root"] = 1;
        expect_parse_error(resp, "N14: coll unknown key", "root");
        resp = base;
        resp["nodes"][0]["coll"]["priority"] = 4294967296ULL;  // 2^32
        expect_parse_error(resp, "N14: priority > INT_MAX",
                           "priority out of range");
        resp = base;
        // 2^31 is inside the old UINT32_MAX bound but turns negative in the
        // int explicit_priority parameter of the Sys::generate_* ->
        // get_priority chain; the tightened parse gate must reject it too
        // (same family as the N13 tag rule).
        resp["nodes"][0]["coll"]["priority"] = 2147483648ULL;  // 2^31
        expect_parse_error(resp, "N14: priority > INT_MAX",
                           "priority out of range");
        resp = base;
        resp["nodes"][0]["coll"]["involved_dim"] =
            nlohmann::json::array({true, 1});
        expect_parse_error(resp, "N14: involved_dim non-bool element",
                           "involved_dim");
        resp = base;
        resp["nodes"][0]["coll"]["involved_dim"] = "xy";
        expect_parse_error(resp, "N14: involved_dim not array",
                           "involved_dim");
    }
    {  // N14/positive: involved_dim is optional (absent == empty)
        nlohmann::json resp = base;
        resp["nodes"][0]["coll"].erase("involved_dim");
        const ParsedGraphBatch parsed =
            parse_ok(resp, "N14: involved_dim absent");
        expect(parsed.nodes[0].node.coll.involved_dim.empty(),
               "N14: absent involved_dim -> empty vector");
    }
    {  // typed getters: >UINT64 integer literal becomes a double ->
       // fail closed
        nlohmann::json resp = base;
        resp["nodes"][0]["compute"]["num_ops"] =
            nlohmann::json::parse("18446744073709551616");  // 2^64
        expect_parse_error(resp, "num_ops > UINT64_MAX",
                           "not an integer");
    }
}

// --------------------------------------------------------------- Part E ----
void test_edge_rules() {
    const nlohmann::json base = max_shape_response();

    {  // E1: not an object / unknown key / missing key
        nlohmann::json resp = base;
        resp["parent_edges"][0] = 5;
        expect_parse_error(resp, "E1: edge not an object", "parent_edge[0]");
        resp = base;
        resp["parent_edges"][0]["weight"] = 1;
        expect_parse_error(resp, "E1: edge unknown key", "weight");
        resp = base;
        resp["parent_edges"][0].erase("kind");
        expect_parse_error(resp, "E1: edge missing kind", "kind");
    }
    {  // E2: rank domain / endpoint domains
        nlohmann::json resp = base;
        resp["parent_edges"][0]["rank"] = 3;
        expect_parse_error(resp, "E2: edge rank out of range", "out of range");
        resp = base;
        resp["parent_edges"][0]["from"] = -2;
        expect_parse_error(resp, "E2: negative from", "negative");
    }
    {  // E3: kind must be "data"
        nlohmann::json resp = base;
        resp["parent_edges"][0]["kind"] = "control";
        expect_parse_error(resp, "E3: unsupported kind", "kind");
    }
    {  // E4: self-loop
        nlohmann::json resp = base;
        resp["parent_edges"][0]["from"] = 6;  // == to
        expect_parse_error(resp, "E4: self-loop", "self-loop");
    }
}

// --------------------------------------------------------------- Part W ----
void test_watch_rules() {
    const nlohmann::json base = max_shape_response();

    {  // W1: shape
        nlohmann::json resp = base;
        resp["watches"][0] = "junk";
        expect_parse_error(resp, "W1: watch not an object", "watch[0]");
        resp = base;
        resp["watches"][0]["extra"] = 1;
        expect_parse_error(resp, "W1: watch unknown key", "extra");
        resp = base;
        resp["watches"][0].erase("statuses");
        expect_parse_error(resp, "W1: watch missing statuses", "statuses");
    }
    {  // W2: identity fields
        nlohmann::json resp = base;
        resp["watches"][0]["request_id"] = "";
        expect_parse_error(resp, "W2: empty watch request_id", "request_id");
        resp = base;
        resp["watches"][0]["stage"] = "other";
        expect_parse_error(resp, "W2: invalid watch stage", "invalid stage");
        resp = base;
        resp["watches"][0]["generation"] = 3;
        expect_parse_error(resp, "W2: watch generation mismatch",
                           "does not match stage");
    }
    {  // W3: member keys/values
        nlohmann::json resp = base;
        resp["watches"][0]["members"] = nlohmann::json::object();
        expect_parse_error(resp, "W3: empty members", "members");
        resp = base;
        resp["watches"][0]["members"] = {{"x", 6}};
        expect_parse_error(resp, "W3: non-decimal member key",
                           "unparsable member rank");
        resp = base;
        resp["watches"][0]["members"] = {{"1x", 6}};
        expect_parse_error(resp, "W3: trailing garbage in member key",
                           "unparsable member rank");
        resp = base;
        resp["watches"][0]["members"] = {{"9", 6}};
        expect_parse_error(resp, "W3: member rank out of range",
                           "out of range");
        resp = base;
        resp["watches"][0]["members"] = {{"0", "6"}};
        expect_parse_error(resp, "W3: member id not integer",
                           "not an integer");
        resp = base;
        resp["watches"][0]["members"] = {{"0", -6}};
        expect_parse_error(resp, "W3: negative member id", "negative");
    }
    {  // W4: statuses
        nlohmann::json resp = base;
        resp["watches"][0]["statuses"] = nlohmann::json::array();
        expect_parse_error(resp, "W4: empty statuses", "statuses");
        resp = base;
        resp["watches"][0]["statuses"] = nlohmann::json::array({"Failed"});
        expect_parse_error(resp, "W4: unknown status", "unknown status");
        resp = base;
        resp["watches"][0]["statuses"] = nlohmann::json::array({1});
        expect_parse_error(resp, "W4: status not string", "not a string");
    }
}

// --------------------------------------------------------------- Part A ----
void test_alarm_rules() {
    const nlohmann::json base = max_shape_response();

    {  // A1: shape
        nlohmann::json resp = base;
        resp["future_alarms"][0] = 9;
        expect_parse_error(resp, "A1: alarm not object", "future_alarm[0]");
        resp = base;
        resp["future_alarms"][0]["extra"] = 1;
        expect_parse_error(resp, "A1: alarm unknown key", "extra");
        resp = base;
        resp["future_alarms"][0].erase("envelope");
        expect_parse_error(resp, "A1: alarm missing envelope", "envelope");
    }
    {  // A2: envelope key set
        nlohmann::json resp = base;
        resp["future_alarms"][0]["envelope"]["extra"] = 1;
        expect_parse_error(resp, "A2: envelope unknown key", "extra");
        resp = base;
        resp["future_alarms"][0]["envelope"].erase("session_id");
        expect_parse_error(resp, "A2: envelope missing session_id",
                           "session_id");
    }
    {  // A3: envelope domains
        nlohmann::json resp = base;
        resp["future_alarms"][0]["envelope"]["request_id"] = "";
        expect_parse_error(resp, "A3: empty envelope request_id",
                           "request_id");
        resp = base;
        resp["future_alarms"][0]["envelope"]["turn_index"] = -1;
        expect_parse_error(resp, "A3: negative turn_index", "turn_index");
        resp = base;
        resp["future_alarms"][0]["envelope"]["queue_index"] = -2;
        expect_parse_error(resp, "A3: queue_index < -1", "queue_index");
        resp = base;
        resp["future_alarms"][0]["arrival_world_ns"] = -1;
        expect_parse_error(resp, "A3: negative arrival", "negative");
    }
}

// ------------------------------------------------------- Part S / O / P ----
void test_opaque_and_order_rules() {
    const nlohmann::json base = max_shape_response();

    {  // S1: assignments -- required structural fields, extra keys legal
        nlohmann::json resp = base;
        resp["assignments"][0].erase("request_id");
        expect_parse_error(resp, "S1: assignment missing request_id",
                           "request_id");
        resp = base;
        resp["assignments"][0]["prefill_instance_index"] = -1;
        expect_parse_error(resp, "S1: negative instance index", "negative");
        resp = base;
        resp["assignments"][0].erase("decode_instance_index");
        expect_parse_error(resp, "S1: missing decode_instance_index",
                           "decode_instance_index");
    }
    {  // S2: kv_actions -- required structural fields, extra keys legal
        nlohmann::json resp = base;
        resp["kv_actions"][0].erase("event_type");
        expect_parse_error(resp, "S2: kv missing event_type",
                           "event_type/trigger_request_id");
        resp = base;
        resp["kv_actions"][0]["trigger_request_id"] = "";
        expect_parse_error(resp, "S2: kv empty trigger_request_id",
                           "event_type/trigger_request_id");
    }
    {  // O1: array order preservation across interleaved ranks
        nlohmann::json resp = max_shape_response();
        resp["nodes"] = nlohmann::json::array(
            {node_json(2, 0, 4, "a"), node_json(0, 5, 4, "b"),
             node_json(1, 3, 4, "c"), node_json(0, 6, 4, "d")});
        resp["touched_ranks"] = nlohmann::json::array({2, 0, 1});
        const ParsedGraphBatch parsed =
            parse_ok(resp, "O1: interleaved ranks");
        expect(parsed.nodes.size() == 4, "O1: four nodes parsed");
        expect(parsed.nodes[0].node.rank == 2 && parsed.nodes[0].json_id == 0 &&
               parsed.nodes[1].node.rank == 0 && parsed.nodes[1].json_id == 5 &&
               parsed.nodes[2].node.rank == 1 && parsed.nodes[3].node.rank == 0,
               "O1: node vector keeps the array order verbatim");
        expect(parsed.touched_ranks == std::vector<int>({2, 0, 1}),
               "O1: touched_ranks keeps the declared order");
    }
    {  // O1: watch member order follows the rank-string key sort; statuses
        //     keep the raw array order including duplicates
        nlohmann::json resp = max_shape_response();
        resp["watches"][0]["members"] = {{"2", 9}, {"0", 7}, {"1", 8}};
        resp["watches"][0]["statuses"] =
            nlohmann::json::array({"Skipped", "Success", "Skipped"});
        const ParsedGraphBatch parsed =
            parse_ok(resp, "O1: member/status order");
        const ParsedWatch& watch = parsed.watches[0];
        expect(watch.members.size() == 3 &&
                   watch.members[0].rank == 0 && watch.members[0].json_id == 7 &&
                   watch.members[1].rank == 1 && watch.members[2].rank == 2,
               "O1: members follow the JSON object key order (ascending)");
        expect(watch.statuses.size() == 3 &&
                   watch.statuses[0] == NodeTerminalStatus::Skipped &&
                   watch.statuses[1] == NodeTerminalStatus::Success &&
                   watch.statuses[2] == NodeTerminalStatus::Skipped,
               "O1: statuses keep the raw array order (duplicates kept)");
    }
    {  // O3: unicode passthrough (no validation either way)
        nlohmann::json resp = max_shape_response();
        resp["nodes"][0]["name"] = "r0_计hdr_α";
        resp["nodes"][0]["request_id"] = "req_中文_1";
        const ParsedGraphBatch parsed =
            parse_ok(resp, "O3: unicode passthrough");
        expect(parsed.nodes[0].node.name == "r0_计hdr_α" &&
                   parsed.nodes[0].node.request_id == "req_中文_1",
               "O3: non-ASCII strings pass through byte-exact");
    }
    {  // O4: the frozen error skeleton parses cleanly, error carried
        nlohmann::json resp;
        resp["schema_version"] = 1;
        resp["batch_id"] = 5;
        resp["source_delivery_sequence"] = 5;
        resp["nodes"] = nlohmann::json::array();
        resp["parent_edges"] = nlohmann::json::array();
        resp["watches"] = nlohmann::json::array();
        resp["assignments"] = nlohmann::json::array();
        resp["kv_actions"] = nlohmann::json::array();
        resp["future_alarms"] = nlohmann::json::array();
        resp["error"] = "handler exploded";
        const ParsedGraphBatch parsed =
            parse_ok(resp, "O4: error skeleton parses");
        expect(parsed.error == "handler exploded",
               "O4: error field carried for the bridge/validate abort");
    }
}

// -------------------------------------------------------- Part P (shape) ----
void test_max_shape_round_trip() {
    const nlohmann::json resp = max_shape_response();
    const ParsedGraphBatch parsed = parse_ok(resp, "P: max shape");

    expect(parsed.batch_id == 42 && parsed.source_delivery_sequence == 42,
           "P: header fields round-trip");
    expect(parsed.nodes.size() == 7, "P: seven nodes");
    expect(parsed.nodes[0].node.rank == 0 && parsed.nodes[0].json_id == 0 &&
               parsed.nodes[0].node.node_type == 1 &&
               parsed.nodes[0].node.kind == NodeKind::Metadata,
           "P: type-1 node maps to NodeKind::Metadata");
    expect(parsed.nodes[1].node.kind == NodeKind::MemLoad &&
               parsed.nodes[2].node.kind == NodeKind::MemStore &&
               parsed.nodes[3].node.kind == NodeKind::Compute &&
               parsed.nodes[4].node.kind == NodeKind::CommSend &&
               parsed.nodes[5].node.kind == NodeKind::CommRecv &&
               parsed.nodes[6].node.kind == NodeKind::CommCollective,
           "P: every NodeKind mapping");
    const ComputeAttrs& compute = parsed.nodes[0].node.compute;
    expect(compute.num_ops == 5000000000ULL &&
               compute.tensor_size == 6000000000ULL &&
               compute.runtime_ns == 12345,
           "P: uint64 compute fields survive > 2^32 values");
    expect(compute.has_remote_weight_bytes && compute.remote_weight_bytes == 777,
           "P: optional remote_weight_bytes present");
    // sh_2.0 adaptation: the HBM attrs live on the TOP-LEVEL OnlineNode
    // (hbm_access_mode), not inside ComputeAttrs; CommAttrs keeps this
    // repo's hbm_charge member.
    expect(parsed.nodes[0].node.hbm_access_mode == 1,
           "P: optional hbm_access_mode");
    const CommAttrs& comm = parsed.nodes[0].node.comm;
    expect(comm.bytes == 128 && comm.src == 0 && comm.dst == 1 &&
               comm.tag == 7 && comm.hbm_charge == false,
           "P: comm fields incl. hbm_charge=false");
    const CollAttrs& coll = parsed.nodes[0].node.coll;
    expect(coll.comm_type == 2 && coll.bytes == 64 && coll.priority == 3 &&
               coll.pg_name == "tp0" &&
               coll.involved_dim == std::vector<bool>({true, false}),
           "P: coll fields incl. involved_dim");
    expect(parsed.nodes[0].node.request_id == "r1" &&
               parsed.nodes[0].node.stage == "prefill" &&
               parsed.nodes[0].node.generation == 0 &&
               parsed.nodes[0].node.is_cpu_op == false &&
               parsed.nodes[0].node.is_timer_op == false &&
               parsed.nodes[0].node.inputs_values.empty() &&
               parsed.nodes[0].node.global_id == 0,
           "P: node identity fields (global_id stays 0 for NodeStore)");
    // sh_2.0-specific optional node key: absent is_local_hbm_kv_restore
    // keeps the false default (the restore-DMA routing bit, §4.3.3); the
    // type-2 fixture node emits it true.
    expect(parsed.nodes[0].node.is_local_hbm_kv_restore == false,
           "P: absent is_local_hbm_kv_restore defaults to false");
    expect(parsed.nodes[1].node.is_local_hbm_kv_restore == true,
           "P: is_local_hbm_kv_restore=true parsed on the type-2 node");
    expect(parsed.parent_edges.size() == 1 &&
               parsed.parent_edges[0].rank == 0 &&
               parsed.parent_edges[0].from_json == 999 &&
               parsed.parent_edges[0].to_json == 6,
           "P: cross-batch parent edge parsed (resolution is preflight's)");
    expect(parsed.watches.size() == 1 && parsed.watches[0].members.size() == 1,
           "P: watch parsed");
    expect(parsed.assignments.size() == 1 &&
               parsed.assignments[0].request_id == "r1" &&
               parsed.assignments[0].prefill_instance_index == 0 &&
               parsed.assignments[0].decode_instance_index == 1,
           "P: assignment structural fields parsed (extra key tolerated)");
    expect(parsed.kv_actions.size() == 1 &&
               parsed.kv_actions[0].event_type == "admit" &&
               parsed.kv_actions[0].trigger_request_id == "r1",
           "P: kv action structural fields parsed (extra key tolerated)");
    expect(parsed.future_alarms.size() == 1 &&
               parsed.future_alarms[0].arrival_world_ns == 200000 &&
               parsed.future_alarms[0].envelope.request_id == "r2" &&
               parsed.future_alarms[0].envelope.turn_index == 1 &&
               parsed.future_alarms[0].envelope.prefill_length == 100 &&
               parsed.future_alarms[0].envelope.decode_length == 10 &&
               parsed.future_alarms[0].envelope.queue_index == 3,
           "P: alarm incl. optional queue_index");
    expect(parsed.has_touched_ranks && parsed.touched_ranks.size() == 1 &&
               parsed.touched_ranks[0] == 0,
           "P: touched_ranks parsed");
    expect(parsed.error.empty(), "P: no error on the happy path");
}

// K (C-3, 2026-09-30): is_kv_route_transfer_leg -- the code-7 memory-anchor
// predicate. The old find("kv") substring heuristic also matched the
// aggregated "..._attention_qkv_projection" comp node emitted later in the
// same request context, so every request's anchor landed on the qkv compute
// tick. The predicate must accept ONLY the exact emission-contract suffixes
// of this repo (_recv_kv / _target_hbm_write_kv) and reject everything else,
// including "kv" embedded mid-name (user data is loader-guarded, M-27, but
// the suffix match itself must not widen).
void test_kv_route_transfer_leg_predicate() {
    OnlineNode leg;
    leg.kind = NodeKind::CommRecv;
    leg.name = "q0001_s1_turn1_r1_history_transfer_action000_s_shard0_recv_kv";
    expect(is_kv_route_transfer_leg(leg), "K: noc-migrate recv_kv leg matches");

    OnlineNode restore_write;
    restore_write.kind = NodeKind::MemLoad;
    restore_write.name = "q0001_s1_turn1_r1_history_transfer_action000_s_shard1_target_hbm_write_kv";
    expect(is_kv_route_transfer_leg(restore_write),
           "K: restore target_hbm_write_kv leg matches");

    OnlineNode qkv_polluter;
    qkv_polluter.kind = NodeKind::Compute;
    qkv_polluter.name =
        "q0001_s1_turn1_r1_current_prefill_request_aggregated_all_layers_"
        "attention_qkv_projection";
    expect(!is_kv_route_transfer_leg(qkv_polluter),
           "K: qkv comp node must NOT match (C-3 pollution regression)");

    OnlineNode joiner_route;
    joiner_route.kind = NodeKind::CommSend;
    joiner_route.name = "q0001_s1_turn0_r0_prefill_to_decode_xfer_send_rank5_to_rank1";
    expect(!is_kv_route_transfer_leg(joiner_route),
           "K: joiner transfer route carries no kv leg");

    OnlineNode kv_mid_name;
    kv_mid_name.kind = NodeKind::CommRecv;
    kv_mid_name.name = "prefix_kv_middle_tail";
    expect(!is_kv_route_transfer_leg(kv_mid_name),
           "K: kv substring mid-name is not a leg (suffix-exact)");

    OnlineNode plain;
    plain.kind = NodeKind::MemStore;
    plain.name = "q0001_s1_turn1_r1_decode_evictions_action000_s_shard0_remote_store";
    expect(!is_kv_route_transfer_leg(plain),
           "K: store branch mem_store is not a transfer-arrival leg");

    OnlineNode bare_suffix;
    bare_suffix.kind = NodeKind::MemLoad;
    bare_suffix.name = "_target_hbm_write_kv";
    expect(is_kv_route_transfer_leg(bare_suffix),
           "K: exact-suffix name still matches (no prefix required)");
}

}  // namespace

int main() {
    test_top_level_rules();
    test_workspace_v2_snapshot_rules();
    test_node_rules();
    test_edge_rules();
    test_watch_rules();
    test_alarm_rules();
    test_opaque_and_order_rules();
    test_max_shape_round_trip();
    test_kv_route_transfer_leg_predicate();
    if (g_ok) {
        std::printf("[parsed_graph_batch_test] ALL PASS\n");
        return 0;
    }
    std::fprintf(stderr, "[parsed_graph_batch_test] FAILURES\n");
    return 1;
}
