/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

ParsedGraphBatch.cc -- C1 (2026-08-29) one-shot structural parse of one
GraphBatch response into the typed ParsedGraphBatch. See ParsedGraphBatch.hh
for the rule catalogue (T/N/E/W/A/S/O) and the ordering/atomicity contract.
Every rule here either replaces a per-walk nlohmann .value()/.find() of the
old four-pass pipeline or tightens it fail-closed (unknown keys, exact key
sets, integer domains); the committer-state-dependent rules (id-stream
contiguity, cycles, p2p pairing, collective membership, MEM ports, watch
eligibility, alarm timing, touched-rank equality) deliberately stay in
GraphBatchCommitter and consume the typed fields.
******************************************************************************/

#include "astra-sim/workload/execution_driven/ParsedGraphBatch.hh"

#include <limits>
#include <string>
#include <vector>

namespace AstraSim {
namespace ExecutionDriven {
namespace {

// Error text prefix: every message names the failing rule family, the entry
// index and the field, so one stderr line fully localizes the violation
// (C1_DESIGN §2.2's error contract; the abort channel itself is the
// bridge's, unchanged).
[[noreturn]] void fail(const std::string& what) {
    throw ParseError(what);
}

// ---- typed getters (strict; the old .value(key, default) silently
// defaulted absent keys and let type mismatches explode downstream) ----

const nlohmann::json& field(const nlohmann::json& obj, const char* key,
                            const std::string& where) {
    const auto it = obj.find(key);
    if (it == obj.end()) {
        fail(where + ": missing field \"" + key + "\"");
    }
    return *it;
}

// Unsigned integer: rejects floats (incl. >UINT64 integer literals, which
// nlohmann stores as double -- rule O6), negative values, bools and strings.
uint64_t get_uint(const nlohmann::json& value, const std::string& key,
                  const std::string& where) {
    if (!value.is_number_integer()) {
        fail(where + ": field \"" + key + "\" is not an integer");
    }
    if (value.is_number_unsigned()) {
        return value.get<uint64_t>();
    }
    const int64_t raw = value.get<int64_t>();
    if (raw < 0) {
        fail(where + ": field \"" + key + "\" is negative (" +
             std::to_string(raw) + ")");
    }
    return static_cast<uint64_t>(raw);
}

// Signed 64-bit integer (queue_index semantics: -1 is the "unknown" value).
int64_t get_int64(const nlohmann::json& value, const std::string& key,
                  const std::string& where) {
    if (!value.is_number_integer()) {
        fail(where + ": field \"" + key + "\" is not an integer");
    }
    if (value.is_number_unsigned()) {
        const uint64_t raw = value.get<uint64_t>();
        if (raw > static_cast<uint64_t>(
                      std::numeric_limits<int64_t>::max())) {
            fail(where + ": field \"" + key + "\" overflows int64");
        }
        return static_cast<int64_t>(raw);
    }
    return value.get<int64_t>();
}

// int32-range integer (rank, turn_index).
int get_int(const nlohmann::json& value, const std::string& key,
            const std::string& where) {
    const int64_t raw = get_int64(value, key, where);
    if (raw < std::numeric_limits<int>::min() ||
        raw > std::numeric_limits<int>::max()) {
        fail(where + ": field \"" + key + "\" overflows int32 (" +
             std::to_string(raw) + ")");
    }
    return static_cast<int>(raw);
}

std::string get_string(const nlohmann::json& value, const std::string& key,
                       const std::string& where) {
    if (!value.is_string()) {
        fail(where + ": field \"" + key + "\" is not a string");
    }
    return value.get<std::string>();
}

bool get_bool(const nlohmann::json& value, const std::string& key,
              const std::string& where) {
    if (!value.is_boolean()) {
        fail(where + ": field \"" + key + "\" is not a boolean");
    }
    return value.get<bool>();
}

// ---- key-set checks (rules N2/N12/N13/N14/E1/W1/A1/A2, top-level T6) ----

// Every present key must belong to `allowed` (unknown-key fail-closed) and
// every `required` key must be present (missing-key fail-closed).
void check_key_set(const nlohmann::json& obj,
                   const std::vector<const char*>& allowed,
                   const std::vector<const char*>& required,
                   const std::string& where) {
    for (const auto& entry : obj.items()) {
        bool known = false;
        for (const char* key : allowed) {
            if (entry.key() == key) {
                known = true;
                break;
            }
        }
        if (!known) {
            fail(where + ": unknown field \"" + entry.key() + "\"");
        }
    }
    for (const char* key : required) {
        if (obj.find(key) == obj.end()) {
            fail(where + ": missing field \"" + std::string(key) + "\"");
        }
    }
}

void check_rank_domain(const int rank, const int num_ranks,
                       const std::string& where) {
    if (num_ranks >= 0 && (rank < 0 || rank >= num_ranks)) {
        fail(where + ": rank " + std::to_string(rank) +
             " out of range [0," + std::to_string(num_ranks) + ")");
    }
}

constexpr uint64_t kUint32Max =
    static_cast<uint64_t>(std::numeric_limits<uint32_t>::max());

ParsedNode parse_node(const nlohmann::json& node_json, size_t index,
                      int num_ranks) {
    const std::string where =
        "node[" + std::to_string(index) + "]";
    if (!node_json.is_object()) {
        fail(where + " is not an object (rule N1)");
    }
    // N2: exactly the frozen 13-key set.
    check_key_set(
        node_json,
        {"rank", "id", "name", "type", "is_cpu_op", "is_timer_op",
         "inputs_values", "request_id", "stage", "generation", "compute",
         "comm", "coll"},
        {"rank", "id", "name", "type", "is_cpu_op", "is_timer_op",
         "inputs_values", "request_id", "stage", "generation", "compute",
         "comm", "coll"},
        where);

    ParsedNode parsed;
    OnlineNode& node = parsed.node;

    const int rank = get_int(node_json["rank"], "rank", where);  // N3
    check_rank_domain(rank, num_ranks, where);
    node.rank = rank;

    parsed.json_id = get_uint(node_json["id"], "id", where);  // N4

    const uint64_t type = get_uint(node_json["type"], "type", where);  // N5
    if (type < 1 || type > 7) {
        fail(where + ": type " + std::to_string(type) +
             " out of range [1,7]");
    }
    node.node_type = type;
    node.kind = node_kind_from_type(type);

    node.name = get_string(node_json["name"], "name", where);  // N6
    if (node.name.empty()) {
        fail(where + ": empty name");
    }
    node.is_cpu_op = get_bool(node_json["is_cpu_op"], "is_cpu_op", where);
    node.is_timer_op =
        get_bool(node_json["is_timer_op"], "is_timer_op", where);  // N7
    node.inputs_values =
        get_string(node_json["inputs_values"], "inputs_values", where);  // N8

    node.request_id =
        get_string(node_json["request_id"], "request_id", where);  // N9
    if (node.request_id.empty()) {
        fail(where + ": empty request_id");
    }
    node.stage = get_string(node_json["stage"], "stage", where);  // N10
    if (node.stage != "prefill" && node.stage != "decode") {
        fail(where + ": invalid stage \"" + node.stage + "\"");
    }
    node.generation =
        get_uint(node_json["generation"], "generation", where);  // N11
    const uint64_t expected_generation = node.stage == "prefill" ? 0 : 1;
    if (node.generation != expected_generation) {
        fail(where + ": generation " + std::to_string(node.generation) +
             " does not match stage " + node.stage);
    }

    // N12: compute -- required {num_ops, tensor_size, runtime_ns}; optional
    // {remote_weight_bytes}. [face adaptation] no hbm_access_mode key: face
    // has no MEM-node local-HBM access-mode contract (NO_MEMORY_EXPANSION;
    // the N-user LocalHbmBandwidthModel derives access from node type), so
    // the sh_1.0 optional key is dropped from the schema here.
    const nlohmann::json& compute = node_json["compute"];
    if (!compute.is_object()) {
        fail(where + ": compute is not an object");
    }
    check_key_set(compute,
                  {"num_ops", "tensor_size", "runtime_ns",
                   "remote_weight_bytes"},
                  {"num_ops", "tensor_size", "runtime_ns"},
                  where + ".compute");
    node.compute.num_ops = get_uint(compute["num_ops"], "num_ops",
                                    where + ".compute");
    node.compute.tensor_size = get_uint(compute["tensor_size"], "tensor_size",
                                        where + ".compute");
    node.compute.runtime_ns = get_uint(compute["runtime_ns"], "runtime_ns",
                                       where + ".compute");
    if (compute.contains("remote_weight_bytes")) {
        node.compute.has_remote_weight_bytes = true;
        node.compute.remote_weight_bytes =
            get_uint(compute["remote_weight_bytes"], "remote_weight_bytes",
                     where + ".compute");
    }

    // N13: comm -- required {bytes, src, dst, tag}; optional {hbm_charge}.
    const nlohmann::json& comm = node_json["comm"];
    if (!comm.is_object()) {
        fail(where + ": comm is not an object");
    }
    check_key_set(comm, {"bytes", "src", "dst", "tag", "hbm_charge"},
                  {"bytes", "src", "dst", "tag"}, where + ".comm");
    node.comm.bytes = get_uint(comm["bytes"], "bytes", where + ".comm");
    node.comm.src = get_int(comm["src"], "src", where + ".comm");
    node.comm.dst = get_int(comm["dst"], "dst", where + ".comm");
    const uint64_t tag = get_uint(comm["tag"], "tag", where + ".comm");
    if (tag > kUint32Max) {
        fail(where + ": comm tag out of range: " + std::to_string(tag));
    }
    node.comm.tag = static_cast<uint32_t>(tag);
    if (comm.contains("hbm_charge")) {
        node.comm.hbm_charge =
            get_bool(comm["hbm_charge"], "hbm_charge", where + ".comm");
    }

    // N14: coll -- required {comm_type, bytes, priority, pg_name}; optional
    // {involved_dim} (absent == empty vector; the real builder always sends
    // all five). Type-7 semantic checks (bytes > 0, pg_name non-empty,
    // completion path) stay in the preflight.
    const nlohmann::json& coll = node_json["coll"];
    if (!coll.is_object()) {
        fail(where + ": coll is not an object");
    }
    check_key_set(coll,
                  {"comm_type", "bytes", "priority", "pg_name",
                   "involved_dim"},
                  {"comm_type", "bytes", "priority", "pg_name"},
                  where + ".coll");
    node.coll.comm_type =
        get_uint(coll["comm_type"], "comm_type", where + ".coll");
    node.coll.bytes = get_uint(coll["bytes"], "bytes", where + ".coll");
    const uint64_t priority =
        get_uint(coll["priority"], "priority", where + ".coll");
    if (priority > kUint32Max) {
        fail(where + ": coll priority out of range: " +
             std::to_string(priority));
    }
    node.coll.priority = static_cast<uint32_t>(priority);
    node.coll.pg_name = get_string(coll["pg_name"], "pg_name", where + ".coll");
    if (coll.contains("involved_dim")) {
        const nlohmann::json& dims = coll["involved_dim"];
        if (!dims.is_array()) {
            fail(where + ": coll.involved_dim is not an array");
        }
        node.coll.involved_dim.reserve(dims.size());
        for (size_t d = 0; d < dims.size(); ++d) {
            node.coll.involved_dim.push_back(
                get_bool(dims[d], "involved_dim[" + std::to_string(d) + "]",
                         where + ".coll"));
        }
    }
    return parsed;
}

ParsedEdge parse_edge(const nlohmann::json& edge_json, size_t index,
                      int num_ranks) {
    const std::string where = "parent_edge[" + std::to_string(index) + "]";
    if (!edge_json.is_object()) {
        fail(where + " is not an object (rule E1)");
    }
    check_key_set(edge_json, {"rank", "from", "to", "kind"},
                  {"rank", "from", "to", "kind"}, where);
    ParsedEdge edge;
    edge.rank = get_int(edge_json["rank"], "rank", where);  // E2
    check_rank_domain(edge.rank, num_ranks, where);
    edge.from_json = get_uint(edge_json["from"], "from", where);
    edge.to_json = get_uint(edge_json["to"], "to", where);
    const std::string kind = get_string(edge_json["kind"], "kind", where);
    if (kind != "data") {  // E3
        fail(where + ": unsupported kind \"" + kind + "\"");
    }
    if (edge.from_json == edge.to_json) {  // E4
        fail(where + ": self-loop on rank " + std::to_string(edge.rank));
    }
    return edge;
}

ParsedWatch parse_watch(const nlohmann::json& watch_json, size_t index,
                        int num_ranks) {
    const std::string where = "watch[" + std::to_string(index) + "]";
    if (!watch_json.is_object()) {
        fail(where + " is not an object (rule W1)");
    }
    check_key_set(watch_json,
                  {"request_id", "stage", "generation", "members", "statuses"},
                  {"request_id", "stage", "generation", "members", "statuses"},
                  where);
    ParsedWatch watch;
    watch.request_id =
        get_string(watch_json["request_id"], "request_id", where);
    if (watch.request_id.empty()) {  // W2
        fail(where + ": empty request_id");
    }
    watch.stage = get_string(watch_json["stage"], "stage", where);
    if (watch.stage != "prefill" && watch.stage != "decode") {
        fail(where + ": invalid stage \"" + watch.stage + "\"");
    }
    watch.generation =
        get_uint(watch_json["generation"], "generation", where);
    const uint64_t expected_generation = watch.stage == "prefill" ? 0 : 1;
    if (watch.generation != expected_generation) {
        fail(where + ": generation " + std::to_string(watch.generation) +
             " does not match stage " + watch.stage);
    }

    // W3: members -- non-empty object; keys are decimal rank strings
    // (std::stoi + no trailing characters, the pre-C1 rule exactly), values
    // are non-negative integer json ids. Insertion order == JSON object key
    // order == the producer's ascending rank-string order.
    const nlohmann::json& members = watch_json["members"];
    if (!members.is_object() || members.empty()) {
        fail(where + ": empty/absent members");
    }
    watch.members.reserve(members.size());
    for (const auto& entry : members.items()) {
        int rank = 0;
        try {
            size_t parsed_chars = 0;
            rank = std::stoi(entry.key(), &parsed_chars);
            if (parsed_chars != entry.key().size()) {
                fail(where + ": unparsable member rank: " + entry.key());
            }
        } catch (const ParseError&) {
            throw;
        } catch (const std::exception&) {
            fail(where + ": unparsable member rank: " + entry.key());
        }
        check_rank_domain(rank, num_ranks, where);
        watch.members.push_back(
            ParsedWatchMember{rank,
                              get_uint(entry.value(), "members[" + entry.key() +
                                                          "]",
                                       where)});
    }

    // W4: statuses -- non-empty array of "Success"/"Skipped", raw order
    // preserved (duplicates included; consumers set-ify exactly like the
    // DOM path did).
    const nlohmann::json& statuses = watch_json["statuses"];
    if (!statuses.is_array() || statuses.empty()) {
        fail(where + ": empty/absent statuses");
    }
    watch.statuses.reserve(statuses.size());
    for (size_t s = 0; s < statuses.size(); ++s) {
        const std::string name = get_string(
            statuses[s], "statuses[" + std::to_string(s) + "]", where);
        if (name == "Success") {
            watch.statuses.push_back(NodeTerminalStatus::Success);
        } else if (name == "Skipped") {
            watch.statuses.push_back(NodeTerminalStatus::Skipped);
        } else {
            fail(where + ": unknown status \"" + name + "\"");
        }
    }
    return watch;
}

ParsedAlarm parse_alarm(const nlohmann::json& alarm_json, size_t index) {
    const std::string where = "future_alarm[" + std::to_string(index) + "]";
    if (!alarm_json.is_object()) {
        fail(where + " is not an object (rule A1)");
    }
    check_key_set(alarm_json, {"arrival_world_ns", "envelope"},
                  {"arrival_world_ns", "envelope"}, where);
    ParsedAlarm alarm;
    alarm.arrival_world_ns =
        get_uint(alarm_json["arrival_world_ns"], "arrival_world_ns", where);
    const nlohmann::json& envelope = alarm_json["envelope"];  // A2
    if (!envelope.is_object()) {
        fail(where + ": envelope is not an object");
    }
    check_key_set(envelope,
                  {"request_id", "session_id", "turn_index", "prefill_length",
                   "decode_length", "inter_request_interval_ns",
                   "queue_index"},
                  {"request_id", "session_id", "turn_index", "prefill_length",
                   "decode_length", "inter_request_interval_ns"},
                  where + ".envelope");
    alarm.envelope.request_id =
        get_string(envelope["request_id"], "request_id", where + ".envelope");
    if (alarm.envelope.request_id.empty()) {  // A3
        fail(where + ": empty envelope request_id");
    }
    alarm.envelope.session_id =
        get_string(envelope["session_id"], "session_id", where + ".envelope");
    if (alarm.envelope.session_id.empty()) {
        fail(where + ": empty envelope session_id");
    }
    alarm.envelope.turn_index =
        get_int(envelope["turn_index"], "turn_index", where + ".envelope");
    if (alarm.envelope.turn_index < 0) {
        fail(where + ": negative envelope turn_index");
    }
    alarm.envelope.prefill_length =
        get_uint(envelope["prefill_length"], "prefill_length",
                 where + ".envelope");
    alarm.envelope.decode_length =
        get_uint(envelope["decode_length"], "decode_length",
                 where + ".envelope");
    alarm.envelope.inter_request_interval_ns = get_uint(
        envelope["inter_request_interval_ns"], "inter_request_interval_ns",
        where + ".envelope");
    if (envelope.contains("queue_index")) {
        alarm.envelope.queue_index =
            get_int64(envelope["queue_index"], "queue_index",
                      where + ".envelope");
        if (alarm.envelope.queue_index < -1) {
            fail(where + ": invalid envelope queue_index " +
                 std::to_string(alarm.envelope.queue_index));
        }
    }
    return alarm;
}

}  // namespace

ParsedGraphBatch parse_graph_batch(const nlohmann::json& resp,
                                   const int num_ranks) {
    if (!resp.is_object()) {
        fail("response is not an object (rule T0)");
    }
    // T6: the top-level key set is frozen (contract ②/④). schema_version
    // and source_delivery_sequence are required; batch_id / error / the six
    // arrays / touched_ranks are optional (T3/T4/T5/T8).
    check_key_set(resp,
                  {"schema_version", "batch_id", "source_delivery_sequence",
                   "error", "nodes", "parent_edges", "watches", "assignments",
                   "kv_actions", "future_alarms", "touched_ranks"},
                  {"schema_version", "source_delivery_sequence"},
                  "response");

    ParsedGraphBatch batch;
    const int64_t schema_version = get_int64(
        resp["schema_version"], "schema_version", "response");
    if (schema_version != 1) {  // T1
        fail("response schema_version mismatch: " +
             std::to_string(schema_version));
    }
    batch.source_delivery_sequence =
        get_uint(resp["source_delivery_sequence"], "source_delivery_sequence",
                 "response");  // T2 (equality with the request seq is the
                               // bridge's check -- it knows the seq)
    if (resp.contains("batch_id")) {
        batch.batch_id = get_uint(resp["batch_id"], "batch_id", "response");
    }
    if (resp.contains("error")) {
        batch.error = get_string(resp["error"], "error", "response");
    }

    const auto array_or_empty = [&resp](const char* key) -> const
        nlohmann::json& {
        static const nlohmann::json kEmpty = nlohmann::json::array();
        const auto it = resp.find(key);
        if (it == resp.end()) {
            return kEmpty;  // T5: absent == empty (zero-node batches)
        }
        if (!it->is_array()) {
            fail(std::string("response: field \"") + key +
                 "\" is not an array (rule T7)");
        }
        return *it;
    };

    const nlohmann::json& nodes = array_or_empty("nodes");
    batch.nodes.reserve(nodes.size());
    for (size_t i = 0; i < nodes.size(); ++i) {
        batch.nodes.push_back(parse_node(nodes[i], i, num_ranks));
    }

    const nlohmann::json& edges = array_or_empty("parent_edges");
    batch.parent_edges.reserve(edges.size());
    for (size_t i = 0; i < edges.size(); ++i) {
        batch.parent_edges.push_back(parse_edge(edges[i], i, num_ranks));
    }

    const nlohmann::json& watches = array_or_empty("watches");
    batch.watches.reserve(watches.size());
    for (size_t i = 0; i < watches.size(); ++i) {
        batch.watches.push_back(parse_watch(watches[i], i, num_ranks));
    }

    // S1: assignments -- structural only (Python-authoritative); unknown
    // sibling keys (e.g. prefill_assignment_key) are tolerated by contract.
    const nlohmann::json& assignments = array_or_empty("assignments");
    batch.assignments.reserve(assignments.size());
    for (size_t i = 0; i < assignments.size(); ++i) {
        const std::string where = "assignment[" + std::to_string(i) + "]";
        const nlohmann::json& entry = assignments[i];
        if (!entry.is_object()) {
            fail(where + " is not an object");
        }
        if (entry.find("request_id") == entry.end() ||
            entry.find("prefill_instance_index") == entry.end() ||
            entry.find("decode_instance_index") == entry.end()) {
            fail(where + ": missing request_id/prefill_instance_index/"
                        "decode_instance_index");
        }
        ParsedAssignment assignment;
        assignment.request_id =
            get_string(entry["request_id"], "request_id", where);
        if (assignment.request_id.empty()) {
            fail(where + ": empty request_id");
        }
        assignment.prefill_instance_index = get_int64(
            entry["prefill_instance_index"], "prefill_instance_index", where);
        assignment.decode_instance_index = get_int64(
            entry["decode_instance_index"], "decode_instance_index", where);
        if (assignment.prefill_instance_index < 0 ||
            assignment.decode_instance_index < 0) {
            fail(where + ": negative instance index");
        }
        batch.assignments.push_back(std::move(assignment));
    }

    // S2: kv_actions -- structural only; the rest of each entry is opaque.
    const nlohmann::json& kv_actions = array_or_empty("kv_actions");
    batch.kv_actions.reserve(kv_actions.size());
    for (size_t i = 0; i < kv_actions.size(); ++i) {
        const std::string where = "kv_action[" + std::to_string(i) + "]";
        const nlohmann::json& entry = kv_actions[i];
        if (!entry.is_object()) {
            fail(where + " is not an object");
        }
        if (entry.find("event_type") == entry.end() ||
            entry.find("trigger_request_id") == entry.end()) {
            fail(where + ": missing event_type/trigger_request_id");
        }
        ParsedKvAction action;
        action.event_type = get_string(entry["event_type"], "event_type", where);
        action.trigger_request_id = get_string(entry["trigger_request_id"],
                                               "trigger_request_id", where);
        if (action.event_type.empty() || action.trigger_request_id.empty()) {
            fail(where + ": missing event_type/trigger_request_id");
        }
        batch.kv_actions.push_back(std::move(action));
    }

    const nlohmann::json& alarms = array_or_empty("future_alarms");
    batch.future_alarms.reserve(alarms.size());
    for (size_t i = 0; i < alarms.size(); ++i) {
        batch.future_alarms.push_back(parse_alarm(alarms[i], i));
    }

    // T8: touched_ranks optional; present => array of in-range integers.
    // Sorted-unique and node-rank-set equality stay in validate.
    const auto touched_it = resp.find("touched_ranks");
    if (touched_it != resp.end()) {
        if (!touched_it->is_array()) {
            fail("response: touched_ranks is not an array (rule T8)");
        }
        batch.has_touched_ranks = true;
        batch.touched_ranks.reserve(touched_it->size());
        for (size_t i = 0; i < touched_it->size(); ++i) {
            const std::string where =
                "touched_ranks[" + std::to_string(i) + "]";
            const int rank =
                get_int((*touched_it)[i], "touched_ranks", where);
            check_rank_domain(rank, num_ranks, where);
            batch.touched_ranks.push_back(rank);
        }
    }
    return batch;
}

}  // namespace ExecutionDriven
}  // namespace AstraSim
