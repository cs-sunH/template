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
// defaulted absent keys and let type mismatches explode downstream).
// (2026-10-01 dead-code removal: a `field()` existence-lookup helper sat
// here with zero callers -- after the C1 key-set whitelist every accessor
// reaches its keys through the allowed/required set checks, so the helper
// was unreachable from the start.) ----

// Unsigned integer: rejects floats (incl. >UINT64 integer literals, which
// nlohmann stores as double), negative values, bools and strings.
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

    // N12: compute -- required {num_ops, tensor_size, runtime_ns};
    // optional {op_role, die_op_spec} (die backend, contract §2).
    // (wscllm repo note vs the sh_1.0 original: this repo has no
    // MEM_LOAD/MEM_STORE remote-pool nodes, so ComputeAttrs carries no
    // hbm_access_mode field and the key is NOT in the allowed set -- a
    // response carrying it fails closed as an unknown key, the same
    // tight-schema posture the wscllm builder has always had: it never
    // writes the key. remote_weight_bytes stays in the allowed set only so
    // its presence gets the dedicated unsupported-key diagnosis below
    // instead of the generic unknown-key one.)
    const nlohmann::json& compute = node_json["compute"];
    if (!compute.is_object()) {
        fail(where + ": compute is not an object");
    }
    check_key_set(compute,
                  {"num_ops", "tensor_size", "runtime_ns",
                   "remote_weight_bytes", "op_role", "die_op_spec"},
                  {"num_ops", "tensor_size", "runtime_ns"},
                  where + ".compute");
    node.compute.num_ops = get_uint(compute["num_ops"], "num_ops",
                                    where + ".compute");
    node.compute.tensor_size = get_uint(compute["tensor_size"], "tensor_size",
                                        where + ".compute");
    node.compute.runtime_ns = get_uint(compute["runtime_ns"], "runtime_ns",
                                       where + ".compute");
    if (compute.contains("op_role")) {
        const std::string op_role =
            get_string(compute["op_role"], "op_role", where + ".compute");
        if (op_role == "compute") {
            node.compute.op_role = OpRole::Compute;
        } else if (op_role == "control_marker") {
            node.compute.op_role = OpRole::ControlMarker;
        } else {
            fail(where + ": compute.op_role \"" + op_role +
                 "\" is not one of compute|control_marker");
        }
    }
    if (compute.contains("die_op_spec")) {
        // Contract §2 conflict rule: a die-spec node must not carry a
        // calibrated runtime (a non-zero runtime_ns would silently bypass
        // the die execution and must fail closed at parse time, not at
        // issue time).
        if (node.compute.runtime_ns != 0) {
            fail(where + ": compute.die_op_spec requires runtime_ns == 0 "
                        "(die nodes execute the compiled plan, never a "
                        "calibrated runtime)");
        }
        const nlohmann::json& spec_json = compute["die_op_spec"];
        if (!spec_json.is_object()) {
            fail(where + ": compute.die_op_spec is not an object");
        }
        node.compute.die_task_json =
            std::make_shared<const nlohmann::json>(spec_json);
        // Keep the legacy parsed form for mesh_die_v1. A new coarse task
        // descriptor has no legacy schema_version and is validated by the
        // single coarse_die_v2 compiler at issue/oracle time. In both cases
        // the original JSON survives unchanged in NodeStore.
        if (!spec_json.contains("schema_version")) {
            if (!spec_json.contains("op_kind")) {
                fail(where + ": compute.die_op_spec is missing op_kind");
            }
        } else {
            try {
                node.compute.die_spec = DieModel::parse_die_op_spec(spec_json);
            } catch (const ParseError&) {
                throw;
            } catch (const std::exception& e) {
                fail(where + ": compute.die_op_spec rejected: " + e.what());
            }
        }
    }
    if (compute.contains("remote_weight_bytes")) {
        // 3.2 fix (2026-09-29): this key used to be parse-legal and only
        // fail at first issue (Workload's remote-operand leg throws for
        // this repo -- it has no remote weight pool), so a stray key in a
        // trace_config aborted the whole simulation mid-run with a
        // diagnosis far from the shape that caused it. Fail here, at the
        // parse boundary, where the batch and node are still nameable.
        // (The tight-schema posture above already rejects the
        // hbm_access_mode cousin; this arm closes the remaining
        // accepted-but-unsupported shape.)
        fail(where + ".compute: remote_weight_bytes is not supported by "
                    "this repo (no remote weight pool; the key aborts at "
                    "first issue) -- remove it from the trace");
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
    // 4.1 tag-domain gate (2026-09-29): the front-end send path panics on
    // any tag at or beyond the NATIVE window ceiling (Sys::COLLECTIVE =
    // 5e8 -- the modulo remap there would fold into another stream's tag
    // space). The old parse bound (uint32 max) admitted tags up to
    // qi*10000+... with queue_index ≈ 429496, so an oversized queue
    // aborted the run mid-flight with a diagnosis far from the node.
    // Enforce the same ceiling here, at the parse boundary.
    if (tag >= 500000000ULL) {
        fail(where + ": comm tag " + std::to_string(tag) +
             " is at or beyond the NATIVE front-end window ceiling "
             "(500000000) -- the p2p tag domain is "
             "queue_index*10000+category+rank, so queue_index must stay "
             "below 50000");
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
    // (strictly [0-9]+: the producer emits str(int) keys, and std::stoi
    // would silently accept leading whitespace and a +/- sign, a wider
    // input class than the rule text), values are non-negative integer
    // json ids. Insertion order == JSON object key order (nlohmann sorts
    // object keys LEXICOGRAPHICALLY -- "10" before "2" at rank >= 10;
    // consumers never assume numeric order).
    const nlohmann::json& members = watch_json["members"];
    if (!members.is_object() || members.empty()) {
        fail(where + ": empty/absent members");
    }
    watch.members.reserve(members.size());
    for (const auto& entry : members.items()) {
        const std::string& key = entry.key();
        if (key.empty() ||
            key.find_first_not_of("0123456789") != std::string::npos) {
            fail(where + ": unparsable member rank: " + key);
        }
        // 10+ decimal digits cannot fit int (old stoi raised out_of_range);
        // reject before accumulating so the manual parse cannot overflow
        // when num_ranks == -1 skips the domain check.
        if (key.size() > 9) {
            fail(where + ": unparsable member rank: " + key);
        }
        int rank = 0;
        for (const char digit : key) {
            rank = rank * 10 + (digit - '0');
        }
        check_rank_domain(rank, num_ranks, where);
        watch.members.push_back(
            ParsedWatchMember{rank,
                              get_uint(entry.value(), "members[" + key +
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
    // D3 positive-length lexicon (4.6 fix, 2026-09-29): the CSV reader and
    // the FIFO Submit port reject zero-length requests; this arm silently
    // admitted them (a zero-length envelope entered the queue the other
    // two ports refuse).
    if (alarm.envelope.prefill_length == 0 ||
        alarm.envelope.decode_length == 0) {
        fail(where + ": envelope prefill_length/decode_length must be "
             "positive (zero-length requests are rejected by the "
             "materializer, the Python loader, and the CSV reader alike)");
    }
    // Reserved-token gate (4.2 fix, same lexicon as the CSV reader and the
    // FIFO Submit port): "first_token" is the WP9 node-name anchor
    // substring and the graph builder embeds the sanitized ids in every
    // regular node name.
    if (alarm.envelope.session_id.find("first_token") != std::string::npos ||
        alarm.envelope.request_id.find("first_token") != std::string::npos) {
        fail(where + ": envelope session_id/request_id must not contain "
             "the reserved marker token \"first_token\"");
    }
    // Reserved-prefix gate (same input-port discipline): a "batch_train_"
    // request_id belongs to the train-sentinel namespace the mechanism
    // layer treats as reserved (watches pinned to prefill, eligibility
    // checks skipped, PREFILL_DRAIN excluded from prefill_drained), so a
    // real request with that id would bypass fail-closed validation. The
    // CSV reader rejects the prefix at its own port; this FIFO Submit
    // alarm envelope port fails closed identically.
    if (alarm.envelope.request_id.rfind("batch_train_", 0) == 0) {
        fail(where + ": envelope request_id must not use the reserved "
             "\"batch_train_\" prefix (train-sentinel namespace)");
    }
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
