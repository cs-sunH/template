/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

ParsedGraphBatch -- execution-driven mechanism layer (C1, 2026-08-29).

The typed view of one GraphBatch response. C1 replaces the old "carry the
nlohmann DOM across the bridge -> committer boundary and re-extract every
field four times" pipeline (validate_impl walk, mandatory_liveness_preflight
walk, commit_after_preflight B-1 assembly walk, anchor-registration walk)
with ONE structural parse here: FileDecisionBridge::deliver_and_receive
parses the response document once into these PODs, and every downstream
consumer (preflight, commit assembly, metrics anchors) reads typed fields.

Two-phase atomicity (方案 §8.1/§8.2, unchanged): parse_graph_batch is a pure
local construction -- no committer, NodeStore, WatchRegistry, ingress or
counter state is touched, and a ParseError unwinds nothing but local
vectors. It runs BEFORE any preflight, so a malformed response fails closed
(schema/structure family -> bridge_fatal -> abort) with zero side effects on
the official state; the structural rules it absorbs from the old DOM walks
are listed below (T/N/E/W/A/S/O rule ids, C1_DESIGN §2.2).

Ordering contract (rule O1): nodes / parent_edges / watches / assignments /
kv_actions / future_alarms keep the ARRAY ORDER (the Python emission order).
Store-id assignment order, issue order, Kahn inputs and anchor first/last
selection all derive from it -- the parser NEVER sorts, dedups or filters
these vectors. Watch members keep the JSON object key order (nlohmann
objects iterate keys in LEXICOGRAPHIC byte order, i.e. the decimal
rank STRINGS sort: "10" precedes "2" once ranks reach double digits --
NOT numeric rank order; consumers build std::set<int> and never rely on
numeric ordering). Watch statuses are stored in raw array order with
duplicates preserved (<= 2 distinct values exist; consumers build the same
std::set they built from the DOM).

Fail-closed family (ParseError, caught by deliver_and_receive ->
bridge_fatal -> stderr one line -> abort, exit code identical to every
other bridge protocol violation):
  T1 schema_version present and == 1 (type/value checked here; the bridge
     also checks it first for message parity with the pre-C1 code);
  T2 source_delivery_sequence present and integer (equality with the
     request seq stays in the bridge, which knows the seq);
  T3 error, when present, is a string (non-empty aborts in the bridge
     BEFORE the structural parse -- the frozen error-response skeleton
     carries all-empty arrays, so the message must be the decision error,
     not a misleading structural one);
  T4 batch_id optional, integer, default 0;
  T5 the six arrays are optional (absent == empty; zero-node batches and
     pre-phase-5 fixtures are legal);
  T6 NO unknown top-level keys (the response schema is frozen with
     DecisionBridge.hh contract ②/④); fix 2.2 (2026-10-17) adds the ONE
     optional top-level key die_spec_tables {contexts[], tp_slices[],
     spans[]} -- the batch-shared die-spec tables. Every node-level
     die_op_spec references entries by index (context_ref/tp_slice_ref/
     spans_ref, bounds-checked at parse) and the assembled DieOpSpec is
     field-identical to the old inline-spec parse (the same strict die_model
     parsers, run once per table entry);
  T7 a present array must be of array type;
  T8 touched_ranks optional; when present it must be an array of integers
     in [0, num_ranks) (sorted-unique and set-equality stay in validate).
  N1..N14 node rules: exact 13-key set, rank/id/type/name/flag/string/
     request/stage/generation typing and domains, compute/comm/coll
     sub-object key sets (required + allowed), integer-overflow rejection
     (a JSON integer literal > UINT64_MAX parses as a double and fails
     is_number_integer), negative-value rejection, tag/priority <=
     UINT32_MAX. MEM-node liveness is gone with the remote memory backend
     (removed 2026-09-24): MEM nodes are structurally legal here (types
     1..7) and fail closed later, at issue time -- Workload::issue throws
     for type 2/3 and the production barriers (commit-period catch and the
     deferred issue-pass wrapper, main_online.cc) turn that into the
     online fatal channel (P2-14).
  E1..E4 edge rules: exact key set {rank, from, to, kind}, rank domain,
     non-negative endpoints, kind == "data", from != to. Endpoint
     RESOLUTION (this batch vs committed history) stays in preflight.
  W1..W4 watch rules: exact key set, non-empty request_id, stage in
     {prefill, decode}, generation == stage, members a non-empty object
     whose keys are decimal rank strings in range with integer json ids,
     statuses a non-empty array of "Success"/"Skipped". Membership,
     identity uniqueness, coverage and eligibility stay in validate/
     preflight (they need the delta facts and cross-array state).
  A1..A3 alarm rules: exact key set {arrival_world_ns, envelope},
     arrival a non-negative integer, envelope an object with key set
     subset of the six frozen envelope fields + optional queue_index
     (integer >= -1, default -1), non-empty request_id/session_id,
     turn_index >= 0, non-negative lengths/intervals. Timing (>= delta
     tick), uniqueness and not-in-flight stay in validate/preflight.
  S1/S2 assignments/kv_actions are half-opaque (Python-authoritative,
     GraphBatchCommitter.hh contract): structural presence checks only --
     non-empty request_id + non-negative instance indices / non-empty
     event_type + trigger_request_id; unknown extra keys tolerated.
  O2 duplicate JSON object keys: nlohmann keeps the LAST value (parser
     behavior, unobservable post-parse); the Python producer (json.dump of
     a dict) structurally cannot emit duplicates. Not separately checked;
     a future SAX parser would upgrade this to fail-closed.
  O3 Unicode: string fields pass through as UTF-8 bytes exactly like the
     old .value<std::string>() path; no validation either way.
  O4 error responses: the frozen skeleton parses cleanly and aborts on T3
     in the bridge (error-before-parse ordering preserved).
  O5 no size cap (byte-parity with the old DOM path).
******************************************************************************/

#ifndef EXECUTION_DRIVEN_PARSEGRAPHBATCH_HH
#define EXECUTION_DRIVEN_PARSEGRAPHBATCH_HH

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include <json/json.hpp>

#include "astra-sim/workload/execution_driven/CompletionObserver.hh"
#include "astra-sim/workload/execution_driven/GraphSource.hh"
#include "astra-sim/workload/execution_driven/RequestIngress.hh"
#include "astra-sim/workload/WorkspaceAdmission.hh"

namespace AstraSim {
namespace ExecutionDriven {

/// Fail-closed structural parse failure. deliver_and_receive catches it and
/// routes it through bridge_fatal (one stderr line + abort), the same
/// channel as every other bridge protocol violation.
class ParseError : public std::runtime_error {
  public:
    explicit ParseError(const std::string& what)
        : std::runtime_error(what) {}
};

/// NodeKind for a raw ChakraProtoMsg::NodeType value (1..7); Invalid outside.
/// Parser-only since C1: the commit assembly deliberately compares the raw
/// node_type numerically (GraphBatchCommitter), so this mapping has a single
/// consumer (parse_graph_batch) plus the tests; kept next to NodeKind so the
/// typed parse can never disagree with the enum's numeric spelling.
inline NodeKind node_kind_from_type(uint64_t type) {
    switch (type) {
        case 1:
            return NodeKind::Metadata;
        case 2:
            return NodeKind::MemLoad;
        case 3:
            return NodeKind::MemStore;
        case 4:
            return NodeKind::Compute;
        case 5:
            return NodeKind::CommSend;
        case 6:
            return NodeKind::CommRecv;
        case 7:
            return NodeKind::CommCollective;
        default:
            return NodeKind::Invalid;
    }
}

/// KV-cache route transfer leg (restore memory-anchor selection; P1 fix,
/// 2026-09-28). The Python producer's paired KV transfers (graph_batch_
/// builder -> _paired_transfer in generate_wsc_llm_trace.py) emit comm legs
/// named exactly
///   "<prefix>_history_kv_send_rank{S}_to_rank{T}"           (+ _recv_rank)
///   "<prefix>_prefill_to_decode_kv_send_rank{S}_to_rank{T}" (+ _recv_rank)
/// with S/T decimal ranks, and those are the ONLY nodes the restore memory
/// anchor (event code 7) may bind to (doc sec.4.3). The former selection was
/// a bare name.find("kv") substring match: the aggregated compute pass emits
/// "..._all_layers_attention_qkv_projection" comp nodes that share the "kv"
/// substring and, inside the same (request_id, rank) context, are emitted
/// AFTER the transfer legs -- they stole the last_transfer slot, so every
/// run registered the restore anchor on a compute node and transfer-free
/// (turn-0 / RECOMPUTE / REUSE) requests got a fabricated anchor. Kept next
/// to node_kind_from_type so the frontend selection and the tests can never
/// disagree on what counts as a KV transfer leg.
inline bool is_kv_route_transfer_leg(const OnlineNode& node) {
    if (node.kind != NodeKind::CommSend && node.kind != NodeKind::CommRecv) {
        return false;  // transfer legs are comm nodes; a "qkv" comp never is
    }
    static const char* const kKvRoutes[] = {"_history_kv",
                                            "_prefill_to_decode_kv"};
    static const char* const kLegs[] = {"_send_rank", "_recv_rank"};
    for (const char* route : kKvRoutes) {
        for (const char* leg : kLegs) {
            const std::string marker = std::string(route) + leg;
            const auto pos = node.name.find(marker);
            if (pos == std::string::npos) {
                continue;
            }
            // The leg must CLOSE the name: marker, source rank digits, the
            // literal "_to_rank", target rank digits, end of string.
            size_t i = pos + marker.size();
            size_t source_digits = 0;
            while (i < node.name.size() && node.name[i] >= '0' &&
                   node.name[i] <= '9') {
                ++i;
                ++source_digits;
            }
            if (source_digits == 0 ||
                node.name.compare(i, 8, "_to_rank") != 0) {
                continue;
            }
            i += 8;
            size_t target_digits = 0;
            while (i < node.name.size() && node.name[i] >= '0' &&
                   node.name[i] <= '9') {
                ++i;
                ++target_digits;
            }
            if (target_digits > 0 && i == node.name.size()) {
                return true;
            }
        }
    }
    return false;
}

/// One parsed node: the response's per-rank json id (the affine-stream
/// anchor) plus the fully assembled OnlineNode (global_id == 0; NodeStore
/// assigns the store id at commit).
struct ParsedNode {
    uint64_t json_id = 0;
    OnlineNode node;
};

/// One parent edge. from may reference an EARLIER batch's json id; to must
/// reference a node of this batch (resolution stays in the preflight).
struct ParsedEdge {
    int rank = -1;
    uint64_t from_json = 0;
    uint64_t to_json = 0;
};

/// One watch member, kept in the JSON object key order (LEXICOGRAPHIC
/// rank-string order -- "10" before "2" at rank >= 10; identical iteration
/// order to the old DOM walk, never assumed numeric).
struct ParsedWatchMember {
    int rank = -1;
    uint64_t json_id = 0;
};

struct ParsedWatch {
    std::string request_id;
    std::string stage;  // "prefill" | "decode"
    uint64_t generation = 0;
    std::vector<ParsedWatchMember> members;    // JSON key order
    std::vector<NodeTerminalStatus> statuses;  // raw array order
};

struct ParsedAlarm {
    uint64_t arrival_world_ns = 0;
    RequestEnvelope envelope;  // queue_index defaults to -1
};

/// Half-opaque entries (Python-authoritative): only the fields the C++
/// structural rules need are extracted; unknown sibling keys are tolerated
/// by design (rule S1/S2, GraphBatchCommitter.hh contract).
struct ParsedAssignment {
    std::string request_id;
    int64_t prefill_instance_index = -1;
    int64_t decode_instance_index = -1;
};

struct ParsedKvAction {
    std::string event_type;
    std::string trigger_request_id;
};

/// The typed GraphBatch. Aliased as GraphBatch in DecisionBridge.hh (the
/// pre-C1 struct name survives so main_online/fixtures keep compiling);
/// this is the canonical type.
struct ParsedGraphBatch {
    uint32_t schema_version = 1;
    uint64_t batch_id = 0;
    uint64_t source_delivery_sequence = 0;
    std::vector<ParsedNode> nodes;              // array order (rule O1)
    std::vector<ParsedEdge> parent_edges;       // array order
    std::vector<ParsedWatch> watches;           // array order
    std::vector<ParsedAssignment> assignments;  // array order (counted)
    std::vector<ParsedKvAction> kv_actions;     // array order (counted)
    std::vector<ParsedAlarm> future_alarms;     // array order
    std::vector<int> touched_ranks;             // empty when absent
    bool has_touched_ranks = false;
    std::string error;  // non-empty => decision failure (bridge aborts)
    std::vector<WorkspaceCapacityRank> workspace_capacity_snapshot;
    bool has_workspace_capacity_snapshot = false;
};

/// Parse one response DOM into the typed batch. num_ranks >= 0 enables the
/// rank-domain checks (N3/E2/W3/T8); -1 skips them (legacy fixtures).
/// Throws ParseError on any structural violation; constructs nothing shared
/// (pure function, exception-safe).
ParsedGraphBatch parse_graph_batch(
    const nlohmann::json& resp, int num_ranks,
    int expected_schema_version = 1,
    const std::vector<WorkspaceRankState>& expected_workspace_state = {});

}  // namespace ExecutionDriven
}  // namespace AstraSim

#endif  // EXECUTION_DRIVEN_PARSEGRAPHBATCH_HH
