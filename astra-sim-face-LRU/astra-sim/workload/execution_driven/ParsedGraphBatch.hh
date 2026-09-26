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
objects iterate keys ascending, i.e. the decimal rank-string sort the
producer emitted). Watch statuses are stored in raw array order with
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
     DecisionBridge.hh contract ②/④);
  T7 a present array must be of array type;
  T8 touched_ranks optional; when present it must be an array of integers
     in [0, num_ranks) (sorted-unique and set-equality stay in validate).
  N1..N15 node rules: exact required 13-key set plus the single optional
     sh_2.0-specific node key is_local_hbm_kv_restore (rule N2),
     rank/id/type/name/flag/string/
     request/stage/generation typing and domains, compute/comm/coll
     sub-object key sets (required + allowed), integer-overflow rejection
     (a JSON integer literal > UINT64_MAX parses as a double and fails
     is_number_integer), negative-value rejection, tag/priority <=
     UINT32_MAX. MEM-node liveness (tensor_size > 0 AND a configured
     remote-memory port) stays in the preflight -- it needs Context.
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
/// Shared by the parser and (pre-C1) the commit assembly; kept here so the
/// typed parse and the committer can never disagree on the mapping.
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

/// One watch member, kept in the JSON object key order (rank-string
/// ascending -- identical iteration order to the old DOM walk).
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
};

/// Parse one response DOM into the typed batch. num_ranks >= 0 enables the
/// rank-domain checks (N3/E2/W3/T8); -1 skips them (legacy fixtures).
/// Throws ParseError on any structural violation; constructs nothing shared
/// (pure function, exception-safe).
ParsedGraphBatch parse_graph_batch(const nlohmann::json& resp, int num_ranks);

}  // namespace ExecutionDriven
}  // namespace AstraSim

#endif  // EXECUTION_DRIVEN_PARSEGRAPHBATCH_HH
