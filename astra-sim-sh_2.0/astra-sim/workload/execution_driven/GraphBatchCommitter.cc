/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

GraphBatchCommitter -- execution-driven mechanism layer (wscllm phase 5).
Two-phase atomic GraphBatch commit (方案 §8.1/§8.2): Phase A validate()
(run the full checklist, touch nothing) then Phase B commit() (add nodes,
edges, watches, alarms, issue the touched ranks, update counters).

The validation rules are the empirical contract of the real 20.csv
first-30s runs -- every rule in the header comment was verified against
strategy (3531 batches) and replay (3491 batches) with zero violations.
*******************************************************************************/

#include "astra-sim/workload/execution_driven/GraphBatchCommitter.hh"

#include "astra-sim/workload/execution_driven/NodeStore.hh"
#include "astra-sim/workload/execution_driven/RequestIngress.hh"
#include "astra-sim/workload/execution_driven/WatchRegistry.hh"

#include <algorithm>
#include <iomanip>
#include <map>
#include <sstream>
#include <stdexcept>
#include <tuple>

namespace AstraSim {
namespace ExecutionDriven {

namespace {

NodeKind node_kind_from_type(const uint64_t type) {
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

std::string stage_generation(const std::string& stage) {
    return stage == "decode" ? "1" : "0";
}

}  // namespace

void GraphBatchCommitter::apply_delta_facts(
    const StateDelta& delta, std::set<std::string>& in_flight,
    std::set<std::string>& prefill_drained) {
    for (const auto& ev : delta.events) {
        switch (ev.reason) {
            case DecisionReason::ARRIVAL:
                in_flight.insert(ev.request_id);
                break;
            case DecisionReason::PREFILL_DRAIN:
                prefill_drained.insert(ev.request_id);
                break;
            case DecisionReason::REQUEST_COMPLETE:
                in_flight.erase(ev.request_id);
                prefill_drained.erase(ev.request_id);
                break;
            case DecisionReason::DECODE_COMPLETION:
            case DecisionReason::RESOURCE_READY:
                // No tracking change (decode completion alone never ends a
                // request; RESOURCE_READY is a reserved bit never produced).
                break;
        }
    }
}

std::vector<int> GraphBatchCommitter::compute_touched_ranks(
    const GraphBatch& batch, const int num_ranks) {
    std::set<int> ranks;
    for (const auto& node : batch.nodes) {
        if (!node.is_object()) {
            continue;  // malformed entries fail validate(); keep this pure
        }
        const int rank = node.value("rank", -1);
        if (rank >= 0 && rank < num_ranks) {
            ranks.insert(rank);
        }
    }
    return std::vector<int>(ranks.begin(), ranks.end());
}

std::optional<std::string> GraphBatchCommitter::validate(
    const StateDelta& delta, const GraphBatch& batch) const {
    try {
        // ---- delta facts first: the batch is validated against the state
        //      AFTER this epoch's arrivals/completions (the commit applies
        //      the same facts before adding anything). Local copies only --
        //      validate() touches no committer state. ----
        std::set<std::string> in_flight = in_flight_;
        std::set<std::string> prefill_drained = prefill_drained_;
        apply_delta_facts(delta, in_flight, prefill_drained);

        // ---- [epoch] ----
        if (batch.batch_id != delta.delivery_sequence) {
            return "batch_id " + std::to_string(batch.batch_id) +
                   " != delivery_sequence " +
                   std::to_string(delta.delivery_sequence);
        }
        if (batch.source_delivery_sequence != delta.delivery_sequence) {
            return "source_delivery_sequence " +
                   std::to_string(batch.source_delivery_sequence) +
                   " != delivery_sequence " +
                   std::to_string(delta.delivery_sequence);
        }
        if (!batch.error.empty()) {
            return "batch carries an error field: " + batch.error;
        }

        // ---- [node] structural pass ----
        std::unordered_map<int, std::set<uint64_t>> batch_ids;
        std::set<std::pair<std::string, std::string>> node_stages;
        std::set<int> touched;
        uint64_t node_count = 0;
        uint64_t node_index = 0;
        for (const auto& node : batch.nodes) {
            if (!node.is_object()) {
                return "node[" + std::to_string(node_index) +
                       "] is not an object";
            }
            const int rank = node.value("rank", -1);
            if (rank < 0 || rank >= ctx_.num_ranks) {
                return "node[" + std::to_string(node_index) +
                       "] rank out of range: " + std::to_string(rank);
            }
            const uint64_t id = node.value("id", uint64_t(-1));
            if (id == uint64_t(-1)) {
                return "node[" + std::to_string(node_index) +
                       "] missing/invalid id";
            }
            if (!batch_ids[rank].insert(id).second) {
                return "node[" + std::to_string(node_index) +
                       "] duplicate id " + std::to_string(id) + " on rank " +
                       std::to_string(rank);
            }
            const uint64_t type = node.value("type", uint64_t{0});
            if (type < 1 || type > 7) {
                return "node[" + std::to_string(node_index) +
                       "] type out of range: " + std::to_string(type);
            }
            if (!node.value("name", std::string()).size()) {
                return "node[" + std::to_string(node_index) +
                       "] empty name";
            }
            if (!node["is_cpu_op"].is_boolean() ||
                !node["is_timer_op"].is_boolean() ||
                !node["inputs_values"].is_string()) {
                return "node[" + std::to_string(node_index) +
                       "] is_cpu_op/is_timer_op/inputs_values malformed";
            }
            const std::string request_id =
                node.value("request_id", std::string());
            if (request_id.empty()) {
                return "node[" + std::to_string(node_index) +
                       "] empty request_id";
            }
            const std::string stage = node.value("stage", std::string());
            if (stage != "prefill" && stage != "decode") {
                return "node[" + std::to_string(node_index) +
                       "] invalid stage: " + stage;
            }
            const uint64_t generation = node.value("generation", uint64_t{0});
            const uint64_t expected_generation =
                stage == "prefill" ? 0 : 1;
            if (generation != expected_generation) {
                return "node[" + std::to_string(node_index) +
                       "] generation " + std::to_string(generation) +
                       " does not match stage " + stage + " (expected " +
                       stage_generation(stage) + ")";
            }
            const auto& compute =
                node.value("compute", nlohmann::json::object());
            if (!compute.is_object()) {
                return "node[" + std::to_string(node_index) +
                       "] compute not an object";
            }
            (void)compute.value("num_ops", uint64_t{0});
            (void)compute.value("tensor_size", uint64_t{0});
            (void)compute.value("runtime_ns", uint64_t{0});
            const auto& comm = node.value("comm", nlohmann::json::object());
            if (!comm.is_object()) {
                return "node[" + std::to_string(node_index) +
                       "] comm not an object";
            }
            // The src/dst/tag range checks are scoped to the comm-typed
            // nodes (types 5/6) -- the ONLY nodes whose comm fields are
            // semantically load-bearing. Real batches carry the comm
            // defaults (src=0/dst=0/tag=0) on every node and pass
            // trivially; the same-tick milestone fixture emits compute/
            // metadata nodes with an EMPTY comm {} and must stay legal
            // (empirical contract: send node comm.src == node.rank, recv
            // node comm.dst == node.rank -- rank-ownership is already
            // type-scoped below).
            if (type == 5 || type == 6) {
                const int src = comm.value("src", -1);
                const int dst = comm.value("dst", -1);
                const int64_t tag = comm.value("tag", int64_t{-1});
                if (src < 0 || src >= ctx_.num_ranks || dst < 0 ||
                    dst >= ctx_.num_ranks) {
                    return "node[" + std::to_string(node_index) +
                           "] comm src/dst out of range: src=" +
                           std::to_string(src) +
                           " dst=" + std::to_string(dst);
                }
                if (tag < 0) {
                    return "node[" + std::to_string(node_index) +
                           "] comm tag out of range: " +
                           std::to_string(tag);
                }
                if (type == 5 && src != rank) {
                    return "node[" + std::to_string(node_index) +
                           "] send node rank " + std::to_string(rank) +
                           " != comm.src " + std::to_string(src);
                }
                if (type == 6 && dst != rank) {
                    return "node[" + std::to_string(node_index) +
                           "] recv node rank " + std::to_string(rank) +
                           " != comm.dst " + std::to_string(dst);
                }
            }
            const auto& coll = node.value("coll", nlohmann::json::object());
            if (!coll.is_object()) {
                return "node[" + std::to_string(node_index) +
                       "] coll not an object";
            }
            (void)coll.value("comm_type", uint64_t{0});
            (void)coll.value("bytes", uint64_t{0});
            (void)coll.value("priority", uint32_t{0});
            if (!coll.value("pg_name", std::string()).size() && type == 7) {
                return "node[" + std::to_string(node_index) +
                       "] collective with empty pg_name";
            }
            if (coll.contains("involved_dim")) {
                if (!coll["involved_dim"].is_array()) {
                    return "node[" + std::to_string(node_index) +
                           "] coll.involved_dim not an array";
                }
                for (const auto& dim : coll["involved_dim"]) {
                    (void)dim.get<bool>();
                }
            }
            node_stages.insert({request_id, stage});
            touched.insert(rank);
            ++node_count;
            ++node_index;
        }

        // ---- [edge] structural pass (collect in-batch edges per rank for
        //      the cycle check; the child endpoint must be a node of THIS
        //      batch, the parent endpoint may be a node of an EARLIER batch
        //      -- the persistent (rank, json id) -> store id map). ----
        std::unordered_map<int, std::vector<std::pair<uint64_t, uint64_t>>>
            in_batch_edges;
        uint64_t edge_index = 0;
        for (const auto& edge : batch.parent_edges) {
            if (!edge.is_object()) {
                return "parent_edge[" + std::to_string(edge_index) +
                       "] is not an object";
            }
            const int rank = edge.value("rank", -1);
            if (rank < 0 || rank >= ctx_.num_ranks) {
                return "parent_edge[" + std::to_string(edge_index) +
                       "] rank out of range: " + std::to_string(rank);
            }
            const std::string kind = edge.value("kind", std::string());
            if (kind != "data") {
                return "parent_edge[" + std::to_string(edge_index) +
                       "] unsupported kind: " + kind +
                       " (only \"data\" in phase 5)";
            }
            const uint64_t from = edge.value("from", uint64_t(-1));
            const uint64_t to = edge.value("to", uint64_t(-1));
            if (from == uint64_t(-1) || to == uint64_t(-1)) {
                return "parent_edge[" + std::to_string(edge_index) +
                       "] missing endpoint";
            }
            if (from == to) {
                return "parent_edge[" + std::to_string(edge_index) +
                       "] self-loop on rank " + std::to_string(rank);
            }
            const bool from_in_batch = batch_ids[rank].count(from) > 0;
            if (!from_in_batch &&
                store_ids_.count(RankNodeKey{rank, from}) == 0) {
                return "parent_edge[" + std::to_string(edge_index) +
                       "] parent (rank=" + std::to_string(rank) + " from=" +
                       std::to_string(from) + ") unresolved";
            }
            if (batch_ids[rank].count(to) == 0) {
                return "parent_edge[" + std::to_string(edge_index) +
                       "] child (rank=" + std::to_string(rank) + " to=" +
                       std::to_string(to) +
                       ") not a node of this batch";
            }
            if (from_in_batch) {
                in_batch_edges[rank].push_back({from, to});
            }
            ++edge_index;
        }

        // ---- [cycle] per-rank Kahn over the in-batch edges ----
        for (const auto& rank_entry : batch_ids) {
            const int rank = rank_entry.first;
            std::unordered_map<uint64_t, std::vector<uint64_t>> adj;
            std::unordered_map<uint64_t, uint64_t> indeg;
            for (const uint64_t id : rank_entry.second) {
                adj[id];
                indeg[id] = 0;
            }
            for (const auto& edge : in_batch_edges[rank]) {
                adj[edge.first].push_back(edge.second);
                indeg[edge.second] += 1;
            }
            std::vector<uint64_t> queue;
            for (const auto& entry : indeg) {
                if (entry.second == 0) {
                    queue.push_back(entry.first);
                }
            }
            uint64_t visited = 0;
            while (!queue.empty()) {
                const uint64_t cur = queue.back();
                queue.pop_back();
                ++visited;
                for (const uint64_t child : adj[cur]) {
                    if (--indeg[child] == 0) {
                        queue.push_back(child);
                    }
                }
            }
            if (visited != rank_entry.second.size()) {
                return "cycle among the in-batch parent edges of rank " +
                       std::to_string(rank);
            }
        }

        // ---- [comm] send/recv pairing + collective group completeness ----
        std::map<std::tuple<int, int, int64_t>, std::pair<bool, bool>> pairs;
        std::map<std::pair<std::string, std::string>, std::set<int>>
            coll_groups;
        std::map<std::string, std::set<int>> pg_ranks;
        for (const auto& node : batch.nodes) {
            const int rank = node.value("rank", -1);
            const uint64_t type = node.value("type", uint64_t{0});
            if (type == 5 || type == 6) {
                const auto& comm =
                    node.value("comm", nlohmann::json::object());
                const auto key = std::make_tuple(
                    comm.value("src", -1), comm.value("dst", -1),
                    comm.value("tag", int64_t{-1}));
                if (type == 5) {
                    pairs[key].first = true;
                } else {
                    pairs[key].second = true;
                }
            } else if (type == 7) {
                const auto& coll =
                    node.value("coll", nlohmann::json::object());
                const std::string pg =
                    coll.value("pg_name", std::string());
                const std::string name =
                    node.value("name", std::string());
                coll_groups[{pg, name}].insert(rank);
                pg_ranks[pg].insert(rank);
            }
        }
        for (const auto& entry : pairs) {
            if (!entry.second.first || !entry.second.second) {
                return "send/recv pair (src=" +
                       std::to_string(std::get<0>(entry.first)) + " dst=" +
                       std::to_string(std::get<1>(entry.first)) + " tag=" +
                       std::to_string(std::get<2>(entry.first)) +
                       ") incomplete within the batch";
            }
        }
        for (const auto& entry : coll_groups) {
            const std::string& pg = entry.first.first;
            if (entry.second != pg_ranks[pg]) {
                return "collective group " + entry.first.second +
                       " of pg_name " + pg +
                       " incomplete within the batch (group ranks differ "
                       "from the batch's " +
                       pg + " collective ranks -- a split collective fails "
                            "closed)";
            }
        }

        // ---- [watch] structural + coverage + eligibility ----
        std::set<std::pair<std::string, std::string>> watch_stages;
        std::set<std::tuple<std::string, std::string, uint64_t>>
            watch_identities;
        uint64_t watch_index = 0;
        for (const auto& watch : batch.watches) {
            if (!watch.is_object()) {
                return "watch[" + std::to_string(watch_index) +
                       "] is not an object";
            }
            const std::string request_id =
                watch.value("request_id", std::string());
            if (request_id.empty()) {
                return "watch[" + std::to_string(watch_index) +
                       "] empty request_id";
            }
            const std::string stage = watch.value("stage", std::string());
            if (stage != "prefill" && stage != "decode") {
                return "watch[" + std::to_string(watch_index) +
                       "] invalid stage: " + stage;
            }
            const uint64_t generation = watch.value("generation", uint64_t{0});
            const uint64_t expected_generation =
                stage == "prefill" ? 0 : 1;
            if (generation != expected_generation) {
                return "watch[" + std::to_string(watch_index) +
                       "] generation " + std::to_string(generation) +
                       " does not match stage " + stage;
            }
            if (!watch_identities
                     .insert({request_id, stage, generation})
                     .second) {
                return "watch[" + std::to_string(watch_index) +
                       "] duplicate identity (request_id, stage, generation) "
                       "in the batch";
            }
            watch_stages.insert({request_id, stage});
            const auto& members_json =
                watch.value("members", nlohmann::json::object());
            if (!members_json.is_object() || members_json.empty()) {
                return "watch[" + std::to_string(watch_index) +
                       "] empty/absent members";
            }
            for (auto it = members_json.begin(); it != members_json.end();
                 ++it) {
                int rank = -1;
                try {
                    rank = std::stoi(it.key());
                } catch (const std::exception&) {
                    return "watch[" + std::to_string(watch_index) +
                           "] unparsable member rank: " + it.key();
                }
                if (rank < 0 || rank >= ctx_.num_ranks) {
                    return "watch[" + std::to_string(watch_index) +
                           "] member rank out of range: " + it.key();
                }
                const uint64_t member_id = it.value().get<uint64_t>();
                if (batch_ids[rank].count(member_id) == 0) {
                    return "watch[" + std::to_string(watch_index) +
                           "] member (rank=" + it.key() + " id=" +
                           std::to_string(member_id) +
                           ") is not a node of this batch";
                }
            }
            const auto& statuses_json =
                watch.value("statuses", nlohmann::json::array());
            if (!statuses_json.is_array() || statuses_json.empty()) {
                return "watch[" + std::to_string(watch_index) +
                       "] empty/absent statuses";
            }
            for (const auto& status_json : statuses_json) {
                const std::string name = status_json.get<std::string>();
                if (name != "Success" && name != "Skipped") {
                    return "watch[" + std::to_string(watch_index) +
                           "] unknown status: " + name;
                }
            }
            // eligibility (against the delta-facts-first tracking state)
            if (stage == "prefill") {
                if (in_flight.count(request_id) == 0) {
                    return "prefill watch[" + std::to_string(watch_index) +
                           "] for request " + request_id +
                           " not in-flight at this epoch (the request must "
                           "have arrived at this or an earlier delivery)";
                }
            } else {
                if (prefill_drained.count(request_id) == 0) {
                    return "decode watch[" + std::to_string(watch_index) +
                           "] for request " + request_id +
                           " whose prefill has not drained at this or an "
                           "earlier epoch";
                }
            }
            ++watch_index;
        }
        // sh_2.0 multi-segment emission extension (contract ①/④): every
        // watch's (request, stage) must be covered by this batch's nodes,
        // but a batch may ALSO emit nodes for a (request, stage) whose watch
        // already fired at an earlier epoch -- sh_2.0 splits each request
        // into prefill / decode / completion segments (completion evictions
        // + next-turn interval gates are decode-stage nodes committed at the
        // REQUEST_COMPLETE boundary, after the decode watch fired). Such
        // node-only segments are admitted when the tracking state proves the
        // stage boundary already passed (decode: prefill_drained + watch
        // already fired, i.e. the request is no longer expected to produce a
        // new watch; prefill: in-flight arrival tracking).
        for (const auto& watch_key : watch_stages) {
            if (node_stages.count(watch_key) == 0) {
                return "watch (request_id, stage) not covered by any node of "
                       "this batch";
            }
        }
        // Requests completed in THIS delta: their completion segment is
        // committed at exactly this epoch (REQUEST_COMPLETE boundary), after
        // apply_delta_facts already removed them from the tracking sets.
        std::set<std::string> completed_this_epoch;
        for (const auto& ev : delta.events) {
            if (ev.reason == DecisionReason::REQUEST_COMPLETE) {
                completed_this_epoch.insert(ev.request_id);
            }
        }
        for (const auto& node_key : node_stages) {
            if (watch_stages.count(node_key) != 0) {
                continue;
            }
            const bool eligible = (node_key.second == "decode")
                ? (prefill_drained.count(node_key.first) != 0 ||
                   completed_this_epoch.count(node_key.first) != 0)
                : (in_flight.count(node_key.first) != 0);
            if (!eligible) {
                return "node (request_id, stage) coverage does not match the "
                       "watch coverage (a batch's nodes and its watches must "
                       "reference the same (request, stage) set; a node-only "
                       "segment requires the stage boundary to have already "
                       "passed)";
            }
        }

        // ---- [assign] (opaque; structural only) ----
        uint64_t assign_index = 0;
        for (const auto& assignment : batch.assignments) {
            if (!assignment.is_object()) {
                return "assignment[" + std::to_string(assign_index) +
                       "] is not an object";
            }
            if (assignment.value("request_id", std::string()).empty()) {
                return "assignment[" + std::to_string(assign_index) +
                       "] empty request_id";
            }
            if (assignment.value("prefill_instance_index", int64_t{-1}) < 0 ||
                assignment.value("decode_instance_index", int64_t{-1}) < 0) {
                return "assignment[" + std::to_string(assign_index) +
                       "] negative instance index";
            }
            ++assign_index;
        }

        // ---- [kv] (opaque to C++; the Python provisional ledger is the
        //      authority -- structural only) ----
        uint64_t kv_index = 0;
        for (const auto& action : batch.kv_actions) {
            if (!action.is_object()) {
                return "kv_action[" + std::to_string(kv_index) +
                       "] is not an object";
            }
            if (action.value("event_type", std::string()).empty() ||
                action.value("trigger_request_id", std::string()).empty()) {
                return "kv_action[" + std::to_string(kv_index) +
                       "] missing event_type/trigger_request_id";
            }
            ++kv_index;
        }

        // ---- [alarm] ----
        std::set<std::string> alarm_ids;
        uint64_t alarm_index = 0;
        for (const auto& alarm : batch.future_alarms) {
            if (!alarm.is_object()) {
                return "future_alarm[" + std::to_string(alarm_index) +
                       "] is not an object";
            }
            const uint64_t arrival =
                alarm.value("arrival_world_ns", uint64_t{0});
            if (arrival < delta.tick) {
                return "future_alarm[" + std::to_string(alarm_index) +
                       "] past arrival_world_ns " +
                       std::to_string(arrival) + " < delta tick " +
                       std::to_string(delta.tick);
            }
            const auto& envelope =
                alarm.value("envelope", nlohmann::json::object());
            if (!envelope.is_object()) {
                return "future_alarm[" + std::to_string(alarm_index) +
                       "] envelope not an object";
            }
            const std::string request_id =
                envelope.value("request_id", std::string());
            if (request_id.empty()) {
                return "future_alarm[" + std::to_string(alarm_index) +
                       "] empty envelope request_id";
            }
            if (!alarm_ids.insert(request_id).second) {
                return "future_alarm[" + std::to_string(alarm_index) +
                       "] duplicate request_id " + request_id +
                       " in the batch";
            }
            // sh_2.0 replay alignment extension: a turn-0 request whose CSV
            // arrival already fired stays in_flight while its prefill segment
            // emission is deferred to an alarm aligned to the offline
            // admission (prefill record) tick -- seconds-scale admission
            // queueing under task-load balancing. Such an alignment alarm is
            // legal while the request is in-flight but NOT yet drained
            // (prefill_drained); a request whose prefill already drained must
            // never receive another arrival alarm.
            if (prefill_drained.count(request_id) != 0) {
                return "future_alarm[" + std::to_string(alarm_index) +
                       "] for request " + request_id +
                       " whose prefill already drained";
            }
            if (envelope.value("session_id", std::string()).empty()) {
                return "future_alarm[" + std::to_string(alarm_index) +
                       "] empty envelope session_id";
            }
            if (envelope.value("turn_index", -1) < 0) {
                return "future_alarm[" + std::to_string(alarm_index) +
                       "] negative envelope turn_index";
            }
            (void)envelope.value("prefill_length", uint64_t{0});
            (void)envelope.value("decode_length", uint64_t{0});
            (void)envelope.value("inter_request_interval_ns", uint64_t{0});
            ++alarm_index;
        }

        // ---- [touched] Python-computed touched_ranks must agree with the
        //      batch's node rank set (sorted unique) ----
        if (batch.has_touched_ranks) {
            if (!batch.touched_ranks.is_array()) {
                return "touched_ranks is not an array";
            }
            std::vector<int> declared;
            for (const auto& rank_json : batch.touched_ranks) {
                if (!rank_json.is_number_integer()) {
                    return "touched_ranks entry is not an integer";
                }
                const int rank = rank_json.get<int>();
                if (rank < 0 || rank >= ctx_.num_ranks) {
                    return "touched_ranks rank out of range: " +
                           std::to_string(rank);
                }
                declared.push_back(rank);
            }
            if (!std::is_sorted(declared.begin(), declared.end()) ||
                std::adjacent_find(declared.begin(), declared.end()) !=
                    declared.end()) {
                return "touched_ranks not sorted unique";
            }
            const std::vector<int> computed =
                compute_touched_ranks(batch, ctx_.num_ranks);
            if (declared != computed) {
                std::ostringstream os;
                os << "touched_ranks mismatch: declared [";
                for (size_t i = 0; i < declared.size(); ++i) {
                    if (i) os << ",";
                    os << declared[i];
                }
                os << "] computed [";
                for (size_t i = 0; i < computed.size(); ++i) {
                    if (i) os << ",";
                    os << computed[i];
                }
                os << "]";
                return os.str();
            }
        }

        return std::nullopt;
    } catch (const std::exception& exc) {
        // Malformed entry types surface as nlohmann type errors; convert to
        // a validation error (fail-closed, zero state mutation).
        return std::string("malformed batch entry: ") + exc.what();
    }
}

void GraphBatchCommitter::commit(const StateDelta& delta,
                                 const GraphBatch& batch) {
    // ---- delta facts first (the same facts validate() used) ----
    apply_delta_facts(delta, in_flight_, prefill_drained_);

    // ---- Phase B-1: nodes (persistent (rank, json id) -> store id) ----
    uint64_t node_count = 0;
    for (const auto& node_json : batch.nodes) {
        if (!node_json.is_object()) {
            throw std::runtime_error(
                "commit: malformed node entry (validate() must have "
                "rejected this)");
        }
        const int rank = node_json.value("rank", -1);
        const uint64_t json_id = node_json.value("id", uint64_t(-1));
        if (rank < 0 || rank >= static_cast<int>(ctx_.graph_sources->size())) {
            throw std::runtime_error(
                "commit: node rank out of range: " + std::to_string(rank));
        }
        OnlineNode node;
        node.global_id = 0;  // NodeStore assigns a fresh id (ascending, per rank)
        node.rank = rank;
        node.kind = node_kind_from_type(node_json.value("type", 0));
        node.node_type = node_json.value("type", 0);
        node.name = node_json.value("name", std::string());
        node.is_cpu_op = node_json.value("is_cpu_op", false);
        node.is_timer_op = node_json.value("is_timer_op", false);
        // sh_2.0: MEM_LOAD/MEM_STORE restore-DMA routing bit (§4.3.3)
        node.is_local_hbm_kv_restore =
            node_json.value("is_local_hbm_kv_restore", false);
        node.inputs_values = node_json.value("inputs_values", std::string());
        node.request_id = node_json.value("request_id", std::string());
        node.stage = node_json.value("stage", std::string());
        node.generation = node_json.value("generation", 0);
        const auto& compute =
            node_json.value("compute", nlohmann::json::object());
        // NOTE: value() deduces the conversion type from the default literal.
        // An int default (0) truncates values >= 2^31 to a signed 32-bit int
        // (sign-extended into the uint64 field) -- kernel num_ops / tensor
        // sizes are routinely > 2^31, and issue_comp then computes a garbage
        // runtime and the node never completes. Always use explicit uint64
        // defaults for 64-bit fields.
        node.compute.num_ops = compute.value("num_ops", uint64_t{0});
        node.compute.tensor_size = compute.value("tensor_size", uint64_t{0});
        node.compute.runtime_ns = compute.value("runtime_ns", uint64_t{0});
        if (compute.contains("remote_weight_bytes")) {
            node.compute.has_remote_weight_bytes = true;
            node.compute.remote_weight_bytes =
                compute.value("remote_weight_bytes", uint64_t{0});
        }
        const auto& comm = node_json.value("comm", nlohmann::json::object());
        node.comm.bytes = comm.value("bytes", uint64_t{0});
        node.comm.src = comm.value("src", 0);
        node.comm.dst = comm.value("dst", 0);
        node.comm.tag = comm.value("tag", uint32_t{0});
        const auto& coll = node_json.value("coll", nlohmann::json::object());
        node.coll.comm_type = coll.value("comm_type", uint64_t{0});
        node.coll.bytes = coll.value("bytes", uint64_t{0});
        node.coll.priority = coll.value("priority", uint32_t{0});
        node.coll.pg_name = coll.value("pg_name", std::string());
        if (coll.contains("involved_dim") && coll["involved_dim"].is_array()) {
            for (const auto& dim : coll["involved_dim"]) {
                node.coll.involved_dim.push_back(dim.get<bool>());
            }
        }
        const uint64_t store_id =
            (*ctx_.graph_sources)[rank]->store().add_node(std::move(node));
        store_ids_[RankNodeKey{rank, json_id}] = store_id;
        ++node_count;
    }

    // ---- Phase B-1.5: side-band metrics anchor hook (phase-7 §10.3). The
    //      anchors must bind to the STORE ids (what on_node_issue /
    //      on_node_complete observe) -- NodeStore hands out store ids
    //      starting at 1 while the online graph's json ids start at 0, so
    //      registering with the json ids off-by-ones every anchor (the id-0
    //      start anchors would never fire at all). The caller's hook receives
    //      the full (rank, json id) -> store id map of this commit and must
    //      register BEFORE the issue pass below lets any of these nodes run.
    if (ctx_.metrics_anchor_hook) {
        ctx_.metrics_anchor_hook(batch, store_ids_);
    }

    // ---- Phase B-2: parent_edges -> add_dependency (Data kind) ----
    for (const auto& edge : batch.parent_edges) {
        if (!edge.is_object()) {
            throw std::runtime_error(
                "commit: malformed parent_edge entry");
        }
        const int rank = edge.value("rank", -1);
        const uint64_t from = edge.value("from", uint64_t(-1));
        const uint64_t to = edge.value("to", uint64_t(-1));
        const auto from_it = store_ids_.find(RankNodeKey{rank, from});
        const auto to_it = store_ids_.find(RankNodeKey{rank, to});
        if (from_it == store_ids_.end() || to_it == store_ids_.end()) {
            throw std::runtime_error(
                "commit: parent edge references an unknown node id "
                "(rank=" + std::to_string(rank) +
                " from=" + std::to_string(from) +
                " to=" + std::to_string(to) + ")");
        }
        (*ctx_.graph_sources)[rank]->store().add_dependency(
            from_it->second, to_it->second, DepKind::Data);
    }

    // ---- Phase B-3: register watches (member ids translated to store ids) ----
    for (const auto& watch : batch.watches) {
        if (!watch.is_object()) {
            throw std::runtime_error("commit: malformed watch entry");
        }
        const std::string request_id =
            watch.value("request_id", std::string());
        const std::string stage = watch.value("stage", std::string());
        const uint64_t generation = watch.value("generation", uint64_t{0});
        std::set<CompletionKey> members;
        const auto& members_json =
            watch.value("members", nlohmann::json::object());
        for (auto it = members_json.begin(); it != members_json.end(); ++it) {
            const int rank = std::stoi(it.key());
            const uint64_t json_id = it.value().get<uint64_t>();
            const auto id_it = store_ids_.find(RankNodeKey{rank, json_id});
            if (id_it == store_ids_.end()) {
                throw std::runtime_error(
                    "commit: watch member references an unknown node id "
                    "(rank=" + it.key() +
                    " id=" + std::to_string(json_id) + ")");
            }
            members.insert(CompletionKey{rank, id_it->second, generation});
        }
        std::set<NodeTerminalStatus> statuses;
        const auto& statuses_json =
            watch.value("statuses", nlohmann::json::array());
        for (const auto& status_json : statuses_json) {
            const std::string name = status_json.get<std::string>();
            if (name == "Success") {
                statuses.insert(NodeTerminalStatus::Success);
            } else if (name == "Skipped") {
                statuses.insert(NodeTerminalStatus::Skipped);
            } else {
                throw std::runtime_error(
                    "commit: unknown watch status: " + name);
            }
        }
        ctx_.watch_registry->register_stage_watch(
            request_id, stage, generation, std::move(members),
            std::move(statuses));
    }

    // ---- Phase B-4: schedule future arrival alarms ----
    for (const auto& alarm : batch.future_alarms) {
        if (!alarm.is_object()) {
            throw std::runtime_error("commit: malformed future_alarm entry");
        }
        const auto& envelope_json =
            alarm.value("envelope", nlohmann::json::object());
        RequestEnvelope envelope;
        envelope.request_id =
            envelope_json.value("request_id", std::string());
        envelope.session_id =
            envelope_json.value("session_id", std::string());
        envelope.turn_index = envelope_json.value("turn_index", 0);
        envelope.prefill_length =
            envelope_json.value("prefill_length", uint64_t{0});
        envelope.decode_length =
            envelope_json.value("decode_length", uint64_t{0});
        envelope.inter_request_interval_ns =
            envelope_json.value("inter_request_interval_ns", uint64_t{0});
        envelope.arrival_world_ns =
            alarm.value("arrival_world_ns", uint64_t{0});
        ctx_.ingress->schedule_future_arrival(envelope);
    }

    // ---- Phase B-5: issue pass over the TOUCHED ranks only (ranks without
    //      new nodes cannot have new free nodes -- the completion hook's
    //      deferred per-rank passes drain every other rank) ----
    for (const int rank : compute_touched_ranks(batch, ctx_.num_ranks)) {
        ctx_.issue_rank(rank);
    }

    // ---- counters ----
    ++counters_.graph_batch_count;
    if (node_count == 1) {
        ++counters_.single_node_bridge_count;
    }
    counters_.total_nodes += node_count;
    counters_.max_nodes_per_batch =
        std::max(counters_.max_nodes_per_batch, node_count);
    counters_.total_watches += batch.watches.size();
    counters_.total_assignments += batch.assignments.size();
    counters_.total_kv_actions += batch.kv_actions.size();
    counters_.total_future_alarms += batch.future_alarms.size();
}

std::string GraphBatchCommitter::counters_report() const {
    std::ostringstream os;
    os << "graph_batch_count=" << counters_.graph_batch_count
       << " single_node_bridge_count=" << counters_.single_node_bridge_count
       << " total_nodes=" << counters_.total_nodes << " avg_nodes_per_batch=";
    if (counters_.graph_batch_count > 0) {
        os << std::fixed << std::setprecision(2)
           << static_cast<double>(counters_.total_nodes) /
                  static_cast<double>(counters_.graph_batch_count);
    } else {
        os << "0";
    }
    os << " max_nodes_per_batch=" << counters_.max_nodes_per_batch
       << " total_watches=" << counters_.total_watches
       << " total_assignments=" << counters_.total_assignments
       << " total_kv_actions=" << counters_.total_kv_actions
       << " total_future_alarms=" << counters_.total_future_alarms;
    return os.str();
}

}  // namespace ExecutionDriven
}  // namespace AstraSim
