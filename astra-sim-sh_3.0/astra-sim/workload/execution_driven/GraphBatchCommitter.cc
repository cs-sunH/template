/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

GraphBatchCommitter -- execution-driven mechanism layer (sh30 (wscllm-blueprint) phase 5).
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

// S1 (2026-08-23): shared empty-object stand-in. The previous
// <json>.value(key, nlohmann::json::object()) calls deep-copied the
// sub-object for every node / watch / alarm of every batch; binding a const
// reference to the found element (or to this shared empty object when the
// key is absent) keeps every downstream is_object()/empty()/value() call
// operating on the exact same value -- identical defaults, identical
// error paths -- while removing the copies. Read-only after static
// initialization.
const nlohmann::json kEmptyObject = nlohmann::json::object();

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
            // sh_3.0: "completion" = the third emission boundary (contract ①)
            if (stage != "prefill" && stage != "decode"
                && stage != "completion") {
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
            // S1 (2026-08-23): bind the sub-objects by const reference
            // instead of the previous per-node deep copies
            // (node.value(key, json::object())); absent keys bind the
            // shared empty object, so every check below sees the exact same
            // value the old copy produced.
            const auto compute_it = node.find("compute");
            const nlohmann::json& compute =
                compute_it == node.end() ? kEmptyObject : *compute_it;
            if (!compute.is_object()) {
                return "node[" + std::to_string(node_index) +
                       "] compute not an object";
            }
            (void)compute.value("num_ops", uint64_t{0});
            (void)compute.value("tensor_size", uint64_t{0});
            (void)compute.value("runtime_ns", uint64_t{0});
            const auto mem_it = node.find("mem");
            const nlohmann::json& mem =
                mem_it == node.end() ? kEmptyObject : *mem_it;
            (void)mem.value("tensor_size", uint64_t{0});
            (void)mem.value("is_local_hbm_kv_restore", false);
            (void)mem.value("hbm_access_mode", int{0});
            const auto comm_it = node.find("comm");
            const nlohmann::json& comm =
                comm_it == node.end() ? kEmptyObject : *comm_it;
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
            const auto coll_it = node.find("coll");
            const nlohmann::json& coll =
                coll_it == node.end() ? kEmptyObject : *coll_it;
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
            // S1: single find for involved_dim (was contains + operator[]
            // re-lookup per access).
            const auto involved_it = coll.find("involved_dim");
            if (involved_it != coll.end()) {
                if (!involved_it->is_array()) {
                    return "node[" + std::to_string(node_index) +
                           "] coll.involved_dim not an array";
                }
                for (const auto& dim : *involved_it) {
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
            // M2 node GC (2026-08-23): a from below the per-rank prune
            // watermark is a committed-then-collected parent (collected =>
            // finished => non-blocking, the NodeStore dead-parent rule).
            // Per-rank json ids are dense from 0 (graph_batch_builder's
            // next_id counter), so below-watermark exactly characterizes
            // "was committed and has since been pruned"; a never-emitted id
            // stays unresolved (fail-closed). With GC off the watermark is
            // always 0 and this reduces to the previous store_ids_ check.
            const uint64_t pruned_watermark =
                rank < static_cast<int>(pruned_json_watermark_.size())
                    ? pruned_json_watermark_[rank]
                    : 0;
            if (!from_in_batch &&
                store_ids_.count(RankNodeKey{rank, from}) == 0 &&
                from >= pruned_watermark) {
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
            // S1 (2026-08-23): pre-size the Kahn scratch maps (capacity
            // only; iteration order and results unchanged).
            adj.reserve(rank_entry.second.size());
            indeg.reserve(rank_entry.second.size());
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
                // S1 (2026-08-23): reference binding, no sub-object copy.
                const auto comm_it = node.find("comm");
                const nlohmann::json& comm =
                    comm_it == node.end() ? kEmptyObject : *comm_it;
                const auto key = std::make_tuple(
                    comm.value("src", -1), comm.value("dst", -1),
                    comm.value("tag", int64_t{-1}));
                if (type == 5) {
                    pairs[key].first = true;
                } else {
                    pairs[key].second = true;
                }
            } else if (type == 7) {
                const auto coll_it = node.find("coll");
                const nlohmann::json& coll =
                    coll_it == node.end() ? kEmptyObject : *coll_it;
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
            // S1 (2026-08-23): reference binding for members/statuses (was
            // per-watch sub-object copies); absent members binds the shared
            // empty object and fails the empty check below exactly like the
            // old fresh-object default did.
            const auto members_it = watch.find("members");
            const nlohmann::json& members_json =
                members_it == watch.end() ? kEmptyObject : *members_it;
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
            const auto statuses_it = watch.find("statuses");
            const nlohmann::json& statuses_json =
                statuses_it == watch.end() ? kEmptyObject : *statuses_it;
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
            // eligibility (against the delta-facts-first tracking state).
            // 拼 batch 适配(2026-08-22,照母本 sh_1.0 阶段 0 定型版):列车
            // 哨兵 watch(request_id = "batch_train_..." 批命名空间,§3.1
            // "批节点归属 + watch 侧成员表"的哨兵形态)不对应任何单请求,
            // 绕过 in-flight/prefill-drained 资格检查——其成员是哨兵标记
            // 节点,fire 后事件经四类 reason 通道送回 Python 侧按 train_id
            // 核销。
            if (request_id.rfind("batch_train_", 0) == 0) {
                // batch sentinel: train-scoped, no request eligibility.
            } else if (stage == "prefill") {
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
        // sh_3.0 third emission boundary (contract ①) + 拼 batch 适配
        // (2026-08-22): the coverage rule is one-directional here. Every
        // watch's (request_id, stage) must be covered by this batch's nodes
        // (an uncovered watch is always a bug), but a batch may carry nodes
        // whose (request_id, stage) has no new watch -- the sh_3.0 completion
        // batch (completion_evictions + next-turn interval gates, stage
        // "completion") AND, since the iteration-train port, the admission
        // batch (KV actions only; the PREFILL_DRAIN watch lives on the train
        // drain marker) and the train batch itself (shared body nodes in the
        // "batch_train_..." namespace + join/pstart anchor markers carry no
        // decision watch; only drain/exit markers do). The former strict
        // reverse direction (with the completion-stage exception, contract
        // ①/⑤) is subsumed by the one-directional rule; deviation mirrored
        // from the sh_1.0 stage-0 committer relaxation (设计文档 §3.1.2).
        for (const auto& entry : watch_stages) {
            if (node_stages.count(entry) == 0) {
                return "watch (request_id, stage) {" + entry.first + "," +
                       entry.second +
                       "} has no node coverage in this batch";
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
            // S1 (2026-08-23): reference binding (was a per-alarm copy).
            const auto envelope_it = alarm.find("envelope");
            const nlohmann::json& envelope =
                envelope_it == alarm.end() ? kEmptyObject : *envelope_it;
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
            if (in_flight.count(request_id) != 0) {
                return "future_alarm[" + std::to_string(alarm_index) +
                       "] for already in-flight request " + request_id;
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
            // S1 (2026-08-23): reuse this validate pass's own touched set
            // instead of re-walking batch.nodes through
            // compute_touched_ranks(): by this point every node is an
            // object with an in-range rank (both conditions fail-closed in
            // the node pass above), so the sorted-unique rank sets are
            // identical by construction.
            const std::vector<int> computed(touched.begin(), touched.end());
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
        const auto& mem = node_json.value("mem", nlohmann::json::object());
        node.mem.tensor_size = mem.value("tensor_size", uint64_t{0});
        node.mem.is_local_hbm_kv_restore =
            mem.value("is_local_hbm_kv_restore", false);
        node.mem.hbm_access_mode =
            mem.value("hbm_access_mode", int{0});
        const auto& comm = node_json.value("comm", nlohmann::json::object());
        node.comm.bytes = comm.value("bytes", uint64_t{0});
        node.comm.src = comm.value("src", 0);
        node.comm.dst = comm.value("dst", 0);
        node.comm.tag = comm.value("tag", uint32_t{0});
        node.comm.hbm_charge = comm.value("hbm_charge", true);
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
        // M2 node GC (2026-08-23): remember the commit order (== per-rank
        // json-id order: per-rank ids are dense and ascending across
        // batches) for the store_ids_ prune pass at the commit tail.
        if (ctx_.node_gc) {
            if (prune_queues_.empty()) {
                prune_queues_.resize(ctx_.num_ranks);
                pruned_json_watermark_.assign(ctx_.num_ranks, 0);
            }
            prune_queues_[rank].entries.emplace_back(json_id, store_id);
        }
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
        if (to_it == store_ids_.end()) {
            throw std::runtime_error(
                "commit: parent edge references an unknown node id "
                "(rank=" + std::to_string(rank) +
                " from=" + std::to_string(from) +
                " to=" + std::to_string(to) + ")");
        }
        if (from_it == store_ids_.end()) {
            // M2 node GC (2026-08-23): a from below the per-rank prune
            // watermark was committed and has since been collected --
            // collected => finished, and add_dependency already treats a
            // finished parent as non-blocking (the NodeStore dead-parent
            // rule), so skipping the edge here is the hoisted, byte-equal
            // form of that no-op. Any other miss is unreachable
            // post-validate (the fail-closed throw is preserved).
            const uint64_t pruned_watermark =
                rank < static_cast<int>(pruned_json_watermark_.size())
                    ? pruned_json_watermark_[rank]
                    : 0;
            if (from < pruned_watermark) {
                continue;
            }
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

    // ---- M2 node GC (2026-08-23): quiescent-point collection. The issue
    //      pass above has fully returned (no Workload callback holds a
    //      NodeView pointer), and any deferred issue passes it scheduled run
    //      later and only ever touch free -- i.e. unfinished -- nodes, so
    //      erasing finished childless nodes here is invisible to every
    //      holder (see NodeStore::collect_garbage). No-op (structurally
    //      absent) when node_gc is off. ----
    if (ctx_.node_gc) {
        collect_node_garbage();
    }
}

void GraphBatchCommitter::collect_node_garbage() {
    // M2 (2026-08-23): end-of-commit quiescent-point collection (the only
    // caller is commit()'s tail, gated on ctx_.node_gc). Two steps:
    //   1. every per-rank store drains its candidate FIFO -- nodes finished
    //      with no unfinished children are erased;
    //   2. store_ids_ is pruned behind the same watermark: the per-rank
    //      commit-order queue (ascending json ids) advances only while its
    //      head matches the dense prune watermark AND its node was already
    //      collected, so the pruned set is always a strict dense prefix
    //      [0, pruned_json_watermark_[rank]) -- validate()'s watermark
    //      disjunct resolves exactly that prefix and nothing else, and a
    //      head blocked on a live node bounds the retained prefix by the
    //      per-rank in-flight window.
    for (auto& source : *ctx_.graph_sources) {
        source->store().collect_garbage();
    }
    // Bound the prune pass by the queue vector, not num_ranks: a zero-node
    // batch (an accounting epoch) can reach here before the first
    // node-bearing commit lazily sized the queues -- nothing to prune then.
    for (size_t rank = 0; rank < prune_queues_.size(); ++rank) {
        RankPruneQueue& queue = prune_queues_[rank];
        uint64_t& watermark = pruned_json_watermark_[rank];
        const NodeStore& store = (*ctx_.graph_sources)[rank]->store();
        while (queue.head < queue.entries.size()) {
            const auto& entry = queue.entries[queue.head];
            if (entry.first != watermark) {
                break;  // keep the pruned set a dense prefix (defensive)
            }
            if (!store.erased(entry.second)) {
                break;  // node still live (unfinished / children pending)
            }
            store_ids_.erase(
                RankNodeKey{static_cast<int>(rank), entry.first});
            ++watermark;
            ++queue.head;
        }
        if (queue.head == queue.entries.size()) {
            queue.entries.clear();
            queue.head = 0;
        }
    }
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
