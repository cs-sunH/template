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
#include <chrono>
#include <iomanip>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>
#include <tuple>
#include <unordered_map>
#include <unordered_set>

namespace AstraSim {
namespace ExecutionDriven {

namespace {

// C1 (2026-08-29): node_kind_from_type moved to ParsedGraphBatch.hh (the
// parser and the committer must share ONE mapping); stage_generation stays
// (a validate_impl diagnostic formatter). The S1 shared empty-object
// stand-ins are gone with the DOM walks -- every consumer reads typed
// fields now.

std::string stage_generation(const std::string& stage) {
    return stage == "decode" ? "1" : "0";
}

constexpr size_t kNoPreflightEdge = std::numeric_limits<size_t>::max();

struct P2pKey {
    int src = -1;
    int dst = -1;
    int64_t tag = -1;
    uint64_t bytes = 0;

    bool operator==(const P2pKey& other) const {
        return src == other.src && dst == other.dst && tag == other.tag &&
               bytes == other.bytes;
    }
};

struct P2pKeyHash {
    size_t operator()(const P2pKey& key) const noexcept {
        size_t result = std::hash<int>{}(key.src);
        const auto combine = [&result](const size_t value) {
            result ^= value + 0x9e3779b9U + (result << 6U) + (result >> 2U);
        };
        combine(std::hash<int>{}(key.dst));
        combine(std::hash<int64_t>{}(key.tag));
        combine(std::hash<uint64_t>{}(key.bytes));
        return result;
    }
};

struct CollectiveKey {
    std::string pg_name;
    std::string name;

    bool operator==(const CollectiveKey& other) const {
        return pg_name == other.pg_name && name == other.name;
    }
};

struct CollectiveKeyHash {
    size_t operator()(const CollectiveKey& key) const noexcept {
        size_t result = std::hash<std::string>{}(key.pg_name);
        result ^= std::hash<std::string>{}(key.name) + 0x9e3779b9U +
                  (result << 6U) + (result >> 2U);
        return result;
    }
};

struct WatchIdentity {
    std::string request_id;
    std::string stage;
    uint64_t generation = 0;

    bool operator==(const WatchIdentity& other) const {
        return request_id == other.request_id && stage == other.stage &&
               generation == other.generation;
    }
};

struct WatchIdentityHash {
    size_t operator()(const WatchIdentity& key) const noexcept {
        size_t result = std::hash<std::string>{}(key.request_id);
        result ^= std::hash<std::string>{}(key.stage) + 0x9e3779b9U +
                  (result << 6U) + (result >> 2U);
        result ^= std::hash<uint64_t>{}(key.generation) + 0x9e3779b9U +
                  (result << 6U) + (result >> 2U);
        return result;
    }
};

struct PreflightEdge {
    size_t to = 0;
    size_t next = kNoPreflightEdge;
};

// C1: the per-rank preflight graph references the parsed nodes instead of
// the response DOM (which no longer exists past deliver_and_receive).
struct PreflightRankGraph {
    std::unordered_map<uint64_t, size_t> node_index_by_json_id;
    std::vector<const ParsedNode*> nodes;
    std::vector<size_t> indegree;
    std::vector<size_t> first_edge;
    std::vector<PreflightEdge> edges;
};

struct P2pCounts {
    uint64_t sends = 0;
    uint64_t recvs = 0;
};

struct CollectiveParticipants {
    std::unordered_map<int, uint32_t> count_by_rank;
    bool has_signature = false;
    uint64_t comm_type = 0;
    uint64_t bytes = 0;
    uint32_t priority = 0;
    // issue_coll_comm() passes this vector into every collective generator;
    // it selects the participating topology dimensions and therefore must be
    // identical for every rank in one logical collective operation.
    std::vector<bool> involved_dims;
};

struct RequestFacts {
    bool in_flight = false;
    bool prefill_drained = false;
};

bool collective_comm_type_has_completion_path(const uint64_t comm_type) {
    // Workload::issue_coll_comm() implements exactly these Chakra enum values.
    // Any other value reaches its unsupported-collective throw after the node
    // has already been taken from a NodeStore, so it is liveness-critical when
    // full validation is off.
    switch (comm_type) {
        case 0:  // ALL_REDUCE
        case 2:  // ALL_GATHER
        case 5:  // BROADCAST
        case 6:  // ALL_TO_ALL
        case 7:  // REDUCE_SCATTER
            return true;
        default:
            return false;
    }
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
                // No tracking change (decode completion alone never ends a
                // request).
                break;
        }
    }
}

std::optional<std::string> GraphBatchCommitter::validate_affine_drift() const {
    if (ctx_.num_ranks < 0) {
        return "negative GraphBatchCommitter rank count";
    }
    if (ctx_.graph_sources == nullptr) {
        return "GraphBatchCommitter has no graph sources";
    }
    const size_t rank_count = static_cast<size_t>(ctx_.num_ranks);
    if (ctx_.graph_sources->size() < rank_count) {
        return "GraphBatchCommitter graph source count " +
               std::to_string(ctx_.graph_sources->size()) +
               " is below num_ranks " + std::to_string(ctx_.num_ranks);
    }
    if (rank_affines_.size() != rank_count) {
        return "GraphBatchCommitter affine metadata has wrong rank count";
    }
    for (size_t rank = 0; rank < rank_count; ++rank) {
        const auto& source = (*ctx_.graph_sources)[rank];
        if (!source) {
            return "GraphBatchCommitter has null graph source for rank " +
                   std::to_string(rank);
        }
        const RankAffine& affine = rank_affines_[rank];
        if (!affine.initialized) {
            continue;
        }
        const uint64_t actual_next = source->store().next_auto_id();
        if (actual_next != affine.store_next) {
            return "NodeStore automatic id stream drift (rank=" +
                   std::to_string(rank) + " expected_next=" +
                   std::to_string(affine.store_next) + " actual_next=" +
                   std::to_string(actual_next) + ")";
        }
    }
    return std::nullopt;
}

std::optional<std::string> GraphBatchCommitter::validate_json_id_stream(
    const GraphBatch& batch, std::vector<int>* const touched_ranks) const {
    // The Python OnlineTraceBuilder owns one next_id counter per rank and
    // emits every id exactly once, in increasing order. Enforcing that
    // producer contract here turns the exact committed history into one
    // bounded [first,next) range per rank. This cheap pass also runs when
    // --online-validate=0, before any delta/store state is touched. It carries
    // the mandatory zero-byte collective safety rule in the same walk: the
    // analytical backend has no valid completion path for such a node.
    if (touched_ranks != nullptr) {
        touched_ranks->clear();
    }
    uint64_t stream_stamp = json_id_stream_stamp_ + 1;
    if (stream_stamp == 0) {
        // Zero is the never-seen marker.  On the (practically unreachable)
        // wrap, clear the fixed scratch once and start a fresh stamp epoch.
        std::fill(json_id_seen_stamp_by_rank_.begin(),
                  json_id_seen_stamp_by_rank_.end(), uint64_t{0});
        stream_stamp = 1;
    }
    json_id_stream_stamp_ = stream_stamp;
    if (touched_ranks != nullptr) {
        json_id_discovered_touched_ranks_.clear();
    }
    uint64_t node_index = 0;
    try {
        // C1 (2026-08-29): typed walk -- the structural shape (object,
        // integer rank/id, coll object) was already guaranteed by
        // parse_graph_batch; the checks kept here are the plain field reads
        // this pass actually owns (id stream contiguity + the zero-byte
        // collective safety rule).
        for (const auto& parsed : batch.nodes) {
            const int rank = parsed.node.rank;
            if (rank < 0 || rank >= ctx_.num_ranks) {
                return "node[" + std::to_string(node_index) +
                       "] rank out of range: " + std::to_string(rank);
            }
            const uint64_t id = parsed.json_id;
            if (id == std::numeric_limits<uint64_t>::max()) {
                return "node[" + std::to_string(node_index) +
                       "] missing/invalid id";
            }

            const uint64_t type = parsed.node.node_type;
            if (type == 7 && parsed.node.coll.bytes == 0) {
                return "node[" + std::to_string(node_index) +
                       "] collective bytes must be positive";
            }

            uint64_t expected = id;  // an arbitrary first id is legal
            if (json_id_seen_stamp_by_rank_[rank] == stream_stamp) {
                expected = json_id_expected_by_rank_[rank];
            } else {
                if (rank < static_cast<int>(rank_affines_.size()) &&
                    rank_affines_[rank].initialized) {
                    expected = rank_affines_[rank].json_next;
                }
                json_id_seen_stamp_by_rank_[rank] = stream_stamp;
                if (touched_ranks != nullptr) {
                    json_id_discovered_touched_ranks_.push_back(rank);
                }
            }
            if (id != expected) {
                return "node[" + std::to_string(node_index) + "] id " +
                       std::to_string(id) + " on rank " +
                       std::to_string(rank) +
                       " breaks the contiguous json-id stream (expected " +
                       std::to_string(expected) + ")";
            }
            json_id_expected_by_rank_[rank] = id + 1;
            ++node_index;
        }
    } catch (const std::exception& exc) {
        return std::string("malformed json-id stream: ") + exc.what();
    }
    if (touched_ranks != nullptr) {
        std::sort(json_id_discovered_touched_ranks_.begin(),
                  json_id_discovered_touched_ranks_.end());
        touched_ranks->insert(touched_ranks->end(),
                              json_id_discovered_touched_ranks_.begin(),
                              json_id_discovered_touched_ranks_.end());
    }
    return std::nullopt;
}

std::optional<std::string> GraphBatchCommitter::mandatory_liveness_preflight(
    const StateDelta& delta, const GraphBatch& batch) const {
    // This is deliberately independent of validate(): production can skip
    // that exhaustive diagnostic pass, but it must never commit a batch
    // that can strand graph nodes, stage watches, p2p callbacks, collectives,
    // or request accounting.  Every container below is local; no fact set,
    // NodeStore, WatchRegistry, ingress queue, counter, or affine record is
    // changed until this returns success.
    const auto fail = [](const std::string& reason)
        -> std::optional<std::string> {
        return std::string("mandatory liveness preflight: ") + reason;
    };

    try {
        // Apply this delta's facts in a compact overlay rather than copying the
        // whole live-request sets.  The result is exactly apply_delta_facts()
        // for every request mentioned by the delta, while untouched requests
        // read straight from the committer's immutable sets.
        std::unordered_map<std::string, RequestFacts> facts_after_delta;
        facts_after_delta.reserve(delta.events.size());
        for (const auto& event : delta.events) {
            auto facts_it = facts_after_delta.find(event.request_id);
            if (facts_it == facts_after_delta.end()) {
                facts_it =
                    facts_after_delta
                        .emplace(event.request_id,
                                 RequestFacts{in_flight_.count(event.request_id) !=
                                                  0,
                                              prefill_drained_.count(
                                                  event.request_id) != 0})
                        .first;
            }
            switch (event.reason) {
                case DecisionReason::ARRIVAL:
                    facts_it->second.in_flight = true;
                    break;
                case DecisionReason::PREFILL_DRAIN:
                    facts_it->second.prefill_drained = true;
                    break;
                case DecisionReason::REQUEST_COMPLETE:
                    facts_it->second.in_flight = false;
                    facts_it->second.prefill_drained = false;
                    break;
                case DecisionReason::DECODE_COMPLETION:
                    break;
            }
        }
        const auto request_facts =
            [this, &facts_after_delta](const std::string& request_id) {
                const auto facts_it = facts_after_delta.find(request_id);
                if (facts_it != facts_after_delta.end()) {
                    return facts_it->second;
                }
                return RequestFacts{in_flight_.count(request_id) != 0,
                                    prefill_drained_.count(request_id) != 0};
            };

        // Build one compact id->index graph per rank.  C1 (2026-08-29): the
        // nodes are the parsed typed entries (referenced, not copied), and
        // each in-batch edge becomes one flat adjacency-list record; that
        // keeps the normal path amortized O(nodes + edges + watches) without
        // one heap allocation per node. The structural shape guarantees the
        // old JSON-is_object/is_boolean lookups used to provide -- the typed
        // reads below are the same rules against the parsed fields.
        std::unordered_map<int, PreflightRankGraph> rank_graphs;
        rank_graphs.reserve(std::min(
            batch.nodes.size(), static_cast<size_t>(std::max(ctx_.num_ranks, 0))));
        std::unordered_map<P2pKey, P2pCounts, P2pKeyHash> p2p_counts;
        p2p_counts.reserve(batch.nodes.size());
        std::unordered_map<CollectiveKey, CollectiveParticipants,
                           CollectiveKeyHash>
            collective_participants;
        collective_participants.reserve(batch.nodes.size());

        uint64_t node_index = 0;
        for (const auto& parsed : batch.nodes) {
            const int rank = parsed.node.rank;
            if (rank < 0 || rank >= ctx_.num_ranks) {
                return fail("node[" + std::to_string(node_index) +
                            "] rank out of range: " + std::to_string(rank));
            }
            const uint64_t json_id = parsed.json_id;
            if (json_id == uint64_t(-1)) {
                return fail("node[" + std::to_string(node_index) +
                            "] missing/invalid id");
            }

            PreflightRankGraph& graph = rank_graphs[rank];
            const auto inserted = graph.node_index_by_json_id.emplace(
                json_id, graph.nodes.size());
            if (!inserted.second) {
                return fail("node[" + std::to_string(node_index) +
                            "] duplicate id " + std::to_string(json_id) +
                            " on rank " + std::to_string(rank));
            }
            graph.nodes.push_back(&parsed);
            graph.indegree.push_back(0);
            graph.first_edge.push_back(kNoPreflightEdge);

            const OnlineNode& node = parsed.node;
            const uint64_t type = node.node_type;
            if (type < 1 || type > 7) {
                return fail("node[" + std::to_string(node_index) +
                            "] type out of range: " +
                            std::to_string(type));
            }
            if (node.name.empty()) {
                return fail("node[" + std::to_string(node_index) +
                            "] empty name");
            }
            const bool is_cpu_op = node.is_cpu_op;
            const std::string& request_id = node.request_id;
            if (request_id.empty()) {
                return fail("node[" + std::to_string(node_index) +
                            "] empty request_id");
            }
            const std::string& stage = node.stage;
            if (stage != "prefill" && stage != "decode") {
                return fail("node[" + std::to_string(node_index) +
                            "] invalid stage: " + stage);
            }
            const uint64_t generation = node.generation;
            const uint64_t expected_generation =
                stage == "prefill" ? 0 : 1;
            if (generation != expected_generation) {
                return fail("node[" + std::to_string(node_index) +
                            "] generation " + std::to_string(generation) +
                            " does not match stage " + stage);
            }
            const uint64_t coll_comm_type = node.coll.comm_type;
            const uint64_t coll_bytes = node.coll.bytes;
            const uint32_t coll_priority = node.coll.priority;
            const std::vector<bool>& coll_involved_dims =
                node.coll.involved_dim;
            if (type == 5 || type == 6) {
                if (is_cpu_op) {
                    return fail("node[" + std::to_string(node_index) +
                                "] comm node cannot be is_cpu_op");
                }
                const int src = node.comm.src;
                const int dst = node.comm.dst;
                const uint64_t tag = node.comm.tag;
                const uint64_t bytes = node.comm.bytes;
                if (src < 0 || src >= ctx_.num_ranks || dst < 0 ||
                    dst >= ctx_.num_ranks) {
                    return fail("node[" + std::to_string(node_index) +
                                "] p2p src/dst/tag out of range");
                }
                if ((type == 5 && src != rank) ||
                    (type == 6 && dst != rank)) {
                    return fail("node[" + std::to_string(node_index) +
                                "] p2p endpoint rank does not match node rank");
                }
                P2pCounts& counts =
                    p2p_counts[P2pKey{src, dst,
                                      static_cast<int64_t>(tag), bytes}];
                if (type == 5) {
                    ++counts.sends;
                } else {
                    ++counts.recvs;
                }
            } else if (type == 7) {
                if (is_cpu_op) {
                    return fail("node[" + std::to_string(node_index) +
                                "] comm node cannot be is_cpu_op");
                }
                const std::string& pg_name = node.coll.pg_name;
                const std::string& name = node.name;
                if (pg_name.empty()) {
                    return fail("node[" + std::to_string(node_index) +
                                "] collective with empty pg_name");
                }
                if (name.empty()) {
                    return fail("node[" + std::to_string(node_index) +
                                "] collective with empty operation name");
                }
                if (coll_bytes == 0) {
                    return fail("node[" + std::to_string(node_index) +
                                "] collective bytes must be positive");
                }
                if (!collective_comm_type_has_completion_path(coll_comm_type)) {
                    return fail("node[" + std::to_string(node_index) +
                                "] unsupported collective comm_type " +
                                std::to_string(coll_comm_type));
                }
                CollectiveParticipants& participants =
                    collective_participants[CollectiveKey{pg_name, name}];
                if (!participants.has_signature) {
                    participants.has_signature = true;
                    participants.comm_type = coll_comm_type;
                    participants.bytes = coll_bytes;
                    participants.priority = coll_priority;
                    participants.involved_dims = coll_involved_dims;
                } else if (participants.comm_type != coll_comm_type ||
                           participants.bytes != coll_bytes ||
                           participants.priority != coll_priority ||
                           participants.involved_dims != coll_involved_dims) {
                    return fail("collective operation " + name +
                                " of pg_name " + pg_name +
                                " has inconsistent comm_type/bytes/priority/"
                                "involved_dim signature on rank " +
                                std::to_string(rank));
                }
                ++participants.count_by_rank[rank];
            }
            ++node_index;
        }

        // Parent edges are structural prerequisites for a meaningful cycle
        // check.  Keep these checks mandatory as well: otherwise commit()
        // could add all nodes before discovering an unresolved edge.
        // C1: typed reads; the shape checks (object/kind/endpoint types)
        // are parse-layer rules now.
        uint64_t edge_index = 0;
        for (const auto& edge_entry : batch.parent_edges) {
            const int rank = edge_entry.rank;
            if (rank < 0 || rank >= ctx_.num_ranks) {
                return fail("parent_edge[" + std::to_string(edge_index) +
                            "] rank out of range: " + std::to_string(rank));
            }
            const uint64_t from = edge_entry.from_json;
            const uint64_t to = edge_entry.to_json;
            if (from == to) {
                return fail("parent_edge[" + std::to_string(edge_index) +
                            "] self-loop on rank " + std::to_string(rank));
            }

            const auto graph_it = rank_graphs.find(rank);
            if (graph_it == rank_graphs.end()) {
                return fail("parent_edge[" + std::to_string(edge_index) +
                            "] child (rank=" + std::to_string(rank) +
                            " to=" + std::to_string(to) +
                            ") not a node of this batch");
            }
            PreflightRankGraph& graph = graph_it->second;
            const auto child_it = graph.node_index_by_json_id.find(to);
            if (child_it == graph.node_index_by_json_id.end()) {
                return fail("parent_edge[" + std::to_string(edge_index) +
                            "] child (rank=" + std::to_string(rank) +
                            " to=" + std::to_string(to) +
                            ") not a node of this batch");
            }
            const auto parent_it = graph.node_index_by_json_id.find(from);
            if (parent_it != graph.node_index_by_json_id.end()) {
                graph.edges.push_back(
                    PreflightEdge{child_it->second,
                                  graph.first_edge[parent_it->second]});
                graph.first_edge[parent_it->second] = graph.edges.size() - 1;
                ++graph.indegree[child_it->second];
            } else {
                const auto parent_store_id = resolve_store_id(rank, from);
                if (!parent_store_id.has_value()) {
                    return fail("parent_edge[" + std::to_string(edge_index) +
                                "] parent (rank=" + std::to_string(rank) +
                                " from=" + std::to_string(from) +
                                ") unresolved");
                }
                if ((*ctx_.graph_sources)[rank]->store().erased(
                        *parent_store_id) &&
                    !ctx_.node_gc) {
                    return fail("parent_edge[" + std::to_string(edge_index) +
                                "] parent (rank=" + std::to_string(rank) +
                                " from=" + std::to_string(from) +
                                ") is erased while node GC is disabled");
                }
            }
            ++edge_index;
        }

        // Kahn's algorithm over the flat per-rank adjacency lists rejects
        // every in-batch cycle before any new node reaches a NodeStore.
        for (auto& rank_entry : rank_graphs) {
            const int rank = rank_entry.first;
            PreflightRankGraph& graph = rank_entry.second;
            std::vector<size_t> ready;
            ready.reserve(graph.nodes.size());
            for (size_t index = 0; index < graph.indegree.size(); ++index) {
                if (graph.indegree[index] == 0) {
                    ready.push_back(index);
                }
            }
            size_t visited = 0;
            while (!ready.empty()) {
                const size_t current = ready.back();
                ready.pop_back();
                ++visited;
                for (size_t edge = graph.first_edge[current];
                     edge != kNoPreflightEdge; edge = graph.edges[edge].next) {
                    const size_t child = graph.edges[edge].to;
                    if (--graph.indegree[child] == 0) {
                        ready.push_back(child);
                    }
                }
            }
            if (visited != graph.nodes.size()) {
                return fail("cycle among the in-batch parent edges of rank " +
                            std::to_string(rank));
            }
        }

        for (const auto& entry : p2p_counts) {
            if (entry.second.sends != entry.second.recvs) {
                const P2pKey& key = entry.first;
                return fail("p2p tuple (src=" + std::to_string(key.src) +
                            " dst=" + std::to_string(key.dst) +
                            " tag=" + std::to_string(key.tag) +
                            " bytes=" + std::to_string(key.bytes) +
                            ") send/recv multiplicity mismatch (send=" +
                            std::to_string(entry.second.sends) + " recv=" +
                            std::to_string(entry.second.recvs) + ")");
            }
        }

        // A collective can complete only if each participant posts exactly one
        // matching operation and that participant set is the declared process
        // group, not merely the batch's self-reported rank union.
        for (const auto& entry : collective_participants) {
            const CollectiveKey& key = entry.first;
            const CollectiveParticipants& participants = entry.second;
            std::vector<int> participant_ranks;
            participant_ranks.reserve(participants.count_by_rank.size());
            for (const auto& participant : participants.count_by_rank) {
                if (participant.second != 1) {
                    return fail("collective operation " + key.name +
                                " of pg_name " + key.pg_name + " has " +
                                std::to_string(participant.second) +
                                " participants on rank " +
                                std::to_string(participant.first) +
                                " (expected exactly one)");
                }
                participant_ranks.push_back(participant.first);
            }
            std::sort(participant_ranks.begin(), participant_ranks.end());
            if (!ctx_.communicator_members_for_pg) {
                return fail("collective operation " + key.name +
                            " of pg_name " + key.pg_name +
                            " cannot verify declared membership: resolver "
                            "is unavailable");
            }
            const auto declared_members = ctx_.communicator_members_for_pg(
                key.pg_name, participant_ranks);
            if (!declared_members.has_value()) {
                return fail("collective operation " + key.name +
                            " of pg_name " + key.pg_name +
                            " has missing or inconsistent declared membership");
            }
            std::unordered_set<int> declared_rank_set;
            declared_rank_set.reserve(declared_members->size());
            for (const int declared_rank : *declared_members) {
                if (declared_rank < 0 || declared_rank >= ctx_.num_ranks) {
                    return fail("collective pg_name " + key.pg_name +
                                " declares an out-of-range rank " +
                                std::to_string(declared_rank));
                }
                if (!declared_rank_set.insert(declared_rank).second) {
                    return fail("collective pg_name " + key.pg_name +
                                " declares rank " +
                                std::to_string(declared_rank) + " more than once");
                }
            }
            if (declared_rank_set.empty() ||
                declared_rank_set.size() != participant_ranks.size()) {
                return fail("collective operation " + key.name +
                            " of pg_name " + key.pg_name +
                            " participant ranks do not exactly match declared "
                            "communicator membership");
            }
            for (const int participant_rank : participant_ranks) {
                if (declared_rank_set.count(participant_rank) == 0) {
                    return fail("collective operation " + key.name +
                                " of pg_name " + key.pg_name +
                                " participant ranks do not exactly match declared "
                                "communicator membership");
                }
            }
        }

        std::unordered_set<WatchIdentity, WatchIdentityHash> watch_identities;
        watch_identities.reserve(batch.watches.size());
        uint64_t watch_index = 0;
        for (const auto& watch : batch.watches) {
            const std::string& request_id = watch.request_id;
            if (request_id.empty()) {
                return fail("watch[" + std::to_string(watch_index) +
                            "] empty request_id");
            }
            const std::string& stage = watch.stage;
            if (stage != "prefill" && stage != "decode") {
                return fail("watch[" + std::to_string(watch_index) +
                            "] invalid stage: " + stage);
            }
            const uint64_t generation = watch.generation;
            const uint64_t expected_generation =
                stage == "prefill" ? 0 : 1;
            if (generation != expected_generation) {
                return fail("watch[" + std::to_string(watch_index) +
                            "] generation " + std::to_string(generation) +
                            " does not match stage " + stage);
            }
            if (!watch_identities
                     .insert(WatchIdentity{request_id, stage, generation})
                     .second) {
                return fail("watch[" + std::to_string(watch_index) +
                            "] duplicate identity (request_id, stage, "
                            "generation) in the batch");
            }

            if (watch.members.empty()) {
                return fail("watch[" + std::to_string(watch_index) +
                            "] empty/absent members");
            }
            // C1: members keep the JSON object key order (rank-string
            // ascending) -- the same iteration order the DOM walk saw.
            for (const auto& member : watch.members) {
                const int rank = member.rank;
                const uint64_t member_id = member.json_id;
                if (rank < 0 || rank >= ctx_.num_ranks) {
                    return fail("watch[" + std::to_string(watch_index) +
                                "] member rank out of range: " +
                                std::to_string(rank));
                }
                const auto graph_it = rank_graphs.find(rank);
                if (graph_it == rank_graphs.end()) {
                    return fail("watch[" + std::to_string(watch_index) +
                                "] member (rank=" + std::to_string(rank) +
                                " id=" + std::to_string(member_id) +
                                ") is not a node of this batch");
                }
                const auto member_node_it =
                    graph_it->second.node_index_by_json_id.find(member_id);
                if (member_node_it ==
                    graph_it->second.node_index_by_json_id.end()) {
                    return fail("watch[" + std::to_string(watch_index) +
                                "] member (rank=" + std::to_string(rank) +
                                " id=" + std::to_string(member_id) +
                                ") is not a node of this batch");
                }
                const OnlineNode& member_node =
                    graph_it->second.nodes[member_node_it->second]->node;
                if (member_node.request_id != request_id ||
                    member_node.stage != stage ||
                    member_node.generation != generation) {
                    return fail("watch[" + std::to_string(watch_index) +
                                "] member (rank=" + std::to_string(rank) +
                                " id=" + std::to_string(member_id) +
                                ") does not match watch request/stage/generation");
                }
            }
            if (watch.statuses.empty()) {
                return fail("watch[" + std::to_string(watch_index) +
                            "] empty/absent statuses");
            }
            if (request_id.rfind("batch_train_", 0) != 0) {
                const RequestFacts facts = request_facts(request_id);
                if (stage == "prefill" && !facts.in_flight) {
                    return fail("prefill watch[" + std::to_string(watch_index) +
                                "] for request " + request_id +
                                " not in-flight at this epoch");
                }
                if (stage == "decode" && !facts.prefill_drained) {
                    return fail("decode watch[" + std::to_string(watch_index) +
                                "] for request " + request_id +
                                " whose prefill has not drained at this epoch");
                }
            }
            ++watch_index;
        }

        std::unordered_set<std::string> alarm_ids;
        alarm_ids.reserve(batch.future_alarms.size());
        std::vector<RequestEnvelope> future_arrival_envelopes;
        future_arrival_envelopes.reserve(batch.future_alarms.size());
        uint64_t alarm_index = 0;
        for (const auto& alarm : batch.future_alarms) {
            const uint64_t arrival = alarm.arrival_world_ns;
            if (arrival < delta.tick) {
                return fail("future_alarm[" + std::to_string(alarm_index) +
                            "] past arrival_world_ns " +
                            std::to_string(arrival) + " < delta tick " +
                            std::to_string(delta.tick));
            }
            const RequestEnvelope& envelope = alarm.envelope;
            const std::string& request_id = envelope.request_id;
            if (request_id.empty()) {
                return fail("future_alarm[" + std::to_string(alarm_index) +
                            "] empty envelope request_id");
            }
            if (!alarm_ids.insert(request_id).second) {
                return fail("future_alarm[" + std::to_string(alarm_index) +
                            "] duplicate request_id " + request_id +
                            " in the batch");
            }
            if (request_facts(request_id).in_flight) {
                return fail("future_alarm[" + std::to_string(alarm_index) +
                            "] for already in-flight request " + request_id);
            }
            if (envelope.session_id.empty()) {
                return fail("future_alarm[" + std::to_string(alarm_index) +
                            "] empty envelope session_id");
            }
            if (envelope.turn_index < 0) {
                return fail("future_alarm[" + std::to_string(alarm_index) +
                            "] negative envelope turn_index");
            }
            if (envelope.queue_index < -1) {
                return fail("future_alarm[" + std::to_string(alarm_index) +
                            "] invalid envelope queue_index " +
                            std::to_string(envelope.queue_index));
            }
            RequestEnvelope future_envelope = envelope;
            future_envelope.arrival_world_ns = arrival;
            future_arrival_envelopes.push_back(std::move(future_envelope));
            ++alarm_index;
        }
        if (!future_arrival_envelopes.empty()) {
            if (ctx_.ingress == nullptr) {
                return fail("future arrivals require a RequestIngress");
            }
            if (const auto ingress_error =
                    ctx_.ingress->validate_future_arrivals(
                        future_arrival_envelopes)) {
                return fail("future arrivals cannot be scheduled: " +
                            *ingress_error);
            }
        }
    } catch (const std::exception& exc) {
        return fail(std::string("malformed batch entry: ") + exc.what());
    }
    return std::nullopt;
}

std::optional<uint64_t> GraphBatchCommitter::resolve_store_id(
    const int rank, const uint64_t json_id) const {
    if (rank < 0 || rank >= static_cast<int>(rank_affines_.size())) {
        return std::nullopt;
    }
    const RankAffine& affine = rank_affines_[rank];
    if (!affine.initialized || json_id < affine.json_first ||
        json_id >= affine.json_next) {
        return std::nullopt;
    }
    return affine.store_first + (json_id - affine.json_first);
}

void GraphBatchCommitter::record_affine_node(const int rank,
                                             const uint64_t json_id,
                                             const uint64_t store_id) {
    if (rank < 0 || rank >= static_cast<int>(rank_affines_.size())) {
        throw std::runtime_error("commit: rank out of range for affine record");
    }
    if (json_id == std::numeric_limits<uint64_t>::max() ||
        store_id == std::numeric_limits<uint64_t>::max()) {
        throw std::runtime_error("commit: cannot advance saturated id stream");
    }
    RankAffine& affine = rank_affines_[rank];
    if (!affine.initialized) {
        affine.initialized = true;
        affine.json_first = json_id;
        affine.json_next = json_id + 1;
        affine.store_first = store_id;
        affine.store_next = store_id + 1;
        return;
    }
    if (json_id != affine.json_next || store_id != affine.store_next) {
        throw std::runtime_error(
            "commit: affine stream mismatch after successful preflight "
            "(rank=" + std::to_string(rank) + " json_id=" +
            std::to_string(json_id) + " expected_json=" +
            std::to_string(affine.json_next) + " store_id=" +
            std::to_string(store_id) + " expected_store=" +
            std::to_string(affine.store_next) + ")");
    }
    ++affine.json_next;
    ++affine.store_next;
}

std::vector<int> GraphBatchCommitter::compute_touched_ranks(
    const GraphBatch& batch, const int num_ranks) {
    std::set<int> ranks;
    for (const auto& parsed : batch.nodes) {
        const int rank = parsed.node.rank;
        if (rank >= 0 && rank < num_ranks) {
            ranks.insert(rank);
        }
    }
    return std::vector<int>(ranks.begin(), ranks.end());
}

std::optional<std::string> GraphBatchCommitter::validate_impl(
    const StateDelta& delta, const GraphBatch& batch,
    std::vector<int>* const json_id_touched_ranks) const {
    try {
        // A committed rank owns an exact affine mapping onto the NodeStore's
        // automatic id stream. Check it before every semantic rule; this is a
        // read-only fail-closed diagnostic and validate() remains pure.
        if (auto drift_error = validate_affine_drift()) {
            return drift_error;
        }

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
        if (auto id_error =
                validate_json_id_stream(batch, json_id_touched_ranks)) {
            return id_error;
        }

        // ---- [node] structural pass (C1: typed reads; the JSON-shape half
        //      of each rule moved to parse_graph_batch, the domain/semantic
        //      half stays here) ----
        std::unordered_map<int, std::set<uint64_t>> batch_ids;
        std::set<std::pair<std::string, std::string>> node_stages;
        std::set<int> touched;
        uint64_t node_count = 0;
        uint64_t node_index = 0;
        for (const auto& parsed : batch.nodes) {
            const int rank = parsed.node.rank;
            if (rank < 0 || rank >= ctx_.num_ranks) {
                return "node[" + std::to_string(node_index) +
                       "] rank out of range: " + std::to_string(rank);
            }
            const uint64_t id = parsed.json_id;
            if (id == uint64_t(-1)) {
                return "node[" + std::to_string(node_index) +
                       "] missing/invalid id";
            }
            if (!batch_ids[rank].insert(id).second) {
                return "node[" + std::to_string(node_index) +
                       "] duplicate id " + std::to_string(id) + " on rank " +
                       std::to_string(rank);
            }
            const OnlineNode& node = parsed.node;
            const uint64_t type = node.node_type;
            if (type < 1 || type > 7) {
                return "node[" + std::to_string(node_index) +
                       "] type out of range: " + std::to_string(type);
            }
            if (node.name.empty()) {
                return "node[" + std::to_string(node_index) +
                       "] empty name";
            }
            const std::string& request_id = node.request_id;
            if (request_id.empty()) {
                return "node[" + std::to_string(node_index) +
                       "] empty request_id";
            }
            const std::string& stage = node.stage;
            if (stage != "prefill" && stage != "decode") {
                return "node[" + std::to_string(node_index) +
                       "] invalid stage: " + stage;
            }
            const uint64_t generation = node.generation;
            const uint64_t expected_generation =
                stage == "prefill" ? 0 : 1;
            if (generation != expected_generation) {
                return "node[" + std::to_string(node_index) +
                       "] generation " + std::to_string(generation) +
                       " does not match stage " + stage + " (expected " +
                       stage_generation(stage) + ")";
            }
            // C1 (2026-08-29): the compute/comm/coll sub-object shape and
            // every field's JSON typing were validated once by
            // parse_graph_batch; this pass reads the typed attrs directly.
            // The src/dst/tag range checks are scoped to the comm-typed
            // nodes (types 5/6) -- the ONLY nodes whose comm fields are
            // semantically load-bearing. Real batches carry the comm
            // defaults (src=0/dst=0/tag=0) on every node and pass trivially
            // (empirical contract: send node comm.src == node.rank, recv
            // node comm.dst == node.rank -- rank-ownership is type-scoped
            // below).
            if (type == 5 || type == 6) {
                const int src = node.comm.src;
                const int dst = node.comm.dst;
                if (src < 0 || src >= ctx_.num_ranks || dst < 0 ||
                    dst >= ctx_.num_ranks) {
                    return "node[" + std::to_string(node_index) +
                           "] comm src/dst out of range: src=" +
                           std::to_string(src) +
                           " dst=" + std::to_string(dst);
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
            // A zero-byte collective has no completion path in the
            // analytical backend: AllGather reaches 0/0 chunking, while
            // other types may synthesize nonzero work. Reject it before
            // commit so no GPU resource, DataSet, or wrapper can be pinned.
            if (type == 7 && node.coll.bytes == 0) {
                return "node[" + std::to_string(node_index) +
                       "] collective bytes must be positive";
            }
            if (node.coll.pg_name.empty() && type == 7) {
                return "node[" + std::to_string(node_index) +
                       "] collective with empty pg_name";
            }
            node_stages.insert({request_id, stage});
            touched.insert(rank);
            ++node_count;
            ++node_index;
        }

        // ---- [edge] structural pass (collect in-batch edges per rank for
        //      the cycle check; the child endpoint must be a node of THIS
        //      batch, the parent endpoint may be a node of an EARLIER batch
        //      -- the persistent per-rank affine json-id translation). ----
        std::unordered_map<int, std::vector<std::pair<uint64_t, uint64_t>>>
            in_batch_edges;
        uint64_t edge_index = 0;
        for (const auto& edge_entry : batch.parent_edges) {
            const int rank = edge_entry.rank;
            if (rank < 0 || rank >= ctx_.num_ranks) {
                return "parent_edge[" + std::to_string(edge_index) +
                       "] rank out of range: " + std::to_string(rank);
            }
            const uint64_t from = edge_entry.from_json;
            const uint64_t to = edge_entry.to_json;
            if (from == to) {
                return "parent_edge[" + std::to_string(edge_index) +
                       "] self-loop on rank " + std::to_string(rank);
            }
            const bool from_in_batch = batch_ids[rank].count(from) > 0;
            const auto from_store_id =
                from_in_batch ? std::optional<uint64_t>()
                              : resolve_store_id(rank, from);
            if (!from_in_batch && !from_store_id.has_value()) {
                return "parent_edge[" + std::to_string(edge_index) +
                       "] parent (rank=" + std::to_string(rank) + " from=" +
                       std::to_string(from) + ") unresolved";
            }
            if (from_store_id.has_value() &&
                (*ctx_.graph_sources)[rank]->store().erased(*from_store_id) &&
                !ctx_.node_gc) {
                return "parent_edge[" + std::to_string(edge_index) +
                       "] parent (rank=" + std::to_string(rank) + " from=" +
                       std::to_string(from) +
                       ") is erased while node GC is disabled";
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
        for (const auto& parsed : batch.nodes) {
            const int rank = parsed.node.rank;
            const uint64_t type = parsed.node.node_type;
            if (type == 5 || type == 6) {
                const auto key = std::make_tuple(
                    parsed.node.comm.src, parsed.node.comm.dst,
                    static_cast<int64_t>(parsed.node.comm.tag));
                if (type == 5) {
                    pairs[key].first = true;
                } else {
                    pairs[key].second = true;
                }
            } else if (type == 7) {
                const std::string& pg = parsed.node.coll.pg_name;
                const std::string& name = parsed.node.name;
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
            const std::string& request_id = watch.request_id;
            if (request_id.empty()) {
                return "watch[" + std::to_string(watch_index) +
                       "] empty request_id";
            }
            const std::string& stage = watch.stage;
            if (stage != "prefill" && stage != "decode") {
                return "watch[" + std::to_string(watch_index) +
                       "] invalid stage: " + stage;
            }
            const uint64_t generation = watch.generation;
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
            // C1: typed members keep the JSON key order (rank ascending).
            if (watch.members.empty()) {
                return "watch[" + std::to_string(watch_index) +
                       "] empty/absent members";
            }
            for (const auto& member : watch.members) {
                const int rank = member.rank;
                if (rank < 0 || rank >= ctx_.num_ranks) {
                    return "watch[" + std::to_string(watch_index) +
                           "] member rank out of range: " +
                           std::to_string(rank);
                }
                const uint64_t member_id = member.json_id;
                if (batch_ids[rank].count(member_id) == 0) {
                    return "watch[" + std::to_string(watch_index) +
                           "] member (rank=" + std::to_string(rank) + " id=" +
                           std::to_string(member_id) +
                           ") is not a node of this batch";
                }
            }
            if (watch.statuses.empty()) {
                return "watch[" + std::to_string(watch_index) +
                       "] empty/absent statuses";
            }
            // eligibility (against the delta-facts-first tracking state).
            // 拼 batch 适配(2026-08-22):列车哨兵 watch(request_id =
            // "batch_train_..." 批命名空间,T_max 截断列车的完成信号)不
            // 对应任何单请求,绕过 in-flight/prefill-drained 资格检查。
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
        // 拼 batch 列车适配(2026-08-22,照 sh_1.0 母本先例): the coverage
        // rule is one-directional here. Every watch's (request_id, stage)
        // must be covered by this batch's nodes (an uncovered watch is
        // always a bug), but a batch may carry nodes whose (request_id,
        // stage) has no new watch -- a D-side iteration train's shared body
        // and end barrier live in the batch namespace ("batch_train_...",
        // not a real request; the member exit watches carry the real
        // request ids) and a joining member's transfer nodes need no watch
        // until it exits in a later train. The historical two-segment
        // batches still satisfy the stricter equality (no behavior change
        // for them); the deviation matches the sh_1.0 relaxation recorded
        // in sh_1.0改造执行实录.md.
        for (const auto& entry : watch_stages) {
            if (node_stages.count(entry) == 0) {
                return "watch (request_id, stage) {" + entry.first + "," +
                       entry.second +
                       "} has no node coverage in this batch";
            }
        }

        // ---- [assign] (opaque; structural only -- S1/S2 are parse-layer
        //      rules now; these typed reads are the same belt-and-suspenders
        //      domain checks against a post-parse mutation) ----
        uint64_t assign_index = 0;
        for (const auto& assignment : batch.assignments) {
            if (assignment.request_id.empty()) {
                return "assignment[" + std::to_string(assign_index) +
                       "] empty request_id";
            }
            if (assignment.prefill_instance_index < 0 ||
                assignment.decode_instance_index < 0) {
                return "assignment[" + std::to_string(assign_index) +
                       "] negative instance index";
            }
            ++assign_index;
        }

        // ---- [kv] (opaque to C++; the Python provisional ledger is the
        //      authority -- structural only) ----
        uint64_t kv_index = 0;
        for (const auto& action : batch.kv_actions) {
            if (action.event_type.empty() ||
                action.trigger_request_id.empty()) {
                return "kv_action[" + std::to_string(kv_index) +
                       "] missing event_type/trigger_request_id";
            }
            ++kv_index;
        }

        // ---- [alarm] ----
        std::set<std::string> alarm_ids;
        uint64_t alarm_index = 0;
        for (const auto& alarm : batch.future_alarms) {
            const uint64_t arrival = alarm.arrival_world_ns;
            if (arrival < delta.tick) {
                return "future_alarm[" + std::to_string(alarm_index) +
                       "] past arrival_world_ns " +
                       std::to_string(arrival) + " < delta tick " +
                       std::to_string(delta.tick);
            }
            const RequestEnvelope& envelope = alarm.envelope;
            const std::string& request_id = envelope.request_id;
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
            if (envelope.session_id.empty()) {
                return "future_alarm[" + std::to_string(alarm_index) +
                       "] empty envelope session_id";
            }
            if (envelope.turn_index < 0) {
                return "future_alarm[" + std::to_string(alarm_index) +
                       "] negative envelope turn_index";
            }
            ++alarm_index;
        }

        // ---- [touched] Python-computed touched_ranks must agree with the
        //      batch's node rank set (sorted unique) ----
        if (batch.has_touched_ranks) {
            const std::vector<int>& declared = batch.touched_ranks;
            for (const int rank : declared) {
                if (rank < 0 || rank >= ctx_.num_ranks) {
                    return "touched_ranks rank out of range: " +
                           std::to_string(rank);
                }
            }
            if (!std::is_sorted(declared.begin(), declared.end()) ||
                std::adjacent_find(declared.begin(), declared.end()) !=
                    declared.end()) {
                return "touched_ranks not sorted unique";
            }
            // S1 (2026-08-23): reuse this validate pass's own touched set
            // instead of re-walking batch.nodes through
            // compute_touched_ranks(): by this point every node has an
            // in-range rank (fail-closed in the node pass above), so the
            // sorted-unique rank sets are identical by construction.
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

        // Keep the full validator's rich diagnostics above, then apply the
        // production-mandatory liveness subset as well. This extends full
        // validation with multiplicity-aware p2p matching, per-member watch
        // generation matching, and declared (not batch-inferred) collective
        // membership without weakening any established diagnosis.
        if (auto liveness_error = mandatory_liveness_preflight(delta, batch)) {
            return liveness_error;
        }

        return std::nullopt;
    } catch (const std::exception& exc) {
        // Malformed entry types surface as nlohmann type errors; convert to
        // a validation error (fail-closed, zero state mutation).
        return std::string("malformed batch entry: ") + exc.what();
    }
}

std::optional<std::string> GraphBatchCommitter::validate(
    const StateDelta& delta, const GraphBatch& batch) const {
    return validate_impl(delta, batch, nullptr);
}

GraphBatchCommitter::ValidateAndCommitResult
GraphBatchCommitter::validate_and_commit(const StateDelta& delta,
                                         const GraphBatch& batch) {
    ValidateAndCommitResult result;
    if (validation_commit_in_progress_) {
        result.error =
            "validate_and_commit reentered while a validation commit is active";
        return result;
    }

    // Phase A and B stay in one call so no mutable gap exists between them.
    // Reuse the rank-bounded scratch allocated by the constructor; callbacks
    // reached by validation must not overwrite it with a nested commit.
    validation_commit_in_progress_ = true;
    try {
        commit_touched_ranks_.clear();
        const auto validation_t0 = std::chrono::steady_clock::now();
        const auto validation_error =
            validate_impl(delta, batch, &commit_touched_ranks_);
        result.validation_ns = static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - validation_t0)
                .count());
        if (validation_error.has_value()) {
            result.error = validation_error;
            validation_commit_in_progress_ = false;
            return result;
        }
        commit_after_preflight(delta, batch, commit_touched_ranks_);
        validation_commit_in_progress_ = false;
        return result;
    } catch (...) {
        validation_commit_in_progress_ = false;
        throw;
    }
}

void GraphBatchCommitter::commit(const StateDelta& delta,
                                 const GraphBatch& batch) {
    if (validation_commit_in_progress_) {
        throw std::runtime_error(
            "commit reentered while validate_and_commit owns preflight scratch");
    }
    // Production may skip the expensive semantic validator. The producer's
    // bounded contiguous-id contract, the zero-byte collective safety rule,
    // and the NodeStore affine stream are still mandatory. They are checked
    // before any delta, NodeStore or affine state change, so a failure has
    // zero commit side effects.
    if (auto drift_error = validate_affine_drift()) {
        throw std::runtime_error("commit preflight: " + *drift_error);
    }
    if (auto id_error = validate_json_id_stream(batch, &commit_touched_ranks_)) {
        throw std::runtime_error("commit preflight: " + *id_error);
    }
    if (auto liveness_error = mandatory_liveness_preflight(delta, batch)) {
        throw std::runtime_error("commit preflight: " + *liveness_error);
    }

    commit_after_preflight(delta, batch, commit_touched_ranks_);
}

void GraphBatchCommitter::commit_after_preflight(
    const StateDelta& delta, const GraphBatch& batch,
    const std::vector<int>& touched_ranks) {

    // ---- delta facts first (the same facts validate() used) ----
    apply_delta_facts(delta, in_flight_, prefill_drained_);

    // ---- Phase B-1: nodes (per-rank affine json id -> store id).
    //      C1 (2026-08-29): the OnlineNode was ALREADY assembled once by
    //      parse_graph_batch (every field's type/domain was validated there,
    //      with explicit uint64 extraction -- the old NOTE about value()'s
    //      int-default truncation is now structurally impossible). This
    //      loop is a copy + store insert + affine advance; nothing here can
    //      throw on the JSON shape anymore, so the B-1 half-commit window
    //      only opens on mechanism bugs, never on data shape. ----
    uint64_t node_count = 0;
    for (const auto& parsed : batch.nodes) {
        const int rank = parsed.node.rank;
        const uint64_t json_id = parsed.json_id;
        if (rank < 0 || rank >= static_cast<int>(ctx_.graph_sources->size())) {
            throw std::runtime_error(
                "commit: node rank out of range: " + std::to_string(rank));
        }
        OnlineNode node = parsed.node;  // global_id already 0 (store assigns)
        const uint64_t store_id =
            (*ctx_.graph_sources)[rank]->store().add_node(std::move(node));
        // The first node captures NodeStore's REAL returned id; later nodes
        // must advance both strictly contiguous streams in lockstep.
        record_affine_node(rank, json_id, store_id);
        ++node_count;
    }

    // ---- Phase B-1.5: side-band metrics anchor hook (phase-7 §10.3). The
    //      anchors must bind to the STORE ids (what on_node_issue /
    //      on_node_complete observe) -- NodeStore hands out store ids
    //      starting at 1 while the online graph's json ids start at 0, so
    //      registering with the json ids off-by-ones every anchor (the id-0
    //      start anchors would never fire at all). The caller's hook resolves
    //      only the ids it needs through this committer's affine mapping and
    //      must register BEFORE the issue pass below lets any nodes run.
    if (ctx_.metrics_anchor_hook) {
        ctx_.metrics_anchor_hook(batch, *this);
    }

    // ---- Phase B-2: parent_edges -> add_dependency (Data kind).
    //      C1: typed edges; kind == "data" is parse-layer-guaranteed. ----
    for (const auto& edge : batch.parent_edges) {
        const int rank = edge.rank;
        const uint64_t from = edge.from_json;
        const uint64_t to = edge.to_json;
        if (rank < 0 || rank >= ctx_.num_ranks) {
            throw std::runtime_error(
                "commit: parent edge rank out of range: " +
                std::to_string(rank));
        }
        const auto from_store_id = resolve_store_id(rank, from);
        const auto to_store_id = resolve_store_id(rank, to);
        if (!to_store_id.has_value()) {
            throw std::runtime_error(
                "commit: parent edge references an unknown node id "
                "(rank=" + std::to_string(rank) +
                " from=" + std::to_string(from) +
                " to=" + std::to_string(to) + ")");
        }
        NodeStore& store = (*ctx_.graph_sources)[rank]->store();
        if (!from_store_id.has_value()) {
            throw std::runtime_error(
                "commit: parent edge references an unknown node id "
                "(rank=" + std::to_string(rank) +
                " from=" + std::to_string(from) +
                " to=" + std::to_string(to) + ")");
        }
        if (store.erased(*from_store_id)) {
            // A translated absent parent is exactly a collected, finished
            // parent. NodeStore never reuses ids, so the dependency is a
            // no-op only under the committer-owned GC mode.
            if (ctx_.node_gc) {
                continue;
            }
            throw std::runtime_error(
                "commit: translated parent is erased while node GC is "
                "disabled (rank=" + std::to_string(rank) +
                " from=" + std::to_string(from) + ")");
        }
        if (store.erased(*to_store_id)) {
            throw std::runtime_error(
                "commit: translated child is erased (rank=" +
                std::to_string(rank) + " to=" + std::to_string(to) + ")");
        }
        store.add_dependency(
            *from_store_id, *to_store_id, DepKind::Data);
    }

    // ---- Phase B-3: register watches (member ids translated to store
    //      ids; C1: typed members in JSON key order, statuses from the
    //      parse-layer enum -- the "unknown status" throw is structurally
    //      gone). ----
    for (const auto& watch : batch.watches) {
        const std::string& request_id = watch.request_id;
        const std::string& stage = watch.stage;
        const uint64_t generation = watch.generation;
        std::set<CompletionKey> members;
        for (const auto& member : watch.members) {
            const int rank = member.rank;
            const uint64_t json_id = member.json_id;
            const auto store_id = resolve_store_id(rank, json_id);
            if (!store_id.has_value()) {
                throw std::runtime_error(
                    "commit: watch member references an unknown node id "
                    "(rank=" + std::to_string(rank) +
                    " id=" + std::to_string(json_id) + ")");
            }
            if ((*ctx_.graph_sources)[rank]->store().erased(*store_id)) {
                throw std::runtime_error(
                    "commit: watch member resolves to an erased node "
                    "(rank=" + std::to_string(rank) +
                    " id=" + std::to_string(json_id) + ")");
            }
            members.insert(CompletionKey{rank, *store_id, generation});
        }
        std::set<NodeTerminalStatus> statuses;
        for (const auto status : watch.statuses) {
            statuses.insert(status);
        }
        ctx_.watch_registry->register_stage_watch(
            request_id, stage, generation, std::move(members),
            std::move(statuses));
    }

    // ---- Phase B-4: schedule future arrival alarms ----
    for (const auto& alarm : batch.future_alarms) {
        RequestEnvelope envelope = alarm.envelope;
        envelope.arrival_world_ns = alarm.arrival_world_ns;
        ctx_.ingress->schedule_future_arrival(envelope);
    }

    // ---- Phase B-5: issue pass over the TOUCHED ranks only (ranks without
    //      new nodes cannot have new free nodes -- the completion hook's
    //      deferred per-rank passes drain every other rank) ----
    // Reuse the sorted-unique rank set collected by the mandatory id-stream
    // preflight.  Re-walking every JSON node here used to add a second
    // O(nodes) pass plus an allocating std::set on every committed batch.
    for (const int rank : touched_ranks) {
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

    // ---- M2 node GC (2026-08-23; A1 amortization 2026-08-28):
    //      quiescent-point collection, amortized. The issue pass above has
    //      fully returned (no Workload callback holds a NodeView pointer),
    //      and any deferred issue passes it scheduled run later and only
    //      ever touch free -- i.e. unfinished -- nodes, so erasing finished
    //      childless nodes here is invisible to every holder (see
    //      NodeStore::collect_garbage). A1: the tail only counts the
    //      pending candidates (O(#ranks)) and drains once
    //      >= kGcAmortizeThreshold have accumulated since the last
    //      collection -- the pre-A1 drain-every-commit is what caused the
    //      light-load wall regression. No-op (structurally absent) when
    //      node_gc is off. ----
    if (ctx_.node_gc) {
        size_t pending = 0;
        for (const auto& source : *ctx_.graph_sources) {
            pending += source->store().pending_gc_count();
        }
        if (pending >= kGcAmortizeThreshold) {
            collect_node_garbage();
        }
    }
}

void GraphBatchCommitter::finalize_node_garbage() {
    // A1 (2026-08-28): run-end forced drain -- the amortized commit tails
    // may leave up to kGcAmortizeThreshold candidates uncollected (plus the
    // post-final-delivery tail nodes that finish after the last commit);
    // draining here once makes the run-end retained-count diagnostics report
    // the true in-flight window. No-op when node_gc is off.
    if (ctx_.node_gc) {
        collect_node_garbage();
    }
}

void GraphBatchCommitter::collect_node_garbage() {
    // M2 (2026-08-23): end-of-commit quiescent-point collection. The affine
    // metadata deliberately survives: it is one fixed record per rank and
    // still translates an erased committed parent to its non-reused store id.
    for (auto& source : *ctx_.graph_sources) {
        source->store().collect_garbage();
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
