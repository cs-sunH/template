/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

NodeStore -- execution-driven mechanism layer (wscllm phase 1).
Implementation (方案 §4 步骤 1-4).
*******************************************************************************/

#include "astra-sim/workload/execution_driven/NodeStore.hh"

#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <iostream>
#include <map>
#include <stdexcept>

#include "extern/graph_frontend/chakra/src/feeder_v3/et_feeder.h"

namespace AstraSim {
namespace ExecutionDriven {

// ---------------------------------------------------------------------------
// NodeStore
// ---------------------------------------------------------------------------

uint64_t NodeStore::add_node(OnlineNode node) {
    if (node.global_id == 0) {
        node.global_id = next_id_++;
    } else {
        next_id_ = std::max(next_id_, node.global_id + 1);
    }
    const auto [it, inserted] =
        nodes_.emplace(node.global_id, NodeRecord{std::move(node)});
    assert(inserted);
    if (it->second.unresolved_parents == 0) {
        free_ids_.insert(node.global_id);
    }
    return node.global_id;
}

void NodeStore::add_dependency(uint64_t parent, uint64_t child,
                               DepKind /*kind*/) {
    // First version: all kinds resolve identically (precedence). The kind is
    // stored per edge for the later fence upgrade (步骤 1-5 note forbids
    // building the general fence early). Dead-parent edges are tolerated:
    // a parent that already finished no longer blocks the child.
    auto child_it = nodes_.find(child);
    if (child_it == nodes_.end()) {
        return;  // unknown child: edge cannot be recorded
    }
    auto parent_it = nodes_.find(parent);
    if (parent_it == nodes_.end() || parent_it->second.finished) {
        return;  // already-finished parent does not block
    }
    // A child added with no parents is free; recording a real parent takes it
    // out of the free set again.
    child_it->second.unresolved_parents++;
    free_ids_.erase(child);
    child_it->second.parents.push_back(parent);
    parent_it->second.children.push_back(child);
    // M2 node GC: count the live child on the parent (the child of a
    // validated edge is always a fresh, unfinished batch node).
    ++parent_it->second.unfinished_children;
}

std::vector<uint64_t> NodeStore::resolve_free_nodes() const {
    std::vector<uint64_t> result;
    fill_free_node_snapshot(result);
    return result;
}

void NodeStore::fill_free_node_snapshot(std::vector<uint64_t>& out) const {
    out.assign(free_ids_.begin(), free_ids_.end());
}

void NodeStore::mark_issued(uint64_t node_id) {
    auto it = nodes_.find(node_id);
    if (it == nodes_.end() || it->second.issued || it->second.finished) {
        return;
    }
    it->second.issued = true;
    free_ids_.erase(node_id);
}

void NodeStore::finish_node(uint64_t node_id) {
    auto it = nodes_.find(node_id);
    if (it == nodes_.end() || it->second.finished) {
        return;  // idempotent: the only dependency-release entry
    }
    it->second.finished = true;
    free_ids_.erase(node_id);
    for (const auto child : it->second.children) {
        auto child_it = nodes_.find(child);
        if (child_it == nodes_.end()) {
            continue;  // M2: child already collected (it finished earlier)
        }
        assert(child_it->second.unresolved_parents > 0);
        if (--child_it->second.unresolved_parents == 0) {
            free_ids_.insert(child);
        }
    }
    // M2 node GC (2026-08-23): a finished node with no unfinished children
    // becomes a GC candidate; a node with pending children is enqueued by
    // the LAST child's finish in the parents loop below (children can only
    // finish after their parents -- a node is issued only from the free set,
    // i.e. after every recorded parent finished), so each node enters the
    // FIFO at most once. finish_node itself NEVER erases: e.g.
    // Workload::skip_invalid looks the record up again AFTER this call, and
    // the issue passes hold NodeViews obtained from the free set -- the
    // actual erase happens only at the committer's quiescent-point
    // collect_garbage().
    if (gc_enabled_ && it->second.unfinished_children == 0) {
        gc_fifo_.push_back(node_id);
    }
    for (const auto parent : it->second.parents) {
        auto parent_it = nodes_.find(parent);
        if (parent_it == nodes_.end()) {
            continue;  // M2: parent already collected
        }
        assert(parent_it->second.unfinished_children > 0);
        if (--parent_it->second.unfinished_children == 0 &&
            parent_it->second.finished && gc_enabled_) {
            gc_fifo_.push_back(parent);
        }
    }
}

std::optional<OnlineNode> NodeStore::node(uint64_t node_id) const {
    const auto it = nodes_.find(node_id);
    if (it == nodes_.end()) {
        return std::nullopt;
    }
    return it->second.node;
}

const OnlineNode* NodeStore::node_ptr(uint64_t node_id) const {
    const auto it = nodes_.find(node_id);
    return it == nodes_.end() ? nullptr : &it->second.node;
}

OnlineStatisticsState* NodeStore::mutable_online_statistics(uint64_t node_id) {
    const auto it = nodes_.find(node_id);
    return it == nodes_.end() ? nullptr : &it->second.node.online_statistics;
}

void NodeStore::set_metric_anchor_flags(uint64_t node_id, bool issue,
                                        bool complete) {
    // R2 anchor fast path: same write-into-stored-record pattern as
    // mutable_online_statistics. OR, not assign: e.g. a watch end anchor and
    // a transfer anchor can both register the same node (each on its own
    // edge), and a re-registration across batches must never clear a flag.
    const auto it = nodes_.find(node_id);
    if (it == nodes_.end()) {
        return;  // unknown node: the anchor itself can never fire either
    }
    it->second.node.metric_issue_anchor |= issue;
    it->second.node.metric_complete_anchor |= complete;
}

bool NodeStore::mark_terminal_observed(uint64_t node_id) {
    const auto it = nodes_.find(node_id);
    if (it == nodes_.end() || !it->second.issued || it->second.finished ||
        it->second.terminal_observed) {
        return false;
    }
    it->second.terminal_observed = true;
    return true;
}

std::optional<NodeStoreMeta> NodeStore::meta_for(uint64_t node_id) const {
    const auto it = nodes_.find(node_id);
    if (it == nodes_.end()) {
        return std::nullopt;
    }
    return NodeStoreMeta{it->second.node.request_id, it->second.node.stage,
                         it->second.node.generation};
}

size_t NodeStore::pending_count() const {
    return std::count_if(nodes_.begin(), nodes_.end(),
                         [](const auto& entry) {
                             return !entry.second.finished;
                         });
}

bool NodeStore::empty() const { return nodes_.empty(); }

void NodeStore::set_gc_enabled(bool enabled) {
    // M2 node GC: the switch is set once by the committer constructor from
    // its Context (the official path always enables it -- the
    // --online-node-gc CLI arm was removed by the B.3 cleanup (2026-09-05);
    // fixtures leave it off and keep the pre-M2 behavior, memory profile
    // included).
    gc_enabled_ = enabled;
}

void NodeStore::collect_garbage() {
    // M2 node GC (2026-08-23): quiescent-point collection. The ONLY caller
    // is the committer's end-of-commit tail --
    // by then every Workload callback triggered by the commit's issue pass
    // has fully returned (no NodeView pointer is alive anywhere) and later
    // deferred issue passes only ever touch free, i.e. unfinished, nodes.
    if (!gc_enabled_) {
        return;
    }
    for (; gc_fifo_head_ < gc_fifo_.size(); ++gc_fifo_head_) {
        const uint64_t node_id = gc_fifo_[gc_fifo_head_];
        auto it = nodes_.find(node_id);
        if (it == nodes_.end()) {
            continue;  // defensive: already collected
        }
        if (!it->second.finished || it->second.unfinished_children != 0) {
            // Defensive: unreachable -- a node becomes a candidate only at
            // (finished && unfinished_children == 0), and post-finish
            // add_dependency early-returns on the finished parent, so the
            // counter can never rise again.
            continue;
        }
        nodes_.erase(it);
        ++gc_erased_count_;
    }
    // All entries consumed (each node enqueues at most once). Keep only a
    // modest reusable FIFO allocation; a single giant completion burst must
    // not pin its historical vector capacity for the rest of the run.
    constexpr size_t kMaxRetainedGcFifoCapacity = 8192;
    gc_fifo_.clear();
    gc_fifo_head_ = 0;
    if (gc_fifo_.capacity() > kMaxRetainedGcFifoCapacity) {
        std::vector<uint64_t>().swap(gc_fifo_);
    }

    // unordered_map::erase never shrinks the bucket array. Rebuild only after
    // a material high-water collapse, at this quiescent point, so RSS follows
    // the live NodeStore window instead of an early historical peak without
    // adding rehash churn to ordinary 4096-node GC cycles.
    constexpr size_t kMinBucketsBeforeCompaction = 4096;
    if (nodes_.bucket_count() > kMinBucketsBeforeCompaction &&
        nodes_.size() < nodes_.bucket_count() / 4) {
        decltype(nodes_) compact;
        compact.max_load_factor(nodes_.max_load_factor());
        compact.reserve(nodes_.size());
        for (auto& entry : nodes_) {
            compact.emplace(entry.first, std::move(entry.second));
        }
        nodes_.swap(compact);
    }
}

bool NodeStore::erased(uint64_t node_id) const {
    // M2: absent == collected for committed store ids (see header). Used by
    // the committer's store_ids_ prune pass only.
    return nodes_.find(node_id) == nodes_.end();
}

RankInjectedSummary NodeStore::injected_unfinished_summary(int rank) const {
    // Phase-3 sensing (方案 §6.2 操作 1 / contract ⑥): classify the
    // committed-but-not-terminal load by compute ops / comm bytes / estimated
    // remaining service / resource state, traceable per request/stage/
    // generation. Pure query over the committed graph; deterministic (the
    // per-request groups are sorted before return).
    RankInjectedSummary summary;
    summary.rank = rank;

    struct GroupKey {
        std::string request_id;
        std::string stage;
        uint64_t generation = 0;

        bool operator<(const GroupKey& other) const {
            if (request_id != other.request_id) {
                return request_id < other.request_id;
            }
            if (stage != other.stage) {
                return stage < other.stage;
            }
            return generation < other.generation;
        }
    };
    std::map<GroupKey, InjectedUnfinishedEntry> groups;

    for (const auto& [node_id, record] : nodes_) {
        (void)node_id;
        if (record.finished) {
            continue;
        }
        const OnlineNode& node = record.node;
        ++summary.node_count;
        if (!record.issued) {
            // Resource state: dependency-satisfied and not yet issued
            // (unresolved_parents == 0 means the node is in the free set).
            if (record.unresolved_parents == 0) {
                ++summary.free_node_count;
            }
        } else {
            ++summary.in_flight_node_count;
        }
        InjectedUnfinishedEntry& entry =
            groups[{node.request_id, node.stage, node.generation}];
        ++entry.node_count;
        switch (node.kind) {
            case NodeKind::Compute:
                summary.compute_ops += node.compute.num_ops;
                summary.estimated_remaining_ns += node.compute.runtime_ns;
                entry.compute_ops += node.compute.num_ops;
                entry.estimated_remaining_ns += node.compute.runtime_ns;
                if (record.issued) {
                    summary.in_flight_gpu_ops += node.compute.num_ops;
                }
                break;
            case NodeKind::CommSend:
            case NodeKind::CommRecv:
                summary.comm_bytes += node.comm.bytes;
                entry.comm_bytes += node.comm.bytes;
                break;
            case NodeKind::CommCollective:
                summary.comm_bytes += node.coll.bytes;
                entry.comm_bytes += node.coll.bytes;
                break;
            case NodeKind::MemLoad:
            case NodeKind::MemStore:
            case NodeKind::Metadata:
            case NodeKind::Invalid:
                // No compute/comm classification contribution; counted in
                // node_count / resource state only.
                break;
        }
    }

    summary.per_request.reserve(groups.size());
    for (auto& [key, entry] : groups) {
        entry.request_id = key.request_id;
        entry.stage = key.stage;
        entry.generation = key.generation;
        summary.per_request.push_back(std::move(entry));
    }
    return summary;
}

// ---------------------------------------------------------------------------
// NodeStoreGraphSource
// ---------------------------------------------------------------------------

std::vector<NodeView> NodeStoreGraphSource::dep_free_nodes() {
    std::vector<NodeView> views;
    for (const auto node_id : store_.resolve_free_nodes()) {
        const auto node = store_.node(node_id);
        if (node.has_value()) {
            views.push_back(*node);
        }
    }
    return views;
}

void NodeStoreGraphSource::for_each_dep_free(
    const std::function<void(const NodeView&)>& consume) {
    // The reusable vector is still a snapshot, so consume() may safely mutate
    // the free set / release dependencies. Newly released children are not
    // visited until the next issue pass.
    store_.fill_free_node_snapshot(dep_free_scratch_ids_);
    for (const auto node_id : dep_free_scratch_ids_) {
        const OnlineNode* node = store_.node_ptr(node_id);
        if (node != nullptr) {
            consume(*node);
        }
    }
}

void NodeStoreGraphSource::finish_node(uint64_t node_id) {
    store_.finish_node(node_id);
}

std::optional<NodeView> NodeStoreGraphSource::lookup(uint64_t node_id) {
    return store_.node(node_id);
}

const NodeView* NodeStoreGraphSource::lookup_ptr(uint64_t node_id) {
    return store_.node_ptr(node_id);
}

OnlineStatisticsState* NodeStoreGraphSource::mutable_online_statistics(
    uint64_t node_id) {
    return store_.mutable_online_statistics(node_id);
}

bool NodeStoreGraphSource::mark_terminal_observed(uint64_t node_id) {
    return store_.mark_terminal_observed(node_id);
}

void NodeStoreGraphSource::take_node(uint64_t node_id) {
    store_.mark_issued(node_id);
}

// ---------------------------------------------------------------------------
// ETFeederGraphSource
// ---------------------------------------------------------------------------

ETFeederGraphSource::ETFeederGraphSource(
    Chakra::FeederV3::ETFeeder* const et_feeder, const int rank)
    : et_feeder_(et_feeder), rank_(rank) {
    assert(et_feeder_ != nullptr);
}

NodeView ETFeederGraphSource::view_of(uint64_t node_id) const {
    const auto node = et_feeder_->lookupNode(node_id);
    assert(node != nullptr);
    NodeView nv;
    nv.global_id = node->id();
    nv.rank = rank_;
    nv.node_type = static_cast<uint64_t>(node->type());
    nv.kind = static_cast<NodeKind>(nv.node_type);
    nv.name = node->name();
    nv.is_cpu_op = node->is_cpu_op();
    nv.is_timer_op = node->get_attr<bool>("is_timer_op", false);
    nv.inputs_values = node->get_inputs_values();
    switch (nv.kind) {
        case NodeKind::Compute:
            // Strict attrs are read eagerly for every free node, but the
            // baseline read them lazily only on the dispatch path that
            // consumes them (e.g. CPU comp nodes replay and never carry
            // num_ops). has_attr guards make the eager read byte-exact with
            // the lazy strict read: whenever the baseline would have read
            // the attr, it exists and the same value is produced.
            if (node->has_attr("num_ops")) {
                nv.compute.num_ops = node->num_ops<uint64_t>();
            }
            if (node->has_attr("tensor_size")) {
                nv.compute.tensor_size = node->tensor_size<uint64_t>();
            }
            nv.compute.runtime_ns = node->runtime() * 1000;  // micros -> ns
            if (node->has_attr("remote_weight_bytes")) {
                nv.compute.has_remote_weight_bytes = true;
                nv.compute.remote_weight_bytes =
                    node->remote_weight_bytes<uint64_t>();
            }
            break;
        case NodeKind::MemLoad:
        case NodeKind::MemStore:
            // issue_remote_mem consumes tensor_size (strict in the baseline).
            if (node->has_attr("tensor_size")) {
                nv.compute.tensor_size = node->tensor_size<uint64_t>();
            }
            break;
        case NodeKind::CommSend:
        case NodeKind::CommRecv:
            // Baseline defaults preserved: comm_src/dst default to the local
            // rank (this rank == adapter rank_), tag defaults 0; size was
            // strict at consumption time, guard keeps eager reads identical.
            if (node->has_attr("comm_size")) {
                nv.comm.bytes = node->comm_size<uint64_t>();
            }
            if (node->has_attr("comm_src")) {
                nv.comm.src = static_cast<int>(node->comm_src<uint32_t>(rank_));
            } else {
                nv.comm.src = rank_;
            }
            if (node->has_attr("comm_dst")) {
                nv.comm.dst = static_cast<int>(node->comm_dst<uint32_t>(rank_));
            } else {
                nv.comm.dst = rank_;
            }
            if (node->has_attr("comm_tag")) {
                nv.comm.tag = node->comm_tag<uint32_t>();
            }
            // hbm-charge (bool, default true): explicit opt-out of the
            // local-HBM endpoint charge under hbm-bandwidth-contention
            // (same get_attr<bool> pattern as is_timer_op above).
            nv.comm.hbm_charge = node->get_attr<bool>("hbm-charge", true);
            break;
        case NodeKind::CommCollective:
            if (node->has_attr("comm_type")) {
                nv.coll.comm_type = node->comm_type<uint64_t>();
            }
            if (node->has_attr("comm_size")) {
                nv.coll.bytes = node->comm_size<uint64_t>();
            }
            nv.coll.priority = node->comm_priority<uint32_t>();  // default 0u
            nv.coll.pg_name = node->pg_name<std::string>("");    // default ""
            // BROADCAST replay runtime (issue_coll_comm), micros -> ns
            nv.compute.runtime_ns = node->runtime() * 1000;
            if (node->has_attr("involved_dim")) {
                const auto& attr = node->get_attr_msg("involved_dim");
                if (attr.has_bool_list()) {
                    const auto& bool_list = attr.bool_list();
                    for (int i = 0; i < bool_list.values_size(); ++i) {
                        nv.coll.involved_dim.push_back(bool_list.values(i));
                    }
                } else {
                    // byte-exact legacy behavior (issue_coll_comm)
                    std::cerr << "Expected bool_list in involved_dim but found"
                                 " another type."
                              << std::endl;
                    std::exit(EXIT_FAILURE);
                }
            } else {
                // legacy default: simulate 4 involved dimensions
                for (int i = 0; i < 4; ++i) {
                    nv.coll.involved_dim.push_back(true);
                }
            }
            break;
        default:
            break;  // Invalid / Metadata / MemLoad / MemStore: no attrs read
    }
    return nv;
}

std::vector<NodeView> ETFeederGraphSource::dep_free_nodes() {
    auto& resolver = et_feeder_->getDependancyResolver();
    // Copy into a std::set (dedup + ascending order) -- byte-exact issue
    // order of the pre-phase-1 issue_dep_free_nodes.
    std::set<uint64_t> node_ids;
    for (const auto node_id : resolver.get_dependancy_free_nodes()) {
        node_ids.insert(node_id);
    }
    std::vector<NodeView> views;
    views.reserve(node_ids.size());
    for (const auto node_id : node_ids) {
        views.push_back(view_of(node_id));
    }
    return views;
}

void ETFeederGraphSource::finish_node(uint64_t node_id) {
    et_feeder_->getDependancyResolver().finish_node(node_id);
}

std::optional<NodeView> ETFeederGraphSource::lookup(uint64_t node_id) {
    try {
        return view_of(node_id);
    } catch (const std::runtime_error&) {
        // Legacy lookupNode semantics: a node id outside the index throws
        // "not found in index" on first read; the optional contract turns
        // that into a miss (byte-exact for every successful baseline, where
        // lookups only ever hit).
        return std::nullopt;
    }
}

void ETFeederGraphSource::take_node(uint64_t node_id) {
    et_feeder_->getDependancyResolver().take_node(node_id);
}

std::shared_ptr<Chakra::FeederV3::ETFeederNode> ETFeederGraphSource::et_node(
    uint64_t node_id) {
    // Same shape as the legacy et_feeder->lookupNode: never nullptr, throws
    // only when the wrapper is dereferenced for an unknown id.
    return et_feeder_->lookupNode(node_id);
}

bool ETFeederGraphSource::static_all_done() {
    auto& resolver = et_feeder_->getDependancyResolver();
    return resolver.get_dependancy_free_nodes().empty() &&
           resolver.get_ongoing_nodes().empty();
}

}  // namespace ExecutionDriven
}  // namespace AstraSim
