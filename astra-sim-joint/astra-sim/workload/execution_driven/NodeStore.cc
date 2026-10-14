/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

NodeStore -- execution-driven mechanism layer (sh30 (wscllm-blueprint) phase 1).
Implementation (方案 §4 步骤 1-4).
*******************************************************************************/

#include "astra-sim/workload/execution_driven/NodeStore.hh"

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <iterator>
#include <limits>
#include <map>
#include <stdexcept>

#include "extern/graph_frontend/chakra/src/feeder_v3/et_feeder.h"

#include "astra-sim/common/Logging.hh"

namespace AstraSim {
namespace ExecutionDriven {

// ---------------------------------------------------------------------------
// NodeStore
// ---------------------------------------------------------------------------

uint64_t NodeStore::add_node(OnlineNode node) {
    uint64_t assigned_id = node.global_id;
    uint64_t next_id_after = next_id_;
    if (node.global_id == 0) {
        if (next_id_ == std::numeric_limits<uint64_t>::max()) {
            std::fprintf(stderr,
                         "[Error] (execution_driven/node_store) add_node: "
                         "automatic global_id stream exhausted (fail-closed)\n");
            std::exit(EXIT_FAILURE);
        }
        assigned_id = next_id_;
        next_id_after = next_id_ + 1;
    } else {
        // R1-A-L4 port (2026-09-30): next_id_ = global_id + 1 wraps to 0 at
        // global_id == UINT64_MAX, silently resetting the id space. Only a
        // synthetic caller can reach the sentinel; fail closed instead of
        // corrupting the id stream.
        if (node.global_id == std::numeric_limits<uint64_t>::max()) {
            std::cerr << "[Error] (execution_driven/node_store) add_node: "
                         "global_id UINT64_MAX would overflow next_id_ "
                         "(fail-closed)"
                      << std::endl;
            std::exit(EXIT_FAILURE);
        }
        next_id_after = std::max(next_id_, node.global_id + 1);
    }
    node.global_id = assigned_id;
    const auto [it, inserted] =
        nodes_.emplace(assigned_id, NodeRecord{std::move(node)});
    if (!inserted) {
        std::fprintf(stderr,
                     "[Error] (execution_driven/node_store) add_node: "
                     "duplicate global_id %llu (fail-closed)\n",
                     static_cast<unsigned long long>(assigned_id));
        std::exit(EXIT_FAILURE);
    }
    next_id_ = next_id_after;
    if (it->second.unresolved_parents == 0) {
        free_ids_.insert(assigned_id);
        record_free_id_insert(assigned_id);
    }
    return assigned_id;
}

bool NodeStore::is_finished_or_erased(const uint64_t node_id) const {
    if (node_id == 0 || node_id >= next_id_) return false;
    const auto found = nodes_.find(node_id);
    return found == nodes_.end() || found->second.finished;
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
    record_free_id_erase(child);
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

bool NodeStore::is_free_id(const uint64_t node_id) const {
    ++ready_set_stats_.membership_queries;
    return free_ids_.find(node_id) != free_ids_.end();
}

const std::vector<uint64_t>& NodeStore::free_delta_since_snapshot() const {
    ++ready_set_stats_.delta_snapshot_reads;
    if (!free_delta_snapshot_cache_valid_) {
        free_delta_snapshot_cache_.assign(free_delta_ids_.begin(),
                                          free_delta_ids_.end());
        ++ready_set_stats_.delta_snapshot_materializations;
        ready_set_stats_.delta_snapshot_ids_copied +=
            static_cast<uint64_t>(free_delta_ids_.size());
        free_delta_snapshot_cache_valid_ = true;
    }
    return free_delta_snapshot_cache_;
}

void NodeStore::record_free_id_insert(const uint64_t node_id) {
    ++ready_set_stats_.free_insert_updates;
    ++ready_set_stats_.delta_set_insert_calls;
    // Advance the generation before the potentially allocating delta-set
    // insertion. free_ids_ was already mutated by the caller; if diagnostics
    // bookkeeping throws, a Workload snapshot must still be invalidated.
    if (free_id_version_ != std::numeric_limits<uint64_t>::max()) {
        ++free_id_version_;
    }
    if (free_delta_ids_.insert(node_id).second) {
        ++ready_set_stats_.delta_insertions;
        free_delta_snapshot_cache_valid_ = false;
    }
}

void NodeStore::record_free_id_erase(const uint64_t node_id) {
    ++ready_set_stats_.free_erase_updates;
    ++ready_set_stats_.delta_set_erase_calls;
    // As above, the authoritative free_ids_ erase has already happened.
    if (free_id_version_ != std::numeric_limits<uint64_t>::max()) {
        ++free_id_version_;
    }
    if (free_delta_ids_.erase(node_id) != 0) {
        // The id was freed and re-taken before any snapshot consumed it.
        ++ready_set_stats_.delta_erasures;
        free_delta_snapshot_cache_valid_ = false;
    } else {
        // v11 Phase 4: the id was not a pending insert, so it may be a
        // member of a caller-held snapshot vector -- record it for the
        // splice. Erases of ids that no snapshot ever captured are
        // harmless spurious entries (the splice set-difference simply
        // never matches them).
        free_erase_delta_ids_.insert(node_id);
    }
}

void NodeStore::fill_free_node_snapshot(std::vector<uint64_t>& out) const {
    // The output owns a stable, ascending copy even when callbacks mutate the
    // authoritative set during the subsequent issue pass. Iterating the set
    // costs O(R), like copying the former vector mirror, and avoids the
    // O(R) suffix shift on every issued id that the mirror required.
    ++ready_set_stats_.snapshot_calls;
    ready_set_stats_.snapshot_ids_copied +=
        static_cast<uint64_t>(free_ids_.size());
    out.assign(free_ids_.begin(), free_ids_.end());
    free_delta_ids_.clear();
    free_erase_delta_ids_.clear();
    free_delta_snapshot_cache_.clear();
    free_delta_snapshot_cache_valid_ = true;
    free_delta_window_base_version_ = free_id_version_;
}

bool NodeStore::splice_free_node_snapshot(
    std::vector<uint64_t>& inout) const {
    // v11 Phase 4 (issue-pass increment, 方案 §6.1). Equivalence argument in
    // NodeStore.hh at the declaration. Implementation notes:
    //   - D_ins materializes ascending from the ordered delta set.
    //   - The merge runs into a scratch buffer and swaps at the end, so
    //     `inout` keeps its previous exact content if any allocation
    //     throws (strong guarantee -- the caller's validity tags may keep
    //     describing it across the exception).
    //   - Both deltas are consumed and the window re-armed at the current
    //     generation, exactly like a full fill.
    std::vector<uint64_t> inserts(free_delta_ids_.begin(),
                                  free_delta_ids_.end());
    std::vector<uint64_t> erases(free_erase_delta_ids_.begin(),
                                 free_erase_delta_ids_.end());

    std::vector<uint64_t> merged;
    merged.reserve(inout.size() + inserts.size());
    size_t base_index = 0;
    size_t erase_index = 0;
    size_t insert_index = 0;
    const uint64_t kSentinel = std::numeric_limits<uint64_t>::max();
    while (base_index < inout.size() || insert_index < inserts.size()) {
        if (base_index < inout.size() && erase_index < erases.size()) {
            if (erases[erase_index] == inout[base_index]) {
                // Erased-not-reinserted base member: drop it. (If it was
                // reinserted later, the insert stream re-adds it at its
                // live-sorted position.)
                ++base_index;
                ++erase_index;
                continue;
            }
            if (erases[erase_index] < inout[base_index]) {
                // Spurious erase entry (the id never entered this vector
                // -- e.g. a no-op partition erase). It can never match any
                // remaining base element either.
                ++erase_index;
                continue;
            }
        }
        const uint64_t base_next =
            base_index < inout.size() ? inout[base_index] : kSentinel;
        const uint64_t insert_next =
            insert_index < inserts.size() ? inserts[insert_index] : kSentinel;
        if (base_next < insert_next) {
            merged.push_back(base_next);
            ++base_index;
        } else if (insert_next < base_next) {
            merged.push_back(insert_next);
            ++insert_index;
        } else {
            // Erased-then-reinserted id: the insert stream (live
            // membership) wins and the id is emitted exactly once.
            merged.push_back(insert_next);
            ++base_index;
            ++insert_index;
        }
    }
    ++ready_set_stats_.splice_calls;
    ready_set_stats_.splice_delta_ids +=
        static_cast<uint64_t>(inserts.size() + erases.size());
    ready_set_stats_.splice_ids_written +=
        static_cast<uint64_t>(merged.size());
    inout.swap(merged);
    free_delta_ids_.clear();
    free_erase_delta_ids_.clear();
    free_delta_snapshot_cache_.clear();
    free_delta_snapshot_cache_valid_ = true;
    free_delta_window_base_version_ = free_id_version_;
    return true;
}

void NodeStore::mark_issued(uint64_t node_id) {
    auto it = nodes_.find(node_id);
    if (it == nodes_.end() || it->second.issued || it->second.finished) {
        return;
    }
    it->second.issued = true;
    free_ids_.erase(node_id);
    record_free_id_erase(node_id);
}

void NodeStore::finish_node(uint64_t node_id) {
    auto it = nodes_.find(node_id);
    if (it == nodes_.end() || it->second.finished) {
        return;  // idempotent: the only dependency-release entry
    }
    it->second.finished = true;
    free_ids_.erase(node_id);
    record_free_id_erase(node_id);
    for (const auto child : it->second.children) {
        auto child_it = nodes_.find(child);
        if (child_it == nodes_.end()) {
            continue;  // M2: child already collected (it finished earlier)
        }
        // F1-CPP C1-5 同款镜像(2026-09-30):原来是 NDEBUG 剥离的 assert;
        // Release 构建下被破坏的不变量会让 uint32 计数器静默回绕而不是
        // 大声失败。改用无条件 fail-closed 通道,点名节点 id。
        if (child_it->second.unresolved_parents == 0) {
            std::fprintf(stderr,
                         "[Error] (execution_driven/nodestore) finish_node "
                         "releases child %llu of node %llu with "
                         "unresolved_parents == 0\n",
                         static_cast<unsigned long long>(child),
                         static_cast<unsigned long long>(node_id));
            std::abort();
        }
        if (--child_it->second.unresolved_parents == 0) {
            free_ids_.insert(child);
            record_free_id_insert(child);
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
        // F1-CPP C1-5 同款镜像(2026-09-30):同上,无条件 fail-closed。
        if (parent_it->second.unfinished_children == 0) {
            std::fprintf(stderr,
                         "[Error] (execution_driven/nodestore) finish_node "
                         "decrements parent %llu of node %llu with "
                         "unfinished_children == 0\n",
                         static_cast<unsigned long long>(parent),
                         static_cast<unsigned long long>(node_id));
            std::abort();
        }
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

// W1 (joint work ledger): stable NodeKind names. The Python ledger keys
// works on these strings; renumbering the enum must not drift the wire.
namespace {
const char* node_kind_name(const NodeKind kind) {
    switch (kind) {
        case NodeKind::Invalid:
            return "invalid";
        case NodeKind::Metadata:
            return "metadata";
        case NodeKind::MemLoad:
            return "mem_load";
        case NodeKind::MemStore:
            return "mem_store";
        case NodeKind::Compute:
            return "compute";
        case NodeKind::CommSend:
            return "comm_send";
        case NodeKind::CommRecv:
            return "comm_recv";
        case NodeKind::CommCollective:
            return "comm_collective";
    }
    return "invalid";
}
}  // namespace

std::vector<NodeStore::NodeWorkView> NodeStore::work_ledger_view() const {
    // W1: per-node const facts for the causal work ledger. Ascending node id
    // (unordered_map iteration order must never leak into the diff). Parents
    // recorded by add_dependency are exactly those that were unfinished when
    // the edge was recorded; finished parents never block, so the exported
    // blocking set filters to resident-and-unfinished records.
    std::vector<std::pair<uint64_t, const NodeRecord*>> records;
    records.reserve(nodes_.size());
    for (const auto& [node_id, record] : nodes_) {
        records.emplace_back(node_id, &record);
    }
    std::sort(records.begin(), records.end(),
              [](const auto& lhs, const auto& rhs) {
                  return lhs.first < rhs.first;
              });
    std::vector<NodeWorkView> views;
    views.reserve(records.size());
    for (const auto& entry : records) {
        const uint64_t node_id = entry.first;
        const NodeRecord& record = *entry.second;
        NodeWorkView view;
        view.node_id = node_id;
        view.request_id = record.node.request_id;
        view.stage = record.node.stage;
        view.generation = record.node.generation;
        view.kind = static_cast<int>(record.node.kind);
        view.kind_name = node_kind_name(record.node.kind);
        view.issued = record.issued;
        view.finished = record.finished;
        view.unresolved_parents = record.unresolved_parents;
        for (const auto parent : record.parents) {
            const auto parent_it = nodes_.find(parent);
            if (parent_it == nodes_.end() || parent_it->second.finished) {
                continue;  // collected or finished: no longer blocks
            }
            view.blocking_parents.push_back(parent);
        }
        switch (record.node.kind) {
            case NodeKind::Compute:
                view.num_ops = record.node.compute.num_ops;
                view.runtime_ns = record.node.compute.runtime_ns;
                view.tensor_size = record.node.compute.tensor_size;
                break;
            case NodeKind::CommSend:
            case NodeKind::CommRecv:
                view.bytes = record.node.comm.bytes;
                break;
            case NodeKind::CommCollective:
                view.bytes = record.node.coll.bytes;
                view.runtime_ns = record.node.compute.runtime_ns;
                break;
            case NodeKind::MemLoad:
            case NodeKind::MemStore:
                view.tensor_size = record.node.mem.tensor_size;
                break;
            case NodeKind::Metadata:
            case NodeKind::Invalid:
                break;  // control/no-work nodes: lifecycle facts only
        }
        views.push_back(std::move(view));
    }
    return views;
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

bool NodeStoreGraphSource::fill_dep_free_id_snapshot(
    std::vector<uint64_t>& out) {
    store_.fill_free_node_snapshot(out);
    return true;
}

bool NodeStoreGraphSource::is_dep_free_member(const uint64_t node_id) {
    return store_.is_free_id(node_id);
}

void NodeStoreGraphSource::for_each_dep_free(
    const std::function<void(const NodeView&)>& consume) {
    // The reusable vector is still a snapshot, so consume() may safely mutate
    // the free set / release dependencies. Newly released children are not
    // visited until the next issue pass.
    fill_dep_free_id_snapshot(dep_free_scratch_ids_);
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
            // Die backend: ET static attributes carry the selected typed
            // payload as JSON text. Legacy die_op_spec keeps its strict
            // parser; coarse die_task_v2 is preserved immutably for the
            // shared Runtime compiler and never normalized through legacy.
            {
                const std::string op_role =
                    node->get_attr<std::string>("op_role", "compute");
                if (op_role == "compute") {
                    nv.compute.op_role = OpRole::Compute;
                } else if (op_role == "control_marker") {
                    nv.compute.op_role = OpRole::ControlMarker;
                } else {
                    std::cerr << "[Error] (execution_driven/node_store) "
                                 "view_of: unknown op_role attribute \""
                              << op_role << "\" (legal: compute|"
                                 "control_marker)" << std::endl;
                    std::exit(EXIT_FAILURE);
                }
                if (node->has_attr("die_op_spec")) {
                    if (node->has_attr("die_task")) {
                        std::cerr << "[Error] (execution_driven/node_store) "
                                     "view_of: die_op_spec and die_task are "
                                     "mutually exclusive" << std::endl;
                        std::exit(EXIT_FAILURE);
                    }
                    const std::string spec_json_text =
                        node->get_attr<std::string>("die_op_spec", "");
                    nlohmann::json spec_json;
                    try {
                        spec_json = nlohmann::json::parse(spec_json_text);
                    } catch (const std::exception& e) {
                        std::cerr << "[Error] (execution_driven/node_store) "
                                     "view_of: die_op_spec attribute is not "
                                     "valid JSON: "
                                  << e.what() << std::endl;
                        std::exit(EXIT_FAILURE);
                    }
                    try {
                        nv.compute.die_spec =
                            DieModel::parse_die_op_spec(spec_json);
                    } catch (const std::exception& e) {
                        std::cerr << "[Error] (execution_driven/node_store) "
                                     "view_of: die_op_spec attribute "
                                     "rejected: "
                                  << e.what() << std::endl;
                        std::exit(EXIT_FAILURE);
                    }
                }
                if (node->has_attr("die_task")) {
                    if (nv.compute.runtime_ns != 0) {
                        std::cerr << "[Error] (execution_driven/node_store) "
                                     "view_of: die_task requires runtime_ns == "
                                     "0" << std::endl;
                        std::exit(EXIT_FAILURE);
                    }
                    if (nv.compute.op_role != OpRole::Compute) {
                        std::cerr << "[Error] (execution_driven/node_store) "
                                     "view_of: die_task requires op_role=compute"
                                  << std::endl;
                        std::exit(EXIT_FAILURE);
                    }
                    const std::string task_json_text =
                        node->get_attr<std::string>("die_task", "");
                    nlohmann::json task_json;
                    try {
                        task_json = nlohmann::json::parse(task_json_text);
                    } catch (const std::exception& e) {
                        std::cerr << "[Error] (execution_driven/node_store) "
                                     "view_of: die_task attribute is not "
                                     "valid JSON: " << e.what() << std::endl;
                        std::exit(EXIT_FAILURE);
                    }
                    if (!task_json.is_object() ||
                        !task_json.contains("schema") ||
                        !task_json["schema"].is_string() ||
                        task_json["schema"].get<std::string>() !=
                            "coarse_die_task_v2") {
                        std::cerr << "[Error] (execution_driven/node_store) "
                                     "view_of: die_task must be an object with "
                                     "schema=coarse_die_task_v2" << std::endl;
                        std::exit(EXIT_FAILURE);
                    }
                    nv.compute.die_task_json =
                        std::make_shared<const nlohmann::json>(
                            std::move(task_json));
                }
            }
            break;
        case NodeKind::MemLoad:
        case NodeKind::MemStore:
            // issue_remote_mem / issue_local_hbm_kv_restore consume
            // tensor_size (strict in the baseline). sh_3.0: the
            // is_local_hbm_kv_restore flag routes MEM_LOAD to the HBM
            // restore model (baseline sh_3.0 Workload MEM-routing semantics).
            // N-way HBM contention: hbm-access-mode (0 none / 1 read /
            // 2 write) charges the pool endpoint's local-HBM half.
            if (node->has_attr("tensor_size")) {
                nv.compute.tensor_size = node->tensor_size<uint64_t>();
                nv.mem.tensor_size = nv.compute.tensor_size;
            }
            nv.mem.is_local_hbm_kv_restore =
                node->get_attr<bool>("is_local_hbm_kv_restore", false);
            // Range-check before narrowing: a uint64 attr truncated to int
            // could wrap into the legal {0,1,2} domain and silently flip the
            // read/write direction (face-LRU audit 2026-09-29, L2 residual).
            {
                const uint64_t raw_mode =
                    node->get_attr<uint64_t>("hbm-access-mode", 0);
                if (raw_mode > 2) {
                    LoggerFactory::get_logger("workload")
                        ->critical("hbm-access-mode must be 0/1/2 (got {})",
                                   raw_mode);
                    std::exit(EXIT_FAILURE);
                }
                nv.mem.hbm_access_mode = static_cast<int>(raw_mode);
            }
            break;
        case NodeKind::CommSend:
        case NodeKind::CommRecv:
            // Baseline defaults preserved: comm_src/dst default to the local
            // rank (this rank == adapter rank_), tag defaults 0; size was
            // strict at consumption time, guard keeps eager reads identical.
            // N-way HBM contention: hbm-charge defaults true (endpoint
            // COMM_READ/COMM_WRITE job); false = pass-through, no job.
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
            nv.comm.hbm_charge =
                node->get_attr<bool>("hbm-charge", true);
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
