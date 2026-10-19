/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

NodeStore -- execution-driven mechanism layer (wscllm phase 1).
Implementation (方案 §4 步骤 1-4).
*******************************************************************************/

#include "astra-sim/workload/execution_driven/NodeStore.hh"

#include <cassert>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <map>
#include <utility>

namespace AstraSim {
namespace ExecutionDriven {

// ---------------------------------------------------------------------------
// NodeStore
// ---------------------------------------------------------------------------

uint64_t NodeStore::add_node(OnlineNode node) {
    if (node.global_id == 0) {
        node.global_id = next_id_++;
    } else {
        // R1-A-L4（2026-10-01 补齐）：next_id_ = global_id + 1 在
        // global_id == UINT64_MAX 处回绕到 0，静默重置 id 空间。只有
        // 合成调用方能触达该哨兵值——fail-closed 而不是腐蚀 id 流。
        if (node.global_id == std::numeric_limits<uint64_t>::max()) {
            std::fprintf(stderr,
                         "[Error] (execution_driven/nodestore) add_node: "
                         "global_id UINT64_MAX would overflow next_id_ "
                         "(fail-closed\n");
            std::abort();
        }
        next_id_ = std::max(next_id_, node.global_id + 1);
    }
    const auto [it, inserted] =
        nodes_.emplace(node.global_id, NodeRecord{std::move(node)});
    // Unconditional fail-closed (not NDEBUG-stripped): a duplicate id must
    // never silently drop the new node AND re-arm the existing record into
    // free_ids_ (an issued node would re-enter the issue set). Same contract
    // as RequestIngress::register_queue_index's duplicate handling.
    if (!inserted) {
        std::fprintf(stderr,
                     "[Error] (execution_driven/nodestore) duplicate node "
                     "id: %llu\n",
                     static_cast<unsigned long long>(it->first));
        std::abort();
    }
    if (it->second.unresolved_parents == 0) {
        free_ids_.insert(FreeEntry{node.global_id, &it->second.node});
    }
    return node.global_id;
}

void NodeStore::add_dependency(uint64_t parent, uint64_t child,
                               DepKind /*kind*/) {
    // First version: every kind resolves identically as plain precedence and
    // the kind is deliberately NOT stored per edge (the 步骤 1-5 note forbids
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
    // S5 守卫（2026-10-01 补齐）：落在已 issued/finished 子节点（或自环）
    // 上的迟到边曾使 free 集与 issue/finish 位失同步——该边抬高
    // unresolved_parents 而子节点已不在 free 集里，父节点随后 finish 又
    // 把已 issued 的子节点重新插回 free_ids_，for_each_dep_free 会把该
    // 节点第二次交给 Workload::issue（double terminal -> fail-closed
    // abort）。提交器的强制 preflight 只给本批新鲜节点（构造上未
    // issued）连边，官方路径永不触发；这么做的调用方是缺陷——点名并
    // 终止（与 add_node 的重复 id 守卫同一 fail-closed 风格）。
    if (child_it->second.issued || child_it->second.finished ||
        parent == child) {
        std::fprintf(stderr,
                     "[Error] (execution_driven/nodestore) add_dependency: "
                     "late edge onto issued/finished node %llu (parent "
                     "%llu) -- free-set accounting would corrupt "
                     "(fail-closed)\n",
                     static_cast<unsigned long long>(child),
                     static_cast<unsigned long long>(parent));
        std::abort();
    }
    // A child added with no parents is free; recording a real parent takes it
    // out of the free set again.
    child_it->second.unresolved_parents++;
    free_ids_.erase(FreeEntry{child, &child_it->second.node});
    child_it->second.parents.push_back(parent);
    parent_it->second.children.push_back(child);
    // M2 node GC: count the live child on the parent (the child of a
    // validated edge is always a fresh, unfinished batch node).
    ++parent_it->second.unfinished_children;
}

void NodeStore::fill_free_node_snapshot(std::vector<uint64_t>& out) const {
    // 2.5-3: ids still come out ascending (entries are ordered by id);
    // the view variant below additionally hands out the node addresses.
    out.clear();
    out.reserve(free_ids_.size());
    for (const auto& entry : free_ids_) {
        out.push_back(entry.id);
    }
}

void NodeStore::fill_free_view_snapshot(
    std::vector<const OnlineNode*>& out) const {
    // 2.5-3: pointer snapshot. The O(F) copy itself stays (the snapshot
    // contract: a consume() callback may mutate the free set mid-pass, and
    // nodes freed mid-pass must NOT join the current pass), but the per-id
    // second hash lookup is gone. Entries point at records that are free,
    // hence unfinished, hence never collected by the quiescent-point GC.
    out.clear();
    out.reserve(free_ids_.size());
    for (const auto& entry : free_ids_) {
        assert(entry.node != nullptr);
        out.push_back(entry.node);
    }
}

void NodeStore::mark_issued(uint64_t node_id) {
    auto it = nodes_.find(node_id);
    if (it == nodes_.end() || it->second.issued || it->second.finished) {
        return;
    }
    it->second.issued = true;
    free_ids_.erase(FreeEntry{node_id, &it->second.node});
}

void NodeStore::finish_node(uint64_t node_id) {
    // 2.5-1: by-id wrapper for the non-handle callers (skip_invalid,
    // fixtures); the shared record body lives in finish_node_record.
    finish_node_record(find_record(node_id));
}

void NodeStore::finish_node_record(const OnlineRecordRef& ref) {
    // 2.5-1: the body below is the former finish_node verbatim, with the
    // by-id find replaced by the caller-minted record. The ref is minted
    // inside the same Workload callback (find_record/take_node_record);
    // NodeStore records are erased ONLY by collect_garbage() at the
    // committer's end-of-commit quiescent point -- never inside a
    // callback -- and unordered_map element references survive
    // insert/rehash of other elements, so the record cannot have died or
    // moved between mint and use.
    if (ref.record == nullptr) {
        return;  // unknown id: same early-out as the by-id path
    }
    NodeRecord& rec = *static_cast<NodeRecord*>(ref.record);
    // The fail-closed messages in the body below name the node id.
    const uint64_t node_id = ref.id;
    assert(ref.node != nullptr);
    assert(ref.id == rec.node.global_id);
    assert(nodes_.find(ref.id) != nodes_.end() &&
           &nodes_.find(ref.id)->second == &rec);
    if (rec.finished) {
        return;  // idempotent: the only dependency-release entry
    }
    rec.finished = true;
    free_ids_.erase(FreeEntry{ref.id, ref.node});
    for (const auto child : rec.children) {
        auto child_it = nodes_.find(child);
        if (child_it == nodes_.end()) {
            continue;  // M2: child already collected (it finished earlier)
        }
        // Static-DAG contract: a child still blocked by this parent can
        // never have been ISSUED from the free set (issue requires
        // unresolved_parents == 0, i.e. every parent finished first), so
        // issued == 1 here means the free-set discipline itself broke.
        // F1-CPP C1-5 同款移植(2026-09-30):原来是 NDEBUG 剥离的 assert;
        // Release 构建下被破坏的不变量会让 uint32 计数器静默回绕而不是
        // 大声失败(永不释放的子节点把 run 楔死进看门狗,永不 GC 的父
        // 节点泄漏 RSS),零诊断。改用与 add_node 的 duplicate-id 门(及
        // ServiceCoordinator/HardwareResource)同款的无条件 fail-closed
        // 通道,点名节点 id 与越界计数。finished == 1 且 issued == 0 是
        // 这里的合法形态(Workload::skip_invalid 会在父节点仍在跑时终结
        // 从未执行的 invalid/metadata/零张量节点,原 Release 行为照常
        // 释放其父边)——原样保留,只把被剥离的 assert 换成无条件守卫。
        if (child_it->second.issued) {
            std::fprintf(stderr,
                         "[Error] (execution_driven/nodestore) finish_node "
                         "releases child %llu of node %llu while it is "
                         "already issued=1 (finished=%d)\n",
                         static_cast<unsigned long long>(child),
                         static_cast<unsigned long long>(node_id),
                         child_it->second.finished ? 1 : 0);
            std::abort();
        }
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
            free_ids_.insert(FreeEntry{child, &child_it->second.node});
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
    if (gc_enabled_ && rec.unfinished_children == 0) {
        gc_fifo_.push_back(node_id);
    }
    for (const auto parent : rec.parents) {
        auto parent_it = nodes_.find(parent);
        if (parent_it == nodes_.end()) {
            continue;  // M2: parent already collected
        }
        // F1-CPP C1-5 同款移植(2026-09-30):原是 NDEBUG 剥离的 assert
        // (见上方 child 侧注释);同款无条件 fail-closed 转换。
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

const OnlineNode* NodeStore::node_ptr(uint64_t node_id) const {
    const auto it = nodes_.find(node_id);
    return it == nodes_.end() ? nullptr : &it->second.node;
}

OnlineRecordRef NodeStore::find_record(uint64_t node_id) {
    // 2.5-1: the single find of the online terminal/issue sequence.
    const auto it = nodes_.find(node_id);
    if (it == nodes_.end()) {
        return OnlineRecordRef{};
    }
    OnlineRecordRef ref;
    ref.id = it->first;
    ref.node = &it->second.node;
    ref.stats = &it->second.node.online_statistics;
    ref.record = &it->second;
    return ref;
}

bool NodeStore::claim_terminal_observed(const OnlineRecordRef& ref) {
    // 2.5-1: mark_terminal_observed's exact semantics, minus the find. The
    // ref is minted inside the same Workload callback; the record cannot
    // have been erased since (GC only at the committer's quiescent point).
    assert(ref.record != nullptr);
    NodeRecord& rec = *static_cast<NodeRecord*>(ref.record);
    assert(&rec.node == ref.node);
    if (!rec.issued || rec.finished || rec.terminal_observed) {
        return false;
    }
    rec.terminal_observed = true;
    return true;
}

OnlineRecordRef NodeStore::take_node_record(uint64_t node_id) {
    // 2.5-1 issue path: exactly baseline take_node(node_id) followed by
    // mutable_online_statistics(node_id) -- mark_issued's no-op guards on
    // the minted record, and the stats pointer handed out regardless of
    // issued state, like mutable_online_statistics does.
    OnlineRecordRef ref = find_record(node_id);
    if (!ref) {
        return ref;
    }
    NodeRecord& rec = *static_cast<NodeRecord*>(ref.record);
    if (!rec.issued && !rec.finished) {
        rec.issued = true;
        free_ids_.erase(FreeEntry{ref.id, ref.node});
    }
    return ref;
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
    // 2.5-1: by-id wrapper (one find) over the shared record body.
    const OnlineRecordRef ref = find_record(node_id);
    if (!ref) {
        return false;
    }
    return claim_terminal_observed(ref);
}

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
    // M2: absent == collected for committed store ids (see header). Consumed
    // only by the committer's liveness preflight and its commit-tail affine /
    // anchor resolution (GraphBatchCommitter.cc).
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

void NodeStoreGraphSource::for_each_dep_free(
    const std::function<void(const NodeView&)>& consume) {
    // The reusable vector is still a snapshot, so consume() may safely mutate
    // the free set / release dependencies. Newly released children are not
    // visited until the next issue pass.
    // 2.5-3: entries carry the record's node address, so the issue pass
    // consumes views with zero extra nodes_ lookups.
    store_.fill_free_view_snapshot(dep_free_scratch_nodes_);
    for (const OnlineNode* node : dep_free_scratch_nodes_) {
        assert(node != nullptr);
        consume(*node);
    }
}

void NodeStore::for_each_unfinished(const std::function<void(
                                          const OnlineNode&, uint32_t, bool)>&
                                          consume) const {
    // DeadEnd-census diagnostic only (2026-10-11): ascending-id walk over
    // the retained records; unfinished nodes only. Read-only.
    std::vector<const NodeRecord*> pending;
    pending.reserve(nodes_.size());
    for (const auto& entry : nodes_) {
        if (!entry.second.finished) {
            pending.push_back(&entry.second);
        }
    }
    std::sort(pending.begin(), pending.end(),
              [](const NodeRecord* lhs, const NodeRecord* rhs) {
                  return lhs->node.global_id < rhs->node.global_id;
              });
    for (const NodeRecord* record : pending) {
        consume(record->node, record->unresolved_parents, record->issued);
    }
}

void NodeStoreGraphSource::finish_node(uint64_t node_id) {
    store_.finish_node(node_id);
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

}  // namespace ExecutionDriven
}  // namespace AstraSim
