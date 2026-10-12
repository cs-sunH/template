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

namespace {

uint64_t gcd_u64(uint64_t a, uint64_t b) {
    while (b != 0) {
        const uint64_t r = a % b;
        a = b;
        b = r;
    }
    return a;
}

uint64_t modular_inverse_u64(uint64_t a, uint64_t modulus) {
    __int128 t = 0;
    __int128 next_t = 1;
    __int128 r = modulus;
    __int128 next_r = a % modulus;
    while (next_r != 0) {
        const __int128 q = r / next_r;
        const __int128 old_t = t;
        t = next_t;
        next_t = old_t - q * next_t;
        const __int128 old_r = r;
        r = next_r;
        next_r = old_r - q * next_r;
    }
    if (r != 1) throw std::logic_error("NodeStore modular inverse failure");
    t %= static_cast<__int128>(modulus);
    if (t < 0) t += modulus;
    return static_cast<uint64_t>(t);
}

bool virtual_progressions_intersect(
    const NodeStore::VirtualIdProgression& a,
    const NodeStore::VirtualIdProgression& b) {
    const unsigned __int128 a_last =
        static_cast<unsigned __int128>(a.first_id) +
        static_cast<unsigned __int128>(a.stride) * (a.count - 1);
    const unsigned __int128 b_last =
        static_cast<unsigned __int128>(b.first_id) +
        static_cast<unsigned __int128>(b.stride) * (b.count - 1);
    const unsigned __int128 low = std::max(a.first_id, b.first_id);
    const unsigned __int128 high = std::min(a_last, b_last);
    if (low > high) return false;

    const uint64_t gcd = gcd_u64(a.stride, b.stride);
    const __int128 delta = static_cast<__int128>(b.first_id) -
                           static_cast<__int128>(a.first_id);
    if (delta % static_cast<__int128>(gcd) != 0) return false;
    const uint64_t modulus = b.stride / gcd;
    uint64_t k = 0;
    if (modulus > 1) {
        __int128 reduced = (delta / static_cast<__int128>(gcd)) %
                           static_cast<__int128>(modulus);
        if (reduced < 0) reduced += modulus;
        const uint64_t inverse = modular_inverse_u64(a.stride / gcd, modulus);
        k = static_cast<uint64_t>(
            (static_cast<unsigned __int128>(reduced) * inverse) % modulus);
    }
    const unsigned __int128 period =
        static_cast<unsigned __int128>(a.stride / gcd) * b.stride;
    unsigned __int128 first =
        static_cast<unsigned __int128>(a.first_id) +
        static_cast<unsigned __int128>(a.stride) * k;
    if (first < low) {
        const unsigned __int128 gap = low - first;
        first += (gap / period + (gap % period != 0 ? 1 : 0)) * period;
    }
    return first <= high;
}

unsigned __int128 virtual_progression_intersection_count(
    const NodeStore::VirtualIdProgression& a,
    const NodeStore::VirtualIdProgression& b) {
    const unsigned __int128 a_last =
        static_cast<unsigned __int128>(a.first_id) +
        static_cast<unsigned __int128>(a.stride) * (a.count - 1);
    const unsigned __int128 b_last =
        static_cast<unsigned __int128>(b.first_id) +
        static_cast<unsigned __int128>(b.stride) * (b.count - 1);
    const unsigned __int128 low = std::max(a.first_id, b.first_id);
    const unsigned __int128 high = std::min(a_last, b_last);
    if (low > high) return 0;

    const uint64_t gcd = gcd_u64(a.stride, b.stride);
    const __int128 delta = static_cast<__int128>(b.first_id) -
                           static_cast<__int128>(a.first_id);
    if (delta % static_cast<__int128>(gcd) != 0) return 0;
    const uint64_t modulus = b.stride / gcd;
    uint64_t k = 0;
    if (modulus > 1) {
        __int128 reduced = (delta / static_cast<__int128>(gcd)) %
                           static_cast<__int128>(modulus);
        if (reduced < 0) reduced += modulus;
        const uint64_t inverse = modular_inverse_u64(a.stride / gcd, modulus);
        k = static_cast<uint64_t>(
            (static_cast<unsigned __int128>(reduced) * inverse) % modulus);
    }
    const unsigned __int128 period =
        static_cast<unsigned __int128>(a.stride / gcd) * b.stride;
    unsigned __int128 first =
        static_cast<unsigned __int128>(a.first_id) +
        static_cast<unsigned __int128>(a.stride) * k;
    if (first < low) {
        const unsigned __int128 gap = low - first;
        first += (gap / period + (gap % period != 0 ? 1 : 0)) * period;
    }
    if (first > high) return 0;
    return (high - first) / period + 1;
}

}  // namespace

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
    if (is_virtual_id(assigned_id)) {
        std::fprintf(stderr,
                     "[Error] (execution_driven/node_store) add_node: "
                     "global_id %llu is a reserved virtual id\n",
                     static_cast<unsigned long long>(assigned_id));
        std::abort();
    }
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
        record_free_id_insert(
            assigned_id,
            it->second.node.compute.exec_role ==
                ComputeExecRole::RemoteReadPipelineAnchor);
    }
    return assigned_id;
}

void NodeStore::reserve_logical_id_span(
    const uint64_t first_store_id, const uint64_t width,
    const std::vector<VirtualIdProgression>& virtual_progressions) {
    reserve_logical_id_span_impl(first_store_id, width, virtual_progressions,
                                 nullptr);
}

void NodeStore::reserve_logical_id_span_impl(
    const uint64_t first_store_id, const uint64_t width,
    const std::vector<VirtualIdProgression>& virtual_progressions,
    VirtualProgressionSweepWork* const work) {
    const uint64_t max_id = std::numeric_limits<uint64_t>::max();
    if (width == 0 || first_store_id == 0 || first_store_id >= max_id ||
        width > max_id - first_store_id) {
        throw std::invalid_argument(
            "NodeStore::reserve_logical_id_span requires a non-empty, "
            "non-overflowing span in the allocatable id domain");
    }
    if (first_store_id != next_id_) {
        throw std::logic_error(
            "NodeStore::reserve_logical_id_span must begin at next_auto_id "
            "(first_id=" + std::to_string(first_store_id) +
            ", next_auto_id=" + std::to_string(next_id_) + ")");
    }

    const uint64_t end_exclusive = first_store_id + width;
    std::vector<VirtualIdProgression> ordered = virtual_progressions;
    std::sort(ordered.begin(), ordered.end(),
              [](const auto& a, const auto& b) {
                  return a.first_id < b.first_id;
              });
    std::map<uint64_t, std::vector<size_t>> expires_at;
    std::set<size_t> active;
    for (size_t i = 0; i < ordered.size(); ++i) {
        const auto& progression = ordered[i];
        if (progression.count == 0 || progression.stride == 0 ||
            progression.first_id < first_store_id ||
            progression.first_id >= end_exclusive) {
            throw std::invalid_argument(
                "NodeStore virtual progression must be non-empty and start "
                "inside the reserved logical span");
        }
        const unsigned __int128 last =
            static_cast<unsigned __int128>(progression.first_id) +
            static_cast<unsigned __int128>(progression.stride) *
                (progression.count - 1);
        if (last >= end_exclusive) {
            throw std::invalid_argument(
                "NodeStore virtual progression exceeds reserved logical span");
        }

        // Inclusive-envelope sweep. Expiring only last < first preserves
        // endpoint contact as a candidate; the exact AP predicate decides
        // whether represented IDs actually intersect.
        while (!expires_at.empty() &&
               expires_at.begin()->first < progression.first_id) {
            for (const size_t expired : expires_at.begin()->second) {
                active.erase(expired);
            }
            expires_at.erase(expires_at.begin());
        }
        for (const size_t active_index : active) {
            if (work != nullptr) {
                ++work->envelope_candidates;
                ++work->exact_intersection_tests;
            }
            if (virtual_progressions_intersect(ordered[active_index],
                                               progression)) {
                throw std::invalid_argument(
                    "NodeStore virtual progressions overlap");
            }
        }
        active.insert(i);
        expires_at[static_cast<uint64_t>(last)].push_back(i);
    }

    if (!ordered.empty()) {
        VirtualIdReservationBlock block;
        block.first_id = first_store_id;
        block.end_exclusive = end_exclusive;
        block.progressions = std::move(ordered);
        block.interval_nodes.reserve(block.progressions.size());
        std::vector<size_t> progression_indexes;
        progression_indexes.reserve(block.progressions.size());
        for (size_t index = 0; index < block.progressions.size(); ++index) {
            progression_indexes.push_back(index);
        }
        block.interval_root = build_virtual_id_interval_tree(
            block, std::move(progression_indexes));

        // Physical-only reservations add no virtual membership metadata.
        // The allocation frontier still advances below, while queries over
        // any resulting gap remain false unless a later span has holes.
        virtual_id_reservation_blocks_.push_back(std::move(block));
    }

    // Do not reserve `history + incoming` on every commit: that exact-growth
    // pattern recopies the full reservation directory on every append. The
    // vector's geometric growth is amortized, and all new block/tree storage
    // has already been constructed off to the side before this publication.
    next_id_ = end_exclusive;
}

size_t NodeStore::build_virtual_id_interval_tree(
    VirtualIdReservationBlock& block,
    std::vector<size_t> progression_indexes) {
    if (progression_indexes.empty()) return kNoVirtualIntervalNode;

    std::vector<uint64_t> midpoints;
    midpoints.reserve(progression_indexes.size());
    for (const size_t index : progression_indexes) {
        const auto& progression = block.progressions[index];
        const unsigned __int128 last =
            static_cast<unsigned __int128>(progression.first_id) +
            static_cast<unsigned __int128>(progression.stride) *
                (progression.count - 1);
        if (last > std::numeric_limits<uint64_t>::max()) {
            throw std::logic_error(
                "validated virtual AP endpoint exceeds uint64_t");
        }
        const uint64_t last_id = static_cast<uint64_t>(last);
        midpoints.push_back(
            progression.first_id + (last_id - progression.first_id) / 2);
    }
    const auto median = midpoints.begin() + midpoints.size() / 2;
    std::nth_element(midpoints.begin(), median, midpoints.end());
    const uint64_t center = *median;

    std::vector<size_t> left_indexes;
    std::vector<size_t> right_indexes;
    std::vector<size_t> center_indexes;
    left_indexes.reserve(progression_indexes.size() / 2);
    right_indexes.reserve(progression_indexes.size() / 2);
    uint64_t min_first = std::numeric_limits<uint64_t>::max();
    uint64_t max_last = 0;
    for (const size_t index : progression_indexes) {
        const auto& progression = block.progressions[index];
        const unsigned __int128 last =
            static_cast<unsigned __int128>(progression.first_id) +
            static_cast<unsigned __int128>(progression.stride) *
                (progression.count - 1);
        const uint64_t last_id = static_cast<uint64_t>(last);
        min_first = std::min(min_first, progression.first_id);
        max_last = std::max(max_last, last_id);
        if (last_id < center) {
            left_indexes.push_back(index);
        } else if (progression.first_id > center) {
            right_indexes.push_back(index);
        } else {
            center_indexes.push_back(index);
        }
    }
    if (center_indexes.empty()) {
        throw std::logic_error(
            "virtual AP interval-tree median has no center interval");
    }

    std::sort(center_indexes.begin(), center_indexes.end(),
              [&block](const size_t left, const size_t right) {
                  const auto& a = block.progressions[left];
                  const auto& b = block.progressions[right];
                  return a.first_id < b.first_id ||
                      (a.first_id == b.first_id && left < right);
              });
    std::vector<size_t> by_last = center_indexes;
    std::sort(by_last.begin(), by_last.end(),
              [&block](const size_t left, const size_t right) {
                  const auto& a = block.progressions[left];
                  const auto& b = block.progressions[right];
                  const uint64_t a_last = static_cast<uint64_t>(
                      static_cast<unsigned __int128>(a.first_id) +
                      static_cast<unsigned __int128>(a.stride) *
                          (a.count - 1));
                  const uint64_t b_last = static_cast<uint64_t>(
                      static_cast<unsigned __int128>(b.first_id) +
                      static_cast<unsigned __int128>(b.stride) *
                          (b.count - 1));
                  return a_last > b_last ||
                      (a_last == b_last && left < right);
              });

    VirtualIdIntervalNode node;
    node.center = center;
    node.min_first = min_first;
    node.max_last = max_last;
    node.by_first = std::move(center_indexes);
    node.by_last = std::move(by_last);
    const size_t node_index = block.interval_nodes.size();
    block.interval_nodes.push_back(std::move(node));
    const size_t left = build_virtual_id_interval_tree(
        block, std::move(left_indexes));
    const size_t right = build_virtual_id_interval_tree(
        block, std::move(right_indexes));
    block.interval_nodes[node_index].left = left;
    block.interval_nodes[node_index].right = right;
    return node_index;
}

void NodeStore::reserve_id_range(const uint64_t first_id,
                                 const uint64_t count) {
    reserve_logical_id_span(first_id, count, {{first_id, count, 1}});
}

bool NodeStore::is_virtual_id(const uint64_t node_id) const {
    return is_virtual_id_impl(node_id, nullptr);
}

bool NodeStore::is_virtual_id_progression(const uint64_t first_id,
                                          const uint64_t count,
                                          const uint64_t stride) const {
    return is_virtual_id_progression_impl(first_id, count, stride, nullptr);
}

bool NodeStore::is_allocated_nonvirtual_progression(
    const uint64_t first_id, const uint64_t count,
    const uint64_t stride) const {
    return is_allocated_nonvirtual_progression_impl(
        first_id, count, stride, nullptr);
}

template <typename Visitor>
bool NodeStore::visit_virtual_id_block_candidates(
    const VirtualIdReservationBlock& block, const uint64_t first_id,
    const uint64_t last_id, VirtualIdQueryWork* const work,
    Visitor&& visitor) {
    if (block.interval_root == kNoVirtualIntervalNode) return false;
    const auto visit_node = [&](const auto& self,
                                const size_t node_index) -> bool {
        if (node_index == kNoVirtualIntervalNode) return false;
        if (work != nullptr) ++work->interval_tree_nodes_examined;
        const VirtualIdIntervalNode& node = block.interval_nodes[node_index];
        if (node.min_first > last_id || node.max_last < first_id) {
            return false;
        }
        const auto visit_progression = [&](const size_t progression_index) {
            const VirtualIdProgression& progression =
                block.progressions[progression_index];
            const unsigned __int128 last =
                static_cast<unsigned __int128>(progression.first_id) +
                static_cast<unsigned __int128>(progression.stride) *
                    (progression.count - 1);
            if (work != nullptr) {
                ++work->envelope_candidates;
                ++work->exact_ap_tests;
            }
            return visitor(progression, static_cast<uint64_t>(last));
        };

        if (last_id < node.center) {
            for (const size_t progression_index : node.by_first) {
                const auto& progression =
                    block.progressions[progression_index];
                if (progression.first_id > last_id) break;
                if (visit_progression(progression_index)) return true;
            }
            return self(self, node.left);
        }
        if (first_id > node.center) {
            for (const size_t progression_index : node.by_last) {
                const auto& progression =
                    block.progressions[progression_index];
                const unsigned __int128 last =
                    static_cast<unsigned __int128>(progression.first_id) +
                    static_cast<unsigned __int128>(progression.stride) *
                        (progression.count - 1);
                if (last < first_id) break;
                if (visit_progression(progression_index)) return true;
            }
            return self(self, node.right);
        }

        for (const size_t progression_index : node.by_first) {
            if (visit_progression(progression_index)) return true;
        }
        return self(self, node.left) || self(self, node.right);
    };
    return visit_node(visit_node, block.interval_root);
}

template <typename Visitor>
bool NodeStore::visit_virtual_id_candidates(
    const uint64_t first_id, const uint64_t last_id,
    VirtualIdQueryWork* const work, Visitor&& visitor) const {
    size_t low = 0;
    size_t high = virtual_id_reservation_blocks_.size();
    while (low < high) {
        const size_t middle = low + (high - low) / 2;
        if (work != nullptr) ++work->reservation_binary_probes;
        if (virtual_id_reservation_blocks_[middle].end_exclusive <= first_id) {
            low = middle + 1;
        } else {
            high = middle;
        }
    }
    for (size_t index = low;
         index < virtual_id_reservation_blocks_.size() &&
             virtual_id_reservation_blocks_[index].first_id <= last_id;
         ++index) {
        if (work != nullptr) ++work->reservation_blocks_examined;
        if (visit_virtual_id_block_candidates(
                virtual_id_reservation_blocks_[index], first_id, last_id,
                work, visitor)) {
            return true;
        }
    }
    return false;
}

bool NodeStore::is_virtual_id_impl(const uint64_t node_id,
                                   VirtualIdQueryWork* const work) const {
    size_t low = 0;
    size_t high = virtual_id_reservation_blocks_.size();
    while (low < high) {
        const size_t middle = low + (high - low) / 2;
        if (work != nullptr) ++work->reservation_binary_probes;
        if (virtual_id_reservation_blocks_[middle].first_id <= node_id) {
            low = middle + 1;
        } else {
            high = middle;
        }
    }
    if (low == 0) return false;
    const VirtualIdReservationBlock& block =
        virtual_id_reservation_blocks_[low - 1];
    if (node_id < block.first_id || node_id >= block.end_exclusive) {
        return false;
    }
    if (work != nullptr) ++work->reservation_blocks_examined;

    bool found = false;
    visit_virtual_id_block_candidates(
        block, node_id, node_id, work,
        [&](const VirtualIdProgression& progression, const uint64_t) {
            if (node_id < progression.first_id) return false;
            const uint64_t delta = node_id - progression.first_id;
            found = delta % progression.stride == 0 &&
                delta / progression.stride < progression.count;
            return found;
        });
    return found;
}

bool NodeStore::is_virtual_id_progression_impl(
    const uint64_t first_id, const uint64_t count, const uint64_t stride,
    VirtualIdQueryWork* const work) const {
    if (first_id == 0 || count == 0 || stride == 0) return false;
    const unsigned __int128 last = static_cast<unsigned __int128>(first_id) +
        static_cast<unsigned __int128>(stride) * (count - 1);
    if (last >= next_id_) return false;
    const VirtualIdProgression query{first_id, count, stride};
    unsigned __int128 covered = 0;
    bool overcovered = false;
    visit_virtual_id_candidates(first_id, static_cast<uint64_t>(last), work,
        [&](const VirtualIdProgression& hole, const uint64_t) {
            covered += virtual_progression_intersection_count(query, hole);
            overcovered = covered > count;
            return overcovered;
        });
    return !overcovered && covered == count;
}

bool NodeStore::is_allocated_nonvirtual_progression_impl(
    const uint64_t first_id, const uint64_t count, const uint64_t stride,
    VirtualIdQueryWork* const work) const {
    if (first_id == 0 || count == 0 || stride == 0) return false;
    const unsigned __int128 last = static_cast<unsigned __int128>(first_id) +
        static_cast<unsigned __int128>(stride) * (count - 1);
    if (last >= next_id_) return false;
    const VirtualIdProgression query{first_id, count, stride};
    bool intersects = false;
    visit_virtual_id_candidates(first_id, static_cast<uint64_t>(last), work,
        [&](const VirtualIdProgression& hole, const uint64_t) {
            intersects = virtual_progressions_intersect(query, hole);
            return intersects;
        });
    return !intersects;
}

uint64_t NodeStore::count_finished_or_erased_nonvirtual_progression(
    const uint64_t first_id, const uint64_t count,
    const uint64_t stride) const {
    if (!is_allocated_nonvirtual_progression(first_id, count, stride)) {
        throw std::invalid_argument(
            "NodeStore finished-count requires an allocated nonvirtual AP");
    }
    uint64_t unfinished = 0;
    for (const auto& entry : nodes_) {
        const uint64_t node_id = entry.first;
        if (node_id < first_id) continue;
        const uint64_t delta = node_id - first_id;
        if (delta % stride == 0 && delta / stride < count &&
            !entry.second.finished) {
            ++unfinished;
        }
    }
    if (unfinished > count) {
        throw std::logic_error(
            "NodeStore retained unfinished count exceeds the AP cardinality");
    }
    return count - unfinished;
}

bool NodeStore::is_finished_or_erased(const uint64_t node_id) const {
    if (node_id == 0 || node_id >= next_id_ || is_virtual_id(node_id)) return false;
    const auto found = nodes_.find(node_id);
    return found == nodes_.end() || found->second.finished;
}

void NodeStore::add_dependency(uint64_t parent, uint64_t child,
                               DepKind /*kind*/) {
    // First version: all kinds resolve identically (precedence). The kind is
    // stored per edge for the later fence upgrade (步骤 1-5 note forbids
    // building the general fence early). Dead-parent edges are tolerated:
    // a parent that already finished no longer blocks the child.
    if (is_virtual_id(parent) || is_virtual_id(child)) {
        throw std::invalid_argument(
            "NodeStore::add_dependency cannot reference a virtual id");
    }
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

void NodeStore::record_free_id_insert(const uint64_t node_id,
                                       const bool is_anchor) {
    ++ready_set_stats_.free_insert_updates;
    ++ready_set_stats_.delta_set_insert_calls;
    // Advance the generation before the potentially allocating delta-set
    // insertion. free_ids_ was already mutated by the caller; if diagnostics
    // bookkeeping throws, a Workload snapshot must still be invalidated.
    if (free_id_version_ != std::numeric_limits<uint64_t>::max()) {
        ++free_id_version_;
    }
    // F (2026-10-11): route into the role partition. Both partitions are
    // plain ordered sets, so the union stays exactly free_ids_ (the insert
    // was already made by the caller) and any partition mutation still
    // bumps the same free_id_version_ above.
    if (is_anchor) {
        free_anchor_ids_.insert(node_id);
    } else {
        free_ordinary_ids_.insert(node_id);
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
    // F: idempotent dual-partition erase -- take/finish paths do not need
    // the node's role (the record may even be gone), and set::erase on a
    // missing key is a no-op.
    free_ordinary_ids_.erase(node_id);
    free_anchor_ids_.erase(node_id);
    if (free_delta_ids_.erase(node_id) != 0) {
        // The id was freed and re-taken before any snapshot consumed it.
        ++ready_set_stats_.delta_erasures;
        free_delta_snapshot_cache_valid_ = false;
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
    free_delta_snapshot_cache_.clear();
    free_delta_snapshot_cache_valid_ = true;
}

void NodeStore::fill_free_ordinary_id_snapshot(
    std::vector<uint64_t>& out) const {
    // F (2026-10-11): same caller-owned-snapshot contract as
    // fill_free_node_snapshot, sourced from the ordinary partition only.
    // This fill consumes the same since-snapshot delta (the delta answers
    // "which ids became free after the pass's fixed member set was
    // captured", regardless of which snapshot kind captured it), so the
    // anchor gate can reject ids that entered the free set mid-pass.
    ++ready_set_stats_.ordinary_snapshot_calls;
    ready_set_stats_.ordinary_snapshot_ids_copied +=
        static_cast<uint64_t>(free_ordinary_ids_.size());
    out.assign(free_ordinary_ids_.begin(), free_ordinary_ids_.end());
    free_delta_ids_.clear();
    free_delta_snapshot_cache_.clear();
    free_delta_snapshot_cache_valid_ = true;
}

bool NodeStore::is_free_anchor_id(const uint64_t node_id) const {
    return free_anchor_ids_.find(node_id) != free_anchor_ids_.end();
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
            record_free_id_insert(
                child,
                child_it->second.node.compute.exec_role ==
                    ComputeExecRole::RemoteReadPipelineAnchor);
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
    // the committer's store_ids_ prune pass only. A descriptor-reserved
    // virtual id is neither a live entity nor a completed/GC-erased parent;
    // callers must check is_virtual_id() and reject it as a graph reference.
    return !is_virtual_id(node_id) && nodes_.find(node_id) == nodes_.end();
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

bool NodeStoreGraphSource::fill_dep_free_ordinary_id_snapshot(
    std::vector<uint64_t>& out) {
    // F (2026-10-11): the online issue pass's physical vector is the
    // ordinary partition; anchors are enumerated through the executor's
    // current-anchor set instead of this snapshot.
    store_.fill_free_ordinary_id_snapshot(out);
    return true;
}

bool NodeStoreGraphSource::is_dep_free_member(const uint64_t node_id) {
    return store_.is_free_id(node_id);
}

bool NodeStoreGraphSource::is_finished_or_erased(const uint64_t node_id) {
    return store_.is_finished_or_erased(node_id);
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
            // Die backend (joint rollout stage 1, contract §2): the ET
            // static path carries op_role / die_op_spec as ordinary string
            // attributes; the spec arrives as the JSON serialization of the
            // same DieOpSpec the online GraphBatch carries (design doc
            // §8.3). Both stay optional; a shape-less historical ET simply
            // leaves die_spec null and the die backend fails closed on real
            // COMP nodes at issue time.
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
