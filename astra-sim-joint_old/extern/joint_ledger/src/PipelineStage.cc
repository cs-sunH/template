// PipelineStage.cc -- native port of the pipeline ledger staging/commit core
// (pipeline_ledger.py PipelineLedger/PipelineLedgerV4: _gate_delivery,
// prepare_envelope's global-sequence merge walk, _stage_delivery,
// commit_prepared and the query surface).
//
// prepare() validates the global logical order, stages every transition
// through COW overlays (rows, counts, indexes, terminal-provenance treaps)
// and drives the coordinated WorkLedger transaction for node events at each
// event's exact logical position.  commit_prepared() publishes the fully
// validated update; a prepare failure destroys every staged overlay before
// returning (the prepared slot is never registered).
//
// Ratified deviations from the Python reference (full list in the report):
//  * the idempotent-replay gate compares the RAW segment bytes of this
//    delivery against the cached bytes of the last accepted delivery
//    (Python compares canonical json.dumps bytes);
//  * dependency-treap priorities use std::hash over the row's canonical
//    decimal serialization (Python uses process-salted hash(); priorities
//    shape the tree only, never the semantics);
//  * pipeline_ledger.py's per-call diagnostic counters ("work" dicts fed to
//    the treap builders) are not ported.
#include "joint_ledger/PipelineStage.hh"

#include <algorithm>
#include <array>
#include <cstdint>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "joint_ledger/LedgerTypes.hh"
#include "joint_ledger/PipelineRows.hh"
#include "joint_ledger/WorkLedgerCore.hh"

namespace jointledger {

namespace {

[[noreturn]] void fail(const std::string& message) {
    throw LedgerError(message);
}

constexpr std::size_t kNoIndex = static_cast<std::size_t>(-1);

std::string num(std::uint64_t value) { return std::to_string(value); }

// ------------------------------------------------------------ parsed rows --
// Readers over the parsed-row JSON conventions of PipelineRows.hh (values
// were fully validated by parse_envelope_v4; these keep malformed direct
// constructions failing closed with a path).

const Json& row_field(const Json& row, const char* key,
                      const std::string& path) {
    if (!row.is_object()) fail(path + " must be an object");
    const auto member = row.find(key);
    if (member == row.end()) fail(path + " is missing '" + key + "'");
    return *member;
}

std::uint64_t row_uint(const Json& row, const char* key,
                       const std::string& path) {
    return read_uint(row_field(row, key, path), path + "." + key);
}

std::optional<std::uint64_t> row_opt_uint(const Json& row, const char* key,
                                          const std::string& path) {
    const Json& value = row_field(row, key, path);
    if (value.is_null()) return std::nullopt;
    return read_uint(value, path + "." + key);
}

bool row_bool(const Json& row, const char* key, const std::string& path) {
    return read_bool(row_field(row, key, path), path + "." + key);
}

std::string row_string(const Json& row, const char* key,
                       const std::string& path) {
    return read_string(row_field(row, key, path), path + "." + key);
}

std::vector<std::uint64_t> row_uint_array(const Json& row, const char* key,
                                          const std::string& path) {
    const Json& value = row_field(row, key, path);
    if (!value.is_array()) fail(path + "." + key + " must be an array");
    std::vector<std::uint64_t> out;
    out.reserve(value.size());
    for (const Json& item : value) {
        out.push_back(read_uint(item, path + "." + key + "[]"));
    }
    return out;
}

bool sorted_uint_array_contains(const Json& row, const char* key,
                                const std::string& path,
                                std::uint64_t target) {
    const std::vector<std::uint64_t> values = row_uint_array(row, key, path);
    return std::binary_search(values.begin(), values.end(), target);
}

// ------------------------------------------------------------------- keys --

struct SlotKey {
    std::string domain;
    std::string role;             // "source" | "target"
    std::uint64_t rank = 0;
    std::int64_t shard = -1;      // -1 encodes Python's null shard ordinal
    std::uint64_t slot_index = 0;

    bool operator<(const SlotKey& other) const {
        if (domain != other.domain) return domain < other.domain;
        if (role != other.role) return role < other.role;
        if (rank != other.rank) return rank < other.rank;
        if (shard != other.shard) return shard < other.shard;
        return slot_index < other.slot_index;
    }
    bool operator==(const SlotKey& other) const {
        return domain == other.domain && role == other.role
            && rank == other.rank && shard == other.shard
            && slot_index == other.slot_index;
    }
};

struct CreditKey {
    std::string domain;
    std::uint64_t source_rank = 0;
    std::uint64_t shard = 0;

    bool operator<(const CreditKey& other) const {
        if (domain != other.domain) return domain < other.domain;
        if (source_rank != other.source_rank) {
            return source_rank < other.source_rank;
        }
        return shard < other.shard;
    }
    bool operator==(const CreditKey& other) const {
        return domain == other.domain && source_rank == other.source_rank
            && shard == other.shard;
    }
};

struct ReadyKey {
    std::string domain;
    std::uint64_t rank = 0;
    std::uint64_t slot_index = 0;
    std::uint64_t target_shard = 0;

    bool operator<(const ReadyKey& other) const {
        if (domain != other.domain) return domain < other.domain;
        if (rank != other.rank) return rank < other.rank;
        if (slot_index != other.slot_index) return slot_index < other.slot_index;
        return target_shard < other.target_shard;
    }
    bool operator==(const ReadyKey& other) const {
        return domain == other.domain && rank == other.rank
            && slot_index == other.slot_index
            && target_shard == other.target_shard;
    }
};

struct ReleaseKey {
    std::string domain;
    std::uint64_t rank = 0;
    std::uint64_t shard = 0;
    std::uint64_t slot_index = 0;

    bool operator<(const ReleaseKey& other) const {
        if (domain != other.domain) return domain < other.domain;
        if (rank != other.rank) return rank < other.rank;
        if (shard != other.shard) return shard < other.shard;
        return slot_index < other.slot_index;
    }
    bool operator==(const ReleaseKey& other) const {
        return domain == other.domain && rank == other.rank
            && shard == other.shard && slot_index == other.slot_index;
    }
};

struct WaiterKey {
    std::uint64_t rank = 0;
    std::uint64_t node_id = 0;
    std::string segment;

    bool operator<(const WaiterKey& other) const {
        if (rank != other.rank) return rank < other.rank;
        if (node_id != other.node_id) return node_id < other.node_id;
        return segment < other.segment;
    }
    bool operator==(const WaiterKey& other) const {
        return rank == other.rank && node_id == other.node_id
            && segment == other.segment;
    }
};

using CursorRunKey = std::pair<std::string, std::uint64_t>;     // (segment, run)
using TargetDomainKey = std::pair<std::string, std::uint64_t>;  // (domain, rank)

// _target_owner_tag: the six owner identity fields, frozen for equality.
struct OwnerTag {
    std::string segment_id;
    std::uint64_t pipeline_generation = 0;
    std::uint64_t stage_epoch = 0;
    std::uint64_t staging_allocation_id = 0;
    std::uint64_t group_ordinal = 0;
    std::uint64_t tile_ordinal = 0;

    bool operator==(const OwnerTag& other) const {
        return segment_id == other.segment_id
            && pipeline_generation == other.pipeline_generation
            && stage_epoch == other.stage_epoch
            && staging_allocation_id == other.staging_allocation_id
            && group_ordinal == other.group_ordinal
            && tile_ordinal == other.tile_ordinal;
    }
    bool operator<(const OwnerTag& other) const {
        if (segment_id != other.segment_id) return segment_id < other.segment_id;
        if (pipeline_generation != other.pipeline_generation) {
            return pipeline_generation < other.pipeline_generation;
        }
        if (stage_epoch != other.stage_epoch) {
            return stage_epoch < other.stage_epoch;
        }
        if (staging_allocation_id != other.staging_allocation_id) {
            return staging_allocation_id < other.staging_allocation_id;
        }
        if (group_ordinal != other.group_ordinal) {
            return group_ordinal < other.group_ordinal;
        }
        return tile_ordinal < other.tile_ordinal;
    }
};

using Lineage = std::array<std::uint64_t, 3>;  // (generation, pred_seq, consumed)

OwnerTag owner_tag_of(const Json& owner, const std::string& path) {
    OwnerTag tag;
    tag.segment_id = row_string(owner, "segment_id", path);
    tag.pipeline_generation = row_uint(owner, "pipeline_generation", path);
    tag.stage_epoch = row_uint(owner, "stage_epoch", path);
    tag.staging_allocation_id = row_uint(owner, "staging_allocation_id", path);
    tag.group_ordinal = row_uint(owner, "group_ordinal", path);
    tag.tile_ordinal = row_uint(owner, "tile_ordinal", path);
    return tag;
}

// ------------------------------------------------- completed-ID interval treap

struct CompletedIntervalNode;
using CompletedPtr = std::shared_ptr<const CompletedIntervalNode>;

struct CompletedIntervalNode {
    std::uint64_t first = 0;
    std::uint64_t last = 0;
    std::uint64_t priority = 0;
    CompletedPtr left;
    CompletedPtr right;
};

// SplitMix64 disperses sequential store IDs (pipeline_ledger.py 638-645).
std::uint64_t completed_interval_priority(std::uint64_t value) {
    std::uint64_t mixed = value + 0x9E3779B97F4A7C15ULL;
    mixed = (mixed ^ (mixed >> 30)) * 0xBF58476D1CE4E5B9ULL;
    mixed = (mixed ^ (mixed >> 27)) * 0x94D049BB133111EBULL;
    return mixed ^ (mixed >> 31);
}

bool completed_heap_less(const CompletedIntervalNode& left,
                         const CompletedIntervalNode& right) {
    if (left.priority != right.priority) return left.priority < right.priority;
    return left.first < right.first;
}

CompletedPtr completed_node(std::uint64_t first, std::uint64_t last,
                            std::uint64_t priority, CompletedPtr left,
                            CompletedPtr right) {
    return std::make_shared<const CompletedIntervalNode>(
        CompletedIntervalNode{first, last, priority, std::move(left),
                              std::move(right)});
}

CompletedPtr completed_rotate_right(const CompletedPtr& root) {
    const CompletedPtr& left = root->left;
    CompletedPtr new_right = completed_node(root->first, root->last,
                                            root->priority, left->right,
                                            root->right);
    return completed_node(left->first, left->last, left->priority, left->left,
                          std::move(new_right));
}

CompletedPtr completed_rotate_left(const CompletedPtr& root) {
    const CompletedPtr& right = root->right;
    CompletedPtr new_left = completed_node(root->first, root->last,
                                           root->priority, root->left,
                                           right->left);
    return completed_node(right->first, right->last, right->priority,
                          std::move(new_left), right->right);
}

CompletedPtr completed_insert(const CompletedPtr& root, std::uint64_t first,
                              std::uint64_t last) {
    if (!root) {
        return completed_node(first, last, completed_interval_priority(first),
                              nullptr, nullptr);
    }
    if (first == root->first) {
        if (last != root->last) {
            fail("completed terminal interval key was reused inconsistently");
        }
        return root;
    }
    if (first < root->first) {
        CompletedPtr left = completed_insert(root->left, first, last);
        CompletedPtr result = completed_node(root->first, root->last,
                                             root->priority, std::move(left),
                                             root->right);
        return completed_heap_less(*result->left, *result)
            ? completed_rotate_right(result) : result;
    }
    CompletedPtr right = completed_insert(root->right, first, last);
    CompletedPtr result = completed_node(root->first, root->last,
                                         root->priority, root->left,
                                         std::move(right));
    return completed_heap_less(*result->right, *result)
        ? completed_rotate_left(result) : result;
}

CompletedPtr completed_merge(const CompletedPtr& left,
                             const CompletedPtr& right) {
    if (!left) return right;
    if (!right) return left;
    if (completed_heap_less(*left, *right)) {
        CompletedPtr merged = completed_merge(left->right, right);
        return completed_node(left->first, left->last, left->priority,
                              left->left, std::move(merged));
    }
    CompletedPtr merged = completed_merge(left, right->left);
    return completed_node(right->first, right->last, right->priority,
                          std::move(merged), right->right);
}

CompletedPtr completed_delete(const CompletedPtr& root, std::uint64_t first) {
    if (!root) return nullptr;
    if (first == root->first) return completed_merge(root->left, root->right);
    if (first < root->first) {
        CompletedPtr left = completed_delete(root->left, first);
        return completed_node(root->first, root->last, root->priority,
                              std::move(left), root->right);
    }
    CompletedPtr right = completed_delete(root->right, first);
    return completed_node(root->first, root->last, root->priority, root->left,
                          std::move(right));
}

// (predecessor, successor-or-equal); pointers into the persistent treap stay
// valid for the whole staging pass (roots are never mutated in place).
struct CompletedNeighbors {
    const CompletedIntervalNode* predecessor = nullptr;
    const CompletedIntervalNode* successor = nullptr;
};

CompletedNeighbors completed_neighbors(const CompletedPtr& root,
                                       std::uint64_t value) {
    CompletedNeighbors out;
    const CompletedIntervalNode* current = root.get();
    while (current != nullptr) {
        if (current->first < value) {
            out.predecessor = current;
            current = current->right.get();
        } else if (current->first > value) {
            out.successor = current;
            current = current->left.get();
        } else {
            out.successor = current;
            return out;
        }
    }
    return out;
}

bool completed_contains(const CompletedPtr& root, std::uint64_t value) {
    const CompletedIntervalNode* current = root.get();
    while (current != nullptr) {
        if (value < current->first) current = current->left.get();
        else if (value > current->last) current = current->right.get();
        else return true;
    }
    return false;
}

// _completed_interval_add (pipeline_ledger.py 766-777).
CompletedPtr completed_add(const CompletedPtr& root, std::uint64_t value) {
    if (completed_contains(root, value)) return root;
    const CompletedNeighbors neighbors = completed_neighbors(root, value);
    std::uint64_t first = value;
    std::uint64_t last = value;
    CompletedPtr next = root;
    if (neighbors.predecessor != nullptr
            && neighbors.predecessor->last + 1 == value) {
        first = neighbors.predecessor->first;
        next = completed_delete(next, neighbors.predecessor->first);
    }
    if (neighbors.successor != nullptr
            && neighbors.successor->first == value + 1) {
        last = neighbors.successor->last;
        next = completed_delete(next, neighbors.successor->first);
    }
    return completed_insert(next, first, last);
}

// _completed_interval_build_from_half_open (pipeline_ledger.py 780-857).
std::pair<CompletedPtr, std::uint64_t> completed_build_from_half_open(
    const std::vector<std::pair<std::uint64_t, std::uint64_t>>& ranges) {
    std::vector<std::pair<std::uint64_t, std::uint64_t>> canonical;
    std::uint64_t completed_count = 0;
    bool have_previous_end = false;
    std::uint64_t previous_end = 0;
    for (const auto& range : ranges) {
        const std::uint64_t first = range.first;
        const std::uint64_t end = range.second;
        if (end <= first || (have_previous_end && first < previous_end)) {
            fail("completed endpoint ranges are empty, overlapping, or unsorted");
        }
        completed_count += end - first;
        if (!canonical.empty() && first == canonical.back().second) {
            canonical.back().second = end;
        } else {
            canonical.emplace_back(first, end);
        }
        previous_end = end;
        have_previous_end = true;
    }

    const std::size_t size = canonical.size();
    if (size == 0) return {nullptr, completed_count};

    std::vector<std::uint64_t> priorities(size);
    for (std::size_t i = 0; i < size; ++i) {
        priorities[i] = completed_interval_priority(canonical[i].first);
    }
    std::vector<std::size_t> left(size, kNoIndex);
    std::vector<std::size_t> right(size, kNoIndex);
    std::vector<std::size_t> stack;
    for (std::size_t index = 0; index < size; ++index) {
        std::size_t last_popped = kNoIndex;
        while (!stack.empty()) {
            const std::size_t top = stack.back();
            const bool key_less =
                priorities[index] != priorities[top]
                    ? priorities[index] < priorities[top]
                    : canonical[index].first < canonical[top].first;
            if (!key_less) break;
            last_popped = stack.back();
            stack.pop_back();
        }
        if (!stack.empty()) right[stack.back()] = index;
        if (last_popped != kNoIndex) left[index] = last_popped;
        stack.push_back(index);
    }

    const std::size_t root_index = stack.front();
    std::vector<std::size_t> postorder;
    postorder.reserve(size);
    std::vector<std::pair<std::size_t, bool>> pending;
    pending.emplace_back(root_index, false);
    while (!pending.empty()) {
        const auto entry = pending.back();
        pending.pop_back();
        if (entry.second) {
            postorder.push_back(entry.first);
            continue;
        }
        pending.emplace_back(entry.first, true);
        if (right[entry.first] != kNoIndex) {
            pending.emplace_back(right[entry.first], false);
        }
        if (left[entry.first] != kNoIndex) {
            pending.emplace_back(left[entry.first], false);
        }
    }

    std::vector<CompletedPtr> nodes(size);
    for (std::size_t index : postorder) {
        nodes[index] = completed_node(
            canonical[index].first, canonical[index].second - 1,
            priorities[index],
            left[index] != kNoIndex ? nodes[left[index]] : nullptr,
            right[index] != kNoIndex ? nodes[right[index]] : nullptr);
    }
    return {nodes[root_index], completed_count};
}

// _completed_interval_add_range (pipeline_ledger.py 860-914).
std::pair<CompletedPtr, std::uint64_t> completed_add_range(
    const CompletedPtr& root, std::uint64_t completed_prefix,
    std::uint64_t total_count, std::uint64_t first, std::uint64_t end) {
    if (first < completed_prefix || end <= first || end > total_count) {
        fail("completed range is duplicate or outside its cursor extent");
    }
    const CompletedNeighbors neighbors = completed_neighbors(root, first);
    if (neighbors.predecessor != nullptr
            && neighbors.predecessor->last >= first) {
        fail("completed range overlaps its predecessor");
    }
    if (neighbors.successor != nullptr && neighbors.successor->first < end) {
        fail("completed range overlaps its successor");
    }

    std::uint64_t merged_first = first;
    std::uint64_t merged_end = end;
    CompletedPtr next = root;
    if (neighbors.predecessor != nullptr
            && neighbors.predecessor->last + 1 == first) {
        merged_first = neighbors.predecessor->first;
        next = completed_delete(next, neighbors.predecessor->first);
    }
    while (true) {
        const CompletedNeighbors at_end = completed_neighbors(next, merged_end);
        if (at_end.successor == nullptr
                || at_end.successor->first != merged_end) {
            break;
        }
        merged_end = at_end.successor->last + 1;
        next = completed_delete(next, at_end.successor->first);
    }

    if (merged_first == completed_prefix) {
        completed_prefix = merged_end;
    } else {
        next = completed_insert(next, merged_first, merged_end - 1);
    }
    // The legacy sorted-list merge folded every completed range beginning at
    // the prefix after each completion, including a canonicalizable interval
    // already present in the seeded cursor row.
    while (true) {
        const CompletedNeighbors at_prefix =
            completed_neighbors(next, completed_prefix);
        if (at_prefix.successor == nullptr
                || at_prefix.successor->first != completed_prefix) {
            break;
        }
        completed_prefix = at_prefix.successor->last + 1;
        next = completed_delete(next, at_prefix.successor->first);
    }
    return {next, completed_prefix};
}

// _completed_interval_export_half_open (pipeline_ledger.py 917-932).
std::vector<std::pair<std::uint64_t, std::uint64_t>> completed_export_half_open(
    const CompletedPtr& root) {
    std::vector<std::pair<std::uint64_t, std::uint64_t>> result;
    std::vector<const CompletedIntervalNode*> stack;
    const CompletedIntervalNode* current = root.get();
    while (current != nullptr || !stack.empty()) {
        while (current != nullptr) {
            stack.push_back(current);
            current = current->left.get();
        }
        current = stack.back();
        stack.pop_back();
        result.emplace_back(current->first, current->last + 1);
        current = current->right.get();
    }
    return result;
}

// ------------------------------------------------- dependency interval treap

using DependencyRow = std::array<std::uint64_t, 9>;

struct DependencyRowHash {
    std::size_t operator()(const DependencyRow& row) const {
        std::string text;
        text.reserve(9 * 20);
        for (std::uint64_t item : row) {
            text += std::to_string(item);
            text += '\x1f';
        }
        return std::hash<std::string>{}(text);
    }
};

struct DependencyIntervalNode;
using DependencyPtr = std::shared_ptr<const DependencyIntervalNode>;

struct DependencyIntervalNode {
    DependencyRow row{};
    std::uint64_t priority = 0;
    std::uint64_t max_last = 0;
    DependencyPtr left;
    DependencyPtr right;
};

std::uint64_t dependency_row_max_last(const DependencyRow& row,
                                      const DependencyPtr& left,
                                      const DependencyPtr& right) {
    std::uint64_t value = row[1];
    if (left && left->max_last > value) value = left->max_last;
    if (right && right->max_last > value) value = right->max_last;
    return value;
}

DependencyPtr dependency_node(const DependencyRow& row,
                              std::uint64_t priority, DependencyPtr left,
                              DependencyPtr right) {
    return std::make_shared<const DependencyIntervalNode>(
        DependencyIntervalNode{row, priority,
                               dependency_row_max_last(row, left, right),
                               std::move(left), std::move(right)});
}

// Priority: std::hash over the row's canonical decimal serialization (the
// documented stand-in for Python's process-salted hash of the tuple).
std::uint64_t dependency_interval_priority(const DependencyRow& row) {
    return static_cast<std::uint64_t>(DependencyRowHash{}(row));
}

DependencyPtr dependency_rotate_right(const DependencyPtr& root) {
    const DependencyPtr& left = root->left;
    DependencyPtr new_right =
        dependency_node(root->row, root->priority, left->right, root->right);
    return dependency_node(left->row, left->priority, left->left,
                           std::move(new_right));
}

DependencyPtr dependency_rotate_left(const DependencyPtr& root) {
    const DependencyPtr& right = root->right;
    DependencyPtr new_left =
        dependency_node(root->row, root->priority, root->left, right->left);
    return dependency_node(right->row, right->priority, std::move(new_left),
                           right->right);
}

DependencyPtr dependency_interval_insert(const DependencyPtr& root,
                                         const DependencyRow& row) {
    if (!root) {
        return dependency_node(row, dependency_interval_priority(row),
                               nullptr, nullptr);
    }
    if (row == root->row) return root;
    if (row < root->row) {
        DependencyPtr left = dependency_interval_insert(root->left, row);
        DependencyPtr result = dependency_node(root->row, root->priority,
                                               std::move(left), root->right);
        return result->left->priority < result->priority
            ? dependency_rotate_right(result) : result;
    }
    DependencyPtr right = dependency_interval_insert(root->right, row);
    DependencyPtr result = dependency_node(root->row, root->priority,
                                           root->left, std::move(right));
    return result->right->priority < result->priority
        ? dependency_rotate_left(result) : result;
}

// _dependency_interval_matches (pipeline_ledger.py 1055-1066).
void dependency_interval_matches(const DependencyPtr& root,
                                 std::uint64_t node_id,
                                 std::set<DependencyRow>& matches) {
    if (!root || root->max_last < node_id) return;
    if (root->left && root->left->max_last >= node_id) {
        dependency_interval_matches(root->left, node_id, matches);
    }
    const DependencyRow& row = root->row;
    if (row[0] <= node_id && node_id <= row[1] && node_id % row[4] == row[5]) {
        matches.insert(row);
    }
    if (row[0] <= node_id && root->right && root->right->max_last >= node_id) {
        dependency_interval_matches(root->right, node_id, matches);
    }
}

// _dependency_node_parts (pipeline_ledger.py 1069-1083): split at the last
// 'n'; the numeric suffix must be canonical decimal.
std::optional<std::pair<std::string, std::uint64_t>> dependency_node_parts(
    const std::string& dependency) {
    const std::size_t cut = dependency.rfind('n');
    if (cut == std::string::npos || cut + 1 == dependency.size()) {
        return std::nullopt;
    }
    const std::string node_text = dependency.substr(cut + 1);
    if (node_text.empty()) return std::nullopt;
    std::uint64_t node_id = 0;
    for (char digit : node_text) {
        if (digit < '0' || digit > '9') return std::nullopt;
        const unsigned value = static_cast<unsigned>(digit - '0');
        if (node_id > (U64_MAX - value) / 10) return std::nullopt;
        node_id = node_id * 10 + value;
    }
    if (node_id < 1) return std::nullopt;
    return std::make_pair(dependency.substr(0, cut + 1), node_id);
}

// ---------------------------------------------- selector index lookups (N4) --

// _descriptor_run_at (pipeline_ledger.py 1134-1143).
std::optional<std::uint64_t> descriptor_run_at(const SelectorIndex& index,
                                               std::uint64_t group,
                                               std::uint64_t rank,
                                               std::uint64_t tile) {
    const auto found = index.run_intervals.find(std::make_tuple(group, rank));
    if (found == index.run_intervals.end()) return std::nullopt;
    const std::vector<std::uint64_t>& starts = found->second.first;
    const auto position_it = std::upper_bound(starts.begin(), starts.end(), tile);
    if (position_it == starts.begin()) return std::nullopt;
    const std::size_t position =
        static_cast<std::size_t>(position_it - starts.begin()) - 1;
    const auto& row = found->second.second[position];
    return (row[0] <= tile && tile < row[1])
        ? std::optional<std::uint64_t>(row[2]) : std::nullopt;
}

// _descriptor_selector_at (pipeline_ledger.py 1146-1155).
std::optional<std::pair<std::uint64_t, std::uint64_t>> descriptor_selector_at(
    const SelectorIndex& index, std::uint64_t group, std::uint64_t rank,
    std::uint64_t tile, std::uint64_t shard, const std::string& endpoint) {
    const auto found = index.selector_intervals.find(
        std::make_tuple(group, rank, shard, endpoint));
    if (found == index.selector_intervals.end()) return std::nullopt;
    const std::vector<std::uint64_t>& starts = found->second.first;
    const auto position_it = std::upper_bound(starts.begin(), starts.end(), tile);
    if (position_it == starts.begin()) return std::nullopt;
    const std::size_t position =
        static_cast<std::size_t>(position_it - starts.begin()) - 1;
    const auto& row = found->second.second[position];
    if (row[0] <= tile && tile < row[1]) return std::make_pair(row[2], row[3]);
    return std::nullopt;
}

// _consume_completion_at (pipeline_ledger.py 1158-1169).
std::optional<std::pair<std::uint64_t, std::uint64_t>> consume_completion_at(
    const SelectorIndex& index, std::uint64_t rank, std::uint64_t group,
    std::uint64_t tile, std::uint64_t shard) {
    const auto found = index.consume_intervals_by_target.find(
        std::make_tuple(rank, group, shard));
    if (found == index.consume_intervals_by_target.end()) return std::nullopt;
    const std::vector<std::uint64_t>& starts = found->second.first;
    const auto position_it = std::upper_bound(starts.begin(), starts.end(), tile);
    if (position_it == starts.begin()) return std::nullopt;
    const std::size_t position =
        static_cast<std::size_t>(position_it - starts.begin()) - 1;
    const auto& row = found->second.second[position];
    if (row[0] <= tile && tile < row[1]) return std::make_pair(row[2], row[3]);
    return std::nullopt;
}

// --------------------------------------------------------------- overlays --

// _RowOverlay (pipeline_ledger.py 520-549); changes() tombstones are nullopt.
template <class K, class V>
struct RowOverlay {
    const std::map<K, V>* base = nullptr;
    std::map<K, V> writes;
    std::set<K> deleted;

    const V* get_raw(const K& key) const {
        const auto write = writes.find(key);
        if (write != writes.end()) return &write->second;
        if (deleted.count(key)) return nullptr;
        if (!base) return nullptr;
        const auto entry = base->find(key);
        return entry == base->end() ? nullptr : &entry->second;
    }
    bool has(const K& key) const {
        return writes.count(key)
            || (!deleted.count(key) && base && base->count(key));
    }
    void set(const K& key, V value) {
        deleted.erase(key);
        writes[key] = std::move(value);
    }
    void del(const K& key) {
        writes.erase(key);
        deleted.insert(key);
    }
    std::map<K, std::optional<V>> changes() const {
        std::map<K, std::optional<V>> out;
        for (const K& key : deleted) out[key] = std::nullopt;
        for (const auto& entry : writes) out[entry.first] = entry.second;
        return out;
    }
};

// _CountOverlay over an ordered-map base.
template <class K>
struct CountOverlay {
    const std::map<K, std::int64_t>* base = nullptr;
    std::map<K, std::int64_t> deltas;

    std::int64_t get(const K& key) const {
        std::int64_t value = 0;
        if (base) {
            const auto entry = base->find(key);
            if (entry != base->end()) value = entry->second;
        }
        const auto delta = deltas.find(key);
        if (delta != deltas.end()) value += delta->second;
        return value;
    }
    void add(const K& key, std::int64_t amount) {
        if (amount == 0) return;
        deltas[key] += amount;
    }
};

// _CountOverlay over a hashed base (string/uint keys).
template <class K>
struct HashCountOverlay {
    const std::unordered_map<K, std::int64_t>* base = nullptr;
    std::unordered_map<K, std::int64_t> deltas;

    std::int64_t get(const K& key) const {
        std::int64_t value = 0;
        if (base) {
            const auto entry = base->find(key);
            if (entry != base->end()) value = entry->second;
        }
        const auto delta = deltas.find(key);
        if (delta != deltas.end()) value += delta->second;
        return value;
    }
    void add(const K& key, std::int64_t amount) {
        if (amount == 0) return;
        deltas[key] += amount;
    }
};

std::uint64_t run_endpoint_total(const PipelineEndpointRun& run) {
    return run.tile_count * run.shard_ordinals.size() * run.legs.size();
}

std::uint64_t cursor_position_ordinal(const Json& row,
                                      const PipelineEndpointRun& run) {
    const std::uint64_t shards = run.shard_ordinals.size();
    const std::uint64_t legs = run.legs.size();
    return row_uint(row, "tile_offset", "cursor run") * shards * legs
        + row_uint(row, "shard_index", "cursor run") * legs
        + row_uint(row, "leg_index", "cursor run");
}

std::size_t index_of(const std::vector<std::uint64_t>& values,
                     std::uint64_t target) {
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (values[i] == target) return i;
    }
    return values.size();
}

// _global_tile_ordinal (pipeline_ledger.py 574-585), index-backed variant.
std::uint64_t global_tile_ordinal(const PipelineDescriptor& descriptor,
                                  std::uint64_t group_ordinal,
                                  std::uint64_t tile_ordinal,
                                  const SelectorIndex* index) {
    const PipelineGroup* group = nullptr;
    if (index != nullptr) {
        const auto found = index->groups_by_ordinal.find(group_ordinal);
        if (found != index->groups_by_ordinal.end()
                && found->second < descriptor.groups.size()) {
            group = &descriptor.groups[found->second];
        }
    }
    if (group == nullptr) group = descriptor.group_by_ordinal(group_ordinal);
    if (group == nullptr || tile_ordinal < 1
            || tile_ordinal > group->tile_axis.tile_count) {
        fail("slot owner points outside descriptor group/tile range");
    }
    return group->tile_axis.global_tile_first + tile_ordinal - 1;
}

// Slot key from a parsed patch entry (the [domain, role, rank, shard|null,
// slot] array form of source patches, or the target_delta key object).
SlotKey parse_slot_key(const Json& key, const std::string& path) {
    SlotKey out;
    if (key.is_array()) {
        if (key.size() != 5) fail(path + " slot key must have five fields");
        out.domain = read_string(key[0], path + ".key[0]");
        out.role = read_string(key[1], path + ".key[1]");
        out.rank = read_rank(key[2], path + ".key[2]");
        out.shard = key[3].is_null()
            ? -1
            : static_cast<std::int64_t>(read_uint(key[3], path + ".key[3]"));
        out.slot_index = read_uint(key[4], path + ".key[4]", 1);
        return out;
    }
    if (key.is_object()) {
        out.domain = row_string(key, "buffer_domain_id", path + ".key");
        out.role = row_string(key, "role", path + ".key");
        out.rank = read_rank(row_field(key, "rank", path + ".key"),
                             path + ".key.rank");
        const Json& shard = row_field(key, "shard_ordinal", path + ".key");
        out.shard = shard.is_null()
            ? -1
            : static_cast<std::int64_t>(
                  read_uint(shard, path + ".key.shard_ordinal"));
        out.slot_index = read_uint(
            row_field(key, "slot_index", path + ".key"),
            path + ".key.slot_index", 1);
        return out;
    }
    fail(path + " slot key must be an array or object");
}

}  // namespace

// ------------------------------------------------------------ staged update --

struct PreparedUpdate {
    ParsedEnvelope parsed;
    std::string segment_bytes;
    bool replayed = false;

    std::unordered_map<std::string, DescriptorPtr> descriptor_additions;
    std::unordered_map<std::string, Json> descriptor_state_updates;
    std::vector<std::string> descriptor_retirements;
    std::map<CursorRunKey, std::optional<Json>> cursor_changes;
    std::unordered_map<std::string, std::uint64_t> cursor_version_updates;
    std::map<CursorRunKey, std::optional<std::map<std::uint64_t, Json>>>
        cursor_transfer_changes;
    std::unordered_map<std::string, std::int64_t> active_endpoint_deltas;
    std::vector<Json> resources;
    std::map<SlotKey, std::optional<Json>> slot_changes;
    std::map<SlotKey, Lineage> target_slot_lineage_changes;
    std::map<ReadyKey, std::optional<OwnerTag>> target_ready_changes;
    std::unordered_map<std::uint64_t, CompletedPtr> terminal_provenance_updates;
    std::map<CreditKey, std::optional<Json>> credit_changes;
    std::map<std::string, std::optional<Json>> obligation_changes;
    std::unordered_map<std::string, std::int64_t> slot_owner_deltas;
    std::map<CreditKey, std::int64_t> source_occupancy_deltas;
    std::map<TargetDomainKey, std::int64_t> target_occupancy_deltas;
    std::unordered_map<std::string, std::int64_t> credit_ack_domain_deltas;
    std::unordered_map<std::string, std::int64_t> ack_pin_deltas;
    std::unordered_map<std::string, std::int64_t> active_descriptor_deltas;
    std::unordered_map<std::string, std::int64_t> slot_row_deltas;
    std::unordered_map<std::string, std::int64_t> credit_row_deltas;
    std::unordered_map<std::string, std::int64_t> obligation_remaining_deltas;
    std::unordered_map<std::string, std::int64_t> dependency_pin_deltas;
    std::map<WaiterKey, std::int64_t> dependency_waiter_deltas;
    std::unordered_map<std::string, std::int64_t> cursor_unissued_deltas;
    std::unordered_map<std::uint64_t, std::int64_t> active_transfer_deltas;
    std::map<ReleaseKey, std::int64_t> release_watermark_deltas;
    std::unordered_map<DependencyRow, std::string, DependencyRowHash>
        dependency_row_segment_additions;
    std::unordered_map<std::string, DependencyPtr> dependency_index_additions;
    std::unordered_map<std::uint64_t, DependencyPtr>
        dependency_rank_index_additions;
    bool changed = false;
};

// --------------------------------------------------------------- ledger state

// The whole accepted state, kept in a plain file-local base so the staging
// core can reference it without naming the private nested Impl.
struct PipelineStageState {
    std::unordered_map<std::string, DescriptorPtr> descriptors;
    std::unordered_map<std::string, SelectorIndex> selector_indexes;
    std::unordered_map<std::string, DependencyPtr> dependency_endpoint_index;
    std::unordered_map<std::uint64_t, DependencyPtr>
        dependency_endpoint_index_by_rank;
    // Segment ownership of dependency rows (Python keeps it inside row[2]).
    std::unordered_map<DependencyRow, std::string, DependencyRowHash>
        dependency_row_segments;
    std::unordered_map<std::uint64_t, CompletedPtr>
        completed_terminal_intervals_by_rank;
    std::unordered_map<std::string, std::int64_t>
        unfinished_dependency_pins_by_segment;
    std::map<WaiterKey, std::int64_t>
        unfinished_dependency_waiters_by_endpoint;
    std::unordered_map<std::string, Json> descriptor_state;
    std::set<std::string> active_segments;
    std::unordered_map<std::string, std::map<std::uint64_t, Json>>
        cursor_runs_by_segment;
    std::unordered_map<std::string, std::uint64_t> cursor_versions_by_segment;
    std::map<CursorRunKey, std::map<std::uint64_t, Json>>
        active_transfers_by_run;
    std::unordered_map<std::uint64_t, std::int64_t> active_transfer_id_counts;
    std::set<std::uint64_t> active_transfer_ids;
    std::unordered_map<std::string, std::int64_t> active_endpoint_counts;
    std::unordered_map<std::string, std::int64_t> cursor_unissued_by_segment;
    std::map<ReleaseKey, std::uint64_t> source_release_watermarks;
    std::unordered_map<std::string, std::set<ReleaseKey>>
        release_watermark_keys_by_domain;
    std::vector<Json> active_resources;
    std::map<SlotKey, Json> slots_by_key;
    std::map<SlotKey, Lineage> target_slot_lineage;
    std::map<ReadyKey, OwnerTag> target_ready_shards_by_slot;
    std::map<CreditKey, Json> credits_by_key;
    std::map<std::string, Json> obligations_by_id;
    std::unordered_map<std::string, std::int64_t> slot_owner_counts;
    std::map<CreditKey, std::int64_t> source_occupancy;
    std::map<TargetDomainKey, std::int64_t> target_occupancy;
    std::unordered_map<std::string, std::int64_t> credit_ack_counts;
    std::unordered_map<std::string, std::int64_t> ack_pins_by_segment;
    std::unordered_map<std::string, std::int64_t>
        active_descriptor_count_by_domain;
    std::unordered_map<std::string, std::int64_t> slot_row_counts_by_domain;
    std::unordered_map<std::string, std::int64_t> credit_row_counts_by_domain;
    std::unordered_map<std::string, std::set<std::string>>
        obligations_by_segment;
    std::unordered_map<std::string, std::int64_t>
        obligation_remaining_by_segment;

    std::vector<std::unique_ptr<PreparedUpdate>> prepared;

    PreparedUpdate* find_prepared(int slot) {
        if (slot < 0 || static_cast<std::size_t>(slot) >= prepared.size()) {
            return nullptr;
        }
        return prepared[static_cast<std::size_t>(slot)].get();
    }
    void release_prepared(int slot) {
        if (slot >= 0 && static_cast<std::size_t>(slot) < prepared.size()) {
            prepared[static_cast<std::size_t>(slot)].reset();
        }
    }
    int store_prepared(std::unique_ptr<PreparedUpdate> update) {
        for (std::size_t i = 0; i < prepared.size(); ++i) {
            if (!prepared[i]) {
                prepared[i] = std::move(update);
                return static_cast<int>(i);
            }
        }
        prepared.push_back(std::move(update));
        return static_cast<int>(prepared.size() - 1);
    }
};

struct PipelineLedgerCore::Impl : PipelineStageState {};

// Side table binding the stage core to the WorkLedgerCore that owns the
// coordinated transactions (the frozen prepare() signature carries only the
// Transaction handle; the orchestrator registers the pair once).  Exposed as
// jointledger::bind_work_ledger for the orchestrator and the unit tests.
namespace {
std::unordered_map<const PipelineLedgerCore*, WorkLedgerCore*>&
work_ledger_bindings() {
    static std::unordered_map<const PipelineLedgerCore*, WorkLedgerCore*> table;
    return table;
}
}  // namespace

void bind_work_ledger(PipelineLedgerCore& pipeline, WorkLedgerCore& work) {
    work_ledger_bindings()[&pipeline] = &work;
}

namespace {

// ------------------------------------------------------------ staging core --

struct TargetWorkset {
    std::optional<Json> row;      // nullopt = Python None
    std::set<std::uint64_t> consumed;
    std::int64_t remaining = 0;
    std::int64_t non_target_count = 0;
};

struct CursorProgress {
    Json row;                     // accepted (or initial) run row
    const PipelineEndpointRun* run = nullptr;
    DescriptorPtr descriptor;     // keeps *run alive
    std::uint64_t total = 0;
    std::uint64_t completed = 0;
    std::uint64_t completed_total = 0;
    std::uint64_t cancelled = 0;
    std::uint64_t remaining = 0;
    CompletedPtr ooo_root;
    std::vector<std::pair<std::uint64_t, std::uint64_t>> ooo_wire_ranges;
    bool ooo_changed = false;
    std::map<std::uint64_t, std::pair<std::uint64_t, std::optional<Json>>>
        active;                   // store_id -> (ordinal, endpoint row|None)
    std::uint64_t position = 0;
    bool closed = false;
};

struct SelectedEndpoint {
    const PipelineEndpointRun* run = nullptr;
    std::uint64_t global_tile_ordinal = 0;
    PipelineShardTile shard;
    std::string endpoint;
};

// _stage_delivery port: one instance per prepare(); the member functions
// mirror the Python closures one-to-one.
class StageDelivery {
public:
    StageDelivery(PipelineStageState& state, PreparedUpdate& update,
                  WorkLedgerCore* work, WorkLedgerCore::Transaction* tx)
        : st_(state), parsed_(update.parsed), update_(update),
          work_(work), tx_(tx) {
        slot_overlay_.base = &st_.slots_by_key;
        target_ready_overlay_.base = &st_.target_ready_shards_by_slot;
        credit_overlay_.base = &st_.credits_by_key;
        obligation_overlay_.base = &st_.obligations_by_id;
        slot_owner_counts_.base = &st_.slot_owner_counts;
        source_occupancy_.base = &st_.source_occupancy;
        target_occupancy_.base = &st_.target_occupancy;
        credit_ack_domains_.base = &st_.credit_ack_counts;
        ack_pins_by_segment_.base = &st_.ack_pins_by_segment;
        active_domain_counts_.base = &st_.active_descriptor_count_by_domain;
        slot_row_counts_.base = &st_.slot_row_counts_by_domain;
        credit_row_counts_.base = &st_.credit_row_counts_by_domain;
        active_transfer_id_counts_.base = &st_.active_transfer_id_counts;
        obligation_remaining_.base = &st_.obligation_remaining_by_segment;
        dependency_pins_.base = &st_.unfinished_dependency_pins_by_segment;
        dependency_waiters_.base = &st_.unfinished_dependency_waiters_by_endpoint;
        cursor_unissued_.base = &st_.cursor_unissued_by_segment;
        release_base_ = &st_.source_release_watermarks;
    }

    void run();

private:
    PipelineStageState& st_;
    const ParsedEnvelope& parsed_;
    PreparedUpdate& update_;
    WorkLedgerCore* work_;
    WorkLedgerCore::Transaction* tx_;

    std::unordered_map<std::string, DescriptorPtr> additions_;
    std::unordered_map<std::string, Json> initial_state_updates_;
    std::unordered_map<std::string, DependencyPtr> dependency_root_changes_;
    std::unordered_map<std::uint64_t, DependencyPtr>
        dependency_rank_root_changes_;

    RowOverlay<SlotKey, Json> slot_overlay_;
    std::map<SlotKey, TargetWorkset> target_worksets_;
    std::map<SlotKey, Lineage> target_slot_lineage_changes_;
    RowOverlay<ReadyKey, OwnerTag> target_ready_overlay_;
    RowOverlay<CreditKey, Json> credit_overlay_;
    RowOverlay<std::string, Json> obligation_overlay_;
    HashCountOverlay<std::string> slot_owner_counts_;
    CountOverlay<CreditKey> source_occupancy_;
    CountOverlay<TargetDomainKey> target_occupancy_;
    HashCountOverlay<std::string> credit_ack_domains_;
    HashCountOverlay<std::string> ack_pins_by_segment_;
    HashCountOverlay<std::string> active_domain_counts_;
    HashCountOverlay<std::string> slot_row_counts_;
    HashCountOverlay<std::string> credit_row_counts_;
    HashCountOverlay<std::uint64_t> active_transfer_id_counts_;
    HashCountOverlay<std::string> obligation_remaining_;
    HashCountOverlay<std::string> dependency_pins_;
    CountOverlay<WaiterKey> dependency_waiters_;
    HashCountOverlay<std::string> cursor_unissued_;
    const std::map<ReleaseKey, std::uint64_t>* release_base_ = nullptr;
    std::map<ReleaseKey, std::int64_t> release_watermark_deltas_;

    std::uint64_t release_watermark_get(const ReleaseKey& key) const {
        std::int64_t value = 0;
        const auto base = release_base_->find(key);
        if (base != release_base_->end()) {
            value = static_cast<std::int64_t>(base->second);
        }
        const auto delta = release_watermark_deltas_.find(key);
        if (delta != release_watermark_deltas_.end()) value += delta->second;
        return static_cast<std::uint64_t>(value);
    }
    void release_watermark_add(const ReleaseKey& key, std::int64_t amount) {
        if (amount != 0) release_watermark_deltas_[key] += amount;
    }

    std::unordered_map<std::string, std::uint64_t> cursor_versions_;
    std::unordered_map<std::string, Json> touched_state_;
    std::vector<std::string> retirements_;
    std::set<std::string> descriptor_commits_;
    std::unordered_map<std::string, std::uint64_t>
        descriptor_transition_counts_;
    std::set<CursorRunKey> cursor_dirty_;
    std::map<CursorRunKey, CursorProgress> cursor_sims_;
    std::map<CursorRunKey, std::optional<Json>> cursor_changes_;
    std::unordered_map<std::string, std::uint64_t> cursor_version_updates_;
    std::map<CursorRunKey, std::optional<std::map<std::uint64_t, Json>>>
        cursor_transfer_changes_;
    std::unordered_map<std::string, std::int64_t> active_endpoint_deltas_;
    std::unordered_map<std::string, std::int64_t> cursor_active_deltas_;
    std::map<std::string, std::optional<Json>> expected_obligation_rows_;
    std::unordered_map<std::string, std::uint64_t> resource_counts_;
    std::set<std::string> domain_retirements_;
    std::unordered_map<std::string, const Json*> parsed_cursor_patches_;
    std::unordered_map<std::string, std::set<std::uint64_t>>
        parsed_cursor_patch_run_indexes_;
    std::set<std::pair<std::uint64_t, std::uint64_t>> node_terminal_frontier_;
    std::unordered_map<std::uint64_t, CompletedPtr> terminal_provenance_roots_;
    std::set<std::pair<ReadyKey, OwnerTag>> pending_ready_after_clear_;
    std::size_t node_index_ = 0;

    // -------------------------------------------------- descriptor views --

    DescriptorPtr find_descriptor(const std::string& segment) const {
        const auto added = additions_.find(segment);
        if (added != additions_.end()) return added->second;
        const auto retained = st_.descriptors.find(segment);
        if (retained != st_.descriptors.end()) return retained->second;
        return nullptr;
    }

    // current_state (pipeline_ledger.py 2653-2663).
    Json& current_state(const std::string& segment) {
        const auto touched = touched_state_.find(segment);
        if (touched != touched_state_.end()) return touched->second;
        const Json* raw = states_get(segment);
        if (raw == nullptr) {
            fail("transition references unknown descriptor '" + segment + "'");
        }
        touched_state_.emplace(segment, *raw);
        return touched_state_.find(segment)->second;
    }

    const Json* states_get(const std::string& segment) const {
        const auto initial = initial_state_updates_.find(segment);
        if (initial != initial_state_updates_.end()) return &initial->second;
        const auto retained = st_.descriptor_state.find(segment);
        return retained == st_.descriptor_state.end() ? nullptr
                                                      : &retained->second;
    }

    static bool state_flag(const Json& state, const char* key) {
        const auto member = state.find(key);
        if (member == state.end() || member->is_null()) return false;
        return member->get<bool>();
    }

    static std::uint64_t state_uint(const Json& state, const char* key) {
        const auto member = state.find(key);
        if (member == state.end()) return 0;
        return read_uint(*member, std::string("descriptor state.") + key);
    }

    // -------------------------------------------------- work node staging --

    static bool active_state(const CommittedWork* work) {
        if (work == nullptr) return false;
        return work->state == WorkState::Committed
            || work->state == WorkState::Ready
            || work->state == WorkState::Issued
            || work->state == WorkState::InService;
    }

    CompletedPtr terminal_root(std::uint64_t rank) const {
        const auto staged = terminal_provenance_roots_.find(rank);
        if (staged != terminal_provenance_roots_.end()) return staged->second;
        const auto persisted =
            st_.completed_terminal_intervals_by_rank.find(rank);
        return persisted == st_.completed_terminal_intervals_by_rank.end()
            ? nullptr : persisted->second;
    }

    bool terminal_proven(std::uint64_t rank, std::uint64_t node_id) const {
        if (node_terminal_frontier_.count({rank, node_id})) return true;
        return completed_contains(terminal_root(rank), node_id);
    }

    DependencyPtr find_dependency_root(const std::string& prefix) const {
        const auto staged = dependency_root_changes_.find(prefix);
        if (staged != dependency_root_changes_.end()) return staged->second;
        const auto persisted = st_.dependency_endpoint_index.find(prefix);
        return persisted == st_.dependency_endpoint_index.end()
            ? nullptr : persisted->second;
    }

    DependencyPtr find_dependency_rank_root(std::uint64_t rank) const {
        const auto staged = dependency_rank_root_changes_.find(rank);
        if (staged != dependency_rank_root_changes_.end()) return staged->second;
        const auto persisted =
            st_.dependency_endpoint_index_by_rank.find(rank);
        return persisted == st_.dependency_endpoint_index_by_rank.end()
            ? nullptr : persisted->second;
    }

    std::string row_segment(const DependencyRow& row) const {
        const auto staged =
            update_.dependency_row_segment_additions.find(row);
        if (staged != update_.dependency_row_segment_additions.end()) {
            return staged->second;
        }
        const auto persisted = st_.dependency_row_segments.find(row);
        if (persisted != st_.dependency_row_segments.end()) {
            return persisted->second;
        }
        return std::string();
    }

    std::set<DependencyRow> dependency_matches(const std::string& dependency) {
        const auto parts = dependency_node_parts(dependency);
        if (!parts) return {};
        std::set<DependencyRow> matches;
        dependency_interval_matches(find_dependency_root(parts->first),
                                    parts->second, matches);
        return matches;
    }

    struct DependencyReference {
        std::string segment;
        std::uint64_t rank = 0;
        std::uint64_t node_id = 0;
    };

    std::optional<DependencyReference> dependency_reference(
        const std::string& dependency) {
        const auto parts = dependency_node_parts(dependency);
        if (!parts) return std::nullopt;
        std::set<DependencyRow> matches = dependency_matches(dependency);
        std::set<std::string> segments;
        for (const DependencyRow& row : matches) {
            segments.insert(row_segment(row));
        }
        if (segments.size() > 1) {
            fail("ordinary dependency maps to multiple pipeline endpoints");
        }
        if (matches.empty()) return std::nullopt;
        const DependencyRow& row = *matches.begin();
        return DependencyReference{row_segment(row), row[3], parts->second};
    }

    // _dependency_segments_for_rank_node (pipeline_ledger.py 1127-1131).
    std::set<std::string> dependency_segments_for_rank_node(
        std::uint64_t rank, std::uint64_t node_id) {
        std::set<DependencyRow> matches;
        dependency_interval_matches(find_dependency_rank_root(rank), node_id,
                                    matches);
        std::set<std::string> segments;
        for (const DependencyRow& row : matches) {
            segments.insert(row_segment(row));
        }
        return segments;
    }

    // _record_dependency_terminal (pipeline_ledger.py 2677-2696).
    void record_dependency_terminal(std::uint64_t rank, std::uint64_t node_id,
                                    const std::set<std::string>* segments) {
        terminal_provenance_roots_[rank] =
            completed_add(terminal_root(rank), node_id);
        std::set<std::string> owned;
        if (segments == nullptr) {
            owned = dependency_segments_for_rank_node(rank, node_id);
            segments = &owned;
        }
        for (const std::string& segment : *segments) {
            const WaiterKey key{rank, node_id, segment};
            const std::int64_t waiters = dependency_waiters_.get(key);
            if (waiters != 0) {
                dependency_waiters_.add(key, -waiters);
                dependency_pins_.add(segment, -waiters);
                if (dependency_waiters_.get(key) < 0) {
                    fail("ordinary dependency waiter aggregate underflow");
                }
                if (dependency_pins_.get(segment) < 0) {
                    fail("unfinished dependency pin aggregate underflow");
                }
            }
        }
    }

    void stage_node_events_before(std::uint64_t logical_seq);

    // -------------------------------------------------- cursor simulation --

    CursorProgress& seed_cursor_progress(const std::string& segment,
                                         std::uint64_t run_index);
    static bool progress_has_completed(const CursorProgress& progress,
                                       std::uint64_t ordinal) {
        return ordinal < progress.completed
            || completed_contains(progress.ooo_root, ordinal);
    }
    const SelectorIndex* selector_index_for(const std::string& segment) const {
        const auto persisted = st_.selector_indexes.find(segment);
        if (persisted != st_.selector_indexes.end()) return &persisted->second;
        const auto added = parsed_.selector_index_additions.find(segment);
        return added == parsed_.selector_index_additions.end() ? nullptr
                                                               : &added->second;
    }
    bool endpoint_completion_proven(const std::string& segment,
                                    const PipelineDescriptor& descriptor,
                                    std::uint64_t rank, std::uint64_t group,
                                    std::uint64_t tile, std::uint64_t shard,
                                    const std::string& endpoint);
    bool consume_terminal_proven(const PipelineDescriptor& descriptor,
                                 const std::string& segment,
                                 std::uint64_t rank, std::uint64_t group,
                                 std::uint64_t tile, std::uint64_t shard);
    std::optional<std::uint64_t> first_pending_store(
        const PipelineDescriptor& descriptor, std::uint64_t run_index,
        const CursorProgress& progress);
    std::optional<Json> expected_obligation(
        const PipelineDescriptor& descriptor, const std::string& segment,
        std::uint64_t run_index, const CursorProgress& progress);
    void merge_completion_range(CursorProgress& progress, std::uint64_t first,
                                std::uint64_t end, const char* path,
                                const std::string& segment,
                                std::uint64_t run_index) {
        CompletedPtr root;
        std::uint64_t completed = progress.completed;
        try {
            const auto merged = completed_add_range(
                progress.ooo_root, progress.completed, progress.total, first,
                end);
            root = merged.first;
            completed = merged.second;
        } catch (const LedgerError&) {
            fail(std::string(path) + " completes an endpoint twice");
        }
        const std::uint64_t completed_total =
            progress.completed_total + end - first;
        if (completed_total > progress.total) {
            fail(std::string(path) + " completed count exceeds its cursor extent");
        }
        progress.ooo_root = root;
        progress.completed = completed;
        progress.completed_total = completed_total;
        progress.ooo_changed = true;
        cursor_dirty_.insert({segment, run_index});
    }
    void merge_completion(CursorProgress& progress, std::uint64_t ordinal,
                          const char* path, const std::string& segment,
                          std::uint64_t run_index) {
        merge_completion_range(progress, ordinal, ordinal + 1, path, segment,
                               run_index);
    }
    void advance_inactive_positions(const PipelineDescriptor& descriptor,
                                    CursorProgress& progress,
                                    const std::string& segment,
                                    std::uint64_t run_index,
                                    const Json& run_transition);
    std::optional<SelectedEndpoint> selected_slot_and_credit(
        const PipelineDescriptor& descriptor, const Json& transition) {
        const Json& selector =
            row_field(transition, "endpoint_selector", "pipeline run");
        if (selector.is_null()) return std::nullopt;
        const auto run_index =
            row_opt_uint(transition, "endpoint_run_index", "pipeline run");
        if (!run_index || *run_index >= descriptor.endpoint_runs.size()) {
            fail("pipeline run selector references an unknown endpoint run");
        }
        SelectedEndpoint selected;
        selected.run = &descriptor.endpoint_runs[*run_index];
        selected.global_tile_ordinal = row_uint(
            transition, "_selected_global_tile_ordinal", "pipeline run");
        const Json& shard =
            row_field(transition, "_selected_shard_tile", "pipeline run");
        selected.shard.shard_ordinal = row_uint(
            shard, "shard_ordinal", "pipeline run._selected_shard_tile");
        selected.shard.source_rank =
            row_uint(shard, "source_rank", "pipeline run._selected_shard_tile");
        selected.shard.target_rank =
            row_uint(shard, "target_rank", "pipeline run._selected_shard_tile");
        selected.shard.data_bytes =
            row_uint(shard, "data_bytes", "pipeline run._selected_shard_tile");
        selected.shard.emit_endpoint = row_bool(
            shard, "emit_endpoint", "pipeline run._selected_shard_tile");
        const auto leg_index =
            row_opt_uint(transition, "endpoint_leg_index", "pipeline run");
        if (!leg_index || *leg_index >= selected.run->legs.size()) {
            fail("pipeline run selector references an unknown endpoint leg");
        }
        selected.endpoint = selected.run->legs[*leg_index].endpoint;
        return selected;
    }

    // -------------------------------------------------- slot/credit rows --

    void add_row_aggregate(const Json& row, std::int64_t sign) {
        const std::string domain =
            row_string(row, "buffer_domain_id", "slot row");
        slot_row_counts_.add(domain, sign);
        if (!row_bool(row, "occupied", "slot row")) return;
        const Json& owner = row_field(row, "owner", "slot row");
        slot_owner_counts_.add(
            row_string(owner, "segment_id", "slot row.owner"), sign);
        if (row_string(row, "role", "slot row") == "source") {
            source_occupancy_.add(
                CreditKey{domain, row_uint(row, "rank", "slot row"),
                          row_uint(row, "shard_ordinal", "slot row")},
                sign);
        } else {
            target_occupancy_.add(
                TargetDomainKey{domain, row_uint(row, "rank", "slot row")},
                sign);
        }
    }

    TargetWorkset* target_workset_for(const SlotKey& key);
    const Json* current_target_row(const SlotKey& key) {
        const auto workset = target_worksets_.find(key);
        if (workset != target_worksets_.end()) {
            return workset->second.row ? &*workset->second.row : nullptr;
        }
        return slot_overlay_.get_raw(key);
    }
    static Json target_row_empty(const SlotKey& key) {
        Json row = Json::object();
        row["buffer_domain_id"] = key.domain;
        row["role"] = "target";
        row["rank"] = key.rank;
        row["shard_ordinal"] = nullptr;
        row["slot_index"] = key.slot_index;
        row["occupied"] = false;
        row["owner"] = nullptr;
        row["consumed"] = false;
        row["target_shards_to_consume"] = Json::array();
        row["consumed_target_shards"] = Json::array();
        return row;
    }
    Lineage target_lineage(const SlotKey& key) const {
        const auto change = target_slot_lineage_changes_.find(key);
        if (change != target_slot_lineage_changes_.end()) return change->second;
        const auto persisted = st_.target_slot_lineage.find(key);
        if (persisted != st_.target_slot_lineage.end()) {
            return persisted->second;
        }
        return Lineage{0, 0, 0};
    }
    std::vector<std::uint64_t> claim_target_members(
        const Json& owner, const SlotKey& key,
        const std::vector<std::uint64_t>& declared);
    std::vector<std::uint64_t> consume_delta_members(
        const TargetWorkset& workset,
        const PipelineDescriptor& owner_descriptor, const std::string& segment,
        const Json& cause);
    bool prove_target_consume_clear(
        const Json& row, const PipelineDescriptor& owner_descriptor,
        const std::optional<SelectedEndpoint>& selected,
        const Json& transition);
    std::set<TargetDomainKey> apply_target_delta(
        const Json& transition, const PipelineDescriptor& descriptor,
        const std::optional<SelectedEndpoint>& selected, const Json& delta);
    std::pair<std::set<CreditKey>, std::set<TargetDomainKey>>
    apply_slot_patches(const Json& transition,
                       const PipelineDescriptor& descriptor,
                       const std::optional<SelectedEndpoint>& selected);
    std::set<CreditKey> apply_credit_patches(
        const Json& transition, const PipelineDescriptor& descriptor,
        const std::pair<std::optional<CreditKey>, std::int64_t>& expected_ack);
    void apply_obligation_patches(
        const Json& transition,
        const std::map<std::string, std::optional<Json>>& required_expected);
    void verify_credit_balance(const std::set<CreditKey>& keys,
                               const std::set<TargetDomainKey>& target_keys) {
        for (const CreditKey& key : keys) {
            const std::int64_t occupancy = source_occupancy_.get(key);
            const Json* credit = credit_overlay_.get_raw(key);
            if (occupancy < 0 || occupancy > 2) {
                fail("source pipeline slot occupancy exceeds two-slot capacity");
            }
            if (credit == nullptr) {
                if (occupancy != 0) {
                    fail("occupied source slots have no matching credit row");
                }
                continue;
            }
            const std::string path = "credit row";
            if (row_uint(*credit, "capacity", path) != 2
                    || row_uint(*credit, "available", path)
                           != 2 - static_cast<std::uint64_t>(occupancy)) {
                fail("source credit balance does not match occupied slots");
            }
            if (static_cast<std::int64_t>(
                    row_uint(*credit, "outstanding_ack_count", path)) < 0) {
                fail("source outstanding ACK count underflow");
            }
        }
        for (const TargetDomainKey& key : target_keys) {
            const std::int64_t occupancy = target_occupancy_.get(key);
            if (occupancy < 0 || occupancy > 2) {
                fail("target pipeline slot occupancy exceeds two-slot capacity");
            }
        }
    }
    std::optional<std::string> simulate_issue_or_completion(
        const Json& run_transition, CursorProgress& progress,
        const std::string& segment, std::uint64_t run_index);
    void run_transition_row(const Json& transition);
    void finalize();
    Json initial_cursor_run(const PipelineDescriptor& descriptor,
                            std::uint64_t run_index);
};

}  // namespace

// ------------------------------------------ StageDelivery member functions --

// _initial_cursor_run (pipeline_ledger.py 1215-1265).
Json StageDelivery::initial_cursor_run(const PipelineDescriptor& descriptor,
                                       std::uint64_t run_index) {
    const PipelineEndpointRun& run = descriptor.endpoint_runs[run_index];
    const std::size_t shard_count = run.shard_ordinals.size();
    const std::size_t leg_count = run.legs.size();
    const std::uint64_t total = run_endpoint_total(run);
    std::uint64_t prefix = 0;
    bool any_active = false;
    for (std::size_t shard_index = 0; shard_index < shard_count; ++shard_index) {
        const PipelineShardTile shard = descriptor.selected_shard_tile(
            run.group_ordinal, run.first_tile - 1,
            run.shard_ordinals[shard_index]).second;
        for (std::size_t leg_index = 0; leg_index < leg_count; ++leg_index) {
            const bool active =
                run.legs[leg_index].endpoint == "body_marker"
                || (shard.emit_endpoint && shard.data_bytes != 0);
            if (active) any_active = true;
            if (prefix == shard_index * leg_count + leg_index && !active) {
                prefix += 1;
            }
        }
    }
    if (!any_active) {
        // The executor's prepare_batch check sees an all-inactive first tile
        // and retires the remaining monotone zero-work suffix compactly.
        prefix = total;
    }
    const std::uint64_t position = prefix;
    std::uint64_t tile_offset = run.tile_count;
    std::uint64_t shard_index_out = 0;
    std::uint64_t leg_index_out = 0;
    std::optional<std::uint64_t> next_store;
    if (position != total) {
        const std::uint64_t per_tile =
            static_cast<std::uint64_t>(shard_count * leg_count);
        tile_offset = position / per_tile;
        const std::uint64_t within = position % per_tile;
        shard_index_out = static_cast<std::uint64_t>(within / leg_count);
        leg_index_out = static_cast<std::uint64_t>(within % leg_count);
        next_store = descriptor.endpoint_id(
            run_index, run.first_tile + tile_offset,
            run.shard_ordinals[shard_index_out],
            static_cast<std::size_t>(leg_index_out)) + 1;
    }
    Json row = Json::object();
    row["rank"] = run.rank;
    row["group_ordinal"] = run.group_ordinal;
    row["first_tile"] = run.first_tile;
    row["tile_count"] = run.tile_count;
    row["completed_endpoint_count"] = prefix;
    row["cancelled_endpoint_count"] = 0;
    row["remaining_endpoint_count"] = total - prefix;
    row["in_flight"] = false;
    row["next_store_id"] = next_store ? Json(*next_store) : Json(nullptr);
    row["tile_offset"] = tile_offset;
    row["shard_index"] = shard_index_out;
    row["leg_index"] = leg_index_out;
    row["initial_parent_store_id"] = run.initial_parent_json_id
        ? Json(*run.initial_parent_json_id + 1) : Json(nullptr);
    row["completed_out_of_order"] = Json::array();
    row["active_endpoints"] = Json::array();
    return row;
}

// stage_node_events_before (pipeline_ledger.py 2698-2769).
void StageDelivery::stage_node_events_before(std::uint64_t logical_seq) {
    // The node source (pre-materialized offline events) owns row validation
    // and WorkLedger staging; this loop keeps only the pipeline-side
    // bookkeeping that interleaves at each event's exact position.
    while (true) {
        if (node_index_ >= parsed_.node_events.size()) return;
        const std::uint64_t seq = parsed_.node_events[node_index_].first;
        if (seq >= logical_seq) return;
        const NodeEvent& event = parsed_.node_events[node_index_].second;
        node_index_ += 1;

        const CommittedWork* before = work_->tx_shadow_work(*tx_, event.work_id);
        const bool before_present = before != nullptr;
        const WorkState before_state =
            before != nullptr ? before->state : WorkState::Released;
        std::vector<std::string> before_dependencies;
        if (before != nullptr) before_dependencies = before->dependencies;
        work_->tx_stage_event(*tx_, seq, event);
        const CommittedWork* after = work_->tx_shadow_work(*tx_, event.work_id);

        const bool was_active =
            before_present
            && (before_state == WorkState::Committed
                || before_state == WorkState::Ready
                || before_state == WorkState::Issued
                || before_state == WorkState::InService);
        if (event.kind == "completion" || event.kind == "release") {
            const Json* rank_value = nullptr;
            const Json* node_value = nullptr;
            const auto rank_member = event.payload.find("rank");
            if (rank_member != event.payload.end()) rank_value = &rank_member.value();
            const auto node_member = event.payload.find("node_id");
            if (node_member != event.payload.end()) node_value = &node_member.value();
            if (rank_value != nullptr && node_value != nullptr
                    && rank_value->is_number_integer()
                    && node_value->is_number_integer()) {
                // NodeStore ids are producer-epoch/rank scoped. The
                // descriptor's affine JSON id is zero-based, while the
                // ordinary node store id is JSON id + 1.
                const auto to_u64 = [](const Json& value) {
                    return value.is_number_unsigned()
                        ? value.get<std::uint64_t>()
                        : static_cast<std::uint64_t>(value.get<std::int64_t>());
                };
                node_terminal_frontier_.insert(
                    {to_u64(*rank_value), to_u64(*node_value)});
                record_dependency_terminal(to_u64(*rank_value),
                                           to_u64(*node_value), nullptr);
            }
        }
        const bool is_active = active_state(after);
        if (was_active != is_active) {
            const std::int64_t direction = was_active ? -1 : 1;
            std::vector<std::string> dependencies;
            if (was_active) {
                dependencies = before_dependencies;
            } else if (after != nullptr) {
                dependencies = after->dependencies;
            }
            for (const std::string& dependency : dependencies) {
                const auto reference = dependency_reference(dependency);
                if (!reference) continue;
                const Json* state = nullptr;
                const auto touched = touched_state_.find(reference->segment);
                if (touched != touched_state_.end()) {
                    state = &touched->second;
                } else {
                    state = states_get(reference->segment);
                }
                if (state == nullptr) {
                    fail("ordinary dependency references an unknown pipeline descriptor");
                }
                if (terminal_proven(reference->rank, reference->node_id)) {
                    continue;
                }
                if (state_flag(*state, "retired")) {
                    fail("ordinary dependency references an incomplete endpoint in a retired pipeline descriptor");
                }
                const bool precommit_declaration =
                    event.kind == "commit"
                    && additions_.count(reference->segment) != 0
                    && !descriptor_commits_.count(reference->segment);
                if (!st_.active_segments.count(reference->segment)
                        && !descriptor_commits_.count(reference->segment)
                        && !precommit_declaration) {
                    fail("ordinary dependency references an uncommitted pipeline descriptor");
                }
                const WaiterKey waiter_key{reference->rank, reference->node_id,
                                           reference->segment};
                if (direction < 0
                        && dependency_waiters_.get(waiter_key) <= 0) {
                    fail("unfinished ordinary dependency waiter underflow");
                }
                dependency_waiters_.add(waiter_key, direction);
                dependency_pins_.add(reference->segment, direction);
                if (dependency_pins_.get(reference->segment) < 0) {
                    fail("unfinished dependency pin aggregate underflow");
                }
            }
        }
    }
}

// seed_cursor_progress (pipeline_ledger.py 2771-2824).
CursorProgress& StageDelivery::seed_cursor_progress(
    const std::string& segment, std::uint64_t run_index) {
    const CursorRunKey key{segment, run_index};
    const auto existing = cursor_sims_.find(key);
    if (existing != cursor_sims_.end()) return existing->second;
    const DescriptorPtr descriptor = find_descriptor(segment);
    if (!descriptor || run_index >= descriptor->endpoint_runs.size()) {
        fail("cursor run references unknown descriptor extent");
    }
    Json row;
    if (additions_.count(segment) != 0) {
        row = initial_cursor_run(*descriptor, run_index);
    } else {
        const auto runs = st_.cursor_runs_by_segment.find(segment);
        if (runs != st_.cursor_runs_by_segment.end()) {
            const auto found = runs->second.find(run_index);
            if (found != runs->second.end()) row = found->second;
        }
        if (row.is_null()) {
            fail("active descriptor cursor run is absent from accepted state");
        }
    }
    const PipelineEndpointRun& run = descriptor->endpoint_runs[run_index];
    std::vector<std::pair<std::uint64_t, std::uint64_t>> original_ooo;
    {
        const Json& ooo = row_field(row, "completed_out_of_order", "cursor run");
        if (!ooo.is_array()) {
            fail("cursor run completed_out_of_order must be an array");
        }
        for (const Json& item : ooo) {
            original_ooo.emplace_back(
                row_uint(item, "first_ordinal",
                         "cursor run.completed_out_of_order"),
                row_uint(item, "end_ordinal_exclusive",
                         "cursor run.completed_out_of_order"));
        }
    }
    const auto built = completed_build_from_half_open(original_ooo);
    std::map<std::uint64_t, std::pair<std::uint64_t, std::optional<Json>>>
        active;
    {
        const Json& endpoints = row_field(row, "active_endpoints", "cursor run");
        if (!endpoints.is_array()) {
            fail("cursor run active_endpoints must be an array");
        }
        for (const Json& endpoint : endpoints) {
            const std::string path = "cursor run.active_endpoints[]";
            const std::uint64_t local_tile =
                row_uint(endpoint, "tile_ordinal", path) - run.first_tile;
            const std::uint64_t shard_ordinal =
                row_uint(endpoint, "shard_ordinal", path);
            const std::size_t shard_index =
                index_of(run.shard_ordinals, shard_ordinal);
            if (shard_index == run.shard_ordinals.size()) {
                fail("cursor active endpoint shard is absent from its run");
            }
            const std::string endpoint_name =
                row_string(endpoint, "endpoint", path);
            std::size_t leg_index = run.legs.size();
            for (std::size_t i = 0; i < run.legs.size(); ++i) {
                if (run.legs[i].endpoint == endpoint_name) {
                    leg_index = i;
                    break;
                }
            }
            if (leg_index == run.legs.size()) {
                fail("cursor active endpoint leg is absent from its run");
            }
            const std::uint64_t ordinal =
                local_tile * run.shard_ordinals.size() * run.legs.size()
                + shard_index * run.legs.size() + leg_index;
            active[row_uint(endpoint, "store_id", path)] =
                std::make_pair(ordinal, std::optional<Json>(endpoint));
        }
    }
    CursorProgress progress;
    progress.row = row;
    progress.run = &run;
    progress.descriptor = descriptor;
    progress.total = run_endpoint_total(run);
    progress.completed =
        row_uint(row, "completed_endpoint_count", "cursor run");
    progress.completed_total = progress.completed + built.second;
    progress.cancelled =
        row_uint(row, "cancelled_endpoint_count", "cursor run");
    progress.remaining =
        row_uint(row, "remaining_endpoint_count", "cursor run");
    progress.ooo_root = built.first;
    progress.ooo_wire_ranges = original_ooo;
    progress.ooo_changed = false;
    progress.active = std::move(active);
    progress.position = cursor_position_ordinal(row, run);
    const Json& state = current_state(segment);
    progress.closed = state_flag(state, "request_closed");
    return cursor_sims_.emplace(key, std::move(progress)).first->second;
}

// endpoint_completion_proven (pipeline_ledger.py 2835-2856).
bool StageDelivery::endpoint_completion_proven(
    const std::string& segment, const PipelineDescriptor& descriptor,
    std::uint64_t rank, std::uint64_t group, std::uint64_t tile,
    std::uint64_t shard, const std::string& endpoint) {
    const SelectorIndex* indexes = selector_index_for(segment);
    if (indexes == nullptr) return false;
    const auto match =
        descriptor_selector_at(*indexes, group, rank, tile, shard, endpoint);
    if (!match) return false;
    const PipelineEndpointRun& run = descriptor.endpoint_runs[match->first];
    const std::size_t shard_position = index_of(run.shard_ordinals, shard);
    if (shard_position == run.shard_ordinals.size()) return false;
    if (tile < run.first_tile || tile >= run.first_tile + run.tile_count) {
        return false;
    }
    const std::uint64_t local_tile = tile - run.first_tile;
    const std::uint64_t ordinal =
        local_tile * run.shard_ordinals.size() * run.legs.size()
        + shard_position * run.legs.size() + match->second;
    return progress_has_completed(seed_cursor_progress(segment, match->first),
                                  ordinal);
}

// consume_terminal_proven (pipeline_ledger.py 2871-2907).
bool StageDelivery::consume_terminal_proven(
    const PipelineDescriptor& descriptor, const std::string& segment,
    std::uint64_t rank, std::uint64_t group, std::uint64_t tile,
    std::uint64_t shard) {
    const SelectorIndex* indexes = selector_index_for(segment);
    if (indexes == nullptr) return false;
    const auto match = consume_completion_at(*indexes, rank, group, tile, shard);
    if (!match) return false;
    const PipelineConsumeCompletionRun& completion =
        descriptor.consume_completion_runs[match->first];
    const std::uint64_t dependency_json_id =
        completion.first_dependency_json_id
        + (tile - completion.first_tile) * completion.dependency_tile_stride
        + match->second * completion.dependency_shard_stride;
    if (completion.dependency_kind == "ordinary_node_terminal") {
        const std::uint64_t store_id = dependency_json_id + 1;
        if (node_terminal_frontier_.count({rank, store_id})) return true;
        return completed_contains(terminal_root(rank), store_id);
    }
    if (completion.dependency_kind == "virtual_endpoint_terminal") {
        const auto marker = descriptor_selector_at(*indexes, group, rank, tile,
                                                   shard, "body_marker");
        if (!marker) return false;
        if (descriptor.endpoint_id(marker->first, tile, shard, marker->second)
                != dependency_json_id) {
            return false;
        }
        return endpoint_completion_proven(segment, descriptor, rank, group,
                                          tile, shard, "body_marker");
    }
    return false;
}

// _first_pending_store (pipeline_ledger.py 2912-2927).
std::optional<std::uint64_t> StageDelivery::first_pending_store(
    const PipelineDescriptor& descriptor, std::uint64_t run_index,
    const CursorProgress& progress) {
    if (progress.closed) {
        if (progress.active.empty()) return std::nullopt;
        return progress.active.begin()->first;
    }
    const std::uint64_t ordinal = progress.completed;
    if (ordinal >= progress.total) return std::nullopt;
    const PipelineEndpointRun& run = descriptor.endpoint_runs[run_index];
    const std::uint64_t per_tile =
        run.shard_ordinals.size() * run.legs.size();
    const std::uint64_t tile_offset = ordinal / per_tile;
    const std::uint64_t within = ordinal % per_tile;
    const std::uint64_t shard_index = within / run.legs.size();
    const std::uint64_t leg_index = within % run.legs.size();
    return descriptor.endpoint_id(run_index, run.first_tile + tile_offset,
                                  run.shard_ordinals[shard_index],
                                  static_cast<std::size_t>(leg_index)) + 1;
}

// expected_obligation (pipeline_ledger.py 2929-2948).
std::optional<Json> StageDelivery::expected_obligation(
    const PipelineDescriptor& descriptor, const std::string& segment,
    std::uint64_t run_index, const CursorProgress& progress) {
    const std::int64_t remaining = progress.closed
        ? static_cast<std::int64_t>(progress.active.size())
        : static_cast<std::int64_t>(progress.total)
              - static_cast<std::int64_t>(progress.completed_total)
              - static_cast<std::int64_t>(progress.cancelled);
    if (remaining <= 0) return std::nullopt;
    const PipelineEndpointRun& run = descriptor.endpoint_runs[run_index];
    const auto first_store_id =
        first_pending_store(descriptor, run_index, progress);
    if (!first_store_id) {
        fail("unfinished cursor run has no first unresolved endpoint");
    }
    Json row = Json::object();
    row["obligation_id"] = segment + "/run/" + num(run_index);
    row["segment_id"] = segment;
    row["rank"] = run.rank;
    row["first_store_id"] = *first_store_id;
    row["remaining_endpoint_count"] = static_cast<std::uint64_t>(remaining);
    row["state"] = progress.active.empty() ? "pending" : "active";
    return row;
}

// advance_inactive_positions (pipeline_ledger.py 2973-3040).
void StageDelivery::advance_inactive_positions(
    const PipelineDescriptor& descriptor, CursorProgress& progress,
    const std::string& segment, std::uint64_t run_index,
    const Json& run_transition) {
    if (progress.position >= progress.total) return;
    const PipelineEndpointRun& run = *progress.run;
    const std::size_t shard_count = run.shard_ordinals.size();
    const std::size_t leg_count = run.legs.size();
    const std::uint64_t per_tile =
        static_cast<std::uint64_t>(shard_count * leg_count);
    const std::uint64_t tile_offset = progress.position / per_tile;
    const std::uint64_t tile_start = tile_offset * per_tile;
    const std::uint64_t tile_end =
        std::min(progress.total, tile_start + per_tile);
    const std::uint64_t skipped_start = progress.position;
    const Json& transition_selector =
        row_field(run_transition, "endpoint_selector", "pipeline run");
    const Json* transition_shard_facts = nullptr;
    {
        const Json& facts =
            row_field(run_transition, "_selected_shard_tile", "pipeline run");
        if (!facts.is_null()) transition_shard_facts = &facts;
    }
    while (progress.position < tile_end) {
        const std::uint64_t within = progress.position - tile_start;
        const std::uint64_t shard_index = within / leg_count;
        const std::uint64_t leg_index = within % leg_count;
        const std::uint64_t shard_ordinal =
            run.shard_ordinals[static_cast<std::size_t>(shard_index)];
        const PipelineEndpointLegRun& leg =
            run.legs[static_cast<std::size_t>(leg_index)];
        bool is_active;
        if (leg.endpoint == "body_marker") {
            is_active = true;
        } else {
            const std::uint64_t candidate_tile_ordinal =
                run.first_tile + tile_offset;
            bool matched = false;
            bool emit_endpoint = false;
            if (!transition_selector.is_null()
                    && transition_shard_facts != nullptr
                    && row_uint(transition_selector, "group_ordinal",
                                "pipeline run.endpoint_selector")
                           == run.group_ordinal
                    && row_uint(transition_selector, "tile_ordinal",
                                "pipeline run.endpoint_selector")
                           == candidate_tile_ordinal
                    && row_uint(transition_selector, "shard_ordinal",
                                "pipeline run.endpoint_selector")
                           == shard_ordinal) {
                matched = true;
                emit_endpoint = row_bool(*transition_shard_facts,
                                         "emit_endpoint", "pipeline run");
            }
            if (!matched) {
                emit_endpoint = descriptor.selected_shard_tile(
                    run.group_ordinal, run.first_tile + tile_offset - 1,
                    shard_ordinal).second.emit_endpoint;
            }
            is_active = emit_endpoint;
        }
        if (is_active) {
            if (progress.position > skipped_start) {
                merge_completion_range(progress, skipped_start,
                                       progress.position,
                                       "inactive cursor skip", segment,
                                       run_index);
                cursor_unissued_.add(
                    segment,
                    -static_cast<std::int64_t>(
                        progress.position - skipped_start));
            }
            return;
        }
        progress.position += 1;
    }
    std::vector<std::pair<std::uint64_t, std::uint64_t>> body_positions;
    for (std::uint64_t si = 0; si < static_cast<std::uint64_t>(shard_count);
         ++si) {
        for (std::uint64_t li = 0; li < static_cast<std::uint64_t>(leg_count);
             ++li) {
            if (run.legs[static_cast<std::size_t>(li)].endpoint
                    == "body_marker") {
                body_positions.emplace_back(si, li);
            }
        }
    }
    std::uint64_t next_position;
    if (!body_positions.empty() && tile_offset + 1 < run.tile_count) {
        next_position = (tile_offset + 1) * per_tile
            + body_positions.front().first * leg_count
            + body_positions.front().second;
        progress.position = next_position;
    } else {
        progress.position = progress.total;
        next_position = progress.position;
    }
    if (next_position > skipped_start) {
        merge_completion_range(progress, skipped_start, next_position,
                               "inactive cursor skip", segment, run_index);
        cursor_unissued_.add(
            segment,
            -static_cast<std::int64_t>(next_position - skipped_start));
    }
}

// target_workset_for (pipeline_ledger.py 3069-3088).
TargetWorkset* StageDelivery::target_workset_for(const SlotKey& key) {
    const auto existing = target_worksets_.find(key);
    if (existing != target_worksets_.end()) return &existing->second;
    const Json* row = slot_overlay_.get_raw(key);
    if (row == nullptr) return nullptr;
    if (row_string(*row, "role", "slot row") != "target") {
        fail("target delta addresses a source slot");
    }
    TargetWorkset workset;
    for (const std::uint64_t shard :
         row_uint_array(*row, "consumed_target_shards", "slot row")) {
        workset.consumed.insert(shard);
    }
    workset.remaining =
        static_cast<std::int64_t>(
            row_uint_array(*row, "target_shards_to_consume", "slot row").size())
        - static_cast<std::int64_t>(workset.consumed.size());
    workset.row = *row;
    return &target_worksets_.emplace(key, std::move(workset)).first->second;
}

// _claim_target_members (pipeline_ledger.py 3110-3151).
std::vector<std::uint64_t> StageDelivery::claim_target_members(
    const Json& owner, const SlotKey& key,
    const std::vector<std::uint64_t>& declared) {
    const std::string owner_segment =
        row_string(owner, "segment_id", "target claim owner");
    const DescriptorPtr owner_descriptor = find_descriptor(owner_segment);
    if (!owner_descriptor) {
        fail("target claim owner references an unknown descriptor");
    }
    const PipelineIdentity& ident = owner_descriptor->identity;
    if (ident.buffer_domain_id != key.domain
            || ident.rank != key.rank
            || ident.generation
                   != row_uint(owner, "pipeline_generation",
                               "target claim owner")
            || ident.stage_epoch
                   != row_uint(owner, "stage_epoch", "target claim owner")
            || ident.staging_allocation_id
                   != row_uint(owner, "staging_allocation_id",
                               "target claim owner")) {
        fail("target claim owner disagrees with descriptor identity");
    }
    const std::uint64_t group_ordinal =
        row_uint(owner, "group_ordinal", "target claim owner");
    const SelectorIndex* owner_index = selector_index_for(owner_segment);
    const PipelineGroup* group = nullptr;
    if (owner_index != nullptr) {
        const auto found = owner_index->groups_by_ordinal.find(group_ordinal);
        if (found != owner_index->groups_by_ordinal.end()
                && found->second < owner_descriptor->groups.size()) {
            group = &owner_descriptor->groups[found->second];
        }
    }
    if (group == nullptr) {
        fail("target claim owner references an unknown tile group");
    }
    const std::uint64_t tile_ordinal =
        row_uint(owner, "tile_ordinal", "target claim owner");
    if (tile_ordinal < 1 || tile_ordinal > group->tile_axis.tile_count) {
        fail("target claim owner points outside descriptor tile range");
    }
    const std::uint64_t global_tile =
        group->tile_axis.global_tile_first + tile_ordinal - 1;
    if (key.slot_index != (global_tile - 1) % 2) {
        fail("target claim slot index disagrees with global tile parity");
    }
    std::vector<std::uint64_t> expected;
    for (std::uint64_t shard_ordinal = 0;
         shard_ordinal < group->shards.size(); ++shard_ordinal) {
        const PipelineShardTile shard = owner_descriptor->selected_shard_tile(
            group_ordinal, tile_ordinal - 1, shard_ordinal).second;
        if (shard.emit_endpoint && shard.target_rank == ident.rank) {
            expected.push_back(shard_ordinal);
        }
    }
    if (expected.empty() || declared != expected) {
        fail("target claim does not declare the exact descriptor membership");
    }
    return expected;
}

// _consume_delta_members (pipeline_ledger.py 3152-3221).
std::vector<std::uint64_t> StageDelivery::consume_delta_members(
    const TargetWorkset& workset, const PipelineDescriptor& owner_descriptor,
    const std::string& segment, const Json& cause) {
    const std::uint64_t run_index =
        row_uint(cause, "consume_run_index", "target consume cause");
    if (run_index >= owner_descriptor.consume_completion_runs.size()) {
        fail("target consume cause run index is outside the descriptor");
    }
    const PipelineConsumeCompletionRun& completion =
        owner_descriptor.consume_completion_runs[run_index];
    const Json& row = *workset.row;
    const Json& owner = row_field(row, "owner", "target consume owner");
    const std::uint64_t tile =
        row_uint(owner, "tile_ordinal", "target consume owner");
    const std::uint64_t terminal_rank =
        row_uint(cause, "terminal_rank", "target consume cause");
    if (completion.rank != terminal_rank
            || completion.rank != row_uint(row, "rank", "slot row")
            || completion.group_ordinal
                   != row_uint(owner, "group_ordinal",
                               "target consume owner")
            || tile < completion.first_tile
            || tile >= completion.first_tile + completion.tile_count) {
        fail("target consume cause run does not cover the slot owner");
    }
    const std::uint64_t tile_delta = tile - completion.first_tile;
    const std::uint64_t terminal_store_id =
        row_uint(cause, "terminal_store_id", "target consume cause");
    const unsigned __int128 wide_json_id =
        static_cast<unsigned __int128>(completion.first_dependency_json_id)
        + static_cast<unsigned __int128>(tile_delta)
              * completion.dependency_tile_stride;
    if (wide_json_id + 1 > static_cast<unsigned __int128>(U64_MAX)) {
        fail("target consume cause affine terminal ID overflows uint64");
    }
    const std::uint64_t affine_store_id =
        static_cast<std::uint64_t>(wide_json_id) + 1;
    std::vector<std::uint64_t> mapped;
    if (completion.dependency_shard_stride != 0) {
        if (terminal_store_id < affine_store_id) {
            fail("target consume cause store ID precedes its affine base");
        }
        const std::uint64_t difference = terminal_store_id - affine_store_id;
        if (difference % completion.dependency_shard_stride != 0) {
            fail("target consume cause store ID is outside its affine shard mapping");
        }
        const std::uint64_t shard_position =
            difference / completion.dependency_shard_stride;
        if (shard_position >= completion.shard_ordinals.size()) {
            fail("target consume cause shard index is outside its run");
        }
        mapped.push_back(completion.shard_ordinals[shard_position]);
    } else {
        if (terminal_store_id != affine_store_id) {
            fail("target consume cause store ID differs from its shared marker");
        }
        mapped = completion.shard_ordinals;
    }
    const SelectorIndex* indexes = selector_index_for(segment);
    if (indexes == nullptr) {
        fail("target consume cause has no retained descriptor index");
    }
    for (const std::uint64_t shard_ordinal : mapped) {
        const auto proof = consume_completion_at(
            *indexes, terminal_rank, completion.group_ordinal, tile,
            shard_ordinal);
        if (!proof || proof->first != run_index) {
            fail("target consume cause does not name the owner's terminal run");
        }
        const std::uint64_t expected_json_id =
            completion.first_dependency_json_id
            + tile_delta * completion.dependency_tile_stride
            + proof->second * completion.dependency_shard_stride;
        if (expected_json_id + 1 != terminal_store_id) {
            fail("target consume cause store ID disagrees with its run member");
        }
        if (!consume_terminal_proven(owner_descriptor, segment, terminal_rank,
                                     completion.group_ordinal, tile,
                                     shard_ordinal)) {
            fail("target consume cause lacks prior terminal proof");
        }
    }
    std::vector<std::uint64_t> expected_new;
    for (const std::uint64_t shard : mapped) {
        if (!workset.consumed.count(shard)) expected_new.push_back(shard);
    }
    std::sort(expected_new.begin(), expected_new.end());
    return expected_new;
}

// _prove_target_consume_clear (pipeline_ledger.py 3223-3263).
bool StageDelivery::prove_target_consume_clear(
    const Json& row, const PipelineDescriptor& owner_descriptor,
    const std::optional<SelectedEndpoint>& selected, const Json& transition) {
    const Json& owner = row_field(row, "owner", "slot row");
    const std::string owner_segment =
        row_string(owner, "segment_id", "slot row.owner");
    const std::vector<std::uint64_t> owner_shards =
        row_uint_array(row, "target_shards_to_consume", "slot row");
    const std::uint64_t row_rank = row_uint(row, "rank", "slot row");
    const std::uint64_t owner_group =
        row_uint(owner, "group_ordinal", "slot row.owner");
    const std::uint64_t owner_tile =
        row_uint(owner, "tile_ordinal", "slot row.owner");
    bool selected_consumer_proof = !selected.has_value();
    if (selected.has_value()) {
        const Json& selector =
            row_field(transition, "endpoint_selector", "pipeline run");
        selected_consumer_proof =
            selected->endpoint == "body_marker"
            && selected->run->rank == row_rank
            && selected->run->group_ordinal == owner_group
            && row_uint(selector, "tile_ordinal",
                        "pipeline run.endpoint_selector") == owner_tile
            && std::find(owner_shards.begin(), owner_shards.end(),
                         selected->shard.shard_ordinal) != owner_shards.end()
            && endpoint_completion_proven(owner_segment, owner_descriptor,
                                          row_rank, owner_group, owner_tile,
                                          selected->shard.shard_ordinal,
                                          "body_marker");
    }
    bool ready_and_terminal = !owner_shards.empty();
    const OwnerTag ready_tag = owner_tag_of(owner, "slot row.owner");
    for (const std::uint64_t target_shard : owner_shards) {
        if (!endpoint_completion_proven(owner_segment, owner_descriptor,
                                        row_rank, owner_group, owner_tile,
                                        target_shard, "data_recv")) {
            ready_and_terminal = false;
            break;
        }
        const ReadyKey ready_key{
            row_string(row, "buffer_domain_id", "slot row"), row_rank,
            row_uint(row, "slot_index", "slot row"), target_shard};
        const OwnerTag* ready_owner = target_ready_overlay_.get_raw(ready_key);
        const bool already_ready =
            ready_owner != nullptr && *ready_owner == ready_tag;
        if (!already_ready) {
            pending_ready_after_clear_.insert({ready_key, ready_tag});
        }
        if (!consume_terminal_proven(owner_descriptor, owner_segment, row_rank,
                                     owner_group, owner_tile, target_shard)) {
            ready_and_terminal = false;
            break;
        }
    }
    return !owner_shards.empty() && selected_consumer_proof
        && ready_and_terminal;
}

// apply_target_delta (pipeline_ledger.py 3265-3469).
std::set<TargetDomainKey> StageDelivery::apply_target_delta(
    const Json& transition, const PipelineDescriptor& descriptor,
    const std::optional<SelectedEndpoint>& selected, const Json& delta) {
    const std::string path = "target delta";
    const SlotKey key = parse_slot_key(row_field(delta, "key", path), path);
    if (key.domain != descriptor.identity.buffer_domain_id) {
        fail(path + " crosses the transition buffer domain");
    }
    if (key.rank != descriptor.identity.rank) {
        fail(path + " rank differs from its buffer-domain descriptor");
    }
    const Lineage lineage = target_lineage(key);
    const std::uint64_t before_generation =
        row_uint(delta, "before_slot_generation", path);
    const std::uint64_t predecessor_seq =
        row_uint(delta, "predecessor_logical_seq", path);
    const std::uint64_t predecessor_count =
        row_uint(delta, "predecessor_consumed_count", path);
    const std::uint64_t transition_seq =
        row_uint(transition, "first_logical_seq", "pipeline run");
    if (before_generation != lineage[0] || predecessor_seq != lineage[1]
            || predecessor_count != lineage[2]
            || predecessor_seq >= transition_seq) {
        fail(path + " predecessor lineage differs from accepted state");
    }
    TargetWorkset* workset_ptr = target_workset_for(key);
    const bool initial_present = workset_ptr != nullptr && workset_ptr->row;
    const bool initial_occupied =
        initial_present && row_bool(*workset_ptr->row, "occupied", path);
    const Json* initial_owner =
        initial_occupied ? &row_field(*workset_ptr->row, "owner", path) : nullptr;
    std::vector<std::uint64_t> initial_targets;
    if (initial_present) {
        initial_targets =
            row_uint_array(*workset_ptr->row, "target_shards_to_consume", path);
    }
    std::uint64_t generation = lineage[0];
    if (workset_ptr == nullptr) {
        workset_ptr = &target_worksets_.emplace(key, TargetWorkset{}).first->second;
    }
    TargetWorkset& workset = *workset_ptr;
    std::optional<Json> row = workset.row;
    const std::string kind =
        row_string(transition, "transition_kind", "pipeline run");
    const Json& operations = row_field(delta, "operations", path);
    if (!operations.is_array()) fail(path + ".operations must be an array");
    for (const Json& op : operations) {
        const std::string op_path = path + ".operations[]";
        const std::string op_kind = row_string(op, "op", op_path);
        const std::uint64_t op_generation =
            row_uint(op, "slot_generation", op_path);
        if (op_kind == "claim") {
            if (generation == U64_MAX) {
                fail("target slot generation overflows uint64");
            }
            if (op_generation != generation + 1 || !row
                    || row_bool(*row, "occupied", op_path)) {
                fail("target claim generation or empty-slot precondition differs");
            }
            const Json& owner = row_field(op, "owner", op_path);
            const std::string owner_segment =
                row_string(owner, "segment_id", op_path + ".owner");
            if (owner_segment
                    != row_string(transition, "descriptor_segment_id",
                                  "pipeline run")) {
                fail("target claim assigns a different descriptor owner");
            }
            const DescriptorPtr owner_descriptor =
                find_descriptor(owner_segment);
            if (!owner_descriptor) {
                fail("target claim owner references an unknown descriptor");
            }
            const std::vector<std::uint64_t> declared =
                row_uint_array(op, "target_shards", op_path);
            claim_target_members(owner, key, declared);
            if (kind != "tile_issue" && kind != "consume_start"
                    && kind != "fill_ack_progress"
                    && kind != "release_ack_progress"
                    && kind != "consume_done") {
                fail("target slot ownership changed outside an issue/consume transition");
            }
            if (selected.has_value()) {
                const Json& selector =
                    row_field(transition, "endpoint_selector", "pipeline run");
                if (row_uint(owner, "tile_ordinal", op_path + ".owner")
                        != row_uint(selector, "tile_ordinal",
                                    "pipeline run.endpoint_selector")
                    || row_uint(owner, "group_ordinal", op_path + ".owner")
                        != selected->run->group_ordinal
                    || selected->shard.target_rank != key.rank) {
                    fail("target claim owner differs from selected endpoint");
                }
            }
            Json new_row = Json::object();
            new_row["buffer_domain_id"] = key.domain;
            new_row["role"] = "target";
            new_row["rank"] = key.rank;
            new_row["shard_ordinal"] = nullptr;
            new_row["slot_index"] = key.slot_index;
            new_row["occupied"] = true;
            new_row["owner"] = owner;
            new_row["consumed"] = false;
            new_row["target_shards_to_consume"] =
                row_field(op, "target_shards", op_path);
            new_row["consumed_target_shards"] = Json::array();
            row = std::move(new_row);
            workset.consumed.clear();
            workset.remaining = static_cast<std::int64_t>(declared.size());
            workset.non_target_count = 0;
            generation = op_generation;
        } else if (op_kind == "initialize") {
            if (op_generation != generation || row.has_value()) {
                fail("target initialize generation or absent-row precondition differs");
            }
            if (kind != "descriptor_commit"
                    || descriptor.identity.buffer_domain_id != key.domain
                    || descriptor.identity.rank != key.rank) {
                fail("target initialize is outside its descriptor commit");
            }
            row = target_row_empty(key);
            workset.consumed.clear();
            workset.remaining = 0;
            workset.non_target_count = 0;
        } else if (op_kind == "consume") {
            if (op_generation != generation || !row
                    || !row_bool(*row, "occupied", op_path) || kind != "consume_done") {
                fail("target consume lacks its occupied owner row");
            }
            const Json& owner = row_field(*row, "owner", op_path);
            const std::string owner_segment =
                row_string(owner, "segment_id", op_path + ".owner");
            if (owner_segment
                    != row_string(transition, "descriptor_segment_id",
                                  "pipeline run")) {
                fail("target consume owner differs from its sequenced descriptor");
            }
            const DescriptorPtr owner_descriptor =
                find_descriptor(owner_segment);
            if (!owner_descriptor) {
                fail("target consume owner references an unknown descriptor");
            }
            const Json& cause = row_field(op, "cause", op_path);
            const std::vector<std::uint64_t> expected_new = consume_delta_members(
                workset, *owner_descriptor, owner_segment, cause);
            const std::vector<std::uint64_t> add_shards =
                row_uint_array(op, "add_shards", op_path);
            if (expected_new.empty() || add_shards != expected_new) {
                fail("target consume additions differ from exact cause mapping");
            }
            const std::vector<std::uint64_t> targets =
                row_uint_array(*row, "target_shards_to_consume", op_path);
            for (const std::uint64_t shard : expected_new) {
                workset.consumed.insert(shard);
                if (std::binary_search(targets.begin(), targets.end(), shard)) {
                    workset.remaining -= 1;
                } else {
                    workset.non_target_count += 1;
                }
            }
            (*row)["consumed"] = !workset.consumed.empty();
            const bool cleared = row_bool(op, "cleared", op_path);
            if (cleared != (workset.remaining == 0)) {
                fail("target consume cleared flag differs from remaining membership");
            }
            if (cleared) {
                if (!prove_target_consume_clear(*row, *owner_descriptor,
                                                selected, transition)) {
                    fail("target consume cleared before its ready/terminal proof");
                }
                row = target_row_empty(key);
                workset.consumed.clear();
                workset.remaining = 0;
                workset.non_target_count = 0;
            }
        } else if (op_kind == "request_close_clear") {
            if (op_generation != generation || !row
                    || !row_bool(*row, "occupied", op_path)) {
                fail("request-close clear lacks its occupied target row");
            }
            const Json& owner = row_field(*row, "owner", op_path);
            const DescriptorPtr owner_descriptor = find_descriptor(
                row_string(owner, "segment_id", op_path + ".owner"));
            if (!owner_descriptor || kind != "request_close"
                    || owner_descriptor->identity.request_id
                        != descriptor.identity.request_id
                    || owner_descriptor->identity.buffer_domain_id
                        != descriptor.identity.buffer_domain_id) {
                fail("request-close clear touches a foreign request/domain");
            }
            row = target_row_empty(key);
            workset.consumed.clear();
            workset.remaining = 0;
            workset.non_target_count = 0;
        } else if (op_kind == "erase") {
            if (op_generation != generation || !row
                    || row_bool(*row, "occupied", op_path)
                    || kind != "descriptor_retire") {
                fail("target erase requires an empty row at descriptor retirement");
            }
            row = std::nullopt;
            workset.consumed.clear();
            workset.remaining = 0;
            workset.non_target_count = 0;
        }
    }
    if (row && row_bool(*row, "occupied", path)) {
        const std::vector<std::uint64_t> targets =
            row_uint_array(*row, "target_shards_to_consume", path);
        if (workset.non_target_count != 0
                || static_cast<std::uint64_t>(
                       workset.consumed.size() + workset.remaining)
                       != targets.size()
                || row_bool(*row, "consumed", path)
                       != !workset.consumed.empty()) {
            fail("occupied target row violates its full-row consume invariant");
        }
    }
    const bool final_present = row.has_value();
    const bool final_occupied =
        final_present && row_bool(*row, "occupied", path);
    const Json* final_owner =
        final_occupied ? &row_field(*row, "owner", path) : nullptr;
    // Public row/owner aggregates are updated once per logical target patch,
    // never once per operation inside its coalesced delta.
    if (initial_present) slot_row_counts_.add(key.domain, -1);
    if (initial_occupied) {
        slot_owner_counts_.add(
            row_string(*initial_owner, "segment_id", path), -1);
        target_occupancy_.add({key.domain, key.rank}, -1);
    }
    if (final_present) slot_row_counts_.add(key.domain, 1);
    if (final_occupied) {
        slot_owner_counts_.add(row_string(*final_owner, "segment_id", path), 1);
        target_occupancy_.add({key.domain, key.rank}, 1);
    }
    const bool have_old_tag = initial_occupied;
    const OwnerTag old_tag =
        initial_occupied ? owner_tag_of(*initial_owner, path) : OwnerTag{};
    const OwnerTag new_tag =
        final_occupied ? owner_tag_of(*final_owner, path) : OwnerTag{};
    if (have_old_tag && (!final_occupied || !(old_tag == new_tag))) {
        for (const std::uint64_t target_shard : initial_targets) {
            target_ready_overlay_.del(
                ReadyKey{key.domain, key.rank, key.slot_index, target_shard});
        }
    }
    // Preserve the old occupied-owner displacement and slot-reuse proof when
    // a coalesced patch keeps both rows occupied.
    if (initial_occupied && final_occupied) {
        const std::string initial_segment =
            row_string(*initial_owner, "segment_id", path);
        const std::string final_segment =
            row_string(*final_owner, "segment_id", path);
        if (initial_segment != final_segment) {
            const Json* old_state = nullptr;
            const auto touched = touched_state_.find(initial_segment);
            if (touched != touched_state_.end()) {
                old_state = &touched->second;
            } else {
                old_state = states_get(initial_segment);
            }
            if (old_state == nullptr
                    || !(state_flag(*old_state, "settle_done")
                         || state_flag(*old_state, "retired"))) {
                fail("foreign target slot owner was displaced before settle");
            }
        } else if (row_uint(*initial_owner, "group_ordinal", path)
                       != row_uint(*final_owner, "group_ordinal", path)
                   || row_uint(*initial_owner, "tile_ordinal", path)
                          != row_uint(*final_owner, "tile_ordinal", path)) {
            fail("target slot owner changed before its legal clear");
        }
    }
    const std::uint64_t final_count =
        final_occupied ? static_cast<std::uint64_t>(workset.consumed.size()) : 0;
    target_slot_lineage_changes_[key] =
        Lineage{generation, transition_seq, final_count};
    workset.row = row;
    return {{key.domain, key.rank}};
}

// apply_slot_patches (pipeline_ledger.py 3471-3721).
std::pair<std::set<CreditKey>, std::set<TargetDomainKey>>
StageDelivery::apply_slot_patches(
    const Json& transition, const PipelineDescriptor& descriptor,
    const std::optional<SelectedEndpoint>& selected) {
    std::set<CreditKey> touched_keys;
    std::set<TargetDomainKey> touched_targets;
    std::vector<const Json*> target_deltas;
    std::vector<std::tuple<SlotKey, const Json*, const Json*>> source_patches;
    const Json& patches = row_field(transition, "slot_patches", "pipeline run");
    if (!patches.is_array()) {
        fail("pipeline run slot_patches must be an array");
    }
    for (const Json& patch : patches) {
        if (patch.is_object() && patch.contains("form")
                && patch["form"] == "target_delta") {
            target_deltas.push_back(&patch);
        } else {
            if (!patch.is_array() || patch.size() != 4
                    || !patch[0].is_string() || patch[0] != "source") {
                fail("slot patch must be a source quad or target_delta object");
            }
            source_patches.emplace_back(parse_slot_key(patch[1], "slot patch"),
                                        &patch[2], &patch[3]);
        }
    }
    const std::string kind =
        row_string(transition, "transition_kind", "pipeline run");
    const std::string path = "slot patch";
    for (const auto& entry : source_patches) {
        const SlotKey& key = std::get<0>(entry);
        const Json* before_field = std::get<1>(entry);
        const Json* after_field = std::get<2>(entry);
        const bool before_null =
            before_field == nullptr || before_field->is_null();
        const bool after_null =
            after_field == nullptr || after_field->is_null();
        const bool present = slot_overlay_.has(key);
        const Json* current = slot_overlay_.get_raw(key);
        if (before_null) {
            if (present) fail(path + " expected an absent before row");
        } else if (!present || !typed_equal(*current, *before_field)) {
            fail(path + " before row differs from accepted COW state");
        }
        const Json* old = present ? current : nullptr;
        const Json* new_row = after_null ? nullptr : after_field;
        if (old != nullptr
                && row_string(*old, "buffer_domain_id", path)
                        != descriptor.identity.buffer_domain_id) {
            fail(path + " crosses the transition buffer domain");
        }
        if (new_row != nullptr
                && row_string(*new_row, "buffer_domain_id", path)
                        != descriptor.identity.buffer_domain_id) {
            fail(path + " crosses the transition buffer domain");
        }
        const std::string* endpoint_name =
            selected.has_value() ? &selected->endpoint : nullptr;
        if (old != nullptr && row_bool(*old, "occupied", path)
                && (new_row == nullptr
                    || !row_bool(*new_row, "occupied", path))) {
            const Json& old_owner = row_field(*old, "owner", path);
            const std::string old_owner_segment =
                row_string(old_owner, "segment_id", path + ".owner");
            bool allowed_clear = false;
            if (kind == "request_close") {
                const DescriptorPtr owner_descriptor =
                    find_descriptor(old_owner_segment);
                allowed_clear =
                    owner_descriptor != nullptr
                    && owner_descriptor->identity.request_id
                        == descriptor.identity.request_id
                    && owner_descriptor->identity.buffer_domain_id
                        == descriptor.identity.buffer_domain_id;
            }
            if (kind == "consume_done"
                    && row_string(*old, "role", path) == "target"
                    && old_owner_segment
                        == row_string(transition, "descriptor_segment_id",
                                      "pipeline run")) {
                const DescriptorPtr owner_descriptor =
                    find_descriptor(old_owner_segment);
                const std::vector<std::uint64_t> owner_shards =
                    row_uint_array(*old, "target_shards_to_consume", path);
                bool selected_consumer_proof = !selected.has_value();
                if (selected.has_value()) {
                    const Json& selector = row_field(
                        transition, "endpoint_selector", "pipeline run");
                    selected_consumer_proof =
                        selected->endpoint == "body_marker"
                        && selected->run->rank == row_uint(*old, "rank", path)
                        && selected->run->group_ordinal
                            == row_uint(old_owner, "group_ordinal", path + ".owner")
                        && row_uint(selector, "tile_ordinal",
                                    "pipeline run.endpoint_selector")
                            == row_uint(old_owner, "tile_ordinal",
                                        path + ".owner")
                        && std::find(owner_shards.begin(), owner_shards.end(),
                                     selected->shard.shard_ordinal)
                            != owner_shards.end()
                        && endpoint_completion_proven(
                            old_owner_segment, *owner_descriptor,
                            row_uint(*old, "rank", path),
                            row_uint(old_owner, "group_ordinal", path + ".owner"),
                            row_uint(old_owner, "tile_ordinal", path + ".owner"),
                            selected->shard.shard_ordinal, "body_marker");
                }
                bool ready_and_terminal = !owner_shards.empty();
                const OwnerTag ready_tag =
                    owner_tag_of(old_owner, path + ".owner");
                for (const std::uint64_t target_shard : owner_shards) {
                    if (!endpoint_completion_proven(
                            old_owner_segment, *owner_descriptor,
                            row_uint(*old, "rank", path),
                            row_uint(old_owner, "group_ordinal", path + ".owner"),
                            row_uint(old_owner, "tile_ordinal", path + ".owner"),
                            target_shard, "data_recv")) {
                        ready_and_terminal = false;
                        break;
                    }
                    const ReadyKey ready_key{
                        row_string(*old, "buffer_domain_id", path),
                        row_uint(*old, "rank", path),
                        row_uint(*old, "slot_index", path), target_shard};
                    const OwnerTag* ready_owner =
                        target_ready_overlay_.get_raw(ready_key);
                    const bool already_ready =
                        ready_owner != nullptr && *ready_owner == ready_tag;
                    if (!already_ready) {
                        // A physical data_recv callback publishes
                        // endpoint_done, consumes terminal watchers, then
                        // publishes data_ready. Its exact cursor completion
                        // may precede this clear in the same callback.
                        pending_ready_after_clear_.insert({ready_key, ready_tag});
                    }
                    if (!consume_terminal_proven(
                            *owner_descriptor, old_owner_segment,
                            row_uint(*old, "rank", path),
                            row_uint(old_owner, "group_ordinal", path + ".owner"),
                            row_uint(old_owner, "tile_ordinal", path + ".owner"),
                            target_shard)) {
                        ready_and_terminal = false;
                        break;
                    }
                }
                allowed_clear =
                    owner_descriptor != nullptr && !owner_shards.empty()
                    && selected_consumer_proof && ready_and_terminal;
            }
            if (kind == "endpoint_done" && endpoint_name != nullptr
                    && *endpoint_name == "release_ack_recv"
                    && row_string(*old, "role", path) == "source"
                    && selected.has_value()) {
                allowed_clear =
                    row_uint(*old, "rank", path) == selected->run->rank
                    && row_uint(*old, "shard_ordinal", path)
                        == selected->shard.shard_ordinal;
                if (allowed_clear) {
                    const std::uint64_t release_lag =
                        descriptor.protocol.release_lag_tiles;
                    const std::uint64_t released_tile =
                        selected->global_tile_ordinal - release_lag;
                    const DescriptorPtr owner_descriptor =
                        find_descriptor(old_owner_segment);
                    const SelectorIndex* owner_index =
                        selector_index_for(old_owner_segment);
                    const std::uint64_t owner_global_tile =
                        owner_descriptor
                            ? global_tile_ordinal(
                                  *owner_descriptor,
                                  row_uint(old_owner, "group_ordinal",
                                           path + ".owner"),
                                  row_uint(old_owner, "tile_ordinal",
                                           path + ".owner"),
                                  owner_index)
                            : 0;
                    const auto owner_run =
                        owner_index != nullptr
                            ? descriptor_run_at(
                                  *owner_index,
                                  row_uint(old_owner, "group_ordinal",
                                           path + ".owner"),
                                  row_uint(*old, "rank", path),
                                  row_uint(old_owner, "tile_ordinal",
                                           path + ".owner"))
                            : std::nullopt;
                    allowed_clear =
                        released_tile > 0
                        && owner_global_tile == released_tile
                        && owner_run.has_value()
                        && owner_descriptor != nullptr
                        && owner_descriptor->identity.request_id
                            == descriptor.identity.request_id
                        && owner_descriptor->identity.buffer_domain_id
                            == descriptor.identity.buffer_domain_id
                        && owner_descriptor->identity.generation
                            == descriptor.identity.generation
                        && owner_descriptor->identity.stage_epoch
                            == descriptor.identity.stage_epoch
                        && owner_descriptor->identity.staging_allocation_id
                            == descriptor.identity.staging_allocation_id;
                }
            }
            if (!allowed_clear) {
                fail(path + " occupied pipeline slot was cleared before its legal release");
            }
        }
        if (new_row != nullptr && row_bool(*new_row, "occupied", path)) {
            const Json& new_owner = row_field(*new_row, "owner", path);
            const std::string owner_segment =
                row_string(new_owner, "segment_id", path + ".owner");
            const DescriptorPtr owner_descriptor =
                find_descriptor(owner_segment);
            if (kind == "request_close") {
                fail("request_close cannot leave an occupied pipeline slot");
            }
            if (kind != "tile_issue" && kind != "consume_start"
                    && kind != "fill_ack_progress"
                    && kind != "release_ack_progress"
                    && kind != "request_close" && kind != "consume_done") {
                fail("pipeline slot ownership changed outside an issue/consume/close transition");
            }
            if (kind == "request_close"
                    && (!owner_descriptor
                        || owner_descriptor->identity.request_id
                            != descriptor.identity.request_id)) {
                fail("request_close touches a foreign request's buffer domain");
            }
            if (kind != "request_close"
                    && owner_segment != row_string(
                           transition, "descriptor_segment_id",
                           "pipeline run")) {
                fail("pipeline slot issue assigned a different descriptor owner");
            }
            if ((kind == "request_close" || kind == "consume_done")
                    && old != nullptr && row_bool(*old, "occupied", path)
                    && !typed_equal(row_field(*old, "owner", path), new_owner)) {
                fail("administrative close/consume cannot transfer slot ownership");
            }
            if (selected.has_value()) {
                const Json& selector =
                    row_field(transition, "endpoint_selector", "pipeline run");
                if (row_uint(new_owner, "tile_ordinal", path + ".owner")
                        != row_uint(selector, "tile_ordinal",
                                    "pipeline run.endpoint_selector")) {
                    fail("pipeline slot owner tile differs from selected endpoint");
                }
                if (row_uint(new_owner, "group_ordinal", path + ".owner")
                        != selected->run->group_ordinal) {
                    fail("pipeline slot owner group differs from selected endpoint");
                }
                if (row_string(*new_row, "role", path) == "source"
                        && (row_uint(*new_row, "rank", path)
                                != selected->run->rank
                            || row_uint(*new_row, "shard_ordinal", path)
                                != selected->shard.shard_ordinal)) {
                    fail("source slot owner differs from selected endpoint rank/shard");
                }
                if (row_string(*new_row, "role", path) == "target"
                        && row_uint(*new_row, "rank", path)
                            != selected->shard.target_rank) {
                    fail("target slot owner differs from selected endpoint rank");
                }
            }
        }
        if (old != nullptr && row_bool(*old, "occupied", path)
                && new_row != nullptr && row_bool(*new_row, "occupied", path)) {
            const Json& old_owner = row_field(*old, "owner", path);
            const Json& new_owner = row_field(*new_row, "owner", path);
            const std::string old_segment =
                row_string(old_owner, "segment_id", path + ".owner");
            const std::string new_segment =
                row_string(new_owner, "segment_id", path + ".owner");
            if (old_segment != new_segment) {
                const Json* old_state = nullptr;
                const auto touched = touched_state_.find(old_segment);
                if (touched != touched_state_.end()) {
                    old_state = &touched->second;
                } else {
                    old_state = states_get(old_segment);
                }
                if (old_state == nullptr
                        || !(state_flag(*old_state, "settle_done")
                             || state_flag(*old_state, "retired"))) {
                    fail("foreign slot owner was displaced before settle");
                }
            } else {
                const DescriptorPtr owner_descriptor =
                    find_descriptor(old_segment);
                const SelectorIndex* owner_index =
                    selector_index_for(old_segment);
                if (!owner_descriptor) {
                    fail("slot owner references an unknown descriptor");
                }
                const std::uint64_t old_global = global_tile_ordinal(
                    *owner_descriptor,
                    row_uint(old_owner, "group_ordinal", path + ".owner"),
                    row_uint(old_owner, "tile_ordinal", path + ".owner"),
                    owner_index);
                const std::uint64_t new_global = global_tile_ordinal(
                    *owner_descriptor,
                    row_uint(new_owner, "group_ordinal", path + ".owner"),
                    row_uint(new_owner, "tile_ordinal", path + ".owner"),
                    owner_index);
                if (old_global != new_global) {
                    // In-domain two-slot reuse requires a completed lagged
                    // ReleaseAckRecv watermark for this slot.
                    if (row_string(*old, "role", path) != "source"
                            || !selected.has_value()
                            || (*endpoint_name != "data_send"
                                && *endpoint_name != "fill_ack_recv")) {
                        fail("slot owner changed before a matching release completion");
                    }
                    const ReleaseKey watermark_key{
                        row_string(*old, "buffer_domain_id", path),
                        row_uint(*old, "rank", path),
                        row_uint(*old, "shard_ordinal", path),
                        row_uint(*old, "slot_index", path)};
                    if (old_global
                            > release_watermark_get(watermark_key)) {
                        fail("source slot was reused before its release watermark");
                    }
                }
            }
        }
        if (old != nullptr) add_row_aggregate(*old, -1);
        if (key.role == "target") {
            const OwnerTag old_tag =
                old != nullptr && row_bool(*old, "occupied", path)
                    ? owner_tag_of(row_field(*old, "owner", path),
                                   path + ".owner")
                    : OwnerTag{};
            const bool have_old_tag =
                old != nullptr && row_bool(*old, "occupied", path);
            const bool new_occupied =
                new_row != nullptr && row_bool(*new_row, "occupied", path);
            const OwnerTag new_tag =
                new_occupied
                    ? owner_tag_of(row_field(*new_row, "owner", path),
                                   path + ".owner")
                    : OwnerTag{};
            if (have_old_tag && (!new_occupied || !(old_tag == new_tag))) {
                for (const std::uint64_t target_shard :
                     row_uint_array(*old, "target_shards_to_consume", path)) {
                    target_ready_overlay_.del(
                        ReadyKey{key.domain, key.rank, key.slot_index,
                                 target_shard});
                }
            }
        }
        if (new_row != nullptr) {
            slot_overlay_.set(key, *new_row);
            add_row_aggregate(*new_row, 1);
        } else {
            slot_overlay_.del(key);
        }
        if (key.role == "source") {
            touched_keys.insert(
                CreditKey{key.domain, key.rank,
                          static_cast<std::uint64_t>(key.shard)});
        } else {
            touched_targets.insert({key.domain, key.rank});
        }
    }
    for (const Json* delta : target_deltas) {
        const std::set<TargetDomainKey> touched = apply_target_delta(
            transition, descriptor, selected, *delta);
        touched_targets.insert(touched.begin(), touched.end());
    }
    return {touched_keys, touched_targets};
}

// apply_credit_patches (pipeline_ledger.py 3723-3766).
std::set<CreditKey> StageDelivery::apply_credit_patches(
    const Json& transition, const PipelineDescriptor& descriptor,
    const std::pair<std::optional<CreditKey>, std::int64_t>& expected_ack) {
    std::set<CreditKey> touched;
    const std::string path = "credit patch";
    const Json& patches = row_field(transition, "credit_patches", "pipeline run");
    if (!patches.is_array()) {
        fail("pipeline run credit_patches must be an array");
    }
    for (const Json& patch : patches) {
        if (!patch.is_array() || patch.size() != 3 || !patch[0].is_array()
                || patch[0].size() != 3) {
            fail(path + " must be [key, before, after]");
        }
        const CreditKey key{read_string(patch[0][0], path + ".key[0]"),
                            read_rank(patch[0][1], path + ".key[1]"),
                            read_uint(patch[0][2], path + ".key[2]")};
        const bool before_null = patch[1].is_null();
        const bool after_null = patch[2].is_null();
        const bool present = credit_overlay_.has(key);
        const Json* current = credit_overlay_.get_raw(key);
        if (before_null) {
            if (present) fail(path + " expected an absent before row");
        } else if (!present || !typed_equal(*current, patch[1])) {
            fail(path + " before row differs from accepted COW state");
        }
        const Json* old = present ? current : nullptr;
        const Json* new_row = after_null ? nullptr : &patch[2];
        if (new_row != nullptr
                && (row_string(*new_row, "buffer_domain_id", path) != key.domain
                    || row_uint(*new_row, "source_rank", path) != key.source_rank
                    || row_uint(*new_row, "shard_ordinal", path) != key.shard)) {
            fail(path + " changes its stable key");
        }
        if (key.domain != descriptor.identity.buffer_domain_id) {
            fail(path + " crosses the transition buffer domain");
        }
        const std::int64_t old_ack =
            old != nullptr
                ? static_cast<std::int64_t>(
                      row_uint(*old, "outstanding_ack_count", path))
                : 0;
        const std::int64_t new_ack =
            new_row != nullptr
                ? static_cast<std::int64_t>(
                      row_uint(*new_row, "outstanding_ack_count", path))
                : 0;
        const std::int64_t expected_delta =
            (expected_ack.first && *expected_ack.first == key)
                ? expected_ack.second : 0;
        if (new_ack != old_ack + expected_delta) {
            fail(path + " ACK count differs from its sequenced send/completion");
        }
        if (old != nullptr && new_row == nullptr
                && (old_ack != 0 || source_occupancy_.get(key) != 0)) {
            fail(path + " tombstone retains occupied slots or ACKs");
        }
        if (old != nullptr) {
            credit_ack_domains_.add(key.domain, -old_ack);
            credit_row_counts_.add(key.domain, -1);
        }
        if (new_row != nullptr) {
            credit_overlay_.set(key, *new_row);
            credit_ack_domains_.add(key.domain, new_ack);
            credit_row_counts_.add(key.domain, 1);
        } else {
            credit_overlay_.del(key);
        }
        touched.insert(key);
    }
    if (expected_ack.second != 0
            && (!expected_ack.first
                || !touched.count(*expected_ack.first))) {
        fail(path + " release ACK count changed without a matching credit row patch");
    }
    return touched;
}

// apply_obligation_patches (pipeline_ledger.py 3768-3826).  The Python
// post-write second pass exists only for custom Mapping/string subtypes;
// plain JSON rows are value-equal after the first comparison, so it is a
// structural no-op here.
void StageDelivery::apply_obligation_patches(
    const Json& transition,
    const std::map<std::string, std::optional<Json>>& required_expected) {
    std::set<std::string> seen;
    const std::string path = "obligation patch";
    const Json& patches =
        row_field(transition, "obligation_patches", "pipeline run");
    if (!patches.is_array()) {
        fail("pipeline run obligation_patches must be an array");
    }
    for (const Json& patch : patches) {
        if (!patch.is_array() || patch.size() != 3) {
            fail(path + " must be [obligation_id, before, after]");
        }
        const std::string key = read_string(patch[0], path + "[0]");
        const bool before_null = patch[1].is_null();
        const bool after_null = patch[2].is_null();
        const bool present = obligation_overlay_.has(key);
        const Json* current = obligation_overlay_.get_raw(key);
        if (before_null) {
            if (present) fail(path + " expected an absent before row");
        } else if (!present || !typed_equal(*current, patch[1])) {
            fail(path + " before row differs from accepted COW state");
        }
        const Json* old = present ? current : nullptr;
        const Json* new_row = after_null ? nullptr : &patch[2];
        if (new_row != nullptr
                && row_string(*new_row, "obligation_id", path) != key) {
            fail(path + " changes its stable obligation_id");
        }
        if (old != nullptr) {
            obligation_remaining_.add(
                row_string(*old, "segment_id", path),
                -static_cast<std::int64_t>(
                    row_uint(*old, "remaining_endpoint_count", path)));
        }
        if (new_row != nullptr) {
            obligation_overlay_.set(key, *new_row);
            obligation_remaining_.add(
                row_string(*new_row, "segment_id", path),
                static_cast<std::int64_t>(
                    row_uint(*new_row, "remaining_endpoint_count", path)));
        } else {
            obligation_overlay_.del(key);
        }
        seen.insert(key);
        const auto expected = required_expected.find(key);
        const Json* expected_row =
            expected != required_expected.end() && expected->second
                ? &*expected->second : nullptr;
        const bool matches =
            (new_row == nullptr) == (expected_row == nullptr)
            && (new_row == nullptr
                || typed_equal(*new_row, *expected_row));
        if (!matches) {
            fail(path + " does not match endpoint completion/close progress");
        }
    }
    if (seen.size() != required_expected.size()) {
        fail(path + " set does not match endpoint progress");
    }
}

// simulate_issue_or_completion (pipeline_ledger.py 3852-3889).
std::optional<std::string> StageDelivery::simulate_issue_or_completion(
    const Json& run_transition, CursorProgress& progress,
    const std::string& segment, std::uint64_t run_index) {
    const std::string kind =
        row_string(run_transition, "transition_kind", "pipeline run");
    const auto store_id =
        row_opt_uint(run_transition, "endpoint_store_id", "pipeline run");
    const std::uint64_t ordinal =
        row_uint(run_transition, "endpoint_ordinal", "pipeline run");
    if (kind == "endpoint_done"
            || (kind == "tile_issue" && store_id
                && progress.active.count(*store_id))) {
        if (!store_id || !progress.active.count(*store_id)) {
            fail("endpoint completion has no active issue");
        }
        const auto active_it = progress.active.find(*store_id);
        const std::uint64_t active_ordinal = active_it->second.first;
        progress.active.erase(active_it);
        if (active_ordinal != ordinal) {
            fail("endpoint completion selector differs from active issue");
        }
        merge_completion(progress, ordinal, "endpoint_done", segment,
                         run_index);
        return "complete";
    }
    if (kind != "tile_issue" && kind != "consume_start"
            && kind != "fill_ack_progress" && kind != "release_ack_progress") {
        return std::nullopt;
    }
    if (progress.closed) {
        fail("request_close forbids new endpoint issue");
    }
    if (ordinal != progress.position) {
        fail("endpoint issue does not match the next legal cursor position");
    }
    if (!store_id) {
        fail("endpoint issue lacks its store id");
    }
    if (progress.active.count(*store_id)) {
        fail("endpoint issue duplicates an active store ID");
    }
    progress.active[*store_id] = {ordinal, std::nullopt};
    progress.position += 1;
    if (progress.position > progress.total) {
        fail("endpoint cursor advanced beyond its run");
    }
    cursor_dirty_.insert({segment, run_index});
    cursor_unissued_.add(segment, -1);
    advance_inactive_positions(*progress.descriptor, progress, segment,
                               run_index, run_transition);
    return "issue";
}

// The pipeline_transition_runs loop body (pipeline_ledger.py 3891-4221).
void StageDelivery::run_transition_row(const Json& transition) {
    const std::string path = "pipeline run";
    const std::uint64_t sequence =
        row_uint(transition, "first_logical_seq", path);
    stage_node_events_before(sequence);
    const std::string segment =
        row_string(transition, "descriptor_segment_id", path);
    const DescriptorPtr descriptor = find_descriptor(segment);
    if (!descriptor) {
        fail(path + " references unknown descriptor");
    }
    Json& state = current_state(segment);
    const std::string kind = row_string(transition, "transition_kind", path);
    const std::string domain = descriptor->identity.buffer_domain_id;
    descriptor_transition_counts_[segment] += 1;
    if (state_flag(state, "retired")) {
        fail("retired descriptor received a later transition");
    }
    if (kind == "descriptor_commit") {
        if (!additions_.count(segment) || descriptor_commits_.count(segment)) {
            fail("descriptor_commit must declare exactly one new descriptor");
        }
        descriptor_commits_.insert(segment);
        if (row_uint(transition, "cursor_version_start", path)
                != state_uint(state, "cursor_version")) {
            fail("descriptor_commit cursor pre-version disagrees with declaration");
        }
        const Json& entry = initial_state_updates_[segment];
        const bool mismatch =
            state_flag(state, "service_done")
                != read_bool(row_field(entry, "service_done", "descriptor commit"),
                             "descriptor commit.service_done")
            || state_flag(state, "settle_done")
                != read_bool(row_field(entry, "settle_done", "descriptor commit"),
                             "descriptor commit.settle_done")
            || state_flag(state, "retired")
                != read_bool(row_field(entry, "retired", "descriptor commit"),
                             "descriptor commit.retired");
        if (mismatch) {
            fail("descriptor_commit metadata disagrees with declaration");
        }
        active_domain_counts_.add(domain, 1);
        for (std::uint64_t initial_run_index = 0;
             initial_run_index < descriptor->endpoint_runs.size();
             ++initial_run_index) {
            CursorProgress& initial_progress =
                seed_cursor_progress(segment, initial_run_index);
            cursor_unissued_.add(
                segment,
                static_cast<std::int64_t>(initial_progress.total)
                    - static_cast<std::int64_t>(initial_progress.position));
            const auto initial_obligation = expected_obligation(
                *descriptor, segment, initial_run_index, initial_progress);
            const std::string obligation_id =
                segment + "/run/" + num(initial_run_index);
            expected_obligation_rows_[obligation_id] =
                initial_obligation ? std::optional<Json>(*initial_obligation)
                                   : std::nullopt;
            active_endpoint_deltas_[segment] +=
                static_cast<std::int64_t>(initial_progress.active.size());
        }
    } else if (!st_.active_segments.count(segment)
               && !descriptor_commits_.count(segment)) {
        fail("pipeline transition precedes descriptor_commit");
    }
    const std::uint64_t current_version =
        cursor_versions_.count(segment)
            ? cursor_versions_[segment]
            : state_uint(state, "cursor_version");
    if (row_uint(transition, "cursor_version_start", path) != current_version) {
        fail("pipeline transition cursor pre-version mismatch");
    }
    const std::uint64_t next_version =
        row_uint(transition, "cursor_version_end", path);
    cursor_versions_[segment] = next_version;
    state["cursor_version"] = next_version;

    const std::optional<SelectedEndpoint> selected =
        selected_slot_and_credit(*descriptor, transition);
    const auto run_index =
        row_opt_uint(transition, "endpoint_run_index", path);
    CursorProgress* progress = nullptr;
    if (run_index) {
        progress = &seed_cursor_progress(segment, *run_index);
        cursor_dirty_.insert({segment, *run_index});
    }

    std::optional<std::string> progress_action;
    if (kind == "request_close") {
        if (state_flag(state, "request_closed")) {
            fail("request_close transition repeated");
        }
        for (std::uint64_t index = 0;
             index < descriptor->endpoint_runs.size(); ++index) {
            CursorProgress& run_progress = seed_cursor_progress(segment, index);
            if (run_progress.closed) {
                fail("request_close found a previously closed cursor run");
            }
            const std::int64_t unresolved =
                static_cast<std::int64_t>(run_progress.total)
                - static_cast<std::int64_t>(run_progress.completed_total)
                - static_cast<std::int64_t>(run_progress.cancelled);
            const std::int64_t active_count =
                static_cast<std::int64_t>(run_progress.active.size());
            if (unresolved < active_count) {
                fail("request_close active endpoints exceed unresolved work");
            }
            cursor_unissued_.add(
                segment,
                -(static_cast<std::int64_t>(run_progress.total)
                  - static_cast<std::int64_t>(run_progress.position)));
            run_progress.cancelled = static_cast<std::uint64_t>(
                static_cast<std::int64_t>(run_progress.cancelled) + unresolved
                - active_count);
            run_progress.remaining = static_cast<std::uint64_t>(active_count);
            run_progress.position = run_progress.total;
            run_progress.closed = true;
            cursor_dirty_.insert({segment, index});
            const auto old =
                expected_obligation(*descriptor, segment, index, run_progress);
            const std::string key = segment + "/run/" + num(index);
            expected_obligation_rows_[key] =
                old ? std::optional<Json>(*old) : std::nullopt;
        }
        state["request_closed"] = true;
        state["service_done"] = true;
    } else if (progress != nullptr) {
        progress_action = simulate_issue_or_completion(
            transition, *progress, segment, *run_index);
        if (progress_action) {
            const auto new_expected = expected_obligation(
                *descriptor, segment, *run_index, *progress);
            const std::string key = segment + "/run/" + num(*run_index);
            expected_obligation_rows_[key] =
                new_expected ? std::optional<Json>(*new_expected)
                             : std::nullopt;
            if (*progress_action == "complete") {
                // endpoint_done carries the concrete descriptor/run/selector
                // and affine store id; it is terminal proof for this exact
                // logical endpoint (request_close cancellation and cursor
                // prefix normalization never enter this path).
                const PipelineEndpointRun& completed_run =
                    descriptor->endpoint_runs[*run_index];
                const auto store_id =
                    row_opt_uint(transition, "endpoint_store_id", path);
                if (!store_id) {
                    fail("endpoint completion has no active issue");
                }
                const std::set<std::string> terminal_segments{segment};
                record_dependency_terminal(completed_run.rank, *store_id,
                                           &terminal_segments);
            }
        }
    }

    if (kind == "data_ready") {
        if (!selected.has_value() || selected->endpoint != "data_recv") {
            fail("data_ready must bind a concrete data_recv endpoint");
        }
        const std::uint64_t ordinal =
            row_uint(transition, "endpoint_ordinal", path);
        if (progress == nullptr
                || !progress_has_completed(*progress, ordinal)) {
            fail("data_ready precedes data_recv endpoint completion");
        }
        const Json& selector =
            row_field(transition, "endpoint_selector", path);
        const std::uint64_t tile_ordinal =
            row_uint(selector, "tile_ordinal", path);
        const std::uint64_t slot_index =
            (selected->global_tile_ordinal - 1) % 2;
        const SlotKey slot_key{domain, "target", selected->run->rank, -1,
                               slot_index};
        const ReadyKey ready_slot_key{domain, selected->run->rank, slot_index,
                                      selected->shard.shard_ordinal};
        const Json* target_row = current_target_row(slot_key);
        const Json* owner =
            target_row != nullptr
                ? &row_field(*target_row, "owner", "slot row") : nullptr;
        if (owner != nullptr && owner->is_null()) owner = nullptr;
        const OwnerTag pending_tag{
            segment, descriptor->identity.generation,
            descriptor->identity.stage_epoch,
            descriptor->identity.staging_allocation_id,
            selected->run->group_ordinal, tile_ordinal};
        const std::pair<ReadyKey, OwnerTag> pending_key{ready_slot_key,
                                                        pending_tag};
        if (state_flag(state, "request_closed")
                && (target_row == nullptr
                    || !row_bool(*target_row, "occupied", "slot row"))) {
            if (target_ready_overlay_.get_raw(ready_slot_key) != nullptr) {
                fail("closed descriptor retains a ready target slot");
            }
            pending_ready_after_clear_.erase(pending_key);
        } else if (target_row != nullptr
                   && row_bool(*target_row, "occupied", "slot row")
                   && row_field(*target_row, "shard_ordinal", "slot row")
                          .is_null()
                   && sorted_uint_array_contains(
                          *target_row, "target_shards_to_consume", "slot row",
                          selected->shard.shard_ordinal)
                   && owner != nullptr
                   && row_string(*owner, "segment_id", "slot row.owner")
                          == segment
                   && row_uint(*owner, "group_ordinal", "slot row.owner")
                          == selected->run->group_ordinal
                   && row_uint(*owner, "tile_ordinal", "slot row.owner")
                          == tile_ordinal
                   && row_uint(*owner, "pipeline_generation", "slot row.owner")
                          == descriptor->identity.generation
                   && row_uint(*owner, "stage_epoch", "slot row.owner")
                          == descriptor->identity.stage_epoch
                   && row_uint(*owner, "staging_allocation_id",
                               "slot row.owner")
                          == descriptor->identity.staging_allocation_id) {
            const OwnerTag tag = owner_tag_of(*owner, "slot row.owner");
            const OwnerTag* ready_owner =
                target_ready_overlay_.get_raw(ready_slot_key);
            if (ready_owner != nullptr && !(*ready_owner == tag)) {
                fail("ready target slot belongs to a different owner");
            }
            if (ready_owner != nullptr && *ready_owner == tag) {
                fail("target shard received duplicate data_ready");
            }
            target_ready_overlay_.set(ready_slot_key, tag);
        } else if (pending_ready_after_clear_.count(pending_key) != 0) {
            pending_ready_after_clear_.erase(pending_key);
        } else {
            fail("data_ready owner mismatch at seq=" + num(sequence));
        }
    } else if (kind == "service_done") {
        if (state_flag(state, "service_done")) {
            fail("service_done transition repeated");
        }
        // Service completion closes issue production; submitted legs still
        // drain until settle_done.
        if (!state_flag(state, "request_closed")
                && cursor_unissued_.get(segment) != 0) {
            fail("service_done precedes cursor issue exhaustion");
        }
        state["service_done"] = true;
    } else if (kind == "settle_done") {
        if (state_flag(state, "settle_done")) {
            fail("settle_done transition repeated");
        }
        if (!state_flag(state, "service_done")) {
            fail("settle_done precedes service_done/request_close");
        }
        if (obligation_remaining_.get(segment) != 0) {
            fail("settle_done retains endpoint obligations");
        }
        const auto delta = active_endpoint_deltas_.find(segment);
        const std::int64_t staged_active =
            delta != active_endpoint_deltas_.end() ? delta->second : 0;
        const auto persisted = st_.active_endpoint_counts.find(segment);
        const std::int64_t persisted_active =
            persisted != st_.active_endpoint_counts.end()
                ? persisted->second : 0;
        if (persisted_active + staged_active != 0) {
            fail("settle_done retains active physical endpoints");
        }
        state["settle_done"] = true;
    } else if (kind == "descriptor_retire") {
        if (!state_flag(state, "service_done")
                || !state_flag(state, "settle_done")) {
            fail("descriptor retirement precedes service and settle");
        }
        if (obligation_remaining_.get(segment) != 0) {
            fail("descriptor retirement retains endpoint obligations");
        }
        if (slot_owner_counts_.get(segment) != 0) {
            fail("descriptor retirement retains a slot owner");
        }
        if (ack_pins_by_segment_.get(segment) != 0) {
            fail("descriptor retirement retains an owned ACK pin");
        }
        const auto delta = active_endpoint_deltas_.find(segment);
        const std::int64_t staged_active =
            delta != active_endpoint_deltas_.end() ? delta->second : 0;
        const auto persisted = st_.active_endpoint_counts.find(segment);
        const std::int64_t persisted_active =
            persisted != st_.active_endpoint_counts.end()
                ? persisted->second : 0;
        if (persisted_active + staged_active != 0) {
            fail("descriptor retirement retains an active endpoint owner");
        }
        const auto resources = resource_counts_.find(segment);
        if (resources != resource_counts_.end()
                && resources->second != 0) {
            fail("descriptor retirement retains an active resource");
        }
        if (dependency_pins_.get(segment) != 0) {
            fail("descriptor retirement retains an unfinished ordinary dependency pin");
        }
        active_domain_counts_.add(domain, -1);
        if (active_domain_counts_.get(domain) < 0) {
            fail("active descriptor buffer-domain count underflow");
        }
        if (active_domain_counts_.get(domain) == 0) {
            domain_retirements_.insert(domain);
        }
        state["retired"] = true;
        retirements_.push_back(segment);
    } else if (kind == "descriptor_commit") {
        // Handled above.
    }

    // Release ACK pins change at the actual completion callback, which
    // precedes endpoint_done in the producer's logical stream.
    std::optional<CreditKey> ack_key;
    std::int64_t ack_delta = 0;
    bool ack_noop_closed_recv = false;
    if (selected.has_value()) {
        const CreditKey key{domain, selected->shard.source_rank,
                            selected->shard.shard_ordinal};
        if (kind == "release_ack_progress"
                && selected->endpoint == "release_ack_send") {
            ack_key = key;
            ack_delta = 1;
        } else if (kind == "endpoint_done"
                && selected->endpoint == "release_ack_recv") {
            const Json* row = credit_overlay_.get_raw(key);
            const std::int64_t current_ack =
                row != nullptr
                    ? static_cast<std::int64_t>(
                          row_uint(*row, "outstanding_ack_count", "credit row"))
                    : 0;
            if (current_ack == 0) {
                if (!state_flag(state, "request_closed")) {
                    fail("release ACK receive lacks a pending source ACK pin");
                }
                ack_noop_closed_recv = true;
            } else {
                ack_key = key;
                ack_delta = -1;
            }
        }
    }
    const auto slot_sets = apply_slot_patches(transition, *descriptor, selected);
    std::set<CreditKey> credit_keys =
        apply_credit_patches(transition, *descriptor, {ack_key, ack_delta});
    for (const CreditKey& key : slot_sets.first) {
        credit_keys.insert(key);
    }
    if (ack_delta != 0) {
        credit_keys.insert(*ack_key);
        ack_pins_by_segment_.add(segment, ack_delta);
        if (ack_pins_by_segment_.get(segment) < 0) {
            fail("segment release ACK pin underflow");
        }
    }
    verify_credit_balance(credit_keys, slot_sets.second);
    if (ack_noop_closed_recv && ack_key.has_value()) {
        fail("closed no-op ACK receive unexpectedly changed a credit row");
    }

    if (selected.has_value()) {
        if (kind == "endpoint_done"
                && selected->endpoint == "release_ack_recv") {
            const std::uint64_t lag =
                descriptor->protocol.release_lag_tiles;
            if (selected->global_tile_ordinal > lag) {
                const std::uint64_t released_tile =
                    selected->global_tile_ordinal - lag;
                const std::uint64_t slot_index = (released_tile - 1) % 2;
                const ReleaseKey watermark_key{
                    domain, selected->run->rank,
                    selected->shard.shard_ordinal, slot_index};
                if (released_tile
                        > release_watermark_get(watermark_key)) {
                    release_watermark_add(
                        watermark_key,
                        static_cast<std::int64_t>(released_tile)
                            - static_cast<std::int64_t>(
                                  release_watermark_get(watermark_key)));
                }
            }
        }
    }

    if (kind == "descriptor_retire" && domain_retirements_.count(domain)) {
        if (slot_row_counts_.get(domain) != 0
                || credit_row_counts_.get(domain) != 0
                || credit_ack_domains_.get(domain) != 0) {
            fail("last descriptor retirement retains buffer-domain rows or ACK pins");
        }
    }

    std::map<std::string, std::optional<Json>> changed_expected;
    for (const auto& expected : expected_obligation_rows_) {
        const Json* current = obligation_overlay_.get_raw(expected.first);
        const Json* expected_row =
            expected.second ? &*expected.second : nullptr;
        const bool matches =
            (current == nullptr) == (expected_row == nullptr)
            && (current == nullptr
                || typed_equal(*current, *expected_row));
        if (!matches) changed_expected[expected.first] = expected.second;
    }
    apply_obligation_patches(transition, changed_expected);
    expected_obligation_rows_.clear();

    if (selected.has_value() && run_index.has_value()) {
        std::int64_t current = 0;
        const auto delta = active_endpoint_deltas_.find(segment);
        if (delta != active_endpoint_deltas_.end()) current = delta->second;
        if (progress_action && *progress_action == "issue") {
            current += 1;
        } else if (progress_action && *progress_action == "complete") {
            current -= 1;
        }
        active_endpoint_deltas_[segment] = current;
    }
}

// The post-loop validation and update assembly (pipeline_ledger.py 4223-4524).
void StageDelivery::finalize() {
    if (descriptor_commits_.size() != additions_.size()) {
        fail("every descriptor declaration needs exactly one descriptor_commit transition");
    }
    std::set<std::string> retired_segment_set(retirements_.begin(),
                                              retirements_.end());
    std::set<std::string> required_cursor_segments;
    for (const auto& addition : additions_) {
        required_cursor_segments.insert(addition.first);
    }
    for (const std::string& segment : retirements_) {
        required_cursor_segments.insert(segment);
    }
    for (const CursorRunKey& key : cursor_dirty_) {
        required_cursor_segments.insert(key.first);
    }
    for (const auto& version : cursor_versions_) {
        std::optional<std::uint64_t> base;
        const auto initial = initial_state_updates_.find(version.first);
        if (initial != initial_state_updates_.end()) {
            base = state_uint(initial->second, "cursor_version");
        } else {
            const auto persisted =
                st_.cursor_versions_by_segment.find(version.first);
            if (persisted != st_.cursor_versions_by_segment.end()) {
                base = persisted->second;
            }
        }
        if (!base || *base != version.second) {
            required_cursor_segments.insert(version.first);
        }
    }
    bool coverage_matches =
        required_cursor_segments.size() == parsed_cursor_patches_.size();
    if (coverage_matches) {
        for (const std::string& segment : required_cursor_segments) {
            if (!parsed_cursor_patches_.count(segment)) {
                coverage_matches = false;
                break;
            }
        }
    }
    if (!coverage_matches) {
        fail("cursor patches do not exactly cover initialized, retired, or changed cursors");
    }
    for (const std::string& segment : retirements_) {
        const auto patch = parsed_cursor_patches_.find(segment);
        if (patch == parsed_cursor_patches_.end()
                || !row_bool(*patch->second, "retired", "cursor patch")) {
            fail("descriptor retirement needs explicit cursor run tombstones");
        }
    }
    for (const auto& addition : additions_) {
        const std::string& segment = addition.first;
        const auto patch = parsed_cursor_patches_.find(segment);
        if (patch == parsed_cursor_patches_.end()) {
            fail("new descriptor requires initial cursor run rows");
        }
        const Json& runs = row_field(*patch->second, "runs", "cursor patch");
        if (runs.size() != addition.second->endpoint_runs.size()) {
            fail("new descriptor cursor patch must initialize every run");
        }
        for (std::size_t index = 0; index < runs.size(); ++index) {
            if (!runs[index].is_array() || runs[index].size() != 2
                    || read_uint(runs[index][0], "cursor patch run index")
                           != index) {
                fail("new descriptor cursor patch must initialize every run");
            }
        }
        if (row_bool(*patch->second, "retired", "cursor patch")) {
            fail("new descriptor cannot start retired");
        }
    }

    for (const Json& patch : parsed_.cursor_patches) {
        const std::string segment =
            row_string(patch, "segment_id", "cursor patch");
        const DescriptorPtr descriptor = find_descriptor(segment);
        if (!descriptor) {
            fail("cursor patch references unknown descriptor");
        }
        Json& state = current_state(segment);
        if (!descriptor_transition_counts_.count(segment)) {
            fail("cursor patch has no sequenced descriptor transition");
        }
        const std::uint64_t patch_version =
            row_uint(patch, "cursor_version", "cursor patch");
        const std::uint64_t current_version =
            cursor_versions_.count(segment)
                ? cursor_versions_[segment]
                : state_uint(state, "cursor_version");
        if (patch_version != current_version) {
            fail("final cursor patch version disagrees with transition sequence");
        }
        if (row_bool(patch, "retired", "cursor patch")
                != state_flag(state, "retired")) {
            fail("cursor retirement marker disagrees with descriptor lifecycle");
        }
        cursor_version_updates_[segment] = patch_version;
        std::optional<std::uint64_t> base_cursor_version;
        {
            const auto initial = initial_state_updates_.find(segment);
            if (initial != initial_state_updates_.end()) {
                base_cursor_version =
                    state_uint(initial->second, "cursor_version");
            } else {
                const auto persisted =
                    st_.cursor_versions_by_segment.find(segment);
                if (persisted != st_.cursor_versions_by_segment.end()) {
                    base_cursor_version = persisted->second;
                }
            }
        }
        const Json& runs = row_field(patch, "runs", "cursor patch");
        if (runs.empty() && !additions_.count(segment) && base_cursor_version
                && patch_version == *base_cursor_version) {
            fail("metadata-only cursor patch has no sequenced version change");
        }
        if (row_bool(patch, "retired", "cursor patch")) {
            if (!retired_segment_set.count(segment)) {
                fail("cursor tombstones require descriptor_retire transition");
            }
            for (const Json& item : runs) {
                const std::uint64_t run_index =
                    read_uint(item[0], "cursor patch run index");
                cursor_changes_[{segment, run_index}] = std::nullopt;
                cursor_transfer_changes_[{segment, run_index}] = std::nullopt;
                const Json* old_row = nullptr;
                Json initial_storage;
                if (additions_.count(segment)) {
                    initial_storage =
                        initial_cursor_run(*descriptor, run_index);
                    old_row = &initial_storage;
                } else {
                    const auto runs_map =
                        st_.cursor_runs_by_segment.find(segment);
                    if (runs_map != st_.cursor_runs_by_segment.end()) {
                        const auto found = runs_map->second.find(run_index);
                        if (found != runs_map->second.end()) {
                            old_row = &found->second;
                        }
                    }
                }
                if (old_row == nullptr) {
                    fail("retirement tombstones a missing accepted cursor run");
                }
                const Json& old_active =
                    row_field(*old_row, "active_endpoints", "cursor run");
                if (!old_active.empty()) {
                    cursor_active_deltas_[segment] -=
                        static_cast<std::int64_t>(old_active.size());
                }
                for (const Json& endpoint : old_active) {
                    const std::uint64_t transfer_id = row_uint(
                        endpoint, "transfer_id", "cursor run.active_endpoints");
                    active_transfer_id_counts_.add(transfer_id, -1);
                    if (active_transfer_id_counts_.get(transfer_id) < 0) {
                        fail("active transfer index underflow during retirement");
                    }
                }
            }
            continue;
        }
        for (const Json& item : runs) {
            const std::uint64_t run_index =
                read_uint(item[0], "cursor patch run index");
            const Json& final_row = item[1];
            const CursorRunKey key{segment, run_index};
            CursorProgress& progress = seed_cursor_progress(segment, run_index);
            if (!cursor_dirty_.count(key) && !additions_.count(segment)) {
                fail("cursor run patch has no matching issue/completion/close transition");
            }
            if (final_row.is_null()) {
                fail("nonretired cursor patch cannot tombstone a run");
            }
            const Json* current_row = nullptr;
            if (!additions_.count(segment)) {
                const auto runs_map =
                    st_.cursor_runs_by_segment.find(segment);
                if (runs_map != st_.cursor_runs_by_segment.end()) {
                    const auto found = runs_map->second.find(run_index);
                    if (found != runs_map->second.end()) {
                        current_row = &found->second;
                    }
                }
            }
            if (current_row != nullptr
                    && typed_equal(*current_row, final_row)) {
                fail("cursor run patch redundantly republishes an unchanged row");
            }
            const std::uint64_t expected_remaining =
                progress.closed
                    ? static_cast<std::uint64_t>(progress.active.size())
                    : progress.total - progress.completed_total
                        - progress.cancelled;
            if (row_uint(final_row, "completed_endpoint_count", "cursor run")
                        != progress.completed
                    || row_uint(final_row, "cancelled_endpoint_count",
                                "cursor run") != progress.cancelled
                    || row_uint(final_row, "remaining_endpoint_count",
                                "cursor run") != expected_remaining
                    || row_bool(final_row, "in_flight", "cursor run")
                        != !progress.active.empty()) {
                fail("cursor run counts disagree with sequenced endpoint progress");
            }
            std::vector<std::pair<std::uint64_t, std::uint64_t>> final_ooo;
            for (const Json& ooo :
                 row_field(final_row, "completed_out_of_order", "cursor run")) {
                final_ooo.emplace_back(
                    row_uint(ooo, "first_ordinal",
                             "cursor run.completed_out_of_order"),
                    row_uint(ooo, "end_ordinal_exclusive",
                             "cursor run.completed_out_of_order"));
            }
            const std::vector<std::pair<std::uint64_t, std::uint64_t>>
                simulated_ooo =
                    progress.ooo_changed
                        ? completed_export_half_open(progress.ooo_root)
                        : progress.ooo_wire_ranges;
            if (final_ooo != simulated_ooo) {
                fail("cursor out-of-order ranges disagree with endpoint completions");
            }
            if (cursor_position_ordinal(final_row, *progress.run)
                    != progress.position) {
                fail("cursor next position disagrees with sequenced endpoint issues");
            }
            if (!row_field(final_row, "next_store_id", "cursor run")
                     .is_null()) {
                const Json& actives =
                    row_field(final_row, "active_endpoints", "cursor run");
                if (actives.is_null()) {
                    fail("cursor has an invalid active endpoint list");
                }
            }
            std::map<std::uint64_t, const Json*> final_active;
            for (const Json& endpoint :
                 row_field(final_row, "active_endpoints", "cursor run")) {
                final_active[row_uint(endpoint, "store_id",
                                      "cursor run.active_endpoints")] =
                    &endpoint;
            }
            {
                bool same = final_active.size() == progress.active.size();
                if (same) {
                    for (const auto& active : progress.active) {
                        if (!final_active.count(active.first)) {
                            same = false;
                            break;
                        }
                    }
                }
                if (!same) {
                    fail("cursor active endpoint set disagrees with sequenced issue/completion events");
                }
            }
            std::map<std::uint64_t, const Json*> old_active_rows;
            if (current_row != nullptr) {
                for (const Json& endpoint :
                     row_field(*current_row, "active_endpoints",
                               "cursor run")) {
                    old_active_rows[row_uint(endpoint, "store_id",
                                             "cursor run.active_endpoints")] =
                        &endpoint;
                }
            }
            for (const auto& active_entry : final_active) {
                const auto old = old_active_rows.find(active_entry.first);
                if (old == old_active_rows.end()) continue;
                const std::string path = "cursor run.active_endpoints[]";
                const Json& active = *active_entry.second;
                const Json& before = *old->second;
                const bool regressed =
                    row_uint(active, "transfer_id", path)
                        != row_uint(before, "transfer_id", path)
                    || !typed_equal(row_field(before, "owner", path),
                                    row_field(active, "owner", path))
                    || row_uint(active, "tile_ordinal", path)
                        != row_uint(before, "tile_ordinal", path)
                    || row_uint(active, "shard_ordinal", path)
                        != row_uint(before, "shard_ordinal", path)
                    || row_string(active, "endpoint", path)
                        != row_string(before, "endpoint", path)
                    || (!row_field(before, "flow_id", path).is_null()
                        && row_field(active, "flow_id", path)
                            != row_field(before, "flow_id", path))
                    || (row_bool(before, "network_done", path)
                        && !row_bool(active, "network_done", path))
                    || (row_bool(before, "hbm_done", path)
                        && !row_bool(active, "hbm_done", path));
                if (regressed) {
                    fail("active physical endpoint identity/progress regressed");
                }
            }
            cursor_changes_[key] = final_row;
            std::map<std::uint64_t, Json> transfer_map;
            for (const auto& active_entry : final_active) {
                transfer_map[active_entry.first] = *active_entry.second;
            }
            cursor_transfer_changes_[key] = std::move(transfer_map);
            const std::int64_t delta =
                static_cast<std::int64_t>(final_active.size())
                - static_cast<std::int64_t>(old_active_rows.size());
            if (delta != 0) {
                cursor_active_deltas_[segment] += delta;
            }
            for (const auto& active_entry : old_active_rows) {
                active_transfer_id_counts_.add(
                    row_uint(*active_entry.second, "transfer_id",
                             "cursor run.active_endpoints"),
                    -1);
            }
            for (const auto& active_entry : final_active) {
                const std::uint64_t transfer_id = row_uint(
                    *active_entry.second, "transfer_id",
                    "cursor run.active_endpoints");
                active_transfer_id_counts_.add(transfer_id, 1);
                if (active_transfer_id_counts_.get(transfer_id) > 1) {
                    fail("active transfer_id is already owned by another cursor");
                }
            }
        }
    }

    for (const auto& delta : active_endpoint_deltas_) {
        const auto cursor = cursor_active_deltas_.find(delta.first);
        const std::int64_t cursor_delta =
            cursor != cursor_active_deltas_.end() ? cursor->second : 0;
        if (delta.second != cursor_delta) {
            fail("active endpoint aggregate differs from final cursor rows");
        }
    }
    for (const auto& cursor : cursor_active_deltas_) {
        const auto delta = active_endpoint_deltas_.find(cursor.first);
        const std::int64_t sequenced =
            delta != active_endpoint_deltas_.end() ? delta->second : 0;
        if (cursor.second != sequenced) {
            fail("final cursor rows contain an unsequenced active endpoint change");
        }
    }
    // Every run touched by a sequenced issue/completion/close must have a
    // corresponding final full-row patch or retirement tombstone.
    for (const CursorRunKey& key : cursor_dirty_) {
        const auto patch = parsed_cursor_patches_.find(key.first);
        const auto indexes =
            parsed_cursor_patch_run_indexes_.find(key.first);
        if (patch == parsed_cursor_patches_.end()
                || indexes == parsed_cursor_patch_run_indexes_.end()
                || !indexes->second.count(key.second)) {
            fail("cursor transition is missing its touched run patch");
        }
    }
    for (const Json& resource : parsed_.resources) {
        if (row_uint(resource, "as_of_tick", "resource row")
                != parsed_.header.tick) {
            fail("active resource frame must match envelope tick");
        }
    }
    for (const auto& delta : target_occupancy_.deltas) {
        const std::int64_t value = target_occupancy_.get(delta.first);
        if (value < 0 || value > 2) {
            fail("target pipeline slot occupancy exceeds two-slot capacity");
        }
    }
    for (const auto& version : cursor_versions_) {
        const Json* state = nullptr;
        const auto touched = touched_state_.find(version.first);
        if (touched != touched_state_.end()) {
            state = &touched->second;
        } else {
            state = states_get(version.first);
        }
        if (state != nullptr
                && state_uint(*state, "cursor_version") != version.second) {
            fail("descriptor state cursor version disagrees with pipeline sequence");
        }
    }

    std::map<SlotKey, std::optional<Json>> slot_changes =
        slot_overlay_.changes();
    for (auto& workset_entry : target_worksets_) {
        const SlotKey& key = workset_entry.first;
        TargetWorkset& workset = workset_entry.second;
        if (!workset.row) {
            slot_changes[key] = std::nullopt;
            continue;
        }
        Json frozen_row = *workset.row;
        Json consumed = Json::array();
        for (const std::uint64_t shard : workset.consumed) {
            consumed.push_back(shard);
        }
        frozen_row["consumed"] = !workset.consumed.empty();
        frozen_row["consumed_target_shards"] = consumed;
        slot_changes[key] = std::move(frozen_row);
    }
    const auto target_ready_changes = target_ready_overlay_.changes();
    const auto credit_changes = credit_overlay_.changes();
    const auto obligation_changes = obligation_overlay_.changes();

    bool resources_equal =
        st_.active_resources.size() == parsed_.resources.size();
    if (resources_equal) {
        for (std::size_t i = 0; i < parsed_.resources.size(); ++i) {
            if (!typed_equal(st_.active_resources[i], parsed_.resources[i])) {
                resources_equal = false;
                break;
            }
        }
    }
    const bool visible_change =
        parsed_.header.event_count != 0
        || !parsed_.descriptors.empty()
        || !slot_changes.empty()
        || !target_ready_changes.empty()
        || !credit_changes.empty()
        || !obligation_changes.empty()
        || !cursor_changes_.empty()
        || !cursor_version_updates_.empty()
        || !resources_equal;
    const std::uint64_t expected_version =
        parsed_.header.base_version + (visible_change ? 1u : 0u);
    if (parsed_.header.result_version != expected_version) {
        fail("result_state_version does not match visible state change");
    }

    std::unordered_map<std::string, Json> state_updates = touched_state_;
    for (const auto& initial : initial_state_updates_) {
        if (!state_updates.count(initial.first)) {
            state_updates[initial.first] = initial.second;
        }
    }
    update_.descriptor_additions = additions_;
    update_.descriptor_state_updates = std::move(state_updates);
    update_.descriptor_retirements = retirements_;
    update_.cursor_changes = std::move(cursor_changes_);
    update_.cursor_version_updates = std::move(cursor_version_updates_);
    update_.cursor_transfer_changes = std::move(cursor_transfer_changes_);
    update_.active_endpoint_deltas = active_endpoint_deltas_;
    update_.resources = parsed_.resources;
    update_.slot_changes = std::move(slot_changes);
    update_.target_slot_lineage_changes = target_slot_lineage_changes_;
    update_.target_ready_changes = std::move(target_ready_changes);
    update_.terminal_provenance_updates = terminal_provenance_roots_;
    update_.credit_changes = std::move(credit_changes);
    update_.obligation_changes = std::move(obligation_changes);
    update_.slot_owner_deltas = slot_owner_counts_.deltas;
    update_.source_occupancy_deltas = source_occupancy_.deltas;
    update_.target_occupancy_deltas = target_occupancy_.deltas;
    update_.credit_ack_domain_deltas = credit_ack_domains_.deltas;
    update_.ack_pin_deltas = ack_pins_by_segment_.deltas;
    update_.active_descriptor_deltas = active_domain_counts_.deltas;
    update_.slot_row_deltas = slot_row_counts_.deltas;
    update_.credit_row_deltas = credit_row_counts_.deltas;
    update_.obligation_remaining_deltas = obligation_remaining_.deltas;
    update_.dependency_pin_deltas = dependency_pins_.deltas;
    update_.dependency_waiter_deltas = dependency_waiters_.deltas;
    update_.cursor_unissued_deltas = cursor_unissued_.deltas;
    update_.active_transfer_deltas = active_transfer_id_counts_.deltas;
    update_.release_watermark_deltas = release_watermark_deltas_;
    update_.dependency_index_additions = dependency_root_changes_;
    update_.dependency_rank_index_additions = dependency_rank_root_changes_;
    update_.changed = visible_change;
}

// _stage_delivery prologue (pipeline_ledger.py 2555-2652) plus the pipeline
// row loop and drain (3891, 4223-4229).
void StageDelivery::run() {
    for (const auto& declaration : parsed_.descriptors) {
        const std::string segment =
            declaration.first->identity.segment_id;
        if (st_.descriptors.count(segment) || additions_.count(segment)) {
            fail("descriptor declaration conflicts with retained descriptor");
        }
        additions_[segment] = declaration.first;
        const Json& entry = declaration.second;
        Json state = Json::object();
        state["commit_tick"] = row_field(entry, "commit_tick",
                                         "descriptor commit");
        state["service_done"] = row_field(entry, "service_done",
                                          "descriptor commit");
        state["settle_done"] = row_field(entry, "settle_done",
                                         "descriptor commit");
        state["retired"] = row_field(entry, "retired", "descriptor commit");
        state["request_closed"] = false;
        state["cursor_version"] = row_field(entry, "cursor_version",
                                            "descriptor commit");
        initial_state_updates_[segment] = std::move(state);
    }
    for (const auto& initial : initial_state_updates_) {
        cursor_versions_[initial.first] =
            state_uint(initial.second, "cursor_version");
    }
    for (const auto& index_addition : parsed_.dependency_index_additions) {
        const std::string& segment = index_addition.first;
        for (const auto& prefix_rows :
             index_addition.second.intervals_by_prefix) {
            DependencyPtr root;
            const auto staged =
                dependency_root_changes_.find(prefix_rows.first);
            if (staged != dependency_root_changes_.end()) {
                root = staged->second;
            } else {
                root = find_dependency_root(prefix_rows.first);
            }
            for (const DependencyRow& interval : prefix_rows.second) {
                root = dependency_interval_insert(root, interval);
                update_.dependency_row_segment_additions[interval] = segment;
            }
            dependency_root_changes_[prefix_rows.first] = root;
        }
        for (const auto& rank_rows :
             index_addition.second.intervals_by_rank) {
            DependencyPtr root;
            const auto staged =
                dependency_rank_root_changes_.find(rank_rows.first);
            if (staged != dependency_rank_root_changes_.end()) {
                root = staged->second;
            } else {
                root = find_dependency_rank_root(rank_rows.first);
            }
            for (const DependencyRow& interval : rank_rows.second) {
                root = dependency_interval_insert(root, interval);
                update_.dependency_row_segment_additions[interval] = segment;
            }
            dependency_rank_root_changes_[rank_rows.first] = root;
        }
    }
    for (const Json& resource : parsed_.resources) {
        const Json& segment_field =
            row_field(resource, "descriptor_segment_id", "resource row");
        if (!segment_field.is_null()) {
            resource_counts_[read_string(
                segment_field, "resource row.descriptor_segment_id")] += 1;
        }
    }
    for (const Json& patch : parsed_.cursor_patches) {
        const std::string segment =
            row_string(patch, "segment_id", "cursor patch");
        parsed_cursor_patches_[segment] = &patch;
        std::set<std::uint64_t> run_indexes;
        const Json& runs = row_field(patch, "runs", "cursor patch");
        if (!runs.is_array()) {
            fail("cursor patch runs must be an array");
        }
        for (const Json& item : runs) {
            if (!item.is_array() || item.size() != 2) {
                fail("cursor patch run entry must be [index, value]");
            }
            run_indexes.insert(
                read_uint(item[0], "cursor patch run index"));
        }
        parsed_cursor_patch_run_indexes_[segment] = std::move(run_indexes);
    }

    for (const Json& transition : parsed_.pipeline_runs) {
        if (!transition.is_object()) {
            fail("pipeline transition run must be an object");
        }
        run_transition_row(transition);
    }

    stage_node_events_before(parsed_.header.seq_end + 1);
    if (!pending_ready_after_clear_.empty()) {
        fail("target consume callback lacks its sequenced data_ready marker");
    }
    if (work_ != nullptr) {
        work_->tx_finish(*tx_, parsed_.header.seq_end);
    }
    finalize();
}

// ------------------------------------------------------ PipelineLedgerCore --

PipelineLedgerCore::PipelineLedgerCore() : impl_(new Impl()) {}

PipelineLedgerCore::~PipelineLedgerCore() {
    work_ledger_bindings().erase(this);
}

int PipelineLedgerCore::prepare(const ParsedEnvelope& parsed,
                                WorkLedgerCore::Transaction& tx,
                                const std::string& segment_bytes) {
    WorkLedgerCore* work = nullptr;
    {
        const auto bound = work_ledger_bindings().find(this);
        if (bound != work_ledger_bindings().end()) work = bound->second;
    }
    auto update = std::make_unique<PreparedUpdate>();
    update->parsed = parsed;
    update->segment_bytes = segment_bytes;

    // validate_envelope's retained-redeclaration gate (pipeline_ledger.py
    // 2397-2403).
    const bool is_last_replay =
        static_cast<std::int64_t>(parsed.header.delivery_sequence)
            == delivery_sequence_;
    for (const auto& declaration : parsed.descriptors) {
        const std::string& segment =
            declaration.first->identity.segment_id;
        if (impl_->descriptors.count(segment) != 0 && !is_last_replay) {
            fail("descriptor segment was declared more than once");
        }
    }

    // _gate_delivery (pipeline_ledger.py 2511-2543); the idempotent-replay
    // branch compares THIS delivery's raw segment bytes against the cached
    // bytes of the last accepted delivery (ratified raw-bytes narrowing).
    if (static_cast<std::int64_t>(parsed.header.delivery_sequence)
                == delivery_sequence_
            && delivery_sequence_ >= 0) {
        if (parsed.header.epoch != epoch_
                || segment_bytes != last_content_raw_) {
            fail("last replay token carries typed-different canonical content");
        }
        update->replayed = true;
        update->resources = impl_->active_resources;
        return impl_->store_prepared(std::move(update));
    }
    if (parsed.header.delivery_sequence
            != static_cast<std::uint64_t>(delivery_sequence_ + 1)) {
        fail("delivery_sequence gap or stale replay");
    }
    if (parsed.header.full) {
        if (initialized_ || parsed.header.base_watermark != 0
                || parsed.header.base_version != 0) {
            fail("full snapshot is only valid for initial state");
        }
    } else {
        if (!initialized_) {
            fail("delta received before initial full snapshot");
        }
        if (parsed.header.epoch != epoch_) {
            fail("v3 epoch changed during run");
        }
        if (parsed.header.base_watermark != watermark_
                || parsed.header.base_version != state_version_) {
            fail("delta base watermark/state version mismatch");
        }
    }
    if (static_cast<std::int64_t>(parsed.header.tick) < last_tick_) {
        fail("v3 tick regressed");
    }

    // Global logical-sequence merge walk (pipeline_ledger.py 2442-2483).
    if (parsed.node_events.size() + parsed.pipeline_runs.size()
            != parsed.header.event_count) {
        fail("logical event count differs from typed transition count");
    }
    {
        std::size_t node_check = 0;
        std::size_t pipeline_check = 0;
        std::uint64_t expected_seq = parsed.header.seq_start;
        std::int64_t previous_tick = last_tick_;
        while (node_check < parsed.node_events.size()
               || pipeline_check < parsed.pipeline_runs.size()) {
            const std::uint64_t* node_seq = nullptr;
            std::uint64_t node_tick = 0;
            if (node_check < parsed.node_events.size()) {
                node_seq = &parsed.node_events[node_check].first;
                node_tick = parsed.node_events[node_check].second.tick;
            }
            std::uint64_t pipeline_seq = 0;
            std::uint64_t pipeline_tick = 0;
            bool have_pipeline = false;
            if (pipeline_check < parsed.pipeline_runs.size()) {
                pipeline_seq = read_uint(
                    row_field(parsed.pipeline_runs[pipeline_check],
                              "first_logical_seq", "pipeline run"),
                    "pipeline run.first_logical_seq");
                pipeline_tick = row_uint(
                    parsed.pipeline_runs[pipeline_check], "tick_first",
                    "pipeline run");
                have_pipeline = true;
            }
            if (node_seq != nullptr && have_pipeline
                    && *node_seq == pipeline_seq) {
                fail("node and pipeline transitions share a logical sequence");
            }
            const bool take_node =
                !have_pipeline || (node_seq != nullptr && *node_seq < pipeline_seq);
            std::uint64_t seq = 0;
            std::uint64_t event_tick = 0;
            if (take_node) {
                seq = *node_seq;
                event_tick = node_tick;
                node_check += 1;
            } else {
                seq = pipeline_seq;
                event_tick = pipeline_tick;
                pipeline_check += 1;
            }
            if (seq != expected_seq) {
                fail("node/pipeline transitions do not form one global logical sequence");
            }
            if ((previous_tick >= 0
                     && static_cast<std::int64_t>(event_tick) < previous_tick)
                    || event_tick > parsed.header.tick) {
                fail("global node/pipeline transition ticks are not monotonic");
            }
            previous_tick = static_cast<std::int64_t>(event_tick);
            expected_seq += 1;
        }
        if (expected_seq != parsed.header.seq_end + 1) {
            fail("logical event count does not cover the declared sequence interval");
        }
    }
    if (!parsed.node_events.empty() && work == nullptr) {
        fail("v3 node transitions require a coordinated WorkLedger COW transaction");
    }

    StageDelivery staging(*impl_, *update, work, &tx);
    staging.run();   // any throw destroys the staged overlays (no slot kept)
    return impl_->store_prepared(std::move(update));
}

PipelineReceipt PipelineLedgerCore::commit_prepared(int prepared_slot) {
    PreparedUpdate* update = impl_->find_prepared(prepared_slot);
    if (update == nullptr) {
        fail("unknown prepared delivery slot");
    }
    const EnvelopeHeader& header = update->parsed.header;
    const std::string snapshot_kind = header.full ? "full" : "delta";
    if (update->replayed) {
        replay_count_ += 1;
        impl_->release_prepared(prepared_slot);
        return PipelineReceipt{snapshot_kind, header.delivery_sequence,
                               header.event_count, header.base_watermark,
                               watermark_, header.base_version,
                               state_version_, true};
    }
    Impl& st = *impl_;
    for (const auto& entry : update->descriptor_additions) {
        st.descriptors[entry.first] = entry.second;
    }
    for (const auto& entry : update->descriptor_state_updates) {
        st.descriptor_state[entry.first] = entry.second;
    }
    for (const auto& entry : update->parsed.selector_index_additions) {
        st.selector_indexes[entry.first] = entry.second;
    }
    for (const auto& entry : update->dependency_index_additions) {
        st.dependency_endpoint_index[entry.first] = entry.second;
    }
    for (const auto& entry : update->dependency_rank_index_additions) {
        st.dependency_endpoint_index_by_rank[entry.first] = entry.second;
    }
    for (const auto& entry : update->dependency_row_segment_additions) {
        st.dependency_row_segments[entry.first] = entry.second;
    }
    for (const auto& entry : update->terminal_provenance_updates) {
        st.completed_terminal_intervals_by_rank[entry.first] = entry.second;
    }
    for (const auto& entry : update->descriptor_additions) {
        st.active_segments.insert(entry.first);
    }
    for (const std::string& segment : update->descriptor_retirements) {
        st.active_segments.erase(segment);
    }

    for (const auto& entry : update->cursor_changes) {
        if (!entry.second) {
            const auto runs =
                st.cursor_runs_by_segment.find(entry.first.first);
            if (runs != st.cursor_runs_by_segment.end()) {
                runs->second.erase(entry.first.second);
            }
        } else {
            st.cursor_runs_by_segment[entry.first.first]
                                   [entry.first.second] = *entry.second;
        }
    }
    for (const auto& entry : update->cursor_version_updates) {
        st.cursor_versions_by_segment[entry.first] = entry.second;
    }
    for (const auto& entry : update->cursor_transfer_changes) {
        if (!entry.second) {
            st.active_transfers_by_run.erase(entry.first);
        } else {
            st.active_transfers_by_run[entry.first] = *entry.second;
        }
    }

    const auto apply_hash_counts =
        [](auto& target, const auto& deltas) {
            for (const auto& delta : deltas) {
                auto current = target.find(delta.first);
                const std::int64_t value =
                    (current == target.end() ? 0 : current->second)
                    + delta.second;
                if (value != 0) {
                    target[delta.first] = value;
                } else {
                    target.erase(delta.first);
                }
            }
        };
    const auto apply_map_counts =
        [](auto& target, const auto& deltas) {
            for (const auto& delta : deltas) {
                auto current = target.find(delta.first);
                const std::int64_t value =
                    (current == target.end() ? 0 : current->second)
                    + delta.second;
                if (value != 0) {
                    target[delta.first] = value;
                } else {
                    target.erase(delta.first);
                }
            }
        };
    const auto apply_rows = [](auto& target, const auto& changes) {
        for (const auto& change : changes) {
            if (!change.second) {
                target.erase(change.first);
            } else {
                target[change.first] = *change.second;
            }
        }
    };
    apply_hash_counts(st.slot_owner_counts, update->slot_owner_deltas);
    apply_map_counts(st.source_occupancy, update->source_occupancy_deltas);
    apply_map_counts(st.target_occupancy, update->target_occupancy_deltas);
    apply_hash_counts(st.credit_ack_counts,
                      update->credit_ack_domain_deltas);
    apply_hash_counts(st.ack_pins_by_segment, update->ack_pin_deltas);
    apply_hash_counts(st.active_descriptor_count_by_domain,
                      update->active_descriptor_deltas);
    apply_hash_counts(st.slot_row_counts_by_domain,
                      update->slot_row_deltas);
    apply_hash_counts(st.credit_row_counts_by_domain,
                      update->credit_row_deltas);
    apply_hash_counts(st.obligation_remaining_by_segment,
                      update->obligation_remaining_deltas);
    apply_hash_counts(st.unfinished_dependency_pins_by_segment,
                      update->dependency_pin_deltas);
    apply_map_counts(st.unfinished_dependency_waiters_by_endpoint,
                     update->dependency_waiter_deltas);
    apply_hash_counts(st.cursor_unissued_by_segment,
                      update->cursor_unissued_deltas);
    apply_hash_counts(st.active_transfer_id_counts,
                      update->active_transfer_deltas);
    apply_map_counts(st.source_release_watermarks,
                     update->release_watermark_deltas);
    for (const auto& delta : update->active_transfer_deltas) {
        if (delta.second != 0) {
            const auto current =
                st.active_transfer_id_counts.find(delta.first);
            if (current != st.active_transfer_id_counts.end()
                    && current->second > 0) {
                st.active_transfer_ids.insert(delta.first);
            } else {
                st.active_transfer_ids.erase(delta.first);
            }
        }
    }

    apply_rows(st.slots_by_key, update->slot_changes);
    apply_rows(st.target_ready_shards_by_slot, update->target_ready_changes);
    apply_rows(st.credits_by_key, update->credit_changes);
    for (const auto& entry : update->target_slot_lineage_changes) {
        st.target_slot_lineage[entry.first] = entry.second;
    }
    for (const auto& entry : update->obligation_changes) {
        const auto old = st.obligations_by_id.find(entry.first);
        if (old != st.obligations_by_id.end()) {
            const std::string old_segment = row_string(
                old->second, "segment_id", "obligation row");
            const auto segment_ids =
                st.obligations_by_segment.find(old_segment);
            if (segment_ids != st.obligations_by_segment.end()) {
                segment_ids->second.erase(entry.first);
                if (segment_ids->second.empty()) {
                    st.obligations_by_segment.erase(old_segment);
                }
            }
        }
        if (!entry.second) {
            st.obligations_by_id.erase(entry.first);
        } else {
            st.obligations_by_id[entry.first] = *entry.second;
            st.obligations_by_segment[row_string(
                                          *entry.second, "segment_id",
                                          "obligation row")]
                .insert(entry.first);
        }
    }

    // Release-watermark domain registration, then the retirement cleanup
    // with the historical guard condition (pipeline_ledger.py 4633-4648).
    for (const auto& delta : update->release_watermark_deltas) {
        const auto current =
            st.source_release_watermarks.find(delta.first);
        if (current != st.source_release_watermarks.end()
                && current->second != 0) {
            st.release_watermark_keys_by_domain[delta.first.domain]
                .insert(delta.first);
        }
    }
    std::set<std::string> cleanup_domains;
    for (const std::string& segment : update->descriptor_retirements) {
        const auto desc = st.descriptors.find(segment);
        if (desc == st.descriptors.end()) continue;
        const std::string domain =
            desc->second->identity.buffer_domain_id;
        const auto delta = update->active_descriptor_deltas.find(domain);
        const std::int64_t domain_delta =
            delta != update->active_descriptor_deltas.end()
                ? delta->second : 0;
        const auto current =
            st.active_descriptor_count_by_domain.find(domain);
        const std::int64_t post =
            current != st.active_descriptor_count_by_domain.end()
                ? current->second : 0;
        if (domain_delta < 0 && post == 0) {
            cleanup_domains.insert(domain);
        }
    }
    for (const std::string& domain : cleanup_domains) {
        const auto keys =
            st.release_watermark_keys_by_domain.find(domain);
        if (keys == st.release_watermark_keys_by_domain.end()) continue;
        for (const ReleaseKey& key : keys->second) {
            st.source_release_watermarks.erase(key);
        }
        st.release_watermark_keys_by_domain.erase(domain);
    }

    for (const auto& entry : update->active_endpoint_deltas) {
        auto current = st.active_endpoint_counts.find(entry.first);
        const std::int64_t value =
            (current == st.active_endpoint_counts.end() ? 0
                                                        : current->second)
            + entry.second;
        if (value != 0) {
            st.active_endpoint_counts[entry.first] = value;
        } else {
            st.active_endpoint_counts.erase(entry.first);
        }
    }

    st.active_resources = update->resources;
    for (const std::string& segment : update->descriptor_retirements) {
        st.cursor_runs_by_segment.erase(segment);
        st.cursor_versions_by_segment.erase(segment);
        st.cursor_unissued_by_segment.erase(segment);
        st.active_endpoint_counts.erase(segment);
        st.ack_pins_by_segment.erase(segment);
        st.selector_indexes.erase(segment);
    }
    epoch_ = header.epoch;
    initialized_ = true;
    delivery_sequence_ =
        static_cast<std::int64_t>(header.delivery_sequence);
    watermark_ = header.seq_end;
    state_version_ = header.result_version;
    last_tick_ = static_cast<std::int64_t>(header.tick);
    last_replay_token_ = header.replay_token;
    last_content_raw_ = update->segment_bytes;
    const PipelineReceipt receipt{snapshot_kind, header.delivery_sequence,
                                  header.event_count,
                                  header.base_watermark, watermark_,
                                  header.base_version, state_version_,
                                  false};
    impl_->release_prepared(prepared_slot);
    return receipt;
}

void PipelineLedgerCore::discard_prepared(int prepared_slot) {
    if (impl_->find_prepared(prepared_slot) == nullptr) {
        fail("unknown prepared delivery slot");
    }
    impl_->release_prepared(prepared_slot);
}

Json PipelineLedgerCore::pricing_json() const {
    Json out = Json::object();
    out["epoch"] = initialized_ ? Json(epoch_) : Json(nullptr);
    out["delivery_sequence"] = delivery_sequence_;
    out["event_watermark"] = watermark_;
    out["state_version"] = state_version_;
    out["tick"] = last_tick_;
    out["active_resources"] = impl_->active_resources;
    return out;
}

// snapshot_state (pipeline_ledger.py 4683-4729).
Json PipelineLedgerCore::snapshot_state_json() const {
    Json out = Json::object();
    out["epoch"] = initialized_ ? Json(epoch_) : Json(nullptr);
    out["delivery_sequence"] = delivery_sequence_;
    out["event_watermark"] = watermark_;
    out["state_version"] = state_version_;
    out["tick"] = last_tick_;
    Json descriptors = Json::object();
    Json descriptor_state = Json::object();
    for (const std::string& segment : impl_->active_segments) {
        const auto desc = impl_->descriptors.find(segment);
        if (desc != impl_->descriptors.end()) {
            descriptors[segment] = desc->second->to_wire();
        }
        const auto state = impl_->descriptor_state.find(segment);
        if (state != impl_->descriptor_state.end()) {
            descriptor_state[segment] = state->second;
        }
    }
    out["descriptors"] = std::move(descriptors);
    out["descriptor_state"] = std::move(descriptor_state);
    Json cursors = Json::array();
    for (const std::string& segment : impl_->active_segments) {
        const auto desc = impl_->descriptors.find(segment);
        if (desc == impl_->descriptors.end()) continue;
        Json entry = Json::object();
        entry["segment_id"] = segment;
        entry["buffer_domain_id"] =
            desc->second->identity.buffer_domain_id;
        entry["generation"] = desc->second->identity.generation;
        const auto version =
            impl_->cursor_versions_by_segment.find(segment);
        if (version != impl_->cursor_versions_by_segment.end()) {
            entry["cursor_version"] = version->second;
        } else {
            const auto state = impl_->descriptor_state.find(segment);
            entry["cursor_version"] =
                state != impl_->descriptor_state.end()
                    ? state->second["cursor_version"] : Json(nullptr);
        }
        entry["as_of_tick"] = last_tick_;
        Json runs = Json::array();
        const auto runs_map =
            impl_->cursor_runs_by_segment.find(segment);
        if (runs_map != impl_->cursor_runs_by_segment.end()) {
            for (const auto& run : runs_map->second) {
                runs.push_back(run.second);
            }
        }
        entry["runs"] = std::move(runs);
        cursors.push_back(std::move(entry));
    }
    out["authoritative_cursors"] = std::move(cursors);
    out["active_resources"] = impl_->active_resources;
    Json obligations = Json::array();
    for (const auto& obligation : impl_->obligations_by_id) {
        obligations.push_back(obligation.second);
    }
    out["obligations"] = std::move(obligations);
    Json slots = Json::array();
    for (const auto& slot : impl_->slots_by_key) {
        slots.push_back(slot.second);
    }
    out["slots"] = std::move(slots);
    Json credits = Json::array();
    for (const auto& credit : impl_->credits_by_key) {
        credits.push_back(credit.second);
    }
    out["credits"] = std::move(credits);
    return out;
}

const std::unordered_map<std::string, DescriptorPtr>&
PipelineLedgerCore::descriptors() const {
    return impl_->descriptors;
}

const std::unordered_map<std::string, SelectorIndex>&
PipelineLedgerCore::selector_indexes() const {
    return impl_->selector_indexes;
}

}  // namespace jointledger
