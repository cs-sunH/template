// PipelineRows.hh -- v4 envelope typed parse surface (pipeline_ledger.py
// _validate_envelope_header/_parse_envelope/_parse_* ports).
//
// The reader consumes the raw work_ledger segment bytes with a nlohmann SAX
// router: header scalars are extracted directly, small sections accumulate
// bounded structures, and the big row arrays stream element-by-element
// through per-row mini-DOMs so validation is a 1:1 port of the Python row
// validators without retaining a whole-envelope DOM.  A duplicate object key
// keeps last-wins semantics (Python dict).
#ifndef JOINT_LEDGER_PIPELINE_ROWS_HH
#define JOINT_LEDGER_PIPELINE_ROWS_HH

#include <array>
#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "LedgerTypes.hh"
#include "PipelineDescriptor.hh"

namespace jointledger {

struct EnvelopeHeader {
    std::string replay_token;
    std::string epoch;
    std::uint64_t delivery_sequence = 0;
    bool full = false;                 // snapshot_kind == "full"
    std::uint64_t tick = 0;
    std::uint64_t base_watermark = 0;
    std::uint64_t seq_start = 0;
    std::uint64_t seq_end = 0;
    std::uint64_t event_count = 0;
    std::uint64_t base_version = 0;
    std::uint64_t result_version = 0;
};

// Selector index for one descriptor (pipeline_ledger
// _DescriptorSelectorIndex: groups_by_ordinal / run_intervals /
// selector_intervals / consume_intervals_by_target).
struct SelectorIndex {
    std::unordered_map<std::uint64_t, std::size_t> groups_by_ordinal;
    // key -> (sorted starts, rows (start, end, run_index[, leg_index or
    // shard_position])) exactly like the frozen Python tuples.
    std::map<std::tuple<std::uint64_t, std::uint64_t>,
             std::pair<std::vector<std::uint64_t>,
                       std::vector<std::array<std::uint64_t, 3>>>>
        run_intervals;
    std::map<std::tuple<std::uint64_t, std::uint64_t, std::uint64_t,
                        std::string>,
             std::pair<std::vector<std::uint64_t>,
                       std::vector<std::array<std::uint64_t, 4>>>>
        selector_intervals;
    std::map<std::tuple<std::uint64_t, std::uint64_t, std::uint64_t>,
             std::pair<std::vector<std::uint64_t>,
                       std::vector<std::array<std::uint64_t, 4>>>>
        consume_intervals_by_target;
};
SelectorIndex build_selector_index(const PipelineDescriptor& descriptor);

// Dependency interval index (prefix/rank treap inputs); rows are the same
// 9-tuples as _build_descriptor_dependency_index.
struct DependencyIndex {
    // prefix -> rows sorted
    std::unordered_map<std::string, std::vector<std::array<std::uint64_t, 9>>>
        intervals_by_prefix;
    std::unordered_map<std::uint64_t, std::vector<std::array<std::uint64_t, 9>>>
        intervals_by_rank;
};
DependencyIndex build_dependency_index(const PipelineDescriptor& descriptor);

struct ParsedEnvelope {
    EnvelopeHeader header;
    // (descriptor, commit-entry JSON with identity/commit_tick/cursor_version/
    // service_done/settle_done/retired exactly as the wire row)
    std::vector<std::pair<DescriptorPtr, Json>> descriptors;
    // New selector/dependency indexes for descriptors declared in this
    // delivery (segment_id -> index).
    std::unordered_map<std::string, SelectorIndex> selector_index_additions;
    std::unordered_map<std::string, DependencyIndex> dependency_index_additions;
    // (logical_seq, event) in wire order; strictly increasing.
    std::vector<std::pair<std::uint64_t, NodeEvent>> node_events;
    // Parsed pipeline run rows: JSON objects with the exact key set produced
    // by _parse_one_pipeline_run (first_logical_seq/descriptor_segment_id/
    // transition_kind/endpoint_selector/endpoint_run_index/endpoint_leg_index/
    // endpoint_ordinal/_selected_global_tile_ordinal/_selected_shard_tile/
    // endpoint_store_id/slot_patches/credit_patches/obligation_patches/
    // tick_first/cursor_version_start/cursor_version_end).
    std::vector<Json> pipeline_runs;
    // Parsed cursor patches: JSON rows as produced by _parse_cursor_patches
    // ({segment_id, runs: [[run_index, parsed_run], ...]}).
    std::vector<Json> cursor_patches;
    // Parsed active_resources rows (dict(row) forms of _parse_resources).
    std::vector<Json> resources;
};

// Retained registry view the reader validates against (accepted descriptors
// and their selector indexes from the live ledger state).
struct RetainedRegistry {
    const std::unordered_map<std::string, DescriptorPtr>* descriptors = nullptr;
    const std::unordered_map<std::string, SelectorIndex>* selector_indexes =
        nullptr;
};

// Parse + validate one work_ledger segment.  ``expected_delivery`` and
// ``expected_tick`` are the outer frame's values (online_scheduler_base
// passes payload delivery/tick; mismatch fails like _validate_payload_
// envelope's cross-check in the Python coordinator).  Throws LedgerError
// (message includes the row path; parse errors include byte offsets).
ParsedEnvelope parse_envelope_v4(const char* data, std::size_t length,
                                 const RetainedRegistry& retained,
                                 bool allow_retained_replay,
                                 std::uint64_t expected_delivery,
                                 std::uint64_t expected_tick);

// Canonical envelope bytes with Python json.dumps(sort_keys=True,
// separators=(",",":"), ensure_ascii=False) formatting -- used ONLY for the
// optional_sha256 verification path and diagnostics.  Re-serializes from a
// DOM parse of the segment; never called on the hot path.
std::string canonical_envelope_bytes(const char* data, std::size_t length);

}  // namespace jointledger

#endif  // JOINT_LEDGER_PIPELINE_ROWS_HH
