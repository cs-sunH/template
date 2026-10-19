// TypedRows.hh -- typed forms of the hot ledger rows (v6 compact round).
//
// These structures are the single in-memory representation of the hot
// pipeline rows (slot/credit/obligation/cursor/resource rows and the parsed
// pipeline run records) shared by:
//   * the compact wire reader (CompactReader.cc) and the legacy JSON row
//     validators (RowParsers.cc) -- both produce these records, so exactly
//     one typed core consumes them;
//   * the staging/commit core (PipelineStage.cc) -- persistent hot state,
//     COW overlays and patch application read and write these fields
//     directly (no Json round-trip on the per-frame path);
//   * the producer (WorkLedgerV2Publisher / RemoteReadExecutor /
//     WorkLedgerV2RuntimeSnapshot) -- callbacks construct these records
//     directly, never a whole-row JSON DOM.
//
// Field-for-field these mirror the parsed-row JSON conventions frozen by the
// v4 round (the exact key sets validated by RowParsers.cc); the compact wire
// (workledger-compact-v1) omits only fields that are uniquely reconstructible
// from already-validated identity (descriptor/run/selector/cursor position)
// and the readers restore them at parse time.
//
// WireNum preserves the bool/int/float distinction of the JSON acceptance
// contract (Python typed_equal never conflates them); every comparison
// between rows uses typed_eq, the structural equality of the same contract.
#ifndef JOINT_LEDGER_TYPED_ROWS_HH
#define JOINT_LEDGER_TYPED_ROWS_HH

#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "LedgerTypes.hh"   // Json (query-boundary serialization only)

namespace jointledger {

// ------------------------------------------------------------- numbers ----

// One JSON-accepted number cell: null, exact int, or float.
struct WireNum {
    enum class Kind : std::uint8_t { Null, Int, Float };
    Kind kind = Kind::Null;
    std::uint64_t integer = 0;  // Python int (non-negative domain of the rows)
    double real = 0.0;

    static WireNum null_num() { return WireNum{}; }
    static WireNum of_uint(std::uint64_t value) {
        WireNum out; out.kind = Kind::Int; out.integer = value; return out;
    }
    static WireNum of_double(double value) {
        WireNum out; out.kind = Kind::Float; out.real = value; return out;
    }
    bool is_null() const { return kind == Kind::Null; }
    bool is_int() const { return kind == Kind::Int; }
    bool is_float() const { return kind == Kind::Float; }
    bool typed_eq(const WireNum& other) const {
        if (kind != other.kind) return false;
        switch (kind) {
            case Kind::Null: return true;
            case Kind::Int: return integer == other.integer;
            case Kind::Float: default: return real == other.real;
        }
    }
    // Numeric value for range comparisons (validated rows only).
    double as_double() const {
        return kind == Kind::Int ? static_cast<double>(integer) : real;
    }
};

// {ops, bytes, latency_ns} with per-dimension null/int/float preservation.
struct WireAmount {
    WireNum ops;
    WireNum bytes;
    WireNum latency_ns;

    bool typed_eq(const WireAmount& other) const {
        return ops.typed_eq(other.ops) && bytes.typed_eq(other.bytes)
            && latency_ns.typed_eq(other.latency_ns);
    }
    bool is_empty() const {
        return ops.is_null() && bytes.is_null() && latency_ns.is_null();
    }
};

// --------------------------------------------------------------- keys -----

struct SlotKey {
    std::string domain;
    bool role_source = true;      // false = "target"
    std::uint64_t rank = 0;
    std::int64_t shard = -1;      // -1 encodes Python's null shard ordinal
    std::uint64_t slot_index = 0;

    bool operator<(const SlotKey& other) const {
        if (domain != other.domain) return domain < other.domain;
        // Python sort key order: "source" < "target".
        if (role_source != other.role_source) {
            return role_source && !other.role_source;
        }
        if (rank != other.rank) return rank < other.rank;
        if (shard != other.shard) return shard < other.shard;
        return slot_index < other.slot_index;
    }
    bool operator==(const SlotKey& other) const {
        return domain == other.domain && role_source == other.role_source
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

// -------------------------------------------------------------- rows ------

// The six owner identity fields (owner_key set of the wire rows).
struct SlotOwner {
    std::string segment_id;
    std::uint64_t pipeline_generation = 0;
    std::uint64_t stage_epoch = 0;
    std::uint64_t staging_allocation_id = 0;
    std::uint64_t group_ordinal = 0;
    std::uint64_t tile_ordinal = 0;

    bool operator==(const SlotOwner& other) const {
        return segment_id == other.segment_id
            && pipeline_generation == other.pipeline_generation
            && stage_epoch == other.stage_epoch
            && staging_allocation_id == other.staging_allocation_id
            && group_ordinal == other.group_ordinal
            && tile_ordinal == other.tile_ordinal;
    }
    bool operator<(const SlotOwner& other) const {
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

enum class ObligationState : std::uint8_t { Pending = 0, Active = 1 };

// obligation_keys wire row (parsed form).
struct ObligationRow {
    std::string obligation_id;     // segment + "/run/" + run_index (canonical)
    std::string segment_id;
    std::uint64_t rank = 0;
    std::uint64_t first_store_id = 0;
    std::uint64_t remaining_endpoint_count = 0;
    ObligationState state = ObligationState::Pending;

    bool typed_eq(const ObligationRow& other) const {
        return obligation_id == other.obligation_id
            && segment_id == other.segment_id && rank == other.rank
            && first_store_id == other.first_store_id
            && remaining_endpoint_count == other.remaining_endpoint_count
            && state == other.state;
    }
};

// credit_keys wire row (parsed form); capacity is the constant 2 of the
// two-slot contract (kept explicit so the legacy JSON domain stays exact).
struct CreditRow {
    std::string domain;
    std::uint64_t source_rank = 0;
    std::uint64_t shard = 0;
    std::uint64_t capacity = 2;
    std::uint64_t available = 0;
    std::uint64_t outstanding_ack_count = 0;

    bool typed_eq(const CreditRow& other) const {
        return domain == other.domain && source_rank == other.source_rank
            && shard == other.shard && capacity == other.capacity
            && available == other.available
            && outstanding_ack_count == other.outstanding_ack_count;
    }
};

// slot_keys wire row (parsed form).
struct SlotRow {
    std::string domain;
    bool role_source = true;
    std::uint64_t rank = 0;
    bool has_shard = false;         // shard_ordinal nullability (source only)
    std::uint64_t shard = 0;
    std::uint64_t slot_index = 0;
    bool occupied = false;
    std::optional<SlotOwner> owner; // present iff occupied
    bool consumed = false;
    std::vector<std::uint64_t> target_shards_to_consume;
    std::vector<std::uint64_t> consumed_target_shards;

    bool typed_eq(const SlotRow& other) const {
        if (domain != other.domain || role_source != other.role_source
                || rank != other.rank || has_shard != other.has_shard
                || shard != other.shard || slot_index != other.slot_index
                || occupied != other.occupied
                || consumed != other.consumed
                || target_shards_to_consume != other.target_shards_to_consume
                || consumed_target_shards != other.consumed_target_shards) {
            return false;
        }
        if (owner.has_value() != other.owner.has_value()) return false;
        if (owner.has_value() && !(*owner == *other.owner)) return false;
        return true;
    }
};

// ------------------------------------------------------------ patches -----

// One operation of a target-slot delta (target_delta_keys row "operations").
struct TargetOp {
    enum class Op : std::uint8_t {
        Initialize = 0, Claim = 1, Consume = 2, RequestCloseClear = 3,
        Erase = 4
    };
    Op op = Op::Initialize;
    std::uint64_t slot_generation = 0;
    // Claim:
    std::optional<SlotOwner> owner;
    std::vector<std::uint64_t> target_shards;
    // Consume:
    std::uint64_t terminal_rank = 0;
    std::uint64_t terminal_store_id = 0;
    std::uint64_t consume_run_index = 0;
    std::vector<std::uint64_t> add_shards;
    bool cleared = false;

    bool typed_eq(const TargetOp& other) const {
        if (op != other.op || slot_generation != other.slot_generation
                || target_shards != other.target_shards
                || terminal_rank != other.terminal_rank
                || terminal_store_id != other.terminal_store_id
                || consume_run_index != other.consume_run_index
                || add_shards != other.add_shards
                || cleared != other.cleared) {
            return false;
        }
        if (owner.has_value() != other.owner.has_value()) return false;
        if (owner.has_value() && !(*owner == *other.owner)) return false;
        return true;
    }
};

// One slot patch: either a source quad [key, before, after] or a
// target_delta object (form discriminator exactly as the wire).
struct SlotPatch {
    bool is_target_delta = false;
    SlotKey key;
    // Source quad form:
    std::optional<SlotRow> before;
    std::optional<SlotRow> after;
    // Target delta form:
    std::uint64_t before_slot_generation = 0;
    std::uint64_t predecessor_logical_seq = 0;
    std::uint64_t predecessor_consumed_count = 0;
    std::vector<TargetOp> operations;
};

struct CreditPatch {
    CreditKey key;
    std::optional<CreditRow> before;
    std::optional<CreditRow> after;
};

struct ObligationPatch {
    std::string obligation_id;     // stable key of the patch
    std::optional<ObligationRow> before;
    std::optional<ObligationRow> after;
};

// ------------------------------------------------------------- cursors ----

// active_endpoint_keys row inside a cursor run.
struct ActiveEndpoint {
    // Run-scoped position ordinal (the wire's carried state for compact
    // frames; the identity fields reconstruct from it).
    std::uint64_t ordinal = 0;
    std::uint64_t store_id = 0;
    std::uint64_t tile_ordinal = 0;
    std::uint64_t shard_ordinal = 0;
    std::string endpoint;
    std::uint64_t transfer_id = 0;
    bool flow_bound = false;
    std::uint64_t flow_id = 0;
    bool network_done = false;
    bool hbm_done = false;
    SlotOwner owner;

    bool typed_eq(const ActiveEndpoint& other) const {
        return store_id == other.store_id
            && tile_ordinal == other.tile_ordinal
            && shard_ordinal == other.shard_ordinal
            && endpoint == other.endpoint
            && transfer_id == other.transfer_id
            && flow_bound == other.flow_bound
            && (!flow_bound || flow_id == other.flow_id)
            && network_done == other.network_done
            && hbm_done == other.hbm_done
            && owner == other.owner;
    }
};

// cursor_run_keys row (parsed form); the static extent fields are the
// descriptor's endpoint-run facts (validated equal on the JSON path,
// reconstructed there on the compact path).
struct CursorRun {
    std::uint64_t rank = 0;
    std::uint64_t group_ordinal = 0;
    std::uint64_t first_tile = 0;
    std::uint64_t tile_count = 0;
    std::uint64_t completed_endpoint_count = 0;
    std::uint64_t cancelled_endpoint_count = 0;
    std::uint64_t remaining_endpoint_count = 0;
    bool in_flight = false;
    bool has_next_store = false;
    std::uint64_t next_store_id = 0;
    std::uint64_t tile_offset = 0;
    std::uint64_t shard_index = 0;
    std::uint64_t leg_index = 0;
    bool has_initial_parent = false;
    std::uint64_t initial_parent_store_id = 0;
    std::vector<std::pair<std::uint64_t, std::uint64_t>> completed_out_of_order;
    std::vector<ActiveEndpoint> active_endpoints;

    bool typed_eq(const CursorRun& other) const {
        return rank == other.rank && group_ordinal == other.group_ordinal
            && first_tile == other.first_tile && tile_count == other.tile_count
            && completed_endpoint_count == other.completed_endpoint_count
            && cancelled_endpoint_count == other.cancelled_endpoint_count
            && remaining_endpoint_count == other.remaining_endpoint_count
            && in_flight == other.in_flight
            && has_next_store == other.has_next_store
            && (!has_next_store || next_store_id == other.next_store_id)
            && tile_offset == other.tile_offset
            && shard_index == other.shard_index
            && leg_index == other.leg_index
            && has_initial_parent == other.has_initial_parent
            && (!has_initial_parent
                    || initial_parent_store_id == other.initial_parent_store_id)
            && completed_out_of_order == other.completed_out_of_order
            && active_endpoints.size() == other.active_endpoints.size()
            && ([](const std::vector<ActiveEndpoint>& left,
                   const std::vector<ActiveEndpoint>& right) {
                   for (std::size_t i = 0; i < left.size(); ++i) {
                       if (!left[i].typed_eq(right[i])) return false;
                   }
                   return true;
               })(active_endpoints, other.active_endpoints);
    }
};

struct CursorPatch {
    std::string segment_id;
    std::uint64_t cursor_version = 0;
    bool retired = false;
    std::vector<std::pair<std::uint64_t, std::optional<CursorRun>>> runs;
};

// --------------------------------------------------------------- runs -----

enum class TransitionKind : std::uint8_t {
    DescriptorCommit = 0, TileIssue = 1, DataReady = 2, EndpointDone = 3,
    FillAckProgress = 4, FillAckDone = 5, ConsumeStart = 6, ConsumeDone = 7,
    ReleaseAckProgress = 8, ReleaseAckDone = 9, CreditReturn = 10,
    ServiceDone = 11, RequestClose = 12, SettleDone = 13, DescriptorRetire = 14
};
const char* transition_kind_name(TransitionKind kind);          // TypedRows.cc
TransitionKind transition_kind_from_name(const std::string& name);  // throws

enum class ResourceLeg : std::uint8_t {
    None = 0, Endpoint = 1, BodyMarker = 2, Noc = 3, HbmCommRead = 4,
    HbmCommWrite = 5
};
const char* resource_leg_name(ResourceLeg leg);                // TypedRows.cc
// Wire string of a resource leg ("endpoint"/"body_marker"/"noc"/
// "hbm:COMM_READ"/"hbm:COMM_WRITE"); throws for None.

struct EndpointSelector {
    std::uint64_t group_ordinal = 0;
    std::uint64_t tile_ordinal = 0;
    std::uint64_t shard_ordinal = 0;
    std::string endpoint;
};

// _selected_shard_tile parsed facts.
struct ShardTileFacts {
    std::uint64_t shard_ordinal = 0;
    std::uint64_t source_rank = 0;
    std::uint64_t target_rank = 0;
    std::uint64_t data_bytes = 0;
    bool emit_endpoint = false;
};

// The parsed pipeline run record (the 16-key parsed-row contract of the v4
// reader, typed).  Endpoint-bound facts are present iff have_selector.
struct PipelineRun {
    std::uint64_t first_logical_seq = 0;
    std::string descriptor_segment_id;
    TransitionKind transition_kind = TransitionKind::DescriptorCommit;
    bool have_selector = false;
    EndpointSelector endpoint_selector;
    std::uint64_t selector_rank = 0;   // rank_first (present with selector)
    // Wire-visible field the staging core does not consume (validated
    // against the endpoint kind by both readers instead).
    ResourceLeg resource_leg = ResourceLeg::None;
    bool has_endpoint_run_index = false;
    std::uint64_t endpoint_run_index = 0;
    bool has_endpoint_leg_index = false;
    std::uint64_t endpoint_leg_index = 0;
    bool has_endpoint_ordinal = false;
    std::uint64_t endpoint_ordinal = 0;
    bool has_selected_global_tile_ordinal = false;
    std::uint64_t selected_global_tile_ordinal = 0;
    bool has_selected_shard_tile = false;
    ShardTileFacts selected_shard_tile;
    bool has_endpoint_store_id = false;
    std::uint64_t endpoint_store_id = 0;
    std::vector<SlotPatch> slot_patches;
    std::vector<CreditPatch> credit_patches;
    std::vector<ObligationPatch> obligation_patches;
    std::uint64_t tick_first = 0;
    std::uint64_t cursor_version_start = 0;
    std::uint64_t cursor_version_end = 0;
};

// ----------------------------------------------------------- resources ----

enum class ResourceKind : std::uint8_t { NocFlow = 0, HbmJob = 1, PoolJob = 2 };
const char* resource_kind_name(ResourceKind kind);             // TypedRows.cc

// active_resources row (parsed form): the common identity plus the
// kind-specific extras exactly as the wire key sets.
struct ResourceRow {
    ResourceKind kind = ResourceKind::NocFlow;
    // common keys
    std::string work_id;            // canonical derivation, validated
    std::string request_id;
    std::string stage;
    std::uint64_t generation = 0;
    std::uint64_t rank = 0;
    std::uint64_t node_id = 0;
    std::string operation;
    std::string leg;
    std::string state;
    WireAmount total;
    WireAmount remaining;
    std::vector<std::uint64_t> route;   // NoC only
    std::uint64_t as_of_tick = 0;
    bool has_issued_tick = false;
    std::uint64_t issued_tick = 0;
    bool bound = false;             // descriptor_segment_id present
    std::string descriptor_segment_id;
    bool has_endpoint_selector = false;
    EndpointSelector endpoint_selector;
    // noc_flow extras
    std::uint64_t flow_id = 0;
    WireNum current_rate_Bpns;
    WireNum last_rate_update_time;
    // hbm_job extras
    std::string job_kind;
    std::uint64_t start_tick = 0;
    // pool_job extras
    std::uint64_t port_index = 0;
    std::uint64_t issue_seq = 0;
    WireNum latency_ready_ns;

    // Per-kind state enumeration for the compact wire (state stays a string
    // in the struct so query serialization is unchanged).
    std::uint8_t state_index() const;                    // TypedRows.cc
    static const char* state_name(ResourceKind kind,
                                  std::uint8_t index);  // TypedRows.cc

    bool typed_eq(const ResourceRow& other) const {
        return kind == other.kind && work_id == other.work_id
            && request_id == other.request_id && stage == other.stage
            && generation == other.generation && rank == other.rank
            && node_id == other.node_id && operation == other.operation
            && leg == other.leg && state == other.state
            && total.typed_eq(other.total)
            && remaining.typed_eq(other.remaining) && route == other.route
            && as_of_tick == other.as_of_tick
            && has_issued_tick == other.has_issued_tick
            && (!has_issued_tick || issued_tick == other.issued_tick)
            && bound == other.bound
            && descriptor_segment_id == other.descriptor_segment_id
            && has_endpoint_selector == other.has_endpoint_selector
            && (!has_endpoint_selector
                    || (endpoint_selector.group_ordinal
                            == other.endpoint_selector.group_ordinal
                        && endpoint_selector.tile_ordinal
                            == other.endpoint_selector.tile_ordinal
                        && endpoint_selector.shard_ordinal
                            == other.endpoint_selector.shard_ordinal
                        && endpoint_selector.endpoint
                            == other.endpoint_selector.endpoint))
            && flow_id == other.flow_id
            && current_rate_Bpns.typed_eq(other.current_rate_Bpns)
            && last_rate_update_time.typed_eq(other.last_rate_update_time)
            && job_kind == other.job_kind && start_tick == other.start_tick
            && port_index == other.port_index && issue_seq == other.issue_seq
            && latency_ready_ns.typed_eq(other.latency_ready_ns);
    }
};

// ------------------------------------------------------------ node rows ---

// Node transition kinds of the node stream (wire order).
enum class NodeKind : std::uint8_t {
    Commit = 0, Issue = 1, Progress = 2, Completion = 3, Ack = 4, Release = 5
};
const char* node_kind_name(NodeKind kind);                     // TypedRows.cc
NodeKind node_kind_from_name(const std::string& name);         // throws

// Query-boundary serialization (pricing/snapshot/export surfaces only).
Json amount_to_json(const WireAmount& amount);
Json slot_row_to_json(const SlotRow& row);
Json credit_row_to_json(const CreditRow& row);
Json obligation_row_to_json(const ObligationRow& row);
Json cursor_run_to_json(const CursorRun& run);
Json resource_row_to_json(const ResourceRow& row);

}  // namespace jointledger

#endif  // JOINT_LEDGER_TYPED_ROWS_HH
