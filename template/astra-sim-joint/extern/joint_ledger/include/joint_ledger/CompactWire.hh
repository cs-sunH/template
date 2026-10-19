// CompactWire.hh -- shared codec surface of the workledger-compact-v1 wire
// (v6 compact round).
//
// One codec, two consumers:
//   * the producer (AstraSim::ExecutionDriven publisher/executor) builds
//     CompactFrameInput records directly in its callbacks and calls
//     encode_compact_frame -- no intermediate whole-row JSON DOM exists on
//     this path;
//   * the native receiver (CompactReader.cc) decodes the same bytes into the
//     typed ParsedEnvelope consumed by the staging core.
//
// The encoding is deterministic for identical logical content (stable
// interning order, fixed section order), which keeps the idempotent-replay
// raw-bytes gate meaningful.  Field layout, nullability and the exact set of
// identity-reconstructed fields are frozen in the v6 interface contract
// (仓外 INTERFACE_CONTRACT.md §2-§3).
#ifndef JOINT_LEDGER_COMPACT_WIRE_HH
#define JOINT_LEDGER_COMPACT_WIRE_HH

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "joint_ledger/LedgerTypes.hh"
#include "joint_ledger/TypedRows.hh"

namespace jointledger {

// -------------------------------------------------------------- identity --

constexpr char kCompactMagic[4] = {'W', 'L', 'C', 'P'};
constexpr std::uint8_t kCompactWireVersion = 1;
// Ledger semantic identity of frames carried by this wire (the semantic
// protocol stays 4; the wire encoding identity is its own version line).
constexpr std::uint8_t kCompactLedgerSchemaVersion = 4;
constexpr std::uint8_t kCompactLedgerProtocolVersion = 4;
constexpr std::uint8_t kCompactDescriptorVersion = 1;
extern const char* const kCompactSchemaName;  // "workledger-compact-v1"

// True when the segment starts with the compact magic.
bool is_compact_frame(const char* data, std::size_t length);
// Total frame length (magic included) of a compact segment at data; throws
// LedgerError when the header is truncated.  Used by the structural scanner.
std::size_t compact_frame_length(const char* data, std::size_t length);

// ------------------------------------------------------------- producer ---

// One node-stream record (payload keeps its JSON body by contract).
struct CompactNodeRecord {
    NodeKind kind = NodeKind::Commit;
    std::uint64_t logical_seq = 0;
    std::uint64_t tick = 0;
    std::string work_id;
    Json payload;   // object; serialized as embedded JSON bytes
};

// The typed frame model handed to the encoder.  The producer fills every
// field directly from callback state.
struct CompactFrameInput {
    bool full = false;                 // snapshot_kind == "full"
    std::string epoch;
    std::uint64_t delivery_sequence = 0;
    std::uint64_t tick = 0;
    std::uint64_t base_watermark = 0;
    std::uint64_t seq_start = 0;
    std::uint64_t seq_end = 0;
    std::uint64_t event_count = 0;
    std::uint64_t base_state_version = 0;
    std::uint64_t result_state_version = 0;
    // Embedded descriptor declaration JSON bytes (the full commit row).
    std::vector<std::string> descriptor_commits_json;
    std::vector<CompactNodeRecord> nodes;
    // Typed pipeline runs.  resource_leg is a wire-visible field the staging
    // core does not consume (validated against the endpoint kind instead).
    std::vector<PipelineRun> runs;
    std::vector<CursorPatch> cursors;
    std::vector<ResourceRow> resources;
};

// Deterministic encoder; throws LedgerError on invalid input (empty epoch,
// unknown enum values, non-object payload).
void encode_compact_frame(const CompactFrameInput& input, std::string& out);

// ------------------------------------------------------------- reader -----

// Bounds-checked cursor over one compact frame (after the magic header).
class WireCursor {
public:
    WireCursor(const char* data, std::size_t length)
        : data_(data), length_(length) {}

    std::size_t offset() const { return pos_; }
    bool exhausted() const { return pos_ >= length_; }
    std::size_t remaining() const { return pos_ <= length_ ? length_ - pos_ : 0; }

    std::uint8_t u8(const char* what);
    std::uint64_t var(const char* what);          // LEB128, <= 9 group bytes
    double f64(const char* what);                 // little-endian bits
    void bytes(std::size_t count, std::string& out, const char* what);
    std::string raw_bytes(std::size_t count, const char* what);

    // Frame-scoped string pool resolution (refs are frame-unique).
    void load_pool(std::vector<std::string>& pool);
    const std::string& pool_str(const std::vector<std::string>& pool,
                                std::uint64_t ref, const char* what);

private:
    const char* data_;
    std::size_t length_;
    std::size_t pos_ = 0;
};

}  // namespace jointledger

#endif  // JOINT_LEDGER_COMPACT_WIRE_HH
