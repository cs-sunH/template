// CompactReader.hh -- workledger-compact-v1 frame decoder (v6 round).
//
// parse_envelope_compact walks the compact bytes in one pass, performs the
// full semantic validation of the v4 row contracts (identity references,
// extents, sorted-unique keys, lifecycle-kind constraints) and reconstructs
// every identity-derived field the wire omits (store ids, owner identities,
// resource quantities, cursor positions), producing exactly the typed
// ParsedEnvelope the staging core consumes.  The legacy JSON reader
// (parse_envelope_v4) feeds the same typed core; there is one online state
// machine and two versioned decoders.
#ifndef JOINT_LEDGER_COMPACT_READER_HH
#define JOINT_LEDGER_COMPACT_READER_HH

#include <cstdint>

#include "joint_ledger/PipelineRows.hh"

namespace jointledger {

ParsedEnvelope parse_envelope_compact(const char* data, std::size_t length,
                                     const RetainedRegistry& retained,
                                     bool allow_retained_replay,
                                     std::uint64_t expected_delivery,
                                     std::uint64_t expected_tick);

}  // namespace jointledger

#endif  // JOINT_LEDGER_COMPACT_READER_HH
