// JsonScanner.hh -- structural scanner over one compact JSON request.
//
// Locates the top-level "work_ledger" member of a bridge request without
// decoding the whole document: a single pass over member boundaries with
// full string-escape/nesting handling.  The splice window it reports
// removes exactly that member (plus one adjacent separator) so the
// remaining bytes are the outer frame JSON.
#ifndef JOINT_LEDGER_JSON_SCANNER_HH
#define JOINT_LEDGER_JSON_SCANNER_HH

#include <cstddef>
#include <cstdint>
#include <string>

namespace jointledger {

struct ScanResult {
    std::uint64_t member_start = 0;  // opening quote of the member name
    std::uint64_t member_end = 0;    // one past the value's last byte
    std::uint64_t value_start = 0;   // first byte of the member value
    std::uint64_t value_end = 0;     // one past the value's last byte
};

// Throws LedgerError with a byte offset on malformed structure or when
// work_ledger is absent/duplicated.  Only boundary-level syntax is checked
// here; full value syntax is validated by the downstream parsers.
ScanResult scan_request(const char* data, std::size_t length);

}  // namespace jointledger

#endif  // JOINT_LEDGER_JSON_SCANNER_HH
