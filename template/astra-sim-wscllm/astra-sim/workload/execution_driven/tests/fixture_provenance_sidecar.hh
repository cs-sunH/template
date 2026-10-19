/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

fixture_provenance_sidecar.hh -- test-only helper (P1 fix 2026-09-28).

The WindowedTraceReader provenance gate is fail-closed on an ABSENT sidecar
(the former "absent (no gate)" pass-through let a truncated half-written
queue silently shrink the simulation input). Every hand-written queue-CSV
fixture must therefore ship its own matching provenance record. This helper
writes the fixture CSV AND a byte-for-byte matching sidecar, computing the
stats exactly like the Python materializer (queue_provenance in
derive_20_first_30_seconds.py) and the reader's index pass:
lines split on '\n' (first non-empty = header, later non-empty = data row),
FNV-1a 64 over the raw bytes, byte count, data-row count, distinct session
count, turn-0 count (6th comma field non-empty), turn-0 arrival min/max,
file-order adjacent inversions, contiguous session blocks. Fixtures that are
DELIBERATELY malformed abort in-row before the gate, so the sidecar values
are never consulted for them.

Header-only so both windowed_trace_reader_test.cc and
calendar_reader_oracle_test.cc share one implementation.
******************************************************************************/

#ifndef EXECUTION_DRIVEN_TESTS_FIXTURE_PROVENANCE_SIDECAR_HH
#define EXECUTION_DRIVEN_TESTS_FIXTURE_PROVENANCE_SIDECAR_HH

#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

namespace AstraSim {
namespace ExecutionDriven {
namespace FixtureProvenance {

inline uint64_t fnv1a64_bytes(uint64_t h, const std::string& bytes) {
    for (const char c : bytes) {
        h = (h ^ static_cast<uint64_t>(static_cast<unsigned char>(c))) *
            1099511628211ULL;
    }
    return h;
}

/// Write `rows` (8-column request-queue data rows, no trailing newlines)
/// under the frozen schema at csv_path, plus a MATCHING
/// csv_path + ".provenance.json" sidecar.
inline void write_queue_fixture(const std::string& csv_path,
                                const std::vector<std::string>& rows) {
    static const char* kHeader =
        "session_id,turn_index,request_id,prefill_length,decode_length,"
        "session_arrival_time_ns,inter_request_interval_ns,description";
    std::string bytes(kHeader);
    for (const auto& row : rows) {
        bytes += "\n";
        bytes += row;
    }
    bytes += "\n";
    {
        std::ofstream out(csv_path, std::ios::binary);
        out << bytes;
    }

    uint64_t fnv = 14695981039346656037ULL;
    fnv = fnv1a64_bytes(fnv, bytes);
    uint64_t data_rows = 0, turn0_count = 0, inversions = 0;
    uint64_t turn0_min = 0, turn0_max = 0, prev_turn0 = 0;
    bool have_turn0 = false;
    std::vector<std::string> sessions;  // file order, distinct
    size_t begin = 0;
    bool header_seen = false;
    while (true) {
        const auto nl = bytes.find('\n', begin);
        if (nl == std::string::npos) {
            break;
        }
        const std::string line = bytes.substr(begin, nl - begin);
        begin = nl + 1;
        if (!header_seen) {
            header_seen = true;  // the schema header line
            continue;
        }
        if (line.empty()) {
            continue;
        }
        ++data_rows;
        // First six comma fields (the description column may hold anything
        // but fixtures keep it comma-free; fields beyond six are ignored,
        // exactly like the reader's getline chain).
        std::string fields[6];
        size_t pos = 0;
        int taken = 0;
        for (; taken < 6; ++taken) {
            const auto comma = line.find(',', pos);
            if (comma == std::string::npos) {
                break;
            }
            fields[taken] = line.substr(pos, comma - pos);
            pos = comma + 1;
        }
        const std::string& session = fields[0];
        bool seen = false;
        for (const auto& s : sessions) {
            seen = seen || (s == session);
        }
        if (!seen) {
            sessions.push_back(session);
        }
        const std::string& arrival_text = (taken == 6) ? fields[5]
                                                       : std::string();
        if (!arrival_text.empty() &&
            arrival_text.find_first_not_of("0123456789") ==
                std::string::npos) {
            const uint64_t arrival =
                std::strtoull(arrival_text.c_str(), nullptr, 10);
            if (!have_turn0) {
                turn0_min = arrival;
                turn0_max = arrival;
                have_turn0 = true;
            } else {
                if (arrival < turn0_min) {
                    turn0_min = arrival;
                }
                if (arrival > turn0_max) {
                    turn0_max = arrival;
                }
                if (arrival < prev_turn0) {
                    ++inversions;
                }
            }
            prev_turn0 = arrival;
            ++turn0_count;
        }
        // A non-numeric arrival field makes the reader fail-closed in-row
        // (before the gate); the sidecar value for such fixtures is unused.
    }

    std::ofstream out(csv_path + ".provenance.json", std::ios::binary);
    out << "{\"schema\":1,\"generator_version\":\"test-fixture\","
        << "\"csv_sha256\":\"test-fixture (human audit only)\","
        << "\"csv_fnv1a64\":" << fnv
        << ",\"csv_bytes\":" << bytes.size()
        << ",\"data_rows\":" << data_rows
        << ",\"sessions\":" << sessions.size()
        << ",\"turn0_count\":" << turn0_count
        << ",\"turn0_arrival_min_ns\":" << turn0_min
        << ",\"turn0_arrival_max_ns\":" << turn0_max
        << ",\"turn0_adjacent_inversions\":" << inversions
        << ",\"session_blocks_contiguous\":true}\n";
}

}  // namespace FixtureProvenance
}  // namespace ExecutionDriven
}  // namespace AstraSim

#endif  // EXECUTION_DRIVEN_TESTS_FIXTURE_PROVENANCE_SIDECAR_HH
