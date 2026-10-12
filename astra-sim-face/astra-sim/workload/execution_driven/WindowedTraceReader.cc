/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

WindowedTraceReader -- execution-driven mechanism layer (wscllm phase 7 §10.4;
P0 turn-0 late-discovery fix 2026-08-30: index pass + turn-0 arrival calendar).
Implementation. Row semantics are byte-for-byte the phase-1 full loader
(the main_online.cc load_request_queue_csv entry it replaced -- the loader
function itself is gone, this class is the only CSV reader): same CSV
schema, same turn-0 Submit / turn>0 future-alarm split, same queue_index and
metrics registration order. The P0 fix changes only WHEN/HOW turn-0 rows are
SUBMITTED: never by file position -- always by the arrival calendar built by
the streaming index pass (see WindowedTraceReader.hh for the full contract
and equivalence argument).
*******************************************************************************/

#include "astra-sim/workload/execution_driven/WindowedTraceReader.hh"

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <sstream>
#include <vector>

#include <json/json.hpp>

#include "astra-sim/workload/MetricCollector.hh"

namespace AstraSim {
namespace ExecutionDriven {

namespace {

uint64_t now_ns() {
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch())
            .count());
}

// FNV-1a 64 over one byte (offset basis 14695981039346656037, prime
// 1099511628211; h ^= b; h *= prime, all arithmetic mod 2^64). Byte-for-byte
// the same algorithm as the Python materializer's provenance sidecar writer
// (derive_20_first_30_seconds.py) -- the cross-language gate depends on the
// two implementations agreeing on every input byte.
// (2026-09-30) kFnv1a64OffsetBasis removed: the member fnv1a64_ carries
// the same offset basis as its initializer literal; the named constant had
// no other reference (dual-source drift risk).
constexpr uint64_t kFnv1a64Prime = 1099511628211ULL;

inline uint64_t fnv1a64_byte(uint64_t h, unsigned char b) {
    return (h ^ static_cast<uint64_t>(b)) * kFnv1a64Prime;
}

[[noreturn]] void reader_fatal(const std::string& what) {
    std::cerr << "[Error] (execution_driven/windowed_reader) " << what
              << std::endl;
    std::exit(EXIT_FAILURE);
}

// M2-3/cpp-ed-core-2 (2026-09-26): structured fail-closed numeric parsing
// for every CSV integer field. The bare std::stoi/std::stoull previously
// used here (a) threw std::invalid_argument/std::out_of_range on a
// malformed field -- an uncaught exception terminating the process with no
// [Error] line, bypassing this module's "[Error] + exit before a single
// Submit" contract -- and (b) accepted strtoull/strtol's whitespace and
// sign shapes (" -1"/"+5": the negative wraps to ~1.8e19 and silently
// enters the calendar or an alarm). The lexicon below is the frozen
// unsigned-integer lexicon, aligned with OnlineCli's is_ascii_digits:
// non-empty, every character an ASCII '0'..'9', then a range-checked
// conversion. Applies to the turn_index / length / session_arrival
// fields alike.
//
// Documented acceptance divergences from the Python loader (kept, not
// silently):
// - R6-V2: an EMPTY inter_request_interval_ns cell on a turn>0 row is
//   legal "unset" here (yields 0) but the Python loader rejects the shape
//   (its timing derivation needs a value). Official products always write
//   the interval; hand-written inputs relying on the empty cell only pass
//   this reader.
// - R7-W2: no %1000 quantization gate on timing values (the Python loader
//   rejects non-multiples of 1000 ns). Official products are pre-quantized
//   by the materializer.
// - (2026-10-01, completing the list) two shapes where THIS reader is the
//   looser side, both kept for the in-repo fixtures that rely on them:
//   (a) a turn-0 row with a NON-EMPTY inter_request_interval_ns cell is
//       accepted (the value only rides the envelope; the Python loader
//       rejects it -- service_vacuum_test writes ",0" on turn-0 rows);
//   (b) an empty session_id on a LATER row legally opens a session block
//       named "" (the Python loader rejects any empty session_id).
//   Inputs relying on either shape pass this reader but not the Python
//   toolchain; new generators must not depend on them.
// The one documented exemption is the turn>0
// inter_request_interval_ns field: an EMPTY interval cell is the schema's
// legal "unset" (the loader semantic) and yields 0 without the lexicon; a
// NON-empty cell goes through the same fail-closed parse.
uint64_t parse_csv_u64(const std::string& value, const char* field,
                       const std::string& row_ctx) {
    const auto fail = [&value, &field, &row_ctx](const char* why) {
        reader_fatal("CSV field " + std::string(field) + " " + why + ": " +
                     value + " (" + row_ctx + ")");
    };
    if (value.empty()) {
        fail("is empty, expected a non-negative integer");
    }
    for (const char c : value) {
        if (c < '0' || c > '9') {
            fail("is not a non-negative integer (ASCII digits only)");
        }
    }
    errno = 0;
    char* end = nullptr;
    const unsigned long long parsed = std::strtoull(value.c_str(), &end, 10);
    if (errno == ERANGE || end != value.c_str() + value.size()) {
        fail("exceeds the 64-bit unsigned range");
    }
    return static_cast<uint64_t>(parsed);
}

int parse_csv_int(const std::string& value, const char* field,
                  const std::string& row_ctx) {
    // Same lexicon; the CSV schema only ever carries non-negative ints
    // here (turn_index), so parse unsigned and range-check into int.
    const uint64_t parsed = parse_csv_u64(value, field, row_ctx);
    if (parsed > static_cast<uint64_t>(std::numeric_limits<int>::max())) {
        reader_fatal("CSV field " + std::string(field) +
                     " exceeds the int range: " + value + " (" + row_ctx +
                     ")");
    }
    return static_cast<int>(parsed);
}

// Nearest-rank percentile (workspace convention: the ceil(rank*N)-th
// smallest value, 1-based; e.g. p99 of 496 samples = the 492nd) over a
// SORTED copy of vals.
uint64_t nearest_rank_percentile(std::vector<uint64_t>& vals, double rank) {
    if (vals.empty()) {
        return 0;
    }
    if (vals.size() == 1) {
        return vals.front();
    }
    const double ordinal =
        std::ceil(rank * static_cast<double>(vals.size()));
    const size_t index = static_cast<size_t>(ordinal) - 1;
    return vals[std::min(index, vals.size() - 1)];
}

}  // namespace

WindowedTraceReader::WindowedTraceReader(const std::string& csv_path,
                                         RequestIngress& ingress,
                                         const uint64_t max_arrival_ns)
    : csv_path_(csv_path),
      ingress_(ingress),
      max_arrival_ns_(max_arrival_ns) {
    if (csv_path.empty()) {
        // No --request-queue-csv: request-neutral IDLE start. The reader is
        // a no-op (already at EOF); the main entry handles the empty path.
        eof_ = true;
        indexed_ = true;
        return;
    }
    // A non-empty CSV is the production indexed-input path. Enable strict
    // one-shot resolution before reading any row so a future alarm cannot
    // silently fall back to queue_index=-1 merely because no turn>0 row has
    // been registered yet. Standalone no-CSV fixtures keep legacy mode.
    ingress_.enable_queue_index_tracking();
    // Binary mode: the index pass hashes and counts the RAW file bytes for
    // the provenance gate, and splits lines on '\n' itself.
    file_.open(csv_path, std::ios::binary);
    if (!file_.is_open()) {
        std::cerr << "[Error] (execution_driven/windowed_reader) cannot open "
                  << "--request-queue-csv: " << csv_path << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

size_t WindowedTraceReader::occupancy() const {
    // Outstanding = submitted-but-unfired turn-0 rows (P0 fix). Turn>0 rows
    // are fully pre-registered during the index pass and never occupy the
    // window; their future alarms consume the ingress one-shot entries, not
    // this set.
    return outstanding_rows_.size();
}

void WindowedTraceReader::run_index_pass() {
    // One streaming pass over the raw bytes: FNV-1a 64 + byte count over
    // every byte exactly once, lines split on '\n', trailing partial line
    // flushed at EOF. No Submit is produced here (the provenance gate must
    // be able to abort with zero submissions).
    const uint64_t start = now_ns();
    constexpr std::streamsize kChunk = 1 << 16;  // 64 KiB
    std::vector<char> chunk(static_cast<size_t>(kChunk));
    std::string pending;
    while (file_) {
        file_.read(chunk.data(), kChunk);
        const std::streamsize got = file_.gcount();
        if (got <= 0) {
            break;
        }
        prov_.csv_bytes += static_cast<uint64_t>(got);
        for (std::streamsize i = 0; i < got; ++i) {
            fnv1a64_ = fnv1a64_byte(fnv1a64_,
                                    static_cast<unsigned char>(chunk[i]));
        }
        pending.append(chunk.data(), static_cast<size_t>(got));
        std::string::size_type begin = 0;
        std::string::size_type nl = pending.find('\n', begin);
        while (nl != std::string::npos) {
            process_indexed_row(pending.substr(begin, nl - begin));
            begin = nl + 1;
            nl = pending.find('\n', begin);
        }
        pending.erase(0, begin);
    }
    // B6 fix (2026-09-28): a low-level read failure (badbit) leaves this
    // loop exactly like a normal EOF, and everything downstream -- the
    // provenance stats and the run-end completion-audit denominator --
    // would then describe a truncated prefix of the CSV as the whole
    // file. Fail loud instead.
    if (file_.bad()) {
        reader_fatal("I/O error while reading --request-queue-csv: " +
                     csv_path_);
    }
    if (!pending.empty()) {
        // Final line without a trailing newline.
        process_indexed_row(pending);
    }
    file_.close();
    // R3-J4 fix (2026-09-29): the Python loader rejects a queue that yields
    // zero requests ("must contain at least one request"); this reader used
    // to accept it and run a silent zero-request simulation (worse after
    // the header validation above: that path now fails loudly too, but an
    // all-comment/blank-row file would still index zero rows). Same
    // fail-closed contract here.
    if (data_rows_ == 0) {
        reader_fatal("request queue CSV contains zero data rows "
                     "(csv=" +
                     csv_path_ + ")");
    }
    prov_.fnv1a64 = fnv1a64_;
    prov_.data_rows = data_rows_;
    prov_.sessions = seen_sessions_.size();
    prov_.turn0_count = turn0_rows_;
    prov_.session_blocks_contiguous = true;  // violations abort in-row
    // The collector copies every AFTER_REQUEST parent queue index during
    // registration. Once this finite CSV is fully indexed no later row can
    // consult the per-session tails, so retaining them only extends their
    // lifetime to run end. The same holds for the M1-2 request-id
    // uniqueness set: the whole file is validated row-by-row in this pass.
    last_queue_index_by_session_.clear();
    seen_request_ids_.clear();

    // Turn-0 arrival calendar: stable sort by (arrival_ns, queue_index).
    // std::stable_sort + the explicit queue_index comparator make the
    // same-arrival group order DETERMINISTIC and equal to the old unbounded
    // arm's row-order insertion sequence (equivalence argument in the
    // header, item 4).
    std::stable_sort(calendar_.begin(), calendar_.end(),
                     [](const CalendarEntry& a, const CalendarEntry& b) {
                         if (a.arrival_ns != b.arrival_ns) {
                             return a.arrival_ns < b.arrival_ns;
                         }
                         return a.queue_index < b.queue_index;
                     });
    io_read_ns_ += now_ns() - start;

    check_provenance_sidecar();
    indexed_ = true;
}

void WindowedTraceReader::process_indexed_row(const std::string& line) {
    // One trailing '\r' is stripped from EVERY line up front (CRLF input;
    // the binary-mode '\n' split leaves it on). Doing this BEFORE the
    // blank-line check (2026-10-01) keeps a CRLF blank line ("\r")
    // skippable exactly like an LF blank line: pre-fix it fell through to
    // the header/data parse and died with a diagnosis unrelated to the
    // real cause (empty-header or empty-field fatals). The provenance
    // FNV/byte counters are computed over the raw chunk bytes before line
    // splitting, so this normalization does not touch them.
    std::string row_line = line;
    if (!row_line.empty() && row_line.back() == '\r') {
        row_line.pop_back();
    }
    if (row_line.empty()) {
        return;  // blank line: not a data row
    }
    if (!header_seen_) {
        header_seen_ = true;
        // R5-N1 fix (2026-09-29, aligned with the wscllm P2-9 canonical
        // check): the header is content-validated against the canonical
        // 8-column schema (kRequestQueueCsvHeader), not blindly skipped.
        // The pre-fix "first non-empty line is the header, whatever it
        // says" silently lost the first data row of a headerless input and
        // turned a single-line file into a zero-request empty run whose
        // run-end audit denominator was equally empty. One trailing '\r'
        // was already tolerated (stripped at entry above).
        // (2026-10-01) one leading UTF-8 BOM (EF BB BF) is stripped before
        // the comparison, matching the Python loader's utf-8-sig semantics:
        // a BOM used to fail the equality with a "got "..."" message whose
        // BOM renders zero-width, making got and expected look identical.
        // The provenance hash/byte counters stay on the RAW bytes (computed
        // before line splitting), so this is a pure parsing-layer
        // acceptance change; the materializer never writes a BOM.
        std::string header = row_line;
        if (header.size() >= 3 &&
            static_cast<unsigned char>(header[0]) == 0xEF &&
            static_cast<unsigned char>(header[1]) == 0xBB &&
            static_cast<unsigned char>(header[2]) == 0xBF) {
            header.erase(0, 3);
        }
        // Two legal header shapes (the joint repo's L11 contract): the
        // canonical 8-column header, and the legacy hand-written-fixture
        // 7-column shape (same names minus the description column -- the
        // ServiceVacuumTest contract). Anything else fails closed.
        const std::string canonical = kRequestQueueCsvHeader;
        const std::string legacy7 =
            canonical.substr(0, canonical.rfind(','));
        if (header != canonical && header != legacy7) {
            reader_fatal("request queue CSV header is not a documented "
                         "header (expected the canonical 8-column header \"" +
                         canonical + "\" or the legacy 7-column shape \"" +
                         legacy7 + "\", got \"" + header + "\"): " +
                         csv_path_);
        }
        return;
    }
    // Data rows: the trailing '\r' was already stripped at entry (the
    // 2026-09-30 per-branch strip is subsumed by it; a legacy 7-column
    // CRLF data row used to parse inter_request_interval_ns as "...\r"
    // and die in the ASCII-digit check with a message unrelated to the
    // real cause).
    std::istringstream row(row_line);
    // Columns 1-7 are consumed below; column 8 ("description") and beyond
    // are documentation-only by schema and deliberately unread (R5-N3
    // note: the Python materializer writes the description column, this
    // reader never consumes it).
    std::string session_id, turn_index_s, request_id, prefill_s, decode_s,
        arrival_s, interval_s;
    std::getline(row, session_id, ',');
    std::getline(row, turn_index_s, ',');
    std::getline(row, request_id, ',');
    std::getline(row, prefill_s, ',');
    std::getline(row, decode_s, ',');
    std::getline(row, arrival_s, ',');
    std::getline(row, interval_s, ',');
    // R3-J1 alignment (2026-09-29): the Python loader skips rows whose
    // request_id starts with '#' (comment convention); this reader counted
    // them as data rows, off-by-one-ing every later queue_index in a file
    // that mixes both conventions. Skip before the frozen queue_index is
    // assigned.
    if (!request_id.empty() && request_id[0] == '#') {
        return;
    }
    // Field-count gate (2026-10-01): the seven consumed columns must EXIST
    // as columns. getline at end-of-stream yields empty strings, so a short
    // row (fewer than 7 columns) was indistinguishable from legal empty
    // CELLS and passed every downstream check -- e.g. a 5-column turn>0
    // row silently registered with interval=0. Both documented schemas
    // (canonical 8-column, legacy 7-column) carry at least 6 commas; the
    // unread description column may legally contain more.
    if (std::count(row_line.begin(), row_line.end(), ',') < 6) {
        reader_fatal(
            "request queue CSV data row has fewer than 7 columns (trailing "
            "cells may be empty, but the columns must exist): data_row=" +
            std::to_string(data_rows_) + " (csv=" + csv_path_ + ")");
    }
    ++data_rows_;
    // Frozen queue index (CSV data-row order, 0-based). Turn-0 carries it
    // directly in the Submit envelope. Turn>0 rows are ALL pre-registered
    // here during the index pass (O(rows), same order as the metrics side;
    // P0 fix note: this is a full pre-registration, no longer bounded by a
    // row window); schedule_future_arrival consumes the entries one-shot.
    const int64_t queue_index = static_cast<int64_t>(data_rows_) - 1;
    // Row context for the structured numeric-parse diagnostics below.
    const std::string row_ctx =
        "csv=" + csv_path_ + " data_row=" + std::to_string(queue_index);

    RequestEnvelope env;
    env.session_id = session_id;
    env.request_id = request_id;
    env.turn_index = parse_csv_int(turn_index_s, "turn_index", row_ctx);
    env.prefill_length = parse_csv_u64(prefill_s, "prefill_length", row_ctx);
    env.decode_length = parse_csv_u64(decode_s, "decode_length", row_ctx);
    env.inter_request_interval_ns =
        interval_s.empty()
            ? 0
            : parse_csv_u64(interval_s, "inter_request_interval_ns",
                            row_ctx);
    env.queue_index = queue_index;

    const bool is_turn0 = !arrival_s.empty();

    // ---- fail-closed session-block structure validation (P0 fix item 2) ----
    // Input-named guard: before the very first session block opens, an empty
    // session_id is a CSV defect. Without this, "" != current_session_ is
    // false (both empty) and the row fell through to the "internal: block
    // state lost" fatal below -- misreporting an input problem as an engine
    // invariant loss. An empty session_id in a LATER row still opens a legal
    // block, exactly as before.
    if (!have_current_session_) {
        if (session_id.empty()) {
            reader_fatal("first data row has an empty session_id (csv=" +
                         csv_path_ + " data_row=" + std::to_string(queue_index) +
                         ")");
        }
        have_current_session_ = true;
    }
    if (session_id != current_session_) {
        if (seen_sessions_.count(session_id) != 0) {
            reader_fatal(
                "session block is not contiguous in the CSV: session_id=" +
                session_id + " reappears at data row " +
                std::to_string(queue_index) + " (csv=" + csv_path_ + ")");
        }
        seen_sessions_.insert(session_id);
        current_session_ = session_id;
        in_block_ = true;
        expected_next_turn_ = 0;
    }
    if (!in_block_) {
        // Defensive: a session change always opens a block above.
        reader_fatal("internal: block state lost while reading session_id=" +
                     session_id);
    }
    if (env.turn_index != expected_next_turn_) {
        reader_fatal("turn_index must increase by exactly 1 within a session "
                     "block: session_id=" +
                     session_id + " data row " +
                     std::to_string(queue_index) + " has turn_index " +
                     std::to_string(env.turn_index) + ", expected " +
                     std::to_string(expected_next_turn_) + " (csv=" +
                     csv_path_ + ")");
    }
    ++expected_next_turn_;
    if (env.turn_index == 0 && !is_turn0) {
        reader_fatal("turn-0 row (turn_index=0) has an empty "
                     "session_arrival_time_ns: session_id=" +
                     session_id + " request_id=" + request_id + " (csv=" +
                     csv_path_ + ")");
    }
    if (env.turn_index > 0 && is_turn0) {
        reader_fatal("turn>0 row has an explicit session_arrival_time_ns "
                     "(only turn-0 rows carry absolute arrivals): session_id=" +
                     session_id + " request_id=" + request_id +
                     " turn_index=" + std::to_string(env.turn_index) +
                     " (csv=" + csv_path_ + ")");
    }
    // M1-2 (2026-09-26): request_id uniqueness, CLASS-AGNOSTIC --
    // root-cause interception. Both duplicate Submits are accounted by
    // RequestIngress/ServiceCoordinator, but the DecisionMailbox's
    // same-epoch same-identity dedup silently drops the second ARRIVAL:
    // C++ and Python bookkeeping diverge permanently and the run only
    // aborts much later at the unrelated lost-wakeup dead-end. The
    // turn>0-side ingress one-shot queue-index registration cannot cover
    // duplicates ACROSS the turn classes either (a turn-0 id never enters
    // queue_index_map_, and the duplicate turn>0 ARRIVAL lands in a later
    // dedup epoch). Fail closed here at the input boundary instead, for
    // every row.
    if (request_id.empty()) {
        // Empty-id symmetry (2026-09-30): the turn>0 side already rejects
        // an empty request_id at RequestIngress registration, and the
        // reference loader (generate_trace.py) synthesizes ids instead of
        // accepting empties -- an empty turn-0 id would enter the calendar,
        // the ARRIVAL mailbox and the Python side as "" and silently
        // contaminate identity bookkeeping.
        reader_fatal("empty request_id in the CSV: session_id=" + session_id +
                     " data row " + std::to_string(queue_index) +
                     " (csv=" + csv_path_ + ")");
    }
    if (!seen_request_ids_.insert(request_id).second) {
        reader_fatal("duplicate request_id in the CSV: request_id=" +
                     request_id + " session_id=" + session_id +
                     " data row " + std::to_string(queue_index) +
                     " (csv=" + csv_path_ + ")");
    }

    // Dynamic request registration (every data row, turn-0 and skipped
    // turn>0 alike, so the request state exists for the node/rank anchors
    // registered at commit time). The online CSV row is authoritative for
    // arrival/session fields. No-op when metrics are disabled. This keeps
    // the phase-1 loader's call order (row order) -- the index pass never
    // reorders the registrations.
    if (MetricCollector::instance().enabled()) {
        if (!is_turn0) {
            // Turn>0 row: AFTER_REQUEST, parent = previous same-session row
            // (the future_alarm scheduling path's parent). Block
            // contiguity (validated above) guarantees the parent was
            // already registered.
            const auto parent = last_queue_index_by_session_.find(session_id);
            if (parent == last_queue_index_by_session_.end()) {
                reader_fatal(
                    "turn>0 metrics row has no preceding same-session row: "
                    "session_id=" +
                    session_id + " request_id=" + request_id);
            }
            MetricCollector::instance().online_register_request(
                queue_index, request_id, session_id, env.turn_index,
                /*absolute_arrival=*/false, 0,
                parent->second,
                env.inter_request_interval_ns);
        } else {
            MetricCollector::instance().online_register_request(
                queue_index, request_id, session_id, env.turn_index,
                /*absolute_arrival=*/true,
                parse_csv_u64(arrival_s, "session_arrival_time_ns", row_ctx),
                /*arrival_parent_queue_index=*/-1, 0);
        }
        last_queue_index_by_session_[session_id] = queue_index;
    }

    ++rows_read_;
    if (!is_turn0) {
        // Turn>0 row: the future_alarm path schedules this arrival from the
        // REQUEST_COMPLETE commit. Never submit it directly. Full
        // pre-registration happens here (P0 fix item 1).
        ingress_.register_queue_index(request_id, queue_index);
        return;
    }

    // Turn-0 row (arrival column non-empty): collected for the calendar,
    // counted separately for the run-end accepted-accounting invariant
    // (accepted + dropped == turn-0 rows; backport fix 2026-08-16).
    ++turn0_rows_;
    const uint64_t arrival_ns =
        parse_csv_u64(arrival_s, "session_arrival_time_ns", row_ctx);
    env.arrival_world_ns = arrival_ns;

    if (!have_prev_turn0_) {
        prov_.turn0_arrival_min_ns = arrival_ns;
        prov_.turn0_arrival_max_ns = arrival_ns;
        have_prev_turn0_ = true;
    } else {
        if (arrival_ns < prov_.turn0_arrival_min_ns) {
            prov_.turn0_arrival_min_ns = arrival_ns;
        }
        if (arrival_ns > prov_.turn0_arrival_max_ns) {
            prov_.turn0_arrival_max_ns = arrival_ns;
        }
        if (arrival_ns < prev_turn0_arrival_) {
            // File-order adjacent inversion: exactly the quantity that made
            // the old row-order window reader discover early arrivals too
            // late (250 on the full TraceLab queue).
            ++prov_.turn0_adjacent_inversions;
        }
    }
    prev_turn0_arrival_ = arrival_ns;

    CalendarEntry entry;
    entry.arrival_ns = arrival_ns;
    entry.queue_index = queue_index;
    entry.envelope = std::move(env);
    calendar_.push_back(std::move(entry));
}

void WindowedTraceReader::check_provenance_sidecar() {
    // Fail-closed provenance gate (P0 fix item 6). Runs AFTER the index
    // stats are complete and BEFORE the first Submit: any mismatch aborts
    // the process with zero submissions.
    // P1 fix (2026-09-28, ported to this repo 2026-10-07): an ABSENT
    // sidecar is now also fail-closed. The former "absent (no gate)"
    // pass-through was exactly the hole a failed materialization falls
    // through: derive_20_first_30_seconds.py used to write the queue
    // row-by-row straight to its final path and the sidecar only after
    // ALL rows succeeded, so a mid-derivation abort left a TRUNCATED
    // queue on disk with no sidecar -- this reader then indexed the
    // shrunken input, the run-end completion audit passed on the
    // truncated denominators, and the simulation ran "green" on silently
    // missing input. (The materializer now publishes queue+sidecar
    // atomically via temp+rename, and this gate refuses any queue that
    // arrives without its provenance record.)
    const std::string sidecar = csv_path_ + ".provenance.json";
    std::ifstream in(sidecar, std::ios::binary);
    if (!in.is_open()) {
        reader_fatal(
            "provenance sidecar is MISSING (fail-closed): " + sidecar +
            " must sit next to the request queue CSV and match it "
            "(re-materialize with "
            "sh_test_mesh/workload/llama2_7b_inference/traces/"
            "derive_20_first_30_seconds.py; a queue without its provenance "
            "record cannot be told apart from a truncated one -- refusing "
            "to submit anything)");
    }
    nlohmann::json doc;
    try {
        in >> doc;
    } catch (const std::exception& e) {
        reader_fatal("provenance sidecar is not valid JSON: " + sidecar +
                     " (" + e.what() + ")");
    }
    auto expect_u64 = [&](const char* field, uint64_t computed) {
        if (!doc.contains(field) || !doc[field].is_number_unsigned()) {
            reader_fatal(std::string("provenance sidecar field missing or not "
                                     "an unsigned integer: ") +
                         field + " (" + sidecar + ")");
        }
        const uint64_t claimed = doc[field].get<uint64_t>();
        if (claimed != computed) {
            reader_fatal("provenance sidecar mismatch on '" +
                         std::string(field) + "': sidecar=" +
                         std::to_string(claimed) + " csv=" +
                         std::to_string(computed) + " (" + sidecar +
                         "); the queue CSV and its provenance record "
                         "disagree -- refusing to submit anything");
        }
    };
    if (!doc.contains("schema") || !doc["schema"].is_number_unsigned() ||
        doc["schema"].get<uint64_t>() != 1) {
        reader_fatal("provenance sidecar schema must be 1: " + sidecar);
    }
    expect_u64("csv_fnv1a64", prov_.fnv1a64);
    expect_u64("csv_bytes", prov_.csv_bytes);
    expect_u64("data_rows", prov_.data_rows);
    expect_u64("sessions", prov_.sessions);
    expect_u64("turn0_count", prov_.turn0_count);
    expect_u64("turn0_arrival_min_ns", prov_.turn0_arrival_min_ns);
    expect_u64("turn0_arrival_max_ns", prov_.turn0_arrival_max_ns);
    expect_u64("turn0_adjacent_inversions", prov_.turn0_adjacent_inversions);
    if (!doc.contains("session_blocks_contiguous") ||
        !doc["session_blocks_contiguous"].is_boolean() ||
        !doc["session_blocks_contiguous"].get<bool>()) {
        // The reader aborts on any non-contiguous input while indexing, so
        // reaching here means the CSV side is true; a sidecar claiming
        // false (or missing the flag) is a mismatch.
        reader_fatal(
            "provenance sidecar session_blocks_contiguous must be true: " +
            sidecar);
    }
    std::cout << "[online] provenance sidecar: matched (fnv1a64=0x" <<
        std::hex << prov_.fnv1a64 << std::dec << " bytes=" << prov_.csv_bytes
              << " data_rows=" << prov_.data_rows
              << " sessions=" << prov_.sessions << "): " << sidecar
              << std::endl;
}

void WindowedTraceReader::submit_from_calendar() {
    // Walk the calendar in arrival order (ties by queue_index). The
    // max_arrival_ns rejection keeps its frozen counting semantics
    // (counted, never submitted, visible at report time, fail-closes the
    // run-end completion audit); a rejected row never enters the
    // outstanding set. Ingress backpressure (full command queue) parks the
    // cursor; the next pump() resumes exactly here.
    while (calendar_cursor_ < calendar_.size()) {
        CalendarEntry& entry = calendar_[calendar_cursor_];
        if (max_arrival_ns_ > 0 && entry.arrival_ns > max_arrival_ns_) {
            ++rejected_out_of_range_;
            entry.rejected = true;
            ++calendar_cursor_;
            continue;
        }
        IngressCommand cmd;
        cmd.kind = IngressCommandKind::Submit;
        cmd.envelope = entry.envelope;
        cmd.envelope.source = RequestSource::StaticCsv;
        cmd.backpressure_deferred = entry.backpressure_deferred;
        if (!ingress_.enqueue_command(cmd)) {
            // false = queue full (backpressure) or input already closed
            // (RequestIngress::enqueue_command has both early-returns).
            // Under the official ordering -- mark_input_closed only after
            // pump() reaches EOF -- only backpressure is reachable, so this
            // is a park, NOT a failure: the cursor stays on this entry and
            // the next pump retries. (A hypothetical close-before-drain
            // would be caught by the run-end completion audit.)
            // M5: mark the park so the eventual drain clamp is disclosed as
            // a capacity artifact and exempted from the run-end
            // static-arrival gate.
            entry.backpressure_deferred = true;
            return;
        }
        entry.submitted = true;
        entry.discovered_tick = ingress_.current_time();
        outstanding_rows_.insert(entry.queue_index);
        ++calendar_cursor_;
    }
}

bool WindowedTraceReader::pump() {
    if (eof_) {
        return false;
    }
    bool did_work = false;
    if (!indexed_) {
        run_index_pass();
        did_work = true;
    }
    const size_t before = calendar_cursor_;
    submit_from_calendar();
    if (calendar_cursor_ != before) {
        did_work = true;
    }
    if (did_work) {
        ++read_pumps_;
    }
    if (occupancy() > peak_occupancy_) {
        peak_occupancy_ = occupancy();
    }
    // eof = fully indexed AND every calendar entry submitted or rejected.
    // The producer-close boundary in main_online relies on this: no later
    // CSV Submit may be rejected by a closed ingress gate.
    if (indexed_ && calendar_cursor_ == calendar_.size()) {
        eof_ = true;
    }
    return !eof_;
}

void WindowedTraceReader::notify_consumed(const int64_t queue_index) {
    // Duplicate notifications for an already-consumed row are harmless; an
    // unknown future index is fail-closed because it would corrupt the
    // accounting. Turn>0 future-alarm notifications erase nothing from the
    // outstanding set (turn>0 rows never entered it after the P0 fix).
    if (queue_index >= static_cast<int64_t>(rows_read_)) {
        std::cerr << "[Error] (execution_driven/windowed_reader) consumed "
                     "queue index was never read: "
                  << queue_index << " rows_read=" << rows_read_ << std::endl;
        std::abort();
    }
    outstanding_rows_.erase(queue_index);
}

std::vector<WindowedTraceReader::ArrivalAuditEntry>
WindowedTraceReader::arrival_audit() const {
    const auto effective = ingress_.static_csv_arrivals();
    std::vector<ArrivalAuditEntry> audit;
    audit.reserve(calendar_.size());
    for (const CalendarEntry& entry : calendar_) {
        if (!entry.submitted && !entry.rejected) {
            // Not yet in the simulation (backpressure-parked cursor). A row
            // still parked at RUN END is caught by the completion audit
            // instead (accepted + dropped != turn-0 rows).
            continue;
        }
        ArrivalAuditEntry row;
        row.queue_index = entry.queue_index;
        row.declared_arrival_ns = entry.arrival_ns;
        row.reader_discovered_tick = entry.discovered_tick;
        const auto it = effective.find(entry.queue_index);
        if (it == effective.end()) {
            if (entry.rejected) {
                // Rejected out-of-range rows are never submitted: no
                // effective arrival exists by design. Excluded from the
                // delay statistics (they never entered the simulation).
                continue;
            }
            // A submitted turn-0 without an ingress record means its Submit
            // never drained -- impossible on a completed run; the gate
            // treats this as a failure (late_source stays
            // "no_effective_record", late_by_ns 0).
            audit.push_back(row);
            continue;
        }
        row.effective_arrival_ns = it->second.effective_arrival_ns;
        row.late_by_ns = row.effective_arrival_ns - row.declared_arrival_ns;
        if (row.declared_arrival_ns == 0 && row.reader_discovered_tick == 0 &&
            row.effective_arrival_ns == 1) {
            row.late_source = "t0_boundary";
        } else if (row.late_by_ns > 0) {
            row.late_source = entry.backpressure_deferred
                ? "capacity_backpressure"
                : "static_submit_late";
        } else {
            row.late_source = "on_time";
        }
        row.backpressure_deferred = entry.backpressure_deferred;
        audit.push_back(row);
    }
    std::sort(audit.begin(), audit.end(),
              [](const ArrivalAuditEntry& a, const ArrivalAuditEntry& b) {
                  return a.queue_index < b.queue_index;
              });
    return audit;
}

WindowedTraceReader::ArrivalAuditSummary
WindowedTraceReader::audit_static_arrivals() const {
    ArrivalAuditSummary summary;
    summary.late_static_submit = ingress_.late_static_submit_count();
    summary.late_external_stream = ingress_.late_external_stream_count();
    summary.late_future_alarm_rounding =
        ingress_.late_future_alarm_rounding_count();
    summary.t0_boundary_clamp = ingress_.t0_boundary_clamp_count();
    summary.late_static_submit_backpressure =
        ingress_.late_static_submit_backpressure_count();

    std::vector<uint64_t> delays;
    delays.reserve(calendar_.size());
    for (const ArrivalAuditEntry& row : arrival_audit()) {
        if (std::string(row.late_source) == "no_effective_record") {
            summary.gate_ok = false;
            summary.gate_why =
                "turn-0 queue_index " + std::to_string(row.queue_index) +
                " has no effective-arrival record (Submit never drained)";
            continue;
        }
        // M5: capacity-backpressure rows carry late_by_ns > 0 but are a
        // bounded-queue artifact, not reader lateness -- disclosed, not
        // gated (mirrors the t0-boundary exemption).
        if (std::string(row.late_source) == "static_submit_late") {
            ++summary.nonzero_delay_rows;
        }
        ++summary.turn0_submitted;
        delays.push_back(row.late_by_ns);
    }
    std::sort(delays.begin(), delays.end());
    summary.delay_p50_ns = nearest_rank_percentile(delays, 0.50);
    summary.delay_p99_ns = nearest_rank_percentile(delays, 0.99);
    summary.delay_max_ns = delays.empty() ? 0 : delays.back();

    if (summary.gate_ok && summary.late_static_submit != 0) {
        summary.gate_ok = false;
        summary.gate_why =
            "late_static_submit=" + std::to_string(summary.late_static_submit) +
            " (a static-CSV turn-0 was submitted after its declared arrival)";
    }
    if (summary.gate_ok && summary.nonzero_delay_rows != 0) {
        summary.gate_ok = false;
        summary.gate_why =
            "turn-0 rows with ingress_delay>0 (excluding t0 boundary): " +
            std::to_string(summary.nonzero_delay_rows);
    }
    if (summary.gate_ok) {
        summary.gate_why = "ok";
    }
    return summary;
}

void WindowedTraceReader::report(std::ostream& os) const {
    os << "[online] windowed reader: max_arrival_ns=" << max_arrival_ns_
       << (max_arrival_ns_ == 0 ? " (unbounded)" : "") << " rows=" << rows_read_
       << " total_rows=" << total_data_rows()
       << " turn0_rows=" << turn0_data_rows()
       << " sessions=" << prov_.sessions
       << " turn0_arrival=[min=" << prov_.turn0_arrival_min_ns
       << " max=" << prov_.turn0_arrival_max_ns << "]"
       << " turn0_file_order_inversions=" << prov_.turn0_adjacent_inversions
       << " rejected_out_of_range=" << rejected_out_of_range_
       << " read_pumps=" << read_pumps_
       << " peak_outstanding_turn0=" << peak_occupancy_
       << " io_read_ns=" << io_read_ns_;
    if (io_read_ns_ > 0) {
        os << " read_throughput_rows_per_s="
           << (static_cast<double>(rows_read_) * 1e9 /
               static_cast<double>(io_read_ns_));
    }
    os << std::endl;
}

CompletionAuditVerdict audit_completion(const CompletionAuditCounts& counts) {
    // Backport fix (2026-08-16, sh_2.0测试 §5.1). The OLD audit
    // (completed == rows-the-window-read) let a stalled/dropping run PASS
    // silently: with the 30e9 default and a longer input, over-window
    // requests were rejected, the window stalled on the un-consumable
    // rejects, EOF (and with it the expected-rows bookkeeping) was never
    // reached, and the audit was skipped entirely -- exit 0 "PASS" with
    // 12.5% of the input silently missing (sh_2.0 strategy-20 measured:
    // 2091 input rows, 1830 completed). Fail-closed order: drops first (the
    // loudest, most actionable), then the turn-0 accepted accounting, then
    // the completed-vs-total shortfall.
    if (counts.dropped > 0) {
        return CompletionAuditVerdict::Dropped;
    }
    if (counts.accepted + counts.dropped != counts.turn0_rows) {
        return CompletionAuditVerdict::AccountMismatch;
    }
    if (counts.completed + counts.dropped != counts.total_rows) {
        return CompletionAuditVerdict::Incomplete;
    }
    return CompletionAuditVerdict::Ok;
}

const char* completion_audit_why(const CompletionAuditVerdict verdict) {
    switch (verdict) {
        case CompletionAuditVerdict::Ok:
            return "ok";
        case CompletionAuditVerdict::Dropped:
            return "input rows rejected by the explicit arrival window "
                   "(dropped_out_of_range > 0; visible drop, fail-closed)";
        case CompletionAuditVerdict::AccountMismatch:
            return "accepted + dropped != turn-0 data rows (a turn-0 row "
                   "was neither accepted by the service nor rejected)";
        case CompletionAuditVerdict::Incomplete:
            return "completed + dropped != total data rows (requests "
                   "missing from the run)";
    }
    return "unknown";
}

}  // namespace ExecutionDriven
}  // namespace AstraSim
