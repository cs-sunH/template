/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

WindowedTraceReader -- execution-driven mechanism layer (wscllm phase 7 §10.4;
P0 turn-0 late-discovery fix 2026-08-30: index pass + turn-0 arrival calendar).
Implementation. Row semantics: same CSV schema, same turn-0 Submit /
turn>0 future-alarm split, same queue_index and metrics registration order
as the phase-1 full loader. Live equivalence anchors (that loader itself is
gone): WindowedTraceReader.hh item 4 (equivalence argument),
tests/windowed_trace_reader_test.cc (Part B row-semantics assertions) and
tests/calendar_reader_oracle_test.cc (LegacyOracleWindowedTraceReader
comparison arm).
The P0 fix changes only WHEN/HOW turn-0 rows are SUBMITTED: never by file
position -- always by the arrival calendar built by the streaming index pass
(see WindowedTraceReader.hh for the full contract and equivalence argument).
*******************************************************************************/

#include "astra-sim/workload/execution_driven/WindowedTraceReader.hh"

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <set>
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
constexpr uint64_t kFnv1a64Prime = 1099511628211ULL;

inline uint64_t fnv1a64_byte(uint64_t h, unsigned char b) {
    return (h ^ static_cast<uint64_t>(b)) * kFnv1a64Prime;
}

[[noreturn]] void reader_fatal(const std::string& what) {
    std::cerr << "[Error] (execution_driven/windowed_reader) " << what
              << std::endl;
    std::exit(EXIT_FAILURE);
}

// Strict numeric field parser for CSV rows (M10 fix). The request-queue
// schema writes every numeric column as a non-negative integer, so a value
// must be non-empty ASCII digits (the online FP1 frozen lexicon -- rejects
// the " -1" / "+1" shapes strtoull would wrap around) and must parse to its
// last character with no range overflow. Any other shape -- empty field,
// sign, whitespace, trailing garbage such as '1000x' -- fail-closes here
// via reader_fatal: the former raw std::stoi/stoull calls silently
// truncated trailing garbage into the accounting or threw an uncaught
// exception past the pump() callers (std::terminate instead of this file's
// [Error]/exit contract). Narrower type bounds (turn_index's int domain)
// are checked at the call site.
uint64_t parse_u64_field(const char* field_name, const std::string& value,
                         const std::string& session_id,
                         const int64_t queue_index,
                         const std::string& csv_path) {
    bool digits_only = !value.empty();
    for (const char c : value) {
        if (c < '0' || c > '9') {
            digits_only = false;
            break;
        }
    }
    if (digits_only) {
        errno = 0;
        char* end = nullptr;
        const unsigned long long parsed =
            std::strtoull(value.c_str(), &end, 10);
        if (errno != ERANGE && end == value.c_str() + value.size()) {
            return static_cast<uint64_t>(parsed);
        }
    }
    reader_fatal(std::string("malformed CSV field ") + field_name +
                 " (expected a non-negative integer): \"" + value +
                 "\" (session_id=" + session_id + " data row " +
                 std::to_string(queue_index) + " csv=" + csv_path + ")");
}

// Strict JSON load for the provenance sidecar: istream operator>> parses
// in NON-strict mode -- it silently consumes the FIRST JSON value and
// leaves trailing bytes unread, and the DOM build last-wins on duplicate
// object keys. The sidecar is machine-written by the paired materializer,
// so trailing content or a duplicate key is producer corruption, not a
// legal shape. json::parse in its default strict mode rejects trailing
// bytes outright; the stateful parse callback rejects a repeated key
// inside one object.
nlohmann::json strict_json_load(std::istream& in, const std::string& what) {
    struct KeyGuard {
        std::vector<std::set<std::string>> stack;
        std::string duplicate;
        bool operator()(int, nlohmann::json::parse_event_t event,
                        nlohmann::json& parsed) {
            using EVT = nlohmann::json::parse_event_t;
            if (event == EVT::object_start) {
                stack.emplace_back();
            } else if (event == EVT::key) {
                if (!stack.back().insert(parsed.get<std::string>()).second) {
                    duplicate = parsed.get<std::string>();
                }
            } else if (event == EVT::object_end && !stack.empty()) {
                stack.pop_back();
            }
            return true;  // observe only; never discard subtree
        }
    } guard;
    nlohmann::json value;
    try {
        value = nlohmann::json::parse(in, guard);
    } catch (const nlohmann::json::exception& e) {
        reader_fatal("invalid JSON in " + what + ": " + e.what());
    }
    if (!guard.duplicate.empty()) {
        reader_fatal("duplicate JSON key \"" + guard.duplicate + "\" in " +
                     what);
    }
    return value;
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
    // Fail-closed on a mid-index I/O error (fix 2026-09-27): the read loop
    // above exits on ANY stream-state failure -- eofbit+failbit at the true
    // EOF, but equally badbit on a disk error / short read. Without this
    // check a mid-file I/O error is silently treated as a clean EOF: the
    // digest and the calendar describe a truncated input that is internally
    // self-consistent, and the run-end completion audit passes on the
    // truncated denominators (the provenance sidecar gate is absent on
    // hand-written fixtures, so nothing else catches it).
    if (file_.bad()) {
        reader_fatal("I/O error while indexing the request queue CSV "
                     "(truncated index, refusing to submit): " +
                     csv_path_ + " (bytes indexed=" +
                     std::to_string(prov_.csv_bytes) + ")");
    }
    if (!pending.empty()) {
        // Final line without a trailing newline.
        process_indexed_row(pending);
    }
    file_.close();
    // P2-9 fix (2026-09-28): a non-empty path that yielded zero lines has
    // no header at all -- the same silent zero-denominator family as the
    // skipped first data row (expected_requests would stay 0 and every
    // gate would run green on an empty run). Fail-closed; the empty PATH
    // (request-neutral no-CSV contract) never reaches the index pass.
    if (!header_seen_) {
        reader_fatal("request queue CSV is empty (no header row): " +
                     csv_path_);
    }
    prov_.fnv1a64 = fnv1a64_;
    prov_.data_rows = data_rows_;
    prov_.sessions = seen_sessions_.size();
    prov_.turn0_count = turn0_rows_;
    prov_.session_blocks_contiguous = true;  // violations abort in-row
    // The collector copies every AFTER_REQUEST parent queue index during
    // registration. Once this finite CSV is fully indexed no later row can
    // consult the per-session tails, so retaining them only extends their
    // lifetime to run end.
    last_queue_index_by_session_.clear();

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

    // R3-J4 port (2026-09-30, from the face repo, aligned with joint): the
    // Python loader (generate_trace.py load_request_queue) rejects a queue
    // that yields zero requests ("must contain at least one request");
    // this reader used to accept one and run a silent zero-request
    // simulation (an all-comment/blank-row file indexes zero rows and the
    // run-end gates all key off expected_requests > 0, so they pass
    // vacuously). Same fail-closed contract on this side; the no-CSV
    // request-neutral mode never reaches this point (the constructor marks
    // it indexed and at EOF).
    if (data_rows_ == 0) {
        reader_fatal("request queue CSV contains zero data rows "
                     "(csv=" + csv_path_ + ")");
    }

    check_provenance_sidecar();
    indexed_ = true;
}

void WindowedTraceReader::process_indexed_row(const std::string& line) {
    if (line.empty()) {
        return;  // blank line: not a data row
    }
    if (!header_seen_) {
        header_seen_ = true;
        // P2-9 fix (2026-09-28): the header is content-validated against
        // the canonical 8-column schema (kRequestQueueCsvHeader), not just
        // blindly skipped. The reader parses every data row POSITIONALLY,
        // so the pre-fix "first non-empty line is the header, whatever it
        // says" behavior silently re-mapped the schema on any hand-written
        // or drifted input: a header-less CSV lost its first data row to
        // the skip (expected/total/audit all derive from the same
        // truncated read, so every downstream gate stayed green), and a
        // renamed/reordered header re-bound the columns with zero
        // diagnostics. Fail-closed, in this file's standard style, before
        // any Submit exists. One trailing '\r' is tolerated (CRLF input;
        // the binary-mode split leaves it on the line).
        std::string header = line;
        if (!header.empty() && header.back() == '\r') {
            header.pop_back();
        }
        if (header != kRequestQueueCsvHeader) {
            reader_fatal("request queue CSV header is not the canonical "
                         "8-column header (expected \"" +
                         std::string(kRequestQueueCsvHeader) +
                         "\", got \"" + header + "\"): " + csv_path_);
        }
        return;
    }
    std::istringstream row(line);
    std::string session_id, turn_index_s, request_id, prefill_s, decode_s,
        arrival_s, interval_s;
    std::getline(row, session_id, ',');
    std::getline(row, turn_index_s, ',');
    std::getline(row, request_id, ',');
    std::getline(row, prefill_s, ',');
    std::getline(row, decode_s, ',');
    std::getline(row, arrival_s, ',');
    std::getline(row, interval_s, ',');
    // Column-shape gate: the reader parses POSITIONALLY, so a row with a
    // missing or extra unquoted comma silently re-maps every later field
    // (e.g. a request_id containing an unquoted comma shifts turn fields
    // into numeric slots or vacates the interval column, whose empty
    // default then moved the next arrival to parent-complete +0). The
    // canonical schema has exactly 8 top-level fields; the 8th
    // (description) is free text ignored here and may ride a trailing
    // '\r' (CRLF input). A 9th top-level field -- a quoted CSV field with
    // an embedded comma, which this reader never supported -- is
    // fail-closed instead of silently absorbed. (2026-10-01 update: a
    // trailing '\r' or any structurally short row now dies
    // at the lower-bound comma gate below, BEFORE field parsing --
    // the older note here said it landed in interval_s and died in
    // parse_u64_field, which is no longer the first gate reached.)
    std::string description_ride;
    // Read the REMAINDER of the line as the 8th (description) field, then
    // reject it if it still contains a comma (= a 9th top-level field).
    // NOTE: `std::getline(row, s, ',')` here would be WRONG -- on an
    // exactly-8-field line the 8th getline consumes "d" and hits EOF, which
    // sets eofbit WITHOUT failbit, so the stream's bool conversion is true
    // and every well-formed row would be rejected.
    std::getline(row, description_ride);
    if (description_ride.find(',') != std::string::npos) {
        reader_fatal("CSV data row has more than the canonical 8 columns "
                     "(unquoted commas inside a field are not supported): "
                     "session_id=" +
                     session_id + " data row " +
                     std::to_string(data_rows_) + " (csv=" + csv_path_ + ")");
    }
    // Lower-bound column gate (2026-10-01): the header is pinned to exactly
    // 8 canonical columns, but a TRUNCATED row (a torn tail line from an
    // interrupted write, or a hand-edited file) used to pass with its
    // trailing columns silently defaulted to empty -- e.g. a 6-column row
    // read interval="" and description="". The canonical materializer always
    // writes exactly 8 columns, so anything under 8 top-level fields is a
    // malformed row and fails closed here, in the same family as the P2-9
    // header gate above. The raw line must carry at least the 7 commas that
    // separate the 8 canonical fields.
    {
        const std::size_t commas =
            std::count(line.begin(), line.end(), ',');
        if (commas < 7) {
            reader_fatal(
                "CSV data row has fewer than the canonical 8 columns "
                "(found " +
                std::to_string(commas + 1) + " of 8; truncated or "
                "hand-edited rows are rejected): session_id=" +
                session_id + " data row " + std::to_string(data_rows_) +
                " (csv=" + csv_path_ + ")");
        }
    }
    ++data_rows_;
    // Frozen queue index (CSV data-row order, 0-based). Turn-0 carries it
    // directly in the Submit envelope. Turn>0 rows are ALL pre-registered
    // here during the index pass (O(rows), same order as the metrics side;
    // P0 fix note: this is a full pre-registration, no longer bounded by a
    // row window); schedule_future_arrival consumes the entries one-shot.
    // Hoisted above the field parsing so strict-parse diagnostics can cite
    // the row.
    const int64_t queue_index = static_cast<int64_t>(data_rows_) - 1;

    RequestEnvelope env;
    env.session_id = session_id;
    const uint64_t turn_index_parsed =
        parse_u64_field("turn_index", turn_index_s, session_id, queue_index,
                        csv_path_);
    if (turn_index_parsed > 2147483647ULL) {
        reader_fatal("CSV field turn_index exceeds the int range: " +
                     std::to_string(turn_index_parsed) + " (session_id=" +
                     session_id + " data row " +
                     std::to_string(queue_index) + " csv=" + csv_path_ + ")");
    }
    env.turn_index = static_cast<int>(turn_index_parsed);
    env.request_id = request_id;
    // C3 fix (2026-09-28): request_id identity gate, ALL data rows. The
    // turn>0 arm was already fail-closed downstream
    // (RequestIngress::register_queue_index aborts on a duplicate or empty
    // id); the turn-0 arm had no uniqueness check anywhere between here and
    // the calendar Submit, so two session blocks reusing one request_id
    // aliased each other downstream (the metric ledger registers
    // request_id -> queue_index last-wins). Check here, before any Submit:
    // empty ids and cross-row duplicates are reader_fatal, symmetric with
    // the session/turn structure gates above.
    if (request_id.empty()) {
        reader_fatal("CSV field request_id is empty: session_id=" +
                     session_id + " data row " +
                     std::to_string(queue_index) + " (csv=" + csv_path_ + ")");
    }
    if (!seen_request_ids_.insert(request_id).second) {
        reader_fatal("request_id is not unique across data rows: "
                     "request_id=" +
                     request_id + " session_id=" + session_id +
                     " data row " + std::to_string(queue_index) +
                     " (csv=" + csv_path_ + ")");
    }
    // Reserved-token gate (4.2 fix, 2026-09-29): the WP9 first-token anchor
    // identifies marker nodes by the "first_token" SUBSTRING of the node
    // name, and the graph builder embeds the sanitized session_id /
    // request_id in EVERY regular node name ("q{qi}_{session}_turn{t}_"
    // "{request}_..."). An id carrying the anchor token therefore hijacked
    // every node of that request: stage start anchors were never
    // registered (the marker arm consumes them) and a bogus first_token
    // event was synthesized from the request's earliest node -- all
    // silently, exit 0. "first_token" is reserved in the id space; the
    // same gate lives at the FIFO Submit port and the Python loader (one
    // lexicon, three ports, same discipline as the D3 length rule).
    if (session_id.find("first_token") != std::string::npos ||
        request_id.find("first_token") != std::string::npos) {
        reader_fatal("session_id/request_id must not contain the reserved "
                     "marker token \"first_token\" (it is the WP9 node-name "
                     "anchor substring): session_id=" +
                     session_id + " request_id=" + request_id +
                     " data row " + std::to_string(queue_index) +
                     " (csv=" + csv_path_ + ")");
    }
    // Reserved-prefix gate (same input-port discipline as the first_token
    // gate above): request_id with the "batch_train_" prefix belongs to the
    // train-sentinel namespace, which the mechanism layer treats as
    // reserved -- such watches are pinned to the prefill stage, skip the
    // in-flight/prefill-drained eligibility checks, and their PREFILL_DRAIN
    // is excluded from prefill_drained. A real request slipping through with
    // that id would silently bypass those fail-closed eligibility gates, so
    // the reader rejects the prefix outright (the producer never issues real
    // requests in this namespace).
    if (request_id.rfind("batch_train_", 0) == 0) {
        reader_fatal("request_id must not use the reserved \"batch_train_\" "
                     "prefix (train-sentinel namespace): request_id=" +
                     request_id + " data row " + std::to_string(queue_index) +
                     " (csv=" + csv_path_ + ")");
    }
    env.prefill_length =
        parse_u64_field("prefill_length", prefill_s, session_id, queue_index,
                        csv_path_);
    env.decode_length =
        parse_u64_field("decode_length", decode_s, session_id, queue_index,
                        csv_path_);
    // D3 fix (2026-09-28): the three ports of the trace chain now share ONE
    // length lexicon. The Python loader (generate_trace.parse_request_length
    // -> parse_int) has always required a positive length, and the derive
    // materializer fail-closes on a non-positive length row since the same
    // batch -- the reader accepting 0 was the only port still admitting the
    // shape, so a hand-written or drifted queue could pass the runtime side
    // while every other port rejects it. Fail-closed, symmetric with the
    // structure gates above.
    if (env.prefill_length == 0 || env.decode_length == 0) {
        reader_fatal("CSV field prefill_length/decode_length must be positive "
                     "(zero-length requests are rejected by the materializer "
                     "and the Python loader alike): prefill_length=" +
                     std::to_string(env.prefill_length) + " decode_length=" +
                     std::to_string(env.decode_length) + " session_id=" +
                     session_id + " data row " +
                     std::to_string(queue_index) + " (csv=" + csv_path_ + ")");
    }
    env.inter_request_interval_ns =
        interval_s.empty()
            ? 0
            : parse_u64_field("inter_request_interval_ns", interval_s,
                              session_id, queue_index, csv_path_);
    // Turn>0 arrivals are computed as parent-complete + interval, so an
    // empty interval column on a turn>0 row silently scheduled the next
    // request at parent-complete +0 (timing distorted, no diagnostic). The
    // materializer always writes the interval on turn>0 rows and leaves it
    // empty ONLY on turn-0 rows (absolute arrival), so the empty default is
    // now legal exclusively there.
    if (env.turn_index > 0 && interval_s.empty()) {
        reader_fatal("CSV field inter_request_interval_ns is empty on a "
                     "turn>0 row (turn arrivals are parent-complete + "
                     "interval): session_id=" +
                     session_id + " turn_index=" +
                     std::to_string(env.turn_index) + " data row " +
                     std::to_string(queue_index) + " (csv=" + csv_path_ + ")");
    }
    // Symmetric twin (2026-09-30): a turn-0 row (absolute arrival) with a
    // NON-empty interval is the mirrored malformed shape -- the Python
    // loader already rejects it ("first request must leave
    // inter_request_interval_ns empty"); accepting it here diverged the two
    // ports on the same input. The value would ride on the ARRIVAL payload
    // to Python unused.
    if (env.turn_index == 0 && !interval_s.empty()) {
        reader_fatal("CSV field inter_request_interval_ns is non-empty on a "
                     "turn-0 row (turn-0 arrivals are absolute; the Python "
                     "loader rejects the same shape): session_id=" +
                     session_id + " data row " +
                     std::to_string(queue_index) + " (csv=" + csv_path_ + ")");
    }
    env.queue_index = queue_index;

    const bool is_turn0 = !arrival_s.empty();

    // ---- fail-closed session-block structure validation (P0 fix item 2) ----
    if (session_id.empty()) {
        // An empty session_id must not fall through into the block logic:
        // current_session_ default-constructs to "" so the row would skip
        // the new-session branch and die later in the misleading
        // "internal: block state lost" defensive arm.
        reader_fatal("CSV field session_id is empty: data row " +
                     std::to_string(queue_index) + " (csv=" + csv_path_ + ")");
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
                parse_u64_field("session_arrival_time_ns", arrival_s,
                                session_id, queue_index, csv_path_),
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
    const uint64_t arrival_ns = parse_u64_field(
        "session_arrival_time_ns", arrival_s, session_id, queue_index,
        csv_path_);
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
    // P1 fix (2026-09-28): an ABSENT sidecar is now also fail-closed. The
    // former "absent (no gate)" pass-through was exactly the hole a failed
    // materialization falls through: derive_20_first_30_seconds.py used to
    // write the queue row-by-row straight to its final path and the sidecar
    // only after ALL rows succeeded, so a mid-derivation abort left a
    // TRUNCATED queue on disk with no sidecar -- this reader then indexed
    // the shrunken input, the run-end completion audit passed on the
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
    nlohmann::json doc = strict_json_load(
        in, "provenance sidecar " + sidecar);
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
        if (!ingress_.enqueue_command(cmd)) {
            // Bounded ingress (capacity 4096): backpressure, NOT a failure.
            // The cursor stays on this entry; the next pump retries.
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
    // index outside the read domain (a negative sentinel or an unknown
    // future index) is fail-closed because it would corrupt the accounting.
    // Turn>0 future-alarm notifications erase nothing from the outstanding
    // set (turn>0 rows never entered it after the P0 fix).
    if (queue_index < 0 || queue_index >= static_cast<int64_t>(rows_read_)) {
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
            row.late_source = "static_submit_late";
        } else {
            row.late_source = "on_time";
        }
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
        // Attribution fix (2026-09-27): the calendar reader tries every
        // turn-0 Submit from tick 0 in (arrival, queue_index) order and has
        // no internal discovery delay, so a submission landing after the
        // declared arrival was parked by ingress backpressure (the bounded
        // command queue) while the clock advanced -- turn-0 (session) scale
        // above the ingress queue capacity, NOT the eradicated
        // late-discovery defect. The gate stays fail-closed either way.
        summary.gate_why =
            "late_static_submit=" + std::to_string(summary.late_static_submit) +
            " (a static-CSV turn-0 was submitted after its declared arrival: "
            "the bounded ingress command queue parked the calendar cursor "
            "past it -- turn-0 scale above ingress capacity)";
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
