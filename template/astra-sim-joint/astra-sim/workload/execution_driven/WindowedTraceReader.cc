/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

WindowedTraceReader -- execution-driven mechanism layer (wscllm phase 7 §10.4;
P0 turn-0 late-discovery fix 2026-08-30: index pass + turn-0 arrival calendar).
Implementation. Row semantics are byte-for-byte the phase-1 full loader
(main_online.cc load_request_queue_csv): same CSV schema, same turn-0 Submit /
turn>0 future-alarm split, same queue_index and metrics registration order.
The P0 fix changes only WHEN/HOW turn-0 rows are SUBMITTED: never by file
position -- always by the arrival calendar built by the streaming index pass
(see WindowedTraceReader.hh for the full contract and equivalence argument).
*******************************************************************************/

#include "astra-sim/workload/execution_driven/WindowedTraceReader.hh"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <set>
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
// (traces/materialize_20_30s.py queue_provenance) -- the cross-language gate
// depends on the two implementations agreeing on every input byte.
constexpr uint64_t kFnv1a64OffsetBasis = 14695981039346656037ULL;
constexpr uint64_t kFnv1a64Prime = 1099511628211ULL;

inline uint64_t fnv1a64_byte(uint64_t h, unsigned char b) {
    return (h ^ static_cast<uint64_t>(b)) * kFnv1a64Prime;
}

[[noreturn]] void reader_fatal(const std::string& what) {
    std::cerr << "[Error] (execution_driven/windowed_reader) " << what
              << std::endl;
    std::exit(EXIT_FAILURE);
}

// M15 (2026-09-23, deep-review): the frozen unsigned integer lexicon for
// the numeric CSV columns -- non-empty and every character an ASCII
// '0'..'9', byte-for-byte OnlineCli's is_ascii_digits rule (FP1/E25). A
// bare std::stoull wraps negatives ("-1" -> ULLONG_MAX) and silently
// truncates trailing garbage ("12x" -> 12); a bare std::stoi additionally
// throws an uncaught std::invalid_argument on an empty or non-numeric
// field. Each numeric column is lexically validated and range-checked
// BEFORE the conversion, so a malformed value aborts through reader_fatal
// (zero submissions) instead of entering the simulation silently. The
// official pipeline CSV is all-digits, so legal input is unchanged.
bool is_ascii_digits(const std::string& s) {
    if (s.empty()) {
        return false;
    }
    for (const char c : s) {
        if (c < '0' || c > '9') {
            return false;
        }
    }
    return true;
}

// For pure-digit strings, lexicographic comparison equals numeric
// comparison; rejects values the target type cannot hold BEFORE the
// (throwing) std::sto* conversion runs.
bool digits_exceed(const std::string& digits, const std::string& type_max) {
    if (digits.size() != type_max.size()) {
        return digits.size() > type_max.size();
    }
    return digits > type_max;
}

uint64_t parse_u64_column(const std::string& column, const std::string& value,
                          const std::string& where) {
    if (!is_ascii_digits(value) ||
        digits_exceed(value, "18446744073709551615")) {
        reader_fatal("CSV column " + column +
                     " is not a representable unsigned integer: "
                     "value='" +
                     value + "' " + where);
    }
    return std::stoull(value);
}

int parse_turn_index_column(const std::string& value, const std::string& where) {
    if (!is_ascii_digits(value) || digits_exceed(value, "2147483647")) {
        reader_fatal("CSV column turn_index is not a representable "
                     "non-negative integer: value='" +
                     value + "' " + where);
    }
    return std::stoi(value);
}

// Strict JSON load for the provenance sidecar (same as the wscllm reader):
// istream operator>> parses in NON-strict mode -- it silently consumes the
// FIRST JSON value and leaves trailing bytes unread, and the DOM build
// last-wins on duplicate object keys. The sidecar is machine-written by
// the paired materializer, so trailing content or a duplicate key is
// producer corruption, not a legal shape. json::parse in its default
// strict mode rejects trailing bytes outright; the stateful parse callback
// rejects a repeated key inside one object.
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
        provenance_status_ = "no-input";
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
    if (!pending.empty()) {
        // Final line without a trailing newline.
        process_indexed_row(pending);
    }
    file_.close();
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

    // R3-J4 port (2026-09-30, from the face repo): the Python loader
    // rejects a queue that yields zero requests ("must contain at least
    // one request"); this reader used to accept it and run a silent
    // zero-request simulation (an all-comment/blank-row file indexes zero
    // rows). Same fail-closed contract here.
    if (data_rows_ == 0) {
        reader_fatal("request queue CSV contains zero data rows "
                     "(csv=" + csv_path_ + ")");
    }

    check_provenance_sidecar();
    indexed_ = true;
}

void WindowedTraceReader::process_indexed_row(const std::string& line) {
    if (line.empty() || line == "\r") {
        // Blank line: not a data row. A CRLF file's blank line is a lone
        // '\r' char -- without this arm it fell through to
        // the comma-count gate and was misreported as a column-count
        // violation (the Python reference reader skips blank rows).
        return;
    }
    if (!header_seen_) {
        header_seen_ = true;
        // R5-N1 port (2026-09-30, from the face repo, aligned with this
        // repo's L11 column-shape contract): the header is content-checked
        // instead of blindly swallowed. Three legal shapes: the canonical
        // 8-column header, the joint canonical 9-column header (the same
        // 8 columns with the typed-eviction next_trigger_type column
        // inserted before description -- the output shape of this repo's
        // traces/materialize_20_30s.py and the input shape of the Python
        // loaders' REQUEST_QUEUE_COLUMNS), and the legacy
        // hand-written-fixture 7-column shape (same names minus the
        // description column). Anything else fails closed -- the pre-fix
        // behavior silently lost the first data row of a headerless input
        // and turned a single-line file into a zero-request empty run. One
        // trailing '\r' is tolerated (CRLF input).
        std::string header = line;
        // (the blank-line arm above guarantees non-empty here)
        if (header.back() == '\r') {
            header.pop_back();
        }
        const std::string canonical =
            "session_id,turn_index,request_id,prefill_length,decode_length,"
            "session_arrival_time_ns,inter_request_interval_ns,description";
        const std::string canonical_joint =
            "session_id,turn_index,request_id,prefill_length,decode_length,"
            "session_arrival_time_ns,inter_request_interval_ns,"
            "next_trigger_type,description";
        const std::string legacy7 =
            canonical.substr(0, canonical.rfind(','));
        if (header != canonical && header != canonical_joint &&
            header != legacy7) {
            reader_fatal("request queue CSV header is not a documented "
                         "header (expected the canonical 8-column header, "
                         "the joint canonical 9-column header, or "
                         "the legacy 7-column shape, got \"" + header +
                         "\"): " + csv_path_);
        }
        return;
    }
    // L11 fix (2026-09-29): explicit CRLF handling. The raw line may ride a
    // trailing '\r' (Windows-produced CSV); strip it HERE, before any field
    // is parsed, so the \r can no longer land silently inside the last
    // column. The provenance hash above already consumed the raw bytes and
    // is unaffected.
    std::string clean_line = line;
    if (clean_line.back() == '\r') {
        clean_line.pop_back();
    }
    // L13 fix (2026-09-29): '#' COMMENT rows. The Python reference reader
    // (generate_trace.py load_request_queue) skips every row whose
    // request_id cell starts with '#', before all other validation; this
    // reader used to count them as data rows and submit them, so a
    // hand-written fixture (no provenance sidecar to catch the divergence)
    // could disagree with Python about what is a request. Skipped rows do
    // not consume a queue index: data_rows_/rows_read_ count only real data
    // rows, exactly Python's accepted-row counting.
    {
        std::istringstream probe(clean_line);
        std::string probe_session, probe_turn, probe_request;
        std::getline(probe, probe_session, ',');
        std::getline(probe, probe_turn, ',');
        std::getline(probe, probe_request, ',');
        if (!probe_request.empty() && probe_request[0] == '#') {
            return;
        }
    }
    // L13 fix (2026-09-29): QUOTED fields. The Python reference reader is
    // quote-aware (csv.DictReader), so a quoted field with an embedded
    // comma parses correctly there while the positional split below would
    // silently re-map every later column. This reader never supported
    // quoted fields -- fail closed instead of mis-parsing.
    if (clean_line.find('"') != std::string::npos) {
        reader_fatal("CSV data row contains a double-quoted field (the "
                     "windowed reader parses positionally and does not "
                     "support quote-aware CSV): data row " +
                     std::to_string(data_rows_) + " (csv=" + csv_path_ + ")");
    }
    // L11 fix (2026-09-29, REPAIR of the earlier column-shape gate): the
    // previous gate ran `if (std::getline(row, description_ride, ','))`
    // after the seven field getlines and fatalled whenever the 8th
    // (description) field was NON-EMPTY -- i.e. it rejected every canonical
    // materializer row (",,d") and only tolerated rows missing the
    // description column outright. The correct gate is a comma count: the
    // canonical queue schema has 8 columns (7 commas) and the joint
    // canonical schema has 9 columns (8 commas -- next_trigger_type rides
    // before the trailing free-text columns); the columns this reader
    // parses positionally are the FIRST SEVEN either way, so both tails
    // are free text ignored here, and the legacy hand-written-fixture
    // shape of 7 columns (no description -- the ServiceVacuumTest
    // contract) stays legal. Anything else fails closed with the data-row
    // index: < 7 columns means a required column (e.g.
    // inter_request_interval_ns) would silently default to 0, and > 9
    // columns means an unquoted comma shifted the whole positional split.
    const size_t comma_count =
        static_cast<size_t>(std::count(clean_line.begin(), clean_line.end(),
                                       ','));
    if (comma_count < 6 || comma_count > 8) {
        reader_fatal("CSV data row does not carry the canonical 8/9 columns "
                     "(got " + std::to_string(comma_count + 1) +
                     " comma-separated fields; 7, 8 or 9 legal): data row " +
                     std::to_string(data_rows_) + " (csv=" + csv_path_ + ")");
    }
    std::istringstream row(clean_line);
    std::string session_id, turn_index_s, request_id, prefill_s, decode_s,
        arrival_s, interval_s;
    std::getline(row, session_id, ',');
    std::getline(row, turn_index_s, ',');
    std::getline(row, request_id, ',');
    std::getline(row, prefill_s, ',');
    std::getline(row, decode_s, ',');
    std::getline(row, arrival_s, ',');
    std::getline(row, interval_s, ',');
    // L13 fix (2026-09-29): EMPTY request_id. Python SYNTHESIZES
    // "request_{n}" for an empty cell; this reader deliberately diverges
    // (fail-closed, no fabrication): the id keys the metrics registration,
    // the ingress one-shot queue-index map and the graph request namespace,
    // and a silently invented id would mask a broken producer. Standard
    // materializer output always carries non-empty ids (zero behavior
    // change on legal input).
    if (request_id.empty()) {
        reader_fatal("CSV field request_id is empty (the Python reference "
                     "reader would synthesize request_{n}; the windowed "
                     "reader refuses to fabricate ids): session_id=" +
                     session_id + " data row " +
                     std::to_string(data_rows_) + " (csv=" + csv_path_ + ")");
    }
    ++data_rows_;
    // Frozen queue index (CSV data-row order, 0-based). Turn-0 carries it
    // directly in the Submit envelope. Turn>0 rows are ALL pre-registered
    // here during the index pass (O(rows), same order as the metrics side;
    // P0 fix note: this is a full pre-registration, no longer bounded by a
    // row window); schedule_future_arrival consumes the entries one-shot.
    const int64_t queue_index = static_cast<int64_t>(data_rows_) - 1;
    const std::string row_where =
        "session_id=" + session_id + " request_id=" + request_id +
        " data row " + std::to_string(queue_index) + " (csv=" + csv_path_ +
        ")";

    RequestEnvelope env;
    env.session_id = session_id;
    env.turn_index = parse_turn_index_column(turn_index_s, row_where);
    env.request_id = request_id;
    // C3 identity gate (ported from the wscllm reader): a cross-row
    // duplicate request_id aliases the metric ledger and the anchor
    // lookups downstream (request_id -> queue_index last-wins). Fail here,
    // before any Submit. (The empty-id leg this gate used to carry was
    // unreachable -- the L13 lexical gate above already rejected it on the
    // same row; removed with the 2026-10-01 dead-branch audit.)
    if (!seen_request_ids_.insert(request_id).second) {
        reader_fatal("request_id is not unique across data rows: "
                     "request_id=" +
                     request_id + " session_id=" + session_id +
                     " data row " + std::to_string(queue_index) +
                     " (csv=" + csv_path_ + ")");
    }
    // Reserved-token gate (4.2 fix, same as the wscllm reader): "first_token"
    // is the WP9 node-name anchor substring and the graph builder embeds the
    // sanitized session_id / request_id in every regular node name -- an id
    // carrying it hijacks every node of that request into the marker arm.
    if (session_id.find("first_token") != std::string::npos ||
        request_id.find("first_token") != std::string::npos) {
        reader_fatal("session_id/request_id must not contain the reserved "
                     "marker token \"first_token\" (it is the WP9 node-name "
                     "anchor substring): session_id=" +
                     session_id + " request_id=" + request_id +
                     " data row " + std::to_string(queue_index) +
                     " (csv=" + csv_path_ + ")");
    }
    env.prefill_length =
        parse_u64_column("prefill_length", prefill_s, row_where);
    env.decode_length =
        parse_u64_column("decode_length", decode_s, row_where);
    env.inter_request_interval_ns =
        interval_s.empty()
            ? 0
            : parse_u64_column("inter_request_interval_ns", interval_s,
                               row_where);
    // Turn>0 arrivals are parent-complete + interval: an empty interval
    // column on a turn>0 row silently scheduled the next request at
    // parent-complete +0. The materializer leaves the column empty ONLY on
    // turn-0 rows (absolute arrival), so the empty default is legal
    // exclusively there.
    if (env.turn_index > 0 && interval_s.empty()) {
        reader_fatal("CSV field inter_request_interval_ns is empty on a "
                     "turn>0 row (turn arrivals are parent-complete + "
                     "interval): " +
                     row_where);
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
                parse_u64_column("session_arrival_time_ns", arrival_s,
                                 row_where),
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
        parse_u64_column("session_arrival_time_ns", arrival_s, row_where);
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
    // M-16 修复(姊妹仓 P1 同款移植,2026-09-30):sidecar 缺席从
    // "absent-no-gate" 放行改为 fail-closed(REQUIRED)。旧缺席通道正是
    // 半途失败的物化会掉进去的洞:截断队列 + 无 sidecar 会被照常索引,
    // run-end 审计在缩水分母上全绿。缺席与在但读不了统一 fail-closed,
    // errno 仍进错误消息便于区分截断物化与权限/IO 故障。
    const std::string sidecar = csv_path_ + ".provenance.json";
    std::ifstream in(sidecar, std::ios::binary);
    if (!in.is_open()) {
        reader_fatal("provenance sidecar is MISSING (fail-closed): " +
                     sidecar +
                     " must sit next to the request queue CSV and match it "
                     "(errno=" + std::to_string(errno) + " " +
                     std::strerror(errno) +
                     "); a queue without its provenance record cannot be "
                     "told apart from a truncated one -- refusing to "
                     "submit anything (re-materialize with "
                     "sh_test_mesh/workload/llama2_7b_inference/traces/"
                     "materialize_20_30s.py)");
    }
    nlohmann::json doc = strict_json_load(in,
                                          "provenance sidecar " + sidecar);
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
    provenance_status_ = "matched";
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
            // Bounded ingress (capacity 4096): backpressure, NOT a failure.
            // The cursor stays on this entry; the next pump retries. The
            // eventual drain clamp is a capacity artifact (M5), not reader
            // lateness.
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
    summary.late_static_submit_backpressure =
        ingress_.late_static_submit_backpressure_count();
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
        // capacity_backpressure rows keep their delay in the percentile
        // stats (informative) but are exempt from the gate counters above:
        // the bounded ingress queue, not the reader, made them late (M5).
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
