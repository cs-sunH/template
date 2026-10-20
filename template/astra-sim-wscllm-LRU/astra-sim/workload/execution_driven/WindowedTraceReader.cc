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
#include <array>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <set>
#include <sstream>
#include <vector>

#include <json/json.hpp>

#include "astra-sim/workload/MetricCollector.hh"

namespace AstraSim {
namespace ExecutionDriven {

namespace {

// Canonical request-queue CSV header (2026-09-29): the first line must be
// exactly this tuple -- same names, same order, same count -- matching the
// Python loader's REQUEST_QUEUE_COLUMNS (generate_trace.py
// load_request_queue, which compares reader.fieldnames against the same
// tuple). The old reader swallowed the header line unconditionally, so a
// swapped/renamed/short header shifted every positional parse silently.
constexpr std::array<const char*, 8> kRequestQueueColumns = {
    "session_id",
    "turn_index",
    "request_id",
    "prefill_length",
    "decode_length",
    "session_arrival_time_ns",
    "inter_request_interval_ns",
    "description",
};

// inter_request_interval_ns safety domain (2026-09-29). The turn>0 path
// adds the interval to the request-completion tick to derive the next
// arrival alarm (uint64 ns arithmetic); a value in the upper half of the
// domain can wrap that addition into a "past" alarm, and the Python
// scheduler overflows tick+interval the same way. Any such interval is
// non-physical for a real trace: fail closed naming the row and the value
// instead of reproducing the overflow.
constexpr uint64_t kMaxInterRequestIntervalNs =
    std::numeric_limits<uint64_t>::max() / 2;

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

// Fail-closed lexicon for the request-queue CSV fields (2026-09-25). The
// row parser used to call std::stoi/std::stoull bare: a malformed token
// threw std::invalid_argument/std::out_of_range, the exception escaped
// pump() into the main_online event loop (no try/catch there) and died as
// a bare std::terminate -- bypassing reader_fatal above and its csv/row
// context. std::stoull additionally SILENTLY accepts shapes no CSV schema
// should honor: a leading '-' wraps around ("-1" ->
// 18446744073709551615) and trailing garbage is ignored ("100x" -> 100),
// letting a distorted value into the arrival/length fields. Frozen rule,
// the same FP1 lexicon as OnlineCli.cc's is_ascii_digits: non-empty pure
// ASCII digits, then std::strtoull with the ERANGE + endptr-at-end checks
// and the per-field target range; any violation is a reader_fatal naming
// the field, the raw token, the queue index and the csv path.
[[noreturn]] void row_field_fatal(const std::string& field,
                                  const std::string& raw,
                                  const int64_t queue_index,
                                  const std::string& csv_path) {
    reader_fatal("CSV field " + field +
                 " must be a non-negative integer (pure ASCII digits), "
                 "got \"" +
                 raw + "\" at data row " + std::to_string(queue_index) +
                 " (csv=" + csv_path + ")");
}

bool ascii_digits_only(const std::string& s) {
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

// Header-name trim: the Python loader compares csv.DictReader fieldnames
// after stripping each one, so a header cell may carry surrounding blanks.
// Only header names are trimmed -- data-row fields keep the frozen
// no-whitespace lexicons below.
std::string trim_header_name(const std::string& s) {
    const auto begin = s.find_first_not_of(" \t");
    if (begin == std::string::npos) {
        return "";
    }
    const auto end = s.find_last_not_of(" \t");
    return s.substr(begin, end - begin + 1);
}

uint64_t parse_row_u64(const std::string& raw, const char* field,
                       const int64_t queue_index,
                       const std::string& csv_path) {
    if (!ascii_digits_only(raw)) {
        row_field_fatal(field, raw, queue_index, csv_path);
    }
    errno = 0;
    char* end = nullptr;
    const uint64_t value = std::strtoull(raw.c_str(), &end, 10);
    if (errno == ERANGE || end != raw.c_str() + raw.size()) {
        row_field_fatal(field, raw, queue_index, csv_path);
    }
    return value;
}

int parse_row_int(const std::string& raw, const char* field,
                  const int64_t queue_index, const std::string& csv_path) {
    const uint64_t value = parse_row_u64(raw, field, queue_index, csv_path);
    if (value > static_cast<uint64_t>(std::numeric_limits<int>::max())) {
        row_field_fatal(field, raw, queue_index, csv_path);
    }
    return static_cast<int>(value);
}

// M13 (2026-09-29): request lengths are STRICTLY POSITIVE. The Python
// loader parses prefill_length/decode_length with parse_request_length
// -> parse_int in generate_trace.py ("must be positive"), so a 0-length request
// never exists on the producing side; the non-negative lexicon above let a
// forged 0 into the node/KV sizing as a silent C++-only extension of the
// schema. Same fatal channel, same message shape as parse_row_u64.
uint64_t parse_row_positive_u64(const std::string& raw, const char* field,
                                const int64_t queue_index,
                                const std::string& csv_path) {
    const uint64_t value = parse_row_u64(raw, field, queue_index, csv_path);
    if (value == 0) {
        reader_fatal("CSV field " + std::string(field) +
                     " must be a positive integer (0 rejected, aligned with "
                     "the Python loader's strict-positive request lengths), "
                     "got \"" +
                     raw + "\" at data row " + std::to_string(queue_index) +
                     " (csv=" + csv_path + ")");
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
    // Bytes of `pending` already scanned without finding a '\n' (2026-09-29
    // fix: the scan used to restart at position 0 after EVERY 64 KiB append,
    // so one data line spanning N chunks was re-scanned O(N) times -- a
    // quadratic index pass on long-line inputs). find() now resumes at the
    // first byte that has not been seen yet; the extracted line boundaries
    // are unchanged.
    std::string::size_type scanned = 0;
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
        std::string::size_type nl = pending.find('\n', scanned);
        while (nl != std::string::npos) {
            process_indexed_row(pending.substr(begin, nl - begin));
            begin = nl + 1;
            nl = pending.find('\n', begin);
        }
        pending.erase(0, begin);
        scanned = pending.size();
    }
    // Fail-closed on a mid-index I/O error (fix 2026-09-27): the read loop
    // above exits on ANY stream-state failure -- eofbit+failbit at the true
    // EOF, but equally badbit on a disk error / short read. Without this
    // check a mid-file I/O error is silently treated as a clean EOF: the
    // digest and the calendar describe a truncated input that is internally
    // self-consistent, and the run-end completion audit passes on the
    // truncated denominators (the provenance sidecar gate below is the
    // second line of defense against exactly that shape since M-16 made it
    // REQUIRED).
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
    prov_.fnv1a64 = fnv1a64_;
    prov_.data_rows = data_rows_;
    prov_.sessions = seen_sessions_.size();
    prov_.turn0_count = turn0_rows_;
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

    check_provenance_sidecar();
    indexed_ = true;
}

void WindowedTraceReader::process_indexed_row(std::string line) {
    // Line terminators (2026-09-29): lines are split on '\n' only, so a
    // CRLF file used to leave one '\r' on every line tail -- it rode the
    // trailing description column (the 8-column count stayed intact, but
    // the first data row of a shorter CRLF file failed with an invisible
    // CR inside the error message). Strip a single trailing '\r' here --
    // exactly the Python universal-newlines behavior -- so CRLF and LF
    // files are indistinguishable downstream.
    if (!line.empty() && line.back() == '\r') {
        line.pop_back();
    }
    if (line.empty()) {
        return;  // blank line: not a data row
    }
    if (!header_seen_) {
        header_seen_ = true;
        // Header contract (2026-09-29): the first line must be exactly the
        // canonical column tuple -- same names, same order, same count --
        // as the Python loader's REQUEST_QUEUE_COLUMNS. Each name may carry
        // surrounding blanks (both ends trim: trim_header_name here, the
        // loader's per-column strip); data-row fields are NOT trimmed on
        // either side. Any other header is a schema mismatch: fail closed
        // naming both sides instead of swallowing the line and shifting
        // every positional parse.
        std::vector<std::string> names;
        std::string::size_type begin = 0;
        while (true) {
            const auto comma = line.find(',', begin);
            if (comma == std::string::npos) {
                names.push_back(trim_header_name(line.substr(begin)));
                break;
            }
            names.push_back(
                trim_header_name(line.substr(begin, comma - begin)));
            begin = comma + 1;
        }
        bool header_ok = names.size() == kRequestQueueColumns.size();
        for (size_t i = 0; header_ok && i < names.size(); ++i) {
            header_ok = names[i] == kRequestQueueColumns[i];
        }
        if (!header_ok) {
            std::string expected;
            for (const char* name : kRequestQueueColumns) {
                if (!expected.empty()) {
                    expected += ",";
                }
                expected += name;
            }
            reader_fatal("request queue CSV header must be exactly: " +
                         expected + "; got: " + line + " (csv=" + csv_path_ +
                         ")");
        }
        // M8 (2026-09-29): the header line defines the row schema -- record
        // its comma-separated column count (normally 8) as the expected
        // field count every data row must carry.
        expected_columns_ = kRequestQueueColumns.size();
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
    // M11 (2026-09-29): '#' comment rows. Since 2026-10-01 the Python
    // loader REJECTS any '#'-prefixed request_id (its stricter end of the
    // three-port lexicon: loader rejects, this reader skips before
    // counting, the derive provenance scan skips before counting); the
    // skip below stays as the defensive classification point so a '#'-led
    // row consumes neither a request position nor a queue
    // index here. Mirror the skip at exactly the same
    // classification point: before data_rows_++ (no queue index consumed),
    // before the column-count and field validations (a comment row may
    // carry arbitrary annotation text in its remaining columns), before any
    // request/metrics registration. Provenance parity: the FNV-1a 64
    // digest and byte count hash the RAW file bytes and are unaffected by
    // the skip; data_rows counts request rows only, which stays
    // sidecar-consistent because the materialized queues that carry a
    // provenance sidecar never contain '#' rows.
    if (!request_id.empty() && request_id.front() == '#') {
        return;
    }
    ++data_rows_;

    // Frozen queue index (CSV data-row order, 0-based). Turn-0 carries it
    // directly in the Submit envelope. Turn>0 rows are ALL pre-registered
    // here during the index pass (O(rows), same order as the metrics side;
    // P0 fix note: this is a full pre-registration, no longer bounded by a
    // row window); schedule_future_arrival consumes the entries one-shot.
    // Computed BEFORE the field parses below so every row_field_fatal
    // message can name the offending row.
    const int64_t queue_index = static_cast<int64_t>(data_rows_) - 1;

    // M8 (2026-09-29): torn-row field-count gate, checked BEFORE any field
    // is parsed. The positional getline parser cannot name fields: a torn
    // row shifts every later column left, so a 5-column turn>0 row left the
    // interval column empty and was silently accepted with
    // inter_request_interval_ns forged to 0 (and a 6-column turn-0 row
    // likewise dropped its interval). The count itself is header-defined
    // (see expected_columns_ above); the trailing-'\r' strip at the top of
    // this function makes a CRLF file's row counts identical to its LF
    // counterpart's.
    const size_t row_columns =
        static_cast<size_t>(std::count(line.begin(), line.end(), ',')) + 1;
    if (row_columns != expected_columns_) {
        reader_fatal("CSV row has " + std::to_string(row_columns) +
                     " fields, expected " +
                     std::to_string(expected_columns_) +
                     " (header-defined column count) at data row " +
                     std::to_string(queue_index) + " (csv=" + csv_path_ + ")");
    }

    // session_id must be non-empty (2026-09-29). The Python loader rejects
    // an empty session_id ("request queue line N is missing session_id").
    // The reader used to misclassify the FIRST data row's empty session_id
    // as "internal: block state lost" (an input-format violation reported as
    // a reader-internal defect) and silently ACCEPT a mid-file empty
    // session_id as a brand-new session block. One input-format gate covers
    // both shapes; it runs before the block structure validation below so
    // the empty id can never enter seen_sessions_ either.
    if (session_id.empty()) {
        reader_fatal("CSV session_id must be non-empty at data row " +
                     std::to_string(queue_index) + " (csv=" + csv_path_ + ")");
    }
    // request_id must be non-empty (2026-09-29; 2026-10-01 两端口径统
    // 一): the Python loader now REJECTS an empty request_id as well (it
    // used to synthesize f"request_{len(requests)}" -- a two-end identity
    // fork this reader never reproduced). Both ends fail closed on the
    // same shape; the id is load-bearing across the turn>0 queue-index
    // registration, the global dedup set and the metrics anchors.
    if (request_id.empty()) {
        reader_fatal("CSV request_id must be non-empty at data row " +
                     std::to_string(queue_index) + " (csv=" + csv_path_ + ")");
    }
    // request_id 全局唯一性（C3 口径，2026-10-01 补齐跨通道）：在
    // turn-0/turn>0 分流之前统一入库查重——两条通道各自的查重
    // （turn>0: RequestIngress::register_queue_index；turn-0: 本读取器）
    // 都看不见对方，同一 id 跨通道复用此前直到运行中后段才以远离根因
    // 的形态爆掉（双生命周期 / M-12 重复 REQUEST_COMPLETE）。
    if (!seen_request_ids_.insert(request_id).second) {
        reader_fatal("request_id is not unique across data rows: "
                     "request_id=" +
                     request_id + " session_id=" + session_id +
                     " data row " + std::to_string(queue_index) +
                     " (csv=" + csv_path_ + ")");
    }
    // 保留前缀门（与 Python 侧调度器构造期拦截同一词表，读取器侧再加
    // 一道）："batch_train_" 前缀属于列车哨兵命名空间——带该前缀的
    // watch 被钉在 prefill 段、跳过在飞/prefill-drained 资格检查、其
    // PREFILL_DRAIN 不入 prefill_drained 账。真实请求顶该前缀会静默
    // 绕过这些 fail-closed 资格门（生产者从不在该命名空间发真实请求）。
    if (request_id.rfind("batch_train_", 0) == 0) {
        reader_fatal("request_id must not use the reserved \"batch_train_\" "
                     "prefix (train-sentinel namespace): request_id=" +
                     request_id + " data row " + std::to_string(queue_index) +
                     " (csv=" + csv_path_ + ")");
    }

    RequestEnvelope env;
    env.session_id = session_id;
    env.turn_index =
        parse_row_int(turn_index_s, "turn_index", queue_index, csv_path_);
    env.request_id = request_id;
    env.prefill_length =
        parse_row_positive_u64(prefill_s, "prefill_length", queue_index,
                               csv_path_);
    env.decode_length =
        parse_row_positive_u64(decode_s, "decode_length", queue_index,
                               csv_path_);
    // inter_request_interval_ns upper bound (2026-09-29): the future-alarm
    // path adds the interval to the request-completion tick (uint64 ns), so
    // a value above the safe domain would wrap into a "past" alarm (the
    // Python scheduler overflows tick+interval the same way). Rejected as a
    // non-physical interval, naming the row and the value.
    if (!interval_s.empty()) {
        const uint64_t interval =
            parse_row_u64(interval_s, "inter_request_interval_ns",
                          queue_index, csv_path_);
        if (interval > kMaxInterRequestIntervalNs) {
            reader_fatal(
                "CSV field inter_request_interval_ns=" +
                std::to_string(interval) +
                " exceeds the uint64 safe domain (max " +
                std::to_string(kMaxInterRequestIntervalNs) +
                "): tick + interval would overflow the arrival-alarm "
                "arithmetic at data row " +
                std::to_string(queue_index) + " (csv=" + csv_path_ + ")");
        }
        env.inter_request_interval_ns = interval;
    } else {
        env.inter_request_interval_ns = 0;
    }
    env.queue_index = queue_index;

    const bool is_turn0 = !arrival_s.empty();

    // ---- fail-closed session-block structure validation (P0 fix item 2) ----
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
    // Interval cross rules (2026-09-29), the exact duals of the Python
    // loader's checks ("first request must leave inter_request_interval_ns
    // empty" / "non-first request requires inter_request_interval_ns",
    // generate_trace.py load_request_queue): turn-0 rows schedule from the
    // absolute arrival, turn>0 rows from the relative interval, so a
    // present/absent interval on the wrong side used to be silently forged
    // to 0 (turn-0) or silently dropped (turn>0).
    if (env.turn_index == 0 && !interval_s.empty()) {
        reader_fatal("turn-0 row must leave inter_request_interval_ns empty "
                     "(first requests schedule from the absolute "
                     "session_arrival_time_ns): session_id=" +
                     session_id + " request_id=" + request_id +
                     " interval=" + interval_s + " (csv=" + csv_path_ + ")");
    }
    if (env.turn_index > 0 && interval_s.empty()) {
        reader_fatal("turn>0 row requires inter_request_interval_ns (the "
                     "relative interval is the only arrival input for a "
                     "non-first request): session_id=" +
                     session_id + " request_id=" + request_id +
                     " turn_index=" + std::to_string(env.turn_index) +
                     " (csv=" + csv_path_ + ")");
    }

    // Timing granularity gate (2026-09-29): the same rule and wording as
    // the Python loader ("timing must be divisible by 1000 ns",
    // generate_trace.py load_request_queue). Ticks are whole microseconds,
    // so a sub-microsecond arrival or interval can never be represented;
    // the gate used to exist only on the Python side and a queue carrying
    // e.g. 500 ns fell through the C++ side silently.
    uint64_t arrival_ns = 0;
    if (is_turn0) {
        arrival_ns = parse_row_u64(arrival_s, "session_arrival_time_ns",
                                   queue_index, csv_path_);
        if (arrival_ns % 1000u != 0u) {
            reader_fatal("timing must be divisible by 1000 ns: "
                         "session_arrival_time_ns=" +
                         std::to_string(arrival_ns) + " at data row " +
                         std::to_string(queue_index) + " (csv=" + csv_path_ +
                         ")");
        }
    } else if (env.inter_request_interval_ns % 1000u != 0u) {
        reader_fatal("timing must be divisible by 1000 ns: "
                     "inter_request_interval_ns=" +
                     std::to_string(env.inter_request_interval_ns) +
                     " at data row " + std::to_string(queue_index) +
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
                /*absolute_arrival=*/true, arrival_ns,
                /*arrival_parent_queue_index=*/-1, 0);
        }
        last_queue_index_by_session_[session_id] = queue_index;
    }

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
    // M9 (2026-09-29) 起 turn-0 侧的 request_id 查重在上方分流前的全行
    // 统一查重（seen_request_ids_）中完成，此处无独立集合。
    // arrival_ns was parsed (and its %1000 granularity checked) by the
    // timing gate above; every turn-0 consumer below reuses that value.
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
    // 半途失败的物化会掉进去的洞:物化器逐行直写最终路径、sidecar 最后
    // 才落盘,中途中断会留下"截断队列 + 无 sidecar"——本读取器照常索引
    // 缩水输入,run-end 完成审计在缩水分母上全绿,仿真对静默缺失的输入
    // "成功"。(物化器侧已按 temp+rename 原子发布 queue+sidecar;本门
    // 拒收任何不带 provenance 记录的队列。)
    const std::string sidecar = csv_path_ + ".provenance.json";
    std::ifstream in(sidecar, std::ios::binary);
    if (!in.is_open()) {
        // "缺席"(ENOENT)与"在但读不了"统一 fail-closed;errno 仍进
        // 错误消息,便于区分截断物化与权限/IO 故障。
        reader_fatal("provenance sidecar is MISSING (fail-closed): " +
                     sidecar +
                     " must sit next to the request queue CSV and match it "
                     "(errno=" + std::to_string(errno) + " " +
                     std::strerror(errno) +
                     "); a queue without its provenance record cannot be "
                     "told apart from a truncated one -- refusing to submit "
                     "anything (re-materialize with "
                     "sh_test_mesh/workload/llama2_7b_inference/traces/"
                     "derive_20_first_30_seconds.py)");
    }
    nlohmann::json doc;
    {
        // Strict JSON load (2026-10-01): same family as DecisionBridge.cc
        // read_json_file. istream operator>> parses in NON-strict mode --
        // it silently consumes the FIRST JSON value and leaves trailing
        // bytes unread, and the DOM build last-wins on duplicate object
        // keys. The sidecar is machine-written by the paired materializer,
        // so trailing content or a duplicate key is producer corruption,
        // not a legal shape; every consumed field is additionally checked
        // by expect_u64/schema below, which stays as the semantic gate.
        struct KeyGuard {
            std::vector<std::set<std::string>> stack;
            std::string duplicate;
            bool operator()(int, nlohmann::json::parse_event_t event,
                            nlohmann::json& parsed) {
                using EVT = nlohmann::json::parse_event_t;
                if (event == EVT::object_start) {
                    stack.emplace_back();
                } else if (event == EVT::key) {
                    if (!stack.back()
                             .insert(parsed.get<std::string>())
                             .second) {
                        duplicate = parsed.get<std::string>();
                    }
                } else if (event == EVT::object_end && !stack.empty()) {
                    stack.pop_back();
                }
                return true;  // observe only; never discard subtree
            }
        } guard;
        try {
            doc = nlohmann::json::parse(in, guard);
        } catch (const std::exception& e) {
            reader_fatal("provenance sidecar is not valid JSON: " + sidecar +
                         " (" + e.what() + ")");
        }
        if (!guard.duplicate.empty()) {
            reader_fatal("provenance sidecar has a duplicate JSON key \"" +
                         guard.duplicate + "\": " + sidecar);
        }
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
    // unknown future index is fail-closed because it would corrupt the
    // accounting. Turn>0 future-alarm notifications erase nothing from the
    // outstanding set (turn>0 rows never entered it after the P0 fix).
    // Negative indices are rejected on the same fail-closed footing: the
    // >= data_rows_ guard alone let a negative index bypass into a silent
    // no-op erase (G1-3).
    if (queue_index < 0 ||
        queue_index >= static_cast<int64_t>(data_rows_)) {
        std::cerr << "[Error] (execution_driven/windowed_reader) consumed "
                     "queue index was never read: "
                  << queue_index << " rows_read=" << data_rows_ << std::endl;
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
        // Attribution corrected 2026-09-27: capacity backpressure, not the
        // eradicated late-discovery defect. The gate stays fail-closed.
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
       << (max_arrival_ns_ == 0 ? " (unbounded)" : "") << " rows=" << data_rows_
       << " total_rows=" << data_rows_
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
           << (static_cast<double>(data_rows_) * 1e9 /
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
