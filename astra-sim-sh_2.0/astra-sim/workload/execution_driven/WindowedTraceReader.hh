/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

WindowedTraceReader -- execution-driven mechanism layer (wscllm phase 7 §10.4;
P0 turn-0 late-discovery fix 2026-08-30: index pass + turn-0 arrival calendar).

P0 FIX CONTEXT (the defect this rewrite eradicates). The request queue CSV is
laid out in SESSION BLOCKS: one contiguous block of rows per session, the
block's first row (turn 0) carrying the session's ABSOLUTE arrival in
session_arrival_time_ns, later turns carrying only relative inter-request
intervals. Turn-0 absolute arrivals are therefore NOT monotonic in file order
(measured on the full TraceLab queue, 22,816 rows / 496 turn-0 rows: 250
adjacent inversions spanning 243.41 days). The pre-fix reader discovered rows
in CSV row order inside a bounded row window and submitted each turn-0 row
per-Submit as it was DISCOVERED: an early-arriving turn-0 sitting late in the
file was submitted after the clock had already passed its declared arrival,
and RequestIngress clamped its alarm to current+1 (491 clamped Submits on the
full run) -- polluting every downstream timing. The fix decouples the turn-0
SUBMISSION ORDER from the file position entirely.

NEW DESIGN (frozen 2026-08-30, 2问题分析与解决方案kimi.md §4.3):

  1. Single streaming INDEX PASS before the first Submit. The whole file is
     read once in binary mode (chunked; the FNV-1a 64 digest and byte count
     are accumulated over the raw bytes on the fly). Per data row, in file
     order, the pass keeps the previous loader semantics EXACTLY: metrics
     online_register_request runs row by row (row order, AFTER_REQUEST
     parent = previous same-session row, frozen queue_index = data-row order
     0-based), and every turn>0 row registers its one-shot queue index in
     RequestIngress up front (O(rows), same order as the metrics side; the
     map is consumed by schedule_future_arrival exactly as before). Turn-0
     rows are COLLECTED, not submitted.
  2. FAIL-CLOSED structure validation during the pass: session rows must be
     contiguous in the file (a session id may never reappear after its block
     ended), each block must start at turn_index 0 with a non-empty arrival,
     and turn_index must increase by exactly 1 per row within a block, with
     empty arrival on every turn>0 row. Any violation: [Error] + exit before
     a single Submit exists.
  3. TURN-0 CALENDAR. The collected turn-0 entries are stable-sorted by
     (arrival_ns, queue_index) IN MEMORY this round (entry count = session
     count; external sort for very large traces is a registered follow-up).
     Submissions then walk the calendar in arrival order, applying the
     max_arrival_ns rejection per entry (same counting semantics as before:
     counted, never submitted, reported; any drop fail-closes the run-end
     completion audit). Ingress backpressure (full command queue) parks a
     cursor; later pump() calls resume from it.
  4. EQUIVALENCE ARGUMENT (why the calendar is not a behavior change for
     inputs that were already correct). EventQueue is a
     std::map<EventTime, EventList>; all events scheduled within one drain at
     a given tick share that tick's EventList and fire in INSERTION order.
     The old UNBOUNDED arm (window 0) read the whole file in row order at
     t=0, queued every turn-0 Submit in ROW order, and one drain scheduled
     them all: within an equal-alarm group the insertion order was therefore
     queue_index ascending (row order). The calendar submits in
     (arrival, queue_index) order: within an equal-arrival group the
     insertion order is again queue_index ascending. Equal alarm groups are
     equal arrival groups (all turn-0 Submits drain at t=0 before any event
     fires, so no Submit is ever clamped except arrival==0 -> 1, and those
     rows form one group whose internal order is row order == queue_index
     order in both arms). Hence the (alarm_time, queue_index) fire sequence
     is ELEMENTWISE IDENTICAL to the old unbounded arm -- for ANY input, not
     just monotonic ones. For inputs that were already arrival-monotonic in
     file order (the 20.csv smoke input) the calendar order IS the row order,
     so the old bounded-window arm is byte-identical too (V3 three-way
     comparison anchors on this). What changes is only that a
     NON-monotonic input (the full TraceLab queue) now submits every turn-0
     BEFORE its declared arrival instead of 491 times after.
  5. `--request-window-rows` is demoted to ADVISORY (design ruling
     2026-08-30, REPORT item). The old window semantics -- bounding how far
     ahead rows may be READ -- is exactly the defect: it made turn-0
     discovery depend on file position. The knob is still parsed, stored,
     checkpointed and reported ("calendar reader: advisory"), including 0,
     but it no longer constrains discovery or submission in any way.
  6. PROVENANCE GATE (fail-closed). The index pass computes over the raw
     file: FNV-1a 64 (offset basis 14695981039346656037, prime 1099511628211,
     per byte: h ^= b; h *= prime, all mod 2^64 -- byte-for-byte the same
     algorithm as the Python materializer's sidecar writer), csv byte count,
     data-row count, session count, turn-0 count, turn-0 arrival min/max,
     turn-0 file-order adjacent inversion count, and session-block
     contiguity. If a sidecar `<queue_csv>.provenance.json` exists, every
     field is compared; any mismatch (or unparsable JSON, or schema != 1)
     fails the run with [Error] + exit BEFORE any Submit. An absent sidecar
     logs one "[online] provenance sidecar: absent (no gate)" line and
     continues (smoke fixtures materialized by older scripts).
  7. ARRIVAL TIME AUDIT + RUN-END GATE. Every turn-0 entry records its
     declared arrival and the simulation tick at which its Submit was
     enqueued (discovered; via RequestIngress::current_time()). The ingress
     records the effective alarm tick per static Submit. At run end the
     reader joins the two (audit_static_arrivals): per-entry
     declared/discovered/effective/late_by_ns/late_source (static_late /
     t0_boundary / on_time) plus summary counters and ingress-delay
     percentiles (nearest-rank p50/p99/max). The GATE for a finite static
     CSV run: late_static_submit_count == 0 AND every turn-0 ingress_delay
     == 0 except t0-boundary rows (declared==0, discovered==0, effective==1
     -- the EventQueue strict-future rule's inherent boundary; the source
     CSV legally contains arrival==0; counted separately, never gated).
     External stream (command FIFO) and future-alarm clamps are never gated.
     Metric origins stay DECLARED (the fix must never mask itself by
     rebasing metrics onto effective arrivals).

Backport fixes retained unchanged (2026-08-16 sh_2.0测试 §5.1; 2026-08-20 中-3):
max_arrival_ns default 0 = UNBOUNDED (explicit experiment knob only; any drop
is visible and fail-closes the run-end completion audit); the completion
audit denominator is the file's TOTAL data rows (a rejected row is consumed
at reject time, so the reader always reaches EOF).

Thread contract: pump()/notify_consumed()/report()/audit_static_arrivals()
are all simulation-thread (notify_consumed is invoked from the arrival hook
installed by the caller, which arrival_cb runs on the simulation thread). The
ingress command queue keeps its own mutex; this reader owns no shared state
with producers.
*******************************************************************************/

#ifndef EXECUTION_DRIVEN_WINDOWEDTRACEREADER_HH
#define EXECUTION_DRIVEN_WINDOWEDTRACEREADER_HH

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <map>
#include <ostream>
#include <string>
#include <unordered_set>
#include <vector>

#include "astra-sim/workload/execution_driven/RequestIngress.hh"

namespace AstraSim {
namespace ExecutionDriven {

class WindowedTraceReader {
  public:
    /// @param csv_path     8-column request queue CSV (same schema as the
    ///                     phase-1 loader).
    /// @param ingress      shared RequestIngress (合同: 共用同一 ingress).
    /// @param high_water   ADVISORY ONLY since the P0 fix (2026-08-30): the
    ///                     turn-0 calendar is always complete and submissions
    ///                     always follow arrival order; this value no longer
    ///                     bounds discovery. Parsed/reported/checkpointed for
    ///                     compatibility (0 included).
    /// @param max_arrival_ns simulation input window upper bound; turn-0
    ///                     rows with arrival > this are rejected (counted,
    ///                     never submitted). Default 0 = UNBOUNDED; a nonzero
    ///                     value is an explicit experiment knob whose drops
    ///                     fail-close the run-end completion audit.
    WindowedTraceReader(const std::string& csv_path, RequestIngress& ingress,
                        size_t high_water = 128,
                        uint64_t max_arrival_ns = 0);

    /// Simulation thread only. First call: run the streaming index pass,
    /// validate structure, check the provenance sidecar, build+sort the
    /// turn-0 calendar, then submit from the calendar cursor while the
    /// ingress has room (backpressure parks the cursor). Later calls: resume
    /// submission. Returns false once the calendar is fully drained
    /// (indexed_ && cursor at end).
    bool pump();

    /// Simulation thread only (arrival hook): mark the row with queue index
    /// `queue_index` as consumed (its arrival alarm fired). Turn-0 rows leave
    /// the outstanding set here; turn>0 notifications (future alarms) are
    /// harmless no-ops for the set.
    void notify_consumed(int64_t queue_index);

    /// Total data rows read (including rejected ones). Equals the whole-file
    /// count after the index pass.
    size_t rows_read() const { return rows_read_; }
    /// Data-row counter (queue indices are 0..data_rows()-1). After the
    /// index pass this equals the file's total data-row count (the run-end
    /// completion assertion target).
    uint64_t data_rows() const { return data_rows_; }
    /// Turn-0 data rows (arrival column non-empty) -- the accepted-request
    /// accounting invariant denominator of the run-end completion audit
    /// (accepted + dropped == turn0 rows).
    uint64_t turn0_data_rows() const { return turn0_rows_; }
    /// True once the file was fully indexed AND every calendar entry was
    /// submitted or rejected (the producer-close boundary).
    bool eof() const { return eof_; }
    /// Turn-0 rows rejected because arrival > max_arrival_ns.
    size_t rejected_out_of_range() const { return rejected_out_of_range_; }
    /// The whole-file data-row count. Equals data_rows() after the index
    /// pass. The run-end completion audit denominator.
    uint64_t total_data_rows() const { return data_rows_; }
    /// Number of pump calls that indexed rows or submitted at least one
    /// calendar entry.
    size_t read_pumps() const { return read_pumps_; }
    /// Accumulated wall time spent reading/parsing the index pass.
    uint64_t io_read_ns() const { return io_read_ns_; }
    /// Peak outstanding (submitted-but-unfired turn-0) count.
    size_t peak_window_occupancy() const { return peak_occupancy_; }
    /// Exact current count of submitted turn-0 rows whose alarm has not
    /// fired yet.
    size_t current_window_occupancy() const { return outstanding_rows_.size(); }
    /// Advisory window knob (see constructor).
    size_t high_water() const { return high_water_; }

    /// Index-pass provenance statistics (all computed over the raw file
    /// bytes; identical definitions on the Python materializer side).
    struct ProvenanceStats {
        uint64_t fnv1a64 = 0;
        uint64_t csv_bytes = 0;
        uint64_t data_rows = 0;
        uint64_t sessions = 0;
        uint64_t turn0_count = 0;
        uint64_t turn0_arrival_min_ns = 0;
        uint64_t turn0_arrival_max_ns = 0;
        uint64_t turn0_adjacent_inversions = 0;
        bool session_blocks_contiguous = true;
    };
    const ProvenanceStats& provenance() const { return prov_; }
    /// Sidecar comparison outcome for the checkpoint JSON.
    /// "matched" / "absent-no-gate".
    const std::string& provenance_sidecar_status() const {
        return provenance_status_;
    }

    /// Per-turn-0 arrival audit record (joined from the reader's calendar
    /// and the ingress's effective-arrival records). late_source:
    /// "on_time" / "t0_boundary" / "static_submit_late" / "no_effective_record"
    /// (the last one is itself a gate failure: a submitted turn-0 whose
    /// Submit never drained -- impossible on a completed run).
    struct ArrivalAuditEntry {
        int64_t queue_index = -1;
        uint64_t declared_arrival_ns = 0;
        uint64_t reader_discovered_tick = 0;
        uint64_t effective_arrival_ns = 0;
        uint64_t late_by_ns = 0;
        const char* late_source = "no_effective_record";
    };

    /// Summary of the arrival audit + the run-end static-arrival gate
    /// (finite static CSV runs; external stream / future alarms never
    /// gated). delay percentiles are nearest-rank over ALL turn-0 entries
    /// (t0-boundary rows contribute delay 1).
    struct ArrivalAuditSummary {
        uint64_t turn0_submitted = 0;
        uint64_t late_static_submit = 0;
        uint64_t late_external_stream = 0;
        uint64_t late_future_alarm_rounding = 0;
        uint64_t t0_boundary_clamp = 0;
        uint64_t delay_p50_ns = 0;
        uint64_t delay_p99_ns = 0;
        uint64_t delay_max_ns = 0;
        uint64_t nonzero_delay_rows = 0;  // delay>0 excluding t0-boundary
        bool gate_ok = true;
        std::string gate_why;
    };

    /// Simulation thread only, run end. Joins the calendar with the ingress
    /// records; entries are returned sorted by queue_index (deterministic
    /// audit encoding).
    std::vector<ArrivalAuditEntry> arrival_audit() const;
    ArrivalAuditSummary audit_static_arrivals() const;

    /// One [online] windowed reader: ... line (rows, sessions, calendar
    /// stats, io ns, throughput, late split, rejected).
    void report(std::ostream& os) const;

    /// Phase 7 §10.5 / P0 fix: run-end audit snapshot (now also carrying
    /// the provenance block and the full arrival audit). JSON, atomically
    /// written (tmp + rename). It deliberately is NOT a restart checkpoint:
    /// the reader alone cannot serialize EventQueue alarms, RequestIngress
    /// commands/one-shot indices, ServiceCoordinator counters, or
    /// MetricCollector parent state. Returns false if the audit snapshot
    /// could not be written; failure never changes simulation state.
    bool write_checkpoint(const std::string& path) const;

    /// Compatibility API retained fail-closed. Run-end snapshots are audit
    /// evidence only, so every call returns false and leaves both the reader
    /// and ingress untouched. A future restart feature must checkpoint the
    /// complete event/ingress/service/metrics state atomically instead of
    /// partially rewinding this reader.
    bool read_checkpoint(const std::string& path);

  private:
    struct CalendarEntry {
        uint64_t arrival_ns = 0;
        int64_t queue_index = -1;
        RequestEnvelope envelope;
        bool submitted = false;
        bool rejected = false;
        uint64_t discovered_tick = 0;  // eq time when its Submit was enqueued
    };

    std::string csv_path_;
    RequestIngress& ingress_;
    size_t high_water_;  // advisory (P0 fix)
    uint64_t max_arrival_ns_;
    bool header_seen_ = false;
    bool indexed_ = false;  // index pass completed (sidecar gate included)
    bool eof_ = false;      // indexed_ && calendar fully drained
    uint64_t data_rows_ = 0;
    // Compatibility/audit watermark only (not an occupancy measure).
    int64_t consumed_idx_ = -1;
    // Submitted-but-unfired turn-0 rows (occupancy).
    std::unordered_set<int64_t> outstanding_rows_;
    size_t rows_read_ = 0;
    size_t rejected_out_of_range_ = 0;
    size_t read_pumps_ = 0;
    size_t peak_occupancy_ = 0;
    uint64_t io_read_ns_ = 0;
    // Previous CSV row per session, for the AFTER_REQUEST metrics
    // registration of turn>0 rows (mirrors the phase-1 loader).
    std::map<std::string, int64_t> last_queue_index_by_session_;

    // ---- index pass state ----
    std::ifstream file_;  // binary; chunked raw-byte reads
    ProvenanceStats prov_;
    std::string provenance_status_ = "not-indexed";
    uint64_t fnv1a64_ = 14695981039346656037ULL;  // FNV-1a 64 offset basis
    // Session-block validation state.
    std::string current_session_;
    bool in_block_ = false;
    int64_t expected_next_turn_ = 0;
    uint64_t prev_turn0_arrival_ = 0;
    bool have_prev_turn0_ = false;
    std::unordered_set<std::string> seen_sessions_;

    // ---- turn-0 calendar ----
    std::vector<CalendarEntry> calendar_;
    size_t calendar_cursor_ = 0;

    size_t occupancy() const;
    void run_index_pass();          // whole file, fail-closed, no Submits
    void process_indexed_row(const std::string& line);
    void check_provenance_sidecar();
    void submit_from_calendar();

    // Turn-0 rows (arrival non-empty): collected by the index pass.
    uint64_t turn0_rows_ = 0;
};

/// Backport fix (2026-08-16, sh_2.0测试 §5.1): the run-end completion-audit
/// arithmetic, factored out so the fail-closed contract is unit-testable
/// without the full online binary (windowed_trace_reader_test.cc part H).
///
/// Denominator = the CSV's TOTAL data rows (not "rows the window happened to
/// read"): a request silently missing from the run must always be able to
/// widen the completed+dropped vs total gap, and any explicit-window drop is
/// itself a loud failure. Verdicts (first match wins):
///   Ok              completed + dropped == total (all rows accounted),
///                   accepted + dropped == turn0_rows (every turn-0 row
///                   either submitted or rejected), dropped == 0.
///   Dropped         dropped > 0 (an explicit --request-max-arrival-ns window
///                   rejected input rows: visible, fail-closed -- never a
///                   silent PASS).
///   AccountMismatch accepted + dropped != turn0_rows (a turn-0 row was read
///                   but neither accepted by the service nor rejected --
///                   e.g. Submit commands lost to an ingress overflow or a
///                   stalled window).
///   Incomplete      completed + dropped != total (requests missing from the
///                   run: never-read tail rows, un-fired future alarms, or
///                   accepted-but-unfinished requests).
///
/// Counter semantics note (measured on every official run): accepted counts
/// ONLY turn-0 submissions (112 for the 20.csv first-30s input); turn>0
/// requests arrive via the future-alarm path and never increment it, so the
/// accepted invariant is against turn-0 rows, never against total rows.
struct CompletionAuditCounts {
    uint64_t total_rows;   // reader.total_data_rows()
    uint64_t turn0_rows;   // reader.turn0_data_rows()
    uint64_t accepted;     // service accepted_request_count()
    uint64_t completed;    // service completed_request_count()
    uint64_t dropped;      // reader rejected_out_of_range()
};

enum class CompletionAuditVerdict {
    Ok,
    Dropped,
    AccountMismatch,
    Incomplete,
};

CompletionAuditVerdict audit_completion(const CompletionAuditCounts& counts);

/// One-line human-readable verdict reason for the audit's [Error] line.
const char* completion_audit_why(const CompletionAuditVerdict verdict);

}  // namespace ExecutionDriven
}  // namespace AstraSim

#endif  // EXECUTION_DRIVEN_WINDOWEDTRACEREADER_HH
