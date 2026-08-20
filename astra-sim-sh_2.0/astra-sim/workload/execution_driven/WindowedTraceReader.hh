/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

WindowedTraceReader -- execution-driven mechanism layer (wscllm phase 7 §10.4).

Bounded-window CSV reader (总体方案 §5.5/§5.6; 方案 §10.4). The request queue
CSV is read in a bounded row window instead of one full pass: the reader keeps
a file position and tops the window up to `high_water` un-consumed rows on
every pump (called from the simulation thread right after drain_commands()).
A row is "consumed" when its arrival alarm fired (RequestIngress::arrival_cb);
turn-0 rows are consumed on their alarm, turn>0 rows are consumed when the
REQUEST_COMPLETE future_alarm fires (the reader never schedules those -- the
commit path does, via RequestIngress::schedule_future_arrival).

Contract (mirrors the phase-1 full loader, byte-for-byte request semantics):

  - rows WITH an explicit session_arrival_time_ns (turn-0) are enqueued as
    Submit commands (alarm-only: no policy decision). A turn-0 arrival beyond
    `max_arrival_ns` (仿真输入窗口上限; backport fix 2026-08-16, sh_2.0测试
    §5.1: default 0 = UNBOUNDED -- the old 30e9 default burned the 30s
    acceptance window into the code and silently dropped over-window
    requests; the bound now exists only as the explicit
    --request-max-arrival-ns experiment knob) is REJECTED: counted in
    rejected_out_of_range(), logged once at report time, never submitted --
    the row still gets its queue_index registered and its metrics request
    registered (conservative: "every data row registered" stays true). The
    rejected row is marked consumed at reject time (a rejected row
    never fires an arrival alarm, so it must not clog the window). Any
    rejection fail-closes the run-end completion audit (the drop is never
    silent; see audit_completion below).
  - rows with an EMPTY arrival (turn>0) are never submitted directly; their
    arrivals are scheduled by the REQUEST_COMPLETE commit through
    future_alarms. The reader registers their queue_index as soon as the row
    is read so the future-arrival scheduling can fill the envelope (the
    high_water >= max consecutive same-session span guarantee: the window
    tops up every pump, so a turn>0 row is always registered before its
    future alarm can fire -- the parent row was consumed earlier, which
    lowered the occupancy and triggered the read of the following rows).
  - ALL data rows are counted toward the run-end completion assertion
    (completed_request_count == data_rows; 1177 for the 20.csv first-30s
    input), exactly like the full loader. Backport fix (2026-08-16,
    unified 2026-08-20 中-3): the assertion denominator is the file's
    TOTAL data rows. A rejected row is consumed at reject time, so the
    window always flows to EOF and total_data_rows() at run end IS the
    whole-file count (no tail scan needed).

Zero decision-sequence perturbation by construction: the same CSV rows are
read in the same order and submitted through the same RequestIngress; the
window only decides WHEN a row is read off disk. In the 20.csv input the 112
turn-0 rows are SCATTERED across the whole file (data rows 2..1165, grouped
by session, arrivals monotonic non-decreasing), so the initial pump alone
never covers them all -- instead consumption advances the window (a turn-0
row is consumed when its arrival alarm fires, turn>0 rows when their future
alarm fires) and every pump tops the window back up to high_water. A turn-0
row is therefore read before its arrival as long as high_water covers the
un-consumed rows ahead of it; measured on the 20.csv input at high_water=128:
read_pumps=101, peak un-consumed occupancy=128, and the frozen 128 arm reads
every turn-0 row in time -- Submit-path late clamps 0, and the run-end late
count (33) matches the unbounded arm exactly, coming entirely from
future_alarm scheduling (the same alarm sequence -> byte-identical decision
log). turn>0 rows never schedule anything by themselves. The queue_index map
holds at most the rows read so far; peak un-consumed envelope count is
bounded by high_water.

Thread contract: pump()/notify_consumed()/report() are all simulation-thread
(notify_consumed is invoked from the arrival hook installed by the caller,
which arrival_cb runs on the simulation thread). The ingress command queue
keeps its own mutex; this reader owns no shared state with producers.
*******************************************************************************/

#ifndef EXECUTION_DRIVEN_WINDOWEDTRACEREADER_HH
#define EXECUTION_DRIVEN_WINDOWEDTRACEREADER_HH

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <map>
#include <ostream>
#include <string>

#include "astra-sim/workload/execution_driven/RequestIngress.hh"

namespace AstraSim {
namespace ExecutionDriven {

class WindowedTraceReader {
  public:
    /// @param csv_path     8-column request queue CSV (same schema as the
    ///                     phase-1 loader).
    /// @param ingress      shared RequestIngress (合同: 共用同一 ingress).
    /// @param high_water   window high watermark: max un-consumed rows read
    ///                     ahead. Frozen at 128 for the phase-7 runs (the
    ///                     20.csv max consecutive same-session span is 72;
    ///                     0 = unbounded, one pump reads the whole file --
    ///                     the full-pass control arm of the phase-7 §10.4
    ///                     window benchmark).
    /// @param max_arrival_ns simulation input window upper bound; turn-0
    ///                     rows with arrival > this are rejected (counted,
    ///                     never submitted). Backport fix 2026-08-16: the
    ///                     default 0 = UNBOUNDED (no cap); a nonzero value is
    ///                     an explicit experiment knob whose drops fail-close
    ///                     the run-end completion audit.
    WindowedTraceReader(const std::string& csv_path, RequestIngress& ingress,
                        size_t high_water = 128,
                        uint64_t max_arrival_ns = 0);

    /// Simulation thread only: read rows off disk until the window occupancy
    /// reaches high_water (or EOF). Returns false once EOF was reached.
    bool pump();

    /// Simulation thread only (arrival hook): mark the row with queue index
    /// `queue_index` as consumed (its arrival alarm fired).
    void notify_consumed(int64_t queue_index);

    /// Total data rows read (including rejected ones).
    size_t rows_read() const { return rows_read_; }
    /// Data-row counter (queue indices are 0..data_rows()-1). After EOF this
    /// equals the file's total data-row count (the run-end completion
    /// assertion target: 1177 for the 20.csv first-30-seconds input).
    uint64_t data_rows() const { return data_rows_; }
    /// Backport fix (2026-08-16): turn-0 data rows (arrival column
    /// non-empty), read by read_one_row. The accepted-request accounting
    /// invariant of the run-end completion audit (accepted + dropped ==
    /// turn-0 rows: every turn-0 row was either submitted to the service
    /// or rejected out-of-window; turn>0 rows arrive via the future-alarm
    /// path and are covered by the completed == total - dropped check).
    uint64_t turn0_data_rows() const { return turn0_rows_; }
    /// True once the file was fully read.
    bool eof() const { return eof_; }
    /// Turn-0 rows rejected because arrival > max_arrival_ns.
    size_t rejected_out_of_range() const { return rejected_out_of_range_; }
    /// The whole-file data-row count. Equals data_rows() once EOF was
    /// reached (consume-at-reject guarantees EOF under any explicit
    /// window). The run-end completion audit denominator (backport fix).
    uint64_t total_data_rows() const { return data_rows_; }
    /// Number of pump calls that actually read at least one row.
    size_t read_pumps() const { return read_pumps_; }
    /// Accumulated wall time spent reading/parsing rows.
    uint64_t io_read_ns() const { return io_read_ns_; }
    /// Peak un-consumed row count inside the window.
    size_t peak_window_occupancy() const { return peak_occupancy_; }
    size_t high_water() const { return high_water_; }

    /// One [online] windowed reader: ... line (rows, pumps, occupancy peak,
    /// io ns, throughput rows/s, late arrivals, rejected).
    void report(std::ostream& os) const;

    /// Phase 7 §10.5: window-position checkpoint (audit state + same-process
    /// window restore). JSON, atomically written (tmp + rename). Returns
    /// false if the file could not be written (run-end reporting only; a
    /// failed checkpoint never fails the run -- it is audit evidence, not a
    /// gate).
    bool write_checkpoint(const std::string& path) const;

    /// Phase 7 §10.5: restore the window state from a checkpoint written by
    /// write_checkpoint. FAIL-CLOSED: a missing or corrupt file, or a
    /// checkpoint whose high_water/max_arrival_ns disagree with this
    /// reader's construction parameters, returns false and leaves the
    /// reader state untouched. Restore semantics: the file position is NOT
    /// persistent across processes, so a restored reader continues pumping
    /// from its current (already open) file position; the checkpoint
    /// carries the window bookkeeping (consumed_idx_, rows_read_, ...) so a
    /// same-process restart can resume with the correct occupancy and
    /// completion accounting.
    bool read_checkpoint(const std::string& path);

  private:
    std::ifstream file_;
    RequestIngress& ingress_;
    size_t high_water_;
    uint64_t max_arrival_ns_;
    bool header_seen_ = false;
    bool eof_ = false;
    uint64_t data_rows_ = 0;
    int64_t consumed_idx_ = -1;  // max queue index whose alarm has fired
    size_t rows_read_ = 0;
    size_t rejected_out_of_range_ = 0;
    size_t read_pumps_ = 0;
    size_t peak_occupancy_ = 0;
    uint64_t io_read_ns_ = 0;
    // Previous CSV row per session, for the AFTER_REQUEST metrics
    // registration of turn>0 rows (mirrors the phase-1 loader).
    std::map<std::string, int64_t> last_queue_index_by_session_;
    // Byte position of the next un-read row (sampled before each read; the
    // checkpoint's file_pos). -1 = never read / stream not open.
    int64_t last_file_pos_ = -1;

    size_t occupancy() const;
    void read_one_row();

    // Turn-0 rows (arrival non-empty): read by read_one_row.
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
