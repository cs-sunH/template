/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

LegacyOracleWindowedTraceReader -- VERBATIM BASELINE COPY of the pre-P0
WindowedTraceReader (md5 of the original at copy time:
WindowedTraceReader.hh 226a34653aa9538c03320f22ff3ffcf5,
WindowedTraceReader.cc a9e1d6bfd6287814c0a4d20acdb7f9b7; archived at
/var/tmp/phase1_reader/baseline_src/ during the 2026-08-30 P0 fix). Test-only
oracle for calendar_reader equivalence (V2): the OLD row-window semantics --
turn-0 rows submitted in CSV row order as the bounded window discovers them
-- is preserved byte-for-byte so the new calendar reader's (alarm_time,
queue_index) fire sequence can be compared against it on the same input. The
audit_completion family was removed (it now lives in the live
WindowedTraceReader.hh); nothing else was changed. NEVER use this class in
production code.
******************************************************************************/

#ifndef EXECUTION_DRIVEN_LEGACYORACLEWINDOWEDTRACEREADER_HH
#define EXECUTION_DRIVEN_LEGACYORACLEWINDOWEDTRACEREADER_HH

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <map>
#include <ostream>
#include <string>
#include <unordered_set>

#include "astra-sim/workload/execution_driven/RequestIngress.hh"

namespace AstraSim {
namespace ExecutionDriven {

class LegacyOracleWindowedTraceReader {
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
    LegacyOracleWindowedTraceReader(const std::string& csv_path, RequestIngress& ingress,
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
    /// Exact current count of rows whose arrival has not fired.
    size_t current_window_occupancy() const { return outstanding_rows_.size(); }
    size_t high_water() const { return high_water_; }

    /// One [online] windowed reader: ... line (rows, pumps, occupancy peak,
    /// io ns, throughput rows/s, late arrivals, rejected).
    void report(std::ostream& os) const;

  private:
    std::ifstream file_;
    RequestIngress& ingress_;
    size_t high_water_;
    uint64_t max_arrival_ns_;
    bool header_seen_ = false;
    bool eof_ = false;
    uint64_t data_rows_ = 0;
    // Compatibility/audit watermark only. It is NOT used for occupancy:
    // arrivals may fire out of CSV order, so a maximum index is not a
    // contiguous consumed prefix.
    int64_t consumed_idx_ = -1;
    // Exact rows read but not yet consumed. Size is the window occupancy and
    // is therefore bounded by high_water (or explicitly unbounded when 0).
    std::unordered_set<int64_t> outstanding_rows_;
    size_t rows_read_ = 0;
    size_t rejected_out_of_range_ = 0;
    size_t read_pumps_ = 0;
    size_t peak_occupancy_ = 0;
    uint64_t io_read_ns_ = 0;
    // Previous CSV row per session, for the AFTER_REQUEST metrics
    // registration of turn>0 rows (mirrors the phase-1 loader).
    std::map<std::string, int64_t> last_queue_index_by_session_;
    // Byte position of the next un-read row (sampled after each successful
    // checkpoint's file_pos). -1 = never read / stream not open.
    int64_t last_file_pos_ = -1;

    size_t occupancy() const;
    void read_one_row();

    // Turn-0 rows (arrival non-empty): read by read_one_row.
    uint64_t turn0_rows_ = 0;
};

}  // namespace ExecutionDriven
}  // namespace AstraSim

#endif  // EXECUTION_DRIVEN_LEGACYORACLEWINDOWEDTRACEREADER_HH
