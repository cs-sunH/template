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
WindowedTraceReader.hh), and the read-only statistics accessors, report(),
and notify_consumed() were trimmed (2026-09-27: the equivalence test only
drives construction + pump()). NEVER use this class in production code.
******************************************************************************/

#ifndef EXECUTION_DRIVEN_LEGACYORACLEWINDOWEDTRACEREADER_HH
#define EXECUTION_DRIVEN_LEGACYORACLEWINDOWEDTRACEREADER_HH

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <map>
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

  private:
    std::ifstream file_;
    RequestIngress& ingress_;
    size_t high_water_;
    uint64_t max_arrival_ns_;
    bool header_seen_ = false;
    bool eof_ = false;
    uint64_t data_rows_ = 0;
    // Exact rows read but not yet consumed. Size is the window occupancy and
    // is therefore bounded by high_water (or explicitly unbounded when 0).
    std::unordered_set<int64_t> outstanding_rows_;
    // Previous CSV row per session, for the AFTER_REQUEST metrics
    // registration of turn>0 rows (mirrors the phase-1 loader).
    std::map<std::string, int64_t> last_queue_index_by_session_;

    size_t occupancy() const;
    void read_one_row();
};

}  // namespace ExecutionDriven
}  // namespace AstraSim

#endif  // EXECUTION_DRIVEN_LEGACYORACLEWINDOWEDTRACEREADER_HH
