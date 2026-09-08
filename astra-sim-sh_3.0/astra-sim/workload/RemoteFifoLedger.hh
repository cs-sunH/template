/******************************************************************************
This source code is licensed under the MIT license found in the LICENSE file in
the root directory of this source tree.
 *******************************************************************************/

#ifndef __REMOTE_FIFO_LEDGER_HH__
#define __REMOTE_FIFO_LEDGER_HH__

#include <cstddef>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace AstraSim {

namespace ExecutionDriven {

// ---------------------------------------------------------------------------
// Remote-memory FIFO real-port ledger (方案 §3.6 / 阶段 E, R3 2026-08-29).
//
// The accounting moved OUT of the Workload layer (which keyed a virtual
// counter per sys_id and could only reconstruct the port state under
// PER_NPU_MEMORY_EXPANSION, where rank == port). The record points now live
// inside the real backend:
//   - issue:      AnalyticalRemoteMemory::issue, right after port_index
//                 resolution and before the busy/enqueue decision;
//   - completion: AnalyticalRemoteMemory::call, from the completion payload
//                 (which carries this transaction's port AND bytes, so a
//                 shared port can never credit the next request's size).
// The ledger keys are therefore the REAL backend port indices under every
// memory architecture:
//   PER_NODE_MEMORY_EXPANSION  port = sys_id / num-npus-per-node (shared
//                              by the node's ranks);
//   PER_NPU_MEMORY_EXPANSION   port = npu-ids array index (or the set_sys
//                              registration order == rank when npu-ids is
//                              absent);
//   MEMORY_POOL                port = 0 (all ranks share one FIFO).
// Because the record points ARE the queue events themselves, the
// single-server invariants hold per real port exactly:
//   in_flight  = issued - completed   (== active + pending)
//   active     = (in_flight > 0) ? 1 : 0
//   pending    = in_flight - active
//
// This header is deliberately self-contained (no Sys/Workload includes): the
// backend translation unit includes it directly, so the backend pair plus
// this hh/cc travel to the other four repositories together in the R5 sync.
// The blueprint repos compile the record calls but never set_enabled(true)
// -- zero behavior change there.
//
// R4-12 sensing gate (fail-closed, default off): record_* are no-ops until
// main_online.cc enables the ledger for a --sensing-enabled run. The gate is
// now the ONLY gate (the old execution-mode check died with the Workload-side
// call): backend accounting is orthogonal to static/online execution mode,
// and non-sensing runs of either mode skip the per-MEM-node bookkeeping
// entirely. Query/audit data only; no simulation semantics are touched
// anywhere. Single-threaded event loop -> no locking.
// ---------------------------------------------------------------------------
class RemoteFifoLedger {
  public:
    struct PortCounters {
        uint64_t issued_count = 0;
        uint64_t issued_bytes = 0;
        uint64_t completed_count = 0;
        uint64_t completed_bytes = 0;
        uint64_t peak_in_flight_count = 0;   // max active+pending requests
        uint64_t peak_in_flight_bytes = 0;   // max active+pending bytes
    };

    // Per-rank issued attribution (the RF3a/b reconcile view: the Python
    // decision side only knows edge ranks, so the sidecar carries the
    // rank -> real-port mapping plus the rank's issued counters).
    struct RankAttribution {
        std::size_t port = 0;
        uint64_t issued_count = 0;
        uint64_t issued_bytes = 0;
    };

    static RemoteFifoLedger& instance();

    // R4-12: sensing gate (fail-closed, default off).
    void set_enabled(bool enabled);
    // R3 (方案 §3.6): the memory-architecture label exported with every
    // sensing row and the run-end mapping evidence line.
    void set_architecture(const std::string& architecture);

    // sys_id is the ISSUING rank (kept for the per-rank attribution view
    // only); the counters land on the real backend port.
    void record_issue(std::size_t port_index, int sys_id,
                      uint64_t tensor_size);
    void record_completion(std::size_t port_index, uint64_t tensor_size);

    /// Real backend ports with any activity, ascending (deterministic dump).
    std::vector<std::size_t> active_ports() const;
    const PortCounters* port(std::size_t port_index) const;
    /// Source ranks that ever issued through this port, ascending.
    const std::vector<int>& port_ranks(std::size_t port_index) const;
    const RankAttribution* rank_attribution(int sys_id) const;
    /// Ranks with issued attribution, ascending.
    std::vector<int> attributed_ranks() const;
    const std::string& architecture() const;
    /// Clear the counters/attribution/mapping data (keeps enabled_ and the
    /// architecture label). Required for the same-process multi-simulation
    /// focused fixture; production runs simulate once and never call it.
    void reset();

    uint64_t total_issued_count() const;
    uint64_t total_issued_bytes() const;
    uint64_t total_completed_count() const;
    uint64_t total_completed_bytes() const;
    /// True iff every real port drained (issued == completed per port). A
    /// run that ends undrained lost a completion -- the caller fails closed
    /// on it.
    bool drained() const;

    /// One sensing-sidecar row (without the trailing newline), exactly the
    /// bytes main_online writes per delivery epoch:
    ///   {"delivery_sequence", "tick", "memory_architecture",
    ///    "ports": [{"port", "ranks", "active", "pending",
    ///               "in_flight_bytes", "issued_count", "completed_count",
    ///               "issued_bytes", "completed_bytes"}],
    ///    "by_rank": [{"rank", "port", "issued_count", "issued_bytes"}]}
    /// Shared by the production export and the focused fixture so the two
    /// formatters can never drift apart (R3 DESIGN §4.5).
    std::string sidecar_row(uint64_t delivery_sequence, uint64_t tick) const;

  private:
    RemoteFifoLedger() = default;

    bool enabled_ = false;
    std::string architecture_;
    // Key = real backend port_index (NOT the rank).
    std::map<std::size_t, PortCounters> ports_;
    std::map<std::size_t, std::vector<int>> port_ranks_;
    std::map<int, RankAttribution> rank_attribution_;
};

}  // namespace ExecutionDriven

}  // namespace AstraSim

#endif /* __REMOTE_FIFO_LEDGER_HH__ */
