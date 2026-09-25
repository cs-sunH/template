/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#ifndef __ANALYTICAL_MEMORY_HH__
#define __ANALYTICAL_MEMORY_HH__

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "astra-sim/common/AstraRemoteMemoryAPI.hh"
#include "astra-sim/system/Callable.hh"
// Kept as in the serial backend header: in-repo consumers (e.g.
// hbm_nway_test.cc) rely on this header providing the complete Sys type.
#include "astra-sim/system/Sys.hh"

namespace AstraSim {
class WorkloadLayerHandlerData;
}  // namespace AstraSim

namespace Analytical {
enum MemoryArchitectureType {
  NO_MEMORY_EXPANSION = 0,
  PER_NODE_MEMORY_EXPANSION,
  PER_NPU_MEMORY_EXPANSION,
  MEMORY_POOL
};

/**
 * Fluid remote-memory port backend (SerDes concurrency redesign, plan V5.3
 * §3).  Replaces the serial one-transaction-at-a-time FIFO (former
 * PendingMemoryRequest / ongoing_transaction / start_request) with:
 *
 *  - Per-port job sets with continuous-ns fluid service: all issued
 *    transactions overlap their fixed latency (latency never consumes
 *    bandwidth); latency-expired positive-byte transactions join the port's
 *    streaming set and equally split remote_mem_bw; the instant a stream
 *    exhausts its bytes it leaves the denominator and the survivors
 *    redistribute immediately.  Observable Workload callbacks land on the
 *    integer Tick ceil(fluid_finish_ns).
 *  - One global earliest-deadline cancellable transition event shared by all
 *    ports, registered on the Sys of the FIRST set_sys call.  The event
 *    payload carries only a generation/sequence number -- never a job pointer
 *    or iterator.  An issue landing exactly on the queued event's Tick keeps
 *    the original event (deadline-diff replan); dispatch harvests ALL due
 *    completions across ports, sorts by (port_index, issue_sequence), settles
 *    per-port statistics, and only then hands each wlhd to its Workload.
 *    Dispatch never recursively dispatches.
 *  - bytes==0 && latency==0 transactions use an independent one-shot timer
 *    event at +1ns instead of the job set (plan §3.2 semantic change from the
 *    old same-tick completion).
 *
 *  Ownership: each PortJob uniquely holds its WorkloadLayerHandlerData until
 *  the completion callback hands it to the Workload; shutdown() cancels the
 *  global event (payload freed via the register_event_cancellable deleter),
 *  frees every timer/completion payload and destroys undelivered wlhd.
 */
class AnalyticalRemoteMemory : public AstraSim::AstraRemoteMemoryAPI,
                               public AstraSim::Callable {
 public:
  AnalyticalRemoteMemory(std::string memory_configuration) noexcept;
  ~AnalyticalRemoteMemory() override;

  void set_sys(int id, AstraSim::Sys* sys) override;
  void issue(uint64_t tensor_size,
             AstraSim::WorkloadLayerHandlerData* wlhd) override;
  void call(AstraSim::EventType type, AstraSim::CallData* data) override;

  // Fail-closed drain check (plan §3.4/§5.1): true iff no port holds any job,
  // no dual-zero timer is pending, no transition event is queued and no
  // dispatch is in progress.  Valid with sensing disabled; backed only by
  // live job state, never by per-transaction telemetry.
  bool is_drained() const;

  // Early-shutdown cleanup (plan §3.4): cancels the global transition event
  // and every dual-zero timer (payloads released through the registration
  // deleter) and destroys every undelivered wlhd.  Idempotent; never
  // fabricates normal completion.  The host Sys must outlive the backend.
  void shutdown();

  // §5.1 transaction-detail telemetry (plan V5.3 §5.1): enables the lazy
  // streaming JSONL writer for bridge_dir/remote_memory_transactions.jsonl.
  // This is NOT a new configuration switch -- the caller gates it on the
  // existing --sensing-enabled flag and supplies the run association key.
  // The file is created on the FIRST completed transaction only, so runs
  // with zero remote transactions leave zero bridge residue.  Rows are
  // written from the issue-time observation copy at completion settlement,
  // strictly BEFORE the Workload callback hands wlhd away (the record is
  // never reconstructed from the cookie after delivery).  Must be called at
  // most once, before the first issue, with a non-empty bridge_dir and a
  // non-empty run association key (metrics-manifest run_id, or the fully
  // normalized bridge directory for runs without one).  Fail-closed fatal
  // on any misuse or write failure.
  void enable_transaction_telemetry(const std::string& bridge_dir,
                                    const std::string& run_id);

  // Read-only per-port statistics snapshot (plan §5.1): the only port fact
  // source for tests and run diagnostics.  All counts are settled at issue,
  // continuous-substep transitions and completion settlement; peaks are
  // event-boundary reconstructions (never fixed-interval samples).
  struct PortStatsSnapshot {
    uint64_t issued_count = 0;
    uint64_t completed_count = 0;
    uint64_t issued_bytes = 0;
    uint64_t completed_bytes = 0;
    uint64_t in_flight_count = 0;  // live: issued, callback not delivered yet
    uint64_t peak_in_flight = 0;   // issue..callback, latency included
    uint64_t streaming_count = 0;  // live: active transmission streams
    uint64_t peak_streaming = 0;   // event-boundary reconstruction
    uint64_t latency_waiting_count = 0;     // live: latency not expired yet
    uint64_t completion_waiting_count = 0;  // live: fluid-complete, callback due
    // Completion-instant share changes (one per continuous instant with
    // survivors, however many streams exhaust together).  Join-driven share
    // changes are counted separately in stream_join_events.
    uint64_t redistribution_events = 0;
    uint64_t stream_join_events = 0;
    double port_busy_ns = 0.0;    // interval integral, any streaming (>=1)
    double shared_busy_ns = 0.0;  // interval integral, streaming (>=2)
    double bytes_served = 0.0;    // interval integral
  };

  [[nodiscard]] std::size_t port_count() const;
  // Out-of-range port_index is fail-closed fatal (same discipline as
  // resolve_port_index).
  [[nodiscard]] PortStatsSnapshot port_stats(std::size_t port_index) const;

 private:
  // Continuous per-port job lifecycle (plan §3.1).  A transaction is
  // LatencyWaiting until its fixed latency expires (latency never consumes
  // bandwidth), then either completes directly (bytes==0) or joins the
  // port's streaming set (bytes>0, equal split of remote_mem_bw), and is
  // FluidCompleteAwaitingCallback from fluid completion until its Workload
  // callback at ceil(fluid_finish_ns).
  enum class PortJobState {
    LatencyWaiting,
    Streaming,
    FluidCompleteAwaitingCallback
  };

  // §5.1 per-transaction observation.  Every field is copied at issue time
  // (or at the continuous transition that produces it) from backend-owned
  // state -- never re-read from the wlhd after the callback.  The record is
  // streamed out at completion settlement, strictly before the Workload
  // callback, and destroyed right after the write.
  struct TransactionRecord {
    uint64_t issue_seq = 0;      // per-port monotonic (dual-zero included)
    int sys_id = 0;              // issuing rank (== rank)
    uint64_t node_id = 0;        // workload node id (copied from wlhd at issue)
    uint64_t bytes = 0;          // original tensor_size copied at issue
    std::size_t port_index = 0;  // mapped port (copied at issue)
    AstraSim::Tick issue_tick = 0;  // issue Tick (copied at issue)
    double latency_ready_ns = 0.0;  // now + remote_mem_latency_ns at issue
    bool streamed = false;          // ever entered the streaming set
    double stream_start_ns = 0.0;   // first stream start (valid if streamed)
    double fluid_finish_ns = 0.0;   // last-byte instant (valid if streamed)
    AstraSim::Tick callback_tick = 0;
  };

  struct PortJob {
    uint64_t issue_seq = 0;  // per-port monotonic; completion order key
    uint64_t bytes = 0;
    double latency_ready_ns = 0.0;
    double remaining_bytes = 0.0;  // streaming only
    double fluid_finish_ns = 0.0;  // valid once FluidCompleteAwaitingCallback
    AstraSim::Tick callback_tick = 0;        // ceil(fluid_finish_ns)
    PortJobState state = PortJobState::LatencyWaiting;
    // Uniquely owned until completion hands it to the Workload (§3.4).
    AstraSim::WorkloadLayerHandlerData* wlhd = nullptr;
    // §5.1 observation copy created at issue when telemetry is enabled; the
    // unique_ptr suppresses PortJob copies (the stats replay copy in
    // earliest_callback_tick clones scalars only, never telemetry or wlhd).
    std::unique_ptr<TransactionRecord> telemetry;
  };

  // Cumulative per-port counters settled at continuous substep transitions
  // and before completion callbacks (plan §5.1).  redistribution_events
  // counts completion-driven share changes (one per instant with survivors);
  // stream_join_events counts join-driven share changes separately.
  struct PortStats {
    uint64_t issued_count = 0;
    uint64_t completed_count = 0;
    uint64_t issued_bytes = 0;
    uint64_t completed_bytes = 0;
    uint64_t in_flight_count = 0;
    uint64_t peak_in_flight = 0;
    // Live per-state counts, maintained at issue, continuous-substep
    // transitions and completion settlement (plan §5.1: state settled at the
    // transition, aggregates at the callback Tick).  Kept with sensing off so
    // the unconditional is_drained()/shutdown settlement audit stays backed.
    uint64_t streaming_count = 0;
    uint64_t latency_waiting_count = 0;
    uint64_t completion_waiting_count = 0;
    uint64_t peak_streaming = 0;
    uint64_t redistribution_events = 0;
    uint64_t stream_join_events = 0;
    double port_busy_ns = 0.0;   // any streaming (>=1)
    double shared_busy_ns = 0.0; // contended streaming (>=2)
    double bytes_served = 0.0;
  };

  struct PortState {
    std::vector<PortJob> jobs;  // live jobs in any non-delivered state
    PortStats stats;
    uint64_t next_issue_seq = 1;
  };

  // Result of one event-point sweep over a port's jobs.
  struct FlipSummary {
    bool stream_completed = false;
    size_t completions = 0;  // streaming exhaustions at this instant
    size_t joins = 0;        // positive-byte latency joins at this instant
  };

  // Payload of every event this backend registers.  Per §3.3 it carries only
  // a kind tag and a sequence/generation number -- never a PortJob*, an
  // iterator or any pointer into the job containers.  The free function
  // free_backend_event_payload is the register_event_cancellable deleter:
  // it releases the payload on cancellation, on shutdown and on the stale
  // generation path; the normal dispatch path releases it at call() entry.
  struct BackendEventPayload : public AstraSim::CallData {
    enum class Kind { Transition, DualZeroTimer };
    BackendEventPayload(const Kind k, const uint64_t s) : kind(k), seq(s) {}
    Kind kind;
    uint64_t seq;
  };

  struct DualZeroJob {
    uint64_t timer_seq = 0;
    uint64_t issue_seq = 0;  // per-port sequence, allocated like any issue
    std::size_t port_index = 0;
    AstraSim::Tick fire_tick = 0;
    // Uniquely owned until the one-shot timer delivers it (§3.4).
    AstraSim::WorkloadLayerHandlerData* wlhd = nullptr;
    // §5.1 observation copy (created at issue when telemetry is enabled).
    std::unique_ptr<TransactionRecord> telemetry;
    AstraSim::SystemEventHandle handle;
  };

  // Cumulative counters of the whole backend (event lifecycle diagnostics).
  struct BackendEventStats {
    uint64_t transition_registrations = 0;
    uint64_t transition_cancels = 0;
    uint64_t transition_keeps = 0;  // deadline-diff replans that kept the event
    uint64_t payloads_freed_at_dispatch = 0;
    uint64_t payloads_freed_by_deleter = 0;
    uint64_t stale_generation_dispatches = 0;  // defensive no-op path
  };

  // [[noreturn]] fail-closed exit for invariant violations and invalid
  // inputs.  Must not throw: Sys::call_events catches std::exception and
  // keeps dispatching (Sys.cc call_events), so only a hard exit enforces
  // backend invariants (plan §3.3).
  [[noreturn]] static void fatal(const std::string& msg);

  // register_event_cancellable deleter: releases the payload on cancellation
  // and shutdown (§3.3).  Suffices on its own because the payload is a plain
  // value struct with no back-references.
  static void free_backend_event_payload(AstraSim::CallData* data);

  static AstraSim::Tick ceil_tick(double t_ns);
  // Flips every job whose event point lies at t (latency joins, zero-byte
  // latency completions, stream exhaustions) and settles the per-instant
  // redistribution/peak counters.  Runs before the rate computation so
  // exhausted streams leave the bandwidth denominator at their exact
  // completion instant (plan §3.1).
  static FlipSummary flip_due_jobs(PortState& port, double t);
  static double advance_port(double bw_bytes_per_ns, PortState& port,
                             double from_ns, double to_ns,
                             bool stop_at_first_completion);
  void advance_all_ports_to(double t_ns);
  AstraSim::Tick earliest_callback_tick() const;
  void replan_transition();
  void dispatch_transitions();
  void deliver_dual_zero(uint64_t timer_seq);
  std::size_t resolve_port_index(int sys_id) const;
  AstraSim::Sys* resolve_sys(int sys_id) const;

  // §5.1 telemetry internals: lazy stream creation, one-row write+flush
  // (fatal on failure), and the read-only replay copy factory that clones
  // job scalars but never telemetry or wlhd ownership.
  void open_telemetry_stream();
  void write_transaction_row(const TransactionRecord& record);
  static PortState make_stats_replay_copy(const PortState& port);

  // §3.4/§5.1 unconditional settlement audit, run by is_drained() and by
  // shutdown() on a fully settled backend -- sensing-independent.  Verifies
  // per-port issued/completed count and byte equality, zero in-flight and
  // zero live per-state counts (cross-checked against a container recount),
  // and -- with telemetry on -- that the number of streamed detail rows
  // equals the settled completions and the stream is not in a failed state.
  void verify_settled_consistency() const;

  MemoryArchitectureType mem_type = NO_MEMORY_EXPANSION;
  double remote_mem_latency_ns = 0.0;  // fixed per-transaction latency in ns
  double remote_mem_bw = 0.0;          // GB/s value consumed as bytes/ns
  int num_nodes = 0;
  int num_npus_per_node = 0;

  std::unordered_map<int, AstraSim::Sys*> sys_map;
  bool per_npu_ids_configured = false;
  std::unordered_map<int, std::size_t> per_npu_port_indices;

  std::vector<PortState> ports_;  // per mapped port (replaces the FIFO deque)

  // Global single transition event (§3.3): bound to the FIRST set_sys host;
  // ports and dual-zero timers never own a second Sys event queue.
  AstraSim::Sys* host_sys_ = nullptr;
  AstraSim::SystemEventHandle transition_handle_;
  AstraSim::Tick transition_tick_ = 0;  // remembered queued Tick (0 == none queued)
  uint64_t transition_generation_ = 0;
  double current_time_ns_ = 0.0;  // continuous fluid clock
  bool in_dispatch_ = false;
  bool shutdown_done_ = false;

  uint64_t next_timer_seq_ = 0;
  std::vector<DualZeroJob> dual_zero_timers_;

  BackendEventStats event_stats_;

  // §5.1 transaction-detail telemetry (lazy; all-zero cost until the first
  // completed transaction of a sensing-enabled run).
  bool telemetry_enabled_ = false;
  bool telemetry_open_ = false;
  uint64_t telemetry_rows_written_ = 0;
  std::string telemetry_bridge_dir_;
  std::string telemetry_run_id_;
  std::ofstream telemetry_stream_;
};

}  // namespace Analytical

#endif /* __ANALYTICAL_MEMORY_HH__ */
