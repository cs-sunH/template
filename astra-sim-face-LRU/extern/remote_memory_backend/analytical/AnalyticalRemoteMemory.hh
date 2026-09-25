/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#ifndef __ANALYTICAL_MEMORY_HH__
#define __ANALYTICAL_MEMORY_HH__

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "astra-sim/common/AstraRemoteMemoryAPI.hh"
#include "astra-sim/system/Callable.hh"
#include "astra-sim/system/Sys.hh"

namespace Analytical {
enum MemoryArchitectureType {
  NO_MEMORY_EXPANSION = 0,
  PER_NODE_MEMORY_EXPANSION,
  PER_NPU_MEMORY_EXPANSION,
  MEMORY_POOL
};

// 2026-09-24 SerDes port fluid-concurrency rework (plan sec.1/sec.3): the
// serial one-transaction-at-a-time FIFO (PendingMemoryRequest,
// pending_requests, ongoing_transaction, start_request and its runtime
// helper) was deleted and replaced by a per-port fluid model.
//
// Model (plan sec.3.1):
//   * Every issue enters the fixed-latency stage immediately; latency
//     stages of all transactions overlap and consume no bandwidth.
//   * When latency expires, a positive-byte transaction joins its port's
//     active stream set. N concurrent streams each receive
//     remote_mem_bw / N; whenever one stream exhausts, the survivors
//     re-split immediately at the new N (streams finishing at the same
//     continuous instant leave the denominator as one batch).
//   * Time advances in continuous ns substeps; the observable Workload
//     callback lands on Tick ceil(fluid_finish_ns). A positive transfer
//     shorter than 1ns still observes its callback on the next Tick.
//   * bytes == 0 && latency > 0 never joins the bandwidth denominator and
//     is delivered asynchronously at ceil(ready) Tick.
//   * bytes == 0 && latency == 0 becomes an independent one-shot dual-zero
//     timer that completes exactly 1ns after issue (plan sec.3.2); it is
//     never synchronously delivered from the issue stack.
//
// Events (plan sec.3.3): all ports and all dual-zero timers share ONE
// cancellable transition event bound to the Sys of the FIRST set_sys call
// (construction precedes set_sys, so the host is not cached at construct
// time). The event payload carries only a global generation number; job
// pointers and iterators never cross the event queue. Completions are
// delivered as one batch per fire, ordered by (port_index, issue_sequence),
// with backend state settled before the first Workload callback. An issue
// landing on a Tick where the transition event is still queued preserves
// that event: the issue advances state first (finished streams become
// awaiting-delivery), adds the new transaction, and lets the preserved
// event harvest the old flows on the very same Tick.
//
// Ownership (plan sec.3.4): a job uniquely owns its wlhd until delivery;
// delivery hands the wlhd to the Workload and then invokes the callback.
// Early shutdown deletes undelivered wlhds, cancels the global event and
// releases every payload. Normal-end teardown fails closed unless every
// backend-owned job, dual-zero timer, awaiting delivery and event handle
// is gone (independent of any sensing switch).
//
// Observation (plan sec.5.1, layered on top of the model without touching
// its semantics): per-port PortStats are rebuilt by event-interval
// integration (peaks at issue/admission instants, busy/shared/bytes
// integrals over the continuous service intervals, redistribution events
// per continuous transition instant) and settle completely BEFORE the
// first Workload callback of a batch. The per-transaction detail JSONL
// streams to <bridge_dir>/remote_memory_transactions.jsonl only when
// main_online.cc arms it from --sensing-enabled; the file is created
// lazily at the first row and each row is written before its wlhd
// callback (no post-hoc observation keys).
class AnalyticalRemoteMemory : public AstraSim::AstraRemoteMemoryAPI,
                               public AstraSim::Callable {
 public:
  AnalyticalRemoteMemory(std::string memory_configuration);
  ~AnalyticalRemoteMemory();

  void set_sys(int id, AstraSim::Sys* sys);
  void issue(
      uint64_t tensor_size,
      AstraSim::WorkloadLayerHandlerData* wlhd);
  void call(AstraSim::EventType type, AstraSim::CallData* data);
  uint64_t get_remote_mem_runtime(uint64_t tensor_size);

  // Four mutually exclusive working states, distinguishable from outside
  // (read-only observation surface for port statistics).
  enum class PortJobState {
    LatencyWaiting = 0,             // issued, fixed latency not yet expired
    ActiveStream,                   // positive bytes sharing port bandwidth
    FluidCompleteAwaitingCallback,  // fluid done, wlhd delivery pending
    DualZeroTimer                   // bytes == 0 && latency == 0 one-shot
  };

  // Immutable per-job observation snapshot (plan sec.5.1 consumer face).
  // fluid_finish_ns is -1.0 and callback_tick is 0 until the fluid finish
  // time exists; remaining_bytes is nonzero only for ActiveStream jobs.
  struct PortJobSnapshot {
    uint64_t tensor_size;
    uint64_t node_id;
    uint64_t issue_sequence;
    std::size_t port_index;
    PortJobState state;
    double issue_ns;
    double ready_ns;         // latency expiry; DualZeroTimer fire time
    double stream_start_ns;  // -1.0 until first bandwidth share
    double fluid_finish_ns;  // -1.0 until known
    AstraSim::Tick callback_tick;
    double remaining_bytes;
  };

  // Read-only per-port statistics snapshot (plan sec.5.1 observation layer;
  // replaces the former lightweight PortCounters). Every field is rebuilt
  // from EVENT-INTERVAL integration on the fluid transitions (continuous
  // substeps at issue/event boundaries) -- never from fixed-interval
  // sampling, which would miss short streams. No FIFO `pending` key exists.
  //
  // Conservation: issued_count == completed_count and issued_bytes ==
  // completed_bytes at normal end (fail-closed audit), and bytes_served
  // matches completed_bytes within the per-transaction completion residue
  // (kByteEps on the bytes path, bw * kTimeEpsNs on the time path; the
  // aggregate bound is audited in verify_drained).
  struct PortStats {
    // count/bytes pairs (conservation-audited at normal end).
    uint64_t issued_count;
    uint64_t completed_count;
    uint64_t issued_bytes;
    uint64_t completed_bytes;
    // Live instantaneous populations (job-state derived at query time).
    uint64_t in_flight_count;          // issued - completed; issue ->
                                       // callback, latency INCLUDED
    uint64_t streaming_count;          // ActiveStream population right now
    uint64_t latency_waiting_count;
    uint64_t completion_waiting_count;
    uint64_t dual_zero_timer_count;
    // Event-interval peaks.
    uint64_t peak_in_flight;   // max undelivered population ever (issue-time
                               // high-water mark)
    uint64_t peak_streaming;   // max simultaneous transfer streams ever
    // Event-interval integrals.
    uint64_t redistribution_events;
    //   Completion-driven: counted per CONTINUOUS completion instant where
    //   at least one stream exhausts AND at least one survivor remains
    //   (survivor share changes bw/N -> bw/(N-k)); simultaneous finishers
    //   count once; distinct continuous instants inside one integer
    //   callback Tick each count.
    uint64_t arrival_redistribution_events;
    //   Arrival-driven (listed separately, never mixed into the counter
    //   above): counted per instant where new streams join while other
    //   streams are already active on the port.
    double port_busy_ns;   // actual service time (streaming_count >= 1)
    double shared_busy_ns; // actual service time with streaming_count >= 2
    double bytes_served;   // bytes actually carried by the port (per-job
                           // exact attribution; equals completed_bytes minus
                           // the <= kByteEps-per-job completion residue)
  };

  std::size_t get_port_count() const;
  std::vector<PortJobSnapshot> get_port_jobs(std::size_t port_index) const;
  PortStats get_port_stats(std::size_t port_index) const;

  // True iff the backend holds no job, no dual-zero timer, no awaiting
  // delivery, no armed event handle and every per-port counter pair is
  // conserved. Unconditional: independent of any sensing switch.
  bool is_drained() const;
  // Fail-closed normal-end audit: sys_panic with a detailed report when
  // is_drained() would return false. Invoked by the destructor.
  void verify_drained() const;
  // Early-shutdown teardown: cancel the global event (deleter releases the
  // payload), delete every undelivered wlhd and job, reset counters. Never
  // pretends a normal completion happened.
  void shutdown();

  // Transaction-detail stream (plan sec.5.1). Armed exclusively through
  // main_online.cc from the --sensing-enabled token; there is no other
  // switch, environment variable or config key. When enabled, every
  // completed transaction streams one JSON line to
  // <bridge_dir>/remote_memory_transactions.jsonl. When disabled, NO
  // per-transaction record is retained anywhere -- only the low-overhead
  // aggregate counters above keep accumulating (they back the
  // unconditional is_drained() check); the detail output is the only
  // thing switched off.
  //
  // Creation semantics (amended 2026-09-24 second round): with the switch
  // OFF the file is never created (zero residue). With the switch ON the
  // artifact is non-empty at every NORMAL end -- rows are created lazily
  // at the first settled row and finalize_transaction_detail() appends one
  // terminal summary row (type=remote_memory_transactions_summary) that
  // carries the per-port PortStats aggregates, so a zero-transaction run
  // still produces exactly one parseable, recomputable line instead of no
  // file. Failure paths never call finalize: a retained partial file
  // without a summary row is the debugging evidence (never a fabricated
  // normal completion).
  //
  // run_id must be non-empty when enabled: the caller resolves the
  // manifest-first priority (metrics manifest run_id, else the fully
  // normalized RUN_DIR); an empty key fails closed here rather than
  // emitting rows without their association key.
  void configure_transaction_detail(bool enabled,
                                    std::string jsonl_path,
                                    std::string run_id);
  // Normal-end only: append the terminal summary row (per-port PortStats
  // aggregates + settled row count) and close the stream. No-op with the
  // detail switch off; idempotent. Creates the file lazily when no
  // transaction settled. Never call it on early-shutdown paths -- those
  // must keep the partial file as evidence.
  void finalize_transaction_detail();
  // Rows actually streamed so far (0 with the detail switch off). The
  // sensing-alive criterion is the artifact itself (the file exists and
  // parses at every normal end of a sensing run); this count distinguishes
  // a zero-transaction run from a broken writer inside the summary row.
  uint64_t transaction_rows_written() const {
    return transaction_rows_written_;
  }

 private:
  // Event-queue payload: only the arming generation (plan sec.3.3 forbids
  // storing PortJob* or container iterators).
  class TransitionEventPayload : public AstraSim::CallData {
   public:
    explicit TransitionEventPayload(uint64_t event_generation)
      : event_generation(event_generation) {
    }

    uint64_t event_generation;
  };

  struct PortJob {
    uint64_t tensor_size;
    uint64_t node_id;
    uint64_t issue_sequence;  // monotonic within its port
    std::size_t port_index;
    int sys_id;  // issuing rank, copied at issue for the detail row key
    AstraSim::WorkloadLayerHandlerData* wlhd;  // uniquely owned until
                                               // delivery
    PortJobState state;
    double issue_ns;
    AstraSim::Tick issue_tick;  // observable Tick at issue, copied at issue
    double ready_ns;          // latency expiry / dual-zero fire instant
    double stream_start_ns;   // -1.0 until first bandwidth share
    double fluid_finish_ns;   // -1.0 until known
    AstraSim::Tick callback_tick;  // 0 until known
    double remaining_bytes;
  };

  struct PortState {
    std::vector<PortJob*> jobs;  // every undelivered job, any state
    uint64_t next_issue_sequence = 0;
    uint64_t issued_count = 0;
    uint64_t completed_count = 0;
    uint64_t issued_bytes = 0;
    uint64_t completed_bytes = 0;
    // Observation-layer accumulators (plan sec.5.1); event-interval
    // integration only -- they are updated at issues, continuous service
    // intervals and transition instants, never by sampling.
    uint64_t peak_in_flight = 0;
    uint64_t peak_streaming = 0;
    uint64_t redistribution_events = 0;
    uint64_t arrival_redistribution_events = 0;
    double port_busy_ns = 0.0;
    double shared_busy_ns = 0.0;
    double bytes_served = 0.0;
  };

  void arm_transition_event(AstraSim::Tick deadline_tick);
  void cancel_transition_event();
  // Reconcile the single global event with the earliest job deadline.
  // Keeping the armed event when its deadline is not later than the new
  // earliest implements the same-Tick preservation rule (plan sec.3.3).
  void rearm_transition_event();
  AstraSim::Tick earliest_deadline_tick() const;
  static AstraSim::Tick ceil_to_tick(double time_ns);
  static void destroy_event_payload(AstraSim::CallData* data);

  // Transaction-detail stream (sensing-gated; plan sec.5.1). The row is
  // written during completion settlement, BEFORE the Workload callback --
  // the wlhd is destroyed after its callback, so a row can never be
  // completed after the fact. The file opens lazily at the first row.
  void settle_transaction_row(const PortJob& job);
  void close_transaction_stream();

  // Continuous fluid advance shared by issue() and the event dispatch: all
  // ports move to target_ns through ns substeps, honoring ready times,
  // dual-zero fires and bandwidth-exhaustion instants in order.
  void advance_all_ports(double target_ns);
  double earliest_transition_ns() const;
  void serve_interval(double dt_ns);
  void process_transitions_at(double t_ns);
  std::size_t active_stream_count(const PortState& port) const;
  std::size_t total_job_count() const;

  MemoryArchitectureType mem_type = NO_MEMORY_EXPANSION;
  double remote_mem_latency = 0.0;  // ns, finite, >= 0 (plan sec.3.2)
  double remote_mem_bw = 0.0;  // bytes/ns numeric convention (GB/s value
                               // used as B/ns), finite, > 0 when enabled

  // per-node memory expansion
  int num_nodes = 0;
  int num_npus_per_node = 0;

  std::vector<PortState> ports;
  bool per_npu_ids_configured = false;
  std::unordered_map<int, std::size_t> per_npu_port_indices;

  AstraSim::Sys* host_sys = nullptr;  // Sys of the FIRST set_sys call; owns
                                      // the single global transition event

  double clock_ns = 0.0;  // continuous backend time, <= observable Tick
  AstraSim::SystemEventHandle transition_event;
  AstraSim::Tick transition_deadline_tick = 0;  // 0 == disarmed
  uint64_t event_generation = 0;                // bumped on every arm

  // Transaction-detail stream state (sensing-gated; plan sec.5.1). The
  // stream opens lazily at the first settled row: with the switch off no
  // file is ever created; with the switch on the terminal summary row
  // guarantees a non-empty artifact at every normal end.
  bool transaction_detail_enabled_ = false;
  bool transaction_detail_finalized_ = false;
  std::string transaction_detail_path_;
  std::string transaction_run_id_;
  std::ofstream transaction_stream_;
  uint64_t transaction_rows_written_ = 0;
  void ensure_transaction_stream_open();
};
} // namespace Analytical

#endif /* __ANALYTICAL_MEMORY_HH__ */
