/******************************************************************************
This source code is licensed under the MIT license found in the LICENSE file in
the root directory of this source tree.
*******************************************************************************/

#include "extern/remote_memory_backend/analytical/AnalyticalRemoteMemory.hh"

#include <json/json.hpp>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <utility>

#include "astra-sim/system/Common.hh"
#include "astra-sim/system/Sys.hh"
#include "astra-sim/system/WorkloadLayerHandlerData.hh"
#include "astra-sim/workload/Workload.hh"

using namespace std;
using namespace AstraSim;
using namespace Analytical;
using json = nlohmann::json;

namespace Analytical {
namespace {

// Explicit residual-clamp tolerances (plan §3.2: clamps must have a declared
// boundary and must not silently swallow observable service).  A stream whose
// remaining bytes fall to <= kByteEps is completed; event instants within
// kTimeEpsNs are treated as simultaneous.
constexpr double kByteEps = 1e-6;   // bytes
constexpr double kTimeEpsNs = 1e-9; // ns

// Peak/clamp helper.
template <typename T>
void bump_to(T& current, const T& candidate) {
  if (candidate > current) {
    current = candidate;
  }
}

}  // namespace

//------------------------------------------------------------------------------
// Construction / configuration
//------------------------------------------------------------------------------

AnalyticalRemoteMemory::AnalyticalRemoteMemory(
    string memory_configuration) noexcept {
  ifstream conf_file;

  conf_file.open(memory_configuration);
  if (!conf_file) {
    cerr << "Unable to open file: " << memory_configuration << endl;
    exit(1);
  }

  json j;
  conf_file >> j;

  if (j.contains("memory-type")) {
    string mem_type_str = j["memory-type"];
    if (mem_type_str.compare("NO_MEMORY_EXPANSION") == 0) {
      mem_type = NO_MEMORY_EXPANSION;
    } else if (mem_type_str.compare("PER_NODE_MEMORY_EXPANSION") == 0) {
      mem_type = PER_NODE_MEMORY_EXPANSION;
    } else if (mem_type_str.compare("PER_NPU_MEMORY_EXPANSION") == 0) {
      mem_type = PER_NPU_MEMORY_EXPANSION;
    } else if (mem_type_str.compare("MEMORY_POOL") == 0) {
      mem_type = MEMORY_POOL;
    } else {
      cerr << "Unsupported memory type: " << mem_type_str << endl;
      exit(1);
    }
  }

  if (mem_type == PER_NODE_MEMORY_EXPANSION) {
    // Fail fast at construction on a missing/invalid node shape: silently
    // defaulting these to 0 used to construct successfully and then hit a
    // divide-by-zero at the first remote access
    // (port_index = sys_id / num_npus_per_node).
    if (!j.contains("num-nodes")) {
      cerr << "PER_NODE_MEMORY_EXPANSION requires the 'num-nodes' key"
           << endl;
      exit(1);
    }
    if (!j.contains("num-npus-per-node")) {
      cerr << "PER_NODE_MEMORY_EXPANSION requires the "
           << "'num-npus-per-node' key" << endl;
      exit(1);
    }
    if (!j["num-nodes"].is_number_unsigned() ||
        j["num-nodes"].get<uint64_t>() == 0 ||
        j["num-nodes"].get<uint64_t>() >
            static_cast<uint64_t>(numeric_limits<int>::max())) {
      cerr << "'num-nodes' must be a positive integer in the int range"
           << endl;
      exit(1);
    }
    if (!j["num-npus-per-node"].is_number_unsigned() ||
        j["num-npus-per-node"].get<uint64_t>() == 0 ||
        j["num-npus-per-node"].get<uint64_t>() >
            static_cast<uint64_t>(numeric_limits<int>::max())) {
      cerr << "'num-npus-per-node' must be a positive integer in the int "
           << "range" << endl;
      exit(1);
    }
    num_nodes = j["num-nodes"].get<int>();
    num_npus_per_node = j["num-npus-per-node"].get<int>();
  } else if (mem_type == PER_NPU_MEMORY_EXPANSION &&
             j.contains("npu-ids")) {
    per_npu_ids_configured = true;
    const json& npu_ids = j["npu-ids"];
    if (!npu_ids.is_array() || npu_ids.empty()) {
      cerr << "npu-ids must be a non-empty array for "
           << "PER_NPU_MEMORY_EXPANSION" << endl;
      exit(1);
    }

    for (const json& npu_id_json : npu_ids) {
      if (!npu_id_json.is_number_integer() &&
          !npu_id_json.is_number_unsigned()) {
        cerr << "Each npu-ids entry must be a non-negative integer" << endl;
        exit(1);
      }

      uint64_t npu_id_value;
      if (npu_id_json.is_number_unsigned()) {
        npu_id_value = npu_id_json.get<uint64_t>();
      } else {
        int64_t signed_npu_id = npu_id_json.get<int64_t>();
        if (signed_npu_id < 0) {
          cerr << "Each npu-ids entry must be a non-negative integer" << endl;
          exit(1);
        }
        npu_id_value = static_cast<uint64_t>(signed_npu_id);
      }

      if (npu_id_value > static_cast<uint64_t>(numeric_limits<int>::max())) {
        cerr << "npu-ids entry is outside the supported NPU rank range: "
             << npu_id_value << endl;
        exit(1);
      }

      int npu_id = static_cast<int>(npu_id_value);
      if (per_npu_port_indices.find(npu_id) !=
          per_npu_port_indices.end()) {
        cerr << "Duplicate NPU rank in npu-ids: " << npu_id << endl;
        exit(1);
      }

      per_npu_port_indices[npu_id] = ports_.size();
      ports_.emplace_back();
    }
  }

  // Plan §3.2: latency must be a finite non-negative number.  The serial
  // backend read this key straight into a uint64_t, which silently wrapped
  // negative JSON values and truncated floats.
  if (j.contains("remote-mem-latency")) {
    if (!j["remote-mem-latency"].is_number()) {
      cerr << "remote-mem-latency must be a number" << endl;
      exit(1);
    }
    const double latency = j["remote-mem-latency"].get<double>();
    if (!isfinite(latency) || latency < 0.0) {
      cerr << "remote-mem-latency must be finite and non-negative" << endl;
      exit(1);
    }
    if (latency > static_cast<double>(numeric_limits<Tick>::max())) {
      // A latency beyond the representable Tick range could never produce a
      // callback Tick; fail closed at construction (§3.2).
      cerr << "remote-mem-latency exceeds the representable Tick range"
           << endl;
      exit(1);
    }
    remote_mem_latency_ns = latency;
  }

  remote_mem_bw = 0;
  if (j.contains("remote-mem-bw")) {
    // §3.2: the value must be a finite positive number.  A bare `> 0` check
    // would admit JSON overflow (e.g. 1e999 parses to +inf) and only reject
    // NaN by accident of comparison semantics.
    if (mem_type != NO_MEMORY_EXPANSION &&
        (!j["remote-mem-bw"].is_number() ||
         !isfinite(j["remote-mem-bw"].get<double>()) ||
         !(j["remote-mem-bw"].get<double>() > 0))) {
      cerr << "remote-mem-bw must be positive and finite for the configured "
           << "memory type" << endl;
      exit(1);
    }
    // Read as double (same consumption type as the Sys.cc side): the old
    // uint64_t read silently truncated json floats (e.g. 512.7 -> 512).
    remote_mem_bw = j["remote-mem-bw"].get<double>();
  }

  if (mem_type != NO_MEMORY_EXPANSION &&
      (!isfinite(remote_mem_bw) || !(remote_mem_bw > 0))) {
    cerr << "remote-mem-bw must be positive and finite for the configured "
         << "memory type" << endl;
    exit(1);
  }

  if (mem_type == PER_NODE_MEMORY_EXPANSION) {
    ports_.resize(static_cast<size_t>(num_nodes));
  } else if (mem_type == MEMORY_POOL) {
    ports_.emplace_back();
  }
}

AnalyticalRemoteMemory::~AnalyticalRemoteMemory() {
  // Defensive: the host Sys must outlive the backend (main deletes/resets the
  // remote API before deleting Sys, plan §3.4); this only releases the
  // backend's own state and never fabricates normal completion.
  shutdown();
}

void AnalyticalRemoteMemory::set_sys(int id, Sys* sys) {
  sys_map[id] = sys;
  // §3.3: the global transition event hangs off the FIRST set_sys host;
  // construction precedes set_sys, so the host can only be bound here.
  if (host_sys_ == nullptr && sys != nullptr) {
    host_sys_ = sys;
  }
  if (mem_type == PER_NPU_MEMORY_EXPANSION && !per_npu_ids_configured &&
      per_npu_port_indices.find(id) == per_npu_port_indices.end()) {
    per_npu_port_indices[id] = ports_.size();
    ports_.emplace_back();
  }
}

//------------------------------------------------------------------------------
// Fail-closed helpers
//------------------------------------------------------------------------------

void AnalyticalRemoteMemory::fatal(const string& msg) {
  // Same hard-exit style as the constructor's config checks: exceptions are
  // swallowed by Sys::call_events and would silently corrupt the dispatch.
  cerr << "[AnalyticalRemoteMemory] FATAL: " << msg << endl;
  exit(1);
}

void AnalyticalRemoteMemory::free_backend_event_payload(CallData* data) {
  // register_event_cancellable deleter (§3.3): sole releaser of the payload
  // on the cancellation/shutdown paths.  The payload is a plain value struct
  // (kind + sequence) with no back-references, so deleting it here is always
  // sufficient.
  delete static_cast<BackendEventPayload*>(data);
}

Tick AnalyticalRemoteMemory::ceil_tick(double t_ns) {
  // §3.2: range-check before ceil so an unrepresentable callback Tick fails
  // closed instead of wrapping.
  if (!isfinite(t_ns)) {
    fatal("non-finite completion time");
  }
  const double c = ceil(t_ns);
  if (c < 0.0 || c > static_cast<double>(numeric_limits<Tick>::max())) {
    fatal("callback Tick out of representable range");
  }
  return static_cast<Tick>(c);
}

//------------------------------------------------------------------------------
// Fluid core
//------------------------------------------------------------------------------

// Flips every job whose event point lies at t: latency joins, zero-byte
// latency completions, and stream exhaustions.  Doing this before the rate
// computation keeps exhausted streams out of the new denominator (plan §3.1)
// and guarantees loop progress (every remaining event point is strictly after
// t).  Also settles the redistribution/peak counters for the instant.
AnalyticalRemoteMemory::FlipSummary
AnalyticalRemoteMemory::flip_due_jobs(PortState& port, const double t) {
  FlipSummary summary;
  const size_t n_stream_before = count_if(
      port.jobs.begin(), port.jobs.end(),
      [](const PortJob& job) { return job.state == PortJobState::Streaming; });
  for (auto& job : port.jobs) {
    if (job.state == PortJobState::LatencyWaiting &&
        job.latency_ready_ns <= t + kTimeEpsNs) {
      if (port.stats.latency_waiting_count == 0) {
        fatal("latency-waiting live count underflow at the latency flip");
      }
      --port.stats.latency_waiting_count;
      if (job.bytes == 0) {
        // Zero-byte positive-latency: no bandwidth share ever (§3.2).
        job.state = PortJobState::FluidCompleteAwaitingCallback;
        job.remaining_bytes = 0.0;
        job.fluid_finish_ns = job.latency_ready_ns;
        job.callback_tick = ceil_tick(job.fluid_finish_ns);
        ++port.stats.completion_waiting_count;
        if (job.telemetry) {
          // Never streamed: the stream fields stay unset (written as null).
          job.telemetry->callback_tick = job.callback_tick;
        }
      } else {
        job.state = PortJobState::Streaming;
        job.remaining_bytes = static_cast<double>(job.bytes);
        ++port.stats.streaming_count;
        ++summary.joins;
        if (job.telemetry && !job.telemetry->streamed) {
          // §5.1: stream_start_ns is the FIRST stream start.
          job.telemetry->streamed = true;
          job.telemetry->stream_start_ns = t;
        }
      }
    } else if (job.state == PortJobState::Streaming &&
               job.remaining_bytes <= kByteEps) {
      // Fluid completion: leaves the denominator now; the Workload callback
      // lands at ceil(fluid_finish_ns) (plan §3.1).
      if (port.stats.streaming_count == 0) {
        fatal("streaming live count underflow at the fluid completion flip");
      }
      --port.stats.streaming_count;
      job.state = PortJobState::FluidCompleteAwaitingCallback;
      job.remaining_bytes = 0.0;
      job.fluid_finish_ns = t;
      job.callback_tick = ceil_tick(t);
      ++port.stats.completion_waiting_count;
      ++summary.completions;
      summary.stream_completed = true;
      if (job.telemetry) {
        // §5.1: fluid_finish_ns is the LAST byte's continuous instant.
        job.telemetry->streamed = true;
        job.telemetry->fluid_finish_ns = t;
        job.telemetry->callback_tick = job.callback_tick;
      }
    }
  }
  if (summary.completions > 0) {
    const size_t survivors =
        n_stream_before + summary.joins - summary.completions;
    // One share change per instant with survivors, however many streams
    // exhausted simultaneously (plan §5.1).
    if (survivors > 0) {
      ++port.stats.redistribution_events;
    }
  }
  if (summary.joins > 0 && n_stream_before > summary.completions) {
    // Join-driven share change, counted separately from completion-driven
    // redistribution (plan §5.1 "另计或另列").
    ++port.stats.stream_join_events;
  }
  size_t n_stream_after = 0;
  for (const auto& job : port.jobs) {
    if (job.state == PortJobState::Streaming) {
      ++n_stream_after;
    }
  }
  if (n_stream_after != port.stats.streaming_count) {
    // Drift between the maintained live counter and the container is an
    // invariant violation, never a benign bookkeeping lag.
    fatal("streaming live-count drift against the job container");
  }
  bump_to(port.stats.peak_streaming, static_cast<uint64_t>(n_stream_after));
  return summary;
}

double AnalyticalRemoteMemory::advance_port(const double bw_bytes_per_ns,
                                            PortState& port, double from_ns,
                                            const double to_ns,
                                            const bool stop_at_first_completion) {
  double t = from_ns;
  while (to_ns - t > kTimeEpsNs || stop_at_first_completion) {
    // (A) Sweep the event point at t.
    const FlipSummary flip = flip_due_jobs(port, t);
    if (stop_at_first_completion && flip.stream_completed) {
      return t;
    }

    // (B) Next event point strictly after t, bounded by to_ns.
    size_t n_stream = 0;
    double next_event = to_ns;
    for (const auto& job : port.jobs) {
      if (job.state == PortJobState::Streaming) {
        ++n_stream;
      }
    }
    double rate = 0.0;
    if (n_stream > 0) {
      rate = bw_bytes_per_ns / static_cast<double>(n_stream);
      // §3.2: a rate that underflowed to (or below) zero cannot produce a
      // representable completion instant; fail closed instead of fabricating
      // +inf event points or zero-step loops.
      if (!(rate > 0.0) || !isfinite(rate)) {
        fatal("equal-share stream rate underflowed or is not finite");
      }
    }
    for (const auto& job : port.jobs) {
      if (job.state == PortJobState::LatencyWaiting &&
          job.latency_ready_ns < next_event) {
        next_event = job.latency_ready_ns;
      }
      if (job.state == PortJobState::Streaming && n_stream > 0) {
        const double completion = t + job.remaining_bytes / rate;
        if (completion < next_event) {
          next_event = completion;
        }
      }
    }
    if (stop_at_first_completion && !isfinite(next_event)) {
      // Peek with no further event: no completion exists ahead.
      return next_event;
    }
    const double t_next = min(next_event, to_ns);
    if (t_next - t <= kTimeEpsNs) {
      // Unreachable after the (A) sweep: every surviving event point is
      // strictly beyond t by at least kByteEps/rate or the remaining latency.
      // If it ever happens, fail closed instead of stalling the loop.
      if (to_ns - t > kTimeEpsNs && t_next < to_ns) {
        fatal("zero-progress fluid advance");
      }
      t = t_next;
      break;
    }

    // (C) Integrate [t, t_next): every stream holds the equal-share rate for
    // the whole interval because all event points are >= t_next (plan §3.1).
    if (n_stream > 0) {
      const double dt = t_next - t;
      port.stats.port_busy_ns += dt;
      if (n_stream >= 2) {
        port.stats.shared_busy_ns += dt;
      }
      port.stats.bytes_served += bw_bytes_per_ns * dt;
      for (auto& job : port.jobs) {
        if (job.state == PortJobState::Streaming) {
          job.remaining_bytes =
              max(0.0, job.remaining_bytes - rate * dt);
        }
      }
    }
    t = t_next;
  }
  // (A') Final sweep: event points exactly at to_ns.  The loop condition
  // stops integration at to_ns before their flip (e.g. a stream whose
  // completion instant is exactly the target Tick).
  flip_due_jobs(port, to_ns);
  return to_ns;
}

void AnalyticalRemoteMemory::advance_all_ports_to(const double t_ns) {
  if (t_ns < current_time_ns_ - kTimeEpsNs) {
    fatal("time regression in fluid advance");
  }
  for (PortState& port : ports_) {
    advance_port(remote_mem_bw, port, current_time_ns_, t_ns,
                 /*stop_at_first_completion=*/false);
  }
  current_time_ns_ = t_ns;
}

Tick AnalyticalRemoteMemory::earliest_callback_tick() const {
  double best_ns = numeric_limits<double>::infinity();
  for (const PortState& port : ports_) {
    for (const PortJob& job : port.jobs) {
      if (job.state == PortJobState::FluidCompleteAwaitingCallback) {
        best_ns = min(best_ns, static_cast<double>(job.callback_tick));
      } else if (job.state == PortJobState::LatencyWaiting &&
                 job.bytes == 0) {
        // Zero-byte positive-latency completion (exact; never streams).
        best_ns = min(best_ns, job.latency_ready_ns);
      }
    }
  }
  // Exact first streaming completion per port: latency joins happen at known
  // instants, so a read-only piecewise replay gives the exact instant and the
  // transition event never fires before any due callback.
  for (const PortState& port : ports_) {
    const bool has_streaming_or_joining = any_of(
        port.jobs.begin(), port.jobs.end(), [](const PortJob& job) {
          return job.state == PortJobState::Streaming ||
                 (job.state == PortJobState::LatencyWaiting &&
                  job.bytes > 0);
        });
    if (!has_streaming_or_joining) {
      continue;
    }
    // Read-only replay on a throwaway scalar clone: telemetry records and
    // wlhd ownership are NEVER copied, so the replay can neither observe nor
    // disturb the per-transaction observation state (plan §5.1).
    PortState peek = make_stats_replay_copy(port);
    const double first_completion =
        advance_port(remote_mem_bw, peek, current_time_ns_,
                     numeric_limits<double>::infinity(),
                     /*stop_at_first_completion=*/true);
    best_ns = min(best_ns, first_completion);
  }
  if (!isfinite(best_ns)) {
    return 0;  // nothing pending
  }
  return ceil_tick(best_ns);
}

//------------------------------------------------------------------------------
// Global transition event
//------------------------------------------------------------------------------

void AnalyticalRemoteMemory::replan_transition() {
  // §3.3: the global transition event hangs on the FIRST set_sys host --
  // never on the issuing rank's Sys (owner-checked cancels would fail and
  // the plan mandates a single host for all backend events).
  Sys* sys = host_sys_;
  if (sys == nullptr) {
    fatal("replan before set_sys");
  }
  const bool any_jobs =
      any_of(ports_.begin(), ports_.end(),
             [](const PortState& port) { return !port.jobs.empty(); });
  if (!any_jobs) {
    if (transition_handle_.valid()) {
      if (!sys->cancel_event(transition_handle_)) {
        // deleter frees the payload; a false return means the queued event
        // was already gone -- a bookkeeping inconsistency.
        fatal("cancel of the tracked transition event failed (empty replan)");
      }
      transition_handle_.reset();
      transition_tick_ = 0;
      ++event_stats_.transition_cancels;
    }
    return;
  }

  const Tick earliest = earliest_callback_tick();
  if (transition_handle_.valid()) {
    if (transition_tick_ <= earliest) {
      // §3.3 keep rule: the queued transition -- including one due exactly at
      // the current Tick -- is kept; a strictly-earlier remembered deadline
      // also keeps covering the new earliest.  Cancelling it could only defer
      // callbacks past their completion Tick, which is forbidden.
      ++event_stats_.transition_keeps;
      return;
    }
    // Deadline-diff replace: the new earliest is strictly earlier than the
    // queued Tick, so the queued event is superseded.  cancel_event runs the
    // registration deleter, releasing the superseded payload (§3.3).
    if (!sys->cancel_event(transition_handle_)) {
      fatal("cancel of the superseded transition event failed");
    }
    transition_handle_.reset();
    transition_tick_ = 0;
    ++event_stats_.transition_cancels;
  }

  const Tick now = Sys::boostedTick();
  if (earliest <= now) {
    // A due completion without a due event is a state inconsistency; the
    // batch at `now` has already been harvested on the dispatch path.
    fatal("transition deadline regressed to or past the current Tick");
  }
  ++transition_generation_;
  transition_tick_ = earliest;
  transition_handle_ = sys->register_event_cancellable(
      this, EventType::General,
      new BackendEventPayload(BackendEventPayload::Kind::Transition,
                              transition_generation_),
      earliest - now, &AnalyticalRemoteMemory::free_backend_event_payload);
  ++event_stats_.transition_registrations;
}

void AnalyticalRemoteMemory::dispatch_transitions() {
  in_dispatch_ = true;
  const Tick now = Sys::boostedTick();
  // §3.3: advance every port's continuous substeps to the dispatch Tick
  // before touching the job sets, so service performed since the last event
  // is never lost.
  advance_all_ports_to(static_cast<double>(now) * CLOCK_PERIOD);

  // Collect ALL due completions across ports and remove them from the active
  // containers as one batch (same-Tick completions leave together, plan
  // §3.1/§3.3).  Per-port scan order preserves issue_sequence order.
  struct DueRef {
    size_t port_index;
    uint64_t issue_seq;
    PortJob job;  // moved out; uniquely holds the wlhd until delivery
  };
  vector<DueRef> due;
  for (size_t p = 0; p < ports_.size(); ++p) {
    PortState& port = ports_[p];
    for (PortJob& job : port.jobs) {
      if (job.state == PortJobState::FluidCompleteAwaitingCallback) {
        due.push_back({p, job.issue_seq, std::move(job)});
      }
    }
    port.jobs.erase(remove_if(port.jobs.begin(), port.jobs.end(),
                              [](const PortJob& job) {
                                return job.state ==
                                       PortJobState::
                                           FluidCompleteAwaitingCallback;
                              }),
                    port.jobs.end());
  }

  // Completion ordering key (§3.3): port_index ascending, then per-port
  // issue_sequence ascending.
  sort(due.begin(), due.end(), [](const DueRef& a, const DueRef& b) {
    if (a.port_index != b.port_index) {
      return a.port_index < b.port_index;
    }
    return a.issue_seq < b.issue_seq;
  });

  // §3.4/§5.1: settle per-port statistics and stream the per-transaction
  // detail rows BEFORE any Workload callback (the callback hands wlhd away;
  // nothing may be observed through it afterwards).
  for (DueRef& d : due) {
    PortStats& stats = ports_[d.port_index].stats;
    ++stats.completed_count;
    stats.completed_bytes += d.job.bytes;
    if (stats.completion_waiting_count > 0) {
      --stats.completion_waiting_count;
    } else {
      fatal("completion for a transaction not awaiting its callback");
    }
    if (stats.in_flight_count > 0) {
      --stats.in_flight_count;
    } else {
      fatal("completion for a transaction not in flight");
    }
    if (d.job.telemetry) {
      write_transaction_row(*d.job.telemetry);
      d.job.telemetry.reset();  // streamed; nothing is kept post-callback
    }
  }

  // Then hand each wlhd to its Workload.  Callbacks may synchronously re-enter
  // Workload and issue new transactions (they only advance state and replan);
  // this loop never re-dispatches and never reorders the batch.
  for (const DueRef& d : due) {
    d.job.wlhd->workload->call(EventType::General, d.job.wlhd);
  }
  in_dispatch_ = false;

  // Register the next transition only after the whole batch is delivered.
  replan_transition();
}

//------------------------------------------------------------------------------
// Dual-zero one-shot timers (§3.2)
//------------------------------------------------------------------------------

void AnalyticalRemoteMemory::deliver_dual_zero(const uint64_t timer_seq) {
  vector<DualZeroJob>::iterator it = find_if(
      dual_zero_timers_.begin(), dual_zero_timers_.end(),
      [timer_seq](const DualZeroJob& job) { return job.timer_seq == timer_seq; });
  if (it == dual_zero_timers_.end()) {
    fatal("dual-zero timer fired without a live job");
  }
  DualZeroJob job = std::move(*it);
  dual_zero_timers_.erase(it);

  // §5.1: statistics and the detail row are settled BEFORE the Workload
  // callback (same discipline as the transition-dispatch batch).
  PortStats& stats = ports_.at(job.port_index).stats;
  ++stats.completed_count;
  if (stats.in_flight_count > 0) {
    --stats.in_flight_count;
  } else {
    fatal("dual-zero completion for a transaction not in flight");
  }
  if (job.telemetry) {
    write_transaction_row(*job.telemetry);
    job.telemetry.reset();  // streamed; nothing is kept post-callback
  }
  // +1ns one-shot delivery, asynchronous from the issue stack (§3.2).
  job.wlhd->workload->call(EventType::General, job.wlhd);
}

//------------------------------------------------------------------------------
// Event dispatch entry (both event kinds)
//------------------------------------------------------------------------------

void AnalyticalRemoteMemory::call(EventType /*type*/, CallData* data) {
  // §3.3: release the payload at callback entry and clear the dequeued
  // handle; a stale generation is a no-op (its payload was already released
  // here); illegal states terminate instead of throwing.
  BackendEventPayload* payload = nullptr;
  if (data == nullptr) {
    fatal("backend event dispatched with a null payload");
  }
  payload = static_cast<BackendEventPayload*>(data);
  const BackendEventPayload::Kind kind = payload->kind;
  const uint64_t seq = payload->seq;
  delete payload;
  ++event_stats_.payloads_freed_at_dispatch;

  if (kind == BackendEventPayload::Kind::Transition) {
    transition_handle_.reset();
    if (seq != transition_generation_ || transition_tick_ == 0) {
      // Superseded/stale generation: normal no-op (§3.3).
      ++event_stats_.stale_generation_dispatches;
      return;
    }
    transition_tick_ = 0;
    dispatch_transitions();
  } else {
    deliver_dual_zero(seq);
  }
}

//------------------------------------------------------------------------------
// Issue path
//------------------------------------------------------------------------------

size_t AnalyticalRemoteMemory::resolve_port_index(const int sys_id) const {
  if (mem_type == PER_NODE_MEMORY_EXPANSION) {
    if (sys_id < 0 || num_npus_per_node <= 0) {
      fatal("invalid sys_id/num_npus_per_node in PER_NODE port resolution");
    }
    const uint64_t port =
        static_cast<uint64_t>(sys_id) /
        static_cast<uint64_t>(num_npus_per_node);
    if (port >= ports_.size()) {
      // Out-of-range sys_id used to index the old boolean vector unchecked;
      // the fluid job containers make the same mistake fatal instead of UB.
      fatal("PER_NODE sys_id maps outside the configured node range");
    }
    return static_cast<size_t>(port);
  }
  if (mem_type == PER_NPU_MEMORY_EXPANSION) {
    const auto it = per_npu_port_indices.find(sys_id);
    if (it == per_npu_port_indices.end()) {
      cerr << "NPU rank " << sys_id
           << " does not have a configured remote-memory port" << endl;
      exit(1);
    }
    return it->second;
  }
  if (mem_type == MEMORY_POOL) {
    return 0;  // all transactions share one logical port
  }
  fatal("port resolution for NO_MEMORY_EXPANSION");
}

Sys* AnalyticalRemoteMemory::resolve_sys(const int sys_id) const {
  const auto it = sys_map.find(sys_id);
  // The serial backend used sys_map[sys_id] here, which silently inserted a
  // nullptr for an unknown id and crashed later; fail closed instead.
  if (it == sys_map.end() || it->second == nullptr) {
    fatal("remote memory issue on a rank without a bound Sys");
  }
  return it->second;
}

void AnalyticalRemoteMemory::issue(uint64_t tensor_size,
                                   WorkloadLayerHandlerData* wlhd) {
  if (mem_type == NO_MEMORY_EXPANSION) {
    cerr << "Remote memory access is not supported in NO_MEMORY_EXPANSION"
         << endl;
    exit(1);
  }
  if (wlhd == nullptr) {
    fatal("issue with a null wlhd");
  }
  if (shutdown_done_) {
    fatal("issue after shutdown");
  }
  // §3.2: huge byte counts fail closed.  Past 2^53 a uint64 byte count is no
  // longer exactly representable as a double, which is the type every fluid
  // service computation consumes; refuse instead of silently rounding.  The
  // check sits before port/sys resolution so it fires even on an unbound
  // rank.
  constexpr uint64_t kMaxExactBytes = 1ULL << 53;  // 2^53: double exact-integer limit
  if (tensor_size > kMaxExactBytes) {
    fatal("tensor_size exceeds the double-exact byte range (2^53)");
  }

  const size_t port_index = resolve_port_index(wlhd->sys_id);
  // Fail-closed rank check: the issuing rank must have a bound Sys even
  // though the events themselves all register on the first-set_sys host.
  resolve_sys(wlhd->sys_id);
  PortState& port = ports_[port_index];
  PortStats& stats = port.stats;

  // §3.3: every external issue first advances all port state to the current
  // integer Tick, then updates the job sets.
  const Tick now = Sys::boostedTick();
  const double now_ns = static_cast<double>(now) * CLOCK_PERIOD;
  advance_all_ports_to(now_ns);
  const double ready_ns = now_ns + remote_mem_latency_ns;
  if (!isfinite(ready_ns)) {
    // latency is finite and Tick-bounded from the constructor and now_ns is
    // finite, so this is defensive; §3.2 forbids NaN/inf delays outright.
    fatal("non-finite latency-ready instant");
  }

  ++stats.issued_count;
  stats.issued_bytes += tensor_size;
  ++stats.in_flight_count;
  bump_to(stats.peak_in_flight, stats.in_flight_count);

  // §5.1: the per-port monotonic issue_sequence is allocated for EVERY issue
  // (dual-zero included) so the transaction-detail row key is total over the
  // port's transaction stream.  Dual-zero jobs never enter the port job set,
  // so the completion-order key is unaffected.
  const uint64_t issue_seq = port.next_issue_seq++;

  // §5.1: the observation copy is created at issue -- original bytes, port,
  // issue Tick, rank and node id are captured here and never re-read from
  // the wlhd after the callback (which destroys the cookie).
  std::unique_ptr<TransactionRecord> telemetry;
  if (telemetry_enabled_) {
    telemetry = std::make_unique<TransactionRecord>();
    telemetry->issue_seq = issue_seq;
    telemetry->sys_id = wlhd->sys_id;
    telemetry->node_id = wlhd->node_id;
    telemetry->bytes = tensor_size;
    telemetry->port_index = port_index;
    telemetry->issue_tick = now;
    telemetry->latency_ready_ns = ready_ns;
  }

  if (tensor_size == 0 && remote_mem_latency_ns == 0.0) {
    // §3.2 semantic change: 0B/0ns no longer completes on the issue stack.
    // It runs as an independent one-shot timer at +1ns, outside the port job
    // set and never inside any bandwidth denominator.
    // §3.3: dual-zero timers share the same single host Sys as the global
    // transition event; delivery itself goes directly to the Workload, so
    // the issuing rank's Sys is never an event host for backend state.
    if (host_sys_ == nullptr) {
      fatal("issue before set_sys");
    }
    DualZeroJob job;
    job.timer_seq = ++next_timer_seq_;
    job.issue_seq = issue_seq;
    job.port_index = port_index;
    job.fire_tick = now + 1;
    job.wlhd = wlhd;
    if (telemetry) {
      telemetry->callback_tick = job.fire_tick;
      job.telemetry = std::move(telemetry);
    }
    job.handle = host_sys_->register_event_cancellable(
        this, EventType::General,
        new BackendEventPayload(BackendEventPayload::Kind::DualZeroTimer,
                                job.timer_seq),
        /*delta_cycles=*/1,
        &AnalyticalRemoteMemory::free_backend_event_payload);
    dual_zero_timers_.push_back(std::move(job));
    return;
  }

  PortJob job;
  job.issue_seq = issue_seq;
  job.bytes = tensor_size;
  job.latency_ready_ns = ready_ns;
  job.remaining_bytes = static_cast<double>(tensor_size);
  job.state = PortJobState::LatencyWaiting;
  job.wlhd = wlhd;
  if (telemetry) {
    job.telemetry = std::move(telemetry);
  }
  ++stats.latency_waiting_count;
  port.jobs.push_back(std::move(job));

  // Deadline-diff replan: keeps the queued transition when it still covers
  // the new earliest deadline (the §3.3 same-Tick keep rule lives here).
  replan_transition();
}

//------------------------------------------------------------------------------
// Read-only statistics snapshot (§5.1)
//------------------------------------------------------------------------------

std::size_t AnalyticalRemoteMemory::port_count() const { return ports_.size(); }

AnalyticalRemoteMemory::PortStatsSnapshot
AnalyticalRemoteMemory::port_stats(const std::size_t port_index) const {
  if (port_index >= ports_.size()) {
    fatal("port_stats index outside the configured port range");
  }
  const PortStats& s = ports_[port_index].stats;
  PortStatsSnapshot snap;
  snap.issued_count = s.issued_count;
  snap.completed_count = s.completed_count;
  snap.issued_bytes = s.issued_bytes;
  snap.completed_bytes = s.completed_bytes;
  snap.in_flight_count = s.in_flight_count;
  snap.peak_in_flight = s.peak_in_flight;
  snap.streaming_count = s.streaming_count;
  snap.peak_streaming = s.peak_streaming;
  snap.latency_waiting_count = s.latency_waiting_count;
  snap.completion_waiting_count = s.completion_waiting_count;
  snap.redistribution_events = s.redistribution_events;
  snap.stream_join_events = s.stream_join_events;
  snap.port_busy_ns = s.port_busy_ns;
  snap.shared_busy_ns = s.shared_busy_ns;
  snap.bytes_served = s.bytes_served;
  return snap;
}

AnalyticalRemoteMemory::PortState
AnalyticalRemoteMemory::make_stats_replay_copy(const PortState& port) {
  PortState copy;
  copy.stats = port.stats;
  copy.next_issue_seq = port.next_issue_seq;
  copy.jobs.reserve(port.jobs.size());
  for (const PortJob& job : port.jobs) {
    PortJob scalar_clone;
    scalar_clone.issue_seq = job.issue_seq;
    scalar_clone.bytes = job.bytes;
    scalar_clone.latency_ready_ns = job.latency_ready_ns;
    scalar_clone.remaining_bytes = job.remaining_bytes;
    scalar_clone.fluid_finish_ns = job.fluid_finish_ns;
    scalar_clone.callback_tick = job.callback_tick;
    scalar_clone.state = job.state;
    // wlhd ownership and the telemetry record are intentionally NOT copied.
    copy.jobs.push_back(std::move(scalar_clone));
  }
  return copy;
}

//------------------------------------------------------------------------------
// Fail-closed settlement audit and drain check (§3.4/§5.1)
//------------------------------------------------------------------------------

void AnalyticalRemoteMemory::verify_settled_consistency() const {
  uint64_t total_completed = 0;
  for (const PortState& port : ports_) {
    const PortStats& s = port.stats;
    if (s.issued_count != s.completed_count) {
      fatal("settled port statistics: issued_count != completed_count");
    }
    if (s.issued_bytes != s.completed_bytes) {
      // Count equality alone must never mask a byte mismatch (§5.1).
      fatal("settled port statistics: issued_bytes != completed_bytes");
    }
    if (s.in_flight_count != 0) {
      fatal("settled port statistics: nonzero in_flight_count");
    }
    // The container must agree with the maintained live counts (both must be
    // empty on a settled port).
    if (!port.jobs.empty() || s.streaming_count != 0 ||
        s.latency_waiting_count != 0 || s.completion_waiting_count != 0) {
      fatal("settled port statistics: nonzero live per-state counts");
    }
    total_completed += s.completed_count;
  }
  if (!dual_zero_timers_.empty()) {
    fatal("settled backend still holds dual-zero timers");
  }
  if (transition_handle_.valid()) {
    fatal("settled backend still holds the transition event");
  }
  if (telemetry_enabled_) {
    // §5.1: every settled completion has exactly one streamed detail row.
    if (telemetry_rows_written_ != total_completed) {
      fatal("transaction-detail row count != settled completions");
    }
    if (telemetry_open_ && !telemetry_stream_.good()) {
      fatal("transaction-detail stream is in a failed state");
    }
  }
}

bool AnalyticalRemoteMemory::is_drained() const {
  if (in_dispatch_) {
    return false;
  }
  if (transition_handle_.valid()) {
    return false;
  }
  if (!dual_zero_timers_.empty()) {
    return false;
  }
  for (const PortState& port : ports_) {
    if (!port.jobs.empty()) {
      return false;
    }
  }
  // §3.4/§5.1: the empty-set answer is only trustworthy when the settled
  // statistics and the telemetry handle are consistent as well.  This runs
  // unconditionally -- sensing-off runs keep the aggregate counters that
  // back this audit.  (After an EARLY shutdown the audit no longer applies:
  // released undelivered transactions intentionally leave issued/completed
  // unequal, and a shutdown() on a settled backend has already audited.)
  if (!shutdown_done_) {
    verify_settled_consistency();
  }
  return true;
}

void AnalyticalRemoteMemory::shutdown() {
  if (shutdown_done_) {
    return;
  }
  shutdown_done_ = true;

  const bool settled =
      !in_dispatch_ && !transition_handle_.valid() &&
      dual_zero_timers_.empty() &&
      all_of(ports_.begin(), ports_.end(),
             [](const PortState& port) { return port.jobs.empty(); });
  if (settled) {
    // Normal end (§3.4): the full unconditional settlement audit -- issued/
    // completed counts and bytes per port, zero live state, telemetry handle
    // consistency -- must pass before the backend is released.  Sensing-off
    // runs are audited from the same aggregate counters.
    verify_settled_consistency();
  }

  // Cancel the global transition event first: cancel_event runs the
  // registration deleter, which releases the payload (§3.3/§3.4).
  if (transition_handle_.valid()) {
    if (host_sys_ == nullptr) {
      fatal("shutdown with a queued event but no host Sys");
    }
    if (!host_sys_->cancel_event(transition_handle_)) {
      fatal("cancel of the tracked transition event failed on shutdown");
    }
    ++event_stats_.transition_cancels;
    transition_handle_.reset();
    transition_tick_ = 0;
  }

  // Undelivered dual-zero timers: cancel each (deleter frees the payload)
  // and destroy the undelivered wlhd the backend still owns.  Their
  // observation records are destroyed unstreamed: the detail file holds
  // completions only, never fabricates them.
  for (DualZeroJob& job : dual_zero_timers_) {
    if (job.handle.valid()) {
      if (host_sys_ == nullptr) {
        fatal("shutdown with a live timer but no host Sys");
      }
      if (!host_sys_->cancel_event(job.handle)) {
        fatal("cancel of a dual-zero timer failed on shutdown");
      }
    }
    delete job.wlhd;
    job.wlhd = nullptr;
    job.telemetry.reset();
  }
  dual_zero_timers_.clear();

  // Undelivered port jobs in any state: the backend owns their wlhd until
  // delivery, so early shutdown destroys them (§3.4).  Cumulative stats are
  // kept for diagnostics; the job sets are emptied and the live per-state
  // counts are recounted so a follow-up is_drained()-style check sees the
  // release.
  for (PortState& port : ports_) {
    for (PortJob& job : port.jobs) {
      delete job.wlhd;
      job.wlhd = nullptr;
      job.telemetry.reset();
    }
    port.jobs.clear();
    port.stats.streaming_count = 0;
    port.stats.latency_waiting_count = 0;
    port.stats.completion_waiting_count = 0;
  }

  // §5.1: the detail stream is flushed and closed on both the normal-end and
  // the early-shutdown path; a flush failure at close is a real data-loss
  // signal and must not pass silently.
  if (telemetry_open_) {
    telemetry_stream_.flush();
    if (!telemetry_stream_.good()) {
      fatal("transaction-detail stream flush failed on shutdown");
    }
    telemetry_stream_.close();
    telemetry_open_ = false;
  }
}

//------------------------------------------------------------------------------
// Transaction-detail telemetry (§5.1, sensing-only)
//------------------------------------------------------------------------------

void AnalyticalRemoteMemory::enable_transaction_telemetry(
    const string& bridge_dir, const string& run_id) {
  if (telemetry_enabled_) {
    fatal("transaction telemetry enabled twice");
  }
  if (shutdown_done_) {
    fatal("transaction telemetry enabled after shutdown");
  }
  if (bridge_dir.empty()) {
    fatal("transaction telemetry requires a non-empty bridge_dir");
  }
  if (run_id.empty()) {
    // The row key's run association must be resolved by the caller
    // (metrics-manifest run_id, or the fully normalized bridge directory).
    fatal("transaction telemetry requires a non-empty run association key");
  }
  for (const PortState& port : ports_) {
    if (port.stats.issued_count != 0) {
      fatal("transaction telemetry enabled after the first issue");
    }
  }
  if (!dual_zero_timers_.empty()) {
    fatal("transaction telemetry enabled with live dual-zero timers");
  }
  telemetry_bridge_dir_ = bridge_dir;
  telemetry_run_id_ = run_id;
  telemetry_enabled_ = true;
}

void AnalyticalRemoteMemory::open_telemetry_stream() {
  // Lazy creation (§5.1): the file appears with the FIRST completed
  // transaction, so runs without remote transactions leave zero residue.
  const string path = telemetry_bridge_dir_ + "/remote_memory_transactions.jsonl";
  telemetry_stream_.open(path, std::ios::out | std::ios::trunc);
  if (!telemetry_stream_.is_open() || !telemetry_stream_.good()) {
    fatal("cannot open the transaction-detail stream at " + path);
  }
  telemetry_open_ = true;
}

void AnalyticalRemoteMemory::write_transaction_row(
    const TransactionRecord& record) {
  if (!telemetry_enabled_) {
    fatal("transaction-detail row write with telemetry disabled");
  }
  if (!telemetry_open_) {
    open_telemetry_stream();
  }
  json row;
  row["schema"] = 1;
  row["type"] = "remote_memory_transaction";
  row["run_id"] = telemetry_run_id_;
  row["sys_id"] = record.sys_id;  // issuing rank
  row["node_id"] = record.node_id;
  row["issue_sequence"] = record.issue_seq;
  row["port_index"] = record.port_index;
  row["bytes"] = record.bytes;
  row["issue_tick"] = record.issue_tick;
  row["latency_ready_ns"] = record.latency_ready_ns;
  if (record.streamed) {
    row["stream_start_ns"] = record.stream_start_ns;  // first stream start
    row["fluid_finish_ns"] = record.fluid_finish_ns;  // last byte
  } else {
    // Zero-byte transactions never join the streaming set: no stream instants.
    row["stream_start_ns"] = nullptr;
    row["fluid_finish_ns"] = nullptr;
  }
  row["callback_tick"] = record.callback_tick;
  telemetry_stream_ << row.dump() << '\n';
  telemetry_stream_.flush();
  if (!telemetry_stream_.good()) {
    fatal("transaction-detail row write failed");
  }
  ++telemetry_rows_written_;
}

}  // namespace Analytical
