/******************************************************************************
This source code is licensed under the MIT license found in the LICENSE file in
the root directory of this source tree.
*******************************************************************************/

// 2026-09-24 SerDes port fluid-concurrency rework (plan sec.1/sec.3). The
// serial one-transaction-at-a-time FIFO (PendingMemoryRequest,
// pending_requests, ongoing_transaction, start_request and its
// get_remote_mem_runtime-driven issue path) was deleted; this file now owns
// the per-port fluid model, the single global cancellable transition event,
// the dual-zero one-shot timers, the numeric fail-closed boundaries and the
// ownership/drain rules. See the class comment in the header for the model
// contract.

#include "extern/remote_memory_backend/analytical/AnalyticalRemoteMemory.hh"
#include <json/json.hpp>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include "astra-sim/system/Common.hh"
#include "astra-sim/system/WorkloadLayerHandlerData.hh"

using namespace std;
using namespace AstraSim;
using namespace Analytical;
using json = nlohmann::json;

namespace {
// plan sec.3.2 explicit numerical boundaries.
constexpr double kTimeEpsNs = 1e-9;  // sub-nanosecond service-time tolerance
constexpr double kByteEps = 1e-6;    // residual byte tolerance on completion
// 2^53: doubles stay integer-exact at or below this many ns/units; anything
// larger would silently lose nanosecond (or byte) granularity and is
// rejected fail-closed instead.
constexpr double kMaxTimeNs = 9007199254740992.0;
constexpr uint64_t kMaxExactBytes = 1ULL << 53;

void add_bytes_fail_closed(uint64_t& total, const uint64_t bytes) {
  if (total > numeric_limits<uint64_t>::max() - bytes) {
    Sys::sys_panic(
        "AnalyticalRemoteMemory: issued-bytes counter would overflow");
  }
  total += bytes;
}
}  // namespace

AnalyticalRemoteMemory::AnalyticalRemoteMemory(
    string memory_configuration) {
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
    num_nodes = 0;
    if (j.contains("num-nodes")) {
      num_nodes = j["num-nodes"];
    }
    num_npus_per_node = 0;
    if (j.contains("num-npus-per-node")) {
      num_npus_per_node = j["num-npus-per-node"];
    }
    if (num_nodes <= 0 || num_npus_per_node <= 0) {
      cerr << "num-nodes and num-npus-per-node must be positive for "
           << "PER_NODE_MEMORY_EXPANSION" << endl;
      exit(1);
    }
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

      per_npu_port_indices[npu_id] = ports.size();
      ports.emplace_back();
    }
  }

  // plan sec.3.2: remote_mem_bw must be positive and finite; latency must
  // be finite and non-negative. The worktree float-bandwidth validation
  // below is preserved, only extended with the finite checks.
  remote_mem_latency = 0;
  if (j.contains("remote-mem-latency")) {
    if (!j["remote-mem-latency"].is_number()) {
      cerr << "remote-mem-latency must be a number" << endl;
      exit(1);
    }
    const double latency_value = j["remote-mem-latency"].get<double>();
    if (!isfinite(latency_value) || latency_value < 0) {
      cerr << "remote-mem-latency must be finite and non-negative" << endl;
      exit(1);
    }
    if (latency_value > kMaxTimeNs) {
      cerr << "remote-mem-latency is outside the supported time range"
           << endl;
      exit(1);
    }
    remote_mem_latency = latency_value;
  }

  remote_mem_bw = 0;
  if (j.contains("remote-mem-bw")) {
    if (mem_type != NO_MEMORY_EXPANSION &&
        (!j["remote-mem-bw"].is_number() ||
         j["remote-mem-bw"].get<double>() <= 0)) {
      cerr << "remote-mem-bw must be positive for the configured memory type"
           << endl;
      exit(1);
    }
    if (j["remote-mem-bw"].is_number()) {
      const double bw_value = j["remote-mem-bw"].get<double>();
      if (!isfinite(bw_value)) {
        cerr << "remote-mem-bw must be finite" << endl;
        exit(1);
      }
      remote_mem_bw = bw_value;
    }
  }

  if (mem_type != NO_MEMORY_EXPANSION && remote_mem_bw <= 0) {
    cerr << "remote-mem-bw must be positive for the configured memory type"
         << endl;
    exit(1);
  }

  if (mem_type == PER_NODE_MEMORY_EXPANSION) {
    ports.resize(static_cast<size_t>(num_nodes));
  } else if (mem_type == MEMORY_POOL) {
    ports.emplace_back();
  }

  conf_file.close();
}

AnalyticalRemoteMemory::~AnalyticalRemoteMemory() {
  // Flush and close the sensing-gated detail stream first, so a completed
  // run's rows are on disk before any drain verdict is rendered.
  close_transaction_stream();
  // plan sec.3.4: normal-end teardown fails closed unless every
  // backend-owned port job, dual-zero timer, awaiting delivery and event
  // handle is gone (unconditional, also with sensing off). Early shutdowns
  // must go through shutdown() before destruction.
  verify_drained();
}

void AnalyticalRemoteMemory::set_sys(const int id, Sys* sys) {
  if (sys == nullptr) {
    Sys::sys_panic("AnalyticalRemoteMemory: set_sys received a null Sys");
  }
  if (host_sys == nullptr) {
    // plan sec.3.3: the single global transition event binds to the FIRST
    // set_sys Sys; construction precedes set_sys, so nothing was cached
    // earlier.
    host_sys = sys;
  }
  if (mem_type == PER_NPU_MEMORY_EXPANSION &&
      !per_npu_ids_configured &&
      per_npu_port_indices.find(id) == per_npu_port_indices.end()) {
    per_npu_port_indices[id] = ports.size();
    ports.emplace_back();
  }
}

void AnalyticalRemoteMemory::issue(
    const uint64_t tensor_size,
    WorkloadLayerHandlerData* wlhd) {
  if (wlhd == nullptr || wlhd->workload == nullptr) {
    Sys::sys_panic(
        "AnalyticalRemoteMemory: issue without a Workload callback target");
  }
  const int sys_id = wlhd->sys_id;
  size_t port_index;

  if (mem_type == NO_MEMORY_EXPANSION) {
    cerr << "Remote memory access is not supported in NO_MEMORY_EXPANSION"
         << endl;
    exit(1);
  } else if (mem_type == PER_NODE_MEMORY_EXPANSION) {
    port_index = static_cast<size_t>(sys_id / num_npus_per_node);
    if (port_index >= ports.size()) {
      cerr << "NPU rank " << sys_id
           << " is outside the configured PER_NODE_MEMORY_EXPANSION range"
           << endl;
      exit(1);
    }
  } else if (mem_type == PER_NPU_MEMORY_EXPANSION) {
    auto port_it = per_npu_port_indices.find(sys_id);
    if (port_it == per_npu_port_indices.end()) {
      cerr << "NPU rank " << sys_id
           << " does not have a configured remote-memory port" << endl;
      exit(1);
    }

    port_index = port_it->second;
  } else if (mem_type == MEMORY_POOL) {
    port_index = 0;
  } else {
    return;
  }

  if (host_sys == nullptr) {
    Sys::sys_panic(
        "AnalyticalRemoteMemory: issue before set_sys bound a host Sys");
  }
  if (tensor_size > kMaxExactBytes) {
    Sys::sys_panic(
        "AnalyticalRemoteMemory: tensor_size exceeds the exactly "
        "representable byte range");
  }

  // plan sec.3.3: every external issue first advances every port to the
  // current observable Tick, so streams that exhausted earlier become
  // awaiting-delivery here and the preserved same-Tick event harvests
  // them.
  const Tick now = Sys::boostedTick();
  advance_all_ports(static_cast<double>(now));

  PortState& port = ports[port_index];
  // Observation layer (plan sec.5.1): an arrival into a port that is
  // already streaming re-splits the survivors' shares; that share change
  // is listed separately from the completion-driven re-splits.
  const bool port_was_streaming = active_stream_count(port) > 0;
  auto* job = new PortJob();
  job->tensor_size = tensor_size;
  job->node_id = wlhd->node_id;
  job->issue_sequence = port.next_issue_sequence++;
  job->port_index = port_index;
  job->sys_id = sys_id;
  job->wlhd = wlhd;
  job->state = PortJobState::LatencyWaiting;
  job->issue_ns = clock_ns;
  job->issue_tick = now;
  job->stream_start_ns = -1.0;
  job->fluid_finish_ns = -1.0;
  job->callback_tick = 0;
  job->remaining_bytes = 0.0;
  job->ready_ns = clock_ns + remote_mem_latency;
  if (!isfinite(job->ready_ns) || job->ready_ns > kMaxTimeNs) {
    Sys::sys_panic(
        "AnalyticalRemoteMemory: latency expiry is outside the "
        "representable time range");
  }

  if (tensor_size == 0 && remote_mem_latency == 0.0) {
    // plan sec.3.2: independent one-shot timer completing exactly 1ns
    // after issue; never a synchronous completion on the issue stack and
    // never a member of the bandwidth set.
    job->state = PortJobState::DualZeroTimer;
    job->ready_ns = clock_ns + 1.0;
    if (job->ready_ns > kMaxTimeNs) {
      Sys::sys_panic(
          "AnalyticalRemoteMemory: dual-zero fire time is outside the "
          "representable time range");
    }
  } else if (tensor_size > 0 && remote_mem_latency == 0.0) {
    // Zero-latency positive-byte transfer: joins the active stream set at
    // the issue instant. Even a sub-ns service keeps its callback on the
    // next observable Tick (ceil of a finish strictly after clock).
    job->state = PortJobState::ActiveStream;
    job->stream_start_ns = clock_ns;
    job->remaining_bytes = static_cast<double>(tensor_size);
  }

  port.jobs.push_back(job);
  port.issued_count += 1;
  add_bytes_fail_closed(port.issued_bytes, tensor_size);

  // Observation layer: in_flight spans issue -> callback (latency
  // included), so the undelivered population right after the push is its
  // high-water-mark candidate; deliveries only shrink it.
  if (port.jobs.size() > port.peak_in_flight) {
    port.peak_in_flight = port.jobs.size();
  }
  if (job->state == PortJobState::ActiveStream) {
    if (port_was_streaming) {
      port.arrival_redistribution_events += 1;
    }
    const std::size_t streaming = active_stream_count(port);
    if (streaming > port.peak_streaming) {
      port.peak_streaming = streaming;
    }
  }

  rearm_transition_event();
}

void AnalyticalRemoteMemory::call(
    const EventType type,
    CallData* data) {
  if (type != EventType::General || data == nullptr) {
    Sys::sys_panic(
        "AnalyticalRemoteMemory: unexpected event dispatch signature");
  }
  auto* payload = static_cast<TransitionEventPayload*>(data);
  const uint64_t generation = payload->event_generation;
  // Normal-dispatch payload release (plan sec.3.3). The stale-generation
  // path releases through this same delete before the guard below.
  delete payload;
  if (generation != event_generation) {
    return;
  }
  // Sys popped this event before invoking us, so it is no longer
  // cancellable. Forget the consumed handle before completion callbacks
  // can re-enter issue() and arm the next transition.
  transition_event.reset();
  transition_deadline_tick = 0;

  const Tick now = Sys::boostedTick();
  advance_all_ports(static_cast<double>(now));

  // Collect the full completion batch of this Tick across all ports and
  // remove it from the active containers as one whole (plan sec.3.3).
  vector<PortJob*> batch;
  for (PortState& port : ports) {
    for (auto job_it = port.jobs.begin(); job_it != port.jobs.end();) {
      PortJob* job = *job_it;
      if (job->state == PortJobState::FluidCompleteAwaitingCallback) {
        batch.push_back(job);
        job_it = port.jobs.erase(job_it);
      } else {
        ++job_it;
      }
    }
  }

  // Delivery order: (port_index ascending, issue_sequence ascending).
  sort(batch.begin(), batch.end(),
       [](const PortJob* lhs, const PortJob* rhs) {
         if (lhs->port_index != rhs->port_index) {
           return lhs->port_index < rhs->port_index;
         }
         return lhs->issue_sequence < rhs->issue_sequence;
       });

  // Backend state settles completely before the first Workload callback;
  // callbacks may synchronously re-enter issue(), but never re-enter this
  // dispatch (plan sec.3.3). The sensing-gated detail row streams out
  // inside this same settlement pass: the wlhd is destroyed after its
  // callback, so a row can never be completed after the fact (plan
  // sec.5.1).
  for (PortJob* job : batch) {
    PortState& port = ports[job->port_index];
    port.completed_count += 1;
    add_bytes_fail_closed(port.completed_bytes, job->tensor_size);
    settle_transaction_row(*job);
  }
  for (PortJob* job : batch) {
    WorkloadLayerHandlerData* wlhd = job->wlhd;
    // Ownership handover (plan sec.3.4): the backend releases the job
    // shell and hands the wlhd to the Workload before the callback.
    delete job;
    wlhd->workload->call(EventType::General, wlhd);
  }

  // Reconcile the next transition. A callback above may already have
  // issued new work and armed an event; rearm keeps that handle unless an
  // earlier deadline exists now.
  rearm_transition_event();
}

uint64_t AnalyticalRemoteMemory::get_remote_mem_runtime(
    const uint64_t tensor_size) {
  // Uncontended service estimate under the fluid model: fixed latency plus
  // full-bandwidth streaming time. The observable completion of a
  // concurrent issue depends on the live stream set, so this is a lower
  // bound only. This is NOT part of the AstraRemoteMemoryAPI interface
  // (which carries just set_sys/issue) and production code never calls it:
  // the sole caller is the remote_port_nway_test fail-closed probe, which
  // invokes it directly on the concrete class to exercise the
  // reject-on-invalid-config exits below.
  if (mem_type == NO_MEMORY_EXPANSION || remote_mem_bw <= 0) {
    cerr << "get_remote_mem_runtime requires a positive remote-mem-bw "
         << "(NO_MEMORY_EXPANSION has no remote port)" << endl;
    exit(1);
  }
  if (tensor_size > kMaxExactBytes) {
    Sys::sys_panic(
        "AnalyticalRemoteMemory: tensor_size exceeds the exactly "
        "representable byte range");
  }
  const double runtime =
      remote_mem_latency + static_cast<double>(tensor_size) / remote_mem_bw;
  if (!isfinite(runtime) || runtime < 0 || runtime > kMaxTimeNs) {
    Sys::sys_panic(
        "AnalyticalRemoteMemory: estimated remote memory runtime is "
        "outside the representable range");
  }
  return static_cast<uint64_t>(runtime);
}

std::size_t AnalyticalRemoteMemory::get_port_count() const {
  return ports.size();
}

std::vector<AnalyticalRemoteMemory::PortJobSnapshot>
AnalyticalRemoteMemory::get_port_jobs(const std::size_t port_index) const {
  if (port_index >= ports.size()) {
    Sys::sys_panic("AnalyticalRemoteMemory: port index out of range");
  }
  vector<PortJobSnapshot> snapshots;
  snapshots.reserve(ports[port_index].jobs.size());
  for (const PortJob* job : ports[port_index].jobs) {
    PortJobSnapshot snapshot;
    snapshot.tensor_size = job->tensor_size;
    snapshot.node_id = job->node_id;
    snapshot.issue_sequence = job->issue_sequence;
    snapshot.port_index = job->port_index;
    snapshot.state = job->state;
    snapshot.issue_ns = job->issue_ns;
    snapshot.ready_ns = job->ready_ns;
    snapshot.stream_start_ns = job->stream_start_ns;
    snapshot.fluid_finish_ns = job->fluid_finish_ns;
    snapshot.callback_tick = job->callback_tick;
    snapshot.remaining_bytes = job->remaining_bytes;
    if (job->state == PortJobState::ActiveStream) {
      // The continuous clock advances lazily (only at issues and event
      // dispatches), and between transitions the active-stream count is
      // constant, so this linear projection to the query Tick is exact.
      // It keeps the read-only surface truthful for queries landing
      // between two transitions.
      const double now_ns = static_cast<double>(Sys::boostedTick());
      if (now_ns > clock_ns) {
        const size_t stream_count = active_stream_count(ports[port_index]);
        const double rate =
            remote_mem_bw / static_cast<double>(stream_count);
        const double projected =
            job->remaining_bytes - rate * (now_ns - clock_ns);
        snapshot.remaining_bytes = max(0.0, projected);
      }
    }
    snapshots.push_back(snapshot);
  }
  return snapshots;
}

AnalyticalRemoteMemory::PortStats
AnalyticalRemoteMemory::get_port_stats(
    const std::size_t port_index) const {
  if (port_index >= ports.size()) {
    Sys::sys_panic("AnalyticalRemoteMemory: port index out of range");
  }
  const PortState& port = ports[port_index];
  PortStats stats = {};  // zero-init: every field is assigned below, this
                         // only guards future field additions
  stats.issued_count = port.issued_count;
  stats.completed_count = port.completed_count;
  stats.issued_bytes = port.issued_bytes;
  stats.completed_bytes = port.completed_bytes;
  stats.latency_waiting_count = 0;
  stats.streaming_count = 0;
  stats.completion_waiting_count = 0;
  stats.dual_zero_timer_count = 0;
  for (const PortJob* job : port.jobs) {
    switch (job->state) {
      case PortJobState::LatencyWaiting:
        stats.latency_waiting_count += 1;
        break;
      case PortJobState::ActiveStream:
        stats.streaming_count += 1;
        break;
      case PortJobState::FluidCompleteAwaitingCallback:
        stats.completion_waiting_count += 1;
        break;
      case PortJobState::DualZeroTimer:
        stats.dual_zero_timer_count += 1;
        break;
    }
  }
  stats.in_flight_count =
      stats.issued_count - stats.completed_count;
  stats.peak_in_flight = port.peak_in_flight;
  stats.peak_streaming = port.peak_streaming;
  stats.redistribution_events = port.redistribution_events;
  stats.arrival_redistribution_events =
      port.arrival_redistribution_events;
  stats.port_busy_ns = port.port_busy_ns;
  stats.shared_busy_ns = port.shared_busy_ns;
  stats.bytes_served = port.bytes_served;
  return stats;
}

bool AnalyticalRemoteMemory::is_drained() const {
  if (transition_event.valid() || transition_deadline_tick != 0) {
    return false;
  }
  for (const PortState& port : ports) {
    if (!port.jobs.empty()) {
      return false;
    }
    if (port.issued_count != port.completed_count ||
        port.issued_bytes != port.completed_bytes) {
      return false;
    }
  }
  return true;
}

void AnalyticalRemoteMemory::verify_drained() const {
  if (is_drained()) {
    return;
  }
  string report =
      "AnalyticalRemoteMemory normal-end drain check failed:";
  if (transition_event.valid() || transition_deadline_tick != 0) {
    report += " armed-transition-event";
  }
  for (size_t index = 0; index < ports.size(); ++index) {
    const PortState& port = ports[index];
    if (!port.jobs.empty()) {
      report += " port[" + to_string(index) + "] undelivered_jobs=" +
                to_string(port.jobs.size());
    }
    if (port.issued_count != port.completed_count) {
      report += " port[" + to_string(index) + "] issued_count=" +
                to_string(port.issued_count) + " completed_count=" +
                to_string(port.completed_count);
    }
    if (port.issued_bytes != port.completed_bytes) {
      report += " port[" + to_string(index) + "] issued_bytes=" +
                to_string(port.issued_bytes) + " completed_bytes=" +
                to_string(port.completed_bytes);
    }
    // Observation-layer conservation (plan sec.5.1): the served-bytes
    // integral must account for every completed byte except the
    // per-transaction completion residue. A completion fires either on
    // the bytes path (at most kByteEps left) or on the time path (at most
    // rate * kTimeEpsNs <= remote_mem_bw * kTimeEpsNs bytes left), so the
    // aggregate gap is bounded by completed_count * (kByteEps + bw *
    // kTimeEpsNs).
    const double residue_bound_per_job =
        kByteEps + remote_mem_bw * kTimeEpsNs;
    const double served_gap =
        static_cast<double>(port.completed_bytes) - port.bytes_served;
    if (std::abs(served_gap) >
        residue_bound_per_job *
            static_cast<double>(port.completed_count)) {
      report += " port[" + to_string(index) + "] bytes_served=" +
                to_string(port.bytes_served) + " completed_bytes=" +
                to_string(port.completed_bytes);
    }
  }
  Sys::sys_panic(report);
}

void AnalyticalRemoteMemory::shutdown() {
  // Early-shutdown teardown (plan sec.3.4): cancel the global event (its
  // deleter releases the queued payload), delete every undelivered wlhd
  // and job, reset the live counters. Workload-owned HBM join cookies are
  // not backend property and are not touched.
  cancel_transition_event();
  // The detail stream closes with whatever rows already settled (an early
  // shutdown keeps the partial file as debugging evidence next to the
  // retained bridge dir; it never fabricates the missing rows).
  close_transaction_stream();
  for (PortState& port : ports) {
    for (PortJob* job : port.jobs) {
      delete job->wlhd;
      delete job;
    }
    port.jobs.clear();
    port.next_issue_sequence = 0;
    port.issued_count = 0;
    port.completed_count = 0;
    port.issued_bytes = 0;
    port.completed_bytes = 0;
    port.peak_in_flight = 0;
    port.peak_streaming = 0;
    port.redistribution_events = 0;
    port.arrival_redistribution_events = 0;
    port.port_busy_ns = 0.0;
    port.shared_busy_ns = 0.0;
    port.bytes_served = 0.0;
  }
  clock_ns = 0.0;
}

void AnalyticalRemoteMemory::configure_transaction_detail(
    const bool enabled,
    std::string jsonl_path,
    std::string run_id) {
  transaction_detail_enabled_ = enabled;
  transaction_detail_path_ = std::move(jsonl_path);
  transaction_run_id_ = std::move(run_id);
  // Fail closed BEFORE the first row: a sensing run whose rows carry no
  // association key could never be joined with its metrics manifest
  // (plan sec.5.1 run_id discipline).
  if (enabled &&
      (transaction_detail_path_.empty() || transaction_run_id_.empty())) {
    Sys::sys_panic(
        "AnalyticalRemoteMemory: transaction detail enabled without a "
        "jsonl path or run_id (manifest run_id, else normalized RUN_DIR)");
  }
}

void AnalyticalRemoteMemory::ensure_transaction_stream_open() {
  if (transaction_stream_.is_open()) {
    return;
  }
  // Lazy creation: the truncating open keeps a re-run against a stale
  // bridge dir from mixing runs; the switch-off case never reaches here.
  transaction_stream_.open(transaction_detail_path_,
                           std::ios::out | std::ios::trunc);
  if (!transaction_stream_.is_open()) {
    Sys::sys_panic(
        "AnalyticalRemoteMemory: cannot open the transaction detail "
        "file: " + transaction_detail_path_);
  }
}

void AnalyticalRemoteMemory::settle_transaction_row(const PortJob& job) {
  if (!transaction_detail_enabled_) {
    return;  // sensing off: no per-transaction record is kept anywhere
  }
  if (transaction_detail_finalized_) {
    // A row settling after the terminal summary means completions were
    // delivered past the drained audit -- a state inconsistency must
    // terminate, not silently extend a closed artifact (plan sec.3.3).
    Sys::sys_panic(
        "AnalyticalRemoteMemory: transaction row settled after the "
        "detail summary row");
  }
  ensure_transaction_stream_open();
  // Row key (run_id, rank, node_id, issue_sequence) plus the issue-time
  // copies (bytes, port, issue Tick) and the fluid timing fields. Rows are
  // written at completion settlement, before the wlhd callback destroys
  // the handler: no observation key can ever be attached after the fact.
  json row;
  row["schema"] = 1;
  row["type"] = "remote_memory_transaction";
  row["run_id"] = transaction_run_id_;
  row["rank"] = job.sys_id;
  row["node_id"] = job.node_id;
  row["issue_sequence"] = job.issue_sequence;
  row["port_index"] = job.port_index;
  row["bytes"] = job.tensor_size;
  row["issue_tick"] = job.issue_tick;
  row["latency_ready_ns"] = job.ready_ns;
  row["stream_start_ns"] = job.stream_start_ns;  // -1.0: never streamed
  row["fluid_finish_ns"] = job.fluid_finish_ns;
  row["callback_tick"] = job.callback_tick;
  transaction_stream_ << row.dump() << '\n';
  transaction_rows_written_ += 1;
  if (!transaction_stream_.good()) {
    Sys::sys_panic(
        "AnalyticalRemoteMemory: transaction detail write failed: " +
        transaction_detail_path_);
  }
}

void AnalyticalRemoteMemory::close_transaction_stream() {
  if (!transaction_stream_.is_open()) {
    return;
  }
  transaction_stream_.flush();
  const bool healthy = transaction_stream_.good();
  transaction_stream_.close();
  if (!healthy) {
    Sys::sys_panic(
        "AnalyticalRemoteMemory: transaction detail stream failed: " +
        transaction_detail_path_);
  }
}

void AnalyticalRemoteMemory::finalize_transaction_detail() {
  if (!transaction_detail_enabled_ || transaction_detail_finalized_) {
    return;
  }
  transaction_detail_finalized_ = true;
  // Terminal summary row (amended 2026-09-24): every normal end of a
  // sensing run leaves a parseable artifact. With zero settled rows this
  // is the file's single line, so the sensing-alive criterion (non-empty,
  // recomputable) holds on every input -- and the per-port PortStats
  // aggregates in the row give the recomputation its backend-side
  // reconciliation target (plan sec.5.1).
  ensure_transaction_stream_open();
  json summary;
  summary["schema"] = 1;
  summary["type"] = "remote_memory_transactions_summary";
  summary["run_id"] = transaction_run_id_;
  summary["rows"] = transaction_rows_written_;
  json port_rows = json::array();
  for (std::size_t index = 0; index < ports.size(); ++index) {
    const PortStats stats = get_port_stats(index);
    json row;
    row["port_index"] = index;
    row["issued_count"] = stats.issued_count;
    row["completed_count"] = stats.completed_count;
    row["issued_bytes"] = stats.issued_bytes;
    row["completed_bytes"] = stats.completed_bytes;
    row["in_flight_count"] = stats.in_flight_count;
    row["streaming_count"] = stats.streaming_count;
    row["latency_waiting_count"] = stats.latency_waiting_count;
    row["completion_waiting_count"] = stats.completion_waiting_count;
    row["dual_zero_timer_count"] = stats.dual_zero_timer_count;
    row["peak_in_flight"] = stats.peak_in_flight;
    row["peak_streaming"] = stats.peak_streaming;
    row["redistribution_events"] = stats.redistribution_events;
    row["arrival_redistribution_events"] =
        stats.arrival_redistribution_events;
    row["port_busy_ns"] = stats.port_busy_ns;
    row["shared_busy_ns"] = stats.shared_busy_ns;
    row["bytes_served"] = stats.bytes_served;
    port_rows.push_back(row);
  }
  summary["ports"] = port_rows;
  transaction_stream_ << summary.dump() << '\n';
  if (!transaction_stream_.good()) {
    Sys::sys_panic(
        "AnalyticalRemoteMemory: transaction detail summary write "
        "failed: " + transaction_detail_path_);
  }
  close_transaction_stream();
}

std::size_t AnalyticalRemoteMemory::active_stream_count(
    const PortState& port) const {
  std::size_t count = 0;
  for (const PortJob* job : port.jobs) {
    if (job->state == PortJobState::ActiveStream) {
      count += 1;
    }
  }
  return count;
}

std::size_t AnalyticalRemoteMemory::total_job_count() const {
  std::size_t count = 0;
  for (const PortState& port : ports) {
    count += port.jobs.size();
  }
  return count;
}

AstraSim::Tick AnalyticalRemoteMemory::ceil_to_tick(const double time_ns) {
  if (!isfinite(time_ns) || time_ns < 0 || time_ns > kMaxTimeNs) {
    Sys::sys_panic(
        "AnalyticalRemoteMemory: fluid completion time is outside the "
        "representable Tick range");
  }
  // The callback Tick is pinned at the TRUE ceil of the completion instant.
  // A former branch snapped a finish lying within kTimeEpsNs below its ceil
  // Tick down to floor -- but at that floor Tick the job has not finished
  // serving, so the transition event re-armed the same Tick with zero
  // progress and re-fired inside the Sys bucket being dispatched forever
  // (simulation hang). The kByteEps / kTimeEpsNs completion tolerances in
  // process_transitions_at() already absorb sub-ns floating-point residue
  // once the true ceil Tick is reached, so no one-tick-early arming is
  // needed; advance_all_ports()' up-snap of t_star lands the continuous
  // clock on that same integer boundary, keeping callback ==
  // ceil(fluid_finish).
  return static_cast<Tick>(ceil(time_ns));
}

double AnalyticalRemoteMemory::earliest_transition_ns() const {
  double earliest = numeric_limits<double>::infinity();
  for (const PortState& port : ports) {
    const size_t stream_count = active_stream_count(port);
    for (const PortJob* job : port.jobs) {
      switch (job->state) {
        case PortJobState::LatencyWaiting:
        case PortJobState::DualZeroTimer:
          earliest = min(earliest, job->ready_ns);
          break;
        case PortJobState::ActiveStream: {
          const double rate =
              remote_mem_bw / static_cast<double>(stream_count);
          earliest =
              min(earliest, clock_ns + job->remaining_bytes / rate);
          break;
        }
        case PortJobState::FluidCompleteAwaitingCallback:
          // Its delivery deadline is covered by earliest_deadline_tick().
          break;
      }
    }
  }
  return earliest;
}

AstraSim::Tick AnalyticalRemoteMemory::earliest_deadline_tick() const {
  Tick earliest = 0;
  for (const PortState& port : ports) {
    if (port.jobs.empty()) {
      continue;
    }
    const size_t stream_count = active_stream_count(port);
    for (const PortJob* job : port.jobs) {
      double deadline_ns;
      switch (job->state) {
        case PortJobState::LatencyWaiting:
        case PortJobState::DualZeroTimer:
          deadline_ns = job->ready_ns;
          break;
        case PortJobState::ActiveStream: {
          const double rate =
              remote_mem_bw / static_cast<double>(stream_count);
          deadline_ns = clock_ns + job->remaining_bytes / rate;
          break;
        }
        case PortJobState::FluidCompleteAwaitingCallback:
          deadline_ns = static_cast<double>(job->callback_tick);
          break;
        default:
          Sys::sys_panic("AnalyticalRemoteMemory: unknown port job state");
          return 0;
      }
      // Clamp at 1 instead of skipping 0: ceil_to_tick yields 0 only for a
      // deadline at instant 0, and treating that Tick as "no deadline"
      // left such a job permanently unarmed (no event, no delivery,
      // is_drained() false at teardown). Tick 1 delivers it at the first
      // observable Tick after the issue Tick.
      const Tick job_deadline =
          max(static_cast<Tick>(1), ceil_to_tick(deadline_ns));
      if (earliest == 0 || job_deadline < earliest) {
        earliest = job_deadline;
      }
    }
  }
  return earliest;
}

void AnalyticalRemoteMemory::serve_interval(const double dt_ns) {
  if (dt_ns <= 0) {
    return;
  }
  if (!isfinite(dt_ns)) {
    Sys::sys_panic(
        "AnalyticalRemoteMemory: non-finite fluid service interval");
  }
  for (PortState& port : ports) {
    const size_t stream_count = active_stream_count(port);
    if (stream_count == 0) {
      continue;
    }
    const double rate = remote_mem_bw / static_cast<double>(stream_count);
    // Observation layer (plan sec.5.1): every field below is EVENT-INTERVAL
    // integration -- one exact contribution per continuous service
    // interval between transition instants -- never fixed-interval
    // sampling (which would miss short streams).
    port.port_busy_ns += dt_ns;
    if (stream_count >= 2) {
      port.shared_busy_ns += dt_ns;
    }
    if (static_cast<uint64_t>(stream_count) > port.peak_streaming) {
      port.peak_streaming = static_cast<uint64_t>(stream_count);
    }
    // Explicit clamp boundary (plan sec.3.2): a surviving stream may be
    // overserved by at most rate * kTimeEpsNs of time, plus the byte
    // tolerance; anything beyond that is an invariant failure, not a
    // silent absorption.
    const double overservice_tolerance = rate * kTimeEpsNs + kByteEps;
    for (PortJob* job : port.jobs) {
      if (job->state != PortJobState::ActiveStream) {
        continue;
      }
      // Per-job exact served-bytes attribution: clamping the credit at the
      // remaining bytes leaves the completion state identical to the
      // former subtract-then-clamp arithmetic (both end at exactly 0.0
      // within the tolerance) while keeping bytes_served conserved. The
      // overservice guard keeps the former fail-closed reach exactly.
      const double raw_served = rate * dt_ns;
      const double served = min(job->remaining_bytes, raw_served);
      job->remaining_bytes -= served;
      if (raw_served - served > overservice_tolerance) {
        Sys::sys_panic(
            "AnalyticalRemoteMemory: bandwidth overservice beyond the "
            "explicit byte tolerance");
      }
      port.bytes_served += served;
    }
  }
}

void AnalyticalRemoteMemory::process_transitions_at(const double t_ns) {
  for (PortState& port : ports) {
    // Order at one continuous instant (plan sec.3.1): exhausted streams
    // leave the bandwidth denominator as one batch before ready jobs are
    // admitted, so an arrival never shares bandwidth with a dying stream.
    uint64_t streams_completed_here = 0;
    for (PortJob* job : port.jobs) {
      if (job->state != PortJobState::ActiveStream) {
        continue;
      }
      const size_t stream_count = active_stream_count(port);
      const double rate =
          remote_mem_bw / static_cast<double>(stream_count);
      const bool bytes_done = job->remaining_bytes <= kByteEps;
      // The time-bounded clamp may absorb at most rate * kTimeEpsNs
      // bytes of residual service (explicit boundary, plan sec.3.2).
      const bool time_done = job->remaining_bytes / rate <= kTimeEpsNs;
      if (!bytes_done && !time_done) {
        continue;
      }
      job->remaining_bytes = 0;
      job->state = PortJobState::FluidCompleteAwaitingCallback;
      job->fluid_finish_ns = t_ns;
      job->callback_tick = ceil_to_tick(t_ns);
      streams_completed_here += 1;
    }
    // Observation layer (plan sec.5.1): the redistribution counters key on
    // the CONTINUOUS completion instants. Simultaneous finishers count
    // once per instant; distinct instants inside one integer callback Tick
    // each count (this function runs once per distinct instant). No
    // survivors -> no share to re-split -> no event.
    const std::size_t survivors = active_stream_count(port);
    if (streams_completed_here > 0 && survivors > 0) {
      port.redistribution_events += 1;
    }
    uint64_t streams_arrived_here = 0;
    for (PortJob* job : port.jobs) {
      if (job->state == PortJobState::LatencyWaiting &&
          job->ready_ns <= t_ns + kTimeEpsNs) {
        if (job->tensor_size > 0) {
          job->state = PortJobState::ActiveStream;
          job->stream_start_ns = t_ns;
          job->remaining_bytes = static_cast<double>(job->tensor_size);
          streams_arrived_here += 1;
        } else {
          // bytes == 0 && latency > 0: never joins the bandwidth
          // denominator; delivers asynchronously at the ceil of its
          // ready instant (plan sec.3.2).
          job->state = PortJobState::FluidCompleteAwaitingCallback;
          job->fluid_finish_ns = t_ns;
          job->callback_tick = ceil_to_tick(t_ns);
        }
      } else if (job->state == PortJobState::DualZeroTimer &&
                 job->ready_ns <= t_ns + kTimeEpsNs) {
        // Dual-zero one-shot fired at its exact issue + 1ns instant.
        job->state = PortJobState::FluidCompleteAwaitingCallback;
        job->fluid_finish_ns = t_ns;
        job->callback_tick = ceil_to_tick(t_ns);
      }
    }
    // Arrival-driven share changes are listed separately (plan sec.5.1):
    // new streams joining while survivors were already streaming re-split
    // their shares again. Arrivals into an idle port get the full
    // bandwidth and are not a redistribution.
    if (streams_arrived_here > 0 && survivors > 0) {
      port.arrival_redistribution_events += 1;
    }
  }
}

void AnalyticalRemoteMemory::advance_all_ports(const double target_ns) {
  if (!isfinite(target_ns) || target_ns < 0 || target_ns > kMaxTimeNs) {
    Sys::sys_panic(
        "AnalyticalRemoteMemory: advance target is outside the "
        "representable time range");
  }
  if (target_ns < clock_ns - kTimeEpsNs) {
    Sys::sys_panic(
        "AnalyticalRemoteMemory: continuous clock cannot move backwards");
  }
  if (target_ns <= clock_ns) {
    process_transitions_at(clock_ns);
    return;
  }

  // Every loop iteration either consumes measurable time up to a
  // transition instant or flips at least one job state at the current
  // instant; both are bounded by the live job population, so this guard
  // turns any zero-step pathology into a fatal instead of a hang (plan
  // sec.3.2).
  const uint64_t step_guard =
      4 * static_cast<uint64_t>(total_job_count()) + 64;
  uint64_t steps = 0;
  while (clock_ns < target_ns - kTimeEpsNs) {
    if (++steps > step_guard) {
      Sys::sys_panic(
          "AnalyticalRemoteMemory: fluid advance exceeded its transition "
          "guard (zero-step loop suspected)");
    }
    double t_star = earliest_transition_ns();
    if (!isfinite(t_star)) {
      break;  // no pending transition anywhere
    }
    if (t_star <= clock_ns + kTimeEpsNs) {
      // Transition numerically at the current instant (a residue whose
      // service time is below the time tolerance): flip states without
      // consuming measurable time.
      process_transitions_at(clock_ns);
      continue;
    }
    if (t_star >= target_ns - kTimeEpsNs) {
      break;  // next transition lies at (or beyond) the target boundary
    }
    serve_interval(t_star - clock_ns);
    // Snap the transition instant into the explicit time tolerance so the
    // stored fluid_finish_ns and the callback Tick stay mutually consistent
    // (a finish landing within kTimeEpsNs of an integer ns is FP residue,
    // not measurable service).
    if (t_star > clock_ns && ceil(t_star) - t_star <= kTimeEpsNs &&
        t_star > 0) {
      t_star = ceil(t_star);
    }
    clock_ns = t_star;
    if (clock_ns > kMaxTimeNs) {
      Sys::sys_panic(
          "AnalyticalRemoteMemory: continuous clock left the "
          "representable time range");
    }
    process_transitions_at(clock_ns);
  }
  if (target_ns > clock_ns) {
    serve_interval(target_ns - clock_ns);
    clock_ns = target_ns;
  }
  process_transitions_at(clock_ns);
}

void AnalyticalRemoteMemory::arm_transition_event(
    const Tick deadline_tick) {
  if (host_sys == nullptr) {
    Sys::sys_panic(
        "AnalyticalRemoteMemory: transition event armed before set_sys");
  }
  const Tick now = Sys::boostedTick();
  if (deadline_tick < now) {
    Sys::sys_panic(
        "AnalyticalRemoteMemory: refusing to arm a transition event in "
        "the past (deadline=" + to_string(deadline_tick) + ", now=" +
        to_string(now) + ")");
  }
  Tick effective_deadline = deadline_tick;
  Tick delay = effective_deadline - now;
  if (delay == 0) {
    // Same-Tick arming is legal only while Sys is draining the bucket of
    // this very Tick: register_event_cancellable appends to that bucket
    // and call_events() pops it later in the same pass, so a survivor
    // finishing at, e.g., ns 300.9 is still delivered at Tick 301 (plan
    // sec.3.1/sec.3.3: no callback later than its completion Tick).
    // Outside that exact context a zero-delay event would outlive the
    // already-erased bucket (stranded handle, drain-audit failure) or
    // re-fire a deadline with zero progress -- e.g. an issue bound to a
    // non-host Sys whose own dispatch does not touch host_sys. Fail
    // closed by deferring to the next Tick instead of panicking. With
    // ceil_to_tick pinned at the true ceil, the Tick a deadline names
    // always harvests that deadline's job, so a same-Tick re-arm with
    // zero progress is impossible and the deferral cannot loop.
    if (!host_sys->dispatching_events ||
        host_sys->dispatching_event_time != now) {
      effective_deadline += 1;
      delay = 1;
    }
  }
  ++event_generation;
  transition_event = host_sys->register_event_cancellable(
      this,
      EventType::General,
      new TransitionEventPayload(event_generation),
      delay,
      &AnalyticalRemoteMemory::destroy_event_payload);
  transition_deadline_tick = effective_deadline;
}

void AnalyticalRemoteMemory::cancel_transition_event() {
  if (transition_event.valid()) {
    // The registered deleter releases the queued payload; a cancelled
    // event can never fire.
    static_cast<void>(host_sys->cancel_event(transition_event));
  }
  transition_event.reset();
  transition_deadline_tick = 0;
}

void AnalyticalRemoteMemory::destroy_event_payload(CallData* data) {
  delete static_cast<TransitionEventPayload*>(data);
}

void AnalyticalRemoteMemory::rearm_transition_event() {
  const Tick earliest = earliest_deadline_tick();
  if (!transition_event.valid()) {
    if (earliest != 0) {
      arm_transition_event(earliest);
    }
    return;
  }
  if (earliest == 0) {
    cancel_transition_event();
    return;
  }
  if (earliest < transition_deadline_tick) {
    cancel_transition_event();
    arm_transition_event(earliest);
  }
  // earliest >= armed deadline: keep the armed event. The equal case is
  // the plan sec.3.3 same-Tick preservation rule: an issue landing on the
  // armed Tick joins its state first and lets the preserved event harvest
  // that Tick's finished streams; cancelling and deferring to the next
  // Tick is explicitly forbidden.
}
