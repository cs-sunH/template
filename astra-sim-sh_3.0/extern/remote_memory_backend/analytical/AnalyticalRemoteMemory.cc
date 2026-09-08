/******************************************************************************
This source code is licensed under the MIT license found in the LICENSE file in
the root directory of this source tree.
 *******************************************************************************/

#include "extern/remote_memory_backend/analytical/AnalyticalRemoteMemory.hh"
#include <json/json.hpp>
#include <fstream>
#include <iostream>
#include <limits>
#include "astra-sim/system/Common.hh"
#include "astra-sim/system/WorkloadLayerHandlerData.hh"
// R3 (方案 §3.6): self-contained observation layer -- the ONLY astra-sim
// workload header this backend translation unit needs (kept minimal so the
// backend pair plus this ledger hh/cc stay trivially syncable across the
// five repositories; the blueprint repos never enable the ledger).
#include "astra-sim/workload/RemoteFifoLedger.hh"

using namespace std;
using namespace AstraSim;
using namespace Analytical;
using json = nlohmann::json;

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
    num_nodes = 0;
    if (j.contains("num-nodes")) {
      num_nodes = j["num-nodes"];
    }
    num_npus_per_node = 0;
    if (j.contains("num-npus-per-node")) {
      num_npus_per_node = j["num-npus-per-node"];
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

      per_npu_port_indices[npu_id] = ongoing_transaction.size();
      ongoing_transaction.push_back(false);
      pending_requests.emplace_back();
    }
  }

  remote_mem_latency = 0;
  if (j.contains("remote-mem-latency")) {
    remote_mem_latency = j["remote-mem-latency"];
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
    remote_mem_bw = j["remote-mem-bw"];
  }

  if (mem_type != NO_MEMORY_EXPANSION && remote_mem_bw == 0) {
    cerr << "remote-mem-bw must be positive for the configured memory type"
         << endl;
    exit(1);
  }

  if (mem_type == PER_NODE_MEMORY_EXPANSION) {
    for (int i = 0; i < num_nodes; i++) {
      ongoing_transaction.push_back(false);
      deque<PendingMemoryRequest> dpmr;
      pending_requests.push_back(dpmr);
    }
  } else if (mem_type == MEMORY_POOL) {
    ongoing_transaction.push_back(false);
    deque<PendingMemoryRequest> dpmr;
    pending_requests.push_back(dpmr);
  }

  conf_file.close();
}

void AnalyticalRemoteMemory::set_sys(int id, Sys* sys) {
  sys_map[id] = sys;
  if (mem_type == PER_NPU_MEMORY_EXPANSION &&
      !per_npu_ids_configured &&
      per_npu_port_indices.find(id) == per_npu_port_indices.end()) {
    per_npu_port_indices[id] = ongoing_transaction.size();
    ongoing_transaction.push_back(false);
    pending_requests.emplace_back();
  }
}

void AnalyticalRemoteMemory::issue(
    uint64_t tensor_size,
    WorkloadLayerHandlerData* wlhd) {
  int sys_id = wlhd->sys_id;
  size_t port_index;

  if (mem_type == NO_MEMORY_EXPANSION) {
    cerr << "Remote memory access is not supported in NO_MEMORY_EXPANSION"
         << endl;
    exit(1);
  } else if (mem_type == PER_NODE_MEMORY_EXPANSION) {
    port_index = static_cast<size_t>(sys_id / num_npus_per_node);
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

  // R3 (方案 §3.6 / 阶段 E): issue accounting at the point where the REAL
  // port_index has been resolved, before the busy/enqueue decision -- both
  // the immediate-start and the queued path count exactly once per request.
  // Pure observation (counters only); the ledger is sensing-gated and
  // fail-closed (default off), so ordinary runs and the blueprint repos
  // (which compile this but never enable it) see zero behavior change.
  AstraSim::ExecutionDriven::RemoteFifoLedger::instance().record_issue(
      port_index, sys_id, tensor_size);

  if (ongoing_transaction[port_index]) {
    pending_requests[port_index].emplace_back(tensor_size, wlhd);
  } else {
    start_request(port_index, tensor_size, wlhd);
  }
}

void AnalyticalRemoteMemory::start_request(
    size_t port_index,
    uint64_t tensor_size,
    WorkloadLayerHandlerData* wlhd) {
  uint64_t runtime = get_remote_mem_runtime(tensor_size);
  Sys* sys = sys_map[wlhd->sys_id];

  sys->register_event(wlhd->workload, EventType::General, wlhd, runtime);
  sys->register_event(
      this,
      EventType::General,
      new RemoteMemoryCompletionData(port_index, tensor_size),
      runtime);

  ongoing_transaction[port_index] = true;
}

void AnalyticalRemoteMemory::call(EventType type, CallData* data) {
  RemoteMemoryCompletionData* completion_data =
      static_cast<RemoteMemoryCompletionData*>(data);
  size_t port_index = completion_data->port_index;
  // R3 (方案 §3.6): the payload carries THIS transaction's bytes; the
  // dequeued pmr.tensor_size below belongs to the NEXT request.
  uint64_t completed_bytes = completion_data->tensor_size;
  delete completion_data;

  // R3 (方案 §3.6 / 阶段 E): completion accounting right after the payload
  // extraction and BEFORE the next pending request starts -- the exact
  // moment the port transaction really finished (no HBM-join deferral).
  // Sensing-gated pure observation; zero behavior change when disabled.
  AstraSim::ExecutionDriven::RemoteFifoLedger::instance().record_completion(
      port_index, completed_bytes);

  if (!pending_requests[port_index].empty()) {
    PendingMemoryRequest pmr = pending_requests[port_index].front();
    pending_requests[port_index].pop_front();
    start_request(port_index, pmr.tensor_size, pmr.wlhd);
  } else {
    ongoing_transaction[port_index] = false;
  }
}

const char* AnalyticalRemoteMemory::architecture_name() const {
  switch (mem_type) {
    case NO_MEMORY_EXPANSION:
      return "NO_MEMORY_EXPANSION";
    case PER_NODE_MEMORY_EXPANSION:
      return "PER_NODE_MEMORY_EXPANSION";
    case PER_NPU_MEMORY_EXPANSION:
      return "PER_NPU_MEMORY_EXPANSION";
    case MEMORY_POOL:
      return "MEMORY_POOL";
  }
  return "UNKNOWN";
}

std::string AnalyticalRemoteMemory::port_mapping_rule() const {
  switch (mem_type) {
    case NO_MEMORY_EXPANSION:
      return "none";
    case PER_NODE_MEMORY_EXPANSION:
      return "sys-id/num-npus-per-node";
    case PER_NPU_MEMORY_EXPANSION:
      return per_npu_ids_configured ? "npu-ids-array-index"
                                    : "set-sys-registration-order(=rank)";
    case MEMORY_POOL:
      return "single-shared-port-0";
  }
  return "unknown";
}

uint64_t AnalyticalRemoteMemory::get_remote_mem_runtime(uint64_t tensor_size) {
  uint64_t runtime = remote_mem_latency
      + static_cast<uint64_t>((static_cast<double>(tensor_size) / remote_mem_bw));
  return runtime;
}
