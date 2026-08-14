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

  if (mem_type == NO_MEMORY_EXPANSION) {
    cerr << "Remote memory access is not supported in NO_MEMORY_EXPANSION"
         << endl;
    exit(1);
  } else if (mem_type == PER_NODE_MEMORY_EXPANSION) {
    int nid = sys_id / num_npus_per_node;
    if (ongoing_transaction[nid]) {
      PendingMemoryRequest pmr(tensor_size, wlhd);
      pending_requests[nid].push_back(pmr);
    } else {
      uint64_t runtime = get_remote_mem_runtime(tensor_size);

      Sys* sys = sys_map[sys_id];
      sys->register_event(wlhd->workload, EventType::General, wlhd, runtime);

      sys->register_event(this, EventType::General, wlhd, runtime);

      ongoing_transaction[nid] = true;
    }
  } else if (mem_type == PER_NPU_MEMORY_EXPANSION) {
    auto port_it = per_npu_port_indices.find(sys_id);
    if (port_it == per_npu_port_indices.end()) {
      cerr << "NPU rank " << sys_id
           << " does not have a configured remote-memory port" << endl;
      exit(1);
    }

    size_t port_index = port_it->second;
    if (ongoing_transaction[port_index]) {
      PendingMemoryRequest pmr(tensor_size, wlhd);
      pending_requests[port_index].push_back(pmr);
    } else {
      uint64_t runtime = get_remote_mem_runtime(tensor_size);

      Sys* sys = sys_map[sys_id];
      sys->register_event(wlhd->workload, EventType::General, wlhd, runtime);

      sys->register_event(this, EventType::General, wlhd, runtime);

      ongoing_transaction[port_index] = true;
    }
  } else if (mem_type == MEMORY_POOL) {
    if (ongoing_transaction[0]) {
      PendingMemoryRequest pmr(tensor_size, wlhd);
      pending_requests[0].push_back(pmr);
    } else {
      uint64_t runtime = get_remote_mem_runtime(tensor_size);

      Sys* sys = sys_map[sys_id];
      sys->register_event(wlhd->workload, EventType::General, wlhd, runtime);

      sys->register_event(this, EventType::General, wlhd, runtime);

      ongoing_transaction[0] = true;
    }
  }
}

void AnalyticalRemoteMemory::call(EventType type, CallData* data) {
  if (mem_type == PER_NODE_MEMORY_EXPANSION) {
    WorkloadLayerHandlerData* wlhd = (WorkloadLayerHandlerData*)data;
    int nid = wlhd->sys_id / num_npus_per_node;
    if (!pending_requests[nid].empty()) {
      PendingMemoryRequest pmr = pending_requests[nid].front();
      pending_requests[nid].pop_front();

      uint64_t runtime = get_remote_mem_runtime(pmr.tensor_size);

      Sys* sys = sys_map[pmr.wlhd->sys_id];
      sys->register_event(
          pmr.wlhd->workload, EventType::General, pmr.wlhd, runtime);

      sys->register_event(this, EventType::General, pmr.wlhd, runtime);

      ongoing_transaction[nid] = true;
    } else {
      ongoing_transaction[nid] = false;
    }
  } else if (mem_type == PER_NPU_MEMORY_EXPANSION) {
    WorkloadLayerHandlerData* wlhd = (WorkloadLayerHandlerData*)data;
    auto port_it = per_npu_port_indices.find(wlhd->sys_id);
    if (port_it == per_npu_port_indices.end()) {
      cerr << "Completed remote-memory request for NPU rank " << wlhd->sys_id
           << " has no configured remote-memory port" << endl;
      exit(1);
    }

    size_t port_index = port_it->second;
    if (!pending_requests[port_index].empty()) {
      PendingMemoryRequest pmr = pending_requests[port_index].front();
      pending_requests[port_index].pop_front();

      uint64_t runtime = get_remote_mem_runtime(pmr.tensor_size);

      Sys* sys = sys_map[pmr.wlhd->sys_id];
      sys->register_event(
          pmr.wlhd->workload, EventType::General, pmr.wlhd, runtime);

      sys->register_event(this, EventType::General, pmr.wlhd, runtime);

      ongoing_transaction[port_index] = true;
    } else {
      ongoing_transaction[port_index] = false;
    }
  } else if (mem_type == MEMORY_POOL) {
    if (!pending_requests[0].empty()) {
      PendingMemoryRequest pmr = pending_requests[0].front();
      pending_requests[0].pop_front();

      uint64_t runtime = get_remote_mem_runtime(pmr.tensor_size);

      Sys* sys = sys_map[pmr.wlhd->sys_id];
      sys->register_event(
          pmr.wlhd->workload, EventType::General, pmr.wlhd, runtime);

      sys->register_event(this, EventType::General, pmr.wlhd, runtime);

      ongoing_transaction[0] = true;
    } else {
      ongoing_transaction[0] = false;
    }
  }
}

uint64_t AnalyticalRemoteMemory::get_remote_mem_runtime(uint64_t tensor_size) {
  uint64_t runtime = remote_mem_latency
      + static_cast<uint64_t>((static_cast<double>(tensor_size) / remote_mem_bw));
  return runtime;
}
