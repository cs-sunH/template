/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#ifndef __ANALYTICAL_MEMORY_HH__
#define __ANALYTICAL_MEMORY_HH__

#include <cstddef>
#include <cstdint>
#include <deque>
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

class PendingMemoryRequest {
 public:
  PendingMemoryRequest(
      uint64_t tensor_size,
      AstraSim::WorkloadLayerHandlerData* wlhd)
    : tensor_size(tensor_size), wlhd(wlhd) {
  }

  uint64_t tensor_size;
  AstraSim::WorkloadLayerHandlerData* wlhd;
};

class AnalyticalRemoteMemory : public AstraSim::AstraRemoteMemoryAPI, public AstraSim::Callable{
 public:
  AnalyticalRemoteMemory(std::string memory_configuration) noexcept;
  void set_sys(int id, AstraSim::Sys* sys);
  void issue(
      uint64_t tensor_size,
      AstraSim::WorkloadLayerHandlerData* wlhd);
  void call(AstraSim::EventType type, AstraSim::CallData* data);
  uint64_t get_remote_mem_runtime(uint64_t tensor_size);

  // R3 (方案 §3.6 / 阶段 E): read-only export helpers for the sensing
  // ledger output -- single source of truth next to the port_index
  // resolution itself (main_online must not re-derive either string).
  const char* architecture_name() const;
  std::string port_mapping_rule() const;

 private:
  class RemoteMemoryCompletionData : public AstraSim::CallData {
   public:
    RemoteMemoryCompletionData(std::size_t port_index, uint64_t tensor_size)
      : port_index(port_index), tensor_size(tensor_size) {
    }

    std::size_t port_index;
    // R3 (方案 §3.6): the completed transaction's byte count, so a shared
    // port's completion accounting can never credit the NEXT request's
    // size (the dequeued pmr.tensor_size in call() belongs to the next
    // request and must never be used for the completion record).
    uint64_t tensor_size;
  };

  void start_request(
      std::size_t port_index,
      uint64_t tensor_size,
      AstraSim::WorkloadLayerHandlerData* wlhd);

  MemoryArchitectureType mem_type = NO_MEMORY_EXPANSION;
  uint64_t remote_mem_latency; // remote memory access latency in nanosec
  uint64_t remote_mem_bw; // remote memory bandwidth in GB/sec
  std::vector<bool> ongoing_transaction;

  // per-node memory expansion
  int num_nodes;
  int num_npus_per_node;

  std::unordered_map<int, AstraSim::Sys*> sys_map;
  bool per_npu_ids_configured = false;
  std::unordered_map<int, std::size_t> per_npu_port_indices;
  std::vector<std::deque<PendingMemoryRequest>> pending_requests;
};
} // namespace Analytical

#endif /* __ANALYTICAL_MEMORY_HH__ */
