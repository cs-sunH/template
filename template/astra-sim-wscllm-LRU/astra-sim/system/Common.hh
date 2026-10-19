/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#ifndef __COMMON_HH__
#define __COMMON_HH__

#include <cstdint>
#include <string>

namespace AstraSim {

typedef unsigned long long Tick;

constexpr uint64_t CLOCK_PERIOD = 1;  // 1ns

// Only NS survives: every other upstream time unit (SE/MS/US/FS) had zero
// users in this repository.
enum time_type_e { NS = 0 };

// Every communication request in the simulator carries UINT8 payload
// granularity; the wider dtypes (BFLOAT16/FP32) had zero users after the
// type-typed removal in sh_test_mesh.
enum req_type_e { UINT8 = 0 };

struct timespec_t {
    time_type_e time_res;
    long double time_val;
};

// Every field carries an NSDMI: several producers fill only a subset of the
// fields before the request is copied or read (Workload p2p endpoints,
// CustomAlgorithm, the Sys rendezvous handshake copies), so a default would
// otherwise propagate indeterminate values.
struct sim_request {
    uint32_t srcRank = 0;
    uint32_t dstRank = 0;
    uint32_t tag = 0;
    req_type_e reqType = UINT8;
    uint32_t vnet = 0;
};

// Empty mixin base of the backend-facing handler data types (Rendezvous*
// Data, WorkloadLayerHandlerData). The former dead `timestamp` field was
// removed with the sim_request NSDMI change.
class MetaData {
  public:
};

enum class ComType {
    None = 0,
    Reduce_Scatter,
    All_Gather,
    All_Reduce,
    All_to_All
};

enum class CollectiveOptimization { Baseline = 0, LocalBWAware };

enum class SchedulingPolicy { LIFO = 0, FIFO, EXPLICIT };

enum class InterDimensionScheduling {
    Ascending = 0,
    RoundRobin,
    OfflineGreedy
};

enum class StreamState {
    Created = 0,
    Ready,
    Executing,
    Zombie,
    Dead
};

enum class EventType {
    CallEvents = 0,
    General,
    RendezvousSend,
    RendezvousRecv,
    PacketReceived,
    PacketSent,
    Rec_Finished,
    Send_Finished,
    Processing_Finished,
    NPU_to_MA,
    MA_to_NPU,
    Consider_Process,
    Consider_Retire,
    Consider_Send_Back,
    StreamInit,
    CommProcessingFinished,
    CollectiveCommunicationFinished
};

}  // namespace AstraSim

#endif /* __COMMON_HH__ */
