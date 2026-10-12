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

constexpr uint64_t CLOCK_PERIOD = 1;           // 1ns

enum time_type_e { SE = 0, MS, US, NS, FS };

// Every communication request in the simulator carries UINT8 payload
// granularity; no wider dtype is produced by any code path.
enum req_type_e { UINT8 = 0 };

struct timespec_t {
    time_type_e time_res;
    long double time_val;
};

struct sim_request {
    // Default member initializers (M1 port, 2026-09-30): the rendezvous
    // path copies this struct by value and reads fields that several issue
    // paths never fill -- the uninitialized copies were formal UB even
    // though the current analytical backend never consumes the payload.
    uint32_t srcRank = 0;
    uint32_t dstRank = 0;
    uint32_t tag = 0;
    req_type_e reqType = UINT8;
    uint64_t reqCount = 0;
    uint32_t vnet = 0;
    uint32_t layerNum = 0;
};

// MetaData survives as an empty marker base for handler-data classes;
// its historic timestamp member had no reader anywhere.
class MetaData {
  public:
};

enum class ComType {
    None = 0,
    Reduce_Scatter,
    All_Gather,
    All_Reduce,
    All_to_All,
    All_Reduce_All_to_All
};

enum class CollectiveOptimization { Baseline = 0, LocalBWAware };

enum class CollectiveBarrier { Blocking = 0, Non_Blocking };

enum class SchedulingPolicy { LIFO = 0, FIFO, EXPLICIT, None };

enum class InjectionPolicy {
    Infinite = 0,
    Aggressive,
    SemiAggressive,
    ExtraAggressive,
    Normal
};

enum class BusType { Both = 0, Shared, Mem };

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
    CollectiveCommunicationFinished,
    CompFinished,
    MemLoadFinished,
    MemStoreFinished
};

}  // namespace AstraSim

#endif /* __COMMON_HH__ */
