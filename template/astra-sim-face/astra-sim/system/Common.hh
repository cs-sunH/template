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

enum time_type_e { NS };

enum req_type_e { UINT8 = 0 };

struct timespec_t {
    time_type_e time_res;
    long double time_val;
};

struct sim_request {
    // Default member initializers: the rendezvous path copies this struct by
    // value and reads every field, while several issue paths only fill part
    // of it (issue_recv_comm filled none) -- the uninitialized copies were
    // formal UB read by rendezvous_sim_send/recv even though the current
    // analytical backend never consumes the payload.
    uint32_t srcRank = 0;
    uint32_t dstRank = 0;
    uint32_t tag = 0;
    req_type_e reqType = UINT8;
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

enum class IntraDimensionScheduling {
    FIFO = 0,
    RG,
    SmallestFirst,
    LessRemainingPhaseFirst
};

enum class InterDimensionScheduling {
    Ascending = 0,
    RoundRobin,
    OfflineGreedy
};

enum class StreamState {
    Created = 0,
    Ready,
    Executing
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
