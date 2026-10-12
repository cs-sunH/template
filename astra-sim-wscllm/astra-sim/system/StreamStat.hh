/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#ifndef __STREAM_STAT_HH__
#define __STREAM_STAT_HH__

#include "astra-sim/system/Common.hh"
#include "astra-sim/system/SharedBusStat.hh"

namespace AstraSim {

// The per-stream/per-dataset network-statistic accumulation
// (net_message_latency / queuing_delay / counters and their update/average
// chain, plus NetworkStat.hh itself) was write-only end to end -- no reader
// ever consumed the data or the averaged result (face-LRU audit 2026-09-29
// mirrored to joint, mirrored to wscllm 2026-09-30) -- and was removed. The
// bus-stat accounting (SharedBusStat) went the same way on 2026-10-01: its
// fields/counters/average had zero consumers, so SharedBusStat is now an
// empty BasicEventHandlerData shell. The class survives as the inheritance
// tag of BaseStream/DataSet (DataSet::notify_stream_finished uses it only as
// a type tag).
class StreamStat : public SharedBusStat {
  public:
    StreamStat() = default;
};

}  // namespace AstraSim

#endif /* __STREAM_STAT_HH__ */
