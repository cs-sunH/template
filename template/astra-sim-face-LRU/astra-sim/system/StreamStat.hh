/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#ifndef __STREAM_STAT_HH__
#define __STREAM_STAT_HH__

#include "astra-sim/system/SharedBusStat.hh"

namespace AstraSim {

// The per-stream/per-dataset network-statistic accumulation
// (net_message_latency / queuing_delay / counters and their update/average
// chain) was write-only end to end -- no reader ever consumed the data or
// the averaged result (face-LRU audit 2026-09-29) -- and was removed. The
// class survives as the inert typed base of BaseStream and as the DataSet
// completion payload identity.
class StreamStat : public SharedBusStat {
  public:
    StreamStat() {}
};

}  // namespace AstraSim

#endif /* __STREAM_STAT_HH__ */
