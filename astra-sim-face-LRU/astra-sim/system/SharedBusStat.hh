/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#ifndef __SHARED_BUS_STAT_HH__
#define __SHARED_BUS_STAT_HH__

#include "astra-sim/system/BasicEventHandlerData.hh"

namespace AstraSim {

// Payload shell only. The eight total_shared_bus_*/total_mem_bus_* delay
// fields together with their update_bus_stats/take_bus_stats_average chain
// were zero-consumer, zero-output dead storage (audit boundary item ② of the
// dead-code inventory) and have been removed. The class survives as the
// non-null CallData payload identity of the MemBus/LogGP event flows
// (CustomAlgorithm::call asserts the payload is non-null) and as the now
// inert base of StreamStat/MemMovRequest.
class SharedBusStat : public BasicEventHandlerData {
  public:
    SharedBusStat() {}
};

}  // namespace AstraSim

#endif /* __SHARED_BUS_STAT_HH__ */
