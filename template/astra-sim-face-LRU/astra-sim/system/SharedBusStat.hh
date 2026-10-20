/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#ifndef __SHARED_BUS_STAT_HH__
#define __SHARED_BUS_STAT_HH__

#include "astra-sim/system/BasicEventHandlerData.hh"

namespace AstraSim {

// Payload shell only. This class exists as the non-null CallData payload
// identity of the MemBus/LogGP event flows (CustomAlgorithm::call requires a
// non-null payload) and as the inert base of StreamStat/MemMovRequest.
class SharedBusStat : public BasicEventHandlerData {
  public:
    SharedBusStat() {}
};

}  // namespace AstraSim

#endif /* __SHARED_BUS_STAT_HH__ */
