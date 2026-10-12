/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#ifndef __SHARED_BUS_STAT_HH__
#define __SHARED_BUS_STAT_HH__

#include "astra-sim/system/BasicEventHandlerData.hh"

namespace AstraSim {

// Payload-only class: bus-transfer events need a CallData object to carry
// the handler identity (sys_id/event) from MemBus to the completing callable.
// The retired accumulate-and-average statistics chain (bus delay fields,
// request counters, update/take accessors) had no terminal consumer — the
// notifier hands the workload layer an IntData and nothing ever read the
// aggregated numbers — so it was removed outright.
class SharedBusStat : public BasicEventHandlerData {
  public:
    SharedBusStat() {}
};

}  // namespace AstraSim

#endif /* __SHARED_BUS_STAT_HH__ */
