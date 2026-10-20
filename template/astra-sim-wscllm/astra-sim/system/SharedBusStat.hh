/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#ifndef __SHARED_BUS_STAT_HH__
#define __SHARED_BUS_STAT_HH__

#include "astra-sim/system/BasicEventHandlerData.hh"

namespace AstraSim {

// The shared/mem bus-stat accounting (8 delay fields, 2 request counters,
// both update_bus_stats overloads and take_bus_stats_average) was write-only
// end to end -- every value written, accumulated and averaged here had no
// reader anywhere in the repo (removed 2026-10-01, mirroring the face repo's
// same-day removal). The class survives as an empty BasicEventHandlerData
// shell because StreamStat (the BaseStream/DataSet mixin) and MemMovRequest
// still inherit it; event payloads that used to be heap-allocated carriers
// of these stats now pass nullptr.
class SharedBusStat : public BasicEventHandlerData {
  public:
    SharedBusStat() = default;
};

}  // namespace AstraSim

#endif /* __SHARED_BUS_STAT_HH__ */
