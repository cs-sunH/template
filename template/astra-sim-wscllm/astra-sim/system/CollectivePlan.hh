/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#ifndef __COLLECTIVE_PLAN_HH__
#define __COLLECTIVE_PLAN_HH__

#include <vector>

#include "astra-sim/system/Common.hh"
#include "astra-sim/system/astraccl/native_collectives/logical_topology/LogicalTopology.hh"

namespace AstraSim {

class CollectivePlan {
  public:
    LogicalTopology* topology;
    std::vector<CollectiveImpl*> implementation_per_dimension;
    std::vector<bool> dimensions_involved;
    bool should_be_removed;
    bool implementations_should_be_removed;
    CollectivePlan(LogicalTopology* topology,
                   std::vector<CollectiveImpl*> implementation_per_dimension,
                   std::vector<bool> dimensions_involved,
                   bool should_be_removed,
                   bool implementations_should_be_removed);
    ~CollectivePlan();
};

}  // namespace AstraSim
#endif /* __COLLECTIVE_PLAN_HH__ */
