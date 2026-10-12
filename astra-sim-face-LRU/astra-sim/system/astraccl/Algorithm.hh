/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#ifndef __ALGORITHM_HH__
#define __ALGORITHM_HH__

#include "astra-sim/system/BaseStream.hh"
#include "astra-sim/system/CallData.hh"
#include "astra-sim/system/Callable.hh"
#include "astra-sim/system/astraccl/native_collectives/logical_topology/LogicalTopology.hh"

namespace AstraSim {

class Algorithm : public Callable {
  public:
    Algorithm();
    virtual ~Algorithm() = default;
    virtual void run(EventType event, CallData* data) = 0;
    virtual void init(BaseStream* stream);
    virtual void call(EventType event, CallData* data);
    virtual void exit();

    // Default member initializers: CollectivePhase reads final_data_size and
    // comType unconditionally right after construction, and algorithm classes
    // whose constructor leaves a member untouched on some ComType paths must
    // not propagate indeterminate values.
    BaseStream* stream = nullptr;
    LogicalTopology* logical_topo = nullptr;
    uint64_t data_size = 0;
    uint64_t final_data_size = 0;
    ComType comType = ComType::None;
    bool enabled = true;
};

}  // namespace AstraSim

#endif /* __ALGORITHM_HH__ */
