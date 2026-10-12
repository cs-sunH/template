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
    virtual ~Algorithm() = default;
    virtual void run(EventType event, CallData* data) = 0;
    virtual void init(BaseStream* stream);
    virtual void call(EventType event, CallData* data);
    virtual void exit();

    // Default-initialized (2026-09-30): subclasses that never use a member
    // (e.g. custom algorithms never touch logical_topo) used to leave it as
    // an uninitialized wild pointer; nullptr/0/None is the honest value.
    BaseStream* stream = nullptr;
    LogicalTopology* logical_topo = nullptr;
    uint64_t data_size = 0;
    uint64_t final_data_size = 0;
    ComType comType = ComType::None;
};

}  // namespace AstraSim

#endif /* __ALGORITHM_HH__ */
