/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#ifndef __WORKLOAD_LAYER_HANDLER_DATA_HH__
#define __WORKLOAD_LAYER_HANDLER_DATA_HH__

#include "astra-sim/common/AstraNetworkAPI.hh"
#include "astra-sim/system/BasicEventHandlerData.hh"

namespace AstraSim {

class Workload;

class WorkloadLayerHandlerData : public BasicEventHandlerData, public MetaData {
  public:
    // sys_id is inherited from BasicEventHandlerData; redeclaring it here
    // would shadow the base field and read back garbage via a base pointer.
    Workload* workload;
    uint64_t node_id;
    WorkloadLayerHandlerData();
};

}  // namespace AstraSim

#endif /* __WORKLOAD_LAYER_HANDLER_DATA_HH__ */
