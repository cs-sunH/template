/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#ifndef __WORKLOAD_LAYER_HANDLER_DATA_HH__
#define __WORKLOAD_LAYER_HANDLER_DATA_HH__

#include "astra-sim/common/AstraNetworkAPI.hh"
#include "astra-sim/system/Callable.hh"
#include "astra-sim/system/BasicEventHandlerData.hh"

namespace AstraSim {

class Workload;

class WorkloadLayerHandlerData : public BasicEventHandlerData, public MetaData {
  public:
    // Default member initializers make the "partially constructed" state
    // explicit: some p2p callback paths fill only node_id, so sys_id and
    // workload must never carry indeterminate values (mirrors the face repo;
    // face-LRU audit 2026-09-29).
    // NOTE: sys_id intentionally shadows BasicEventHandlerData::sys_id. It is
    // load-bearing: the remote memory backend reads it as the issuing rank
    // (AnalyticalRemoteMemory::issue), which the base member (set from the
    // handler ctor) does not carry. Do not remove it in favor of the base;
    // rename both only together.
    int sys_id = -1;
    Workload* workload = nullptr;
    uint64_t node_id = 0;
    /// Optional descriptor-transport sink. Workload::call transfers ownership
    /// of this cookie to the sink instead of looking up node_id.
    Callable* pipeline_sink = nullptr;
    WorkloadLayerHandlerData();
};

}  // namespace AstraSim

#endif /* __WORKLOAD_LAYER_HANDLER_DATA_HH__ */
