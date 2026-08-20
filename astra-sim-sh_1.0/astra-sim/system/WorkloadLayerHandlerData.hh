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
    int sys_id;
    Workload* workload;
    uint64_t node_id;
    // LocalHbmBandwidthModel endpoint-job completions arrive at
    // Workload::call through this same handler; the flag distinguishes the
    // HBM-side arrival of a joined node completion (network / remote-port
    // side uses the default false). Set by Workload when it allocates the
    // endpoint job's handler; never read elsewhere.
    bool is_local_hbm_job;
    WorkloadLayerHandlerData();
};

}  // namespace AstraSim

#endif /* __WORKLOAD_LAYER_HANDLER_DATA_HH__ */
