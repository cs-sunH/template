/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#ifndef __RECV_PACKET_EVENT_HANDLER_DATA_HH__
#define __RECV_PACKET_EVENT_HANDLER_DATA_HH__

#include "astra-sim/system/BaseStream.hh"
#include "astra-sim/system/BasicEventHandlerData.hh"
#include "astra-sim/system/astraccl/custom_collectives/CustomAlgorithm.hh"

namespace AstraSim {

class WorkloadLayerHandlerData;

class RecvPacketEventHandlerData : public BasicEventHandlerData {
  public:
    RecvPacketEventHandlerData();
    // vnet/stream_id inputs are kept for the astraccl call sites but are no
    // longer stored: both fields had zero readers.
    RecvPacketEventHandlerData(BaseStream* owner,
                               int sys_id,
                               EventType event,
                               int /*vnet*/,
                               int /*stream_id*/);

    Workload* workload;
    WorkloadLayerHandlerData* wlhd;
    BaseStream* owner;
    CustomAlgorithm* custom_algorithm;
    Tick ready_time = 0;
};

}  // namespace AstraSim

#endif /* __RECV_PACKET_EVENT_HANDLER_DATA_HH__ */
