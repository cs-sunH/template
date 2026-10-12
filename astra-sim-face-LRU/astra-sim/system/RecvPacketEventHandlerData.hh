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
    // The vnet/stream_id parameters were removed (dead value chain cleanup
    // 2026-09-30): both were ignored anonymous arguments fed from
    // MyPacket::preferred_vnet / BaseStream::stream_id at the astraccl call
    // sites and had zero readers anywhere in the tree.
    RecvPacketEventHandlerData(BaseStream* owner,
                               int sys_id,
                               EventType event);

    Workload* workload;
    WorkloadLayerHandlerData* wlhd;
    BaseStream* owner;
    CustomAlgorithm* custom_algorithm;
    // ready_time was removed (dead code cleanup 2026-09-30): the member was
    // only ever written by the constructor and had zero readers anywhere in
    // the tree.
};

}  // namespace AstraSim

#endif /* __RECV_PACKET_EVENT_HANDLER_DATA_HH__ */
