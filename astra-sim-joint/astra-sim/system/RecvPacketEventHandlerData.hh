/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#ifndef __RECV_PACKET_EVENT_HANDLER_DATA_HH__
#define __RECV_PACKET_EVENT_HANDLER_DATA_HH__

#include "astra-sim/system/BaseStream.hh"
#include "astra-sim/system/BasicEventHandlerData.hh"
#include "astra-sim/system/Callable.hh"

namespace AstraSim {

class WorkloadLayerHandlerData;

class RecvPacketEventHandlerData : public BasicEventHandlerData {
  public:
    RecvPacketEventHandlerData();
    RecvPacketEventHandlerData(BaseStream* owner,
                               int sys_id,
                               EventType event,
                               int vnet);

    Workload* workload;
    /// Optional receiver-side callable for descriptor-driven transport.
    /// Legacy Workload and collective paths leave this null.
    Callable* callable;
    WorkloadLayerHandlerData* wlhd;
    BaseStream* owner;
    int vnet;
    bool message_end;
};

}  // namespace AstraSim

#endif /* __RECV_PACKET_EVENT_HANDLER_DATA_HH__ */
