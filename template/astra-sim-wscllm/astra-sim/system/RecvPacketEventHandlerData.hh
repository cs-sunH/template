/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#ifndef __RECV_PACKET_EVENT_HANDLER_DATA_HH__
#define __RECV_PACKET_EVENT_HANDLER_DATA_HH__

#include "astra-sim/system/BaseStream.hh"
#include "astra-sim/system/BasicEventHandlerData.hh"
#include <cstddef>

namespace AstraSim {

class Workload;
class WorkloadLayerHandlerData;

class RecvPacketEventHandlerData : public BasicEventHandlerData {
  public:
    RecvPacketEventHandlerData();
    RecvPacketEventHandlerData(BaseStream* owner, int sys_id, EventType event);

    // Perf item 4.1 (bounded freelist pool): class-scoped operator
    // new/delete recycle storage between the per-chunk allocations of
    // the collective algorithms (Ring/HalvingDoubling/
    // DoubleBinaryTreeAllReduce ready(), plus the Workload/
    // CustomAlgorithm recv arms) and the single existing delete point
    // (Sys::handleEvent's PacketReceived branch). Behavior-identical
    // by construction: call sites, constructor/destructor order,
    // event ordering and cancel_event semantics are all untouched --
    // only the storage source changes. See the .cc for the bounded
    // freelist (empty pool -> heap, full pool -> heap release).
    static void* operator new(std::size_t size);
    static void operator delete(void* ptr, std::size_t size) noexcept;
    static void operator delete(void* ptr) noexcept;

    Workload* workload;
    WorkloadLayerHandlerData* wlhd;
    BaseStream* owner;
};

}  // namespace AstraSim

#endif /* __RECV_PACKET_EVENT_HANDLER_DATA_HH__ */
