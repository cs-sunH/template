/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#ifndef __PACKET_BUNDLE_HH__
#define __PACKET_BUNDLE_HH__

#include "astra-sim/system/BaseStream.hh"
#include "astra-sim/system/Callable.hh"
#include "astra-sim/system/Common.hh"
#include "astra-sim/system/MemBus.hh"
#include <cstddef>

namespace AstraSim {

class Sys;
class PacketBundle : public Callable {
  public:
    PacketBundle(Sys* sys,
                 BaseStream* stream,
                 bool needs_processing,
                 bool send_back,
                 uint64_t size,
                 MemBus::Transmition transmition);
    void send_to_MA();
    void send_to_NPU();
    void call(EventType event, CallData* data);

    // Perf item 4.1 (bounded freelist pool): class-scoped operator
    // new/delete recycle storage between the per-packet allocations of the
    // collective algorithms (Ring/HalvingDoubling/DoubleBinaryTreeAllReduce
    // release_packets()) and the single existing delete point (the
    // `delete this;` at the end of PacketBundle::call). Behavior-identical
    // by construction: call sites, destructor order, event ordering and
    // cancel_event semantics are all untouched -- only the storage source
    // changes. See the .cc for the bounded freelist (empty pool -> heap,
    // full pool -> heap release).
    static void* operator new(std::size_t size);
    static void operator delete(void* ptr, std::size_t size) noexcept;
    static void operator delete(void* ptr) noexcept;

    Sys* sys;
    bool needs_processing;
    bool send_back;
    uint64_t size;
    BaseStream* stream;
    MemBus::Transmition transmition;
    Tick delay;
};

}  // namespace AstraSim

#endif /* __PACKET_BUNDLE_HH__ */
