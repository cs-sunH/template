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
#include "astra-sim/system/MyPacket.hh"

namespace AstraSim {

class Sys;
class PacketBundle : public Callable {
  public:
    // (The locked_packets constructor parameter is gone: it was passed by
    // value and immediately discarded -- call() used to stamp each packet's
    // ready_time here, and that loop was retired together with the removal
    // of MyPacket::ready_time.)
    PacketBundle(Sys* sys,
                 BaseStream* stream,
                 bool needs_processing,
                 bool send_back,
                 uint64_t size,
                 MemBus::Transmition transmition);
    void send_to_MA();
    void send_to_NPU();
    void call(EventType event, CallData* data);

    Sys* sys;
    bool needs_processing;
    bool send_back;
    uint64_t size;
    BaseStream* stream;
    MemBus::Transmition transmition;
};

}  // namespace AstraSim

#endif /* __PACKET_BUNDLE_HH__ */
