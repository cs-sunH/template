/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#include "astra-sim/system/PacketBundle.hh"

using namespace AstraSim;

// (The locked_packets constructor parameter is gone: it was passed by value
// and immediately discarded -- call() used to stamp each packet's ready_time
// here, and that loop was retired together with the removal of
// MyPacket::ready_time.)
PacketBundle::PacketBundle(Sys* sys,
                           BaseStream* stream,
                           bool needs_processing,
                           bool send_back,
                           uint64_t size,
                           MemBus::Transmition transmition) {
    this->sys = sys;
    this->needs_processing = needs_processing;
    this->send_back = send_back;
    this->size = size;
    this->stream = stream;
    this->transmition = transmition;
}

void PacketBundle::send_to_MA() {
    sys->memBus->send_from_NPU_to_MA(transmition, size, needs_processing,
                                     send_back, this);
}

void PacketBundle::send_to_NPU() {
    sys->memBus->send_from_MA_to_NPU(transmition, size, needs_processing,
                                     send_back, this);
}

void PacketBundle::call(EventType event, CallData* data) {
    if (needs_processing == true) {
        needs_processing = false;
        if (sys->local_mem_bw <= 0) {
            // A missing or non-positive local-mem-bw would turn the transfer
            // delay below into inf/NaN and, after the uint64_t cast, an
            // astronomical tick that stalls the simulation.  Fail fast
            // instead of silently hanging.
            Sys::sys_panic(
                "local-mem-bw must be set to a positive bandwidth (GB/s) in "
                "the system configuration before collective processing can "
                "compute a local memory access delay");
        }
        // delay[ns], size[bytes], local_mem_bw[bytes/s]. Each local
        // HBM write/read pays the configured fixed access latency.
        // delay stays a local: try_register_event consumes it through a
        // non-const Tick& and zeroes it, so storing it on the object
        // would be dead state.
        const auto local_mem_access_delay =
            sys->local_mem_latency +
            static_cast<uint64_t>(static_cast<double>(size) /
                                  sys->local_mem_bw * 1e9);
        Tick delay = 3 * local_mem_access_delay;  // write + read + read
        sys->try_register_event(this, EventType::CommProcessingFinished, data,
                                delay);
        return;
    }
    stream->call(EventType::General, data);
    delete this;
}
