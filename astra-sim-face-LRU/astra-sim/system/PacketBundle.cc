/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#include "astra-sim/system/PacketBundle.hh"

using namespace AstraSim;

PacketBundle::PacketBundle(Sys* sys,
                           BaseStream* stream,
                           std::list<MyPacket*> /*locked_packets*/,
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
        // Fail closed: local_mem_bw defaults to 0 and stays 0 when the system
        // config omits local-mem-bw. Dividing by it yields +inf whose cast to
        // an integer type is UB and would park this event in the far future,
        // hanging the simulation silently.
        if (sys->local_mem_bw <= 0) {
            sys->sys_panic(
                "PacketBundle collective processing requires a positive "
                "local-mem-bw in the system config");
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
