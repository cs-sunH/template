/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#include "astra-sim/system/PacketBundle.hh"

#include <limits>

using namespace AstraSim;

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
    creation_time = Sys::boostedTick();
}

void PacketBundle::send_to_MA() {
    if (size > (uint64_t)std::numeric_limits<int>::max()) {
        Sys::sys_panic("memory transfer size exceeds int range");
    }
    sys->memBus->send_from_NPU_to_MA(transmition, size, needs_processing,
                                     send_back, this);
}

void PacketBundle::send_to_NPU() {
    if (size > (uint64_t)std::numeric_limits<int>::max()) {
        Sys::sys_panic("memory transfer size exceeds int range");
    }
    sys->memBus->send_from_MA_to_NPU(transmition, size, needs_processing,
                                     send_back, this);
}

void PacketBundle::call(EventType event, CallData* data) {
    if (needs_processing == true) {
        needs_processing = false;
        if (sys->local_mem_bw <= 0) {
            Sys::sys_panic(
                "needs_processing memory access requires a positive "
                "local-mem-bw");
        }
        // this->delay[ns], size[bytes], local_mem_bw[bytes/s]. Each local
        // HBM write/read pays the configured fixed access latency.
        const auto local_mem_access_delay =
            sys->local_mem_latency +
            static_cast<uint64_t>(static_cast<double>(size) /
                                  sys->local_mem_bw * 1e9);
        this->delay = 3 * local_mem_access_delay;  // write + read + read
        sys->try_register_event(this, EventType::CommProcessingFinished, data,
                                this->delay);
        return;
    }
    stream->call(EventType::General, data);
    delete this;
}
