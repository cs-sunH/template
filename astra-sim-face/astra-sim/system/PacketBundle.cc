/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#include "astra-sim/system/PacketBundle.hh"

#include <cmath>
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
        // (2026-10-01) overflow guard, same shape as the endpoint-delay x
        // injection-scale guard in Sys.cc: local-mem-latency accepts the
        // full uint64 domain, so both the latency+transfer addition and the
        // 3x multiplication would wrap on extreme-but-legal values and
        // silently mistime the collective. Check in the double domain
        // before any conversion (3x <= uint64 max also proves the sum and
        // the cast fit).
        const double transfer_ns =
            static_cast<double>(size) / sys->local_mem_bw * 1e9;
        const double total_3x =
            3.0 * (static_cast<double>(sys->local_mem_latency) + transfer_ns);
        if (!std::isfinite(total_3x) ||
            total_3x > static_cast<double>(
                           std::numeric_limits<uint64_t>::max())) {
            Sys::sys_panic(
                "PacketBundle 3x local memory access delay overflows uint64");
        }
        const auto local_mem_access_delay =
            sys->local_mem_latency + static_cast<uint64_t>(transfer_ns);
        // 3x covers the (needs_processing=true, send_back=false) shape:
        // write + read + read. DBT reduce+send-back nodes
        // (needs_processing=true, send_back=true) pay the SAME 3x here --
        // the extra write-back HBM access is not modeled separately
        // (upstream-inherited approximation; MemBus's active path never
        // consumes the send_back flag). Changing the multiplier changes
        // collective timing baselines -- treat as frozen model input
        // (documented 2026-09-30).
        this->delay = 3 * local_mem_access_delay;
        sys->try_register_event(this, EventType::CommProcessingFinished, data,
                                this->delay);
        return;
    }
    stream->call(EventType::General, data);
    delete this;
}
