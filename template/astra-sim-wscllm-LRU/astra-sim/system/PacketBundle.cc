/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#include "astra-sim/system/PacketBundle.hh"

using namespace AstraSim;

// ---------------------------------------------------------------------------
// Perf item 4.1: bounded freelist pool for PacketBundle.
//
// Every collective packet allocates one PacketBundle in
// Ring/HalvingDoubling/DoubleBinaryTreeAllReduce::release_packets() and
// releases it at the single delete point (`delete this;` at the end of
// PacketBundle::call -- reached exactly once per bundle on every path:
// the direct Sys event arm and the shared-bus LogGP arm both funnel into
// call(), which either re-registers itself once for local-memory
// processing or finishes and deletes). Class-scoped operator new/delete
// route the allocation pair through a bounded LIFO freelist: no call-site
// changes, destruction still happens exactly at the existing delete point
// -- by which time the event carrying the bundle has already been popped
// -- so storage is never reused while any queue still references it. No
// scheduling, ordering or cancel_event semantics are involved: only the
// storage source changes.
//
// The simulator event loop is single-threaded; no locking. An empty
// freelist falls back to the heap (never slower than the pre-pool path
// beyond one null check); a full freelist releases to the heap, so
// retained memory is bounded by capacity * sizeof(PacketBundle).
// ---------------------------------------------------------------------------
#include <cstddef>
#include <new>

namespace {
constexpr std::size_t kPacketBundlePoolCapacity = 8192;

struct PacketBundleFreeNode {
    PacketBundleFreeNode* next;
};

PacketBundleFreeNode* packet_bundle_freelist = nullptr;
std::size_t packet_bundle_pool_size = 0;
}  // namespace

void* PacketBundle::operator new(const std::size_t size) {
    if (size != sizeof(PacketBundle)) {
        // No subclass exists today; never pool foreign sizes.
        return ::operator new(size);
    }
    if (packet_bundle_freelist == nullptr) {
        return ::operator new(size);
    }
    auto* const node = packet_bundle_freelist;
    packet_bundle_freelist = node->next;
    --packet_bundle_pool_size;
    return node;
}

void PacketBundle::operator delete(void* const ptr,
                                   const std::size_t size) noexcept {
    if (ptr == nullptr) {
        return;
    }
    if (size != sizeof(PacketBundle) ||
        packet_bundle_pool_size >= kPacketBundlePoolCapacity) {
        ::operator delete(ptr);
        return;
    }
    auto* const node = static_cast<PacketBundleFreeNode*>(ptr);
    node->next = packet_bundle_freelist;
    packet_bundle_freelist = node;
    ++packet_bundle_pool_size;
}

void PacketBundle::operator delete(void* const ptr) noexcept {
    // Unsized fallback (completeness only): every delete site (`delete
    // this;` inside call()) goes through the exact static type.
    PacketBundle::operator delete(ptr, sizeof(PacketBundle));
}

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
