/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#include "astra-sim/system/PacketBundle.hh"

#include <cmath>
#include <limits>

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
