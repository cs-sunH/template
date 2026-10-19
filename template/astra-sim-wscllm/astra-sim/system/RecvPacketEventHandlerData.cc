/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#include "astra-sim/system/RecvPacketEventHandlerData.hh"

#include "astra-sim/system/WorkloadLayerHandlerData.hh"

using namespace AstraSim;

// ---------------------------------------------------------------------------
// Perf item 4.1: bounded freelist pool for RecvPacketEventHandlerData.
//
// Every collective chunk allocates one RecvPacketEventHandlerData in
// Ring/HalvingDoubling/DoubleBinaryTreeAllReduce::ready() (plus the
// Workload/CustomAlgorithm recv arms) and releases it at the single delete
// point (Sys::handleEvent's PacketReceived branch). Class-scoped operator
// new/delete route that allocation pair through a bounded LIFO freelist:
// no call site changes, destruction still happens exactly at the existing
// delete point -- by which time every event-queue reference to the object
// has already been popped -- so storage is never reused while any queue
// still references it. No scheduling, ordering or cancel_event semantics
// are involved: only the storage source changes.
//
// The simulator event loop is single-threaded; no locking. An empty
// freelist falls back to the heap (never slower than the pre-pool path
// beyond one null check); a full freelist releases to the heap, so
// retained memory is bounded by capacity * sizeof(RecvPacketEventHandlerData).
// ---------------------------------------------------------------------------
#include <cstddef>
#include <new>

namespace {
constexpr std::size_t kRecvPacketEventPoolCapacity = 8192;

struct RecvPacketEventFreeNode {
    RecvPacketEventFreeNode* next;
};

RecvPacketEventFreeNode* recv_packet_event_freelist = nullptr;
std::size_t recv_packet_event_pool_size = 0;
}  // namespace

void* RecvPacketEventHandlerData::operator new(const std::size_t size) {
    if (size != sizeof(RecvPacketEventHandlerData)) {
        // No subclass exists today; never pool foreign sizes.
        return ::operator new(size);
    }
    if (recv_packet_event_freelist == nullptr) {
        return ::operator new(size);
    }
    auto* const node = recv_packet_event_freelist;
    recv_packet_event_freelist = node->next;
    --recv_packet_event_pool_size;
    return node;
}

void RecvPacketEventHandlerData::operator delete(void* const ptr,
                                                 const std::size_t size) noexcept {
    if (ptr == nullptr) {
        return;
    }
    if (size != sizeof(RecvPacketEventHandlerData) ||
        recv_packet_event_pool_size >= kRecvPacketEventPoolCapacity) {
        ::operator delete(ptr);
        return;
    }
    auto* const node = static_cast<RecvPacketEventFreeNode*>(ptr);
    node->next = recv_packet_event_freelist;
    recv_packet_event_freelist = node;
    ++recv_packet_event_pool_size;
}

void RecvPacketEventHandlerData::operator delete(void* const ptr) noexcept {
    // Unsized fallback (completeness only): every delete site goes through
    // the exact static type, so the sized form above carries the truth.
    RecvPacketEventHandlerData::operator delete(
        ptr, sizeof(RecvPacketEventHandlerData));
}

RecvPacketEventHandlerData::RecvPacketEventHandlerData() {
    this->workload = nullptr;
    this->wlhd = nullptr;
    this->owner = nullptr;
}

RecvPacketEventHandlerData::RecvPacketEventHandlerData(BaseStream* owner,
                                                       int sys_id,
                                                       EventType event)
    : BasicEventHandlerData(sys_id, event) {
    this->workload = nullptr;
    this->wlhd = nullptr;
    this->owner = owner;
}
