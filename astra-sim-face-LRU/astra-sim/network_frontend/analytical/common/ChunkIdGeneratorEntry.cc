/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#include "common/ChunkIdGeneratorEntry.hh"
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <limits>

using namespace AstraSimAnalytical;

namespace {

// Same diagnostic channel/shape as ChunkIdGenerator.cc's
// fail_chunk_lifecycle (kept local: the entry class does not carry the
// (tag,src,dest,size) key context its owner has).
[[noreturn]] void fail_chunk_id_overflow(const char* which) noexcept {
    std::fprintf(stderr,
                 "[Error] (network/analytical) ChunkIdGenerator %s id "
                 "overflow: %s id reached INT_MAX -- unpaired sim_%s calls "
                 "would wrap the id space and corrupt the completion "
                 "accounting; failing closed\n",
                 which, which, which);
    std::abort();
}

}  // namespace

ChunkIdGeneratorEntry::ChunkIdGeneratorEntry() noexcept
    : send_id(-1),
      recv_id(-1),
      outstanding_(0) {}

int ChunkIdGeneratorEntry::get_send_id() const noexcept {
    assert(send_id >= 0);

    return send_id;
}

int ChunkIdGeneratorEntry::get_recv_id() const noexcept {
    assert(recv_id >= 0);

    return recv_id;
}

void ChunkIdGeneratorEntry::increment_send_id() noexcept {
    // INT_MAX fail-closed guard (2026-09-30, mirroring the joint repo
    // ChunkIdGeneratorEntry.cc; type deliberately unchanged): the counters
    // are int, so one more increment past INT_MAX is signed overflow (UB)
    // and a wrapped id silently corrupts the tracker/generator pairing.
    if (send_id == std::numeric_limits<int>::max()) {
        fail_chunk_id_overflow("send");
    }
    send_id++;
}

void ChunkIdGeneratorEntry::increment_recv_id() noexcept {
    // Same guard as increment_send_id.
    if (recv_id == std::numeric_limits<int>::max()) {
        fail_chunk_id_overflow("recv");
    }
    recv_id++;
}

uint64_t ChunkIdGeneratorEntry::outstanding() const noexcept {
    return outstanding_;
}

void ChunkIdGeneratorEntry::increment_outstanding() noexcept {
    ++outstanding_;
}

void ChunkIdGeneratorEntry::decrement_outstanding() noexcept {
    assert(outstanding_ > 0);
    --outstanding_;
}

bool ChunkIdGeneratorEntry::mark_chunk_completed(const int chunk_id) noexcept {
    return completed_ids_.insert(chunk_id).second;
}

size_t ChunkIdGeneratorEntry::completed_count() const noexcept {
    return completed_ids_.size();
}
