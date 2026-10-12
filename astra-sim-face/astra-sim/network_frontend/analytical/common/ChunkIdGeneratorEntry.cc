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

[[noreturn]] void fail_chunk_id_overflow(const char* stream,
                                         const int id) noexcept {
    std::fprintf(stderr,
                 "[Error] (network/analytical) ChunkIdGeneratorEntry %s id "
                 "exhausted: next id would overflow INT_MAX (current=%d); "
                 "failing closed\n",
                 stream, id);
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
    if (send_id == std::numeric_limits<int>::max()) {
        fail_chunk_id_overflow("send", send_id);
    }
    ++send_id;
}

void ChunkIdGeneratorEntry::increment_recv_id() noexcept {
    if (recv_id == std::numeric_limits<int>::max()) {
        fail_chunk_id_overflow("recv", recv_id);
    }
    ++recv_id;
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
