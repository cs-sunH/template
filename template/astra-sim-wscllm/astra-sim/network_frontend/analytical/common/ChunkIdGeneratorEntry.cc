/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#include "common/ChunkIdGeneratorEntry.hh"

#include <cstdio>
#include <cstdlib>
#include <cassert>

using namespace AstraSimAnalytical;

ChunkIdGeneratorEntry::ChunkIdGeneratorEntry() noexcept
    : send_id(0),
      recv_id(0),
      outstanding_(0) {}

int ChunkIdGeneratorEntry::get_send_id() const noexcept {
    // Cap guard in increment_send_id keeps the counter below 2^31, so the
    // narrowing to the public int contract is safe.
    return static_cast<int>(send_id);
}

int ChunkIdGeneratorEntry::get_recv_id() const noexcept {
    return static_cast<int>(recv_id);
}

void ChunkIdGeneratorEntry::increment_send_id() noexcept {
    if (send_id >= INT32_MAX) {
        // Unpaired sends would wrap the id and corrupt the completion
        // accounting; fail closed instead of overflowing.
        std::fprintf(stderr,
                     "[ChunkIdGenerator] send id overflow (unpaired sends "
                     "exceed INT32_MAX): failing closed\n");
        std::abort();
    }
    send_id++;
}

void ChunkIdGeneratorEntry::increment_recv_id() noexcept {
    if (recv_id >= INT32_MAX) {
        std::fprintf(stderr,
                     "[ChunkIdGenerator] recv id overflow (unpaired recvs "
                     "exceed INT32_MAX): failing closed\n");
        std::abort();
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
