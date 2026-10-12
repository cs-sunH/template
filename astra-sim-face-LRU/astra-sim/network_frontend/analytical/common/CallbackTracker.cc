/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#include "common/CallbackTracker.hh"
#include <cassert>
#include <cstdio>
#include <cstdlib>

using namespace AstraSimAnalytical;

CallbackTracker::CallbackTracker() noexcept {
    // initialize tracker
    tracker = {};
}

std::optional<CallbackTrackerEntry*> CallbackTracker::search_entry(
    const int tag,
    const int src,
    const int dest,
    const ChunkSize chunk_size,
    const int chunk_id) noexcept {
    assert(tag >= 0);
    assert(src >= 0);
    assert(dest >= 0);
    assert(chunk_size > 0);
    assert(chunk_id >= 0);

    // create key and search entry
    const auto key = std::make_tuple(tag, src, dest, chunk_size, chunk_id);
    const auto entry = tracker.find(key);

    // no entry exists
    if (entry == tracker.end()) {
        return std::nullopt;
    }

    // return pointer to entry
    return &(entry->second);
}

CallbackTrackerEntry* CallbackTracker::create_new_entry(
    const int tag,
    const int src,
    const int dest,
    const ChunkSize chunk_size,
    const int chunk_id) noexcept {
    assert(tag >= 0);
    assert(src >= 0);
    assert(dest >= 0);
    assert(chunk_size > 0);
    assert(chunk_id >= 0);

    // create key
    const auto key = std::make_tuple(tag, src, dest, chunk_size, chunk_id);

    // create new emtpy entry
    const auto entry = tracker.emplace(key, CallbackTrackerEntry()).first;

    // return pointer to entry
    return &(entry->second);
}

void CallbackTracker::pop_entry(const int tag,
                                const int src,
                                const int dest,
                                const ChunkSize chunk_size,
                                const int chunk_id) noexcept {
    assert(tag >= 0);
    assert(src >= 0);
    assert(dest >= 0);
    assert(chunk_size > 0);
    assert(chunk_id >= 0);

    // create key
    const auto key = std::make_tuple(tag, src, dest, chunk_size, chunk_id);

    // find entry
    const auto entry = tracker.find(key);
    if (entry == tracker.end()) {
        std::fprintf(stderr,
                     "[Error] (network/analytical) CallbackTracker pop "
                     "without an entry (tag=%d src=%d dest=%d size=%llu "
                     "chunk=%d)\n",
                     tag, src, dest,
                     static_cast<unsigned long long>(chunk_size), chunk_id);
        std::abort();
    }

    // erase entry from the tracker
    tracker.erase(entry);
}

bool CallbackTracker::has_entry_with_mismatched_count(
    const int tag,
    const int src,
    const int dest,
    const ChunkSize chunk_size) noexcept {
    assert(tag >= 0);
    assert(src >= 0);
    assert(dest >= 0);
    assert(chunk_size > 0);

    // Linear scan: the tracker only holds OUTSTANDING chunks, so the scan
    // is bounded by the in-flight window, not by the transfer size.
    // (2026-09-30, ported from the joint repo CallbackTracker.cc)
    for (const auto& keyed_entry : tracker) {
        const auto& key = keyed_entry.first;
        if (std::get<0>(key) == tag && std::get<1>(key) == src &&
            std::get<2>(key) == dest &&
            std::get<3>(key) != chunk_size) {
            return true;
        }
    }
    return false;
}
