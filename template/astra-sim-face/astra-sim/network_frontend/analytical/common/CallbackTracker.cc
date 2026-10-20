/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#include "common/CallbackTracker.hh"
#include <algorithm>
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

    // create new empty entry
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

CallbackTrackerEntry* CallbackTracker::search_or_create_entry(
    const int tag,
    const int src,
    const int dest,
    const ChunkSize chunk_size,
    const int chunk_id,
    bool& created) noexcept {
    assert(tag >= 0);
    assert(src >= 0);
    assert(dest >= 0);
    assert(chunk_size > 0);
    assert(chunk_id >= 0);

    // Perf item 4.4: one lower_bound traversal; on a miss the same iterator
    // is the emplace_hint position (amortized O(1) insert), eliminating the
    // second traversal of the search_entry()+create_new_entry() pair. The
    // node lands exactly where plain emplace(key, CallbackTrackerEntry())
    // would, so the map contents and order are unchanged.
    const auto key = std::make_tuple(tag, src, dest, chunk_size, chunk_id);
    auto candidate = tracker.lower_bound(key);
    if (candidate == tracker.end() ||
        tracker.key_comp()(key, candidate->first)) {
        candidate =
            tracker.emplace_hint(candidate, key, CallbackTrackerEntry());
        created = true;
    } else {
        created = false;
    }
    return &(candidate->second);
}

CallbackTracker::FoundEntry CallbackTracker::find_entry_with_handle(
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

    // Perf item 4.4: the same single find as search_entry(), but the
    // iterator is handed back so the arrival path can pop with an O(1)
    // erase. std::map iterators remain valid while other keys are inserted
    // or erased, which is what makes holding the handle across the entry's
    // callback invocations safe.
    const auto key = std::make_tuple(tag, src, dest, chunk_size, chunk_id);
    const auto found = tracker.find(key);
    if (found == tracker.end()) {
        return {nullptr, tracker.end()};
    }
    return {&(found->second), found};
}

void CallbackTracker::pop_entry(const EntryHandle handle) noexcept {
    if (handle == tracker.end()) {
        // Defensive: every valid producer pairs a non-null entry with a
        // dereferenceable handle.
        std::fprintf(stderr,
                     "[Error] (network/analytical) CallbackTracker handle "
                     "pop without an entry\n");
        std::abort();
    }
    tracker.erase(handle);
}

size_t CallbackTracker::size() const noexcept {
    return tracker.size();
}

std::vector<std::tuple<int, int, int, ChunkSize, int>>
CallbackTracker::resident_key_samples(const size_t max_samples) const {
    std::vector<std::tuple<int, int, int, ChunkSize, int>> samples;
    samples.reserve(std::min(max_samples, tracker.size()));
    for (const auto& entry : tracker) {
        if (samples.size() >= max_samples) {
            break;
        }
        samples.push_back(entry.first);
    }
    return samples;
}
