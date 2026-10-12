/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#pragma once

#include "common/CallbackTrackerEntry.hh"
// ChunkSize is visible through CallbackTrackerEntry.hh -> Type.h; the
// direct ChunkIdGenerator.hh include (net-frontend-D2) is retired.
#include <map>
#include <tuple>

namespace AstraSimAnalytical {

/**
 * CallbackTracker keeps track of sim_send() and sim_recv() callbacks of each
 * chunk identified by (tag, src, dest, chunk_size, chunk_id) tuple.
 */
class CallbackTracker {
  public:
    /// Key = (tag, src, dest, chunk_size, chunk_id)
    using Key = std::tuple<int, int, int, ChunkSize, int>;

    /**
     * Search for the entry identified by (tag, src, dest, chunk_size, chunk_id)
     * tuple.
     *
     * @param tag tag of the sim_send() or sim_recv() call
     * @param src src NPU ID of the sim_send() or sim_recv() call
     * @param dest dest NPU ID of the sim_send() or sim_recv() call
     * @param chunk_size chunk size of the sim_send() or sim_recv() call
     * @param chunk_id id of the chunk
     * @return the found entry if exists, std::nullopt otherwise
     */
    std::optional<CallbackTrackerEntry*> search_entry(int tag,
                                                      int src,
                                                      int dest,
                                                      ChunkSize chunk_size,
                                                      int chunk_id) noexcept;

    /**
     * Create a new entry identified by (tag, src, dest, chunk_size, chunk_id)
     * tuple.
     *
     * @param tag tag of the sim_send() or sim_recv() call
     * @param src src NPU ID of the sim_send() or sim_recv() call
     * @param dest dest NPU ID of the sim_send() or sim_recv() call
     * @param chunk_size chunk size of the sim_send() or sim_recv() call
     * @param chunk_id id of the chunk
     * @return the created entry
     */
    CallbackTrackerEntry* create_new_entry(int tag,
                                           int src,
                                           int dest,
                                           ChunkSize chunk_size,
                                           int chunk_id) noexcept;

    /**
     * Remove the entry identified by (tag, src, dest, chunk_size, chunk_id)
     * tuple.
     *
     * @param tag tag of the sim_send() or sim_recv() call
     * @param src src NPU ID of the sim_send() or sim_recv() call
     * @param dest dest NPU ID of the sim_send() or sim_recv() call
     * @param chunk_size chunk size of the sim_send() or sim_recv() call
     * @param chunk_id id of the chunk
     */
    void pop_entry(int tag,
                   int src,
                   int dest,
                   ChunkSize chunk_size,
                   int chunk_id) noexcept;

    /**
     * C3 fix (2026-09-28): count-mismatch diagnostic. Reports whether any
     * outstanding entry shares (tag, src, dest) with the given registration
     * but carries a DIFFERENT chunk_size -- the signature of a send/recv
     * count mismatch. A mismatched pair registers two half-filled entries
     * that wait on each other forever (the send's arrival only completes
     * its own key, so the wrong-count recv is never woken): the run used to
     * hang with zero diagnostics. Diagnostic only -- callers log a warning;
     * no simulation behavior changes.
     *
     * @param tag tag of the sim_send() or sim_recv() call
     * @param src src NPU ID of the sim_send() or sim_recv() call
     * @param dest dest NPU ID of the sim_send() or sim_recv() call
     * @param chunk_size chunk size of the registering call
     * @return true if a same-key-different-count entry is outstanding
     */
    bool has_entry_with_mismatched_count(int tag,
                                         int src,
                                         int dest,
                                         ChunkSize chunk_size) noexcept;

  private:
    /// map from (tag, src, dest, chunk_size, chunk_id) tuple to
    /// CallbackTrackerEntry
    std::map<Key, CallbackTrackerEntry> tracker;
};

}  // namespace AstraSimAnalytical
