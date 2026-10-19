/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#pragma once

#include "common/CallbackTrackerEntry.hh"
#include "common/ChunkIdGenerator.hh"
#include <map>
#include <tuple>
#include <vector>

namespace AstraSimAnalytical {

/**
 * CallbackTracker keeps track of sim_send() and sim_recv() callbacks of each
 * chunk identified by (tag, src, dest, chunk_size, chunk_id) tuple.
 */
class CallbackTracker {
  public:
    CallbackTracker() noexcept;

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

    /// Perf item 4.4 (single-traversal search-or-create): returns the entry
    /// for the key, creating a default one when absent; `created` reports
    /// which happened. Replaces the search_entry()+create_new_entry() pair
    /// (two tree traversals on the create path) at call sites that branch
    /// on existence and then use the entry either way. The lower_bound +
    /// hinted-emplace implementation lands the node exactly where plain
    /// emplace(key, CallbackTrackerEntry()) would, so the map contents and
    /// their order are unchanged.
    CallbackTrackerEntry* search_or_create_entry(int tag,
                                                 int src,
                                                 int dest,
                                                 ChunkSize chunk_size,
                                                 int chunk_id,
                                                 bool& created) noexcept;

    /// Opaque iterator handle to one resident entry (perf item 4.4).
    using EntryHandle = std::map<std::tuple<int, int, int, ChunkSize, int>,
                 CallbackTrackerEntry>::iterator;

    /// Search result of find_entry_with_handle(): the entry pointer (null
    /// on miss) plus the handle for an O(1) pop_entry(). std::map iterators
    /// stay valid while OTHER keys are inserted or erased, so the handle
    /// may be held across the entry's callback invocations.
    struct FoundEntry {
        CallbackTrackerEntry* entry;
        EntryHandle handle;
    };

    /// Perf item 4.4 (handle search): one find that returns the entry and
    /// an O(1)-pop handle, so the chunk-arrival path and sim_recv's
    /// finished path skip the second key-based traversal of pop_entry().
    /// Miss is reported as entry == nullptr (handle is tracker.end()).
    FoundEntry find_entry_with_handle(int tag,
                                      int src,
                                      int dest,
                                      ChunkSize chunk_size,
                                      int chunk_id) noexcept;

    /// O(1) erase through a handle returned by find_entry_with_handle().
    /// Fails closed on an end() handle (unreachable for valid producers).
    void pop_entry(EntryHandle handle) noexcept;

    /// Run-end residual audit accessors: entry count and up to max_samples
    /// keys still resident in the tracker. A quiescent run end should have
    /// drained every entry (each completion pops its entry); any resident
    /// entry is an anomaly worth reporting. Callers own the warn decision.
    [[nodiscard]] size_t size() const noexcept;
    /// Not noexcept: returns a freshly allocated vector.
    [[nodiscard]] std::vector<std::tuple<int, int, int, ChunkSize, int>>
    resident_key_samples(size_t max_samples) const;

  private:
    /// map from (tag, src, dest, chunk_size, chunk_id) tuple to
    /// CallbackTrackerEntry
    std::map<std::tuple<int, int, int, ChunkSize, int>, CallbackTrackerEntry>
        tracker;
};

}  // namespace AstraSimAnalytical
