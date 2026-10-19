/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#ifndef __OFFLINE_GREEDY_HH__
#define __OFFLINE_GREEDY_HH__

#include <cstddef>
#include <cstdint>
#include <map>
#include <vector>

#include "astra-sim/system/Common.hh"
#include "astra-sim/system/Sys.hh"

namespace AstraSim {

struct OfflineGreedyScheduleKey {
    int64_t communicator_namespace;
    int64_t chunk_id;

    bool operator<(const OfflineGreedyScheduleKey& other) const {
        if (communicator_namespace != other.communicator_namespace) {
            return communicator_namespace < other.communicator_namespace;
        }
        return chunk_id < other.chunk_id;
    }
};

struct OfflineGreedyScheduleEntry {
    std::vector<int> schedule;
    uint64_t chunk_size;
    size_t consumed;
    size_t expected_consumers;
};

// All ranks participating in one collective must reuse the exact schedule and
// chunk size produced by the first rank.  Keep that rendezvous state in one
// composite-keyed journal so communicator-local stream ids cannot collide and
// an entry is reclaimed as soon as the real communicator has consumed it.
class OfflineGreedyScheduleJournal {
  public:
    static bool consume(const OfflineGreedyScheduleKey& key,
                        size_t expected_consumers,
                        std::vector<int>& schedule,
                        uint64_t& chunk_size);
    static void publish(const OfflineGreedyScheduleKey& key,
                        std::vector<int> schedule,
                        uint64_t chunk_size,
                        size_t expected_consumers);
    // Test-only observability: production code only uses publish/consume.
    // These two accessors exist for the OfflineGreedyScheduleJournal
    // regression test's assertions.
    static size_t pending_entry_count();
    static void reset();

  private:
    static std::map<OfflineGreedyScheduleKey, OfflineGreedyScheduleEntry>
        entries;
};

class DimElapsedTime {
  public:
    int dim_num;
    double elapsed_time;
    DimElapsedTime(int dim_num);
    bool operator<(const DimElapsedTime& dimElapsedTime) const {
        return (elapsed_time < dimElapsedTime.elapsed_time);
    }
};
class OfflineGreedy {
  public:
    Sys* sys;
    std::vector<DimElapsedTime> dim_elapsed_time;
    std::vector<double> dim_BW;
    std::vector<int> dim_size;
    OfflineGreedy(Sys* sys);
    void reset_loads();
    // The historical inter_dim_scheduling parameter was always
    // OfflineGreedy here (Sys only routes into this scheduler under that
    // mode); it was removed as an always-true branch (2026-09-30).
    std::vector<int> get_chunk_scheduling(
        int64_t communicator_namespace,
        long long chunk_id,
        size_t expected_consumers,
        uint64_t& remaining_data_size,
        uint64_t recommended_chunk_size,
        std::vector<bool>& dimensions_involved,
        ComType comm_type);
    uint64_t get_chunk_size_from_elapsed_time(double elapsed_time,
                                              DimElapsedTime dim,
                                              ComType comm_type);
};

}  // namespace AstraSim

#endif /* __OFFLINE_GREEDY_HH__ */
