/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#ifndef __COMMUNICATOR_GROUP_HH__
#define __COMMUNICATOR_GROUP_HH__

#include <map>
#include <vector>

#include "astra-sim/system/Common.hh"

namespace AstraSim {

class Sys;
class CollectivePlan;

/// Communicator identity is order-sensitive: custom-collective rank mapping
/// and shaped dimensions both depend on the exact sequence. Comparing sorted
/// rank sets would incorrectly collapse distinct definitions.
bool same_communicator_definition(
    const std::vector<int>& lhs_ranks,
    const std::vector<int>& lhs_dimensions,
    const std::vector<int>& rhs_ranks,
    const std::vector<int>& rhs_dimensions);

/// Return the algorithm-rank position of `rank` in the communicator's
/// order-sensitive rank vector, or -1 when the rank is not a member.
/// Test-only helper: its production consumer (the retired custom
/// collectives' algorithm-rank mapping) is gone, and the only remaining
/// reference is the communicator_group_order regression test.
int communicator_rank_position(const std::vector<int>& ordered_ranks,
                               int rank) noexcept;

class CommunicatorGroup {
  public:
    // Width of one collective stream-id window. The world group owns
    // [0, STREAM_WINDOW); the group with logical id N owns
    // [N * STREAM_WINDOW, (N + 1) * STREAM_WINDOW). A stream id doubles as
    // the COLLECTIVE tag of every packet the collective emits, so the window
    // MUST be derived from the logical group id: every rank constructs its
    // own CommunicatorGroup copy of the same logical group, and the native
    // algorithm layer (Ring/HalvingDoubling/...) pairs packets across ranks
    // by tag -- identical tags on every rank are therefore a hard invariant
    // (DB1 2026-09-28: a per-object process-wide slot lease broke it and
    // dead-ended every multi-rank collective; see the lease statics below).
    static constexpr int STREAM_WINDOW = 1000000;

    // Lease this logical group's stream window (idempotent per id; returns
    // the shared base). Fail-closed on a group id that would fold in the
    // COLLECTIVE tag remap, or on a second live group with a DIFFERENT
    // membership claiming the same id. F1-CPP C1-2 (2026-09-29): the
    // registry fingerprint is the ordered ranks only -- dimensions are
    // per-view plan-shape metadata (a pg-metadata group and a comm-group
    // file entry over the same membership differ exactly there) that never
    // reaches a stream id or packet tag, so a dims-only difference is
    // reference-counted as the same logical communicator, not panicked on.
    static int lease_stream_window(int id,
                                   const std::vector<int>& ranks,
                                   const std::vector<int>& dimensions);
    // Release one live copy's lease; the registry entry (and its identity
    // record) goes away when the last copy dies.
    static void release_stream_window(int id,
                                      const std::vector<int>& ranks,
                                      const std::vector<int>& dimensions);

    CommunicatorGroup(int comm_group_id,
                      std::vector<int> involved_NPUs,
                      Sys* generator,
                      std::vector<int> dimension_sizes = {});
    // `workload_node_id` is a retired lookup key (see the [operation
    // specific collective selection -- RETIRED mechanism] note in Sys.hh):
    // the custom per-operator selection no longer exists and the parameter
    // is kept only for call-site compatibility.
    CollectivePlan* get_collective_plan(ComType comm_type, uint64_t workload_node_id = -1);
    void set_id(int id);
    [[nodiscard]] bool matches_definition(
        const std::vector<int>& ranks,
        const std::vector<int>& dimensions = {}) const;
    [[nodiscard]] const std::vector<int>& get_dimension_sizes() const {
        return dimension_sizes;
    }
    [[nodiscard]] int get_id() const { return id; }
    ~CommunicatorGroup();

    std::vector<int> involved_NPUs;
    int num_streams;
    // Exclusive upper bound of this group's stream-id window; Sys fails
    // closed when num_streams reaches it instead of spilling into the next
    // group's tags.
    [[nodiscard]] int stream_tag_ceiling() const {
        return stream_base_ + STREAM_WINDOW;
    }

  private:
    int id;
    Sys* generator;
    std::vector<int> dimension_sizes;
    std::map<ComType, CollectivePlan*> comm_plans;
    // Base of this logical group's stream-id window (id * STREAM_WINDOW);
    // 0 while unleased.
    int stream_base_ = 0;
};

}  // namespace AstraSim

#endif /* __COMMUNICATOR_GROUP_HH__ */
