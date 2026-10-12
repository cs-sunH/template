/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#ifndef __COMMUNICATOR_GROUP_HH__
#define __COMMUNICATOR_GROUP_HH__

#include <map>
#include <utility>
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
int communicator_rank_position(const std::vector<int>& ordered_ranks,
                               int rank) noexcept;

class CommunicatorGroup {
  public:
    CommunicatorGroup(int comm_group_id,
                      std::vector<int> involved_NPUs,
                      Sys* generator,
                      std::vector<int> dimension_sizes = {});
    // For a detailed description on why we need `workload_node_id`,
    // Refer to the comment titled [operation specific custom collective].
    CollectivePlan* get_collective_plan(ComType comm_type, uint64_t workload_node_id = -1);
    int get_position_in_group();
    void set_id(int id);
    // (matches_definition was removed (2026-10-01): zero callers; Workload's
    // group dedup compares involved_NPUs directly -- see the note in
    // CommunicatorGroup.cc.)
    [[nodiscard]] const std::vector<int>& get_dimension_sizes() const {
        return dimension_sizes;
    }
    [[nodiscard]] int get_id() const { return id; }
    ~CommunicatorGroup();

    std::vector<int> involved_NPUs;
    int num_streams;
    // 本组流 id 窗口的独占上界；num_streams 触达它时 Sys fail-closed，
    // 而不是静默溢进下一组的 tag 块（跨组 (src,dst,tag) 三元组别名 =
    // 错配对或永久互等）。
    [[nodiscard]] int stream_tag_ceiling() const {
        return id * 1000000 + 1000000;
    }

  private:
    int id;
    int pos_in_group;
    Sys* generator;
    std::vector<int> dimension_sizes;
    // Keyed by (collective type, workload node id): per-node custom
    // implementations make the node id part of the plan identity.
    std::map<std::pair<ComType, uint64_t>, CollectivePlan*> comm_plans;
};

}  // namespace AstraSim

#endif /* __COMMUNICATOR_GROUP_HH__ */
