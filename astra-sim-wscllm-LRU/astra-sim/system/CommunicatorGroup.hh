/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#ifndef __COMMUNICATOR_GROUP_HH__
#define __COMMUNICATOR_GROUP_HH__

#include <assert.h>
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

  private:
    int id;
    int pos_in_group;
    Sys* generator;
    std::vector<int> dimension_sizes;
    std::map<ComType, CollectivePlan*> comm_plans;
};

}  // namespace AstraSim

#endif /* __COMMUNICATOR_GROUP_HH__ */
