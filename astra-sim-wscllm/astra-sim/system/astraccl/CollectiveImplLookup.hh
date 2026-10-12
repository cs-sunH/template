/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#ifndef __COLLECTIVEIMPLLOOKUP_HH__
#define __COLLECTIVEIMPLLOOKUP_HH__

#include <json/json.hpp>

#include "astra-sim/system/Common.hh"
#include "astra-sim/system/astraccl/CollectiveImpl.hh"

namespace AstraSim {

// The GeneralComplexTopology class is needed when native algorithm is chosen.
// When *initializing* GeneralComplexTopology, we need to provide only the native algorithms.
// Therefore, set some enumerations to define bypassing rules.
enum class BypassRule {
    NO_BYPASS = 0,
    BYPASS_ALL_CUSTOM, // Bypass custom priority tiers and look only at native
                       // algorithms; an empty native table yields an empty
                       // result instead of an error.
};

/*
 * CollectiveImplLookup decides which algorithm to use for the given collective
 * operation. Only native algorithm specifications ("ring", "direct[N]", ...)
 * remain: the custom collectives / ET feeder path was retired together with
 * the offline static arm (2026-09-27). Configuring any of the retired custom
 * keys ("all-reduce/all-gather/reduce-scatter/all-to-all-implementation-custom",
 * "per-node-custom-implementation") fails closed at parse time.
 */
class CollectiveImplLookup {
    public:
        CollectiveImplLookup(int rank_);
        // The map below owns the CollectiveImpl objects it points to; free
        // them here so per-rank lookups do not leak for the process lifetime.
        ~CollectiveImplLookup();

        void setup_collective_impl_from_config(nlohmann::json j);

        // `workload_node_id` is retained in the signature for call-site
        // compatibility (it no longer selects anything: the per-node custom
        // tier it used to address was removed with the custom path).
        std::vector<CollectiveImpl*> get_collective_impl(
            ComType comm_type,
            uint64_t workload_node_id,
            BypassRule bypass_rule = BypassRule::NO_BYPASS
        );

    private:
        // Map from Collective Type to a vector of native algorithms, where
        // each element in the vector corresponds to a dimension.
        // The algorithm for each dimension is specified in the system input with the key "all-reduce-implementation".
        std::map<ComType, std::vector<CollectiveImpl*>> native_impl_per_coll_dim;
};

} // namespace AstraSim

#endif /* __COLLECTIVEIMPLLOOKUP_HH__ */