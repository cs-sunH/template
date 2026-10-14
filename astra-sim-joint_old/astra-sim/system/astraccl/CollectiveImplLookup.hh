#ifndef __COLLECTIVEIMPLLOOKUP_HH__
#define __COLLECTIVEIMPLLOOKUP_HH__

#include <json/json.hpp>

#include "astra-sim/common/Common.hh"
#include "astra-sim/system/astraccl/CollectiveImpl.hh"

using json = nlohmann::json;

namespace AstraSim {

// The GeneralComplexTopology class is needed when native algorithm is chosen.
// When *initializing* GeneralComplexTopology, we need to provide only the native algorithms.
// Therefore, set some enumerations to define bypassing rules.
enum class BypassRule {
    NO_BYPASS = 0,
    BYPASS_ALL_CUSTOM, // Bypass any non-native tier. Look only at native algorithm.
};

/*
 * CollectiveImplLookup decides which algorithm to use for the given collective
 * operation: the native algorithm for that collective specified by the user
 * (traditional algorithm specification).
 *
 * The custom tiers this lookup used to consult first (per-node and
 * global *-implementation-custom / per-node-custom-implementation) were
 * RETIRED: configuring any of the retired custom keys fails closed at parse
 * time in setup_collective_impl_from_config (see reject_retired_custom_key
 * in CollectiveImplLookup.cc).
 */
class CollectiveImplLookup {
    public:
        CollectiveImplLookup(int rank_);
        // The map below owns the CollectiveImpl objects it points to; free
        // them here so per-rank lookups do not leak for the process lifetime.
        ~CollectiveImplLookup();

        void setup_collective_impl_from_config(json j);

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
