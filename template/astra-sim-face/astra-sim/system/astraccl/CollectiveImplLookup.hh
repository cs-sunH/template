#ifndef __COLLECTIVEIMPLLOOKUP_HH__
#define __COLLECTIVEIMPLLOOKUP_HH__

#include <json/json.hpp>

#include "astra-sim/system/Common.hh"
#include "astra-sim/system/astraccl/CollectiveImpl.hh"

using json = nlohmann::json;

namespace AstraSim {

// The GeneralComplexTopology class is needed when native algorithm is chosen.
// When *initializing* GeneralComplexTopology, we need to provide only the native algorithms.
// Therefore, set some enumerations to define bypassing rules.
enum class BypassRule {
    NO_BYPASS = 0,
    BYPASS_ALL_CUSTOM, // Bypass priority 1 and 2. below. Look only at native algorithm.
};

/*
 * CollectiveImplLookup decides which algorithm to use for the given collective operation.
 * We use the algorithms in the following order:
 * 1. If the user specifies which custom algorithm to use for a specific operation, use it.
 * 2. If the user specifies a custom algorithm to use for all operations of that collective (e.g. all AllReduce, etc.), use it.
 * 3. Finally, use the native algorithm for that collective specified by the user. (Traditional algorithm specification)
 */

/*
 * Known limitation (2026-09-26 review, replaces the bare upstream TODO):
 * there is no safeguard preventing a custom algorithm whose size does not
 * fit the communication group (no commgroup specified / all-rank default
 * group / a smaller group alike). This is upstream behavior kept as-is.
 */
class CollectiveImplLookup {
    public:
        CollectiveImplLookup(int rank_);
        ~CollectiveImplLookup();

        void setup_collective_impl_from_config(json j);

        // Notes on `workload_node_id`: Ctrl+F for [operation specific custom collective].
        std::vector<CollectiveImpl*> get_collective_impl(
            ComType comm_type,
            uint64_t workload_node_id,
            BypassRule bypass_rule = BypassRule::NO_BYPASS
        );

        // Phase-time re-validation of the (native implementation x collective
        // type) whitelist enforced inside get_collective_impl: the All-Reduce
        // split-phase paths in Sys (collective-optimization: localBWAware /
        // inter-dimension-scheduling: OfflineGreedy) reuse the All-Reduce
        // implementation for their Reduce_Scatter/All_Gather phases, so the
        // lookup-time check -- which only sees the original collective type
        // -- cannot catch such a rewritten pairing. Sys calls this from
        // generate_collective_phase with the *phase* type; a violating
        // pairing fails loud.
        static void ensure_native_impl_supports(ComType comm_type,
                                                CollectiveImplType impl_type);

    private:
        // Map from Chakra node id (integer) to a custom algorithm identifier/filepath.
        // Populated from YAML file. The filename is specified in the system input with the key "per-node-custom-implementation".
        std::map<int, CollectiveImpl*> per_node_custom_impl;

        // Map from Collective Type to a custom algorithm to be applied
        // on all collective of that type, unless overridden by per node configuration.
        // The filename of the Chakra ET for the specific custom algorithm is specified in the system input with the key "all-reduce-implementation-custom".
        std::map<ComType, CollectiveImpl*> global_custom_impl_per_coll;

        // Map from Collective Type to a vector of native algorithms, where
        // each element in the vector corresponds to a dimension.
        // To be applied only if no custom algorithm is specified.
        // The algorithm for each dimension is specified in the system input with the key "all-reduce-implementation".
        std::map<ComType, std::vector<CollectiveImpl*>> native_impl_per_coll_dim;
};

} // namespace AstraSim

#endif /* __COLLECTIVEIMPLLOOKUP_HH__ */