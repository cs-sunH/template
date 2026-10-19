/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#include "astra-sim/system/astraccl/CollectiveImplLookup.hh"

#include <cstdlib>
#include <stdexcept>
#include <string>
#include <yaml-cpp/yaml.h>

#include "astra-sim/common/Logging.hh"
#include "astra-sim/system/Sys.hh"

using namespace std;
using json = nlohmann::json;

namespace AstraSim {

    namespace {

        // Whitelist of (native implementation x collective type) pairings the
        // native algorithms actually implement. A native algorithm only models
        // the data movement of the types listed for it; assigned to any other
        // type it silently mis-simulates instead of crashing:
        // - Ring keeps its ring neighbor fixed at construction (Ring.cc stores
        //   curr_sender/curr_receiver once; only the AllToAll subclass rotates
        //   them per step), so as all-to-all every message would go to a single
        //   neighbor with N/2x over-sent bytes.
        // - DoubleBinaryTreeAllReduce hard-wires comType=All_Reduce and
        //   all-reduce data sizes, so any other type would run with all-reduce
        //   semantics.
        // The lookup rejects such pairings whenever it hands out the native
        // implementations (Sys.cc builds one topology per ComType eagerly at
        // construction, so an invalid pairing fails loudly at startup).
        bool native_impl_supports(ComType comm_type,
                                  CollectiveImplType impl_type) {
            switch (impl_type) {
            case CollectiveImplType::Ring:
            case CollectiveImplType::OneRing:
            case CollectiveImplType::HalvingDoubling:
            case CollectiveImplType::OneHalvingDoubling:
                return comm_type == ComType::All_Reduce ||
                       comm_type == ComType::All_Gather ||
                       comm_type == ComType::Reduce_Scatter;
            case CollectiveImplType::DoubleBinaryTree:
                return comm_type == ComType::All_Reduce;
            default:
                // direct/oneDirect (the AllToAll algorithm) and custom
                // collectives carry no native-implemented-type restriction.
                return true;
            }
        }

        std::string collective_impl_name(CollectiveImplType impl_type) {
            switch (impl_type) {
            case CollectiveImplType::Ring: return "ring";
            case CollectiveImplType::OneRing: return "oneRing";
            case CollectiveImplType::Direct: return "direct";
            case CollectiveImplType::OneDirect: return "oneDirect";
            case CollectiveImplType::DoubleBinaryTree:
                return "doubleBinaryTree";
            case CollectiveImplType::HalvingDoubling:
                return "halvingDoubling";
            case CollectiveImplType::OneHalvingDoubling:
                return "oneHalvingDoubling";
            case CollectiveImplType::CustomCollectiveImpl:
                return "custom";
            }
            return "unknown";
        }

        std::string com_type_name(ComType comm_type) {
            switch (comm_type) {
            case ComType::All_Reduce: return "all-reduce";
            case ComType::Reduce_Scatter: return "reduce-scatter";
            case ComType::All_Gather: return "all-gather";
            case ComType::All_to_All: return "all-to-all";
            case ComType::None: return "none";
            }
            return "unknown";
        }

    }  // namespace

    namespace {

        // Strict window parser for the direct/oneDirect AllToAll variants.
        // Legal spellings: the bare prefix ("direct" / "oneDirect", unbounded
        // window), prefix + "-1" (the same unbounded sentinel), or prefix +
        // "<N>" with a window N >= 1. Anything else throws at parse time
        // (startup, from setup_collective_impl_from_config) instead of the
        // old lenient read's behaviors: stoi's partial-prefix parse turned
        // "direct2x" into window 2, substr(6,5)/substr(9,5) truncated windows
        // past five digits, a non-numeric suffix escaped as an uncaught
        // std::invalid_argument, and a non-positive window (other than the
        // -1 sentinel) passed construction and only blew up at the first
        // collective inside Ring::ready with the misleading "should not
        // inject nothing!" panic (AllToAll computes parallel_reduce =
        // min(window, N-1), which injects no packets for window <= 0).
        int parse_direct_window(const string& impl_name,
                                const string& prefix) {
            if (impl_name == prefix) {
                return -1;  // unbounded-window sentinel
            }
            const string suffix = impl_name.substr(prefix.size());
            int window = 0;
            try {
                size_t pos = 0;
                window = stoi(suffix, &pos);
                if (pos != suffix.size()) {
                    throw std::invalid_argument("trailing characters");
                }
            } catch (const std::exception&) {
                throw std::invalid_argument(
                    "cannot parse direct collective variant '" + impl_name +
                    "': legal spellings are '" + prefix + "', '" + prefix +
                    "-1' (unbounded window) or '" + prefix +
                    "<N>' with a window N >= 1");
            }
            // -1 is the legal unbounded-window sentinel (same contract as
            // the bare prefix above), so it passes through here; every
            // other non-positive window is rejected.
            if (window != -1 && window <= 0) {
                throw std::invalid_argument(
                    "direct collective variant '" + impl_name +
                    "' declares window " + to_string(window) +
                    ": only -1 (unbounded window) or a positive window is "
                    "legal. A non-positive window injects no packets and "
                    "used to fail only later, at the first collective, with "
                    "a misleading panic");
            }
            return window;
        }

    }  // namespace

    void CollectiveImplLookup::ensure_native_impl_supports(
        ComType comm_type, CollectiveImplType impl_type) {
        if (native_impl_supports(comm_type, impl_type)) {
            return;
        }
        Sys::sys_panic(
            "Cannot use native collective implementation {" +
            collective_impl_name(impl_type) + "} for collective type {" +
            com_type_name(comm_type) +
            "}: that implementation does not implement this collective type "
            "and would silently mis-simulate it (ring keeps a single fixed "
            "ring neighbor; doubleBinaryTree hard-wires all-reduce "
            "semantics). Re-checked at phase construction because the "
            "All-Reduce split phases (localBWAware / OfflineGreedy) reuse "
            "the All-Reduce implementation for Reduce_Scatter/All_Gather "
            "phases, which the lookup-time whitelist cannot see.");
    }

    CollectiveImplLookup::CollectiveImplLookup(
        int /*rank_ -- parameter kept for the unchanged call sites*/) {}

    CollectiveImplLookup::~CollectiveImplLookup() {
        // Every CollectiveImpl in these maps was independently new'd by
        // setup_collective_impl_from_config and is owned solely by this
        // lookup: receivers of get_collective_impl results never delete them
        // (CollectivePlan gets clone_collective_impl copies instead), so
        // deleting here releases everything exactly once.
        for (auto& kv : per_node_custom_impl) {
            delete kv.second;
        }
        for (auto& kv : global_custom_impl_per_coll) {
            delete kv.second;
        }
        for (auto& kv : native_impl_per_coll_dim) {
            for (CollectiveImpl* ci : kv.second) {
                delete ci;
            }
        }
    }

    CollectiveImpl* generate_collective_impl_from_input(
        string collective_impl_str) {
        if (collective_impl_str == "ring") {
            return new CollectiveImpl(CollectiveImplType::Ring);
        } else if (collective_impl_str == "oneRing") {
            return new CollectiveImpl(CollectiveImplType::OneRing);
        } else if (collective_impl_str == "doubleBinaryTree") {
            return new CollectiveImpl(CollectiveImplType::DoubleBinaryTree);
        } else if (collective_impl_str.rfind("direct", 0) == 0) {
            return new DirectCollectiveImpl(
                CollectiveImplType::Direct,
                parse_direct_window(collective_impl_str, "direct"));
        } else if (collective_impl_str.rfind("oneDirect", 0) == 0) {
            return new DirectCollectiveImpl(
                CollectiveImplType::OneDirect,
                parse_direct_window(collective_impl_str, "oneDirect"));
        } else if (collective_impl_str == "halvingDoubling") {
            return new CollectiveImpl(CollectiveImplType::HalvingDoubling);
        } else if (collective_impl_str == "oneHalvingDoubling") {
            return new CollectiveImpl(CollectiveImplType::OneHalvingDoubling);
        } else {
            auto logger = LoggerFactory::get_logger("astraccl");
            logger->critical("Cannot interpret collective implementations. "
                            "Please check the collective implementations in the sys"
                            "input file");
            exit(1);
        }
    }

    CollectiveImpl* generate_custom_collective_impl(string chakra_filepath) {
        return new CustomCollectiveImpl(CollectiveImplType::CustomCollectiveImpl,
                                        chakra_filepath);
    }

    std::map<int, std::string> parse_per_node_yaml_file(string yaml_filepath) {
        YAML::Node root;
        try {
            root = YAML::LoadFile(yaml_filepath);
        } catch (const YAML::BadFile& e) {
            throw std::runtime_error("Failed to open YAML file: " + yaml_filepath);
        } catch (const YAML::ParserException& e) {
            throw std::runtime_error(std::string("YAML parse error: ") + e.what());
        }

        if (!root || !root.IsMap()) {
            throw std::runtime_error("Top-level YAML must be a mapping of int -> string.");
        }

        std::map<int, std::string> result;
        for (const auto&kv: root) {
            if (!kv.first.IsScalar() || !kv.second.IsScalar()) {
                throw std::runtime_error("YAML mapping keys and values must be scalars.");
            }
            int node_id;
            try {
                node_id = kv.first.as<int>();
            } catch (const YAML::BadConversion& e) {
                throw std::runtime_error("YAML mapping keys must be integers.");
            }
            std::string chakra_filepath = kv.second.as<std::string>();
            result[node_id] = chakra_filepath;
        }

        return result;
    }

    void CollectiveImplLookup::setup_collective_impl_from_config(json j) {
        // This function is intentionally rolled out (multiple if statements
        // instead of for loop across colls) to provide better code clarity.

        // R6-T2 fix (2026-09-29): an explicitly empty list ("[]") registered
        // an empty per-dimension vector; the three consumer paths that
        // check .empty() rejected it, but the default whole-cluster path
        // (Sys.cc implementation_per_dimension[0]) dereferenced end() --
        // UB. An empty implementation list is a configuration error at
        // parse time, same contract as the custom-implementation arity
        // checks below. (Omitting the key entirely stays legal: an unused
        // collective type needs no implementation.)

        // 1. Parse the native collectives. These have the lowest priority.
        if (j.contains("all-reduce-implementation")) {
            vector<string> collective_impl_str_vec = j["all-reduce-implementation"];
            if (collective_impl_str_vec.empty()) {
                throw logic_error(
                    "'all-reduce-implementation' is empty; either remove the "
                    "key or list at least one implementation per dimension");
            }
            for (auto collective_impl_str : collective_impl_str_vec) {
                CollectiveImpl* ci =
                    generate_collective_impl_from_input(collective_impl_str);
                native_impl_per_coll_dim[ComType::All_Reduce].push_back(ci);
            }
        }
        if (j.contains("reduce-scatter-implementation")) {
            vector<string> collective_impl_str_vec =
                j["reduce-scatter-implementation"];
            if (collective_impl_str_vec.empty()) {
                throw logic_error(
                    "'reduce-scatter-implementation' is empty; either remove "
                    "the key or list at least one implementation per "
                    "dimension");
            }
            for (auto collective_impl_str : collective_impl_str_vec) {
                CollectiveImpl* ci =
                    generate_collective_impl_from_input(collective_impl_str);
                native_impl_per_coll_dim[ComType::Reduce_Scatter].push_back(ci);
            }
        }
        if (j.contains("all-gather-implementation")) {
            vector<string> collective_impl_str_vec = j["all-gather-implementation"];
            if (collective_impl_str_vec.empty()) {
                throw logic_error(
                    "'all-gather-implementation' is empty; either remove the "
                    "key or list at least one implementation per dimension");
            }
            for (auto collective_impl_str : collective_impl_str_vec) {
                CollectiveImpl* ci =
                    generate_collective_impl_from_input(collective_impl_str);
                native_impl_per_coll_dim[ComType::All_Gather].push_back(ci);
            }
        }
        if (j.contains("all-to-all-implementation")) {
            vector<string> collective_impl_str_vec = j["all-to-all-implementation"];
            if (collective_impl_str_vec.empty()) {
                throw logic_error(
                    "'all-to-all-implementation' is empty; either remove the "
                    "key or list at least one implementation per dimension");
            }
            for (auto collective_impl_str : collective_impl_str_vec) {
                CollectiveImpl* ci =
                    generate_collective_impl_from_input(collective_impl_str);
                native_impl_per_coll_dim[ComType::All_to_All].push_back(ci);
            }
        }

        // 2. Parse the custom collectives to be applied on all collectives, if defined.
        // These have the next highest priority.
        if (j.contains("all-to-all-implementation-custom")) {
            vector<string> chakra_filepath_str_vec =
                j["all-to-all-implementation-custom"];
            if (chakra_filepath_str_vec.size() != 1) {
                throw logic_error(
                    "There should be 1 Chakra ET only. In multi-dim collectives, "
                    "that 1 ET file covers all dimensions");
            }
            CollectiveImpl* ci =
                generate_custom_collective_impl(chakra_filepath_str_vec[0]);
            global_custom_impl_per_coll[ComType::All_to_All] = ci;
        }
        if (j.contains("all-gather-implementation-custom")) {
            vector<string> chakra_filepath_str_vec =
                j["all-gather-implementation-custom"];
            if (chakra_filepath_str_vec.size() != 1) {
                throw logic_error(
                    "There should be 1 Chakra ET only. In multi-dim collectives, "
                    "that 1 ET file covers all dimensions");
            }
            CollectiveImpl* ci =
                generate_custom_collective_impl(chakra_filepath_str_vec[0]);
            global_custom_impl_per_coll[ComType::All_Gather] = ci;
        }
        if (j.contains("reduce-scatter-implementation-custom")) {
            vector<string> chakra_filepath_str_vec =
                j["reduce-scatter-implementation-custom"];
            if (chakra_filepath_str_vec.size() != 1) {
                throw logic_error(
                    "There should be 1 Chakra ET only. In multi-dim collectives, "
                    "that 1 ET file covers all dimensions");
            }
            CollectiveImpl* ci =
                generate_custom_collective_impl(chakra_filepath_str_vec[0]);
            global_custom_impl_per_coll[ComType::Reduce_Scatter] = ci;
        }
        if (j.contains("all-reduce-implementation-custom")) {
            vector<string> chakra_filepath_str_vec =
                j["all-reduce-implementation-custom"];
            if (chakra_filepath_str_vec.size() != 1) {
                throw logic_error(
                    "There should be 1 Chakra ET only. In multi-dim collectives, "
                    "that 1 ET file covers all dimensions");
            }
            CollectiveImpl* ci =
                generate_custom_collective_impl(chakra_filepath_str_vec[0]);
            global_custom_impl_per_coll[ComType::All_Reduce] = ci;
        }

        // 3. Finally, parse the list of per-chakra-node custom collective algorithm.
        // These have the highest priority.
        if (j.contains("per-node-custom-implementation")) {
            string per_node_custom_impl_filepath = j["per-node-custom-implementation"];
            std::map<int, std::string> per_node_custom_impl_filename =
                parse_per_node_yaml_file(per_node_custom_impl_filepath);

            for (auto const& [node_id, chakra_filepath] : per_node_custom_impl_filename) {
                CollectiveImpl* ci =
                    generate_custom_collective_impl(chakra_filepath);
                per_node_custom_impl[node_id] = ci;
            }
        }
    }

    std::vector<CollectiveImpl*> CollectiveImplLookup::get_collective_impl(
        ComType comm_type,
        uint64_t workload_node_id,
        BypassRule bypass_rule) {

        // Check if there is a per-node custom implementation first.
        if (bypass_rule != BypassRule::BYPASS_ALL_CUSTOM) {
            auto it = per_node_custom_impl.find(static_cast<int>(workload_node_id));
            if (it != per_node_custom_impl.end()) {
                return std::vector<CollectiveImpl*>{it->second};
            }
        }

        // Next, check if there is a global custom implementation for this collective type.
        if (bypass_rule != BypassRule::BYPASS_ALL_CUSTOM) {
            auto git = global_custom_impl_per_coll.find(comm_type);
            if (git != global_custom_impl_per_coll.end()) {
                return std::vector<CollectiveImpl*>{git->second};
            }
        }

        // Finally, return the native implementation per dimension.
        auto nit = native_impl_per_coll_dim.find(comm_type);
        if (nit != native_impl_per_coll_dim.end()) {
            for (CollectiveImpl* ci : nit->second) {
                if (!native_impl_supports(comm_type, ci->type)) {
                    throw std::runtime_error(
                        "Cannot use native collective implementation {" +
                        collective_impl_name(ci->type) +
                        "} for collective type {" + com_type_name(comm_type) +
                        "}: that implementation does not implement this "
                        "collective type and would silently mis-simulate it "
                        "(ring keeps a single fixed ring neighbor; "
                        "doubleBinaryTree hard-wires all-reduce semantics). "
                        "Please use a supported combination, e.g. "
                        "direct/oneDirect or a custom implementation for "
                        "all-to-all, and doubleBinaryTree for all-reduce "
                        "only.");
                }
            }
            return nit->second;
        }

        // Traditional native, it was okay to not define a collective in system input
        // as long as that collective was not actually present.
        // Still, GeneralComplexTopology was utilized for all collectives.
        if (bypass_rule == BypassRule::BYPASS_ALL_CUSTOM) {
            return {};
        }

        // If no implementation is found, throw an error.
        throw std::runtime_error("No collective implementation found for the given type {" + std::to_string(static_cast<int>(comm_type)) + "} and node ID {" + std::to_string(workload_node_id) + "}.");
    }

} // namespace AstraSim
