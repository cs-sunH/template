/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#include "astra-sim/system/astraccl/CollectiveImplLookup.hh"

#include <cstdlib>
#include <yaml-cpp/yaml.h>

#include "astra-sim/common/Logging.hh"

using namespace std;
using json = nlohmann::json;

namespace AstraSim {

    CollectiveImplLookup::CollectiveImplLookup(int rank_) : rank(rank_) {}

    CollectiveImplLookup::~CollectiveImplLookup() {
        for (auto& it : per_node_custom_impl) {
            delete it.second;
        }
        for (auto& it : global_custom_impl_per_coll) {
            delete it.second;
        }
        for (auto& it : native_impl_per_coll_dim) {
            for (auto* ci : it.second) {
                delete ci;
            }
        }
    }

    // Parses the window suffix of "direct<window>"/"oneDirect<window>" (e.g.
    // "direct4"). A bare "direct"/"oneDirect" requests an unlimited window,
    // which is encoded as -1. Anything else (non-numeric, trailing garbage or
    // non-positive window) is a fatal configuration error: such a window
    // would either be silently truncated by stoi or degenerate the Direct
    // implementation into a zero-packet stream that can never complete.
    int parse_direct_collective_window(const string& prefix,
                                       const string& collective_impl_str) {
        if (collective_impl_str == prefix) {
            return -1;
        }
        string window_str = collective_impl_str.substr(prefix.size());
        if (window_str.empty() ||
            window_str.find_first_not_of("0123456789") != string::npos) {
            LoggerFactory::get_logger("astraccl")
                ->critical(
                    "Cannot interpret the window of the direct collective "
                    "implementation '{}'. The window must be a positive "
                    "integer.",
                    collective_impl_str);
            exit(1);
        }
        int window = 0;
        try {
            window = stoi(window_str);
        } catch (const std::exception&) {
            LoggerFactory::get_logger("astraccl")
                ->critical(
                    "The window of the direct collective implementation '{}' "
                    "is out of range.",
                    collective_impl_str);
            exit(1);
        }
        if (window <= 0) {
            LoggerFactory::get_logger("astraccl")
                ->critical(
                    "The window of the direct collective implementation '{}' "
                    "must be a positive integer.",
                    collective_impl_str);
            exit(1);
        }
        return window;
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
            int window =
                parse_direct_collective_window("direct", collective_impl_str);
            return new DirectCollectiveImpl(CollectiveImplType::Direct, window);
        } else if (collective_impl_str.rfind("oneDirect", 0) == 0) {
            int window = parse_direct_collective_window("oneDirect",
                                                        collective_impl_str);
            return new DirectCollectiveImpl(CollectiveImplType::OneDirect,
                                            window);
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

        // 1. Parse the native collectives. These have the lowest priority.
        if (j.contains("all-reduce-implementation")) {
            vector<string> collective_impl_str_vec = j["all-reduce-implementation"];
            for (auto collective_impl_str : collective_impl_str_vec) {
                CollectiveImpl* ci =
                    generate_collective_impl_from_input(collective_impl_str);
                native_impl_per_coll_dim[ComType::All_Reduce].push_back(ci);
            }
        }
        if (j.contains("reduce-scatter-implementation")) {
            vector<string> collective_impl_str_vec =
                j["reduce-scatter-implementation"];
            for (auto collective_impl_str : collective_impl_str_vec) {
                CollectiveImpl* ci =
                    generate_collective_impl_from_input(collective_impl_str);
                native_impl_per_coll_dim[ComType::Reduce_Scatter].push_back(ci);
            }
        }
        if (j.contains("all-gather-implementation")) {
            vector<string> collective_impl_str_vec = j["all-gather-implementation"];
            for (auto collective_impl_str : collective_impl_str_vec) {
                CollectiveImpl* ci =
                    generate_collective_impl_from_input(collective_impl_str);
                native_impl_per_coll_dim[ComType::All_Gather].push_back(ci);
            }
        }
        if (j.contains("all-to-all-implementation")) {
            vector<string> collective_impl_str_vec = j["all-to-all-implementation"];
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
            return nit->second;
        }

        // Traditional native, it was okay to not define a collective in system input
        // as long as that collective was not actually present.
        // Still, GeneralLogicalTopo was utilized for all collectives.
        if (bypass_rule == BypassRule::BYPASS_ALL_CUSTOM) {
            return {};
        }

        // If no implementation is found, throw an error.
        throw std::runtime_error("No collective implementation found for the given type {" + std::to_string(static_cast<int>(comm_type)) + "} and node ID {" + std::to_string(workload_node_id) + "}.");
    }

} // namespace AstraSim
