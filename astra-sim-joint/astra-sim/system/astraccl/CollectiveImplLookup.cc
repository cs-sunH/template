/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#include "astra-sim/system/astraccl/CollectiveImplLookup.hh"

#include <cstdlib>

#include "astra-sim/common/Logging.hh"

using namespace std;
using json = nlohmann::json;

namespace AstraSim {

    // The historic `rank` member this constructor used to initialize had no
    // reader anywhere; the parameter is kept because Sys.cc still passes the
    // owning rank at construction.
    CollectiveImplLookup::CollectiveImplLookup(int /*rank_*/) {}

    CollectiveImplLookup::~CollectiveImplLookup() {
        for (const auto& type_and_impls : native_impl_per_coll_dim) {
            for (auto* impl : type_and_impls.second) {
                delete impl;
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
            int window = -1;
            if (collective_impl_str != "direct") {
                window = stoi(collective_impl_str.substr(6, 5));
            }
            return new DirectCollectiveImpl(CollectiveImplType::Direct, window);
        } else if (collective_impl_str.rfind("oneDirect", 0) == 0) {
            int window = -1;
            if (collective_impl_str != "oneDirect") {
                window = stoi(collective_impl_str.substr(9, 5));
            }
            return new DirectCollectiveImpl(CollectiveImplType::OneDirect, window);
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

    // The custom collectives / ET feeder path (per-node and global custom
    // implementations) is retired in this repository: the online entry
    // never consumes a workload ET for collectives, and Sys.cc's known-key
    // gate already lists the retired keys with this rejection as their
    // documented handler. Any retired custom key present in the system
    // JSON must fail closed here with an explicit retirement message
    // instead of being silently half-parsed (2026-10-01 audit: the
    // rejection this file's callers documented never existed here, so the
    // keys silently re-enabled the retired path -- fail-open).
    void reject_retired_custom_key(const string& key) {
        auto logger = LoggerFactory::get_logger("astraccl");
        logger->critical(
            "custom collectives are retired in this repository: system "
            "input key '" + key + "' is no longer supported. Remove it and "
            "use the native implementation keys "
            "(all-reduce/all-gather/reduce-scatter/all-to-all-implementation) "
            "instead.");
        exit(1);
    }

    void CollectiveImplLookup::setup_collective_impl_from_config(json j) {
        // This function is intentionally rolled out (multiple if statements
        // instead of for loop across colls) to provide better code clarity.

        // 0. Fail closed on the retired custom-configuration keys (checked
        // before anything else so a stray custom key is reported as
        // retired, never half-honored).
        if (j.contains("all-reduce-implementation-custom")) {
            reject_retired_custom_key("all-reduce-implementation-custom");
        }
        if (j.contains("all-gather-implementation-custom")) {
            reject_retired_custom_key("all-gather-implementation-custom");
        }
        if (j.contains("reduce-scatter-implementation-custom")) {
            reject_retired_custom_key("reduce-scatter-implementation-custom");
        }
        if (j.contains("all-to-all-implementation-custom")) {
            reject_retired_custom_key("all-to-all-implementation-custom");
        }
        if (j.contains("per-node-custom-implementation")) {
            reject_retired_custom_key("per-node-custom-implementation");
        }

        // 1. Parse the native collectives. The empty-array check sits BEFORE
        // the loop: inside the loop body it was unreachable (a for over an
        // empty vector never evaluates the body), so an explicitly empty
        // list fell through to the generic "No collective implementation
        // found" error at first use instead of this dedicated message.
        if (j.contains("all-reduce-implementation")) {
            vector<string> collective_impl_str_vec = j["all-reduce-implementation"];
            if (collective_impl_str_vec.empty()) {
                throw logic_error(
                    "'all-reduce-implementation' is empty; either remove the key or list at "
                    "least one implementation per dimension");
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
                    "'reduce-scatter-implementation' is empty; either remove the key or list at "
                    "least one implementation per dimension");
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
                    "'all-gather-implementation' is empty; either remove the key or list at "
                    "least one implementation per dimension");
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
                    "'all-to-all-implementation' is empty; either remove the key or list at "
                    "least one implementation per dimension");
            }
            for (auto collective_impl_str : collective_impl_str_vec) {
                CollectiveImpl* ci =
                    generate_collective_impl_from_input(collective_impl_str);
                native_impl_per_coll_dim[ComType::All_to_All].push_back(ci);
            }
        }
    }

    std::vector<CollectiveImpl*> CollectiveImplLookup::get_collective_impl(
        ComType comm_type,
        uint64_t workload_node_id,
        BypassRule bypass_rule) {

        // Return the native implementation per dimension. (The per-node and
        // global custom tiers this lookup used to consult first were
        // removed with the retired custom path; workload_node_id is
        // retained in the signature for call-site compatibility.)
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
