/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#include "astra-sim/system/astraccl/CollectiveImplLookup.hh"

#include <cstdlib>
#include <stdexcept>
#include <string>

#include "astra-sim/common/Logging.hh"

using namespace std;
using json = nlohmann::json;

namespace AstraSim {

    namespace {
    // Parse the numeric window suffix of a "direct<window>"/"oneDirect<window>"
    // implementation string. A fixed-width substr() silently truncates wider
    // windows and an uncaught stoi() exception would terminate the process
    // outright, so parse the full remainder and fail through the unified
    // critical+exit path instead.
    int parse_window_suffix(const string& impl_str, size_t prefix_len) {
        try {
            size_t pos = 0;
            int window = stoi(impl_str.substr(prefix_len), &pos);
            if (pos != impl_str.size() - prefix_len) {
                throw std::invalid_argument("trailing characters");
            }
            return window;
        } catch (const std::exception&) {
            auto logger = LoggerFactory::get_logger("astraccl");
            logger->critical("Cannot interpret collective implementations. "
                            "Please check the collective implementations in the sys"
                            "input file");
            exit(1);
        }
    }

    // Parse one native implementation list from the system JSON. An
    // adversarial value (a non-array, or an array holding non-strings) makes
    // nlohmann throw type_error; the throw must not escape into the Sys
    // constructor as a bare terminate, so fail closed here with the same
    // critical+exit standard as parse_window_suffix.
    vector<string> parse_impl_list(const json& j, const string& key) {
        try {
            return j.at(key).get<vector<string>>();
        } catch (const std::exception&) {
            auto logger = LoggerFactory::get_logger("astraccl");
            logger->critical("System input key '" + key +
                             "' must be an array of implementation strings "
                             "(e.g. [\"ring\", \"ring\"]).");
            exit(1);
        }
    }

    // The custom collectives / ET feeder path (per-node and global custom
    // implementations driving the retired feeder_v3 ETFeeder) was removed
    // together with the offline static arm (2026-09-27). Any retired custom
    // key present in the system JSON must fail closed here with an explicit
    // retirement message instead of being silently ignored or half-parsed.
    void reject_retired_custom_key(const string& key) {
        auto logger = LoggerFactory::get_logger("astraccl");
        logger->critical(
            "custom collectives are retired in this repository (the offline "
            "static ET path was removed with ETFeederGraphSource): system "
            "input key '" + key + "' is no longer supported. Remove it and "
            "use the native implementation keys "
            "(all-reduce/all-gather/reduce-scatter/all-to-all-implementation) "
            "instead.");
        exit(1);
    }

    }  // namespace

    // The historic `rank` member this constructor used to initialize had no
    // reader anywhere; the parameter is kept because Sys.cc still passes the
    // owning rank at construction.
    CollectiveImplLookup::CollectiveImplLookup(int /*rank_*/) {}

    CollectiveImplLookup::~CollectiveImplLookup() {
        for (auto& type_and_impls : native_impl_per_coll_dim) {
            for (auto* impl : type_and_impls.second) {
                delete impl;
            }
        }
    }

    CollectiveImpl* generate_collective_impl_from_input(
        string collective_impl_str) {
        // A direct window of 0 (or any negative value) would zero
        // parallel_reduce: no packet would ever be inserted and the stream
        // would wait forever with no diagnosis. Reject it at parse time.
        // (2026-10-01) The former `window != -1` escape hatch let an
        // EXPLICIT "direct-1"/"oneDirect-1" literal parse to -1 and ride the
        // auto-window path; the -1 sentinel is only reachable through the
        // omitted-suffix arms above, which never call this rejector -- so
        // every parsed suffix can require a strictly positive window.
        const auto reject_bad_window = [](int window) {
            if (window <= 0) {
                auto logger = LoggerFactory::get_logger("astraccl");
                logger->critical(
                    "direct/oneDirect collective window must be a positive "
                    "integer (or omitted for the automatic n-1 window); got " +
                    std::to_string(window));
                exit(1);
            }
        };
        if (collective_impl_str == "ring") {
            return new CollectiveImpl(CollectiveImplType::Ring);
        } else if (collective_impl_str == "oneRing") {
            return new CollectiveImpl(CollectiveImplType::OneRing);
        } else if (collective_impl_str == "doubleBinaryTree") {
            return new CollectiveImpl(CollectiveImplType::DoubleBinaryTree);
        } else if (collective_impl_str.rfind("direct", 0) == 0) {
            int window = -1;
            if (collective_impl_str != "direct") {
                window = parse_window_suffix(collective_impl_str, 6);
                reject_bad_window(window);
            }
            return new DirectCollectiveImpl(CollectiveImplType::Direct, window);
        } else if (collective_impl_str.rfind("oneDirect", 0) == 0) {
            int window = -1;
            if (collective_impl_str != "oneDirect") {
                window = parse_window_suffix(collective_impl_str, 9);
                reject_bad_window(window);
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

    void CollectiveImplLookup::setup_collective_impl_from_config(json j) {
        // This function is intentionally rolled out (multiple if statements
        // instead of for loop across colls) to provide better code clarity.

        // 0. Fail closed on the retired custom-configuration keys (checked
        // before anything else so a stray custom key is reported as retired,
        // never half-honored).
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

        // 1. Parse the native collectives. parse_impl_list fails closed on
        // adversarial JSON types instead of leaking a type_error.
        if (j.contains("all-reduce-implementation")) {
            vector<string> collective_impl_str_vec =
                parse_impl_list(j, "all-reduce-implementation");
            for (auto collective_impl_str : collective_impl_str_vec) {
                CollectiveImpl* ci =
                    generate_collective_impl_from_input(collective_impl_str);
                native_impl_per_coll_dim[ComType::All_Reduce].push_back(ci);
            }
        }
        if (j.contains("reduce-scatter-implementation")) {
            vector<string> collective_impl_str_vec =
                parse_impl_list(j, "reduce-scatter-implementation");
            for (auto collective_impl_str : collective_impl_str_vec) {
                CollectiveImpl* ci =
                    generate_collective_impl_from_input(collective_impl_str);
                native_impl_per_coll_dim[ComType::Reduce_Scatter].push_back(ci);
            }
        }
        if (j.contains("all-gather-implementation")) {
            vector<string> collective_impl_str_vec =
                parse_impl_list(j, "all-gather-implementation");
            for (auto collective_impl_str : collective_impl_str_vec) {
                CollectiveImpl* ci =
                    generate_collective_impl_from_input(collective_impl_str);
                native_impl_per_coll_dim[ComType::All_Gather].push_back(ci);
            }
        }
        if (j.contains("all-to-all-implementation")) {
            vector<string> collective_impl_str_vec =
                parse_impl_list(j, "all-to-all-implementation");
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

        // Return the native implementation per dimension.
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
