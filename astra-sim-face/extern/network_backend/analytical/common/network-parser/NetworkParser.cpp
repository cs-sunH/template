/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#include "common/NetworkParser.h"
#include <cassert>
#include <iostream>

using namespace NetworkAnalytical;

NetworkParser::NetworkParser(const std::string& path) noexcept
    : dims_count(-1),
      fluid_max_active_flows(20'000),
      fluid_max_route_memberships(500'000),
      progress_report_event_interval(100'000) {
    // initialize values
    npus_count_per_dim = {};
    bandwidth_per_dim = {};
    latency_per_dim = {};
    topology_per_dim = {};

    try {
        // load network config file
        const auto network_config = YAML::LoadFile(path);

        // parse network configs
        parse_network_config_yml(network_config);
    } catch (const YAML::BadFile& e) {
        // loading network config file failed
        std::cerr << "[Error] (network/analytical) " << e.what() << std::endl;
        std::exit(-1);
    }
}

int NetworkParser::get_dims_count() const noexcept {
    assert(dims_count > 0);

    return dims_count;
}

std::vector<int> NetworkParser::get_npus_counts_per_dim() const noexcept {
    assert(dims_count > 0);
    assert(npus_count_per_dim.size() == dims_count);

    return npus_count_per_dim;
}

std::vector<Bandwidth> NetworkParser::get_bandwidths_per_dim() const noexcept {
    assert(dims_count > 0);
    assert(bandwidth_per_dim.size() == dims_count);

    return bandwidth_per_dim;
}

std::vector<Latency> NetworkParser::get_latencies_per_dim() const noexcept {
    assert(dims_count > 0);
    assert(latency_per_dim.size() == dims_count);

    return latency_per_dim;
}

std::vector<TopologyBuildingBlock> NetworkParser::get_topologies_per_dim() const noexcept {
    assert(dims_count > 0);
    assert(topology_per_dim.size() == dims_count);

    return topology_per_dim;
}

uint64_t NetworkParser::get_fluid_max_active_flows() const noexcept {
    return fluid_max_active_flows;
}

uint64_t NetworkParser::get_fluid_max_route_memberships() const noexcept {
    return fluid_max_route_memberships;
}

uint64_t NetworkParser::get_progress_report_event_interval() const noexcept {
    return progress_report_event_interval;
}

void NetworkParser::parse_network_config_yml(const YAML::Node& network_config) noexcept {
    // parse topology_per_dim
    const auto topology_names = parse_vector<std::string>(network_config["topology"]);
    for (const auto& topology_name : topology_names) {
        const auto topology_dim = NetworkParser::parse_topology_name(topology_name);
        topology_per_dim.push_back(topology_dim);
    }

    // set dims_count
    dims_count = static_cast<int>(topology_per_dim.size());

    // parse vector values
    npus_count_per_dim = parse_vector<int>(network_config["npus_count"]);
    bandwidth_per_dim = parse_vector<Bandwidth>(network_config["bandwidth"]);
    latency_per_dim = parse_vector<Latency>(network_config["latency"]);

    if (network_config["transmission-mode"]) {
        auto mode = std::string();
        try {
            mode = network_config["transmission-mode"].as<std::string>();
        } catch (const YAML::BadConversion& e) {
            std::cerr << "[Error] (network/analytical) " << e.what() << std::endl;
            std::exit(-1);
        }
        if (mode != "fluid-pipelined") {
            std::cerr << "[Error] (network/analytical) congestion-aware only supports "
                      << "transmission-mode: fluid-pipelined; got: " << mode << std::endl;
            std::exit(-1);
        }
    }

    try {
        if (network_config["fluid-max-active-flows"]) {
            fluid_max_active_flows = network_config["fluid-max-active-flows"].as<uint64_t>();
        }
        if (network_config["fluid-max-route-memberships"]) {
            fluid_max_route_memberships = network_config["fluid-max-route-memberships"].as<uint64_t>();
        }
        if (network_config["progress-report-event-interval"]) {
            progress_report_event_interval = network_config["progress-report-event-interval"].as<uint64_t>();
        }
    } catch (const YAML::BadConversion& e) {
        std::cerr << "[Error] (network/analytical) " << e.what() << std::endl;
        std::exit(-1);
    }

    // check the validity of the parsed network config
    check_validity();
}

TopologyBuildingBlock NetworkParser::parse_topology_name(const std::string& topology_name) noexcept {
    assert(!topology_name.empty());

    if (topology_name == "Ring") {
        return TopologyBuildingBlock::Ring;
    }

    if (topology_name == "FullyConnected") {
        return TopologyBuildingBlock::FullyConnected;
    }

    if (topology_name == "Switch") {
        return TopologyBuildingBlock::Switch;
    }

    if (topology_name == "Line" || topology_name == "Mesh") {
        return TopologyBuildingBlock::Mesh;
    }

    // shouldn't reach here
    std::cerr << "[Error] (network/analytical) " << "Topology name " << topology_name << " not supported" << std::endl;
    std::exit(-1);
}

void NetworkParser::check_validity() const noexcept {
    if (dims_count == 0) {
        std::cerr << "[Error] (network/analytical) "
                  << "topology/dimensions must define at least one dimension"
                  << std::endl;
        std::exit(-1);
    }
    if (fluid_max_active_flows == 0 || fluid_max_route_memberships == 0 ||
        progress_report_event_interval == 0) {
        std::cerr << "[Error] (network/analytical) fluid resource limits and progress interval must be positive"
                  << std::endl;
        std::exit(-1);
    }
    // dims_count should match
    if (dims_count != npus_count_per_dim.size()) {
        std::cerr << "[Error] (network/analytical) " << "length of npus_count (" << npus_count_per_dim.size()
                  << ") doesn't match with dimensions (" << dims_count << ")" << std::endl;
        std::exit(-1);
    }

    if (dims_count != bandwidth_per_dim.size()) {
        std::cerr << "[Error] (network/analytical) " << "length of bandwidth (" << bandwidth_per_dim.size()
                  << ") doesn't match with dims_count (" << dims_count << ")" << std::endl;
        std::exit(-1);
    }

    if (dims_count != latency_per_dim.size()) {
        std::cerr << "[Error] (network/analytical) " << "length of latency (" << latency_per_dim.size()
                  << ") doesn't match with dims_count (" << dims_count << ")" << std::endl;
        std::exit(-1);
    }

    // npus_count should be all positive. A size-1 dimension represents a
    // degenerate mesh axis, e.g. a 1-row x 2-column 2D mesh.
    for (const auto& npus_count : npus_count_per_dim) {
        if (npus_count < 1) {
            std::cerr << "[Error] (network/analytical) " << "npus_count (" << npus_count << ") should be positive"
                      << std::endl;
            std::exit(-1);
        }
    }

    // bandwidths should be all positive
    for (const auto& bandwidth : bandwidth_per_dim) {
        if (bandwidth <= 0) {
            std::cerr << "[Error] (network/analytical) " << "bandwidth (" << bandwidth << ") should be larger than 0"
                      << std::endl;
            std::exit(-1);
        }
    }

    // latency should be non-negative
    for (const auto& latency : latency_per_dim) {
        if (latency < 0) {
            std::cerr << "[Error] (network/analytical) " << "latency (" << latency << ") should be non-negative"
                      << std::endl;
            std::exit(-1);
        }
    }
}
