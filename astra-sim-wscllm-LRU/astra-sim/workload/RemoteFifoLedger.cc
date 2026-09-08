/******************************************************************************
This source code is licensed under the MIT license found in the LICENSE file in
the root directory of this source tree.
 *******************************************************************************/

#include "astra-sim/workload/RemoteFifoLedger.hh"

#include <algorithm>

namespace AstraSim {
namespace ExecutionDriven {

RemoteFifoLedger& RemoteFifoLedger::instance() {
    static RemoteFifoLedger ledger;
    return ledger;
}

void RemoteFifoLedger::set_enabled(const bool enabled) {
    enabled_ = enabled;
}

void RemoteFifoLedger::set_architecture(const std::string& architecture) {
    architecture_ = architecture;
}

void RemoteFifoLedger::record_issue(const std::size_t port_index,
                                    const int sys_id,
                                    const uint64_t tensor_size) {
    if (!enabled_) {
        return;
    }
    PortCounters& port = ports_[port_index];
    port.issued_count += 1;
    port.issued_bytes += tensor_size;
    const uint64_t in_flight =
        port.issued_count - port.completed_count;
    const uint64_t in_flight_bytes =
        port.issued_bytes - port.completed_bytes;
    if (in_flight > port.peak_in_flight_count) {
        port.peak_in_flight_count = in_flight;
    }
    if (in_flight_bytes > port.peak_in_flight_bytes) {
        port.peak_in_flight_bytes = in_flight_bytes;
    }

    // Source-rank view of this real port (deduplicated, ascending insert;
    // a rank maps to exactly one port under every supported architecture).
    std::vector<int>& ranks = port_ranks_[port_index];
    if (std::find(ranks.begin(), ranks.end(), sys_id) == ranks.end()) {
        ranks.insert(std::upper_bound(ranks.begin(), ranks.end(), sys_id),
                     sys_id);
    }

    RankAttribution& attribution = rank_attribution_[sys_id];
    attribution.port = port_index;
    attribution.issued_count += 1;
    attribution.issued_bytes += tensor_size;
}

void RemoteFifoLedger::record_completion(const std::size_t port_index,
                                         const uint64_t tensor_size) {
    if (!enabled_) {
        return;
    }
    PortCounters& port = ports_[port_index];
    port.completed_count += 1;
    port.completed_bytes += tensor_size;
}

std::vector<std::size_t> RemoteFifoLedger::active_ports() const {
    std::vector<std::size_t> result;
    result.reserve(ports_.size());
    for (const auto& [port_index, counters] : ports_) {
        (void)counters;
        result.push_back(port_index);
    }
    return result;  // std::map iterates in key order -- deterministic
}

const RemoteFifoLedger::PortCounters* RemoteFifoLedger::port(
    const std::size_t port_index) const {
    const auto it = ports_.find(port_index);
    return it == ports_.end() ? nullptr : &it->second;
}

const std::vector<int>& RemoteFifoLedger::port_ranks(
    const std::size_t port_index) const {
    static const std::vector<int> kNoRanks;
    const auto it = port_ranks_.find(port_index);
    return it == port_ranks_.end() ? kNoRanks : it->second;
}

const RemoteFifoLedger::RankAttribution* RemoteFifoLedger::rank_attribution(
    const int sys_id) const {
    const auto it = rank_attribution_.find(sys_id);
    return it == rank_attribution_.end() ? nullptr : &it->second;
}

std::vector<int> RemoteFifoLedger::attributed_ranks() const {
    std::vector<int> result;
    result.reserve(rank_attribution_.size());
    for (const auto& [sys_id, attribution] : rank_attribution_) {
        (void)attribution;
        result.push_back(sys_id);
    }
    return result;
}

const std::string& RemoteFifoLedger::architecture() const {
    return architecture_;
}

void RemoteFifoLedger::reset() {
    ports_.clear();
    port_ranks_.clear();
    rank_attribution_.clear();
}

uint64_t RemoteFifoLedger::total_issued_count() const {
    uint64_t total = 0;
    for (const auto& [port_index, counters] : ports_) {
        (void)port_index;
        total += counters.issued_count;
    }
    return total;
}

uint64_t RemoteFifoLedger::total_issued_bytes() const {
    uint64_t total = 0;
    for (const auto& [port_index, counters] : ports_) {
        (void)port_index;
        total += counters.issued_bytes;
    }
    return total;
}

uint64_t RemoteFifoLedger::total_completed_count() const {
    uint64_t total = 0;
    for (const auto& [port_index, counters] : ports_) {
        (void)port_index;
        total += counters.completed_count;
    }
    return total;
}

uint64_t RemoteFifoLedger::total_completed_bytes() const {
    uint64_t total = 0;
    for (const auto& [port_index, counters] : ports_) {
        (void)port_index;
        total += counters.completed_bytes;
    }
    return total;
}

bool RemoteFifoLedger::drained() const {
    for (const auto& [port_index, counters] : ports_) {
        (void)port_index;
        if (counters.issued_count != counters.completed_count) {
            return false;
        }
    }
    return true;
}

std::string RemoteFifoLedger::sidecar_row(const uint64_t delivery_sequence,
                                          const uint64_t tick) const {
    std::string row = "{\"delivery_sequence\": ";
    row += std::to_string(delivery_sequence);
    row += ", \"tick\": ";
    row += std::to_string(tick);
    row += ", \"memory_architecture\": \"";
    row += architecture_;
    row += "\", \"ports\": [";
    bool first_port = true;
    for (const std::size_t port_index : active_ports()) {
        const PortCounters* counters = port(port_index);
        if (counters == nullptr) {
            continue;
        }
        const uint64_t in_flight =
            counters->issued_count - counters->completed_count;
        const uint64_t in_flight_bytes =
            counters->issued_bytes - counters->completed_bytes;
        const uint64_t active = in_flight > 0 ? 1u : 0u;
        if (!first_port) {
            row += ", ";
        }
        first_port = false;
        row += "{\"port\": ";
        row += std::to_string(port_index);
        row += ", \"ranks\": [";
        bool first_rank = true;
        for (const int rank : port_ranks(port_index)) {
            if (!first_rank) {
                row += ", ";
            }
            first_rank = false;
            row += std::to_string(rank);
        }
        row += "], \"active\": ";
        row += std::to_string(active);
        row += ", \"pending\": ";
        row += std::to_string(in_flight - active);
        row += ", \"in_flight_bytes\": ";
        row += std::to_string(in_flight_bytes);
        row += ", \"issued_count\": ";
        row += std::to_string(counters->issued_count);
        row += ", \"completed_count\": ";
        row += std::to_string(counters->completed_count);
        row += ", \"issued_bytes\": ";
        row += std::to_string(counters->issued_bytes);
        row += ", \"completed_bytes\": ";
        row += std::to_string(counters->completed_bytes);
        row += "}";
    }
    row += "], \"by_rank\": [";
    bool first_rank = true;
    for (const int sys_id : attributed_ranks()) {
        const RankAttribution* attribution = rank_attribution(sys_id);
        if (attribution == nullptr) {
            continue;
        }
        if (!first_rank) {
            row += ", ";
        }
        first_rank = false;
        row += "{\"rank\": ";
        row += std::to_string(sys_id);
        row += ", \"port\": ";
        row += std::to_string(attribution->port);
        row += ", \"issued_count\": ";
        row += std::to_string(attribution->issued_count);
        row += ", \"issued_bytes\": ";
        row += std::to_string(attribution->issued_bytes);
        row += "}";
    }
    row += "]}";
    return row;
}

}  // namespace ExecutionDriven
}  // namespace AstraSim
