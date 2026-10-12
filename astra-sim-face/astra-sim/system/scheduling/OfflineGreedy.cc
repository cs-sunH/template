/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#include "astra-sim/system/scheduling/OfflineGreedy.hh"
#include "astra-sim/common/Logging.hh"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <utility>

using namespace AstraSim;

std::map<OfflineGreedyScheduleKey, OfflineGreedyScheduleEntry>
    OfflineGreedyScheduleJournal::entries;

bool OfflineGreedyScheduleJournal::consume(
    const OfflineGreedyScheduleKey& key,
    size_t expected_consumers,
    std::vector<int>& schedule,
    uint64_t& chunk_size) {
    if (expected_consumers == 0) {
        throw std::invalid_argument(
            "OfflineGreedy requires at least one schedule consumer");
    }
    auto it = entries.find(key);
    if (it == entries.end()) {
        return false;
    }
    OfflineGreedyScheduleEntry& entry = it->second;
    if (entry.expected_consumers != expected_consumers) {
        throw std::runtime_error(
            "OfflineGreedy communicator consumer count changed for a live "
            "schedule");
    }
    if (entry.consumed >= entry.expected_consumers) {
        throw std::runtime_error(
            "OfflineGreedy schedule was consumed more than once per rank");
    }
    schedule = entry.schedule;
    chunk_size = entry.chunk_size;
    ++entry.consumed;
    if (entry.consumed == entry.expected_consumers) {
        entries.erase(it);
    }
    return true;
}

void OfflineGreedyScheduleJournal::publish(
    const OfflineGreedyScheduleKey& key,
    std::vector<int> schedule,
    uint64_t chunk_size,
    size_t expected_consumers) {
    if (expected_consumers == 0) {
        throw std::invalid_argument(
            "OfflineGreedy requires at least one schedule consumer");
    }
    if (entries.find(key) != entries.end()) {
        throw std::runtime_error(
            "OfflineGreedy attempted to replace a live schedule");
    }
    // The producer is the first real consumer even when a nonzero-rank Sys
    // delegates schedule calculation to rank zero.  A singleton communicator
    // therefore needs no journal entry at all.
    if (expected_consumers == 1) {
        return;
    }
    entries.emplace(
        key, OfflineGreedyScheduleEntry{std::move(schedule), chunk_size, 1,
                                        expected_consumers});
}

DimElapsedTime::DimElapsedTime(int dim_num) {
    this->dim_num = dim_num;
    this->elapsed_time = 0;
}
OfflineGreedy::OfflineGreedy(Sys* sys) {
    this->sys = sys;
    this->dim_size = sys->physical_dims;
    this->dim_BW.resize(this->dim_size.size());
    for (uint64_t i = 0; i < this->dim_size.size(); i++) {
        this->dim_BW[i] = sys->comm_NI->get_BW_at_dimension(i);
        this->dim_elapsed_time.push_back(DimElapsedTime(i));
    }
    // dim_BW[0] is the normalization divisor of every chunk-size/load
    // formula below and the other entries are its numerators; a backend that
    // reports no bandwidth (the AstraNetworkAPI default is -1) would
    // otherwise silently poison every scheduled chunk. Fail closed instead.
    for (uint64_t i = 0; i < this->dim_BW.size(); i++) {
        if (this->dim_BW[i] <= 0) {
            Sys::sys_panic("OfflineGreedy requires a positive bandwidth for "
                           "every dimension (dim_BW[0] normalizes the chunk "
                           "sizes), but the network backend reported " +
                           std::to_string(this->dim_BW[i]) + " for dimension " +
                           std::to_string(i));
        }
    }
    if (sys->id == 0) {
        auto logger = LoggerFactory::get_logger("themis");
        logger->info("Themis is configured with the following parameters:");
        std::stringstream buffer;
        buffer << "Dim size: ";
        for (uint64_t i = 0; i < this->dim_size.size(); i++) {
            buffer << this->dim_size[i] << ", ";
        }
        logger->info(buffer.str());
        buffer.str("");
        buffer << "BW per dim: ";
        for (uint64_t i = 0; i < this->dim_BW.size(); i++) {
            buffer << this->dim_BW[i] << ", ";
        }
        logger->info(buffer.str());
    }
}
uint64_t OfflineGreedy::get_chunk_size_from_elapsed_time(double elapsed_time,
                                                         DimElapsedTime dim,
                                                         ComType comm_type) {
    if (comm_type == ComType::Reduce_Scatter) {
        uint64_t result = ((elapsed_time * (dim_BW[dim.dim_num] / dim_BW[0])) /
                           (((double)(dim_size[dim.dim_num] - 1)) /
                            (dim_size[dim.dim_num]))) *
                          1048576;
        return result;
    } else {
        uint64_t result = ((elapsed_time * (dim_BW[dim.dim_num] / dim_BW[0])) /
                           (((double)(dim_size[dim.dim_num] - 1)) / (1))) *
                          1048576;
        return result;
    }
}
void OfflineGreedy::reset_loads() {
    int i = 0;
    for (auto& dim : dim_elapsed_time) {
        dim.elapsed_time = 0;
        dim.dim_num = i;
        i++;
    }
}
std::vector<int> OfflineGreedy::get_chunk_scheduling(
    int64_t communicator_namespace,
    long long chunk_id,
    size_t expected_consumers,
    uint64_t& remaining_data_size,
    uint64_t recommended_chunk_size,
    std::vector<bool>& dimensions_involved,
    InterDimensionScheduling inter_dim_scheduling,
    ComType comm_type) {
    // Every dimensions_involved access below is indexed by a physical
    // dimension (dim.dim_num over dim_elapsed_time, which is sized from
    // sys->physical_dims). An involved_dim attribute present but shorter
    // than the physical dimension count (the producer only pads four true
    // entries when the attribute is absent) would read the vector<bool> out
    // of bounds; fail loud with the actual length instead.
    if (dimensions_involved.size() < dim_elapsed_time.size()) {
        Sys::sys_panic("dimensions_involved has " +
                       std::to_string(dimensions_involved.size()) +
                       " entries but OfflineGreedy schedules " +
                       std::to_string(dim_elapsed_time.size()) +
                       " physical dimensions");
    }
    const OfflineGreedyScheduleKey schedule_key{communicator_namespace,
                                                 chunk_id};
    std::vector<int> cached_schedule;
    uint64_t cached_chunk_size = 0;
    if (OfflineGreedyScheduleJournal::consume(
            schedule_key, expected_consumers, cached_schedule,
            cached_chunk_size)) {
        if (cached_chunk_size > remaining_data_size) {
            throw std::runtime_error(
                "OfflineGreedy cached chunk exceeds remaining collective data");
        }
        remaining_data_size -= cached_chunk_size;
        return cached_schedule;
    }
    if (sys->id != 0) {
        // (2026-10-01, mirrored from the wscllm sister repo) the cross-rank
        // delegation below used to be a bare dereference; a null rank-0 slot
        // (teardown ordering, inconsistent per-rank construction) crashed
        // with no diagnosis. Same fail-closed guard as Sys.cc's
        // ask_for_schedule null handling.
        if (sys->all_sys.empty() || sys->all_sys[0] == nullptr ||
            sys->all_sys[0]->offline_greedy == nullptr) {
            Sys::sys_panic(
                "rank " + std::to_string(sys->id) +
                " has inter-dimension-scheduling=OfflineGreedy but rank 0 "
                "does not; the cross-rank schedule delegation requires the "
                "same inter-dimension-scheduling on every rank");
        }
        return sys->all_sys[0]->offline_greedy->get_chunk_scheduling(
            communicator_namespace, chunk_id, expected_consumers,
            remaining_data_size, recommended_chunk_size, dimensions_involved,
            inter_dim_scheduling, comm_type);
    } else {
        if (comm_type == ComType::All_Reduce) {
            comm_type = ComType::Reduce_Scatter;
        }
        std::sort(dim_elapsed_time.begin(), dim_elapsed_time.end());
        if (comm_type == ComType::All_Gather) {
            std::reverse(dim_elapsed_time.begin(), dim_elapsed_time.end());
        }
        std::vector<int> result;
        uint64_t chunk_size = recommended_chunk_size;
        uint64_t scheduled_chunk_size = 0;
        bool chunk_size_calculated = false;
        if (inter_dim_scheduling == InterDimensionScheduling::OfflineGreedy) {
            scheduled_chunk_size =
                std::min(remaining_data_size, chunk_size);
            remaining_data_size -= scheduled_chunk_size;
            // The clamped value is what this chunk actually carries, so every
            // load accumulation below (both the balanced-dimension branch and
            // the tail loop) must use it; accounting with the unclamped
            // recommended size overestimated the last, partially-filled
            // chunk and skewed the greedy ordering. Only the load bookkeeping
            // is affected -- data volume and communication timing use
            // remaining_data_size/scheduled_chunk_size, not chunk_size.
            chunk_size = scheduled_chunk_size;
        }
        int dim_elapsed_time_pointer = -1;
        for (auto& dim : dim_elapsed_time) {
            dim_elapsed_time_pointer++;
            if (!dimensions_involved[dim.dim_num] ||
                dim_size[dim.dim_num] == 1) {
                result.push_back(dim.dim_num);
                continue;
            } else if (inter_dim_scheduling ==
                           InterDimensionScheduling::OfflineGreedy &&
                       !chunk_size_calculated) {
                chunk_size_calculated = true;
                uint64_t diff_size = 0;
                if (comm_type == ComType::Reduce_Scatter) {
                    double load_difference =
                        fabs(dim_elapsed_time.back().elapsed_time -
                             dim.elapsed_time);
                    diff_size = get_chunk_size_from_elapsed_time(
                        load_difference, dim, ComType::Reduce_Scatter);
                } else {
                    int lastIndex = dim_elapsed_time.size() - 1;
                    while (!dimensions_involved[dim_elapsed_time[lastIndex]
                                                    .dim_num] ||
                           dim_size[dim_elapsed_time[lastIndex].dim_num] == 1) {
                        lastIndex--;
                    }
                    double load_difference =
                        fabs(dim_elapsed_time[lastIndex].elapsed_time -
                             dim.elapsed_time);
                    diff_size = get_chunk_size_from_elapsed_time(
                        load_difference, dim_elapsed_time[lastIndex],
                        ComType::All_Gather);
                    lastIndex--;
                    while (dim_elapsed_time_pointer <= lastIndex) {
                        if (dimensions_involved[dim_elapsed_time[lastIndex]
                                                    .dim_num] &&
                            dim_size[dim_elapsed_time[lastIndex].dim_num] > 1) {
                            diff_size /=
                                dim_size[dim_elapsed_time[lastIndex].dim_num];
                        }
                        lastIndex--;
                    }
                }
                if (diff_size < (recommended_chunk_size / 16)) {
                    result.resize(dim_elapsed_time.size());
                    std::iota(std::begin(result), std::end(result), 0);
                    std::vector<DimElapsedTime> myReordered;
                    myReordered.resize(dim_elapsed_time.size(),
                                       dim_elapsed_time[0]);
                    for (uint64_t myDim = 0; myDim < dim_elapsed_time.size();
                         myDim++) {
                        for (uint64_t searchDim = 0;
                             searchDim < dim_elapsed_time.size(); searchDim++) {
                            if (dim_elapsed_time[searchDim].dim_num ==
                                static_cast<uint64_t>(myDim)) {
                                myReordered[myDim] =
                                    dim_elapsed_time[searchDim];
                                break;
                            }
                        }
                    }
                    dim_elapsed_time = myReordered;
                    if (comm_type == ComType::All_Gather) {
                        std::reverse(dim_elapsed_time.begin(),
                                     dim_elapsed_time.end());
                    }
                    for (uint64_t myDim = 0; myDim < dim_elapsed_time.size();
                         myDim++) {
                        // The All-Gather path reverses dim_elapsed_time above,
                        // so the loop position (myDim) no longer equals the
                        // physical dimension. Index the physical tables
                        // (dimensions_involved/dim_size/dim_BW) through
                        // dim_num; only the load accumulation stays keyed by
                        // the loop position.
                        const uint64_t phys = dim_elapsed_time[myDim].dim_num;
                        if (!dimensions_involved[phys] ||
                            dim_size[phys] == 1) {
                            continue;
                        }
                        if (comm_type == ComType::Reduce_Scatter) {
                            dim_elapsed_time[myDim].elapsed_time +=
                                ((((double)chunk_size) / 1048576) *
                                 (((double)(dim_size[phys] - 1)) /
                                  (dim_size[phys]))) /
                                (dim_BW[phys] / dim_BW[0]);
                            chunk_size /= dim_size[phys];
                        } else {
                            dim_elapsed_time[myDim].elapsed_time +=
                                ((((double)chunk_size) / 1048576) *
                                 (((double)(dim_size[phys] - 1)))) /
                                (dim_BW[phys] / dim_BW[0]);
                            chunk_size *= dim_size[phys];
                        }
                    }
                    OfflineGreedyScheduleJournal::publish(
                        schedule_key, result, scheduled_chunk_size,
                        expected_consumers);
                    return result;
                }
            }
            result.push_back(dim.dim_num);
            if (comm_type == ComType::Reduce_Scatter) {
                dim.elapsed_time += ((((double)chunk_size) / 1048576) *
                                     (((double)(dim_size[dim.dim_num] - 1)) /
                                      (dim_size[dim.dim_num]))) /
                                    (dim_BW[dim.dim_num] / dim_BW[0]);
                chunk_size /= dim_size[dim.dim_num];
            } else {
                dim.elapsed_time += ((((double)chunk_size) / 1048576) *
                                     (((double)(dim_size[dim.dim_num] - 1)))) /
                                    (dim_BW[dim.dim_num] / dim_BW[0]);
                chunk_size *= dim_size[dim.dim_num];
            }
        }
        OfflineGreedyScheduleJournal::publish(
            schedule_key, result, scheduled_chunk_size, expected_consumers);
        return result;
    }
}
