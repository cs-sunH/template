/******************************************************************************
Shared builder for the seven-rank P0->D1 remote-read fixture program.

Extracted verbatim from remote_read_executor_test.cc (2026-10-06) so the
cold execution fixture case generator can produce the SAME proven schema-2
batch wire without duplicating the program. Pure JSON construction: no
simulator state is touched here. The executor test keeps its own main and
assertions; this header carries only the wire builder.
******************************************************************************/

#ifndef EXECUTION_DRIVEN_TESTS_REMOTE_READ_EXECUTOR_PROGRAM_HH
#define EXECUTION_DRIVEN_TESTS_REMOTE_READ_EXECUTOR_PROGRAM_HH

#include "astra-sim/common/Logging.hh"
#include "astra-sim/system/Sys.hh"
#include "astra-sim/workload/LocalHbmBandwidthModel.hh"
#include "astra-sim/workload/MetricCollector.hh"
#include "astra-sim/workload/Workload.hh"
#include "astra-sim/workload/execution_driven/CompletionObserver.hh"
#include "astra-sim/workload/execution_driven/DecisionMailbox.hh"
#include "astra-sim/workload/execution_driven/GraphBatchCommitter.hh"
#include "astra-sim/workload/execution_driven/NodeStore.hh"
#include "astra-sim/workload/execution_driven/RemoteReadExecutor.hh"
#include "astra-sim/workload/execution_driven/TileTransportAdapter.hh"
#include "astra-sim/workload/execution_driven/WatchRegistry.hh"
#include "astra-sim/workload/execution_driven/WorkLedgerV2Publisher.hh"
#include "astra-sim/workload/execution_driven/WorkLedgerV2RuntimeSnapshot.hh"
#include "congestion_aware/CongestionAwareNetworkApi.hh"
#include "extern/remote_memory_backend/analytical/AnalyticalRemoteMemory.hh"
#include <astra-network-analytical/common/EventQueue.h>
#include <astra-network-analytical/common/NetworkParser.h>
#include <astra-network-analytical/congestion_aware/Helper.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <unistd.h>

using namespace AstraSim;
using namespace AstraSim::ExecutionDriven;
using namespace Analytical;
using namespace AstraSimAnalytical;
using namespace AstraSimAnalyticalCongestionAware;
using namespace NetworkAnalytical;
using namespace NetworkAnalyticalCongestionAware;

namespace {

constexpr const char* kRequestId = "fixture-remote-read";
constexpr const char* kTrainContext = "batch_train_i0_1";
constexpr const char* kRequestHex = "666978747572652d72656d6f74652d72656164";
constexpr int kOwnerRank = 6;
constexpr int kSourceRanks[] = {0, 5};

enum BodyId { PrefillService = 0, DecodeBodyA = 1, DecodeBodyB = 2 };

struct EventKey {
    int descriptor = 0;  // 0 = P0, 1 = D1
    uint64_t tile = 0;   // one-based within the descriptor's group
    uint64_t shard = 0;
    std::string endpoint;

    bool operator<(const EventKey& other) const {
        return std::tie(descriptor, tile, shard, endpoint) <
               std::tie(other.descriptor, other.tile, other.shard,
                        other.endpoint);
    }
};

struct ParentRef {
    bool is_body = false;
    EventKey event;
    BodyId body = PrefillService;
};

ParentRef event_parent(const EventKey& key) {
    ParentRef parent;
    parent.event = key;
    return parent;
}

ParentRef body_parent(const BodyId id) {
    ParentRef parent;
    parent.is_body = true;
    parent.body = id;
    return parent;
}

struct EndpointEvent {
    EventKey key;
    int rank = -1;
    uint64_t json_id = 0;
    uint32_t tag = 0;
    uint64_t tag_ordinal = 0;
    bool physical_anchor = false;
    std::string anchor_role;
    std::vector<ParentRef> parents;
};

struct DescriptorProgram {
    int index = 0;
    uint64_t tile_count = 0;
    uint64_t global_tile_first = 1;
    std::vector<int> source_ranks;
    std::map<EventKey, EndpointEvent> events;
    std::vector<EventKey> emission_order;
    nlohmann::json wire;
};

struct BuiltFixture {
    nlohmann::json batch;
    std::vector<DescriptorProgram> programs;
    std::map<BodyId, uint64_t> body_json_ids;
};

[[noreturn]] void fail(const std::string& message) {
    std::fprintf(stderr, "[remote_read_executor_test] FAIL: %s\n",
                 message.c_str());
    std::fflush(stderr);
    std::exit(1);
}

void expect(const bool condition, const std::string& message) {
    if (!condition) {
        fail(message);
    }
}

std::string temporary_directory() {
    const std::string pattern = "/tmp/joint_remote_read_executor_XXXXXX";
    std::vector<char> buffer(pattern.begin(), pattern.end());
    buffer.push_back('\0');
    const char* const path = ::mkdtemp(buffer.data());
    if (path == nullptr) {
        fail("mkdtemp failed");
    }
    return std::string(path);
}

void write_file(const std::string& path, const std::string& content) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) {
        fail("cannot create fixture input " + path);
    }
    out << content;
    out.close();
    if (!out.good()) {
        fail("failed writing fixture input " + path);
    }
}

void write_configs(const std::string& directory) {
    write_file(directory + "/system.json", R"({
  "scheduling-policy": "FIFO",
  "endpoint-delay": 1,
  "active-chunks-per-dimension": 1,
  "preferred-dataset-splits": 6,
  "all-reduce-implementation": ["ring", "ring"],
  "all-gather-implementation": ["ring", "ring"],
  "reduce-scatter-implementation": ["ring", "ring"],
  "all-to-all-implementation": ["ring", "ring"],
  "collective-optimization": "localBWAware",
  "roofline-enabled": 1,
  "replay-only": 0,
  "track-local-mem": 0,
  "trace-enabled": 0,
  "hbm-bandwidth-contention": 1,
  "peak-perf": 1000,
  "local-mem-bw": 6.0,
  "local-mem-latency": 10,
  "remote-mem-bw": 6.0,
  "remote-mem-latency": 10
})");
    write_file(directory + "/comm_group.json", "{}\n");
    write_file(directory + "/network.yml",
               "# seven-rank line: real multi-shard Sys/FluidScheduler path\n"
               "topology: [ Line, Line ]\n"
               "npus_count: [ 7, 1 ]\n"
               "bandwidth: [ 4000.0, 4000.0 ]\n"
               "latency: [ 5, 5 ]\n");
    write_file(directory + "/remote_memory.json", R"({
  "memory-type": "PER_NPU_MEMORY_EXPANSION",
  "remote-mem-bw": 6.0,
  "remote-mem-latency": 10,
  "npu-ids": [0, 1, 2, 3, 4, 5, 6]
})");
}

nlohmann::json contract_wire() {
    return nlohmann::json{
        {"projection_abi_version", 1},
        {"descriptor_version", 1},
        {"pipeline_protocol_version", 1},
        {"model_version", "v6-compact-window-20261005"},
        {"timer_whitelist_version", "timer-whitelist-v1-20261005"},
        {"window_tile_limit", 1},
        {"window_local_error_ppm", 10000},
        {"window_refine_margin_ppm", 20000},
        {"prediction_config_sha256", std::string(64, 'a')}};
}

nlohmann::json projection_chain(const uint64_t tile_count,
                                const std::string& suffix) {
    // The two affine registration-order progressions are disjoint and
    // preserve the original fill/consume registration order. The target side
    // owns the two-slot staging/consume gate.
    return nlohmann::json{
        {"group_ordinal", 0},
        {"rank", kOwnerRank},
        {"tile_count", tile_count},
        {"fill", {{"leg_id_prefix", "fill:r6:" + suffix + ":"},
                  {"first_index", 0},
                  {"fill_order_first", 0},
                  {"fill_order_stride", 2},
                  {"completion_order_first", nullptr},
                  {"completion_order_stride", 0},
                  {"route", nlohmann::json::array({"hbm:6"})},
                  {"bytes_per_tile", 2},
                  {"latency_ns", 0},
                  {"start_deps", nlohmann::json::array()},
                  {"gate_id", "credit:r6:" + suffix},
                  {"gate_capacity", 2}}},
        {"consume", {{"leg_id_prefix", "consume:r6:" + suffix + ":"},
                     {"first_index", 0},
                     {"consume_order_first", 1},
                     {"consume_order_stride", 2},
                     {"completion_order_first", nullptr},
                     {"completion_order_stride", 0},
                     {"kind", "fixed"},
                     {"route", nlohmann::json::array()},
                     {"group", nullptr},
                     {"latency_first_ns", 0},
                     {"latency_rest_ns", 0},
                     {"consume_deps", nlohmann::json::array()},
                     {"consume_start_deps", nlohmann::json::array()},
                     {"gate_id", "credit:r6:" + suffix},
                     {"compute_share", nullptr},
                     {"hbm_share", nullptr}}}};
}
nlohmann::json anchor_node(const EndpointEvent& event,
                           const DescriptorProgram& program) {
    const bool decode = program.index == 1;
    const std::string segment_id = std::string("pp1/") + kRequestHex + "/" +
        (decode ? "decode/1/4/6/1" : "prefill/0/4/6/0");
    const std::string request_id = decode ? kTrainContext : kRequestId;
    const std::string stage = decode ? "decode" : "prefill";
    const uint64_t generation = decode ? 1 : 0;
    return nlohmann::json{
        {"rank", event.rank}, {"id", event.json_id}, {"type", 4},
        {"name", std::string("pipeline-anchor-") + event.anchor_role + "-r" +
                     std::to_string(event.rank) + "-" + event.key.endpoint},
        {"is_cpu_op", false}, {"is_timer_op", false},
        {"inputs_values", ""}, {"request_id", request_id},
        {"stage", stage}, {"generation", generation},
        {"compute", {{"num_ops", 0}, {"tensor_size", 0},
                      {"runtime_ns", 0}, {"die_op_spec", nlohmann::json::object()},
                      {"exec_role", "remote_read_pipeline_anchor"},
                      {"pipeline_anchor", {{"segment_id", segment_id},
                                            {"role", event.anchor_role}}}}},
        {"comm", {{"bytes", 0}, {"src", 0}, {"dst", 0}, {"tag", 0},
                   {"hbm_charge", false}}},
        {"coll", {{"comm_type", 0}, {"bytes", 0}, {"priority", 0},
                   {"pg_name", ""}, {"involved_dim", nlohmann::json::array()}}}};
}

nlohmann::json body_node(const BodyId body, const uint64_t json_id) {
    const char* name = body == PrefillService ? "prefill-service"
        : (body == DecodeBodyA ? "decode-consumer-a" : "decode-consumer-b");
    const char* stage = body == PrefillService ? "prefill" : "decode";
    const uint64_t generation = body == PrefillService ? 0 : 1;
    const uint64_t runtime_ns = body == PrefillService ? 4
        : (body == DecodeBodyA ? 30 : 20);
    return nlohmann::json{
        {"rank", kOwnerRank}, {"id", json_id}, {"type", 4},
        {"name", name},
        {"is_cpu_op", false}, {"is_timer_op", false},
        {"inputs_values", ""}, {"request_id", kRequestId},
        {"stage", stage}, {"generation", generation},
        {"compute", {{"num_ops", 1}, {"tensor_size", 1},
                      {"runtime_ns", runtime_ns}, {"op_role", "compute"}}},
        {"comm", {{"bytes", 0}, {"src", 0}, {"dst", 0}, {"tag", 0},
                   {"hbm_charge", false}}},
        {"coll", {{"comm_type", 0}, {"bytes", 0}, {"priority", 0},
                   {"pg_name", ""}, {"involved_dim", nlohmann::json::array()}}}};
}

std::string segment_id(const int descriptor) {
    return std::string("pp1/") + kRequestHex +
        (descriptor == 0 ? "/prefill/0/4/6/0" : "/decode/1/4/6/1");
}

EventKey key(const int descriptor, const uint64_t tile,
             const uint64_t shard, const char* endpoint) {
    return EventKey{descriptor, tile, shard, endpoint};
}

bool is_source_endpoint(const std::string& endpoint) {
    return endpoint == "data_send" || endpoint == "fill_ack_recv" ||
           endpoint == "release_ack_recv";
}

std::string flow_family(const std::string& endpoint) {
    if (endpoint == "data_send" || endpoint == "data_recv") return "data";
    if (endpoint == "fill_ack_send" || endpoint == "fill_ack_recv") {
        return "fill_ack";
    }
    if (endpoint == "release_ack_send" || endpoint == "release_ack_recv") {
        return "release_ack";
    }
    return "body_marker";
}

EndpointEvent& add_event(DescriptorProgram& program, const uint64_t tile,
                         const uint64_t shard, const char* endpoint,
                         std::vector<ParentRef> parents = {},
                         const bool physical = false,
                         std::string anchor_role = {}) {
    EventKey event_key = key(program.index, tile, shard, endpoint);
    const int rank = endpoint == std::string("body_marker")
        ? kOwnerRank
        : (is_source_endpoint(endpoint)
               ? program.source_ranks.at(static_cast<size_t>(shard))
               : kOwnerRank);
    EndpointEvent event;
    event.key = event_key;
    event.rank = rank;
    event.physical_anchor = physical;
    event.anchor_role = std::move(anchor_role);
    event.parents = std::move(parents);
    auto inserted = program.events.emplace(event_key, std::move(event));
    if (!inserted.second) fail("duplicate endpoint key in fixture program");
    program.emission_order.push_back(event_key);
    return inserted.first->second;
}

void mark_anchor(DescriptorProgram& program, const EventKey& event_key,
                 const char* role) {
    auto it = program.events.find(event_key);
    if (it == program.events.end()) fail("anchor references missing event");
    it->second.physical_anchor = true;
    it->second.anchor_role = role;
}

uint64_t parent_json_id(const ParentRef& parent,
                        const std::vector<DescriptorProgram>& programs,
                        const std::map<BodyId, uint64_t>& body_ids) {
    if (parent.is_body) return body_ids.at(parent.body);
    return programs.at(static_cast<size_t>(parent.event.descriptor))
        .events.at(parent.event).json_id;
}

bool parent_is_ordinary(const ParentRef& parent,
                        const std::vector<DescriptorProgram>& programs) {
    return parent.is_body ||
        programs.at(static_cast<size_t>(parent.event.descriptor))
            .events.at(parent.event).physical_anchor;
}

nlohmann::json make_projection(const uint64_t tile_count,
                               const std::string& suffix) {
    return {{"version", 1},
            {"chains", nlohmann::json::array({
                 projection_chain(tile_count, suffix)})}};
}

nlohmann::json logical_ranges(const DescriptorProgram& program) {
    std::map<int, std::vector<uint64_t>> holes;
    for (const EventKey& event_key : program.emission_order) {
        const EndpointEvent& event = program.events.at(event_key);
        if (!event.physical_anchor) holes[event.rank].push_back(event.json_id);
    }
    nlohmann::json ranges = nlohmann::json::array();
    for (auto& entry : holes) {
        auto& ids = entry.second;
        std::sort(ids.begin(), ids.end());
        for (size_t i = 0; i < ids.size();) {
            size_t end = i + 1;
            while (end < ids.size() && ids[end] == ids[end - 1] + 1) ++end;
            ranges.push_back({{"rank", entry.first},
                              {"first_json_id", ids[i]},
                              {"count", end - i}, {"stride", 1},
                              {"kind", "virtual_endpoint_hole"}});
            i = end;
        }
    }
    return ranges;
}

nlohmann::json make_descriptor_wire(
    DescriptorProgram& program,
    const std::vector<DescriptorProgram>& programs,
    const std::map<BodyId, uint64_t>& body_ids) {
    const bool decode = program.index == 1;
    const std::string owner_stage = decode ? "decode" : "prefill";
    const std::string legacy_request = decode ? kTrainContext : kRequestId;
    const uint64_t legacy_generation = decode ? 1 : 0;
    const uint32_t tag_start = decode ? 12 : 1;
    uint32_t next_tag = tag_start;
    uint64_t next_ordinal = tag_start;
    using FlowKey = std::tuple<uint64_t, uint64_t, std::string>;
    std::map<FlowKey, std::pair<uint32_t, uint64_t>> allocation_by_flow;
    nlohmann::json tag_runs = nlohmann::json::array();
    for (const EventKey& event_key : program.emission_order) {
        EndpointEvent& event = program.events.at(event_key);
        if (event.key.endpoint == "body_marker") continue;
        const FlowKey flow{event.key.tile, event.key.shard,
                           flow_family(event.key.endpoint)};
        auto allocation = allocation_by_flow.find(flow);
        if (allocation == allocation_by_flow.end()) {
            const uint32_t tag = next_tag++;
            const uint64_t ordinal = next_ordinal++;
            allocation = allocation_by_flow.emplace(
                flow, std::make_pair(tag, ordinal)).first;
            tag_runs.push_back({{"allocator_scope_id", "fixture-global-tags"},
                                {"emission_ordinal", ordinal},
                                {"first_tag", tag}, {"count", 1},
                                {"group_ordinal", 0},
                                {"first_tile", event.key.tile},
                                {"tile_count", 1},
                                {"first_shard", event.key.shard},
                                {"shard_count", 1}});
        }
        event.tag = allocation->second.first;
        event.tag_ordinal = allocation->second.second;
    }

    nlohmann::json endpoint_runs = nlohmann::json::array();
    nlohmann::json dependency_runs = nlohmann::json::array();
    for (const EventKey& event_key : program.emission_order) {
        const EndpointEvent& event = program.events.at(event_key);
        nlohmann::json leg{{"endpoint", event.key.endpoint},
                           {"json_offset", 0}};
        if (event.key.endpoint == "body_marker") {
            leg["tag_emission_ordinal"] = nullptr;
            leg["tag_offset"] = nullptr;
            leg["tag_tile_stride"] = nullptr;
            leg["tag_shard_stride"] = nullptr;
        } else {
            leg["tag_emission_ordinal"] = event.tag_ordinal;
            leg["tag_offset"] = 0;
            leg["tag_tile_stride"] = 1;
            leg["tag_shard_stride"] = 1;
        }
        nlohmann::json initial_parent = nullptr;
        if (!event.parents.empty()) {
            initial_parent = parent_json_id(event.parents.front(), programs,
                                            body_ids);
        }
        endpoint_runs.push_back({{"rank", event.rank},
                                 {"group_ordinal", 0},
                                 {"first_tile", event.key.tile},
                                 {"tile_count", 1},
                                 {"shard_ordinals", {event.key.shard}},
                                 {"first_json_id", event.json_id},
                                 {"tile_stride", 1},
                                 {"shard_stride", 1},
                                 {"initial_parent_json_id", initial_parent},
                                 {"legacy_context", {
                                     {"request_id", legacy_request},
                                     {"stage", owner_stage},
                                     {"generation", legacy_generation}}},
                                 {"legs", nlohmann::json::array({leg})}});
        for (const ParentRef& parent : event.parents) {
            dependency_runs.push_back({
                {"rank", event.rank}, {"group_ordinal", 0},
                {"first_tile", event.key.tile}, {"tile_count", 1},
                {"shard_ordinals", {event.key.shard}},
                {"endpoint", event.key.endpoint},
                {"dependency_kind", parent_is_ordinary(parent, programs)
                                        ? "ordinary_node_terminal"
                                        : "virtual_endpoint_terminal"},
                {"first_dependency_json_id",
                 parent_json_id(parent, programs, body_ids)},
                {"dependency_tile_stride", 0},
                {"dependency_shard_stride", 0}});
        }
    }

    nlohmann::json anchors = nlohmann::json::array();
    for (const EventKey& event_key : program.emission_order) {
        const EndpointEvent& event = program.events.at(event_key);
        if (!event.physical_anchor) continue;
        nlohmann::json selector;
        if (event.key.endpoint == "body_marker") {
            selector = {{"kind", "internal_marker_done"},
                        {"group_ordinal", 0},
                        {"tile_ordinal", event.key.tile},
                        {"shard_ordinal", event.key.shard},
                        {"endpoint", "body_marker"}};
        } else {
            selector = {{"kind", event.key.endpoint == "data_recv"
                                      ? "tile_endpoint_terminal"
                                      : "tile_endpoint_leg"},
                        {"group_ordinal", 0},
                        {"tile_ordinal", event.key.tile},
                        {"shard_ordinal", event.key.shard},
                        {"endpoint", event.key.endpoint},
                        {"tag_allocation_emission_ordinal", event.tag_ordinal},
                        {"tag_offset", 0}};
        }
        anchors.push_back({{"role", event.anchor_role}, {"rank", event.rank},
                           {"binding", "physical_endpoint_slot"},
                           {"node_id", event.json_id},
                           {"selector", std::move(selector)}});
    }
    anchors.push_back({{"role", "service"}, {"rank", kOwnerRank},
                       {"binding", "descriptor_cursor_selector"},
                       {"node_id", nullptr},
                       {"selector", {{"kind", "existing_service_gate"},
                                     {"existing_gate", decode
                                          ? "decode_service_gate"
                                          : "prefill_service_gate"}}}});
    anchors.push_back({{"role", "settle"}, {"rank", kOwnerRank},
                       {"binding", "descriptor_cursor_selector"},
                       {"node_id", nullptr},
                       {"selector", {{"kind", "all_obligations_settled"}}}});

    nlohmann::json consume_runs = nlohmann::json::array();
    if (!decode) {
        const EventKey marker1 = key(0, 1, 0, "body_marker");
        const EventKey marker2 = key(0, 2, 0, "body_marker");
        consume_runs.push_back({{"rank", kOwnerRank}, {"group_ordinal", 0},
                                {"first_tile", 1}, {"tile_count", 1},
                                {"shard_ordinals", {0, 1}},
                                {"dependency_kind", "ordinary_node_terminal"},
                                {"first_dependency_json_id",
                                 programs[0].events.at(marker1).json_id},
                                {"dependency_tile_stride", 0},
                                {"dependency_shard_stride", 0}});
        consume_runs.push_back({{"rank", kOwnerRank}, {"group_ordinal", 0},
                                {"first_tile", 2}, {"tile_count", 1},
                                {"shard_ordinals", {0, 1}},
                                {"dependency_kind", "virtual_endpoint_terminal"},
                                {"first_dependency_json_id",
                                 programs[0].events.at(marker2).json_id},
                                {"dependency_tile_stride", 0},
                                {"dependency_shard_stride", 0}});
    } else {
        consume_runs.push_back({{"rank", kOwnerRank}, {"group_ordinal", 0},
                                {"first_tile", 1}, {"tile_count", 1},
                                {"shard_ordinals", {0, 1}},
                                {"dependency_kind", "ordinary_node_terminal"},
                                {"first_dependency_json_id",
                                 body_ids.at(DecodeBodyA)},
                                {"dependency_tile_stride", 0},
                                {"dependency_shard_stride", 0}});
        consume_runs.push_back({{"rank", kOwnerRank}, {"group_ordinal", 0},
                                {"first_tile", 2}, {"tile_count", 2},
                                {"shard_ordinals", {0, 1}},
                                {"dependency_kind", "ordinary_node_terminal"},
                                {"first_dependency_json_id",
                                 body_ids.at(DecodeBodyB)},
                                {"dependency_tile_stride", 0},
                                {"dependency_shard_stride", 0}});
    }

    const uint64_t total_units = program.tile_count * 2;
    nlohmann::json shards = nlohmann::json::array();
    for (uint64_t shard = 0; shard < 2; ++shard) {
        const int source = program.source_ranks.at(static_cast<size_t>(shard));
        shards.push_back({{"shard_ordinal", shard}, {"source_rank", source},
                          {"target_rank", kOwnerRank},
                          {"total_data_bytes", total_units},
                          {"bytes_per_unit", 1},
                          {"noc_path", nlohmann::json::array({source, kOwnerRank})}});
    }
    const std::string stage = decode ? "decode" : "prefill";
    const uint64_t generation = decode ? 1 : 0;
    const uint64_t segment_ordinal = decode ? 1 : 0;
    const std::string domain_id = std::string("pb1/") + kRequestHex + "/6/1";
    program.wire = {
        {"descriptor_version", 1},
        {"identity", {{"request_id", kRequestId}, {"stage", stage},
                       {"generation", generation}, {"stage_epoch", 4},
                       {"segment_ordinal", segment_ordinal},
                       {"rank", kOwnerRank}, {"staging_allocation_id", 1},
                       {"segment_id", segment_id(program.index)},
                       {"buffer_domain_id", domain_id}}},
        {"stream_kind", decode ? "decode_remote_credit" : "prefill_read_window"},
        {"groups", nlohmann::json::array({
            {{"group_ordinal", 0}, {"pass_or_window_ordinal", segment_ordinal},
             {"source_ordinal", segment_ordinal},
             {"source_key", decode ? "fixture-decode-source" : "fixture-prefill-source"},
             {"layer_start", 0}, {"layer_end", 1},
             {"tile_axis", {{"unit", "stream_bytes"},
                            {"total_units", total_units},
                            {"units_per_tile", 2},
                            {"tile_count", program.tile_count},
                            {"tail_units", 2},
                            {"global_tile_first", program.global_tile_first}}},
             {"shards", std::move(shards)}}})},
        {"protocol", {{"buffer_slots", 2}, {"release_lag_tiles", 2},
                       {"initial_tiles_without_release", 2},
                       {"fill_ack_bytes", 1}, {"release_ack_bytes", 1},
                       {"hbm_charge", true},
                       {"target_rank_order", "release_ack_send_i_if_required_then_data_recv_i_then_fill_ack_send_i_then_data_recv_i_plus_1"},
                       {"source_rank_order", "release_ack_recv_i_if_required_then_data_send_i_then_fill_ack_recv_i"},
                       {"tile_branching", "tile1_head_main_chain_tile2_immediate_side_branch_tile3_plus_release_gated"}}},
        {"prediction_profile", {{"prefill_consume", "zero_duration_no_extra_consume"},
                                 {"decode_consume", {{"kind", "profiled_consume"},
                                                      {"profile_id", "fixture-profile"},
                                                      {"compute_ns_total", 0.0},
                                                      {"hbm_bytes_total", 0},
                                                      {"resource_key", "hbm:6"},
                                                      {"share_rule", "existing_action_work_graph_byte_share_vector"}}}}},
        {"execution_profile", {{"data_legs", "ordinary_comm_send_recv_path"},
                                {"endpoint_hbm_rule", "one_original_COMM_READ_or_COMM_WRITE_job_per_charged_endpoint"},
                                {"prefill_consume", "existing_body_or_marker_only"},
                                {"decode_consume", "existing_body_nodes_only"}}},
        {"logical_id_ranges", logical_ranges(program)},
        {"tag_allocation_runs", std::move(tag_runs)},
        {"endpoint_runs", std::move(endpoint_runs)},
        {"endpoint_dependency_runs", std::move(dependency_runs)},
        {"consume_completion_runs", std::move(consume_runs)},
        {"projection", make_projection(program.tile_count,
                                        decode ? "D1" : "P0")},
        {"anchors", std::move(anchors)}};
    return program.wire;
}

void create_events(DescriptorProgram& prefill, DescriptorProgram& decode) {
    prefill.index = 0;
    prefill.tile_count = 2;
    prefill.global_tile_first = 1;
    prefill.source_ranks = {0, 5};
    decode.index = 1;
    decode.tile_count = 3;
    decode.global_tile_first = 3;
    decode.source_ranks = {5, 5};

    // Preserve the old side branch: tile 2 source sends are emitted before
    // the tile-1 main-chain sends. Every other leg keeps its original rank
    // order and dependency, with one canonical marker shared by two shards.
    add_event(prefill, 2, 0, "data_send");
    add_event(prefill, 2, 1, "data_send");
    add_event(prefill, 1, 0, "data_send");
    add_event(prefill, 1, 1, "data_send");
    for (uint64_t tile = 1; tile <= 2; ++tile) {
        for (uint64_t shard = 0; shard < 2; ++shard) {
            add_event(prefill, tile, shard, "fill_ack_recv",
                      {event_parent(key(0, tile, shard, "data_send"))});
        }
    }
    add_event(prefill, 1, 0, "data_recv");
    add_event(prefill, 1, 0, "fill_ack_send",
              {event_parent(key(0, 1, 0, "data_recv"))});
    add_event(prefill, 1, 1, "data_recv",
              {event_parent(key(0, 1, 0, "fill_ack_send"))});
    add_event(prefill, 1, 1, "fill_ack_send",
              {event_parent(key(0, 1, 1, "data_recv"))});
    add_event(prefill, 1, 0, "body_marker",
              {event_parent(key(0, 1, 0, "fill_ack_send")),
               event_parent(key(0, 1, 1, "fill_ack_send"))}, true, "frontier");
    add_event(prefill, 2, 0, "data_recv",
              {event_parent(key(0, 1, 1, "fill_ack_send"))});
    add_event(prefill, 2, 0, "fill_ack_send",
              {event_parent(key(0, 2, 0, "data_recv"))});
    add_event(prefill, 2, 1, "data_recv",
              {event_parent(key(0, 2, 0, "fill_ack_send"))});
    add_event(prefill, 2, 1, "fill_ack_send",
              {event_parent(key(0, 2, 1, "data_recv"))});
    add_event(prefill, 2, 0, "body_marker",
              {event_parent(key(0, 2, 0, "fill_ack_send")),
               event_parent(key(0, 2, 1, "fill_ack_send"))});
    mark_anchor(prefill, key(0, 2, 0, "data_send"), "start");
    mark_anchor(prefill, key(0, 1, 0, "data_recv"), "frontier");

    EventKey previous_source;
    bool have_previous_source = false;
    for (uint64_t tile = 1; tile <= 3; ++tile) {
        for (uint64_t shard = 0; shard < 2; ++shard) {
            std::vector<ParentRef> release_parents;
            if (have_previous_source) {
                release_parents.push_back(event_parent(previous_source));
            }
            const EventKey release = key(1, tile, shard, "release_ack_recv");
            add_event(decode, tile, shard, "release_ack_recv",
                      std::move(release_parents),
                      tile == 1 && shard == 0, tile == 1 && shard == 0
                          ? "start" : "");
            const EventKey data_send = key(1, tile, shard, "data_send");
            add_event(decode, tile, shard, "data_send",
                      {event_parent(release)});
            const EventKey fill_recv = key(1, tile, shard, "fill_ack_recv");
            add_event(decode, tile, shard, "fill_ack_recv",
                      {event_parent(data_send)});
            previous_source = fill_recv;
            have_previous_source = true;
        }
    }

    EventKey previous_target;
    bool have_previous_target = false;
    for (uint64_t tile = 1; tile <= 3; ++tile) {
        std::vector<ParentRef> release0_parents;
        if (tile == 1) {
            release0_parents.push_back(
                event_parent(key(0, 1, 0, "body_marker")));
        } else if (tile == 2) {
            release0_parents.push_back(
                event_parent(key(0, 2, 0, "body_marker")));
        } else {
            release0_parents.push_back(body_parent(DecodeBodyA));
        }
        if (have_previous_target) {
            release0_parents.push_back(event_parent(previous_target));
        }
        add_event(decode, tile, 0, "release_ack_send",
                  std::move(release0_parents));
        add_event(decode, tile, 0, "data_recv",
                  {event_parent(key(1, tile, 0, "release_ack_send"))},
                  (tile == 1 || tile == 3), "frontier");
        add_event(decode, tile, 0, "fill_ack_send",
                  {event_parent(key(1, tile, 0, "data_recv"))});
        add_event(decode, tile, 1, "release_ack_send",
                  {event_parent(key(1, tile, 0, "fill_ack_send"))});
        add_event(decode, tile, 1, "data_recv",
                  {event_parent(key(1, tile, 1, "release_ack_send"))},
                  (tile == 1 || tile == 3), "frontier");
        add_event(decode, tile, 1, "fill_ack_send",
                  {event_parent(key(1, tile, 1, "data_recv"))});
        previous_target = key(1, tile, 1, "fill_ack_send");
        have_previous_target = true;
    }
    mark_anchor(decode, key(1, 1, 0, "data_recv"), "frontier");
    mark_anchor(decode, key(1, 1, 1, "data_recv"), "frontier");
    mark_anchor(decode, key(1, 3, 0, "data_recv"), "frontier");
    mark_anchor(decode, key(1, 3, 1, "data_recv"), "frontier");
}

nlohmann::json body_node_json(const BodyId body, const uint64_t id) {
    return body_node(body, id);
}

BuiltFixture build_fixture() {
    BuiltFixture fixture;
    fixture.programs.resize(2);
    create_events(fixture.programs[0], fixture.programs[1]);
    std::map<int, uint64_t> next_id{{0, 0}, {5, 0}, {kOwnerRank, 0}};
    for (const EventKey& event_key : fixture.programs[0].emission_order) {
        EndpointEvent& event = fixture.programs[0].events.at(event_key);
        event.json_id = next_id[event.rank]++;
    }
    fixture.body_json_ids[PrefillService] = next_id[kOwnerRank]++;
    for (const EventKey& event_key : fixture.programs[1].emission_order) {
        EndpointEvent& event = fixture.programs[1].events.at(event_key);
        event.json_id = next_id[event.rank]++;
    }
    fixture.body_json_ids[DecodeBodyA] = next_id[kOwnerRank]++;
    fixture.body_json_ids[DecodeBodyB] = next_id[kOwnerRank]++;
    for (size_t i = 0; i < fixture.programs.size(); ++i) {
        make_descriptor_wire(fixture.programs[i], fixture.programs,
                             fixture.body_json_ids);
    }

    struct NodeEntry { int rank; uint64_t id; nlohmann::json node; };
    std::vector<NodeEntry> nodes;
    for (const DescriptorProgram& program : fixture.programs) {
        for (const EventKey& event_key : program.emission_order) {
            const EndpointEvent& event = program.events.at(event_key);
            if (event.physical_anchor) {
                nodes.push_back({event.rank, event.json_id,
                                 anchor_node(event, program)});
            }
        }
    }
    for (const auto& entry : fixture.body_json_ids) {
        nodes.push_back({kOwnerRank, entry.second,
                         body_node_json(entry.first, entry.second)});
    }
    std::sort(nodes.begin(), nodes.end(), [](const NodeEntry& a,
                                            const NodeEntry& b) {
        return std::tie(a.rank, a.id) < std::tie(b.rank, b.id);
    });
    nlohmann::json node_wire = nlohmann::json::array();
    for (auto& node : nodes) node_wire.push_back(std::move(node.node));

    nlohmann::json edges = nlohmann::json::array();
    const auto add_body_edge = [&](const EventKey& parent, const BodyId child) {
        const EndpointEvent& event = fixture.programs.at(
            static_cast<size_t>(parent.descriptor)).events.at(parent);
        if (!event.physical_anchor) {
            fail("ordinary body edge tried to reference a virtual endpoint hole");
        }
        edges.push_back({{"rank", kOwnerRank}, {"from", event.json_id},
                         {"to", fixture.body_json_ids.at(child)},
                         {"kind", "data"}});
    };
    add_body_edge(key(0, 1, 0, "body_marker"), PrefillService);
    add_body_edge(key(1, 1, 0, "data_recv"), DecodeBodyA);
    add_body_edge(key(1, 1, 1, "data_recv"), DecodeBodyA);
    add_body_edge(key(1, 3, 0, "data_recv"), DecodeBodyB);
    add_body_edge(key(1, 3, 1, "data_recv"), DecodeBodyB);
    edges.push_back({{"rank", kOwnerRank},
                     {"from", fixture.body_json_ids.at(DecodeBodyA)},
                     {"to", fixture.body_json_ids.at(DecodeBodyB)},
                     {"kind", "data"}});

    const nlohmann::json watches = nlohmann::json::array({
        {{"request_id", kRequestId}, {"stage", "prefill"},
         {"generation", 0},
         {"members", {{"6", fixture.body_json_ids.at(PrefillService)}}},
         {"statuses", nlohmann::json::array({"Success"})}},
        {{"request_id", kRequestId}, {"stage", "decode"},
         {"generation", 1},
         {"members", {{"6", fixture.body_json_ids.at(DecodeBodyB)}}},
         {"statuses", nlohmann::json::array({"Success"})}}});

    fixture.batch = {{"schema_version", 2}, {"batch_id", 1},
                     {"source_delivery_sequence", 1}, {"error", ""},
                     {"nodes", std::move(node_wire)},
                     {"parent_edges", std::move(edges)},
                     {"watches", std::move(watches)},
                     {"assignments", nlohmann::json::array()},
                     {"kv_actions", nlohmann::json::array()},
                     {"future_alarms", nlohmann::json::array()},
                     {"touched_ranks", {0, 5, 6}},
                     {"pipeline_descriptors", nlohmann::json::array({
                          fixture.programs[0].wire,
                          fixture.programs[1].wire})},
                     {"kernel_contract", contract_wire()}};
    return fixture;
}

struct TerminalRecord {
    int rank = -1;
    uint64_t node_id = 0;
    uint64_t tick = 0;
};

struct FlowRecord {
    std::string descriptor_identity;
    uint64_t generation = 0;
    std::string leg;
    uint64_t tile = 0;
    uint64_t shard = 0;
    int rank = -1;
    uint32_t tag = 0;
    uint64_t flow_id = 0;
    uint64_t bound_tick = 0;
};

struct WatchFireRecord {
    std::string request_id;
    std::string stage;
    uint64_t generation = 0;
    uint64_t tick = 0;
};

struct IssuePassState {
    std::vector<Workload*>* workloads = nullptr;
};

void issue_pass_callback(void* opaque, const int rank) {
    auto* state = static_cast<IssuePassState*>(opaque);
    if (state == nullptr || state->workloads == nullptr || rank < 0 ||
        static_cast<size_t>(rank) >= state->workloads->size()) {
        fail("corrupt deferred issue pass");
    }
    (*state->workloads)[rank]->issue_dep_free_nodes();
}

struct TerminalHookContext {
    std::vector<TerminalRecord>* records = nullptr;
    OnlineCompletionHookContext* online_hook = nullptr;
};

void terminal_hook(void* opaque, const int rank, const uint64_t node_id,
                   const char* request_id, const char* stage,
                   const uint64_t generation, const uint64_t tick,
                   const int terminal_status) {
    auto* context = static_cast<TerminalHookContext*>(opaque);
    if (context == nullptr || context->records == nullptr ||
        context->online_hook == nullptr ||
        terminal_status != static_cast<int>(NodeTerminalStatus::Success)) {
        fail("invalid physical terminal callback");
    }
    context->records->push_back(TerminalRecord{rank, node_id, tick});
    online_completion_hook(context->online_hook, rank, node_id, request_id,
                           stage, generation, tick, terminal_status);
}

struct TransferObservation {
    TileTransportAdapter::ActiveTransfer transfer;
    uint64_t issue_tick = 0;
    bool completed = false;
    uint64_t completion_tick = 0;
};

void sample_transfers(TileTransportAdapter& adapter, const uint64_t tick,
                      std::map<uint64_t, TransferObservation>& observations,
                      size_t* peak_active_per_rank,
                      size_t* peak_target_data_tiles,
                      size_t* peak_source_shard_data_tiles) {
    const auto active = adapter.active_snapshot();
    std::map<int, size_t> active_by_rank;
    std::set<uint64_t> target_data_tiles;
    std::map<std::pair<int, uint64_t>, std::set<uint64_t>> source_data_tiles;
    std::set<uint64_t> active_ids;
    for (const auto& transfer : active) {
        active_ids.insert(transfer.transfer_id);
        ++active_by_rank[transfer.rank];
        if (transfer.logical_key.leg == "data_recv" &&
            transfer.rank == kOwnerRank) {
            target_data_tiles.insert(transfer.logical_key.tile);
        } else if (transfer.logical_key.leg == "data_send") {
            source_data_tiles[{transfer.rank, transfer.logical_key.shard}]
                .insert(transfer.logical_key.tile);
        }
        auto it = observations.find(transfer.transfer_id);
        if (it == observations.end()) {
            observations.emplace(
                transfer.transfer_id,
                TransferObservation{transfer, transfer.submit_tick, false, 0});
        } else {
            it->second.transfer = transfer;
        }
    }
    for (const auto& entry : active_by_rank) {
        *peak_active_per_rank = std::max(*peak_active_per_rank, entry.second);
    }
    *peak_target_data_tiles = std::max(*peak_target_data_tiles,
                                       target_data_tiles.size());
    for (const auto& entry : source_data_tiles) {
        *peak_source_shard_data_tiles = std::max(
            *peak_source_shard_data_tiles, entry.second.size());
    }
    for (auto& entry : observations) {
        if (!entry.second.completed && active_ids.count(entry.first) == 0) {
            entry.second.completed = true;
            entry.second.completion_tick = tick;
        }
    }
}

void require_one_terminal(const std::vector<TerminalRecord>& records,
                          const int rank, const uint64_t node_id) {
    const size_t count = static_cast<size_t>(std::count_if(
        records.begin(), records.end(), [rank, node_id](const auto& record) {
            return record.rank == rank && record.node_id == node_id;
        }));
    expect(count == 1, "physical node terminal count rank=" +
                           std::to_string(rank) + " id=" +
                           std::to_string(node_id) + " is " +
                           std::to_string(count));
}

}  // namespace

nlohmann::json stage_batch_wire(const BuiltFixture& fixture,
                                const int descriptor_index) {
    const std::string stage = descriptor_index == 0 ? "prefill" : "decode";
    nlohmann::json result = fixture.batch;
    result["batch_id"] = descriptor_index + 1;
    result["source_delivery_sequence"] = descriptor_index + 1;
    result["nodes"] = nlohmann::json::array();
    result["parent_edges"] = nlohmann::json::array();
    result["watches"] = nlohmann::json::array();
    result["pipeline_descriptors"] = nlohmann::json::array({
        fixture.programs.at(static_cast<size_t>(descriptor_index)).wire});
    std::set<uint64_t> stage_node_ids;
    std::set<int> touched_ranks;
    for (const auto& node : fixture.batch.at("nodes")) {
        if (node.at("stage").get<std::string>() != stage) continue;
        result["nodes"].push_back(node);
        stage_node_ids.insert(node.at("id").get<uint64_t>());
        touched_ranks.insert(node.at("rank").get<int>());
    }
    for (const auto& edge : fixture.batch.at("parent_edges")) {
        if (stage_node_ids.count(edge.at("to").get<uint64_t>()) != 0) {
            result["parent_edges"].push_back(edge);
        }
    }
    for (const auto& watch : fixture.batch.at("watches")) {
        if (watch.at("stage").get<std::string>() == stage) {
            result["watches"].push_back(watch);
        }
    }
    // Signature 5 convention (2026-10-06 ruling): touched_ranks declares
    // ordinary-node ranks UNION the compact virtual endpoint ranks of the
    // included descriptors (the committer computes exactly that union).
    std::set<int> declared_ranks = touched_ranks;
    for (const auto& descriptor :
         result.at("pipeline_descriptors")) {
        for (const auto& range : descriptor.at("logical_id_ranges")) {
            declared_ranks.insert(range.at("rank").get<int>());
        }
        for (const auto& run : descriptor.at("endpoint_runs")) {
            declared_ranks.insert(run.at("rank").get<int>());
        }
    }
    result["touched_ranks"] = nlohmann::json::array();
    for (const int rank : declared_ranks) {
        result["touched_ranks"].push_back(rank);
    }
    return result;
}

nlohmann::json descriptor_for(const nlohmann::json& snapshot,
                              const std::string& segment) {
    for (const auto& entry : snapshot.at("descriptor_commits")) {
        if (entry.at("identity").at("segment_id") == segment) {
            return entry;
        }
    }
    fail("snapshot has no descriptor " + segment);
}

#endif  // EXECUTION_DRIVEN_TESTS_REMOTE_READ_EXECUTOR_PROGRAM_HH
