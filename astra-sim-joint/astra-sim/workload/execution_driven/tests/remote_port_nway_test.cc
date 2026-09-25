/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

/**

remote_port_nway_test.cc -- remote-memory port N-way fluid model precise
fixture (SerDes片外链路并发化改造执行方案 V5.3 阶段 5.1 item 1 + item 2
数学锚点; test-only, never part of the production gates).

Self-contained: the test itself writes ALL fixture inputs (remote_memory.json
for the three port mappings, system.json / comm_group.json / network.yml and
the per-rank .et graphs) into isolated temp directories under
/tmp/serdes-joint-work/.  system.json/comm_group.json keep the official
template shape (preferred-dataset-splits / scheduling-policy /
collective-optimization present, the four *-implementation keys all
["ring","ring"]): the joint deep-dive H1/H2/H3 landmines (custom-impl ETFeeder
leak, doubleBinaryTree nullptr, dimensionless comm groups) are never armed.
comm_group.json is "{}" -- no groups at all, exactly like the in-repo
make_hbm_nway_fixture.py precedent -- so no dimensionless group can exist.

Every scenario is a FULL static ET simulation (entry replicated from
congestion_aware/main.cc, same skeleton as hbm_nway_test.cc) over plain
unannotated MEM_LOAD nodes (Workload::issue_remote_mem ->
AnalyticalRemoteMemory::issue, single-sided completion), run in a forked
child process for state isolation; the parent only forks, waits and checks
exit codes.  Terminals are captured through the CompletionObserver hook in
exact callback-delivery order; port facts come exclusively from the read-only
PortStatsSnapshot; is_drained() closes every scenario (its settled audit also
re-runs when the backend is destroyed).

Coverage (task-book list -> scenario):
  PER_NPU / PER_NODE / MEMORY_POOL mapping      S1/S5/S6, S3/S4, S2
  same-tick latency / staggered latency         S1/S5, S4 (chain re-issue)
  2-flow long/short + completion re-split       S3 (600/1200 -> 300/400,
                                                 NOT 500), S1, S2 chain
  latency-ready == stream completion same Tick  S4 (B ready@400 == C fluid
                                                 completion instant 400)
  same-Tick multi-port / multi-job ordering     S5 ((port_index, issue_seq)
                                                 batch order), S9 (dual-zero
                                                 before same-Tick streaming
                                                 completion + §3.3 keep rule)
  single transaction non-divisible              S6 (100 B @ 6 B/ns -> ceil
                                                 116.67 = 117)
  tiny residual                                 S7 (1 B @ 6 B/ns: fluid ~1/6 ns,
                                                 callback next Tick), S6 pair
  dual zero (0 B / 0 ns)                        S9 (+1 ns one-shot timer, NOT
                                                 same-Tick completion)
  zero-byte positive latency                    S8 (never joins the bandwidth
                                                 denominator: 600 B partner
                                                 finishes solo at 200 not 300)
  huge value / NaN-class / invalid bw fail-closed S10 (14 forked cases:
                                                 tensor_size > 2^53 runtime
                                                 fatal; inf (JSON 1e999),
                                                 negative, zero, string and
                                                 missing remote-mem-bw;
                                                 negative/string/inf latency;
                                                 PER_NODE missing/zero shape
                                                 keys; bogus memory-type;
                                                 NO_MEMORY_EXPANSION issue;
                                                 PER_NPU unbound rank)
  service-rate / capacity boundaries + count/bytes conservation
                                                every scenario: per-port
                                                issued==completed (count AND
                                                bytes), live per-state counts
                                                zero, |bytes_served -
                                                bw*port_busy_ns| <= 1e-6,
                                                shared_busy <= port_busy,
                                                peak_streaming <=
                                                peak_in_flight, is_drained.

Math anchors (阶段 5.1 item 2; segment-by-segment service, no hand-copied
mixed-rate formulas):
  bw=6 B/ns, latency=100 ns, two same-Tick 600 B flows -> each 3 B/ns,
    callback t=300 (serial old-behavior reference 200/400 is deliberately
    NOT asserted -- it is the removed FIFO semantics).        [S1]
  bw=100 B/ns, latency=50 ns, four flows 1k/2k/3k/4 kB, continuous
    work-conserving re-split: segments 4-way@25 B/ns [50,90], 3-way@(100/3)
    [90,120], 2-way@50 [120,140], 1-way@100 [140,150] -> callbacks exactly
    90/120/140/150.                                           [S2]
  1 B @ 6 B/ns: continuous fluid finish ~1/6 ns, observable callback at the
    NEXT integer Tick (1) -- the two time notions stay distinct. [S7]

--stress mode: multi-port / large-flow count pressure run (PER_NPU,
16 ports x 128 dep-free flows, deterministic LCG sizes) writing ports / max
active streams / redistribution+join transition counts / issue+completion
counts / CPU time / wall time / peak RSS to
/tmp/serdes-joint-work/stress_results.txt.  The backend's exact transition
registration counter (BackendEventStats) is private, so the file reports the
public PortStats transition proxies (redistribution_events +
stream_join_events) next to the issued/completed counts, clearly labeled.

Build (task 4 of the pipeline registers the identical target name in
astra-sim/network_frontend/analytical/CMakeLists.txt):
  AstraSim_Analytical_Congestion_Aware_RemotePortNwayTest
Run: build/.../bin/AstraSim_Analytical_Congestion_Aware_RemotePortNwayTest
     [--stress]
Exit code 0 on ALL PASS.
*******************************************************************************/

#include "astra-sim/common/Logging.hh"
#include "astra-sim/system/Sys.hh"
#include "astra-sim/system/WorkloadLayerHandlerData.hh"
#include "astra-sim/workload/MetricCollector.hh"
#include "astra-sim/workload/execution_driven/CompletionObserver.hh"
#include "congestion_aware/CongestionAwareNetworkApi.hh"
#include "extern/remote_memory_backend/analytical/AnalyticalRemoteMemory.hh"
#include <astra-network-analytical/common/EventQueue.h>
#include <astra-network-analytical/common/NetworkParser.h>
#include <astra-network-analytical/congestion_aware/Helper.h>
#include <et_def.pb.h>
#include <json/json.hpp>

#include <sys/resource.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cerrno>
#include <chrono>
#include <ctime>
#include <functional>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

using namespace AstraSim;
using namespace Analytical;
using namespace AstraSimAnalytical;
using namespace AstraSimAnalyticalCongestionAware;
using namespace NetworkAnalytical;
using namespace NetworkAnalyticalCongestionAware;

namespace {

//****************************************************************************
// Failure reporting: fail-fast, NDEBUG-proof (never assert()).
//****************************************************************************

[[noreturn]] void fail(const std::string& what) {
    std::fprintf(stderr, "[remote_port_nway_test] FAIL: %s\n", what.c_str());
    std::fflush(stderr);
    std::exit(1);
}

void expect_true(bool got, const std::string& what) {
    if (!got) {
        fail(what + " (expected true)");
    }
}

void expect_eq_u64(uint64_t got, uint64_t want, const std::string& what) {
    if (got != want) {
        fail(what + ": got " + std::to_string(got) + ", expected " +
             std::to_string(want));
    }
}

void expect_eq_d(double got, double want, const std::string& what,
                 double tol = 1e-6) {
    if (got < want - tol || got > want + tol) {
        char buf[256];
        std::snprintf(buf, sizeof(buf), "%s: got %.12f, expected %.12f",
                      what.c_str(), got, want);
        fail(buf);
    }
}

//****************************************************************************
// Temp-dir plumbing.  Everything lives under /tmp/serdes-joint-work/.
//****************************************************************************

const char* kTmpBase = "/tmp/serdes-joint-work";
const char* kStressResultsPath =
    "/tmp/serdes-joint-work/stress_results.txt";

std::string scenario_dir(const std::string& name) {
    return std::string(kTmpBase) + "/portnway_" + name;
}

void write_text(const std::string& path, const std::string& content) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out.is_open()) {
        fail("cannot write fixture file " + path);
    }
    out << content;
    out.close();
    if (!out.good()) {
        fail("failed writing fixture file " + path);
    }
}

//****************************************************************************
// Minimal ET writer (varint32 length + serialized message, the exact wire
// format Chakra::FeederV3::ProtobufUtils::readMessage consumes).
//****************************************************************************

void write_varint32(std::ofstream& out, uint32_t value) {
    uint8_t byte;
    while (value > 0x7f) {
        byte = static_cast<uint8_t>(value & 0x7f) | 0x80;
        out.write(reinterpret_cast<const char*>(&byte), 1);
        value >>= 7;
    }
    byte = static_cast<uint8_t>(value);
    out.write(reinterpret_cast<const char*>(&byte), 1);
}

template <typename T>
void write_message(std::ofstream& out, const T& msg) {
    const std::string serialized = msg.SerializeAsString();
    write_varint32(out, static_cast<uint32_t>(serialized.size()));
    out.write(serialized.data(), static_cast<std::streamsize>(serialized.size()));
}

// One fixture MEM node: plain remote-memory transaction (no local-HBM KV
// restore annotation, no hbm-access-mode endpoint charge).  tensor_size == 0
// is legal and lands in the zero-byte backend paths (S8/S9).
struct MemNodeSpec {
    uint64_t id = 0;
    uint64_t tensor_size = 0;
    std::vector<uint64_t> data_deps;  // static ET Data-dependency parents
    bool is_replay_comp = false;      // idle-rank 1 ns replay COMP filler
};

void write_et_file(const std::string& path,
                   const std::vector<MemNodeSpec>& nodes) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out.is_open()) {
        fail("cannot write ET file " + path);
    }

    ChakraProtoMsg::GlobalMetadata metadata;
    metadata.set_version("0.0.4");
    ChakraProtoMsg::AttributeProto* schema = metadata.add_attr();
    schema->set_name("schema");
    schema->set_string_val("1.0.2-chakra.0.0.4");
    ChakraProtoMsg::AttributeProto* mode = metadata.add_attr();
    mode->set_name("execution_mode");
    mode->set_string_val("remote_port_nway_fixture");
    ChakraProtoMsg::AttributeProto* granularity = metadata.add_attr();
    granularity->set_name("trace_granularity");
    granularity->set_string_val("synthetic");
    write_message(out, metadata);

    for (const MemNodeSpec& spec : nodes) {
        ChakraProtoMsg::Node node;
        node.set_id(spec.id);
        node.set_name(spec.is_replay_comp
                          ? "idle_replay_comp"
                          : "remote_mem_" + std::to_string(spec.id));
        node.set_type(spec.is_replay_comp ? ChakraProtoMsg::COMP_NODE
                                          : ChakraProtoMsg::MEM_LOAD_NODE);
        if (spec.is_replay_comp) {
            // roofline is disabled in this fixture: COMP routes to
            // issue_replay with runtime = 1 ns and never touches a port.
            ChakraProtoMsg::AttributeProto* num_ops = node.add_attr();
            num_ops->set_name("num_ops");
            num_ops->set_uint64_val(1);
            ChakraProtoMsg::AttributeProto* tensor = node.add_attr();
            tensor->set_name("tensor_size");
            tensor->set_uint64_val(64);
        } else {
            ChakraProtoMsg::AttributeProto* tensor = node.add_attr();
            tensor->set_name("tensor_size");
            tensor->set_uint64_val(spec.tensor_size);
            for (const uint64_t parent : spec.data_deps) {
                node.add_data_deps(parent);
            }
        }
        write_message(out, node);
    }
    out.close();
    if (!out.good()) {
        fail("failed writing ET file " + path);
    }
}

//****************************************************************************
// Config writers.  system.json / comm_group.json keep the official template
// shape (H1/H2/H3 避雷: no custom implementations, no doubleBinaryTree, no
// comm groups at all -> no dimensionless group can exist).
//****************************************************************************

void write_system_json(const std::string& path, double remote_bw,
                       double remote_latency_ns) {
    nlohmann::json j;
    j["scheduling-policy"] = "LIFO";
    j["endpoint-delay"] = 10;
    j["active-chunks-per-dimension"] = 1;
    j["preferred-dataset-splits"] = 6;
    j["all-reduce-implementation"] = {"ring", "ring"};
    j["all-gather-implementation"] = {"ring", "ring"};
    j["reduce-scatter-implementation"] = {"ring", "ring"};
    j["all-to-all-implementation"] = {"ring", "ring"};
    j["collective-optimization"] = "localBWAware";
    j["roofline-enabled"] = 0;
    j["replay-only"] = 0;
    j["track-local-mem"] = 0;
    j["trace-enabled"] = 0;
    j["hbm-bandwidth-contention"] = 0;
    j["peak-perf"] = 1000;
    j["local-mem-bw"] = 1640.0;
    j["local-mem-latency"] = 100;
    j["remote-mem-bw"] = remote_bw;
    j["remote-mem-latency"] = remote_latency_ns;
    write_text(path, j.dump(2) + "\n");
}

void write_comm_group_json(const std::string& path) {
    // Same shape as make_hbm_nway_fixture.py / the ledger fixture: empty
    // object.  The fixture has no collectives, and an empty group set cannot
    // carry a dimensions-less group.
    write_text(path, "{}\n");
}

void write_network_yml(const std::string& path, int num_ranks) {
    std::string body =
        "# remote_port_nway synthetic line topology (2 dims: the system\n"
        "# template carries per-dimension collective implementations)\n"
        "topology: [ Line, Line ]\n"
        "npus_count: [ " +
        std::to_string(num_ranks) + ", 1 ]\n"
        "bandwidth: [ 4000.0, 4000.0 ]\n"
        "latency: [ 5, 5 ]\n";
    write_text(path, body);
}

void write_remote_memory_json(const std::string& path,
                              const std::string& memory_type, double bw,
                              double latency_ns,
                              const std::vector<uint64_t>& npu_ids,
                              int num_nodes, int npus_per_node) {
    nlohmann::json j;
    j["memory-type"] = memory_type;
    j["remote-mem-bw"] = bw;
    j["remote-mem-latency"] = latency_ns;
    if (!npu_ids.empty()) {
        j["npu-ids"] = npu_ids;
    }
    if (memory_type == "PER_NODE_MEMORY_EXPANSION") {
        j["num-nodes"] = num_nodes;
        j["num-npus-per-node"] = npus_per_node;
    }
    write_text(path, j.dump(2) + "\n");
}

//****************************************************************************
// Terminal capture (delivery-order preserving).
//****************************************************************************

struct TermRecord {
    int rank;
    uint64_t node_id;
    uint64_t tick;
};

struct HookContext {
    std::vector<TermRecord> records;
};

void terminal_hook(void* ctx, int rank, uint64_t node_id, const char*,
                   const char*, uint64_t, uint64_t tick, int terminal_status) {
    auto* context = static_cast<HookContext*>(ctx);
    // Every fixture node must terminate as Success; Skipped would mean the
    // node never executed (zero-tensor COMP skip / invalid node) and the
    // anchors below would silently rot.
    if (terminal_status !=
        static_cast<int>(ExecutionDriven::NodeTerminalStatus::Success)) {
        fail("unexpected terminal status " + std::to_string(terminal_status) +
             " for rank " + std::to_string(rank) + " node " +
             std::to_string(node_id));
    }
    context->records.push_back(TermRecord{rank, node_id, tick});
}

uint64_t terminal_of(const std::vector<TermRecord>& records, int rank,
                     uint64_t node_id) {
    uint64_t hits = 0;
    uint64_t tick = 0;
    for (const TermRecord& rec : records) {
        if (rec.rank == rank && rec.node_id == node_id) {
            ++hits;
            tick = rec.tick;
        }
    }
    if (hits != 1) {
        fail("node (rank=" + std::to_string(rank) + ", id=" +
             std::to_string(node_id) + ") has " + std::to_string(hits) +
             " terminal records (expected exactly 1)");
    }
    return tick;
}

// The (rank, node_id) sequence of all terminals with `tick`, in exact
// callback-delivery order.
std::vector<std::pair<std::pair<int, uint64_t>, uint64_t>>
terminals_at(const std::vector<TermRecord>& records, uint64_t tick) {
    std::vector<std::pair<std::pair<int, uint64_t>, uint64_t>> out;
    for (const TermRecord& rec : records) {
        if (rec.tick == tick) {
            out.push_back({{rec.rank, rec.node_id}, rec.tick});
        }
    }
    return out;
}

//****************************************************************************
// Full static-ET pipeline runner (entry replicated from
// congestion_aware/main.cc, hbm_nway_test.cc skeleton).  Runs ONE simulation
// in the CURRENT process; scenarios execute it inside forked children.
//****************************************************************************

struct ScenarioSpec {
    std::string name;
    std::string memory_type;  // remote_memory.json memory-type value
    double bw = 0.0;          // B/ns
    double latency_ns = 0.0;
    std::vector<uint64_t> npu_ids;  // PER_NPU
    int num_nodes = 0;              // PER_NODE
    int npus_per_node = 0;          // PER_NODE
    int num_ranks = 0;
    std::vector<std::vector<MemNodeSpec>> per_rank_nodes;
    bool telemetry = false;  // enable §5.1 transaction-detail JSONL
};

struct RunOutput {
    std::vector<TermRecord> terminals;  // exact delivery order
    std::size_t port_count = 0;
    std::vector<AnalyticalRemoteMemory::PortStatsSnapshot> ports;
    bool drained = false;
    std::string bridge_dir;
    uint64_t total_issued = 0;
    uint64_t total_completed = 0;
};

RunOutput run_scenario(const ScenarioSpec& spec) {
    const std::string dir = scenario_dir(spec.name);
    std::error_code ec;
    std::filesystem::remove_all(dir, ec);
    std::filesystem::create_directories(dir, ec);
    if (!std::filesystem::is_directory(dir)) {
        fail("cannot create fixture dir " + dir);
    }

    const std::string system_configuration = dir + "/system.json";
    const std::string comm_group_configuration = dir + "/comm_group.json";
    const std::string network_configuration = dir + "/network.yml";
    const std::string remote_memory_configuration =
        dir + "/remote_memory.json";
    const std::string workload_prefix = dir + "/fixture";

    write_system_json(system_configuration, spec.bw, spec.latency_ns);
    write_comm_group_json(comm_group_configuration);
    write_network_yml(network_configuration, spec.num_ranks);
    write_remote_memory_json(remote_memory_configuration, spec.memory_type,
                             spec.bw, spec.latency_ns, spec.npu_ids,
                             spec.num_nodes, spec.npus_per_node);
    if (spec.per_rank_nodes.size() != static_cast<std::size_t>(spec.num_ranks)) {
        fail("scenario " + spec.name + ": per-rank node list size mismatch");
    }
    for (int rank = 0; rank < spec.num_ranks; ++rank) {
        write_et_file(workload_prefix + "." + std::to_string(rank) + ".et",
                      spec.per_rank_nodes[rank]);
    }

    AstraSim::LoggerFactory::init("empty", "off");

    static HookContext hook_ctx;
    hook_ctx.records.clear();
    ExecutionDriven::CompletionObserver::instance().set_hook(&terminal_hook,
                                                             &hook_ctx);
    MetricCollector::instance().initialize("empty", "off");

    const auto event_queue = std::make_shared<EventQueue>();
    const auto network_parser = NetworkParser(network_configuration);
    const auto topology = construct_topology(network_parser);
    const auto npus_count = topology->get_npus_count();
    const auto npus_count_per_dim = topology->get_npus_count_per_dim();
    const auto dims_count = topology->get_dims_count();
    if (static_cast<int>(npus_count) != spec.num_ranks) {
        fail("scenario " + spec.name + ": topology rank count mismatch");
    }

    CongestionAwareNetworkApi::set_event_queue(event_queue);
    CongestionAwareNetworkApi::set_topology(topology);
    const auto fluid_scheduler = std::make_shared<FluidScheduler>(
        event_queue, topology->get_directed_links(),
        network_parser.get_fluid_max_active_flows(),
        network_parser.get_fluid_max_route_memberships(),
        network_parser.get_progress_report_event_interval());
    CongestionAwareNetworkApi::set_fluid_scheduler(fluid_scheduler);

    auto network_apis =
        std::vector<std::unique_ptr<CongestionAwareNetworkApi>>();
    // Declared BEFORE the Sys vector so the backend (and its §3.4 settled
    // audit on destruction) outlives every Sys.
    const auto memory_api =
        std::make_unique<AnalyticalRemoteMemory>(remote_memory_configuration);
    if (spec.telemetry) {
        memory_api->enable_transaction_telemetry(dir,
                                                 "port-nway-" + spec.name);
    }
    auto systems = std::vector<Sys*>();

    auto queues_per_dim = std::vector<int>();
    for (auto i = 0; i < dims_count; i++) {
        queues_per_dim.push_back(1);
    }

    for (int i = 0; i < spec.num_ranks; i++) {
        auto network_api = std::make_unique<CongestionAwareNetworkApi>(i);
        auto* const system =
            new Sys(i, workload_prefix, comm_group_configuration,
                    system_configuration, memory_api.get(), network_api.get(),
                    npus_count_per_dim, queues_per_dim, 1.0, 1.0, false);
        network_apis.push_back(std::move(network_api));
        systems.push_back(system);
    }

    for (int i = 0; i < spec.num_ranks; i++) {
        systems[i]->workload->fire();
    }
    fluid_scheduler->flush_pending_starts();
    fluid_scheduler->mark_event_loop_started();
    while (!event_queue->finished()) {
        event_queue->proceed();
    }

    RunOutput out;
    out.terminals = hook_ctx.records;
    out.port_count = memory_api->port_count();
    for (std::size_t p = 0; p < out.port_count; ++p) {
        const auto snap = memory_api->port_stats(p);
        out.total_issued += snap.issued_count;
        out.total_completed += snap.completed_count;
        out.ports.push_back(snap);
    }
    out.drained = memory_api->is_drained();
    out.bridge_dir = dir;

    for (auto it : systems) {
        delete it;
    }
    systems.clear();
    // memory_api destroyed at scope exit -> backend shutdown() runs the
    // §3.4 normal-end settlement audit (fatal exit on any leak).
    return out;
}

//****************************************************************************
// Generic conservation / capacity-bound checks, applied to every port of
// every scenario (方案 §5.1: counts AND bytes, never count-equality alone).
//****************************************************************************

void expect_port_conserved(const AnalyticalRemoteMemory::PortStatsSnapshot& s,
                           double bw, const std::string& tag) {
    expect_eq_u64(s.issued_count, s.completed_count,
                  tag + ": issued_count == completed_count");
    expect_eq_u64(s.issued_bytes, s.completed_bytes,
                  tag + ": issued_bytes == completed_bytes");
    expect_eq_u64(s.in_flight_count, 0, tag + ": in_flight drained");
    expect_eq_u64(s.streaming_count, 0, tag + ": streaming drained");
    expect_eq_u64(s.latency_waiting_count, 0, tag + ": latency-waiting drained");
    expect_eq_u64(s.completion_waiting_count, 0,
                  tag + ": completion-waiting drained");
    // Capacity boundary: the port's integrated service equals exactly
    // bw * busy time (it streams at full port bandwidth, split N ways).
    expect_eq_d(s.bytes_served, bw * s.port_busy_ns,
                tag + ": bytes_served == bw * port_busy_ns");
    if (s.shared_busy_ns > s.port_busy_ns + 1e-9) {
        fail(tag + ": shared_busy_ns exceeds port_busy_ns");
    }
    if (s.peak_streaming > s.peak_in_flight) {
        fail(tag + ": peak_streaming exceeds peak_in_flight");
    }
}

//****************************************************************************
// Scenarios.  Each runs inside a forked child; the function returns normally
// on success and exit(1)s through fail() on any mismatch.
//****************************************************************************

// ---- S1: anchor A + PER_NPU mapping + cross-port isolation ----------------
// bw=6 B/ns, latency=100 ns, two same-Tick 600 B flows on port0 -> each
// 3 B/ns, callback t=300; port1's 600 B flow is independent -> t=200 while
// port0 is still streaming (ports never block each other).
void scenario_s1() {
    ScenarioSpec spec;
    spec.name = "s1_anchor_per_npu";
    spec.memory_type = "PER_NPU_MEMORY_EXPANSION";
    spec.bw = 6.0;
    spec.latency_ns = 100.0;
    spec.npu_ids = {0, 1};
    spec.num_ranks = 2;
    spec.per_rank_nodes = {
        {{0, 600, {}, false}, {1, 600, {}, false}},
        {{0, 600, {}, false}},
    };

    const RunOutput out = run_scenario(spec);
    expect_eq_u64(out.terminals.size(), 3, "S1 terminal count");
    expect_eq_u64(terminal_of(out.terminals, 0, 0), 300,
                  "S1 anchor A: rank0 node0 (600 B shared) callback t=300");
    expect_eq_u64(terminal_of(out.terminals, 0, 1), 300,
                  "S1 anchor A: rank0 node1 (600 B shared) callback t=300");
    expect_eq_u64(terminal_of(out.terminals, 1, 0), 200,
                  "S1: rank1 port-isolated 600 B callback t=200");

    expect_eq_u64(out.port_count, 2, "S1 PER_NPU port count");
    const auto& p0 = out.ports[0];
    const auto& p1 = out.ports[1];
    expect_eq_u64(p0.issued_count, 2, "S1 port0 issued");
    expect_eq_u64(p0.completed_count, 2, "S1 port0 completed");
    expect_eq_u64(p0.issued_bytes, 1200, "S1 port0 issued bytes");
    expect_eq_u64(p0.completed_bytes, 1200, "S1 port0 completed bytes");
    expect_eq_u64(p0.peak_in_flight, 2, "S1 port0 peak_in_flight");
    expect_eq_u64(p0.peak_streaming, 2, "S1 port0 peak_streaming (nway)");
    expect_eq_u64(p0.redistribution_events, 0,
                  "S1 port0: simultaneous exhaustion leaves no survivor");
    expect_eq_u64(p0.stream_join_events, 0,
                  "S1 port0: first joins create no prior share to change");
    expect_eq_d(p0.port_busy_ns, 200.0, "S1 port0 port_busy_ns");
    expect_eq_d(p0.shared_busy_ns, 200.0, "S1 port0 shared_busy_ns");
    expect_eq_d(p0.bytes_served, 1200.0, "S1 port0 bytes_served");
    expect_eq_u64(p1.issued_count, 1, "S1 port1 issued");
    expect_eq_u64(p1.peak_streaming, 1, "S1 port1 peak_streaming");
    expect_eq_d(p1.port_busy_ns, 100.0, "S1 port1 port_busy_ns");
    expect_eq_d(p1.bytes_served, 600.0, "S1 port1 bytes_served");
    expect_true(out.drained, "S1 is_drained");
    expect_port_conserved(p0, spec.bw, "S1 port0");
    expect_port_conserved(p1, spec.bw, "S1 port1");
    std::printf("[remote_port_nway_test] S1 anchor-A/PER_NPU/isolation PASS\n");
}

// ---- S2: anchor B + MEMORY_POOL mapping (all ranks -> one logical port) ---
// bw=100 B/ns, latency=50 ns, four flows 1k/2k/3k/4 kB, continuous
// work-conserving re-split: segments 4-way@25 [50,90], 3-way@(100/3)
// [90,120], 2-way@50 [120,140], 1-way@100 [140,150] -> 90/120/140/150.
void scenario_s2() {
    ScenarioSpec spec;
    spec.name = "s2_anchor_pool";
    spec.memory_type = "MEMORY_POOL";
    spec.bw = 100.0;
    spec.latency_ns = 50.0;
    spec.num_ranks = 4;
    spec.per_rank_nodes = {
        {{0, 1000, {}, false}},
        {{0, 2000, {}, false}},
        {{0, 3000, {}, false}},
        {{0, 4000, {}, false}},
    };

    const RunOutput out = run_scenario(spec);
    expect_eq_u64(out.terminals.size(), 4, "S2 terminal count");
    expect_eq_u64(terminal_of(out.terminals, 0, 0), 90, "S2 anchor: 1 kB @90");
    expect_eq_u64(terminal_of(out.terminals, 1, 0), 120,
                  "S2 anchor: 2 kB @120");
    expect_eq_u64(terminal_of(out.terminals, 2, 0), 140,
                  "S2 anchor: 3 kB @140");
    expect_eq_u64(terminal_of(out.terminals, 3, 0), 150,
                  "S2 anchor: 4 kB @150");
    // Completion order equals flow order (each in its own transition batch).
    expect_eq_u64(out.terminals[0].rank, 0, "S2 delivery order 0");
    expect_eq_u64(out.terminals[1].rank, 1, "S2 delivery order 1");
    expect_eq_u64(out.terminals[2].rank, 2, "S2 delivery order 2");
    expect_eq_u64(out.terminals[3].rank, 3, "S2 delivery order 3");

    expect_eq_u64(out.port_count, 1, "S2 MEMORY_POOL port count");
    const auto& p0 = out.ports[0];
    expect_eq_u64(p0.issued_count, 4, "S2 issued");
    expect_eq_u64(p0.issued_bytes, 10000, "S2 issued bytes");
    expect_eq_u64(p0.completed_bytes, 10000, "S2 completed bytes");
    expect_eq_u64(p0.peak_in_flight, 4,
                  "S2 peak_in_flight (all ranks on one port)");
    expect_eq_u64(p0.peak_streaming, 4, "S2 peak_streaming");
    // Re-split counts: completions with survivors at 90 (3 survivors), 120
    // (2), 140 (1); the 150 completion leaves none.  Joins at 50 start from
    // an empty set -> no join-driven share change.
    expect_eq_u64(p0.redistribution_events, 3, "S2 redistribution_events");
    expect_eq_u64(p0.stream_join_events, 0, "S2 stream_join_events");
    expect_eq_d(p0.port_busy_ns, 100.0, "S2 port_busy_ns (50..150)");
    expect_eq_d(p0.shared_busy_ns, 90.0, "S2 shared_busy_ns (50..140)");
    expect_eq_d(p0.bytes_served, 10000.0, "S2 bytes_served (work-conserving)");
    expect_true(out.drained, "S2 is_drained");
    expect_port_conserved(p0, spec.bw, "S2 port0");
    std::printf("[remote_port_nway_test] S2 anchor-B/MEMORY_POOL PASS\n");
}

// ---- S3: PER_NODE mapping + 2-flow long/short completion re-split ---------
// port0 = ranks {0,1}: 600 B and 1200 B both stream at 3 B/ns from 100;
// short completes at 300, survivor re-splits to full 6 B/ns -> long at 400
// (500 would be the no-re-split FIFO-fluid value).  port1 = ranks {2,3}:
// rank2's 600 B finishes solo at 200 while port0 is mid-flight.
void scenario_s3() {
    ScenarioSpec spec;
    spec.name = "s3_per_node_resplit";
    spec.memory_type = "PER_NODE_MEMORY_EXPANSION";
    spec.bw = 6.0;
    spec.latency_ns = 100.0;
    spec.num_nodes = 2;
    spec.npus_per_node = 2;
    spec.num_ranks = 4;
    spec.per_rank_nodes = {
        {{0, 600, {}, false}},
        {{0, 1200, {}, false}},
        {{0, 600, {}, false}},
        {{100, 0, {}, true}},  // idle rank: 1 ns replay COMP filler
    };

    const RunOutput out = run_scenario(spec);
    expect_eq_u64(out.terminals.size(), 4, "S3 terminal count");
    expect_eq_u64(terminal_of(out.terminals, 0, 0), 300,
                  "S3: short 600 B callback t=300");
    expect_eq_u64(terminal_of(out.terminals, 1, 0), 400,
                  "S3: long 1200 B callback t=400 (completion re-split; "
                  "without it the tail would end at 500)");
    expect_eq_u64(terminal_of(out.terminals, 2, 0), 200,
                  "S3: rank2 port-isolated 600 B callback t=200");
    expect_eq_u64(terminal_of(out.terminals, 3, 100), 1,
                  "S3: idle-rank replay filler");

    expect_eq_u64(out.port_count, 2, "S3 PER_NODE port count");
    const auto& p0 = out.ports[0];
    const auto& p1 = out.ports[1];
    expect_eq_u64(p0.issued_count, 2, "S3 port0 issued");
    expect_eq_u64(p0.issued_bytes, 1800, "S3 port0 issued bytes");
    expect_eq_u64(p0.peak_in_flight, 2, "S3 port0 peak_in_flight");
    expect_eq_u64(p0.peak_streaming, 2, "S3 port0 peak_streaming");
    expect_eq_u64(p0.redistribution_events, 1,
                  "S3 port0: one completion-driven re-split at 300");
    expect_eq_d(p0.port_busy_ns, 300.0, "S3 port0 port_busy_ns (100..400)");
    expect_eq_d(p0.shared_busy_ns, 200.0, "S3 port0 shared_busy_ns");
    expect_eq_d(p0.bytes_served, 1800.0, "S3 port0 bytes_served");
    expect_eq_u64(p1.issued_count, 1, "S3 port1 issued");
    expect_eq_d(p1.port_busy_ns, 100.0, "S3 port1 port_busy_ns");
    expect_eq_d(p1.bytes_served, 600.0, "S3 port1 bytes_served");
    expect_true(out.drained, "S3 is_drained");
    expect_port_conserved(p0, spec.bw, "S3 port0");
    expect_port_conserved(p1, spec.bw, "S3 port1");
    std::printf("[remote_port_nway_test] S3 PER_NODE/long-short resplit PASS\n");
}

// ---- S4: latency-ready joins exactly at a stream-completion instant -------
// PER_NODE port0 = ranks {0,1}, bw=6, latency=100.
//   X (600 B, rank0) and C (1200 B, rank1) issue at 0 -> both 3 B/ns from
//   100; X completes at 300 and its chained successor B (600 B) is issued at
//   tick 300 (staggered latency window [300,400] overlapping C's stream).
//   C streams SOLO at 6 B/ns over [300,400] (latency eats no bandwidth) and
//   its fluid completion instant is exactly 400 == B's latency-ready instant:
//   one flip handles C's exhaustion and B's join together.  B then streams
//   solo [400,500] -> 500.
void scenario_s4() {
    ScenarioSpec spec;
    spec.name = "s4_latency_ready_collision";
    spec.memory_type = "PER_NODE_MEMORY_EXPANSION";
    spec.bw = 6.0;
    spec.latency_ns = 100.0;
    spec.num_nodes = 2;
    spec.npus_per_node = 2;
    spec.num_ranks = 2;
    spec.per_rank_nodes = {
        {{0, 600, {}, false}, {1, 600, {0}, false}},
        {{0, 1200, {}, false}},
    };

    const RunOutput out = run_scenario(spec);
    expect_eq_u64(out.terminals.size(), 3, "S4 terminal count");
    expect_eq_u64(terminal_of(out.terminals, 0, 0), 300, "S4: X callback 300");
    expect_eq_u64(terminal_of(out.terminals, 1, 0), 400,
                  "S4: C callback 400 (unshaken by B's join at the same "
                  "instant)");
    expect_eq_u64(terminal_of(out.terminals, 0, 1), 500,
                  "S4: B callback 500 (solo tail after joining at 400)");

    const auto& p0 = out.ports[0];
    expect_eq_u64(p0.issued_count, 3, "S4 issued");
    expect_eq_u64(p0.issued_bytes, 2400, "S4 issued bytes");
    expect_eq_u64(p0.peak_in_flight, 2,
                  "S4 peak_in_flight (X settles at 300 before its chained "
                  "successor B issues; C spans the whole window)");
    expect_eq_u64(p0.peak_streaming, 2,
                  "S4 peak_streaming (B joins only as C leaves)");
    expect_eq_u64(p0.redistribution_events, 2,
                  "S4: re-splits at 300 (X done, C survives) and at 400 "
                  "(C done, B joined at the same instant)");
    expect_eq_u64(p0.stream_join_events, 0,
                  "S4: join at 400 coincides with a completion, not a pure "
                  "share change");
    expect_eq_d(p0.port_busy_ns, 400.0, "S4 port_busy_ns (100..500)");
    expect_eq_d(p0.shared_busy_ns, 200.0, "S4 shared_busy_ns (100..300)");
    expect_eq_d(p0.bytes_served, 2400.0, "S4 bytes_served");
    expect_true(out.drained, "S4 is_drained");
    expect_port_conserved(p0, spec.bw, "S4 port0");
    std::printf(
        "[remote_port_nway_test] S4 latency-ready==completion instant PASS\n");
}

// ---- S5: same-Tick multi-port / multi-job delivery ordering ---------------
// PER_NPU, bw=6, latency=100: port0's two 600 B flows (issue_seq 1,2) and
// port1's solo 1200 B flow all complete at t=300.  One dispatch batch must
// deliver in (port_index asc, issue_seq asc) order: r0#0, r0#1, r1#0.
void scenario_s5() {
    ScenarioSpec spec;
    spec.name = "s5_same_tick_order";
    spec.memory_type = "PER_NPU_MEMORY_EXPANSION";
    spec.bw = 6.0;
    spec.latency_ns = 100.0;
    spec.npu_ids = {0, 1};
    spec.num_ranks = 2;
    spec.per_rank_nodes = {
        {{0, 600, {}, false}, {1, 600, {}, false}},
        {{0, 1200, {}, false}},
    };

    const RunOutput out = run_scenario(spec);
    expect_eq_u64(out.terminals.size(), 3, "S5 terminal count");
    expect_eq_u64(terminal_of(out.terminals, 0, 0), 300, "S5 r0#0 @300");
    expect_eq_u64(terminal_of(out.terminals, 0, 1), 300, "S5 r0#1 @300");
    expect_eq_u64(terminal_of(out.terminals, 1, 0), 300, "S5 r1#0 @300");

    // Exact batch delivery order at the shared Tick.
    const auto batch = terminals_at(out.terminals, 300);
    expect_eq_u64(batch.size(), 3, "S5 same-Tick batch size");
    expect_true(batch[0].first == std::make_pair(0, (uint64_t)0) &&
                    batch[1].first == std::make_pair(0, (uint64_t)1) &&
                    batch[2].first == std::make_pair(1, (uint64_t)0),
                "S5 batch order must be (port0,seq1), (port0,seq2), "
                "(port1,seq1)");

    const auto& p0 = out.ports[0];
    const auto& p1 = out.ports[1];
    expect_eq_u64(p0.completed_count, 2, "S5 port0 completed");
    expect_eq_u64(p1.completed_count, 1, "S5 port1 completed");
    expect_eq_u64(p1.peak_streaming, 1, "S5 port1 solo (1200 B @6 -> 200 ns)");
    expect_eq_d(p0.port_busy_ns, 200.0, "S5 port0 port_busy_ns");
    expect_eq_d(p1.port_busy_ns, 200.0, "S5 port1 port_busy_ns");
    expect_true(out.drained, "S5 is_drained");
    expect_port_conserved(p0, spec.bw, "S5 port0");
    expect_port_conserved(p1, spec.bw, "S5 port1");
    std::printf(
        "[remote_port_nway_test] S5 same-Tick (port,seq) ordering PASS\n");
}

// ---- S6: single non-divisible transaction + non-divisible pair ------------
// PER_NPU, bw=6, latency=100.  port0: 100 B solo -> fluid 100 + 100/6 =
// 116.67 -> callback ceil = 117.  port1: 100 B + 200 B share 3 B/ns from
// 100; the short's fluid is 133.33 (callback 134); the survivor re-splits to
// 6 B/ns with a 99.99.. B residual and lands on fluid 150.0 (callback 150).
void scenario_s6() {
    ScenarioSpec spec;
    spec.name = "s6_non_divisible";
    spec.memory_type = "PER_NPU_MEMORY_EXPANSION";
    spec.bw = 6.0;
    spec.latency_ns = 100.0;
    spec.npu_ids = {0, 1};
    spec.num_ranks = 2;
    spec.per_rank_nodes = {
        {{0, 100, {}, false}},
        {{0, 100, {}, false}, {1, 200, {}, false}},
    };

    const RunOutput out = run_scenario(spec);
    expect_eq_u64(out.terminals.size(), 3, "S6 terminal count");
    expect_eq_u64(terminal_of(out.terminals, 0, 0), 117,
                  "S6: single 100 B -> ceil(116.67) = 117");
    expect_eq_u64(terminal_of(out.terminals, 1, 0), 134,
                  "S6: shared short 100 B -> ceil(133.33) = 134");
    expect_eq_u64(terminal_of(out.terminals, 1, 1), 150,
                  "S6: shared long 200 B -> fluid 150.0 after residual "
                  "re-split, callback 150");

    const auto& p0 = out.ports[0];
    const auto& p1 = out.ports[1];
    expect_eq_d(p0.port_busy_ns, 100.0 / 6.0, "S6 port0 port_busy (100/6 ns)");
    expect_eq_d(p0.bytes_served, 100.0, "S6 port0 bytes_served");
    expect_eq_d(p1.port_busy_ns, 50.0, "S6 port1 port_busy (33.33 + 16.67)");
    expect_eq_d(p1.shared_busy_ns, 100.0 / 3.0, "S6 port1 shared_busy");
    expect_eq_d(p1.bytes_served, 300.0, "S6 port1 bytes_served");
    expect_eq_u64(p1.redistribution_events, 1,
                  "S6 port1: re-split at the short's fluid completion");
    expect_true(out.drained, "S6 is_drained");
    expect_port_conserved(p0, spec.bw, "S6 port0");
    expect_port_conserved(p1, spec.bw, "S6 port1");
    std::printf("[remote_port_nway_test] S6 non-divisible/residual PASS\n");
}

// ---- S7: 1 B @ 6 B/ns -- fluid sub-Tick time vs integer callback Tick -----
// latency 0: the 1 B flow streams at t=0 and exhausts at ~1/6 ns; the
// observable Workload callback lands on the NEXT integer Tick (1).  Telemetry
// (§5.1) proves fluid_finish_ns ~ 1/6 while callback_tick == 1, and the row
// count equals the settled completions.
void scenario_s7() {
    ScenarioSpec spec;
    spec.name = "s7_one_byte_fluid";
    spec.memory_type = "PER_NPU_MEMORY_EXPANSION";
    spec.bw = 6.0;
    spec.latency_ns = 0.0;
    spec.npu_ids = {0};
    spec.num_ranks = 2;
    spec.telemetry = true;
    spec.per_rank_nodes = {
        {{0, 1, {}, false}},
        {{100, 0, {}, true}},  // idle-rank filler
    };

    const RunOutput out = run_scenario(spec);
    expect_eq_u64(out.terminals.size(), 2, "S7 terminal count");
    expect_eq_u64(terminal_of(out.terminals, 0, 0), 1,
                  "S7: 1 B callback at the NEXT Tick (1), not Tick 0");

    const auto& p0 = out.ports[0];
    expect_eq_d(p0.port_busy_ns, 1.0 / 6.0, "S7 port_busy == fluid 1/6 ns",
                1e-9);
    expect_eq_d(p0.bytes_served, 1.0, "S7 bytes_served == 1", 1e-9);
    expect_eq_u64(p0.peak_streaming, 1, "S7 peak_streaming");
    expect_true(out.drained, "S7 is_drained");
    expect_port_conserved(p0, spec.bw, "S7 port0");

    // §5.1 transaction-detail rows: lazy file, one row per settled
    // completion, fluid finish strictly below the callback Tick.
    const std::string telemetry_path =
        out.bridge_dir + "/remote_memory_transactions.jsonl";
    std::ifstream rows(telemetry_path);
    expect_true(rows.is_open(), "S7 telemetry file exists");
    std::string line;
    std::vector<nlohmann::json> parsed;
    while (std::getline(rows, line)) {
        if (!line.empty()) {
            parsed.push_back(nlohmann::json::parse(line));
        }
    }
    expect_eq_u64(parsed.size(), 1, "S7 telemetry row count == completions");
    expect_eq_u64(parsed[0]["schema"].get<uint64_t>(), 1, "S7 row schema");
    expect_eq_u64(parsed[0]["sys_id"].get<uint64_t>(), 0, "S7 row sys_id");
    expect_eq_u64(parsed[0]["node_id"].get<uint64_t>(), 0, "S7 row node_id");
    expect_eq_u64(parsed[0]["issue_sequence"].get<uint64_t>(), 1,
                  "S7 row issue_sequence");
    expect_eq_u64(parsed[0]["port_index"].get<uint64_t>(), 0,
                  "S7 row port_index");
    expect_eq_u64(parsed[0]["bytes"].get<uint64_t>(), 1, "S7 row bytes");
    expect_eq_u64(parsed[0]["issue_tick"].get<uint64_t>(), 0,
                  "S7 row issue_tick");
    expect_eq_u64(parsed[0]["callback_tick"].get<uint64_t>(), 1,
                  "S7 row callback_tick");
    expect_eq_d(parsed[0]["fluid_finish_ns"].get<double>(), 1.0 / 6.0,
                "S7 row fluid_finish_ns ~ 1/6 ns", 1e-9);
    expect_eq_d(parsed[0]["stream_start_ns"].get<double>(), 0.0,
                "S7 row stream_start_ns");
    std::printf(
        "[remote_port_nway_test] S7 1B fluid(~1/6ns)/callback(Tick 1) PASS\n");
}

// ---- S8: zero-byte positive-latency never joins the denominator -----------
// bw=6, latency=100: the 0 B transaction completes at ceil(100)=100 and must
// never consume bandwidth -- its 600 B partner streams SOLO at 6 B/ns and
// finishes at 200 (a shared denominator would push it to 300).
void scenario_s8() {
    ScenarioSpec spec;
    spec.name = "s8_zero_byte_positive_latency";
    spec.memory_type = "PER_NPU_MEMORY_EXPANSION";
    spec.bw = 6.0;
    spec.latency_ns = 100.0;
    spec.npu_ids = {0};
    spec.num_ranks = 2;
    spec.per_rank_nodes = {
        {{0, 0, {}, false}, {1, 600, {}, false}},
        {{100, 0, {}, true}},  // idle-rank filler
    };

    const RunOutput out = run_scenario(spec);
    expect_eq_u64(out.terminals.size(), 3, "S8 terminal count");
    expect_eq_u64(terminal_of(out.terminals, 0, 0), 100,
                  "S8: 0 B + 100 ns latency -> callback at ceil(100) = 100");
    expect_eq_u64(terminal_of(out.terminals, 0, 1), 200,
                  "S8: 600 B partner streams SOLO (0 B absent from the "
                  "denominator) -> 200, not 300");

    const auto& p0 = out.ports[0];
    expect_eq_u64(p0.issued_count, 2, "S8 issued");
    expect_eq_u64(p0.issued_bytes, 600, "S8 issued bytes (0 B contributes 0)");
    expect_eq_u64(p0.peak_in_flight, 2, "S8 peak_in_flight");
    expect_eq_u64(p0.peak_streaming, 1,
                  "S8 peak_streaming == 1: the 0 B job never streams");
    expect_eq_u64(p0.redistribution_events, 0, "S8 redistribution_events");
    expect_eq_d(p0.port_busy_ns, 100.0, "S8 port_busy (100..200)");
    expect_eq_d(p0.bytes_served, 600.0, "S8 bytes_served");
    expect_true(out.drained, "S8 is_drained");
    expect_port_conserved(p0, spec.bw, "S8 port0");
    std::printf(
        "[remote_port_nway_test] S8 zero-byte positive-latency PASS\n");
}

// ---- S9: dual-zero + same-Tick dual-zero/streaming ordering + §3.3 keep ---
// bw=6, latency=0, PER_NPU 2 ranks.  rank0: id0 = 0 B/0 ns -> independent
// one-shot timer at +1 ns (NOT the old same-Tick completion); id1 = 6 B
// dep-free (fluid 1 -> transition event queued at Tick 1); id2 = 6 B chained
// after id0.  At Tick 1 the dual-zero timer (registered first) fires, and its
// callback chains id2's issue INTO the same Tick while the streaming
// transition for id1 is still queued: the §3.3 keep rule must retain that
// event so id1's callback is NOT deferred past its completion Tick.
// rank1: 600 B solo -> 100 (port-capacity companion).
void scenario_s9() {
    ScenarioSpec spec;
    spec.name = "s9_dual_zero_keep_rule";
    spec.memory_type = "PER_NPU_MEMORY_EXPANSION";
    spec.bw = 6.0;
    spec.latency_ns = 0.0;
    spec.npu_ids = {0, 1};
    spec.num_ranks = 2;
    spec.per_rank_nodes = {
        {{0, 0, {}, false}, {1, 6, {}, false}, {2, 6, {0}, false}},
        {{0, 600, {}, false}},
    };

    const RunOutput out = run_scenario(spec);
    expect_eq_u64(out.terminals.size(), 4, "S9 terminal count");
    expect_eq_u64(terminal_of(out.terminals, 0, 0), 1,
                  "S9: dual-zero delivered at +1 ns (Tick 1), not same-Tick");
    expect_eq_u64(terminal_of(out.terminals, 0, 1), 1,
                  "S9: 6 B streaming completion kept its Tick-1 callback "
                  "despite the mid-Tick chained issue (§3.3 keep rule)");
    expect_eq_u64(terminal_of(out.terminals, 0, 2), 2,
                  "S9: chained 6 B streams solo [1,2] -> 2");
    expect_eq_u64(terminal_of(out.terminals, 1, 0), 100,
                  "S9: rank1 600 B solo -> 100");

    // Same-Tick delivery order on rank0: the dual-zero timer was registered
    // before the transition event, so its callback precedes id1's.
    const auto batch = terminals_at(out.terminals, 1);
    expect_eq_u64(batch.size(), 2, "S9 Tick-1 batch size");
    expect_true(batch[0].first == std::make_pair(0, (uint64_t)0) &&
                    batch[1].first == std::make_pair(0, (uint64_t)1),
                "S9 Tick-1 order: dual-zero job before the streaming "
                "completion");

    const auto& p0 = out.ports[0];
    expect_eq_u64(p0.issued_count, 3, "S9 port0 issued");
    expect_eq_u64(p0.issued_bytes, 12, "S9 port0 bytes (0 B contributes 0)");
    expect_eq_u64(p0.peak_in_flight, 2, "S9 port0 peak_in_flight");
    expect_eq_u64(p0.peak_streaming, 1,
                  "S9 port0: the two 6 B streams never overlap (id2 joins as "
                  "id1 leaves at the same instant)");
    expect_eq_d(p0.port_busy_ns, 2.0, "S9 port0 port_busy ([0,1] + [1,2])");
    expect_eq_u64(p0.redistribution_events, 0,
                  "S9 port0: id1 exhausts in the issue-advance sweep BEFORE "
                  "the chained id2 joins at the dispatch sweep, and no two "
                  "streams ever actually share -- no share change occurs "
                  "(§5.1: one count per instant with survivors)");
    const auto& p1 = out.ports[1];
    expect_eq_d(p1.port_busy_ns, 100.0, "S9 port1 port_busy");
    expect_true(out.drained, "S9 is_drained");
    expect_port_conserved(p0, spec.bw, "S9 port0");
    expect_port_conserved(p1, spec.bw, "S9 port1");
    std::printf(
        "[remote_port_nway_test] S9 dual-zero/same-Tick keep-rule PASS\n");
}

//****************************************************************************
// S10: fail-closed children.  The backend enforces its invariants with
// exit(1) (never exceptions: Sys::call_events swallows std::exception), so
// every case runs in a forked child and the parent requires a clean
// exit status 1.  JSON has no NaN literal; the JSON-representable non-finite
// is overflow to +inf ("1e999"), which nlohmann may reject at parse time
// (abort) -- both paths are fail-closed, and for those two cases alone the
// parent accepts either a clean exit 1 or a signal death, recording which.
//****************************************************************************

struct FailCaseResult {
    bool clean_exit1 = false;
    bool signaled = false;
};

FailCaseResult run_fail_child(const std::function<void()>& body,
                              bool quiet_stderr) {
    std::fflush(nullptr);  // flush parent buffers before fork
    const pid_t pid = fork();
    if (pid < 0) {
        fail("fork failed for a fail-closed child");
    }
    if (pid == 0) {
        if (quiet_stderr) {
            if (freopen("/dev/null", "w", stderr) == nullptr) {
                _exit(2);
            }
        }
        body();  // must exit(1) from inside; reaching the end means survive
        _exit(0);
    }
    int status = 0;
    if (waitpid(pid, &status, 0) < 0) {
        fail("waitpid failed for a fail-closed child");
    }
    FailCaseResult res;
    if (WIFEXITED(status)) {
        res.clean_exit1 = (WEXITSTATUS(status) == 1);
    } else if (WIFSIGNALED(status)) {
        res.signaled = true;
    }
    return res;
}

// Writes a remote_memory.json and constructs the backend in the child.
void expect_config_rejected(const std::string& tag, const std::string& body) {
    const std::string dir = scenario_dir("s10_fail_" + tag);
    std::error_code ec;
    std::filesystem::create_directories(dir, ec);
    const std::string path = dir + "/remote_memory.json";
    write_text(path, body);
    const FailCaseResult res = run_fail_child(
        [&path]() {
            AnalyticalRemoteMemory memory(path);
            // Surviving construction is a failure.
        },
        true);
    if (!(res.clean_exit1 || res.signaled)) {
        fail("fail-closed config case '" + tag +
             "' was NOT rejected (child survived)");
    }
    if (res.signaled) {
        std::printf("[remote_port_nway_test] S10 '%s' rejected (JSON parse "
                    "abort; fail-closed)\n",
                    tag.c_str());
    } else {
        std::printf("[remote_port_nway_test] S10 '%s' rejected (exit 1)\n",
                    tag.c_str());
    }
}

// Constructs a VALID backend in the child, then performs a runtime action
// that must hit a fail-closed fatal (exit 1).
void expect_runtime_rejected(const std::string& tag,
                             const std::string& config_body,
                             const std::function<void(
                                 AnalyticalRemoteMemory&)>& action) {
    const std::string dir = scenario_dir("s10_fail_" + tag);
    std::error_code ec;
    std::filesystem::create_directories(dir, ec);
    const std::string path = dir + "/remote_memory.json";
    write_text(path, config_body);
    const FailCaseResult res = run_fail_child(
        [&path, &action]() {
            AnalyticalRemoteMemory memory(path);
            action(memory);
        },
        true);
    if (!res.clean_exit1) {
        fail("fail-closed runtime case '" + tag +
             "' did not exit 1 (child survived or died abnormally)");
    }
    std::printf("[remote_port_nway_test] S10 '%s' rejected (exit 1)\n",
                tag.c_str());
}

void scenario_s10() {
    // --- constructor-time config rejections (§3.2 / PER_NODE fail-closed) --
    expect_config_rejected(
        "bw_inf", R"({"memory-type": "PER_NPU_MEMORY_EXPANSION",
                      "remote-mem-bw": 1e999, "remote-mem-latency": 100,
                      "npu-ids": [0]})");
    expect_config_rejected(
        "bw_negative", R"({"memory-type": "PER_NPU_MEMORY_EXPANSION",
                           "remote-mem-bw": -6, "remote-mem-latency": 100,
                           "npu-ids": [0]})");
    expect_config_rejected(
        "bw_zero", R"({"memory-type": "PER_NPU_MEMORY_EXPANSION",
                       "remote-mem-bw": 0, "remote-mem-latency": 100,
                       "npu-ids": [0]})");
    expect_config_rejected(
        "bw_string", R"({"memory-type": "PER_NPU_MEMORY_EXPANSION",
                         "remote-mem-bw": "abc", "remote-mem-latency": 100,
                         "npu-ids": [0]})");
    expect_config_rejected(
        "bw_missing", R"({"memory-type": "PER_NPU_MEMORY_EXPANSION",
                          "remote-mem-latency": 100, "npu-ids": [0]})");
    expect_config_rejected(
        "latency_negative", R"({"memory-type": "PER_NPU_MEMORY_EXPANSION",
                                "remote-mem-bw": 6,
                                "remote-mem-latency": -5, "npu-ids": [0]})");
    expect_config_rejected(
        "latency_inf", R"({"memory-type": "PER_NPU_MEMORY_EXPANSION",
                           "remote-mem-bw": 6, "remote-mem-latency": 1e999,
                           "npu-ids": [0]})");
    expect_config_rejected(
        "latency_string", R"({"memory-type": "PER_NPU_MEMORY_EXPANSION",
                              "remote-mem-bw": 6,
                              "remote-mem-latency": "soon", "npu-ids": [0]})");
    expect_config_rejected(
        "per_node_missing_shape",
        R"({"memory-type": "PER_NODE_MEMORY_EXPANSION", "remote-mem-bw": 6})");
    expect_config_rejected(
        "per_node_zero_nodes",
        R"({"memory-type": "PER_NODE_MEMORY_EXPANSION", "remote-mem-bw": 6,
            "num-nodes": 0, "num-npus-per-node": 2})");
    expect_config_rejected(
        "bogus_memory_type", R"({"memory-type": "BOGUS_TYPE",
                                 "remote-mem-bw": 6})");

    // --- runtime fail-closed (§3.2 huge values / NO_MEMORY / unbound rank) -
    const std::string valid_per_npu =
        R"({"memory-type": "PER_NPU_MEMORY_EXPANSION", "remote-mem-bw": 6,
            "remote-mem-latency": 100, "npu-ids": [0]})";
    // tensor_size beyond the double-exact range (2^53) must never reach the
    // fluid core; the fatal fires before port/sys resolution, so no Sys is
    // needed in the child.
    expect_runtime_rejected(
        "huge_tensor_size", valid_per_npu,
        [](AnalyticalRemoteMemory& memory) {
            WorkloadLayerHandlerData wlhd;
            wlhd.sys_id = 0;
            wlhd.workload = nullptr;
            wlhd.node_id = 0;
            memory.issue((1ULL << 53) + 1, &wlhd);
        });
    // NO_MEMORY_EXPANSION keeps fail-closed issue behavior.
    expect_runtime_rejected(
        "no_memory_issue",
        R"({"memory-type": "NO_MEMORY_EXPANSION"})",
        [](AnalyticalRemoteMemory& memory) {
            WorkloadLayerHandlerData wlhd;
            wlhd.sys_id = 0;
            wlhd.workload = nullptr;
            wlhd.node_id = 0;
            memory.issue(100, &wlhd);
        });
    // A PER_NPU rank without a configured port must fail closed.
    expect_runtime_rejected(
        "per_npu_unbound_rank",
        R"({"memory-type": "PER_NPU_MEMORY_EXPANSION", "remote-mem-bw": 6,
            "remote-mem-latency": 100, "npu-ids": [5]})",
        [](AnalyticalRemoteMemory& memory) {
            WorkloadLayerHandlerData wlhd;
            wlhd.sys_id = 7;  // not in npu-ids
            wlhd.workload = nullptr;
            wlhd.node_id = 0;
            memory.issue(100, &wlhd);
        });
    std::printf("[remote_port_nway_test] S10 fail-closed (14 cases) PASS\n");
}

//****************************************************************************
// --stress mode (阶段 4/5.1 pressure fixture): multi-port, large flow count.
//****************************************************************************

constexpr int kStressPorts = 16;
constexpr int kStressFlowsPerPort = 128;

int run_stress() {
    std::error_code ec;
    std::filesystem::create_directories(kTmpBase, ec);

    ScenarioSpec spec;
    spec.name = "stress";
    spec.memory_type = "PER_NPU_MEMORY_EXPANSION";
    spec.bw = 100.0;
    spec.latency_ns = 50.0;
    spec.num_ranks = kStressPorts;
    for (int r = 0; r < kStressPorts; ++r) {
        spec.npu_ids.push_back(static_cast<uint64_t>(r));
    }
    // Deterministic LCG sizes: varied -> completion instants spread across
    // many continuous transitions (re-splits) instead of aliasing onto one
    // Tick.
    uint64_t lcg = 0x9E3779B97F4A7C15ULL;
    for (int r = 0; r < kStressPorts; ++r) {
        std::vector<MemNodeSpec> nodes;
        for (int i = 0; i < kStressFlowsPerPort; ++i) {
            lcg = lcg * 6364136223846793005ULL + 1442695040888963407ULL;
            nodes.push_back({static_cast<uint64_t>(i),
                             64 + (lcg % 8192),
                             {},
                             false});
        }
        spec.per_rank_nodes.push_back(std::move(nodes));
    }

    const auto wall_start = std::chrono::steady_clock::now();
    timespec cpu_start{};
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &cpu_start);
    struct rusage usage_start {};
    getrusage(RUSAGE_SELF, &usage_start);

    const RunOutput out = run_scenario(spec);

    timespec cpu_end{};
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &cpu_end);
    const auto wall_end = std::chrono::steady_clock::now();
    struct rusage usage_end {};
    getrusage(RUSAGE_SELF, &usage_end);

    const double wall_ns =
        std::chrono::duration<double, std::nano>(wall_end - wall_start)
            .count();
    const double cpu_ns =
        static_cast<double>(cpu_end.tv_sec - cpu_start.tv_sec) * 1e9 +
        static_cast<double>(cpu_end.tv_nsec - cpu_start.tv_nsec);
    const long peak_rss_kib = usage_end.ru_maxrss;  // Linux: KiB, peak

    // ---- correctness under load ----
    const uint64_t expected_transactions =
        static_cast<uint64_t>(kStressPorts) * kStressFlowsPerPort;
    expect_eq_u64(out.terminals.size(), expected_transactions,
                  "stress terminal count");
    expect_eq_u64(out.port_count,
                  static_cast<std::size_t>(kStressPorts),
                  "stress port count");
    uint64_t sum_issued_bytes = 0;
    uint64_t sum_redistribution = 0;
    uint64_t sum_join = 0;
    uint64_t max_peak_streaming = 0;
    for (std::size_t p = 0; p < out.port_count; ++p) {
        const auto& s = out.ports[p];
        expect_eq_u64(s.issued_count, kStressFlowsPerPort,
                      "stress per-port issued");
        expect_port_conserved(s, spec.bw,
                              "stress port " + std::to_string(p));
        sum_issued_bytes += s.issued_bytes;
        sum_redistribution += s.redistribution_events;
        sum_join += s.stream_join_events;
        if (s.peak_streaming > max_peak_streaming) {
            max_peak_streaming = s.peak_streaming;
        }
    }
    expect_true(out.drained, "stress is_drained");
    // Every flow completed exactly once, in order-preserving delivery.
    if (out.terminals.size() != expected_transactions) {
        fail("stress delivery count mismatch");
    }

    std::ofstream report(kStressResultsPath, std::ios::trunc);
    if (!report.is_open()) {
        fail(std::string("cannot open ") + kStressResultsPath);
    }
    report << "remote_port_nway_test --stress results\n";
    report << "ports=" << kStressPorts << "\n";
    report << "flows_per_port=" << kStressFlowsPerPort << "\n";
    report << "transactions_issued=" << out.total_issued << "\n";
    report << "transactions_completed=" << out.total_completed << "\n";
    report << "total_issued_bytes=" << sum_issued_bytes << "\n";
    report << "max_active_streams_peak=" << max_peak_streaming << "\n";
    report << "redistribution_events_total=" << sum_redistribution << "\n";
    report << "stream_join_events_total=" << sum_join << "\n";
    report << "transition_proxy_total="
           << (sum_redistribution + sum_join) << "\n";
    report << "cpu_time_ns=" << static_cast<uint64_t>(cpu_ns) << "\n";
    report << "wall_time_ns=" << static_cast<uint64_t>(wall_ns) << "\n";
    report << "peak_rss_kib=" << peak_rss_kib << "\n";
    report.close();
    if (!report.good()) {
        fail("failed writing stress results");
    }

    std::printf(
        "[remote_port_nway_test] stress: %d ports x %d flows, "
        "peak_streaming=%llu, transitions(redist+join)=%llu, "
        "cpu=%.3f ms, wall=%.3f ms, peak_rss=%ld KiB -> %s\n",
        kStressPorts, kStressFlowsPerPort,
        static_cast<unsigned long long>(max_peak_streaming),
        static_cast<unsigned long long>(sum_redistribution + sum_join),
        cpu_ns / 1e6, wall_ns / 1e6, peak_rss_kib, kStressResultsPath);
    std::printf("[remote_port_nway_test] --stress ALL PASS\n");
    return 0;
}

}  // namespace

//****************************************************************************
// Entry: default = precision suite (S1..S10, each scenario in a forked child
// for full static-state isolation); --stress = pressure mode.
//****************************************************************************

namespace {
// Fork wrapper: the child runs one scenario function; any fail() inside it
// exits 1.  Forking per scenario gives each static-ET simulation a pristine
// process image (fresh Sys registry / ETFeeder statics / observer hook).
void run_scenario_isolated(const char* name, void (*fn)()) {
    std::fflush(nullptr);
    const pid_t pid = fork();
    if (pid < 0) {
        fail(std::string("fork failed for scenario ") + name);
    }
    if (pid == 0) {
        fn();
        std::exit(0);
    }
    int status = 0;
    if (waitpid(pid, &status, 0) < 0) {
        fail(std::string("waitpid failed for scenario ") + name);
    }
    if (WIFSIGNALED(status)) {
        fail(std::string("scenario ") + name + " died by signal " +
             std::to_string(WTERMSIG(status)));
    }
    if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
        fail(std::string("scenario ") + name + " exited with status " +
             std::to_string(WIFEXITED(status) ? WEXITSTATUS(status) : -1));
    }
}
}  // namespace

int main(int argc, char* argv[]) {
    std::error_code ec;
    std::filesystem::create_directories(kTmpBase, ec);

    if (argc > 1 && std::strcmp(argv[1], "--stress") == 0) {
        std::fflush(nullptr);
        const pid_t pid = fork();
        if (pid < 0) {
            fail("fork failed for the stress run");
        }
        if (pid == 0) {
            std::exit(run_stress());
        }
        int status = 0;
        if (waitpid(pid, &status, 0) < 0) {
            fail("waitpid failed for the stress run");
        }
        if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
            fail("stress run failed");
        }
        return 0;
    }

    // S10 first: pure config/runtime rejections, no simulation state around.
    run_scenario_isolated("s10_fail_closed", &scenario_s10);

    run_scenario_isolated("s1", &scenario_s1);
    run_scenario_isolated("s2", &scenario_s2);
    run_scenario_isolated("s3", &scenario_s3);
    run_scenario_isolated("s4", &scenario_s4);
    run_scenario_isolated("s5", &scenario_s5);
    run_scenario_isolated("s6", &scenario_s6);
    run_scenario_isolated("s7", &scenario_s7);
    run_scenario_isolated("s8", &scenario_s8);
    run_scenario_isolated("s9", &scenario_s9);

    std::printf(
        "[remote_port_nway_test] ALL PASS (precision suite: S1..S10, "
        "PER_NPU/PER_NODE/MEMORY_POOL, anchors 300 | 90/120/140/150 | "
        "1B fluid 1/6 ns)\n");
    return 0;
}
