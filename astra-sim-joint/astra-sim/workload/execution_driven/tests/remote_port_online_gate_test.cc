/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

/**

remote_port_online_gate_test.cc -- ONLINE NodeView/NodeStore system-path
gate fixture for the remote-memory port (SerDes片外链路并发化改造执行方案
V5.3 阶段 5.3; test-only, never part of the production gates).

Proves on the real online issue path (NodeStoreGraphSource / NodeView ->
Workload::issue_dep_free_nodes -> HardwareResource occupy -> backend issue)
that the former comm-single-slot gate no longer serializes remote MEM
emission.  This is NOT an extension of the retired RemoteFifoLedgerTest
(deleted from this repo's working tree; its FIFO-accounting assertions are
obsolete) -- it is a new standalone fixture over the current NodeStore
mechanism layer, in the alarm_cancellation_test.cc online-harness style.

Scenarios (each a full online simulation in a forked child; configs written
by the test itself into isolated dirs under /tmp/serdes-joint-work/;
system.json keeps the official template shape -- preferred-dataset-splits /
scheduling-policy / collective-optimization present, the four
*-implementation keys all ["ring","ring"]; comm_group.json is "{}" so no
dimensionless group can exist):

 A. same-rank two DEPENDENCY-FREE MEM_LOAD nodes (600 B each, PER_NPU
    npu-ids [0,1], remote-mem-bw 6 B/ns, remote-mem-latency 100 ns) plus a
    one-node control rank.  After ONE issue pass, BEFORE the event loop:
      - rank0's port (port0) PortStats: issued_count == 2 AND
        in_flight_count == 2 (both transactions live, latency overlapping),
        latency_waiting_count == 2, streaming_count == 0 (latency consumes
        no bandwidth);
      - the rank0 NodeStore free set is EMPTY (neither MEM was held back by
        a gate -- under the retired comm-single-slot behavior the second MEM
        would still be sitting in the free set);
      - port1 control: issued == 1, in_flight == 1.
    After the simulation drains:
      - terminals: rank0 nodes at t=300 (2-way 3 B/ns split over [100,300];
        the old serial FIFO would give 200/400); rank1 control at 200;
      - same-Tick delivery order on port0: issue_seq order (r0#1, r0#2);
      - interval-integral evidence: peak_streaming == 2 (>= 2) and
        shared_busy_ns == 200 ns (> 0, the exact [100,300] two-stream
        service integral), port_busy_ns == 200, bytes_served == 1200;
      - count/bytes conservation on both ports + is_drained().

 B. independent MEM + COMM_SEND on one rank (PER_NPU [0,1], same port
    physics): rank0 issues MEM_LOAD 600 B AND COMM_SEND (dst rank1, tag 7,
    64 B) dependency-free in the same pass; rank1 holds the matching
    COMM_RECV.  After ONE issue pass:
      - port0 PortStats: issued == 1, in_flight == 1 (the MEM transaction
        is in flight on the port);
      - rank0's NodeStore free set is EMPTY -- the COMM_SEND was issued in
        the SAME pass, not deferred behind the MEM (MEM takes the remote-MEM
        node slot, COMM takes the comm slot: independent gates).
    After the drain:
      - the MEM terminal is exactly 200 (100 ns latency + 600 B at 6 B/ns);
      - the SEND and RECV terminals are strictly EARLIER than 200: the
        transfer completed while the MEM was still in flight.  Under the
        retired comm-single-slot gating the send could not have issued until
        the MEM released the slot at 200, pushing the recv terminal past
        200 -- this inequality is the gate-independence red line;
      - port0 conservation + is_drained().

Build: the CMake target AstraSim_Analytical_Congestion_Aware_RemotePortOnlineGateTest
(same shape as the in-repo online fixtures: frontend congestion_aware shared
sources + execution_driven layer compiled into the test target).
Run: build/.../bin/AstraSim_Analytical_Congestion_Aware_RemotePortOnlineGateTest
Exit code 0 on ALL PASS.
*******************************************************************************/

#include "astra-sim/common/Logging.hh"
#include "astra-sim/system/Sys.hh"
#include "astra-sim/workload/MetricCollector.hh"
#include "astra-sim/workload/Workload.hh"
#include "astra-sim/workload/execution_driven/CompletionObserver.hh"
#include "astra-sim/workload/execution_driven/NodeStore.hh"
#include "congestion_aware/CongestionAwareNetworkApi.hh"
#include "extern/remote_memory_backend/analytical/AnalyticalRemoteMemory.hh"
#include <astra-network-analytical/common/EventQueue.h>
#include <astra-network-analytical/common/NetworkParser.h>
#include <astra-network-analytical/congestion_aware/Helper.h>

#include <sys/wait.h>
#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

using namespace AstraSim;
using namespace AstraSim::ExecutionDriven;
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
    std::fprintf(stderr, "[remote_port_online_gate_test] FAIL: %s\n",
                 what.c_str());
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
// Fixture plumbing (isolated dirs under /tmp/serdes-joint-work/).
//****************************************************************************

const char* kTmpBase = "/tmp/serdes-joint-work";

std::string scenario_dir(const std::string& name) {
    return std::string(kTmpBase) + "/online_gate_" + name;
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

// Official-template-shape system config (H1/H2/H3 避雷: ring/ring everywhere,
// no custom implementations, no collectives in the fixture -> the empty
// comm_group cannot carry a dimensions-less group).
void write_configs(const std::string& dir, double remote_bw,
                   double remote_latency_ns) {
    std::string system = "{\n";
    system += "  \"scheduling-policy\": \"LIFO\",\n";
    system += "  \"endpoint-delay\": 10,\n";
    system += "  \"active-chunks-per-dimension\": 1,\n";
    system += "  \"preferred-dataset-splits\": 6,\n";
    system += "  \"all-reduce-implementation\": [\"ring\", \"ring\"],\n";
    system += "  \"all-gather-implementation\": [\"ring\", \"ring\"],\n";
    system += "  \"reduce-scatter-implementation\": [\"ring\", \"ring\"],\n";
    system += "  \"all-to-all-implementation\": [\"ring\", \"ring\"],\n";
    system += "  \"collective-optimization\": \"localBWAware\",\n";
    system += "  \"roofline-enabled\": 0,\n";
    system += "  \"replay-only\": 0,\n";
    system += "  \"track-local-mem\": 0,\n";
    system += "  \"trace-enabled\": 0,\n";
    system += "  \"hbm-bandwidth-contention\": 0,\n";
    system += "  \"peak-perf\": 1000,\n";
    system += "  \"local-mem-bw\": 1640.0,\n";
    system += "  \"local-mem-latency\": 100,\n";
    system += "  \"remote-mem-bw\": " + std::to_string(remote_bw) + ",\n";
    system += "  \"remote-mem-latency\": " +
              std::to_string(remote_latency_ns) + "\n";
    system += "}\n";
    write_text(dir + "/system.json", system);
    write_text(dir + "/comm_group.json", "{}\n");
    write_text(dir + "/network.yml",
               "# remote_port_online_gate synthetic 2-rank line topology\n"
               "topology: [ Line, Line ]\n"
               "npus_count: [ 2, 1 ]\n"
               "bandwidth: [ 4000.0, 4000.0 ]\n"
               "latency: [ 5, 5 ]\n");
    write_text(dir + "/remote_memory.json",
               "{\n"
               "  \"memory-type\": \"PER_NPU_MEMORY_EXPANSION\",\n"
               "  \"remote-mem-bw\": " +
                   std::to_string(remote_bw) +
                   ",\n"
                   "  \"remote-mem-latency\": " +
                   std::to_string(remote_latency_ns) +
                   ",\n"
                   "  \"npu-ids\": [0, 1]\n"
                   "}\n");
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

//****************************************************************************
// Node builders (plain unannotated MEM -> remote port transaction; comm with
// explicit src/dst/tag matching).
//****************************************************************************

OnlineNode make_mem_node(uint64_t id, int rank, uint64_t bytes) {
    OnlineNode node;
    node.global_id = id;  // never 0: NodeStore would auto-assign
    node.rank = rank;
    node.kind = NodeKind::MemLoad;
    node.node_type = 2;  // ChakraProtoMsg::NodeType::MEM_LOAD_NODE
    node.name = "gate_mem_" + std::to_string(id);
    node.is_cpu_op = false;
    node.is_timer_op = false;
    node.request_id = "online-gate-fixture";
    node.stage = "prefill";
    node.generation = 0;
    node.mem.tensor_size = bytes;
    return node;
}

OnlineNode make_comm_node(uint64_t id, int rank, bool is_send, int src,
                          int dst, uint32_t tag, uint64_t bytes) {
    OnlineNode node;
    node.global_id = id;
    node.rank = rank;
    node.kind = is_send ? NodeKind::CommSend : NodeKind::CommRecv;
    node.node_type = is_send ? 5 : 6;  // COMM_SEND_NODE / COMM_RECV_NODE
    node.name = std::string(is_send ? "gate_send_" : "gate_recv_") +
                std::to_string(id);
    node.is_cpu_op = false;
    node.is_timer_op = false;
    node.request_id = "online-gate-fixture";
    node.stage = "prefill";
    node.generation = 0;
    node.comm.bytes = bytes;
    node.comm.src = src;
    node.comm.dst = dst;
    node.comm.tag = tag;
    node.comm.hbm_charge = true;  // contention disabled in this fixture
    return node;
}

//****************************************************************************
// One online simulation: harness + injection + ONE issue pass + the caller's
// post-pass inspector, then the drain and the caller's post-run checks.
//****************************************************************************

struct OnlineRun {
    std::vector<TermRecord> terminals;  // exact delivery order
    std::vector<AnalyticalRemoteMemory::PortStatsSnapshot> ports;
    bool drained = false;
};

using PostPassInspect = void (*)(const std::vector<NodeStoreGraphSource*>&,
                                 AnalyticalRemoteMemory*);

OnlineRun run_online(const std::string& name,
                     const std::vector<std::vector<OnlineNode>>& per_rank_nodes,
                     PostPassInspect inspect_after_pass) {
    const std::string dir = scenario_dir(name);
    std::error_code ec;
    std::filesystem::remove_all(dir, ec);
    std::filesystem::create_directories(dir, ec);
    if (!std::filesystem::is_directory(dir)) {
        fail("cannot create fixture dir " + dir);
    }
    constexpr double kBw = 6.0;          // B/ns
    constexpr double kLatencyNs = 100.0; // ns
    write_configs(dir, kBw, kLatencyNs);

    AstraSim::LoggerFactory::init("empty", "off");

    static HookContext hook_ctx;
    hook_ctx.records.clear();
    ExecutionDriven::CompletionObserver::instance().set_hook(&terminal_hook,
                                                             &hook_ctx);
    MetricCollector::instance().initialize("empty", "off");

    const auto event_queue = std::make_shared<EventQueue>();
    const auto network_parser = NetworkParser(dir + "/network.yml");
    const auto topology = construct_topology(network_parser);
    CongestionAwareNetworkApi::set_event_queue(event_queue);
    CongestionAwareNetworkApi::set_topology(topology);
    const auto fluid_scheduler = std::make_shared<FluidScheduler>(
        event_queue, topology->get_directed_links(),
        network_parser.get_fluid_max_active_flows(),
        network_parser.get_fluid_max_route_memberships(),
        network_parser.get_progress_report_event_interval());
    // Online mode: comm emission goes through the deferred channel (the
    // coordinator flushes post-commit); the fixture flushes once after its
    // single manual issue pass.
    fluid_scheduler->set_deferred_flush_mode(true);
    CongestionAwareNetworkApi::set_fluid_scheduler(fluid_scheduler);

    // Declared BEFORE the Sys vector: the backend (and its §3.4 settled
    // audit at destruction) must outlive every Sys.
    const auto memory_api =
        std::make_unique<AnalyticalRemoteMemory>(dir + "/remote_memory.json");

    const auto npus_count_per_dim = network_parser.get_npus_counts_per_dim();
    std::vector<int> queues_per_dim;
    for (size_t i = 0; i < npus_count_per_dim.size(); ++i) {
        queues_per_dim.push_back(1);
    }

    const int num_ranks = static_cast<int>(per_rank_nodes.size());
    auto network_apis = std::vector<std::unique_ptr<CongestionAwareNetworkApi>>();
    auto graph_sources = std::vector<NodeStoreGraphSource*>();
    auto systems = std::vector<Sys*>();
    auto workloads = std::vector<Workload*>();

    for (int rank = 0; rank < num_ranks; ++rank) {
        auto network_api = std::make_unique<CongestionAwareNetworkApi>(rank);
        auto graph_source = std::make_shared<NodeStoreGraphSource>();
        auto* const system =
            new Sys(rank, dir + "/workload", dir + "/comm_group.json",
                    dir + "/system.json", memory_api.get(), network_api.get(),
                    npus_count_per_dim, queues_per_dim, 1.0, 1.0, false,
                    ExecutionDriven::ExecutionMode::Online, graph_source);
        network_apis.push_back(std::move(network_api));
        graph_sources.push_back(graph_source.get());
        systems.push_back(system);
        workloads.push_back(system->workload);
    }

    // ---- inject the nodes straight into each rank's NodeStore ----
    for (int rank = 0; rank < num_ranks; ++rank) {
        auto& store = graph_sources[rank]->store();
        for (const OnlineNode& node : per_rank_nodes[rank]) {
            store.add_node(node);
        }
    }

    // ---- ONE issue pass (the online emission path under test) ----
    for (int rank = 0; rank < num_ranks; ++rank) {
        workloads[rank]->issue_dep_free_nodes();
    }

    // Post-pass inspector runs BEFORE the event loop: this is the
    // "一次 issue pass 后" PortStats/free-set observation point.
    inspect_after_pass(graph_sources, memory_api.get());

    // ---- drain ----
    fluid_scheduler->flush_pending_starts();
    fluid_scheduler->mark_event_loop_started();
    uint64_t guard = 0;
    while (!event_queue->finished()) {
        event_queue->proceed();
        if (++guard > 1000000ULL) {
            fail("scenario " + name +
                 ": event loop did not terminate (guard exceeded)");
        }
    }

    OnlineRun out;
    out.terminals = hook_ctx.records;
    const auto port_count = memory_api->port_count();
    for (std::size_t p = 0; p < port_count; ++p) {
        out.ports.push_back(memory_api->port_stats(p));
    }
    out.drained = memory_api->is_drained();

    for (auto it : systems) {
        delete it;
    }
    systems.clear();
    // memory_api destroyed at scope exit -> §3.4 settled audit.
    return out;
}

//****************************************************************************
// Generic conservation check.
//****************************************************************************

void expect_port_conserved(const AnalyticalRemoteMemory::PortStatsSnapshot& s,
                           double bw, const std::string& tag) {
    expect_eq_u64(s.issued_count, s.completed_count,
                  tag + ": issued_count == completed_count");
    expect_eq_u64(s.issued_bytes, s.completed_bytes,
                  tag + ": issued_bytes == completed_bytes");
    expect_eq_u64(s.in_flight_count, 0, tag + ": in_flight drained");
    expect_eq_u64(s.streaming_count, 0, tag + ": streaming drained");
    expect_eq_u64(s.latency_waiting_count, 0,
                  tag + ": latency-waiting drained");
    expect_eq_u64(s.completion_waiting_count, 0,
                  tag + ": completion-waiting drained");
    if (s.bytes_served < bw * s.port_busy_ns - 1e-6 ||
        s.bytes_served > bw * s.port_busy_ns + 1e-6) {
        fail(tag + ": bytes_served deviates from bw * port_busy_ns");
    }
    if (s.shared_busy_ns > s.port_busy_ns + 1e-9) {
        fail(tag + ": shared_busy_ns exceeds port_busy_ns");
    }
    if (s.peak_streaming > s.peak_in_flight) {
        fail(tag + ": peak_streaming exceeds peak_in_flight");
    }
}

//****************************************************************************
// Scenario A: same-rank two dependency-free MEMs -- the remote-MEM node slot
// must let BOTH issue in one pass and share the rank's port.
//****************************************************************************

void scenario_a_inspect(const std::vector<NodeStoreGraphSource*>& sources,
                        AnalyticalRemoteMemory* memory) {
    // Free sets must be empty: no node was held back by any gate.
    for (size_t rank = 0; rank < sources.size(); ++rank) {
        const auto free_ids = sources[rank]->store().resolve_free_nodes();
        if (!free_ids.empty()) {
            fail("scenario A: rank " + std::to_string(rank) +
                 " still holds " + std::to_string(free_ids.size()) +
                 " un-issued nodes after one issue pass (gate blocked "
                 "emission)");
        }
    }
    // The task-book core assertion: same port, issued == in_flight == 2.
    const auto port0 = memory->port_stats(0);
    expect_eq_u64(port0.issued_count, 2,
                  "A: port0 issued_count == 2 after one issue pass");
    expect_eq_u64(port0.in_flight_count, 2,
                  "A: port0 in_flight_count == 2 after one issue pass");
    expect_eq_u64(port0.latency_waiting_count, 2,
                  "A: both transactions in the latency phase (not streaming)");
    expect_eq_u64(port0.streaming_count, 0,
                  "A: latency consumes no bandwidth (0 streams at t=0)");
    expect_eq_u64(port0.peak_in_flight, 2, "A: port0 peak_in_flight == 2");
    const auto port1 = memory->port_stats(1);
    expect_eq_u64(port1.issued_count, 1,
                  "A: control rank port issued == 1");
    expect_eq_u64(port1.in_flight_count, 1,
                  "A: control rank port in_flight == 1");
}

void scenario_a() {
    // PER_NPU [0,1]: rank0 -> port0 (the gate subject), rank1 -> port1.
    const std::vector<std::vector<OnlineNode>> nodes = {
        {make_mem_node(1, 0, 600), make_mem_node(2, 0, 600)},
        {make_mem_node(1, 1, 600)},
    };

    const OnlineRun out = run_online("a_two_mem_same_rank", nodes,
                                     &scenario_a_inspect);

    expect_eq_u64(out.terminals.size(), 3, "A terminal count");
    // 2-way split: 600 B at 3 B/ns over [100,300] -> both at 300 (the old
    // serial FIFO would give 200/400).
    expect_eq_u64(terminal_of(out.terminals, 0, 1), 300,
                  "A: rank0 node1 callback t=300 (shared 3 B/ns)");
    expect_eq_u64(terminal_of(out.terminals, 0, 2), 300,
                  "A: rank0 node2 callback t=300 (shared 3 B/ns)");
    expect_eq_u64(terminal_of(out.terminals, 1, 1), 200,
                  "A: control rank solo 600 B -> 200");

    // Same-Tick delivery order on port0 at the shared Tick 300 (the control
    // rank's t=200 terminal legitimately precedes the batch in the record
    // vector): within the Tick-300 batch, (port, issue_seq) ascending.
    std::vector<std::pair<int, uint64_t>> batch;
    for (const TermRecord& rec : out.terminals) {
        if (rec.tick == 300) {
            batch.emplace_back(rec.rank, rec.node_id);
        }
    }
    if (batch.size() != 2 || batch[0] != std::make_pair(0, (uint64_t)1) ||
        batch[1] != std::make_pair(0, (uint64_t)2)) {
        fail("A: Tick-300 batch delivery order must be (r0#1) then (r0#2) "
             "((port, issue_seq) ascending)");
    }

    expect_eq_u64(out.ports.size(), 2, "A PER_NPU port count");
    const auto& p0 = out.ports[0];
    const auto& p1 = out.ports[1];
    // Interval-integral evidence (阶段 5.3 red line): two concurrent streams
    // really shared the port service interval.
    expect_eq_u64(p0.peak_streaming, 2,
                  "A: peak_streaming >= 2 (two concurrent streams)");
    if (p0.peak_streaming < 2) {
        fail("A: peak_streaming below 2");
    }
    expect_eq_d(p0.shared_busy_ns, 200.0,
                "A: shared_busy_ns == 200 > 0 (the [100,300] two-stream "
                "integral)");
    if (p0.shared_busy_ns <= 0.0) {
        fail("A: shared_busy_ns must be > 0");
    }
    expect_eq_d(p0.port_busy_ns, 200.0, "A: port_busy_ns == 200");
    expect_eq_d(p0.bytes_served, 1200.0, "A: bytes_served == 1200");
    expect_eq_u64(p0.redistribution_events, 0,
                  "A: simultaneous exhaustion leaves no survivor");
    expect_eq_u64(p1.peak_streaming, 1, "A: control port peak_streaming == 1");
    expect_eq_d(p1.port_busy_ns, 100.0, "A: control port_busy_ns == 100");
    expect_eq_d(p1.bytes_served, 600.0, "A: control bytes_served == 600");
    expect_true(out.drained, "A is_drained");
    expect_port_conserved(p0, 6.0, "A port0");
    expect_port_conserved(p1, 6.0, "A port1");
    std::printf(
        "[remote_port_online_gate_test] A same-rank 2xMEM: issued/in_flight="
        "2 post-pass, peak_streaming=2, shared_busy_ns=200 PASS\n");
}

//****************************************************************************
// Scenario B: independent MEM + COMM_SEND on one rank -- MEM occupies the
// remote-MEM node slot, COMM the comm slot; neither gate blocks the other.
//****************************************************************************

void scenario_b_inspect(const std::vector<NodeStoreGraphSource*>& sources,
                        AnalyticalRemoteMemory* memory) {
    for (size_t rank = 0; rank < sources.size(); ++rank) {
        const auto free_ids = sources[rank]->store().resolve_free_nodes();
        if (!free_ids.empty()) {
            fail("scenario B: rank " + std::to_string(rank) +
                 " still holds " + std::to_string(free_ids.size()) +
                 " un-issued nodes after one issue pass (the COMM/MEM gates "
                 "are not independent)");
        }
    }
    const auto port0 = memory->port_stats(0);
    expect_eq_u64(port0.issued_count, 1,
                  "B: port0 issued == 1 (the MEM transaction)");
    expect_eq_u64(port0.in_flight_count, 1,
                  "B: port0 in_flight == 1 while the COMM pair is live on "
                  "the comm gate");
}

void scenario_b() {
    // rank0: MEM 600 B + COMM_SEND -> rank1 (tag 7, 64 B); rank1: matching
    // COMM_RECV.  All three dependency-free -> one pass.
    const std::vector<std::vector<OnlineNode>> nodes = {
        {make_mem_node(1, 0, 600),
         make_comm_node(2, 0, /*is_send=*/true, /*src=*/0, /*dst=*/1,
                        /*tag=*/7, /*bytes=*/64)},
        {make_comm_node(1, 1, /*is_send=*/false, /*src=*/0, /*dst=*/1,
                        /*tag=*/7, /*bytes=*/64)},
    };

    const OnlineRun out = run_online("b_mem_plus_comm_send", nodes,
                                     &scenario_b_inspect);

    expect_eq_u64(out.terminals.size(), 3, "B terminal count");
    const uint64_t mem_terminal = terminal_of(out.terminals, 0, 1);
    const uint64_t send_terminal = terminal_of(out.terminals, 0, 2);
    const uint64_t recv_terminal = terminal_of(out.terminals, 1, 1);
    expect_eq_u64(mem_terminal, 200,
                  "B: MEM terminal exactly 200 (100 latency + 600 B @ 6 "
                  "B/ns)");
    // Gate-independence red line: the comm pair completed while the MEM was
    // still in flight.  Under the retired comm-single-slot behavior the
    // send could not have issued before the MEM released the shared slot at
    // t=200, so the recv terminal would land beyond 200.
    if (send_terminal >= mem_terminal) {
        fail("B: COMM_SEND terminal " + std::to_string(send_terminal) +
             " not earlier than the MEM terminal 200 (comm was gated behind "
             "the remote MEM)");
    }
    if (recv_terminal >= mem_terminal) {
        fail("B: COMM_RECV terminal " + std::to_string(recv_terminal) +
             " not earlier than the MEM terminal 200 (comm was gated behind "
             "the remote MEM)");
    }
    std::printf(
        "[remote_port_online_gate_test] B MEM+COMM_SEND: MEM=200, "
        "send=%llu, recv=%llu (both < 200; independent gates) PASS\n",
        static_cast<unsigned long long>(send_terminal),
        static_cast<unsigned long long>(recv_terminal));

    expect_eq_u64(out.ports.size(), 2, "B PER_NPU port count");
    const auto& p0 = out.ports[0];
    expect_eq_u64(p0.issued_count, 1, "B port0 issued == 1");
    expect_eq_u64(p0.issued_bytes, 600, "B port0 issued bytes == 600");
    expect_eq_u64(p0.completed_bytes, 600, "B port0 completed bytes == 600");
    expect_eq_u64(p0.peak_streaming, 1, "B port0 peak_streaming == 1");
    expect_eq_d(p0.port_busy_ns, 100.0, "B port_busy_ns == 100");
    expect_eq_d(p0.bytes_served, 600.0, "B bytes_served == 600");
    expect_true(out.drained, "B is_drained");
    expect_port_conserved(p0, 6.0, "B port0");
    expect_port_conserved(out.ports[1], 6.0, "B port1 (idle)");
}

//****************************************************************************
// Entry: each scenario in a forked child for full static-state isolation.
//****************************************************************************

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

int main() {
    run_scenario_isolated("a", &scenario_a);
    run_scenario_isolated("b", &scenario_b);
    std::printf(
        "[remote_port_online_gate_test] ALL PASS (online gate: same-rank "
        "2xMEM issued/in_flight=2 post-pass, peak_streaming>=2, "
        "shared_busy_ns>0; MEM+COMM_SEND independent gates)\n");
    return 0;
}
