/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

*******************************************************************************/

/**

pipeline_concurrent_gate_test.cc -- ONLINE gate fixture for the PARTIAL
跨实例 copy 流水化 (2026-09-25): after the first KV chunk's barrier, D2D
sends, KV restores and prefill COMPUTE must form a real pipeline -- the
same-rank HardwareResource gates for COMM_SEND (gpu_comm) and KV restore
(hbm_dma) are counted UNLIMITED slots, so multiple transactions may be in
flight at once and arbitration belongs to LocalHbmBandwidthModel's N-way
equal split (requirement 1 execution side) which must show strict N-way
dynamic sharing over ALL active HBM streams (requirement 2).

Test-only, never part of the production gates.  Online NodeView/NodeStore
harness identical to remote_port_online_gate_test.cc (fork isolation,
self-written configs under /tmp/joint-pipeline-work/, ONE manual issue
pass, terminal hook).  System config: roofline-enabled=1,
hbm-bandwidth-contention=1, hbm-kv-restore-bandwidth-sharing=1,
local-mem-bw=3 B/ns, local-mem-latency=100 ns; all compute nodes carry
runtime_ns=0 (production online shape) so COMPUTE joins the HBM pool.

Scenarios (each a full online simulation in a forked child):

 P. same-rank pipeline mix (requirement 1 + 2): rank0 issues FOUR
    dependency-free nodes in ONE pass -- COMP(300 B), KV-restore(600 B),
    COMM_SEND A(450 B, tag 1) and COMM_SEND B(450 B, tag 2), both charged
    (COMM_READ endpoint jobs) with matching recvs on rank1.  After ONE
    issue pass every rank's NodeStore free set is EMPTY -- under the
    retired comm single slot, send B would still sit in the free set.
    Exact HBM-pool timeline (4-way -> 3-way -> 2-way -> solo):
      t=100 4-way @ 0.75 B/ns: COMP+READ legs exhaust 300 B each at 500;
      t=500 3-way @ 1.0: sends' 150 B remnants exhaust at 650;
      t=650 2-way @ 1.5: restore's 150 B remnant exhausts at 700.
    Assertions:
      - COMP terminal == 500 (1/4-rate numeric: 100 + 300/0.75);
      - both send terminals == 650 (join: HBM leg dominates, network
        side arrives at ~6) -- both STRICTLY BEFORE the restore terminal
        700: sends and restore overlapped in flight, neither serialized
        behind the other;
      - recv terminals == 400 (rank1 2-way COMM_WRITE split): the remote
        side also drains while the sends/restore are still in flight;
      - peak_concurrent_jobs == 4; restore_bytes_served == 600;
        comm_read_bytes_served == 900; compute_bytes_served == 300;
      - redistribution_events == 5 (batch-level rule: issue side
        restore/sendA/sendB each join a non-empty set (+3); completion
        batches at 500 (survivors +1) and 650 (survivor +1), the 700
        batch leaves no survivor).

 Q. two concurrent KV restores (requirement 2 -- first time two RESTORE
    jobs can share the pool, impossible under the retired hbm_dma single
    slot): rank0 issues COMP(300 B), RESTORE_A(300 B), RESTORE_B(600 B)
    and a charged COMM_SEND(450 B, tag 9) in ONE pass; rank1 holds the
    matching recv.  After ONE pass the free set is empty -- RESTORE_B was
    NOT stranded behind RESTORE_A.  Timeline (4 -> 2 -> 1-way):
      t=100 4-way @ 0.75: COMP/RA exhaust 300 B at 500; RB 300 left;
      READ 150 left;
      t=500 2-way @ 1.5: READ exhausts at 600; t=600 RB solo @ 3:
      150 B remnant exhausts at 650.
    Assertions:
      - COMP == 500, RESTORE_A == 500, send == 600, RESTORE_B == 650
        (dynamic re-split evidence: RB's rate rises twice);
      - restore_bytes_served == 900 (both restores served by ONE pool),
        comm_read == 450, compute == 300, peak_concurrent_jobs == 4,
        redistribution_events == 5 (issue +3; batches 500 and 600 leave
        survivors +2).

Build: the CMake target
AstraSim_Analytical_Congestion_Aware_PipelineConcurrentGateTest.
Run: build/.../bin/AstraSim_Analytical_Congestion_Aware_PipelineConcurrentGateTest
Exit code 0 on ALL PASS.
*******************************************************************************/

#include "astra-sim/common/Logging.hh"
#include "astra-sim/system/Sys.hh"
#include "astra-sim/workload/LocalHbmBandwidthModel.hh"
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
    std::fprintf(stderr, "[pipeline_concurrent_gate_test] FAIL: %s\n",
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
// Fixture plumbing (isolated dirs under /tmp/joint-pipeline-work/).
//****************************************************************************

const char* kTmpBase = "/tmp/joint-pipeline-work";

std::string scenario_dir(const std::string& name) {
    return std::string(kTmpBase) + "/pipeline_gate_" + name;
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

// Production-shape system config: the LocalHbmBandwidthModel exists and owns
// COMP/restore/comm endpoints (contention on), roofline on (COMP issue
// contract), local-mem-bw 3 GB/s == 3 B/ns, latency 100 ns.
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
    system += "  \"roofline-enabled\": 1,\n";
    system += "  \"hbm-kv-restore-bandwidth-sharing\": 1,\n";
    system += "  \"hbm-bandwidth-contention\": 1,\n";
    system += "  \"replay-only\": 0,\n";
    system += "  \"track-local-mem\": 0,\n";
    system += "  \"trace-enabled\": 0,\n";
    system += "  \"peak-perf\": 1000,\n";
    system += "  \"local-mem-bw\": 3.0,\n";
    system += "  \"local-mem-latency\": 100,\n";
    system += "  \"remote-mem-bw\": " + std::to_string(remote_bw) + ",\n";
    system += "  \"remote-mem-latency\": " +
              std::to_string(remote_latency_ns) + "\n";
    system += "}\n";
    write_text(dir + "/system.json", system);
    write_text(dir + "/comm_group.json", "{}\n");
    write_text(dir + "/network.yml",
               "# pipeline_concurrent_gate synthetic 2-rank line topology\n"
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
// Terminal capture.
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
// Node builders (production online shapes: runtime_ns=0 COMP, KV-restore
// MEM_LOAD, charged p2p comm).
//****************************************************************************

OnlineNode make_comp_node(uint64_t id, int rank, uint64_t tensor_size) {
    OnlineNode node;
    node.global_id = id;
    node.rank = rank;
    node.kind = NodeKind::Compute;
    node.node_type = 4;  // ChakraProtoMsg::NodeType::COMP_NODE
    node.name = "pipe_comp_" + std::to_string(id);
    node.is_cpu_op = false;
    node.is_timer_op = false;
    node.request_id = "pipeline-gate-fixture";
    node.stage = "prefill";
    node.generation = 0;
    node.compute.num_ops = 1;
    node.compute.tensor_size = tensor_size;  // runtime_ns stays 0 (online)
    return node;
}

OnlineNode make_restore_node(uint64_t id, int rank, uint64_t tensor_size) {
    OnlineNode node;
    node.global_id = id;
    node.rank = rank;
    node.kind = NodeKind::MemLoad;
    node.node_type = 2;  // MEM_LOAD_NODE
    node.name = "pipe_kv_restore_" + std::to_string(id);
    node.is_cpu_op = false;
    node.is_timer_op = false;
    node.request_id = "pipeline-gate-fixture";
    node.stage = "prefill";
    node.generation = 0;
    node.mem.tensor_size = tensor_size;
    node.mem.is_local_hbm_kv_restore = true;
    return node;
}

OnlineNode make_comm_node(uint64_t id, int rank, bool is_send, int src,
                          int dst, uint32_t tag, uint64_t bytes) {
    OnlineNode node;
    node.global_id = id;
    node.rank = rank;
    node.kind = is_send ? NodeKind::CommSend : NodeKind::CommRecv;
    node.node_type = is_send ? 5 : 6;  // COMM_SEND_NODE / COMM_RECV_NODE
    node.name = std::string(is_send ? "pipe_send_" : "pipe_recv_") +
                std::to_string(id);
    node.is_cpu_op = false;
    node.is_timer_op = false;
    node.request_id = "pipeline-gate-fixture";
    node.stage = "prefill";
    node.generation = 0;
    node.comm.bytes = bytes;
    node.comm.src = src;
    node.comm.dst = dst;
    node.comm.tag = tag;
    node.comm.hbm_charge = true;  // charged: endpoint HBM jobs join the pool
    return node;
}

//****************************************************************************
// One online simulation: harness + injection + ONE issue pass + the caller's
// post-pass inspector, then the drain and model snapshots.
//****************************************************************************

struct ModelSnapshot {
    double compute_bytes = 0.0;
    double restore_bytes = 0.0;
    double comm_read_bytes = 0.0;
    double comm_write_bytes = 0.0;
    uint64_t peak_concurrent_jobs = 0;
    uint64_t redistribution_events = 0;
};

struct OnlineRun {
    std::vector<TermRecord> terminals;
    ModelSnapshot model0;
    ModelSnapshot model1;
};

using PostPassInspect = void (*)(const std::vector<NodeStoreGraphSource*>&);

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
    write_configs(dir, /*remote_bw=*/6.0, /*remote_latency_ns=*/100.0);

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
    fluid_scheduler->set_deferred_flush_mode(true);
    CongestionAwareNetworkApi::set_fluid_scheduler(fluid_scheduler);

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

    // Post-pass inspector runs BEFORE the event loop.
    inspect_after_pass(graph_sources);

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
    const auto snapshot_of = [](const Sys* system) {
        const auto* model =
            system->workload->local_hbm_bandwidth_model.get();
        if (model == nullptr) {
            fail("LocalHbmBandwidthModel missing (contention must be on)");
        }
        ModelSnapshot snap;
        snap.compute_bytes = model->compute_bytes_served();
        snap.restore_bytes = model->restore_bytes_served();
        snap.comm_read_bytes = model->comm_read_bytes_served();
        snap.comm_write_bytes = model->comm_write_bytes_served();
        snap.peak_concurrent_jobs = model->peak_concurrent_jobs();
        snap.redistribution_events = model->redistribution_events();
        return snap;
    };
    out.model0 = snapshot_of(systems[0]);
    out.model1 = snapshot_of(systems[1]);

    for (auto it : systems) {
        delete it;
    }
    systems.clear();
    return out;
}

void expect_free_sets_empty(const std::vector<NodeStoreGraphSource*>& sources,
                            const std::string& tag) {
    for (size_t rank = 0; rank < sources.size(); ++rank) {
        const auto free_ids = sources[rank]->store().resolve_free_nodes();
        if (!free_ids.empty()) {
            fail(tag + ": rank " + std::to_string(rank) + " still holds " +
                 std::to_string(free_ids.size()) +
                 " un-issued nodes after one issue pass (a resource gate "
                 "blocked emission -- the single-slot behavior is back)");
        }
    }
}

//****************************************************************************
// Scenario P: COMP + KV-restore + 2 charged sends, one rank, one pass.
//****************************************************************************

void scenario_p_inspect(const std::vector<NodeStoreGraphSource*>& sources) {
    expect_free_sets_empty(sources, "P");
}

void scenario_p() {
    const std::vector<std::vector<OnlineNode>> nodes = {
        {make_comp_node(1, 0, 300),
         make_restore_node(2, 0, 600),
         make_comm_node(3, 0, /*is_send=*/true, /*src=*/0, /*dst=*/1,
                        /*tag=*/1, /*bytes=*/450),
         make_comm_node(4, 0, /*is_send=*/true, /*src=*/0, /*dst=*/1,
                        /*tag=*/2, /*bytes=*/450)},
        {make_comm_node(1, 1, /*is_send=*/false, /*src=*/0, /*dst=*/1,
                        /*tag=*/1, /*bytes=*/450),
         make_comm_node(2, 1, /*is_send=*/false, /*src=*/0, /*dst=*/1,
                        /*tag=*/2, /*bytes=*/450)},
    };

    const OnlineRun out = run_online("p_comp_restore_two_sends", nodes,
                                     &scenario_p_inspect);

    expect_eq_u64(out.terminals.size(), 6, "P terminal count");
    const uint64_t comp = terminal_of(out.terminals, 0, 1);
    const uint64_t restore = terminal_of(out.terminals, 0, 2);
    const uint64_t send_a = terminal_of(out.terminals, 0, 3);
    const uint64_t send_b = terminal_of(out.terminals, 0, 4);
    const uint64_t recv_a = terminal_of(out.terminals, 1, 1);
    const uint64_t recv_b = terminal_of(out.terminals, 1, 2);
    // 1/4-rate numeric: 100 ns latency + 300 B at 0.75 B/ns.
    expect_eq_u64(comp, 500, "P: COMP terminal (4-way split 0.75 B/ns)");
    // HBM-pool joins: sends exhaust their COMM_READ jobs at 650 (4-way then
    // 3-way), the restore's remaining 150 B drain solo-equivalent at 700
    // (2-way @ 1.5 from 650).  Network sides arrive far earlier.
    expect_eq_u64(send_a, 650, "P: send A terminal (HBM join dominates)");
    expect_eq_u64(send_b, 650, "P: send B terminal (HBM join dominates)");
    expect_eq_u64(restore, 700, "P: restore terminal (2-way split tail)");
    // Pipeline red line: BOTH D2D sends terminated strictly before the KV
    // restore -- under the retired comm single slot, send B could not have
    // issued until send A released the slot at 650, pushing its HBM join
    // past 700 (>= restore).
    if (send_a >= restore || send_b >= restore) {
        fail("P: send terminal not earlier than restore terminal 700 "
             "(sends were serialized behind/against the restore)");
    }
    // Remote side drains while rank0 is still busy: recvs complete at the
    // rank1 2-way COMM_WRITE split (100 + 450/1.5).
    expect_eq_u64(recv_a, 400, "P: recv A terminal (rank1 2-way split)");
    expect_eq_u64(recv_b, 400, "P: recv B terminal (rank1 2-way split)");
    // HBM pool evidence (requirement 2).
    expect_eq_u64(out.model0.peak_concurrent_jobs, 4,
                  "P: rank0 peak_concurrent_jobs == 4");
    expect_eq_d(out.model0.compute_bytes, 300.0,
                "P: rank0 compute bytes served");
    expect_eq_d(out.model0.restore_bytes, 600.0,
                "P: rank0 restore bytes served");
    expect_eq_d(out.model0.comm_read_bytes, 900.0,
                "P: rank0 comm_read bytes served");
    expect_eq_u64(out.model0.redistribution_events, 5,
                  "P: rank0 redistribution events (issue +3, completion +2)");
    expect_eq_u64(out.model1.peak_concurrent_jobs, 2,
                  "P: rank1 peak_concurrent_jobs == 2");
    expect_eq_d(out.model1.comm_write_bytes, 900.0,
                "P: rank1 comm_write bytes served");

    std::printf(
        "[pipeline_concurrent_gate_test] P COMP+restore+2 sends: free sets "
        "empty post-pass, comp=500, sends=650<restore=700, peak_jobs=4, "
        "redistribution=5 PASS\n");
}

//****************************************************************************
// Scenario Q: TWO concurrent KV restores share the pool with COMP + comm.
//****************************************************************************

void scenario_q_inspect(const std::vector<NodeStoreGraphSource*>& sources) {
    expect_free_sets_empty(sources, "Q");
}

void scenario_q() {
    const std::vector<std::vector<OnlineNode>> nodes = {
        {make_comp_node(1, 0, 300),
         make_restore_node(2, 0, 300),
         make_restore_node(3, 0, 600),
         make_comm_node(4, 0, /*is_send=*/true, /*src=*/0, /*dst=*/1,
                        /*tag=*/9, /*bytes=*/450)},
        {make_comm_node(1, 1, /*is_send=*/false, /*src=*/0, /*dst=*/1,
                        /*tag=*/9, /*bytes=*/450)},
    };

    const OnlineRun out = run_online("q_two_restores", nodes,
                                     &scenario_q_inspect);

    expect_eq_u64(out.terminals.size(), 5, "Q terminal count");
    const uint64_t comp = terminal_of(out.terminals, 0, 1);
    const uint64_t restore_a = terminal_of(out.terminals, 0, 2);
    const uint64_t restore_b = terminal_of(out.terminals, 0, 3);
    const uint64_t send = terminal_of(out.terminals, 0, 4);
    expect_eq_u64(comp, 500, "Q: COMP terminal (4-way split)");
    expect_eq_u64(restore_a, 500,
                  "Q: RESTORE_A terminal (4-way split, exhausted with COMP)");
    expect_eq_u64(send, 600, "Q: send terminal (3-way split tail + join)");
    expect_eq_u64(restore_b, 650,
                  "Q: RESTORE_B terminal (dynamic 4->2->1-way re-split)");
    // Both restores were in flight at once: their jobs overlapped on
    // [100, 500] and split rates diverged afterwards (A done at 500 while
    // B kept draining at the re-split rate).
    if (restore_a == restore_b) {
        fail("Q: the two restores did not produce distinct re-split tails");
    }
    // HBM pool evidence: one pool served BOTH restores.
    expect_eq_d(out.model0.restore_bytes, 900.0,
                "Q: rank0 restore bytes served (both restores, one pool)");
    expect_eq_d(out.model0.comm_read_bytes, 450.0,
                "Q: rank0 comm_read bytes served");
    expect_eq_d(out.model0.compute_bytes, 300.0,
                "Q: rank0 compute bytes served");
    expect_eq_u64(out.model0.peak_concurrent_jobs, 4,
                  "Q: rank0 peak_concurrent_jobs == 4");
    expect_eq_u64(out.model0.redistribution_events, 5,
                  "Q: rank0 redistribution events (issue +3, completion +2)");
    expect_eq_u64(terminal_of(out.terminals, 1, 1), 250,
                  "Q: recv terminal (solo 450 B at 3 B/ns + 100 latency)");
    expect_eq_d(out.model1.comm_write_bytes, 450.0,
                "Q: rank1 comm_write bytes served");

    std::printf(
        "[pipeline_concurrent_gate_test] Q 2xRESTORE+comm+COMP: free sets "
        "empty post-pass, restores 500/650 (re-split), peak_jobs=4, "
        "redistribution=5 PASS\n");
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
    run_scenario_isolated("p", &scenario_p);
    run_scenario_isolated("q", &scenario_q);
    std::printf(
        "[pipeline_concurrent_gate_test] ALL PASS (pipeline gate: same-rank "
        "COMP+KV-restore+2 sends issued in one pass, sends terminate before "
        "restore; 2 concurrent restores share the N-way pool; HBM served "
        "bytes/peak/redistribution pinned)\n");
    return 0;
}
