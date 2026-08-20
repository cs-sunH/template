/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

/**
local_hbm_model_test.cc -- multi-user local-HBM bandwidth contention
fixture (hbm-bandwidth-contention / LocalHbmBandwidthModel).

Self-contained online-mode fixture (no .et files, no Python): the test
writes its own tiny system/network/remote-memory configs to a temp dir,
builds 4 ranks over a real congestion-aware line topology, injects nodes
straight into the per-rank NodeStore, and drives the event queue with the
same post-commit deferred issue pass the online ServiceCoordinator uses
(WatchRegistry.cc online_issue_pass_cb semantics). Terminal facts are
recorded through the CompletionObserver hook.

Two sequential simulations:

  SIM 1 -- flag ON (hbm-bandwidth-contention: 1):
   rank2 (fluid numerics, COMP roofline jobs, local-mem-latency 100 ns,
         full rate 1640 B/ns):
     A COMP(246000 B) + B/C COMP(492000 B) concurrent -> each 1/3 rate
       (546.67 B/ns): A = 100 + 450 = 550; B/C = 550 + 300 = 850
       (A's completion reallocates the bus to the survivors at 1/2).
   rank0/rank1 (COMM join both orders; d2d 400 B/ns slower than HBM):
     small send (8200 B): network (~50 ns) completes BEFORE the HBM read
       (105 ns) -> join fires at 105 (network-first order);
     large send (8.2 MB) chained after it: HBM read (100+5000 ns) completes
       BEFORE the network (~20.6 us) -> join fires at the network arrival
       (HBM-first order);
     a third small send with hbm-charge=false: no HBM job at all
       (terminal = pure network time; comm_read bytes exclude it).
   rank3 (MEM join both orders; remote port 1000 B/ns + 100 ns):
     MEM_STORE(mode 2) + COMP co-runner: port (1740) before the shared
       HBM write (2100) -> port-first order;
     a second MEM_STORE(mode 2) chained after the first, bus now idle:
       HBM (3200) before the port (3840) -> HBM-first order;
     a MEM_LOAD with hbm-access-mode absent: no HBM job (pool bytes
       exclude it).
   Cross-checks: served-byte sums per category, peak concurrency 3,
   reallocation events >= 1, every joined node fires exactly once.

  SIM 2 -- flag OFF (hbm-bandwidth-contention: 0):
     the model is null; the same rank2 COMP A completes at the legacy
     single-owner roofline time 100 + 246000/1640 = 250 ns (full rate),
     proving the complete fallback to the legacy timing.

  NAIL (R2-1) -- same-tick double issue: two model issues at one
     boostedTick (the first job is still inside its memory-latency phase
     because advance_to's elapsed is 0 between them) must count exactly one
     redistribution event -- the unified face rule (!jobs.empty()); the
     pre-R2-1 draining-survivor rule (a joining job counts only when some
     survivor is already streaming bytes) saw a latency-phase-only set and
     counted 0, so this nail fails on the pre-fix code. Independent of the
     local-mem-latency value.

Build: cmake target AstraSim_Analytical_Congestion_Aware_LocalHbmTest.
Run: build/astra_analytical/build_congestion_aware/bin/\
     AstraSim_Analytical_Congestion_Aware_LocalHbmTest
Exit code 0 on ALL PASS.
*/

#include "astra-sim/common/Logging.hh"
#include "astra-sim/system/WorkloadLayerHandlerData.hh"
#include "astra-sim/workload/Workload.hh"
#include "astra-sim/workload/execution_driven/CompletionObserver.hh"
#include "astra-sim/workload/execution_driven/NodeStore.hh"
#include "congestion_aware/CongestionAwareNetworkApi.hh"
#include <astra-network-analytical/common/EventQueue.h>
#include <astra-network-analytical/common/NetworkParser.h>
#include <astra-network-analytical/congestion_aware/Helper.h>
#include <remote_memory_backend/analytical/AnalyticalRemoteMemory.hh>

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <string>
#include <tuple>
#include <vector>

using namespace AstraSim;
using namespace AstraSim::ExecutionDriven;
using namespace Analytical;
using namespace AstraSimAnalytical;
using namespace AstraSimAnalyticalCongestionAware;
using namespace NetworkAnalytical;
using namespace NetworkAnalyticalCongestionAware;

namespace {

// ---- fixture config (d2d 400 B/ns < HBM 1640 B/ns lets both join orders
// happen; remote port 1000 B/ns sits between uncontended 1640 and the
// 2-user 820 share so both MEM join orders happen) ----
constexpr int kRanks = 4;
constexpr double kD2dBw = 400.0;        // GB/s -> B/ns
constexpr double kHbmBw = 1640.0;       // GB/s -> B/ns
constexpr uint64_t kHbmLatency = 100;   // ns
constexpr double kRemoteBw = 1000.0;    // B/ns (remote_memory.json units)
constexpr uint64_t kRemoteLatency = 100;
constexpr double kPeakPerfTflops = 261.12;

struct TerminalRecord {
    int rank;
    uint64_t node_id;
    uint64_t tick;
};

struct HookContext {
    std::vector<TerminalRecord>* records = nullptr;
    EventQueue* event_queue = nullptr;
    std::vector<Workload*>* workloads = nullptr;
};

struct IssuePassArg {
    Workload* workload = nullptr;
};

void issue_pass_cb(void* arg) {
    auto* pass = static_cast<IssuePassArg*>(arg);
    pass->workload->issue_dep_free_nodes();
    delete pass;
}

void terminal_hook(void* ctx, int rank, uint64_t node_id,
                   const char*, const char*, uint64_t, uint64_t tick,
                   int) {
    auto* context = static_cast<HookContext*>(ctx);
    context->records->push_back(TerminalRecord{rank, node_id, tick});
    // Same-tick deferred per-rank re-issue (the online coordinator's
    // post-commit drain, WatchRegistry.cc step 1-8): chained / slot-blocked
    // nodes must re-scan when their predecessor released them.
    if (context->event_queue != nullptr && context->workloads != nullptr &&
        rank >= 0 && rank < static_cast<int>(context->workloads->size())) {
        auto* pass = new IssuePassArg;
        pass->workload = (*context->workloads)[rank];
        context->event_queue->schedule_event_deferred(issue_pass_cb, pass);
    }
}

uint64_t terminal_tick(const std::vector<TerminalRecord>& records,
                       int rank, uint64_t node_id) {
    uint64_t hits = 0;
    uint64_t tick = 0;
    for (const auto& rec : records) {
        if (rec.rank == rank && rec.node_id == node_id) {
            ++hits;
            tick = rec.tick;
        }
    }
    assert(hits == 1);  // exactly-once node completion
    return tick;
}

void write_text(const std::string& path, const std::string& content) {
    std::ofstream out(path);
    assert(out.is_open());
    out << content;
}

std::string system_json(bool contention) {
    std::string flag = contention ? "1" : "0";
    return std::string(R"({
  "scheduling-policy": "LIFO",
  "endpoint-delay": 10,
  "active-chunks-per-dimension": 1,
  "preferred-dataset-splits": 6,
  "all-reduce-implementation": ["ring", "ring"],
  "all-gather-implementation": ["ring", "ring"],
  "reduce-scatter-implementation": ["ring", "ring"],
  "all-to-all-implementation": ["ring", "ring"],
  "collective-optimization": "localBWAware",
  "boost-mode": 0,
  "roofline-enabled": 1,
  "replay-only": 0,
  "track-local-mem": 0,
  "trace-enabled": 0,
  "hbm-bandwidth-contention": )") + flag + "\n" + R"(,
  "peak-perf": 261.12,
  "local-mem-bw": 1640.0,
  "local-mem-latency": 100,
  "remote-mem-bw": 1000.0,
  "remote-mem-latency": 100
}
)";
}

const char* kNetworkYaml = R"(topology: [ Line, Line ]
npus_count: [ 4, 1 ]
bandwidth: [ 400.0, 400.0 ]
latency: [ 5, 5 ]
)";

const char* kRemoteMemoryJson = R"({
  "memory-type": "PER_NPU_MEMORY_EXPANSION",
  "remote-mem-bw": 1000.0,
  "remote-mem-latency": 100,
  "npu-ids": [0, 1, 2, 3]
}
)";

OnlineNode make_node(uint64_t id, int rank, NodeKind kind, uint64_t type,
                     const std::string& name) {
    OnlineNode node;
    node.global_id = id;
    node.rank = rank;
    node.kind = kind;
    node.node_type = type;
    node.name = name;
    node.is_cpu_op = false;
    node.is_timer_op = false;
    node.request_id = "fixture";
    node.stage = "prefill";
    node.generation = 0;
    return node;
}

// One full online simulation over the fixture configs; returns the systems
// (kept alive: the caller reads the HBM model counters) and fills
// `records` with every node terminal fact.
std::vector<Sys*> run_simulation(const std::string& config_dir,
                                 bool contention,
                                 std::vector<TerminalRecord>& records) {
    const std::string system_configuration =
        config_dir + (contention ? "/system_on.json" : "/system_off.json");
    const std::string network_configuration = config_dir + "/network.yml";
    const std::string remote_memory_configuration =
        config_dir + "/remote_memory.json";
    const std::string comm_group_configuration = config_dir + "/comm_group.json";
    const std::string workload_configuration = config_dir + "/workload";

    AstraSim::LoggerFactory::init("empty", "off");

    const auto event_queue = std::make_shared<EventQueue>();
    const auto network_parser = NetworkParser(network_configuration);
    const auto topology = construct_topology(network_parser);
    CongestionAwareNetworkApi::set_event_queue(event_queue);
    CongestionAwareNetworkApi::set_topology(topology);
    const auto fluid_scheduler = std::make_shared<FluidScheduler>(
        event_queue, topology->get_directed_links(),
        network_parser.get_fluid_max_active_flows(),
        network_parser.get_fluid_max_route_memberships(),
        network_parser.get_progress_report_event_interval());
    // Online semantics (main_online.cc): comm starts emitted from a
    // deferred/tick-end context flush through the same-tick deferred drain,
    // never as a current_time EventList in the main queue.
    fluid_scheduler->set_deferred_flush_mode(true);
    CongestionAwareNetworkApi::set_fluid_scheduler(fluid_scheduler);

    const auto memory_api =
        std::make_unique<AnalyticalRemoteMemory>(remote_memory_configuration);

    auto network_apis =
        std::vector<std::unique_ptr<CongestionAwareNetworkApi>>();
    auto graph_sources =
        std::vector<std::shared_ptr<NodeStoreGraphSource>>();
    auto systems = std::vector<Sys*>();
    auto workloads = std::vector<Workload*>();

    const auto npus_count_per_dim = network_parser.get_npus_counts_per_dim();
    std::vector<int> queues_per_dim;
    for (size_t i = 0; i < npus_count_per_dim.size(); ++i) {
        queues_per_dim.push_back(1);
    }

    for (int i = 0; i < kRanks; ++i) {
        auto network_api = std::make_unique<CongestionAwareNetworkApi>(i);
        auto graph_source = std::make_shared<NodeStoreGraphSource>();
        auto* const system =
            new Sys(i, workload_configuration, comm_group_configuration,
                    system_configuration, memory_api.get(), network_api.get(),
                    npus_count_per_dim, queues_per_dim, 1.0, 1.0, false,
                    ExecutionMode::Online, graph_source);
        network_apis.push_back(std::move(network_api));
        graph_sources.push_back(std::move(graph_source));
        systems.push_back(system);
        workloads.push_back(system->workload);
    }

    HookContext hook_ctx{&records, event_queue.get(), &workloads};
    CompletionObserver::instance().set_hook(terminal_hook, &hook_ctx);

    // ---- inject the fixture nodes (per-rank NodeStore) ----
    // Occupancy reality (HardwareResource online): COMP and COMM_SEND/MEM
    // are single-slot each (different slots), COMM_RECV never occupies -- so
    // the concurrency scenarios use co-issueable kinds: COMP + SEND + RECV
    // on one rank, and chained nodes for the rest.
    auto& s0 = graph_sources[0]->store();
    auto& s1 = graph_sources[1]->store();
    auto& s2 = graph_sources[2]->store();
    auto& s3 = graph_sources[3]->store();

    // rank0: chained COMM_SENDs 10 (net-first join), 11 (hbm-first join),
    // 12 (hbm-charge=false -> no HBM job); all go to rank1.
    {
        OnlineNode n = make_node(10, 0, NodeKind::CommSend, 5, "send_small");
        n.comm.bytes = 8200;
        n.comm.src = 0;
        n.comm.dst = 1;
        n.comm.tag = 1;
        s0.add_node(n);
    }
    {
        OnlineNode n = make_node(11, 0, NodeKind::CommSend, 5, "send_large");
        n.comm.bytes = 8200000;
        n.comm.src = 0;
        n.comm.dst = 1;
        n.comm.tag = 2;
        s0.add_node(n);
        s0.add_dependency(10, 11, DepKind::Data);
    }
    {
        OnlineNode n = make_node(12, 0, NodeKind::CommSend, 5, "send_free");
        n.comm.bytes = 8200;
        n.comm.src = 0;
        n.comm.dst = 1;
        n.comm.tag = 3;
        n.comm.hbm_charge = false;
        s0.add_node(n);
        s0.add_dependency(11, 12, DepKind::Data);
    }

    // rank1: the matching COMM_RECVs (no occupancy -> all issue at t=0 and
    // their COMM_WRITE jobs share rank1's bus from t=0) plus one send of
    // its own feeding rank2's numeric recv.
    {
        OnlineNode n = make_node(20, 1, NodeKind::CommRecv, 6, "recv_small");
        n.comm.bytes = 8200;
        n.comm.src = 0;
        n.comm.dst = 1;
        n.comm.tag = 1;
        s1.add_node(n);
    }
    {
        OnlineNode n = make_node(21, 1, NodeKind::CommRecv, 6, "recv_large");
        n.comm.bytes = 8200000;
        n.comm.src = 0;
        n.comm.dst = 1;
        n.comm.tag = 2;
        s1.add_node(n);
    }
    {
        OnlineNode n = make_node(22, 1, NodeKind::CommRecv, 6, "recv_free");
        n.comm.bytes = 8200;
        n.comm.src = 0;
        n.comm.dst = 1;
        n.comm.tag = 3;
        s1.add_node(n);
    }
    {
        OnlineNode n = make_node(23, 1, NodeKind::CommSend, 5, "send_num");
        n.comm.bytes = 492000;
        n.comm.src = 1;
        n.comm.dst = 2;
        n.comm.tag = 5;
        s1.add_node(n);
    }

    // rank2: the equal-share numerics -- three co-issued HBM jobs (COMP
    // comp-slot + SEND comm-slot + RECV unoccupied), equal-ish byte totals:
    //   COMP A 246000 B, SEND 246000 B, RECV 492000 B -> each 1/3 rate
    //   (546.67 B/ns after the 100 ns latency): A and the send's read
    //   drain at 100+450 = 550; the recv write's remaining 246000 B then
    //   drains at 1/2 rate (820 B/ns) -> 550+300 = 850.
    {
        OnlineNode n = make_node(30, 2, NodeKind::Compute, 4, "comp_a");
        n.compute.num_ops = 1;
        n.compute.tensor_size = 246000;
        s2.add_node(n);
    }
    {
        OnlineNode n = make_node(31, 2, NodeKind::CommSend, 5, "send_num");
        n.comm.bytes = 246000;
        n.comm.src = 2;
        n.comm.dst = 3;
        n.comm.tag = 4;
        s2.add_node(n);
    }
    {
        OnlineNode n = make_node(32, 2, NodeKind::CommRecv, 6, "recv_num");
        n.comm.bytes = 492000;
        n.comm.src = 1;
        n.comm.dst = 2;
        n.comm.tag = 5;
        s2.add_node(n);
    }

    // rank3: MEM join both orders + an uncharged MEM_LOAD. MEM A and COMP X
    // co-issue at t=0 (different slots); MEM B chains after A.
    {
        OnlineNode n = make_node(40, 3, NodeKind::MemStore, 3, "mem_a");
        n.compute.tensor_size = 1640000;
        n.compute.hbm_access_mode = 2;
        s3.add_node(n);
    }
    {
        OnlineNode n = make_node(41, 3, NodeKind::Compute, 4, "comp_x");
        n.compute.num_ops = 1;
        n.compute.tensor_size = 1640000;
        s3.add_node(n);
    }
    {
        OnlineNode n = make_node(42, 3, NodeKind::MemStore, 3, "mem_b");
        n.compute.tensor_size = 1640000;
        n.compute.hbm_access_mode = 2;
        s3.add_node(n);
        s3.add_dependency(40, 42, DepKind::Data);
    }
    {
        OnlineNode n = make_node(43, 3, NodeKind::MemLoad, 2, "mem_free");
        n.compute.tensor_size = 100000;  // hbm_access_mode stays 0
        s3.add_node(n);
        s3.add_dependency(42, 43, DepKind::Data);
    }
    {
        // rank2's numeric send (tag 4) needs a receiver on rank3; the recv
        // never occupies hardware, so rank3's MEM/COMP timing above is
        // untouched (its COMM_WRITE shares rank3's bus with A and X).
        OnlineNode n = make_node(44, 3, NodeKind::CommRecv, 6, "recv_num");
        n.comm.bytes = 246000;
        n.comm.src = 2;
        n.comm.dst = 3;
        n.comm.tag = 4;
        s3.add_node(n);
    }

    // ---- run (online mode: no fire(); drain the stores manually, then the
    // event loop; the hook's deferred pass re-drains after each terminal) --
    for (auto* workload : workloads) {
        workload->issue_dep_free_nodes();
    }
    fluid_scheduler->flush_pending_starts();
    fluid_scheduler->mark_event_loop_started();
    size_t guard = 0;
    while (!event_queue->finished()) {
        event_queue->proceed();
        if (++guard > 1000000) {
            std::fprintf(stderr, "[LocalHbmTest] event loop did not drain\n");
            std::exit(1);
        }
    }
    return systems;
}

void expect_near(uint64_t actual, uint64_t expected, uint64_t slack,
                 const char* what) {
    const int64_t diff = static_cast<int64_t>(actual) -
        static_cast<int64_t>(expected);
    const bool ok = diff >= -static_cast<int64_t>(slack) &&
        diff <= static_cast<int64_t>(slack);
    std::printf("%-46s expected=%llu actual=%llu -> %s\n", what,
                static_cast<unsigned long long>(expected),
                static_cast<unsigned long long>(actual), ok ? "PASS" : "FAIL");
    if (!ok) {
        std::exit(1);
    }
}

void expect_true(bool cond, const char* what) {
    std::printf("%-46s -> %s\n", what, cond ? "PASS" : "FAIL");
    if (!cond) {
        std::exit(1);
    }
}

// R2-1 regression nail: same-tick double issue on a fresh model. Between
// the two issue calls no event fires, so advance_to()'s elapsed is 0 and
// the first job is still inside its memory-latency phase when the second
// joins. The unified face rule (!jobs.empty()) counts that join; the
// pre-R2-1 rule required a survivor already streaming bytes and counted 0
// here (a true would-fail-before-the-fix nail). The scheduled transition
// events are intentionally never fired: the fixture asserts and exits
// without draining this event queue.
void same_tick_double_issue_nail(const std::string& config_dir) {
    AstraSim::LoggerFactory::init("empty", "off");

    const std::string system_configuration = config_dir + "/system_on.json";
    const std::string network_configuration = config_dir + "/network.yml";
    const std::string remote_memory_configuration =
        config_dir + "/remote_memory.json";
    const std::string comm_group_configuration = config_dir + "/comm_group.json";
    const std::string workload_configuration = config_dir + "/workload";

    const auto event_queue = std::make_shared<EventQueue>();
    const auto network_parser = NetworkParser(network_configuration);
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
        std::make_unique<AnalyticalRemoteMemory>(remote_memory_configuration);

    // A single rank suffices (the model is per-rank and nothing here uses
    // the network); same Sys construction as run_simulation.
    auto network_api = std::make_unique<CongestionAwareNetworkApi>(0);
    auto graph_source = std::make_shared<NodeStoreGraphSource>();
    const auto npus_count_per_dim = network_parser.get_npus_counts_per_dim();
    std::vector<int> queues_per_dim;
    for (size_t i = 0; i < npus_count_per_dim.size(); ++i) {
        queues_per_dim.push_back(1);
    }
    auto* const system =
        new Sys(0, workload_configuration, comm_group_configuration,
                system_configuration, memory_api.get(), network_api.get(),
                npus_count_per_dim, queues_per_dim, 1.0, 1.0, false,
                ExecutionMode::Online, graph_source);
    auto* const model = system->workload->local_hbm_bandwidth_model.get();
    expect_true(model != nullptr, "nail: contention on -> model exists");

    auto* wlhd = new WorkloadLayerHandlerData;
    wlhd->sys_id = system->id;
    wlhd->workload = system->workload;
    wlhd->node_id = 9001;

    model->issue_comm_read(246000, wlhd);
    model->issue_comm_read(246000, wlhd);
    expect_true(model->redistribution_events() == 1,
                "nail: same-tick double issue -> 1 event");
}

}  // namespace

int main() {
    setvbuf(stdout, nullptr, _IONBF, 0);
    // ---- write the fixture configs to a temp dir ----
    const std::string tmp_template = "/tmp/local_hbm_test_XXXXXX";
    std::vector<char> buffer(tmp_template.begin(), tmp_template.end());
    buffer.push_back('\0');
    const char* made = mkdtemp(buffer.data());
    assert(made != nullptr);
    const std::string config_dir(made);
    write_text(config_dir + "/system_on.json", system_json(true));
    write_text(config_dir + "/system_off.json", system_json(false));
    write_text(config_dir + "/network.yml", kNetworkYaml);
    write_text(config_dir + "/remote_memory.json", kRemoteMemoryJson);
    write_text(config_dir + "/comm_group.json", "{}");

    // ================= SIM 1: contention ON =================
    std::vector<TerminalRecord> records;
    auto systems = run_simulation(config_dir, true, records);
    const auto model = [](Sys* s) {
        return s->workload->local_hbm_bandwidth_model.get();
    };

    std::printf("[SIM1] contention on: %zu terminals\n", records.size());

    // --- rank2 equal-share numerics (COMP A + send read + recv write) ---
    // Each of the three jobs gets 1640/3 = 546.67 B/ns after the 100 ns
    // latency: COMP A drains 246000 B at 550; the recv write's remaining
    // 246000 B then drains at 1/2 rate -> 850 (checked via served bytes +
    // reallocation count; its node terminal is net-bound).
    expect_near(terminal_tick(records, 2, 30), 550, 2,
                "rank2 comp_a (3-way 1/3 share)");
    expect_true(model(systems[2])->peak_concurrent_jobs() == 3,
                "rank2 peak concurrent jobs == 3");
    // 中-4④ derivation: the unified face-rule condition (!jobs.empty() after
    // the batch) is satisfied whenever the old draining_after>0 was (any
    // draining survivor implies a survivor), and additionally on the rare
    // all-survivors-in-latency-phase edge -- the count can only grow, so the
    // weak >= 1 assertion holds under both definitions unchanged.
    expect_true(model(systems[2])->redistribution_events() >= 1,
                "rank2 redistribution events >= 1");
    expect_near(static_cast<uint64_t>(
                    model(systems[2])->compute_bytes_served()),
                246000, 10, "rank2 comp served bytes");
    expect_near(static_cast<uint64_t>(
                    model(systems[2])->comm_read_bytes_served()),
                246000, 10, "rank2 comm_read served bytes");
    expect_near(static_cast<uint64_t>(
                    model(systems[2])->comm_write_bytes_served()),
                492000, 10, "rank2 comm_write served bytes");
    expect_near(static_cast<uint64_t>(model(systems[2])->hbm_busy_ns()),
                600, 10, "rank2 hbm_busy_ns (450 3-way + 150 full)");

    // --- rank0 COMM join, network-first order ---
    // HBM read = 100 + 8200/1640 = 105; the network (~50 ns) is earlier, so
    // the node completes only at the HBM side (join waited).
    expect_near(terminal_tick(records, 0, 10), 105, 4,
                "rank0 send_small (network-first join)");

    // --- rank0 COMM join, HBM-first order ---
    // HBM read = 105 + 100 + 8.2e6/1640 ~ 5205; the network (~20.6 us) is
    // later, so the node completes at the network side.
    const uint64_t t_send_large = terminal_tick(records, 0, 11);
    expect_true(t_send_large > 6000 && t_send_large < 22000,
                "rank0 send_large (hbm-first join, net-limited)");
    std::printf("%-46s terminal=%llu (hbm side ~5205)\n",
                "  detail send_large",
                static_cast<unsigned long long>(t_send_large));

    // --- hbm-charge=false: no COMM_READ job ---
    expect_near(static_cast<uint64_t>(
                    model(systems[0])->comm_read_bytes_served()),
                8200 + 8200000, 10,
                "rank0 comm_read bytes (excl. uncharged)");
    // its terminal is net-only (~50 ns after send_large's terminal).
    expect_near(terminal_tick(records, 0, 12), t_send_large + 50, 100,
                "rank0 send_free terminal (net-only)");

    // --- rank1: recv joins (net-first) + byte sums ---
    // The three COMM_WRITEs + rank1's own send read share the bus at 1/4
    // rate: recv_small's write drains at 100 + 8200/410 = 120 (the network
    // arrival ~50 ns is earlier -> network-first join on the recv side).
    expect_near(terminal_tick(records, 1, 20), 120, 6,
                "rank1 recv_small (network-first join)");
    const uint64_t t_recv_large = terminal_tick(records, 1, 21);
    expect_true(t_recv_large > 15000,
                "rank1 recv_large (hbm-first join, net-limited)");
    expect_near(static_cast<uint64_t>(
                    model(systems[1])->comm_write_bytes_served()),
                8200 + 8200000 + 8200, 10, "rank1 comm_write bytes");
    expect_near(static_cast<uint64_t>(
                    model(systems[1])->comm_read_bytes_served()),
                492000, 10, "rank1 comm_read bytes");

    // --- rank3 MEM join, port-first order ---
    // port A = 100 + 1640000/1000 = 1740; the HBM write shares rank3's bus
    // (with COMP X and the numeric recv's write) -> drains at 2250. Port
    // completes first; the node terminal waits for the HBM side.
    expect_near(terminal_tick(records, 3, 40), 2250, 10,
                "rank3 mem_a (port-first join)");
    expect_near(terminal_tick(records, 3, 41), 2250, 10,
                "rank3 comp_x (shared with mem_a)");
    // --- rank3 MEM join, HBM-first order ---
    // B issues at 2250: HBM write = 2250 + 100 + 1640000/1640 = 3350;
    // port B = 2250 + 1740 = 3990. HBM side first; terminal at the port.
    expect_near(terminal_tick(records, 3, 42), 3990, 12,
                "rank3 mem_b (hbm-first join)");
    // --- hbm-access-mode absent: no POOL job ---
    expect_near(static_cast<uint64_t>(
                    model(systems[3])->pool_write_bytes_served()),
                1640000 + 1640000, 10,
                "rank3 pool_write bytes (excl. uncharged)");
    expect_near(terminal_tick(records, 3, 43), 3990 + 200, 12,
                "rank3 mem_free terminal (port-only)");
    expect_true(model(systems[3])->pool_read_bytes_served() == 0.0,
                "rank3 pool_read bytes == 0");

    // utilization sanity: rank2's bus was busy for a positive fraction of
    // the observed window (the max terminal tick; avoids poking the global
    // all_sys clock after the loop).
    uint64_t max_tick = 0;
    for (const auto& rec : records) {
        if (rec.tick > max_tick) {
            max_tick = rec.tick;
        }
    }
    expect_true(
        model(systems[2])->hbm_utilization(max_tick) > 0.0,
        "rank2 hbm utilization > 0");

    // ================= SIM 2: contention OFF =================
    std::vector<TerminalRecord> records_off;
    auto systems_off = run_simulation(config_dir, false, records_off);
    std::printf("[SIM2] contention off: %zu terminals\n",
                records_off.size());
    expect_true(systems_off[0]->workload->local_hbm_bandwidth_model ==
                    nullptr,
                "flag off -> model is null");
    expect_true(systems_off[0]->hbm_bandwidth_contention == false,
                "flag off -> sys flag false");
    // Legacy single-owner roofline: 100 + 246000/1640 = 250 ns at FULL rate
    // (no sharing with the co-issued send/recv jobs).
    expect_near(terminal_tick(records_off, 2, 30), 250, 2,
                "comp_a legacy full-rate terminal");
    // No COMM join exists: send_small completes at its network time (~50 ns).
    expect_near(terminal_tick(records_off, 0, 10), 50, 60,
                "send_small legacy net-only terminal");
    // No MEM join: mem_a completes at its port transaction (1740), not at
    // the (no longer modeled) HBM time.
    expect_near(terminal_tick(records_off, 3, 40), 1740, 10,
                "mem_a legacy port-only terminal");

    // ================= NAIL: R2-1 same-tick double issue =================
    same_tick_double_issue_nail(config_dir);

    std::printf("ALL PASS\n");
    // Best-effort temp-config cleanup (keep on failure for debugging).
    std::error_code ec;
    std::filesystem::remove_all(config_dir, ec);
    return 0;
}
