/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

*******************************************************************************/

/**

noc_shared_link_nway_test.cc -- NoC shared-link N-way equal-split fixture
for the PARTIAL 跨实例 copy 流水化 (2026-09-25, requirement 3): home→exec
copy legs and edge→exec remote-load legs are plain p2p COMM_SENDs through
the single FluidScheduler entry (CongestionAwareNetworkApi::sim_send ->
start_flow), so concurrent D2D traffic sharing one physical link must split
that link's capacity strictly N-way over its active flows -- with dynamic
re-split as flows join/complete.

Test-only, never part of the production gates.  ONLINE NodeView/NodeStore
harness (fork isolation, self-written configs under /tmp/joint-pipeline-
work/, ONE manual issue pass, terminal hook), contention OFF -- pure NoC
timing, no HBM jobs (hbm-charge=false everywhere).

Topology: 4-NPU line (npus_count [4,1]), per-link bandwidth 4000 B/ns,
per-link latency 5 ns.  Three flows start together at t=0:

  F1: 0 -> 1, 8000 B  (route: link01 only)
  F2: 0 -> 2, 4000 B  (route: link01 + link12 -- SHARES link01 with F1)
  F3: 2 -> 3, 4000 B  (route: link23, SOLO control)

Fluid math (rate = min over route links of capacity / active flows;
service = ceil(bytes / rate); arrival = service end + sum of route link
latencies; every service completion re-splits the survivors):

  F1: [0,2] link01 carries {F1,F2} -> 2000 B/ns -> 4000 B served; at t=2
      F2's service ends and link01 re-splits to {F1} -> 4000 B/ns; the
      remaining 4000 B drain over [2,3]; arrival 3 + 5 = 8.
  F2: min(link01 2000, link12 4000) = 2000 B/ns; service ends t=2;
      arrival 2 + (5+5) = 12.
  F3: link23 solo -> 4000 B/ns; service ends t=1; arrival 1 + 5 = 6.

Equal-split + dynamic-re-split evidence carried by the pins:
  - F1 and F2 both ran at exactly half the link01 capacity while both
    were active (F2's 4000 B took 2 ns at 2000 B/ns) -- per-flow equal
    split, not per-byte fair share;
  - F1's arrival 8 is FASTER than its static-2-way counterfactual (9)
    and slower than its solo counterfactual (7): the completion of F2
    dynamically re-split the released capacity to F1 (requirement 3's
    "flows joining/completing re-allocate" made observable);
  - F3 terminates first: a solo link is untouched by the contention
    elsewhere in the fabric (6 < 8 < 12).

Build: the CMake target
AstraSim_Analytical_Congestion_Aware_NocSharedLinkNwayTest.
Run: build/.../bin/AstraSim_Analytical_Congestion_Aware_NocSharedLinkNwayTest
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

[[noreturn]] void fail(const std::string& what) {
    std::fprintf(stderr, "[noc_shared_link_nway_test] FAIL: %s\n",
                 what.c_str());
    std::fflush(stderr);
    std::exit(1);
}

void expect_eq_u64(uint64_t got, uint64_t want, const std::string& what) {
    if (got != want) {
        fail(what + ": got " + std::to_string(got) + ", expected " +
             std::to_string(want));
    }
}

const char* kTmpBase = "/tmp/joint-pipeline-work";

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

void write_configs(const std::string& dir) {
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
    system += "  \"remote-mem-bw\": 6.0,\n";
    system += "  \"remote-mem-latency\": 100\n";
    system += "}\n";
    write_text(dir + "/system.json", system);
    write_text(dir + "/comm_group.json", "{}\n");
    write_text(dir + "/network.yml",
               "# noc_shared_link_nway synthetic 4-NPU line topology\n"
               "topology: [ Line, Line ]\n"
               "npus_count: [ 4, 1 ]\n"
               "bandwidth: [ 4000.0, 4000.0 ]\n"
               "latency: [ 5, 5 ]\n");
    write_text(dir + "/remote_memory.json",
               "{\n"
               "  \"memory-type\": \"PER_NPU_MEMORY_EXPANSION\",\n"
               "  \"remote-mem-bw\": 6.0,\n"
               "  \"remote-mem-latency\": 100,\n"
               "  \"npu-ids\": [0, 1, 2, 3]\n"
               "}\n");
}

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

OnlineNode make_pass_send(uint64_t id, int rank, int src, int dst,
                          uint32_t tag, uint64_t bytes) {
    OnlineNode node;
    node.global_id = id;
    node.rank = rank;
    node.kind = NodeKind::CommSend;
    node.node_type = 5;  // COMM_SEND_NODE
    node.name = "noc_send_" + std::to_string(id);
    node.is_cpu_op = false;
    node.is_timer_op = false;
    node.request_id = "noc-shared-link-fixture";
    node.stage = "prefill";
    node.generation = 0;
    node.comm.bytes = bytes;
    node.comm.src = src;
    node.comm.dst = dst;
    node.comm.tag = tag;
    node.comm.hbm_charge = false;  // pure NoC pass-through
    return node;
}

OnlineNode make_pass_recv(uint64_t id, int rank, int src, int dst,
                          uint32_t tag, uint64_t bytes) {
    OnlineNode node;
    node.global_id = id;
    node.rank = rank;
    node.kind = NodeKind::CommRecv;
    node.node_type = 6;  // COMM_RECV_NODE
    node.name = "noc_recv_" + std::to_string(id);
    node.is_cpu_op = false;
    node.is_timer_op = false;
    node.request_id = "noc-shared-link-fixture";
    node.stage = "prefill";
    node.generation = 0;
    node.comm.bytes = bytes;
    node.comm.src = src;
    node.comm.dst = dst;
    node.comm.tag = tag;
    node.comm.hbm_charge = false;
    return node;
}

void scenario_shared_link() {
    const std::string dir = std::string(kTmpBase) + "/noc_shared_link";
    std::error_code ec;
    std::filesystem::remove_all(dir, ec);
    std::filesystem::create_directories(dir, ec);
    if (!std::filesystem::is_directory(dir)) {
        fail("cannot create fixture dir " + dir);
    }
    write_configs(dir);

    AstraSim::LoggerFactory::init("empty", "off");

    static HookContext hook_ctx;
    hook_ctx.records.clear();
    ExecutionDriven::CompletionObserver::instance().set_hook(&terminal_hook,
                                                             &hook_ctx);
    MetricCollector::instance().initialize("empty", "off");

    const auto event_queue = std::make_shared<EventQueue>();
    const auto network_parser = NetworkParser(dir + "/network.yml");
    const auto topology = construct_topology(network_parser);
    if (topology->get_npus_count() != 4) {
        fail("expected a 4-NPU line topology");
    }
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

    auto network_apis =
        std::vector<std::unique_ptr<CongestionAwareNetworkApi>>();
    auto graph_sources = std::vector<NodeStoreGraphSource*>();
    auto systems = std::vector<Sys*>();
    auto workloads = std::vector<Workload*>();

    for (int rank = 0; rank < 4; ++rank) {
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

    // F1: 0->1 (8000 B), F2: 0->2 (4000 B, shares link01), F3: 2->3
    // (4000 B, solo link23).  All dependency-free; matching recvs parked on
    // the destination ranks.
    const std::vector<std::vector<OnlineNode>> per_rank_nodes = {
        {make_pass_send(1, 0, 0, 1, 11, 8000),
         make_pass_send(2, 0, 0, 2, 12, 4000)},
        {make_pass_recv(1, 1, 0, 1, 11, 8000)},
        {make_pass_recv(1, 2, 0, 2, 12, 4000),
         make_pass_send(2, 2, 2, 3, 13, 4000)},
        {make_pass_recv(1, 3, 2, 3, 13, 4000)},
    };
    for (int rank = 0; rank < 4; ++rank) {
        auto& store = graph_sources[rank]->store();
        for (const OnlineNode& node : per_rank_nodes[rank]) {
            store.add_node(node);
        }
    }

    for (int rank = 0; rank < 4; ++rank) {
        workloads[rank]->issue_dep_free_nodes();
    }
    for (int rank = 0; rank < 4; ++rank) {
        const auto free_ids = graph_sources[rank]->store().resolve_free_nodes();
        if (!free_ids.empty()) {
            fail("rank " + std::to_string(rank) + " still holds " +
                 std::to_string(free_ids.size()) +
                 " un-issued nodes after one issue pass");
        }
    }

    fluid_scheduler->flush_pending_starts();
    fluid_scheduler->mark_event_loop_started();
    uint64_t guard = 0;
    while (!event_queue->finished()) {
        event_queue->proceed();
        if (++guard > 1000000ULL) {
            fail("event loop did not terminate (guard exceeded)");
        }
    }

    // Sends and recvs land on the shared arrival ticks (6 total terminals).
    expect_eq_u64(hook_ctx.records.size(), 6, "terminal count");
    const uint64_t f1 = terminal_of(hook_ctx.records, 0, 1);
    const uint64_t f2 = terminal_of(hook_ctx.records, 0, 2);
    const uint64_t f3 = terminal_of(hook_ctx.records, 2, 2);
    expect_eq_u64(f1, 8,
                  "F1 (0->1, 8000 B) arrival: [0,2] at the 2-way split 2000 "
                  "B/ns, then dynamic re-split to 4000 B/ns -> service end 3 "
                  "+ 5 latency (static 2-way would be 9, solo 7)");
    expect_eq_u64(f2, 12,
                  "F2 (0->2, 4000 B) arrival: bottleneck link01 2000 B/ns -> "
                  "service end 2 + 2-hop latency 10");
    expect_eq_u64(f3, 6,
                  "F3 (2->3, 4000 B) arrival: solo link23 4000 B/ns -> "
                  "service end 1 + 5 latency");
    if (!(f3 < f1 && f1 < f2)) {
        fail("expected the solo control to finish first and the shared flows "
             "to order by their equal-split service times (6 < 8 < 12)");
    }
    // Recv sides mirror the same arrivals.
    expect_eq_u64(terminal_of(hook_ctx.records, 1, 1), f1, "F1 recv arrival");
    expect_eq_u64(terminal_of(hook_ctx.records, 2, 1), f2, "F2 recv arrival");
    expect_eq_u64(terminal_of(hook_ctx.records, 3, 1), f3, "F3 recv arrival");

    for (auto it : systems) {
        delete it;
    }
    systems.clear();

    std::printf(
        "[noc_shared_link_nway_test] ALL PASS (shared link01: F2=12 at 2000 "
        "B/ns while F1 active; F1=8 after F2's completion re-split the link; "
        "solo control F3=6)\n");
}

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
    run_scenario_isolated("shared_link", &scenario_shared_link);
    return 0;
}
