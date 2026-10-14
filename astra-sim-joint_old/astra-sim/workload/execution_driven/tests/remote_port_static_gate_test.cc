/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

/**

remote_port_static_gate_test.cc -- STATIC/ETFeeder system-path gate fixture
for the remote-memory port (SerDes片外链路并发化改造执行方案 V5.3 阶段 5.4;
test-only, never part of the production gates).

Subject: the static ETFeeder issue path (Workload::fire ->
issue_dep_free_nodes -> HardwareResource occupy -> backend issue) with TWO
dependency-free plain remote MEM_LOAD nodes on ONE rank.  The joint-unique
HbmNwayTest is also a static ET entry but carries only ONE remote MEM per
rank -- it cannot substitute this double-MEM gate fixture.

Fixture contract (plan 阶段 5.7, fixed):
    python3 make_remote_port_static_fixture_et.py --out-dir <临时目录>
    <this binary> --fixture-dir <同一临时目录>

The generator writes fixture.{0,1}.et + system.json + comm_group.json +
network.yml + remote_memory.json (official template shape: the three
scheduling/collective keys present, the four *-implementation keys all
["ring","ring"], comm_group "{}" so no dimensionless group can exist; the
joint deep-dive H1/H2/H3 landmines stay unarmed).

Physics (remote_memory.json: PER_NPU npu-ids [0,1], remote-mem-bw 6 B/ns,
remote-mem-latency 100 ns; rank0 = two dep-free 600 B MEMs, rank1 = one
control 600 B MEM):

 1. comm 单槽未阻止第二个发射: both rank0 MEMs issue at t=0 and share port0
    -- 3 B/ns each over [100,300], so BOTH terminals land at t=300 and the
    interval integrals show peak_in_flight == 2, peak_streaming == 2 and
    shared_busy_ns == 200.  Under the retired comm-single-slot fall-through
    the second MEM would have waited for the first's release: serial
    terminals 200/400 with peak_streaming == 1.  The control rank finishes
    solo at t=200 on port1 (cross-port isolation).
 2. 计数/集合释放恰一次: every terminal is recorded exactly once per (rank,
    node); after the drain every HardwareResource slot counter is zero and
    the remote_mem_ops_node set is EMPTY (each occupy paired with exactly
    one release -- a double release would have hit the release fatal guards
    mid-run, a leak would show here); backend per-port issued==completed
    counts and bytes; is_drained() (its settled audit also re-runs at
    backend destruction).
 3. finish gate 等到全部终结: every rank's workload reports is_finished and
    all three terminals exist exactly once -- the static finish gate could
    only fire after the remote-MEM class drained (if it had fired early,
    the event loop would have ended with terminals missing).

Build: the CMake target AstraSim_Analytical_Congestion_Aware_RemotePortStaticGateTest
(same shape as HbmNwayTest: frontend congestion_aware shared sources +
execution_driven layer compiled into the test target).
Run:
    python3 astra-sim/workload/execution_driven/tests/\
        make_remote_port_static_fixture_et.py --out-dir <临时目录>
    build/.../bin/AstraSim_Analytical_Congestion_Aware_RemotePortStaticGateTest \
        --fixture-dir <同一临时目录>
Exit code 0 on ALL PASS.
*******************************************************************************/

#include "astra-sim/common/Logging.hh"
#include "astra-sim/system/Sys.hh"
#include "astra-sim/workload/MetricCollector.hh"
#include "astra-sim/workload/Workload.hh"
#include "astra-sim/workload/execution_driven/CompletionObserver.hh"
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
#include <memory>
#include <string>
#include <utility>
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
    std::fprintf(stderr, "[remote_port_static_gate_test] FAIL: %s\n",
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
             " terminal records (expected exactly 1 -- terminal/join must "
             "fire exactly once)");
    }
    return tick;
}

//****************************************************************************
// Entry.
//****************************************************************************

void print_usage(const char* argv0) {
    std::fprintf(stderr,
                 "usage: %s --fixture-dir <dir generated by "
                 "make_remote_port_static_fixture_et.py --out-dir <dir>>\n"
                 "       (also accepts --fixture-dir=<dir>)\n",
                 argv0);
}

}  // namespace

int main(int argc, char* argv[]) {
    std::string fixture_dir;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--fixture-dir" && i + 1 < argc) {
            fixture_dir = argv[++i];
        } else if (arg.rfind("--fixture-dir=", 0) == 0) {
            fixture_dir = arg.substr(std::strlen("--fixture-dir="));
        } else {
            print_usage(argv[0]);
            return 1;
        }
    }
    if (fixture_dir.empty()) {
        print_usage(argv[0]);
        return 1;
    }
    // Fail closed on missing fixture inputs (bad generator invocation must
    // not masquerade as a simulator failure).
    for (const char* name :
         {"fixture.0.et", "fixture.1.et", "system.json", "comm_group.json",
          "network.yml", "remote_memory.json"}) {
        if (!std::filesystem::exists(
                std::filesystem::path(fixture_dir) / name)) {
            fail(std::string("fixture input missing: ") + fixture_dir + "/" +
                 name);
        }
    }

    AstraSim::LoggerFactory::init("empty", "off");

    static HookContext hook_ctx;
    ExecutionDriven::CompletionObserver::instance().set_hook(&terminal_hook,
                                                             &hook_ctx);
    MetricCollector::instance().initialize("empty", "off");

    const std::string workload_configuration = fixture_dir + "/fixture";
    const auto event_queue = std::make_shared<EventQueue>();
    const auto network_parser =
        NetworkParser(fixture_dir + "/network.yml");
    const auto topology = construct_topology(network_parser);
    const auto npus_count = topology->get_npus_count();
    const auto npus_count_per_dim = topology->get_npus_count_per_dim();
    const auto dims_count = topology->get_dims_count();

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
    // Declared BEFORE the Sys vector: the backend (and its §3.4 settled
    // audit at destruction) must outlive every Sys.
    const auto memory_api = std::make_unique<AnalyticalRemoteMemory>(
        fixture_dir + "/remote_memory.json");
    auto systems = std::vector<Sys*>();

    auto queues_per_dim = std::vector<int>();
    for (auto i = 0; i < dims_count; i++) {
        queues_per_dim.push_back(1);
    }

    for (int i = 0; i < npus_count; i++) {
        auto network_api = std::make_unique<CongestionAwareNetworkApi>(i);
        auto* const system =
            new Sys(i, workload_configuration,
                    fixture_dir + "/comm_group.json",
                    fixture_dir + "/system.json", memory_api.get(),
                    network_api.get(), npus_count_per_dim, queues_per_dim,
                    1.0, 1.0, false);
        network_apis.push_back(std::move(network_api));
        systems.push_back(system);
    }

    for (int i = 0; i < npus_count; i++) {
        systems[i]->workload->fire();
    }
    fluid_scheduler->flush_pending_starts();
    fluid_scheduler->mark_event_loop_started();
    while (!event_queue->finished()) {
        event_queue->proceed();
    }

    // ---- 1. comm 单槽未阻止第二个发射: shared-service terminals -------
    expect_eq_u64(hook_ctx.records.size(), 3, "terminal count");
    expect_eq_u64(terminal_of(hook_ctx.records, 0, 0), 300,
                  "rank0 MEM-a terminal t=300 (shared 3 B/ns; serial gate "
                  "would give 200)");
    expect_eq_u64(terminal_of(hook_ctx.records, 0, 1), 300,
                  "rank0 MEM-b terminal t=300 (second emission NOT blocked; "
                  "serial gate would give 400)");
    expect_eq_u64(terminal_of(hook_ctx.records, 1, 0), 200,
                  "rank1 control terminal t=200 (solo, cross-port)");
    // Tick-300 batch delivery order: (port, issue_seq) ascending.  The
    // records vector is in cross-Tick delivery order and the t=200 control
    // completes EARLIER, so assert the two rank0 entries are adjacent and in
    // issue order wherever the control sits.
    bool batch_order_ok = false;
    for (size_t i = 0; i + 1 < hook_ctx.records.size(); ++i) {
        if (hook_ctx.records[i].tick == 300 &&
            hook_ctx.records[i + 1].tick == 300 &&
            hook_ctx.records[i].rank == 0 &&
            hook_ctx.records[i].node_id == 0 &&
            hook_ctx.records[i + 1].rank == 0 &&
            hook_ctx.records[i + 1].node_id == 1) {
            batch_order_ok = true;
            break;
        }
    }
    expect_true(batch_order_ok,
                "Tick-300 batch order must be (r0#0) then (r0#1) "
                "((port, issue_seq) ascending)");

    // ---- port statistics (read-only PortStats snapshot) ----------------
    expect_eq_u64(memory_api->port_count(), 2, "PER_NPU port count");
    const auto p0 = memory_api->port_stats(0);
    const auto p1 = memory_api->port_stats(1);

    // Interval-integral evidence that BOTH rank0 MEMs streamed together.
    expect_eq_u64(p0.issued_count, 2, "port0 issued_count");
    expect_eq_u64(p0.completed_count, 2, "port0 completed_count");
    expect_eq_u64(p0.issued_bytes, 1200, "port0 issued_bytes");
    expect_eq_u64(p0.completed_bytes, 1200, "port0 completed_bytes");
    expect_eq_u64(p0.peak_in_flight, 2, "port0 peak_in_flight == 2");
    expect_eq_u64(p0.peak_streaming, 2,
                  "port0 peak_streaming == 2 (both MEMs in flight together "
                  "-- the comm single slot did not block the second)");
    expect_eq_d(p0.shared_busy_ns, 200.0,
                "port0 shared_busy_ns == 200 (the [100,300] two-stream "
                "integral)");
    expect_eq_d(p0.port_busy_ns, 200.0, "port0 port_busy_ns == 200");
    expect_eq_d(p0.bytes_served, 1200.0, "port0 bytes_served == 1200");
    expect_eq_u64(p0.redistribution_events, 0,
                  "port0: simultaneous exhaustion leaves no survivor");
    expect_eq_u64(p0.stream_join_events, 0,
                  "port0: first joins create no prior share to change");
    expect_eq_u64(p1.issued_count, 1, "port1 issued_count");
    expect_eq_u64(p1.completed_count, 1, "port1 completed_count");
    expect_eq_u64(p1.peak_streaming, 1, "port1 peak_streaming == 1");
    expect_eq_d(p1.port_busy_ns, 100.0, "port1 port_busy_ns == 100");
    expect_eq_d(p1.bytes_served, 600.0, "port1 bytes_served == 600");

    // ---- 2. 计数/集合释放恰一次 -----------------------------------------
    for (int rank = 0; rank < static_cast<int>(npus_count); ++rank) {
        const HardwareResource* hw = systems[rank]->workload->hw_resource;
        expect_eq_u64(hw->num_in_flight_remote_mem_ops, 0,
                      "rank " + std::to_string(rank) +
                          ": remote-MEM slot count released to zero");
        expect_true(hw->remote_mem_ops_node.empty(),
                    "rank " + std::to_string(rank) +
                        ": remote-MEM node set empty (each occupy released "
                        "exactly once)");
        expect_eq_u64(hw->num_in_flight_cpu_ops, 0,
                      "rank " + std::to_string(rank) +
                          ": CPU slot count zero");
        expect_eq_u64(hw->num_in_flight_gpu_comp_ops, 0,
                      "rank " + std::to_string(rank) +
                          ": GPU comp slot count zero");
        expect_eq_u64(hw->num_in_flight_gpu_comm_ops, 0,
                      "rank " + std::to_string(rank) +
                          ": comm slot count zero");
        expect_eq_u64(hw->num_in_flight_hbm_dma_ops, 0,
                      "rank " + std::to_string(rank) +
                          ": HBM DMA slot count zero");
    }
    // Conservation helper (counts AND bytes, never count-equality alone).
    for (const auto* snapp : {&p0, &p1}) {
        const auto& s = *snapp;
        expect_eq_u64(s.issued_count, s.completed_count,
                      "port issued == completed");
        expect_eq_u64(s.issued_bytes, s.completed_bytes,
                      "port issued_bytes == completed_bytes");
        expect_eq_u64(s.in_flight_count, 0, "port in_flight drained");
        expect_eq_u64(s.latency_waiting_count, 0,
                      "port latency-waiting drained");
        expect_eq_u64(s.completion_waiting_count, 0,
                      "port completion-waiting drained");
    }
    expect_true(memory_api->is_drained(), "backend is_drained");

    // ---- 3. finish gate 等到全部终结 ------------------------------------
    for (int rank = 0; rank < static_cast<int>(npus_count); ++rank) {
        expect_true(systems[rank]->workload->is_finished,
                    "rank " + std::to_string(rank) +
                        ": static finish gate fired only after every node "
                        "terminated (is_finished)");
    }
    // The gate cannot have fired early: all three terminals exist exactly
    // once (checked above), and the latest terminal is the t=300 batch.
    uint64_t last_terminal = 0;
    for (const TermRecord& rec : hook_ctx.records) {
        if (rec.tick > last_terminal) {
            last_terminal = rec.tick;
        }
    }
    expect_eq_u64(last_terminal, 300,
                  "simulation spanned to the last remote-MEM terminal (300)");

    for (auto it : systems) {
        delete it;
    }
    systems.clear();
    // memory_api destroyed at scope exit -> §3.4 settled audit re-runs.

    std::printf(
        "[remote_port_static_gate_test] ALL PASS (static gate: same-rank "
        "double MEM issued together (300/300, peak_streaming=2, "
        "shared_busy_ns=200); exactly-once occupy/release; finish gate "
        "waited for all terminals)\n");
    return 0;
}
