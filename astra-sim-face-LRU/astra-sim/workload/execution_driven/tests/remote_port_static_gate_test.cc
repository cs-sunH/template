/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

/*

remote_port_static_gate_test.cc -- static/ETFeeder issue-gating system
fixture for the SerDes off-chip-link rework (plan: 片外共享内存端口并发化
改造执行方案 V5.3, stage 5.4; target
AstraSim_Analytical_Congestion_Aware_RemotePortStaticGateTest).

Contract (plan stage 5.7; the workflow runs exactly this pair):
  python3 astra-sim/workload/execution_driven/tests/\
      make_remote_port_static_fixture_et.py --out-dir <temporary dir>
  build/astra_analytical/build_congestion_aware/bin/\
      AstraSim_Analytical_Congestion_Aware_RemotePortStaticGateTest \
      --fixture-dir <the same temporary dir>

The fixture graph (generator products in --fixture-dir) is two
dependency-free remote MEM loads plus one COMM_SEND on rank 0, and the
matching COMM_RECV on rank 1. The static ET path is exercised end to end:
Sys in its default ExecutionMode::Static (ETFeederGraphSource built inside
Workload), the first issue pass triggered by Workload::fire(), the static
finish gate in Workload::call, and the HardwareResource ETFeederNode
overloads with node-id retention (static keeps the exact id sets).

What is asserted:

  1. comm-single-slot gate is gone for remote MEMs: after the FIRST issue
     pass on rank 0 both MEM loads hold the remote-MEM slot at the same
     time (num_in_flight_remote_mem_ops == 2, num_remote_mem_ops == 2,
     remote_mem_ops_node == {1, 2}) while the COMM_SEND holds the legacy
     comm slot (num_in_flight_gpu_comm_ops == 1). Under the old hidden
     gate the second MEM would have been left in the dep-free set and the
     in-flight counts could never reach 2.
  2. each remote-MEM node occupies and releases its slot EXACTLY ONCE:
     probe at Tick 150 shows the sets still populated; after the run the
     in-flight counters are 0, the cumulative counters are exactly
     num_remote_mem_ops == 2 / num_gpu_comms == 1, and both retained id
     sets are empty (any double release would have aborted in
     HardwareResource release_class).
  3. the static finish gate waits for every remote node: the gate
     (Workload::call, static branch) requires num_in_flight_remote_mem_ops
     == 0 before report()/notify; the test asserts is_finished on both
     ranks, every terminal recorded exactly once with the last remote-port
     completions at Tick 300, and the event loop end Tick >= 300 (the
     loop cannot drain before the remote legs terminate).

Timing anchor (analytical remote-port model; PER_NPU bw=6 B/ns, latency
100 ns): both 600 B MEMs stream at 3 B/ns over [100,300) -> 600/3 = 200 ->
fluid 300.0, callback Tick 300, peak_streaming 2, shared_busy_ns 200.0,
port_busy_ns 200.0 (event-interval integrals). The p2p pair completes via
the network (~247 ns transfer + 25 ns latency) and is asserted only as
"completed exactly once".

Config discipline (plan stage 5 避雷): the generator's system.json carries
scheduling-policy / preferred-dataset-splits / collective-optimization and
all four *-implementation keys as ["ring","ring"]; no
*-implementation-custom, no doubleBinaryTree, no comm groups; local-mem
keys absent so the local-HBM contention model auto-disables. The retired
dead key "boost-mode" is not resurrected.

Build (registered in astra-sim/network_frontend/analytical/CMakeLists.txt):
  cmake --build build/astra_analytical/build_congestion_aware --target \
      AstraSim_Analytical_Congestion_Aware_RemotePortStaticGateTest
Exit code 0 on ALL PASS.
*/

#include "astra-sim/common/Logging.hh"
#include "astra-sim/workload/Workload.hh"
#include "astra-sim/workload/execution_driven/CompletionObserver.hh"
#include "congestion_aware/CongestionAwareNetworkApi.hh"
#include <astra-network-analytical/common/EventQueue.h>
#include <astra-network-analytical/common/NetworkParser.h>
#include <astra-network-analytical/congestion_aware/Helper.h>
#include <remote_memory_backend/analytical/AnalyticalRemoteMemory.hh>

#include <unistd.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <map>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

using namespace AstraSim;
using namespace Analytical;
using namespace AstraSimAnalytical;
using namespace AstraSimAnalyticalCongestionAware;
using namespace NetworkAnalytical;
using namespace NetworkAnalyticalCongestionAware;

namespace {

bool g_ok = true;

void expect(bool cond, const std::string& what) {
    if (!cond) {
        std::fprintf(stderr,
                     "[remote_port_static_gate_test] FAIL: %s\n",
                     what.c_str());
        g_ok = false;
    }
}

void expect_near(double value, double expected, double tol,
                 const std::string& what) {
    const double diff =
        value > expected ? value - expected : expected - value;
    if (!(diff <= tol)) {
        std::fprintf(stderr,
                     "[remote_port_static_gate_test] FAIL: %s: got %.12f "
                     "expected %.12f (tol %.3e)\n",
                     what.c_str(), value, expected, tol);
        g_ok = false;
    }
}

// ---------------------------------------------------------------------------
// Terminal observation: the CompletionObserver hook fires synchronously on
// the Workload::call terminal path (static ET path included), in backend
// delivery order.
// ---------------------------------------------------------------------------

struct Recorder {
    std::vector<std::tuple<int, uint64_t, uint64_t>> order;
    std::map<std::pair<int, uint64_t>, uint64_t> tick_of;
    std::map<std::pair<int, uint64_t>, uint64_t> count_of;

    uint64_t tick(int rank, uint64_t node) const {
        auto it = tick_of.find({rank, node});
        return it == tick_of.end() ? 0 : it->second;
    }
    uint64_t count(int rank, uint64_t node) const {
        auto it = count_of.find({rank, node});
        return it == count_of.end() ? 0 : it->second;
    }
    bool rank_touched(int rank) const {
        for (const auto& rec : order) {
            if (std::get<0>(rec) == rank) {
                return true;
            }
        }
        return false;
    }
} g_recorder;

void terminal_hook(void* ctx, int rank, uint64_t node_id, const char*,
                   const char*, uint64_t, uint64_t tick, int status) {
    auto* rec = static_cast<Recorder*>(ctx);
    if (status !=
        static_cast<int>(ExecutionDriven::NodeTerminalStatus::Success)) {
        std::fprintf(stderr,
                     "[remote_port_static_gate_test] FAIL: rank %d node "
                     "%llu terminal status %d\n",
                     rank, static_cast<unsigned long long>(node_id),
                     status);
        g_ok = false;
    }
    rec->order.emplace_back(rank, node_id, tick);
    rec->tick_of[{rank, node_id}] = tick;
    rec->count_of[{rank, node_id}] += 1;
}

// Mid-flight probe from the real event queue at Tick == delay.
class ProbeEvent : public Callable {
  public:
    explicit ProbeEvent(std::function<void()> fn) : fn_(std::move(fn)) {}

    void call(EventType, CallData*) override {
        fn_();
        delete this;
    }

  private:
    std::function<void()> fn_;
};

void schedule_probe_at(Sys* sys, Tick delay, std::function<void()> fn) {
    sys->register_event(new ProbeEvent(std::move(fn)), EventType::General,
                        nullptr, delay);
}

// plan sec.5.1 conservation: count/bytes pairs closed and the served-bytes
// integral within the per-job completion residue bound.
void expect_port_conservation(const AnalyticalRemoteMemory& mem,
                              std::size_t port, double bw,
                              const std::string& tag) {
    const auto st = mem.get_port_stats(port);
    expect(st.issued_count == st.completed_count,
           tag + ": port " + std::to_string(port) + " issued_count " +
               std::to_string(st.issued_count) + " == completed_count " +
               std::to_string(st.completed_count));
    expect(st.issued_bytes == st.completed_bytes,
           tag + ": port " + std::to_string(port) + " issued_bytes " +
               std::to_string(st.issued_bytes) + " == completed_bytes " +
               std::to_string(st.completed_bytes));
    expect(st.in_flight_count == 0,
           tag + ": port " + std::to_string(port) + " drained in-flight");
    const double signed_gap = static_cast<double>(st.completed_bytes) -
                              st.bytes_served;
    const double gap = signed_gap < 0 ? -signed_gap : signed_gap;
    const double bound = static_cast<double>(st.completed_count) *
                         (1e-6 + bw * 1e-9);
    expect(gap <= bound,
           tag + ": port " + std::to_string(port) + " bytes_served " +
               std::to_string(st.bytes_served) + " vs completed_bytes " +
               std::to_string(st.completed_bytes) + " within residue bound");
}

std::string parse_fixture_dir(int argc, char* argv[]) {
    const std::string kFlag = "--fixture-dir";
    std::string value;
    for (int i = 1; i < argc; i++) {
        const std::string arg = argv[i];
        if (arg.rfind(kFlag + "=", 0) == 0) {
            value = arg.substr(kFlag.size() + 1);
        } else if (arg == kFlag && i + 1 < argc) {
            value = argv[++i];
        }
    }
    return value;
}

}  // namespace

int main(int argc, char* argv[]) {
    const std::string fixture_dir = parse_fixture_dir(argc, argv);
    if (fixture_dir.empty()) {
        std::fprintf(stderr,
                     "usage: %s --fixture-dir <dir produced by "
                     "make_remote_port_static_fixture_et.py>\n",
                     argv[0]);
        return 2;
    }
    const std::string workload_prefix =
        fixture_dir + "/remote_port_static_gate";

    AstraSim::LoggerFactory::init("empty", "off");
    ExecutionDriven::CompletionObserver::instance().set_hook(terminal_hook,
                                                             &g_recorder);

    // ---- stack assembly (generator products; static mode by default) ----
    const auto event_queue = std::make_shared<EventQueue>();
    const auto network_parser = NetworkParser(fixture_dir + "/network.yml");
    const auto topology = construct_topology(network_parser);
    CongestionAwareNetworkApi::set_event_queue(event_queue);
    CongestionAwareNetworkApi::set_topology(topology);
    const auto fluid_scheduler = std::make_shared<FluidScheduler>(
        event_queue, topology->get_directed_links(),
        network_parser.get_fluid_max_active_flows(),
        network_parser.get_fluid_max_route_memberships(),
        network_parser.get_progress_report_event_interval());
    CongestionAwareNetworkApi::set_fluid_scheduler(fluid_scheduler);

    const auto memory_api = std::make_unique<AnalyticalRemoteMemory>(
        fixture_dir + "/remote_memory.json");

    auto network_apis =
        std::vector<std::unique_ptr<CongestionAwareNetworkApi>>();
    auto systems = std::vector<Sys*>();
    const auto npus_count = static_cast<int>(topology->get_npus_count());
    expect(npus_count == 2, "fixture topology has 2 ranks");
    const auto dims = topology->get_npus_count_per_dim();
    const std::vector<int> queues_per_dim(dims.size(), 1);
    for (int i = 0; i < npus_count; i++) {
        auto net = std::make_unique<CongestionAwareNetworkApi>(i);
        // Default ExecutionMode::Static: Workload builds its own
        // ETFeederGraphSource over <prefix>.<id>.et from the fixture dir.
        systems.push_back(new Sys(i, workload_prefix, "empty",
                                  fixture_dir + "/system.json",
                                  memory_api.get(), net.get(), dims,
                                  queues_per_dim, 1.0, 1.0, false));
        network_apis.push_back(std::move(net));
    }

    // Static path proof: node-id retention is static-only.
    for (int i = 0; i < npus_count; i++) {
        expect(systems[i]->workload->hw_resource->tracks_node_ids(),
               "rank " + std::to_string(i) + ": static node-id retention");
    }

    // ---- first issue pass (static entry: Workload::fire) ----------------
    for (int i = 0; i < npus_count; i++) {
        systems[i]->workload->fire();
    }

    // ---- gate check right after the first pass ---------------------------
    // rank0: BOTH remote MEMs hold the remote-MEM slot simultaneously and
    // the COMM_SEND holds the comm slot; nothing was left un-issued.
    {
        auto* hw = systems[0]->workload->hw_resource;
        expect(hw->num_in_flight_remote_mem_ops == 2,
               "first pass: both MEMs in flight (got " +
                   std::to_string(hw->num_in_flight_remote_mem_ops) + ")");
        expect(hw->num_remote_mem_ops == 2,
               "first pass: two remote-MEM occupies, got " +
                   std::to_string(hw->num_remote_mem_ops));
        expect(hw->remote_mem_ops_node.size() == 2,
               "first pass: remote-MEM id set == {mem_a, mem_b}");
        expect(hw->num_in_flight_gpu_comm_ops == 1,
               "first pass: COMM_SEND issued on the comm slot");
        expect(hw->num_in_flight_remote_mem_ops == 2 &&
                   hw->remote_mem_ops_node.count(1) == 1 &&
                   hw->remote_mem_ops_node.count(2) == 1,
               "first pass: comm slot did not gate the second MEM");
        const auto st = memory_api->get_port_stats(0);
        expect(st.issued_count == 2,
               "first pass: port saw both MEM transactions, got " +
                   std::to_string(st.issued_count));
        expect(st.in_flight_count == 2,
               "first pass: port in_flight == 2, got " +
                   std::to_string(st.in_flight_count));
    }

    // ---- mid-flight probe ------------------------------------------------
    // Tick 150: both MEM streams active ([100,300) window), the send still
    // in the network, no terminal recorded anywhere yet.
    schedule_probe_at(
        systems[0], 150, [mem = memory_api.get()]() {
            const auto st = mem->get_port_stats(0);
            expect(st.streaming_count == 2,
                   "probe: both MEM streams active at Tick 150");
            expect(st.in_flight_count == 2,
                   "probe: both MEM transactions in flight at Tick 150");
            expect(g_recorder.order.empty(),
                   "probe: nothing terminal before Tick 150 "
                   "(finish gate held)");
        });

    fluid_scheduler->flush_pending_starts();
    fluid_scheduler->mark_event_loop_started();

    // Two watchdogs (proceed count + wall clock): convert a hypothetical
    // zero-progress pathology into a failed assertion with diagnostics.
    const uint64_t kProceedCap = 50000000;
    const auto kWallCap = std::chrono::seconds(300);
    const auto t0 = std::chrono::steady_clock::now();
    uint64_t proceeds = 0;
    while (!event_queue->finished()) {
        event_queue->proceed();
        proceeds++;
        if (proceeds > kProceedCap ||
            std::chrono::steady_clock::now() - t0 > kWallCap) {
            std::fprintf(stderr,
                         "[remote_port_static_gate_test] watchdog fired "
                         "after %llu proceeds; port0 issued=%llu "
                         "completed=%llu in_flight=%llu streaming=%llu\n",
                         static_cast<unsigned long long>(proceeds),
                         static_cast<unsigned long long>(
                             memory_api->get_port_stats(0).issued_count),
                         static_cast<unsigned long long>(
                             memory_api->get_port_stats(0).completed_count),
                         static_cast<unsigned long long>(
                             memory_api->get_port_stats(0).in_flight_count),
                         static_cast<unsigned long long>(
                             memory_api->get_port_stats(0).streaming_count));
            expect(false, "event loop watchdog fired");
            break;
        }
    }
    const uint64_t end_tick = Sys::boostedTick();

    // ---- terminals -------------------------------------------------------
    // Remote-port anchor: both 600 B MEMs share the port at 3 B/ns over
    // [100,300) -> fluid 300.0, callback Tick 300, delivered in issue
    // (id) order.
    expect(g_recorder.tick(0, 1) == 300,
           "rank0 MEM_A callback Tick 300, got " +
               std::to_string(g_recorder.tick(0, 1)));
    expect(g_recorder.tick(0, 2) == 300,
           "rank0 MEM_B callback Tick 300, got " +
               std::to_string(g_recorder.tick(0, 2)));
    expect(g_recorder.count(0, 3) == 1,
           "rank0 COMM_SEND terminal exactly once");
    expect(g_recorder.count(1, 1) == 1,
           "rank1 COMM_RECV terminal exactly once");
    expect(g_recorder.order.size() == 4,
           "exactly four terminals (2 MEM + send + recv), got " +
               std::to_string(g_recorder.order.size()));

    // ---- static finish gate ---------------------------------------------
    // The gate fires only after every remote node terminated and the
    // remote-MEM in-flight count reached zero (Workload::call static
    // branch); the loop end cannot precede the last remote completion.
    expect(systems[0]->workload->is_finished,
           "rank0 static finish gate reached (is_finished)");
    expect(systems[1]->workload->is_finished,
           "rank1 static finish gate reached (is_finished)");
    expect(end_tick >= 300,
           "event loop ended at Tick " + std::to_string(end_tick) +
               ", not before the Tick-300 remote completions");

    // ---- counters / sets released exactly once ---------------------------
    {
        auto* hw0 = systems[0]->workload->hw_resource;
        expect(hw0->num_in_flight_remote_mem_ops == 0,
               "rank0 remote-MEM in-flight drained");
        expect(hw0->num_remote_mem_ops == 2,
               "rank0 cumulative remote-MEM occupies == 2 (each node "
               "occupied once), got " +
                   std::to_string(hw0->num_remote_mem_ops));
        expect(hw0->remote_mem_ops_node.empty(),
               "rank0 remote-MEM id set fully released");
        expect(hw0->num_in_flight_gpu_comm_ops == 0,
               "rank0 comm slot released");
        expect(hw0->num_gpu_comms == 1,
               "rank0 exactly one comm occupy/release cycle");
        expect(hw0->gpu_comms_node.empty(),
               "rank0 comm id set fully released");
        auto* hw1 = systems[1]->workload->hw_resource;
        expect(hw1->num_in_flight_remote_mem_ops == 0 &&
                   hw1->num_remote_mem_ops == 0 &&
                   hw1->remote_mem_ops_node.empty(),
               "rank1 remote-MEM counters untouched");
        expect(hw1->num_in_flight_gpu_comm_ops == 0 &&
                   hw1->gpu_comms_node.empty(),
               "rank1 COMM_RECV is a gate no-op");
    }

    // ---- port integrals + conservation -----------------------------------
    {
        const auto st = memory_api->get_port_stats(0);
        expect(st.issued_count == 2 && st.issued_bytes == 1200,
               "port0 totals 2 / 1200");
        expect(st.peak_streaming >= 2,
               "port0 peak_streaming >= 2 (event-interval)");
        expect(st.peak_streaming == 2, "port0 peak_streaming == 2");
        expect(st.shared_busy_ns > 0.0, "port0 shared_busy_ns > 0");
        expect_near(st.shared_busy_ns, 200.0, 1e-9,
                    "port0 shared window [100,300) = 200 ns");
        expect_near(st.port_busy_ns, 200.0, 1e-9,
                    "port0 port_busy 200 ns");
        expect(st.redistribution_events == 0,
               "port0 simultaneous finish, no survivor");
        expect(st.arrival_redistribution_events == 0,
               "port0 single same-instant arrival wave");
        const auto st1 = memory_api->get_port_stats(1);
        expect(st1.issued_count == 0,
               "port1 untouched (rank1 has only the recv)");
    }
    expect_port_conservation(*memory_api, 0, 6.0, "static_gate");
    expect_port_conservation(*memory_api, 1, 6.0, "static_gate");
    expect(memory_api->is_drained(), "backend fully drained");

    std::printf(
        "[remote_port_static_gate_test] gates: end_tick=%llu "
        "event_loop_proceeds=%llu terminals=%zu\n",
        static_cast<unsigned long long>(end_tick),
        static_cast<unsigned long long>(proceeds), g_recorder.order.size());
    std::fflush(stdout);

    for (auto it : systems) {
        delete it;
    }
    systems.clear();

    AstraSim::LoggerFactory::shutdown();

    if (!g_ok) {
        std::fprintf(stderr, "[remote_port_static_gate_test] FAIL\n");
        return 1;
    }
    std::printf("[remote_port_static_gate_test] ALL PASS: static first pass "
                "issues both remote MEMs beside the comm send, counters/"
                "sets released exactly once, static finish gate waited for "
                "all remote terminals\n");
    return 0;
}
