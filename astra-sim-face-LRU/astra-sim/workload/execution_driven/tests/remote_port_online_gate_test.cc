/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

/*

remote_port_online_gate_test.cc -- online NodeView issue-gating system
fixture for the SerDes off-chip-link rework (plan: 片外共享内存端口并发化
改造执行方案 V5.3, stage 5.3; target
AstraSim_Analytical_Congestion_Aware_RemotePortOnlineGateTest).

This file is a NEW fixture built directly on NodeStoreGraphSource / NodeView.
The former RemoteFifoLedgerTest this plan family once referenced was deleted
from this repository (2026-09-24 worktree) and is NOT extended or referenced
here. The gate semantics under test are the reworked HardwareResource
classification (HardwareResource.hh:57-105): remote MEM nodes occupy an
independent, count-based, UNBOUNDED remote-MEM slot (is_available never
blocks on it, and it is NOT a comm slot), while CPU/COMP/COMM keep their
legacy slots.

One PER_NPU phase (bw=6 B/ns, latency=100 ns, ranks 0..5 on the two-dim
Mesh [3,2] -- the topology shape the in-repo p2p reference fixture
local_hbm_model_test.cc is validated with), real online stack (Sys +
Workload + HardwareResource + NodeStoreGraphSource + real
AnalyticalRemoteMemory + real fluid network + real Sys event queue):

  rank0  SAME-RANK TWO INDEPENDENT MEMs, no dependency edges, ONE issue
         pass. Immediately after the pass (PortStats): issued_count == 2,
         in_flight_count == 2 (latency included), latency_waiting_count == 2
         and the source's dep-free set is empty (both nodes were taken and
         issued). After the run the EVENT-INTERVAL integrals of the backend
         (PortStats, rebuilt per continuous service interval -- never
         sampled) assert the concurrent sharing: peak_streaming >= 2
         (exactly 2) and shared_busy_ns > 0 (exactly 200.0 = the [100,300)
         window at 3 B/ns each; 600/3 = 200 -> both callbacks Tick 300).
  rank1  TWO MEMs ISSUED BACK-TO-BACK while the first is still in flight:
         A=1200B at t0; B=600B issued from the real event queue at Tick 150
         (A streaming). Probe at Tick 160: in_flight_count == 2 with
         streaming_count == 1 (B still in its latency stage). Segment table
         (per-stream rate 6/N):
           [100,150) N=1 rate 6: A serves 300, rem 900
           [150,250) N=1 rate 6: A serves 600, rem 300
           [250,350) N=2 rate 3: A 300 -> fluid 350 (survivor B ->
                     redistribution +1); B serves 300, rem 300
           [350,400] N=1 rate 6: B 300 -> fluid 400
         busy [100,400) = 300; shared [250,350) = 100; arrival re-split +1.
  rank2  MEM FIRST, then COMM_SEND on the SAME rank, same Tick: the MEM
         occupies the remote-MEM slot while the COMM_SEND must still issue
         immediately (a remote MEM that wrongly held the comm single slot
         would leave the send in the dep-free set -- asserted empty right
         after the pass). Probe at Tick 150: port in_flight == 1 (MEM in
         flight) and no terminal record yet (COMM also in flight): the two
         gates coexist. The MEM runs ALONE: 600 B at 6 B/ns over [100,200)
         -> callback Tick 200; COMM completes exactly once.
  rank3  COMM_SEND FIRST, then MEM on the SAME rank, same Tick: the mirror
         direction. A COMM_SEND occupying the comm slot must not block the
         remote MEM (asserted via the empty dep-free set + MEM callback
         Tick 200 + port stats). rank3 store layout: id 1 = COMM_RECV for
         rank2's send (tag 21), id 2 = COMM_SEND (tag 22), id 3 = the MEM.
  rank4  destination rank of rank3's send: carries only the matching
         COMM_RECV; its remote port stays untouched (issued_count == 0).

Config discipline (plan stage 5 避雷): system.json follows the official
template shape (the since-removed
inputs/system/analytical/dgx_v100_4gpu.json) with
"scheduling-policy", "preferred-dataset-splits" and "collective-optimization"
present and all four *-implementation keys as ["ring","ring"]; no
*-implementation-custom, no doubleBinaryTree, no comm_group.json involved
(no collective issued); the retired dead key "boost-mode" is not resurrected.

Build (registered in astra-sim/network_frontend/analytical/CMakeLists.txt):
  cmake --build build/astra_analytical/build_congestion_aware --target \
      AstraSim_Analytical_Congestion_Aware_RemotePortOnlineGateTest
Run:
  build/astra_analytical/build_congestion_aware/bin/\
      AstraSim_Analytical_Congestion_Aware_RemotePortOnlineGateTest
Exit code 0 on ALL PASS.
*/

#include "astra-sim/common/Logging.hh"
#include "astra-sim/workload/Workload.hh"
#include "astra-sim/workload/execution_driven/CompletionObserver.hh"
#include "astra-sim/workload/execution_driven/NodeStore.hh"
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
#include <fstream>
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
                     "[remote_port_online_gate_test] FAIL: %s\n",
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
                     "[remote_port_online_gate_test] FAIL: %s: got %.12f "
                     "expected %.12f (tol %.3e)\n",
                     what.c_str(), value, expected, tol);
        g_ok = false;
    }
}

void write_file(const std::string& path, const std::string& content) {
    std::ofstream out(path, std::ios::out | std::ios::trunc);
    if (!out) {
        std::fprintf(stderr,
                     "[remote_port_online_gate_test] FAIL: cannot write "
                     "%s\n",
                     path.c_str());
        g_ok = false;
        return;
    }
    out << content;
}

// ---------------------------------------------------------------------------
// Terminal observation: the CompletionObserver hook fires synchronously on
// the Workload::call terminal path, in backend delivery order.
// ---------------------------------------------------------------------------

struct Recorder {
    std::vector<std::tuple<int, uint64_t, uint64_t>> order;
    std::map<std::pair<int, uint64_t>, uint64_t> tick_of;
    std::map<std::pair<int, uint64_t>, uint64_t> count_of;

    void clear() {
        order.clear();
        tick_of.clear();
        count_of.clear();
    }
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
                     "[remote_port_online_gate_test] FAIL: rank %d node "
                     "%llu terminal status %d\n",
                     rank, static_cast<unsigned long long>(node_id),
                     status);
        g_ok = false;
    }
    rec->order.emplace_back(rank, node_id, tick);
    rec->tick_of[{rank, node_id}] = tick;
    rec->count_of[{rank, node_id}] += 1;
}

void expect_terminal(const Recorder& rec, int rank, uint64_t node,
                     uint64_t tick, const std::string& what) {
    expect(rec.count(rank, node) == 1,
           what + ": terminal exactly once (rank " + std::to_string(rank) +
               " node " + std::to_string(node) + ", got " +
               std::to_string(rec.count(rank, node)) + ")");
    expect(rec.tick(rank, node) == tick,
           what + ": callback tick (rank " + std::to_string(rank) +
               " node " + std::to_string(node) + ") got " +
               std::to_string(rec.tick(rank, node)) + " expected " +
               std::to_string(tick));
}

// ---------------------------------------------------------------------------
// Node builders + issue plumbing (mirrors local_hbm_model_test.cc). MEM
// nodes carry no hbm-access-mode / kv-restore flag: exactly one remote-port
// transaction each.
// ---------------------------------------------------------------------------

ExecutionDriven::OnlineNode make_mem_node(int rank, uint64_t bytes,
                                          const std::string& name) {
    ExecutionDriven::OnlineNode node;
    node.rank = rank;
    node.kind = ExecutionDriven::NodeKind::MemLoad;
    node.node_type = 2;  // ChakraNodeType::MEM_LOAD_NODE
    node.name = name;
    node.compute.tensor_size = bytes;
    return node;
}

ExecutionDriven::OnlineNode make_send_node(int rank, int dst,
                                           uint64_t bytes, uint32_t tag,
                                           const std::string& name) {
    ExecutionDriven::OnlineNode node;
    node.rank = rank;
    node.kind = ExecutionDriven::NodeKind::CommSend;
    node.node_type = 5;  // ChakraNodeType::COMM_SEND_NODE
    node.name = name;
    node.comm.src = rank;
    node.comm.dst = dst;
    node.comm.bytes = bytes;
    node.comm.tag = tag;
    node.comm.hbm_charge = false;  // pure gate test: no local-HBM endpoint
    return node;
}

ExecutionDriven::OnlineNode make_recv_node(int rank, int src,
                                           uint64_t bytes, uint32_t tag,
                                           const std::string& name) {
    ExecutionDriven::OnlineNode node;
    node.rank = rank;
    node.kind = ExecutionDriven::NodeKind::CommRecv;
    node.node_type = 6;  // ChakraNodeType::COMM_RECV_NODE
    node.name = name;
    node.comm.src = src;
    node.comm.dst = rank;
    node.comm.bytes = bytes;
    node.comm.tag = tag;
    node.comm.hbm_charge = false;
    return node;
}

void issue_all_dep_free(Sys* sys,
                        ExecutionDriven::NodeStoreGraphSource* source) {
    for (const auto& nv : source->dep_free_nodes()) {
        if (sys->workload->hw_resource->is_available(nv)) {
            sys->workload->issue(nv);
        }
    }
}

void add_and_issue(Sys* sys, ExecutionDriven::NodeStoreGraphSource* source,
                   ExecutionDriven::OnlineNode node) {
    source->store().add_node(std::move(node));
    issue_all_dep_free(sys, source);
}

// Staggered issue from the real Sys event queue at Tick == delay.
class StaggeredIssueEvent : public Callable {
  public:
    StaggeredIssueEvent(Sys* sys,
                        ExecutionDriven::NodeStoreGraphSource* source,
                        ExecutionDriven::OnlineNode node)
        : sys_(sys), source_(source), node_(std::move(node)) {}

    void call(EventType, CallData*) override {
        add_and_issue(sys_, source_, std::move(node_));
        delete this;
    }

  private:
    Sys* sys_;
    ExecutionDriven::NodeStoreGraphSource* source_;
    ExecutionDriven::OnlineNode node_;
};

void schedule_issue_at(Sys* sys,
                       ExecutionDriven::NodeStoreGraphSource* source,
                       ExecutionDriven::OnlineNode node, Tick delay) {
    sys->register_event(new StaggeredIssueEvent(sys, source, std::move(node)),
                        EventType::General, nullptr, delay);
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

// ---------------------------------------------------------------------------
// Phase assembly (fresh event queue + topology + fluid scheduler + memory;
// single phase in this fixture).
// ---------------------------------------------------------------------------

// Official template shape (the since-removed
// inputs/system/analytical/dgx_v100_4gpu.json).
const char* kSystemJson =
    "{\n"
    "  \"scheduling-policy\": \"LIFO\",\n"
    "  \"endpoint-delay\": 1,\n"
    "  \"active-chunks-per-dimension\": 1,\n"
    "  \"preferred-dataset-splits\": 1,\n"
    "  \"all-reduce-implementation\": [\"ring\", \"ring\"],\n"
    "  \"all-gather-implementation\": [\"ring\", \"ring\"],\n"
    "  \"reduce-scatter-implementation\": [\"ring\", \"ring\"],\n"
    "  \"all-to-all-implementation\": [\"ring\", \"ring\"],\n"
    "  \"collective-optimization\": \"localBWAware\"\n"
    "}\n";

struct PhaseStack {
    std::shared_ptr<EventQueue> event_queue;
    std::shared_ptr<NetworkParser> parser;
    std::shared_ptr<Topology> topology;
    std::shared_ptr<FluidScheduler> fluid_scheduler;
    std::unique_ptr<AnalyticalRemoteMemory> memory;
    std::vector<std::unique_ptr<CongestionAwareNetworkApi>> network_apis;
    std::vector<Sys*> systems;
    std::vector<std::shared_ptr<ExecutionDriven::NodeStoreGraphSource>>
        sources;
};

void bootstrap_phase(const std::string& dir, const std::string& tag,
                     int ranks, const std::string& remote_cfg,
                     PhaseStack& stack) {
    g_recorder.clear();  // rank/node ids repeat across fixtures
    const std::string system_path = dir + "/system_" + tag + ".json";
    const std::string network_path = dir + "/network_" + tag + ".yml";
    write_file(system_path, kSystemJson);
    // Two-dimension Mesh, the exact topology shape the in-repo p2p
    // reference fixture (make_local_hbm_fixture_et.py /
    // local_hbm_model_test.cc) is validated with: the first failing run
    // showed every PacketSent/PacketReceived completion missing on a
    // single-dimension Mesh, while the port-model events were fine.
    write_file(network_path,
               "topology: [ Mesh, Mesh ]\n"
               "npus_count: [ 3, 2 ]\n"
               "bandwidth: [ 4050, 4050 ]\n"
               "latency: [ 25, 25 ]\n");

    stack.event_queue = std::make_shared<EventQueue>();
    stack.parser = std::make_shared<NetworkParser>(network_path);
    stack.topology = construct_topology(*stack.parser);
    CongestionAwareNetworkApi::set_event_queue(stack.event_queue);
    CongestionAwareNetworkApi::set_topology(stack.topology);
    stack.fluid_scheduler = std::make_shared<FluidScheduler>(
        stack.event_queue, stack.topology->get_directed_links(),
        stack.parser->get_fluid_max_active_flows(),
        stack.parser->get_fluid_max_route_memberships(),
        stack.parser->get_progress_report_event_interval());
    CongestionAwareNetworkApi::set_fluid_scheduler(stack.fluid_scheduler);

    stack.memory = std::make_unique<AnalyticalRemoteMemory>(remote_cfg);

    const auto dims = stack.topology->get_npus_count_per_dim();
    const std::vector<int> queues_per_dim(dims.size(), 1);
    expect(static_cast<int>(stack.topology->get_npus_count()) == ranks,
           "phase " + tag + ": topology rank count");
    for (int i = 0; i < ranks; i++) {
        auto source =
            std::make_shared<ExecutionDriven::NodeStoreGraphSource>();
        stack.sources.push_back(source);
        auto net = std::make_unique<CongestionAwareNetworkApi>(i);
        stack.systems.push_back(new Sys(
            i, dir + "/none.et", "empty", system_path, stack.memory.get(),
            net.get(), dims, queues_per_dim, 1.0, 1.0, false,
            ExecutionDriven::ExecutionMode::Online, source));
        stack.network_apis.push_back(std::move(net));
    }
}

void teardown_phase(PhaseStack& stack) {
    for (Sys* sys : stack.systems) {
        delete sys;
    }
    stack.systems.clear();
    stack.sources.clear();
    stack.network_apis.clear();
    // ~AnalyticalRemoteMemory runs the unconditional verify_drained()
    // fail-closed audit (plan sec.3.4).
    stack.memory.reset();
    stack.fluid_scheduler.reset();
    stack.topology.reset();
    stack.parser.reset();
    stack.event_queue.reset();
}

uint64_t run_event_loop(EventQueue& queue, const std::string& tag) {
    const uint64_t kProceedCap = 50000000;
    uint64_t proceeds = 0;
    while (!queue.finished()) {
        queue.proceed();
        proceeds++;
        if (proceeds > kProceedCap) {
            expect(false, "phase " + tag + ": event loop watchdog fired");
            return proceeds;
        }
    }
    return proceeds;
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

}  // namespace

int main() {
    char dir_template[] = "/tmp/remote_port_online_gate_XXXXXX";
    const char* dir_name = mkdtemp(dir_template);
    if (dir_name == nullptr) {
        std::perror("mkdtemp");
        return 1;
    }
    const std::string dir(dir_name);

    AstraSim::LoggerFactory::init("empty", "off");
    ExecutionDriven::CompletionObserver::instance().set_hook(terminal_hook,
                                                             &g_recorder);

    const std::string tag = "onlinegate";
    const std::string cfg_path = dir + "/remote_" + tag + ".json";
    write_file(cfg_path,
               "{\n"
               "  \"memory-type\": \"PER_NPU_MEMORY_EXPANSION\",\n"
               "  \"npu-ids\": [0, 1, 2, 3, 4, 5],\n"
               "  \"remote-mem-bw\": 6,\n"
               "  \"remote-mem-latency\": 100\n"
               "}\n");

    PhaseStack stack;
    bootstrap_phase(dir, tag, 6, cfg_path, stack);
    auto* mem = stack.memory.get();
    auto* sys0 = stack.systems[0];
    auto* src0 = stack.sources[0].get();

    // ---- rank0: two same-rank independent MEMs, ONE issue pass ----------
    // Both nodes enter the store with no dependency edges, then a single
    // issue pass takes and issues BOTH (plan stage 5.3).
    // [0,100) latency overlapped; [100,300) N=2 -> 3 B/ns each;
    // 600/3 = 200 -> both fluid 300.0, callback Tick 300.
    src0->store().add_node(make_mem_node(0, 600, "gate_m0"));
    src0->store().add_node(make_mem_node(0, 600, "gate_m1"));
    issue_all_dep_free(sys0, src0);  // ONE pass: issues both MEMs
    {
        // plan stage 5.3 literal check, BEFORE any event fires.
        const auto st = mem->get_port_stats(0);
        expect(st.issued_count == 2,
               tag + ": rank0 one-pass issued_count == 2, got " +
                   std::to_string(st.issued_count));
        expect(st.in_flight_count == 2,
               tag + ": rank0 one-pass in_flight_count == 2, got " +
                   std::to_string(st.in_flight_count));
        expect(st.latency_waiting_count == 2,
               tag + ": rank0 both in latency stage");
        expect(st.completed_count == 0,
               tag + ": rank0 nothing completed yet");
        expect(src0->dep_free_nodes().empty(),
               tag + ": rank0 dep-free set drained by the issue pass");
    }

    // ---- rank1: second MEM issued while the first is in flight ----------
    // A=1200B at t0; B=600B issued at Tick 150 from the real event queue.
    // Segment table (per-stream rate = 6/N):
    //   [100,150) N=1 rate 6: A serves 300, rem 900
    //   [150,250) N=1 rate 6: A serves 600, rem 300
    //   [250,350) N=2 rate 3: A serves 300 -> fluid 350 (cb 350, a
    //             completion WITH survivor B -> redistribution +1);
    //             B serves 300, rem 300
    //   [350,400] N=1 rate 6: B serves 300 -> fluid 400 (cb 400)
    // busy [100,400) = 300; shared [250,350) = 100; arrival re-split +1
    // (B joins while A streams).
    auto* sys1 = stack.systems[1];
    auto* src1 = stack.sources[1].get();
    add_and_issue(sys1, src1, make_mem_node(1, 1200, "gate_a"));
    schedule_issue_at(sys1, src1, make_mem_node(1, 600, "gate_b"), 150);
    schedule_probe_at(
        sys1, 160, [mem]() {
            const auto st = mem->get_port_stats(1);
            expect(st.issued_count == 2,
                   "rank1: back-to-back issue accepted (issued 2)");
            expect(st.in_flight_count == 2,
                   "rank1: both transactions in flight at Tick 160");
            expect(st.streaming_count == 1,
                   "rank1: only A streaming at Tick 160 (B in latency)");
            expect(st.latency_waiting_count == 1,
                   "rank1: B latency-waiting at Tick 160");
        });

    // ---- rank2: MEM first, then COMM_SEND (same rank, same Tick) --------
    // If a remote MEM wrongly held the legacy comm single slot, the send
    // below could not issue and would stay in the dep-free set.
    auto* sys2 = stack.systems[2];
    auto* src2 = stack.sources[2].get();
    add_and_issue(sys2, src2, make_mem_node(2, 600, "gate_mem_first"));
    add_and_issue(sys2, src2,
                  make_send_node(2, 3, 1000000, 21, "gate_send_after_mem"));
    // rank3's matching recv for rank2's send (tag 21).
    add_and_issue(stack.systems[3], stack.sources[3].get(),
                  make_recv_node(3, 2, 1000000, 21, "gate_recv21"));

    // ---- rank3/4: COMM_SEND first, then MEM (mirror direction) ----------
    // If the remote MEM wrongly consumed the comm slot, it could not issue
    // behind the in-flight send and would stay in the dep-free set.
    auto* sys3 = stack.systems[3];
    auto* src3 = stack.sources[3].get();
    add_and_issue(sys3, src3,
                  make_send_node(3, 4, 1000000, 22, "gate_send_first"));
    add_and_issue(sys3, src3, make_mem_node(3, 600, "gate_mem_after_send"));
    add_and_issue(stack.systems[4], stack.sources[4].get(),
                  make_recv_node(4, 3, 1000000, 22, "gate_recv22"));

    {
        // Both mixed-gate ranks: every node taken and issued immediately.
        expect(src2->dep_free_nodes().empty(),
               tag + ": rank2 MEM+SEND both issued (dep-free empty)");
        expect(src3->dep_free_nodes().empty(),
               tag + ": rank3 SEND+MEM both issued (dep-free empty)");
        const auto s2 = mem->get_port_stats(2);
        const auto s3 = mem->get_port_stats(3);
        expect(s2.issued_count == 1 && s2.in_flight_count == 1,
               tag + ": rank2 exactly the MEM entered the remote port");
        expect(s3.issued_count == 1 && s3.in_flight_count == 1,
               tag + ": rank3 exactly the MEM entered the remote port");
    }

    // Coexistence probes: at Tick 150 the MEM is inside its port latency /
    // service and the COMM is still in the network -- both gates held at
    // the same time, no terminal record anywhere yet (rank0 callbacks are
    // at Tick 300, rank1 at 350/400, the 1MB sends need ~247ns + 25ns
    // latency).
    schedule_probe_at(
        sys2, 150, [mem]() {
            const auto st = mem->get_port_stats(2);
            expect(st.in_flight_count == 1,
                   "rank2 probe: MEM still in flight at Tick 150");
            expect(!g_recorder.rank_touched(2),
                   "rank2 probe: COMM not terminal yet (both in flight)");
        });
    schedule_probe_at(
        sys3, 150, [mem]() {
            const auto st = mem->get_port_stats(3);
            expect(st.in_flight_count == 1,
                   "rank3 probe: MEM still in flight at Tick 150");
            expect(!g_recorder.rank_touched(3),
                   "rank3 probe: COMM not terminal yet (both in flight)");
        });

    // Root cause of the previous run's missing p2p completions: fluid flows
    // register as pending starts at issue time and are only launched by
    // these two calls (local_hbm_model_test.cc does the same). Without the
    // flush the PacketSent/PacketReceived callbacks never enter the event
    // queue and the loop ends while the sends are still pending.
    stack.fluid_scheduler->flush_pending_starts();
    stack.fluid_scheduler->mark_event_loop_started();

    const uint64_t proceeds = run_event_loop(*stack.event_queue, tag);
    std::fflush(stdout);

    // ---- callbacks -------------------------------------------------------
    // rank0: the one-pass pair completes together at Tick 300.
    expect_terminal(g_recorder, 0, 1, 300, tag + ": rank0 MEM n1");
    expect_terminal(g_recorder, 0, 2, 300, tag + ": rank0 MEM n2");
    // rank1: back-to-back pair per the segment table above (A completes at
    // 350 while B survives; B re-splits to full rate and lands at 400).
    expect_terminal(g_recorder, 1, 1, 350, tag + ": rank1 MEM A");
    expect_terminal(g_recorder, 1, 2, 400, tag + ": rank1 MEM B");
    // rank2/3: each MEM runs ALONE on its port: 600 B at 6 B/ns over
    // [100,200) -> fluid 200.0, callback Tick 200. rank3 store layout:
    // id 1 = recv21 (added with rank2's pair), id 2 = send22, id 3 = MEM.
    expect_terminal(g_recorder, 2, 1, 200, tag + ": rank2 MEM");
    expect(g_recorder.count(2, 2) == 1,
           tag + ": rank2 COMM_SEND completed exactly once");
    expect_terminal(g_recorder, 3, 3, 200, tag + ": rank3 MEM");
    expect(g_recorder.count(3, 1) == 1,
           tag + ": rank3 COMM_RECV (tag 21) completed exactly once");
    expect(g_recorder.count(3, 2) == 1,
           tag + ": rank3 COMM_SEND (tag 22) completed exactly once");
    expect(g_recorder.count(4, 1) == 1,
           tag + ": rank4 COMM_RECV (tag 22) completed exactly once");
    expect(g_recorder.count(4, 2) == 0,
           tag + ": rank4 has no second node");

    // ---- event-interval integrals: the concurrent-sharing evidence -------
    {
        // rank0: peak_streaming >= 2 (exactly 2) and shared_busy_ns > 0
        // (exactly the [100,300) window at N=2).
        const auto st = mem->get_port_stats(0);
        expect(st.peak_streaming >= 2,
               tag + ": rank0 peak_streaming >= 2, got " +
                   std::to_string(st.peak_streaming));
        expect(st.peak_streaming == 2, tag + ": rank0 peak_streaming == 2");
        expect(st.shared_busy_ns > 0.0,
               tag + ": rank0 shared_busy_ns > 0");
        expect_near(st.shared_busy_ns, 200.0, 1e-9,
                    tag + ": rank0 shared window [100,300) = 200 ns");
        expect_near(st.port_busy_ns, 200.0, 1e-9,
                    tag + ": rank0 port_busy 200");
        expect(st.peak_in_flight == 2, tag + ": rank0 peak_in_flight 2");
        expect(st.redistribution_events == 0,
               tag + ": rank0 simultaneous finish, no survivor");
    }
    {
        const auto st = mem->get_port_stats(1);
        expect(st.peak_streaming == 2, tag + ": rank1 peak_streaming 2");
        expect(st.shared_busy_ns > 0.0,
               tag + ": rank1 shared_busy_ns > 0");
        expect_near(st.shared_busy_ns, 100.0, 1e-9,
                    tag + ": rank1 shared window [250,350) = 100 ns");
        expect_near(st.port_busy_ns, 300.0, 1e-9,
                    tag + ": rank1 port_busy [100,400) = 300");
        expect(st.arrival_redistribution_events == 1,
               tag + ": rank1 arrival-driven re-split +1 (B joins at 250)");
        expect(st.redistribution_events == 1,
               tag + ": rank1 completion re-split +1 (A ends at 350, "
                     "B survives)");
    }
    {
        // Mixed-gate ranks: only the MEM entered the port; the COMM leg
        // never charged the remote port. Each MEM runs alone: 600 B at
        // 6 B/ns over [100,200).
        const auto s2 = mem->get_port_stats(2);
        const auto s3 = mem->get_port_stats(3);
        const auto s4 = mem->get_port_stats(4);
        expect(s2.issued_count == 1 && s2.issued_bytes == 600,
               tag + ": rank2 port carried only the MEM");
        expect(s3.issued_count == 1 && s3.issued_bytes == 600,
               tag + ": rank3 port carried only the MEM");
        expect(s4.issued_count == 0,
               tag + ": rank4 remote port untouched");
        expect_near(s2.port_busy_ns, 100.0, 1e-9,
                    tag + ": rank2 MEM ran alone [100,200)");
        expect_near(s3.port_busy_ns, 100.0, 1e-9,
                    tag + ": rank3 MEM ran alone [100,200)");
        expect(s2.shared_busy_ns == 0.0 && s3.shared_busy_ns == 0.0,
               tag + ": mixed ranks never share the port");
    }
    for (std::size_t p = 0; p < 6; p++) {
        expect_port_conservation(*mem, p, 6.0, tag);
    }

    // Sensing stays off: no per-transaction rows, no lazy detail file.
    expect(mem->transaction_rows_written() == 0,
           tag + ": sensing off -> zero transaction rows");
    expect(access((dir + "/txn_" + tag + ".jsonl").c_str(), F_OK) != 0,
           tag + ": sensing off -> no detail file created");

    expect(mem->is_drained(), tag + ": backend fully drained");

    std::printf(
        "[remote_port_online_gate_test] gates: event_loop_proceeds=%llu\n",
        static_cast<unsigned long long>(proceeds));

    teardown_phase(stack);

    ExecutionDriven::CompletionObserver::instance().set_hook(nullptr,
                                                             nullptr);
    AstraSim::LoggerFactory::shutdown();

    // Workspace hygiene: remove every file this test created.
    const char* files[] = {
        "system_onlinegate.json", "network_onlinegate.yml",
        "remote_onlinegate.json",
    };
    for (const char* f : files) {
        std::remove((dir + "/" + f).c_str());
    }
    ::rmdir(dir.c_str());

    if (!g_ok) {
        std::fprintf(stderr, "[remote_port_online_gate_test] FAIL\n");
        return 1;
    }
    std::printf("[remote_port_online_gate_test] ALL PASS: same-rank dual "
                "MEM one-pass issue (issued/in_flight 2), back-to-back "
                "issue in flight, MEM+COMM_SEND gates coexist in both "
                "orders, event-interval shared/streaming evidence\n");
    return 0;
}
