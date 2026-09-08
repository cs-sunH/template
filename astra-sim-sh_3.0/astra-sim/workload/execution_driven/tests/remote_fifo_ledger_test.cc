/******************************************************************************
This source code is licensed under the MIT license found in the LICENSE file in
the root directory of this source tree.
 *******************************************************************************/

/**
remote_fifo_ledger_test.cc -- R3 (方案 §3.6 / 阶段 E): RemoteFifoLedger
real-backend-port accounting fixture.

Self-contained online-mode fixture (no .et files, no Python; the
local_hbm_model_test.cc skeleton): tiny self-written configs in a temp dir,
4 ranks over a real congestion-aware line topology, one shared
AnalyticalRemoteMemory, MEM nodes injected straight into the per-rank
NodeStore, terminals recorded through the CompletionObserver hook with the
same-tick deferred re-issue pass. hbm-bandwidth-contention: 0 keeps the MEM
terminal exactly at the port-transaction completion (no HBM join), so the
node-terminal ticks are the analytical port-FIFO service ticks.

remote-mem-latency 50 ns + remote-mem-bw 100 B/ns make every runtime
hand-computable: runtime = 50 + bytes/100 -> 1000/2000/3000/4000 B map to
60/70/80/90 ns. Each backend port is a strict single server, so a port with
queued requests r1..rk started at t completes them serially at
t+r1, t+r1+r2, ... -- the closed-form ground truth the fixtures assert.

Three architectures (each one simulation in this process; the ledger
singleton is reset() before each):

  A. PER_NPU, sparse OUT-OF-ORDER npu-ids [3, 1]  (key-label defect)
     rank3 -> port0 (array INDEX, not rank 3!), rank1 -> port1.
     rank3: chained MEM_LOAD 1000 B then 2000 B (the in-rank MEM slot
     serializes same-rank MEMs, so the chain is also the physical order);
     rank1: one MEM_STORE 3000 B. rank0/2 issue no MEM node.
     Proves: active_ports() == {0,1} (the old sys_id keys were {1,3} --
     rank3's account was mislabeled port 3); per-port counters, by_rank
     attribution (rank3->port0), serial completion ticks 60/130 (port0)
     and 80 (port1); peak_in_flight_count 1 per port (a PER_NPU port has
     exactly one rank and the rank's MEM slot serializes -- peak>1 is only
     reachable on a SHARED port, covered by B/C below).

  B. PER_NODE, 2 nodes x 2 ranks  (shared-port active/pending/peak defect)
     rank0,1 -> port0; rank2,3 -> port1. All four ranks issue one MEM_LOAD
     at t=0 (rank loop order -> rank0 precedes rank1 on port0, etc.).
     Proves the t=0 REAL port state: port0 in_flight=2, active=1, pending=1,
     peak_in_flight_count=2, peak_in_flight_bytes=3000. The OLD per-rank
     accounting would print two keys each active=1/pending=0 (sum 2 != 1,
     peak_pending 0 != 1) -- recorded as a fixed evidence assertion.
     Serial completion ground truth: port0 60 (rank0, 1000 B) then 130
     (rank1, 2000 B); port1 80 (rank2) then 170 (rank3). The completion
     payload carries THIS transaction's bytes: port0 completed_bytes == 3000
     (1000 + 2000) even though the dequeued next request's tensor_size is
     visible at the same moment -- crediting the next request's size is
     exactly the mix-up the payload's tensor_size prevents.

  C. MEMORY_POOL, 4 ranks -> single port 0  (single shared port)
     rank0..3 each issue one MEM_STORE 1000/2000/3000/4000 B at t=0.
     Proves: active_ports() == {0} (the old accounting had 4 rank keys);
     t=0 in_flight=4, active=1, pending=3, peak_in_flight_count=4,
     peak_in_flight_bytes=10000; serial completion ticks 60/130/210/300
     (60 + 70 + 80 + 90); port_ranks(0) == {0,1,2,3}; four by_rank rows.

Common per architecture: issued/completed count and byte conservation,
per-port drained, architecture() label, and the sensing sidecar row built
by RemoteFifoLedger::sidecar_row (the SAME formatter main_online uses) --
substring checks for the real port keys, the source-rank lists and the
by_rank view.

Negative gate: after set_enabled(false), record_issue/record_completion
are no-ops (counters unchanged, no new keys).

Build: cmake target AstraSim_Analytical_Congestion_Aware_RemoteFifoLedgerTest.
Run: build/astra_analytical/build_congestion_aware/bin/\
     AstraSim_Analytical_Congestion_Aware_RemoteFifoLedgerTest
Exit code 0 on ALL PASS.
*/

#include "astra-sim/common/Logging.hh"
#include "astra-sim/workload/RemoteFifoLedger.hh"
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
#include <filesystem>
#include <fstream>
#include <functional>
#include <initializer_list>
#include <string>
#include <vector>

using namespace AstraSim;
using namespace AstraSim::ExecutionDriven;
using namespace Analytical;
using namespace AstraSimAnalytical;
using namespace AstraSimAnalyticalCongestionAware;
using namespace NetworkAnalytical;
using namespace NetworkAnalyticalCongestionAware;

namespace {

// ---- fixture config: remote port 100 B/ns + 50 ns so that
// runtime = 50 + bytes/100 (1000/2000/3000/4000 B -> 60/70/80/90 ns) ----
constexpr int kRanks = 4;
constexpr double kRemoteBw = 100.0;
constexpr uint64_t kRemoteLatency = 50;

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
    // post-commit drain): the chained PER_NPU MEM_LOAD must re-scan when
    // its predecessor released it -- this is what serializes node31's
    // port0 transaction right after node30's completion at t=60.
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
    // Exactly-once node completion. Explicit (NOT assert): the Release
    // build defines NDEBUG, which would silently zero-init the tick.
    if (hits != 1) {
        std::fprintf(stderr,
                     "[RemoteFifoLedgerTest] node (rank=%d, id=%llu) has "
                     "%llu terminal records (expected exactly 1)\n",
                     rank, static_cast<unsigned long long>(node_id),
                     static_cast<unsigned long long>(hits));
        std::exit(1);
    }
    return tick;
}

void write_text(const std::string& path, const std::string& content) {
    std::ofstream out(path);
    assert(out.is_open());
    out << content;
}

// hbm-bandwidth-contention: 0 -- no HBM join: a MEM node's terminal IS the
// port-transaction completion tick. No COMP nodes -> roofline stays unused.
const char* kSystemJson = R"({
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
  "roofline-enabled": 0,
  "replay-only": 0,
  "track-local-mem": 0,
  "trace-enabled": 0,
  "hbm-bandwidth-contention": 0,
  "peak-perf": 261.12,
  "local-mem-bw": 1640.0,
  "local-mem-latency": 100,
  "remote-mem-bw": 100.0,
  "remote-mem-latency": 50
}
)";

const char* kNetworkYaml = R"(topology: [ Line, Line ]
npus_count: [ 4, 1 ]
bandwidth: [ 400.0, 400.0 ]
latency: [ 5, 5 ]
)";

// Out-of-order sparse npu-ids: the PORT is the array INDEX (rank3 -> 0,
// rank1 -> 1) -- the key-label defect the old sys_id accounting had.
const char* kRemotePerNpu = R"({
  "memory-type": "PER_NPU_MEMORY_EXPANSION",
  "remote-mem-bw": 100.0,
  "remote-mem-latency": 50,
  "npu-ids": [3, 1]
}
)";

const char* kRemotePerNode = R"({
  "memory-type": "PER_NODE_MEMORY_EXPANSION",
  "remote-mem-bw": 100.0,
  "remote-mem-latency": 50,
  "num-nodes": 2,
  "num-npus-per-node": 2
}
)";

const char* kRemotePool = R"({
  "memory-type": "MEMORY_POOL",
  "remote-mem-bw": 100.0,
  "remote-mem-latency": 50
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

// One MEM node spec: (bytes, is_load). Node id = rank*10 + chain index.
struct MemSpec {
    uint64_t bytes;
    bool is_load;
};

// One full online simulation over the fixture configs. mems_per_rank[rank]
// lists the rank's MEM nodes in chain order (a same-rank chain of Data
// dependencies serializes them through the in-rank MEM slot). The
// after_issue callback runs once at t=0 -- right after the initial issue
// pass and before the event loop -- so the fixture can snapshot the real
// port state at the congestion peak. Fills `records` with every terminal.
void run_arch(const std::string& config_dir,
              const std::string& remote_memory_file,
              const std::vector<std::vector<MemSpec>>& mems_per_rank,
              const std::function<void()>& after_issue,
              std::vector<TerminalRecord>& records) {
    const std::string system_configuration = config_dir + "/system.json";
    const std::string network_configuration = config_dir + "/network.yml";
    const std::string remote_memory_configuration =
        config_dir + "/" + remote_memory_file;
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

    // ---- inject the MEM nodes (per-rank NodeStore; chain within a rank) --
    // Node ids start at 100: NodeStore::add_node treats global_id == 0 as
    // "assign a fresh id", so 0 must never be used as a fixture id.
    for (int rank = 0; rank < kRanks; ++rank) {
        auto& store = graph_sources[rank]->store();
        const std::vector<MemSpec>& mems = mems_per_rank[rank];
        for (size_t i = 0; i < mems.size(); ++i) {
            const uint64_t id = 100 + static_cast<uint64_t>(rank) * 10 + i;
            const MemSpec& spec = mems[i];
            OnlineNode node =
                spec.is_load
                    ? make_node(id, rank, NodeKind::MemLoad, 2, "mem_load")
                    : make_node(id, rank, NodeKind::MemStore, 3, "mem_store");
            // sh_3.0 variant: issue_remote_mem reads MemAttrs::tensor_size
            // (the ET adapter copies compute.tensor_size into mem during
            // parsing); this fixture bypasses the adapter, so it fills the
            // mem member directly.  sh_1.0 sets compute.tensor_size here.
            node.mem.tensor_size = spec.bytes;
            store.add_node(node);
            if (i > 0) {
                // Same-rank chain (Data): the in-rank MEM slot serializes
                // same-rank MEM nodes; the chain makes the order explicit.
                store.add_dependency(id - 1, id, DepKind::Data);
            }
        }
    }

    // ---- run: issue pass in rank order (the port queue order for the
    // shared-port architectures IS this loop order), t=0 snapshot, loop. --
    for (auto* workload : workloads) {
        workload->issue_dep_free_nodes();
    }
    fluid_scheduler->flush_pending_starts();
    fluid_scheduler->mark_event_loop_started();
    if (after_issue) {
        after_issue();  // t=0: all t=0 issues accounted, nothing completed
    }
    size_t guard = 0;
    while (!event_queue->finished()) {
        event_queue->proceed();
        if (++guard > 1000000) {
            std::fprintf(stderr,
                         "[RemoteFifoLedgerTest] event loop did not drain\n");
            std::exit(1);
        }
    }
}

void expect_true(bool cond, const char* what) {
    std::printf("%-56s -> %s\n", what, cond ? "PASS" : "FAIL");
    if (!cond) {
        std::exit(1);
    }
}

void expect_eq_u64(uint64_t actual, uint64_t expected, const char* what) {
    const bool ok = actual == expected;
    std::printf("%-56s expected=%llu actual=%llu -> %s\n", what,
                static_cast<unsigned long long>(expected),
                static_cast<unsigned long long>(actual),
                ok ? "PASS" : "FAIL");
    if (!ok) {
        std::exit(1);
    }
}

bool same_ports(const std::vector<std::size_t>& actual,
                std::initializer_list<std::size_t> expected) {
    return std::vector<std::size_t>(expected) == actual;
}

bool same_ranks(const std::vector<int>& actual,
                std::initializer_list<int> expected) {
    return std::vector<int>(expected) == actual;
}

bool contains(const std::string& haystack, const std::string& needle) {
    return haystack.find(needle) != std::string::npos;
}

// Per-port counter snapshot helper (in_flight/active/pending view).
struct PortView {
    uint64_t issued_count;
    uint64_t issued_bytes;
    uint64_t completed_count;
    uint64_t completed_bytes;
    uint64_t in_flight;
    uint64_t active;
    uint64_t pending;
    uint64_t in_flight_bytes;
};
PortView view(const RemoteFifoLedger& fifo, std::size_t port_index) {
    const auto* p = fifo.port(port_index);
    assert(p != nullptr);
    PortView v{};
    v.issued_count = p->issued_count;
    v.issued_bytes = p->issued_bytes;
    v.completed_count = p->completed_count;
    v.completed_bytes = p->completed_bytes;
    v.in_flight = v.issued_count - v.completed_count;
    v.in_flight_bytes = v.issued_bytes - v.completed_bytes;
    v.active = v.in_flight > 0 ? 1u : 0u;
    v.pending = v.in_flight - v.active;
    return v;
}

// ---- shared conservation / drained / label block per architecture ----
void check_conservation(const RemoteFifoLedger& fifo, uint64_t issued,
                        uint64_t bytes, const char* arch) {
    expect_eq_u64(fifo.total_issued_count(), issued, "total issued count");
    expect_eq_u64(fifo.total_completed_count(), issued,
                  "total completed count");
    expect_eq_u64(fifo.total_issued_bytes(), bytes, "total issued bytes");
    expect_eq_u64(fifo.total_completed_bytes(), bytes,
                  "total completed bytes");
    expect_true(fifo.drained(), "all real ports drained");
    expect_true(fifo.architecture() == arch, "architecture label");
}

}  // namespace

int main() {
    setvbuf(stdout, nullptr, _IONBF, 0);
    // ---- write the fixture configs to a temp dir ----
    const std::string tmp_template = "/tmp/remote_fifo_ledger_XXXXXX";
    std::vector<char> buffer(tmp_template.begin(), tmp_template.end());
    buffer.push_back('\0');
    const char* made = mkdtemp(buffer.data());
    assert(made != nullptr);
    const std::string config_dir(made);
    write_text(config_dir + "/system.json", kSystemJson);
    write_text(config_dir + "/network.yml", kNetworkYaml);
    write_text(config_dir + "/remote_per_npu.json", kRemotePerNpu);
    write_text(config_dir + "/remote_per_node.json", kRemotePerNode);
    write_text(config_dir + "/remote_pool.json", kRemotePool);
    write_text(config_dir + "/comm_group.json", "{}");

    auto& fifo = RemoteFifoLedger::instance();
    fifo.set_enabled(true);  // fail-closed default off; fixture opens it

    // ================= A. PER_NPU (out-of-order npu-ids [3, 1]) ==========
    // rank3 -> port0 (ARRAY INDEX), rank1 -> port1; rank0/2 issue no MEM.
    {
        std::printf("[A] PER_NPU npu-ids [3,1]: rank3->port0, rank1->port1\n");
        fifo.reset();
        fifo.set_architecture("PER_NPU_MEMORY_EXPANSION");

        // t=0 snapshot state, captured inside the run at the peak moment.
        PortView port0_at_zero{};
        PortView port1_at_zero{};
        std::string sidecar_zero;
        const auto snapshot = [&]() {
            port0_at_zero = view(fifo, 0);
            port1_at_zero = view(fifo, 1);
            sidecar_zero = fifo.sidecar_row(0, 0);
        };

        std::vector<TerminalRecord> records;
        // rank1: one 3000 B store; rank3: chained 1000 B then 2000 B loads.
        const std::vector<std::vector<MemSpec>> mems = {
            {},                          // rank0: none
            {{3000, false}},             // rank1 -> port1
            {},                          // rank2: none
            {{1000, true}, {2000, true}} // rank3 -> port0 (chained)
        };
        run_arch(config_dir, "remote_per_npu.json", mems, snapshot, records);

        // Real port keys: {0,1}. The old sys_id-keyed ledger printed {1,3}
        // (rank3's account mislabeled as "port 3") -- the label defect.
        expect_true(same_ports(fifo.active_ports(), {0, 1}),
                    "A: active_ports == {0,1} (old keys were {1,3})");

        // t=0: one in-flight transaction per port, nothing completed.
        expect_eq_u64(port0_at_zero.in_flight, 1, "A: t=0 port0 in_flight");
        expect_eq_u64(port0_at_zero.active, 1, "A: t=0 port0 active");
        expect_eq_u64(port0_at_zero.pending, 0, "A: t=0 port0 pending");
        expect_eq_u64(port0_at_zero.issued_bytes, 1000,
                      "A: t=0 port0 issued_bytes");
        expect_eq_u64(port1_at_zero.in_flight, 1, "A: t=0 port1 in_flight");
        expect_eq_u64(port1_at_zero.in_flight_bytes, 3000,
                      "A: t=0 port1 in_flight_bytes");

        // Final per-port counters: port0 carries BOTH of rank3's requests
        // (3000 B total), port1 rank1's single one.
        const PortView p0 = view(fifo, 0);
        const PortView p1 = view(fifo, 1);
        expect_eq_u64(p0.issued_count, 2, "A: port0 issued_count");
        expect_eq_u64(p0.issued_bytes, 3000, "A: port0 issued_bytes");
        expect_eq_u64(p0.completed_count, 2, "A: port0 completed_count");
        expect_eq_u64(p0.completed_bytes, 3000, "A: port0 completed_bytes");
        expect_eq_u64(fifo.port(0)->peak_in_flight_count, 1,
                      "A: port0 peak_in_flight_count");
        expect_eq_u64(fifo.port(0)->peak_in_flight_bytes, 2000,
                      "A: port0 peak_in_flight_bytes");
        expect_eq_u64(p1.issued_count, 1, "A: port1 issued_count");
        expect_eq_u64(p1.completed_bytes, 3000, "A: port1 completed_bytes");
        expect_eq_u64(fifo.port(1)->peak_in_flight_count, 1,
                      "A: port1 peak_in_flight_count");

        // Mapping evidence: port_ranks / by_rank (RF3 view).
        expect_true(same_ranks(fifo.port_ranks(0), {3}),
                    "A: port_ranks(0) == {3}");
        expect_true(same_ranks(fifo.port_ranks(1), {1}),
                    "A: port_ranks(1) == {1}");
        expect_true(same_ranks(fifo.attributed_ranks(), {1, 3}),
                    "A: attributed_ranks == {1,3}");
        expect_true(fifo.rank_attribution(3) != nullptr &&
                    fifo.rank_attribution(3)->port == 0 &&
                    fifo.rank_attribution(3)->issued_count == 2 &&
                    fifo.rank_attribution(3)->issued_bytes == 3000,
                    "A: by_rank rank3 -> port0 (2 req, 3000 B)");
        expect_true(fifo.rank_attribution(1) != nullptr &&
                    fifo.rank_attribution(1)->port == 1 &&
                    fifo.rank_attribution(1)->issued_bytes == 3000,
                    "A: by_rank rank1 -> port1");

        // Serial service ground truth: port0 [0,60) then [60,130);
        // port1 [0,80). Terminals == the port completion ticks.
        expect_eq_u64(terminal_tick(records, 3, 130), 60,
                      "A: rank3 mem#1 terminal == 60");
        expect_eq_u64(terminal_tick(records, 3, 131), 130,
                      "A: rank3 mem#2 terminal == 130");
        expect_eq_u64(terminal_tick(records, 1, 110), 80,
                      "A: rank1 store terminal == 80");
        expect_eq_u64(records.size(), 3, "A: exactly 3 MEM terminals");

        check_conservation(fifo, 3, 6000, "PER_NPU_MEMORY_EXPANSION");

        // Sidecar row (shared formatter): architecture label + real port
        // keys + source ranks + by_rank view.
        expect_true(contains(sidecar_zero,
                             "\"memory_architecture\": "
                             "\"PER_NPU_MEMORY_EXPANSION\""),
                    "A: sidecar has memory_architecture");
        expect_true(contains(sidecar_zero, "\"port\": 0"),
                    "A: sidecar uses real port keys");
        expect_true(contains(sidecar_zero, "\"ranks\": [3]"),
                    "A: sidecar port0 source ranks");
        expect_true(contains(sidecar_zero,
                             "\"by_rank\": [{\"rank\": 1, \"port\": 1, "
                             "\"issued_count\": 1, \"issued_bytes\": 3000}, "
                             "{\"rank\": 3, \"port\": 0, \"issued_count\": "
                             "1, \"issued_bytes\": 1000}]"),
                    "A: sidecar by_rank view");
    }

    // ================= B. PER_NODE (2 nodes x 2 ranks) ===================
    // rank0,1 -> port0; rank2,3 -> port1; all four issue at t=0.
    {
        std::printf("[B] PER_NODE 2x2: rank0,1->port0; rank2,3->port1\n");
        fifo.reset();
        fifo.set_architecture("PER_NODE_MEMORY_EXPANSION");

        PortView port0_at_zero{};
        PortView port1_at_zero{};
        const auto snapshot = [&]() {
            port0_at_zero = view(fifo, 0);
            port1_at_zero = view(fifo, 1);
        };

        std::vector<TerminalRecord> records;
        const std::vector<std::vector<MemSpec>> mems = {
            {{1000, true}}, // rank0 -> port0 (first in queue)
            {{2000, true}}, // rank1 -> port0 (queued)
            {{3000, true}}, // rank2 -> port1 (first in queue)
            {{4000, true}}  // rank3 -> port1 (queued)
        };
        run_arch(config_dir, "remote_per_node.json", mems, snapshot, records);

        expect_true(same_ports(fifo.active_ports(), {0, 1}),
                    "B: active_ports == {0,1} (old keys were {0,1,2,3})");

        // t=0 REAL shared-port state: ONE server busy, ONE queued.
        expect_eq_u64(port0_at_zero.in_flight, 2, "B: t=0 port0 in_flight");
        expect_eq_u64(port0_at_zero.active, 1, "B: t=0 port0 active");
        expect_eq_u64(port0_at_zero.pending, 1, "B: t=0 port0 pending");
        expect_eq_u64(port0_at_zero.in_flight_bytes, 3000,
                      "B: t=0 port0 in_flight_bytes");
        expect_eq_u64(port1_at_zero.in_flight, 2, "B: t=0 port1 in_flight");
        expect_eq_u64(port1_at_zero.pending, 1, "B: t=0 port1 pending");
        // Old per-rank accounting evidence: two keys each active=1/pending=0
        // (sum 2 != real 1; peak_pending 0 != real 1) -- fixed constants of
        // the OLD formula on this same scenario, no old code runs here.
        expect_true(1 + 1 != port0_at_zero.active &&
                    0 + 0 != port0_at_zero.pending,
                    "B: old rank-view contradicts real port state");

        // Peaks recorded on the real shared port.
        expect_eq_u64(fifo.port(0)->peak_in_flight_count, 2,
                      "B: port0 peak_in_flight_count");
        expect_eq_u64(fifo.port(0)->peak_in_flight_bytes, 3000,
                      "B: port0 peak_in_flight_bytes");
        expect_eq_u64(fifo.port(1)->peak_in_flight_count, 2,
                      "B: port1 peak_in_flight_count");
        expect_eq_u64(fifo.port(1)->peak_in_flight_bytes, 7000,
                      "B: port1 peak_in_flight_bytes");

        // Final counters per shared port.
        const PortView p0 = view(fifo, 0);
        const PortView p1 = view(fifo, 1);
        expect_eq_u64(p0.issued_count, 2, "B: port0 issued_count");
        expect_eq_u64(p0.issued_bytes, 3000, "B: port0 issued_bytes");
        expect_eq_u64(p0.completed_bytes, 3000, "B: port0 completed_bytes");
        expect_eq_u64(p1.issued_count, 2, "B: port1 issued_count");
        expect_eq_u64(p1.issued_bytes, 7000, "B: port1 issued_bytes");
        expect_eq_u64(p1.completed_bytes, 7000, "B: port1 completed_bytes");

        // Mapping evidence.
        expect_true(same_ranks(fifo.port_ranks(0), {0, 1}),
                    "B: port_ranks(0) == {0,1}");
        expect_true(same_ranks(fifo.port_ranks(1), {2, 3}),
                    "B: port_ranks(1) == {2,3}");
        expect_true(fifo.rank_attribution(0) != nullptr &&
                    fifo.rank_attribution(0)->port == 0 &&
                    fifo.rank_attribution(0)->issued_bytes == 1000,
                    "B: by_rank rank0 -> port0");
        expect_true(fifo.rank_attribution(3) != nullptr &&
                    fifo.rank_attribution(3)->port == 1 &&
                    fifo.rank_attribution(3)->issued_bytes == 4000,
                    "B: by_rank rank3 -> port1");

        // Serial service ground truth: port0 60 (rank0 1000B) then 130
        // (rank1 2000B); port1 80 (rank2) then 170 (rank3). The second
        // port0 completion is rank1's 2000 B -- completed_bytes 3000 ==
        // 1000 + 2000 proves the payload carried THIS transaction's bytes
        // (the dequeued next-request tensor_size must never leak in here).
        expect_eq_u64(terminal_tick(records, 0, 100), 60,
                      "B: rank0 terminal == 60 (1000B)");
        expect_eq_u64(terminal_tick(records, 1, 110), 130,
                      "B: rank1 terminal == 130 (2000B, queued)");
        expect_eq_u64(terminal_tick(records, 2, 120), 80,
                      "B: rank2 terminal == 80 (3000B)");
        expect_eq_u64(terminal_tick(records, 3, 130), 170,
                      "B: rank3 terminal == 170 (4000B, queued)");
        expect_eq_u64(records.size(), 4, "B: exactly 4 MEM terminals");

        check_conservation(fifo, 4, 10000, "PER_NODE_MEMORY_EXPANSION");
    }

    // ================= C. MEMORY_POOL (all ranks -> port 0) ==============
    {
        std::printf("[C] MEMORY_POOL: rank0..3 -> port0\n");
        fifo.reset();
        fifo.set_architecture("MEMORY_POOL");

        PortView port0_at_zero{};
        std::string sidecar_zero;
        const auto snapshot = [&]() {
            port0_at_zero = view(fifo, 0);
            sidecar_zero = fifo.sidecar_row(0, 0);
        };

        std::vector<TerminalRecord> records;
        const std::vector<std::vector<MemSpec>> mems = {
            {{1000, false}}, // rank0 -> port0 (queue head)
            {{2000, false}}, // rank1 -> port0
            {{3000, false}}, // rank2 -> port0
            {{4000, false}}  // rank3 -> port0 (queue tail)
        };
        run_arch(config_dir, "remote_pool.json", mems, snapshot, records);

        // THE single shared real port (old accounting: 4 rank keys).
        expect_true(same_ports(fifo.active_ports(), {0}),
                    "C: active_ports == {0} (old keys were {0,1,2,3})");

        // t=0: one server busy, THREE queued.
        expect_eq_u64(port0_at_zero.in_flight, 4, "C: t=0 port0 in_flight");
        expect_eq_u64(port0_at_zero.active, 1, "C: t=0 port0 active");
        expect_eq_u64(port0_at_zero.pending, 3, "C: t=0 port0 pending");
        expect_eq_u64(port0_at_zero.in_flight_bytes, 10000,
                      "C: t=0 port0 in_flight_bytes");
        // Old per-rank accounting evidence: four keys each active=1
        // (sum 4 != real 1) and pending 0 (!= real 3).
        expect_true(1 + 1 + 1 + 1 != port0_at_zero.active &&
                    0 + 0 + 0 + 0 != port0_at_zero.pending,
                    "C: old rank-view contradicts real port state");

        expect_eq_u64(fifo.port(0)->peak_in_flight_count, 4,
                      "C: port0 peak_in_flight_count");
        expect_eq_u64(fifo.port(0)->peak_in_flight_bytes, 10000,
                      "C: port0 peak_in_flight_bytes");
        expect_eq_u64(fifo.port(0)->issued_count, 4, "C: port0 issued_count");
        expect_eq_u64(fifo.port(0)->issued_bytes, 10000,
                      "C: port0 issued_bytes");
        expect_eq_u64(fifo.port(0)->completed_bytes, 10000,
                      "C: port0 completed_bytes");

        expect_true(same_ranks(fifo.port_ranks(0), {0, 1, 2, 3}),
                    "C: port_ranks(0) == {0,1,2,3}");
        bool four_attributions = true;
        const uint64_t expected_bytes[4] = {1000, 2000, 3000, 4000};
        for (int rank = 0; rank < 4; ++rank) {
            const auto* attribution = fifo.rank_attribution(rank);
            if (attribution == nullptr || attribution->port != 0 ||
                attribution->issued_count != 1 ||
                attribution->issued_bytes != expected_bytes[rank]) {
                four_attributions = false;
            }
        }
        expect_true(four_attributions,
                    "C: by_rank four rows, each 1 req on port0");

        // Serial service ground truth: 60, 130, 210, 300
        // (60 + 70 + 80 + 90 across the single server).
        expect_eq_u64(terminal_tick(records, 0, 100), 60, "C: rank0 == 60");
        expect_eq_u64(terminal_tick(records, 1, 110), 130, "C: rank1 == 130");
        expect_eq_u64(terminal_tick(records, 2, 120), 210, "C: rank2 == 210");
        expect_eq_u64(terminal_tick(records, 3, 130), 300, "C: rank3 == 300");
        expect_eq_u64(records.size(), 4, "C: exactly 4 MEM terminals");

        check_conservation(fifo, 4, 10000, "MEMORY_POOL");

        expect_true(contains(sidecar_zero,
                             "\"memory_architecture\": \"MEMORY_POOL\""),
                    "C: sidecar architecture label");
        expect_true(contains(sidecar_zero, "\"pending\": 3"),
                    "C: sidecar t=0 real pending");
        expect_true(contains(sidecar_zero, "\"ranks\": [0, 1, 2, 3]"),
                    "C: sidecar port0 source ranks");
    }

    // ================= Negative gate: set_enabled(false) =================
    {
        fifo.set_enabled(false);
        const uint64_t issued_before = fifo.port(0)->issued_count;
        const uint64_t completed_before = fifo.port(0)->completed_count;
        const std::size_t ranks_before = fifo.port_ranks(0).size();
        fifo.record_issue(0, 9, 12345);        // must be a no-op
        fifo.record_completion(0, 12345);      // must be a no-op
        expect_true(fifo.port(0)->issued_count == issued_before &&
                    fifo.port(0)->completed_count == completed_before,
                    "gate off: record_* leave counters untouched");
        expect_true(fifo.port_ranks(0).size() == ranks_before &&
                    fifo.rank_attribution(9) == nullptr,
                    "gate off: no new keys / attribution rows");
        fifo.set_enabled(true);
    }

    std::printf("ALL PASS\n");
    // Best-effort temp-config cleanup (keep on failure for debugging).
    std::error_code ec;
    std::filesystem::remove_all(config_dir, ec);
    return 0;
}
