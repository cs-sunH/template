/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

/*

local_hbm_model_test.cc -- sh_2.0 N-way HBM contention fixture.

Standalone test (no baseline artifacts touched). Boots 6 ranks of the real
online-mode stack (Sys + Workload + LocalHbmBandwidthModel + real fluid
network + real AnalyticalRemoteMemory FIFO) from self-written tiny configs,
then issues hand-built nodes through Workload::issue and asserts the N-way
equal-split semantics numerically:

  Rank 0  three-way equal split (COMP + KV restore + comm send endpoint):
          each job drains its bytes at exactly full_rate/3, all served-byte
          counters ~= issued bytes, peak concurrency 3.
  Rank 1  two-way 50/50 then full-rate reallocation after the restore
          completes (the legacy two-user case as the N=2 instance of the
          N-way model).
  Rank 2  pool endpoint join, HBM side first: the MEM node (hbm-access-mode
          2) terminates at max(FIFO, HBM job) -- the FIFO time, proving the
          join waited for the slower side and completed exactly once.
  Rank 4  pool endpoint join, FIFO side first (restore+COMP+MEM three-way
          dilution): termination at the HBM-side time; also the NoC transit
          rank of rank5->rank3 traffic -- transit charges nothing (comm
          served counters stay 0).
  Rank 5  hbm-charge=false send: no COMM_READ job is ever created (all model
          counters 0), the node still completes via the network alone.
  Rank 3  helper receiver: charged recv creates exactly one COMM_WRITE job,
          uncharged recv creates none.

Config used by the math (written to a temp dir at startup):
  local-mem-bw = 1000 GB/s (1000 B/ns full rate), local-mem-latency 100 ns,
  peak-perf 1000 TFLOPS, remote-mem-bw 500 GB/s, remote-mem-latency 500 ns.
Tick assertions tolerate +-3 ns of ceil()/residue-clamp rounding in the
transition scheduler; byte assertions tolerate 0.01 bytes.

Build: cmake --build build/astra_analytical/build_congestion_aware -j
  --target AstraSim_Analytical_Congestion_Aware_LocalHbmModelTest
Run:
  build/astra_analytical/build_congestion_aware/bin/\
      AstraSim_Analytical_Congestion_Aware_LocalHbmModelTest
Exit code 0 on ALL PASS.
*/

#include "astra-sim/common/Logging.hh"
#include "astra-sim/workload/LocalHbmBandwidthModel.hh"
#include "astra-sim/workload/Workload.hh"
#include "astra-sim/workload/execution_driven/CompletionObserver.hh"
#include "astra-sim/workload/execution_driven/NodeStore.hh"
#include "common/CmdLineParser.hh"
#include "congestion_aware/CongestionAwareNetworkApi.hh"
#include <astra-network-analytical/common/EventQueue.h>
#include <astra-network-analytical/common/NetworkParser.h>
#include <astra-network-analytical/congestion_aware/Helper.h>
#include <remote_memory_backend/analytical/AnalyticalRemoteMemory.hh>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <unistd.h>
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
        std::fprintf(stderr, "[local_hbm_model_test] FAIL: %s\n",
                     what.c_str());
        g_ok = false;
    }
}

void expect_near(double value, double expected, double tol,
                 const std::string& what) {
    const double diff = value > expected ? value - expected
                                         : expected - value;
    if (!(diff <= tol)) {
        std::fprintf(stderr,
                     "[local_hbm_model_test] FAIL: %s: got %.6f expected "
                     "%.6f (tol %.6f)\n",
                     what.c_str(), value, expected, tol);
        g_ok = false;
    }
}

struct TerminalCounters {
    // (rank << 32) | node_id -> (count, tick of last terminal)
    std::map<uint64_t, std::pair<uint64_t, uint64_t>> records;
};

void terminal_hook(void* ctx, int rank, uint64_t node_id,
                   const char*, const char*, uint64_t, uint64_t tick,
                   int status) {
    auto* counters = static_cast<TerminalCounters*>(ctx);
    const uint64_t key =
        (static_cast<uint64_t>(rank) << 32) | node_id;
    auto& entry = counters->records[key];
    entry.first += 1;
    entry.second = tick;
    if (status !=
        static_cast<int>(ExecutionDriven::NodeTerminalStatus::Success)) {
        std::fprintf(stderr,
                     "[local_hbm_model_test] FAIL: unexpected terminal "
                     "status %d for rank %d node %llu\n",
                     status, rank, static_cast<unsigned long long>(node_id));
        g_ok = false;
    }
}

ExecutionDriven::OnlineNode make_node(int rank,
                                      ExecutionDriven::NodeKind kind,
                                      uint64_t node_type,
                                      const std::string& name) {
    ExecutionDriven::OnlineNode node;
    node.rank = rank;
    node.kind = kind;
    node.node_type = node_type;
    node.name = name;
    return node;
}

}  // namespace

int main() {
    // ---- self-written tiny configs in a temp dir (no generated inputs) ----
    char dir_template[] = "/tmp/hbm_model_test_XXXXXX";
    const char* dir_name = mkdtemp(dir_template);
    if (dir_name == nullptr) {
        std::perror("mkdtemp");
        return 1;
    }
    const std::string dir(dir_name);
    const std::string system_path = dir + "/system.json";
    const std::string remote_path = dir + "/remote_memory.json";
    const std::string network_path = dir + "/network.yml";

    {
        FILE* f = std::fopen(system_path.c_str(), "w");
        // Official-template fixture shape (阶段5夹具避雷): scheduling-policy /
        // preferred-dataset-splits / collective-optimization present, all four
        // *-implementation keys = ["ring","ring"] (2 topology dims). With any
        // *-implementation key missing, Sys's CONSTRUCTOR builds its per-ComType
        // logical topologies (Sys.cc:266-272) from an empty native-impl list and
        // GeneralComplexTopology throws "requires at least one collective
        // implementation" before this test's own assertions can run.
        std::fputs("{\n"
                   "  \"scheduling-policy\": \"LIFO\",\n"
                   "  \"preferred-dataset-splits\": 6,\n"
                   "  \"collective-optimization\": \"localBWAware\",\n"
                   "  \"all-reduce-implementation\": [\"ring\", \"ring\"],\n"
                   "  \"all-gather-implementation\": [\"ring\", \"ring\"],\n"
                   "  \"reduce-scatter-implementation\": [\"ring\", \"ring\"],\n"
                   "  \"all-to-all-implementation\": [\"ring\", \"ring\"],\n"
                   "  \"roofline-enabled\": 1,\n"
                   "  \"peak-perf\": 1000,\n"
                   "  \"local-mem-bw\": 1000,\n"
                   "  \"local-mem-latency\": 100,\n"
                   "  \"remote-mem-bw\": 500,\n"
                   "  \"remote-mem-latency\": 500,\n"
                   "  \"hbm-bandwidth-contention\": 1,\n"
                   "  \"hbm-kv-restore-bandwidth-sharing\": 1\n"
                   "}\n",
                   f);
        std::fclose(f);
    }
    {
        FILE* f = std::fopen(remote_path.c_str(), "w");
        // Per-rank ports: the FIFO of one rank never queues behind another
        // rank's transaction (keeps the join-order math per rank exact).
        std::fputs("{\n"
                   "  \"memory-type\": \"PER_NPU_MEMORY_EXPANSION\",\n"
                   "  \"npu-ids\": [0, 1, 2, 3, 4, 5],\n"
                   "  \"remote-mem-bw\": 500,\n"
                   "  \"remote-mem-latency\": 500\n"
                   "}\n",
                   f);
        std::fclose(f);
    }
    {
        FILE* f = std::fopen(network_path.c_str(), "w");
        std::fputs("topology: [ Mesh, Mesh ]\n"
                   "npus_count: [ 3, 2 ]\n"
                   "bandwidth: [ 4050, 4050 ]\n"
                   "latency: [ 25, 25 ]\n",
                   f);
        std::fclose(f);
    }

    AstraSim::LoggerFactory::init("empty", "off");

    static TerminalCounters counters;
    ExecutionDriven::CompletionObserver::instance().set_hook(
        terminal_hook, &counters);

    const auto event_queue = std::make_shared<EventQueue>();
    const auto network_parser = NetworkParser(network_path);
    const auto topology = construct_topology(network_parser);
    CongestionAwareNetworkApi::set_event_queue(event_queue);
    CongestionAwareNetworkApi::set_topology(topology);
    const auto fluid_scheduler = std::make_shared<FluidScheduler>(
        event_queue, topology->get_directed_links(),
        network_parser.get_fluid_max_active_flows(),
        network_parser.get_fluid_max_route_memberships(),
        network_parser.get_progress_report_event_interval());
    CongestionAwareNetworkApi::set_fluid_scheduler(fluid_scheduler);

    const auto memory_api =
        std::make_unique<AnalyticalRemoteMemory>(remote_path);
    auto network_apis =
        std::vector<std::unique_ptr<CongestionAwareNetworkApi>>();
    auto systems = std::vector<Sys*>();
    auto sources =
        std::vector<std::shared_ptr<ExecutionDriven::NodeStoreGraphSource>>();

    const int npus_count = static_cast<int>(topology->get_npus_count());
    expect(npus_count == 6, "topology has 6 ranks");
    const std::vector<int> queues_per_dim{1, 1};
    for (int i = 0; i < npus_count; i++) {
        auto source =
            std::make_shared<ExecutionDriven::NodeStoreGraphSource>();
        sources.push_back(source);
        auto network_api = std::make_unique<CongestionAwareNetworkApi>(i);
        systems.push_back(new Sys(
            i, dir + "/none.et", "empty", system_path, memory_api.get(),
            network_api.get(), topology->get_npus_count_per_dim(),
            queues_per_dim, 1.0, 1.0, false,
            ExecutionDriven::ExecutionMode::Online, source));
        network_apis.push_back(std::move(network_api));
    }

    // Flag plumbing sanity on every rank.
    for (int i = 0; i < npus_count; i++) {
        expect(systems[i]->hbm_bandwidth_contention,
               "hbm-bandwidth-contention parsed true");
        expect(systems[i]->workload->local_hbm_bandwidth_model != nullptr,
               "N-way model assembled");
    }

    // ---- helper: add + issue one node on a rank ----
    auto issue_node = [](Sys* sys,
                         ExecutionDriven::NodeStoreGraphSource* source,
                         ExecutionDriven::OnlineNode node) {
        source->store().add_node(std::move(node));
        for (const auto& nv : source->dep_free_nodes()) {
            if (sys->workload->hw_resource->is_available(nv)) {
                sys->workload->issue(nv);
            }
        }
    };

    constexpr uint64_t kB = 1000000;  // 1 MB; 1000 B/ns -> 1000 ns exclusive
    constexpr uint64_t kOps = 1000000;  // 1e6 ops at 1e6 ops/ns -> 1 ns

    // Rank 0: COMP + restore + comm_send -- strict three-way 1/3 split.
    {
        auto comp = make_node(0, ExecutionDriven::NodeKind::Compute, 4,
                              "s1_comp");
        comp.compute.num_ops = kOps;
        comp.compute.tensor_size = kB;
        issue_node(systems[0], sources[0].get(), comp);

        auto restore = make_node(0, ExecutionDriven::NodeKind::MemLoad, 2,
                                 "s1_restore");
        restore.compute.tensor_size = kB;
        restore.is_local_hbm_kv_restore = true;
        issue_node(systems[0], sources[0].get(), restore);

        auto send = make_node(0, ExecutionDriven::NodeKind::CommSend, 5,
                              "s1_send");
        send.comm.src = 0;
        send.comm.dst = 3;
        send.comm.bytes = kB;
        send.comm.tag = 10;
        issue_node(systems[0], sources[0].get(), send);
    }

    // Rank 1: COMP (2 MB) + restore (1 MB) -- 50/50 then full-rate.
    {
        auto comp = make_node(1, ExecutionDriven::NodeKind::Compute, 4,
                              "s2_comp");
        comp.compute.num_ops = kOps;
        comp.compute.tensor_size = 2 * kB;
        issue_node(systems[1], sources[1].get(), comp);

        auto restore = make_node(1, ExecutionDriven::NodeKind::MemLoad, 2,
                                 "s2_restore");
        restore.compute.tensor_size = kB;
        restore.is_local_hbm_kv_restore = true;
        issue_node(systems[1], sources[1].get(), restore);
    }

    // Rank 2: COMP (4 MB) + pool write (1 MB, hbm-access-mode 2) -- the HBM
    // side of the join (t ~= 2100) finishes BEFORE the FIFO (t = 2500), so
    // the MEM node must terminate at the FIFO time (join, not first-wins).
    {
        auto comp = make_node(2, ExecutionDriven::NodeKind::Compute, 4,
                              "s3a_comp");
        comp.compute.num_ops = kOps;
        comp.compute.tensor_size = 4 * kB;
        issue_node(systems[2], sources[2].get(), comp);

        auto pool = make_node(2, ExecutionDriven::NodeKind::MemLoad, 2,
                              "s3a_pool_write");
        pool.compute.tensor_size = kB;
        pool.hbm_access_mode = 2;
        issue_node(systems[2], sources[2].get(), pool);
    }

    // Rank 4: restore (1 MB) + COMP (6 MB) + pool write (3 MB) -- three-way
    // dilution makes the HBM side (t ~= 7100) finish AFTER the FIFO
    // (t = 6500): FIFO-first join order. Rank 4 is also the NoC transit
    // rank of the rank5 -> rank3 flow: transit must charge nothing.
    {
        auto restore = make_node(4, ExecutionDriven::NodeKind::MemLoad, 2,
                                 "s3b_restore");
        restore.compute.tensor_size = kB;
        restore.is_local_hbm_kv_restore = true;
        issue_node(systems[4], sources[4].get(), restore);

        auto comp = make_node(4, ExecutionDriven::NodeKind::Compute, 4,
                              "s3b_comp");
        comp.compute.num_ops = kOps;
        comp.compute.tensor_size = 6 * kB;
        issue_node(systems[4], sources[4].get(), comp);

        auto pool = make_node(4, ExecutionDriven::NodeKind::MemLoad, 2,
                              "s3b_pool_write");
        pool.compute.tensor_size = 3 * kB;
        pool.hbm_access_mode = 2;
        issue_node(systems[4], sources[4].get(), pool);
    }

    // Rank 5 -> rank 3: uncharged send (hbm-charge false).
    {
        auto send = make_node(5, ExecutionDriven::NodeKind::CommSend, 5,
                              "s4_send_uncharged");
        send.comm.src = 5;
        send.comm.dst = 3;
        send.comm.bytes = kB;
        send.comm.tag = 20;
        send.comm.hbm_charge = false;
        issue_node(systems[5], sources[5].get(), send);
    }

    // Rank 3: helper receiver -- charged recv (tag 10, one COMM_WRITE job)
    // and uncharged recv (tag 20, no job).
    {
        auto recv_charged = make_node(3, ExecutionDriven::NodeKind::CommRecv,
                                      6, "helper_recv_charged");
        recv_charged.comm.src = 0;
        recv_charged.comm.dst = 3;
        recv_charged.comm.bytes = kB;
        recv_charged.comm.tag = 10;
        issue_node(systems[3], sources[3].get(), recv_charged);

        auto recv_uncharged =
            make_node(3, ExecutionDriven::NodeKind::CommRecv, 6,
                      "helper_recv_uncharged");
        recv_uncharged.comm.src = 5;
        recv_uncharged.comm.dst = 3;
        recv_uncharged.comm.bytes = kB;
        recv_uncharged.comm.tag = 20;
        recv_uncharged.comm.hbm_charge = false;
        issue_node(systems[3], sources[3].get(), recv_uncharged);
    }

    // ---- run the real event loop ----
    fluid_scheduler->flush_pending_starts();
    fluid_scheduler->mark_event_loop_started();
    while (!event_queue->finished()) {
        event_queue->proceed();
    }

    const uint64_t end_tick = Sys::boostedTick();

    // ---- assertions ----
    auto terminal = [&](int rank, uint64_t node_id)
        -> std::pair<uint64_t, uint64_t> {
        const auto it = counters.records.find(
            (static_cast<uint64_t>(rank) << 32) | node_id);
        if (it == counters.records.end()) {
            return {0, 0};
        }
        return it->second;
    };

    // Every issued node terminated exactly once (join included).
    const std::vector<std::pair<int, uint64_t>> all_nodes = {
        {0, 1}, {0, 2}, {0, 3},
        {1, 1}, {1, 2},
        {2, 1}, {2, 2},
        {3, 1}, {3, 2},
        {4, 1}, {4, 2}, {4, 3},
        {5, 1},
    };
    for (const auto& [rank, node_id] : all_nodes) {
        const auto term = terminal(rank, node_id);
        expect(term.first == 1,
               "rank " + std::to_string(rank) + " node " +
                   std::to_string(node_id) + " terminal exactly once (got " +
                   std::to_string(term.first) + ")");
    }
    // Rank 0 issued exactly three nodes (ids 1..3).
    expect(terminal(0, 4).first == 0, "rank 0 has no fourth node");

    // Global quiescence: nothing in flight, no active HBM jobs.
    for (int i = 0; i < npus_count; i++) {
        auto* hw = systems[i]->workload->hw_resource;
        expect(hw->num_in_flight_cpu_ops == 0 &&
                   hw->num_in_flight_gpu_comp_ops == 0 &&
                   hw->num_in_flight_gpu_comm_ops == 0 &&
                   hw->num_in_flight_hbm_dma_ops == 0,
               "rank " + std::to_string(i) + " fully released");
        expect(!systems[i]->workload->local_hbm_bandwidth_model
                    ->has_active_jobs(),
               "rank " + std::to_string(i) + " HBM model drained");
    }

    // Rank 0 -- three-way equal split (full_rate/3 = 333.33 B/ns each):
    // each 1 MB job takes ~3000 ns after the shared 100 ns latency, so all
    // terminals sit at ~3101 (3100 + ceil rounding), and every category
    // served exactly its issued bytes.
    {
        const auto* model =
            systems[0]->workload->local_hbm_bandwidth_model.get();
        expect_near(model->compute_bytes_served(), 1e6, 0.01,
                    "rank0 compute served ~= 1MB at 1/3 rate");
        expect_near(model->restore_bytes_served(), 1e6, 0.01,
                    "rank0 restore served ~= 1MB at 1/3 rate");
        expect_near(model->comm_read_bytes_served(), 1e6, 0.01,
                    "rank0 comm_read served ~= 1MB at 1/3 rate");
        expect(model->comm_write_bytes_served() == 0.0 &&
                   model->pool_read_bytes_served() == 0.0 &&
                   model->pool_write_bytes_served() == 0.0,
               "rank0 no other categories");
        expect(model->peak_concurrent_jobs() == 3,
               "rank0 peak concurrency 3");
        expect(model->redistribution_events() == 2,
               "rank0 two redistributions (restore joins, comm joins)");
        expect_near(model->hbm_busy_ns(), 3000.0, 3.0,
                    "rank0 busy ~= 3000 ns (1MB at full/3)");
        expect_near(model->hbm_shared_ns(), 3000.0, 3.0,
                    "rank0 shared ~= 3000 ns");
        for (uint64_t id = 1; id <= 3; id++) {
            const uint64_t tick = terminal(0, id).second;
            expect(tick >= 3099 && tick <= 3105,
                   "rank0 node " + std::to_string(id) +
                       " completes at ~3101 (three-way split), got " +
                       std::to_string(tick));
        }
    }

    // Rank 1 -- two-way 50/50 (500 B/ns each), then the COMP returns to the
    // full 1000 B/ns when the restore completes: restore terminal 2100,
    // COMP terminal 3100.
    {
        const auto* model =
            systems[1]->workload->local_hbm_bandwidth_model.get();
        expect_near(model->compute_bytes_served(), 2e6, 0.01,
                    "rank1 compute served ~= 2MB");
        expect_near(model->restore_bytes_served(), 1e6, 0.01,
                    "rank1 restore served ~= 1MB");
        expect(model->peak_concurrent_jobs() == 2, "rank1 peak 2");
        expect(model->redistribution_events() == 2,
               "rank1 two redistributions (join + completion-with-survivor)");
        expect_near(model->hbm_busy_ns(), 3000.0, 2.0, "rank1 busy 3000");
        expect_near(model->hbm_shared_ns(), 2000.0, 2.0,
                    "rank1 shared 2000 (50/50 window)");
        expect(terminal(1, 2).second == 2100,
               "rank1 restore completes at 2100 (half rate), got " +
                   std::to_string(terminal(1, 2).second));
        expect(terminal(1, 1).second == 3100,
               "rank1 COMP completes at 3100 (full rate after), got " +
                   std::to_string(terminal(1, 1).second));
    }

    // Rank 2 -- pool write join with the HBM side first: HBM job done ~2100
    // (half rate), FIFO done exactly 2500; the MEM node must wait for the
    // FIFO (2500), not complete at the first arrival (2100).
    {
        const auto* model =
            systems[2]->workload->local_hbm_bandwidth_model.get();
        expect_near(model->pool_write_bytes_served(), 1e6, 0.01,
                    "rank2 pool_write served ~= 1MB");
        expect_near(model->compute_bytes_served(), 4e6, 0.01,
                    "rank2 compute served ~= 4MB");
        expect(model->peak_concurrent_jobs() == 2, "rank2 peak 2");
        const uint64_t tick = terminal(2, 2).second;
        expect(tick == 2500,
               "rank2 pool MEM terminal at FIFO time 2500 (HBM side "
               "finished first), got " +
                   std::to_string(tick));
        expect(terminal(2, 1).second == 5100,
               "rank2 COMP terminal 5100 (full rate after pool job), got " +
                   std::to_string(terminal(2, 1).second));
    }

    // Rank 4 -- FIFO-first join: the FIFO (6500) completes before the
    // diluted HBM side (~7100); the MEM node terminates at the HBM-side
    // time. Transit traffic from rank5 -> rank3 crosses rank 4 without
    // charging it (comm counters stay zero).
    {
        const auto* model =
            systems[4]->workload->local_hbm_bandwidth_model.get();
        expect_near(model->pool_write_bytes_served(), 3e6, 0.01,
                    "rank4 pool_write served ~= 3MB");
        expect_near(model->restore_bytes_served(), 1e6, 0.01,
                    "rank4 restore served ~= 1MB");
        expect_near(model->compute_bytes_served(), 6e6, 0.01,
                    "rank4 compute served ~= 6MB");
        expect(model->peak_concurrent_jobs() == 3, "rank4 peak 3");
        expect(model->comm_read_bytes_served() == 0.0 &&
                   model->comm_write_bytes_served() == 0.0,
               "rank4 NoC transit charges nothing");
        const uint64_t tick = terminal(4, 3).second;
        expect(tick >= 7099 && tick <= 7105,
               "rank4 pool MEM terminal at HBM-side time ~7101 (FIFO "
               "finished first at 6500), got " +
                   std::to_string(tick));
        expect(terminal(4, 1).second >= 3099 &&
                   terminal(4, 1).second <= 3105,
               "rank4 restore terminal ~3101");
    }

    // Rank 5 -- hbm-charge=false: no job was ever created; the node still
    // completed via the network (terminal recorded above).
    {
        const auto* model =
            systems[5]->workload->local_hbm_bandwidth_model.get();
        expect(model->compute_bytes_served() == 0.0 &&
                   model->restore_bytes_served() == 0.0 &&
                   model->comm_read_bytes_served() == 0.0 &&
                   model->comm_write_bytes_served() == 0.0 &&
                   model->pool_read_bytes_served() == 0.0 &&
                   model->pool_write_bytes_served() == 0.0,
               "rank5 uncharged send created no HBM job");
        expect(model->peak_concurrent_jobs() == 0, "rank5 peak 0");
        expect(model->redistribution_events() == 0, "rank5 no redistributions");
        expect(terminal(5, 1).first == 1,
               "rank5 uncharged node completed via network alone");
    }

    // Rank 3 -- charged recv: exactly one COMM_WRITE job.
    {
        const auto* model =
            systems[3]->workload->local_hbm_bandwidth_model.get();
        expect_near(model->comm_write_bytes_served(), 1e6, 0.01,
                    "rank3 comm_write served ~= 1MB (charged recv)");
        expect(model->peak_concurrent_jobs() == 1, "rank3 peak 1");
    }

    std::printf("[local_hbm_model_test] end_tick=%llu\n",
                static_cast<unsigned long long>(end_tick));

    for (auto it : systems) {
        delete it;
    }
    systems.clear();
    AstraSim::LoggerFactory::shutdown();

    std::remove(system_path.c_str());
    std::remove(remote_path.c_str());
    std::remove(network_path.c_str());
    ::rmdir(dir.c_str());

    if (!g_ok) {
        std::fprintf(stderr, "[local_hbm_model_test] FAIL\n");
        return 1;
    }
    std::printf("[local_hbm_model_test] ALL PASS: N-way equal split, "
                "reallocation, both join orders exactly once, "
                "hbm-charge=false, NoC transit uncharged\n");
    return 0;
}
