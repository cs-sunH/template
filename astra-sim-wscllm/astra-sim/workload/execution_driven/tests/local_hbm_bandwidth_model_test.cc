/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

/****************************************************************************

local_hbm_bandwidth_model_test.cc -- hbm-bandwidth-contention unit fixture.

Standalone deterministic test (same fixture style as the other
execution_driven tests; the repo has no googletest infrastructure) driving
one online-mode Sys + Workload through a minimal fake network/memory
backend. No .et files, no analytical backend, no request CSV: system.json
variants are written by the test itself into a temp directory and removed
at exit.

Scenarios (all ticks are exact ns; local-mem-bw = 3000 GB/s =>
full_rate = 3000 B/ns unless stated otherwise):

  S1  Fair sharing: COMP(3000B) + COMM_SEND(6000B) + COMM_RECV(9000B)
      issued together at t=0 -> 3 users, 1000 B/ns each:
        COMP completes t=3; then 2 users, 1500 B/ns each:
        SEND completes t=5; then RECV alone, 3000 B/ns: t=6.
      Network callbacks (delay 1ns) arrive BEFORE the HBM legs -> join
      "network first" order; each node terminal record fires exactly once.
      Stats: busy 6ns; served 3000/6000/9000 (comp/read/write);
      peak concurrent 3; redistribution events 4 (2 arrivals + 2
      departures leaving survivors).
  S2  Join "HBM first" order: single SEND(3000B), network delay 50ns.
      HBM leg completes t=1, network t=50 -> node completes exactly once
      at t=50.
  S3  local-mem-latency once per job: SEND(3000B), latency 100ns, network
      delay 5ns -> HBM leg 100 + 3000/3000 = 101 -> node completes t=101.
  S4  Flag off fallback ("hbm-bandwidth-contention": 0): model == nullptr;
      COMP(30000B, negligible ops) uses the closed-form roofline ->
      30000/3000 = 10ns; SEND completes at the network tick only (7ns).
  S5  Single-user COMP parity with roofline max(): peak-perf 1000 TFLOPS
      (1e6 ops/ns), ops 6e6 (6ns) vs bytes 3000 (1ns) -> completes t=6.
  S6  ALL_REDUCE endpoint job (P2, 2026-09-28): the coll node charges
      comm_size through the model (comm_read) and completes at the join
      of the collective callback and the endpoint HBM leg, the same
      two-event join as p2p: coll degenerate self-event t=1, HBM leg
      100 + 3000/3000 = 101 -> node completes t=101.

Build: cmake --build build/astra_analytical/build_congestion_aware -j
       (target AstraSim_Analytical_Congestion_Aware_LocalHbmBandwidthModelTest)
Run:  build/astra_analytical/build_congestion_aware/bin/\
         AstraSim_Analytical_Congestion_Aware_LocalHbmBandwidthModelTest
Exit code 0 on ALL PASS.

****************************************************************************/

#include "astra-sim/common/AstraNetworkAPI.hh"
#include "astra-sim/common/Logging.hh"
#include "astra-sim/system/Sys.hh"
#include "astra-sim/workload/LocalHbmBandwidthModel.hh"
#include "astra-sim/workload/Workload.hh"
#include "astra-sim/workload/execution_driven/CompletionObserver.hh"
#include "astra-sim/workload/execution_driven/NodeStore.hh"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <map>
#include <string>
#include <vector>

#include <unistd.h>  // getpid (tests-F6 pid-namespaced temp dir)

using namespace AstraSim;
using AstraSim::ExecutionDriven::NodeKind;
using AstraSim::ExecutionDriven::NodeStoreGraphSource;
using AstraSim::ExecutionDriven::NodeTerminalStatus;
using AstraSim::ExecutionDriven::OnlineNode;

namespace {

int g_failures = 0;

#define CHECK(cond, msg)                                            \
    do {                                                            \
        if (!(cond)) {                                              \
            std::fprintf(stderr, "[FAIL] %s:%d: %s\n", __FILE__,    \
                         __LINE__, msg);                            \
            g_failures++;                                           \
        }                                                           \
    } while (0)

#define CHECK_EQ_T(actual, expected, msg)                           \
    do {                                                            \
        const auto _a = (actual);                                   \
        const auto _e = (expected);                                 \
        if (!(_a == _e)) {                                          \
            std::fprintf(stderr,                                    \
                         "[FAIL] %s:%d: %s (actual=%llu expected="  \
                         "%llu)\n",                                 \
                         __FILE__, __LINE__, msg,                   \
                         static_cast<unsigned long long>(_a),       \
                         static_cast<unsigned long long>(_e));      \
            g_failures++;                                           \
        }                                                           \
    } while (0)

// ---------------------------------------------------------------------------
// Minimal deterministic backends
// ---------------------------------------------------------------------------

// Fake network: own tick + FIFO event queue. sim_send/sim_recv deliver the
// Sys::handleEvent callback after a test-chosen delay; sim_schedule backs
// Sys::register_event transitions.
class FakeNetworkApi : public AstraNetworkAPI {
  public:
    explicit FakeNetworkApi(int rank)
        : AstraNetworkAPI(rank), send_delay_ns_(1), recv_delay_ns_(1) {}

    int sim_send(void* /*buffer*/, uint64_t /*count*/, int /*type*/,
                 int /*dst*/, int /*tag*/, sim_request* /*request*/,
                 void (*msg_handler)(void*), void* fun_arg) override {
        this->schedule(this->send_delay_ns_, msg_handler, fun_arg);
        return 0;
    }

    int sim_recv(void* /*buffer*/, uint64_t /*count*/, int /*type*/,
                 int /*src*/, int /*tag*/, sim_request* /*request*/,
                 void (*msg_handler)(void*), void* fun_arg) override {
        this->schedule(this->recv_delay_ns_, msg_handler, fun_arg);
        return 0;
    }

    void sim_schedule(timespec_t delta, void (*fun_ptr)(void*),
                      void* fun_arg) override {
        // Sys always passes NS resolution here; the other scales are
        // converted explicitly (tests-F4, 2026-10-01: the old conditional
        // had two semantically identical arms -- time_val is already long
        // double -- so a non-NS scale would have been silently consumed
        // as ns).
        long double ns = delta.time_val;
        switch (delta.time_res) {
            case NS: break;
            case US: ns *= 1e3L; break;
            case MS: ns *= 1e6L; break;
            case SE: ns *= 1e9L; break;
            default:
                std::fprintf(stderr,
                             "[FakeNetworkApi] sim_schedule: unsupported time "
                             "resolution %d (Sys always passes NS)\n",
                             static_cast<int>(delta.time_res));
                std::abort();
        }
        this->schedule(static_cast<uint64_t>(ns), fun_ptr, fun_arg);
    }

    timespec_t sim_get_time() override {
        timespec_t now;
        now.time_res = NS;
        now.time_val = static_cast<long double>(this->now_ns_);
        return now;
    }

    // AstraNetworkAPI grew this pure virtual in the working-tree interface
    // cleanup; the HBM-model fixture has no link model, so any finite
    // bandwidth answer will do (the model under test reads its own config).
    double get_BW_at_dimension(int /*dim*/) override { return 1.0; }

    // Deterministic pump: earliest (time, seq) first.
    void run_until_empty() {
        while (!this->events_.empty()) {
            auto it = std::min_element(
                this->events_.begin(), this->events_.end(),
                [](const Event& a, const Event& b) {
                    return std::make_pair(a.time_ns, a.seq) <
                        std::make_pair(b.time_ns, b.seq);
                });
            const Event ev = *it;
            this->events_.erase(it);
            if (ev.time_ns > this->now_ns_) {
                this->now_ns_ = ev.time_ns;
            }
            ev.fn(ev.arg);
        }
    }

    uint64_t now_ns() const {
        return this->now_ns_;
    }

    uint64_t send_delay_ns_;
    uint64_t recv_delay_ns_;

  private:
    struct Event {
        uint64_t time_ns;
        uint64_t seq;
        void (*fn)(void*);
        void* arg;
    };

    void schedule(uint64_t delay_ns, void (*fn)(void*), void* arg) {
        this->events_.push_back(
            {this->now_ns_ + delay_ns, this->seq_++, fn, arg});
    }

    std::vector<Event> events_;
    uint64_t seq_ = 0;
    uint64_t now_ns_ = 0;
};

// ---------------------------------------------------------------------------
// Terminal-record ledger (CompletionObserver hook)
// ---------------------------------------------------------------------------

struct TerminalLedger {
    // node_id -> (tick, count of Success records)
    std::map<uint64_t, std::pair<uint64_t, uint64_t>> success;

    static void hook(void* ctx, int /*rank*/, uint64_t node_id,
                     const char* /*request_id*/, const char* /*stage*/,
                     uint64_t /*generation*/, uint64_t tick,
                     int terminal_status) {
        auto* const ledger = static_cast<TerminalLedger*>(ctx);
        if (terminal_status ==
            static_cast<int>(NodeTerminalStatus::Success)) {
            auto& entry = ledger->success[node_id];
            entry.first = tick;
            entry.second++;
        }
    }
};

// ---------------------------------------------------------------------------
// Fixture: one scenario = one online Sys (rank 0) + NodeStore graph source
// ---------------------------------------------------------------------------

// tests-F6 (2026-10-01): pid-namespaced temp dir, the same convention as the
// sibling fixtures (bridge_loopback / calendar_reader_oracle / cli_online /
// service_vacuum / windowed_trace_reader all suffix getpid()). The single
// fixed directory let two concurrent instances delete each other's
// system.json once the suite runs under ctest.
std::string temp_dir() {
    return "/tmp/wscllm_hbm_contention_test_" +
           std::to_string(static_cast<long>(::getpid()));
}

std::string write_system_json(const std::string& name,
                              bool contention,
                              uint64_t local_mem_latency_ns) {
    const std::string path = temp_dir() + "/" + name;
    std::ofstream out(path);
    out << "{\n"
        << "  \"roofline-enabled\": 1,\n"
        << "  \"peak-perf\": 1000,\n"
        << "  \"local-mem-bw\": 3000,\n"
        << "  \"local-mem-latency\": "
        << static_cast<unsigned long long>(local_mem_latency_ns) << ",\n"
        << "  \"hbm-bandwidth-contention\": "
        << (contention ? "1" : "0") << ",\n"
        // The four *-implementation keys are mandatory: Sys fails closed
        // (GeneralComplexTopology, empty implementation table) without
        // them; a single "ring" matches this fixture's {1} shape.
        << "  \"all-reduce-implementation\": [\"ring\"],\n"
        << "  \"all-gather-implementation\": [\"ring\"],\n"
        << "  \"reduce-scatter-implementation\": [\"ring\"],\n"
        << "  \"all-to-all-implementation\": [\"ring\"]\n"
        << "}\n";
    out.close();
    return path;
}

struct Fixture {
    FakeNetworkApi net{0};
    std::shared_ptr<NodeStoreGraphSource> source =
        std::make_shared<NodeStoreGraphSource>();
    Sys* sys = nullptr;
    TerminalLedger ledger;

    Fixture(const std::string& system_json_path) {
        ExecutionDriven::CompletionObserver::instance().set_hook(
            &TerminalLedger::hook, &this->ledger);
        // Leaked deliberately: Sys::all_sys is process-global static state;
        // keeping every scenario's Sys alive avoids teardown-order hazards
        // while each scenario's own FakeNetworkApi owns its event queue.
        this->sys = new Sys(
            0, "workload-unused", "empty", system_json_path,
            &this->net, std::vector<int>{1}, std::vector<int>{1},
            1.0 /*injection*/, 1.0 /*comm scale*/,
            false /*rendezvous*/, ExecutionDriven::ExecutionMode::Online,
            this->source);
    }

    ~Fixture() {
        // Uninstall the hook (fix 2026-09-27): the ctx points at this stack
        // Fixture's ledger while CompletionObserver::instance() is a
        // process-wide singleton; leaving it installed would leave a
        // dangling pointer behind (any late terminal record would be a
        // use-after-free). A null hook is the documented zero-overhead
        // default, and every scenario reinstalls its own hook on entry.
        ExecutionDriven::CompletionObserver::instance().set_hook(nullptr,
                                                                 nullptr);
    }

    Workload* workload() const {
        return this->sys->workload;
    }

    // Build a node in the store and hand the read view to Workload::issue.
    uint64_t issue_node(OnlineNode node) {
        node.rank = 0;
        node.global_id = 0;  // store assigns a fresh id
        const uint64_t id =
            this->source->store().add_node(std::move(node));
        const auto* view = this->source->lookup_ptr(id);
        this->workload()->issue(*view);
        return id;
    }

    static OnlineNode comp_node(uint64_t num_ops, uint64_t tensor_size) {
        OnlineNode node;
        node.kind = NodeKind::Compute;
        node.node_type = 4;  // ChakraProtoMsg::COMP_NODE
        node.name = "comp";
        node.compute.num_ops = num_ops;
        node.compute.tensor_size = tensor_size;
        return node;
    }

    static OnlineNode send_node(uint64_t bytes, int dst) {
        OnlineNode node;
        node.kind = NodeKind::CommSend;
        node.node_type = 5;  // ChakraProtoMsg::COMM_SEND_NODE
        node.name = "comm_send";
        node.comm.bytes = bytes;
        node.comm.src = 0;
        node.comm.dst = dst;
        node.comm.tag = 1;
        return node;
    }

    static OnlineNode recv_node(uint64_t bytes, int src) {
        OnlineNode node;
        node.kind = NodeKind::CommRecv;
        node.node_type = 6;  // ChakraProtoMsg::COMM_RECV_NODE
        node.name = "comm_recv";
        node.comm.bytes = bytes;
        node.comm.src = src;
        node.comm.dst = 0;
        node.comm.tag = 1;
        return node;
    }

    // Four-type collective node builder (2026-10-01): the endpoint HBM
    // charge lives BEFORE the comm_type dispatch in Workload::issue_coll_comm,
    // so AG/A2A/RS symmetry is structural -- this builder pins it per type
    // (proto values: ALL_REDUCE=0, ALL_GATHER=2, ALL_TO_ALL=6,
    // REDUCE_SCATTER=7).
    static OnlineNode coll_node(uint64_t bytes, int comm_type) {
        OnlineNode node;
        node.kind = NodeKind::CommCollective;
        node.node_type = 7;  // ChakraProtoMsg::COMM_COLL_NODE
        node.name = "comm_coll";
        node.coll.comm_type = comm_type;
        node.coll.bytes = bytes;
        node.coll.pg_name = "";  // default communicator (all ranks)
        return node;
    }

    void expect_success(uint64_t node_id, uint64_t tick,
                        const char* what) {
        const auto it = this->ledger.success.find(node_id);
        if (it == this->ledger.success.end()) {
            std::fprintf(stderr, "[FAIL] %s: node %llu never completed\n",
                         what, static_cast<unsigned long long>(node_id));
            g_failures++;
            return;
        }
        CHECK_EQ_T(it->second.first, tick, what);
        CHECK_EQ_T(it->second.second, 1ull, what);
    }
};

void remove_temp_dir() {
    const std::string cmd =
        std::string("rm -rf ") + temp_dir();
    if (std::system(cmd.c_str()) != 0) {
        std::fprintf(stderr, "[warn] failed to remove %s\n",
                     temp_dir().c_str());
    }
}

// ---------------------------------------------------------------------------
// Scenarios
// ---------------------------------------------------------------------------

// S1: 3 concurrent users each get full_rate/3; after the first completion
// the two survivors get full_rate/2; after the second the survivor gets the
// full rate. Network callbacks arrive before the HBM legs (join network
// first). Exactly one Success terminal record per node.
void scenario_fair_sharing_and_reallocation() {
    std::printf("[S1] fair sharing / reallocation / join network-first\n");
    Fixture fx(write_system_json("system_s1.json", true, 0));

    CHECK(fx.workload()->local_hbm_bandwidth_model != nullptr,
          "S1: model must exist with contention on");

    fx.net.send_delay_ns_ = 1;
    fx.net.recv_delay_ns_ = 1;
    const uint64_t comp = fx.issue_node(Fixture::comp_node(1, 3000));
    const uint64_t send = fx.issue_node(Fixture::send_node(6000, 1));
    const uint64_t recv = fx.issue_node(Fixture::recv_node(9000, 1));

    fx.net.run_until_empty();

    // 3 users * 1000 B/ns: COMP(3000B) at t=3; 2 users * 1500 B/ns:
    // SEND(6000B: 3 + 3000/1500) at t=5; RECV alone 3000 B/ns:
    // (9000-3000-2*1500)/3000 -> t=6.
    fx.expect_success(comp, 3, "S1 COMP t=3 (3 users, 1/3 rate each)");
    fx.expect_success(send, 5, "S1 SEND t=5 (2 users, 1/2 rate each)");
    fx.expect_success(recv, 6, "S1 RECV t=6 (sole user, full rate)");

    const LocalHbmBandwidthModel* const model =
        fx.workload()->local_hbm_bandwidth_model.get();
    CHECK_EQ_T(static_cast<uint64_t>(model->hbm_busy_ns() + 0.5), 6ull,
               "S1 busy_ns == 6");
    CHECK_EQ_T(static_cast<uint64_t>(model->compute_bytes_served() + 0.5),
               3000ull, "S1 comp bytes served");
    CHECK_EQ_T(static_cast<uint64_t>(model->comm_read_bytes_served() + 0.5),
               6000ull, "S1 comm_read bytes served");
    CHECK_EQ_T(static_cast<uint64_t>(model->comm_write_bytes_served() + 0.5),
               9000ull, "S1 comm_write bytes served");
    CHECK_EQ_T(model->peak_concurrent_jobs(), 3ull, "S1 peak concurrency 3");
    CHECK_EQ_T(model->redistribution_events(), 4ull,
               "S1 redistribution events (2 arrivals + 2 departures)");
    CHECK(!fx.workload()->local_hbm_bandwidth_model->has_active_jobs(),
          "S1 no active jobs at the end");
    CHECK_EQ_T(fx.source->store().injected_unfinished_summary(0).node_count,
               0ull, "S1 all nodes terminal (0 unfinished in the store)");
}

// S2: join HBM-first order -- the HBM leg finishes long before the network
// callback; the node completes exactly once, at the network tick.
void scenario_join_hbm_first() {
    std::printf("[S2] join HBM-first order\n");
    Fixture fx(write_system_json("system_s2.json", true, 0));

    fx.net.send_delay_ns_ = 50;  // HBM leg: 3000/3000 = 1ns
    const uint64_t send = fx.issue_node(Fixture::send_node(3000, 1));

    fx.net.run_until_empty();

    fx.expect_success(send, 50,
                      "S2 SEND t=50 (HBM done t=1, network done t=50)");
    CHECK(!fx.workload()->local_hbm_bandwidth_model->has_active_jobs(),
          "S2 no active jobs at the end");
}

// S3: local-mem-latency is charged once per job at its start.
void scenario_latency_once_per_job() {
    std::printf("[S3] local-mem-latency once per job\n");
    Fixture fx(write_system_json("system_s3.json", true, 100));

    fx.net.send_delay_ns_ = 5;  // network long done; HBM leg dominates
    const uint64_t send = fx.issue_node(Fixture::send_node(3000, 1));

    fx.net.run_until_empty();

    // HBM leg = 100 (latency) + 3000/3000 (bytes) = 101.
    fx.expect_success(send, 101, "S3 SEND t=101 (latency 100 + 1ns bytes)");
}

// S4: flag off -- full legacy fallback (closed-form roofline, comm without
// HBM accounting).
void scenario_flag_off_fallback() {
    std::printf("[S4] flag off fallback\n");
    Fixture fx(write_system_json("system_s4.json", false, 0));

    CHECK(fx.workload()->local_hbm_bandwidth_model == nullptr,
          "S4: model must NOT exist with contention off");
    CHECK(!fx.sys->hbm_bandwidth_contention, "S4: sys flag parsed as false");

    fx.net.send_delay_ns_ = 7;
    const uint64_t comp = fx.issue_node(Fixture::comp_node(1, 30000));
    const uint64_t send = fx.issue_node(Fixture::send_node(3000, 1));

    fx.net.run_until_empty();

    // Closed-form roofline: 30000B / 3000GB/s = 10ns; comm completes at the
    // network tick only (no HBM join).
    fx.expect_success(comp, 10, "S4 COMP closed-form roofline t=10");
    fx.expect_success(send, 7, "S4 SEND network-only completion t=7");
}

// S5: single COMP user keeps the roofline max(compute, memory) semantics.
void scenario_single_comp_roofline_parity() {
    std::printf("[S5] single COMP roofline max() parity\n");
    Fixture fx(write_system_json("system_s5.json", true, 0));

    // peak-perf 1000 TFLOPS => 1e6 ops/ns; 6e6 ops = 6ns vs 3000B = 1ns.
    const uint64_t comp = fx.issue_node(Fixture::comp_node(6000000, 3000));

    fx.net.run_until_empty();

    fx.expect_success(comp, 6, "S5 COMP t=6 (compute-bound max())");
}

// S6: EVERY collective type's endpoint charges its bytes through the shared
// HBM model, exactly like the COMP and p2p endpoints (the 2026-09-29 4.2
// symmetry fix).  On this {1}-shaped fixture the collective degenerates to a
// self event at t=1; the endpoint HBM job (100ns latency + 3000/3000 =
// 101ns) must HOLD the node open to t=101 via the same two-event join the
// p2p callbacks use.  Pre-fix the node completed at the collective tick
// alone (t=1) and the endpoint bytes never entered the model.
// 2026-10-01: parametrized over all four charged types (ALL_REDUCE /
// ALL_GATHER / ALL_TO_ALL / REDUCE_SCATTER) -- the charge sits before the
// comm_type dispatch, and this loop pins that no future refactor moves it
// back into a single type's branch.
void scenario_all_reduce_endpoint_job() {
    struct CollTypeCase {
        const char* name;
        int comm_type;
    };
    const CollTypeCase cases[] = {
        {"ALL_REDUCE", 0}, {"ALL_GATHER", 2}, {"ALL_TO_ALL", 6},
        {"REDUCE_SCATTER", 7}};
    for (const CollTypeCase& c : cases) {
        std::printf("[S6] %s endpoint HBM job / coll-join\n", c.name);
        Fixture fx(write_system_json("system_s6.json", true, 100));

        LocalHbmBandwidthModel* const model =
            fx.workload()->local_hbm_bandwidth_model.get();
        CHECK(fx.workload()->local_hbm_bandwidth_model != nullptr,
              "S6: model must exist with contention on");

        const uint64_t coll =
            fx.issue_node(Fixture::coll_node(3000, c.comm_type));

        fx.net.run_until_empty();

        std::string what = std::string("S6 ") + c.name +
                           " t=101 (coll done t=1, HBM leg t=101)";
        fx.expect_success(coll, 101, what.c_str());
        CHECK_EQ_T(static_cast<uint64_t>(model->comm_read_bytes_served() + 0.5),
                   3000ull, (std::string("S6 ") + c.name +
                            " endpoint bytes charged through the model")
                       .c_str());
        CHECK(!fx.workload()->local_hbm_bandwidth_model->has_active_jobs(),
              "S6 no active jobs at the end");
        CHECK_EQ_T(
            fx.source->store().injected_unfinished_summary(0).node_count,
            0ull, (std::string("S6 ") + c.name +
                   " node terminal exactly once (0 unfinished)")
                      .c_str());
    }
}

}  // namespace

int main() {
    LoggerFactory::init("empty", "off");
    // tests-F6 (2026-10-01): no remove-at-entry -- the pid-suffixed
    // directory cannot pre-exist from a same-pid stale run in practice, and
    // mkdir -p tolerates a leftover; only the exit path cleans up.
    if (std::system(std::string("mkdir -p " + temp_dir())
                        .c_str()) != 0) {
        std::fprintf(stderr, "[FAIL] cannot create %s\n", temp_dir().c_str());
        return 1;
    }

    scenario_fair_sharing_and_reallocation();
    scenario_join_hbm_first();
    scenario_latency_once_per_job();
    scenario_flag_off_fallback();
    scenario_single_comp_roofline_parity();
    scenario_all_reduce_endpoint_job();

    remove_temp_dir();

    if (g_failures == 0) {
        std::printf("LocalHbmBandwidthModel tests: ALL PASS\n");
        return 0;
    }
    std::printf("LocalHbmBandwidthModel tests: %d FAILURE(S)\n", g_failures);
    return 1;
}
