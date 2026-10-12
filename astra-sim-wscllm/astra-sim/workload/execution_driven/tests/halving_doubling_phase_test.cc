/******************************************************************************
halving_doubling_phase_test.cc -- HalvingDoubling phase-arithmetic 1-byte
floor regression (F1-CPP C1-1, 2026-09-29).

The DB1 constructor clamp pinned only the INITIAL msg_size. The
process_max_count halving chain (uint64 /= double truncation) re-zeroes
msg_size mid-schedule whenever the remaining phase size drops below the
divisor, so the ready()/release emissions of the LATER phases carried
count == 0 -- the fluid backend rejects that fail-closed ("invalid fluid
flow"). For an n-rank All_Reduce every d < 2n was hit (n=8, d=8 ->
[4,2,1,0,0,0]: the RS->AG turning point divides 1/2 -> 0 and the /0.5
turning step doubles the zero back to zero); for Reduce_Scatter every
d < n.

Fixture: 1D 8-NPU ring (power-of-two, the exact_log2 guard's requirement),
all four native implementation keys = "halvingDoubling", splits=1. A
recording fake network captures every front_end_sim_send byte count. The
workload path clamps an All_Reduce/Reduce_Scatter chunk up to
total_nodes, so the smallest reachable All_Reduce chunk on 8 NPUs is
d=8 -- i.e. the RV1 trigger "history barrier ALL_REDUCE bytes=1" lands on
exactly the pinned d=8 case.

Pins per case: the send COUNT is the phase count (6 for AR n=8, 3 for
RS/AG), NO send carries 0 bytes, and the exact post-fix phase sequences
are pinned for d=8 (the RV1 example) and d=15:

  All_Reduce d=8  -> sends {4, 2, 1, 1, 2, 4}
  All_Reduce d=15 -> sends {7, 3, 1, 1, 2, 4}
  All_Gather d=1  -> sends {1, 2, 4}            (msg only doubles: safe)
  Reduce_Scatter d=8 -> sends {4, 2, 1}         (d >= n: safe)

The 2026-10-01 F1 fix (HalvingDoubling.cc, RS->AG turning point) derives
the first All_Gather phase from final_data_size / nodes_in_ring instead
of doubling the truncation-corrupted msg_size accumulator; the AR
sequences above pin that fix. Pre-fix, the AG half ran at 2x byte volume
(d=8 -> {4, 2, 1, 2, 4, 8}); before the 2026-09-29 C1-1 clamp the
d=8/d=15 cases emitted 0-byte sends outright.

Build: cmake target
       AstraSim_Analytical_Congestion_Aware_HalvingDoublingPhaseTest.
Run: build/astra_analytical/build_congestion_aware/bin/\
     AstraSim_Analytical_Congestion_Aware_HalvingDoublingPhaseTest
Exit code 0 on ALL PASS.
******************************************************************************/

#include "astra-sim/common/AstraNetworkAPI.hh"
#include "astra-sim/system/Sys.hh"
#include "astra-sim/workload/Workload.hh"
#include "astra-sim/workload/execution_driven/NodeStore.hh"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <memory>
#include <string>
#include <vector>

using namespace AstraSim;
using AstraSim::ExecutionDriven::NodeKind;
using AstraSim::ExecutionDriven::NodeStoreGraphSource;
using AstraSim::ExecutionDriven::OnlineNode;

namespace {

bool g_ok = true;
int g_checks = 0;

void expect(bool condition, const char* message) {
    ++g_checks;
    if (!condition) {
        std::fprintf(stderr,
                     "[halving_doubling_phase_test] FAIL: %s\n", message);
        g_ok = false;
    }
}

// Recording fake network: sim_send captures the payload byte count (the
// collective algorithms pass msg_size there); events are drained FIFO.
class RecordingNetworkApi : public AstraNetworkAPI {
  public:
    RecordingNetworkApi() : AstraNetworkAPI(0) {}

    int sim_send(void* /*buffer*/, uint64_t count, int /*type*/,
                 int /*dst*/, int /*tag*/, sim_request* /*request*/,
                 void (*msg_handler)(void*), void* fun_arg) override {
        this->send_bytes.push_back(count);
        this->schedule(msg_handler, fun_arg);
        return 0;
    }

    // AstraNetworkAPI 纯虚面:桩无链路带宽语义,返回有限占位值即可。
    double get_BW_at_dimension(int /*dim*/) override { return 1.0; }

    int sim_recv(void* /*buffer*/, uint64_t /*count*/, int /*type*/,
                 int /*src*/, int /*tag*/, sim_request* /*request*/,
                 void (*msg_handler)(void*), void* fun_arg) override {
        this->schedule(msg_handler, fun_arg);
        return 0;
    }

    void sim_schedule(AstraSim::timespec_t delta, void (*fun_ptr)(void*),
                      void* fun_arg) override {
        // Sys always passes NS resolution here; the other scales are
        // converted explicitly (the old ternary had two token-identical
        // arms, so a non-NS scale would have been silently consumed as ns).
        long double ns = delta.time_val;
        switch (delta.time_res) {
            case NS: break;
            case US: ns *= 1e3L; break;
            case MS: ns *= 1e6L; break;
            case SE: ns *= 1e9L; break;
            default:
                std::fprintf(stderr,
                             "[RecordingNetworkApi] sim_schedule: unsupported "
                             "time resolution %d (Sys always passes NS)\n",
                             static_cast<int>(delta.time_res));
                std::abort();
        }
        this->schedule_at(this->now_ns_ + static_cast<uint64_t>(ns),
                          fun_ptr, fun_arg);
    }

    AstraSim::timespec_t sim_get_time() override {
        return {AstraSim::NS, static_cast<long double>(this->now_ns_)};
    }

    void run_until_empty() {
        while (!this->events_.empty()) {
            const Event ev = this->events_.front();
            this->events_.erase(this->events_.begin());
            if (ev.time_ns > this->now_ns_) {
                this->now_ns_ = ev.time_ns;
            }
            ev.fn(ev.arg);
        }
    }

    std::vector<uint64_t> send_bytes;

  private:
    struct Event {
        uint64_t time_ns;
        void (*fn)(void*);
        void* arg;
    };

    void schedule(void (*fn)(void*), void* arg) {
        this->schedule_at(this->now_ns_, fn, arg);
    }

    void schedule_at(uint64_t time_ns, void (*fn)(void*), void* arg) {
        this->events_.push_back({time_ns, fn, arg});
    }

    std::vector<Event> events_;
    uint64_t now_ns_ = 0;
};

std::string write_system_config() {
    char pattern[] = "/tmp/astra-hd-phase-XXXXXX";
    const int fd = ::mkstemp(pattern);
    if (fd < 0) {
        std::perror("mkstemp");
        std::exit(EXIT_FAILURE);
    }
    ::close(fd);
    std::ofstream out(pattern);
    out << "{\n"
        << "  \"scheduling-policy\": \"FIFO\",\n"
        << "  \"preferred-dataset-splits\": 1,\n"
        << "  \"roofline-enabled\": 1,\n"
        << "  \"peak-perf\": 1000,\n"
        << "  \"local-mem-bw\": 3000,\n"
        << "  \"local-mem-latency\": 100,\n"
        << "  \"hbm-bandwidth-contention\": false,\n"
        << "  \"all-reduce-implementation\": [\"halvingDoubling\"],\n"
        << "  \"all-gather-implementation\": [\"halvingDoubling\"],\n"
        << "  \"reduce-scatter-implementation\": [\"halvingDoubling\"],\n"
        << "  \"all-to-all-implementation\": [\"direct\"]\n"
        << "}\n";
    out.close();
    return pattern;
}

struct Fixture {
    RecordingNetworkApi net;
    std::shared_ptr<NodeStoreGraphSource> source =
        std::make_shared<NodeStoreGraphSource>();
    // Leaked deliberately (Sys::all_sys is process-global static state;
    // same pattern as collective_tail_chunk_test.cc).
    Sys* sys;

    Fixture(const std::string& system_config)
        : sys(new Sys(0, "workload-unused", "empty", system_config, &net,
                      std::vector<int>{8}, std::vector<int>{1}, 1.0, 1.0,
                      false,
                      AstraSim::ExecutionDriven::ExecutionMode::Online,
                      source)) {}

    void issue_collective(uint64_t comm_type, uint64_t bytes,
                          const char* name) {
        OnlineNode node;
        node.rank = 0;
        node.kind = NodeKind::CommCollective;
        node.node_type = 7;  // ChakraProtoMsg::COMM_COLL_NODE
        node.name = name;
        node.coll.comm_type = comm_type;  // ChakraProtoMsg enum value
        node.coll.bytes = bytes;
        node.coll.pg_name = "";  // default communicator (all 8 ranks)
        const uint64_t id = this->source->store().add_node(std::move(node));
        const auto* view = this->source->lookup_ptr(id);
        this->sys->workload->issue(*view);
    }
};

std::vector<uint64_t> sorted_copy(std::vector<uint64_t> sends) {
    std::sort(sends.begin(), sends.end());
    return sends;
}

void expect_no_zero_send(const std::vector<uint64_t>& sends, const char* tag) {
    for (const uint64_t bytes : sends) {
        if (bytes == 0) {
            expect(false, (std::string(tag) +
                           ": a phase emission carries count == 0 (the "
                           "fluid backend rejects it fail-closed)")
                              .c_str());
            return;
        }
    }
    expect(true, (std::string(tag) + ": no zero-byte phase emission").c_str());
}

void run_case(const std::string& system_config, uint64_t comm_type,
              uint64_t bytes, const char* tag,
              const std::vector<uint64_t>& expected_sends) {
    Fixture fixture(system_config);
    fixture.issue_collective(comm_type, bytes, tag);
    fixture.net.run_until_empty();
    expect(fixture.net.send_bytes.size() == expected_sends.size(),
           (std::string(tag) + ": send count equals the phase count")
               .c_str());
    expect_no_zero_send(fixture.net.send_bytes, tag);
    expect(sorted_copy(fixture.net.send_bytes) ==
               sorted_copy(expected_sends),
           (std::string(tag) + ": the full phase sequence matches the "
            "1-byte-floored schedule")
               .c_str());
    std::printf("    %s: sends =", tag);
    for (const uint64_t bytes_sent : fixture.net.send_bytes) {
        std::printf(" %llu", static_cast<unsigned long long>(bytes_sent));
    }
    std::printf("\n");
}

}  // namespace

int main() {
    setvbuf(stdout, nullptr, _IONBF, 0);
    const std::string system_config = write_system_config();
    constexpr uint64_t kAllReduce = 0;      // ChakraProtoMsg::ALL_REDUCE
    constexpr uint64_t kReduceScatter = 7;  // ChakraProtoMsg::REDUCE_SCATTER
    constexpr uint64_t kAllGather = 2;      // ChakraProtoMsg::ALL_GATHER

    // RV1's exact trigger: n=8, d=8 (< 2n = 16) folded the turning point
    // into [4,2,1,0,0,0]. bytes=8 with splits=1 is also what the workload
    // path produces for ANY All_Reduce request below 8 bytes (the chunk is
    // clamped up to total_nodes), e.g. the history barrier bytes=1.
    run_case(system_config, kAllReduce, 8, "ar_d8", {4, 2, 1, 1, 2, 4});
    // A second d < 2n shape: the RS half truncates to {7,3,1} and the AG
    // half restarts from the 1-byte floor.
    run_case(system_config, kAllReduce, 15, "ar_d15", {7, 3, 1, 1, 2, 4});
    // All_Gather only doubles msg_size: never zero, sanity pin.
    run_case(system_config, kAllGather, 1, "ag_d1", {1, 2, 4});
    // Reduce_Scatter d=8 = n: the halving chain bottoms out at exactly 1.
    run_case(system_config, kReduceScatter, 8, "rs_d8", {4, 2, 1});

    (void)::unlink(system_config.c_str());

    std::fprintf(stderr,
                 "[halving_doubling_phase_test] %d checks: %s\n", g_checks,
                 g_ok ? "PASS" : "FAIL");
    return g_ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
