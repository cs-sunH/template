/******************************************************************************
collective_tail_chunk_test.cc -- preferred-dataset-splits tail-block byte
regression (P2, 2026-09-28).

When preferred-dataset-splits does not divide the collective size, the
last stream's chunk is the REMAINDER (size % chunk_size), but the
pre-fix generate_collective loop set remain_size = chunk_size for every
stream including the tail: the tail stream emitted a FULL padded chunk
of collective traffic.  With the production splits=6 the collection
traffic was systematically overstated (the report's +16.7% figure).
All_Gather shares the same root (the common remain_size assignment).

Fixture: 1D 4-NPU ring, splits=6, collective size 610 B ->
chunk 101, streams ceil(610/101) = 7, tail chunk 4 B.
A recording fake network captures every front_end_sim_send byte count:

  AllReduce (Ring): msg_size = remain/4 per send -> full streams send
    25 B, the tail stream sends floor(4/4) = 1 B;
  AllGather (Ring): msg_size = remain per send -> full streams send
    101 B, the tail stream sends 4 B.

Every stream runs the same algorithm with the same send count, so the
invariants are count-relative (robust to algorithm-internal send
multiplicity).  Pre-fix, no sub-chunk send exists at all -> RED.

Build: cmake target
       AstraSim_Analytical_Congestion_Aware_CollectiveTailChunkTest.
Run: build/astra_analytical/build_congestion_aware/bin/\
     AstraSim_Analytical_Congestion_Aware_CollectiveTailChunkTest
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
        std::fprintf(stderr, "[collective_tail_chunk_test] FAIL: %s\n",
                     message);
        g_ok = false;
    }
}

// Recording fake network: sim_send captures the payload byte count (the
// Ring algorithms pass msg_size there); events are drained FIFO.
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
        // converted explicitly (tests-F4, 2026-10-01: the old ternary had
        // two token-identical arms, so a non-NS scale would have been
        // silently consumed as ns).
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

    // Bounded twin of run_until_empty for termination assertions: returns
    // false if the protocol is still generating events after max_events
    // dispatches (a non-terminating collective would hang the unbounded
    // drain forever).
    bool run_until_empty_bounded(size_t max_events) {
        for (size_t dispatched = 0; !this->events_.empty(); ++dispatched) {
            if (dispatched >= max_events) {
                return false;
            }
            const Event ev = this->events_.front();
            this->events_.erase(this->events_.begin());
            if (ev.time_ns > this->now_ns_) {
                this->now_ns_ = ev.time_ns;
            }
            ev.fn(ev.arg);
        }
        return true;
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

std::string write_system_config(const char* all_reduce_impl = "ring") {
    char pattern[] = "/tmp/astra-tail-chunk-XXXXXX";
    const int fd = ::mkstemp(pattern);
    if (fd < 0) {
        std::perror("mkstemp");
        std::exit(EXIT_FAILURE);
    }
    ::close(fd);
    std::ofstream out(pattern);
    out << "{\n"
        << "  \"scheduling-policy\": \"FIFO\",\n"
        << "  \"preferred-dataset-splits\": 6,\n"
        << "  \"roofline-enabled\": 1,\n"
        << "  \"peak-perf\": 1000,\n"
        << "  \"local-mem-bw\": 3000,\n"
        << "  \"local-mem-latency\": 100,\n"
        << "  \"hbm-bandwidth-contention\": false,\n"
        << "  \"all-reduce-implementation\": [\"" << all_reduce_impl
        << "\"],\n"
        << "  \"all-gather-implementation\": [\"ring\"],\n"
        << "  \"reduce-scatter-implementation\": [\"ring\"],\n"
        << "  \"all-to-all-implementation\": [\"ring\"]\n"
        << "}\n";
    out.close();
    return pattern;
}

struct Fixture {
    RecordingNetworkApi net;
    std::shared_ptr<NodeStoreGraphSource> source =
        std::make_shared<NodeStoreGraphSource>();
    // Leaked deliberately (Sys::all_sys is process-global static state;
    // same pattern as mem_dispatch_throw_test.cc).
    Sys* sys;

    Fixture(const std::string& system_config)
        : sys(new Sys(0, "workload-unused", "empty", system_config, &net,
                      std::vector<int>{4}, std::vector<int>{1}, 1.0, 1.0,
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
        node.coll.pg_name = "";  // default communicator (all ranks)
        const uint64_t id = this->source->store().add_node(std::move(node));
        const auto* view = this->source->lookup_ptr(id);
        this->sys->workload->issue(*view);
    }
};

size_t count_sends(const std::vector<uint64_t>& sends, uint64_t bytes) {
    return static_cast<size_t>(
        std::count(sends.begin(), sends.end(), bytes));
}

void print_send_histogram(const std::vector<uint64_t>& sends) {
    std::map<uint64_t, size_t> histogram;
    for (const uint64_t bytes : sends) {
        histogram[bytes]++;
    }
    for (const auto& [bytes, count] : histogram) {
        std::printf("    send %llu B x %zu\n",
                    static_cast<unsigned long long>(bytes), count);
    }
}

}  // namespace

int main() {
    setvbuf(stdout, nullptr, _IONBF, 0);
    const std::string system_config = write_system_config();
    // 610 B with splits=6: chunk 101, streams ceil(610/101) = 7, tail 4 B.
    constexpr uint64_t kCollectiveBytes = 610;
    constexpr uint64_t kAllReduce = 0;  // ChakraProtoMsg::ALL_REDUCE
    constexpr uint64_t kAllGather = 2;  // ChakraProtoMsg::ALL_GATHER

    {
        Fixture fixture(system_config);
        fixture.issue_collective(kAllReduce, kCollectiveBytes,
                                 "tail_ar");
        fixture.net.run_until_empty();
        std::printf("[AR] send histogram:\n");
        print_send_histogram(fixture.net.send_bytes);
        const size_t full = count_sends(fixture.net.send_bytes, 25);
        const size_t tail = count_sends(fixture.net.send_bytes, 1);
        expect(tail > 0,
               "AR: the 4-byte tail stream emits sub-chunk sends (1 B)");
        expect(full > 0 && full == 6 * tail,
               "AR: six full-chunk streams (25 B sends) per tail stream");
        fixture.net.send_bytes.clear();
    }
    {
        Fixture fixture(system_config);
        fixture.issue_collective(kAllGather, kCollectiveBytes,
                                 "tail_ag");
        fixture.net.run_until_empty();
        std::printf("[AG] send histogram:\n");
        print_send_histogram(fixture.net.send_bytes);
        const size_t full = count_sends(fixture.net.send_bytes, 101);
        const size_t tail = count_sends(fixture.net.send_bytes, 4);
        expect(tail > 0,
               "AG: the 4-byte tail stream emits its actual bytes (4 B)");
        expect(full > 0 && full == 6 * tail,
               "AG: six full-chunk streams (101 B sends) per tail stream");
    }
    (void)::unlink(system_config.c_str());

    // R4-cpp-F2 (2026-10-01): the All_Reduce-via-AllToAll subpath
    // (all-reduce-implementation "direct") had zero test coverage -- round-2
    // verification derived its termination by hand for n=2/n=4; this case
    // machine-pins termination and the byte discipline on the same 4-NPU
    // ring. The event budget bounds a would-be non-terminating protocol
    // instead of hanging the drain.
    {
        const std::string direct_ar_config = write_system_config("direct");
        Fixture fixture(direct_ar_config);
        fixture.issue_collective(kAllReduce, kCollectiveBytes, "direct_ar");
        constexpr size_t kMaxEvents = 100000;
        expect(fixture.net.run_until_empty_bounded(kMaxEvents),
               "AR-direct: protocol terminates within the event budget");
        std::printf("[AR-direct] send histogram:\n");
        print_send_histogram(fixture.net.send_bytes);
        expect(!fixture.net.send_bytes.empty(),
               "AR-direct: the direct path emits network sends");
        expect(count_sends(fixture.net.send_bytes, 0) == 0,
               "AR-direct: no zero-byte send reaches the network");
        // Pinned send volume (610 B, splits=6, chunk 25 B full / 1 B tail,
        // 4-rank direct fan-out): every chunk reaches each peer and the
        // tail stream emits sub-chunk sends, so any change to the direct
        // path's chunk/peer arithmetic shows up here.
        expect(fixture.net.send_bytes.size() == 42 &&
                   count_sends(fixture.net.send_bytes, 25) == 36 &&
                   count_sends(fixture.net.send_bytes, 1) == 6,
               "AR-direct: pinned 36x25B + 6x1B send volume");
        (void)::unlink(direct_ar_config.c_str());
    }

    std::fprintf(stderr, "[collective_tail_chunk_test] %d checks: %s\n",
                 g_checks, g_ok ? "PASS" : "FAIL");
    return g_ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
