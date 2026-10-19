/******************************************************************************
BaseStream bounded-lifecycle regression fixture.

BaseStream used to append one entry per stream id to three static maps.  The
maps had no production consumers, so the ready-list front check in Sys::schedule
is the sole readiness decision.  This fixture proves stream lifecycles are
self-contained, including subgroup-shaped lifetimes and same-local-id
collisions, then stress-constructs one million streams.
*******************************************************************************/

#include <unistd.h>

#include <cerrno>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <list>
#include <memory>
#include <string>
#include <vector>

#include "astra-sim/common/AstraNetworkAPI.hh"
#include "astra-sim/common/AstraRemoteMemoryAPI.hh"
#include "astra-sim/system/BaseStream.hh"
#include "astra-sim/system/Sys.hh"
#include "astra-sim/workload/execution_driven/NodeStore.hh"

namespace {

bool g_ok = true;

void expect(bool condition, const char* message) {
    if (!condition) {
        std::fprintf(stderr,
                     "[system_history_lifecycle_test] FAIL: %s\n",
                     message);
        g_ok = false;
    }
}

class TestRemoteMemoryApi final : public AstraSim::AstraRemoteMemoryAPI {
  public:
    void set_sys(int, AstraSim::Sys*) override {}
    void issue(uint64_t, AstraSim::WorkloadLayerHandlerData*) override {}
};

class TestNetworkApi final : public AstraSim::AstraNetworkAPI {
  public:
    TestNetworkApi() : AstraNetworkAPI(0) {}

    int sim_send(void*, uint64_t, int, int, int, AstraSim::sim_request*,
                 void (*)(void*), void*) override {
        return 0;
    }

    int sim_recv(void*, uint64_t, int, int, int, AstraSim::sim_request*,
                 void (*)(void*), void*) override {
        return 0;
    }

    void sim_schedule(AstraSim::timespec_t, void (*)(void*), void*) override {}

    AstraSim::timespec_t sim_get_time() override {
        return {AstraSim::NS, 0};
    }
};

class TestStream final : public AstraSim::BaseStream {
  public:
    TestStream(int stream_id, AstraSim::Sys* owner)
        : BaseStream(stream_id, owner, std::list<AstraSim::CollectivePhase>()) {}

    void call(AstraSim::EventType, AstraSim::CallData*) override {}
    void consume(AstraSim::RecvPacketEventHandlerData*) override {}
    void init() override {}
};

std::string write_minimal_system_config() {
    char path[] = "/tmp/astra_system_history_XXXXXX";
    const int fd = ::mkstemp(path);
    if (fd < 0) {
        std::fprintf(stderr,
                     "[system_history_lifecycle_test] mkstemp failed: %s\n",
                     std::strerror(errno));
        std::exit(EXIT_FAILURE);
    }

    // The Sys constructor builds per-ComType logical topologies from the
    // config (the Sys::Sys constructor's logical_topologies build-out, via
    // CollectiveImplLookup::get_collective_impl with
    // BypassRule::BYPASS_ALL_CUSTOM); a missing or empty *-implementation key no
    // longer aborts construction (GeneralComplexTopology tolerates the empty
    // list as a dummy topology) but would make any use of that collective
    // type fail inside CollectiveImplLookup::get_collective_impl. The four
    // keys below are the proven one-topology-dimension shape
    // (physical_dims={1}, same as local_hbm_bandwidth_model_test.cc).
    constexpr char contents[] =
        "{\"scheduling-policy\":\"FIFO\",\"preferred-dataset-splits\":1,"
        "\"collective-optimization\":\"localBWAware\","
        "\"all-reduce-implementation\":[\"ring\"],"
        "\"all-gather-implementation\":[\"ring\"],"
        "\"reduce-scatter-implementation\":[\"ring\"],"
        "\"all-to-all-implementation\":[\"ring\"],"
        "\"hbm-bandwidth-contention\":false}";
    size_t offset = 0;
    while (offset < sizeof(contents) - 1) {
        const ssize_t written =
            ::write(fd, contents + offset, sizeof(contents) - 1 - offset);
        if (written <= 0) {
            std::fprintf(stderr,
                         "[system_history_lifecycle_test] config write failed: %s\n",
                         std::strerror(errno));
            ::close(fd);
            ::unlink(path);
            std::exit(EXIT_FAILURE);
        }
        offset += static_cast<size_t>(written);
    }
    if (::close(fd) != 0) {
        std::fprintf(stderr,
                     "[system_history_lifecycle_test] config close failed: %s\n",
                     std::strerror(errno));
        ::unlink(path);
        std::exit(EXIT_FAILURE);
    }
    return path;
}

std::unique_ptr<AstraSim::Sys> make_online_system(
    int id,
    const std::string& system_config,
    TestRemoteMemoryApi& remote_memory,
    TestNetworkApi& network) {
    return std::make_unique<AstraSim::Sys>(
        id, "unused", "empty", system_config, &remote_memory, &network,
        std::vector<int>{1}, std::vector<int>{1}, 1.0, 1.0, false,
        AstraSim::ExecutionDriven::ExecutionMode::Online,
        // Empty-store NodeStoreGraphSource: the retired EmptyGraphSource
        // placeholder is gone; an empty store never yields nodes either.
        std::make_shared<AstraSim::ExecutionDriven::NodeStoreGraphSource>());
}

void test_subgroup_lifetimes_and_id_collisions(
    const std::array<AstraSim::Sys*, 4>& ranks) {
    // A 2-of-4 subgroup can finish before its peer constructs.  No global
    // BaseStream table exists, so this cannot retain a partial barrier.
    {
        TestStream first(77, ranks[0]);
    }
    {
        TestStream late_peer(77, ranks[2]);
    }

    // Two independent 2-of-4 groups may use the same local stream id at the
    // same time.  Their lifetimes are independent because no id-keyed static
    // state remains.
    {
        TestStream group_a_rank_zero(88, ranks[0]);
        TestStream group_a_rank_two(88, ranks[2]);
        TestStream group_b_rank_one(88, ranks[1]);
        TestStream group_b_rank_three(88, ranks[3]);
    }
    for (auto* rank : ranks) {
        expect(rank->ready_list.empty() && rank->active_Streams.at(0).empty(),
               "subgroup/id-collision lifecycles leave no scheduler linkage");
    }
}

void test_stream_lifecycle_stress(const std::array<AstraSim::Sys*, 4>& ranks) {
    constexpr uint64_t kIterations = 1000000;
    for (uint64_t index = 0; index < kIterations; ++index) {
        TestStream stream(static_cast<int>(index & 1023u),
                          ranks[index & 3u]);
    }
    for (auto* rank : ranks) {
        expect(rank->ready_list.empty() && rank->active_Streams.at(0).empty(),
               "one million BaseStream lifecycles retain no scheduler state");
    }
}

}  // namespace

int main() {
    const std::string system_config = write_minimal_system_config();
    {
        TestRemoteMemoryApi remote_zero;
        TestRemoteMemoryApi remote_one;
        TestRemoteMemoryApi remote_two;
        TestRemoteMemoryApi remote_three;
        TestNetworkApi network_zero;
        TestNetworkApi network_one;
        TestNetworkApi network_two;
        TestNetworkApi network_three;
        const auto rank_zero = make_online_system(
            0, system_config, remote_zero, network_zero);
        const auto rank_one = make_online_system(
            1, system_config, remote_one, network_one);
        const auto rank_two = make_online_system(
            2, system_config, remote_two, network_two);
        const auto rank_three = make_online_system(
            3, system_config, remote_three, network_three);
        const std::array<AstraSim::Sys*, 4> ranks = {
            rank_zero.get(), rank_one.get(), rank_two.get(), rank_three.get()};

        test_subgroup_lifetimes_and_id_collisions(ranks);
        test_stream_lifecycle_stress(ranks);
    }
    if (::unlink(system_config.c_str()) != 0) {
        std::fprintf(stderr,
                     "[system_history_lifecycle_test] unlink failed: %s\n",
                     std::strerror(errno));
        return EXIT_FAILURE;
    }
    return g_ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
