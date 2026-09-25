/******************************************************************************
BaseStream and UsageTracker bounded-lifecycle regression fixture.

BaseStream used to append one entry per stream id to three static maps.  The
maps had no production consumers: ask_for_schedule already proves readiness by
checking every rank's ready-list front, and ready_counter/suspended_streams
have no writers/readers.  This fixture proves the retained ready-list decision
is exact, including subgroup-shaped lifetimes and same-local-id collisions,
then stress-constructs one million streams.  It also checks that online
UsageTracker keeps its precise level without retaining transition history.
*******************************************************************************/

#include <unistd.h>

#include <cerrno>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <list>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "astra-sim/common/AstraNetworkAPI.hh"
#include "astra-sim/system/BaseStream.hh"
#include "astra-sim/system/DataSet.hh"
#include "astra-sim/system/StreamBaseline.hh"
#include "astra-sim/system/Sys.hh"
#include "astra-sim/system/UsageTracker.hh"

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

    constexpr char contents[] =
        "{\"scheduling-policy\":\"FIFO\",\"hbm-bandwidth-contention\":false}";
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
    TestNetworkApi& network) {
    return std::make_unique<AstraSim::Sys>(
        id, "unused", "empty", system_config, &network,
        std::vector<int>{1}, std::vector<int>{1}, 1.0, 1.0, false,
        AstraSim::ExecutionDriven::ExecutionMode::Online,
        std::make_shared<AstraSim::ExecutionDriven::EmptyGraphSource>());
}

struct ReadyStream {
    std::unique_ptr<AstraSim::DataSet> dataset;
    std::unique_ptr<AstraSim::StreamBaseline> stream;
};

ReadyStream make_ready_stream(AstraSim::Sys& sys, int stream_id) {
    AstraSim::CollectivePhase phase;
    phase.queue_id = 0;
    phase.algorithm = nullptr;
    phase.initial_data_size = 0;
    phase.final_data_size = 0;
    phase.enabled = false;
    phase.comm_type = AstraSim::ComType::None;
    auto dataset = std::make_unique<AstraSim::DataSet>(1, 0);
    auto stream = std::make_unique<AstraSim::StreamBaseline>(
        &sys, dataset.get(), stream_id, std::list<AstraSim::CollectivePhase>{
                                           phase},
        0);
    sys.ready_list.push_back(stream.get());
    return {std::move(dataset), std::move(stream)};
}

void clear_ready_lists(const std::array<AstraSim::Sys*, 4>& ranks) {
    for (auto* rank : ranks) {
        rank->ready_list.clear();
    }
}

void expect_not_scheduled(const std::array<AstraSim::Sys*, 4>& ranks,
                          const std::array<size_t, 4>& expected_ready,
                          const char* message) {
    for (size_t index = 0; index < ranks.size(); ++index) {
        expect(ranks[index]->ready_list.size() == expected_ready[index] &&
                   ranks[index]->total_running_streams == 0 &&
                   ranks[index]->first_phase_streams == 0,
               message);
    }
}

void reset_schedule_counters(const std::array<AstraSim::Sys*, 4>& ranks) {
    for (auto* rank : ranks) {
        rank->first_phase_streams = 0;
        rank->total_running_streams = 0;
    }
}

void test_ready_list_equivalence(const std::array<AstraSim::Sys*, 4>& ranks) {
    // The caller has no front: old and new code both return before any work.
    ranks[0]->ask_for_schedule(3);
    expect_not_scheduled(ranks, {0, 0, 0, 0},
                         "empty caller ready list never schedules");

    // At least one rank is empty: the all-rank front scan rejects scheduling.
    {
        std::vector<ReadyStream> streams;
        streams.push_back(make_ready_stream(*ranks[0], 10));
        ranks[0]->ask_for_schedule(3);
        expect_not_scheduled(ranks, {1, 0, 0, 0},
                             "an empty peer ready list blocks scheduling");
        clear_ready_lists(ranks);
    }

    // Every rank has work but their fronts disagree: the same scan rejects it.
    {
        std::vector<ReadyStream> streams;
        for (size_t index = 0; index < ranks.size(); ++index) {
            streams.push_back(make_ready_stream(
                *ranks[index], index == 1 ? 21 : 20));
        }
        ranks[0]->ask_for_schedule(3);
        expect_not_scheduled(ranks, {1, 1, 1, 1},
                             "mismatched all-rank ready fronts block scheduling");
        clear_ready_lists(ranks);
    }

    // Matching fronts schedule no more than max, even when each rank has more.
    {
        std::vector<ReadyStream> streams;
        for (auto* rank : ranks) {
            streams.push_back(make_ready_stream(*rank, 30));
            streams.push_back(make_ready_stream(*rank, 31));
        }
        ranks[0]->ask_for_schedule(1);
        for (auto* rank : ranks) {
            expect(rank->ready_list.size() == 1 &&
                       rank->total_running_streams == 1 &&
                       rank->first_phase_streams == 1,
                   "matching fronts honor ask_for_schedule max");
        }
        clear_ready_lists(ranks);
        reset_schedule_counters(ranks);
    }

    // With max above every queue depth, the smallest ready-list length wins.
    {
        std::vector<ReadyStream> streams;
        streams.push_back(make_ready_stream(*ranks[0], 40));
        streams.push_back(make_ready_stream(*ranks[0], 41));
        streams.push_back(make_ready_stream(*ranks[1], 40));
        streams.push_back(make_ready_stream(*ranks[2], 40));
        streams.push_back(make_ready_stream(*ranks[2], 41));
        streams.push_back(make_ready_stream(*ranks[2], 42));
        streams.push_back(make_ready_stream(*ranks[3], 40));
        streams.push_back(make_ready_stream(*ranks[3], 41));
        ranks[0]->ask_for_schedule(8);
        expect(ranks[0]->ready_list.size() == 1 &&
                   ranks[1]->ready_list.empty() &&
                   ranks[2]->ready_list.size() == 2 &&
                   ranks[3]->ready_list.size() == 1,
               "matching fronts honor the global ready-list minimum");
        for (auto* rank : ranks) {
            expect(rank->total_running_streams == 1 &&
                       rank->first_phase_streams == 1,
                   "minimum scheduling advances all ranks exactly once");
        }
        clear_ready_lists(ranks);
        reset_schedule_counters(ranks);
    }
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

void test_usage_tracker_contract(const std::array<AstraSim::Sys*, 4>& ranks) {
    // The default constructor remains history preserving for static/reporting
    // paths, including the legacy transition contents.
    AstraSim::UsageTracker history(2);
    history.increase_usage();
    history.decrease_usage();
    expect(history.current_level == 0 && history.usage.size() == 2,
           "default UsageTracker preserves legacy transition history");
    const auto& first = history.usage.front();
    const auto& second = history.usage.back();
    expect(first.level == 0 && first.start == 0 && first.end == 0 &&
               second.level == 1 && second.start == 0 && second.end == 0,
           "default UsageTracker record contents remain unchanged");

    auto& online = ranks[0]->scheduler_unit->usage.at(0);
    online.set_usage(1);
    expect(online.current_level == 1 && online.usage.empty(),
           "online scheduler changes level without retaining history");
    online.set_usage(0);

    const int level_before_report = online.current_level;
    const AstraSim::Tick tick_before_report = online.last_tick;
    // UsageTracker::report(CSVWriter*, int) was removed with the CSVWriter
    // chain; only report_percentage keeps an online fail-closed contract.
    bool percentage_failed_closed = false;
    try {
        (void)online.report_percentage(100);
    } catch (const std::logic_error&) {
        percentage_failed_closed = true;
    }
    expect(percentage_failed_closed,
           "online UsageTracker::report_percentage rejects unavailable history");
    expect(online.current_level == level_before_report &&
               online.last_tick == tick_before_report && online.usage.empty(),
           "rejected online history reports do not mutate live state");

    constexpr uint64_t kIterations = 1000000;
    for (uint64_t index = 0; index < kIterations; ++index) {
        online.increase_usage();
        online.decrease_usage();
        if ((index & 65535u) == 0) {
            expect(online.current_level == 0 && online.usage.empty(),
                   "online UsageTracker remains O(1) during stress");
        }
    }
    expect(online.current_level == 0 && online.usage.empty(),
           "online run-end retains zero UsageTracker history");
}

}  // namespace

int main() {
    const std::string system_config = write_minimal_system_config();
    {
        TestNetworkApi network_zero;
        TestNetworkApi network_one;
        TestNetworkApi network_two;
        TestNetworkApi network_three;
        const auto rank_zero = make_online_system(
            0, system_config, network_zero);
        const auto rank_one = make_online_system(
            1, system_config, network_one);
        const auto rank_two = make_online_system(
            2, system_config, network_two);
        const auto rank_three = make_online_system(
            3, system_config, network_three);
        const std::array<AstraSim::Sys*, 4> ranks = {
            rank_zero.get(), rank_one.get(), rank_two.get(), rank_three.get()};

        test_ready_list_equivalence(ranks);
        test_subgroup_lifetimes_and_id_collisions(ranks);
        test_stream_lifecycle_stress(ranks);
        test_usage_tracker_contract(ranks);
    }
    if (::unlink(system_config.c_str()) != 0) {
        std::fprintf(stderr,
                     "[system_history_lifecycle_test] unlink failed: %s\n",
                     std::strerror(errno));
        return EXIT_FAILURE;
    }
    return g_ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
