/******************************************************************************
Statistics online-compaction regression fixture.

Exercises the compact online-service API directly. Service execution keeps its
short-lived state in NodeStore records, never in Statistics'
operator_statistics map; the separate history-preserving case retains the
legacy NodeView/map behavior for microbenchmark window queries.
*******************************************************************************/

#include "astra-sim/common/AstraNetworkAPI.hh"
#include "astra-sim/system/Sys.hh"
#include "astra-sim/workload/Statistics.hh"
#include "astra-sim/workload/Workload.hh"

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <string>
#include <vector>
#include <sys/wait.h>
#include <unistd.h>

using AstraSim::Statistics;
using AstraSim::Tick;
using AstraSim::ExecutionDriven::NodeKind;
using AstraSim::ExecutionDriven::NodeView;
using AstraSim::ExecutionDriven::OnlineStatisticsState;

namespace {

bool g_ok = true;

void expect(bool condition, const char* message) {
    if (!condition) {
        std::fprintf(stderr, "[statistics_online_compaction_test] FAIL: %s\n",
                     message);
        g_ok = false;
    }
}

template <typename Fn>
void expect_child_failure(const char* message, Fn&& child_body) {
    const pid_t child = fork();
    expect(child >= 0, message);
    if (child == 0) {
        child_body();
        _exit(EXIT_SUCCESS);
    }
    if (child < 0) {
        return;
    }
    int status = 0;
    expect(waitpid(child, &status, 0) == child, message);
    expect(!(WIFEXITED(status) && WEXITSTATUS(status) == EXIT_SUCCESS),
           message);
}

NodeView make_node(uint64_t id, NodeKind kind = NodeKind::Compute,
                   bool is_cpu_op = false) {
    NodeView node;
    node.global_id = id;
    node.kind = kind;
    node.is_cpu_op = is_cpu_op;
    return node;
}

NodeView gpu_node(uint64_t id) { return make_node(id); }

uint64_t double_bits(double value) {
    uint64_t bits = 0;
    static_assert(sizeof(bits) == sizeof(value),
                  "double bit comparison requires 64-bit doubles");
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

void expect_same_double_bits(double actual, double expected,
                             const char* message) {
    expect(double_bits(actual) == double_bits(expected), message);
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

std::string write_minimal_system_config() {
    char path[] = "/tmp/astra_statistics_guard_XXXXXX";
    const int fd = ::mkstemp(path);
    if (fd < 0) {
        std::fprintf(stderr,
                     "[statistics_online_compaction_test] mkstemp failed: %s\n",
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
            std::fprintf(
                stderr,
                "[statistics_online_compaction_test] system config write failed: %s\n",
                std::strerror(errno));
            ::close(fd);
            ::unlink(path);
            std::exit(EXIT_FAILURE);
        }
        offset += static_cast<size_t>(written);
    }
    if (::close(fd) != 0) {
        std::fprintf(
            stderr,
            "[statistics_online_compaction_test] system config close failed: %s\n",
            std::strerror(errno));
        ::unlink(path);
        std::exit(EXIT_FAILURE);
    }
    return path;
}

std::unique_ptr<AstraSim::Sys> make_online_test_system(
    const std::string& system_config,
    TestNetworkApi& network) {
    return std::make_unique<AstraSim::Sys>(
        0, "unused", "empty", system_config, &network,
        std::vector<int>{1}, std::vector<int>{1}, 1.0, 1.0, false,
        AstraSim::ExecutionDriven::ExecutionMode::Online,
        std::make_shared<AstraSim::ExecutionDriven::EmptyGraphSource>());
}

struct CompactStatisticsSnapshot {
    Tick wall_time = 0;
    size_t retained_operator_count = 0;
    Tick gpu_busy_time = 0;
    uint64_t compute_weighted_sum_bits = 0;
    uint64_t memory_weighted_sum_bits = 0;
    Tick total_compute_time = 0;
};

CompactStatisticsSnapshot snapshot_compact_statistics(const Statistics& stats) {
    constexpr Tick kFullWindowEnd = std::numeric_limits<Tick>::max();
    CompactStatisticsSnapshot snapshot;
    snapshot.wall_time = stats.get_wall_time();
    snapshot.retained_operator_count = stats.retained_online_operator_count();
    snapshot.gpu_busy_time = stats.calculate_type_time_in_window(
        Statistics::OperatorStatistics::OperatorType::GPU, 0, kFullWindowEnd);
    const auto roofline = stats.calculate_roofline_utilization_in_window(
        0, kFullWindowEnd);
    snapshot.compute_weighted_sum_bits =
        double_bits(roofline.compute_utilization_weighted_sum);
    snapshot.memory_weighted_sum_bits =
        double_bits(roofline.memory_utilization_weighted_sum);
    snapshot.total_compute_time = roofline.total_comp_time;
    return snapshot;
}

bool compact_statistics_unchanged(const Statistics& stats,
                                  const CompactStatisticsSnapshot& expected) {
    const CompactStatisticsSnapshot actual = snapshot_compact_statistics(stats);
    return actual.wall_time == expected.wall_time &&
           actual.retained_operator_count == expected.retained_operator_count &&
           actual.gpu_busy_time == expected.gpu_busy_time &&
           actual.compute_weighted_sum_bits ==
               expected.compute_weighted_sum_bits &&
           actual.memory_weighted_sum_bits == expected.memory_weighted_sum_bits &&
           actual.total_compute_time == expected.total_compute_time;
}

Statistics* g_exit_snapshot_statistics = nullptr;
CompactStatisticsSnapshot g_exit_snapshot;

void verify_compact_state_at_exit() {
    const bool unchanged =
        g_exit_snapshot_statistics != nullptr &&
        compact_statistics_unchanged(*g_exit_snapshot_statistics,
                                     g_exit_snapshot);
    ::_exit(unchanged ? EXIT_FAILURE : EXIT_SUCCESS);
}

void require_compact_state_unchanged_on_exit(Statistics& stats) {
    g_exit_snapshot_statistics = &stats;
    g_exit_snapshot = snapshot_compact_statistics(stats);
    if (std::atexit(verify_compact_state_at_exit) != 0) {
        ::_exit(EXIT_SUCCESS);
    }
}

template <typename Fn>
void expect_child_failure_without_compact_mutation(const char* message,
                                                   Statistics& stats,
                                                   Fn&& child_body) {
    expect_child_failure(message, [&] {
        require_compact_state_unchanged_on_exit(stats);
        child_body();
    });
}

void compact_start(Statistics& stats, const NodeView& node,
                   OnlineStatisticsState& state, Tick tick) {
    stats.record_online_service_start(node, state, tick);
}

void compact_end(Statistics& stats, const NodeView& node,
                 OnlineStatisticsState& state, Tick tick) {
    stats.complete_online_service_operator(node, state, tick);
}

void test_live_gpu_fails_closed() {
    expect_child_failure("live compact GPU query fails closed", [] {
        Statistics stats(nullptr);
        stats.configure_online_history_preservation(false);
        const NodeView live = gpu_node(1);
        OnlineStatisticsState state;
        compact_start(stats, live, state, 10);
        (void)stats.calculate_type_time_in_window(
            Statistics::OperatorStatistics::OperatorType::GPU, 0, 20);
    });
}

void test_direct_completed_gpu_union() {
    Statistics stats(nullptr);
    stats.configure_online_history_preservation(false);
    const NodeView gpu = gpu_node(2);
    OnlineStatisticsState state;
    compact_start(stats, gpu, state, 10);
    compact_end(stats, gpu, state, 30);

    expect(stats.retained_online_operator_count() == 0,
           "direct service GPU completion retains no operator map entry");
    expect(stats.calculate_type_time_in_window(
               Statistics::OperatorStatistics::OperatorType::GPU, 0, 31) ==
               20,
           "direct service GPU union remains exact");
}

void test_compact_post_processing_fails_closed_without_mutation() {
    Statistics stats(nullptr);
    stats.configure_online_history_preservation(false);
    const NodeView gpu = gpu_node(3);
    OnlineStatisticsState state;
    compact_start(stats, gpu, state, 10);
    state.compute_utilization = 0.5;
    state.memory_utilization = 0.25;
    compact_end(stats, gpu, state, 30);

    const CompactStatisticsSnapshot before = snapshot_compact_statistics(stats);
    expect(before.wall_time == 30 && before.gpu_busy_time == 20 &&
               before.total_compute_time == 21,
           "compact post-processing fixture has nonzero aggregate state");
    expect_child_failure_without_compact_mutation(
        "compact Statistics::post_processing fails closed without mutation",
        stats, [&] { stats.post_processing(); });
}

void test_history_preserving_legacy_post_processing_succeeds(
    const std::string& system_config) {
    TestNetworkApi network;
    const auto system =
        make_online_test_system(system_config, network);
    Statistics& stats = *system->workload->stats;
    stats.configure_online_history_preservation(true);

    const NodeView gpu = gpu_node(4);
    stats.record_start(gpu, 10);
    stats.record_end(gpu, 30);
    stats.post_processing();

    expect(stats.get_wall_time() == 30,
           "history-preserving legacy post-processing keeps wall time");
    expect(stats.get_type_time(
               Statistics::OperatorStatistics::OperatorType::GPU) == 20,
           "history-preserving legacy post-processing keeps GPU union");
}

void test_compact_workload_report_fails_closed_without_mutation(
    const std::string& system_config) {
    expect_child_failure(
        "compact Workload::report fails closed without mutation", [&] {
            TestNetworkApi network;
            const auto system =
                make_online_test_system(system_config, network);
            Statistics& stats = *system->workload->stats;
            const NodeView gpu = gpu_node(5);
            OnlineStatisticsState state;
            compact_start(stats, gpu, state, 10);
            state.compute_utilization = 0.5;
            state.memory_utilization = 0.25;
            compact_end(stats, gpu, state, 30);

            require_compact_state_unchanged_on_exit(stats);
            system->workload->report();
        });
}

void test_direct_overlapping_gpu_union_and_roofline() {
    Statistics stats(nullptr);
    stats.configure_online_history_preservation(false);
    const NodeView first = gpu_node(10);
    const NodeView second = gpu_node(11);
    OnlineStatisticsState first_state;
    OnlineStatisticsState second_state;

    compact_start(stats, first, first_state, 10);
    first_state.compute_utilization = 0.5;
    first_state.memory_utilization = 0.125;

    compact_start(stats, second, second_state, 20);
    second_state.compute_utilization = 0.25;
    second_state.memory_utilization = 0.5;

    compact_end(stats, first, first_state, 30);
    expect(stats.retained_online_operator_count() == 0,
           "overlapping direct service leaves no first-node map record");
    compact_end(stats, second, second_state, 50);

    expect(stats.retained_online_operator_count() == 0,
           "overlapping completed direct GPU records stay compact");
    expect(stats.calculate_type_time_in_window(
               Statistics::OperatorStatistics::OperatorType::GPU, 0, 51) ==
               40,
           "overlapping direct GPU intervals preserve their exact union");

    const auto roofline =
        stats.calculate_roofline_utilization_in_window(0, 51);
    expect(roofline.compute_utilization_weighted_sum == 17.5,
           "direct compaction preserves weighted compute utilization");
    expect(roofline.memory_utilization_weighted_sum == 17.5,
           "direct compaction preserves weighted memory utilization");
    expect(roofline.total_comp_time == 51,
           "direct compaction preserves compute duration plus 1ns guard");
}

void test_cpu_no_util_and_comm_state_stay_compact() {
    Statistics stats(nullptr);
    stats.configure_online_history_preservation(false);

    const NodeView cpu = make_node(20, NodeKind::Compute, true);
    OnlineStatisticsState cpu_state;
    compact_start(stats, cpu, cpu_state, 0);
    cpu_state.compute_utilization = 0.25;
    compact_end(stats, cpu, cpu_state, 10);

    const NodeView gpu = gpu_node(21);
    OnlineStatisticsState gpu_state;
    compact_start(stats, gpu, gpu_state, 10);
    // No utilization attributes: it still contributes its duration, but no
    // weighted numerator, exactly like the legacy Statistics behavior.
    compact_end(stats, gpu, gpu_state, 20);

    const NodeView comm = make_node(22, NodeKind::CommSend);
    OnlineStatisticsState comm_state;
    compact_start(stats, comm, comm_state, 20);
    comm_state.comm_size = 4096;
    compact_end(stats, comm, comm_state, 30);

    expect(comm_state.completed && comm_state.comm_size.has_value() &&
               comm_state.comm_size.value() == 4096,
           "compact NodeStore state retains terminal communication facts");
    expect(stats.retained_online_operator_count() == 0,
           "CPU, no-util GPU, and comm state never create map entries");
    expect(stats.calculate_type_time_in_window(
               Statistics::OperatorStatistics::OperatorType::GPU, 0, 31) ==
               10,
           "direct compact path retains GPU accounting with no utilization");

    const auto roofline =
        stats.calculate_roofline_utilization_in_window(0, 31);
    expect(roofline.compute_utilization_weighted_sum == 2.5,
           "CPU compact contribution is retained without a map record");
    expect(roofline.memory_utilization_weighted_sum == 0.0,
           "missing utilization produces no compact weighted numerator");
    expect(roofline.total_comp_time == 21,
           "CPU and no-util GPU durations both remain in compact denominator");
}

void test_exact_sum_matches_history_bit_for_bit() {
    constexpr Tick kLargeDuration = Tick{1} << 53;
    constexpr Tick kEnd = kLargeDuration + 2;

    Statistics compact(nullptr);
    compact.configure_online_history_preservation(false);
    Statistics history(nullptr);
    history.configure_online_history_preservation(true);

    for (uint64_t index = 0; index < 3; ++index) {
        const Tick duration = index == 0 ? kLargeDuration : Tick{1};
        const Tick start = index == 0 ? 0 : kLargeDuration + index - 1;
        const Tick end = start + duration;
        const NodeView compact_node = gpu_node(100 + index);
        OnlineStatisticsState state;
        compact_start(compact, compact_node, state, start);
        state.compute_utilization = 1.0;
        state.memory_utilization = 1.0;
        compact_end(compact, compact_node, state, end);

        const NodeView history_node = gpu_node(200 + index);
        history.record_start(history_node, start);
        auto& history_stat =
            history.get_operator_statistics(history_node.global_id);
        history_stat.compute_utilization = 1.0;
        history_stat.memory_utilization = 1.0;
        history.record_end(history_node, end);
    }

    const auto compact_roofline =
        compact.calculate_roofline_utilization_in_window(0, kEnd);
    const auto history_roofline =
        history.calculate_roofline_utilization_in_window(0, kEnd);
    const double expected = static_cast<double>(kLargeDuration + 2);
    expect_same_double_bits(compact_roofline.compute_utilization_weighted_sum,
                            history_roofline.compute_utilization_weighted_sum,
                            "2^53 + 1 + 1 compute sum matches history bits");
    expect_same_double_bits(compact_roofline.memory_utilization_weighted_sum,
                            history_roofline.memory_utilization_weighted_sum,
                            "2^53 + 1 + 1 memory sum matches history bits");
    expect_same_double_bits(compact_roofline.compute_utilization_weighted_sum,
                            expected,
                            "2^53 + 1 + 1 rounds once to 2^53 + 2");
    expect(compact_roofline.total_comp_time == kLargeDuration + 3 &&
               history_roofline.total_comp_time == kLargeDuration + 3,
           "adversarial exact sums preserve the denominator in both modes");
    expect(compact.retained_online_operator_count() == 0,
           "adversarial compact service still retains no map records");
}

void run_long_direct_service(uint64_t node_count, uint64_t id_base,
                             const char* message) {
    Statistics stats(nullptr);
    stats.configure_online_history_preservation(false);
    NodeView gpu = gpu_node(id_base);
    for (uint64_t index = 1; index <= node_count; ++index) {
        gpu.global_id = id_base + index;
        OnlineStatisticsState state;
        const Tick start = static_cast<Tick>(index * 2);
        compact_start(stats, gpu, state, start);
        compact_end(stats, gpu, state, start + 1);
        if ((index & 65535u) == 0) {
            expect(stats.retained_online_operator_count() == 0, message);
        }
    }
    expect(stats.retained_online_operator_count() == 0, message);
    expect(stats.calculate_type_time_in_window(
               Statistics::OperatorStatistics::OperatorType::GPU, 0,
               static_cast<Tick>(node_count * 2 + 2)) == node_count,
           "long direct service preserves exact GPU busy union");
}

void test_long_direct_service_stays_bounded() {
    run_long_direct_service(200000, 1000,
                            "200k direct service retains no operator records");
    run_long_direct_service(1000000, 1000000,
                            "1M direct service retains no operator records");
}

void test_duplicate_direct_start_and_end_fail_closed() {
    expect_child_failure("duplicate direct GPU start fails closed", [] {
        Statistics stats(nullptr);
        stats.configure_online_history_preservation(false);
        const NodeView gpu = gpu_node(300);
        OnlineStatisticsState state;
        compact_start(stats, gpu, state, 0);
        compact_start(stats, gpu, state, 1);
    });
    expect_child_failure("duplicate direct GPU end fails closed", [] {
        Statistics stats(nullptr);
        stats.configure_online_history_preservation(false);
        const NodeView gpu = gpu_node(301);
        OnlineStatisticsState state;
        compact_start(stats, gpu, state, 0);
        compact_end(stats, gpu, state, 1);
        compact_end(stats, gpu, state, 2);
    });
}

void test_microbenchmark_history_is_unchanged() {
    Statistics stats(nullptr);
    stats.configure_online_history_preservation(true);
    const NodeView completed = gpu_node(400);
    const NodeView live = gpu_node(401);
    stats.record_start(completed, 10);
    stats.record_end(completed, 30);
    stats.record_start(live, 40);
    expect(stats.retained_online_operator_count() == 2,
           "history microbenchmark retains completed and live node records");
    expect(stats.calculate_type_time_in_window(
               Statistics::OperatorStatistics::OperatorType::GPU, 15, 25) ==
               10,
           "history microbenchmark retains arbitrary clipped GPU windows");
}

}  // namespace

int main() {
    test_live_gpu_fails_closed();
    test_direct_completed_gpu_union();
    test_compact_post_processing_fails_closed_without_mutation();
    test_direct_overlapping_gpu_union_and_roofline();
    test_cpu_no_util_and_comm_state_stay_compact();
    test_exact_sum_matches_history_bit_for_bit();
    test_long_direct_service_stays_bounded();
    test_duplicate_direct_start_and_end_fail_closed();
    test_microbenchmark_history_is_unchanged();
    const std::string system_config = write_minimal_system_config();
    test_history_preserving_legacy_post_processing_succeeds(system_config);
    test_compact_workload_report_fails_closed_without_mutation(system_config);
    expect(::unlink(system_config.c_str()) == 0,
           "remove temporary system config");
    return g_ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
