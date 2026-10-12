/******************************************************************************
MetricCollector online memory-anchor spool regression fixture.

The static collector historically retained every code-7 completion in
memory_anchor_ticks_.  Online service mode must emit the exact same raw records
without retaining a terminal-node-sized vector: raw records are spooled in
arrival order and the ledger resolver retains only the action-referenced
(subject, rank) / subject maxima.
*******************************************************************************/

#include "astra-sim/workload/MetricCollector.hh"

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <string>
#include <sys/resource.h>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>

#include <json/json.hpp>

using AstraSim::MetricCollector;
using AstraSim::Tick;
using json = nlohmann::json;

namespace AstraSim {

// The production API deliberately exposes no storage implementation details.
// This friend is solely the regression fixture's proof that online raw-anchor
// memory is bounded by action cardinality rather than terminal-node count.
struct MetricCollectorTestAccess {
    static void record_memory_anchor(MetricCollector& collector,
                                     int64_t subject_id,
                                     int rank,
                                     uint64_t node_id,
                                     Tick tick) {
        collector.apply_event(
            MetricCollector::NodeMetricEvent{7, subject_id}, rank, node_id,
            tick);
    }

    static size_t retained_raw_anchor_count(const MetricCollector& collector) {
        return collector.memory_anchor_ticks_.size();
    }

    static bool spool_open(const MetricCollector& collector) {
        return collector.memory_anchor_spool_ != nullptr;
    }

    static uint64_t spool_record_count(const MetricCollector& collector) {
        return collector.memory_anchor_spool_record_count_;
    }

    static size_t action_key_count(const MetricCollector& collector) {
        return collector.online_transfer_anchor_keys_needed_.size();
    }

    static size_t action_subject_count(const MetricCollector& collector) {
        return collector.online_transfer_anchor_subjects_needed_.size();
    }

    static size_t indexed_rank_max_count(const MetricCollector& collector) {
        return collector.online_transfer_anchor_ticks_.size();
    }

    static size_t indexed_subject_max_count(const MetricCollector& collector) {
        return collector.online_transfer_anchor_by_subject_.size();
    }

    static bool finalized(const MetricCollector& collector) {
        return collector.finalized_;
    }
};

}  // namespace AstraSim

namespace {

bool g_ok = true;

void expect(bool condition, const char* message) {
    if (!condition) {
        std::fprintf(stderr,
                     "[metric_collector_anchor_spool_test] FAIL: %s\n",
                     message);
        g_ok = false;
    }
}

std::string write_manifest(const json& manifest) {
    char pattern[] = "/tmp/astra-metric-anchor-XXXXXX";
    const int fd = ::mkstemp(pattern);
    if (fd < 0) {
        std::fprintf(stderr, "mkstemp failed: %s\n", std::strerror(errno));
        std::exit(EXIT_FAILURE);
    }
    const std::string contents = manifest.dump();
    size_t offset = 0;
    while (offset < contents.size()) {
        const ssize_t written = ::write(
            fd, contents.data() + offset, contents.size() - offset);
        if (written > 0) {
            offset += static_cast<size_t>(written);
            continue;
        }
        if (written < 0 && errno == EINTR) {
            continue;
        }
        std::fprintf(stderr, "write manifest failed: %s\n", std::strerror(errno));
        ::close(fd);
        ::unlink(pattern);
        std::exit(EXIT_FAILURE);
    }
    if (::close(fd) != 0) {
        std::fprintf(stderr, "close manifest failed: %s\n", std::strerror(errno));
        ::unlink(pattern);
        std::exit(EXIT_FAILURE);
    }
    return pattern;
}

json make_manifest(bool with_transfer_actions) {
    json request;
    request["queue_index"] = 10;
    request["request_id"] = "r10";
    request["session_id"] = "s0";
    request["turn_index"] = 0;
    request["prefill_ranks"] = json::array();
    request["decode_ranks"] = json::array();
    request["arrival"] = {{"kind", "absolute"}, {"value_ns", 0}};

    json manifest;
    manifest["schema_version"] = 1;
    manifest["repo_variant"] = "metric-anchor-spool-fixture";
    manifest["run_mode"] = "service";
    manifest["run_id"] = "metric-anchor-spool-fixture";
    manifest["requests"] = json::array({request});
    manifest["node_events_by_rank"] = json::object();
    manifest["planner_memory_peaks"] = json::array();
    manifest["memory_actions"] = json::array();
    if (with_transfer_actions) {
        manifest["memory_actions"].push_back({
            {"sequence_index", 1},
            {"anchor_kind", "transfer_complete"},
            {"anchor_quality", "exact"},
            {"trigger_queue_index", 10},
            {"rank", 1},
            {"weight_delta_bytes", 10},
            {"resident_kv_delta_bytes", 0},
            {"reserved_kv_delta_bytes", 0},
            {"cause", "rank-max"},
        });
        manifest["memory_actions"].push_back({
            {"sequence_index", 2},
            {"anchor_kind", "transfer_complete"},
            {"anchor_quality", "fallback"},
            {"trigger_queue_index", 10},
            {"rank", 3},
            {"weight_delta_bytes", 0},
            {"resident_kv_delta_bytes", 20},
            {"reserved_kv_delta_bytes", 0},
            {"cause", "subject-max-fallback"},
        });
    }
    return manifest;
}

template <typename Action>
std::string capture_stdout(Action&& action) {
    std::fflush(stdout);
    std::FILE* const capture = std::tmpfile();
    if (capture == nullptr) {
        std::fprintf(stderr, "tmpfile capture failed: %s\n", std::strerror(errno));
        std::exit(EXIT_FAILURE);
    }
    const int saved_stdout = ::dup(STDOUT_FILENO);
    if (saved_stdout < 0 || ::dup2(::fileno(capture), STDOUT_FILENO) < 0) {
        std::fprintf(stderr, "stdout redirect failed: %s\n", std::strerror(errno));
        std::exit(EXIT_FAILURE);
    }

    action();
    MetricCollector::instance().flush_emit_buffer();
    std::fflush(stdout);

    if (::dup2(saved_stdout, STDOUT_FILENO) < 0 || ::close(saved_stdout) != 0) {
        std::fprintf(stderr, "stdout restore failed: %s\n", std::strerror(errno));
        std::exit(EXIT_FAILURE);
    }
    if (std::fseek(capture, 0, SEEK_SET) != 0) {
        std::fprintf(stderr, "capture rewind failed: %s\n", std::strerror(errno));
        std::exit(EXIT_FAILURE);
    }
    std::string output;
    char buffer[4096];
    for (;;) {
        const size_t read = std::fread(buffer, 1, sizeof(buffer), capture);
        output.append(buffer, read);
        if (read == sizeof(buffer)) {
            continue;
        }
        if (std::feof(capture) != 0) {
            break;
        }
        std::fprintf(stderr, "capture read failed: %s\n", std::strerror(errno));
        std::exit(EXIT_FAILURE);
    }
    if (std::fclose(capture) != 0) {
        std::fprintf(stderr, "capture close failed: %s\n", std::strerror(errno));
        std::exit(EXIT_FAILURE);
    }
    return output;
}

std::vector<json> records_of_type(const std::string& output,
                                  const std::string& type) {
    std::vector<json> records;
    size_t cursor = 0;
    while (cursor < output.size()) {
        const size_t newline = output.find('\n', cursor);
        const std::string line = output.substr(
            cursor, newline == std::string::npos ? std::string::npos
                                                  : newline - cursor);
        constexpr const char kPrefix[] = "[METRIC] ";
        if (line.compare(0, sizeof(kPrefix) - 1, kPrefix) == 0) {
            const json record = json::parse(line.substr(sizeof(kPrefix) - 1));
            if (record.value("type", std::string()) == type) {
                records.push_back(record);
            }
        }
        if (newline == std::string::npos) {
            break;
        }
        cursor = newline + 1;
    }
    return records;
}

const json* record_for_rank(const std::vector<json>& records, int rank) {
    for (const auto& record : records) {
        if (record.value("rank", -1) == rank) {
            return &record;
        }
    }
    return nullptr;
}

void record_four_anchors(MetricCollector& collector) {
    // Arrival order is intentionally not tick order.  This exercises exact
    // raw record ordering and both duplicate-key maxima independently.
    AstraSim::MetricCollectorTestAccess::record_memory_anchor(
        collector, 10, 1, 100, 5);
    AstraSim::MetricCollectorTestAccess::record_memory_anchor(
        collector, 10, 2, 101, 22);
    AstraSim::MetricCollectorTestAccess::record_memory_anchor(
        collector, 10, 1, 102, 12);
    AstraSim::MetricCollectorTestAccess::record_memory_anchor(
        collector, 10, 2, 103, 7);
}

void expect_anchor_payload(const std::vector<json>& records) {
    expect(records.size() == 4,
           "memory_anchor count preserves every raw completion");
    if (records.size() != 4) {
        return;
    }
    const uint64_t nodes[] = {100, 101, 102, 103};
    const int ranks[] = {1, 2, 1, 2};
    const Tick ticks[] = {5, 22, 12, 7};
    for (size_t i = 0; i < records.size(); ++i) {
        expect(records[i].value("subject_id", int64_t(-1)) == 10,
               "memory_anchor subject field preserved");
        expect(records[i].value("rank", -1) == ranks[i],
               "memory_anchor original event order preserved");
        expect(records[i].value("node_id", uint64_t(0)) == nodes[i],
               "memory_anchor node field preserved");
        expect(records[i].value("tick_ns", Tick(-1)) == ticks[i],
               "memory_anchor tick field preserved");
    }
}

void expect_transfer_replay(const std::vector<json>& records) {
    const json* const rank_one = record_for_rank(records, 1);
    const json* const rank_three = record_for_rank(records, 3);
    expect(rank_one != nullptr, "rank-max transfer action was replayed");
    expect(rank_three != nullptr,
           "subject-max fallback transfer action was replayed");
    if (rank_one != nullptr) {
        expect(rank_one->value("resident_byte_ns", std::string()) == "180",
               "(subject,rank) duplicate anchors use their maximum tick");
        expect(rank_one->value("transfer_anchor_request_level_fallback", 99) ==
                   0,
               "exact rank transfer anchor does not count fallback");
    }
    if (rank_three != nullptr) {
        expect(rank_three->value("resident_byte_ns", std::string()) == "160",
               "missing rank falls back to the subject maximum tick");
        expect(rank_three->value("transfer_anchor_request_level_fallback", 99) ==
                   1,
               "subject-max fallback counter remains exact");
    }
}

void test_static_online_equivalence_and_maxima() {
    const std::string manifest_path = write_manifest(make_manifest(true));
    MetricCollector& collector = MetricCollector::instance();

    const std::string static_output = capture_stdout([&] {
        collector.initialize(manifest_path, "full");
        record_four_anchors(collector);
        expect(AstraSim::MetricCollectorTestAccess::retained_raw_anchor_count(
                   collector) == 4,
               "static path retains the legacy raw-anchor vector");
        expect(!AstraSim::MetricCollectorTestAccess::spool_open(collector),
               "static path does not create an online spool");
        collector.finalize({}, 30);
        collector.finalize({}, 30);  // repeated finalize must be idempotent
    });
    const std::vector<json> static_anchors =
        records_of_type(static_output, "memory_anchor");
    const std::vector<json> static_capacity =
        records_of_type(static_output, "capacity_timeavg");
    expect_anchor_payload(static_anchors);
    expect_transfer_replay(static_capacity);

    const std::string online_output = capture_stdout([&] {
        collector.initialize(manifest_path, "full");
        collector.clear_static_node_events();
        collector.online_register_request(10, "r10", "s0", 0, true, 0, -1,
                                          0);
        record_four_anchors(collector);
        expect(AstraSim::MetricCollectorTestAccess::retained_raw_anchor_count(
                   collector) == 0,
               "online raw anchors are not retained in a vector");
        expect(AstraSim::MetricCollectorTestAccess::spool_open(collector),
               "online raw anchors use an anonymous spool");
        expect(AstraSim::MetricCollectorTestAccess::spool_record_count(
                   collector) == 4,
               "online spool preserves raw anchor multiplicity");
        expect(AstraSim::MetricCollectorTestAccess::action_key_count(
                   collector) == 2,
               "online rank-max index is bounded by transfer actions");
        expect(AstraSim::MetricCollectorTestAccess::action_subject_count(
                   collector) == 1,
               "online subject-max index is bounded by transfer actions");
        expect(AstraSim::MetricCollectorTestAccess::indexed_rank_max_count(
                   collector) == 1,
               "only action-referenced rank maxima are retained");
        expect(AstraSim::MetricCollectorTestAccess::indexed_subject_max_count(
                   collector) == 1,
               "only action-referenced subject maxima are retained");
        collector.finalize({}, 30);
        expect(!AstraSim::MetricCollectorTestAccess::spool_open(collector),
               "finalize closes the online spool exactly once");
        expect(AstraSim::MetricCollectorTestAccess::finalized(collector),
               "finalize records its idempotent lifecycle state");
        collector.finalize({}, 30);
    });
    const std::vector<json> online_anchors =
        records_of_type(online_output, "memory_anchor");
    const std::vector<json> online_capacity =
        records_of_type(online_output, "capacity_timeavg");
    expect_anchor_payload(online_anchors);
    expect_transfer_replay(online_capacity);
    expect(static_anchors == online_anchors,
           "static and online memory_anchor JSON records are byte-schema equivalent");
    expect(static_capacity == online_capacity,
           "static and online transfer replay records are exactly equivalent");

    ::unlink(manifest_path.c_str());
}

void test_long_timeline_storage_bound_and_reinitialize() {
    const std::string manifest_path = write_manifest(make_manifest(false));
    MetricCollector& collector = MetricCollector::instance();
    constexpr uint64_t kAnchorCount = 200000;

    (void)capture_stdout([&] {
        collector.initialize(manifest_path, "summary");
        collector.clear_static_node_events();
        for (uint64_t i = 1; i <= kAnchorCount; ++i) {
            AstraSim::MetricCollectorTestAccess::record_memory_anchor(
                collector, 999, static_cast<int>(i % 8), i, i);
            if ((i & 32767u) == 0) {
                expect(AstraSim::MetricCollectorTestAccess::retained_raw_anchor_count(
                           collector) == 0,
                       "long online run retains no raw anchor vector entries");
                expect(AstraSim::MetricCollectorTestAccess::spool_record_count(
                           collector) == i,
                       "long online run stores every raw record only in spool");
                expect(AstraSim::MetricCollectorTestAccess::indexed_rank_max_count(
                           collector) == 0,
                       "no transfer action means no rank-max memory growth");
                expect(AstraSim::MetricCollectorTestAccess::indexed_subject_max_count(
                           collector) == 0,
                       "no transfer action means no subject-max memory growth");
            }
        }
        collector.finalize({}, kAnchorCount + 1);
        expect(!AstraSim::MetricCollectorTestAccess::spool_open(collector),
               "summary finalize closes long-run spool without replaying raw rows");
        collector.finalize({}, kAnchorCount + 1);
    });

    // A new initialize must close/reset all prior lifecycle state before the
    // next online transition.  This catches stale spool pointers and stale
    // record counters across reused singleton test runs.
    (void)capture_stdout([&] {
        collector.initialize(manifest_path, "summary");
        expect(AstraSim::MetricCollectorTestAccess::spool_record_count(
                   collector) == 0,
               "reinitialize clears old spool record count");
        expect(!AstraSim::MetricCollectorTestAccess::finalized(collector),
               "reinitialize clears finalization state");
        collector.clear_static_node_events();
        expect(AstraSim::MetricCollectorTestAccess::spool_open(collector),
               "reinitialize opens a fresh online spool");
        collector.finalize({}, 1);
    });

    ::unlink(manifest_path.c_str());
}

int highest_open_fd() {
    int highest = 2;
    for (int fd = 3; fd < 1024; ++fd) {
        if (::fcntl(fd, F_GETFD) != -1 || errno != EBADF) {
            highest = fd;
        }
    }
    return highest;
}

void test_tmpfile_failure_is_fail_closed() {
    const std::string manifest_path = write_manifest(make_manifest(false));
    const pid_t child = ::fork();
    expect(child >= 0, "fork tmpfile-failure child");
    if (child == 0) {
        MetricCollector& collector = MetricCollector::instance();
        collector.initialize(manifest_path, "summary");

        struct rlimit limit {};
        if (::getrlimit(RLIMIT_NOFILE, &limit) != 0) {
            _exit(77);
        }
        const int highest = highest_open_fd();
        const rlim_t desired = static_cast<rlim_t>(highest) + 1;
        if (desired > limit.rlim_max) {
            _exit(77);
        }
        limit.rlim_cur = desired;
        if (::setrlimit(RLIMIT_NOFILE, &limit) != 0) {
            _exit(77);
        }
        while (::open("/dev/null", O_RDONLY) >= 0) {
        }
        if (errno != EMFILE && errno != ENFILE) {
            _exit(78);
        }
        // clear_static_node_events must try tmpfile and terminate rather than
        // silently retain an unbounded vector when no descriptor is available.
        collector.clear_static_node_events();
        _exit(EXIT_SUCCESS);
    }
    if (child >= 0) {
        int status = 0;
        expect(::waitpid(child, &status, 0) == child,
               "wait for tmpfile-failure child");
        expect(WIFEXITED(status), "tmpfile-failure child exits normally");
        if (WIFEXITED(status)) {
            const int code = WEXITSTATUS(status);
            expect(code != EXIT_SUCCESS && code != 77 && code != 78,
                   "tmpfile allocation failure is fail-closed, not a setup skip");
        }
    }
    ::unlink(manifest_path.c_str());
}

}  // namespace

int main() {
    test_static_online_equivalence_and_maxima();
    test_long_timeline_storage_bound_and_reinitialize();
    test_tmpfile_failure_is_fail_closed();
    return g_ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
