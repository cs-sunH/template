/******************************************************************************
metric_ledger_domain_test.cc -- MetricCollector ledger-domain regressions
(P3-3.2 batch C2, 2026-09-28).

Nine pins over the manifest-ledger side of the collector, each aligned
with the fail-closed semantics of the authoritative Python journal replay
(slo_tools/hbm_watermark.py) or with cross-record honesty:

  C1  prefill_ranks/decode_ranks must be JSON arrays: a scalar was
      iterated by nlohmann as a one-element set, silently shrinking the
      expected-rank domain (the request could then never complete) --
      now a load-time fatal (fork child, exit 1).
  C2  manifest request_id must be unique: the online registration path
      keys anchors/watch ranks by request_id last-wins, so a duplicate
      hung anchors on the wrong queue_index -- load-time fatal (fork).
  C3  memory_actions sequence_index strictly increasing (the journal's
      "sequence 回退" rule) -- load-time fatal (fork), never a silent
      re-sort into a clean-looking replay.
  C4  a negative intermediate ledger value (release without acquire) is
      corruption: the old direct integral clamped it to 0 and kept
      integrating while the Python replay fail-closes the same stream
      (exit 2) -- now finalize-time fatal (fork), never fabricated
      byte*ns areas.
  C5  anchored ticks that invert the ledger sequence order are reported
      as a consistency violation instead of being silently re-sorted
      (the replay itself still proceeds, so the totals stay checkable).
  C6  ledger rows targeting a rank outside the simulated systems are
      ghost rows across the four record families -- reported as
      consistency violations (the ghost capacity_timeavg row is still
      emitted, but no longer silently).
  C7  a fallback rank whose ledger rows are ALL unresolved reports its
      real actions_total/actions_unresolved in the flat capacity_timeavg
      record instead of hardcoded zeros that contradict the summary.
  C8  the init and summary records' same-named input_requests diverge
      whenever requests are dynamically registered between initialize()
      and finalize(); outside online mode (no registration path) that is
      a consistency violation, and the summary always echoes the init
      value as input_requests_at_init.
  C9  hbm_watermark bucket and summary records declare their sampling
      window (window_start_ns/window_end_ns = the finalize sim_end_tick),
      which is a DIFFERENT source than the WP6 link_bucket rows (the
      FluidScheduler observer's last tick).
  C10 (F1-CPP C1-4, 2026-09-29) a fresh dynamic registration re-claiming a
      request_id already bound to a different queue_index (manifest <-> CSV
      cross-source duplicate) fails closed via exit(1) instead of
      rebinding the identity last-wins.

Red -> green: C1-C4/C8/C9 assert behavior that did not exist pre-fix
(accepted input / silent clamp / missing fields); C5/C6/C7 assert
records/violations that were silently absent pre-fix.

Build: cmake target AstraSim_Analytical_Congestion_Aware_MetricLedgerDomainTest.
Run: build/astra_analytical/build_congestion_aware/bin/\
     AstraSim_Analytical_Congestion_Aware_MetricLedgerDomainTest
Exit code 0 on ALL PASS.
******************************************************************************/

#include "astra-sim/common/AstraNetworkAPI.hh"
#include "astra-sim/system/Sys.hh"
#include "astra-sim/workload/MetricCollector.hh"
#include "astra-sim/workload/execution_driven/NodeStore.hh"

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <functional>
#include <memory>
#include <string>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>

#include <json/json.hpp>

using namespace AstraSim;
using AstraSim::MetricCollector;
using json = nlohmann::json;

namespace {

bool g_ok = true;
int g_checks = 0;

void expect(bool condition, const char* message) {
    ++g_checks;
    if (!condition) {
        std::fprintf(stderr, "[metric_ledger_domain_test] FAIL: %s\n",
                     message);
        g_ok = false;
    }
}

std::string write_manifest(const json& manifest) {
    char pattern[] = "/tmp/astra-metric-ledger-domain-XXXXXX";
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
        std::fprintf(stderr, "write manifest failed: %s\n",
                     std::strerror(errno));
        ::close(fd);
        ::unlink(pattern);
        std::exit(EXIT_FAILURE);
    }
    if (::close(fd) != 0) {
        std::fprintf(stderr, "close manifest failed: %s\n",
                     std::strerror(errno));
        ::unlink(pattern);
        std::exit(EXIT_FAILURE);
    }
    return pattern;
}

json base_manifest() {
    json manifest;
    manifest["schema_version"] = 1;
    manifest["repo_variant"] = "metric-ledger-domain-fixture";
    manifest["run_mode"] = "service";
    manifest["run_id"] = "metric-ledger-domain-fixture";
    manifest["requests"] = json::array();
    manifest["node_events_by_rank"] = json::object();
    return manifest;
}

json make_request(int64_t queue_index, const char* request_id) {
    json request;
    request["queue_index"] = queue_index;
    request["request_id"] = request_id;
    request["session_id"] = "s0";
    request["turn_index"] = 0;
    request["prefill_ranks"] = json::array();
    request["decode_ranks"] = json::array();
    request["arrival"] = {{"kind", "absolute"}, {"value_ns", 0}};
    return request;
}

json ledger_action(int64_t sequence, const char* anchor_kind,
                   int64_t trigger, int rank, int64_t weight_delta) {
    return json{
        {"sequence_index", sequence},
        {"anchor_kind", anchor_kind},
        {"anchor_quality", "exact"},
        {"trigger_queue_index", trigger},
        {"rank", rank},
        {"weight_delta_bytes", weight_delta},
        {"resident_kv_delta_bytes", 0},
        {"reserved_kv_delta_bytes", 0},
        {"cause", "ledger-domain-fixture"},
    };
}

// ---------------------------------------------------------------------------
// Fork harness for the load-time/finalize-time fail-closed pins. The child
// redirects stderr to a probe file (message discrimination), runs its action
// and either dies through the collector's fatal channel (exit 1) or reports
// "accepted" with a POSIX _exit(0) (no duplicated-atexit noise).
// ---------------------------------------------------------------------------

void run_child_case(const json& manifest, bool run_finalize,
                    const char* expected_message, const char* label) {
    const std::string manifest_path = write_manifest(manifest);
    char err_pattern[] = "/tmp/astra-ledger-domain-err-XXXXXX";
    const int err_fd = ::mkstemp(err_pattern);
    if (err_fd < 0) {
        std::fprintf(stderr, "mkstemp failed: %s\n", std::strerror(errno));
        std::exit(EXIT_FAILURE);
    }
    ::close(err_fd);
    const pid_t pid = ::fork();
    if (pid < 0) {
        std::fprintf(stderr, "fork failed: %s\n", std::strerror(errno));
        std::exit(EXIT_FAILURE);
    }
    if (pid == 0) {
        // Child: stderr -> probe file, then the action. The fatal channel
        // inside the collector exits 1; reaching the end means accepted.
        if (std::freopen(err_pattern, "w", stderr) == nullptr) {
            ::_exit(2);
        }
        auto& collector = MetricCollector::instance();
        collector.initialize(manifest_path, "full");
        if (run_finalize) {
            collector.finalize({}, 60);
        }
        std::fflush(nullptr);
        ::_exit(0);
    }
    int status = 0;
    if (::waitpid(pid, &status, 0) < 0) {
        std::fprintf(stderr, "waitpid failed: %s\n", std::strerror(errno));
        std::exit(EXIT_FAILURE);
    }
    std::string exit_label = label;
    exit_label += " [exit code 1]";
    expect(WIFEXITED(status) && WEXITSTATUS(status) == 1,
           exit_label.c_str());
    // The message must name the corruption, not a generic failure.
    std::string err_text;
    if (FILE* f = std::fopen(err_pattern, "r")) {
        char buffer[4096];
        size_t read_bytes = 0;
        while ((read_bytes = std::fread(buffer, 1, sizeof(buffer), f)) > 0) {
            err_text.append(buffer, read_bytes);
        }
        std::fclose(f);
    }
    std::string message_label = label;
    message_label += " [fail-closed message]";
    expect(err_text.find(expected_message) != std::string::npos,
           message_label.c_str());
    ::unlink(err_pattern);
    ::unlink(manifest_path.c_str());
}

// F1-CPP C1-4 (2026-09-29): a FRESH dynamic registration whose request_id
// is already bound to a different queue_index is a cross-source identity
// collision (the CSV row re-claims an id the manifest or an earlier dynamic
// row owns). The fresh branch used to rebind request_index_by_request_id_
// last-wins, silently moving every anchor of that id to the wrong subject.
// Same fail-closed contract and channel as C2's manifest-internal
// duplicate gate (fork child, exit 1).
void run_cross_source_duplicate_id_case() {
    const json manifest = base_manifest();
    const std::string manifest_path = write_manifest(manifest);
    char err_pattern[] = "/tmp/astra-ledger-domain-err-XXXXXX";
    const int err_fd = ::mkstemp(err_pattern);
    if (err_fd < 0) {
        std::fprintf(stderr, "mkstemp failed: %s\n", std::strerror(errno));
        std::exit(EXIT_FAILURE);
    }
    ::close(err_fd);
    const pid_t pid = ::fork();
    if (pid < 0) {
        std::fprintf(stderr, "fork failed: %s\n", std::strerror(errno));
        std::exit(EXIT_FAILURE);
    }
    if (pid == 0) {
        if (std::freopen(err_pattern, "w", stderr) == nullptr) {
            ::_exit(2);
        }
        auto& collector = MetricCollector::instance();
        collector.initialize(manifest_path, "full");
        // First registration: fresh id, legitimately bound to queue 99.
        collector.online_register_request(99, "r99", "s99", 0, true, 0, -1,
                                          0);
        // Second registration at a DIFFERENT queue_index re-claims "r99":
        // pre-fix this rebound the identity last-wins and the child exited
        // 0; post-fix it is a load-time-style fatal (exit 1).
        collector.online_register_request(77, "r99", "s99", 0, true, 0, -1,
                                          0);
        std::fflush(nullptr);
        ::_exit(0);  // last-wins regression would land here
    }
    int status = 0;
    if (::waitpid(pid, &status, 0) < 0) {
        std::fprintf(stderr, "waitpid failed: %s\n", std::strerror(errno));
        std::exit(EXIT_FAILURE);
    }
    expect(WIFEXITED(status) && WEXITSTATUS(status) == 1,
           "C10: cross-source duplicate request_id fails closed via "
           "exit(1)");
    std::string err_text;
    if (FILE* f = std::fopen(err_pattern, "r")) {
        char buffer[4096];
        size_t read_bytes = 0;
        while ((read_bytes = std::fread(buffer, 1, sizeof(buffer), f)) > 0) {
            err_text.append(buffer, read_bytes);
        }
        std::fclose(f);
    }
    expect(err_text.find("duplicate request_id across sources") !=
               std::string::npos,
           "C10: the fail-closed message names the cross-source duplicate");
    ::unlink(err_pattern);
    ::unlink(manifest_path.c_str());
}

// ---------------------------------------------------------------------------
// In-process harness (records_of_type/capture_stdout, same recipe as
// metric_capacity_replay_test).
// ---------------------------------------------------------------------------

std::string capture_stdout(std::function<void()> action) {
    std::fflush(stdout);
    std::FILE* const capture = std::tmpfile();
    if (capture == nullptr) {
        std::fprintf(stderr, "tmpfile capture failed: %s\n",
                     std::strerror(errno));
        std::exit(EXIT_FAILURE);
    }
    const int saved_stdout = ::dup(STDOUT_FILENO);
    if (saved_stdout < 0 || ::dup2(::fileno(capture), STDOUT_FILENO) < 0) {
        std::fprintf(stderr, "stdout redirect failed: %s\n",
                     std::strerror(errno));
        std::exit(EXIT_FAILURE);
    }

    action();
    MetricCollector::instance().flush_emit_buffer();
    std::fflush(stdout);

    if (::dup2(saved_stdout, STDOUT_FILENO) < 0 ||
        ::close(saved_stdout) != 0) {
        std::fprintf(stderr, "stdout restore failed: %s\n",
                     std::strerror(errno));
        std::exit(EXIT_FAILURE);
    }
    if (std::fseek(capture, 0, SEEK_SET) != 0) {
        std::fprintf(stderr, "capture rewind failed: %s\n",
                     std::strerror(errno));
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
        std::fprintf(stderr, "capture read failed: %s\n",
                     std::strerror(errno));
        std::exit(EXIT_FAILURE);
    }
    if (std::fclose(capture) != 0) {
        std::fprintf(stderr, "capture close failed: %s\n",
                     std::strerror(errno));
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

bool violations_mention(const std::string& output, const char* needle) {
    for (const auto& record : records_of_type(output, "consistency")) {
        for (const auto& violation : record.value("violations",
                                                  json::array())) {
            if (violation.get<std::string>().find(needle) !=
                std::string::npos) {
                return true;
            }
        }
    }
    return false;
}

bool consistency_ok(const std::string& output) {
    const std::vector<json> records = records_of_type(output, "consistency");
    return !records.empty() && records.back().value("ok", true);
}

// A minimal constructed Sys (the custom_only_system_construct recipe) so a
// case can pass a non-empty simulated rank set to finalize.
class FakeNetworkApi : public AstraNetworkAPI {
  public:
    FakeNetworkApi() : AstraNetworkAPI(0) {}

    int sim_send(void*, uint64_t, int, int, int, sim_request*,
                 void (*)(void*), void*) override {
        return 0;
    }

    double get_BW_at_dimension(int /*dim*/) override { return 1.0; }

    int sim_recv(void*, uint64_t, int, int, int, sim_request*,
                 void (*)(void*), void*) override {
        return 0;
    }

    void sim_schedule(AstraSim::timespec_t, void (*)(void*), void*) override {}

    AstraSim::timespec_t sim_get_time() override {
        return {AstraSim::NS, 0};
    }
};

Sys* make_rank_zero_sys() {
    char pattern[] = "/tmp/astra-ledger-domain-sys-XXXXXX";
    const int fd = ::mkstemp(pattern);
    if (fd < 0) {
        std::fprintf(stderr, "mkstemp failed: %s\n", std::strerror(errno));
        std::exit(EXIT_FAILURE);
    }
    const char config[] =
        "{\n"
        "  \"scheduling-policy\": \"FIFO\",\n"
        "  \"preferred-dataset-splits\": 1,\n"
        "  \"all-reduce-implementation\": [\"ring\"],\n"
        "  \"local-mem-bw\": 50\n"
        "}\n";
    if (::write(fd, config, sizeof(config) - 1) <= 0 ||
        ::close(fd) != 0) {
        std::fprintf(stderr, "write system config failed\n");
        std::exit(EXIT_FAILURE);
    }
    auto* net = new FakeNetworkApi();
    auto source =
        std::make_shared<AstraSim::ExecutionDriven::NodeStoreGraphSource>();
    Sys* sys = new Sys(0, "workload-unused", "empty", pattern, net,
                       std::vector<int>{1}, std::vector<int>{1}, 1.0, 1.0,
                       false,
                       AstraSim::ExecutionDriven::ExecutionMode::Online,
                       source);
    ::unlink(pattern);
    return sys;
}

// C5: anchored ticks invert the ledger sequence order -> reported, replay
// still proceeds and the totals stay checkable.
void run_anchor_inversion_case() {
    json manifest = base_manifest();
    manifest["requests"] = json::array({make_request(10, "r10"),
                                        make_request(11, "r11")});
    manifest["node_events_by_rank"] = json::object({
        {"0", json::array({json::array({100, 4, 10}),
                           json::array({200, 4, 11})})},
    });
    manifest["memory_actions"] = json::array({
        ledger_action(1, "completion", 10, 1, 10),  // resolves to tick 100
        ledger_action(2, "completion", 11, 1, 5),   // resolves to tick 50
    });
    manifest["planner_memory_peaks"] = json::array({
        json{{"rank", 1},
             {"ledger", json{{"capacity_bytes", 1000},
                             {"physical_used_bytes", 15},
                             {"committed_used_bytes", 15}}}},
    });
    const std::string manifest_path = write_manifest(manifest);
    MetricCollector& collector = MetricCollector::instance();
    const std::string output = capture_stdout([&] {
        collector.initialize(manifest_path, "full");
        collector.on_node_complete(0, 200, 50);  // q11 completes at 50
        collector.on_node_complete(0, 100, 100);  // q10 completes at 100
        collector.finalize({}, 200);
    });
    expect(!consistency_ok(output),
           "C5: anchored-tick inversion of the sequence order is a "
           "consistency violation (not a silent re-sort)");
    expect(violations_mention(output, "re-sort"),
           "C5: the violation names the re-sort and the offending pair");
    // The replay itself still proceeded, sorted by (tick, sequence).
    const std::vector<json> capacity =
        records_of_type(output, "capacity_timeavg");
    const json* const rank_one = record_for_rank(capacity, 1);
    expect(rank_one != nullptr, "C5: capacity_timeavg still emitted");
    if (rank_one != nullptr) {
        expect(rank_one->value("resident_byte_ns", std::string()) == "1750",
               "C5: replay totals unchanged by the report (5 B x [50,100) + "
               "15 B x [100,200) = 1750)");
        expect(rank_one->value("final_state_matches_planner", false),
               "C5: final ledger still matches the planner");
    }
    ::unlink(manifest_path.c_str());
}

// C6: ledger rows targeting a rank outside the simulated systems.
void run_ghost_rank_case(Sys* sys) {
    json manifest = base_manifest();
    manifest["requests"] = json::array({make_request(10, "r10")});
    manifest["memory_actions"] = json::array({
        ledger_action(1, "tick_zero", -1, 5, 7),  // rank 5: not simulated
    });
    manifest["planner_memory_peaks"] = json::array({
        json{{"rank", 5},
             {"ledger", json{{"capacity_bytes", 100},
                             {"physical_used_bytes", 7},
                             {"committed_used_bytes", 7}}}},
    });
    const std::string manifest_path = write_manifest(manifest);
    MetricCollector& collector = MetricCollector::instance();
    const std::string output = capture_stdout([&] {
        collector.initialize(manifest_path, "full");
        collector.finalize({sys}, 60);
    });
    expect(!consistency_ok(output),
           "C6: ghost ledger rank outside the simulated systems is a "
           "consistency violation");
    expect(violations_mention(output, "not a simulated rank"),
           "C6: both the action row and the peaks row are named");
    const std::vector<json> capacity =
        records_of_type(output, "capacity_timeavg");
    expect(record_for_rank(capacity, 5) != nullptr,
           "C6: the ghost capacity_timeavg row is still emitted (reported, "
           "not dropped)");
    ::unlink(manifest_path.c_str());
}

// C7: a simulated rank whose ledger rows are ALL unresolved lands in the
// fallback branch and must report its real per-rank totals.
void run_fallback_totals_case(Sys* sys) {
    json manifest = base_manifest();
    manifest["requests"] = json::array({make_request(10, "r10")});
    manifest["memory_actions"] = json::array({
        ledger_action(1, "completion", 99, 0, 7),  // trigger 99: unknown
    });
    manifest["planner_memory_peaks"] = json::array({
        json{{"rank", 0},
             {"ledger", json{{"capacity_bytes", 500},
                             {"physical_used_bytes", 0},
                             {"committed_used_bytes", 0}}}},
    });
    const std::string manifest_path = write_manifest(manifest);
    MetricCollector& collector = MetricCollector::instance();
    const std::string output = capture_stdout([&] {
        collector.initialize(manifest_path, "full");
        collector.finalize({sys}, 60);
    });
    const std::vector<json> capacity =
        records_of_type(output, "capacity_timeavg");
    const json* const rank_zero = record_for_rank(capacity, 0);
    expect(rank_zero != nullptr,
           "C7: flat capacity_timeavg emitted for the simulated rank");
    if (rank_zero != nullptr) {
        expect(rank_zero->value("actions_total", uint64_t(0)) == 1,
               "C7: fallback record actions_total is the rank's real total "
               "(1), not a hardcoded 0");
        expect(rank_zero->value("actions_unresolved", uint64_t(0)) == 1,
               "C7: fallback record actions_unresolved matches the summary "
               "(1), not a hardcoded 0");
    }
    expect(consistency_ok(output),
           "C7: unresolved anchors alone do not flag the run inconsistent");
    ::unlink(manifest_path.c_str());
}

// C8: requests registered between initialize() and finalize() outside
// online mode -> divergence is a violation; the summary echoes the init
// count either way (here: reported, because online mode was never latched).
void run_input_requests_case() {
    json manifest = base_manifest();
    manifest["requests"] = json::array({make_request(10, "r10")});
    const std::string manifest_path = write_manifest(manifest);
    MetricCollector& collector = MetricCollector::instance();
    const std::string output = capture_stdout([&] {
        collector.initialize(manifest_path, "full");
        collector.online_register_request(99, "r99", "s99", 0, true, 0, -1,
                                          0);
        collector.finalize({}, 60);
    });
    const std::vector<json> summaries = records_of_type(output, "summary");
    expect(summaries.size() == 1, "C8: one summary record");
    if (!summaries.empty()) {
        expect(summaries.front().value("input_requests_at_init", uint64_t(0)) ==
                   1,
               "C8: summary echoes the init record's input_requests");
    }
    expect(!consistency_ok(output),
           "C8: init/summary input_requests divergence outside online mode "
           "is a consistency violation");
    expect(violations_mention(output, "input_requests"),
           "C8: the violation names both counts");
    ::unlink(manifest_path.c_str());
}

// C9: watermark records declare their sampling window.
void run_window_fields_case() {
    json manifest = base_manifest();
    manifest["memory_actions"] = json::array({
        ledger_action(1, "tick_zero", -1, 1, 5),
    });
    manifest["planner_memory_peaks"] = json::array({
        json{{"rank", 1},
             {"ledger", json{{"capacity_bytes", 100},
                             {"physical_used_bytes", 5},
                             {"committed_used_bytes", 5}}}},
    });
    const std::string manifest_path = write_manifest(manifest);
    MetricCollector& collector = MetricCollector::instance();
    const std::string output = capture_stdout([&] {
        collector.initialize(manifest_path, "full");
        collector.finalize({}, 60);
    });
    const std::vector<json> buckets =
        records_of_type(output, "hbm_watermark");
    expect(!buckets.empty(), "C9: bucket record emitted");
    for (const auto& bucket : buckets) {
        expect(bucket.value("window_start_ns", uint64_t(1)) == 0 &&
                   bucket.value("window_end_ns", uint64_t(0)) == 60,
               "C9: hbm_watermark rows declare [0, sim_end_tick)");
    }
    const std::vector<json> summaries =
        records_of_type(output, "hbm_watermark_summary");
    expect(!summaries.empty(), "C9: summary record emitted");
    for (const auto& summary : summaries) {
        expect(summary.value("window_start_ns", uint64_t(1)) == 0 &&
                   summary.value("window_end_ns", uint64_t(0)) == 60,
               "C9: hbm_watermark_summary rows declare the same window");
    }
    expect(consistency_ok(output), "C9: fixture run stays consistent");
    ::unlink(manifest_path.c_str());
}

}  // namespace

int main() {
    // -- fork cases (fail-closed channels) --------------------------------
    json scalar_ranks = base_manifest();
    json scalar_request = make_request(10, "r10");
    scalar_request["prefill_ranks"] = 3;  // scalar, not an array
    scalar_ranks["requests"] = json::array({scalar_request});
    run_child_case(scalar_ranks, /*run_finalize=*/false,
                   "prefill_ranks must be an array",
                   "C1: scalar prefill_ranks fails closed via exit(1) with "
                   "the array-type message");

    json scalar_decode = base_manifest();
    json scalar_decode_request = make_request(10, "r10");
    scalar_decode_request["decode_ranks"] = 4;  // scalar, not an array
    scalar_decode["requests"] = json::array({scalar_decode_request});
    run_child_case(scalar_decode, /*run_finalize=*/false,
                   "decode_ranks must be an array",
                   "C1b: scalar decode_ranks fails closed via exit(1)");

    json dup_ids = base_manifest();
    dup_ids["requests"] = json::array({make_request(10, "dup"),
                                       make_request(11, "dup")});
    run_child_case(dup_ids, /*run_finalize=*/false,
                   "duplicate request_id",
                   "C2: duplicate manifest request_id fails closed via "
                   "exit(1)");

    json bad_sequence = base_manifest();
    bad_sequence["memory_actions"] = json::array({
        ledger_action(2, "tick_zero", -1, 1, 5),
        ledger_action(1, "tick_zero", -1, 1, 5),  // sequence regression
    });
    run_child_case(bad_sequence, /*run_finalize=*/false,
                   "not strictly increasing",
                   "C3: memory_actions sequence regression fails closed via "
                   "exit(1)");

    json negative_ledger = base_manifest();
    negative_ledger["memory_actions"] = json::array({
        ledger_action(1, "tick_zero", -1, 1, -50),  // release w/o acquire
    });
    run_child_case(negative_ledger, /*run_finalize=*/true,
                   "went negative",
                   "C4: negative intermediate ledger value fails closed via "
                   "exit(1), never clamped-and-integrated");

    run_cross_source_duplicate_id_case();

    // -- in-process cases -------------------------------------------------
    // One shared minimal Sys (id 0) for the cases that need a simulated
    // rank universe; deliberately not joined (same lifecycle as the
    // custom_only_system_construct fixture).
    Sys* shared_sys = make_rank_zero_sys();
    run_anchor_inversion_case();
    run_ghost_rank_case(shared_sys);
    run_fallback_totals_case(shared_sys);
    run_input_requests_case();
    run_window_fields_case();

    std::fprintf(stderr, "[metric_ledger_domain_test] %d checks: %s\n",
                 g_checks, g_ok ? "PASS" : "FAIL");
    return g_ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
