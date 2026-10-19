/******************************************************************************
metric_capacity_replay_test.cc -- same-tick capacity-replay sampling
regression (P2, 2026-09-28).

The WP8 watermark walk must sample the ledger step function at EVERY
ledger delta, including same-tick delta groups (each delta's post-value
sampled individually) and deltas anchored exactly at the window end.
The pre-fix walk folded a same-tick group into one post-group sample, so
an intermediate transient that exceeds the planner HBM capacity inside
one tick was invisible to the bucket peaks and to capacity_violations --
while the Python journal replay (slo_tools/hbm_watermark.py, consume())
checks physical_after > capacity per journal row and DID flag it.  The
same ledger could therefore be certified clean by C++ and dirty by
Python.

Fixture (rank 1, capacity 100 B, window [0, 60)):
  seq 1  tick_zero  resident_kv +110   -> post-delta committed 110 > 100
  seq 2  tick_zero  resident_kv -20    -> post-group value 90 <= 100
  seq 3  transfer_complete @t=60 (sim end) weight +30
             -> end-of-window instant resident/committed 120 > 100
Planner final ledger 120/120 matches the replay, so the only failing
checks before the fix are the peaks and the violation count.

Expected (red -> green):
  pre-fix : committed_peak == 90 (group-folded), 0 violations, and the
            end-of-window delta is never sampled at all;
  post-fix: committed_peak == 120, capacity_violation_count == 2.

Build: cmake target AstraSim_Analytical_Congestion_Aware_MetricCapacityReplayTest.
Run: build/astra_analytical/build_congestion_aware/bin/\
     AstraSim_Analytical_Congestion_Aware_MetricCapacityReplayTest
Exit code 0 on ALL PASS.
******************************************************************************/

#include "astra-sim/workload/MetricCollector.hh"

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <string>
#include <unistd.h>
#include <vector>

#include <json/json.hpp>

using AstraSim::MetricCollector;
using json = nlohmann::json;

namespace {

bool g_ok = true;
int g_checks = 0;

void expect(bool condition, const char* message) {
    ++g_checks;
    if (!condition) {
        std::fprintf(stderr, "[metric_capacity_replay_test] FAIL: %s\n",
                     message);
        g_ok = false;
    }
}

std::string write_manifest(const json& manifest) {
    char pattern[] = "/tmp/astra-metric-capacity-replay-XXXXXX";
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

json make_manifest() {
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
    manifest["repo_variant"] = "metric-capacity-replay-fixture";
    manifest["run_mode"] = "service";
    manifest["run_id"] = "metric-capacity-replay-fixture";
    manifest["requests"] = json::array({request});
    manifest["node_events_by_rank"] = json::object({
        {"1", json::array({json::array({300, 7, 10})})},
    });
    manifest["planner_memory_peaks"] = json::array({
        json{
            {"rank", 1},
            {"ledger", json{
                {"capacity_bytes", 100},
                {"physical_used_bytes", 120},
                {"committed_used_bytes", 120},
            }},
        },
    });
    // Same-tick group (seq 1/2 anchored at tick 0) whose intermediate
    // value exceeds capacity, plus a delta anchored exactly at the window
    // end (the transfer anchor fires with the last code-7 complete).
    manifest["memory_actions"] = json::array({
        json{
            {"sequence_index", 1},
            {"anchor_kind", "tick_zero"},
            {"anchor_quality", "exact"},
            {"trigger_queue_index", -1},
            {"rank", 1},
            {"weight_delta_bytes", 0},
            {"resident_kv_delta_bytes", 110},
            {"reserved_kv_delta_bytes", 0},
            {"cause", "same-tick-transient"},
        },
        json{
            {"sequence_index", 2},
            {"anchor_kind", "tick_zero"},
            {"anchor_quality", "exact"},
            {"trigger_queue_index", -1},
            {"rank", 1},
            {"weight_delta_bytes", 0},
            {"resident_kv_delta_bytes", -20},
            {"reserved_kv_delta_bytes", 0},
            {"cause", "same-tick-release"},
        },
        json{
            {"sequence_index", 3},
            {"anchor_kind", "transfer_complete"},
            {"anchor_quality", "exact"},
            {"trigger_queue_index", 10},
            {"rank", 1},
            {"weight_delta_bytes", 30},
            {"resident_kv_delta_bytes", 0},
            {"reserved_kv_delta_bytes", 0},
            {"cause", "window-end-delta"},
        },
    });
    return manifest;
}

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

void run_same_tick_transient_case() {
    const std::string manifest_path = write_manifest(make_manifest());
    MetricCollector& collector = MetricCollector::instance();

    const std::string output = capture_stdout([&] {
        collector.initialize(manifest_path, "full");
        // The end-of-window weight delta anchors on the code-7 complete
        // at tick 60 == sim_end_tick.
        collector.on_node_complete(1, 300, 60);
        collector.finalize({}, 60);
    });

    // The direct capacity_timeavg integral is per-delta already and must
    // be unchanged by the fix: 90 resident bytes over [0, 60).
    const std::vector<json> capacity =
        records_of_type(output, "capacity_timeavg");
    const json* const rank_one = record_for_rank(capacity, 1);
    expect(rank_one != nullptr, "capacity_timeavg record emitted for rank 1");
    if (rank_one != nullptr) {
        expect(rank_one->value("resident_byte_ns", std::string()) == "5400",
               "direct integral unchanged (90 B x 60 ns = 5400)");
        expect(rank_one->value("final_state_matches_planner", false),
               "replay final ledger matches the planner ledger");
    }

    // The WP8 walk must have sampled every delta's post-value.
    const std::vector<json> summaries =
        records_of_type(output, "hbm_watermark_summary");
    const json* const summary = record_for_rank(summaries, 1);
    expect(summary != nullptr, "hbm_watermark_summary emitted for rank 1");
    if (summary != nullptr) {
        expect(summary->value("committed_peak_bytes", uint64_t(0)) == 120,
               "same-tick transient (110) and window-end delta (120) are "
               "sampled: committed peak is 120");
        expect(summary->value("resident_peak_bytes", uint64_t(0)) == 120,
               "resident peak follows the same per-delta samples (120)");
        expect(summary->value("capacity_violation_count", uint64_t(0)) == 2,
               "both over-capacity instants counted (post-seq-1 and "
               "window-end)");
    }

    ::unlink(manifest_path.c_str());
}

}  // namespace

int main() {
    run_same_tick_transient_case();
    std::fprintf(stderr, "[metric_capacity_replay_test] %d checks: %s\n",
                 g_checks, g_ok ? "PASS" : "FAIL");
    return g_ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
