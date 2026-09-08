/******************************************************************************
MetricCollector one-shot bucket erase + OnlineNode anchor fast-path
regression fixture (R2, 2026-08-29; frozen plan §4-B2 / §6.3-1/2).

Covers, in order:
  A. on_node_issue/on_node_complete consume the matched node bucket in full
     and erase it immediately (empty rank bucket dropped too): bucket counts
     strictly decrease per fired node, unfired entries stay, a second call
     for a fired edge is a silent miss (no drop, no double record).
  B. Output equivalence: manifest load + one-shot fire path vs. a
     direct-apply baseline (the pre-R2 semantics) produce byte-identical
     [METRIC] output, including the multi-event bucket multiplicity, the
     retained raw memory anchors and the transfer_complete replay.  This is
     also the static-path red line: the static fire path keeps its
     unconditional call semantics.
  C. Online dynamic registration: return values (true for registered AND
     idempotent duplicates; false for disabled/unknown request/kind), the
     online spool precision, and the A2 request-level release coexisting
     with the one-shot erase (already-erased keys are harmless no-ops,
     unfired residue is freed, the reverse ledger drains).
  D. OnlineNode anchor flags: sizeof unchanged (padding reuse, locked by
     two static_asserts against a pre-R2 mirror struct), default-false on
     every NodeStore view, OR semantics of set_metric_anchor_flags, and
     flag visibility through the NodeStoreGraphSource views (the exact data
     path Workload::issue consumes).  The ETFeederGraphSource builds its
     views the same way (default-constructed NodeView, per-field assign,
     never touching the metric flags), so static views are structurally
     flag-false -- the gate `execution_mode_ != Online || flag` keeps the
     static path on the original unconditional logic.
*******************************************************************************/

#include "astra-sim/workload/MetricCollector.hh"
#include "astra-sim/workload/execution_driven/GraphSource.hh"
#include "astra-sim/workload/execution_driven/NodeStore.hh"

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unistd.h>
#include <vector>

#include <json/json.hpp>

using AstraSim::MetricCollector;
using AstraSim::Tick;
using json = nlohmann::json;

// ---------------------------------------------------------------------------
// D (part 1): pre-R2 mirror of OnlineNode (field list identical to the
// struct BEFORE the two metric flags were added).  The mirror assert is the
// ABI-agnostic proof that the flags landed in the former bool tail padding;
// the numeric assert anchors the current toolchain value (GCC 15 / libstdc++
// x86-64, std::vector = 40).  If a different toolchain changes the numeric
// value while the mirror assert still holds, update the number and re-prove
// -- that failure mode is protective, not an equivalence break.
// sh_2.0 adaptation (R5, 2026-08-29): this variant's pre-R2 OnlineNode
// carries three N-way-HBM members between is_timer_op and inputs_values
// (is_local_hbm_kv_restore / hbm_access_mode / hbm_charge) and its
// ComputeAttrs has no hbm_access_mode member (that lives on the node
// here), so the mirror reproduces THIS repository's pre-R2 field list.
// The two metric flags slot into the variant bool group's tail padding
// (the group ends at offset 64, the next member aligns to 72), so
// sizeof stays 448 exactly as in sh_1.0.
// ---------------------------------------------------------------------------
struct OnlineNodePreR2Mirror {
    uint64_t global_id = 0;
    int rank = 0;
    AstraSim::ExecutionDriven::NodeKind kind =
        AstraSim::ExecutionDriven::NodeKind::Invalid;
    uint64_t node_type = 0;
    std::string name;
    bool is_cpu_op = false;
    bool is_timer_op = false;
    bool is_local_hbm_kv_restore = false;
    int hbm_access_mode = 0;
    bool hbm_charge = true;
    std::string inputs_values;
    std::string request_id;
    std::string stage;
    uint64_t generation = 0;
    AstraSim::ExecutionDriven::ComputeAttrs compute;
    AstraSim::ExecutionDriven::CommAttrs comm;
    AstraSim::ExecutionDriven::CollAttrs coll;
    AstraSim::ExecutionDriven::OnlineStatisticsState online_statistics;
};
static_assert(sizeof(AstraSim::ExecutionDriven::OnlineNode) ==
                  sizeof(OnlineNodePreR2Mirror),
              "R2 metric anchor flags must reuse the OnlineNode bool tail "
              "padding; sizeof(OnlineNode) is unchanged");
static_assert(sizeof(AstraSim::ExecutionDriven::OnlineNode) == 448,
              "OnlineNode layout anchor for this toolchain (see the mirror "
              "assert above for the ABI-agnostic bound)");

namespace AstraSim {

// Same friend pattern as metric_collector_anchor_spool_test.cc: the
// production API stays clean, the fixture proves storage behavior.
struct MetricCollectorTestAccess {
    static void apply_event_direct(MetricCollector& collector,
                                   uint8_t event_code,
                                   int64_t subject_id,
                                   int rank,
                                   uint64_t node_id,
                                   Tick tick) {
        collector.apply_event(
            MetricCollector::NodeMetricEvent{event_code, subject_id}, rank,
            node_id, tick);
    }

    static size_t issue_rank_bucket_count(const MetricCollector& collector) {
        return collector.issue_events_.size();
    }

    static size_t complete_rank_bucket_count(const MetricCollector& collector) {
        return collector.complete_events_.size();
    }

    static size_t issue_node_bucket_count(const MetricCollector& collector) {
        size_t count = 0;
        for (const auto& entry : collector.issue_events_) {
            count += entry.second.size();
        }
        return count;
    }

    static size_t complete_node_bucket_count(const MetricCollector& collector) {
        size_t count = 0;
        for (const auto& entry : collector.complete_events_) {
            count += entry.second.size();
        }
        return count;
    }

    static bool issue_has_node(const MetricCollector& collector, int rank,
                               uint64_t node_id) {
        const auto rank_it = collector.issue_events_.find(rank);
        return rank_it != collector.issue_events_.end() &&
               rank_it->second.count(node_id) > 0;
    }

    static bool complete_has_node(const MetricCollector& collector, int rank,
                                  uint64_t node_id) {
        const auto rank_it = collector.complete_events_.find(rank);
        return rank_it != collector.complete_events_.end() &&
               rank_it->second.count(node_id) > 0;
    }

    static uint64_t dropped_event_count(const MetricCollector& collector) {
        return collector.dropped_events_;
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

    static size_t online_anchor_request_count(const MetricCollector& collector) {
        return collector.online_anchor_nodes_.size();
    }
};

}  // namespace AstraSim

namespace {

bool g_ok = true;
int g_checks = 0;

void expect(bool condition, const char* message) {
    ++g_checks;
    if (!condition) {
        std::fprintf(stderr, "[metric_one_shot_erase_test] FAIL: %s\n",
                     message);
        g_ok = false;
    }
}

std::string write_manifest(const json& manifest) {
    char pattern[] = "/tmp/astra-metric-one-shot-XXXXXX";
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

// Anchor layout (subject queue_index 10 everywhere):
//   rank 1  node 100: issue  code 1 (prefill_start) + complete code 2
//                    (prefill_end)                          -> 2 edges
//   rank 1  node 101: complete code 7 (memory anchor)       -> 1 edge
//   rank 1  node 102: complete codes 2 + 7 (MULTI-EVENT bucket)
//   rank 2  node 200: complete code 7                       -> 1 edge
// Two transfer_complete actions replay against those code-7 anchors
// (rank 1 exact max, rank 3 subject-max fallback), mirroring the anchor
// spool fixture's coverage so the replay arithmetic is exercised too.
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
    manifest["repo_variant"] = "metric-one-shot-erase-fixture";
    manifest["run_mode"] = "service";
    manifest["run_id"] = "metric-one-shot-erase-fixture";
    manifest["requests"] = json::array({request});
    json rank1_events = json::array({
        json::array({100, 1, 10}),
        json::array({100, 2, 10}),
        json::array({101, 7, 10}),
        json::array({102, 2, 10}),
        json::array({102, 7, 10}),
    });
    json rank2_events = json::array({
        json::array({200, 7, 10}),
    });
    manifest["node_events_by_rank"] =
        json::object({{"1", rank1_events}, {"2", rank2_events}});
    manifest["planner_memory_peaks"] = json::array();
    manifest["memory_actions"] = json::array({
        json{
            {"sequence_index", 1},
            {"anchor_kind", "transfer_complete"},
            {"anchor_quality", "exact"},
            {"trigger_queue_index", 10},
            {"rank", 1},
            {"weight_delta_bytes", 10},
            {"resident_kv_delta_bytes", 0},
            {"reserved_kv_delta_bytes", 0},
            {"cause", "rank-max"},
        },
        json{
            {"sequence_index", 2},
            {"anchor_kind", "transfer_complete"},
            {"anchor_quality", "fallback"},
            {"trigger_queue_index", 10},
            {"rank", 3},
            {"weight_delta_bytes", 0},
            {"resident_kv_delta_bytes", 20},
            {"reserved_kv_delta_bytes", 0},
            {"cause", "subject-max-fallback"},
        },
    });
    return manifest;
}

template <typename Action>
std::string capture_stdout(Action&& action) {
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

// A + B + C(static half): the one-shot fire path against the direct-apply
// baseline.  Runs on the SAME manifest so the only variable is the routing
// consumption; byte-identical output is the equivalence proof.
void test_one_shot_erase_and_output_equivalence() {
    const std::string manifest_path = write_manifest(make_manifest());
    MetricCollector& collector = MetricCollector::instance();

    const std::string one_shot_output = capture_stdout([&] {
        collector.initialize(manifest_path, "full");
        // Loaded shape: 1 issue node bucket (rank 1 / node 100), 4 complete
        // node buckets across 2 rank buckets.
        expect(AstraSim::MetricCollectorTestAccess::issue_rank_bucket_count(
                   collector) == 1,
               "manifest loads one issue rank bucket");
        expect(AstraSim::MetricCollectorTestAccess::issue_node_bucket_count(
                   collector) == 1,
               "manifest loads one issue node bucket");
        expect(AstraSim::MetricCollectorTestAccess::
                   complete_rank_bucket_count(collector) == 2,
               "manifest loads two complete rank buckets");
        expect(AstraSim::MetricCollectorTestAccess::
                   complete_node_bucket_count(collector) == 4,
               "manifest loads four complete node buckets");
        const uint64_t dropped_before =
            AstraSim::MetricCollectorTestAccess::dropped_event_count(collector);

        // Issue fire: bucket consumed and erased, rank bucket dropped.
        collector.on_node_issue(1, 100, 5);
        expect(AstraSim::MetricCollectorTestAccess::issue_node_bucket_count(
                   collector) == 0,
               "issue fire erases its node bucket");
        expect(AstraSim::MetricCollectorTestAccess::issue_rank_bucket_count(
                   collector) == 0,
               "empty issue rank bucket is dropped");

        // Complete fires: node-bucket count strictly -1 per fired node; the
        // rank-1 bucket survives until its last node fires.
        collector.on_node_complete(1, 100, 20);
        expect(AstraSim::MetricCollectorTestAccess::
                   complete_node_bucket_count(collector) == 3,
               "complete fire strictly decreases the node-bucket count");
        expect(AstraSim::MetricCollectorTestAccess::
                   complete_rank_bucket_count(collector) == 2,
               "rank bucket retained while other nodes are pending");
        collector.on_node_complete(1, 101, 30);
        expect(AstraSim::MetricCollectorTestAccess::
                   complete_node_bucket_count(collector) == 2,
               "second complete fire strictly decreases the count again");
        expect(AstraSim::MetricCollectorTestAccess::complete_has_node(
                   collector, 1, 102),
               "multi-event bucket stays pending until its node fires");
        expect(AstraSim::MetricCollectorTestAccess::
                   retained_raw_anchor_count(collector) == 1,
               "code-7 event copied into the retained raw anchor vector");
        collector.on_node_complete(1, 102, 40);
        expect(AstraSim::MetricCollectorTestAccess::
                   complete_node_bucket_count(collector) == 1,
               "multi-event bucket erased once after applying both events");
        expect(AstraSim::MetricCollectorTestAccess::
                   complete_rank_bucket_count(collector) == 1,
               "emptied rank-1 bucket dropped; rank 2 remains");
        expect(AstraSim::MetricCollectorTestAccess::
                   retained_raw_anchor_count(collector) == 2,
               "both rank-1 code-7 events retained (multi-event multiplicity)");
        expect(AstraSim::MetricCollectorTestAccess::complete_has_node(
                   collector, 2, 200),
               "unfired anchor entry is retained");

        // One-shot semantics: a second call for a fired edge is a silent
        // miss -- no crash, no drop, no double record, no count change.
        collector.on_node_issue(1, 100, 5);
        collector.on_node_complete(1, 100, 20);
        collector.on_node_complete(1, 102, 40);
        expect(AstraSim::MetricCollectorTestAccess::
                   complete_node_bucket_count(collector) == 1,
               "refire of a consumed edge changes nothing");
        expect(AstraSim::MetricCollectorTestAccess::
                   retained_raw_anchor_count(collector) == 2,
               "refire of a consumed edge records nothing");
        expect(AstraSim::MetricCollectorTestAccess::dropped_event_count(
                   collector) == dropped_before,
               "refire of a consumed edge is a miss, not a drop");

        // Final fire empties both tables while the raw anchors stay.
        collector.on_node_complete(2, 200, 50);
        expect(AstraSim::MetricCollectorTestAccess::
                   complete_node_bucket_count(collector) == 0 &&
                   AstraSim::MetricCollectorTestAccess::
                       complete_rank_bucket_count(collector) == 0,
               "all routing tables drain after every anchor fires");
        expect(AstraSim::MetricCollectorTestAccess::
                   retained_raw_anchor_count(collector) == 3,
               "finalize-bound raw anchors survive the routing erase");

        collector.finalize({}, 60);
        collector.finalize({}, 60);  // idempotent
    });

    const std::string baseline_output = capture_stdout([&] {
        collector.initialize(manifest_path, "full");
        // Pre-R2 semantics equivalent: every event applied exactly once
        // through apply_event directly (never through the routing tables),
        // in the same order the one-shot run fired them.  The manifest's
        // routing entries are never consumed and stay retained -- the old
        // "static buckets live until process end" behavior.
        AstraSim::MetricCollectorTestAccess::apply_event_direct(collector, 1,
                                                                10, 1, 100, 5);
        AstraSim::MetricCollectorTestAccess::apply_event_direct(collector, 2,
                                                                10, 1, 100,
                                                                20);
        AstraSim::MetricCollectorTestAccess::apply_event_direct(collector, 7,
                                                                10, 1, 101,
                                                                30);
        AstraSim::MetricCollectorTestAccess::apply_event_direct(collector, 2,
                                                                10, 1, 102,
                                                                40);
        AstraSim::MetricCollectorTestAccess::apply_event_direct(collector, 7,
                                                                10, 1, 102,
                                                                40);
        AstraSim::MetricCollectorTestAccess::apply_event_direct(collector, 7,
                                                                10, 2, 200,
                                                                50);
        expect(AstraSim::MetricCollectorTestAccess::issue_node_bucket_count(
                   collector) == 1,
               "baseline keeps its unconsumed issue entries");
        expect(AstraSim::MetricCollectorTestAccess::
                   complete_node_bucket_count(collector) == 4,
               "baseline keeps its unconsumed complete entries");
        expect(AstraSim::MetricCollectorTestAccess::
                   retained_raw_anchor_count(collector) == 3,
               "baseline retains the same raw anchor count");
        collector.finalize({}, 60);
    });

    expect(one_shot_output == baseline_output,
           "one-shot fire path [METRIC] output is byte-identical to the "
           "direct-apply baseline (static red line)");

    const std::vector<json> anchors =
        records_of_type(one_shot_output, "memory_anchor");
    const std::vector<json> capacity =
        records_of_type(one_shot_output, "capacity_timeavg");
    expect(anchors.size() == 3,
           "every raw memory anchor is emitted despite the routing erase");
    if (anchors.size() == 3) {
        const uint64_t nodes[] = {101, 102, 200};
        for (size_t i = 0; i < 3; ++i) {
            expect(anchors[i].value("node_id", uint64_t(0)) == nodes[i],
                   "memory_anchor node order preserved");
        }
    }
    const json* const rank_one = record_for_rank(capacity, 1);
    const json* const rank_three = record_for_rank(capacity, 3);
    expect(rank_one != nullptr, "rank-max transfer action replayed");
    expect(rank_three != nullptr, "subject-max fallback replayed");
    if (rank_one != nullptr) {
        // weight 10 anchored at the rank-1 code-7 max tick 40, integrated
        // over [40, 60] -> 10 * 20 = 200.
        expect(rank_one->value("resident_byte_ns", std::string()) == "200",
               "rank-max transfer replay value unchanged by the erase");
        expect(rank_one->value("transfer_anchor_request_level_fallback", 99) ==
                   0,
               "exact rank transfer anchor does not count fallback");
    }
    if (rank_three != nullptr) {
        // resident 20 anchored at the subject max tick 50 over [50, 60] ->
        // 20 * 10 = 200.
        expect(rank_three->value("resident_byte_ns", std::string()) == "200",
               "subject-max fallback value unchanged by the erase");
        expect(rank_three->value("transfer_anchor_request_level_fallback", 99) ==
                   1,
               "subject-max fallback counter remains exact");
    }

    ::unlink(manifest_path.c_str());
}

// C (online half): dynamic registration return values, spool precision, and
// the A2 request-level release coexisting with the one-shot erase.
void test_online_registration_and_a2_coexistence() {
    const std::string manifest_path = write_manifest(make_manifest());
    MetricCollector& collector = MetricCollector::instance();

    (void)capture_stdout([&] {
        collector.initialize(manifest_path, "full");
        collector.clear_static_node_events();
        expect(AstraSim::MetricCollectorTestAccess::spool_open(collector),
               "online transition opens the spool");
        collector.online_register_request(10, "r10", "s0", 0, true, 0, -1, 0);

        // Registration return values.
        expect(collector.online_register_node_anchor(1, 100, "r10",
                                                     "prefill_start", false),
               "start anchor registration returns true");
        expect(AstraSim::MetricCollectorTestAccess::issue_has_node(collector,
                                                                   1, 100),
               "start anchor lands in the issue table");
        expect(collector.online_register_node_anchor(1, 101, "r10",
                                                     "prefill_end", false),
               "end anchor registration returns true");
        expect(AstraSim::MetricCollectorTestAccess::complete_has_node(
                   collector, 1, 101),
               "end anchor lands in the complete table");
        expect(collector.online_register_node_anchor(1, 102, "r10", "", true),
               "transfer anchor registration returns true");
        expect(AstraSim::MetricCollectorTestAccess::complete_has_node(
                   collector, 1, 102),
               "transfer anchor lands in the complete table");
        expect(collector.online_register_node_anchor(1, 101, "r10",
                                                     "prefill_end", false),
               "idempotent duplicate registration also returns true (the "
               "fast-path flag must stay set)");
        expect(AstraSim::MetricCollectorTestAccess::
                   online_anchor_request_count(collector) == 1,
               "reverse ledger holds one request");
        expect(!collector.online_register_node_anchor(
                   1, 103, "no-such-request", "prefill_start", false),
               "unknown request returns false");
        expect(!collector.online_register_node_anchor(1, 103, "r10",
                                                      "bogus-kind", false),
               "unknown kind returns false");
        expect(!AstraSim::MetricCollectorTestAccess::issue_has_node(
                   collector, 1, 103) &&
                   !AstraSim::MetricCollectorTestAccess::complete_has_node(
                       collector, 1, 103),
               "rejected registrations leave no routing entry");

        // Fire the issue and the end anchor: one-shot erases both; the
        // transfer anchor is the unfired residue of the window.
        collector.on_node_issue(1, 100, 5);
        expect(AstraSim::MetricCollectorTestAccess::issue_node_bucket_count(
                   collector) == 0,
               "online issue fire erases its bucket");
        collector.on_node_complete(1, 101, 20);
        expect(!AstraSim::MetricCollectorTestAccess::complete_has_node(
                   collector, 1, 101),
               "online complete fire erases its bucket");
        expect(AstraSim::MetricCollectorTestAccess::complete_has_node(
                   collector, 1, 102),
               "unfired transfer anchor stays until release");
        expect(AstraSim::MetricCollectorTestAccess::spool_record_count(
                   collector) == 0,
               "unfired transfer anchor writes nothing to the spool");

        // A2 release AFTER one-shot already erased two of the three keys:
        // already-erased keys are harmless no-ops, the unfired residue is
        // freed, the reverse ledger drains.
        collector.online_release_request_anchors("r10");
        expect(AstraSim::MetricCollectorTestAccess::
                   complete_node_bucket_count(collector) == 0,
               "A2 frees the unfired residue after one-shot erases the rest");
        expect(AstraSim::MetricCollectorTestAccess::
                   online_anchor_request_count(collector) == 0,
               "A2 drains the reverse ledger");
        collector.online_release_request_anchors("r10");  // idempotent no-op

        // A post-release fire of the released node is a silent miss
        // (documented one-shot/A2 interplay; normal runs fire before the
        // REQUEST_COMPLETE release).
        collector.on_node_complete(1, 102, 30);
        expect(AstraSim::MetricCollectorTestAccess::spool_record_count(
                   collector) == 0,
               "post-release fire is a silent miss, not a crash");

        // Fire the transfer anchor in the normal order on a fresh request
        // cycle to prove the spool precision with the erase active.
        expect(collector.online_register_node_anchor(1, 300, "r10",
                                                     "completion", false),
               "fresh registration after release returns true");
        expect(collector.online_register_node_anchor(1, 301, "r10", "",
                                                     true),
               "fresh transfer registration returns true");
        collector.on_node_complete(1, 300, 60);
        collector.on_node_complete(1, 301, 70);
        expect(AstraSim::MetricCollectorTestAccess::spool_record_count(
                   collector) == 1,
               "fired transfer anchor writes exactly one spool record");
        expect(AstraSim::MetricCollectorTestAccess::
                   complete_node_bucket_count(collector) == 0,
               "online tables drain once every anchor fires");
        collector.online_release_request_anchors("r10");

        collector.finalize({}, 100);
    });

    ::unlink(manifest_path.c_str());
}

// Metrics-off registration and re-initialization.
void test_disabled_registration_returns_false() {
    const std::string manifest_path = write_manifest(make_manifest());
    MetricCollector& collector = MetricCollector::instance();
    (void)capture_stdout([&] {
        collector.initialize(manifest_path, "off");
        expect(!collector.enabled(), "off detail level disables metrics");
        expect(!collector.online_register_node_anchor(1, 100, "r10",
                                                      "prefill_start", false),
               "disabled registration returns false");
        expect(AstraSim::MetricCollectorTestAccess::issue_node_bucket_count(
                   collector) == 0,
               "disabled registration writes no entry");
        // Restore for any later singleton reuse.
        collector.initialize(manifest_path, "full");
        expect(collector.enabled(), "re-initialize restores metrics");
    });
    ::unlink(manifest_path.c_str());
}

// D (part 2): NodeStore flag plumbing -- defaults, OR semantics, unknown-id
// no-op, and visibility through the exact view APIs Workload consumes.
void test_online_node_flag_plumbing() {
    using AstraSim::ExecutionDriven::NodeKind;
    using AstraSim::ExecutionDriven::NodeStore;
    using AstraSim::ExecutionDriven::NodeStoreGraphSource;
    using AstraSim::ExecutionDriven::NodeView;
    using AstraSim::ExecutionDriven::OnlineNode;

    NodeStore store;
    OnlineNode plain;
    plain.kind = NodeKind::Compute;
    const uint64_t plain_id = store.add_node(plain);

    // Every view of a freshly added (unanchored) node is flag-false: the
    // structural premise that ordinary online nodes never enter the
    // MetricCollector hash lookups once the Workload gate consults the
    // flags.
    const OnlineNode* ptr = store.node_ptr(plain_id);
    expect(ptr != nullptr && !ptr->metric_issue_anchor &&
               !ptr->metric_complete_anchor,
           "node_ptr view of an unanchored node is flag-false");
    const auto opt = store.node(plain_id);
    expect(opt.has_value() && !opt->metric_issue_anchor &&
               !opt->metric_complete_anchor,
           "by-value lookup view of an unanchored node is flag-false");

    // ETFeederGraphSource::view_of constructs its views exactly this way
    // (default-constructed NodeView, per-field assignment, never touching
    // the metric flags), so static/ET views are structurally flag-false and
    // the `execution_mode_ != Online || flag` gate keeps the static path on
    // the original unconditional call.
    NodeView et_style_view;
    et_style_view.global_id = 7;
    et_style_view.name = "static_et_node";
    expect(!et_style_view.metric_issue_anchor &&
               !et_style_view.metric_complete_anchor,
           "ET-style (default-constructed) view is flag-false");

    // OR semantics: a second registration on the other edge adds, never
    // clears; a re-registration never clears either.
    store.set_metric_anchor_flags(plain_id, /*issue=*/true,
                                  /*complete=*/false);
    expect(ptr->metric_issue_anchor && !ptr->metric_complete_anchor,
           "issue-only registration sets only the issue flag");
    store.set_metric_anchor_flags(plain_id, /*issue=*/false,
                                  /*complete=*/true);
    expect(ptr->metric_issue_anchor && ptr->metric_complete_anchor,
           "complete registration ORs in without clearing the issue flag");
    store.set_metric_anchor_flags(plain_id, /*issue=*/false,
                                  /*complete=*/false);
    expect(ptr->metric_issue_anchor && ptr->metric_complete_anchor,
           "re-registration never clears already-set flags");
    // Unknown id: no-op, no crash.
    store.set_metric_anchor_flags(987654321u, true, true);
    expect(store.node_ptr(987654321u) == nullptr,
           "unknown id remains unknown after a no-op flag set");

    // Visibility through the source views (Workload::issue's data path).
    NodeStoreGraphSource source;
    OnlineNode anchored;
    anchored.kind = NodeKind::MemLoad;
    anchored.name = "kv_route_tail";
    const uint64_t anchored_id = source.store().add_node(anchored);
    source.store().set_metric_anchor_flags(anchored_id, /*issue=*/false,
                                           /*complete=*/true);
    size_t seen = 0;
    source.for_each_dep_free([&](const NodeView& nv) {
        ++seen;
        expect(nv.global_id == anchored_id,
               "for_each_dep_free yields the anchored node");
        expect(nv.metric_complete_anchor && !nv.metric_issue_anchor,
               "for_each_dep_free view carries the registered flag");
    });
    expect(seen == 1, "one dep-free node visited");
    const NodeView* looked_up = source.lookup_ptr(anchored_id);
    expect(looked_up != nullptr && looked_up->metric_complete_anchor,
           "lookup_ptr view carries the registered flag");
}

}  // namespace

int main() {
    test_one_shot_erase_and_output_equivalence();
    test_online_registration_and_a2_coexistence();
    test_disabled_registration_returns_false();
    test_online_node_flag_plumbing();
    std::fprintf(stderr, "[metric_one_shot_erase_test] %d checks: %s\n",
                 g_checks, g_ok ? "PASS" : "FAIL");
    return g_ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
