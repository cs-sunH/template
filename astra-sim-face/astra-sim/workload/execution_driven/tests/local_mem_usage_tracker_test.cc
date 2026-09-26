/******************************************************************************
LocalMemUsageTracker regression fixture.

The legacy tracker held a complete unordered_set<TensorId> snapshot for every
timeline tick, then duplicated every trace event in a nlohmann::json vector.
This fixture keeps a small, independently-built legacy reference for exact
trace-byte and peak comparison, and drives a high-overlap timeline where the
old O(ticks * live_tensors) snapshot representation is impractical.
*******************************************************************************/

#include "astra-sim/workload/LocalMemUsageTracker.hh"

#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <map>
#include <set>
#include <string>
#include <sys/resource.h>
#include <unistd.h>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

using AstraSim::LocalMemUsageTracker;
using AstraSim::MemActivity;
using AstraSim::TensorId;
using AstraSim::Tick;
using json = nlohmann::json;

namespace AstraSim {

// The production API deliberately exposes no synthetic-node ingress.  This
// friend is limited to the regression fixture so it can compare the current
// implementation against a small reference without constructing ET protobufs.
struct LocalMemUsageTrackerTestAccess {
    static void seed(
        LocalMemUsageTracker& tracker,
        std::unordered_map<TensorId, uint64_t> tensor_sizes,
        std::unordered_map<TensorId, MemActivity> writes,
        std::unordered_map<TensorId, std::vector<MemActivity>> reads) {
        tracker.tensorSize = std::move(tensor_sizes);
        tracker.memWrites = std::move(writes);
        tracker.memReads = std::move(reads);
        tracker.tensorMapId.clear();
        tracker.peak_memory_usage_ = 0;
        tracker.has_timeline_tick_ = false;
        tracker.last_timeline_tick_ = 0;
    }

    static const std::unordered_map<TensorId, uint64_t>& tensor_sizes(
        const LocalMemUsageTracker& tracker) {
        return tracker.tensorSize;
    }

    static const std::unordered_map<TensorId, MemActivity>& writes(
        const LocalMemUsageTracker& tracker) {
        return tracker.memWrites;
    }

    static const std::unordered_map<TensorId, std::vector<MemActivity>>& reads(
        const LocalMemUsageTracker& tracker) {
        return tracker.memReads;
    }

    static uint64_t trace_event_count(const LocalMemUsageTracker& tracker) {
        return tracker.trace_spool_event_count_;
    }

    static bool trace_spool_open(const LocalMemUsageTracker& tracker) {
        return tracker.trace_spool_ != nullptr;
    }
};

}  // namespace AstraSim

namespace {

bool g_ok = true;

void expect(const bool condition, const char* const message) {
    if (!condition) {
        std::fprintf(stderr, "[local_mem_usage_tracker_test] FAIL: %s\n",
                     message);
        g_ok = false;
    }
}

MemActivity activity(const Tick start, const Tick end, std::string node_name,
                     const uint64_t node_id) {
    return MemActivity{start, end, std::move(node_name), node_id};
}

std::string temporary_prefix() {
    char pattern[] = "/tmp/astra-local-mem-tracker-XXXXXX";
    const int fd = ::mkstemp(pattern);
    if (fd < 0) {
        std::fprintf(stderr, "mkstemp failed: %s\n", std::strerror(errno));
        std::exit(EXIT_FAILURE);
    }
    if (::close(fd) != 0 || ::unlink(pattern) != 0) {
        std::fprintf(stderr, "temporary-prefix cleanup failed: %s\n",
                     std::strerror(errno));
        std::exit(EXIT_FAILURE);
    }
    return pattern;
}

std::string read_file(const std::string& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input.is_open()) {
        std::fprintf(stderr, "failed to open %s\n", path.c_str());
        std::exit(EXIT_FAILURE);
    }
    return std::string((std::istreambuf_iterator<char>(input)),
                       std::istreambuf_iterator<char>());
}

struct LegacyResult {
    std::string trace;
    uint64_t peak = 0;
};

// A deliberately local copy of the retired materialized-vector algorithm.  It
// is used only for a tiny fixture, so its historic O(K*T) snapshots remain a
// useful independent oracle rather than a production cost.
LegacyResult legacy_reference(const LocalMemUsageTracker& tracker) {
    const auto& tensor_sizes =
        AstraSim::LocalMemUsageTrackerTestAccess::tensor_sizes(tracker);
    const auto& writes = AstraSim::LocalMemUsageTrackerTestAccess::writes(tracker);
    const auto& reads = AstraSim::LocalMemUsageTrackerTestAccess::reads(tracker);

    std::unordered_map<TensorId, uint64_t> tensor_map_id;
    uint64_t id_count = 0;
    for (const auto& item : tensor_sizes) {
        tensor_map_id.insert(std::make_pair(item.first, id_count++));
    }

    std::vector<json> serialized;
    for (const auto& item : tensor_map_id) {
        const TensorId& tensor_name = item.first;
        const auto reads_it = reads.find(tensor_name);
        if (reads_it == reads.end()) {
            continue;
        }
        for (const auto& read_activity : reads_it->second) {
            const std::string node_name = read_activity.nodeName;
            const uint64_t node_id = read_activity.nodeId;
            serialized.push_back({
                {"name", tensor_name},
                {"cat", "tensorRead"},
                {"ph", "B"},
                {"ts", 1e-3 * read_activity.start},
                {"pid", tracker.sysId},
                {"tid", tensor_map_id.at(tensor_name)},
                {"args", json{{"size", tensor_sizes.at(tensor_name)},
                              {"node_name", node_name},
                              {"node_id", node_id}}},
            });
            serialized.push_back({
                {"name", tensor_name},
                {"cat", "tensorRead"},
                {"ph", "E"},
                {"ts", 1e-3 * read_activity.end},
                {"pid", tracker.sysId},
                {"tid", tensor_map_id.at(tensor_name)},
                {"args", json{{"size", tensor_sizes.at(tensor_name)},
                              {"node_name", node_name},
                              {"node_id", node_id}}},
            });
        }
    }
    for (const auto& item : tensor_map_id) {
        const TensorId& tensor_name = item.first;
        const MemActivity write_activity = writes.at(tensor_name);
        const std::string node_name = write_activity.nodeName;
        const uint64_t node_id = write_activity.nodeId;
        serialized.push_back({
            {"name", tensor_name},
            {"cat", "tensorWrite"},
            {"ph", "B"},
            {"ts", 1e-3 * write_activity.start},
            {"pid", tracker.sysId},
            {"tid", tensor_map_id.at(tensor_name)},
            {"args", json{{"size", tensor_sizes.at(tensor_name)},
                          {"node_name", node_name},
                          {"node_id", node_id}}},
        });
        serialized.push_back({
            {"name", tensor_name},
            {"cat", "tensorWrite"},
            {"ph", "E"},
            {"ts", 1e-3 * write_activity.end},
            {"pid", tracker.sysId},
            {"tid", tensor_map_id.at(tensor_name)},
            {"args", json{{"size", tensor_sizes.at(tensor_name)},
                          {"node_name", node_name},
                          {"node_id", node_id}}},
        });
    }

    std::set<Tick> ticks;
    std::map<Tick, std::vector<TensorId>> tensor_writes;
    std::map<Tick, std::vector<TensorId>> tensor_last_reads;
    for (const auto& item : writes) {
        const TensorId& tensor_name = item.first;
        const MemActivity& write_activity = item.second;
        ticks.insert(write_activity.start);
        tensor_writes[write_activity.start].push_back(tensor_name);
        serialized.push_back({
            {"name", tensor_name},
            {"cat", "tensorLifetime"},
            {"ph", "B"},
            {"ts", 1e-3 * write_activity.start},
            {"pid", tracker.sysId + 1000000ul},
            {"tid", tensor_map_id.at(tensor_name)},
            {"args", json{{"size", tensor_sizes.at(tensor_name)}}},
        });
    }
    for (const auto& item : reads) {
        const TensorId& tensor_name = item.first;
        const MemActivity& latest_read = item.second.back();
        ticks.insert(latest_read.end);
        tensor_last_reads[latest_read.end].push_back(tensor_name);
        serialized.push_back({
            {"name", tensor_name},
            {"cat", "tensorLifetime"},
            {"ph", "E"},
            {"ts", 1e-3 * latest_read.end},
            {"pid", tracker.sysId + 1000000ul},
            {"tid", tensor_map_id.at(tensor_name)},
            {"args", json{{"size", tensor_sizes.at(tensor_name)}}},
        });
    }

    std::map<Tick, std::unordered_set<TensorId>> memory_contents;
    std::map<Tick, uint64_t> memory_usage;
    for (auto it = ticks.begin(); it != ticks.end(); ++it) {
        if (it != ticks.begin()) {
            const auto previous = std::prev(it);
            memory_contents.emplace(*it, memory_contents.at(*previous));
        } else {
            memory_contents.emplace(*it, std::unordered_set<TensorId>());
        }
        const auto writes_it = tensor_writes.find(*it);
        if (writes_it != tensor_writes.end()) {
            for (const auto& tensor_name : writes_it->second) {
                memory_contents.at(*it).insert(tensor_name);
            }
        }
        const auto reads_it = tensor_last_reads.find(*it);
        if (reads_it != tensor_last_reads.end()) {
            for (const auto& tensor_name : reads_it->second) {
                memory_contents.at(*it).erase(tensor_name);
            }
        }
        uint64_t total_size_bytes = 0;
        for (const auto& tensor_name : memory_contents.at(*it)) {
            total_size_bytes += tensor_sizes.at(tensor_name);
        }
        const double total_size_mb =
            static_cast<double>(total_size_bytes) / (1024.0 * 1024.0);
        serialized.push_back({
            {"name", "GPU Memory Usage (MiB)"},
            {"cat", "GPU Memory"},
            {"ph", "C"},
            {"ts", 1e-3 * (*it)},
            {"pid", tracker.sysId + 2000000ul},
            {"args", json{{"Memory_MiB", total_size_mb}}},
        });
        memory_usage.emplace(*it, total_size_bytes);
    }

    std::vector<std::tuple<TensorId, Tick, Tick, uint64_t>> lifetimes;
    for (const auto& item : writes) {
        const TensorId& tensor_name = item.first;
        const Tick start = item.second.start;
        Tick end;
        const auto reads_it = reads.find(tensor_name);
        if (reads_it != reads.end() && !reads_it->second.empty()) {
            end = reads_it->second.back().end;
        } else if (!memory_usage.empty()) {
            end = memory_usage.rbegin()->first;
        } else {
            end = item.second.end;
        }
        lifetimes.emplace_back(tensor_name, start, end,
                               tensor_sizes.at(tensor_name));
    }
    std::sort(lifetimes.begin(), lifetimes.end(),
              [](const auto& left, const auto& right) {
                  return std::get<2>(left) - std::get<1>(left) >
                         std::get<2>(right) - std::get<1>(right);
              });

    const uint64_t heatmap_process_id = tracker.sysId + 3000000ul;
    serialized.push_back({
        {"name", "process_name"},
        {"ph", "M"},
        {"pid", heatmap_process_id},
        {"args", json{{"name", "Tensor Lifetime Heap"}}},
    });
    serialized.push_back({
        {"name", "thread_name"},
        {"ph", "M"},
        {"pid", heatmap_process_id},
        {"tid", 0},
        {"args", json{{"name", "Longest Lifetime → Shortest Lifetime"}}},
    });

    const int count = static_cast<int>(lifetimes.size());
    uint64_t min_lifetime = std::numeric_limits<uint64_t>::max();
    uint64_t max_lifetime = 0;
    for (int i = 0; i < count; ++i) {
        const uint64_t duration =
            std::get<2>(lifetimes[i]) - std::get<1>(lifetimes[i]);
        min_lifetime = std::min(min_lifetime, duration);
        max_lifetime = std::max(max_lifetime, duration);
    }
    uint64_t max_tensor_size = 0;
    for (const auto& item : tensor_sizes) {
        max_tensor_size = std::max(max_tensor_size, item.second);
    }
    for (int i = 0; i < count; ++i) {
        const auto& [tensor_name, start, end, size] = lifetimes[i];
        const uint64_t duration = end - start;
        int heap_position = 0;
        if (max_lifetime != min_lifetime) {
            heap_position = static_cast<int>(
                (static_cast<double>(duration - min_lifetime) /
                 (max_lifetime - min_lifetime)) *
                (count - 1));
        }
        // R5 (2026-08-29, authorized micro-item): the same provable
        // [0, 255] intensity range as the production getSizeColor (every
        // size scales by max_tensor_size, the maximum over all entries) is
        // restated here with the production-side clamp + unsigned args +
        // bounded snprintf, eliminating the format-overflow warning without
        // changing a single output byte (the byte-exact golden comparison
        // against the production implementation still holds; the
        // max_tensor_size == 0 path used to be a division-by-zero UB and
        // is now the well-defined blue, same as production).
        const double heatmap_ratio =
            (max_tensor_size == 0)
                ? 0.0
                : std::min(1.0, static_cast<double>(size) /
                                    static_cast<double>(max_tensor_size));
        unsigned int intensity = static_cast<unsigned int>(heatmap_ratio *
                                                           255.0);
        if (intensity > 255u) {
            intensity = 255u;
        }
        char color[8];
        std::snprintf(color, sizeof(color), "#%02X%02X%02X", intensity, 100u,
                      255u - intensity);
        const double size_mb = static_cast<double>(size) / (1024.0 * 1024.0);
        std::string display_name = tensor_name;
        if (display_name.length() > 20) {
            display_name = display_name.substr(0, 17) + "...";
        }
        display_name += " (" + std::to_string(size_mb).substr(0, 5) + " MiB)";
        serialized.push_back({
            {"name", display_name},
            {"cat", "tensorHeatmap"},
            {"ph", "X"},
            {"ts", 1e-3 * start},
            {"dur", 1e-3 * duration},
            {"pid", heatmap_process_id},
            {"tid", heap_position},
            {"cname", color},
            {"args", json{{"tensor_name", tensor_name},
                          {"size_bytes", size},
                          {"size_mib", size_mb},
                          {"lifetime_ns", duration},
                          {"position", heap_position}}},
        });
    }
    serialized.push_back({
        {"name", "Tensor Lifetime Heatmap"},
        {"cat", "tensorHeatmap"},
        {"ph", "i"},
        {"ts", 0},
        {"pid", heatmap_process_id},
        {"s", "p"},
        {"args", json{{"description", "Tensors arranged by lifetime duration (longest at bottom)"},
                      {"total_tensors", lifetimes.size()},
                      {"displayed_tensors", count}}},
    });

    uint64_t peak = 0;
    for (const auto& item : memory_usage) {
        peak = std::max(peak, item.second);
    }
    json trace;
    trace["traceEvents"] = serialized;
    return {trace.dump(2), peak};
}

void test_byte_exact_reference_and_boundaries() {
    LocalMemUsageTracker tracker(17);
    std::unordered_map<TensorId, uint64_t> sizes{
        {"alpha", 1024}, {"beta", 2048}, {"gamma", 512}, {"orphan", 128}};
    std::unordered_map<TensorId, MemActivity> writes{
        {"alpha", activity(10, 11, "writer-alpha", 101)},
        {"beta", activity(10, 12, "writer-beta", 102)},
        {"gamma", activity(25, 26, "writer-gamma", 103)},
        // This is the exact synthetic write record recordReads() creates for
        // a read-before-write tensor.
        {"orphan", activity(0, 10, "UNDEFINED", UINT64_MAX)},
    };
    std::unordered_map<TensorId, std::vector<MemActivity>> reads{
        {"alpha", {activity(15, 20, "reader-a0", 201),
                    activity(30, 40, "reader-a1", 202)}},
        // beta's free shares gamma's allocation tick; write-before-free order
        // is part of the legacy timeline semantics.
        {"beta", {activity(12, 25, "reader-b", 203)}},
        {"orphan", {activity(1, 17, "reader-orphan", 204)}},
    };
    AstraSim::LocalMemUsageTrackerTestAccess::seed(
        tracker, std::move(sizes), std::move(writes), std::move(reads));

    const LegacyResult expected = legacy_reference(tracker);
    tracker.buildMemoryTrace();
    tracker.buildMemoryTimeline();
    expect(AstraSim::LocalMemUsageTrackerTestAccess::trace_spool_open(tracker),
           "nonempty trace uses an anonymous spool instead of a JSON vector");
    expect(tracker.getPeakMemUsage() == expected.peak,
           "single-sweep peak equals legacy snapshot peak");

    const std::string prefix = temporary_prefix();
    const std::string trace_path = prefix + ".17.json";
    tracker.dumpMemoryTrace(prefix);
    const std::string actual = read_file(trace_path);
    expect(actual == expected.trace,
           "streamed pretty trace is byte-identical to the legacy JSON dump");
    try {
        expect(json::parse(actual) == json::parse(expected.trace),
               "streamed trace remains structurally valid JSON");
    } catch (const std::exception& error) {
        std::fprintf(stderr, "JSON parse failed: %s\n", error.what());
        g_ok = false;
    }
    if (::unlink(trace_path.c_str()) != 0) {
        std::fprintf(stderr, "trace cleanup failed: %s\n", std::strerror(errno));
        g_ok = false;
    }

    LocalMemUsageTracker empty_tracker(3);
    const std::string empty_prefix = temporary_prefix();
    const std::string empty_path = empty_prefix + ".3.json";
    empty_tracker.dumpMemoryTrace(empty_prefix);
    expect(read_file(empty_path) == "{\n  \"traceEvents\": []\n}",
           "empty trace keeps the legacy pretty JSON spelling");
    if (::unlink(empty_path.c_str()) != 0) {
        std::fprintf(stderr, "empty trace cleanup failed: %s\n",
                     std::strerror(errno));
        g_ok = false;
    }
}

void test_high_overlap_pressure() {
    // Every tensor is live at tick 0; one departs at each later tick.  The
    // old map of complete snapshots held roughly N*(N+1)/2 TensorId strings.
    // The production implementation holds one N-sized live set plus O(N)
    // event pointers while sweeping, and all rendered events live on disk.
    constexpr uint64_t kTensorCount = 4096;
    std::unordered_map<TensorId, uint64_t> sizes;
    std::unordered_map<TensorId, MemActivity> writes;
    std::unordered_map<TensorId, std::vector<MemActivity>> reads;
    sizes.reserve(kTensorCount);
    writes.reserve(kTensorCount);
    reads.reserve(kTensorCount);
    for (uint64_t index = 0; index < kTensorCount; ++index) {
        const TensorId name = "overlap_tensor_" + std::to_string(index);
        sizes.emplace(name, 1);
        writes.emplace(name, activity(0, 0, "stress-writer", index));
        reads.emplace(name, std::vector<MemActivity>{
                                activity(0, index + 1, "stress-reader", index)});
    }

    struct rusage before {};
    struct rusage after {};
    if (::getrusage(RUSAGE_SELF, &before) != 0) {
        std::fprintf(stderr, "getrusage(before) failed: %s\n", std::strerror(errno));
        std::exit(EXIT_FAILURE);
    }
    LocalMemUsageTracker tracker(9);
    AstraSim::LocalMemUsageTrackerTestAccess::seed(
        tracker, std::move(sizes), std::move(writes), std::move(reads));
    tracker.buildMemoryTrace();
    tracker.buildMemoryTimeline();
    if (::getrusage(RUSAGE_SELF, &after) != 0) {
        std::fprintf(stderr, "getrusage(after) failed: %s\n", std::strerror(errno));
        std::exit(EXIT_FAILURE);
    }

    expect(tracker.getPeakMemUsage() == kTensorCount,
           "high-overlap sweep preserves the all-live peak");
    expect(AstraSim::LocalMemUsageTrackerTestAccess::trace_event_count(tracker) ==
               8 * kTensorCount + 4,
           "high-overlap trace is spooled as individual events");
#ifdef __linux__
    const long rss_delta_kib = after.ru_maxrss - before.ru_maxrss;
    // The 4,096-timestamp / 4,096-live-tensor legacy layout is hundreds of
    // MiB before its JSON DOM.  This deliberately generous cap leaves room
    // for the linked simulator but rejects a return to quadratic snapshots.
    expect(rss_delta_kib >= 0 && rss_delta_kib < 256L * 1024L,
           "high-overlap tracker stays below the quadratic-snapshot RSS range");
#endif
}

}  // namespace

int main() {
    test_byte_exact_reference_and_boundaries();
    test_high_overlap_pressure();
    if (!g_ok) {
        return EXIT_FAILURE;
    }
    std::printf("local_mem_usage_tracker_test: PASS\n");
    return EXIT_SUCCESS;
}
