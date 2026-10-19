#include "astra-sim/workload/LocalMemUsageTracker.hh"

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <stdexcept>
#include <memory>
#include <unordered_set>
#include <vector>
#include <tuple>
#include <string>
#include <set>
#include <map>
#include <fstream>
#include <algorithm>
#include <limits>

#include "astra-sim/system/Common.hh"
#include "astra-sim/common/Logging.hh"

// Using the new feeder v3 APIs.
using namespace AstraSim;
using namespace Chakra;
using namespace Chakra::FeederV3;  // Bring ChakraAttr and other FeederV3 types into scope

namespace {

struct TimelineActions {
  std::vector<const TensorId*> writes;
  std::vector<const TensorId*> last_reads;
};

using TimelineActionsByTick = std::map<Tick, TimelineActions>;

void apply_timeline_actions(
    const TimelineActions& actions,
    const std::unordered_map<TensorId, uint64_t>& tensor_sizes,
    std::unordered_set<TensorId>& live_tensors,
    uint64_t& current_bytes) {
  // The legacy code applies all writes at a tick before all final reads at the
  // same tick.  Preserve that ordering, including its insert/erase semantics.
  for (const TensorId* const tensor_name : actions.writes) {
    const auto insertion = live_tensors.insert(*tensor_name);
    if (insertion.second) {
      current_bytes += tensor_sizes.at(*tensor_name);
    }
  }
  for (const TensorId* const tensor_name : actions.last_reads) {
    if (live_tensors.erase(*tensor_name) != 0) {
      current_bytes -= tensor_sizes.at(*tensor_name);
    }
  }
}

void write_indented_event(std::ostream& output,
                          const std::string& encoded_event) {
  output << "    ";
  for (const char character : encoded_event) {
    output.put(character);
    if (character == '\n') {
      output << "    ";
    }
  }
}

[[noreturn]] void throw_trace_spool_error(const char* const operation) {
  const int error_number = errno;
  throw std::runtime_error(
      std::string("LocalMemUsageTracker trace spool failed while ") + operation +
      ": " + (error_number == 0 ? "unknown error" : std::strerror(error_number)));
}

}  // namespace

void LocalMemUsageTracker::parseIOInfos(
  const google::protobuf::RepeatedPtrField<std::string>& values,
  std::vector<std::tuple<TensorId, uint64_t>>& IOinfos) {
  if (values.size() % 2 != 0) {
    throw std::runtime_error("IO infos list size is not even.");
  }
  for (int i = 0; i < values.size(); i += 2) {
    const TensorId& tensorName = values.Get(i);
    const std::string& sizeStr = values.Get(i + 1);
    // R8-W1 fix (2026-09-29): the bare std::stoull accepted strtoull's
    // lexical family with silent poisoning: "-5" wrapped to ~1.8e19 and
    // entered every memory statistic, leading whitespace/trailing garbage
    // were silently ignored, and an empty/out-of-range string escaped as an
    // uncaught exception with no field context. Same strict ASCII-digit
    // lexicon as the repo's other fail-closed parses.
    if (sizeStr.empty()) {
      throw std::runtime_error(
        "IO infos size field for tensor '" + tensorName + "' is empty");
    }
    for (const char c : sizeStr) {
      if (c < '0' || c > '9') {
        throw std::runtime_error(
          "IO infos size field for tensor '" + tensorName +
          "' is not a non-negative integer (ASCII digits only): " + sizeStr);
      }
    }
    errno = 0;
    char* end = nullptr;
    const unsigned long long parsed =
      std::strtoull(sizeStr.c_str(), &end, 10);
    if (errno == ERANGE || end != sizeStr.c_str() + sizeStr.size()) {
      throw std::runtime_error(
        "IO infos size field for tensor '" + tensorName +
        "' exceeds the 64-bit unsigned range: " + sizeStr);
    }
    IOinfos.emplace_back(tensorName, static_cast<uint64_t>(parsed));
  }
}

void LocalMemUsageTracker::recordReads(
  const std::shared_ptr<Chakra::ETFeederNode> node,
  Tick start,
  Tick end) {
  // Scratch buffer for this call only (was a function-local static shared
  // across calls for reuse; the simulator drives this tracker from a single
  // event thread, so a plain local is equivalent and stateless).
  std::vector<std::tuple<TensorId, uint64_t>> IOinfos;
  uint64_t nodeId = node->id();

  if (!node->has_attr("inputs")) {
    throw std::runtime_error("Attribute 'inputs' not found in node " +
      std::to_string(nodeId) +
      " (LocalMemUsageTracker::recordReads)");
  }

  // use the explicit template type.
  auto attr = node->get_attr<ChakraAttr>("inputs");
  if (!attr.has_string_list() || attr.string_list().values().empty()) {
      return;
  }
  const auto& values = attr.string_list().values();
  parseIOInfos(values, IOinfos);

  for (const auto& iter : IOinfos) {
    TensorId tensorName = std::get<0>(iter);
    uint64_t tensorSize = std::get<1>(iter);

    // Ignore tensors with size 0
    if (tensorSize == 0) {
      continue;
    }

    MemActivity readActivity;
    readActivity.start = start;
    readActivity.end = end;
    readActivity.nodeName = node->name();
    readActivity.nodeId = node->id();

    if (this->tensorSize.find(tensorName) == this->tensorSize.end()) {
      AstraSim::LoggerFactory::get_logger("workload::LocalMemUsageTracker")
          ->trace("tracker record read before write node.id={} tensor.name={} start={} end={}",
                  node->id(), tensorName, start, end);
      MemActivity writeActivity;
      writeActivity.start = 0ul;
      writeActivity.end = 10ul;
      writeActivity.nodeName = "UNDEFINED";
      writeActivity.nodeId = UINT64_MAX;
      this->tensorSize.insert({tensorName, tensorSize});
      this->memWrites.insert({tensorName, writeActivity});
    }
    if (this->memReads.find(tensorName) == this->memReads.end()) {
      this->memReads.emplace(tensorName, std::vector<MemActivity>());
    }
    AstraSim::LoggerFactory::get_logger("workload::LocalMemUsageTracker")
        ->trace("tracker record read node.id={} tensor.name={} start={} end={}",
                node->id(), tensorName, start, end);
    this->memReads[tensorName].emplace_back(readActivity);
  }
}

void LocalMemUsageTracker::recordWrites(
  const std::shared_ptr<Chakra::ETFeederNode> node,
  Tick start,
  Tick end) {
  // Scratch buffer for this call only (was a function-local static shared
  // across calls for reuse; the simulator drives this tracker from a single
  // event thread, so a plain local is equivalent and stateless).
  std::vector<std::tuple<TensorId, uint64_t>> IOinfos;
  uint64_t nodeId = node->id();

  if (!node->has_attr("outputs")) {
    throw std::runtime_error("Attribute 'outputs' not found in node " +
      std::to_string(nodeId) + " (LocalMemUsageTracker::recordWrites)");
  }

  auto attr = node->get_attr<ChakraAttr>("outputs");
  if (!attr.has_string_list() || attr.string_list().values().empty()) {
      return;
  }

  const auto& values = attr.string_list().values();
  parseIOInfos(values, IOinfos);

  for (const auto& iter : IOinfos) {
    TensorId tensorName = std::get<0>(iter);
    uint64_t tensorSize = std::get<1>(iter);

    // Ignore tensors with size 0
    if (tensorSize == 0) {
      continue;
    }

    MemActivity writeActivity;
    writeActivity.start = start;
    writeActivity.end = end;
    writeActivity.nodeName = node->name();
    writeActivity.nodeId = node->id();
    // S4 fix (2026-09-28): the per-write unconditional logger flush is gone
    // (方案 §15: no per-node forced flush in official logging; the flush ran
    // below the trace level gate, one sink refresh per tensor write on the
    // --track-local-mem path). The logger keeps its own batching; the offline
    // trace_spool_ fflush at dump time is unrelated and unchanged.
    AstraSim::LoggerFactory::get_logger("workload::LocalMemUsageTracker")
        ->trace("tracker record write node.id={} tensor.name={} start={} end={}",
                node->id(), tensorName, start, end);
    if (this->tensorSize.find(tensorName) == this->tensorSize.end()) {
      // first write
      this->tensorSize.insert({tensorName, tensorSize});
      this->memWrites.insert({tensorName, writeActivity});
    } else {
      // Each tensor should only be written once.  The sole legal exception
      // is the synthetic "read before write" placeholder recordReads()
      // seeds for a tensor whose read event was reported first: the real
      // write supersedes that placeholder instead of tripping the guard
      // (otherwise the tolerated read-before-write ordering would turn
      // into a hard duplicate-write crash here).
      const auto placeholder = this->memWrites.find(tensorName);
      if (placeholder != this->memWrites.end() &&
          placeholder->second.nodeName == "UNDEFINED" &&
          placeholder->second.nodeId == UINT64_MAX) {
        this->tensorSize[tensorName] = tensorSize;
        placeholder->second = writeActivity;
      } else {
        throw std::runtime_error("LocalMemUsageTracker: tensor '" +
                                 tensorName + "' written more than once");
      }
    }
  }
}

void LocalMemUsageTracker::recordStart(
    const std::shared_ptr<Chakra::ETFeederNode> node,
    Tick tick) {
  uint64_t nodeId = node->id();
  AstraSim::LoggerFactory::get_logger("workload::LocalMemUsageTracker")
      ->trace("tracker record start of node.id={} at tick={}", nodeId, tick);
  this->activityStartTime[nodeId] = tick;
}

void LocalMemUsageTracker::recordEnd(
    const std::shared_ptr<Chakra::ETFeederNode> node,
    Tick tick) {
  uint64_t nodeId = node->id();
  if (this->activityStartTime.find(nodeId) == this->activityStartTime.end()) {
    // (2026-09-30) A node ending without a recorded start would silently
    // drop its entire read/write tensor accounting. Every other malformed
    // input in this file throws; this historical early-return at least
    // leaves a forensic warn line now.
    AstraSim::LoggerFactory::get_logger("workload::LocalMemUsageTracker")
        ->warn(
            "recordEnd without recordStart: node.id={} -- its tensor "
            "accounting is dropped",
            nodeId);
    return;
  }
  Tick start = this->activityStartTime[nodeId];
  Tick end = tick;
  AstraSim::LoggerFactory::get_logger("workload::LocalMemUsageTracker")
      ->trace("tracker record end of node.id={} at start={} end={}", nodeId, start, end);
  this->recordReads(node, start, end);
  this->recordWrites(node, start, end);
  this->activityStartTime.erase(nodeId);
}

void LocalMemUsageTracker::close_trace_spool() noexcept {
  if (this->trace_spool_ != nullptr) {
    std::fclose(this->trace_spool_);
    this->trace_spool_ = nullptr;
  }
}

void LocalMemUsageTracker::reset_trace_spool() {
  close_trace_spool();
  this->trace_spool_event_count_ = 0;
}

void LocalMemUsageTracker::append_trace_event(const json& event) {
  if (this->trace_spool_ == nullptr) {
    errno = 0;
    this->trace_spool_ = std::tmpfile();
    if (this->trace_spool_ == nullptr) {
      throw_trace_spool_error("creating an anonymous spool");
    }
  }

  const std::string encoded_event = event.dump(2);
  if (this->trace_spool_event_count_ ==
      std::numeric_limits<uint64_t>::max()) {
    throw std::runtime_error("LocalMemUsageTracker trace event count overflow");
  }

  const uint64_t encoded_size =
      static_cast<uint64_t>(encoded_event.size());
  if (std::fwrite(&encoded_size, 1, sizeof(encoded_size),
                  this->trace_spool_) != sizeof(encoded_size) ||
      std::fwrite(encoded_event.data(), 1, encoded_event.size(),
                  this->trace_spool_) != encoded_event.size()) {
    throw_trace_spool_error("writing an event");
  }
  ++this->trace_spool_event_count_;
}

void LocalMemUsageTracker::buildMemoryTrace() {
  this->tensorMapId.clear();
  uint64_t idCnt = 0ul;
  for (const auto& item : tensorSize) {
    const TensorId& tensorName = item.first;
    uint64_t id = idCnt++;
    tensorMapId.insert(std::make_pair(tensorName, id));
  }
  this->reset_trace_spool();
  for (const auto& item : tensorMapId) {
    const TensorId& tensorName = item.first;
    if (this->memReads.find(tensorName) == this->memReads.end()) {
      continue;
    }
    for (const auto& readActivity : this->memReads.at(tensorName)) {
      std::string nodeName = readActivity.nodeName;
      uint64_t nodeId = readActivity.nodeId;
      json objStart = {
          {"name", tensorName},
          {"cat", "tensorRead"},
          {"ph", "B"},
          {"ts", 1e-3 * readActivity.start},
          {"pid", this->sysId},
          {"tid", this->tensorMapId.at(tensorName)},
          {"args", json{{"size", this->tensorSize.at(tensorName)},
                        {"node_name", nodeName},
                        {"node_id", nodeId}}}};
      json objEnd = {
          {"name", tensorName},
          {"cat", "tensorRead"},
          {"ph", "E"},
          {"ts", 1e-3 * readActivity.end},
          {"pid", this->sysId},
          {"tid", this->tensorMapId.at(tensorName)},
          {"args", json{{"size", this->tensorSize.at(tensorName)},
                        {"node_name", nodeName},
                        {"node_id", nodeId}}}};
      this->append_trace_event(objStart);
      this->append_trace_event(objEnd);
    }
  }
  for (const auto& item : tensorMapId) {
    const TensorId& tensorName = item.first;
    auto writeActivity = this->memWrites.at(tensorName);
    std::string nodeName = writeActivity.nodeName;
    uint64_t nodeId = writeActivity.nodeId;
    json objStart = {
        {"name", tensorName},
        {"cat", "tensorWrite"},
        {"ph", "B"},
        {"ts", 1e-3 * writeActivity.start},
        {"pid", this->sysId},
        {"tid", this->tensorMapId.at(tensorName)},
        {"args", json{{"size", this->tensorSize.at(tensorName)},
                      {"node_name", nodeName},
                      {"node_id", nodeId}}}};
    json objEnd = {
        {"name", tensorName},
        {"cat", "tensorWrite"},
        {"ph", "E"},
        {"ts", 1e-3 * writeActivity.end},
        {"pid", this->sysId},
        {"tid", this->tensorMapId.at(tensorName)},
        {"args", json{{"size", this->tensorSize.at(tensorName)},
                      {"node_name", nodeName},
                      {"node_id", nodeId}}}};
    this->append_trace_event(objStart);
    this->append_trace_event(objEnd);
  }
}

void LocalMemUsageTracker::dumpMemoryTrace(const std::string& filename) {
  std::string local_mem_trace_filename =
      fmt::format(filename + ".{}.json", this->sysId);
  std::ofstream file(local_mem_trace_filename);
  if (!file.is_open()) {
    AstraSim::LoggerFactory::get_logger("workload::LocalMemUsageTracker")
        ->error("failed to open file {}", local_mem_trace_filename);
    return;
  }
  if (this->trace_spool_event_count_ == 0) {
    file << "{\n  \"traceEvents\": []\n}";
    file.close();
    return;
  }
  if (this->trace_spool_ == nullptr) {
    AstraSim::LoggerFactory::get_logger("workload::LocalMemUsageTracker")
        ->error("trace spool is missing despite {} serialized events",
                this->trace_spool_event_count_);
    file.close();
    return;
  }
  if (std::fflush(this->trace_spool_) != 0) {
    AstraSim::LoggerFactory::get_logger("workload::LocalMemUsageTracker")
        ->error("failed to flush trace spool: {}", std::strerror(errno));
    file.close();
    return;
  }
  std::clearerr(this->trace_spool_);
  if (std::fseek(this->trace_spool_, 0, SEEK_SET) != 0) {
    AstraSim::LoggerFactory::get_logger("workload::LocalMemUsageTracker")
        ->error("failed to rewind trace spool: {}", std::strerror(errno));
    file.close();
    return;
  }

  file << "{\n  \"traceEvents\": [\n";
  for (uint64_t event_index = 0;
       event_index < this->trace_spool_event_count_;
       ++event_index) {
    uint64_t encoded_size = 0;
    if (std::fread(&encoded_size, 1, sizeof(encoded_size),
                   this->trace_spool_) != sizeof(encoded_size)) {
      AstraSim::LoggerFactory::get_logger("workload::LocalMemUsageTracker")
          ->error("failed to read trace spool event header");
      file.close();
      return;
    }
    std::string encoded_event(static_cast<size_t>(encoded_size), '\0');
    if (std::fread(encoded_event.data(), 1, encoded_event.size(),
                   this->trace_spool_) != encoded_event.size()) {
      AstraSim::LoggerFactory::get_logger("workload::LocalMemUsageTracker")
          ->error("failed to read trace spool event body");
      file.close();
      return;
    }
    if (event_index != 0) {
      file << ",\n";
    }
    write_indented_event(file, encoded_event);
  }
  file << "\n  ]\n}";
  file.close();
}

void LocalMemUsageTracker::buildMemoryTimeline() {
  // tensorMapId is normally populated by buildMemoryTrace(), which the sole
  // production caller always runs first (Workload.cc).  Populate it the same
  // way when the timeline is requested without a preceding trace build so
  // the .at() lookups below cannot go out_of_range; with the production
  // call order this is a no-op (map already filled over the same unchanged
  // tensorSize container, so the ids stay byte-identical).
  if (this->tensorMapId.empty()) {
    uint64_t idCnt = 0ul;
    for (const auto& item : this->tensorSize) {
      this->tensorMapId.insert(std::make_pair(item.first, idCnt++));
    }
  }
  TimelineActionsByTick actions_by_tick;
  // R1-C5 fix (2026-09-29): a read-before-write tensor (its last read ends
  // strictly before its write starts -- a malformed input shape) used to
  // stay in the live-byte sweep forever: the write inserted it into
  // live_tensors, but its removal action (the last read) sat at an earlier
  // tick where the erase had already been consumed as a no-op, so its bytes
  // inflated the peak to the end of the trace. The heatmap clamps the same
  // backwards-lifetime shape to a zero-duration row; mirror that here by
  // excluding such tensors from the byte accounting (their B/E lifetime
  // events are still emitted, unchanged).
  std::unordered_set<TensorId> read_before_write_tensors;
  for (const auto& item : this->memReads) {
    const auto write_it = this->memWrites.find(item.first);
    if (write_it != this->memWrites.end() && !item.second.empty() &&
        item.second.back().end < write_it->second.start) {
      read_before_write_tensors.insert(item.first);
    }
  }
  for (const auto& item : this->memWrites) {
    const TensorId& tensorName = item.first;
    const MemActivity& writeActivity = item.second;
    if (read_before_write_tensors.count(tensorName) == 0) {
      actions_by_tick[writeActivity.start].writes.push_back(&tensorName);
    }
    json tensorWriteEvent = {
        {"name", tensorName},
        {"cat", "tensorLifetime"},
        {"ph", "B"},
        {"ts", 1e-3 * writeActivity.start},
        {"pid", this->sysId + 1000000ul},
        {"tid", this->tensorMapId.at(tensorName)},
        {"args", json{{"size", this->tensorSize.at(tensorName)}}}};
    this->append_trace_event(tensorWriteEvent);
  }
  for (const auto& item : this->memReads) {
    const TensorId& tensorName = item.first;
    const MemActivity& latestReadActivity = item.second.back();
    if (read_before_write_tensors.count(tensorName) == 0) {
      actions_by_tick[latestReadActivity.end].last_reads.push_back(
          &tensorName);
    }
    json tensorLastReadEvent = {
        {"name", tensorName},
        {"cat", "tensorLifetime"},
        {"ph", "E"},
        {"ts", 1e-3 * latestReadActivity.end},
        {"pid", this->sysId + 1000000ul},
        {"tid", this->tensorMapId.at(tensorName)},
        {"args", json{{"size", this->tensorSize.at(tensorName)}}}};
    this->append_trace_event(tensorLastReadEvent);
  }

  // Build the memory timeline as a counter event so that it appears as a line
  // chart.  Unlike the old implementation, this is a single sweep: only the
  // current live tensor set, byte total, and peak are retained.
  std::unordered_set<TensorId> live_tensors;
  uint64_t current_bytes = 0ul;
  uint64_t peak_bytes = 0ul;
  for (const auto& [tick, actions] : actions_by_tick) {
    apply_timeline_actions(actions, this->tensorSize, live_tensors,
                           current_bytes);
    peak_bytes = std::max(peak_bytes, current_bytes);

    // Convert bytes to mebibytes (1 MiB = 1024*1024 bytes)
    double totalSizeMB = static_cast<double>(current_bytes) / (1024.0 * 1024.0);
    json memoryTimelineEvent = {
        {"name", "GPU Memory Usage (MiB)"},
        {"cat", "GPU Memory"},
        {"ph", "C"},
        {"ts", 1e-3 * tick},
        {"pid", this->sysId + 2000000ul},
        {"args", json{{"Memory_MiB", totalSizeMB}}}};
    this->append_trace_event(memoryTimelineEvent);
  }
  this->peak_memory_usage_ = peak_bytes;
  this->has_timeline_tick_ = !actions_by_tick.empty();
  if (this->has_timeline_tick_) {
    this->last_timeline_tick_ = actions_by_tick.rbegin()->first;
  }
  
  // Build the tensor lifetime heatmap after building the memory timeline
  this->buildTensorLifetimeHeatmap();
}

void LocalMemUsageTracker::buildTensorLifetimeHeatmap() {
  // Calculate lifetime for each tensor
  std::vector<std::tuple<TensorId, Tick, Tick, uint64_t>> tensorLifetimes; // tensor, start, end, size
  bool warned_backwards_lifetime = false;

  for (const auto& item : this->memWrites) {
    const TensorId& tensorName = item.first;
    Tick start = item.second.start;
    Tick end;

    // Find the last read time
    if (this->memReads.find(tensorName) != this->memReads.end() && !this->memReads.at(tensorName).empty()) {
      end = this->memReads.at(tensorName).back().end;
    } else {
      // No reads; tensor didn't end. Use simulation's last tick if available
      if (this->has_timeline_tick_) {
          end = this->last_timeline_tick_;
      } else {
          end = item.second.end;
      }
    }

    // Tolerated-order ETs may report reads completing before the real
    // write begins; unsigned Tick would wrap the (end - start) lifetime
    // around and poison the shared min/max scale of the whole heatmap.
    // Clamp such a tensor to a zero-duration row instead.
    if (end < start) {
      if (!warned_backwards_lifetime) {
        AstraSim::LoggerFactory::get_logger("workload::LocalMemUsageTracker")
            ->warn("tensor lifetime heatmap: last read end={} precedes "
                   "write start={} for tensor '{}' (tolerated-order "
                   "input); clamping such rows to zero duration",
                   end, start, tensorName);
        warned_backwards_lifetime = true;
      }
      end = start;
    }

    uint64_t size = this->tensorSize.at(tensorName);
    tensorLifetimes.emplace_back(tensorName, start, end, size);
  }
  
  // Sort tensors by lifetime duration (longest first)
  std::sort(tensorLifetimes.begin(), tensorLifetimes.end(), 
    [](const auto& a, const auto& b) {
      Tick durationA = std::get<2>(a) - std::get<1>(a);
      Tick durationB = std::get<2>(b) - std::get<1>(b);
      return durationA > durationB; // Longest lifetime first
    });
  
  // Generate heatmap events for Perfetto
  const uint64_t heatmapProcessId = this->sysId + 3000000ul; // Use a different process ID for the heatmap view
  
  // Add a process name metadata event to label the heatmap view in Perfetto
  json processNameEvent = {
    {"name", "process_name"},
    {"ph", "M"},  // Metadata event
    {"pid", heatmapProcessId},
    {"args", json{{"name", "Tensor Lifetime Heap"}}}
  };
  this->append_trace_event(processNameEvent);
  
  // Add thread name metadata for the legend/scale
  json threadNameEvent = {
    {"name", "thread_name"},
    {"ph", "M"},  // Metadata event
    {"pid", heatmapProcessId},
    {"tid", 0},
    {"args", json{{"name", "Longest Lifetime → Shortest Lifetime"}}}
  };
  this->append_trace_event(threadNameEvent);
  
  // Every tensor contributes one heatmap row (the legend maps row index to
  // lifetime rank); the count below is the full lifetime list.
  int count = static_cast<int>(tensorLifetimes.size());
  uint64_t minLifetime = std::numeric_limits<uint64_t>::max();
  uint64_t maxLifetime = 0;
  for (int i = 0; i < count; i++) {
      uint64_t duration = std::get<2>(tensorLifetimes[i]) - std::get<1>(tensorLifetimes[i]);
      if (duration < minLifetime) minLifetime = duration;
      if (duration > maxLifetime) maxLifetime = duration;
  }
  
  // Generate color gradient for size visualization
  auto getSizeColor = [](uint64_t size, uint64_t maxSize) -> std::string {
    // Simple heat gradient: small tensors are blue, large are red
    //
    // R4b (frozen plan §6.3-8, 2026-08-29): intensity is provably in
    // [0, 255] -- every caller scales by maxTensorSize, the maximum over
    // all tensorSize entries (>= every size passed here), so the ratio is
    // in [0, 1]. The explicit clamp plus UNSIGNED arguments additionally
    // (a) pin the degenerate all-zero case (maxSize == 0 used to divide
    // by zero and cast NaN -- undefined behavior) to a well-defined
    // blue, and (b) bound snprintf's "#RRGGBB" output to exactly 7 chars
    // + NUL in the 8-byte buffer, eliminating the old sprintf
    // format-overflow warning. Byte output for every well-defined input
    // is unchanged: truncation semantics and the %02X field widths are
    // preserved exactly.
    const double ratio =
        (maxSize == 0)
            ? 0.0
            : std::min(1.0, static_cast<double>(size) /
                                static_cast<double>(maxSize));
    unsigned int intensity = static_cast<unsigned int>(ratio * 255.0);
    if (intensity > 255u) {
      intensity = 255u;  // defensive clamp; ratio <= 1.0 keeps it <= 255
    }
    char color[8];
    std::snprintf(color, sizeof(color), "#%02X%02X%02X", intensity, 100u,
                  255u - intensity);
    return std::string(color);
  };
  
  // Find maximum tensor size for color scaling
  uint64_t maxTensorSize = 0;
  for (const auto& item : this->tensorSize) {
    maxTensorSize = std::max(maxTensorSize, item.second);
  }
  
  // Generate the heatmap events using tensor lifetime to compute heap position
  for (int i = 0; i < count; i++) {
    const auto& [tensorName, start, end, size] = tensorLifetimes[i];
    uint64_t duration = end - start;
    int heapPos = 0;
    if (maxLifetime != minLifetime) {
      heapPos = static_cast<int>((static_cast<double>(duration - minLifetime) / (maxLifetime - minLifetime)) * (count - 1));
    }
    std::string color = getSizeColor(size, maxTensorSize);
    double sizeMB = static_cast<double>(size) / (1024.0 * 1024.0);
    std::string displayName = tensorName;
    if (displayName.length() > 20) {
      displayName = displayName.substr(0, 17) + "...";
    }
    displayName += " (" + std::to_string(sizeMB).substr(0, 5) + " MiB)";

    json heatmapEvent = {
      {"name", displayName},
      {"cat", "tensorHeatmap"},
      {"ph", "X"},
      {"ts", 1e-3 * start},
      {"dur", 1e-3 * duration},
      {"pid", heatmapProcessId},
      {"tid", heapPos},
      {"cname", color},
      {"args", json{
        {"tensor_name", tensorName},
        {"size_bytes", size},
        {"size_mib", sizeMB},
        {"lifetime_ns", duration},
        {"position", heapPos}
      }}
    };
    this->append_trace_event(heatmapEvent);
  }
  
  // Add additional metadata to describe the view
  json heatmapInfoEvent = {
    {"name", "Tensor Lifetime Heatmap"},
    {"cat", "tensorHeatmap"},
    {"ph", "i"},  // Instant event
    {"ts", 0},  // Start of trace
    {"pid", heatmapProcessId},
    {"s", "p"},  // Process scoped
    {"args", json{
      {"description", "Tensors arranged by lifetime duration (longest at bottom)"},
      {"total_tensors", tensorLifetimes.size()},
      {"displayed_tensors", count}
    }}
  };
  this->append_trace_event(heatmapInfoEvent);
}

std::tuple<float, std::string> LocalMemUsageTracker::getPeakMemUsageFormatted() const {
  const uint64_t peakMemUsage = this->peak_memory_usage_;

  float value;
  std::string unit;

  if (peakMemUsage < 1024ull) {
      value = static_cast<float>(peakMemUsage);
      unit = "B";
  } else if (peakMemUsage < 1024ull * 1024) {
      value = static_cast<float>(peakMemUsage) / 1024.0f;
      unit = "KiB";
  } else if (peakMemUsage < 1024ull * 1024 * 1024) {
      value = static_cast<float>(peakMemUsage) / (1024.0f * 1024);
      unit = "MiB";
  } else if (peakMemUsage < 1024ull * 1024 * 1024 * 1024) {
      value = static_cast<float>(peakMemUsage) / (1024.0f * 1024 * 1024);
      unit = "GiB";
  } else {
      value = static_cast<float>(peakMemUsage) / (1024.0f * 1024 * 1024 * 1024);
      unit = "TiB";
  }

  return std::make_tuple(value, unit);
}
LocalMemUsageTracker::~LocalMemUsageTracker() {
  this->memReads.clear();
  this->memWrites.clear();
  this->tensorSize.clear();
  this->activityStartTime.clear();
  this->tensorMapId.clear();
  this->close_trace_spool();
}
