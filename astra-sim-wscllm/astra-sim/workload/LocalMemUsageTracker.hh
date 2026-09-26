#ifndef __LOCAL_MEM_USAGE_TRACKER__
#define __LOCAL_MEM_USAGE_TRACKER__

#include <cstdio>
#include <json/json.hpp>
#include <map>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <tuple>
#include "astra-sim/system/Common.hh"

#include "extern/graph_frontend/chakra/src/feeder_v3/et_feeder.h"

using json = nlohmann::json;

namespace AstraSim {

typedef struct {
  Tick start;
  Tick end;
  std::string nodeName;
  uint64_t nodeId;
} MemActivity;

typedef std::string TensorId;

class LocalMemUsageTracker {
 public:
  LocalMemUsageTracker(uint64_t sysId) : sysId(sysId) {}
  ~LocalMemUsageTracker();
  void recordStart(const std::shared_ptr<Chakra::ETFeederNode> node, Tick tick);
  void recordEnd(const std::shared_ptr<Chakra::ETFeederNode> node, Tick tick);
  void buildMemoryTrace();
  void dumpMemoryTrace(const std::string& filename);
  void buildMemoryTimeline();
  void buildTensorLifetimeHeatmap();
  std::tuple<float, std::string> getPeakMemUsageFormatted() const;
  uint64_t getPeakMemUsage() const;
  
  uint64_t sysId;

 private:
  void recordReads(const std::shared_ptr<Chakra::ETFeederNode> node, Tick start, Tick end);
  void recordWrites(const std::shared_ptr<Chakra::ETFeederNode> node, Tick start, Tick end);
  void reset_trace_spool();
  void close_trace_spool() noexcept;
  void append_trace_event(const json& event);

  uint64_t parseIOInfos(const google::protobuf::RepeatedPtrField<std::string>& values, std::vector<std::tuple<TensorId, uint64_t>>& IOinfos);
  std::unordered_map<TensorId, std::vector<MemActivity>> memReads;
  std::unordered_map<TensorId, MemActivity> memWrites;
  std::unordered_map<TensorId, uint64_t> tensorSize;
  std::unordered_map<uint64_t, Tick> activityStartTime;
  std::unordered_map<TensorId, uint64_t> tensorMapId;
  // Trace events are serialized individually into an anonymous spool.  This
  // preserves the legacy event order and pretty JSON bytes without keeping a
  // second, full nlohmann::json tree alive through report().
  std::FILE* trace_spool_ = nullptr;
  uint64_t trace_spool_event_count_ = 0;
  // The timeline is still an exact single sweep, but it retains only this
  // aggregate result instead of one complete live-tensor snapshot per tick.
  uint64_t peak_memory_usage_ = 0;
  bool has_timeline_tick_ = false;
  Tick last_timeline_tick_ = 0;

  friend struct LocalMemUsageTrackerTestAccess;
};

} // namespace AstraSim

#endif
