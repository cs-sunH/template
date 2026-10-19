/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

/*
Capi.cc -- the C ABI implementation (contract §4; design doc §8.1).

Every entry point is exception-free at the boundary: all C++ exceptions are
caught here and turned into NULL/-1 plus the die_last_error() slot (a single
process-wide string; see include/DieCApi.h for the thread-safety caveat).
JSON buffers returned by die_hardware_summary()/die_estimate_batch() are
allocated by this library and must be released with die_free_result().

CostResult serialization field names (frozen, snake_case; stage A3 report):
  profile, total_seconds, pe_seconds, vec_seconds, noc_seconds,
  mem_seconds, launch_seconds, launch_instances, flops, eqops,
  matrix_actual_bytes, vector_actual_bytes, matrix_service_units,
  vector_service_units, noc_link_bytes_total, noc_edge_bytes_max,
  sram_peak_per_core, fused, fallback, total_ns_suggested
*/

#include "DieCApi.h"

#include "DieModel.hh"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <list>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <unistd.h>

#include <json/json.hpp>

namespace {

// ---------------------------------------------------------------------------
// Last-error slot (single process-wide, documented as not thread-safe)

std::string& last_error_slot() {
  static std::string slot;
  return slot;
}

void set_last_error(const std::string& message) { last_error_slot() = message; }

// ---------------------------------------------------------------------------
// Persistent estimate cache (C2, joint提速改造 2026-10-17)
//
// plan_cache_key -> CostResult value copy, opt-in per model handle via the
// env var DIE_ESTIMATE_CACHE_PATH: unset (or empty) keeps est_cache null and
// every code path below is byte-for-byte the pre-cache behavior.  Loaded at
// die_create_model, saved (best effort, only when dirty) at die_destroy_model.
// Fail-closed throughout: absent file, fingerprint mismatch, or ANY parse
// anomaly discards the whole file; save failures are swallowed.  No locks:
// per-handle state under the DieCApi.h single-thread contract.  Key reuse:
// the estimate path computes exactly the cached_compile key (parse ->
// DieModel::plan_cache_key), so key semantics inherit the library's own
// versioning discipline; the FILE-level fingerprint (configs/VERSION +
// compiler/flag digest + source SHA256 fold, stamped at configure time)
// invalidates the whole store when the .so is rebuilt differently.

constexpr std::size_t kEstimateCacheCap = 4096;
constexpr std::uint32_t kEstimateCacheFmt = 1;
const char kEstimateCacheMagic[8] = {'D', 'I', 'E', 'E', 'S', 'T', 'C', '1'};

std::uint64_t fnv1a64(const char* data, std::size_t n) {
  std::uint64_t h = 14695981039346656037ULL;
  for (std::size_t i = 0; i < n; ++i) {
    h ^= static_cast<std::uint64_t>(static_cast<unsigned char>(data[i]));
    h *= 1099511628211ULL;
  }
  return h;
}

std::string fnv1a_16hex(const std::string& s) {
  const std::uint64_t h = fnv1a64(s.data(), s.size());
  static const char* kHex = "0123456789abcdef";
  std::string out(16, '0');
  std::uint64_t v = h;
  for (int i = 15; i >= 0; --i) {
    out[static_cast<std::size_t>(i)] = kHex[v & 0xF];
    v >>= 4;
  }
  return out;
}

// Build fingerprint rendered as 16 hex digits (same primitive as the plan
// keys).  Inputs are stamped by CMakeLists.txt: configs/VERSION content, a
// compiler/build-type/flags digest, and a per-source SHA256 fold so any edit
// to the estimate-relevant sources invalidates persisted stores.
std::string estimate_fingerprint() {
  std::string s = "die_model-est-fp-v1|";
  s += DIE_MODEL_VERSION_STAMP;
  s += "|";
  s += DIE_MODEL_BUILD_STAMP;
  return fnv1a_16hex(s);
}

// LRU over (16-hex plan key -> CostResult copy); eviction mirrors
// DieModel::PlanCache (CostModel.cc).  insert_loaded() warms without marking
// dirty so read-only handles (config_resolver's summary-only models) never
// rewrite the file.
class EstimateCache {
 public:
  explicit EstimateCache(std::size_t capacity)
      : capacity_(capacity == 0 ? 1 : capacity) {}

  bool get(const std::string& key, DieModel::CostResult* out) {
    auto it = index_.find(key);
    if (it == index_.end()) {
      return false;
    }
    lru_.splice(lru_.begin(), lru_, it->second);
    *out = it->second->second;
    return true;
  }

  void put(const std::string& key, const DieModel::CostResult& value) {
    insert_loaded(key, value);
    dirty_ = true;
  }

  void insert_loaded(const std::string& key,
                     const DieModel::CostResult& value) {
    auto it = index_.find(key);
    if (it != index_.end()) {
      it->second->second = value;
      lru_.splice(lru_.begin(), lru_, it->second);
      return;
    }
    lru_.emplace_front(key, value);
    index_[key] = lru_.begin();
    while (lru_.size() > capacity_) {
      index_.erase(lru_.back().first);
      lru_.pop_back();
    }
  }

  bool dirty() const { return dirty_; }

  template <typename Fn>
  void for_each_lru_oldest_first(Fn f) const {
    for (auto it = lru_.rbegin(); it != lru_.rend(); ++it) {
      f(it->first, it->second);
    }
  }

 private:
  std::size_t capacity_;
  bool dirty_ = false;
  std::list<std::pair<std::string, DieModel::CostResult>> lru_;
  std::unordered_map<
      std::string,
      std::list<std::pair<std::string, DieModel::CostResult>>::iterator>
      index_;
};

// --- little-endian binary helpers (exact, byte-addressed, locale-free) ----

void put_u32le(std::string& out, std::uint32_t v) {
  for (int i = 0; i < 4; ++i) {
    out.push_back(static_cast<char>((v >> (8 * i)) & 0xFFu));
  }
}

void put_u64le(std::string& out, std::uint64_t v) {
  for (int i = 0; i < 8; ++i) {
    out.push_back(static_cast<char>((v >> (8 * i)) & 0xFFu));
  }
}

void put_f64le(std::string& out, double d) {
  std::uint64_t bits = 0;
  std::memcpy(&bits, &d, sizeof bits);  // exact IEEE-754 bit pattern
  put_u64le(out, bits);
}

struct Cursor {
  const std::string& buf;
  std::size_t pos = 0;
  bool ok = true;

  void take(std::size_t n, std::string* out) {
    if (!ok || pos + n > buf.size() || pos + n < pos) {
      ok = false;
      return;
    }
    if (out != nullptr) {
      out->assign(buf, pos, n);
    }
    pos += n;
  }

  std::uint8_t u8() {
    if (!ok || pos + 1 > buf.size()) {
      ok = false;
      return 0;
    }
    return static_cast<std::uint8_t>(static_cast<unsigned char>(buf[pos++]));
  }

  std::uint32_t u32() {
    if (!ok || pos + 4 > buf.size()) {
      ok = false;
      return 0;
    }
    std::uint32_t v = 0;
    for (int i = 0; i < 4; ++i) {
      v |= static_cast<std::uint32_t>(
               static_cast<unsigned char>(buf[pos + i]))
           << (8 * i);
    }
    pos += 4;
    return v;
  }

  std::uint64_t u64() {
    if (!ok || pos + 8 > buf.size()) {
      ok = false;
      return 0;
    }
    std::uint64_t v = 0;
    for (int i = 0; i < 8; ++i) {
      v |= static_cast<std::uint64_t>(
               static_cast<unsigned char>(buf[pos + i]))
           << (8 * i);
    }
    pos += 8;
    return v;
  }

  double f64() {
    const std::uint64_t bits = u64();
    double d = 0.0;
    std::memcpy(&d, &bits, sizeof d);  // bit-exact restore (incl. -0.0/NaN)
    return d;
  }
};

// File layout (little-endian): magic[8] "DIEESTC1" | u32 fmt=1 | u32 fp_len
// | fp bytes | u32 count | count * entry | u64 fnv1a64 checksum over all
// preceding bytes.  Entry: u8 key_len(=16) | key ascii | u8 profile_idx |
// 15 * f64 (field order frozen from Capi.cc cost_result_json) | u64
// launch_instances | u8 fused | u8 fallback | u64 total_ns_suggested.
// Entries are stored oldest-first; load replays them into the LRU.
std::unique_ptr<EstimateCache> load_estimate_cache(
    const std::string& path, const std::string& fingerprint) {
  auto empty = [] {
    return std::make_unique<EstimateCache>(kEstimateCacheCap);
  };
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    return empty();
  }
  std::string buf((std::istreambuf_iterator<char>(in)),
                  std::istreambuf_iterator<char>());
  if (in.bad()) {
    return empty();
  }
  Cursor c{buf};
  std::string magic;
  c.take(8, &magic);
  if (!c.ok || magic != std::string(kEstimateCacheMagic, 8)) return empty();
  if (c.u32() != kEstimateCacheFmt) return empty();
  const std::uint32_t fp_len = c.u32();
  if (!c.ok || fp_len == 0 || fp_len > 256) return empty();
  std::string fp;
  c.take(fp_len, &fp);
  if (!c.ok || fp != fingerprint) return empty();  // stale build: drop all
  const std::uint32_t count = c.u32();
  if (!c.ok || count > kEstimateCacheCap) return empty();
  struct Entry {
    std::string key;
    DieModel::CostResult value;
  };
  std::vector<Entry> entries;
  entries.reserve(count);
  for (std::uint32_t i = 0; i < count && c.ok; ++i) {
    const std::uint8_t key_len = c.u8();
    if (!c.ok || key_len == 0 || key_len > 64) {
      c.ok = false;
      break;
    }
    Entry e;
    c.take(key_len, &e.key);
    const std::uint8_t profile_idx = c.u8();
    if (c.ok && profile_idx <= 1) {  // DieTypes.hh Profile enum order
      e.value.profile = profile_idx == 0
          ? DieModel::Profile::MeshDieV1
          : DieModel::Profile::Tx8ReferenceV1;
    } else {
      c.ok = false;
      break;
    }
    e.value.total_seconds = c.f64();
    e.value.pe_seconds = c.f64();
    e.value.vec_seconds = c.f64();
    e.value.noc_seconds = c.f64();
    e.value.mem_seconds = c.f64();
    e.value.launch_seconds = c.f64();
    e.value.flops = c.f64();
    e.value.eqops = c.f64();
    e.value.matrix_actual_bytes = c.f64();
    e.value.vector_actual_bytes = c.f64();
    e.value.matrix_service_units = c.f64();
    e.value.vector_service_units = c.f64();
    e.value.noc_link_bytes_total = c.f64();
    e.value.noc_edge_bytes_max = c.f64();
    e.value.sram_peak_per_core = c.f64();
    e.value.launch_instances = c.u64();
    e.value.fused = c.u8() != 0;
    e.value.fallback = c.u8() != 0;
    e.value.total_ns_suggested = c.u64();
    if (c.ok) {
      entries.push_back(std::move(e));
    }
  }
  const std::uint64_t stored_sum = c.u64();
  if (!c.ok || c.pos != buf.size()) return empty();  // short / trailing junk
  if (fnv1a64(buf.data(), c.pos - 8) != stored_sum) return empty();
  auto cache = empty();
  for (const Entry& e : entries) {
    cache->insert_loaded(e.key, e.value);
  }
  return cache;
}

// Best-effort save: temp file + atomic rename, swallows every failure so it
// can run from the non-throwing die_destroy_model path.
void save_estimate_cache(const EstimateCache& cache, const std::string& path,
                         const std::string& fingerprint) noexcept {
  try {
    std::string entries;
    std::size_t cnt = 0;
    cache.for_each_lru_oldest_first(
        [&](const std::string& key, const DieModel::CostResult& r) {
          entries.push_back(static_cast<char>(key.size()));
          entries.append(key);
          entries.push_back(static_cast<char>(
              r.profile == DieModel::Profile::Tx8ReferenceV1 ? 1 : 0));
          put_f64le(entries, r.total_seconds);
          put_f64le(entries, r.pe_seconds);
          put_f64le(entries, r.vec_seconds);
          put_f64le(entries, r.noc_seconds);
          put_f64le(entries, r.mem_seconds);
          put_f64le(entries, r.launch_seconds);
          put_f64le(entries, r.flops);
          put_f64le(entries, r.eqops);
          put_f64le(entries, r.matrix_actual_bytes);
          put_f64le(entries, r.vector_actual_bytes);
          put_f64le(entries, r.matrix_service_units);
          put_f64le(entries, r.vector_service_units);
          put_f64le(entries, r.noc_link_bytes_total);
          put_f64le(entries, r.noc_edge_bytes_max);
          put_f64le(entries, r.sram_peak_per_core);
          put_u64le(entries, r.launch_instances);
          entries.push_back(r.fused ? 1 : 0);
          entries.push_back(r.fallback ? 1 : 0);
          put_u64le(entries, r.total_ns_suggested);
          ++cnt;
        });
    std::string out;
    out.append(kEstimateCacheMagic, 8);
    put_u32le(out, kEstimateCacheFmt);
    put_u32le(out, static_cast<std::uint32_t>(fingerprint.size()));
    out.append(fingerprint);
    put_u32le(out, static_cast<std::uint32_t>(cnt));
    out.append(entries);
    put_u64le(out, fnv1a64(out.data(), out.size()));
    const std::string tmp =
        path + ".tmp." + std::to_string(static_cast<long>(::getpid()));
    {
      std::ofstream f(tmp, std::ios::binary | std::ios::trunc);
      if (!f) {
        return;
      }
      f.write(out.data(), static_cast<std::streamsize>(out.size()));
      if (!f) {
        f.close();
        std::remove(tmp.c_str());
        return;
      }
    }
    if (std::rename(tmp.c_str(), path.c_str()) != 0) {
      std::remove(tmp.c_str());
    }
  } catch (...) {
    // Persistence is best effort; never propagate out of the destroy path.
  }
}

// ---------------------------------------------------------------------------
// Handles

struct ModelHandle {
  const std::uint64_t magic = 0x4449454d4f444c31ULL;  // "DIEMODL1"
  DieModel::Hardware hw;
  DieModel::PlanCache cache;
  // C2 persistent estimate cache (opt-in via DIE_ESTIMATE_CACHE_PATH): both
  // stay null/empty when the env is unset -- zero overhead, zero behavior
  // change on the default path.
  std::unique_ptr<EstimateCache> est_cache;
  std::string est_cache_path;
  ModelHandle() : cache(4096) {}
};

struct ExecHandle {
  const std::uint64_t magic = 0x4449454558454331ULL;  // "DIEEXEC1"
  std::shared_ptr<const DieModel::DiePlan> plan;
  std::unique_ptr<DieModel::Execution> exec;
};

ModelHandle* as_model(void* p) {
  auto* m = static_cast<ModelHandle*>(p);
  return (m && m->magic == 0x4449454d4f444c31ULL) ? m : nullptr;
}

ExecHandle* as_exec(void* p) {
  auto* e = static_cast<ExecHandle*>(p);
  return (e && e->magic == 0x4449454558454331ULL) ? e : nullptr;
}

// ---------------------------------------------------------------------------
// Serialization

nlohmann::json cost_result_json(const DieModel::CostResult& r) {
  nlohmann::json j;
  j["profile"] = DieModel::to_string(r.profile);
  j["total_seconds"] = r.total_seconds;
  j["pe_seconds"] = r.pe_seconds;
  j["vec_seconds"] = r.vec_seconds;
  j["noc_seconds"] = r.noc_seconds;
  j["mem_seconds"] = r.mem_seconds;
  j["launch_seconds"] = r.launch_seconds;
  j["launch_instances"] = r.launch_instances;
  j["flops"] = r.flops;
  j["eqops"] = r.eqops;
  j["matrix_actual_bytes"] = r.matrix_actual_bytes;
  j["vector_actual_bytes"] = r.vector_actual_bytes;
  j["matrix_service_units"] = r.matrix_service_units;
  j["vector_service_units"] = r.vector_service_units;
  j["noc_link_bytes_total"] = r.noc_link_bytes_total;
  j["noc_edge_bytes_max"] = r.noc_edge_bytes_max;
  j["sram_peak_per_core"] = r.sram_peak_per_core;
  j["fused"] = r.fused;
  j["fallback"] = r.fallback;
  j["total_ns_suggested"] = r.total_ns_suggested;
  return j;
}

const char* internal_slice_class_name(DieModel::InternalSliceClass value) {
  using Class = DieModel::InternalSliceClass;
  switch (value) {
    case Class::Idle: return "idle";
    case Class::PurePe: return "pure_pe";
    case Class::PureVec: return "pure_vec";
    case Class::InternalNoc: return "internal_noc";
    case Class::MemoryLatency: return "memory_latency";
    case Class::MemoryService: return "memory_service";
    case Class::MemNocResidual: return "mem_noc_residual";
    case Class::MixedFrontier: return "mixed_frontier";
    case Class::Unknown: return "unknown";
  }
  return "unknown";
}

const char* stage_resource_name(DieModel::StageResource value) {
  using Resource = DieModel::StageResource;
  switch (value) {
    case Resource::PE: return "pe";
    case Resource::VEC: return "vec";
    case Resource::NOC: return "noc";
    case Resource::MEM: return "memory";
    case Resource::MEM_NOC: return "mem_noc";
    case Resource::TIMER: return "timer";
  }
  return "unknown";
}

nlohmann::json internal_slice_json(const DieModel::InternalSliceInfo& info) {
  nlohmann::json j;
  j["action_token"] = info.action_token.valid
      ? nlohmann::json{{"template_key", info.action_token.template_key},
                       {"flat_instance_id", info.action_token.flat_instance_id},
                       {"frontier_generation", info.action_token.frontier_generation}}
      : nlohmann::json(nullptr);
  j["template_key"] = info.template_key;
  j["classification"] = internal_slice_class_name(info.classification);
  j["now_ns"] = info.now_ns;
  j["has_next_boundary"] = info.has_next_boundary;
  j["next_boundary_ns"] = info.has_next_boundary
      ? nlohmann::json(info.next_boundary_ns) : nlohmann::json(nullptr);
  j["partial_service_supported"] = info.partial_service_supported;
  j["reason"] = info.reason;
  j["active_stages"] = nlohmann::json::array();
  for (const auto& stage : info.active_stages) {
    j["active_stages"].push_back({
        {"stage_id", stage.stage_id}, {"wave_index", stage.wave_index},
        {"resource", stage_resource_name(stage.resource)},
        {"start_ns", stage.start_ns}, {"end_ns", stage.end_ns},
        {"original_service_ns", stage.original_service_ns},
        {"remaining_service_ns", stage.remaining_service_ns},
        {"original_work", stage.original_work},
        {"remaining_work", stage.remaining_work},
        {"memory_actual_bytes", stage.memory_actual_bytes},
        {"memory_service_units", stage.memory_service_units},
        {"memory_pending", stage.memory_pending},
        {"hardware_min_ns", stage.hardware_min_ns}});
  }
  return j;
}

char* alloc_cstring(const std::string& s) {
  char* buf = new char[s.size() + 1];
  std::memcpy(buf, s.c_str(), s.size() + 1);
  return buf;
}

// Compiles one spec through the model's plan cache.  Throws DieModelError /
// nlohmann parse errors to the caller's catch.
std::shared_ptr<const DieModel::DiePlan> cached_compile(
    ModelHandle& m, const nlohmann::json& spec) {
  std::shared_ptr<const DieModel::DieOpSpec> parsed =
      DieModel::parse_die_op_spec(spec);
  const std::string key = DieModel::plan_cache_key(*parsed, m.hw);
  std::shared_ptr<const DieModel::DiePlan> plan = m.cache.get(key);
  if (!plan) {
    plan = DieModel::compile(*parsed, m.hw);
    m.cache.put(key, plan);
  }
  return plan;
}

// C2: the die_estimate_batch body per spec.  Computes exactly the
// cached_compile key (parse -> plan_cache_key), consults the persistent
// estimate cache first -- a hit returns the stored CostResult directly and
// skips compile + estimate_alone -- and on miss takes the original
// compile -> plan cache -> estimate_alone road, recording the result.
// Throws exactly what the old inline expression threw (parse/compile/
// estimate errors land in the per-spec catch unchanged).
nlohmann::json estimate_result_json(ModelHandle& m,
                                     const nlohmann::json& spec) {
  std::shared_ptr<const DieModel::DieOpSpec> parsed =
      DieModel::parse_die_op_spec(spec);
  const std::string key = DieModel::plan_cache_key(*parsed, m.hw);
  if (m.est_cache != nullptr) {
    DieModel::CostResult hit;
    if (m.est_cache->get(key, &hit)) {
      return cost_result_json(hit);
    }
  }
  std::shared_ptr<const DieModel::DiePlan> plan = m.cache.get(key);
  if (!plan) {
    plan = DieModel::compile(*parsed, m.hw);
    m.cache.put(key, plan);
  }
  const DieModel::CostResult fresh = DieModel::estimate_alone(*plan);
  if (m.est_cache != nullptr) {
    m.est_cache->put(key, fresh);
  }
  return cost_result_json(fresh);
}

}  // namespace

// ---------------------------------------------------------------------------
// Model lifecycle

void* die_create_model(const char* config_json) {
  try {
    if (config_json == nullptr) {
      set_last_error("die_create_model: config_json is NULL");
      return nullptr;
    }
    const nlohmann::json j = nlohmann::json::parse(config_json);
    auto* model = new ModelHandle();
    try {
      model->hw = DieModel::make_hardware(j);
    } catch (...) {
      delete model;
      throw;
    }
    // C2: opt-in persistent estimate cache.  Env unset/empty -> est_cache
    // stays null and the estimate path is bit-identical to the previous
    // behavior.  A load failure inside load_estimate_cache is fail-closed
    // (empty cache), never a create failure.
    const char* est_cache_path = std::getenv("DIE_ESTIMATE_CACHE_PATH");
    if (est_cache_path != nullptr && est_cache_path[0] != '\0') {
      model->est_cache_path = est_cache_path;
      model->est_cache =
          load_estimate_cache(model->est_cache_path, estimate_fingerprint());
    }
    return model;
  } catch (const std::exception& e) {
    set_last_error(std::string("die_create_model: ") + e.what());
    return nullptr;
  } catch (...) {
    set_last_error("die_create_model: unknown error");
    return nullptr;
  }
}

const char* die_hardware_summary(void* model) {
  try {
    ModelHandle* m = as_model(model);
    if (m == nullptr) {
      set_last_error("die_hardware_summary: invalid model handle");
      return nullptr;
    }
    return alloc_cstring(DieModel::hardware_summary_json(m->hw).dump());
  } catch (const std::exception& e) {
    set_last_error(std::string("die_hardware_summary: ") + e.what());
    return nullptr;
  } catch (...) {
    set_last_error("die_hardware_summary: unknown error");
    return nullptr;
  }
}

const char* die_estimate_batch(void* model, const char* specs_json) {
  try {
    ModelHandle* m = as_model(model);
    if (m == nullptr) {
      set_last_error("die_estimate_batch: invalid model handle");
      return nullptr;
    }
    if (specs_json == nullptr) {
      set_last_error("die_estimate_batch: specs_json is NULL");
      return nullptr;
    }
    const nlohmann::json input = nlohmann::json::parse(specs_json);
    if (!input.is_object() || input.size() != 1 || !input.contains("specs") ||
        !input.at("specs").is_array()) {
      throw DieModel::DieModelError(
          "specs_json must be exactly {\"specs\":[spec,...]}");
    }
    const nlohmann::json& specs = input.at("specs");
    nlohmann::json results = nlohmann::json::array();
    nlohmann::json errors = nlohmann::json::array();
    for (const nlohmann::json& spec : specs) {
      std::string error;
      nlohmann::json result;
      try {
        result = estimate_result_json(*m, spec);
      } catch (const std::exception& e) {
        error = e.what();
      }
      results.push_back(error.empty() ? result : nlohmann::json());
      errors.push_back(error.empty() ? nlohmann::json()
                                     : nlohmann::json(error));
    }
    nlohmann::json out;
    out["results"] = std::move(results);
    out["errors"] = std::move(errors);
    return alloc_cstring(out.dump());
  } catch (const std::exception& e) {
    set_last_error(std::string("die_estimate_batch: ") + e.what());
    return nullptr;
  } catch (...) {
    set_last_error("die_estimate_batch: unknown error");
    return nullptr;
  }
}

void* die_create_execution(void* model, const char* spec_json) {
  try {
    ModelHandle* m = as_model(model);
    if (m == nullptr) {
      set_last_error("die_create_execution: invalid model handle");
      return nullptr;
    }
    if (spec_json == nullptr) {
      set_last_error("die_create_execution: spec_json is NULL");
      return nullptr;
    }
    const nlohmann::json spec = nlohmann::json::parse(spec_json);
    auto* exec = new ExecHandle();
    try {
      exec->plan = cached_compile(*m, spec);
      exec->exec = std::make_unique<DieModel::Execution>(exec->plan);
    } catch (...) {
      delete exec;
      throw;
    }
    return exec;
  } catch (const std::exception& e) {
    set_last_error(std::string("die_create_execution: ") + e.what());
    return nullptr;
  } catch (...) {
    set_last_error("die_create_execution: unknown error");
    return nullptr;
  }
}

// ---------------------------------------------------------------------------
// Execution protocol

int die_next_action(void* exec, char* out, size_t cap) {
  try {
    ExecHandle* e = as_exec(exec);
    if (e == nullptr) {
      set_last_error("die_next_action: invalid execution handle");
      return -1;
    }
    if (out == nullptr && cap != 0) {
      set_last_error("die_next_action: out is NULL");
      return -1;
    }
    const DieModel::ExecutionAction a = e->exec->next_action();
    nlohmann::json j;
    switch (a.kind) {
      case DieModel::ExecutionAction::Kind::Timer:
        j["kind"] = "timer";
        j["ns"] = a.ns;
        break;
      case DieModel::ExecutionAction::Kind::MemoryRequest:
        j["kind"] = "memory";
        j["stage_id"] = a.stage_id;
        j["actual_bytes"] = a.actual_bytes;
        j["service_units"] = a.service_units;
        break;
      case DieModel::ExecutionAction::Kind::Done:
        j["kind"] = "done";
        j["elapsed_ns"] = a.ns;
        break;
    }
    const std::string s = j.dump();
    if (s.size() + 1 > cap) {
      set_last_error("die_next_action: buffer too small (need " +
                     std::to_string(s.size() + 1) + " bytes)");
      return -2;
    }
    if (out != nullptr) {
      std::memcpy(out, s.c_str(), s.size() + 1);
    }
    return 0;
  } catch (const std::exception& e) {
    set_last_error(std::string("die_next_action: ") + e.what());
    return -1;
  } catch (...) {
    set_last_error("die_next_action: unknown error");
    return -1;
  }
}

int die_advance_time(void* exec, unsigned long long ns) {
  try {
    ExecHandle* e = as_exec(exec);
    if (e == nullptr) {
      set_last_error("die_advance_time: invalid execution handle");
      return -1;
    }
    e->exec->advance_time(static_cast<std::uint64_t>(ns));
    return 0;
  } catch (const std::exception& e) {
    set_last_error(std::string("die_advance_time: ") + e.what());
    return -1;
  } catch (...) {
    set_last_error("die_advance_time: unknown error");
    return -1;
  }
}

int die_complete_memory(void* exec, unsigned long long stage_id) {
  try {
    ExecHandle* e = as_exec(exec);
    if (e == nullptr) {
      set_last_error("die_complete_memory: invalid execution handle");
      return -1;
    }
    e->exec->complete_memory(static_cast<std::uint64_t>(stage_id));
    return 0;
  } catch (const std::exception& e) {
    set_last_error(std::string("die_complete_memory: ") + e.what());
    return -1;
  } catch (...) {
    set_last_error("die_complete_memory: unknown error");
    return -1;
  }
}

const char* die_describe_internal_slice(void* exec) {
  try {
    ExecHandle* e = as_exec(exec);
    if (e == nullptr) {
      set_last_error("die_describe_internal_slice: invalid execution handle");
      return nullptr;
    }
    return alloc_cstring(internal_slice_json(
        e->exec->describe_internal_slice()).dump());
  } catch (const std::exception& e) {
    set_last_error(std::string("die_describe_internal_slice: ") + e.what());
    return nullptr;
  } catch (...) {
    set_last_error("die_describe_internal_slice: unknown error");
    return nullptr;
  }
}

int die_advance_internal_service(void* exec,
                                 const char* plan_template_key,
                                 unsigned long long flat_instance_id,
                                 unsigned long long frontier_generation,
                                 unsigned long long delta_original_ns) {
  try {
    ExecHandle* e = as_exec(exec);
    if (e == nullptr) {
      set_last_error("die_advance_internal_service: invalid execution handle");
      return -1;
    }
    if (plan_template_key == nullptr) {
      set_last_error("die_advance_internal_service: plan_template_key is NULL");
      return -1;
    }
    e->exec->advance_internal_service(DieModel::InternalActionToken{
        plan_template_key, static_cast<std::uint64_t>(flat_instance_id),
        static_cast<std::uint64_t>(frontier_generation), true},
        static_cast<std::uint64_t>(delta_original_ns));
    return 0;
  } catch (const std::exception& e) {
    set_last_error(std::string("die_advance_internal_service: ") + e.what());
    return -1;
  } catch (...) {
    set_last_error("die_advance_internal_service: unknown error");
    return -1;
  }
}

int die_advance_external_wait(void* exec,
                              unsigned long long delta_global_ns,
                              unsigned long long expected_memory_stage) {
  try {
    ExecHandle* e = as_exec(exec);
    if (e == nullptr) {
      set_last_error("die_advance_external_wait: invalid execution handle");
      return -1;
    }
    e->exec->advance_external_wait(
        static_cast<std::uint64_t>(delta_global_ns),
        static_cast<std::uint64_t>(expected_memory_stage));
    return 0;
  } catch (const std::exception& e) {
    set_last_error(std::string("die_advance_external_wait: ") + e.what());
    return -1;
  } catch (...) {
    set_last_error("die_advance_external_wait: unknown error");
    return -1;
  }
}

const char* die_export_progress(void* exec) {
  try {
    ExecHandle* e = as_exec(exec);
    if (e == nullptr) {
      set_last_error("die_export_progress: invalid execution handle");
      return nullptr;
    }
    const DieModel::ExecutionProgress progress = e->exec->export_progress();
    nlohmann::json j;
    j["now_ns"] = progress.now_ns;
    j["elapsed_ns"] = progress.elapsed_ns;
    j["finished"] = progress.finished;
    j["memory_pending"] = progress.memory_pending;
    j["memory_stage_id"] = progress.memory_pending
        ? nlohmann::json(progress.memory_stage_id) : nlohmann::json(nullptr);
    j["memory_instance_id"] = progress.has_memory_instance_id
        ? nlohmann::json(progress.memory_instance_id) : nlohmann::json(nullptr);
    j["pending_memory_actual_bytes"] = progress.pending_memory_actual_bytes;
    j["pending_memory_service_units"] = progress.pending_memory_service_units;
    j["external_memory_wait_elapsed_ns"] =
        progress.external_memory_wait_elapsed_ns;
    j["frontier"] = internal_slice_json(progress.frontier);
    j["stages"] = nlohmann::json::array();
    for (const auto& stage : progress.stages) {
      j["stages"].push_back({{"stage_id", stage.stage_id},
                             {"resource", stage_resource_name(stage.resource)},
                             {"completed", stage.completed},
                             {"total", stage.total},
                             {"remaining_work", stage.remaining_work},
                             {"remaining_noc_bytes", stage.remaining_noc_bytes},
                             {"remaining_actual_bytes", stage.remaining_actual_bytes},
                             {"remaining_service_units", stage.remaining_service_units},
                             {"remaining_internal_service_ns",
                              stage.remaining_internal_service_ns}});
    }
    return alloc_cstring(j.dump());
  } catch (const std::exception& e) {
    set_last_error(std::string("die_export_progress: ") + e.what());
    return nullptr;
  } catch (...) {
    set_last_error("die_export_progress: unknown error");
    return nullptr;
  }
}

void die_destroy_execution(void* exec) {
  delete as_exec(exec);
}

void die_destroy_model(void* model) {
  ModelHandle* m = as_model(model);
  if (m != nullptr && m->est_cache != nullptr && !m->est_cache_path.empty() &&
      m->est_cache->dirty()) {
    save_estimate_cache(*m->est_cache, m->est_cache_path,
                        estimate_fingerprint());
  }
  delete m;
}

void die_free_result(const char* buf) { delete[] const_cast<char*>(buf); }

const char* die_last_error(void) { return last_error_slot().c_str(); }
