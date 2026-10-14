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
#include <cstring>
#include <memory>
#include <string>
#include <utility>
#include <vector>

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
// Handles

struct ModelHandle {
  const std::uint64_t magic = 0x4449454d4f444c31ULL;  // "DIEMODL1"
  DieModel::Hardware hw;
  DieModel::PlanCache cache;
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
        result = cost_result_json(
            DieModel::estimate_alone(*cached_compile(*m, spec)));
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

const char* die_query_template_slice(void* exec,
                                     unsigned int stage_id,
                                     unsigned long long position,
                                     unsigned long long count) {
  try {
    ExecHandle* e = as_exec(exec);
    if (e == nullptr) {
      set_last_error("die_query_template_slice: invalid execution handle");
      return nullptr;
    }
    nlohmann::json out = nlohmann::json::array();
    for (const auto& entry : e->exec->query_template_slice(
             stage_id, static_cast<std::uint64_t>(position),
             static_cast<std::uint64_t>(count))) {
      out.push_back({
          {"stage_id", entry.stage_id}, {"wave_index", entry.wave_index},
          {"resource", stage_resource_name(entry.resource)},
          {"original_work", entry.original_work},
          {"original_service_ns", entry.original_service_ns},
          {"hardware_min_ns", entry.hardware_min_ns}});
    }
    return alloc_cstring(out.dump());
  } catch (const std::exception& e) {
    set_last_error(std::string("die_query_template_slice: ") + e.what());
    return nullptr;
  } catch (...) {
    set_last_error("die_query_template_slice: unknown error");
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

void die_destroy_model(void* model) { delete as_model(model); }

void die_free_result(const char* buf) { delete[] const_cast<char*>(buf); }

const char* die_last_error(void) { return last_error_slot().c_str(); }
