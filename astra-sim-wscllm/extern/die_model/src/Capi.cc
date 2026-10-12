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

void die_destroy_execution(void* exec) {
  delete as_exec(exec);
}

void die_destroy_model(void* model) { delete as_model(model); }

void die_free_result(const char* buf) { delete[] const_cast<char*>(buf); }

const char* die_last_error(void) { return last_error_slot().c_str(); }
