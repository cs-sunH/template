/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

/*
capi_test.cc -- stage A3 unit tests of the C ABI (include/DieCApi.h,
src/Capi.cc): model lifecycle, batch estimation, the execution protocol and
every error path, called as pure C (the die_* symbols live only in the
die_model_cabi shared target).
*/

#include "DieCApi.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include <json/json.hpp>

namespace {

struct Checker {
  bool ok = true;
  void expect(bool cond, const std::string& what) {
    if (!cond) {
      ok = false;
      std::fprintf(stderr, "FAIL: %s\n", what.c_str());
    }
  }
};

// The §3.5 default die point in the wrapped §3.3 form.
const char* kConfig = R"JSON({
  "schema_version": 1,
  "profile": "mesh_die_v1",
  "die": {
    "sram-mb-per-core": 0.75, "core-type": "large",
    "compute-area-mm2": 800, "hbm-per-side": 4, "hbm-sides": 1,
    "noc-width-bits": 512
  }
})JSON";

// One-span prefill norm spec (full unsharded extent).
const char* kNormSpec = R"JSON({
  "schema_version": 1,
  "op_kind": "norm", "phase": "prefill", "dtype": "bf16",
  "model_dims": { "hidden": 4096, "ffn": 14336, "vocab": 32000,
    "mlp_variant": "swiglu", "norm_type": "rmsnorm",
    "position_encoding": "rope", "num_q_heads": 32, "num_kv_heads": 8,
    "head_dim": 128 },
  "tp_slice": { "tensor_parallel": 1, "tensor_parallel_rank": 0,
    "q_heads_local": 32, "kv_heads_local": 8, "ffn_local": 14336,
    "vocab_local": 32000 },
  "spans": [ { "request_id": "r0", "q_tokens": 1024, "kv_tokens": 1024,
    "q_start_pos": 0, "kv_start_pos": 0 } ],
  "layer_count": 1, "weight_passes": 1,
  "kv_access": { "identity": "local" },
  "fusion_policy": "auto"
})JSON";

const char* kOprojSpec = R"JSON({
  "schema_version": 1,
  "op_kind": "o_proj", "phase": "prefill", "dtype": "bf16",
  "model_dims": { "hidden": 4096, "ffn": 14336, "vocab": 32000,
    "mlp_variant": "swiglu", "norm_type": "rmsnorm",
    "position_encoding": "rope", "num_q_heads": 32, "num_kv_heads": 8,
    "head_dim": 128 },
  "tp_slice": { "tensor_parallel": 1, "tensor_parallel_rank": 0,
    "q_heads_local": 32, "kv_heads_local": 8, "ffn_local": 14336,
    "vocab_local": 32000 },
  "spans": [ { "request_id": "r0", "q_tokens": 1024, "kv_tokens": 1024,
    "q_start_pos": 0, "kv_start_pos": 0 } ],
  "layer_count": 1, "weight_passes": 1,
  "kv_access": { "identity": "local" },
  "fusion_policy": "auto"
})JSON";

// Strict-parser violation: op_kind spelled wrong.
const char* kBadSpec = R"JSON({
  "schema_version": 1,
  "op_kind": "normmm", "phase": "prefill", "dtype": "bf16",
  "model_dims": { "hidden": 4096, "ffn": 14336, "vocab": 32000,
    "mlp_variant": "swiglu", "norm_type": "rmsnorm",
    "position_encoding": "rope", "num_q_heads": 32, "num_kv_heads": 8,
    "head_dim": 128 },
  "tp_slice": { "tensor_parallel": 1, "tensor_parallel_rank": 0,
    "q_heads_local": 32, "kv_heads_local": 8, "ffn_local": 14336,
    "vocab_local": 32000 },
  "spans": [ { "request_id": "r0", "q_tokens": 1024, "kv_tokens": 1024,
    "q_start_pos": 0, "kv_start_pos": 0 } ],
  "layer_count": 1, "weight_passes": 1,
  "kv_access": { "identity": "local" },
  "fusion_policy": "auto"
})JSON";

const char* kFrozenCostFields[] = {
    "profile",          "total_seconds",     "pe_seconds",
    "vec_seconds",      "noc_seconds",       "mem_seconds",
    "launch_seconds",   "launch_instances",  "flops",
    "eqops",            "matrix_actual_bytes", "vector_actual_bytes",
    "matrix_service_units", "vector_service_units", "noc_link_bytes_total",
    "noc_edge_bytes_max", "sram_peak_per_core", "fused",
    "fallback",         "total_ns_suggested",
};

std::string specs_json_of(const char* a, const char* b, const char* c) {
  nlohmann::json arr = nlohmann::json::array();
  arr.push_back(nlohmann::json::parse(a));
  arr.push_back(nlohmann::json::parse(b));
  arr.push_back(nlohmann::json::parse(c));
  nlohmann::json wrap;
  wrap["specs"] = std::move(arr);
  return wrap.dump();
}

}  // namespace

int main() {
  Checker ck;

  // --- error path: invalid config -> NULL + die_last_error ---
  {
    void* bad = die_create_model("{ not json");
    ck.expect(bad == nullptr, "create_model: malformed JSON returns NULL");
    ck.expect(die_last_error() != nullptr &&
                  std::strlen(die_last_error()) > 0,
              "create_model: last error message present");
    die_destroy_model(bad);  // NULL is a no-op
    void* bad2 = die_create_model(
        R"({"die": {"sram-mb-per-core": 0.75, "core-type": "large",
                     "compute-area-mm2": 799, "hbm-per-side": 4,
                     "hbm-sides": 1, "noc-width-bits": 512}})");
    ck.expect(bad2 == nullptr,
              "create_model: illegal layout returns NULL");
    ck.expect(std::strstr(die_last_error(), "legal layouts") != nullptr,
              "create_model: last error names the violation");
  }

  // --- model lifecycle: wrapped and bare forms ---
  void* model = die_create_model(kConfig);
  ck.expect(model != nullptr, "create_model: wrapped form accepted");
  void* bare = die_create_model(
      R"({"sram-mb-per-core": 0.75, "core-type": "large",
          "compute-area-mm2": 800, "hbm-per-side": 4, "hbm-sides": 1,
          "noc-width-bits": 512})");
  ck.expect(bare != nullptr, "create_model: bare die node accepted");

  // --- hardware summary (contract §3.3 derived block) ---
  double bandwidth = 0.0;
  {
    const char* summary = die_hardware_summary(model);
    ck.expect(summary != nullptr, "hardware_summary: non-NULL");
    const nlohmann::json j = nlohmann::json::parse(summary);
    ck.expect(j.at("schema_version") == 1 &&
                  j.at("profile") == "mesh_die_v1",
              "hardware_summary: schema_version/profile");
    ck.expect(j.at("die").at("hbm-per-side") == 4u,
              "hardware_summary: die node verbatim");
    ck.expect(j.at("derived").at("mesh_n") == 35u &&
                  j.at("derived").at("cores") == 1225u,
              "hardware_summary: mesh 35x35");
    bandwidth = j.at("derived").at("hbm_bandwidth_bytes");
    ck.expect(bandwidth == 1.64e12,
              "hardware_summary: hbm bandwidth = 4x410 GB/s");
    die_free_result(summary);
    die_free_result(nullptr);  // no-op
  }

  // --- estimate_batch: results/errors arrays; bad spec does not kill batch ---
  std::string norm_total_ns;
  {
    ck.expect(die_estimate_batch(model, "not json") == nullptr,
              "estimate_batch: malformed wrapper returns NULL");
    ck.expect(die_estimate_batch(model, "{\"nope\": 1}") == nullptr,
              "estimate_batch: missing specs key returns NULL");
    const std::string in = specs_json_of(kNormSpec, kOprojSpec, kBadSpec);
    const char* out = die_estimate_batch(model, in.c_str());
    ck.expect(out != nullptr, "estimate_batch: batch succeeds");
    const nlohmann::json j = nlohmann::json::parse(out);
    ck.expect(j.at("results").size() == 3 && j.at("errors").size() == 3,
              "estimate_batch: parallel results/errors arrays");
    ck.expect(!j.at("errors").at(0).is_string() &&
                  !j.at("errors").at(1).is_string(),
              "estimate_batch: good specs have null errors");
    ck.expect(j.at("errors").at(2).is_string() &&
                  j.at("results").at(2).is_null(),
              "estimate_batch: bad spec yields null result + message");
    ck.expect(std::strstr(j.at("errors").at(2).get<std::string>().c_str(),
                          "op_kind") != nullptr,
              "estimate_batch: error message names the field");
    const nlohmann::json& r0 = j.at("results").at(0);
    for (const char* field : kFrozenCostFields) {
      ck.expect(r0.contains(field),
                std::string("estimate_batch: CostResult field ") + field);
    }
    ck.expect(r0.at("profile") == "mesh_die_v1" &&
                  r0.at("total_ns_suggested").is_number_unsigned(),
              "estimate_batch: field types");
    ck.expect(r0.at("total_seconds").get<double>() > 0.0,
              "estimate_batch: positive total");
    norm_total_ns = j.at("results").at(0).at("total_ns_suggested").dump();
    die_free_result(out);

    // Determinism (and plan-cache reuse under the hood).
    const char* out2 = die_estimate_batch(model, in.c_str());
    ck.expect(out2 != nullptr &&
                  nlohmann::json::parse(out2)
                          .at("results")
                          .at(0)
                          .at("total_ns_suggested")
                          .dump() == norm_total_ns,
              "estimate_batch: deterministic results");
    die_free_result(out2);
  }

  // --- execution protocol over the C ABI: full norm run to Done ---
  {
    void* exec = die_create_execution(model, kNormSpec);
    ck.expect(exec != nullptr, "create_execution: norm spec compiles");
    ck.expect(die_create_execution(model, kBadSpec) == nullptr,
              "create_execution: bad spec returns NULL");
    ck.expect(std::strlen(die_last_error()) > 0,
              "create_execution: last error present after failure");

    char buf[512];
    std::uint64_t memreqs = 0;
    long long elapsed = -1;
    long long estimate = std::atoll(norm_total_ns.c_str());
    ck.expect(estimate > 0, "capi execution: positive estimate");
    for (int guard = 0; guard < 100000; ++guard) {
      const int rc = die_next_action(exec, buf, sizeof(buf));
      ck.expect(rc == 0, "next_action: rc == 0");
      if (rc != 0) {
        break;
      }
      const nlohmann::json a = nlohmann::json::parse(buf);
      const std::string kind = a.at("kind");
      if (kind == "done") {
        elapsed = a.at("elapsed_ns");
        break;
      }
      if (kind == "timer") {
        ck.expect(a.at("ns").is_number_unsigned() && a.at("ns") >= 1u,
                  "next_action: timer ns >= 1");
        ck.expect(die_advance_time(exec, a.at("ns")) == 0,
                  "advance_time rc == 0");
      } else {
        ck.expect(kind == "memory", "next_action: known kind");
        ck.expect(a.at("stage_id").is_number_unsigned() &&
                      a.at("actual_bytes").is_number() &&
                      a.at("service_units").is_number(),
                  "next_action: memory payload fields");
        ++memreqs;
        const double svc = a.at("service_units");
        // Product first: exactly-representable integer service durations
        // survive the ceil without +1 ns of double noise.
        const double ns = svc > 0.0 ? std::ceil(svc * 1e9 / bandwidth) : 0.0;
        const unsigned long long t =
            ns > 0.0 && ns < 1.0 ? 1ull : (unsigned long long)ns;
        if (t != 0) {
          ck.expect(die_advance_time(exec, t) == 0, "advance_time rc == 0");
        }
        ck.expect(die_complete_memory(exec, a.at("stage_id")) == 0,
                  "complete_memory rc == 0");
      }
    }
    ck.expect(elapsed >= 0, "capi execution: reached Done");
    ck.expect(elapsed >= estimate,
              "capi execution: elapsed >= estimate (monotone rounding)");
    ck.expect(elapsed - estimate <= 2LL * (long long)memreqs + 2,
              "capi execution: parity within the memory-link rounding budget");
    std::printf("  capi norm run: elapsed=%lld estimate=%lld memreqs=%llu\n",
                elapsed, estimate, (unsigned long long)memreqs);

    // Truncation: a Done action serialized into an 8-byte buffer is -2
    // (the call itself is legal; only the capacity is insufficient).
    char small[8];
    ck.expect(die_next_action(exec, small, sizeof(small)) == -2,
              "next_action: -2 when cap is too small for the action");
    die_destroy_execution(exec);

    // A fresh execution: first action is the launch timer; a 16-byte buffer
    // is too small, a 256-byte buffer works.
    void* e2 = die_create_execution(model, kOprojSpec);
    ck.expect(e2 != nullptr, "create_execution: second handle");
    ck.expect(die_complete_memory(e2, 3) == -1,
              "complete_memory: -1 with nothing outstanding");
    ck.expect(die_advance_time(e2, 0) == 0, "advance_time(0) rc == 0");
    char mid[16];
    ck.expect(die_next_action(e2, mid, sizeof(mid)) == -2,
              "next_action: -2 on a 16-byte buffer");
    char big[256];
    ck.expect(die_next_action(e2, big, sizeof(big)) == 0,
              "next_action: 0 with a large enough buffer");
    ck.expect(nlohmann::json::parse(big).at("kind") == "timer",
              "next_action: first o_proj action is a timer");
    // Mismatching stage_id on a real outstanding request.
    {
      char buf2[256];
      bool saw_memory = false;
      for (int i = 0; i < 4 && !saw_memory; ++i) {
        if (die_next_action(e2, buf2, sizeof(buf2)) != 0) {
          break;
        }
        const nlohmann::json a = nlohmann::json::parse(buf2);
        if (a.at("kind") == "timer") {
          die_advance_time(e2, a.at("ns"));
        } else if (a.at("kind") == "memory") {
          saw_memory = true;
          const unsigned long long wrong = a.at("stage_id") == 1ull ? 2ull : 1ull;
          ck.expect(die_complete_memory(e2, wrong) == -1,
                    "complete_memory: -1 on a mismatching stage_id");
        }
      }
    }
    die_destroy_execution(e2);
  }

  die_destroy_model(bare);
  die_destroy_model(model);
  die_destroy_model(nullptr);  // no-op
  die_free_result(nullptr);    // no-op
  ck.expect(die_last_error() != nullptr, "last_error never NULL");

  if (ck.ok) {
    std::printf("[capi_test] ALL PASS\n");
    return 0;
  }
  return 1;
}
