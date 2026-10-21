/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

/*
Shared explicit hardware fixtures for legacy ABI compatibility tests.
0.75 MiB / 800 mm2 / 512 bit / 23x23 and 0.25 MiB / 300 mm2 /
256 bit / 23x23 are test inputs, not selected or recommended architectures.
The production typed model and this compatibility layer use one Core design.
*/

#pragma once

#include "DieModel.hh"

#include <cmath>
#include <cstdio>
#include <functional>
#include <string>

namespace die_test {

struct Checker {
  bool ok = true;
  void expect(bool cond, const std::string& what) {
    if (!cond) {
      ok = false;
      std::fprintf(stderr, "FAIL: %s\n", what.c_str());
    }
  }
  void expect_near(double actual, double expected, double rel_tol,
                   const std::string& what) {
    const double denom = expected == 0.0 ? 1.0 : std::abs(expected);
    if (!(std::abs(actual - expected) <= rel_tol * denom)) {
      ok = false;
      std::fprintf(stderr,
                   "FAIL: %s = %.17g, expected %.17g (rel err %.3e)\n",
                   what.c_str(), actual, expected,
                   std::abs(actual - expected) / denom);
    }
  }
  void expect_exact(double actual, double expected, const std::string& what) {
    if (actual != expected) {
      ok = false;
      std::fprintf(stderr, "FAIL: %s = %.17g, expected exactly %.17g\n",
                   what.c_str(), actual, expected);
    }
  }
  void expect_uint(std::uint64_t actual, std::uint64_t expected,
                   const std::string& what) {
    if (actual != expected) {
      ok = false;
      std::fprintf(stderr, "FAIL: %s = %llu, expected %llu\n", what.c_str(),
                   static_cast<unsigned long long>(actual),
                   static_cast<unsigned long long>(expected));
    }
  }
  void expect_throw(const std::function<void()>& fn, const std::string& what,
                    const std::string& contains = "") {
    try {
      fn();
      ok = false;
      std::fprintf(stderr, "FAIL: %s (no DieModelError thrown)\n",
                   what.c_str());
    } catch (const DieModel::DieModelError& e) {
      if (!contains.empty() &&
          std::string(e.what()).find(contains) == std::string::npos) {
        ok = false;
        std::fprintf(stderr, "FAIL: %s (message \"%s\" lacks \"%s\")\n",
                     what.c_str(), e.what(), contains.c_str());
      }
    } catch (const std::exception& e) {
      ok = false;
      std::fprintf(stderr, "FAIL: %s (wrong exception type: %s)\n",
                   what.c_str(), e.what());
    }
  }
};

inline nlohmann::json die_node(double sram, double area,
                               std::uint64_t per_side, std::uint64_t sides,
                               std::uint64_t width, std::uint64_t nx = 23,
                               std::uint64_t ny = 23) {
  nlohmann::json die;
  die["sram-mb-per-core"] = sram;
  die["mesh-nx"] = nx;
  die["mesh-ny"] = ny;
  die["compute-area-mm2"] = area;
  die["hbm-per-side"] = per_side;
  die["hbm-sides"] = sides;
  die["noc-width-bits"] = width;
  return die;
}

inline nlohmann::json wrapped(const nlohmann::json& die,
                              const char* profile) {
  nlohmann::json j;
  j["schema_version"] = 1u;  // unsigned storage (strict parser)
  j["profile"] = profile;
  j["die"] = die;
  return j;
}

// The explicit 23x23 compatibility-test fixture.
inline DieModel::Hardware hw_default() {
  return DieModel::make_hardware(wrapped(
      die_node(0.75, 800, 4, 1, 512), "mesh_die_v1"));
}

// A small-SRAM / small-area legal point.
inline DieModel::Hardware hw_low_sram() {
  return DieModel::make_hardware(
      wrapped(die_node(0.25, 300, 2, 1, 256), "mesh_die_v1"));
}

// A legal point with six HBM stacks but the SAME mesh geometry as
// hw_default() (800 mm2, 0.75 MiB, 512 bit) for the sweep-validity check.
inline DieModel::Hardware hw_default_6hbm() {
  return DieModel::make_hardware(
      wrapped(die_node(0.75, 800, 3, 2, 512), "mesh_die_v1"));
}

// One-span spec builder.
inline DieModel::DieOpSpec spec_of(
    DieModel::OpKind kind, DieModel::Phase phase, std::uint64_t q,
    std::uint64_t kv, std::uint64_t q_start, std::uint64_t kv_start,
    std::uint64_t layers = 1, std::uint64_t passes = 1,
    std::uint64_t q_heads = 32, std::uint64_t kv_heads = 8,
    std::uint64_t hidden = 4096, std::uint64_t ffn = 14336,
    std::uint64_t vocab = 32000, std::uint64_t head_dim = 128) {
  DieModel::DieOpSpec s;
  s.op_kind = kind;
  s.phase = phase;
  s.dtype = DieModel::Dtype::Bf16;
  s.model_dims = DieModel::DieModelDims{
      hidden, ffn, vocab, DieModel::MlpVariant::Swiglu,
      DieModel::NormType::Rmsnorm, DieModel::PositionEncoding::Rope, q_heads,
      kv_heads, head_dim};
  s.tp_slice = DieModel::DieTpSlice{1, 0, q_heads, kv_heads, ffn, vocab};
  DieModel::DieSpan sp;
  sp.request_id = "r0";
  sp.q_tokens = q;
  sp.kv_tokens = kv;
  sp.q_start_pos = q_start;
  sp.kv_start_pos = kv_start;
  s.spans.push_back(sp);
  s.layer_count = layers;
  s.weight_passes = passes;
  return s;
}

inline DieModel::DieOpSpec add_span(DieModel::DieOpSpec s,
                                    const std::string& id, std::uint64_t q,
                                    std::uint64_t kv, std::uint64_t q_start,
                                    std::uint64_t kv_start) {
  DieModel::DieSpan sp;
  sp.request_id = id;
  sp.q_tokens = q;
  sp.kv_tokens = kv;
  sp.q_start_pos = q_start;
  sp.kv_start_pos = kv_start;
  s.spans.push_back(sp);
  return s;
}

}  // namespace die_test
