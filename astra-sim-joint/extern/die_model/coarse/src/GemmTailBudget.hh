#pragma once

// Internal O(1) GEMM slowest-cohort compute budget helper.
//
// This models the current §7.2 conservative COMPUTE_GATE contract: for each
// cohort/wave, the old aggregate work/(active_cores * per-core peak) average is
// replaced by the slowest per-core PE+Vec serial budget.  A tile cost is thus
// divided by a single-core rate, and a cohort contributes max(tile_cost); do
// not divide that maximum by the core count a second time.
//
// The helper has no tile/core/K-wave-sized loop.  OS uses at most four
// full/tail tile classes and threshold cohort counts; WR uses bounded full/tail
// column-group arithmetic.  uint64 indices/counts and checked u128 products
// reject shapes that cannot be represented by this interface.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace coarse_die::detail::gemm_tail_budget {

using u64 = std::uint64_t;
using u128 = unsigned __int128;

enum class Dataflow { OutputStationary, WeightResident };
enum class MatrixPath { Int4PE, StrictFp32Vec };

struct Input {
  u64 M = 0, N = 0, K = 0;
  u64 tm = 0, tn = 0, tk = 0;
  u64 ncore = 0;
  Dataflow dataflow = Dataflow::OutputStationary;
  MatrixPath matrix_path = MatrixPath::Int4PE;

  // Per-core peak rates.  Efficiencies follow the current hardware profile.
  long double p_int4_per_core_ops_s = 0;
  long double p_vec_per_core_eqops_s = 0;
  long double eta_pe = 1;
  long double eta_vec = 1;

  // Nonnegative equivalent-op cost per actually converted element.  Supply
  // zero for native/no-conversion roles.  These are intentionally fractional
  // long doubles: e.g. strict-FP32 -> INT4 output can use kappa_quant here.
  long double a_input_conversion_eqops_per_element = 0;
  long double w_input_conversion_eqops_per_element = 0;
  long double int32_acc_conversion_eqops_per_result = 0;
  long double output_cast_eqops_per_element = 0;
};

struct Result {
  u64 m_tiles = 0, n_tiles = 0, k_tiles = 0;
  u64 output_round_count = 0;
  u64 consumption_wave_count = 0;
  u64 last_cohort_active_cores = 0;
  long double serial_slowest_compute_gate_seconds = 0;
  long double last_cohort_max_seconds = 0;
};

namespace detail {

inline u128 max_u128() { return ~static_cast<u128>(0); }

inline u128 mul128(u128 a, u128 b, const char* what) {
  if (a && b > max_u128() / a)
    throw std::overflow_error(std::string("u128 overflow: ") + what);
  return a * b;
}

inline u128 add128(u128 a, u128 b, const char* what) {
  if (b > max_u128() - a)
    throw std::overflow_error(std::string("u128 overflow: ") + what);
  return a + b;
}

inline u64 as_u64(u128 value, const char* what) {
  if (value > std::numeric_limits<u64>::max())
    throw std::overflow_error(std::string("u64 overflow: ") + what);
  return static_cast<u64>(value);
}

inline u64 ceildiv(u64 value, u64 divisor) {
  if (!divisor) throw std::invalid_argument("ceildiv divisor is zero");
  return value / divisor + static_cast<u64>(value % divisor != 0);
}

inline void require_nonnegative_finite(long double x, const char* what) {
  if (!std::isfinite(x) || x < 0)
    throw std::invalid_argument(std::string(what) + " must be finite and nonnegative");
}

inline long double require_rate(long double rate, const char* what) {
  if (!std::isfinite(rate) || !(rate > 0))
    throw std::invalid_argument(std::string(what) + " must be finite and positive");
  return rate;
}

inline long double require_efficiency(long double eta, const char* what) {
  if (!std::isfinite(eta) || !(eta > 0) || eta > 1)
    throw std::invalid_argument(std::string(what) + " must be in (0,1]");
  return eta;
}

struct Shape {
  u64 mt = 0, nt = 0, kt = 0;
  u64 mfull = 0, nfull = 0;
  u64 mtail = 0, ntail = 0;
  bool has_mtail = false, has_ntail = false;
  u64 mfull_len = 0, nfull_len = 0;
  u64 q = 0;
};

inline Shape shape(const Input& p, bool needs_os_index_space) {
  if (!p.M || !p.N || !p.K || !p.tm || !p.tn || !p.tk || !p.ncore)
    throw std::invalid_argument("M,N,K,tm,tn,tk,ncore must all be positive");
  Shape s;
  s.mt = ceildiv(p.M, p.tm);
  s.nt = ceildiv(p.N, p.tn);
  s.kt = ceildiv(p.K, p.tk);
  s.mfull = p.M / p.tm;
  s.nfull = p.N / p.tn;
  s.mtail = p.M % p.tm;
  s.ntail = p.N % p.tn;
  s.has_mtail = s.mtail != 0 && s.mfull != 0;
  s.has_ntail = s.ntail != 0 && s.nfull != 0;
  s.mfull_len = s.mfull ? p.tm : p.M;
  s.nfull_len = s.nfull ? p.tn : p.N;
  if (!s.has_mtail) {
    s.mfull = s.mt;
    s.mtail = 0;
  }
  if (!s.has_ntail) {
    s.nfull = s.nt;
    s.ntail = 0;
  }
  if (needs_os_index_space)
    s.q = as_u64(mul128(s.mt, s.nt, "OS output tile index space"),
                 "OS output tile index space");
  return s;
}

inline u128 product3(u64 a, u64 b, u64 c, const char* what) {
  return mul128(mul128(a, b, what), c, what);
}

inline long double to_ld(u128 x) { return static_cast<long double>(x); }

inline long double p_int4_rate(const Input& p) {
  return p.p_int4_per_core_ops_s * p.eta_pe;
}

inline long double p_vec_rate(const Input& p) {
  return p.p_vec_per_core_eqops_s * p.eta_vec;
}

inline long double per_core_tile_seconds(const Input& p, const Shape& s,
                                         u64 m, u64 n,
                                         bool include_w_conversion) {
  const u128 mn = mul128(m, n, "tile M*N");
  const u128 mnk = mul128(mn, p.K, "tile M*N*K");
  const u128 matrix_ops = mul128(2, mnk, "tile matrix ops");
  const u128 a_elements = mul128(m, p.K, "tile A elements");
  const u128 w_elements = mul128(n, p.K, "tile W elements");
  const u128 acc_elements = mul128(mn, s.kt, "tile accumulator conversions");
  const u128 acc_adds = mul128(mn, s.kt - 1, "tile accumulator additions");

  long double pe_ops = 0;
  long double vec_ops = 0;
  if (p.matrix_path == MatrixPath::Int4PE) {
    pe_ops = to_ld(matrix_ops);
    vec_ops += to_ld(acc_elements) * p.int32_acc_conversion_eqops_per_result;
    vec_ops += to_ld(acc_adds);
  } else {
    vec_ops += to_ld(matrix_ops);
  }
  vec_ops += to_ld(a_elements) * p.a_input_conversion_eqops_per_element;
  if (include_w_conversion)
    vec_ops += to_ld(w_elements) * p.w_input_conversion_eqops_per_element;
  vec_ops += to_ld(mn) * p.output_cast_eqops_per_element;

  const long double pe_s = pe_ops == 0 ? 0 : pe_ops / p_int4_rate(p);
  const long double vec_s = vec_ops / p_vec_rate(p);
  const long double total = pe_s + vec_s;
  if (!std::isfinite(total))
    throw std::overflow_error("non-finite per-core tile seconds");
  return total;
}

struct TileClass {
  u64 m_len = 0, n_len = 0;
  u64 m_count = 0, n_count = 0;
  long double seconds = 0;
};

inline std::array<TileClass, 4> os_classes(const Input& p, const Shape& s) {
  std::array<TileClass, 4> c{};
  c[0] = {s.mfull_len, s.nfull_len, s.mfull, s.nfull,
          per_core_tile_seconds(p, s, s.mfull_len, s.nfull_len, true)};
  if (s.has_ntail)
    c[1] = {s.mfull_len, s.ntail, s.mfull, 1,
            per_core_tile_seconds(p, s, s.mfull_len, s.ntail, true)};
  if (s.has_mtail)
    c[2] = {s.mtail, s.nfull_len, 1, s.nfull,
            per_core_tile_seconds(p, s, s.mtail, s.nfull_len, true)};
  if (s.has_mtail && s.has_ntail)
    c[3] = {s.mtail, s.ntail, 1, 1,
            per_core_tile_seconds(p, s, s.mtail, s.ntail, true)};
  return c;
}

inline u64 tile_count(const TileClass& c) {
  return as_u64(mul128(c.m_count, c.n_count, "tile class count"),
                "tile class count");
}

inline u64 os_coverage_count(const Input& p, const Shape& s,
                             const std::array<TileClass, 4>& classes,
                             long double threshold) {
  const auto cost = [&](u64 m, u64 n) {
    return per_core_tile_seconds(p, s, m, n, true);
  };
  if (classes[0].seconds < threshold) return 0;

  const bool m_all = !s.has_mtail || cost(s.mtail, s.nfull_len) >= threshold;
  const bool n_all = !s.has_ntail || cost(s.mfull_len, s.ntail) >= threshold;
  const bool bottom_right_too_slow = s.has_mtail && s.has_ntail &&
      cost(s.mtail, s.ntail) < threshold;
  const u64 c = p.ncore;

  if (c == 1) {
    if (m_all && n_all)
      return s.q - static_cast<u64>(bottom_right_too_slow);
    if (m_all)
      return as_u64(mul128(s.mt, s.nfull, "OS qualified tiles"),
                    "OS qualified tiles");
    if (n_all)
      return as_u64(mul128(s.mfull, s.nt, "OS qualified tiles"),
                    "OS qualified tiles");
    return as_u64(mul128(s.mfull, s.nfull, "OS qualified tiles"),
                  "OS qualified tiles");
  }

  const u64 all_cohorts = ceildiv(s.q, c);
  if (m_all && n_all) {
    if (bottom_right_too_slow) return ceildiv(s.q - 1, c);
    return all_cohorts;
  }
  if (m_all) {
    u64 cohorts = all_cohorts;
    if (s.has_ntail && (s.q - 1) % c == 0) --cohorts;
    return cohorts;
  }
  if (n_all) {
    const u64 prefix = as_u64(mul128(s.mfull, s.nt, "OS qualified prefix"),
                              "OS qualified prefix");
    return ceildiv(prefix, c);
  }
  const u64 prefix = as_u64(mul128(s.mfull, s.nt, "OS qualified prefix"),
                            "OS qualified prefix");
  u64 cohorts = ceildiv(prefix, c);
  if (s.has_ntail && prefix % c == 1) --cohorts;
  return cohorts;
}

inline u64 os_class_last_index(const Shape& s, unsigned which) {
  u128 index = 0;
  switch (which) {
    case 0: // final full-M/full-N tile
      index = add128(mul128(s.mfull - 1, s.nt, "OS full/full endpoint"),
                     s.nfull - 1, "OS full/full endpoint");
      break;
    case 1: // final full-M/N-tail tile
      index = add128(mul128(s.mfull - 1, s.nt, "OS full/tail endpoint"),
                     s.nfull, "OS full/tail endpoint");
      break;
    case 2: // final M-tail/full-N tile
      index = add128(mul128(s.mfull, s.nt, "OS tail/full endpoint"),
                     s.nfull - 1, "OS tail/full endpoint");
      break;
    case 3: // bottom-right tail tile
      index = s.q - 1;
      break;
    default:
      throw std::logic_error("bad OS class");
  }
  return as_u64(index, "OS class endpoint index");
}

inline long double os_total(const Input& p, const Shape& s,
                            const std::array<TileClass, 4>& classes,
                            u64& last_active, long double& last_max) {
  std::array<long double, 4> costs{};
  std::size_t cost_count = 0;
  for (const auto& c : classes) {
    if (!c.m_count || !c.n_count) continue;
    costs[cost_count++] = c.seconds;
  }
  // At most four classes: keep the bounded insertion sort explicit so the
  // compiler does not assume std::sort's generic insertion-sort threshold.
  for (std::size_t i = 1; i < cost_count; ++i) {
    const long double value = costs[i];
    std::size_t j = i;
    while (j && value < costs[j - 1]) {
      costs[j] = costs[j - 1];
      --j;
    }
    costs[j] = value;
  }
  const auto end = std::unique(costs.begin(), costs.begin() + cost_count);
  cost_count = static_cast<std::size_t>(end - costs.begin());
  if (!cost_count) throw std::logic_error("OS has no tile class");

  const u64 cohorts = ceildiv(s.q, p.ncore);
  long double total = costs[0] * static_cast<long double>(cohorts);
  for (std::size_t j = 1; j < cost_count; ++j) {
    const u64 covered = os_coverage_count(p, s, classes, costs[j]);
    total += (costs[j] - costs[j - 1]) * static_cast<long double>(covered);
  }
  last_active = s.q - (cohorts - 1) * p.ncore;
  const u64 last_start = (cohorts - 1) * p.ncore;
  last_max = 0;
  for (unsigned i = 0; i < classes.size(); ++i) {
    if (!classes[i].m_count || !classes[i].n_count) continue;
    if (os_class_last_index(s, i) >= last_start)
      last_max = std::max(last_max, classes[i].seconds);
  }
  if (!std::isfinite(total) || !std::isfinite(last_max))
    throw std::overflow_error("non-finite OS gate seconds");
  return total;
}

inline long double wr_weight_setup_seconds(const Input& p, u64 n_len) {
  const u128 elements = mul128(n_len, p.K, "WR weight conversion elements");
  const long double eqops = to_ld(elements) * p.w_input_conversion_eqops_per_element;
  const long double seconds = eqops / p_vec_rate(p);
  if (!std::isfinite(seconds))
    throw std::overflow_error("non-finite WR weight setup seconds");
  return seconds;
}

inline long double wr_column_group_seconds(const Input& p, const Shape& s,
                                           u64 n_len) {
  long double sum_m = static_cast<long double>(s.mfull) *
      per_core_tile_seconds(p, s, s.mfull_len, n_len, false);
  if (s.has_mtail)
    sum_m += per_core_tile_seconds(p, s, s.mtail, n_len, false);
  return sum_m + wr_weight_setup_seconds(p, n_len);
}

}  // namespace detail

inline Result calculate(const Input& p) {
  using namespace detail;
  if (p.dataflow != Dataflow::OutputStationary &&
      p.dataflow != Dataflow::WeightResident)
    throw std::invalid_argument("unsupported GEMM dataflow");
  if (p.matrix_path != MatrixPath::Int4PE &&
      p.matrix_path != MatrixPath::StrictFp32Vec)
    throw std::invalid_argument("unsupported GEMM matrix path");
  const Shape s = shape(p, p.dataflow == Dataflow::OutputStationary);
  require_rate(p.p_int4_per_core_ops_s, "P_int4_per_core_ops_s");
  require_rate(p.p_vec_per_core_eqops_s, "P_vec_per_core_eqops_s");
  require_efficiency(p.eta_pe, "eta_pe");
  require_efficiency(p.eta_vec, "eta_vec");
  require_nonnegative_finite(p.a_input_conversion_eqops_per_element,
                             "A input conversion coefficient");
  require_nonnegative_finite(p.w_input_conversion_eqops_per_element,
                             "W input conversion coefficient");
  require_nonnegative_finite(p.int32_acc_conversion_eqops_per_result,
                             "INT32 accumulator conversion coefficient");
  require_nonnegative_finite(p.output_cast_eqops_per_element,
                             "output cast coefficient");
  require_rate(p_int4_rate(p), "effective PE rate");
  require_rate(p_vec_rate(p), "effective Vec rate");

  Result r;
  r.m_tiles = s.mt;
  r.n_tiles = s.nt;
  r.k_tiles = s.kt;
  if (p.dataflow == Dataflow::OutputStationary) {
    const auto classes = os_classes(p, s);
    r.output_round_count = ceildiv(s.q, p.ncore);
    r.last_cohort_active_cores = s.q -
        (r.output_round_count - 1) * p.ncore;
    r.serial_slowest_compute_gate_seconds =
        os_total(p, s, classes, r.last_cohort_active_cores,
                 r.last_cohort_max_seconds);
  } else {
    const u64 full_groups = ceildiv(s.nfull, p.ncore);
    const bool tail_only_group = s.has_ntail && s.nfull % p.ncore == 0;
    const long double full_group_s =
        wr_column_group_seconds(p, s, s.nfull_len);
    const long double tail_group_s = s.has_ntail
        ? wr_column_group_seconds(p, s, s.ntail) : 0;
    r.output_round_count = as_u64(
        mul128(ceildiv(s.nt, p.ncore), s.mt, "WR output round count"),
        "WR output round count");
    r.serial_slowest_compute_gate_seconds =
        static_cast<long double>(full_groups) * full_group_s +
        static_cast<long double>(tail_only_group ? 1 : 0) * tail_group_s;
    r.last_cohort_active_cores =
        s.nt - (ceildiv(s.nt, p.ncore) - 1) * p.ncore;
    r.last_cohort_max_seconds = tail_only_group ? tail_group_s : full_group_s;
    if (!std::isfinite(r.serial_slowest_compute_gate_seconds) ||
        !std::isfinite(r.last_cohort_max_seconds))
      throw std::overflow_error("non-finite WR gate seconds");
  }
  r.consumption_wave_count = as_u64(
      mul128(r.output_round_count, s.kt, "GEMM consumption wave count"),
      "GEMM consumption wave count");
  return r;
}

}  // namespace coarse_die::detail::gemm_tail_budget
