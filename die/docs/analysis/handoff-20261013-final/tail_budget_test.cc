#include "tail_budget.hh"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

namespace {

using tail_budget::Dataflow;
using tail_budget::Input;
using tail_budget::MatrixPath;
using tail_budget::Result;
using tail_budget::u64;

u64 ceildiv(u64 x, u64 y) { return x / y + static_cast<u64>(x % y != 0); }

long double brute_tile_seconds(const Input& p, u64 m, u64 n,
                               bool include_w_conversion) {
  const long double mn = static_cast<long double>(m) * n;
  const long double matrix_ops = 2 * mn * p.K;
  const long double k_tiles = ceildiv(p.K, p.tk);
  long double pe_ops = 0;
  long double vec_ops = 0;
  if (p.matrix_path == MatrixPath::Int4PE) {
    pe_ops = matrix_ops;
    vec_ops += mn * k_tiles * p.int32_acc_conversion_eqops_per_result;
    vec_ops += mn * (k_tiles - 1);
  } else {
    vec_ops += matrix_ops;
  }
  vec_ops += m * static_cast<long double>(p.K) *
      p.a_input_conversion_eqops_per_element;
  if (include_w_conversion)
    vec_ops += n * static_cast<long double>(p.K) *
        p.w_input_conversion_eqops_per_element;
  vec_ops += mn * p.output_cast_eqops_per_element;
  return pe_ops / (p.p_int4_per_core_ops_s * p.eta_pe) +
      vec_ops / (p.p_vec_per_core_eqops_s * p.eta_vec);
}

u64 m_len(const Input& p, u64 i) {
  return std::min(p.tm, p.M - i * p.tm);
}

u64 n_len(const Input& p, u64 j) {
  return std::min(p.tn, p.N - j * p.tn);
}

Result brute(const Input& p) {
  const u64 mt = ceildiv(p.M, p.tm);
  const u64 nt = ceildiv(p.N, p.tn);
  const u64 kt = ceildiv(p.K, p.tk);
  Result r;
  r.m_tiles = mt;
  r.n_tiles = nt;
  r.k_tiles = kt;
  if (p.dataflow == Dataflow::OutputStationary) {
    long double total = 0;
    long double last = 0;
    u64 active = 0;
    for (u64 start = 0; start < mt * nt; start += p.ncore) {
      const u64 end = std::min<u64>(mt * nt, start + p.ncore);
      long double wave_max = 0;
      for (u64 q = start; q < end; ++q) {
        const u64 i = q / nt;
        const u64 j = q % nt;
        wave_max = std::max(wave_max,
                            brute_tile_seconds(p, m_len(p, i), n_len(p, j), true));
      }
      total += wave_max;
      active = end - start;
      last = wave_max;
    }
    r.output_round_count = ceildiv(mt * nt, p.ncore);
    r.last_cohort_active_cores = active;
    r.serial_slowest_compute_gate_seconds = total;
    r.last_cohort_max_seconds = last;
  } else {
    long double total = 0;
    long double last = 0;
    u64 active = 0;
    const u64 groups = ceildiv(nt, p.ncore);
    for (u64 g = 0; g < groups; ++g) {
      const u64 start = g * p.ncore;
      const u64 end = std::min<u64>(nt, start + p.ncore);
      long double group_max = 0;
      for (u64 j = start; j < end; ++j) {
        long double column_s = 0;
        for (u64 i = 0; i < mt; ++i)
          column_s += brute_tile_seconds(p, m_len(p, i), n_len(p, j), false);
        column_s += n_len(p, j) * static_cast<long double>(p.K) *
            p.w_input_conversion_eqops_per_element /
            (p.p_vec_per_core_eqops_s * p.eta_vec);
        group_max = std::max(group_max, column_s);
      }
      total += group_max;
      active = end - start;
      last = group_max;
    }
    r.output_round_count = groups * mt;
    r.last_cohort_active_cores = active;
    r.serial_slowest_compute_gate_seconds = total;
    r.last_cohort_max_seconds = last;
  }
  r.consumption_wave_count = r.output_round_count * kt;
  return r;
}

bool close(long double a, long double b) {
  const long double scale = std::max({1.0L, std::fabs(a), std::fabs(b)});
  return std::fabs(a - b) <= scale * 2e-12L;
}

void check_case(const Input& p, u64& cases) {
  const Result got = tail_budget::calculate(p);
  const Result expected = brute(p);
  if (got.m_tiles != expected.m_tiles || got.n_tiles != expected.n_tiles ||
      got.k_tiles != expected.k_tiles ||
      got.output_round_count != expected.output_round_count ||
      got.consumption_wave_count != expected.consumption_wave_count ||
      got.last_cohort_active_cores != expected.last_cohort_active_cores ||
      !close(got.serial_slowest_compute_gate_seconds,
             expected.serial_slowest_compute_gate_seconds) ||
      !close(got.last_cohort_max_seconds, expected.last_cohort_max_seconds)) {
    throw std::runtime_error("analytic/brute mismatch: M=" + std::to_string(p.M) +
        " N=" + std::to_string(p.N) + " K=" + std::to_string(p.K) +
        " tm=" + std::to_string(p.tm) + " tn=" + std::to_string(p.tn) +
        " tk=" + std::to_string(p.tk) + " C=" + std::to_string(p.ncore));
  }
  ++cases;
}

template <typename Fn>
void expect_throw(Fn fn, const char* label) {
  try {
    fn();
  } catch (const std::exception&) {
    return;
  }
  throw std::runtime_error(std::string("expected rejection: ") + label);
}

}  // namespace

int main() {
  u64 cases = 0;
  for (u64 M = 1; M <= 6; ++M)
    for (u64 N = 1; N <= 6; ++N)
      for (u64 K = 1; K <= 5; ++K)
        for (u64 tm = 1; tm <= 4; ++tm)
          for (u64 tn = 1; tn <= 4; ++tn)
            for (u64 tk = 1; tk <= 3; ++tk)
              for (u64 cores = 1; cores <= 5; ++cores)
                for (unsigned path = 0; path < 2; ++path)
                  for (unsigned df = 0; df < 2; ++df)
                    for (unsigned coeffs = 0; coeffs < 3; ++coeffs) {
                      Input p;
                      p.M = M; p.N = N; p.K = K;
                      p.tm = tm; p.tn = tn; p.tk = tk; p.ncore = cores;
                      p.matrix_path = path ? MatrixPath::StrictFp32Vec
                                           : MatrixPath::Int4PE;
                      p.dataflow = df ? Dataflow::WeightResident
                                      : Dataflow::OutputStationary;
                      p.p_int4_per_core_ops_s = 79;
                      p.p_vec_per_core_eqops_s = 113;
                      p.eta_pe = 0.83L;
                      p.eta_vec = 0.71L;
                      if (coeffs == 0) {
                        p.a_input_conversion_eqops_per_element = 0;
                        p.w_input_conversion_eqops_per_element = 0;
                        p.int32_acc_conversion_eqops_per_result = 0;
                        p.output_cast_eqops_per_element = 0;
                      } else if (coeffs == 1) {
                        p.a_input_conversion_eqops_per_element = 0.625L;
                        p.w_input_conversion_eqops_per_element = 0.875L;
                        p.int32_acc_conversion_eqops_per_result = 1.375L;
                        p.output_cast_eqops_per_element = 0.625L;
                      } else {
                        p.a_input_conversion_eqops_per_element = 1.0L;
                        p.w_input_conversion_eqops_per_element = 0.25L;
                        p.int32_acc_conversion_eqops_per_result = 0.5L;
                        p.output_cast_eqops_per_element = 1.0L;
                      }
                      check_case(p, cases);
                    }

  // The WR path need not materialize the OS row-major tile index product.
  Input large_wr;
  large_wr.M = std::numeric_limits<u64>::max();
  large_wr.N = std::numeric_limits<u64>::max();
  large_wr.K = 1;
  large_wr.tm = large_wr.tn = 1;
  large_wr.ncore = std::numeric_limits<u64>::max();
  large_wr.tk = 1;
  large_wr.dataflow = Dataflow::WeightResident;
  large_wr.p_int4_per_core_ops_s = 2;
  large_wr.p_vec_per_core_eqops_s = 3;
  const auto large_result = tail_budget::calculate(large_wr);
  if (large_result.output_round_count != std::numeric_limits<u64>::max() ||
      large_result.last_cohort_active_cores != std::numeric_limits<u64>::max())
    throw std::runtime_error("large WR index-free case failed");

  Input os_index_overflow = large_wr;
  os_index_overflow.tm = os_index_overflow.tn = 1;
  os_index_overflow.dataflow = Dataflow::OutputStationary;
  expect_throw([&] { tail_budget::calculate(os_index_overflow); },
               "OS u64 tile index space");

  Input op_overflow;
  op_overflow.M = op_overflow.N = op_overflow.K =
      std::numeric_limits<u64>::max();
  op_overflow.tm = op_overflow.M;
  op_overflow.tn = op_overflow.N;
  op_overflow.tk = op_overflow.K;
  op_overflow.ncore = 1;
  op_overflow.p_int4_per_core_ops_s = 1;
  op_overflow.p_vec_per_core_eqops_s = 1;
  expect_throw([&] { tail_budget::calculate(op_overflow); },
               "u128 matrix op product");

  Input bad_coefficient;
  bad_coefficient.M = bad_coefficient.N = bad_coefficient.K = 1;
  bad_coefficient.tm = bad_coefficient.tn = bad_coefficient.tk = 1;
  bad_coefficient.ncore = 1;
  bad_coefficient.p_int4_per_core_ops_s = 1;
  bad_coefficient.p_vec_per_core_eqops_s = 1;
  bad_coefficient.output_cast_eqops_per_element = -0.1L;
  expect_throw([&] { tail_budget::calculate(bad_coefficient); },
               "negative conversion coefficient");

  std::cout << "passed exhaustive_small_shape_cases=" << cases
            << " plus overflow/validation checks; scope=independent model-only formula check\n";
  return 0;
}
