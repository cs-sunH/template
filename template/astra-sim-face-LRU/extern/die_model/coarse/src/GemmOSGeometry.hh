#pragma once

// Bounded symbolic geometry and exact access ledger for one OS GEMM.
//
// This header deliberately owns no per-tile, per-wave, or per-core storage.
// Output tile q is decoded from q / Tn and q % Tn, then mapped to round/core
// by division and remainder. The at-most-12 access classes (M full/tail x N
// full/tail x K first/middle/final) summarize arbitrarily large repeats.

#include "GemmOSAccessCounter.hh"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace coarse_die::detail::gemm_os_geometry {

using u64 = std::uint64_t;
using u128 = unsigned __int128;
namespace access = gemm_os_access_counter;
using EqWork = access::EqWork;

inline constexpr std::size_t kMaxAccessClasses = 32;

struct Input {
  u64 M = 0, N = 0, K = 0;
  u64 tm = 0, tn = 0, tk = 0;
  u64 core_count = 0;
  access::StorageDType a_storage = access::StorageDType::Fp32;
  access::StorageDType w_storage = access::StorageDType::Fp32;
  access::StorageDType output_storage = access::StorageDType::Fp32;
  access::MatrixPath matrix_path = access::MatrixPath::Int4PE;
  double kappa_quant = 4.0;
  double kappa_acc = 2.0;
};

enum class KRole : std::uint8_t { First, Middle, Final, FirstFinal };

struct Ledger {
  u128 hbm_read_bytes = 0;
  u128 hbm_write_bytes = 0;
  u128 load_sram_write_bytes = 0;
  u128 compute_sram_read_bytes = 0;
  u128 compute_sram_write_bytes = 0;
  u128 store_sram_read_bytes = 0;
  u128 sram_read_bytes = 0;
  u128 sram_write_bytes = 0;
  u128 local_bytes = 0;
  u128 matrix_int4_ops = 0;
  EqWork vec_equivalent_ops;
  u128 conversion_elements = 0;
  u128 source_math_ops = 0;
};

struct AccessClass {
  std::size_t index = 0;
  u64 m_tile_begin = 0, m_tile_count = 0;
  u64 n_tile_begin = 0, n_tile_count = 0;
  u64 k_tile_begin = 0, k_tile_count = 0;
  u64 m = 0, n = 0, k = 0;
  bool m_tail = false, n_tail = false, k_tail = false;
  bool first_k = false, final_k = false;
  KRole k_role = KRole::FirstFinal;
  u128 output_tile_count = 0;
  u128 repeat_count = 0;  // output tiles * K positions represented here
  access::Result per_consumption;
  Ledger aggregate;
};

struct CohortFact {
  u128 round = 0;
  u128 q_begin = 0;
  u128 q_end = 0;  // exclusive
  u64 active_cores = 0;
};

struct WholeRowClasswordFact {
  u64 m_tile = 0;
  u64 m = 0;
  bool m_tail = false;
  u128 q_begin = 0;
  u128 q_end = 0;  // exclusive
  u64 full_n_tile_count = 0;
  bool has_n_tail = false;
  u128 n_full_q_end = 0;
  u128 n_tail_q_begin = 0;
  u128 full_m_q_end = 0;
  u128 m_tail_q_begin = 0;
  bool has_m_tail = false;
};

struct OutputFact {
  u128 q = 0;
  u64 m_tile = 0, n_tile = 0;
  u128 round = 0;
  u64 core = 0;
  u64 active_cores = 0;
  u64 m = 0, n = 0;
  bool m_tail = false, n_tail = false;
};

struct WaveFact {
  OutputFact output;
  u128 global_wave = 0;
  u64 k_tile = 0;
  u64 k = 0;
  bool k_tail = false;
  bool first_k = false, final_k = false;
  std::size_t class_index = 0;
};

namespace detail {

inline constexpr u128 max_u128() noexcept { return ~u128{0}; }

inline u128 checked_add(u128 a, u128 b, const char* what) {
  if (b > max_u128() - a)
    throw std::overflow_error(std::string("overflow: GEMM OS geometry ") + what);
  return a + b;
}

inline u128 checked_mul(u128 a, u128 b, const char* what) {
  if (a != 0 && b > max_u128() / a)
    throw std::overflow_error(std::string("overflow: GEMM OS geometry ") + what);
  return a * b;
}

inline u64 ceil_div(u64 value, u64 divisor) {
  if (divisor == 0) throw std::invalid_argument("GEMM OS tile divisor is zero");
  return value / divisor + static_cast<u64>(value % divisor != 0);
}

inline u128 ceil_div(u128 value, u64 divisor) {
  if (divisor == 0) throw std::invalid_argument("GEMM OS core count is zero");
  return value / divisor + static_cast<u128>(value % divisor != 0);
}

struct AxisClass {
  u64 begin = 0;
  u64 count = 0;
  u64 extent = 0;
  bool tail = false;
};

struct AxisClasses {
  std::array<AxisClass, 2> values{};
  std::size_t size = 0;
};

inline AxisClasses axis_classes(u64 dim, u64 tile) {
  AxisClasses result;
  const u64 full_count = dim / tile;
  const u64 tail = dim % tile;
  if (full_count != 0) {
    result.values[result.size++] = {0, full_count, tile, false};
  }
  if (tail != 0) {
    result.values[result.size++] = {full_count, 1, tail, true};
  }
  // If the whole dimension is shorter than one nominal tile, it is the one
  // short (tail) class and still occupies tile index zero.
  if (result.size == 0) {
    result.values[result.size++] = {0, 1, dim, dim != tile};
  }
  return result;
}

struct KClass {
  u64 begin = 0;
  u64 count = 0;
  u64 extent = 0;
  bool tail = false;
  bool first = false;
  bool final = false;
  KRole role = KRole::FirstFinal;
};

struct KClasses {
  std::array<KClass, 3> values{};
  std::size_t size = 0;
};

inline KClasses k_classes(u64 dim, u64 tile) {
  KClasses result;
  const u64 full_count = dim / tile;
  const u64 tail = dim % tile;
  if (full_count == 0) {
    result.values[result.size++] = {
        0, 1, dim, true, true, true, KRole::FirstFinal};
    return result;
  }

  result.values[result.size++] = {0, 1, tile, false, true, false, KRole::First};
  if (tail != 0) {
    if (full_count > 1) {
      result.values[result.size++] = {
          1, full_count - 1, tile, false, false, false, KRole::Middle};
    }
    result.values[result.size++] = {
        full_count, 1, tail, true, false, true, KRole::Final};
  } else if (full_count == 1) {
    result.values[0].final = true;
    result.values[0].role = KRole::FirstFinal;
  } else {
    if (full_count > 2) {
      result.values[result.size++] = {
          1, full_count - 2, tile, false, false, false, KRole::Middle};
    }
    result.values[result.size++] = {
        full_count - 1, 1, tile, false, false, true, KRole::Final};
  }
  return result;
}

inline EqWork multiply_work_u128(EqWork value, u128 count) {
  EqWork result;
  while (count != 0) {
    if ((count & 1) != 0) result += value;
    count >>= 1;
    if (count != 0) value += value;
  }
  return result;
}

inline u128 scaled(u64 value, u128 count, const char* what) {
  return checked_mul(static_cast<u128>(value), count, what);
}

inline Ledger scale(const access::Result& value, u128 count) {
  Ledger result;
  result.hbm_read_bytes = scaled(value.hbm_read_bytes, count, "HBM read aggregate");
  result.hbm_write_bytes = scaled(value.hbm_write_bytes, count, "HBM write aggregate");
  result.load_sram_write_bytes = scaled(value.load_sram_write_bytes, count,
                                        "Load SRAM aggregate");
  result.compute_sram_read_bytes = scaled(value.compute_sram_read_bytes, count,
                                          "Compute SRAM read aggregate");
  result.compute_sram_write_bytes = scaled(value.compute_sram_write_bytes, count,
                                           "Compute SRAM write aggregate");
  result.store_sram_read_bytes = scaled(value.store_sram_read_bytes, count,
                                        "Store SRAM aggregate");
  result.sram_read_bytes = scaled(value.sram_read_bytes, count,
                                  "SRAM read aggregate");
  result.sram_write_bytes = scaled(value.sram_write_bytes, count,
                                   "SRAM write aggregate");
  result.local_bytes = scaled(value.local_bytes, count, "LOCAL aggregate");
  result.matrix_int4_ops = scaled(value.matrix_int4_ops, count,
                                  "INT4 operation aggregate");
  result.vec_equivalent_ops = multiply_work_u128(value.vec_equivalent_ops, count);
  result.conversion_elements = scaled(value.conversion_elements, count,
                                      "conversion aggregate");
  result.source_math_ops = scaled(value.source_math_ops, count,
                                  "source operation aggregate");
  return result;
}

inline void add(Ledger& into, const Ledger& value) {
  into.hbm_read_bytes = checked_add(into.hbm_read_bytes, value.hbm_read_bytes,
                                    "HBM read sum");
  into.hbm_write_bytes = checked_add(into.hbm_write_bytes, value.hbm_write_bytes,
                                     "HBM write sum");
  into.load_sram_write_bytes = checked_add(into.load_sram_write_bytes,
                                           value.load_sram_write_bytes,
                                           "Load SRAM sum");
  into.compute_sram_read_bytes = checked_add(into.compute_sram_read_bytes,
                                             value.compute_sram_read_bytes,
                                             "Compute SRAM read sum");
  into.compute_sram_write_bytes = checked_add(into.compute_sram_write_bytes,
                                              value.compute_sram_write_bytes,
                                              "Compute SRAM write sum");
  into.store_sram_read_bytes = checked_add(into.store_sram_read_bytes,
                                           value.store_sram_read_bytes,
                                           "Store SRAM sum");
  into.sram_read_bytes = checked_add(into.sram_read_bytes, value.sram_read_bytes,
                                     "SRAM read sum");
  into.sram_write_bytes = checked_add(into.sram_write_bytes, value.sram_write_bytes,
                                      "SRAM write sum");
  into.local_bytes = checked_add(into.local_bytes, value.local_bytes,
                                 "LOCAL sum");
  into.matrix_int4_ops = checked_add(into.matrix_int4_ops, value.matrix_int4_ops,
                                     "INT4 operation sum");
  into.vec_equivalent_ops += value.vec_equivalent_ops;
  into.conversion_elements = checked_add(into.conversion_elements,
                                         value.conversion_elements,
                                         "conversion sum");
  into.source_math_ops = checked_add(into.source_math_ops, value.source_math_ops,
                                     "source operation sum");
}

}  // namespace detail

class GemmOSGeometry {
 public:
  explicit GemmOSGeometry(const Input& input) : input_(input) {
    if (!input_.M || !input_.N || !input_.K || !input_.tm || !input_.tn ||
        !input_.tk || !input_.core_count) {
      throw std::invalid_argument(
          "GEMM OS geometry M,N,K,tm,tn,tk,core_count must be positive");
    }
    m_tiles_ = detail::ceil_div(input_.M, input_.tm);
    n_tiles_ = detail::ceil_div(input_.N, input_.tn);
    k_tiles_ = detail::ceil_div(input_.K, input_.tk);
    output_tile_count_ = detail::checked_mul(m_tiles_, n_tiles_,
                                             "output tile count");
    output_round_count_ = detail::ceil_div(output_tile_count_, input_.core_count);
    consumption_wave_count_ = detail::checked_mul(
        output_round_count_, k_tiles_, "consumption wave count");
    build_classes();
  }

  const Input& input() const noexcept { return input_; }
  u64 m_tiles() const noexcept { return m_tiles_; }
  u64 n_tiles() const noexcept { return n_tiles_; }
  u64 k_tiles() const noexcept { return k_tiles_; }
  u128 output_tile_count() const noexcept { return output_tile_count_; }
  u128 output_round_count() const noexcept { return output_round_count_; }
  u128 consumption_wave_count() const noexcept { return consumption_wave_count_; }
  std::size_t class_count() const noexcept { return class_count_; }
  const AccessClass& access_class(std::size_t index) const {
    if (index >= class_count_) throw std::out_of_range("GEMM OS class index");
    return classes_[index];
  }
  const Ledger& total_ledger() const noexcept { return total_; }

  CohortFact cohort(u128 round) const {
    if (round >= output_round_count_)
      throw std::out_of_range("GEMM OS output round");
    const u128 begin = detail::checked_mul(round, input_.core_count,
                                           "cohort q begin");
    const u128 remain = output_tile_count_ - begin;
    const u64 active = remain < input_.core_count
        ? static_cast<u64>(remain) : input_.core_count;
    return {round, begin, begin + active, active};
  }

  WholeRowClasswordFact whole_row_classword(u64 m_tile) const {
    if (m_tile >= m_tiles_) throw std::out_of_range("GEMM OS M row");
    const u128 q_begin = detail::checked_mul(m_tile, n_tiles_,
                                             "row q begin");
    const u128 q_end = detail::checked_add(q_begin, n_tiles_, "row q end");
    const u64 full_n = input_.N / input_.tn;
    const u64 full_m = input_.M / input_.tm;
    const bool n_tail = input_.N % input_.tn != 0;
    const bool m_tail = input_.M % input_.tm != 0;
    const u128 n_full_end = detail::checked_add(q_begin, full_n,
                                                "row full-N boundary");
    const u128 full_m_end = detail::checked_mul(full_m, n_tiles_,
                                                "full-M q boundary");
    return {m_tile,
            axis_extent(input_.M, input_.tm, m_tile),
            axis_extent(input_.M, input_.tm, m_tile) != input_.tm,
            q_begin,
            q_end,
            full_n,
            n_tail,
            n_full_end,
            n_full_end,
            full_m_end,
            full_m_end,
            m_tail};
  }

  // Exclusive end of the maximal contiguous q run with the same actual M/N
  // tile extents as q. A short N-tail following full N tiles splits each row;
  // if N is only one short tile, equal full-M rows share a run across rows.
  u128 same_shape_run_end(u128 q) const {
    const OutputFact output = output_fact(q);
    const bool has_n_tail = input_.N % input_.tn != 0;
    const u64 full_n_tiles = input_.N / input_.tn;
    if (has_n_tail && full_n_tiles != 0) {
      if (output.n_tail) return q + 1;
      const u128 row_begin = detail::checked_mul(output.m_tile, n_tiles_,
                                                 "shape run row begin");
      return detail::checked_add(row_begin, full_n_tiles,
                                 "shape run N-tail boundary");
    }

    if (input_.M % input_.tm != 0) {
      const u64 full_m_tiles = input_.M / input_.tm;
      if (full_m_tiles != 0 && output.m_tile < full_m_tiles) {
        return detail::checked_mul(full_m_tiles, n_tiles_,
                                   "shape run M-tail boundary");
      }
    }
    return output_tile_count_;
  }

  OutputFact output_fact(u128 q) const {
    if (q >= output_tile_count_) throw std::out_of_range("GEMM OS output q");
    const u128 m_index = q / n_tiles_;
    const u64 n_index = static_cast<u64>(q % n_tiles_);
    const u64 m_tile = static_cast<u64>(m_index);
    const u128 round = q / input_.core_count;
    const CohortFact current = cohort(round);
    const u64 m = axis_extent(input_.M, input_.tm, m_tile);
    const u64 n = axis_extent(input_.N, input_.tn, n_index);
    return {q, m_tile, n_index, round,
            static_cast<u64>(q % input_.core_count), current.active_cores,
            m, n, m != input_.tm, n != input_.tn};
  }

  WaveFact wave_fact(u128 q, u64 k_tile) const {
    if (k_tile >= k_tiles_) throw std::out_of_range("GEMM OS K tile");
    const OutputFact output = output_fact(q);
    const u64 k = axis_extent(input_.K, input_.tk, k_tile);
    const bool first = k_tile == 0;
    const bool final = k_tile + 1 == k_tiles_;
    const std::size_t index = find_class(output, k_tile);
    const u128 wave = detail::checked_add(
        detail::checked_mul(output.round, k_tiles_, "global wave index"),
        k_tile, "global wave index");
    return {output, wave, k_tile, k, k != input_.tk, first, final, index};
  }

  // global_wave orders rounds first and K positions within each round.
  // lane is the output's core slot in that round.
  WaveFact wave_slot_fact(u128 global_wave, u64 lane) const {
    if (global_wave >= consumption_wave_count_)
      throw std::out_of_range("GEMM OS global consumption wave");
    const u128 round = global_wave / k_tiles_;
    const u64 k_tile = static_cast<u64>(global_wave % k_tiles_);
    const CohortFact current = cohort(round);
    if (lane >= current.active_cores)
      throw std::out_of_range("GEMM OS inactive cohort lane");
    return wave_fact(current.q_begin + lane, k_tile);
  }

  // Count output q values from one M/N access class in [q_begin, q_end).
  // This is a fixed number of arithmetic operations, including across row
  // boundaries; it never walks output tiles or cores.
  u128 class_output_count_in_range(std::size_t class_index,
                                   u128 q_begin, u128 q_end) const {
    const AccessClass& cls = access_class(class_index);
    if (q_begin > q_end || q_end > output_tile_count_)
      throw std::out_of_range("GEMM OS output q range");
    return prefix_count(cls, q_end) - prefix_count(cls, q_begin);
  }

  // Sum the existing per-consumption counter for the active output slots in
  // one (round,K) cohort. The loop is over the fixed class table, never cores.
  Ledger wave_ledger(u128 global_wave) const {
    if (global_wave >= consumption_wave_count_)
      throw std::out_of_range("GEMM OS global consumption wave");
    Ledger result;
    for (std::size_t i = 0; i < class_count_; ++i) {
      const u128 count = class_output_count_in_wave(global_wave, i);
      if (count == 0) continue;
      detail::add(result, detail::scale(classes_[i].per_consumption, count));
    }
    return result;
  }

  u128 class_consumption_count_in_cohort(u128 round,
                                         std::size_t class_index) const {
    const AccessClass& cls = access_class(class_index);
    const CohortFact current = cohort(round);
    return detail::checked_mul(
        class_output_count_in_range(class_index, current.q_begin, current.q_end),
        cls.k_tile_count, "cohort class consumption count");
  }

  // Number of output slots in this single (round,K) wave that use class_index.
  u128 class_output_count_in_wave(u128 global_wave,
                                  std::size_t class_index) const {
    const AccessClass& cls = access_class(class_index);
    if (global_wave >= consumption_wave_count_)
      throw std::out_of_range("GEMM OS global consumption wave");
    const u64 k_tile = static_cast<u64>(global_wave % k_tiles_);
    if (k_tile < cls.k_tile_begin ||
        k_tile - cls.k_tile_begin >= cls.k_tile_count) {
      return 0;
    }
    const CohortFact current = cohort(global_wave / k_tiles_);
    return class_output_count_in_range(class_index, current.q_begin,
                                       current.q_end);
  }

 private:
  static u64 axis_extent(u64 dim, u64 tile, u64 tile_index) {
    const u128 begin_wide = detail::checked_mul(tile_index, tile,
                                                "tile coordinate");
    const u64 begin = static_cast<u64>(begin_wide);
    return std::min(tile, dim - begin);
  }

  void build_classes() {
    const detail::AxisClasses m_classes = detail::axis_classes(input_.M, input_.tm);
    const detail::AxisClasses n_classes = detail::axis_classes(input_.N, input_.tn);
    const detail::KClasses k_classes = detail::k_classes(input_.K, input_.tk);
    for (std::size_t mi = 0; mi < m_classes.size; ++mi) {
      const auto& m = m_classes.values[mi];
      for (std::size_t ni = 0; ni < n_classes.size; ++ni) {
        const auto& n = n_classes.values[ni];
        for (std::size_t ki = 0; ki < k_classes.size; ++ki) {
          const auto& k = k_classes.values[ki];
          if (class_count_ >= kMaxAccessClasses)
            throw std::logic_error("GEMM OS access class bound exceeded");
          AccessClass cls;
          cls.index = class_count_;
          cls.m_tile_begin = m.begin;
          cls.m_tile_count = m.count;
          cls.n_tile_begin = n.begin;
          cls.n_tile_count = n.count;
          cls.k_tile_begin = k.begin;
          cls.k_tile_count = k.count;
          cls.m = m.extent;
          cls.n = n.extent;
          cls.k = k.extent;
          cls.m_tail = m.tail;
          cls.n_tail = n.tail;
          cls.k_tail = k.tail;
          cls.first_k = k.first;
          cls.final_k = k.final;
          cls.k_role = k.role;
          cls.output_tile_count = detail::checked_mul(
              m.count, n.count, "class output tile count");
          cls.repeat_count = detail::checked_mul(
              cls.output_tile_count, k.count, "class repeat count");

          access::Input counter_input;
          counter_input.m = cls.m;
          counter_input.n = cls.n;
          counter_input.k = cls.k;
          counter_input.first_k = cls.first_k;
          counter_input.final_k = cls.final_k;
          counter_input.a_storage = input_.a_storage;
          counter_input.w_storage = input_.w_storage;
          counter_input.output_storage = input_.output_storage;
          counter_input.matrix_path = input_.matrix_path;
          counter_input.dataflow = access::Dataflow::OutputStationary;
          counter_input.kappa_quant = input_.kappa_quant;
          counter_input.kappa_acc = input_.kappa_acc;
          cls.per_consumption = access::count(counter_input);
          cls.aggregate = detail::scale(cls.per_consumption, cls.repeat_count);
          detail::add(total_, cls.aggregate);
          classes_[class_count_++] = std::move(cls);
        }
      }
    }
  }

  u128 prefix_count(const AccessClass& cls, u128 q_end) const {
    if (q_end == 0) return 0;
    const u128 full_rows = q_end / n_tiles_;
    const u64 partial_columns = static_cast<u64>(q_end % n_tiles_);
    const u128 m_begin = cls.m_tile_begin;
    const u128 m_end = m_begin + cls.m_tile_count;
    const u128 complete_m_end = std::min(full_rows, m_end);
    const u128 complete_m_begin = std::min(full_rows, m_begin);
    const u128 selected_rows = complete_m_end > complete_m_begin
        ? complete_m_end - complete_m_begin : 0;
    u128 count = detail::checked_mul(selected_rows, cls.n_tile_count,
                                     "class prefix count");

    if (partial_columns != 0 && full_rows >= m_begin && full_rows < m_end) {
      const u128 n_begin = cls.n_tile_begin;
      const u128 n_end = n_begin + cls.n_tile_count;
      const u128 selected_n_end = std::min<u128>(partial_columns, n_end);
      const u128 selected_n_begin = std::min<u128>(partial_columns, n_begin);
      if (selected_n_end > selected_n_begin) {
        count = detail::checked_add(count, selected_n_end - selected_n_begin,
                                    "class partial-row count");
      }
    }
    return count;
  }

  std::size_t find_class(const OutputFact& output, u64 k_tile) const {
    const u64 k_extent = axis_extent(input_.K, input_.tk, k_tile);
    const bool k_tail = k_extent != input_.tk;
    for (std::size_t i = 0; i < class_count_; ++i) {
      const AccessClass& cls = classes_[i];
      if (cls.m_tail == output.m_tail && cls.n_tail == output.n_tail &&
          cls.k_tail == k_tail && k_tile >= cls.k_tile_begin &&
          k_tile - cls.k_tile_begin < cls.k_tile_count) {
        return i;
      }
    }
    throw std::logic_error("GEMM OS tile has no access class");
  }

  Input input_;
  u64 m_tiles_ = 0, n_tiles_ = 0, k_tiles_ = 0;
  u128 output_tile_count_ = 0;
  u128 output_round_count_ = 0;
  u128 consumption_wave_count_ = 0;
  std::array<AccessClass, kMaxAccessClasses> classes_{};
  std::size_t class_count_ = 0;
  Ledger total_;
};

}  // namespace coarse_die::detail::gemm_os_geometry
