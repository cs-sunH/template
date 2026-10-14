#include "../src/GemmOSGeometry.hh"

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace geometry = coarse_die::detail::gemm_os_geometry;
namespace counter = coarse_die::detail::gemm_os_access_counter;
using u64 = std::uint64_t;
using u128 = unsigned __int128;

namespace {

void require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}

u64 ceil_div(u64 value, u64 divisor) {
  return value / divisor + static_cast<u64>(value % divisor != 0);
}

u64 extent(u64 dim, u64 tile, u64 index) {
  return std::min(tile, dim - index * tile);
}

struct Key {
  u64 m = 0, n = 0, k = 0;
  bool first = false, final = false;
};

bool same(const Key& a, const Key& b) {
  return a.m == b.m && a.n == b.n && a.k == b.k &&
         a.first == b.first && a.final == b.final;
}

Key key_for(const geometry::Input& input, u64 q, u64 k_tile) {
  const u64 n_tiles = ceil_div(input.N, input.tn);
  const u64 m_tile = q / n_tiles;
  const u64 n_tile = q % n_tiles;
  const u64 k_tiles = ceil_div(input.K, input.tk);
  return {extent(input.M, input.tm, m_tile),
          extent(input.N, input.tn, n_tile),
          extent(input.K, input.tk, k_tile),
          k_tile == 0, k_tile + 1 == k_tiles};
}

struct OracleClass {
  Key key;
  u128 repeat_count = 0;
  u128 output_tile_count = 0;
  u128 k_tile_count = 0;
  counter::Result sample;
  geometry::Ledger aggregate;
};

void add_one(geometry::Ledger& to, const counter::Result& from) {
  to.hbm_read_bytes += from.hbm_read_bytes;
  to.hbm_write_bytes += from.hbm_write_bytes;
  to.load_sram_write_bytes += from.load_sram_write_bytes;
  to.compute_sram_read_bytes += from.compute_sram_read_bytes;
  to.compute_sram_write_bytes += from.compute_sram_write_bytes;
  to.store_sram_read_bytes += from.store_sram_read_bytes;
  to.sram_read_bytes += from.sram_read_bytes;
  to.sram_write_bytes += from.sram_write_bytes;
  to.local_bytes += from.local_bytes;
  to.matrix_int4_ops += from.matrix_int4_ops;
  to.vec_equivalent_ops += from.vec_equivalent_ops;
  to.conversion_elements += from.conversion_elements;
  to.source_math_ops += from.source_math_ops;
}

void require_work(const counter::EqWork& got, const counter::EqWork& expected,
                  const std::string& where) {
  require(got.whole() == expected.whole(), where + " Vec whole mismatch");
  require(got.fraction() == expected.fraction(), where + " Vec fraction mismatch");
}

void require_ledger(const geometry::Ledger& got,
                    const geometry::Ledger& expected,
                    const std::string& where) {
  require(got.hbm_read_bytes == expected.hbm_read_bytes, where + " HBM read");
  require(got.hbm_write_bytes == expected.hbm_write_bytes, where + " HBM write");
  require(got.load_sram_write_bytes == expected.load_sram_write_bytes,
          where + " Load SRAM write");
  require(got.compute_sram_read_bytes == expected.compute_sram_read_bytes,
          where + " Compute SRAM read");
  require(got.compute_sram_write_bytes == expected.compute_sram_write_bytes,
          where + " Compute SRAM write");
  require(got.store_sram_read_bytes == expected.store_sram_read_bytes,
          where + " Store SRAM read");
  require(got.sram_read_bytes == expected.sram_read_bytes, where + " SRAM read");
  require(got.sram_write_bytes == expected.sram_write_bytes, where + " SRAM write");
  require(got.local_bytes == expected.local_bytes, where + " LOCAL");
  require(got.matrix_int4_ops == expected.matrix_int4_ops, where + " INT4 ops");
  require_work(got.vec_equivalent_ops, expected.vec_equivalent_ops, where);
  require(got.conversion_elements == expected.conversion_elements,
          where + " conversion elements");
  require(got.source_math_ops == expected.source_math_ops, where + " source ops");
}

void require_counter(const counter::Result& got,
                     const counter::Result& expected,
                     const std::string& where) {
  require(got.hbm_read_bytes == expected.hbm_read_bytes, where + " per-K HBM read");
  require(got.hbm_write_bytes == expected.hbm_write_bytes, where + " per-K HBM write");
  require(got.load_sram_write_bytes == expected.load_sram_write_bytes,
          where + " per-K Load write");
  require(got.compute_sram_read_bytes == expected.compute_sram_read_bytes,
          where + " per-K Compute read");
  require(got.compute_sram_write_bytes == expected.compute_sram_write_bytes,
          where + " per-K Compute write");
  require(got.store_sram_read_bytes == expected.store_sram_read_bytes,
          where + " per-K Store read");
  require(got.sram_read_bytes == expected.sram_read_bytes,
          where + " per-K SRAM read");
  require(got.sram_write_bytes == expected.sram_write_bytes,
          where + " per-K SRAM write");
  require(got.local_bytes == expected.local_bytes, where + " per-K LOCAL");
  require(got.matrix_int4_ops == expected.matrix_int4_ops,
          where + " per-K INT4 ops");
  require_work(got.vec_equivalent_ops, expected.vec_equivalent_ops,
               where + " per-K");
  require(got.conversion_elements == expected.conversion_elements,
          where + " per-K conversions");
  require(got.source_math_ops == expected.source_math_ops,
          where + " per-K source ops");
}

counter::Result count_one(const geometry::Input& input, u64 m, u64 n, u64 k,
                          bool first, bool final) {
  counter::Input one;
  one.m = m;
  one.n = n;
  one.k = k;
  one.first_k = first;
  one.final_k = final;
  one.a_storage = input.a_storage;
  one.w_storage = input.w_storage;
  one.output_storage = input.output_storage;
  one.matrix_path = input.matrix_path;
  one.dataflow = counter::Dataflow::OutputStationary;
  one.kappa_quant = input.kappa_quant;
  one.kappa_acc = input.kappa_acc;
  return counter::count(one);
}

const OracleClass& oracle_class_for(const std::vector<OracleClass>& classes,
                                    const Key& key) {
  for (const auto& item : classes) {
    if (same(item.key, key)) return item;
  }
  throw std::runtime_error("oracle class missing");
}

std::vector<OracleClass> enumerate_and_check(const geometry::Input& input) {
  const geometry::GemmOSGeometry actual(input);
  const u64 mt = ceil_div(input.M, input.tm);
  const u64 nt = ceil_div(input.N, input.tn);
  const u64 kt = ceil_div(input.K, input.tk);
  const u128 q_count = static_cast<u128>(mt) * nt;
  const u128 rounds = q_count / input.core_count +
      static_cast<u128>(q_count % input.core_count != 0);

  require(actual.m_tiles() == mt && actual.n_tiles() == nt &&
              actual.k_tiles() == kt,
          "tile grid dimensions");
  require(actual.output_tile_count() == q_count, "output tile count");
  require(actual.output_round_count() == rounds, "output round count");
  require(actual.consumption_wave_count() == rounds * kt,
          "consumption wave count");

  const u64 full_m_rows = input.M / input.tm;
  const u64 full_n_tiles = input.N / input.tn;
  for (u64 m_tile = 0; m_tile < mt; ++m_tile) {
    const auto row = actual.whole_row_classword(m_tile);
    const u128 row_begin = static_cast<u128>(m_tile) * nt;
    const u128 row_end = row_begin + nt;
    const u128 full_m_boundary = static_cast<u128>(full_m_rows) * nt;
    require(row.m_tile == m_tile &&
                row.m == extent(input.M, input.tm, m_tile) &&
                row.m_tail == (row.m != input.tm) &&
                row.q_begin == row_begin && row.q_end == row_end,
            "whole-row classword facts");
    require(row.full_n_tile_count == full_n_tiles &&
                row.has_n_tail == (input.N % input.tn != 0) &&
                row.n_full_q_end == row_begin + full_n_tiles &&
                row.n_tail_q_begin == row.n_full_q_end,
            "whole-row N full/tail boundary facts");
    require(row.full_m_q_end == full_m_boundary &&
                row.m_tail_q_begin == full_m_boundary &&
                row.has_m_tail == (input.M % input.tm != 0),
            "whole-row M full/tail boundary facts");
  }

  std::vector<OracleClass> expected;
  geometry::Ledger all;
  for (u128 q = 0; q < q_count; ++q) {
    const u64 m_tile = static_cast<u64>(q / nt);
    const u64 n_tile = static_cast<u64>(q % nt);
    const auto output = actual.output_fact(q);
    const u128 round = q / input.core_count;
    const u64 active = static_cast<u64>(
        std::min<u128>(input.core_count, q_count - round * input.core_count));
    require(output.q == q && output.m_tile == m_tile &&
                output.n_tile == n_tile && output.round == round &&
                output.core == q % input.core_count &&
                output.active_cores == active,
            "O(1) output position mapping");
    const u64 m = extent(input.M, input.tm, m_tile);
    const u64 n = extent(input.N, input.tn, n_tile);
    require(output.m == m && output.n == n &&
                output.m_tail == (m != input.tm) &&
                output.n_tail == (n != input.tn),
            "full/tail output tile facts");

    u128 expected_run_end = q + 1;
    while (expected_run_end < q_count) {
      const u64 next_m_tile = static_cast<u64>(expected_run_end / nt);
      const u64 next_n_tile = static_cast<u64>(expected_run_end % nt);
      if (extent(input.M, input.tm, next_m_tile) != m ||
          extent(input.N, input.tn, next_n_tile) != n) {
        break;
      }
      ++expected_run_end;
    }
    require(actual.same_shape_run_end(q) == expected_run_end,
            "contiguous same-shape run end");

    for (u64 k_tile = 0; k_tile < kt; ++k_tile) {
      const Key key = key_for(input, static_cast<u64>(q), k_tile);
      const counter::Result one = count_one(input, key.m, key.n, key.k,
                                            key.first, key.final);
      auto found = expected.end();
      for (auto it = expected.begin(); it != expected.end(); ++it) {
        if (same(it->key, key)) {
          found = it;
          break;
        }
      }
      if (found == expected.end()) {
        expected.push_back({key, 0, 0, 0, one, {}});
        found = expected.end() - 1;
      }
      ++found->repeat_count;
      add_one(found->aggregate, one);
      add_one(all, one);

      const auto wave = actual.wave_fact(q, k_tile);
      const u128 global_wave = round * kt + k_tile;
      require(wave.output.q == q && wave.k_tile == k_tile &&
                  wave.global_wave == global_wave && wave.k == key.k &&
                  wave.k_tail == (key.k != input.tk) &&
                  wave.first_k == key.first && wave.final_k == key.final,
              "O(1) wave facts");
      const auto& cls = actual.access_class(wave.class_index);
      require(cls.m == key.m && cls.n == key.n && cls.k == key.k &&
                  cls.first_k == key.first && cls.final_k == key.final,
              "wave class descriptor");
      require_counter(cls.per_consumption, one, "wave class");
    }
  }

  for (auto& item : expected) {
    // Independently count the distinct output and K positions represented by
    // this class, rather than deriving either count from geometry metadata.
    for (u64 m_tile = 0; m_tile < mt; ++m_tile) {
      if (extent(input.M, input.tm, m_tile) != item.key.m) continue;
      for (u64 n_tile = 0; n_tile < nt; ++n_tile) {
        if (extent(input.N, input.tn, n_tile) == item.key.n)
          ++item.output_tile_count;
      }
    }
    for (u64 k_tile = 0; k_tile < kt; ++k_tile) {
      const Key key = key_for(input, 0, k_tile);
      if (key.k == item.key.k && key.first == item.key.first &&
          key.final == item.key.final) {
        ++item.k_tile_count;
      }
    }
    require(item.repeat_count == item.output_tile_count * item.k_tile_count,
            "oracle repeat multiplicity");
  }

  require(actual.class_count() == expected.size(), "bounded class count");
  require(actual.class_count() <= geometry::kMaxAccessClasses,
          "class storage bound");
  require_ledger(actual.total_ledger(), all, "enumerated total ledger");
  for (std::size_t i = 0; i < actual.class_count(); ++i) {
    const auto& cls = actual.access_class(i);
    const Key key{cls.m, cls.n, cls.k, cls.first_k, cls.final_k};
    const auto& oracle = oracle_class_for(expected, key);
    require(cls.index == i && cls.repeat_count == oracle.repeat_count &&
                cls.output_tile_count == oracle.output_tile_count &&
                cls.k_tile_count == oracle.k_tile_count,
            "class multiplicity and dimensions");
    require(cls.m_tile_count * cls.n_tile_count == cls.output_tile_count,
            "class M/N multiplicity");
    require(cls.k_role == (cls.first_k && cls.final_k
                               ? geometry::KRole::FirstFinal
                               : cls.first_k ? geometry::KRole::First
                                             : cls.final_k ? geometry::KRole::Final
                                                           : geometry::KRole::Middle),
            "K first/middle/final role");
    require(cls.m_tail == (cls.m != input.tm) &&
                cls.n_tail == (cls.n != input.tn) &&
                cls.k_tail == (cls.k != input.tk),
            "class full/tail tags");
    require_counter(cls.per_consumption, oracle.sample, "class per-K counter");
    require_ledger(cls.aggregate, oracle.aggregate, "class aggregate ledger");
  }

  // Brute force cohort and arbitrary interval queries. These short shapes
  // deliberately include N tails at row boundaries and cohorts that cross
  // those boundaries.
  for (u128 round = 0; round < rounds; ++round) {
    const u128 q_begin = round * input.core_count;
    const u128 q_end = std::min<u128>(q_count, q_begin + input.core_count);
    const auto cohort = actual.cohort(round);
    require(cohort.round == round && cohort.q_begin == q_begin &&
                cohort.q_end == q_end && cohort.active_cores == q_end - q_begin,
            "cohort facts");
    for (std::size_t ci = 0; ci < actual.class_count(); ++ci) {
      const auto& cls = actual.access_class(ci);
      u128 expected_cohort_consumptions = 0;
      for (u128 q = q_begin; q < q_end; ++q) {
        for (u64 k_tile = 0; k_tile < kt; ++k_tile) {
          if (same(key_for(input, static_cast<u64>(q), k_tile),
                   Key{cls.m, cls.n, cls.k, cls.first_k, cls.final_k})) {
            ++expected_cohort_consumptions;
          }
        }
      }
      require(actual.class_consumption_count_in_cohort(round, ci) ==
                  expected_cohort_consumptions,
              "cohort class consumption query");

      for (u128 begin = 0; begin <= q_count; ++begin) {
        for (u128 end = begin; end <= q_count; ++end) {
          u128 expected_range = 0;
          for (u128 q = begin; q < end; ++q) {
            const u64 m_tile = static_cast<u64>(q / nt);
            const u64 n_tile = static_cast<u64>(q % nt);
            if (extent(input.M, input.tm, m_tile) == cls.m &&
                extent(input.N, input.tn, n_tile) == cls.n) {
              ++expected_range;
            }
          }
          require(actual.class_output_count_in_range(ci, begin, end) ==
                      expected_range,
                  "class interval query");
        }
      }
    }
    for (u64 k_tile = 0; k_tile < kt; ++k_tile) {
      const u128 global_wave = round * kt + k_tile;
      geometry::Ledger expected_wave;
      for (u128 q = q_begin; q < q_end; ++q) {
        const Key key = key_for(input, static_cast<u64>(q), k_tile);
        add_one(expected_wave,
                count_one(input, key.m, key.n, key.k, key.first, key.final));
      }
      require_ledger(actual.wave_ledger(global_wave), expected_wave,
                     "independently enumerated wave ledger");
      for (u64 lane = 0; lane < cohort.active_cores; ++lane) {
        const auto slot = actual.wave_slot_fact(global_wave, lane);
        const u128 q = q_begin + lane;
        require(slot.output.q == q && slot.global_wave == global_wave &&
                    slot.k_tile == k_tile,
                "random wave/cohort slot facts");
      }
      for (std::size_t ci = 0; ci < actual.class_count(); ++ci) {
        const auto& cls = actual.access_class(ci);
        u128 expected_outputs = 0;
        for (u128 q = q_begin; q < q_end; ++q) {
          if (same(key_for(input, static_cast<u64>(q), k_tile),
                   Key{cls.m, cls.n, cls.k, cls.first_k, cls.final_k})) {
            ++expected_outputs;
          }
        }
        require(actual.class_output_count_in_wave(global_wave, ci) ==
                    expected_outputs,
                "wave class output query");
      }
    }
  }

  return expected;
}

geometry::Input odd_int4_input() {
  geometry::Input input;
  input.M = 5;
  input.N = 7;
  input.K = 5;
  input.tm = 3;
  input.tn = 4;
  input.tk = 2;
  input.core_count = 3;
  input.a_storage = counter::StorageDType::Bf16;
  input.w_storage = counter::StorageDType::Int4;
  input.output_storage = counter::StorageDType::Int4;
  input.matrix_path = counter::MatrixPath::Int4PE;
  input.kappa_quant = 1.25;
  input.kappa_acc = 0.5;
  return input;
}

geometry::Input strict_input() {
  geometry::Input input;
  input.M = 3;
  input.N = 5;
  input.K = 4;
  input.tm = 4;
  input.tn = 3;
  input.tk = 2;
  input.core_count = 4;
  input.a_storage = counter::StorageDType::Int32;
  input.w_storage = counter::StorageDType::Int4;
  input.output_storage = counter::StorageDType::Bf16;
  input.matrix_path = counter::MatrixPath::StrictFp32Vec;
  input.kappa_quant = 1.375;
  input.kappa_acc = 99.0;  // strict path must not charge accumulator conversion.
  return input;
}

void small_shape_oracles() {
  const auto odd = odd_int4_input();
  const auto odd_classes = enumerate_and_check(odd);
  require(odd_classes.size() == 12,
          "all M/N full-tail x K first-middle-final classes are represented");
  const geometry::GemmOSGeometry odd_geometry(odd);
  require(odd_geometry.total_ledger().source_math_ops == 2 * 5 * 7 * 5,
          "INT4 path source work conserves the actual GEMM shape");

  const auto strict = strict_input();
  const auto strict_classes = enumerate_and_check(strict);
  require(!strict_classes.empty(), "strict path classes enumerated");
  const geometry::GemmOSGeometry strict_geometry(strict);
  require(strict_geometry.total_ledger().matrix_int4_ops == 0 &&
              strict_geometry.total_ledger().source_math_ops == 2 * 3 * 5 * 4,
          "strict FP32 path keeps source math in Vec work");

  geometry::Input one;
  one.M = 2;
  one.N = 3;
  one.K = 1;
  one.tm = 5;
  one.tn = 4;
  one.tk = 8;
  one.core_count = 8;
  one.a_storage = counter::StorageDType::Int4;
  one.w_storage = counter::StorageDType::Int4;
  one.output_storage = counter::StorageDType::Fp32;
  one.matrix_path = counter::MatrixPath::Int4PE;
  const auto one_classes = enumerate_and_check(one);
  require(one_classes.size() == 1 && one_classes[0].key.first &&
              one_classes[0].key.final,
          "single short K tile is a first-final tail class");

  geometry::Input narrow_n;
  narrow_n.M = 5;
  narrow_n.N = 3;
  narrow_n.K = 2;
  narrow_n.tm = 2;
  narrow_n.tn = 4;
  narrow_n.tk = 2;
  narrow_n.core_count = 3;
  narrow_n.a_storage = counter::StorageDType::Bf16;
  narrow_n.w_storage = counter::StorageDType::Int4;
  narrow_n.output_storage = counter::StorageDType::Bf16;
  narrow_n.matrix_path = counter::MatrixPath::StrictFp32Vec;
  (void)enumerate_and_check(narrow_n);
}

void giant_shape_stays_bounded() {
  static_assert(sizeof(geometry::GemmOSGeometry) < 32 * 1024,
                "geometry state remains a small fixed-size object");
  geometry::Input input;
  input.M = u64{1} << 32;
  input.N = u64{1} << 32;
  input.K = 1;
  input.tm = input.tn = input.tk = 1;
  input.core_count = 64;
  input.a_storage = counter::StorageDType::Int4;
  input.w_storage = counter::StorageDType::Int4;
  input.output_storage = counter::StorageDType::Int4;
  input.matrix_path = counter::MatrixPath::Int4PE;
  input.kappa_quant = 0.5;
  input.kappa_acc = 0.25;
  const geometry::GemmOSGeometry actual(input);
  const u128 q_count = u128{1} << 64;
  const u128 rounds = u128{1} << 58;
  require(actual.class_count() == 1 &&
              actual.access_class(0).repeat_count == q_count,
          "huge repeated shape uses a single access class");
  require(actual.output_tile_count() == q_count &&
              actual.output_round_count() == rounds &&
              actual.consumption_wave_count() == rounds,
          "huge output and wave counts are arithmetic only");
  require(actual.total_ledger().hbm_read_bytes == (u128{1} << 65) &&
              actual.total_ledger().hbm_write_bytes == q_count &&
              actual.total_ledger().source_math_ops == (u128{1} << 65),
          "huge shape exact aggregate ledger");
  require(actual.class_output_count_in_range(0, 0, q_count) == q_count,
          "huge interval query is constant-storage");
  require(actual.same_shape_run_end(q_count / 2) == q_count &&
              actual.same_shape_run_end(q_count - 1) == q_count,
          "huge u128 q same-shape run boundary");
  const auto huge_last_row = actual.whole_row_classword(
      actual.m_tiles() - 1);
  require(huge_last_row.q_begin == q_count - (u64{1} << 32) &&
              huge_last_row.q_end == q_count && !huge_last_row.m_tail &&
              !huge_last_row.has_m_tail && !huge_last_row.has_n_tail,
          "huge u128 whole-row boundary facts");

  const u128 last_round = rounds - 1;
  const auto last_cohort = actual.cohort(last_round);
  require(last_cohort.q_begin == q_count - 64 &&
              last_cohort.q_end == q_count && last_cohort.active_cores == 64,
          "huge final cohort facts");
  const auto last = actual.wave_slot_fact(last_round, 63);
  require(last.output.q == q_count - 1 && last.output.m_tile == (u64{1} << 32) - 1 &&
              last.output.n_tile == (u64{1} << 32) - 1 &&
              last.output.core == 63 && last.output.round == last_round,
          "huge random output position facts");
  const auto last_wave = actual.wave_ledger(last_round);
  require(last_wave.hbm_read_bytes == 128 &&
              last_wave.hbm_write_bytes == 64 &&
              last_wave.source_math_ops == 128,
          "huge final wave ledger aggregates one fixed cohort");

  geometry::Input tail_grid;
  tail_grid.M = (u64{1} << 32) + 3;
  tail_grid.N = (u64{1} << 32) + 4;
  tail_grid.K = 1;
  tail_grid.tm = 2;
  tail_grid.tn = 3;
  tail_grid.tk = 1;
  tail_grid.core_count = 7;
  tail_grid.a_storage = counter::StorageDType::Int4;
  tail_grid.w_storage = counter::StorageDType::Int4;
  tail_grid.output_storage = counter::StorageDType::Int4;
  tail_grid.matrix_path = counter::MatrixPath::Int4PE;
  const geometry::GemmOSGeometry tails(tail_grid);
  const u128 tail_q_count = tails.output_tile_count();
  const u64 tail_nt = tails.n_tiles();
  const u64 n_full = tail_grid.N / tail_grid.tn;
  const u128 last_row_begin = static_cast<u128>(tails.m_tiles() - 1) * tail_nt;
  require(tail_q_count > (u128{1} << 60),
          "tail boundary facts exercise a huge q index");
  require(tails.same_shape_run_end(0) == n_full &&
              tails.same_shape_run_end(n_full - 1) == n_full &&
              tails.same_shape_run_end(n_full) == n_full + 1,
          "huge first-row full-N and N-tail run ends");
  require(tails.same_shape_run_end(last_row_begin) == last_row_begin + n_full &&
              tails.same_shape_run_end(tail_q_count - 1) == tail_q_count,
          "huge final M/N tail run ends");
  const auto last_row = tails.whole_row_classword(tails.m_tiles() - 1);
  require(last_row.q_begin == last_row_begin && last_row.q_end == tail_q_count &&
              last_row.m_tail && last_row.has_m_tail && last_row.has_n_tail &&
              last_row.full_m_q_end ==
                  static_cast<u128>(tail_grid.M / tail_grid.tm) * tail_nt,
          "huge final whole-row M/N classword boundaries");
}

template <class Exception, class Fn>
void require_throws(Fn&& fn, const std::string& message) {
  try {
    fn();
  } catch (const Exception&) {
    return;
  }
  throw std::runtime_error(message);
}

void checked_overflow_and_bounds() {
  geometry::Input overflow;
  overflow.M = u64{1} << 32;
  overflow.N = u64{1} << 32;
  overflow.K = u64{1} << 63;
  overflow.tm = overflow.tn = overflow.tk = 1;
  overflow.core_count = std::numeric_limits<u64>::max();
  overflow.a_storage = counter::StorageDType::Int4;
  overflow.w_storage = counter::StorageDType::Int4;
  overflow.output_storage = counter::StorageDType::Fp32;
  overflow.matrix_path = counter::MatrixPath::StrictFp32Vec;
  overflow.kappa_quant = 0.0;
  require_throws<std::overflow_error>(
      [&] { (void)geometry::GemmOSGeometry(overflow); },
      "u128 aggregate overflow must be rejected");

  geometry::Input invalid;
  invalid.M = invalid.N = invalid.K = 1;
  invalid.tm = invalid.tn = invalid.tk = 1;
  invalid.core_count = 0;
  require_throws<std::invalid_argument>(
      [&] { (void)geometry::GemmOSGeometry(invalid); },
      "zero core count must be rejected");

  geometry::Input one = odd_int4_input();
  const geometry::GemmOSGeometry actual(one);
  require_throws<std::out_of_range>(
      [&] { (void)actual.output_fact(actual.output_tile_count()); },
      "out-of-range output q must be rejected");
  require_throws<std::out_of_range>(
      [&] { (void)actual.wave_slot_fact(actual.consumption_wave_count(), 0); },
      "out-of-range wave must be rejected");
}

}  // namespace

int main() {
  try {
    small_shape_oracles();
    giant_shape_stays_bounded();
    checked_overflow_and_bounds();
    std::cout << "gemm_os_geometry_test: PASS\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "gemm_os_geometry_test: FAIL: " << error.what() << '\n';
    return 1;
  }
}
