#include "../src/GemmOSRouteCounter.hh"

#include <algorithm>
#include <array>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace route = coarse_die::detail::gemm_os_route_counter;
using u64 = std::uint64_t;
using u128 = unsigned __int128;

namespace {

void require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}

std::string show(u128 value) {
  if (value == 0) return "0";
  std::string text;
  while (value != 0) {
    text.push_back(static_cast<char>('0' + value % 10));
    value /= 10;
  }
  std::reverse(text.begin(), text.end());
  return text;
}

struct Oracle {
  u128 output_tiles = 0;
  u128 bytes = 0;
  u128 x_byte_sum = 0;
  u128 y_byte_sum = 0;
  u128 x_left_bytes = 0;
  u128 x_right_bytes = 0;
  u128 y_low_bytes = 0;
  u128 y_high_bytes = 0;
};

Oracle enumerate(const route::Rect& rect, u64 n_tiles, u64 nx, u64 ny,
                 u128 q_begin, u128 q_end, u128 bytes_per_tile) {
  Oracle result;
  const u64 m_end = rect.m_tile_begin + rect.m_tile_count;
  const u64 n_end = rect.n_tile_begin + rect.n_tile_count;
  for (u64 m = rect.m_tile_begin; m < m_end; ++m) {
    for (u64 n = rect.n_tile_begin; n < n_end; ++n) {
      const u128 q = static_cast<u128>(m) * n_tiles + n;
      if (q < q_begin || q >= q_end) continue;
      const u64 lane = static_cast<u64>(q % (nx * ny));
      const u64 x = lane % nx;
      const u64 y = lane / nx;
      ++result.output_tiles;
      result.bytes += bytes_per_tile;
      result.x_byte_sum += static_cast<u128>(x) * bytes_per_tile;
      result.y_byte_sum += static_cast<u128>(y) * bytes_per_tile;
      if (x < nx / 2) result.x_left_bytes += bytes_per_tile;
      else result.x_right_bytes += bytes_per_tile;
      if (y < ny / 2) result.y_low_bytes += bytes_per_tile;
      else result.y_high_bytes += bytes_per_tile;
    }
  }
  return result;
}

void compare(const route::ByteMoments& got, const Oracle& want,
             const std::string& where) {
  require(got.output_tiles == want.output_tiles,
          where + " output count got=" + show(got.output_tiles) +
              " want=" + show(want.output_tiles));
  require(got.bytes == want.bytes, where + " byte total");
  require(got.x_byte_sum == want.x_byte_sum, where + " x first moment");
  require(got.y_byte_sum == want.y_byte_sum, where + " y first moment");
  require(got.x_left_bytes == want.x_left_bytes, where + " x-left cut");
  require(got.x_right_bytes == want.x_right_bytes, where + " x-right cut");
  require(got.y_low_bytes == want.y_low_bytes, where + " y-low cut");
  require(got.y_high_bytes == want.y_high_bytes, where + " y-high cut");
  require(!got.has_second_moments, where + " must expose partial status");
}

void check_residue_floor_sum() {
  std::size_t cases = 0;
  for (u64 modulus = 1; modulus <= 13; ++modulus) {
    for (u64 count = 0; count <= 31; ++count) {
      for (u64 step = 0; step <= 17; ++step) {
        for (u64 first = 0; first <= 19; ++first) {
          for (u64 cut = 0; cut <= modulus; ++cut) {
            u128 expected = 0;
            for (u64 i = 0; i < count; ++i)
              expected += (first + static_cast<u128>(step) * i) % modulus < cut;
            const u128 actual = route::detail::residue_count_lt(
                count, modulus, step, first, cut);
            require(actual == expected, "residue count mismatch");
            ++cases;
          }
        }
      }
    }
  }
  require(cases > 100000, "residue floor-sum oracle coverage");
}

void check_class_rectangles() {
  struct Shape { u64 mt, nt; bool m_tail, n_tail; };
  const std::array<Shape, 6> shapes{{
      {1, 1, false, false},
      {2, 3, false, false},
      {4, 3, true, false},
      {3, 4, false, true},
      {4, 5, true, true},
      {2, 1, true, true},
  }};
  std::size_t cases = 0;
  for (const auto& shape : shapes) {
    std::vector<std::pair<u64, u64>> mclasses;
    if (shape.m_tail && shape.mt > 1)
      mclasses.emplace_back(0, shape.mt - 1);
    else
      mclasses.emplace_back(0, shape.mt);
    if (shape.m_tail) mclasses.emplace_back(shape.mt - 1, 1);

    std::vector<std::pair<u64, u64>> nclasses;
    if (shape.n_tail && shape.nt > 1)
      nclasses.emplace_back(0, shape.nt - 1);
    else if (!shape.n_tail)
      nclasses.emplace_back(0, shape.nt);
    if (shape.n_tail) nclasses.emplace_back(shape.nt - 1, 1);

    const u128 output_count = static_cast<u128>(shape.mt) * shape.nt;
    const route::GemmOSRouteCounter counter(shape.mt, shape.nt, 3, 4);
    for (const auto& [mb, mc] : mclasses) {
      for (const auto& [nb, nc] : nclasses) {
        const route::Rect rect{mb, mc, nb, nc};
        for (u128 q_begin = 0; q_begin <= output_count; ++q_begin) {
          for (u128 q_end = q_begin; q_end <= output_count; ++q_end) {
            const u128 byte_weight = 1 + (q_begin + q_end + mb + nb) % 7;
            const auto actual = counter.class_bytes(rect, q_begin, q_end,
                                                     byte_weight);
            const auto expected = enumerate(rect, shape.nt, 3, 4,
                                             q_begin, q_end, byte_weight);
            compare(actual, expected,
                    "class q-range mt=" + std::to_string(shape.mt) +
                        " nt=" + std::to_string(shape.nt));
            ++cases;
          }
        }
      }
    }
  }
  require(cases > 500, "class rectangle/q-range oracle coverage");
}

void check_core_grids() {
  std::size_t cases = 0;
  for (u64 nx = 1; nx <= 6; ++nx) {
    for (u64 ny = 1; ny <= 5; ++ny) {
      for (u64 n_tiles = 1; n_tiles <= 8; ++n_tiles) {
        const u64 m_tiles = 5;
        const route::GemmOSRouteCounter counter(m_tiles, n_tiles, nx, ny);
        const route::Rect all_m{0, m_tiles, 0, n_tiles};
        const u128 total = static_cast<u128>(m_tiles) * n_tiles;
        for (u128 q_begin = 0; q_begin <= total; ++q_begin) {
          const u128 q_end = total - q_begin;
          if (q_begin > q_end) continue;
          const auto actual = counter.class_bytes(all_m, q_begin, q_end, 13);
          const auto expected = enumerate(all_m, n_tiles, nx, ny,
                                           q_begin, q_end, 13);
          compare(actual, expected, "grid cut/moment oracle");
          ++cases;
        }
      }
    }
  }
  require(cases > 1000, "grid moment oracle coverage");
}

void check_large_repeat_constant_state() {
  constexpr u64 m_tiles = 1000000000000ULL;
  constexpr u64 n_tiles = 5;
  constexpr u64 nx = 8, ny = 8;
  const route::GemmOSRouteCounter counter(m_tiles, n_tiles, nx, ny);
  // This is the N-tail-column class: one AP spanning a trillion M rows.
  const route::Rect tail{0, m_tiles, n_tiles - 1, 1};
  const u128 total = static_cast<u128>(m_tiles) * n_tiles;
  // Scalar payload multiplier stands for repeated K/executions; it does not
  // expand the trillion-row AP into individual consumptions.
  constexpr u128 bytes_per_tile = 1000000000000ULL;
  const auto whole = counter.class_bytes(tail, 0, total, bytes_per_tile);
  const u128 split_q = total / 3;
  const auto left = counter.class_bytes(tail, 0, split_q, bytes_per_tile);
  const auto right = counter.class_bytes(tail, split_q, total, bytes_per_tile);
  require(whole.output_tiles == m_tiles, "large AP count");
  require(whole.bytes == static_cast<u128>(m_tiles) * bytes_per_tile,
          "large AP bytes");
  require(left.output_tiles + right.output_tiles == whole.output_tiles &&
              left.bytes + right.bytes == whole.bytes &&
              left.x_byte_sum + right.x_byte_sum == whole.x_byte_sum &&
              left.y_byte_sum + right.y_byte_sum == whole.y_byte_sum &&
              left.x_left_bytes + right.x_left_bytes == whole.x_left_bytes &&
              left.y_low_bytes + right.y_low_bytes == whole.y_low_bytes,
          "large AP prefix partition must be additive");
  require(whole.x_left_bytes + whole.x_right_bytes == whole.bytes &&
              whole.y_low_bytes + whole.y_high_bytes == whole.bytes,
          "large AP cuts conserve bytes");
}

void verify() {
  static_assert(!route::GemmOSRouteCounter::has_second_moments(),
                "partial API must not imply second-moment support");
  check_residue_floor_sum();
  check_class_rectangles();
  check_core_grids();
  check_large_repeat_constant_state();
}

}  // namespace

int main() {
  try {
    verify();
    std::cout << "gemm_os_route_counter_test: PASS (partial first moments; no second moments)\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "gemm_os_route_counter_test: FAIL: " << error.what() << '\n';
    return 1;
  }
}
