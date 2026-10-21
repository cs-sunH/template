#pragma once

// Bounded arithmetic for byte-weighted OS GEMM destination statistics.
//
// This helper is intentionally independent of Mapping.cc and GemmOSGeometry.
// Its Rect input matches one M/N rectangle from an AccessClass, and q is the
// row-major OS output-tile index. It never allocates or walks per-core or
// per-tile state. Current version provides exact byte-weighted counts,
// first coordinate moments, and x/y cut counts. It deliberately does NOT
// provide second moments yet; uniform-source hop distances must not consume
// this partial API until those moments have a verified bounded derivation.

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace coarse_die::detail::gemm_os_route_counter {

using u64 = std::uint64_t;
using u128 = unsigned __int128;

inline constexpr bool kSecondMomentsAvailable = false;

struct Rect {
  u64 m_tile_begin = 0;
  u64 m_tile_count = 0;
  u64 n_tile_begin = 0;
  u64 n_tile_count = 0;
};

struct ByteMoments {
  u128 output_tiles = 0;
  u128 bytes = 0;
  u128 x_byte_sum = 0;
  u128 y_byte_sum = 0;
  u128 x_left_bytes = 0;
  u128 x_right_bytes = 0;
  u128 y_low_bytes = 0;
  u128 y_high_bytes = 0;
  // Explicit marker: x/y second moments are intentionally unavailable.
  bool has_second_moments = false;
};

namespace detail {

inline u128 checked_add(u128 a, u128 b, const char* what) {
  if (b > ~u128{0} - a)
    throw std::overflow_error(std::string("overflow: OS route ") + what);
  return a + b;
}

inline u128 checked_mul(u128 a, u128 b, const char* what) {
  if (a != 0 && b > ~u128{0} / a)
    throw std::overflow_error(std::string("overflow: OS route ") + what);
  return a * b;
}

inline u128 ceil_div(u128 value, u64 divisor) {
  if (divisor == 0) throw std::invalid_argument("OS route divisor is zero");
  return value / divisor + static_cast<u128>(value % divisor != 0);
}

// Sum floor((a*i+b)/m), 0 <= i < n, by Euclidean reduction. Inputs are
// nonnegative. The caller's checked workload bounds keep every intermediate
// representable; this routine still checks each accumulation/product.
inline u128 floor_sum(u128 n, u128 m, u128 a, u128 b) {
  if (m == 0) throw std::invalid_argument("OS route floor_sum modulus is zero");
  u128 answer = 0;
  while (true) {
    if (a >= m) {
      const u128 quotient = a / m;
      u128 ta = n, tb = n ? n - 1 : 0;
      if ((ta & 1) == 0) ta /= 2;
      else tb /= 2;
      const u128 triangle = checked_mul(ta, tb, "floor_sum triangle");
      answer = checked_add(answer,
          checked_mul(triangle, quotient, "floor_sum slope reduction"),
          "floor_sum slope reduction");
      a %= m;
    }
    if (b >= m) {
      answer = checked_add(answer,
          checked_mul(n, b / m, "floor_sum offset reduction"),
          "floor_sum offset reduction");
      b %= m;
    }
    const u128 y_max = checked_add(checked_mul(a, n, "floor_sum y max"), b,
                                   "floor_sum y max");
    if (y_max < m) return answer;
    n = y_max / m;
    b = y_max % m;
    const u128 old_m = m;
    m = a;
    a = old_m;
  }
}

// Number of i with (b+a*i) mod m < t. This is a threshold difference of two
// floor sums, valid for t in [0,m] and for non-coprime strides.
inline u128 residue_count_lt(u128 n, u64 m, u128 a, u128 b, u64 t) {
  if (m == 0 || t > m)
    throw std::invalid_argument("OS route residue threshold is invalid");
  if (n == 0 || t == 0) return 0;
  if (t == m) return n;
  const u128 b_mod = b % m;
  const u128 a_mod = a % m;
  const u128 upper = floor_sum(n, m, a_mod, b_mod + m - t);
  const u128 lower = floor_sum(n, m, a_mod, b_mod);
  if (upper < lower || upper - lower > n)
    throw std::logic_error("OS route residue threshold invariant failed");
  return n - (upper - lower);
}

struct Ap {
  u128 first = 0;
  u64 step = 1;
  u128 count = 0;
  int sign = 1;
};

struct Terms {
  std::array<Ap, 4> values{};
  std::size_t size = 0;

  void add(u128 first, u64 step, u128 count, int sign = 1) {
    if (count == 0) return;
    if (sign != 1 && sign != -1)
      throw std::invalid_argument("OS route AP sign must be +/-1");
    if (size == values.size())
      throw std::logic_error("OS route fixed AP term bound exceeded");
    values[size++] = {first, step, count, sign};
  }
};

inline void add_interval(Terms& terms, u128 begin, u128 end, int sign = 1) {
  if (end > begin) {
    terms.add(begin, 1, end - begin, sign);
  }
}

inline u128 count_for_m_range(u64 m_begin, u64 m_end, u64 n_tiles,
                              u64 n_offset, u128 q_begin, u128 q_end,
                              Terms& terms, int sign = 1) {
  if (m_end <= m_begin || q_end <= q_begin) return 0;
  const u128 first_q = checked_add(
      checked_mul(m_begin, n_tiles, "row AP first"), n_offset,
      "row AP first");
  const auto m_at_or_after = [&](u128 q) -> u128 {
    if (q <= n_offset) return 0;
    return ceil_div(q - n_offset, n_tiles);
  };
  const u128 lo_wide = m_at_or_after(q_begin);
  const u128 hi_wide = m_at_or_after(q_end);
  const u64 lo = static_cast<u64>(lo_wide < m_begin ? m_begin :
      (lo_wide > m_end ? m_end : lo_wide));
  const u64 hi = static_cast<u64>(hi_wide < m_begin ? m_begin :
      (hi_wide > m_end ? m_end : hi_wide));
  if (hi <= lo) return 0;
  const u128 first = checked_add(checked_mul(lo, n_tiles, "row AP start"),
                                 n_offset, "row AP start");
  const u128 count = hi - lo;
  (void)first_q;
  terms.add(first, n_tiles, count, sign);
  return count;
}

inline Terms class_terms(const Rect& rect, u64 n_tiles, u128 q_begin,
                         u128 q_end) {
  if (n_tiles == 0 || rect.m_tile_count == 0 || rect.n_tile_count == 0 ||
      q_begin > q_end)
    throw std::invalid_argument("OS route class rectangle/range is invalid");
  const u128 m_end_wide = checked_add(rect.m_tile_begin, rect.m_tile_count,
                                      "M class end");
  const u128 n_end_wide = checked_add(rect.n_tile_begin, rect.n_tile_count,
                                      "N class end");
  if (m_end_wide > std::numeric_limits<u64>::max() ||
      n_end_wide > std::numeric_limits<u64>::max())
    throw std::overflow_error("OS route class coordinate exceeds u64");
  const u64 m_end = static_cast<u64>(m_end_wide);
  const u64 n_end = static_cast<u64>(n_end_wide);
  if (n_end > n_tiles)
    throw std::invalid_argument("OS route N class exceeds tile row");

  Terms terms;
  // Entire N row: one contiguous output-q range, clipped by the class M range.
  if (rect.n_tile_begin == 0 && rect.n_tile_count == n_tiles) {
    const u128 row_begin = checked_mul(rect.m_tile_begin, n_tiles,
                                       "full-N class q begin");
    const u128 row_end = checked_mul(m_end, n_tiles,
                                     "full-N class q end");
    add_interval(terms, q_begin > row_begin ? q_begin : row_begin,
                 q_end < row_end ? q_end : row_end);
    return terms;
  }

  // A full-N prefix with exactly one final tail column omitted. The complete
  // rows are one interval minus the tail-column AP; at most two clipped
  // boundary rows remain as separate intervals.
  if (rect.n_tile_begin == 0 && rect.n_tile_count < n_tiles &&
      rect.n_tile_count == n_tiles - 1) {
    if (rect.n_tile_count == 0) return terms;
    u128 full_begin = ceil_div(q_begin, n_tiles);
    u128 full_end = q_end / n_tiles;
    if (full_begin < rect.m_tile_begin) full_begin = rect.m_tile_begin;
    if (full_end > m_end) full_end = m_end;
    if (full_end > full_begin) {
      const u128 q0 = checked_mul(full_begin, n_tiles,
                                  "full-row interval begin");
      const u128 q1 = checked_mul(full_end, n_tiles,
                                  "full-row interval end");
      add_interval(terms, q0, q1);
      terms.add(checked_add(q0, rect.n_tile_count,
                            "tail-column AP first"),
                n_tiles, full_end - full_begin, -1);
    }

    std::array<u64, 2> boundary_rows{};
    std::size_t boundary_count = 0;
    if (q_begin < q_end && q_begin % n_tiles != 0) {
      const u128 row = q_begin / n_tiles;
      if (row >= rect.m_tile_begin && row < m_end)
        boundary_rows[boundary_count++] = static_cast<u64>(row);
    }
    if (q_begin < q_end && q_end % n_tiles != 0) {
      const u128 row = (q_end - 1) / n_tiles;
      if (row >= rect.m_tile_begin && row < m_end &&
          (boundary_count == 0 || boundary_rows[0] != row))
        boundary_rows[boundary_count++] = static_cast<u64>(row);
    }
    for (std::size_t i = 0; i < boundary_count; ++i) {
      const u128 row_begin = checked_mul(boundary_rows[i], n_tiles,
                                         "partial-row begin");
      const u128 class_end = checked_add(row_begin, rect.n_tile_count,
                                         "partial-row class end");
      add_interval(terms, q_begin > row_begin ? q_begin : row_begin,
                   q_end < class_end ? q_end : class_end);
    }
    return terms;
  }

  // N-tail class: one q per M row, hence one clipped AP.
  if (rect.n_tile_count == 1 && rect.n_tile_begin == n_tiles - 1) {
    count_for_m_range(rect.m_tile_begin, m_end, n_tiles,
                      rect.n_tile_begin, q_begin, q_end, terms);
    return terms;
  }

  throw std::invalid_argument(
      "OS route class must be full N, full-N prefix, or N-tail column");
}

struct RawMoments {
  u128 count = 0;
  u128 x_sum = 0;
  u128 y_sum = 0;
  u128 x_left = 0;
  u128 y_low = 0;
};

inline RawMoments ap_moments(const Ap& ap, u64 nx, u64 ny) {
  if (nx == 0 || ny == 0)
    throw std::invalid_argument("OS route grid dimensions must be positive");
  const u128 core_count_wide = checked_mul(nx, ny, "core grid");
  if (core_count_wide > std::numeric_limits<u64>::max())
    throw std::overflow_error("OS route core grid exceeds u64");
  const u64 core_count = static_cast<u64>(core_count_wide);
  if (ap.step == 0) throw std::invalid_argument("OS route AP step is zero");
  RawMoments result;
  result.count = ap.count;
  if (ap.count == 0) return result;

  const u128 x_step = ap.step % nx;
  const u128 x_first = ap.first % nx;
  const u128 x_floor = floor_sum(ap.count, nx, x_step, x_first);
  u128 ta = ap.count, tb = ap.count - 1;
  if ((ta & 1) == 0) ta /= 2;
  else tb /= 2;
  const u128 x_linear = checked_add(
      checked_mul(ap.count, x_first, "AP x coordinate sum"),
      checked_mul(x_step, checked_mul(ta, tb, "AP triangle"),
                  "AP x coordinate sum"),
      "AP x coordinate sum");
  const u128 x_floor_scaled = checked_mul(nx, x_floor, "x coordinate sum");
  if (x_linear < x_floor_scaled)
    throw std::logic_error("OS route x moment became negative");
  result.x_sum = x_linear - x_floor_scaled;

  // C is a multiple of nx, so the common quotient of q/C cancels. Reducing
  // the AP base and stride modulo C keeps the intermediate sums bounded.
  const u128 grid_step = ap.step % core_count;
  const u128 grid_first = ap.first % core_count;
  const u128 x_floor_for_y = floor_sum(ap.count, nx, grid_step, grid_first);
  const u128 grid_floor = floor_sum(ap.count, core_count, grid_step, grid_first);
  const u128 y_floor_scaled = checked_mul(ny, grid_floor,
                                          "y coordinate sum");
  if (x_floor_for_y < y_floor_scaled)
    throw std::logic_error("OS route y moment became negative");
  result.y_sum = x_floor_for_y - y_floor_scaled;

  const u64 cx = nx / 2;
  const u64 cy = ny / 2;
  result.x_left = residue_count_lt(ap.count, nx, x_step, x_first, cx);
  result.y_low = residue_count_lt(ap.count, core_count, grid_step, grid_first,
                                  static_cast<u64>(nx * cy));
  return result;
}

inline u128 signed_sum(const Terms& terms, u128 (*select)(const RawMoments&),
                       u64 nx, u64 ny) {
  u128 positive = 0, negative = 0;
  for (std::size_t i = 0; i < terms.size; ++i) {
    const RawMoments raw = ap_moments(terms.values[i], nx, ny);
    if (terms.values[i].sign > 0)
      positive = checked_add(positive, select(raw), "positive AP sum");
    else
      negative = checked_add(negative, select(raw), "negative AP sum");
  }
  if (negative > positive)
    throw std::logic_error("OS route signed AP sum became negative");
  return positive - negative;
}

inline u128 select_count(const RawMoments& m) { return m.count; }
inline u128 select_x_sum(const RawMoments& m) { return m.x_sum; }
inline u128 select_y_sum(const RawMoments& m) { return m.y_sum; }
inline u128 select_x_left(const RawMoments& m) { return m.x_left; }
inline u128 select_y_low(const RawMoments& m) { return m.y_low; }

}  // namespace detail

class GemmOSRouteCounter {
 public:
  GemmOSRouteCounter(u64 m_tiles, u64 n_tiles, u64 nx, u64 ny)
      : m_tiles_(m_tiles), n_tiles_(n_tiles), nx_(nx), ny_(ny),
        output_tiles_(detail::checked_mul(m_tiles, n_tiles,
                                          "output tile count")) {
    if (!m_tiles_ || !n_tiles_ || !nx_ || !ny_)
      throw std::invalid_argument("OS route geometry dimensions must be positive");
    if (detail::checked_mul(nx_, ny_, "core grid") >
        std::numeric_limits<u64>::max())
      throw std::overflow_error("OS route core grid exceeds u64");
  }

  static constexpr bool has_second_moments() noexcept {
    return kSecondMomentsAvailable;
  }

  // bytes_per_output_tile already includes the AccessCounter per-consumption
  // bytes times the selected K multiplicity and execution repeat. q_begin/end
  // select the cumulative primitive's output-tile range.
  ByteMoments class_bytes(const Rect& rect, u128 q_begin, u128 q_end,
                          u128 bytes_per_output_tile) const {
    if (q_begin > q_end || q_end > output_tiles_)
      throw std::out_of_range("OS route q range exceeds output geometry");
    const u128 m_end = detail::checked_add(rect.m_tile_begin,
        rect.m_tile_count, "M class end");
    if (m_end > m_tiles_)
      throw std::out_of_range("OS route M class exceeds output geometry");
    const detail::Terms terms =
        detail::class_terms(rect, n_tiles_, q_begin, q_end);
    const u128 count = detail::signed_sum(terms, detail::select_count,
                                          nx_, ny_);
    const u128 x_sum = detail::signed_sum(terms, detail::select_x_sum,
                                          nx_, ny_);
    const u128 y_sum = detail::signed_sum(terms, detail::select_y_sum,
                                          nx_, ny_);
    const u128 x_left = detail::signed_sum(terms, detail::select_x_left,
                                           nx_, ny_);
    const u128 y_low = detail::signed_sum(terms, detail::select_y_low,
                                          nx_, ny_);
    ByteMoments result;
    result.output_tiles = count;
    result.bytes = detail::checked_mul(count, bytes_per_output_tile,
                                       "class byte count");
    result.x_byte_sum = detail::checked_mul(x_sum, bytes_per_output_tile,
                                            "weighted x moment");
    result.y_byte_sum = detail::checked_mul(y_sum, bytes_per_output_tile,
                                            "weighted y moment");
    result.x_left_bytes = detail::checked_mul(x_left, bytes_per_output_tile,
                                              "weighted x cut");
    result.x_right_bytes = detail::checked_mul(count - x_left,
                                               bytes_per_output_tile,
                                               "weighted x cut");
    result.y_low_bytes = detail::checked_mul(y_low, bytes_per_output_tile,
                                             "weighted y cut");
    result.y_high_bytes = detail::checked_mul(count - y_low,
                                              bytes_per_output_tile,
                                              "weighted y cut");
    result.has_second_moments = false;
    return result;
  }

 private:
  u64 m_tiles_;
  u64 n_tiles_;
  u64 nx_;
  u64 ny_;
  u128 output_tiles_;
};

}  // namespace coarse_die::detail::gemm_os_route_counter
