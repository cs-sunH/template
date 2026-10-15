#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <map>
#include <string>
#include <tuple>
#include <vector>

namespace coarse_die::detail {

struct RatioRow {
  std::vector<double> a;
  double b = 0.0;
};

namespace ratio_allocation_internal {

using Wide = long double;

struct Constraint {
  std::vector<Wide> a;
  Wide b = 0.0L;
};

struct Edge {
  std::size_t from = 0;
  std::size_t to = 0;
  Wide gain = 0.0L;
};

inline bool finite(Wide value) { return std::isfinite(value); }

// The inputs are binary64 model values. Near-ties closer than this relative
// envelope can change a breakpoint/bottleneck under ordinary input rounding;
// the caller should use the canonical LP for those cases.
inline bool nearly_equal(Wide lhs, Wide rhs) {
  if (lhs == rhs) return false;
  const Wide scale = std::max(std::abs(lhs), std::abs(rhs));
  if (scale == 0.0L) return false;
  return std::abs(lhs - rhs) <=
      64.0L * static_cast<Wide>(std::numeric_limits<double>::epsilon()) * scale;
}

inline bool arithmetic_nearly_equal(Wide lhs, Wide rhs) {
  if (lhs == rhs) return false;
  const Wide scale = std::max(std::abs(lhs), std::abs(rhs));
  if (scale == 0.0L) return false;
  return std::abs(lhs - rhs) <=
      64.0L * std::numeric_limits<Wide>::epsilon() * scale;
}

inline bool checked_product(Wide lhs, Wide rhs, Wide& result) {
  if (!finite(lhs) || !finite(rhs) || lhs < 0.0L || rhs < 0.0L) return false;
  result = lhs * rhs;
  return finite(result) && (lhs == 0.0L || rhs == 0.0L || result > 0.0L);
}

inline bool checked_sum(Wide lhs, Wide rhs, Wide& result) {
  if (!finite(lhs) || !finite(rhs) || lhs < 0.0L || rhs < 0.0L) return false;
  result = lhs + rhs;
  return finite(result);
}

inline bool maximum(Wide lhs, Wide rhs, Wide& result) {
  if (lhs != rhs && nearly_equal(lhs, rhs)) return false;
  result = std::max(lhs, rhs);
  return finite(result);
}

inline bool cycle_is_nonamplifying(const std::vector<std::vector<Wide>>& edge,
                                   const std::vector<std::size_t>& nodes) {
  Wide product = 1.0L;
  for (std::size_t i = 0; i < nodes.size(); ++i) {
    const Wide gain = edge[nodes[i]][nodes[(i + 1) % nodes.size()]];
    Wide next = 0.0L;
    if (!(gain > 0.0L) || !checked_product(product, gain, next)) return false;
    product = next;
  }
  if (product == 1.0L) return true;
  if (nearly_equal(product, 1.0L)) return false;
  // A gain product above one forces the whole cycle to zero. This bounded
  // fast path deliberately leaves zero exclusion and owner-Q restart to LP.
  return product < 1.0L;
}

inline bool build_gain_closure(std::size_t count,
                               const std::vector<Edge>& edges,
                               std::vector<std::vector<Wide>>& gain) {
  std::vector<std::vector<Wide>> direct(
      count, std::vector<Wide>(count, 0.0L));
  for (const Edge& edge : edges) {
    Wide& old = direct[edge.from][edge.to];
    if (old != 0.0L && edge.gain != old && nearly_equal(edge.gain, old))
      return false;
    old = std::max(old, edge.gain);
  }

  for (std::size_t i = 0; i < count; ++i) {
    for (std::size_t j = i + 1; j < count; ++j) {
      if (direct[i][j] > 0.0L && direct[j][i] > 0.0L &&
          !cycle_is_nonamplifying(direct, {i, j})) return false;
    }
  }
  if (count == 3) {
    if (direct[0][1] > 0.0L && direct[1][2] > 0.0L && direct[2][0] > 0.0L &&
        !cycle_is_nonamplifying(direct, {0, 1, 2})) return false;
    if (direct[0][2] > 0.0L && direct[2][1] > 0.0L && direct[1][0] > 0.0L &&
        !cycle_is_nonamplifying(direct, {0, 2, 1})) return false;
  }

  gain.assign(count, std::vector<Wide>(count, 0.0L));
  for (std::size_t i = 0; i < count; ++i) {
    gain[i][i] = 1.0L;
    for (std::size_t j = 0; j < count; ++j) {
      if (direct[i][j] > 0.0L) gain[i][j] = direct[i][j];
    }
  }
  // Floyd-Warshall over the max-times semiring. With at most three vertices,
  // non-amplifying cycles never improve a path; the closure is fixed work.
  for (std::size_t k = 0; k < count; ++k) {
    for (std::size_t i = 0; i < count; ++i) {
      if (gain[i][k] == 0.0L) continue;
      for (std::size_t j = 0; j < count; ++j) {
        if (gain[k][j] == 0.0L) continue;
        Wide through = 0.0L;
        if (!checked_product(gain[i][k], gain[k][j], through)) return false;
        Wide closed = 0.0L;
        if (!maximum(gain[i][j], through, closed)) return false;
        gain[i][j] = closed;
      }
    }
  }
  return true;
}

inline bool closure_lines_valid(const std::vector<Wide>& constant,
                                const std::vector<Wide>& slope,
                                const std::vector<Edge>& edges) {
  for (const Edge& edge : edges) {
    Wide lower_constant = 0.0L, lower_slope = 0.0L;
    if (!checked_product(edge.gain, constant[edge.from], lower_constant) ||
        !checked_product(edge.gain, slope[edge.from], lower_slope)) return false;
    if (lower_constant > constant[edge.to] || lower_slope > slope[edge.to]) {
      const bool ambiguous_constant =
          lower_constant != constant[edge.to] &&
          nearly_equal(lower_constant, constant[edge.to]);
      const bool ambiguous_slope =
          lower_slope != slope[edge.to] &&
          nearly_equal(lower_slope, slope[edge.to]);
      if (ambiguous_constant || ambiguous_slope ||
          lower_constant > constant[edge.to] || lower_slope > slope[edge.to])
        return false;
    }
  }
  return true;
}

inline bool closure_for_tier(const std::vector<std::vector<Wide>>& gain,
                             const std::vector<Edge>& edges,
                             const std::vector<bool>& frozen,
                             const std::vector<Wide>& frozen_value,
                             std::vector<Wide>& constant,
                             std::vector<Wide>& slope) {
  const std::size_t count = frozen.size();
  constant.assign(count, 0.0L);
  slope.assign(count, 0.0L);
  for (std::size_t destination = 0; destination < count; ++destination) {
    for (std::size_t source = 0; source < count; ++source) {
      if (gain[source][destination] == 0.0L) continue;
      if (frozen[source]) {
        Wide path_floor = 0.0L, next = 0.0L;
        if (!checked_product(frozen_value[source], gain[source][destination], path_floor) ||
            !maximum(constant[destination], path_floor, next)) return false;
        constant[destination] = next;
      } else {
        Wide next = 0.0L;
        if (!maximum(slope[destination], gain[source][destination], next)) return false;
        slope[destination] = next;
      }
    }
  }
  return closure_lines_valid(constant, slope, edges);
}

inline bool closure_for_probe(const std::vector<std::vector<Wide>>& gain,
                              const std::vector<Edge>& edges,
                              const std::vector<bool>& frozen,
                              const std::vector<Wide>& frozen_value,
                              std::size_t raised_activity,
                              Wide tier_floor,
                              std::vector<Wide>& constant,
                              std::vector<Wide>& slope) {
  const std::size_t count = frozen.size();
  constant.assign(count, 0.0L);
  slope.assign(count, 0.0L);
  for (std::size_t destination = 0; destination < count; ++destination) {
    for (std::size_t source = 0; source < count; ++source) {
      const Wide path_gain = gain[source][destination];
      if (path_gain == 0.0L) continue;
      if (frozen[source]) {
        Wide path_floor = 0.0L, next = 0.0L;
        if (!checked_product(frozen_value[source], path_gain, path_floor) ||
            !maximum(constant[destination], path_floor, next)) return false;
        constant[destination] = next;
      } else if (source == raised_activity) {
        Wide next = 0.0L;
        if (!maximum(slope[destination], path_gain, next)) return false;
        slope[destination] = next;
      } else {
        Wide path_floor = 0.0L, next = 0.0L;
        if (!checked_product(tier_floor, path_gain, path_floor) ||
            !maximum(constant[destination], path_floor, next)) return false;
        constant[destination] = next;
      }
    }
  }
  return closure_lines_valid(constant, slope, edges);
}

inline bool row_use(const Constraint& row,
                    const std::vector<Wide>& constant,
                    const std::vector<Wide>& slope,
                    Wide parameter, Wide& use) {
  use = 0.0L;
  for (std::size_t i = 0; i < row.a.size(); ++i) {
    if (row.a[i] == 0.0L) continue;
    Wide rising = 0.0L;
    if (!checked_product(slope[i], parameter, rising)) return false;
    const Wide value = std::max(constant[i], rising);
    Wide term = 0.0L, next = 0.0L;
    if (!checked_product(row.a[i], value, term) ||
        !checked_sum(use, term, next)) return false;
    use = next;
  }
  return true;
}

// Return the last feasible parameter for one nonnegative physical/private row.
// Each variable contributes max(constant, slope*x), so at most count<=3
// breakpoints divide the row into affine segments; no iterative search is used.
inline bool row_limit(const Constraint& row,
                      const std::vector<Wide>& constant,
                      const std::vector<Wide>& slope,
                      Wide lower,
                      Wide& limit,
                      bool& bounded) {
  bounded = false;
  limit = std::numeric_limits<Wide>::infinity();
  Wide initial_use = 0.0L;
  if (!row_use(row, constant, slope, lower, initial_use) ||
      initial_use > row.b) return false;

  if (row.a.size() > 3) return false;
  std::array<Wide, 3> breaks{};
  std::size_t break_count = 0;
  for (std::size_t i = 0; i < row.a.size(); ++i) {
    if (row.a[i] <= 0.0L || slope[i] <= 0.0L) continue;
    const Wide point = constant[i] / slope[i];
    if (!finite(point) || point < 0.0L) return false;
    if (point != lower && arithmetic_nearly_equal(point, lower)) return false;
    if (point > lower) breaks[break_count++] = point;
  }
  std::sort(breaks.begin(), breaks.begin() + break_count);
  for (std::size_t i = 1; i < break_count; ++i) {
    if (breaks[i] != breaks[i - 1] &&
        arithmetic_nearly_equal(breaks[i], breaks[i - 1]))
      return false;
  }
  break_count = static_cast<std::size_t>(
      std::unique(breaks.begin(), breaks.begin() + break_count) - breaks.begin());

  Wide start = lower;
  for (std::size_t segment = 0; segment <= break_count; ++segment) {
    const bool has_end = segment < break_count;
    const Wide end = has_end ? breaks[segment]
                             : std::numeric_limits<Wide>::infinity();
    Wide intercept = 0.0L, row_slope = 0.0L;
    for (std::size_t i = 0; i < row.a.size(); ++i) {
      if (row.a[i] == 0.0L) continue;
      Wide rising = 0.0L;
      if (!checked_product(slope[i], start, rising)) return false;
      Wide term = 0.0L, next = 0.0L;
      if (slope[i] > 0.0L && rising >= constant[i]) {
        if (!checked_product(row.a[i], slope[i], term) ||
            !checked_sum(row_slope, term, next)) return false;
        row_slope = next;
      } else {
        if (!checked_product(row.a[i], constant[i], term) ||
            !checked_sum(intercept, term, next)) return false;
        intercept = next;
      }
    }
    Wide start_use = 0.0L, rising = 0.0L;
    if (!checked_product(row_slope, start, rising) ||
        !checked_sum(intercept, rising, start_use) ||
        start_use > row.b) return false;

    if (row_slope > 0.0L) {
      const Wide candidate = (row.b - intercept) / row_slope;
      if (!finite(candidate)) return false;
      if (candidate < start) {
        if (!arithmetic_nearly_equal(candidate, start) || start_use != row.b)
          return false;
        limit = start;
        bounded = true;
        return true;
      }
      if (!has_end || candidate <= end) {
        limit = candidate;
        bounded = true;
        return true;
      }
    } else if (intercept > row.b) {
      return false;
    }
    if (!has_end) break;
    start = end;
  }
  return true;
}

inline bool maximum_feasible_parameter(
    const std::vector<Constraint>& rows,
    const std::vector<Wide>& constant,
    const std::vector<Wide>& slope,
    Wide lower,
    Wide& maximum) {
  maximum = std::numeric_limits<Wide>::infinity();
  bool found_bound = false;
  for (const Constraint& row : rows) {
    Wide row_max = 0.0L;
    bool bounded = false;
    if (!row_limit(row, constant, slope, lower, row_max, bounded)) return false;
    if (!bounded) continue;
    if (!found_bound) {
      maximum = row_max;
      found_bound = true;
    } else {
      if (row_max != maximum &&
          arithmetic_nearly_equal(row_max, maximum)) return false;
      maximum = std::min(maximum, row_max);
    }
  }
  if (!found_bound || !finite(maximum) || maximum < lower) return false;
  for (const Constraint& row : rows) {
    Wide used = 0.0L;
    if (!row_use(row, constant, slope, maximum, used) || used > row.b) return false;
  }
  return true;
}

inline bool rows_feasible(const std::vector<Constraint>& rows,
                          const std::vector<Wide>& value) {
  for (const Constraint& row : rows) {
    Wide used = 0.0L;
    for (std::size_t i = 0; i < value.size(); ++i) {
      if (row.a[i] == 0.0L) continue;
      Wide term = 0.0L, next = 0.0L;
      if (!checked_product(row.a[i], value[i], term) ||
          !checked_sum(used, term, next)) return false;
      used = next;
    }
    if (used > row.b) return false;
  }
  return true;
}

inline bool certify_and_round(const std::vector<RatioRow>& input_rows,
                              std::vector<double>& rates) {
  constexpr std::size_t kMaxCorrectionUlps = 8;
  const std::size_t count = rates.size();
  std::vector<std::size_t> corrections(count, 0);
  const auto row_use_binary64 = [&](const RatioRow& row, Wide& used) {
    used = 0.0L;
    for (std::size_t i = 0; i < count; ++i) {
      const Wide term = static_cast<Wide>(row.a[i]) *
                        static_cast<Wide>(rates[i]);
      if (!finite(term)) return false;
      used += term;
      if (!finite(used)) return false;
    }
    return true;
  };

  // A correction always lowers a positive row contributor. Since signed
  // rows can couple opposite directions, recertify every original row after
  // each pass and fall back if a bounded ulp correction does not settle.
  for (std::size_t pass = 0; pass <= count * 8; ++pass) {
    bool changed = false;
    for (const RatioRow& row : input_rows) {
      Wide used = 0.0L;
      if (!row_use_binary64(row, used)) return false;
      if (used <= static_cast<Wide>(row.b)) continue;

      std::size_t chosen = count;
      double chosen_rate = 0.0;
      std::size_t chosen_steps = kMaxCorrectionUlps + 1;
      for (std::size_t i = 0; i < count; ++i) {
        if (row.a[i] <= 0.0 || rates[i] <= 0.0 ||
            corrections[i] >= kMaxCorrectionUlps) continue;
        const Wide other = used - static_cast<Wide>(row.a[i]) * rates[i];
        const Wide bound = (static_cast<Wide>(row.b) - other) /
                           static_cast<Wide>(row.a[i]);
        if (!finite(bound) || bound < 0.0L) continue;
        double candidate = rates[i];
        for (std::size_t steps = 1;
             steps <= kMaxCorrectionUlps - corrections[i]; ++steps) {
          candidate = std::nextafter(candidate, 0.0);
          if (static_cast<Wide>(candidate) <= bound) {
            if (steps < chosen_steps) {
              chosen = i;
              chosen_rate = candidate;
              chosen_steps = steps;
            }
            break;
          }
        }
      }
      if (chosen == count) return false;
      rates[chosen] = chosen_rate;
      corrections[chosen] += chosen_steps;
      changed = true;
    }
    if (!changed) return true;
  }
  return false;
}

}  // namespace ratio_allocation_internal

// Lexicographic max-min for at most three activities coupled by zero-bound
// signed pair rows. The owner utility is W_owner * rate; physical/private
// nonnegative rows are solved in those utility coordinates. Unsupported rows,
// zero exclusions, gain-amplifying cycles, numeric ambiguity, and any failed
// final certificate leave outputs untouched for the caller's LP fallback.
inline bool allocate_ratio(const std::vector<RatioRow>& rows,
                           const std::vector<std::string>& owners,
                           const std::vector<double>& canonical_q,
                           std::vector<double>& rates,
                           std::vector<double>& alpha) {
  using namespace ratio_allocation_internal;
  try {
    const std::size_t count = owners.size();
    if (count < 2 || count > 3 || canonical_q.size() != count ||
        rows.empty() || rows.size() > 128) return false;

    std::map<std::string, Wide> owner_totals;
    for (std::size_t i = 0; i < count; ++i) {
      if (owners[i].empty() || !std::isfinite(canonical_q[i]) ||
          canonical_q[i] <= 0.0) return false;
      owner_totals[owners[i]] += static_cast<Wide>(canonical_q[i]);
      if (!finite(owner_totals[owners[i]]) || owner_totals[owners[i]] <= 0.0L)
        return false;
    }
    std::vector<Wide> owner_weight(count, 0.0L);
    std::vector<double> result_alpha(count, 0.0);
    for (std::size_t i = 0; i < count; ++i) {
      owner_weight[i] = owner_totals.at(owners[i]);
      const double share = static_cast<double>(
          static_cast<Wide>(canonical_q[i]) / owner_weight[i]);
      if (!std::isfinite(share) || share <= 0.0 || share > 1.0) return false;
      result_alpha[i] = share;
    }

    std::vector<Constraint> physical_rows;
    std::vector<Edge> edges;
    std::vector<bool> individually_bounded(count, false);
    for (const RatioRow& row : rows) {
      if (row.a.size() != count || !std::isfinite(row.b) || row.b < 0.0)
        return false;
      std::size_t positive = count, negative = count;
      std::size_t positive_count = 0, negative_count = 0;
      for (std::size_t i = 0; i < count; ++i) {
        const double coefficient = row.a[i];
        if (!std::isfinite(coefficient)) return false;
        if (coefficient > 0.0) { positive = i; ++positive_count; }
        if (coefficient < 0.0) { negative = i; ++negative_count; }
      }
      if (negative_count == 0) {
        if (positive_count == 0) continue;
        // A zero-bound positive row requires zero exclusion and owner-Q
        // renormalization, which is intentionally left to the LP.
        if (row.b == 0.0) return false;
        Constraint physical;
        physical.a.assign(count, 0.0L);
        physical.b = static_cast<Wide>(row.b);
        for (std::size_t i = 0; i < count; ++i) {
          if (row.a[i] <= 0.0) continue;
          physical.a[i] = static_cast<Wide>(row.a[i]) / owner_weight[i];
          if (!finite(physical.a[i]) || physical.a[i] <= 0.0L) return false;
          individually_bounded[i] = true;
        }
        physical_rows.push_back(std::move(physical));
        continue;
      }
      if (positive_count == 0) {
        // A nonpositive row with nonnegative bound is redundant for rates>=0.
        continue;
      }
      if (positive_count != 1 || negative_count != 1 || row.b != 0.0)
        return false;

      const Wide positive_utility_coefficient =
          static_cast<Wide>(row.a[positive]) / owner_weight[positive];
      const Wide negative_utility_coefficient =
          -static_cast<Wide>(row.a[negative]) / owner_weight[negative];
      if (!(positive_utility_coefficient > 0.0L) ||
          !(negative_utility_coefficient > 0.0L) ||
          !finite(positive_utility_coefficient) ||
          !finite(negative_utility_coefficient)) return false;
      const Wide gain = positive_utility_coefficient /
                        negative_utility_coefficient;
      if (!(gain > 0.0L) || !finite(gain)) return false;
      edges.push_back({positive, negative, gain});
    }
    if (edges.empty()) return false;  // Leave nonnegative rows to PackingAllocation.
    if (std::any_of(individually_bounded.begin(), individually_bounded.end(),
                    [](bool value) { return !value; })) return false;

    std::vector<std::vector<Wide>> gain;
    if (!build_gain_closure(count, edges, gain)) return false;

    std::vector<bool> frozen(count, false);
    std::vector<Wide> frozen_value(count, 0.0L);
    Wide floor = 0.0L;
    std::vector<Wide> solution(count, 0.0L);
    std::size_t remaining = count;
    while (remaining != 0) {
      std::vector<Wide> constant, slope;
      if (!closure_for_tier(gain, edges, frozen, frozen_value,
                            constant, slope)) return false;

      std::vector<Constraint> constraints = physical_rows;
      for (std::size_t i = 0; i < count; ++i) {
        if (!frozen[i]) continue;
        Constraint fixed_upper{std::vector<Wide>(count, 0.0L), frozen_value[i]};
        fixed_upper.a[i] = 1.0L;
        constraints.push_back(std::move(fixed_upper));
      }

      Wide tier = 0.0L;
      if (!maximum_feasible_parameter(constraints, constant, slope, floor, tier) ||
          !(tier > 0.0L)) return false;  // Zero-tier restart is LP-only here.

      std::vector<std::size_t> forced;
      for (std::size_t raised = 0; raised < count; ++raised) {
        if (frozen[raised]) continue;
        std::vector<Wide> probe_constant, probe_slope;
        if (!closure_for_probe(gain, edges, frozen, frozen_value, raised,
                               tier, probe_constant, probe_slope)) return false;
        Wide max_floor = 0.0L;
        if (!maximum_feasible_parameter(constraints, probe_constant,
                                        probe_slope, tier, max_floor)) return false;
        if (max_floor < tier) return false;
        if (max_floor == tier) {
          forced.push_back(raised);
        } else if (arithmetic_nearly_equal(max_floor, tier)) {
          return false;
        }
      }
      if (forced.empty()) return false;
      for (std::size_t i : forced) {
        frozen[i] = true;
        frozen_value[i] = tier;
        solution[i] = tier;
        --remaining;
      }
      floor = tier;
    }

    // All utilities are now fixed. Recompute the exact ratio closure and
    // require every frozen value to satisfy its own propagated lower bound.
    std::vector<Wide> final_constant, final_slope;
    if (!closure_for_tier(gain, edges, frozen, frozen_value,
                          final_constant, final_slope)) return false;
    for (std::size_t i = 0; i < count; ++i) {
      if (final_constant[i] != frozen_value[i]) return false;
      solution[i] = frozen_value[i];
    }
    if (!rows_feasible(physical_rows, solution)) return false;

    std::vector<double> result_rates(count, 0.0);
    for (std::size_t i = 0; i < count; ++i) {
      const Wide rate = solution[i] / owner_weight[i];
      const double rounded = static_cast<double>(rate);
      if (!finite(rate) || !(rate > 0.0L) || !std::isfinite(rounded) ||
          !(rounded > 0.0) || (static_cast<Wide>(rounded) > rate &&
                               static_cast<Wide>(std::nextafter(rounded, 0.0)) > rate))
        return false;
      result_rates[i] = rounded;
    }
    if (!certify_and_round(rows, result_rates)) return false;
    rates.swap(result_rates);
    alpha.swap(result_alpha);
    return true;
  } catch (...) {
    return false;
  }
}

}  // namespace coarse_die::detail
