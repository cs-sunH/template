#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <map>
#include <string>
#include <vector>

namespace coarse_die::detail {

struct PackingRow {
  std::vector<double> a;
  double b = 0.0;
};

// Solve the weighted max-min problem for nonnegative packing constraints.
// Activity i's utility is owner_Q * rate_i, so its progress rate grows at
// 1 / owner_Q while it remains unfrozen. Signed rows and inputs outside the
// bounded numerical domain are left to the general LP solver.
inline bool allocate_packing(const std::vector<PackingRow>& rows,
                             const std::vector<std::string>& owners,
                             const std::vector<double>& canonical_q,
                             std::vector<double>& rates,
                             std::vector<double>& alpha) {
  try {
    const std::size_t count = owners.size();
    if (count != canonical_q.size() || count > 96 || rows.size() > 128) return false;

    for (std::size_t i = 0; i < count; ++i) {
      if (owners[i].empty() || !std::isfinite(canonical_q[i]) ||
          canonical_q[i] <= 0.0) return false;
    }
    for (const PackingRow& row : rows) {
      if (row.a.size() != count || !std::isfinite(row.b) || row.b < 0.0) return false;
      for (double coefficient : row.a) {
        if (!std::isfinite(coefficient) || coefficient < 0.0) return false;
      }
    }

    using Wide = long double;
    const Wide wide_max = std::numeric_limits<Wide>::max();
    std::vector<bool> excluded(count, false);
    // A zero-bound nonnegative row forces every activity with a positive
    // coefficient in that row to zero. Remove those activities before
    // computing owner_Q so they do not dilute the remaining activities.
    for (const PackingRow& row : rows) {
      if (row.b != 0.0) continue;
      for (std::size_t i = 0; i < count; ++i) {
        if (row.a[i] > 0.0) excluded[i] = true;
      }
    }

    std::map<std::string, Wide> owner_q;
    for (std::size_t i = 0; i < count; ++i) {
      if (excluded[i]) continue;
      owner_q[owners[i]] += static_cast<Wide>(canonical_q[i]);
      if (!std::isfinite(owner_q[owners[i]]) || owner_q[owners[i]] <= 0.0L ||
          owner_q[owners[i]] > wide_max) return false;
    }

    std::vector<Wide> weights(count, 0.0L);
    std::vector<Wide> wide_alpha(count, 0.0L);
    std::vector<Wide> wide_rates(count, 0.0L);
    std::vector<bool> frozen = excluded;
    for (std::size_t i = 0; i < count; ++i) {
      if (excluded[i]) continue;
      const Wide total = owner_q.at(owners[i]);
      weights[i] = 1.0L / total;
      wide_alpha[i] = static_cast<Wide>(canonical_q[i]) / total;
      if (!std::isfinite(weights[i]) || weights[i] <= 0.0L ||
          !std::isfinite(wide_alpha[i]) || wide_alpha[i] <= 0.0L) return false;
    }

    std::vector<Wide> residual(rows.size(), 0.0L);
    for (std::size_t r = 0; r < rows.size(); ++r)
      residual[r] = static_cast<Wide>(rows[r].b);

    std::size_t active = 0;
    for (bool value : frozen) if (!value) ++active;
    while (active != 0) {
      std::vector<Wide> slopes(rows.size(), 0.0L);
      std::vector<Wide> increments(rows.size(),
                                    std::numeric_limits<Wide>::infinity());
      Wide step = std::numeric_limits<Wide>::infinity();
      std::size_t binding_row = rows.size();

      for (std::size_t r = 0; r < rows.size(); ++r) {
        Wide slope = 0.0L;
        for (std::size_t i = 0; i < count; ++i) {
          if (frozen[i]) continue;
          slope += static_cast<Wide>(rows[r].a[i]) * weights[i];
        }
        if (!std::isfinite(slope) || slope < 0.0L) return false;
        slopes[r] = slope;
        if (slope == 0.0L) continue;
        if (!std::isfinite(residual[r]) || residual[r] < 0.0L) return false;
        const Wide candidate = residual[r] / slope;
        if (!std::isfinite(candidate) || candidate < 0.0L) return false;
        increments[r] = candidate;
        if (candidate < step) {
          step = candidate;
          binding_row = r;
        }
      }

      // With no positive-slope bounded row the remaining activities are
      // unbounded, so this helper cannot produce a finite LP-equivalent rate.
      if (binding_row == rows.size() || !std::isfinite(step) || step < 0.0L)
        return false;
      // All exact zero-capacity blockers were removed above. A later zero
      // increment therefore indicates numerical ambiguity, not a new fair
      // bottleneck that should silently zero-rate active activities.
      if (step == 0.0L) return false;

      for (std::size_t i = 0; i < count; ++i) {
        if (frozen[i]) continue;
        const Wide increment = step * weights[i];
        if (!std::isfinite(increment) || increment < 0.0L) return false;
        wide_rates[i] += increment;
        if (!std::isfinite(wide_rates[i]) || wide_rates[i] < 0.0L) return false;
      }

      std::vector<bool> saturated(rows.size(), false);
      for (std::size_t r = 0; r < rows.size(); ++r) {
        if (increments[r] == step) {
          saturated[r] = true;
          residual[r] = 0.0L;
        } else if (slopes[r] > 0.0L) {
          residual[r] -= slopes[r] * step;
          // Do not widen the feasibility tolerance: uncertain roundoff goes
          // to the caller's LP fallback.
          if (!std::isfinite(residual[r]) || residual[r] < 0.0L) return false;
        }
      }

      std::size_t newly_frozen = 0;
      // Row and activity order are the stable tie break for simultaneous
      // saturations. Every activity touching any saturated row is frozen.
      for (std::size_t r = 0; r < rows.size(); ++r) {
        if (!saturated[r]) continue;
        for (std::size_t i = 0; i < count; ++i) {
          if (!frozen[i] && rows[r].a[i] > 0.0) {
            frozen[i] = true;
            ++newly_frozen;
          }
        }
      }
      if (newly_frozen == 0 || newly_frozen > active) return false;
      active -= newly_frozen;
    }

    std::vector<double> result_rates(count, 0.0);
    std::vector<double> result_alpha(count, 0.0);
    for (std::size_t i = 0; i < count; ++i) {
      if (excluded[i]) continue;
      const double rate = static_cast<double>(wide_rates[i]);
      const double share = static_cast<double>(wide_alpha[i]);
      if (!std::isfinite(rate) || rate < 0.0 || (wide_rates[i] > 0.0L && rate == 0.0) ||
          !std::isfinite(share) || share <= 0.0 || share > 1.0) return false;
      result_rates[i] = static_cast<Wide>(rate)>wide_rates[i] ? std::nextafter(rate,0.0) : rate;
      result_alpha[i] = share;
    }

    // Round each positive scalar downward once, preserving equal-rate ties.
    // If the original packing rows still cannot be certified, use the LP.
    for (const PackingRow& row : rows) {
      Wide used=0;
      for(std::size_t i=0;i<count;++i)
        used+=static_cast<Wide>(row.a[i])*result_rates[i];
      if(!std::isfinite(used)||used>static_cast<Wide>(row.b))return false;
    }
    rates.swap(result_rates);alpha.swap(result_alpha);
    return true;
  } catch (...) {
    return false;
  }
}

}  // namespace coarse_die::detail
