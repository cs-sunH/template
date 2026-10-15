#include "../src/RatioAllocation.hh"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <random>
#include <string>
#include <vector>

namespace {

using coarse_die::detail::RatioRow;
using coarse_die::detail::allocate_ratio;

void require(bool condition, const std::string& message) {
  if (condition) return;
  std::cerr << "FAIL ratio_allocator_test: " << message << '\n';
  std::exit(1);
}

void close(double actual, double expected, const std::string& message,
           double tolerance = 2.0e-13) {
  const double scale = std::max(1.0, std::abs(expected));
  require(std::isfinite(actual) && std::abs(actual - expected) <= tolerance * scale,
          message + ": got " + std::to_string(actual) +
              ", expected " + std::to_string(expected));
}

void signed_pair_manual_solution_and_owner_alpha() {
  // Q=(1,3), W=4. The signed row gives u2>=2*u1, and the
  // physical row gives (u1+u2)/4<=1. Thus u=(4/3,8/3),
  // r=(1/3,2/3), alpha=(1/4,3/4).
  const std::vector<RatioRow> rows{
      {{1.0, 1.0}, 1.0},
      {{2.0, -1.0}, 0.0},
      {{-1.0, 0.0}, 0.0},  // redundant nonpositive zero-bound row
  };
  std::vector<double> rates, alpha;
  require(allocate_ratio(rows, {"owner", "owner"}, {1.0, 3.0}, rates, alpha),
          "manual one-edge signed system allocates");
  close(rates.at(0), 1.0 / 3.0, "manual first rate");
  close(rates.at(1), 2.0 / 3.0, "manual second rate");
  close(alpha.at(0), 1.0 / 4.0, "owner-Q alpha for first activity");
  close(alpha.at(1), 3.0 / 4.0, "owner-Q alpha for second activity");
  for (const RatioRow& row : rows) {
    const long double used = static_cast<long double>(row.a[0]) * rates[0] +
                             static_cast<long double>(row.a[1]) * rates[1];
    require(used <= static_cast<long double>(row.b),
            "published binary64 rates strictly certify every original row");
  }
}

void three_activity_closure_freezes_forced_tiers_and_keeps_private_rows() {
  // W_A=4 and W_B=2. The first ratio gives u1>=2*u0; the
  // second gives u2>=u1/2. The physical row is tight at u=(1,2,1).
  // u0 and u2 are forced at tier 1; u1 then rises to tier 2.
  const std::vector<RatioRow> rows{
      {{1.0, 1.0, 1.0}, 1.25},
      {{2.0, -1.0, 0.0}, 0.0},
      {{0.0, 1.0, -1.0}, 0.0},
  };
  std::vector<double> rates, alpha;
  require(allocate_ratio(rows, {"A", "A", "B"}, {1.0, 3.0, 2.0},
                         rates, alpha),
          "three-activity ratio closure allocates");
  close(rates.at(0), 0.25, "three-activity first rate");
  close(rates.at(1), 0.5, "three-activity second rate");
  close(rates.at(2), 0.5, "three-activity third rate");
  close(alpha.at(0), 0.25, "A activity one alpha");
  close(alpha.at(1), 0.75, "A activity two alpha");
  close(alpha.at(2), 1.0, "B activity alpha");

  // A's private cap is the first bottleneck. The ratio requires B to be at
  // least A; after A freezes at 1/2, B uses the remaining physical capacity.
  const std::vector<RatioRow> private_rows{
      {{1.0, 1.0}, 10.0},
      {{1.0, 0.0}, 0.5},
      {{0.0, 1.0}, 10.0},
      {{1.0, -1.0}, 0.0},
  };
  require(allocate_ratio(private_rows, {"A", "B"}, {1.0, 1.0}, rates, alpha),
          "PRIVATE utility cap allocates through ratio closure");
  close(rates.at(0), 0.5, "private first rate");
  close(rates.at(1), 9.5, "private second rate uses residual row capacity");
}

void exact_ratio_cycle_closes_without_zeroing() {
  // The two pair rows imply u1=2*u0. Their cycle gain is exactly one,
  // so closure is finite and the physical row fixes the scale.
  const std::vector<RatioRow> rows{
      {{1.0, 1.0}, 1.0},
      {{2.0, -1.0}, 0.0},
      {{-2.0, 1.0}, 0.0},
  };
  std::vector<double> rates, alpha;
  require(allocate_ratio(rows, {"owner", "owner"}, {1.0, 1.0}, rates, alpha),
          "product-one ratio cycle closes");
  close(rates.at(0), 1.0 / 3.0, "product-one cycle first rate");
  close(rates.at(1), 2.0 / 3.0, "product-one cycle second rate");

  const std::vector<RatioRow> contracting_cycle{
      {{1.0, 1.0}, 2.0},
      {{1.0, -2.0}, 0.0},
      {{-1.0, 1.0}, 0.0},
  };
  require(allocate_ratio(contracting_cycle, {"owner", "owner"}, {1.0, 1.0},
                         rates, alpha),
          "strictly contracting ratio cycle remains feasible");
  close(rates.at(0), 1.0, "contracting cycle first rate");
  close(rates.at(1), 1.0, "contracting cycle second rate");
}

// Small exact rational arithmetic used only by the test oracle. Random inputs
// are small integers, so signed 64-bit storage has a wide safety margin.
struct Fraction {
  std::int64_t n = 0;
  std::int64_t d = 1;

  Fraction() = default;
  Fraction(std::int64_t numerator, std::int64_t denominator = 1)
      : n(numerator), d(denominator) { normalize(); }

  void normalize() {
    require(d != 0, "oracle fraction has nonzero denominator");
    if (d < 0) { n = -n; d = -d; }
    const std::int64_t divisor = std::gcd(n < 0 ? -n : n, d);
    if (divisor > 1) { n /= divisor; d /= divisor; }
  }

  long double value() const {
    return static_cast<long double>(n) / static_cast<long double>(d);
  }
};

Fraction operator+(Fraction lhs, Fraction rhs) {
  return {lhs.n * rhs.d + rhs.n * lhs.d, lhs.d * rhs.d};
}
Fraction operator-(Fraction lhs, Fraction rhs) {
  return {lhs.n * rhs.d - rhs.n * lhs.d, lhs.d * rhs.d};
}
Fraction operator-(Fraction value) { return {-value.n, value.d}; }
Fraction operator*(Fraction lhs, Fraction rhs) {
  return {lhs.n * rhs.n, lhs.d * rhs.d};
}
Fraction operator/(Fraction lhs, Fraction rhs) {
  require(rhs.n != 0, "oracle does not divide by zero");
  return {lhs.n * rhs.d, lhs.d * rhs.n};
}
bool operator==(Fraction lhs, Fraction rhs) {
  return lhs.n == rhs.n && lhs.d == rhs.d;
}
bool operator<(Fraction lhs, Fraction rhs) {
  return static_cast<__int128>(lhs.n) * rhs.d <
         static_cast<__int128>(rhs.n) * lhs.d;
}

struct Inequality {
  std::vector<Fraction> a;
  Fraction b;
};

bool solve_square(std::vector<std::vector<Fraction>> matrix,
                  std::vector<Fraction> rhs,
                  std::vector<Fraction>& solution) {
  const std::size_t size = matrix.size();
  for (const auto& row : matrix) if (row.size() != size) return false;
  for (std::size_t column = 0; column < size; ++column) {
    std::size_t pivot = column;
    while (pivot < size && matrix[pivot][column].n == 0) ++pivot;
    if (pivot == size) return false;
    std::swap(matrix[pivot], matrix[column]);
    std::swap(rhs[pivot], rhs[column]);
    const Fraction divisor = matrix[column][column];
    for (std::size_t c = column; c < size; ++c)
      matrix[column][c] = matrix[column][c] / divisor;
    rhs[column] = rhs[column] / divisor;
    for (std::size_t row = 0; row < size; ++row) {
      if (row == column || matrix[row][column].n == 0) continue;
      const Fraction factor = matrix[row][column];
      for (std::size_t c = column; c < size; ++c)
        matrix[row][c] = matrix[row][c] - factor * matrix[column][c];
      rhs[row] = rhs[row] - factor * rhs[column];
    }
  }
  solution = std::move(rhs);
  return true;
}

bool feasible(const std::vector<Inequality>& rows,
              const std::vector<Fraction>& point) {
  for (const Inequality& row : rows) {
    Fraction used;
    for (std::size_t i = 0; i < point.size(); ++i)
      used = used + row.a[i] * point[i];
    if (row.b < used) return false;
  }
  return true;
}

bool maximize_at_vertices(const std::vector<Inequality>& rows,
                          const std::vector<Fraction>& objective,
                          Fraction& optimum) {
  const std::size_t dimension = objective.size();
  if (dimension == 0 || rows.size() < dimension) return false;
  bool found = false;
  std::vector<std::size_t> selected(dimension);
  std::function<void(std::size_t, std::size_t)> visit =
      [&](std::size_t depth, std::size_t begin) {
    if (depth == dimension) {
      std::vector<std::vector<Fraction>> matrix(
          dimension, std::vector<Fraction>(dimension));
      std::vector<Fraction> rhs(dimension);
      for (std::size_t r = 0; r < dimension; ++r) {
        const auto& row = rows[selected[r]];
        matrix[r] = row.a;
        rhs[r] = row.b;
      }
      std::vector<Fraction> point;
      if (!solve_square(std::move(matrix), std::move(rhs), point) ||
          !feasible(rows, point)) return;
      Fraction value;
      for (std::size_t i = 0; i < dimension; ++i)
        value = value + objective[i] * point[i];
      if (!found || optimum < value) {
        optimum = value;
        found = true;
      }
      return;
    }
    const std::size_t needed = dimension - depth;
    for (std::size_t i = begin; i + needed <= rows.size(); ++i) {
      selected[depth] = i;
      visit(depth + 1, i + 1);
    }
  };
  visit(0, 0);
  return found;
}

std::vector<Inequality> exact_original_rows(
    const std::vector<RatioRow>& rows,
    const std::vector<std::string>& owners,
    const std::vector<double>& quantities) {
  std::map<std::string, std::int64_t> total;
  for (std::size_t i = 0; i < owners.size(); ++i)
    total[owners[i]] += static_cast<std::int64_t>(quantities[i]);
  std::vector<Inequality> result;
  for (const auto& row : rows) {
    Inequality exact;
    exact.a.resize(owners.size());
    exact.b = Fraction(static_cast<std::int64_t>(row.b));
    require(static_cast<double>(exact.b.n) == row.b,
            "random oracle rows use integer coefficients and bounds");
    for (std::size_t i = 0; i < owners.size(); ++i) {
      const auto coefficient = static_cast<std::int64_t>(row.a[i]);
      require(static_cast<double>(coefficient) == row.a[i],
              "random oracle row coefficients are integral");
      exact.a[i] = Fraction(coefficient, total.at(owners[i]));
    }
    result.push_back(std::move(exact));
  }
  for (std::size_t i = 0; i < owners.size(); ++i) {
    Inequality nonnegative;
    nonnegative.a.assign(owners.size(), Fraction());
    nonnegative.a[i] = Fraction(-1);
    nonnegative.b = Fraction(0);
    result.push_back(std::move(nonnegative));
  }
  return result;
}

struct OracleSolution {
  std::vector<Fraction> utility;
};

OracleSolution exact_lexicographic_oracle(
    const std::vector<RatioRow>& input,
    const std::vector<std::string>& owners,
    const std::vector<double>& quantities) {
  const std::size_t count = owners.size();
  const auto base = exact_original_rows(input, owners, quantities);
  std::vector<bool> frozen(count, false);
  std::vector<Fraction> fixed(count);
  Fraction tier;
  std::size_t remaining = count;
  while (remaining != 0) {
    const std::size_t dimension = count + 1;
    std::vector<Inequality> tier_rows;
    for (const auto& row : base) {
      Inequality extended;
      extended.a = row.a;
      extended.a.push_back(Fraction(0));
      extended.b = row.b;
      tier_rows.push_back(std::move(extended));
    }
    for (std::size_t i = 0; i < count; ++i) {
      if (frozen[i]) {
        Inequality upper, lower;
        upper.a.assign(dimension, Fraction());
        lower.a.assign(dimension, Fraction());
        upper.a[i] = Fraction(1);
        lower.a[i] = Fraction(-1);
        upper.b = fixed[i];
        lower.b = -fixed[i];
        tier_rows.push_back(std::move(upper));
        tier_rows.push_back(std::move(lower));
      } else {
        Inequality floor;
        floor.a.assign(dimension, Fraction());
        floor.a[i] = Fraction(-1);
        floor.a[count] = Fraction(1);
        floor.b = Fraction(0);
        tier_rows.push_back(std::move(floor));
      }
    }
    Inequality prior_tier;
    prior_tier.a.assign(dimension, Fraction());
    prior_tier.a[count] = Fraction(-1);
    prior_tier.b = -tier;
    tier_rows.push_back(std::move(prior_tier));
    std::vector<Fraction> objective(dimension);
    objective[count] = Fraction(1);
    Fraction next_tier;
    require(maximize_at_vertices(tier_rows, objective, next_tier),
            "independent exact tier LP has a vertex optimum");
    tier = next_tier;
    require(tier.n > 0, "random generated cases have a positive fair tier");

    std::vector<std::size_t> forced;
    for (std::size_t raised = 0; raised < count; ++raised) {
      if (frozen[raised]) continue;
      std::vector<Inequality> face = base;
      for (std::size_t i = 0; i < count; ++i) {
        if (frozen[i]) {
          Inequality upper, lower;
          upper.a.assign(count, Fraction());
          lower.a.assign(count, Fraction());
          upper.a[i] = Fraction(1);
          lower.a[i] = Fraction(-1);
          upper.b = fixed[i];
          lower.b = -fixed[i];
          face.push_back(std::move(upper));
          face.push_back(std::move(lower));
        } else {
          Inequality floor;
          floor.a.assign(count, Fraction());
          floor.a[i] = Fraction(-1);
          floor.b = -tier;
          face.push_back(std::move(floor));
        }
      }
      std::vector<Fraction> individual_objective(count);
      individual_objective[raised] = Fraction(1);
      Fraction maximum;
      require(maximize_at_vertices(face, individual_objective, maximum),
              "independent exact forced-activity LP has a vertex optimum");
      if (maximum == tier) forced.push_back(raised);
    }
    require(!forced.empty(), "independent oracle identifies a forced activity");
    for (std::size_t i : forced) {
      frozen[i] = true;
      fixed[i] = tier;
      --remaining;
    }
  }
  return {fixed};
}

void randomized_exact_vertex_oracle() {
  std::mt19937 random(0x72617469u);
  auto draw = [&](int low, int high) {
    return std::uniform_int_distribution<int>(low, high)(random);
  };
  std::size_t successes = 0;
  for (std::size_t sample = 0; sample < 64; ++sample) {
    const std::size_t count = sample % 2 == 0 ? 2 : 3;
    std::vector<std::string> owners(count);
    std::vector<double> quantities(count, 1.0);
    for (std::size_t i = 0; i < count; ++i)
      owners[i] = "owner_" + std::to_string(i);

    std::vector<RatioRow> rows;
    // Independent finite physical/private caps plus one shared physical row.
    for (std::size_t i = 0; i < count; ++i) {
      std::vector<double> cap(count, 0.0);
      cap[i] = 1.0;
      rows.push_back({cap, static_cast<double>(draw(2, 8))});
    }
    std::vector<double> shared(count);
    for (double& coefficient : shared)
      coefficient = static_cast<double>(draw(1, 4));
    rows.push_back({shared, static_cast<double>(draw(5, 16))});
    // An acyclic pair-ratio chain avoids gain cycles while exercising both
    // two- and three-activity closure and the lexicographic freeze step.
    for (std::size_t i = 0; i + 1 < count; ++i) {
      std::vector<double> pair(count, 0.0);
      pair[i] = static_cast<double>(draw(1, 3));
      pair[i + 1] = -static_cast<double>(draw(1, 3));
      rows.push_back({pair, 0.0});
    }

    const auto oracle = exact_lexicographic_oracle(rows, owners, quantities);
    std::vector<double> rates, alpha;
    if (!allocate_ratio(rows, owners, quantities, rates, alpha)) continue;
    ++successes;
    require(rates.size() == count && alpha.size() == count,
            "successful randomized allocation has one rate and alpha per activity");
    for (std::size_t i = 0; i < count; ++i) {
      const long double expected = oracle.utility[i].value();
      const long double actual = rates[i];  // W_owner=1 in these cases.
      const long double error = std::abs(actual - expected);
      require(error <= 64.0L * std::numeric_limits<double>::epsilon() *
                          std::max(1.0L, std::abs(expected)),
              "ratio allocator matches independent Fraction vertex oracle");
      close(alpha[i], 1.0, "single-activity owner alpha in random case");
    }
  }
  require(successes >= 40,
          "bounded random acyclic cases exercise the fast path despite safe fallbacks");
}

void unsupported_rows_and_cycles_fall_back_without_mutation() {
  const std::vector<std::string> owners{"A", "B"};
  const std::vector<double> q{1.0, 1.0};
  const auto check_fallback = [&](const std::vector<RatioRow>& rows,
                                  const std::string& label) {
    std::vector<double> rates{17.0}, alpha{23.0};
    require(!allocate_ratio(rows, owners, q, rates, alpha), label);
    require(rates == std::vector<double>{17.0} &&
                alpha == std::vector<double>{23.0},
            "LP fallback leaves caller outputs untouched");
  };
  check_fallback({{{1.0, 0.0}, 0.0}, {{1.0, -1.0}, 0.0}},
                 "positive zero-bound row requests zero-exclusion LP fallback");
  check_fallback({{{1.0, 1.0}, 1.0}, {{1.0, -1.0}, 1.0}},
                 "nonzero signed bound requests LP fallback");
  check_fallback({{{1.0, 1.0}, 1.0}, {{1.0, -2.0}, 0.0},
                  {{-1.0, 3.0}, 0.0}},
                 "gain-amplifying ratio cycle requests LP fallback");
  const double next = std::nextafter(1.0, 2.0);
  check_fallback({{{1.0, 1.0}, 1.0}, {{1.0, -1.0}, 0.0},
                  {{-1.0, next}, 0.0}},
                 "near-unit gain cycle requests numerical-ambiguity fallback");
  check_fallback({{{1.0, 1.0}, 1.0}},
                 "nonnegative-only constraints remain on PackingAllocation");
  check_fallback({{{1.0, 0.0}, 0.0}, {{1.0, -1.0}, 0.0}},
                 "zero exclusion is not approximated by a signed fast path");

  std::vector<double> rates{17.0}, alpha{23.0};
  const std::vector<RatioRow> multiple_signs{
      {{1.0, 1.0, 1.0}, 1.0},
      {{1.0, 1.0, -1.0}, 0.0},
  };
  require(!allocate_ratio(multiple_signs, {"A", "B", "C"},
                          {1.0, 1.0, 1.0}, rates, alpha),
          "multiple-positive signed row requests LP fallback");
  require(rates == std::vector<double>{17.0} &&
              alpha == std::vector<double>{23.0},
          "three-variable unsupported fallback leaves outputs untouched");

  const std::vector<RatioRow> four_variables{
      {{1.0, 0.0, 0.0, 0.0}, 1.0},
      {{0.0, 1.0, 0.0, 0.0}, 1.0},
      {{0.0, 0.0, 1.0, 0.0}, 1.0},
      {{0.0, 0.0, 0.0, 1.0}, 1.0},
      {{1.0, -1.0, 0.0, 0.0}, 0.0},
  };
  rates = {17.0};
  alpha = {23.0};
  require(!allocate_ratio(four_variables, {"A", "B", "C", "D"},
                          {1.0, 1.0, 1.0, 1.0}, rates, alpha),
          "more than three active variables use the LP");
  require(rates == std::vector<double>{17.0} &&
              alpha == std::vector<double>{23.0},
          "over-limit fallback leaves outputs untouched");
}

}  // namespace

int main() {
  signed_pair_manual_solution_and_owner_alpha();
  three_activity_closure_freezes_forced_tiers_and_keeps_private_rows();
  exact_ratio_cycle_closes_without_zeroing();
  randomized_exact_vertex_oracle();
  unsupported_rows_and_cycles_fall_back_without_mutation();
  std::cout << "PASS ratio_allocator_test\n";
  return 0;
}
