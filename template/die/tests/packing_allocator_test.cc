#include "../src/PackingAllocation.hh"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace {

using coarse_die::detail::PackingRow;
using coarse_die::detail::allocate_packing;

void require(bool condition, const char* message) {
  if (condition) return;
  std::cerr << "FAIL packing_allocator_test: " << message << '\n';
  std::exit(1);
}

void close(double actual, double expected, const char* message) {
  const double scale = std::max(1.0, std::abs(expected));
  require(std::isfinite(actual) && std::abs(actual - expected) <= 1.0e-12 * scale,
          message);
}

void multi_resource_heterogeneous_owner() {
  // A's two activities have Q=(1,3), while B has Q=1. The two resource
  // limits first bind row 1 at utility 4/3, freezing A2 and B; A1 then uses
  // the remaining row-0 capacity. This gives rates (2/3,1/3,4/3).
  const std::vector<PackingRow> rows{
      {{1.0, 0.0, 1.0}, 2.0},
      {{0.0, 2.0, 1.0}, 2.0},
  };
  std::vector<double> rates, alpha;
  require(allocate_packing(rows, {"A", "A", "B"}, {1.0, 3.0, 1.0},
                           rates, alpha),
          "heterogeneous two-resource packing allocates");
  close(rates.at(0), 2.0 / 3.0, "A1 uses its remaining resource capacity");
  close(rates.at(1), 1.0 / 3.0, "A2 freezes at the first bottleneck");
  close(rates.at(2), 4.0 / 3.0, "B freezes at the first bottleneck");
  close(alpha.at(0), 1.0 / 4.0, "A1 canonical owner share is preserved");
  close(alpha.at(1), 3.0 / 4.0, "A2 canonical owner share is preserved");
  close(alpha.at(2), 1.0, "B is the sole activity for its owner");
  close(rates[0] + rates[2], 2.0, "row 0 is tight at completion");
  close(2.0 * rates[1] + rates[2], 2.0, "row 1 remains tight at completion");
}

void private_bounds_and_independent_tiers() {
  // The private row limits A to 1/2. B then fills the residual shared
  // physical capacity, reaching 19/2. Private bounds are independent rows.
  const std::vector<PackingRow> rows{
      {{1.0, 1.0}, 10.0},
      {{1.0, 0.0}, 0.5},
      {{0.0, 1.0}, 10.0},
  };
  std::vector<double> rates, alpha;
  require(allocate_packing(rows, {"A", "B"}, {1.0, 1.0}, rates, alpha),
          "private bounds allocate");
  close(rates.at(0), 0.5, "A stops at its private limit");
  close(rates.at(1), 9.5, "B continues after A freezes");
  close(alpha.at(0), 1.0, "A owner share remains normalized");
  close(alpha.at(1), 1.0, "B owner share remains normalized");
}

void zero_capacity_exclusion_renormalizes_owner() {
  const std::vector<PackingRow> rows{
      {{1.0, 0.0, 0.0}, 0.0},
      {{0.0, 1.0, 1.0}, 2.0},
  };
  std::vector<double> rates, alpha;
  require(allocate_packing(rows, {"A", "A", "B"}, {9.0, 1.0, 1.0},
                           rates, alpha),
          "zero-capacity blocker is excluded before owner normalization");
  close(rates.at(0), 0.0, "blocked activity receives zero rate");
  close(alpha.at(0), 0.0, "blocked activity does not dilute owner alpha");
  close(rates.at(1), 1.0, "remaining A activity shares the resource equally");
  close(rates.at(2), 1.0, "B shares the resource equally");
  close(alpha.at(1), 1.0, "remaining A activity is renormalized to one");
  close(alpha.at(2), 1.0, "B alpha remains one");
}

void physical_multiplicity_is_in_the_row() {
  const std::vector<PackingRow> rows{{{3.0, 1.0}, 8.0}};
  std::vector<double> rates, alpha;
  require(allocate_packing(rows, {"A", "B"}, {1.0, 1.0}, rates, alpha),
          "multiplicity-weighted physical row allocates");
  close(rates.at(0), 2.0, "multiplicity limits A to rate two");
  close(rates.at(1), 2.0, "B receives equal owner utility");
  close(3.0 * rates[0] + rates[1], 8.0,
        "multiplicity is accounted in the physical capacity");
}

void signed_and_invalid_rows_fall_back_without_mutation() {
  std::vector<double> rates{17.0}, alpha{23.0};
  require(!allocate_packing({{{-1.0}, 1.0}}, {"A"}, {1.0}, rates, alpha),
          "coupled signed row requests LP fallback");
  require(rates == std::vector<double>{17.0} && alpha == std::vector<double>{23.0},
          "signed fallback leaves output vectors untouched");
  require(!allocate_packing({{{1.0}, std::numeric_limits<double>::infinity()}},
                            {"A"}, {1.0}, rates, alpha),
          "non-finite bound requests LP fallback");
  require(!allocate_packing({{{1.0, 2.0}, 1.0}}, {"A"}, {1.0}, rates, alpha),
          "dimension mismatch requests LP fallback");
  require(rates == std::vector<double>{17.0} && alpha == std::vector<double>{23.0},
          "invalid fallback leaves output vectors untouched");
}

void unbounded_rows_fall_back() {
  std::vector<double> rates, alpha;
  require(!allocate_packing({}, {"A"}, {1.0}, rates, alpha),
          "unbounded positive-rate activity requests LP fallback");
}

}  // namespace

int main() {
  multi_resource_heterogeneous_owner();
  private_bounds_and_independent_tiers();
  zero_capacity_exclusion_renormalizes_owner();
  physical_multiplicity_is_in_the_row();
  signed_and_invalid_rows_fall_back_without_mutation();
  unbounded_rows_fall_back();
  std::cout << "PASS packing_allocator_test\n";
  return 0;
}
