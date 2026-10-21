#include "../src/EquivalentWork.hh"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

namespace {
using coarse_die::detail::equivalent_work::Work;
using u128 = coarse_die::detail::equivalent_work::u128;

int failures = 0;

void check(bool condition, const std::string& message) {
  if (!condition) {
    ++failures;
    std::cerr << "FAIL: " << message << '\n';
  }
}

template <class Exception, class Callable>
void check_throws(Callable&& callable, const std::string& message) {
  try {
    callable();
    check(false, message + " (did not throw)");
  } catch (const Exception&) {
  } catch (const std::exception& error) {
    check(false, message + " (wrong exception: " + error.what() + ")");
  }
}

long double close_value(const Work& work) {
  return work.as_long_double();
}

void check_close(long double actual, long double expected, long double tolerance,
                 const std::string& message) {
  check(std::fabs(actual - expected) <= tolerance, message);
}

void weighted_cases() {
  check(Work::weighted(99, 0.0).integral(), "κ=0 is zero work");
  check(Work::weighted(99, 0.0).whole() == 0, "κ=0 retains zero whole part");

  const Work half = Work::weighted(3, 0.5);
  check(half.whole() == 1 && half.fraction() == 0.5L,
        "κ=0.5 preserves the half-element result");

  const Work one_and_quarter = Work::weighted(4, 1.25);
  check(one_and_quarter.whole() == 5 && one_and_quarter.integral(),
        "κ=1.25 yields an exact integer product when applicable");

  const Work two_and_three_quarters = Work::weighted(2, 2.75);
  check(two_and_three_quarters.whole() == 5 &&
            two_and_three_quarters.fraction() == 0.5L,
        "κ=2.75 preserves the fractional contribution");

  const Work huge_zero = Work::weighted(0, std::numeric_limits<double>::max());
  check(huge_zero.integral() && huge_zero.whole() == 0,
        "zero elements short-circuit a huge finite κ");
  check_throws<std::overflow_error>(
      [] { (void)Work::weighted(1, std::numeric_limits<double>::max()); },
      "huge finite κ is checked before converting the whole part");

  const u128 large_integer = (u128{1} << 100) + 123;
  const Work integer_product = Work::weighted(large_integer, 3.0);
  check(integer_product.whole() == large_integer * 3 && integer_product.integral(),
        "integer κ multiplication remains exact above 2^53");

  const double tiny = std::numeric_limits<double>::denorm_min();
  const Work tiny_exact = Work::weighted(1, tiny);
  check(tiny_exact.whole() == 0 && tiny_exact.canonical_fraction_double() == tiny,
        "binary64 subnormal κ is retained when representable");
  check_throws<std::overflow_error>(
      [] { (void)Work::weighted(~u128{0}, std::numeric_limits<double>::denorm_min()); },
      "unrepresentable low-order fractional product is rejected explicitly");

  check_throws<std::invalid_argument>(
      [] { (void)Work::weighted(1, std::numeric_limits<double>::quiet_NaN()); },
      "NaN κ is rejected");
  check_throws<std::invalid_argument>(
      [] { (void)Work::weighted(1, std::numeric_limits<double>::infinity()); },
      "infinite κ is rejected");
  check_throws<std::invalid_argument>(
      [] { (void)Work::weighted(1, -0.25); }, "negative κ is rejected");

  check_throws<std::overflow_error>(
      [] { (void)Work::weighted(~u128{0}, 2.0); },
      "128-bit weighted product overflow is rejected");
}

void integer_domain_cases() {
  const std::uint64_t max_u64 = std::numeric_limits<std::uint64_t>::max();
  const Work above_binary64_integer((u128{1} << 53) + 1);
  check(above_binary64_integer.whole() == (u128{1} << 53) + 1,
        "integer work above 2^53 is retained exactly");
  check(Work(max_u64).checked_u64_whole() == max_u64,
        "uint64 maximum fits the checked whole accessor");
  check_throws<std::overflow_error>(
      [] { (void)Work(u128{1} << 64).checked_u64_whole(); },
      "checked uint64 accessor rejects values above uint64");

  const u128 max_u128 = ~u128{0};
  check_throws<std::overflow_error>([&] { (void)(Work(max_u128) + Work(1)); },
                                    "128-bit sum overflow is rejected");
  check_throws<std::overflow_error>([&] { (void)Work(max_u128).multiplied(2); },
                                    "128-bit repeated work overflow is rejected");

  const auto [base, remainder] = Work(17).divide(5);
  check(base.whole() == 3 && base.integral(),
        "integer divide returns exact floor quotient");
  check(remainder.whole() == 2 && remainder.integral(),
        "integer divide returns exact remainder");
  check((base.multiplied(5) + remainder).whole() == 17,
        "integer quotient and remainder reconstruct the original exactly");
  check_throws<std::invalid_argument>([] { (void)Work(1).divide(0); },
                                      "zero divisor is rejected");
}

void fractional_operations() {
  const Work sum = Work::weighted(3, 0.5) + Work::from_parts(0, 0.25);
  check(sum.whole() == 1 && sum.fraction() == 0.75L,
        "fractional addition normalizes without changing work");

  const Work product = sum.multiplied(3);
  check(product.whole() == 5 && product.fraction() == 0.25L,
        "fractional multiplication carries the integer portion exactly");

  const auto [base, remainder] = product.divide(3);
  const Work reconstructed = base.multiplied(3) + remainder;
  check_close(close_value(reconstructed), close_value(product), 1e-18L,
              "fractional divide conserves work within numeric tolerance");
  check(remainder.fraction() >= 0.0L,
        "fractional divide never returns a negative remainder");
  check(base.fraction() > 0.0L,
        "fractional divide distributes a share across the base");

  // Runtime combines repeated subsets and the final remainder into aggregate
  // activities. Every such quantity must retain its stored fraction exactly.
  for (const std::uint64_t whole : {17u,32769u,1000001u}) {
    const Work total = Work::from_parts(whole,0.75L);
    for (const std::uint64_t count : {3u,7u,19u}) {
      const auto split = total.divide(count);
      const Work restored = split.first.multiplied(count) + split.second;
      check(restored.whole() == total.whole() &&
                restored.fraction() == total.fraction(),
            "repeated split conserves exact whole and dyadic fraction");
      for (std::uint64_t copies = 0; copies <= count; ++copies) {
        const Work subset = split.first.multiplied(copies);
        (void)subset.as_long_double();
        (void)(subset + split.second).as_long_double();
      }
    }
  }

  const Work large_with_fraction = Work::from_parts(u128{1} << 127, 0.5);
  check(large_with_fraction.whole() == (u128{1} << 127) &&
            large_with_fraction.fraction() == 0.5L,
        "parts retain a small fraction separately from a large exact whole");
  check_throws<std::overflow_error>([&] { (void)large_with_fraction.as_long_double(); },
                                    "aggregate conversion rejects a lost small fraction");

  check_throws<std::invalid_argument>(
      [] { (void)Work::from_parts(0, std::numeric_limits<double>::quiet_NaN()); },
      "from_parts rejects NaN");
  check_throws<std::invalid_argument>(
      [] { (void)Work::from_parts(0, -0.25); }, "from_parts rejects negative fractions");
  check_throws<std::invalid_argument>(
      [] { (void)Work::from_parts(0, 1.0); }, "from_parts rejects fractions at one");

  const Work noncanonical = Work::from_parts(0, 0.1) + Work::from_parts(0, 0.2);
  check_throws<std::overflow_error>([&] { (void)noncanonical.canonical_fraction_double(); },
                                    "binary64 serialization rejects a noncanonical fraction");
}

}  // namespace

int main() {
  weighted_cases();
  integer_domain_cases();
  fractional_operations();
  if (failures != 0) {
    std::cerr << failures << " equivalent-work checks failed\n";
    return 1;
  }
  std::cout << "equivalent-work checks passed\n";
  return 0;
}

