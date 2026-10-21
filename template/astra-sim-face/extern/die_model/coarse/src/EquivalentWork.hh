#pragma once

#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace coarse_die::detail::equivalent_work {

using u128 = unsigned __int128;

static_assert(std::numeric_limits<double>::radix == 2,
              "equivalent work requires binary floating-point coefficients");
static_assert(std::numeric_limits<long double>::radix == 2,
              "equivalent work requires binary long double");

// A checked, nonnegative work quantity. The integer part remains exact across
// the full u128 range. The fractional part is a separately stored long double
// in [0, 1), so callers must not combine it with `whole()` prematurely.
//
// `weighted` treats its binary64 coefficient as the exact value represented by
// that double. It rejects products whose integer part exceeds u128 or whose
// fractional part cannot be represented exactly by this target's long double.
// Exact dyadic decomposition is implemented for binary long double precision
// up to 64 bits (including binary64 and x87 extended precision); wider formats
// fail at compile time rather than silently losing fractional bits.
// Plan v3 encodes this fraction as an exact odd significand and exponent.
// Consumers that require a binary64 scalar can use `canonical_fraction_double()`
// to reject values not exactly representable at that narrower boundary.
class Work {
 public:
  constexpr Work() noexcept = default;
  constexpr Work(u128 whole) noexcept : whole_(whole) {}

  static Work from_parts(u128 whole, long double fraction) {
    if (!std::isfinite(fraction) || fraction < 0.0 || fraction >= 1.0)
      throw std::invalid_argument("equivalent work fraction must be finite and in [0, 1)");
    if (fraction == 0.0) return Work(whole);
    return Work(whole, fraction, PartsTag{});
  }

  static Work weighted(u128 elements, double coefficient) {
    if (!std::isfinite(coefficient) || coefficient < 0.0)
      throw std::invalid_argument("equivalent work coefficient must be finite and nonnegative");
    if (elements == 0 || coefficient == 0.0) return Work{};

    // A double is an exact integer significand times a power of two. Multiply
    // that integer significand by `elements` before shifting, so large integer
    // κ values never pass through a lossy floating-point product.
    static_assert(std::numeric_limits<double>::digits <= 64,
                  "equivalent work expects a binary64-or-narrower double");
    const int coefficient_digits = std::numeric_limits<double>::digits;
    int exponent = 0;
    const double mantissa = std::frexp(coefficient, &exponent);
    const auto significand = static_cast<std::uint64_t>(
        std::ldexp(mantissa, coefficient_digits));
    const Wide192 product = multiply_u128_u64(elements, significand);
    const int denominator_shift = coefficient_digits - exponent;

    if (denominator_shift <= 0) {
      const unsigned left_shift = static_cast<unsigned>(-denominator_shift);
      const unsigned bits = bit_length(product);
      if (bits + left_shift > 128)
        throw std::overflow_error("overflow: equivalent work weighted product exceeds u128");
      return Work(shift_left_to_u128(product, left_shift));
    }

    const unsigned shift = static_cast<unsigned>(denominator_shift);
    const u128 whole = shift_right_to_u128(product, shift,
                                          "equivalent work weighted product exceeds u128");
    const Wide192 residue = low_bits(product, shift);
    const long double fraction = exact_fraction(residue, shift,
                                                "equivalent work fractional product is not exactly representable as long double");
    return Work(whole, fraction, PartsTag{});
  }

  u128 whole() const noexcept { return whole_; }
  long double fraction() const noexcept { return fraction_; }
  bool integral() const noexcept { return fraction_ == 0.0L; }

  // The aggregate conversion is approximate for large integer values. If the
  // fractional contribution changes when combined with the converted whole,
  // fail instead of silently rounding it.
  long double as_long_double() const {
    const long double whole_value = static_cast<long double>(whole_);
    const long double result = whole_value + fraction_;
    if (fraction_ != 0.0L && result - whole_value != fraction_)
      throw std::overflow_error("overflow: equivalent work fraction is lost in long double aggregate conversion");
    if (!std::isfinite(result))
      throw std::overflow_error("overflow: equivalent work aggregate exceeds long double range");
    return result;
  }

  double canonical_fraction_double() const {
    const double result = static_cast<double>(fraction_);
    if (!std::isfinite(result) || static_cast<long double>(result) != fraction_)
      throw std::overflow_error("overflow: equivalent work fraction is not exactly representable as binary64");
    return result;
  }

  std::uint64_t checked_u64_whole() const {
    if (whole_ > static_cast<u128>(std::numeric_limits<std::uint64_t>::max()))
      throw std::overflow_error("overflow: equivalent work whole part exceeds uint64");
    return static_cast<std::uint64_t>(whole_);
  }

  Work& operator+=(const Work& other) {
    const FractionSum fraction_sum = add_fractions(fraction_, other.fraction_);
    u128 result_whole = checked_add(whole_, other.whole_, "equivalent work sum exceeds u128");
    if (fraction_sum.carry)
      result_whole = checked_add(result_whole, 1, "equivalent work sum exceeds u128");
    whole_ = result_whole;
    fraction_ = fraction_sum.fraction;
    return *this;
  }

  Work multiplied(std::uint64_t count) const {
    if (count == 0) return Work{};
    const u128 whole_product = checked_multiply(
        whole_, static_cast<u128>(count), "equivalent work product exceeds u128");
    const FractionProduct fraction_product = multiply_fraction(fraction_, count);
    return Work(checked_add(whole_product, fraction_product.carry,
                            "equivalent work product exceeds u128"),
                fraction_product.fraction, PartsTag{});
  }

  Work multiply(std::uint64_t count) const { return multiplied(count); }

  std::pair<Work, Work> divide(std::uint64_t divisor) const {
    if (divisor == 0)
      throw std::invalid_argument("equivalent work division by zero");
    const u128 whole_quotient = whole_ / divisor;
    const u128 whole_remainder = whole_ % divisor;
    if (fraction_ == 0.0L)
      return {Work(whole_quotient), Work(whole_remainder)};

    // Preserve the old integer quotient/remainder rule, while distributing the
    // fractional share across the repeated base. Use the total's scalar
    // precision so any later repeated subset or base+remainder combination
    // can retain that share. Any undistributed residue stays in the exact,
    // nonnegative remainder; no equivalent work is discarded.
    const long double divisor_value = static_cast<long double>(divisor);
    long double base_fraction = fraction_ / divisor_value;
    FractionProduct allocated{};
    bool found = false;
    for (int retained_bits = std::numeric_limits<long double>::digits;
         retained_bits >= 0; --retained_bits) {
      const long double candidate = truncate_fraction(base_fraction, retained_bits);
      if (candidate == 0.0L) {
        allocated = {};
        base_fraction = 0.0L;
        found = true;
        break;
      }
      try {
        const FractionProduct product = multiply_fraction(candidate, divisor);
        const auto combines_exactly = [](u128 whole, long double fraction) {
          const long double value = static_cast<long double>(whole);
          return (value + fraction) - value == fraction;
        };
        if (product.carry == 0 && product.fraction <= fraction_ &&
            combines_exactly(whole_, candidate) &&
            combines_exactly(whole_remainder, fraction_ - product.fraction)) {
          base_fraction = candidate;
          allocated = product;
          found = true;
          break;
        }
      } catch (const std::overflow_error&) {
        // Drop a low significand bit and retry. The loop is bounded by the
        // target's fixed long-double precision (at most 64 bits here).
      }
    }
    if (!found)
      throw std::overflow_error("overflow: equivalent work fractional division cannot be represented");

    const long double remainder_fraction = fraction_ - allocated.fraction;
    if (remainder_fraction < 0.0L)
      throw std::overflow_error("overflow: equivalent work division produced a negative remainder");
    return {Work(whole_quotient, base_fraction, PartsTag{}),
            Work(whole_remainder, remainder_fraction, PartsTag{})};
  }

  friend Work operator+(Work left, const Work& right) {
    left += right;
    return left;
  }

  friend Work operator*(const Work& work, std::uint64_t count) {
    return work.multiplied(count);
  }

  friend Work operator*(std::uint64_t count, const Work& work) {
    return work.multiplied(count);
  }

 private:
  struct PartsTag {};
  struct FractionSum {
    long double fraction = 0.0L;
    bool carry = false;
  };
  struct FractionProduct {
    u128 carry = 0;
    long double fraction = 0.0L;
  };
  struct Wide192 {
    std::uint64_t limb[3] = {0, 0, 0};
  };

  constexpr Work(u128 whole, long double fraction, PartsTag) noexcept
      : whole_(whole), fraction_(fraction) {}

  static constexpr u128 max_u128() noexcept { return ~u128{0}; }

  static u128 checked_add(u128 left, u128 right, const char* message) {
    if (right > max_u128() - left) throw std::overflow_error(std::string("overflow: ") + message);
    return left + right;
  }

  static u128 checked_multiply(u128 left, u128 right, const char* message) {
    if (left != 0 && right > max_u128() / left)
      throw std::overflow_error(std::string("overflow: ") + message);
    return left * right;
  }

  static FractionSum add_fractions(long double left, long double right) {
    const long double sum = left + right;
    // Knuth TwoSum: detect whether the exact sum of the two stored binary
    // fractions survived this addition without losing a low-order term.
    const long double virtual_right = sum - left;
    const long double error = (left - (sum - virtual_right)) +
                              (right - virtual_right);
    if (error != 0.0L)
      throw std::overflow_error("overflow: equivalent work fractional sum is not exactly representable");
    if (sum >= 1.0L) return {sum - 1.0L, true};
    return {sum, false};
  }

  static Wide192 multiply_u128_u64(u128 value, std::uint64_t factor) noexcept {
    const std::uint64_t low = static_cast<std::uint64_t>(value);
    const std::uint64_t high = static_cast<std::uint64_t>(value >> 64);
    const u128 low_product = static_cast<u128>(low) * factor;
    const u128 high_product = static_cast<u128>(high) * factor;
    const u128 middle_sum = static_cast<u128>(static_cast<std::uint64_t>(low_product >> 64)) +
                            static_cast<std::uint64_t>(high_product);
    return {{static_cast<std::uint64_t>(low_product),
             static_cast<std::uint64_t>(middle_sum),
             static_cast<std::uint64_t>(high_product >> 64) +
                 static_cast<std::uint64_t>(middle_sum >> 64)}};
  }

  static unsigned bit_length(std::uint64_t value) noexcept {
    return value == 0 ? 0U : 64U - static_cast<unsigned>(__builtin_clzll(value));
  }

  static unsigned bit_length(u128 value) noexcept {
    const std::uint64_t high = static_cast<std::uint64_t>(value >> 64);
    return high != 0 ? 64U + bit_length(high)
                     : bit_length(static_cast<std::uint64_t>(value));
  }

  static unsigned bit_length(const Wide192& value) noexcept {
    if (value.limb[2] != 0) return 128U + bit_length(value.limb[2]);
    if (value.limb[1] != 0) return 64U + bit_length(value.limb[1]);
    return bit_length(value.limb[0]);
  }

  static Wide192 low_bits(const Wide192& value, unsigned count) noexcept {
    if (count >= 192U) return value;
    Wide192 result{};
    const unsigned full_limbs = count / 64U;
    const unsigned partial_bits = count % 64U;
    for (unsigned i = 0; i < full_limbs; ++i) result.limb[i] = value.limb[i];
    if (partial_bits != 0U)
      result.limb[full_limbs] = value.limb[full_limbs] &
                                ((std::uint64_t{1} << partial_bits) - 1U);
    return result;
  }

  static unsigned trailing_zeros(const Wide192& value) {
    for (unsigned i = 0; i < 3U; ++i) {
      if (value.limb[i] != 0)
        return i * 64U + static_cast<unsigned>(__builtin_ctzll(value.limb[i]));
    }
    return 0U;
  }

  static long double exact_fraction(const Wide192& numerator, unsigned denominator_shift,
                                   const char* message) {
    if (bit_length(numerator) == 0U) return 0.0L;
    const unsigned zeros = trailing_zeros(numerator);
    const unsigned significant_bits = bit_length(numerator) - zeros;
    if (significant_bits > static_cast<unsigned>(std::numeric_limits<long double>::digits))
      throw std::overflow_error(std::string("overflow: ") + message);
    const int least_exponent = static_cast<int>(zeros) -
                               static_cast<int>(denominator_shift);
    const int minimum_subnormal_exponent =
        std::numeric_limits<long double>::has_denorm == std::denorm_absent
            ? std::numeric_limits<long double>::min_exponent - 1
            : std::numeric_limits<long double>::min_exponent -
                  std::numeric_limits<long double>::digits;
    if (least_exponent < minimum_subnormal_exponent)
      throw std::overflow_error(std::string("overflow: ") + message);

    long double integer = 0.0L;
    for (int limb = 2; limb >= 0; --limb) {
      integer = std::ldexp(integer, 64) +
                static_cast<long double>(numerator.limb[limb]);
    }
    const long double result = std::ldexp(integer, -static_cast<int>(denominator_shift));
    if (!std::isfinite(result) || result == 0.0L)
      throw std::overflow_error(std::string("overflow: ") + message);
    return result;
  }

  static u128 shift_right_to_u128(const Wide192& value, unsigned shift,
                                  const char* overflow_message) {
    const unsigned bits = bit_length(value);
    if (bits > shift + 128U) throw std::overflow_error(std::string("overflow: ") + overflow_message);
    if (shift >= 192U) return 0;
    const unsigned word = shift / 64U;
    const unsigned offset = shift % 64U;
    std::uint64_t low = 0;
    std::uint64_t high = 0;
    if (word == 0U) {
      low = value.limb[0] >> offset;
      if (offset != 0U) low |= value.limb[1] << (64U - offset);
      high = offset == 0U ? value.limb[1]
                          : (value.limb[1] >> offset) |
                                (value.limb[2] << (64U - offset));
    } else if (word == 1U) {
      low = value.limb[1] >> offset;
      if (offset != 0U) low |= value.limb[2] << (64U - offset);
      high = offset == 0U ? value.limb[2] : value.limb[2] >> offset;
    } else {
      low = value.limb[2] >> offset;
    }
    return (static_cast<u128>(high) << 64) | low;
  }

  static u128 shift_left_to_u128(const Wide192& value, unsigned shift) {
    if (shift >= 128U) return 0;
    const u128 low_value = static_cast<u128>(value.limb[0]) |
                           (static_cast<u128>(value.limb[1]) << 64);
    return low_value << shift;
  }

  static unsigned trailing_zeros(u128 value) {
    const std::uint64_t low = static_cast<std::uint64_t>(value);
    if (low != 0) return static_cast<unsigned>(__builtin_ctzll(low));
    return 64U + static_cast<unsigned>(__builtin_ctzll(static_cast<std::uint64_t>(value >> 64)));
  }

  static u128 multiply_u64(std::uint64_t left, std::uint64_t right) noexcept {
    return static_cast<u128>(left) * right;
  }

  static FractionProduct multiply_fraction(long double fraction, std::uint64_t factor) {
    if (fraction == 0.0L || factor == 0) return {};
    static_assert(std::numeric_limits<long double>::digits <= 64,
                  "equivalent work expects long double precision no wider than 64 bits");
    constexpr int digits = std::numeric_limits<long double>::digits;
    int exponent = 0;
    const long double mantissa = std::frexp(fraction, &exponent);
    const auto significand = static_cast<std::uint64_t>(std::ldexp(mantissa, digits));
    const u128 product = multiply_u64(significand, factor);
    const unsigned denominator_shift = static_cast<unsigned>(digits - exponent);

    u128 whole = 0;
    u128 remainder = product;
    if (denominator_shift < 128U) {
      whole = product >> denominator_shift;
      remainder = product & ((u128{1} << denominator_shift) - 1U);
    }
    const unsigned remainder_bits = bit_length(remainder);
    if (remainder_bits != 0U) {
      const unsigned zeros = trailing_zeros(remainder);
      const unsigned significant_bits = remainder_bits - zeros;
      const int least_exponent = static_cast<int>(zeros) -
                                 static_cast<int>(denominator_shift);
      const int minimum_subnormal_exponent =
          std::numeric_limits<long double>::has_denorm == std::denorm_absent
              ? std::numeric_limits<long double>::min_exponent - 1
              : std::numeric_limits<long double>::min_exponent -
                    std::numeric_limits<long double>::digits;
      if (significant_bits > static_cast<unsigned>(digits) ||
          least_exponent < minimum_subnormal_exponent)
        throw std::overflow_error("overflow: equivalent work fractional product is not exactly representable");
    }
    const long double numerator = static_cast<long double>(remainder);
    const long double result = std::ldexp(numerator, -static_cast<int>(denominator_shift));
    if (remainder != 0 && result == 0.0L)
      throw std::overflow_error("overflow: equivalent work fractional product underflows long double");
    return {whole, result};
  }

  static long double truncate_fraction(long double value, int retained_bits) {
    if (value == 0.0L || retained_bits >= std::numeric_limits<long double>::digits)
      return value;
    if (retained_bits <= 0) return 0.0L;
    constexpr int digits = std::numeric_limits<long double>::digits;
    int exponent = 0;
    const long double mantissa = std::frexp(value, &exponent);
    const auto significand = static_cast<std::uint64_t>(std::ldexp(mantissa, digits));
    const unsigned dropped = static_cast<unsigned>(digits - retained_bits);
    const std::uint64_t mask = dropped == 64U
                                   ? std::numeric_limits<std::uint64_t>::max()
                                   : ((std::uint64_t{1} << dropped) - 1U);
    const std::uint64_t truncated = significand & ~mask;
    return std::ldexp(static_cast<long double>(truncated), exponent - digits);
  }

  u128 whole_ = 0;
  long double fraction_ = 0.0L;
};

}  // namespace coarse_die::detail::equivalent_work

