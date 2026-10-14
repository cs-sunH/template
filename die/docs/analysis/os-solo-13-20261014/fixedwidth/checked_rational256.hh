#pragma once

// Internal feasibility prototype only.  A canonical rational uses signed
// 256-bit magnitude/denominator slots (8x32-bit limbs each); every operation
// uses fixed 768-bit scratch (24x32-bit limbs).  This header has no dependency
// beyond the C++17 standard library and does not touch the public Runtime ABI.

#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <string_view>

namespace os_fixedwidth {

template <std::size_t Limbs>
class UInt {
 public:
  std::array<std::uint32_t, Limbs> limb{};

  bool zero() const {
    for (std::uint32_t value : limb) if (value != 0) return false;
    return true;
  }

  std::size_t bit_width() const {
    for (std::size_t i = Limbs; i-- > 0;) {
      if (limb[i] == 0) continue;
      std::uint32_t v = limb[i];
      unsigned bits = 0;
      while (v != 0) { ++bits; v >>= 1; }
      return i * 32 + bits;
    }
    return 0;
  }

  static int compare(const UInt& a, const UInt& b) {
    for (std::size_t i = Limbs; i-- > 0;) {
      if (a.limb[i] < b.limb[i]) return -1;
      if (a.limb[i] > b.limb[i]) return 1;
    }
    return 0;
  }

  bool add_small(std::uint32_t value) {
    std::uint64_t carry = value;
    for (std::size_t i = 0; i < Limbs && carry != 0; ++i) {
      const std::uint64_t sum = static_cast<std::uint64_t>(limb[i]) + carry;
      limb[i] = static_cast<std::uint32_t>(sum);
      carry = sum >> 32;
    }
    return carry == 0;
  }

  bool multiply_small(std::uint32_t value) {
    std::uint64_t carry = 0;
    for (std::size_t i = 0; i < Limbs; ++i) {
      const std::uint64_t product =
          static_cast<std::uint64_t>(limb[i]) * value + carry;
      limb[i] = static_cast<std::uint32_t>(product);
      carry = product >> 32;
    }
    return carry == 0;
  }

  static bool add(const UInt& a, const UInt& b, UInt* out) {
    std::uint64_t carry = 0;
    for (std::size_t i = 0; i < Limbs; ++i) {
      const std::uint64_t sum = static_cast<std::uint64_t>(a.limb[i]) +
                                b.limb[i] + carry;
      out->limb[i] = static_cast<std::uint32_t>(sum);
      carry = sum >> 32;
    }
    return carry == 0;
  }

  // Subtract b from a modulo the fixed limb width.  Callers either establish
  // a >= b or deliberately consume a high carry from a prior left shift.
  static bool subtract_mod(const UInt& a, const UInt& b, UInt* out) {
    std::uint64_t borrow = 0;
    for (std::size_t i = 0; i < Limbs; ++i) {
      const std::uint64_t subtrahend =
          static_cast<std::uint64_t>(b.limb[i]) + borrow;
      const std::uint64_t minuend = a.limb[i];
      out->limb[i] = static_cast<std::uint32_t>(minuend - subtrahend);
      borrow = minuend < subtrahend ? 1 : 0;
    }
    return borrow == 0;
  }

  static bool multiply(const UInt& a, const UInt& b, UInt* out) {
    *out = UInt{};
    const std::size_t aw = a.bit_width();
    const std::size_t bw = b.bit_width();
    if (aw == 0 || bw == 0) return true;
    if (aw + bw > Limbs * 32 + 1) return false;
    std::size_t an = (aw + 31) / 32;
    std::size_t bn = (bw + 31) / 32;
    for (std::size_t i = 0; i < an; ++i) {
      std::uint64_t carry = 0;
      for (std::size_t j = 0; j < bn; ++j) {
        const std::size_t k = i + j;
        if (k >= Limbs) return false;
        // The sum is at most 2^64-1: one 32x32 product plus two 32-bit terms.
        const std::uint64_t value =
            static_cast<std::uint64_t>(a.limb[i]) * b.limb[j] +
            out->limb[k] + carry;
        out->limb[k] = static_cast<std::uint32_t>(value);
        carry = value >> 32;
      }
      std::size_t k = i + bn;
      while (carry != 0) {
        if (k >= Limbs) return false;
        const std::uint64_t value =
            static_cast<std::uint64_t>(out->limb[k]) + carry;
        out->limb[k] = static_cast<std::uint32_t>(value);
        carry = value >> 32;
        ++k;
      }
    }
    return true;
  }

  bool set_bit(std::size_t bit) {
    if (bit >= Limbs * 32) return false;
    limb[bit / 32] |= static_cast<std::uint32_t>(1u) << (bit % 32);
    return true;
  }

  bool bit(std::size_t bit_index) const {
    if (bit_index >= Limbs * 32) return false;
    return (limb[bit_index / 32] &
            (static_cast<std::uint32_t>(1u) << (bit_index % 32))) != 0;
  }

  bool shift_left_one() {
    std::uint32_t carry = 0;
    for (std::size_t i = 0; i < Limbs; ++i) {
      const std::uint32_t next = limb[i] >> 31;
      limb[i] = static_cast<std::uint32_t>((limb[i] << 1) | carry);
      carry = next;
    }
    return carry == 0;
  }

  void shift_right_one() {
    std::uint32_t carry = 0;
    for (std::size_t i = Limbs; i-- > 0;) {
      const std::uint32_t next = limb[i] & 1u;
      limb[i] = static_cast<std::uint32_t>((limb[i] >> 1) | (carry << 31));
      carry = next;
    }
  }

  bool shift_left(std::size_t count) {
    for (std::size_t i = 0; i < count; ++i)
      if (!shift_left_one()) return false;
    return true;
  }

  void shift_right(std::size_t count) {
    for (std::size_t i = 0; i < count; ++i) shift_right_one();
  }

  std::size_t trailing_zero_bits() const {
    for (std::size_t i = 0; i < Limbs; ++i) {
      if (limb[i] == 0) continue;
      std::uint32_t value = limb[i];
      unsigned bits = 0;
      while ((value & 1u) == 0) { ++bits; value >>= 1; }
      return i * 32 + bits;
    }
    return Limbs * 32;
  }

  static bool divide_small(const UInt& input, std::uint32_t divisor,
                           UInt* quotient, std::uint32_t* remainder) {
    if (divisor == 0) return false;
    std::uint64_t rem = 0;
    for (std::size_t i = Limbs; i-- > 0;) {
      const std::uint64_t value = (rem << 32) | input.limb[i];
      quotient->limb[i] = static_cast<std::uint32_t>(value / divisor);
      rem = value % divisor;
    }
    *remainder = static_cast<std::uint32_t>(rem);
    return true;
  }

  static bool divide(const UInt& dividend, const UInt& divisor,
                     UInt* quotient, UInt* remainder) {
    if (divisor.zero()) return false;
    *quotient = UInt{};
    *remainder = UInt{};
    const std::size_t width = dividend.bit_width();
    for (std::size_t i = width; i-- > 0;) {
      const bool high_carry = !remainder->shift_left_one();
      if (dividend.bit(i)) remainder->limb[0] |= 1u;
      if (high_carry || compare(*remainder, divisor) >= 0) {
        UInt difference;
        const bool no_borrow = subtract_mod(*remainder, divisor, &difference);
        if (high_carry && no_borrow)
          return false;  // impossible if the long-division invariant holds
        *remainder = difference;
        if (!quotient->set_bit(i)) return false;
      }
    }
    return true;
  }

  static bool parse_decimal(std::string_view text, UInt* output) {
    const std::size_t max_digits = (Limbs * 32 * 30103) / 100000 + 1;
    if (text.empty() || text.size() > max_digits) return false;
    *output = UInt{};
    for (char c : text) {
      if (c < '0' || c > '9') return false;
      if (!output->multiply_small(10) ||
          !output->add_small(static_cast<std::uint32_t>(c - '0')))
        return false;
    }
    return true;
  }

  std::string decimal() const {
    if (zero()) return "0";
    UInt value = *this;
    std::string digits;
    while (!value.zero()) {
      UInt quotient;
      std::uint32_t remainder = 0;
      (void)divide_small(value, 10, &quotient, &remainder);
      digits.push_back(static_cast<char>('0' + remainder));
      value = quotient;
    }
    for (std::size_t i = 0, j = digits.size() - 1; i < j; ++i, --j) {
      const char temp = digits[i]; digits[i] = digits[j]; digits[j] = temp;
    }
    return digits;
  }
};

using UInt256 = UInt<8>;
using UInt768 = UInt<24>;

inline bool copy_to_scratch(const UInt256& source, UInt768* output) {
  *output = UInt768{};
  for (std::size_t i = 0; i < source.limb.size(); ++i)
    output->limb[i] = source.limb[i];
  return true;
}

inline bool copy_to_state(const UInt768& source, UInt256* output) {
  if (source.bit_width() > 256) return false;
  *output = UInt256{};
  for (std::size_t i = 0; i < output->limb.size(); ++i)
    output->limb[i] = source.limb[i];
  return true;
}

inline bool gcd(UInt768 a, UInt768 b, UInt768* output) {
  if (a.zero()) { *output = b; return true; }
  if (b.zero()) { *output = a; return true; }
  const std::size_t common = a.trailing_zero_bits() < b.trailing_zero_bits()
      ? a.trailing_zero_bits() : b.trailing_zero_bits();
  a.shift_right(a.trailing_zero_bits());
  do {
    b.shift_right(b.trailing_zero_bits());
    if (UInt768::compare(a, b) > 0) {
      const UInt768 temp = a; a = b; b = temp;
    }
    UInt768 difference;
    if (!UInt768::subtract_mod(b, a, &difference)) return false;
    b = difference;
  } while (!b.zero());
  if (!a.shift_left(common)) return false;
  *output = a;
  return true;
}

class Rational256 {
 public:
  Rational256() { denominator_.limb[0] = 1; }

  bool negative() const { return negative_; }
  bool zero() const { return numerator_.zero(); }
  std::size_t numerator_bits() const { return numerator_.bit_width(); }
  std::size_t denominator_bits() const { return denominator_.bit_width(); }

  std::string to_string() const {
    const std::string n = numerator_.decimal();
    return std::string(negative_ && !zero() ? "-" : "") + n + "/" +
           denominator_.decimal();
  }

  bool equal(const Rational256& other) const {
    return negative_ == other.negative_ &&
        UInt256::compare(numerator_, other.numerator_) == 0 &&
        UInt256::compare(denominator_, other.denominator_) == 0;
  }

  static bool from_integer(std::string_view text, Rational256* output,
                           std::string* error) {
    std::string_view digits = text;
    bool negative = false;
    if (!digits.empty() && (digits.front() == '-' || digits.front() == '+')) {
      negative = digits.front() == '-';
      digits.remove_prefix(1);
    }
    UInt768 numerator;
    if (!UInt768::parse_decimal(digits, &numerator))
      return fail(error, "integer input exceeds 768-bit parse scratch");
    UInt768 denominator;
    denominator.limb[0] = 1;
    return normalize(numerator, negative, denominator, output, error);
  }

  static bool from_fraction(std::string_view text, Rational256* output,
                            std::string* error) {
    if (text.size() > 468)
      return fail(error, "rational text exceeds fixed parser input bound");
    const std::size_t slash = text.find('/');
    if (slash == std::string_view::npos)
      return from_integer(text, output, error);
    std::string_view ntext = text.substr(0, slash);
    std::string_view dtext = text.substr(slash + 1);
    bool negative = false;
    if (!ntext.empty() && (ntext.front() == '-' || ntext.front() == '+')) {
      negative = ntext.front() == '-';
      ntext.remove_prefix(1);
    }
    if (!dtext.empty() && dtext.front() == '+') dtext.remove_prefix(1);
    if (!dtext.empty() && dtext.front() == '-')
      return fail(error, "denominator must be positive");
    UInt768 numerator, denominator;
    if (!UInt768::parse_decimal(ntext, &numerator) ||
        !UInt768::parse_decimal(dtext, &denominator))
      return fail(error, "rational input exceeds 768-bit parse scratch");
    return normalize(numerator, negative, denominator, output, error);
  }

  static bool from_double(double value, Rational256* output,
                          std::string* error) {
    static_assert(sizeof(double) == sizeof(std::uint64_t),
                  "binary64 conversion requires 64-bit double storage");
    static_assert(std::numeric_limits<double>::is_iec559,
                  "binary64 conversion requires IEC 60559 doubles");
    std::uint64_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    const std::uint64_t exponent = (bits >> 52) & 0x7ffu;
    const std::uint64_t fraction = bits & ((std::uint64_t{1} << 52) - 1);
    const bool negative = (bits >> 63) != 0;
    if (exponent == 0x7ffu)
      return fail(error, "non-finite binary64 input");
    if (exponent == 0 && fraction == 0)
      return from_integer("0", output, error);
    std::uint64_t significand = fraction;
    int power = 1 - 1023 - 52;
    if (exponent != 0) {
      significand |= (std::uint64_t{1} << 52);
      power = static_cast<int>(exponent) - 1023 - 52;
    }
    UInt768 numerator, denominator;
    numerator.limb[0] = static_cast<std::uint32_t>(significand);
    numerator.limb[1] = static_cast<std::uint32_t>(significand >> 32);
    if (power >= 0) {
      denominator.limb[0] = 1;
      if (!numerator.shift_left(static_cast<std::size_t>(power)))
        return fail(error, "binary64 significand exceeds fixed parse scratch");
    } else {
      const std::size_t shift = static_cast<std::size_t>(-power);
      if (!denominator.set_bit(shift))
        return fail(error, "binary64 denominator exceeds fixed parse scratch");
    }
    return normalize(numerator, negative, denominator, output, error);
  }

  static bool add(const Rational256& a, const Rational256& b,
                  Rational256* output, std::string* error) {
    UInt768 an, ad, bn, bd, left, right, denominator;
    copy_to_scratch(a.numerator_, &an); copy_to_scratch(a.denominator_, &ad);
    copy_to_scratch(b.numerator_, &bn); copy_to_scratch(b.denominator_, &bd);
    if (!UInt768::multiply(an, bd, &left) ||
        !UInt768::multiply(bn, ad, &right) ||
        !UInt768::multiply(ad, bd, &denominator))
      return fail(error, "768-bit addition cross-product overflow");
    bool negative = false;
    UInt768 numerator;
    if (a.negative_ == b.negative_) {
      if (!UInt768::add(left, right, &numerator))
        return fail(error, "768-bit addition carry overflow");
      negative = a.negative_;
    } else {
      const int cmp = UInt768::compare(left, right);
      if (cmp >= 0) {
        (void)UInt768::subtract_mod(left, right, &numerator);
        negative = a.negative_;
      } else {
        (void)UInt768::subtract_mod(right, left, &numerator);
        negative = b.negative_;
      }
    }
    return normalize(numerator, negative, denominator, output, error);
  }

  static bool subtract(const Rational256& a, const Rational256& b,
                       Rational256* output, std::string* error) {
    Rational256 negated = b;
    if (!negated.zero()) negated.negative_ = !negated.negative_;
    return add(a, negated, output, error);
  }

  static bool multiply(const Rational256& a, const Rational256& b,
                       Rational256* output, std::string* error) {
    UInt768 an, ad, bn, bd, numerator, denominator;
    copy_to_scratch(a.numerator_, &an); copy_to_scratch(a.denominator_, &ad);
    copy_to_scratch(b.numerator_, &bn); copy_to_scratch(b.denominator_, &bd);
    if (!UInt768::multiply(an, bn, &numerator) ||
        !UInt768::multiply(ad, bd, &denominator))
      return fail(error, "768-bit multiplication overflow");
    return normalize(numerator, a.negative_ != b.negative_, denominator,
                     output, error);
  }

  static bool divide(const Rational256& a, const Rational256& b,
                     Rational256* output, std::string* error) {
    if (b.zero()) return fail(error, "division by zero");
    UInt768 an, ad, bn, bd, numerator, denominator;
    copy_to_scratch(a.numerator_, &an); copy_to_scratch(a.denominator_, &ad);
    copy_to_scratch(b.numerator_, &bn); copy_to_scratch(b.denominator_, &bd);
    if (!UInt768::multiply(an, bd, &numerator) ||
        !UInt768::multiply(ad, bn, &denominator))
      return fail(error, "768-bit division cross-product overflow");
    return normalize(numerator, a.negative_ != b.negative_, denominator,
                     output, error);
  }

  static bool compare(const Rational256& a, const Rational256& b,
                      int* result, std::string* error) {
    if (a.zero() && b.zero()) { *result = 0; return true; }
    if (a.negative_ != b.negative_) {
      *result = a.negative_ ? -1 : 1;
      return true;
    }
    UInt768 an, ad, bn, bd, left, right;
    copy_to_scratch(a.numerator_, &an); copy_to_scratch(a.denominator_, &ad);
    copy_to_scratch(b.numerator_, &bn); copy_to_scratch(b.denominator_, &bd);
    const bool left_ok = UInt768::multiply(an, bd, &left);
    const bool right_ok = UInt768::multiply(bn, ad, &right);
    if (!left_ok || !right_ok)
      return fail(error, "768-bit comparison cross-product overflow");
    const int cmp = UInt768::compare(left, right);
    *result = a.negative_ ? -cmp : cmp;
    return true;
  }

  static bool multiply_repeat(const Rational256& value,
                              std::string_view repeat_decimal,
                              Rational256* output, std::string* error) {
    UInt768 repeat, numerator, denominator;
    if (!UInt768::parse_decimal(repeat_decimal, &repeat) ||
        repeat.bit_width() > 128)
      return fail(error, "repeat count exceeds checked u128 range");
    UInt768 n;
    copy_to_scratch(value.numerator_, &n);
    copy_to_scratch(value.denominator_, &denominator);
    if (!UInt768::multiply(n, repeat, &numerator))
      return fail(error, "768-bit repeated-ledger product overflow");
    return normalize(numerator, value.negative_, denominator, output, error);
  }

 private:
  bool negative_ = false;
  UInt256 numerator_{};
  UInt256 denominator_{};

  static bool fail(std::string* error, const char* message) {
    if (error != nullptr) *error = message;
    return false;
  }

  static bool normalize(const UInt768& numerator, bool negative,
                        const UInt768& denominator, Rational256* output,
                        std::string* error) {
    if (denominator.zero()) return fail(error, "zero denominator");
    if (numerator.zero()) {
      *output = Rational256{};
      return true;
    }
    UInt768 common, reduced_n, nrem, reduced_d, drem;
    if (!gcd(numerator, denominator, &common) || common.zero() ||
        !UInt768::divide(numerator, common, &reduced_n, &nrem) ||
        !UInt768::divide(denominator, common, &reduced_d, &drem) ||
        !nrem.zero() || !drem.zero())
      return fail(error, "fixed-limb rational normalization failed");
    UInt256 nslot, dslot;
    if (!copy_to_state(reduced_n, &nslot) ||
        !copy_to_state(reduced_d, &dslot))
      return fail(error, "normalized rational exceeds 256-bit state slot");
    output->negative_ = negative;
    output->numerator_ = nslot;
    output->denominator_ = dslot;
    return true;
  }
};

}  // namespace os_fixedwidth
