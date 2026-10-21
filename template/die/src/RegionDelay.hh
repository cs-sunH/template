#pragma once

#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace coarse_die::detail::region_delay {

struct Result {
  std::uint64_t whole=0;
  std::uint64_t ceil=0;
  bool integral() const noexcept {return whole==ceil;}
};

namespace implementation {
using u64=std::uint64_t;
using u128=__uint128_t;

struct Term {
  u64 whole=0;
  u128 remainder=0;
  unsigned denominator_shift=0;
};

[[noreturn]] inline void overflow() {
  throw std::overflow_error("overflow: region delay exceeds uint64 nanoseconds");
}

inline u64 checked_add(u64 a,u64 b) {
  if(b>std::numeric_limits<u64>::max()-a)overflow();
  return a+b;
}

// binary64 = an exact 53-bit integer times a power of two. Even subnormal
// inputs use this representation without rounding: frexp normalizes them.
// A 53-bit significand times uint64 occupies at most 117 bits.
inline Term multiply(double value,u64 count) {
  if(value==0 || count==0)return {};
  int exponent=0;
  const double fraction=std::frexp(value,&exponent);
  const u64 significand=static_cast<u64>(std::ldexp(fraction,53));
  const u128 product=static_cast<u128>(significand)*count;
  exponent-=53;
  if(exponent>=0) {
    if(exponent>=64 || product>(std::numeric_limits<u64>::max()>>exponent))overflow();
    return {static_cast<u64>(product)<<exponent,0,0};
  }
  const unsigned shift=static_cast<unsigned>(-exponent);
  const u128 whole=shift>=128 ? 0 : product>>shift;
  if(whole>std::numeric_limits<u64>::max())overflow();
  const u128 remainder=shift>=128 ? product : product-(whole<<shift);
  return {static_cast<u64>(whole),remainder,shift};
}

// Compare the two proper fractions' sum with one. No common denominator,
// large shift, floating sum or repeated iteration is needed.
inline int compare_fraction_sum_to_one(Term a,Term b) {
  if(a.remainder==0 || b.remainder==0)return -1;
  if(a.denominator_shift>b.denominator_shift) {
    const Term old=a;a=b;b=old;
  }
  // Each numerator has at most 117 bits. If both denominators are at least
  // 2^128, even their combined upper bound has fewer than 128 bits.
  if(a.denominator_shift>=128)return -1;
  const u128 gap=(static_cast<u128>(1)<<a.denominator_shift)-a.remainder;
  const unsigned shift=b.denominator_shift-a.denominator_shift;
  if(shift>=128)return -1;  // gap*2^shift exceeds any possible numerator.
  const u128 quotient=b.remainder>>shift;
  if(quotient<gap)return -1;
  if(quotient>gap)return 1;
  // Equality of the quotient can still leave a positive low-bit residue.
  return b.remainder-(quotient<<shift)==0 ? 0 : 1;
}
}  // namespace implementation

// Return exact floor/ceil of (a+b)*count for the input binary64 values.
// Only two <=117-bit products and fixed exponent comparisons are required;
// count never controls a loop or the size of an intermediate representation.
inline Result sum(double a,double b,std::uint64_t count) {
  static_assert(std::numeric_limits<double>::is_iec559 &&
      std::numeric_limits<double>::radix==2 && std::numeric_limits<double>::digits==53,
      "region delay requires IEEE-754 binary64");
  if(!std::isfinite(a) || !std::isfinite(b) || a<0 || b<0)
    throw std::invalid_argument("invalid_input: region delay coefficients must be finite and nonnegative");
  const auto first=implementation::multiply(a,count);
  const auto second=implementation::multiply(b,count);
  auto whole=implementation::checked_add(first.whole,second.whole);
  const bool any_fraction=first.remainder!=0 || second.remainder!=0;
  const int comparison=implementation::compare_fraction_sum_to_one(first,second);
  if(comparison>=0)whole=implementation::checked_add(whole,1);
  const bool remaining_fraction=any_fraction && comparison!=0;
  return {whole,remaining_fraction ? implementation::checked_add(whole,1) : whole};
}
}  // namespace coarse_die::detail::region_delay
