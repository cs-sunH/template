#pragma once
// Standalone research core. No JSON, Runtime, model state, or arbitrary precision.
#include <array>
#include <cfenv>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
namespace norm_dyadic128 {
using Real=long double;
__extension__ using U128=unsigned __int128;
constexpr U128 max128=~U128{0};
inline bool supported_abi(){return std::numeric_limits<Real>::digits==64&&std::numeric_limits<double>::is_iec559&&std::numeric_limits<double>::digits==53&&sizeof(double)==8&&std::fegetround()==FE_TONEAREST;}
inline unsigned bits(U128 x){if(!x)return 0;const auto hi=static_cast<std::uint64_t>(x>>64);return hi?128-__builtin_clzll(hi):64-__builtin_clzll(static_cast<std::uint64_t>(x));}
inline unsigned zeros(U128 x){const auto lo=static_cast<std::uint64_t>(x);return lo?__builtin_ctzll(lo):64+__builtin_ctzll(static_cast<std::uint64_t>(x>>64));}
struct Term {U128 mantissa=0;int exponent=0;bool negative=false;};
inline void strip(Term& t){if(!t.mantissa){t.exponent=0;t.negative=false;return;}const auto z=zeros(t.mantissa);t.mantissa>>=z;t.exponent+=static_cast<int>(z);}
inline bool decompose_double(double x,Term& t,std::string& reason){
 if(!std::isfinite(x)){reason="nonfinite_binary64";return false;}if(x==0){t={};return true;}
 std::uint64_t word=0;std::memcpy(&word,&x,sizeof word);const auto exp=static_cast<unsigned>((word>>52)&2047);if(exp==0){reason="nonzero_subnormal_binary64";return false;}
 t={U128{(word&((std::uint64_t{1}<<52)-1))|(std::uint64_t{1}<<52)},static_cast<int>(exp)-1023-52,(word>>63)!=0};strip(t);return true;
}
inline bool decompose_progress(Real p,Term& t,std::string& reason){
 if(!std::isfinite(p)||p<0||p>1){reason="invalid_progress";return false;}if(p==0){t={};return true;}
 if(std::fpclassify(p)!=FP_NORMAL){reason="nonzero_subnormal_progress";return false;}
 int exp=0;const Real fraction=std::frexp(p,&exp);const Real scaled=std::ldexp(fraction,64);
 if(!(scaled>=std::ldexp(1.0L,63)&&scaled<std::ldexp(1.0L,64))){reason="invalid_progress_significand";return false;}
 t={static_cast<std::uint64_t>(scaled),exp-64,false};strip(t);return true;
}
inline Term product(Term a,Term b){Term result;if(!a.mantissa||!b.mantissa)return result;result={a.mantissa*b.mantissa,a.exponent+b.exponent,a.negative!=b.negative};strip(result);return result;}
struct Result {bool supported=false;const char* reason="unset";int sign=0,base_exponent=0,exponent=0;U128 positive=0,negative=0,magnitude=0;Real lower=0,upper=0;};
inline bool normal_scale(std::uint64_t m,int exponent){if(!m)return true;const int top=exponent+static_cast<int>(64-__builtin_clzll(m))-1;return top>=std::numeric_limits<Real>::min_exponent-1&&top<std::numeric_limits<Real>::max_exponent;}
inline Result sum_terms(const std::array<Term,5>& terms,std::size_t count,bool make_interval){
 Result r;if(count>terms.size()){r.reason="unsupported_term_count";return r;}for(std::size_t i=0;i<count;++i)if(terms[i].mantissa&&(terms[i].exponent < -32768 || terms[i].exponent > 32768)){r.reason="unsupported_term_exponent";return r;}bool any=false;int base=0;for(std::size_t i=0;i<count;++i)if(terms[i].mantissa){if(!any||terms[i].exponent<base)base=terms[i].exponent;any=true;}
 r.base_exponent=base;
 for(std::size_t i=0;i<count;++i){const auto& t=terms[i];if(!t.mantissa)continue;const int shift=t.exponent-base;
  if(shift<0||shift>=128||bits(t.mantissa)+static_cast<unsigned>(shift)>128){r.reason="alignment_exceeds_128_bits";return r;}
  const U128 value=t.mantissa<<static_cast<unsigned>(shift);U128& group=t.negative?r.negative:r.positive;
  if(value>max128-group){r.reason=t.negative?"negative_accumulator_overflow":"positive_accumulator_overflow";return r;}group+=value;
 }
 r.sign=(r.positive>r.negative)-(r.positive<r.negative);r.magnitude=r.sign<0?r.negative-r.positive:r.positive-r.negative;r.exponent=base;
 if(!r.magnitude){r.supported=true;r.reason="supported_exact_zero";return r;}
 Term normalized{r.magnitude,r.exponent,false};strip(normalized);r.magnitude=normalized.mantissa;r.exponent=normalized.exponent;
 const auto nbits=bits(r.magnitude);std::uint64_t floor_m=0,ceil_m=0;int floor_e=r.exponent,ceil_e=r.exponent;
 if(nbits<=64){floor_m=ceil_m=static_cast<std::uint64_t>(r.magnitude);}
 else {const unsigned shift=nbits-64;floor_m=static_cast<std::uint64_t>(r.magnitude>>shift);const U128 remainder=r.magnitude&((U128{1}<<shift)-1);floor_e+=static_cast<int>(shift);ceil_e=floor_e;ceil_m=floor_m;
  if(remainder){if(floor_m==std::numeric_limits<std::uint64_t>::max()){ceil_m=std::uint64_t{1}<<63;++ceil_e;}else ++ceil_m;}
 }
 // Check integer significand/exponent bounds BEFORE any floating conversion
 // or power-of-two scale. There is no cast-back overflow test.
 if(!normal_scale(floor_m,floor_e)||!normal_scale(ceil_m,ceil_e)){const int top=r.exponent+static_cast<int>(nbits)-1;r.reason=top<std::numeric_limits<Real>::min_exponent-1?"result_or_bound_subnormal":"result_or_bound_overflow";return r;}
 if(make_interval){const Real lo=std::ldexp(static_cast<Real>(floor_m),floor_e),hi=std::ldexp(static_cast<Real>(ceil_m),ceil_e);r.lower=r.sign<0?-hi:lo;r.upper=r.sign<0?-lo:hi;}
 r.supported=true;r.reason="supported";return r;
}
struct Sample {double initial=0,capacity=0;std::array<double,3> d{};std::array<Real,3> p{};std::size_t n=0;};
struct Evaluation {Result result;std::array<Term,5> terms{};std::size_t count=0;};
inline Evaluation evaluate(const Sample& s,bool head,bool interval){
 Evaluation e;if(!supported_abi()){e.result.reason="unsupported_numeric_ABI";return e;}if(s.n>3){e.result.reason="unsupported_activity_count";return e;}std::string reason;std::array<Term,3> pd{},cd{};Term initial,capacity;
 for(std::size_t i=0;i<s.n;++i)if(!decompose_progress(s.p[i],pd[i],reason)){e.result.reason="nonzero_subnormal_or_invalid_progress";return e;}
 if(!decompose_double(s.initial,initial,reason)||!decompose_double(s.capacity,capacity,reason)){e.result.reason="nonfinite_or_subnormal_constant";return e;}
 for(std::size_t i=0;i<s.n;++i)if(!decompose_double(s.d[i],cd[i],reason)){e.result.reason="nonfinite_or_subnormal_coefficient";return e;}
 if(head){e.terms[e.count++]=capacity;if(initial.mantissa)initial.negative=!initial.negative;e.terms[e.count++]=initial;}else e.terms[e.count++]=initial;
 for(std::size_t i=0;i<s.n;++i){Term t=product(cd[i],pd[i]);if(head&&t.mantissa)t.negative=!t.negative;e.terms[e.count++]=t;}
 e.result=sum_terms(e.terms,e.count,interval);return e;
}

} // namespace norm_dyadic128
