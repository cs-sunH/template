// Fixed-precision, fail-closed interval evaluator for nested_k_output.py.
// This is an independent scratch prototype, not production certificate code.
#include <array>
#include <algorithm>
#include <cfenv>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <locale>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr std::size_t kLimbs = 8;
constexpr std::size_t kLedgerCount = 10;
constexpr long double kBase32 = 4294967296.0L;
const long double kInf = std::numeric_limits<long double>::infinity();

[[noreturn]] void fail(const std::string& message) {
  throw std::runtime_error(message);
}

long double rounded_add(long double x, long double y) {
  volatile long double z = x + y;
  return z;
}
long double rounded_sub(long double x, long double y) {
  volatile long double z = x - y;
  return z;
}
long double rounded_mul(long double x, long double y) {
  volatile long double z = x * y;
  return z;
}
long double rounded_div(long double x, long double y) {
  volatile long double z = x / y;
  return z;
}

long double down(long double x) {
  if (!std::isfinite(x)) fail("nonfinite lower endpoint / overflow");
  const long double y = std::nextafter(x, -kInf);
  if (!std::isfinite(y)) fail("outward lower endpoint overflow");
  return y;
}
long double up(long double x) {
  if (!std::isfinite(x)) fail("nonfinite upper endpoint / overflow");
  const long double y = std::nextafter(x, kInf);
  if (!std::isfinite(y)) fail("outward upper endpoint overflow");
  return y;
}

struct Interval {
  long double lo;
  long double hi;

  Interval() : lo(0.0L), hi(0.0L) {}
  Interval(long double lower, long double upper) : lo(lower), hi(upper) {
    if (!std::isfinite(lo) || !std::isfinite(hi) || lo > hi)
      fail("invalid or overflowing interval");
  }
  static Interval point(long double x) { return Interval(x, x); }
};

Interval add(const Interval& x, const Interval& y) {
  return Interval(down(rounded_add(x.lo, y.lo)),
                  up(rounded_add(x.hi, y.hi)));
}
Interval sub(const Interval& x, const Interval& y) {
  return Interval(down(rounded_sub(x.lo, y.hi)),
                  up(rounded_sub(x.hi, y.lo)));
}
Interval neg(const Interval& x) { return Interval(-x.hi, -x.lo); }

Interval mul(const Interval& x, const Interval& y) {
  const std::array<long double, 4> p{
      rounded_mul(x.lo, y.lo), rounded_mul(x.lo, y.hi),
      rounded_mul(x.hi, y.lo), rounded_mul(x.hi, y.hi)};
  for (long double v : p)
    if (!std::isfinite(v)) fail("interval multiplication overflow");
  const auto mm = std::minmax_element(p.begin(), p.end());
  return Interval(down(*mm.first), up(*mm.second));
}

Interval div(const Interval& x, const Interval& y) {
  if (y.lo <= 0.0L && y.hi >= 0.0L)
    fail("division by an interval containing zero");
  const std::array<long double, 4> q{
      rounded_div(x.lo, y.lo), rounded_div(x.lo, y.hi),
      rounded_div(x.hi, y.lo), rounded_div(x.hi, y.hi)};
  for (long double v : q)
    if (!std::isfinite(v)) fail("interval division overflow");
  const auto mm = std::minmax_element(q.begin(), q.end());
  return Interval(down(*mm.first), up(*mm.second));
}

bool strictly_inside(const Interval& x, const Interval& lo,
                     const Interval& hi) {
  return x.lo > lo.hi && x.hi < hi.lo;
}

struct UInt256 {
  std::array<std::uint32_t, kLimbs> word{};  // little endian

  bool mul10_add(unsigned digit) {
    std::uint64_t carry = digit;
    for (std::size_t i = 0; i < kLimbs; ++i) {
      const std::uint64_t v = std::uint64_t(word[i]) * 10 + carry;
      word[i] = static_cast<std::uint32_t>(v);
      carry = v >> 32;
    }
    return carry == 0;
  }

  static UInt256 decimal(const std::string& text) {
    if (text.empty()) fail("empty integer in descriptor");
    UInt256 value;
    for (char c : text) {
      if (c < '0' || c > '9') fail("invalid unsigned decimal in descriptor");
      if (!value.mul10_add(static_cast<unsigned>(c - '0')))
        fail("descriptor integer exceeds fixed 256-bit limb capacity");
    }
    return value;
  }

  bool zero() const {
    for (auto v : word)
      if (v != 0) return false;
    return true;
  }

  Interval to_interval() const {
    Interval acc = Interval::point(0.0L);
    const Interval base = Interval::point(kBase32);
    for (std::size_t i = kLimbs; i-- > 0;) {
      acc = add(mul(acc, base),
                Interval::point(static_cast<long double>(word[i])));
    }
    return acc;
  }
};

Interval rational_interval(const std::string& input) {
  const auto slash = input.find('/');
  std::string numerator = input.substr(0, slash);
  std::string denominator = slash == std::string::npos
                                ? "1"
                                : input.substr(slash + 1);
  bool negative = false;
  if (!numerator.empty() && (numerator[0] == '-' || numerator[0] == '+')) {
    negative = numerator[0] == '-';
    numerator.erase(0, 1);
  }
  const UInt256 n = UInt256::decimal(numerator);
  const UInt256 d = UInt256::decimal(denominator);
  if (d.zero()) fail("zero rational denominator");
  Interval quotient = div(n.to_interval(), d.to_interval());
  return negative ? neg(quotient) : quotient;
}

std::uint64_t parse_u64(const std::string& s) {
  if (s.empty()) fail("empty unsigned integer metadata");
  std::uint64_t value = 0;
  for (char c : s) {
    if (c < '0' || c > '9') fail("invalid unsigned integer metadata");
    const unsigned d = static_cast<unsigned>(c - '0');
    if (value > (std::numeric_limits<std::uint64_t>::max() - d) / 10)
      fail("metadata integer exceeds uint64 range");
    value = value * 10 + d;
  }
  return value;
}

std::uint64_t checked_add_u64(std::uint64_t x, std::uint64_t y,
                              const std::string& what) {
  if (x > std::numeric_limits<std::uint64_t>::max() - y)
    fail("uint64 addition overflow while checking " + what);
  return x + y;
}
std::uint64_t checked_mul_u64(std::uint64_t x, std::uint64_t y,
                              const std::string& what) {
  if (y != 0 && x > std::numeric_limits<std::uint64_t>::max() / y)
    fail("uint64 multiplication overflow while checking " + what);
  return x * y;
}

using Fields = std::map<std::string, std::string>;
Fields read_fields(const std::string& path) {
  std::ifstream in(path);
  if (!in) fail("cannot open affine descriptor: " + path);
  Fields fields;
  std::string line;
  std::size_t line_no = 0;
  while (std::getline(in, line)) {
    ++line_no;
    if (line.empty()) continue;
    const auto equal = line.find('=');
    if (equal == std::string::npos || equal == 0)
      fail("malformed descriptor line " + std::to_string(line_no));
    const std::string key = line.substr(0, equal);
    const std::string value = line.substr(equal + 1);
    if (!fields.emplace(key, value).second)
      fail("duplicate descriptor field: " + key);
  }
  return fields;
}
const std::string& need(const Fields& f, const std::string& key) {
  auto it = f.find(key);
  if (it == f.end()) fail("missing descriptor field: " + key);
  return it->second;
}

const std::array<std::string, kLedgerCount> kLedgerNames{
    "time_s", "compute_ops", "compute_gate_s", "local_bytes",
    "hbm0_bytes", "hbm1_bytes", "busy_compute_s", "busy_hbm0_s",
    "busy_hbm1_s", "busy_hbm_union_s"};

struct Transform {
  Interval a;
  Interval b;
  std::array<Interval, kLedgerCount> c;
  std::array<Interval, kLedgerCount> d;
  explicit Transform(Interval aa, Interval bb)
      : a(aa), b(bb), c{}, d{} {
    for (auto& x : c) x = Interval::point(0.0L);
    for (auto& x : d) x = Interval::point(0.0L);
  }
};

std::uint64_t g_compose_count = 0;
Transform compose(const Transform& first, const Transform& second) {
  ++g_compose_count;
  Transform out(mul(second.a, first.a),
                add(mul(second.a, first.b), second.b));
  for (std::size_t i = 0; i < kLedgerCount; ++i) {
    out.c[i] = add(first.c[i], mul(second.c[i], first.a));
    out.d[i] = add(add(first.d[i], mul(second.c[i], first.b)), second.d[i]);
  }
  return out;
}

Transform identity() {
  Transform out(Interval::point(1.0L), Interval::point(0.0L));
  return out;
}

Transform power(Transform base, std::uint64_t count) {
  Transform result = identity();
  while (count != 0) {
    if (count & 1U) result = compose(result, base);
    count >>= 1;
    if (count != 0) base = compose(base, base);
  }
  return result;
}

struct Applied {
  Interval x;
  std::array<Interval, kLedgerCount> y;
};
Applied apply(const Transform& t, const Interval& x) {
  Applied out{add(mul(t.a, x), t.b), {}};
  for (std::size_t i = 0; i < kLedgerCount; ++i)
    out.y[i] = add(mul(t.c[i], x), t.d[i]);
  return out;
}

std::string decimal_outward(long double x, bool lower) {
  // One extra ulp leaves room for decimal serialization rounding. max_digits10
  // round-trips the long double; the printed exact decimal remains outward.
  const long double safe = lower ? down(x) : up(x);
  std::ostringstream out;
  out.imbue(std::locale::classic());
  out << std::scientific
      << std::setprecision(std::numeric_limits<long double>::max_digits10)
      << safe;
  return out.str();
}
void emit_interval(const std::string& key, const Interval& x) {
  std::cout << key << '=' << decimal_outward(x.lo, true) << '|'
            << decimal_outward(x.hi, false) << '\n';
}

std::uint64_t safe_count(const Transform& base, const Interval& x0,
                         const Interval& stop, std::uint64_t upper) {
  std::uint64_t lo = 0;
  std::uint64_t hi = upper + 1;
  while (lo + 1 < hi) {
    const std::uint64_t mid = lo + (hi - lo) / 2;
    const Applied v = apply(power(base, mid), x0);
    const Interval& elapsed = v.y[0];
    if (elapsed.hi <= stop.lo) {
      lo = mid;
    } else if (elapsed.lo > stop.hi) {
      hi = mid;
    } else {
      fail("external stop overlaps an elapsed-time interval; refusing ambiguous count");
    }
  }
  const Applied at = apply(power(base, lo), x0);
  const Applied next = apply(power(base, lo + 1), x0);
  if (!(at.y[0].hi <= stop.lo && next.y[0].lo > stop.hi))
    fail("external stop lacks strict safe/next-segment certificate");
  return lo;
}

void require(bool condition, const std::string& message) {
  if (!condition) fail(message);
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc != 2) fail("usage: fixed_interval_affine DESCRIPTOR.affine");
    require(std::numeric_limits<long double>::is_iec559,
            "long double is not IEC 60559 binary floating point");
    require(std::numeric_limits<long double>::radix == 2,
            "long double radix is not binary");
    require(std::numeric_limits<long double>::digits == 64,
            "prototype requires the checked 64-bit-significand x87 long double format");
    require(std::fesetround(FE_TONEAREST) == 0,
            "cannot set FE_TONEAREST");

    const Fields f = read_fields(argv[1]);
    require(need(f, "version") == "1", "unsupported descriptor version");
    require(parse_u64(need(f, "state_dimension")) == 1,
            "prototype is limited to one affine state coordinate");
    require(parse_u64(need(f, "ledger_count")) == kLedgerCount,
            "unexpected additive-ledger dimension");
    const std::uint64_t template_steps = parse_u64(need(f, "event_template_steps"));
    const std::uint64_t tail = parse_u64(need(f, "tail_half_bound"));
    const std::uint64_t first_nonfull = parse_u64(need(f, "first_nonfull_wave"));
    const std::uint64_t k_tiles = parse_u64(need(f, "k_tiles"));
    const std::uint64_t cursor0 = parse_u64(need(f, "initial_cursor_owner0"));
    const std::uint64_t cursor1 = parse_u64(need(f, "initial_cursor_owner1"));
    const std::uint64_t stop_ns = parse_u64(need(f, "external_stop_ns"));
    require(tail < std::numeric_limits<std::uint64_t>::max(),
            "tail bound leaves no representable successor count");
    require(template_steps == 32, "descriptor category-template event count changed");
    require(k_tiles != 0 && first_nonfull >= cursor0 && first_nonfull >= cursor1,
            "invalid lifecycle tail metadata");
    const std::uint64_t cursor_bound0 = (first_nonfull - cursor0) / k_tiles;
    const std::uint64_t cursor_bound1 = (first_nonfull - cursor1) / k_tiles;
    require(tail == std::min(cursor_bound0, cursor_bound1),
            "tail bound does not equal the two-owner cursor minimum");
    const std::uint64_t tail_waves = checked_mul_u64(tail, k_tiles, "tail bound");
    require(checked_add_u64(cursor0, tail_waves, "owner0 tail bound") <= first_nonfull &&
                checked_add_u64(cursor1, tail_waves, "owner1 tail bound") <= first_nonfull,
            "claimed tail bound crosses the first non-full wave");
    const std::uint64_t successor = checked_add_u64(tail, 1, "tail successor");
    bool owner0_crosses = false;
    bool owner1_crosses = false;
    if (successor <= (std::numeric_limits<std::uint64_t>::max() - cursor0) / k_tiles)
      owner0_crosses = checked_add_u64(
          cursor0, checked_mul_u64(successor, k_tiles, "owner0 tail successor"),
          "owner0 tail successor") > first_nonfull;
    if (successor <= (std::numeric_limits<std::uint64_t>::max() - cursor1) / k_tiles)
      owner1_crosses = checked_add_u64(
          cursor1, checked_mul_u64(successor, k_tiles, "owner1 tail successor"),
          "owner1 tail successor") > first_nonfull;
    require(owner0_crosses || owner1_crosses,
            "tail bound is not maximal");

    Interval a = rational_interval(need(f, "a"));
    Interval b = rational_interval(need(f, "b"));
    Interval x0 = rational_interval(need(f, "x0"));
    Interval domain_lo = rational_interval(need(f, "domain_lo"));
    Interval domain_hi = rational_interval(need(f, "domain_hi"));
    Interval expected_fixed = rational_interval(need(f, "fixed"));
    require(a.lo > 0.0L && a.hi < 1.0L,
            "state map is not a strict positive contraction");
    require(b.lo > 0.0L, "state-map translation is not strictly positive");
    const Interval fixed = div(b, sub(Interval::point(1.0L), a));
    require(fixed.lo <= expected_fixed.hi && fixed.hi >= expected_fixed.lo,
            "computed fixed-point interval does not overlap descriptor's exact fixed point");
    require(fixed.hi < x0.lo, "initial state is not strictly above the fixed point");
    require(strictly_inside(fixed, domain_lo, domain_hi) &&
                strictly_inside(x0, domain_lo, domain_hi),
            "contraction orbit endpoints are not strictly inside event-order domain");
    const Interval x0_next = add(mul(a, x0), b);
    require(x0_next.hi < x0.lo,
            "initial-state map does not strictly descend toward its fixed point");
    const Interval orbit(fixed.lo, x0.hi);
    const Interval orbit_image = add(mul(a, orbit), b);
    require(orbit_image.lo >= domain_lo.hi && orbit_image.hi <= domain_hi.lo,
            "one-step orbit image is not inside strict event-order domain");

    Transform base(a, b);
    for (std::size_t i = 0; i < kLedgerCount; ++i) {
      const std::string prefix = "ledger." + kLedgerNames[i] + ".";
      base.c[i] = rational_interval(need(f, prefix + "slope"));
      base.d[i] = rational_interval(need(f, prefix + "intercept"));
      const Interval increment = add(mul(base.c[i], orbit), base.d[i]);
      require(increment.lo > 0.0L,
              "ledger increment is not strictly positive throughout the orbit: " +
                  kLedgerNames[i]);
    }

    const std::array<std::uint64_t, 6> samples{1, 10, 100, 995,
                                               1000000, 1000000000000ULL};
    for (std::uint64_t n : samples) {
      g_compose_count = 0;
      const Transform powered = power(base, n);
      const Applied value = apply(powered, x0);
      const bool lifecycle_bounded = n <= tail;
      if (lifecycle_bounded) {
        require(strictly_inside(value.x, domain_lo, domain_hi),
                "bounded output state escaped the strict event-order domain");
      }
      if (n == tail) {
        const Interval elapsed_width = sub(Interval::point(value.y[0].hi),
                                            Interval::point(value.y[0].lo));
        const Interval one_nanosecond = rational_interval("1/1000000000");
        require(elapsed_width.hi < one_nanosecond.lo,
                "995-segment elapsed-time enclosure exceeds one nanosecond");
      }
      std::cout << "case." << n << ".scope="
                << (lifecycle_bounded ? "lifecycle-bounded" : "numeric-only") << '\n';
      std::cout << "case." << n << ".composition_ops=" << g_compose_count << '\n';
      emit_interval("case." + std::to_string(n) + ".state", value.x);
      for (std::size_t i = 0; i < kLedgerCount; ++i)
        emit_interval("case." + std::to_string(n) + ".ledger." +
                          kLedgerNames[i], value.y[i]);
    }

    const Interval stop = div(rational_interval(std::to_string(stop_ns)),
                               rational_interval("1000000000"));
    g_compose_count = 0;
    const std::uint64_t count = safe_count(base, x0, stop, tail);
    require(count == 123, "external stop did not certify 123 whole segments");
    std::cout << "external_stop.count=" << count << '\n';
    std::cout << "external_stop.search_composition_ops=" << g_compose_count << '\n';
    emit_interval("external_stop.stop_s", stop);
    g_compose_count = 0;
    const Applied safe = apply(power(base, count), x0);
    emit_interval("external_stop.elapsed_safe_s", safe.y[0]);
    g_compose_count = 0;
    const Applied next = apply(power(base, count + 1), x0);
    emit_interval("external_stop.elapsed_next_s", next.y[0]);
    require(safe.y[0].hi <= stop.lo && next.y[0].lo > stop.hi,
            "external stop strict predecessor/successor inequalities failed");

    std::cout << "state_dimension=1\n"
              << "ledger_count=" << kLedgerCount << '\n'
              << "event_template_steps_per_segment=" << template_steps << '\n'
              << "runtime_expanded_events=0\n"
              << "tail_half_bound=" << tail << '\n'
              << "long_double_digits=" << std::numeric_limits<long double>::digits << '\n'
              << "status=PASS\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "FAIL: " << e.what() << '\n';
    return 2;
  }
}
