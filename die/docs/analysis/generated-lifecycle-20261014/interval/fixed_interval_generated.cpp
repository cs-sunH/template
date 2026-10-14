// Fixed-width interval lowering for one generated guarded affine return.
// This is a scratch evaluator, not production certification code.
#include <algorithm>
#include <array>
#include <cfenv>
#include <cmath>
#include <cstdint>
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
    if (text.empty()) fail("empty integer in rational descriptor");
    UInt256 value;
    for (char c : text) {
      if (c < '0' || c > '9') fail("invalid unsigned integer in descriptor");
      if (!value.mul10_add(static_cast<unsigned>(c - '0')))
        fail("descriptor integer exceeds fixed 256-bit capacity");
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
    for (std::size_t i = kLimbs; i-- > 0;)
      acc = add(mul(acc, base),
                Interval::point(static_cast<long double>(word[i])));
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
  const Interval q = div(n.to_interval(), d.to_interval());
  return negative ? neg(q) : q;
}

using Fields = std::map<std::string, std::string>;
Fields read_fields(const std::string& path) {
  std::ifstream in(path);
  if (!in) fail("cannot open generated affine descriptor: " + path);
  Fields fields;
  std::string line;
  std::size_t line_no = 0;
  while (std::getline(in, line)) {
    ++line_no;
    if (line.empty()) continue;
    const auto equal = line.find('=');
    if (equal == std::string::npos || equal == 0)
      fail("malformed descriptor line " + std::to_string(line_no));
    if (!fields.emplace(line.substr(0, equal), line.substr(equal + 1)).second)
      fail("duplicate descriptor field: " + line.substr(0, equal));
  }
  return fields;
}
const std::string& need(const Fields& f, const std::string& key) {
  const auto it = f.find(key);
  if (it == f.end()) fail("missing descriptor field: " + key);
  return it->second;
}
std::uint64_t parse_u64(const std::string& s) {
  if (s.empty()) fail("empty unsigned metadata");
  std::uint64_t value = 0;
  for (char c : s) {
    if (c < '0' || c > '9') fail("invalid unsigned metadata");
    const unsigned digit = static_cast<unsigned>(c - '0');
    if (value > (std::numeric_limits<std::uint64_t>::max() - digit) / 10)
      fail("unsigned metadata exceeds uint64 range");
    value = value * 10 + digit;
  }
  return value;
}
std::uint64_t checked_add(std::uint64_t x, std::uint64_t y,
                          const std::string& what) {
  if (x > std::numeric_limits<std::uint64_t>::max() - y)
    fail("uint64 addition overflow while checking " + what);
  return x + y;
}
std::uint64_t checked_mul(std::uint64_t x, std::uint64_t y,
                          const std::string& what) {
  if (y != 0 && x > std::numeric_limits<std::uint64_t>::max() / y)
    fail("uint64 multiplication overflow while checking " + what);
  return x * y;
}
void require(bool condition, const std::string& message) {
  if (!condition) fail(message);
}

struct Transform {
  Interval a;
  Interval b;
  std::vector<Interval> c;
  std::vector<Interval> d;
  Transform(Interval aa, Interval bb, std::size_t dimensions)
      : a(aa), b(bb), c(dimensions, Interval::point(0.0L)),
        d(dimensions, Interval::point(0.0L)) {}
};
std::uint64_t g_compose_count = 0;
Transform compose(const Transform& first, const Transform& second) {
  ++g_compose_count;
  require(first.c.size() == second.c.size(), "ledger dimensions do not match");
  const std::size_t n = first.c.size();
  Transform out(mul(second.a, first.a),
                add(mul(second.a, first.b), second.b), n);
  for (std::size_t i = 0; i < n; ++i) {
    out.c[i] = add(first.c[i], mul(second.c[i], first.a));
    out.d[i] = add(add(first.d[i], mul(second.c[i], first.b)), second.d[i]);
  }
  return out;
}
Transform identity(std::size_t dimensions) {
  return Transform(Interval::point(1.0L), Interval::point(0.0L), dimensions);
}
Transform power(Transform base, std::uint64_t count) {
  Transform result = identity(base.c.size());
  while (count != 0) {
    if (count & 1U) result = compose(result, base);
    count >>= 1;
    if (count != 0) base = compose(base, base);
  }
  return result;
}
struct Applied {
  Interval x;
  std::vector<Interval> y;
};
Applied apply(const Transform& t, const Interval& x) {
  Applied out{add(mul(t.a, x), t.b), {}};
  out.y.reserve(t.c.size());
  for (std::size_t i = 0; i < t.c.size(); ++i)
    out.y.push_back(add(mul(t.c[i], x), t.d[i]));
  return out;
}

std::string decimal_outward(long double x, bool lower) {
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

struct OwnerLife {
  std::uint64_t next_load = 0;
  std::uint64_t retired = 0;
  std::vector<std::string> active_stage;
  std::vector<std::uint64_t> active_wave;
  std::vector<std::uint64_t> loaded;
  std::vector<std::uint64_t> computed;
  std::uint64_t max_reference() const {
    std::uint64_t out = next_load;
    for (auto x : active_wave) out = std::max(out, x);
    for (auto x : loaded) out = std::max(out, x);
    for (auto x : computed) out = std::max(out, x);
    return out;
  }
  std::uint64_t reference_slots() const {
    return 1 + active_wave.size() + loaded.size() + computed.size();
  }
};
OwnerLife read_life(const Fields& f, const std::string& phase,
                    std::uint64_t owner) {
  const std::string root = "lifecycle." + phase + "." +
                           std::to_string(owner) + ".";
  OwnerLife life;
  life.next_load = parse_u64(need(f, root + "next_load"));
  life.retired = parse_u64(need(f, root + "retired"));
  const std::uint64_t active_count =
      parse_u64(need(f, root + "active_count"));
  const std::uint64_t loaded_count =
      parse_u64(need(f, root + "loaded_count"));
  const std::uint64_t computed_count =
      parse_u64(need(f, root + "computed_count"));
  require(active_count <= 256 && loaded_count <= 256 && computed_count <= 256,
          "lifecycle slot count exceeds bounded descriptor cap");
  for (std::uint64_t i = 0; i < active_count; ++i) {
    const std::string item = root + "active." + std::to_string(i) + ".";
    life.active_stage.push_back(need(f, item + "stage"));
    life.active_wave.push_back(parse_u64(need(f, item + "wave")));
  }
  for (std::uint64_t i = 0; i < loaded_count; ++i)
    life.loaded.push_back(parse_u64(
        need(f, root + "loaded." + std::to_string(i) + ".wave")));
  for (std::uint64_t i = 0; i < computed_count; ++i)
    life.computed.push_back(parse_u64(
        need(f, root + "computed." + std::to_string(i) + ".wave")));
  return life;
}
std::uint64_t lifecycle_bound(const std::vector<OwnerLife>& life,
                              const std::vector<std::uint64_t>& shifts,
                              std::uint64_t first_nonfull,
                              const std::string& label) {
  require(life.size() == shifts.size(), "lifecycle/shift owner counts differ");
  std::uint64_t bound = std::numeric_limits<std::uint64_t>::max();
  for (std::size_t i = 0; i < life.size(); ++i) {
    require(shifts[i] > 0, "zero lifecycle translation");
    const std::uint64_t max_wave = life[i].max_reference();
    require(first_nonfull > max_wave,
            label + " already references the first non-full wave");
    bound = std::min(bound, (first_nonfull - 1 - max_wave) / shifts[i]);
  }
  return bound;
}
void validate_lifecycle_shift(const std::vector<OwnerLife>& initial,
                              const std::vector<OwnerLife>& ending,
                              const std::vector<std::uint64_t>& shifts,
                              const std::vector<std::uint64_t>& retired_outputs) {
  require(initial.size() == ending.size() && initial.size() == shifts.size() &&
              initial.size() == retired_outputs.size(),
          "lifecycle arrays have inconsistent owner counts");
  for (std::size_t i = 0; i < initial.size(); ++i) {
    const auto& before = initial[i];
    const auto& after = ending[i];
    const auto shift = shifts[i];
    require(after.next_load == checked_add(before.next_load, shift, "next_load"),
            "next_load does not translate by K for owner " + std::to_string(i));
    require(after.retired == checked_add(before.retired, retired_outputs[i],
                                         "retired cursor"),
            "retired cursor does not match generated return");
    require(before.active_stage == after.active_stage &&
                before.active_wave.size() == after.active_wave.size(),
            "active lifecycle shape changed across return");
    require(before.loaded.size() == after.loaded.size() &&
                before.computed.size() == after.computed.size(),
            "loaded/computed lifecycle shape changed across return");
    for (std::size_t j = 0; j < before.active_wave.size(); ++j)
      require(after.active_wave[j] ==
                  checked_add(before.active_wave[j], shift, "active wave"),
              "active wave does not translate by K");
    for (std::size_t j = 0; j < before.loaded.size(); ++j)
      require(after.loaded[j] ==
                  checked_add(before.loaded[j], shift, "loaded wave"),
              "loaded wave does not translate by K");
    for (std::size_t j = 0; j < before.computed.size(); ++j)
      require(after.computed[j] ==
                  checked_add(before.computed[j], shift, "computed wave"),
              "computed wave does not translate by K");
  }
}

std::pair<std::uint64_t, Transform> safe_count(
    const Transform& base, const Interval& x0, const Interval& stop,
    std::uint64_t upper, std::size_t time_index) {
  std::vector<Transform> powers;
  Transform current = base;
  std::uint64_t bit_value = 1;
  for (;;) {
    powers.push_back(current);
    if (bit_value > upper / 2) break;
    current = compose(current, current);
    bit_value *= 2;
  }
  std::uint64_t count = 0;
  Transform accumulated = identity(base.c.size());
  for (std::size_t bit = powers.size(); bit-- > 0;) {
    const std::uint64_t step = std::uint64_t(1) << bit;
    if (step > upper - count) continue;
    Transform candidate = compose(accumulated, powers[bit]);
    const Interval elapsed = apply(candidate, x0).y[time_index];
    if (elapsed.hi <= stop.lo) {
      count += step;
      accumulated = std::move(candidate);
    } else if (elapsed.lo > stop.hi) {
      // This prefix is too late; smaller binary steps can still be safe.
    } else {
      fail("external stop overlaps an elapsed-time interval; refusing ambiguity");
    }
  }
  const Interval elapsed = apply(accumulated, x0).y[time_index];
  require(elapsed.hi <= stop.lo, "computed external-stop prefix is not safe");
  return {count, std::move(accumulated)};
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc != 2) fail("usage: fixed_interval_generated DESCRIPTOR.affine");
    require(std::numeric_limits<long double>::is_iec559 &&
                std::numeric_limits<long double>::radix == 2 &&
                std::numeric_limits<long double>::digits == 64,
            "requires IEC 60559 x87 long double with a 64-bit significand");
    require(std::fesetround(FE_TONEAREST) == 0, "cannot set FE_TONEAREST");

    const Fields f = read_fields(argv[1]);
    require(need(f, "version") == "1", "unsupported descriptor version");
    require(parse_u64(need(f, "state_dimension")) == 1,
            "this lowering requires one proven affine state coordinate");
    const std::uint64_t ledger_count = parse_u64(need(f, "ledger_count"));
    require(ledger_count > 0 && ledger_count <= 256,
            "ledger dimension outside the bounded descriptor cap");
    const std::uint64_t template_steps =
        parse_u64(need(f, "event_template_steps"));
    const std::uint64_t guard_count = parse_u64(need(f, "guard_count"));
    require(template_steps > 0 && template_steps <= 256,
            "event template exceeds the 256-step cap");
    require(guard_count > 0, "guarded return has no guard records");
    const std::uint64_t owner_count =
        parse_u64(need(f, "lifecycle_owner_count"));
    require(owner_count > 0 && owner_count <= 64,
            "lifecycle owner count outside bounded descriptor cap");
    const std::uint64_t first_nonfull =
        parse_u64(need(f, "first_nonfull_wave"));
    std::vector<std::uint64_t> shifts, retired_outputs;
    std::vector<OwnerLife> initial, ending;
    for (std::uint64_t owner = 0; owner < owner_count; ++owner) {
      const std::string key = std::to_string(owner);
      shifts.push_back(parse_u64(need(f, "owner." + key + ".shift")));
      retired_outputs.push_back(
          parse_u64(need(f, "owner." + key + ".retired_output")));
      initial.push_back(read_life(f, "initial", owner));
      ending.push_back(read_life(f, "end", owner));
    }
    validate_lifecycle_shift(initial, ending, shifts, retired_outputs);
    const std::uint64_t tail =
        lifecycle_bound(initial, shifts, first_nonfull, "initial lifecycle");
    const std::uint64_t additional =
        lifecycle_bound(ending, shifts, first_nonfull, "end lifecycle");
    require(additional == parse_u64(need(f, "declared_additional_returns")),
            "derived end-state lookahead disagrees with discovery metadata");
    require(tail < std::numeric_limits<std::uint64_t>::max(),
            "tail bound has no representable successor");
    bool successor_crosses = false;
    for (std::size_t i = 0; i < initial.size(); ++i) {
      const std::uint64_t max_wave = initial[i].max_reference();
      const auto nplus = checked_add(tail, 1, "tail successor");
      const auto wave = checked_add(
          max_wave, checked_mul(nplus, shifts[i], "tail successor wave"),
          "tail successor wave");
      if (wave >= first_nonfull) successor_crosses = true;
    }
    require(successor_crosses, "derived initial tail bound is not maximal");

    const std::uint64_t ref_slots_initial = [&]() {
      std::uint64_t n = 0;
      for (const auto& life : initial)
        n = checked_add(n, life.reference_slots(), "reference slot count");
      return n;
    }();
    const std::uint64_t ref_slots_end = [&]() {
      std::uint64_t n = 0;
      for (const auto& life : ending)
        n = checked_add(n, life.reference_slots(), "reference slot count");
      return n;
    }();
    require(ref_slots_initial == ref_slots_end,
            "reference slot count changed across generated return");

    Interval a = rational_interval(need(f, "a"));
    Interval b = rational_interval(need(f, "b"));
    Interval x0 = rational_interval(need(f, "x0"));
    Interval domain_lo = rational_interval(need(f, "domain_lo"));
    Interval domain_hi = rational_interval(need(f, "domain_hi"));
    Interval expected_fixed = rational_interval(need(f, "fixed"));
    require(need(f, "domain_lo_strict") == "1" &&
                need(f, "domain_hi_strict") == "1",
            "guard-domain endpoints are not marked strict");
    require(a.lo > 0.0L && a.hi < 1.0L,
            "state map is not a strict positive contraction");
    const Interval fixed = div(b, sub(Interval::point(1.0L), a));
    require(fixed.lo <= expected_fixed.hi && fixed.hi >= expected_fixed.lo,
            "calculated fixed point does not overlap exact descriptor value");
    require(strictly_inside(expected_fixed, domain_lo, domain_hi) &&
                strictly_inside(x0, domain_lo, domain_hi),
            "initial/fixed-point enclosure is not strictly within guard domain");
    const Interval orbit(std::min(expected_fixed.lo, x0.lo),
                         std::max(expected_fixed.hi, x0.hi));
    require(strictly_inside(orbit, domain_lo, domain_hi),
            "contraction-orbit hull is not strictly within guard domain");
    const Interval orbit_image = add(mul(a, orbit), b);
    require(orbit_image.lo >= domain_lo.hi && orbit_image.hi <= domain_hi.lo,
            "one-step orbit image escaped strict guard domain");

    std::vector<std::string> ledger_names;
    Transform base(a, b, static_cast<std::size_t>(ledger_count));
    std::size_t time_index = static_cast<std::size_t>(ledger_count);
    for (std::size_t i = 0; i < ledger_count; ++i) {
      const std::string prefix = "ledger." + std::to_string(i) + ".";
      const std::string name = need(f, prefix + "name");
      require(std::find(ledger_names.begin(), ledger_names.end(), name) ==
                  ledger_names.end(),
              "duplicate ledger name");
      ledger_names.push_back(name);
      base.c[i] = rational_interval(need(f, prefix + "slope"));
      base.d[i] = rational_interval(need(f, prefix + "intercept"));
      if (name == need(f, "time_ledger_name")) time_index = i;
      // Ledger slopes may be negative and increments may be exactly zero.
      // Only elapsed time needs a strictly positive increment for stop search.
    }
    require(time_index < ledger_count, "time ledger is absent");
    const Interval time_increment =
        add(mul(base.c[time_index], orbit), base.d[time_index]);
    require(time_increment.lo > 0.0L,
            "elapsed-time increment is not provably positive on the orbit");

    const std::uint64_t stop_ns = parse_u64(need(f, "external_stop_ns"));
    const Interval stop = div(
        rational_interval(std::to_string(stop_ns)),
        rational_interval("1000000000"));
    const std::array<std::uint64_t, 6> fixed_samples{
        1, 10, 100, tail, 1000000, 1000000000000ULL};
    std::vector<std::uint64_t> samples;
    for (auto n : fixed_samples)
      if (std::find(samples.begin(), samples.end(), n) == samples.end())
        samples.push_back(n);
    for (std::uint64_t n : samples) {
      g_compose_count = 0;
      const Transform powered = power(base, n);
      const Applied value = apply(powered, x0);
      const bool bounded = n <= tail;
      if (bounded)
        require(strictly_inside(value.x, domain_lo, domain_hi),
                "bounded repeated state escaped strict guard domain");
      if (n == tail) {
        const Interval width = sub(Interval::point(value.y[time_index].hi),
                                   Interval::point(value.y[time_index].lo));
        const Interval one_ns = rational_interval("1/1000000000");
        require(width.hi < one_ns.lo,
                "tail elapsed-time enclosure is at least one nanosecond wide");
      }
      std::cout << "case." << n << ".scope="
                << (bounded ? "lifecycle-bounded" : "numeric-only") << '\n'
                << "case." << n << ".composition_ops=" << g_compose_count
                << '\n';
      emit_interval("case." + std::to_string(n) + ".state", value.x);
      for (std::size_t i = 0; i < ledger_count; ++i)
        emit_interval("case." + std::to_string(n) + ".ledger." +
                          ledger_names[i], value.y[i]);
      if (n == tail)
        emit_interval("tail.elapsed_width_s",
                      sub(Interval::point(value.y[time_index].hi),
                          Interval::point(value.y[time_index].lo)));
    }

    g_compose_count = 0;
    auto stop_result = safe_count(base, x0, stop, tail, time_index);
    const std::uint64_t count = stop_result.first;
    Transform safe_transform = std::move(stop_result.second);
    const Applied safe = apply(safe_transform, x0);
    const Transform next_transform = compose(safe_transform, base);
    const Applied next = apply(next_transform, x0);
    require(safe.y[time_index].hi <= stop.lo &&
                next.y[time_index].lo > stop.hi,
            "external stop lacks strict safe/next-segment inequalities");
    const Interval partial = sub(stop, safe.y[time_index]);
    std::cout << "external_stop.count=" << count << '\n'
              << "external_stop.search_composition_ops=" << g_compose_count
              << '\n'
              << "external_stop.partial_handler=ordinary_integrator\n";
    emit_interval("external_stop.stop_s", stop);
    emit_interval("external_stop.elapsed_safe_s", safe.y[time_index]);
    emit_interval("external_stop.elapsed_next_s", next.y[time_index]);
    emit_interval("external_stop.state_safe", safe.x);
    emit_interval("external_stop.partial_residual_s", partial);

    std::uint64_t max_initial = 0;
    std::uint64_t max_end = 0;
    for (std::size_t i = 0; i < initial.size(); ++i) {
      max_initial = std::max(max_initial, initial[i].max_reference());
      max_end = std::max(max_end, ending[i].max_reference());
    }
    std::cout << "state_dimension=1\n"
              << "ledger_count=" << ledger_count << '\n'
              << "event_template_steps_per_return=" << template_steps << '\n'
              << "guard_count=" << guard_count << '\n'
              << "lifecycle_owner_count=" << owner_count << '\n'
              << "initial_lifecycle_reference_slots=" << ref_slots_initial << '\n'
              << "end_lifecycle_reference_slots=" << ref_slots_end << '\n'
              << "initial_max_reference_wave=" << max_initial << '\n'
              << "end_max_reference_wave=" << max_end << '\n'
              << "first_nonfull_wave=" << first_nonfull << '\n'
              << "initial_lifecycle_tail_bound=" << tail << '\n'
              << "tail_successor_count=" << checked_add(tail, 1, "tail successor")
              << '\n'
              << "end_additional_complete_returns=" << additional << '\n'
              << "runtime_expanded_events=0\n"
              << "long_double_digits=" << std::numeric_limits<long double>::digits
              << '\n'
              << "status=PASS\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "FAIL: " << e.what() << '\n';
    return 2;
  }
}
