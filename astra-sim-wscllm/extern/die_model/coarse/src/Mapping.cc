#include "coarse_die/Plan.hh"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <numeric>
#include <set>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace coarse_die::detail {
namespace {

using u64 = std::uint64_t;
using u128 = unsigned __int128;

[[noreturn]] void fail(const std::string& message) {
  throw std::invalid_argument(message);
}

u128 add128(u128 a, u128 b, const std::string& field) {
  constexpr u128 max = ~u128{0};
  if (b > max - a) fail("overflow: " + field);
  return a + b;
}

u128 mul128(u128 a, u128 b, const std::string& field) {
  constexpr u128 max = ~u128{0};
  if (a && b > max / a) fail("overflow: " + field);
  return a * b;
}

u64 as_u64(u128 value, const std::string& field) {
  if (value > std::numeric_limits<u64>::max())
    fail("overflow: " + field + " exceeds uint64 output range");
  return static_cast<u64>(value);
}

double as_double(long double value, const std::string& field) {
  if (!std::isfinite(value) || value < 0.0L ||
      value > static_cast<long double>(std::numeric_limits<double>::max()))
    fail("overflow: " + field + " cannot be represented as a finite double");
  return static_cast<double>(value);
}

double as_signed_double(long double value, const std::string& field) {
  const long double limit =
      static_cast<long double>(std::numeric_limits<double>::max());
  if (!std::isfinite(value) || value < -limit || value > limit)
    fail("overflow: " + field + " cannot be represented as a finite double");
  return static_cast<double>(value);
}

u64 getu(const nlohmann::json& j, const char* key, u64 fallback = 0) {
  auto it = j.find(key);
  if (it == j.end()) return fallback;
  if (it->is_number_unsigned()) return it->get<u64>();
  if (it->is_number_integer()) {
    const auto value = it->get<std::int64_t>();
    if (value >= 0) return static_cast<u64>(value);
  }
  fail(std::string("task: ") + key + " must be a non-negative integer");
}

double getd(const nlohmann::json& j, const char* key, double fallback) {
  auto it = j.find(key);
  if (it == j.end()) return fallback;
  if (!it->is_number()) fail(std::string("invalid_hardware: ") + key + " must be numeric");
  const double value = it->get<double>();
  if (!std::isfinite(value)) fail(std::string("invalid_hardware: ") + key + " must be finite");
  return value;
}

std::string gets(const nlohmann::json& j, const char* key,
                 const std::string& fallback = {}) {
  auto it = j.find(key);
  if (it == j.end()) return fallback;
  if (!it->is_string()) fail(std::string("task: ") + key + " must be a string");
  return it->get<std::string>();
}

u64 ceildiv(u64 a, u64 b) {
  if (!b) fail("mapping: zero tile dimension");
  return a / b + (a % b != 0);
}

u64 bpe(const std::string& dtype) {
  if (dtype == "int4") return 0;
  if (dtype == "bf16") return 2;
  if (dtype == "fp32" || dtype == "int32") return 4;
  fail("unsupported_dtype: " + dtype);
}

u128 bytes(u128 elements, const std::string& dtype) {
  if (dtype == "int4") return elements / 2 + (elements % 2 != 0);
  return mul128(elements, bpe(dtype), "tensor bytes");
}

u128 tile_bytes(u64 rows, u64 cols, const std::string& dtype) {
  return bytes(mul128(rows, cols, "tile elements"), dtype);
}

u128 tile_grid_bytes(u64 rows, u64 tr, u64 cols, u64 tc,
                     const std::string& dtype) {
  if (!tr || !tc) fail("mapping: zero tile size");
  const u64 rf = rows / tr, rr = rows % tr;
  const u64 cf = cols / tc, cr = cols % tc;
  u128 result = 0;
  if (rf && cf) result = add128(result, mul128(mul128(rf, cf, "tile count"), tile_bytes(tr, tc, dtype), "tile payload"), "tile payload");
  if (rf && cr) result = add128(result, mul128(rf, tile_bytes(tr, cr, dtype), "tail payload"), "tile payload");
  if (rr && cf) result = add128(result, mul128(cf, tile_bytes(rr, tc, dtype), "tail payload"), "tile payload");
  if (rr && cr) result = add128(result, tile_bytes(rr, cr, dtype), "tile payload");
  return result;
}

u64 shape_u(const nlohmann::json& task, const char* key, u64 fallback = 0) {
  const auto& shape = task.at("tensor_shapes");
  auto it = shape.find(key);
  if (it == shape.end()) return fallback;
  if (it->is_number_unsigned()) return it->get<u64>();
  if (it->is_number_integer()) {
    const auto value = it->get<std::int64_t>();
    if (value >= 0) return static_cast<u64>(value);
  }
  fail(std::string("task.tensor_shapes.") + key + " must be a non-negative integer");
}

std::string dtype(const nlohmann::json& task, const char* role,
                  const std::string& fallback = "bf16") {
  const auto it = task.find("storage_dtype");
  if (it == task.end() || !it->is_object()) return fallback;
  auto value = it->find(role);
  if (value == it->end()) return fallback;
  if (!value->is_string()) fail(std::string("task.storage_dtype.") + role + " must be a string");
  const std::string result = value->get<std::string>();
  (void)bpe(result);
  return result;
}

double perf(const Hardware& hw, const char* key, double fallback) {
  const auto it = hw.input.find("performance_profile");
  if (it == hw.input.end() || !it->is_object()) return fallback;
  return getd(*it, key, fallback);
}

u64 du(const Hardware& hw, const char* group, const char* key, u64 fallback = 0) {
  auto g = hw.derived.find(group);
  if (g == hw.derived.end() || !g->is_object()) return fallback;
  return getu(*g, key, fallback);
}

double dd(const Hardware& hw, const char* group, const char* key,
          double fallback = 0.0) {
  auto g = hw.derived.find(group);
  if (g == hw.derived.end() || !g->is_object()) return fallback;
  return getd(*g, key, fallback);
}

struct Span {
  u64 q = 0, kv = 0, count = 1;
  u64 qs = 0, ks = 0, kv_stride = 0, qs_stride = 0;
};

std::vector<Span> spans_of(const nlohmann::json& task) {
  std::vector<Span> result;
  auto it = task.find("spans");
  if (it == task.end()) return result;
  if (!it->is_array()) fail("task.spans must be an array");
  for (const auto& j : *it) {
    Span s{getu(j, "q_tokens"), getu(j, "kv_tokens"), getu(j, "count", 1),
           getu(j, "q_start_pos"), getu(j, "kv_start_pos"),
           getu(j, "kv_stride"), getu(j, "q_start_stride")};
    if (!s.q || !s.kv || !s.count) fail("task.spans: q_tokens, kv_tokens and count must be positive");
    if (s.count > 1 && s.q != 1 &&
        !(s.kv >= s.q && s.ks == 0 && s.qs == s.kv - s.q && s.qs_stride == s.kv_stride))
      fail("unsupported_mapping: compact attention span progression is not a supported causal run");
    result.push_back(s);
  }
  return result;
}

u128 valid_pairs_one(const Span& s) {
  const u128 q0 = s.qs, k0 = s.ks, q = s.q, kv = s.kv;
  if (q0 + q - 1 < k0) return 0;
  const u128 first = k0 > q0 ? k0 - q0 : 0;
  if (first >= q) return 0;
  const u128 terms = q - first;
  const u128 reach = q0 + first - k0 + 1;
  const u128 rising = reach < kv ? std::min(terms, kv - reach) : 0;
  u128 total = mul128(rising, reach, "causal valid pairs");
  if (rising > 1) total = add128(total, mul128(rising, rising - 1, "causal valid pairs") / 2, "causal valid pairs");
  return add128(total, mul128(terms - rising, kv, "causal valid pairs"), "causal valid pairs");
}

u128 valid_pairs(const Span& s, bool causal = true) {
  if (!causal) {
    const u128 sum_kv=add128(mul128(s.count,s.kv,"attention KV span sum"),
        mul128(s.kv_stride,mul128(s.count,s.count-1,"attention KV span sum")/2,"attention KV span sum"),
        "attention KV span sum");
    return mul128(s.q,sum_kv,"noncausal score pairs");
  }
  if (s.count == 1) return valid_pairs_one(s);
  if(s.kv<s.q || s.ks!=0 || s.qs!=s.kv-s.q || s.qs_stride!=s.kv_stride)
    fail("unsupported_mapping: causal compact span requires q_start_pos=kv_len-q_len and kv_start_pos=0");
  if (s.q == 1) return add128(mul128(s.count, s.kv, "causal run"),
      mul128(s.kv_stride, mul128(s.count, s.count - 1, "causal run") / 2, "causal run"), "causal run");
  const u128 q = s.q, n = s.count;
  const u128 first = mul128(q, s.kv - q + 1, "causal run") + mul128(q, q - 1, "causal run") / 2;
  return add128(mul128(n, first, "causal run"),
      mul128(mul128(n, n - 1, "causal run") / 2, mul128(q, s.kv_stride, "causal run"), "causal run"), "causal run");
}

u128 floor_sum(u128 n, u128 mod, u128 a, u128 b) {
  if (!mod) fail("overflow: floor_sum zero modulus");
  u128 answer = 0;
  for (;;) {
    if (a >= mod) {
      const u128 q = a / mod;
      answer = add128(answer, mul128(mul128(n, n - (n != 0), "floor_sum"), q, "floor_sum") / 2, "floor_sum");
      a %= mod;
    }
    if (b >= mod) {
      answer = add128(answer, mul128(n, b / mod, "floor_sum"), "floor_sum");
      b %= mod;
    }
    const u128 top = add128(mul128(a, n, "floor_sum"), b, "floor_sum");
    if (top < mod) return answer;
    n = top / mod;
    b = top % mod;
    std::swap(mod, a);
  }
}

// Sum the key prefix at each query tile. The residue loop is bounded by tk/gcd(tq,tk),
// never by the number of query tiles.
std::pair<u128, u128> prefix_slots_one(const Span& s, u64 tq, u64 tk) {
  if (s.count != 1) {
    if (s.kv < s.q || s.ks != 0 || s.qs != s.kv - s.q || s.qs_stride != s.kv_stride)
      fail("unsupported_mapping: compact attention run must be a causal kv-length progression");
    if (!s.kv_stride) {
      Span one=s; one.count=1;
      const auto part=prefix_slots_one(one,tq,tk);
      return {mul128(part.first,s.count,"attention fixed run slots"),
              mul128(part.second,s.count,"attention fixed run prefixes")};
    }
    const u64 qfull=s.q/tq, qtail=s.q%tq;
    const u64 period=tk/std::gcd(s.kv_stride,tk);
    const u128 step=mul128(period,s.kv_stride,"attention run residue stride");
    u128 all_slots=0, all_prefixes=0;
    for(u64 residue=0;residue<std::min(period,s.count);++residue){
      const u64 count=1+(s.count-1-residue)/period;
      const u128 first_kv=add128(s.kv,mul128(residue,s.kv_stride,"attention run kv"),"attention run kv");
      const u64 rem=static_cast<u64>(first_kv%tk);
      const u128 sum_kv=add128(mul128(count,first_kv,"attention run kv sum"),
          mul128(step,mul128(count,count-1,"attention run kv sum")/2,"attention run kv sum"),"attention run kv sum");
      u64 u=0;
      if(qfull && s.q>=tq && s.q-tq>=rem)
        u=std::min<u64>(qfull,(s.q-tq-rem)/tq+1);
      u128 sum_s=0;
      if(u){
        const u128 delta_last=static_cast<u128>(s.q)-tq-static_cast<u128>(u-1)*tq;
        sum_s=add128(mul128(u,rem,"attention residue correction"),
            mul128(tk,floor_sum(u,tk,tq,delta_last-rem),"attention residue correction"),
            "attention residue correction");
      }
      const u128 full_prefix=mul128(qfull,sum_kv,"attention run prefixes")-
          mul128(count,sum_s,"attention run causal correction");
      all_prefixes=add128(all_prefixes,full_prefix,"attention run prefixes");
      all_slots=add128(all_slots,mul128(tq,full_prefix,"attention run slots"),"attention run slots");
      if(qtail){
        all_prefixes=add128(all_prefixes,sum_kv,"attention tail prefixes");
        all_slots=add128(all_slots,mul128(qtail,sum_kv,"attention tail slots"),"attention tail slots");
      }
    }
    return {all_slots,all_prefixes};
  }
  const u64 full = s.q / tq, tail = s.q % tq;
  const u64 period = tk / std::gcd(tq, tk);
  const u128 stride = mul128(period, tq, "attention residue stride");
  const __int128 base = static_cast<__int128>(s.qs) - static_cast<__int128>(s.ks);
  u128 prefixes = 0;
  for (u64 r = 0; r < std::min(full, period); ++r) {
    const u64 n = 1 + (full - 1 - r) / period;
    const __int128 first_reach = base + static_cast<__int128>(r + 1) * tq;
    u64 first_positive = 0;
    if (first_reach <= 0) {
      const __int128 numerator = 1 - first_reach;
      first_positive = static_cast<u64>((numerator + static_cast<__int128>(stride) - 1) / static_cast<__int128>(stride));
    }
    if (first_positive >= n) continue;
    const u64 positive_n = n - first_positive;
    const __int128 positive_reach = first_reach + static_cast<__int128>(first_positive) * static_cast<__int128>(stride);
    const u128 first_len = std::min<u128>(s.kv, ((static_cast<u128>(positive_reach) + tk - 1) / tk) * tk);
    const u128 delta = stride;
    u64 unsaturated = positive_n;
    if (first_len >= s.kv) unsaturated = 0;
    else unsaturated = std::min<u64>(positive_n,
         static_cast<u64>((s.kv - first_len + delta - 1) / delta));
    u128 sum = 0;
    if (unsaturated) sum = add128(mul128(unsaturated, first_len, "attention prefix"),
        mul128(delta, mul128(unsaturated, unsaturated - 1, "attention prefix") / 2, "attention prefix"), "attention prefix");
    sum = add128(sum, mul128(positive_n - unsaturated, s.kv, "attention prefix"), "attention prefix");
    prefixes = add128(prefixes, sum, "attention prefix");
  }
  if (tail) {
    const __int128 reach = base + s.q;
    if (reach > 0) {
      const u128 len = std::min<u128>(s.kv, ((static_cast<u128>(reach) + tk - 1) / tk) * tk);
      prefixes = add128(prefixes, len, "attention tail prefix");
    }
  }
  u128 slots = mul128(prefixes, tq, "attention score slots");
  if (tail) {
    const __int128 reach = base + s.q;
    if (reach > 0) {
      const u128 len = std::min<u128>(s.kv, ((static_cast<u128>(reach) + tk - 1) / tk) * tk);
      // The final partial query tile was included once at tq above; replace its padding.
      slots -= mul128(tq - tail, len, "attention tail correction");
    }
  }
  return {slots, prefixes};
}

u128 attention_slots(const std::vector<Span>& spans, u64 tq, u64 tk) {
  u128 total = 0;
  for (const auto& s : spans) total = add128(total, prefix_slots_one(s, tq, tk).first, "attention slots");
  return total;
}

struct AxisDist {
  u64 nx = 1, ny = 1, active = 1;
  long double ex = 0, ex2 = 0, ey = 0, ey2 = 0;
  long double xleft = 0, xright = 0, ylow = 0, yhigh = 0;
};

long double sum0(u64 n) { return static_cast<long double>(n) * (n - 1) / 2; }
long double sumsq0(u64 n) { return static_cast<long double>(n) * (n - 1) * (2.0L * n - 1) / 6; }

AxisDist destination(u64 count, u64 nx, u64 ny) {
  if (!nx || !ny) fail("invalid_hardware: core grid must be positive");
  const u64 total = as_u64(mul128(nx, ny, "core grid"), "core grid");
  AxisDist d; d.nx = nx; d.ny = ny; d.active = std::max<u64>(1, std::min(count, total));
  const u64 rows = d.active / nx, rem = d.active % nx;
  d.ex = (static_cast<long double>(rows) * sum0(nx) + sum0(rem)) / d.active;
  d.ex2 = (static_cast<long double>(rows) * sumsq0(nx) + sumsq0(rem)) / d.active;
  d.ey = (static_cast<long double>(nx) * sum0(rows) + static_cast<long double>(rows) * rem) / d.active;
  d.ey2 = (static_cast<long double>(nx) * sumsq0(rows) + static_cast<long double>(rows) * rows * rem) / d.active;
  const u64 cx = nx / 2, cy = ny / 2;
  const u64 full = d.active / nx, tail = d.active % nx;
  const u64 xr = full * (nx - cx) + (tail > cx ? tail - cx : 0);
  u64 yh = 0;
  if (full > cy) yh = (full - cy) * nx;
  if (tail && full >= cy) yh += tail;
  d.xright = static_cast<long double>(xr) / d.active; d.xleft = 1 - d.xright;
  d.yhigh = static_cast<long double>(yh) / d.active; d.ylow = 1 - d.yhigh;
  return d;
}

long double point_abs_axis(const AxisDist& d, u64 point, bool xaxis) {
  const u64 nx = d.nx;
  if (xaxis) {
    const u64 rows = d.active / nx, rem = d.active % nx;
    auto prefix_abs = [point](u64 n) -> long double {
      const u64 left = std::min(n, point + 1);
      const long double l = static_cast<long double>(left) * point - sum0(left);
      const u64 start = std::min(n, point + 1), rn = n - start;
      const long double r = (sum0(n) - sum0(start)) - static_cast<long double>(rn) * point;
      return l + r;
    };
    return (static_cast<long double>(rows) * prefix_abs(nx) + prefix_abs(rem)) / d.active;
  }
  const u64 rows = d.active / nx, rem = d.active % nx;
  const long double e = d.ey;
  (void)rows; (void)rem;
  // E|point-y| over the row-major active prefix, in closed form.
  const u64 left_n = std::min<u64>(d.active / nx, point + 1);
  long double sum = static_cast<long double>(nx) *
      (static_cast<long double>(left_n) * point - sum0(left_n));
  const u64 start = left_n;
  const u64 full_rows = d.active / nx;
  if (full_rows > start) {
    const u64 n = full_rows - start;
    const long double ys = sum0(full_rows) - sum0(start);
    sum += static_cast<long double>(nx) * (ys - static_cast<long double>(n) * point);
  }
  const u64 tail = d.active % nx;
  if (tail) sum += static_cast<long double>(tail) * std::abs(static_cast<long double>(full_rows) - point);
  (void)e;
  return sum / d.active;
}

long double uniform_source_mean_abs(const AxisDist& d, u64 dim, bool xaxis) {
  if (dim <= 1) return 0;
  const long double e = xaxis ? d.ex : d.ey;
  const long double e2 = xaxis ? d.ex2 : d.ey2;
  return (2 * e2 - 2 * (dim - 1) * e + static_cast<long double>(dim) * (dim - 1)) / (2 * dim);
}

struct Topology {
  std::string profile;
  std::array<int, 2> ingress{{-1, -1}};
  std::array<int, 2> row_for_y{{0, 1}};
  std::array<std::vector<std::pair<u64, long double>>, 2> source_x;
  std::array<bool, 2> edge{{false, false}};
  u64 nx = 1, ny = 1;
};

Topology topology(const Hardware& hw, u64 nx, u64 ny) {
  Topology t; t.nx = nx; t.ny = ny;
  const auto pp = hw.input.find("performance_profile");
  if (pp == hw.input.end() || !pp->is_object()) fail("missing_topology_profile: performance_profile absent");
  t.profile = gets(*pp, "timing_topology_profile");
  const auto hbm = hw.derived.find("hbm");
  if (hbm == hw.derived.end() || !hbm->is_object()) fail("invalid_hardware: missing derived.hbm");
  if (t.profile == "distributed_ideal_v2") {
    return t;
  }
  if (t.profile == "edge_single_word_v2") {
    const auto map = hbm->find("row_to_ingress");
    if (map != hbm->end() && map->is_array() && map->size() == 2 && !(*map)[0].is_null()) {
      t.ingress[0] = static_cast<int>((*map)[0].get<u64>());
      t.ingress[1] = static_cast<int>((*map)[1].get<u64>());
    } else {
      const auto arch = hw.input.find("architecture");
      const std::string placement = arch != hw.input.end() && arch->is_object()
          ? gets(*arch, "hbm_placement_mode") : "";
      t.ingress = placement == "two_sides_one_row_each" ? std::array<int,2>{{0,1}} : std::array<int,2>{{0,0}};
    }
    t.edge = {{true, true}};
    return t;
  }
  if (t.profile == "explicit_ingress_v2") {
    const auto ei = pp->find("explicit_ingress");
    if (ei == pp->end() || !ei->is_object()) fail("missing_topology_profile: explicit_ingress object required");
    const auto& e = *ei;
    const auto source = e.find("source");
    if (source == e.end() || !source->is_string() || source->get<std::string>().empty()) fail("invalid_hardware: explicit ingress source required");
    const auto rows = e.find("row_to_ingress"), positions = e.find("source_positions");
    if (rows == e.end() || !rows->is_array() || rows->size() != 2 || positions == e.end() || !positions->is_array() || positions->size() != 2)
      fail("invalid_hardware: explicit ingress mapping requires two rows");
    for (std::size_t r = 0; r < 2; ++r) {
      t.ingress[r] = static_cast<int>((*rows)[r].get<u64>());
      const auto& list = (*positions)[r];
      if (!list.is_array() || list.empty()) fail("invalid_hardware: explicit source position row must be non-empty");
      long double total = 0;
      for (const auto& p : list) {
        if (!p.is_array() || p.size() != 2) fail("invalid_hardware: source position must be [column,weight]");
        const u64 x = p[0].get<u64>(); const double w = p[1].get<double>();
        if (x >= nx || !std::isfinite(w) || w < 0) fail("invalid_hardware: invalid source position");
        t.source_x[r].emplace_back(x, w); total += w;
      }
      if (std::abs(total - 1.0L) > 1e-9L) fail("invalid_hardware: source position weights must sum to one");
      t.edge[r] = true;
    }
    return t;
  }
  fail("missing_topology_profile: expected distributed_ideal_v2, edge_single_word_v2, or explicit_ingress_v2");
}

struct Flow {
  std::array<long double, resource_count> work{};
  std::array<std::array<u128, 2>, 2> hbm{};
  u128 int4 = 0, vec = 0;
  long double compute_s = 0;
  std::array<long double, 2> credit{};
};

struct Metrics {
  u128 source_mops = 0, executed_mops = 0, source_vops = 0, vecops = 0;
  u128 int4ops = 0, int4bytes = 0, conversion_elements = 0;
  u128 hbm_read = 0, hbm_write = 0, logical = 0, workspace = 0;
  u128 sram_peak = 0, sram_read = 0, sram_write = 0, local = 0;
  u128 score_slots = 0, valid_pairs = 0, score_bytes = 0, probability_bytes = 0;
  u128 output_rounds = 0, waves = 0;
  u64 active_cores = 1, tm = 0, tn = 0, tk = 0, th = 0, tf = 0;
  std::string dataflow = "none", template_id = "";
  bool recompute = false;
  std::vector<std::string> assumptions;
};

struct GemmChoice { u64 tm=1, tn=64, tk=64; bool wr=false; };

std::pair<long double,long double> route_moments(const Topology& topo,
                                                 const AxisDist& d,
                                                 u64 row, bool writing) {
  const u64 cutx = topo.nx / 2, cuty = topo.ny / 2;
  long double sx_left = 0.5L, sy_low = 0.5L, sx_abs = 0, sy_abs = 0;
  if (topo.profile == "edge_single_word_v2") {
    sx_left = static_cast<long double>(cutx) / topo.nx;
    sy_low = row == 0 ? 1.0L : 0.0L;
    sx_abs = uniform_source_mean_abs(d, topo.nx, true);
    sy_abs = row == 0 ? d.ey : (topo.ny - 1 - d.ey);
  } else if (topo.profile == "explicit_ingress_v2") {
    long double absx = 0, left = 0;
    if (topo.source_x[row].empty()) fail("invalid_hardware: missing explicit source positions");
    for (const auto& p : topo.source_x[row]) {
      left += p.second * (p.first < cutx ? 1.0L : 0.0L);
      absx += p.second * point_abs_axis(d, p.first, true);
    }
    sx_left = left; sx_abs = absx;
    sy_low = row == 0 ? 1.0L : 0.0L;
    sy_abs = row == 0 ? d.ey : (topo.ny - 1 - d.ey);
  } else {
    sx_left = static_cast<long double>(cutx) / topo.nx;
    sy_low = static_cast<long double>(cuty) / topo.ny;
    sx_abs = uniform_source_mean_abs(d, topo.nx, true);
    sy_abs = uniform_source_mean_abs(d, topo.ny, false);
  }
  const long double dx = writing ? d.xleft * (1 - sx_left) + d.xright * sx_left
                                 : sx_left * d.xright + (1 - sx_left) * d.xleft;
  const long double dy = writing ? d.ylow * (1 - sy_low) + d.yhigh * sy_low
                                 : sy_low * d.yhigh + (1 - sy_low) * d.ylow;
  long double meanx = sx_abs, meany = sy_abs;
  if (writing) {
    if (topo.profile == "edge_single_word_v2") {
      meanx = sx_abs; meany = sy_abs;
    } else if (topo.profile == "explicit_ingress_v2") {
      meanx = sx_abs; meany = sy_abs;
    } else {
      meanx = uniform_source_mean_abs(d, topo.nx, true);
      meany = uniform_source_mean_abs(d, topo.ny, false);
    }
  }
  (void)dx; (void)dy;
  return {meanx, meany};
}

void route_bytes(Flow& flow, const Hardware& hw, const Topology& topo,
                 u64 row, u64 nbytes, u64 active_cores, bool writing) {
  if (!nbytes) return;
  const AxisDist d = destination(active_cores, topo.nx, topo.ny);
  const u64 cx = topo.nx / 2, cy = topo.ny / 2;
  long double source_x_left = 0.5L, source_y_low = 0.5L, meanx = 0, meany = 0;
  if (topo.profile == "edge_single_word_v2") {
    source_x_left = static_cast<long double>(cx) / topo.nx;
    source_y_low = row == 0 ? 1.0L : 0.0L;
    meanx = uniform_source_mean_abs(d, topo.nx, true);
    meany = row == 0 ? d.ey : (topo.ny - 1 - d.ey);
  } else if (topo.profile == "explicit_ingress_v2") {
    source_x_left = 0; meanx = 0;
    for (const auto& p : topo.source_x[row]) {
      if (p.first < cx) source_x_left += p.second;
      meanx += p.second * point_abs_axis(d, p.first, true);
    }
    source_y_low = row == 0 ? 1.0L : 0.0L;
    meany = row == 0 ? d.ey : (topo.ny - 1 - d.ey);
  } else {
    source_x_left = static_cast<long double>(cx) / topo.nx;
    source_y_low = static_cast<long double>(cy) / topo.ny;
    meanx = uniform_source_mean_abs(d, topo.nx, true);
    meany = uniform_source_mean_abs(d, topo.ny, false);
  }
  long double xpos = 0, xneg = 0, ypos = 0, yneg = 0;
  if (!writing) {
    xpos = source_x_left * d.xright; xneg = (1 - source_x_left) * d.xleft;
    ypos = source_y_low * d.yhigh; yneg = (1 - source_y_low) * d.ylow;
  } else {
    xpos = d.xleft * (1 - source_x_left); xneg = d.xright * source_x_left;
    ypos = d.ylow * (1 - source_y_low); yneg = d.yhigh * source_y_low;
  }
  flow.work[XPOS] += nbytes * xpos;
  flow.work[XNEG] += nbytes * xneg;
  flow.work[YPOS] += nbytes * ypos;
  flow.work[YNEG] += nbytes * yneg;
  flow.work[HOP] += nbytes * (meanx + meany);

  const int ingress = topo.ingress[row];
  if (ingress >= 0) {
    const std::size_t resource = ingress == 0 ? INJECT0 : INJECT1;
    const auto& desc = hw.derived.at("resources").at(resource);
    if (desc.at("enabled").get<bool>()) flow.work[resource] += nbytes;
  }
}

void add_hbm(Flow& flow, const Hardware& hw, const Topology& topo,
             u128 total, u64 active, bool writing, double eta) {
  const u64 total64 = as_u64(total, "HBM payload");
  const u64 row0 = total64 / 2 + total64 % 2, row1 = total64 / 2;
  const std::array<u64,2> row_bytes{{row0,row1}};
  for (u64 row = 0; row < 2; ++row) {
    const u64 value = row_bytes[row];
    if (!value) continue;
    const std::size_t dir = writing ? 1 : 0;
    flow.hbm[row][dir] = add128(flow.hbm[row][dir], value, "HBM row bytes");
    flow.work[row == 0 ? HBM0 : HBM1] += static_cast<long double>(value) / eta;
    route_bytes(flow, hw, topo, row, value, active, writing);
  }
}

void add_bytes_work(Flow& flow, std::size_t resource, u128 amount,
                    const std::string& field) {
  flow.work[resource] += static_cast<long double>(as_u64(amount, field));
}

double eta(const Hardware& hw, const char* key) {
  const double value = perf(hw, key, 1.0);
  if (!(value > 0.0) || value > 1.0) fail(std::string("invalid_hardware: ") + key + " must be in (0,1]");
  return value;
}

double compute_time(const Hardware& hw, u128 int4ops, u128 vecops,
                    u64 active) {
  const long double ppe = dd(hw, "core", "P_int4_per_core_ops_per_s");
  const long double pvec = dd(hw, "core", "P_vec_per_core_ops_per_s");
  if (!(ppe > 0) || !(pvec > 0) || !active) fail("invalid_hardware: compute rates must be positive");
  const long double tpe = static_cast<long double>(int4ops) /
      (ppe * active * eta(hw, "eta_pe"));
  const long double tv = static_cast<long double>(vecops) /
      (pvec * active * eta(hw, "eta_vec"));
  return as_double(tpe + tv, "compute seconds");
}

void seal_activity(Activity& a, Flow& f, const Hardware& hw, u64 active) {
  a.hbm_bytes = {};
  for (std::size_t row = 0; row < 2; ++row)
    for (std::size_t dir = 0; dir < 2; ++dir)
      a.hbm_bytes[row][dir] = as_u64(f.hbm[row][dir], "activity HBM bytes");
  a.int4_ops = as_u64(f.int4, "activity INT4 ops");
  a.vec_eqops = as_u64(f.vec, "activity Vec eqops");
  a.private_time_s = as_double(f.compute_s, "activity private time");
  const double local_one_core = active ? hw.rates[LOCAL] *
      (static_cast<double>(active) / std::max<u64>(1, du(hw,"die","core_count",1))) : 0;
  const double sr_one_core = active ? hw.rates[SRAM_READ] *
      (static_cast<double>(active) / std::max<u64>(1, du(hw,"die","core_count",1))) : 0;
  const double sw_one_core = active ? hw.rates[SRAM_WRITE] *
      (static_cast<double>(active) / std::max<u64>(1, du(hw,"die","core_count",1))) : 0;
  long double private_time = a.private_time_s;
  if (local_one_core > 0) private_time = std::max(private_time, f.work[LOCAL] / local_one_core);
  if (sr_one_core > 0) private_time = std::max(private_time, f.work[SRAM_READ] / sr_one_core);
  if (sw_one_core > 0) private_time = std::max(private_time, f.work[SRAM_WRITE] / sw_one_core);
  a.private_time_s = as_double(private_time, "activity private time");
  for (std::size_t r = 0; r < resource_count; ++r) {
    if (r == COMPUTE_GATE) continue;
    a.work[r] = as_double(f.work[r], "activity resource work");
  }
  a.work[COMPUTE_GATE] = as_double(f.compute_s, "compute gate work");
  a.canonical_q = canonical_basis(a);
  a.credit_delta = {{as_signed_double(f.credit[0], "input credit delta"),
                     as_signed_double(f.credit[1], "output credit delta")}};
}

Flow add_flow(const Flow& a, const Flow& b) {
  Flow r;
  for (std::size_t i=0;i<resource_count;++i) r.work[i]=a.work[i]+b.work[i];
  for (std::size_t row=0;row<2;++row) for(std::size_t dir=0;dir<2;++dir)
    r.hbm[row][dir]=add128(a.hbm[row][dir],b.hbm[row][dir],"activity byte split");
  r.int4=add128(a.int4,b.int4,"activity op split");r.vec=add128(a.vec,b.vec,"activity op split");r.compute_s=a.compute_s+b.compute_s;
  return r;
}

Flow scale_flow(const Flow& a,u64 count) {
  Flow r;
  for(std::size_t i=0;i<resource_count;++i) r.work[i]=a.work[i]*count;
  for(std::size_t row=0;row<2;++row)for(std::size_t dir=0;dir<2;++dir)r.hbm[row][dir]=mul128(a.hbm[row][dir],count,"repeated activity bytes");
  r.int4=mul128(a.int4,count,"repeated INT4 ops");r.vec=mul128(a.vec,count,"repeated Vec ops");r.compute_s=a.compute_s*count;
  return r;
}

std::pair<Flow,Flow> divide_flow(const Flow& a,u64 count) {
  if(!count) fail("mapping: zero frame count");
  Flow base;
  for(std::size_t i=0;i<resource_count;++i) base.work[i]=a.work[i]/count;
  for(std::size_t row=0;row<2;++row)for(std::size_t dir=0;dir<2;++dir)base.hbm[row][dir]=a.hbm[row][dir]/count;
  base.int4=a.int4/count;base.vec=a.vec/count;base.compute_s=a.compute_s/count;
  Flow remainder;
  for(std::size_t i=0;i<resource_count;++i) remainder.work[i]=a.work[i]-base.work[i]*count;
  for(std::size_t row=0;row<2;++row)for(std::size_t dir=0;dir<2;++dir)remainder.hbm[row][dir]=a.hbm[row][dir]-base.hbm[row][dir]*count;
  remainder.int4=a.int4-base.int4*count;remainder.vec=a.vec-base.vec*count;remainder.compute_s=a.compute_s-base.compute_s*count;
  return {base,remainder};
}

void push_epoch(Plan& plan,const Hardware& hw,Flow load,Flow compute,Flow store,
                u64 active,u64 repeat,u64 in_units,u64 out_units,
                double input_initial,double output_initial,const std::string& name) {
  Epoch epoch;epoch.name=name;epoch.credit_initial={{input_initial,output_initial}};epoch.credit_capacity={{2.0,2.0}};epoch.repeat_count=std::max<u64>(1,repeat);
  load.credit={{static_cast<long double>(in_units),0}};
  compute.credit={{-static_cast<long double>(in_units),static_cast<long double>(out_units)}};
  store.credit={{0,-static_cast<long double>(out_units)}};
  Activity l;l.id=plan.epochs.size()*3+1;l.role=Role::Load;
  Activity c;c.id=plan.epochs.size()*3+2;c.role=Role::Compute;
  Activity s;s.id=plan.epochs.size()*3+3;s.role=Role::Store;
  seal_activity(l,load,hw,active);seal_activity(c,compute,hw,active);seal_activity(s,store,hw,active);
  auto has=[](const Activity& a){return std::any_of(a.work.begin(),a.work.end(),[](double x){return x>0.0;});};
  if(has(l))epoch.activities.push_back(l);if(has(c))epoch.activities.push_back(c);if(has(s))epoch.activities.push_back(s);
  plan.epochs.push_back(std::move(epoch));
}

void append_epoch(Plan& plan,const Hardware& hw,Flow load,Flow compute,Flow store,
                  u64 active,u64 frames,const std::string& name) {
  const u64 W=std::max<u64>(1,frames);
  auto [lb,lr]=divide_flow(load,W);auto [cb,cr]=divide_flow(compute,W);auto [sb,sr]=divide_flow(store,W);
  if(W==1){
    push_epoch(plan,hw,load,Flow{},Flow{},active,1,1,0,0,0,name+"_load_first");
    push_epoch(plan,hw,Flow{},compute,Flow{},active,1,1,1,1,0,name+"_compute_last");
    push_epoch(plan,hw,Flow{},Flow{},store,active,1,0,1,0,1,name+"_store_last");
    return;
  }
  if(W==2){
    push_epoch(plan,hw,lb,Flow{},Flow{},active,1,1,0,0,0,name+"_load_first");
    push_epoch(plan,hw,lb,cb,Flow{},active,1,1,1,1,0,name+"_first_overlap");
    push_epoch(plan,hw,Flow{},add_flow(cb,cr),sb,active,1,1,1,1,1,name+"_last_overlap");
    push_epoch(plan,hw,Flow{},Flow{},add_flow(sb,sr),active,1,0,1,0,1,name+"_store_last");
    return;
  }
  push_epoch(plan,hw,lb,Flow{},Flow{},active,1,1,0,0,0,name+"_load_first");
  push_epoch(plan,hw,lb,cb,Flow{},active,1,1,1,1,0,name+"_first_overlap");
  push_epoch(plan,hw,add_flow(scale_flow(lb,W-2),lr),scale_flow(cb,W-2),scale_flow(sb,W-2),
             active,W-2,W-2,W-2,1,1,name+"_interior_repeat");
  push_epoch(plan,hw,Flow{},add_flow(cb,cr),sb,active,1,1,1,1,1,name+"_last_overlap");
  push_epoch(plan,hw,Flow{},Flow{},add_flow(sb,sr),active,1,0,1,0,1,name+"_store_last");
}

std::vector<u64> options(u64 dim, std::initializer_list<u64> choices) {
  if (!dim) fail("unsupported_shape: dimensions must be positive");
  std::set<u64> result;
  for (u64 value : choices) result.insert(std::min(dim, value));
  return {result.begin(), result.end()};
}

void add_gemm_epoch(Plan& plan, Metrics& m, const Hardware& hw,
                    const Topology& topo, u64 M, u64 N, u64 K,
                    const std::string& ad, const std::string& wd,
                    const std::string& od, u64 layers, u64 passes,
                    const GemmChoice& choice, const std::string& label,
                    bool count_source = true) {
  if (!M || !N || !K) fail("unsupported_shape: GEMM M/N/K must be positive");
  const u64 tm = choice.tm, tn = choice.tn, tk = choice.tk;
  const u64 tM = ceildiv(M, tm), tN = ceildiv(N, tn), tK = ceildiv(K, tk);
  const u64 ncore = du(hw, "die", "core_count");
  const bool wr = choice.wr;
  const u64 active = std::max<u64>(1, std::min(ncore, wr ? tN : as_u64(mul128(tM,tN,"output tiles"),"output tiles")));
  const u128 ops = mul128(2, mul128(mul128(M,N,"matrix ops"),K,"matrix ops"),"matrix ops");
  const u128 executions = mul128(layers, passes, "GEMM executions");
  const u128 layer_ops = mul128(ops, executions, "layer matrix ops");
  if (count_source) m.source_mops = add128(m.source_mops, layer_ops, "source matrix ops");
  m.executed_mops = add128(m.executed_mops, layer_ops, "executed matrix ops");
  const u128 a_data = mul128(tile_grid_bytes(M,tm,K,tk,ad), tN, "GEMM A physical bytes");
  const u128 w_data = mul128(tile_grid_bytes(K,tk,N,tn,wd), wr ? 1 : tM, "GEMM W physical bytes");
  const u128 out_data = tile_grid_bytes(M,tm,N,tn,od);
  const u128 a_total = mul128(a_data,executions,"GEMM A physical bytes");
  const u128 w_total = mul128(mul128(w_data,passes,"GEMM W passes"),layers,"GEMM W physical bytes");
  const u128 c_total = mul128(out_data,executions,"GEMM output bytes");
  const u128 read_total = add128(a_total,w_total,"GEMM HBM read");
  m.hbm_read = add128(m.hbm_read,read_total,"HBM read total");
  m.hbm_write = add128(m.hbm_write,c_total,"HBM write total");
  const u128 logical_one = add128(add128(bytes(mul128(M,K,"logical A"),ad),
      bytes(mul128(K,N,"logical W"),wd),"logical GEMM"),
      bytes(mul128(M,N,"logical C"),od),"logical GEMM");
  m.logical = add128(m.logical,mul128(logical_one,executions,"logical GEMM executions"),"logical bytes");

  const bool int4_path = gets(plan.task,"execution_profile") == "int4_matrix_vec_fp32_assumed_v1";
  const std::string execution_dtype=int4_path?"int4":"fp32";
  const bool convert_a=ad!=execution_dtype,convert_w=wd!=execution_dtype;
  const u128 quant_a_elems = !convert_a ? 0 : mul128(mul128(M,K,"A conversion"),mul128(tN,executions,"A conversion repeats"),"A conversion");
  const u128 quant_w_elems = !convert_w ? 0 : mul128(mul128(K,N,"W conversion"),mul128(wr?1:tM,executions,"W conversion repeats"),"W conversion");
  const u128 conversions = add128(quant_a_elems,quant_w_elems,"conversion elements");
  const u128 a_i4 = mul128(tile_grid_bytes(M,tm,K,tk,"int4"),tN,"A INT4 bytes");
  const u128 w_i4 = mul128(tile_grid_bytes(K,tk,N,tn,"int4"),wr?1:tM,"W INT4 bytes");
  const u128 i4total = int4_path ? add128(mul128(a_i4,executions,"A INT4 bytes"),mul128(w_i4,executions,"W INT4 bytes"),"INT4 input bytes") : 0;
  m.int4bytes = add128(m.int4bytes,i4total,"INT4 input bytes");
  const u128 scale_a = int4_path&&convert_a ? mul128(mul128(tM,tK,"A scale tiles"),tN,"A scale tiles") : 0;
  const u128 scale_w = int4_path&&convert_w ? mul128(mul128(tK,tN,"W scale tiles"),wr?1:tM,"W scale tiles") : 0;
  const u128 scales = mul128(add128(mul128(scale_a,passes,"A scale passes"),mul128(scale_w,passes,"W scale passes"),"scale tiles"),mul128(layers,4,"scale bytes"),"scale bytes");
  const u128 a_exec=mul128(tile_grid_bytes(M,tm,K,tk,execution_dtype),mul128(tN,executions,"A execution repeats"),"A execution bytes");
  const u128 w_exec=mul128(tile_grid_bytes(K,tk,N,tn,execution_dtype),mul128(wr?1:tM,executions,"W execution repeats"),"W execution bytes");
  const u128 quant_buf=add128(add128(convert_a?a_exec:0,convert_w?w_exec:0,"converted buffers"),scales,"converted buffers");
  const u128 conversion_reads=add128(convert_a?a_total:0,convert_w?w_total:0,"conversion source reads");
  const u128 scale_reads=mul128(add128(scale_a,mul128(scale_w,wr?tM:1,"resident weight scale reads"),"scale reads"),mul128(executions,4,"scale read bytes"),"scale read bytes");
  const u128 partial = int4_path ? mul128(mul128(mul128(M,N,"partial"),tK,"partial K"),mul128(executions,4,"partial bytes"),"partial bytes") : 0;
  const u128 accum_write=mul128(mul128(mul128(M,N,"accumulator"),tK,"accumulator K"),mul128(executions,4,"accumulator bytes"),"accumulator bytes");
  const u128 accum_read=mul128(mul128(mul128(M,N,"accumulator"),tK-1,"accumulator reuse"),mul128(executions,4,"accumulator bytes"),"accumulator read");
  const u128 output_cast_elements=od=="fp32"?0:mul128(mul128(M,N,"output conversions"),executions,"output conversions");
  const u128 output_cast_reads=mul128(output_cast_elements,4,"output conversion reads");
  const u128 output_scales=od=="int4"?mul128(mul128(tM,tN,"output scale tiles"),mul128(executions,4,"output scale bytes"),"output scale bytes"):0;
  const u128 consumed_inputs=add128(a_exec,mul128(w_exec,wr?tM:1,"resident weight consumption"),"matrix operand reads");
  const u128 sram_r=add128(add128(add128(consumed_inputs,conversion_reads,"matrix source reads"),add128(partial,accum_read,"matrix state reads"),"matrix reads"),add128(scale_reads,output_cast_reads,"matrix conversion reads"),"SRAM reads");
  const u128 sram_w=add128(add128(quant_buf,partial,"matrix writes"),add128(accum_write,add128(output_cast_elements?c_total:0,output_scales,"output conversion writes"),"matrix writes"),"SRAM writes");
  const u128 local = add128(sram_r,sram_w,"LOCAL traffic");
  m.sram_read = add128(m.sram_read,add128(sram_r,c_total,"GEMM store SRAM read"),"SRAM reads"); m.sram_write = add128(m.sram_write,add128(sram_w,read_total,"GEMM load SRAM write"),"SRAM writes"); m.local = add128(m.local,local,"LOCAL traffic");

  u128 vecops = 0;
  u128 peops = 0;
  if (int4_path) {
    peops = layer_ops;
    vecops = add128(vecops,mul128(conversions,static_cast<u64>(perf(hw,"kappa_quant_eqops",4)),"quant eqops"),"GEMM Vec ops");
    const u128 converted = mul128(mul128(mul128(M,N,"acc convert"),tK,"acc convert"),executions,"acc convert");
    vecops = add128(vecops,mul128(converted,static_cast<u64>(perf(hw,"kappa_acc_convert_eqops",2)),"acc convert eqops"),"GEMM Vec ops");
    vecops = add128(vecops,mul128(mul128(mul128(M,N,"acc add"),tK-1,"acc add"),executions,"acc add"),"GEMM Vec ops");
    if (od != "fp32") vecops = add128(vecops,mul128(output_cast_elements,od=="int4"?static_cast<u64>(perf(hw,"kappa_quant_eqops",4)):1,"output conversion"),"GEMM Vec ops");
    m.conversion_elements=add128(m.conversion_elements,add128(add128(conversions,converted,"GEMM conversions"),output_cast_elements,"GEMM conversions"),"conversion elements");
  } else {
    // BF16/INT32 inputs are converted once per loaded tile. INT4 decoding
    // uses the explicit template cost; FP32 operands require no conversion.
    vecops=add128(layer_ops,add128(mul128(quant_a_elems,ad=="int4"?static_cast<u64>(perf(hw,"kappa_quant_eqops",4)):1,"A FP32 conversion"),mul128(quant_w_elems,wd=="int4"?static_cast<u64>(perf(hw,"kappa_quant_eqops",4)):1,"W FP32 conversion"),"FP32 conversions"),"GEMM Vec ops");
    vecops=add128(vecops,output_cast_elements,"output conversion");
    m.conversion_elements=add128(m.conversion_elements,add128(conversions,output_cast_elements,"GEMM conversions"),"conversion elements");
  }
  m.int4ops = add128(m.int4ops,peops,"INT4 ops"); m.vecops = add128(m.vecops,vecops,"Vec eqops");
  // Quantization, accumulator and storage conversion are execution work;
  // the source GEMM graph has matrix operations, not these Vec operations.
  const double comp_s = compute_time(hw,peops,vecops,active);

  Flow load, compute, store;
  const double hbm_eta = eta(hw,"eta_hbm_matrix");
  add_hbm(load,hw,topo,read_total,active,false,hbm_eta);
  add_bytes_work(load,SRAM_WRITE,read_total,"load SRAM bytes");
  compute.work[SRAM_READ] = static_cast<long double>(as_u64(sram_r,"GEMM SRAM reads"));
  compute.work[SRAM_WRITE] = static_cast<long double>(as_u64(sram_w,"GEMM SRAM writes"));
  compute.work[LOCAL] = static_cast<long double>(as_u64(local,"GEMM LOCAL traffic"));
  compute.compute_s = comp_s; compute.int4 = peops; compute.vec = vecops;
  add_bytes_work(store,SRAM_READ,c_total,"store SRAM reads");
  add_hbm(store,hw,topo,c_total,active,true,hbm_eta);
  const u128 rounds = wr ? mul128(ceildiv(tN,ncore),tM,"WR output rounds") : ceildiv(as_u64(mul128(tM,tN,"OS output tiles"),"OS output tiles"),ncore);
  const u128 wave = mul128(rounds,tK,"GEMM consumption waves");
  append_epoch(plan,hw,std::move(load),std::move(compute),std::move(store),active,as_u64(rounds,"GEMM output rounds"),label);
  const u128 max_m = std::min(M,tm), max_n = std::min(N,tn), max_k = std::min(K,tk);
  const u128 a_exec_tile = tile_bytes(static_cast<u64>(max_m),static_cast<u64>(max_k),int4_path?"int4":"fp32");
  const u128 w_exec_tile = tile_bytes(static_cast<u64>(max_k),static_cast<u64>(max_n),int4_path?"int4":"fp32");
  const u128 a_raw_tile = tile_bytes(static_cast<u64>(max_m),static_cast<u64>(max_k),ad);
  const u128 w_raw_tile = tile_bytes(static_cast<u64>(max_k),static_cast<u64>(max_n),wd);
  const u128 a_tile=convert_a?add128(add128(a_exec_tile,a_raw_tile,"A raw and converted slots"),int4_path?4:0,"A conversion scale"):a_exec_tile;
  const u128 w_tile=convert_w?add128(add128(w_exec_tile,w_raw_tile,"W raw and converted slots"),int4_path?4:0,"W conversion scale"):w_exec_tile;
  const u128 accum = mul128(mul128(max_m,max_n,"acc tile"),int4_path?8:4,"accumulator and partial");
  const u128 output = mul128(2,add128(tile_bytes(static_cast<u64>(max_m),static_cast<u64>(max_n),od),od=="int4"?4:0,"output tile and scale"),"pending output slots");
  const u128 resident_weight=add128(tile_grid_bytes(K,tk,static_cast<u64>(max_n),static_cast<u64>(max_n),execution_dtype),convert_w?add128(tile_grid_bytes(K,tk,static_cast<u64>(max_n),static_cast<u64>(max_n),wd),int4_path?mul128(tK,4,"WR resident scales"):0,"WR conversion buffers"):0,"WR resident weight");
  const u128 peak = wr ? add128(resident_weight,add128(mul128(2,a_tile,"WR input slots"),add128(accum,output,"WR live buffers"),"WR live buffers"),"WR peak")
                       : add128(mul128(2,add128(a_tile,w_tile,"OS tile inputs"),"OS double inputs"),add128(accum,output,"OS live outputs"),"OS peak");
  m.sram_peak = std::max(m.sram_peak,peak);
  m.output_rounds = add128(m.output_rounds,rounds,"output rounds"); m.waves = add128(m.waves,wave,"consumption waves");
  m.active_cores = active; m.tm=tm; m.tn=tn; m.tk=tk; m.dataflow=wr?"weight_resident_v2":"output_stationary_v2"; m.template_id=label;
}

void finalize_summary(Plan& p, const Metrics& m, const Hardware& hw) {
  p.summary["kind"] = gets(p.task,"op_kind");
  p.summary["template_id"] = m.template_id;
  p.summary["dataflow_id"] = m.dataflow;
  p.summary["source_matrix_ops"] = as_u64(m.source_mops,"source matrix ops");
  p.summary["executed_matrix_ops"] = as_u64(m.executed_mops,"executed matrix ops");
  p.summary["recomputed_matrix_ops"] = as_u64(m.executed_mops >= m.source_mops ? m.executed_mops-m.source_mops : 0,"recomputed matrix ops");
  p.summary["recompute_matrix_ops"] = p.summary["recomputed_matrix_ops"];
  p.summary["executed_int4_ops"] = as_u64(m.int4ops,"executed INT4 ops");
  p.summary["int4_input_bytes"] = as_u64(m.int4bytes,"INT4 input bytes");
  p.summary["source_vec_ops"] = as_u64(m.source_vops,"source Vec ops");
  p.summary["vec_equivalent_ops"] = as_u64(m.vecops,"Vec eqops");
  p.summary["conversion_elements"] = as_u64(m.conversion_elements,"conversion elements");
  p.summary["logical_tensor_bytes"] = as_u64(m.logical,"logical tensor bytes");
  p.summary["hbm_read_bytes_total"] = as_u64(m.hbm_read,"HBM read total");
  p.summary["hbm_write_bytes_total"] = as_u64(m.hbm_write,"HBM write total");
  p.summary["physical_transfer_bytes"] = as_u64(add128(m.hbm_read,m.hbm_write,"HBM total"),"HBM total");
  p.summary["hbm_workspace_peak_bytes"] = as_u64(m.workspace,"HBM workspace");
  p.summary["sram_peak_bytes_per_core"] = as_u64(m.sram_peak,"SRAM peak");
  p.summary["sram_capacity_bytes_per_core"] = du(hw,"core","sram_bytes_per_core");
  p.summary["sram_read_bytes"] = as_u64(m.sram_read,"SRAM reads");
  p.summary["sram_write_bytes"] = as_u64(m.sram_write,"SRAM writes");
  p.summary["local_bus_bytes"] = as_u64(m.local,"LOCAL bytes");
  p.summary["output_round_count"] = as_u64(m.output_rounds,"output rounds");
  p.summary["consumption_wave_count"] = as_u64(m.waves,"consumption waves");
  p.summary["wave_count"] = p.summary["consumption_wave_count"];
  p.summary["active_cores"] = m.active_cores;
  p.summary["tm"] = m.tm; p.summary["tn"] = m.tn; p.summary["tk"] = m.tk;
  p.summary["th"] = m.th; p.summary["tf"] = m.tf;
  p.summary["executed_score_slots"] = as_u64(m.score_slots,"executed score slots");
  p.summary["valid_score_pairs"] = as_u64(m.valid_pairs,"valid score pairs");
  p.summary["score_bytes"] = as_u64(m.score_bytes,"score bytes");
  p.summary["probability_bytes"] = as_u64(m.probability_bytes,"probability bytes");
  p.summary["granularity_validation"] = "not_run";
  p.summary["mapping_assumptions"] = m.assumptions;
  p.summary["activity_units"] = "work/hbm_bytes/ops are totals over this epoch; repeat_count is audit metadata only; signed credit_delta is total over progress [0,1]";
  p.summary["routing_evidence"] = {{"profile", gets(hw.input.at("performance_profile"),"timing_topology_profile")}, {"status","engineering_assumption"}};
}

std::shared_ptr<Plan> new_plan(const Hardware& hw, const nlohmann::json& task) {
  auto p = std::make_shared<Plan>(); p->task = task;
  p->hardware_fingerprint = hw.fingerprint;
  if (p->hardware_fingerprint.empty()) p->hardware_fingerprint = stable_hash(hw.input.dump());
  p->launch_ns = 0;
  return p;
}

void add_vector(Plan& plan, Metrics& m, const Hardware& hw,
                const Topology& topo, u128 elements, u128 eqops,
                u128 input_bytes, u128 output_bytes, u64 active,
                const std::string& label, u128 extra_input = 0) {
  if (!elements) fail("unsupported_shape: vector element count must be positive");
  const u128 in = add128(input_bytes,extra_input,"vector input bytes");
  m.logical = add128(m.logical,add128(in,output_bytes,"vector logical bytes"),"logical bytes");
  m.hbm_read = add128(m.hbm_read,in,"vector HBM reads"); m.hbm_write = add128(m.hbm_write,output_bytes,"vector HBM writes");
  m.source_vops = add128(m.source_vops,eqops,"source Vec ops"); m.vecops = add128(m.vecops,eqops,"Vec eqops");
  const u128 sramr = in, sramw = output_bytes;
  m.sram_read = add128(m.sram_read,add128(sramr,output_bytes,"vector store SRAM read"),"SRAM read bytes"); m.sram_write = add128(m.sram_write,add128(sramw,in,"vector load SRAM write"),"SRAM write bytes");
  m.local = add128(m.local,add128(sramr,sramw,"LOCAL traffic"),"LOCAL traffic");
  Flow load, compute, store;
  add_hbm(load,hw,topo,in,active,false,eta(hw,"eta_hbm_vector"));
  load.work[SRAM_WRITE] = static_cast<long double>(as_u64(in,"vector load"));
  compute.work[SRAM_READ] = static_cast<long double>(as_u64(sramr,"vector SRAM reads"));
  compute.work[SRAM_WRITE] = static_cast<long double>(as_u64(sramw,"vector SRAM writes"));
  compute.work[LOCAL] = static_cast<long double>(as_u64(add128(sramr,sramw,"vector LOCAL"),"vector LOCAL"));
  compute.vec = eqops; compute.compute_s = compute_time(hw,0,eqops,active);
  store.work[SRAM_READ] = static_cast<long double>(as_u64(output_bytes,"vector store"));
  add_hbm(store,hw,topo,output_bytes,active,true,eta(hw,"eta_hbm_vector"));
  // Vector kernels stream bounded per-core tiles. The aggregate tensor still
  // determines total service work, but it is never charged as one resident
  // SRAM allocation (which would reject ordinary large Norms).
  const u128 tile_elems = std::min<u128>(elements,256);
  const u128 tile_input = elements ? (mul128(input_bytes,tile_elems,"vector input tile") + elements - 1) / elements : 0;
  const u128 tile_output = elements ? (mul128(output_bytes,tile_elems,"vector output tile") + elements - 1) / elements : 0;
  u128 tile_extra = elements ? (mul128(extra_input,tile_elems,"vector extra tile") + elements - 1) / elements : 0;
  if (gets(plan.task,"op_kind")=="norm") {
    const u64 hidden=shape_u(plan.task,"hidden",shape_u(plan.task,"h",0));
    if (hidden) {
      const std::string norm=gets(plan.task.at("tensor_shapes"),"norm_type","rmsnorm");
      tile_extra=mul128(norm=="layernorm"?2:1,bytes(std::min<u128>(hidden,tile_elems),dtype(plan.task,"parameters")),"Norm parameter tile");
    }
  }
  const u128 tile_peak=add128(add128(mul128(2,add128(tile_input,tile_extra,"vector input tile"),"vector input slots"),mul128(2,tile_output,"vector pending output slots"),"vector live tensors"),
      mul128(tile_elems,4,"vector FP32 working tile"),"vector peak");
  const u64 elems64=as_u64(elements,"vector elements");
  const u64 per_round=std::max<u64>(1,as_u64(mul128(std::max<u64>(1,active),256,"vector tile throughput"),"vector tile throughput"));
  const u64 rounds=std::max<u64>(1,ceildiv(elems64,per_round));
  append_epoch(plan,hw,std::move(load),std::move(compute),std::move(store),active,rounds,label);
  m.sram_peak = std::max<u128>(m.sram_peak,tile_peak);
  m.output_rounds = add128(m.output_rounds,rounds,"vector rounds");
  const u64 waves=rounds;
  m.waves = add128(m.waves,waves,"vector waves"); m.active_cores=active; m.template_id=label;
}

std::vector<std::shared_ptr<Plan>> enumerate_gemm(const Hardware& hw,
                                                   const nlohmann::json& task,
                                                   const Topology& topo,
                                                   u64 M,u64 N,u64 K) {
  const auto ts=options(M,{1,16,64,256}); const auto ns=options(N,{64,256,1024,4096});
  const auto ks=options(K,{64,256,1024,4096});
  const u64 layers=getu(task,"layer_instances",1), passes=getu(task,"weight_passes",1);
  const std::string ad=dtype(task,"activations"), wd=dtype(task,"weights"), od=dtype(task,"output");
  const u64 ncore=du(hw,"die","core_count");
  const double cap=perf(hw,"rho_sram",0.9)*dd(hw,"core","sram_bytes_per_core");
  std::vector<std::shared_ptr<Plan>> result;
  for (bool wr : {false,true}) for (u64 tm:ts) for (u64 tn:ns) for (u64 tk:ks) {
    auto p=new_plan(hw,task); Metrics m; m.assumptions={"engineering_assumption","bounded_tile_candidates","exact_per_tile_INT4_tail_packing"};
    GemmChoice c{tm,tn,tk,wr};
    add_gemm_epoch(*p,m,hw,topo,M,N,K,ad,wd,od,layers,passes,c,wr?"weight_resident_v2":"output_stationary_v2");
    if (static_cast<long double>(as_u64(m.sram_peak,"SRAM peak")) > cap) continue;
    finalize_summary(*p,m,hw);
    p->summary["template_id"] = "gemm_" + m.dataflow;
    p->summary["dataflow_id"] = m.dataflow;
    p->summary["source_matrix_ops"] = as_u64(m.source_mops,"source matrix ops");
    p->summary["executed_matrix_ops"] = as_u64(m.executed_mops,"executed matrix ops");
    p->summary["recomputed_matrix_ops"] = 0; p->summary["recompute_matrix_ops"] = 0;
    p->summary["tm"]=tm; p->summary["tn"]=tn; p->summary["tk"]=tk;
    p->fingerprint=plan_to_json(*p).at("fingerprint").get<std::string>();
    result.push_back(std::move(p));
  }
  return result;
}

std::vector<std::shared_ptr<Plan>> enumerate_attention(const Hardware& hw,
                                                        const nlohmann::json& task,
                                                        const Topology& topo,
                                                        bool full,
                                                        const std::string& mode = "both") {
  const auto spans=spans_of(task);
  if (spans.empty()) fail("task.spans: attention requires at least one span");
  const u64 qh=shape_u(task,"q_heads",shape_u(task,"q_heads_local",1));
  const u64 kvh=shape_u(task,"kv_heads",shape_u(task,"kv_heads_local",1));
  const u64 hd=shape_u(task,"head_dim",1);
  if (!qh || !kvh || qh%kvh) fail("unsupported_head_mapping: standard GQA requires q_heads divisible by kv_heads");
  if (mode!="both" && mode!="qk" && mode!="av") fail("unsupported_attention_mode: "+mode);
  bool causal=true;
  if (task.contains("causal")) {
    if (!task.at("causal").is_boolean()) fail("task.causal: expected boolean");
    causal=task.at("causal").get<bool>();
  }
  const u128 executions=mul128(getu(task,"layer_instances",1),getu(task,"weight_passes",1),"attention executions");
  const u64 ncore=du(hw,"die","core_count");
  const std::string dtype_k=dtype(task,"kv"), dtype_q=dtype(task,"activations"), dtype_o=dtype(task,"output");
  std::vector<std::shared_ptr<Plan>> result;
  u64 maxq=1,maxkv=1;
  for(const auto& span:spans){
    maxq=std::max(maxq,span.q);
    const u128 last_kv=add128(span.kv,mul128(span.count-1,span.kv_stride,"attention last KV length"),"attention last KV length");
    maxkv=std::max(maxkv,as_u64(last_kv,"attention maximum KV length"));
  }
  const auto tq_opts=full?std::vector<u64>{16}:options(maxq,{16,64,256,1024});
  const auto tk_opts=full?std::vector<u64>{64}:options(maxkv,{16,64,256,1024});
  const bool int4path=gets(task,"execution_profile")=="int4_matrix_vec_fp32_assumed_v1";
  const std::string score_dtype=dtype(task,"scores","fp32");
  const std::string accumulator_dtype=dtype(task,"accumulator","fp32");
  if(score_dtype!="fp32"||accumulator_dtype!="fp32")
    fail("unsupported_dtype: attention score and accumulator workspaces are fixed FP32");
  const u128 kappa_quant=static_cast<u64>(perf(hw,"kappa_quant_eqops",4));
  const u128 kappa_acc=static_cast<u64>(perf(hw,"kappa_acc_convert_eqops",2));
  if(kappa_quant>std::numeric_limits<u64>::max()||kappa_acc>std::numeric_limits<u64>::max())
    fail("overflow: attention conversion coefficient");
  auto sum_kv_lengths=[](const Span& s)->u128 {
    return add128(mul128(s.count,s.kv,"attention KV span sum"),
        mul128(s.kv_stride,mul128(s.count,s.count-1,"attention KV span sum")/2,"attention KV span sum"),
        "attention KV span sum");
  };
  auto sum_storage_tensor_bytes=[&](const Span& s,u64 features,const std::string& dt)->u128 {
    const u128 elements=mul128(sum_kv_lengths(s),features,"attention source tensor elements");
    if(dt!="int4")return bytes(elements,dt);
    if(features%2==0)return elements/2;
    const u64 period=2/std::gcd<u64>(2,s.kv_stride%2);
    u128 odd_tensors=0;
    for(u64 r=0;r<std::min(period,s.count);++r){
      const u64 n=1+(s.count-1-r)/period;
      const u128 first=add128(s.kv,mul128(r,s.kv_stride,"attention INT4 logical parity"),"attention INT4 logical parity");
      if(first%2)odd_tensors=add128(odd_tensors,n,"attention INT4 logical parity");
    }
    return add128(elements,odd_tensors,"attention INT4 logical bytes")/2;
  };
  auto progression_tiles=[](const Span& s,u64 tile)->std::pair<u128,u128> {
    const u64 period=tile/std::gcd(s.kv_stride,tile);
    const u128 step=mul128(period,s.kv_stride,"attention compact tile stride");
    u128 all=0,full=0;
    for(u64 r=0;r<std::min(period,s.count);++r){
      const u64 n=1+(s.count-1-r)/period;
      const u128 first=add128(s.kv,mul128(r,s.kv_stride,"attention compact tile length"),"attention compact tile length");
      const u128 group_full=add128(mul128(n,first/tile,"attention compact full tile count"),
          mul128(step/tile,mul128(n,n-1,"attention compact full tile count")/2,"attention compact full tile count"),
          "attention compact full tile count");
      const bool tail=(first%tile)!=0;
      full=add128(full,group_full,"attention compact full tile count");
      all=add128(all,add128(group_full,tail?n:0,"attention compact tile count"),"attention compact tile count");
    }
    return {all,full};
  };
  auto packed_kv_grid=[&](const Span& s,u64 tile,u64 features,u64 feature_tile)->u128 {
    const u64 period=tile/std::gcd(s.kv_stride,tile);
    const u128 step=mul128(period,s.kv_stride,"attention INT4 tile stride");
    const u128 full_tile=tile_grid_bytes(tile,tile,features,feature_tile,"int4");
    u128 total=0;
    for(u64 r=0;r<std::min(period,s.count);++r){
      const u64 n=1+(s.count-1-r)/period;
      const u128 first=add128(s.kv,mul128(r,s.kv_stride,"attention INT4 KV length"),"attention INT4 KV length");
      const u128 full_count=add128(mul128(n,first/tile,"attention INT4 full tiles"),
          mul128(step/tile,mul128(n,n-1,"attention INT4 full tiles")/2,"attention INT4 full tiles"),
          "attention INT4 full tiles");
      total=add128(total,mul128(full_count,full_tile,"attention INT4 full bytes"),"attention INT4 bytes");
      const u64 tail=static_cast<u64>(first%tile);
      if(tail){
        const u128 tail_bytes=tile_grid_bytes(tail,tail,features,feature_tile,"int4");
        total=add128(total,mul128(n,tail_bytes,"attention INT4 tail bytes"),"attention INT4 bytes");
      }
    }
    return total;
  };
  auto packed_causal_prefix=[&](const Span& s,u64 tq,u64 tk,u64 features,u64 feature_tile,u128 prefix)->u128 {
    const u64 qfull=s.q/tq,qtail=s.q%tq;
    u128 tail_tokens=0,tail_payload=0;
    auto add_tail_class=[&](u64 requests,u64 kv_remainder,u64 sat_tiles){
      if(!requests||!sat_tiles||!kv_remainder)return;
      const u128 count=mul128(requests,sat_tiles,"attention causal packed tail count");
      tail_tokens=add128(tail_tokens,mul128(count,kv_remainder,"attention causal packed tail tokens"),"attention causal packed tail tokens");
      tail_payload=add128(tail_payload,mul128(count,tile_grid_bytes(kv_remainder,kv_remainder,features,feature_tile,"int4"),"attention causal packed tail bytes"),"attention causal packed tail bytes");
    };
    if(s.count==1){
      const u64 rem=s.kv%tk;
      if(rem){
        const __int128 threshold=static_cast<__int128>(((s.kv-1)/tk)*tk+1)-
            (static_cast<__int128>(s.qs)-static_cast<__int128>(s.ks));
        u64 sat_full=0;
        if(threshold<=0)sat_full=qfull;
        else if(static_cast<u128>(threshold)<=s.q){
          const u64 first=static_cast<u64>((static_cast<u128>(threshold)+tq-1)/tq);
          if(first<=qfull)sat_full=qfull-first+1;
        }
        const bool sat_tail=qtail&&threshold<=static_cast<__int128>(s.q);
        add_tail_class(1,rem,as_u64(add128(sat_full,sat_tail?1:0,"attention causal packed tails"),"attention causal packed tails"));
      }
    }else{
      const u64 period=tk/std::gcd(s.kv_stride,tk);
      for(u64 r=0;r<std::min(period,s.count);++r){
        const u64 requests=1+(s.count-1-r)/period;
        const u64 rem=static_cast<u64>((s.kv+static_cast<u128>(r)*s.kv_stride)%tk);
        if(!rem)continue;
        const __int128 threshold=static_cast<__int128>(s.q)-static_cast<__int128>(rem)+1;
        u64 sat_full=0;
        if(threshold<=0)sat_full=qfull;
        else if(static_cast<u128>(threshold)<=s.q){
          const u64 first=static_cast<u64>((static_cast<u128>(threshold)+tq-1)/tq);
          if(first<=qfull)sat_full=qfull-first+1;
        }
        const bool sat_tail=qtail&&threshold<=static_cast<__int128>(s.q);
        add_tail_class(requests,rem,as_u64(add128(sat_full,sat_tail?1:0,"attention causal packed tails"),"attention causal packed tails"));
      }
    }
    if(tail_tokens>prefix||(prefix-tail_tokens)%tk)
      fail("mapping_invariant: causal packed prefix does not decompose into full and tail tiles");
    const u128 full_tile=tile_grid_bytes(tk,tk,features,feature_tile,"int4");
    return add128(mul128((prefix-tail_tokens)/tk,full_tile,"attention causal packed full bytes"),tail_payload,"attention causal packed bytes");
  };
  struct AttentionTileCounts {u128 slots=0,prefix=0,blocks=0,row_blocks=0,active_rows=0;};
  auto causal_counts=[&](const Span& s,u64 tq,u64 tk)->AttentionTileCounts {
    const auto base=prefix_slots_one(s,tq,tk);
    AttentionTileCounts out;out.slots=base.first;out.prefix=base.second;
    u128 saturated_tiles=0,saturated_rows=0,tail_sum=0,tail_count=0,weighted_tail=0,weighted_tail_blocks=0;
    if(s.count>1){
      const u64 period=tk/std::gcd(s.kv_stride,tk);
      const u64 qfull=s.q/tq,qtail=s.q%tq;
      for(u64 r=0;r<std::min(period,s.count);++r){
        const u64 n=1+(s.count-1-r)/period;
        const u128 first=add128(s.kv,mul128(r,s.kv_stride,"attention causal residue"),"attention causal residue");
        const u64 rem=static_cast<u64>(first%tk);
        const __int128 threshold=static_cast<__int128>(s.q)-static_cast<__int128>(rem)+1;
        u64 sat_full=0;
        if(threshold<=0)sat_full=qfull;
        else if(static_cast<u128>(threshold)<=s.q){
          const u64 first_tile=static_cast<u64>((static_cast<u128>(threshold)+tq-1)/tq);
          if(first_tile<=qfull)sat_full=qfull-first_tile+1;
        }
        const bool sat_tail=qtail&&threshold<=static_cast<__int128>(s.q);
        const u128 sat_tiles=add128(sat_full,sat_tail?1:0,"attention saturated tiles");
        const u128 sat_rows=add128(mul128(sat_full,tq,"attention saturated rows"),sat_tail?qtail:0,"attention saturated rows");
        saturated_tiles=add128(saturated_tiles,mul128(n,sat_tiles,"attention saturated tiles"),"attention saturated tiles");
        saturated_rows=add128(saturated_rows,mul128(n,sat_rows,"attention saturated rows"),"attention saturated rows");
        if(rem){
          tail_sum=add128(tail_sum,mul128(mul128(n,sat_tiles,"attention tail count"),rem,"attention tail tokens"),"attention tail tokens");
          tail_count=add128(tail_count,mul128(n,sat_tiles,"attention tail blocks"),"attention tail blocks");
          weighted_tail=add128(weighted_tail,mul128(mul128(n,sat_rows,"attention weighted tail rows"),rem,"attention weighted tail tokens"),"attention weighted tail tokens");
          weighted_tail_blocks=add128(weighted_tail_blocks,mul128(n,sat_rows,"attention weighted tail blocks"),"attention weighted tail blocks");
        }
      }
      out.active_rows=mul128(s.q,s.count,"attention active query rows");
    }else{
      const u64 tail_k=s.kv%tk;
      const __int128 key_block_end=static_cast<__int128>(ceildiv(s.kv,tk)-1)*static_cast<__int128>(tk)+1;
      const __int128 threshold=key_block_end-(static_cast<__int128>(s.qs)-static_cast<__int128>(s.ks));
      const u64 full=s.q/tq,tail=s.q%tq;
      u64 sat_full=0;
      if(threshold<=0)sat_full=full;
      else if(static_cast<u128>(threshold)<=s.q){
        const u64 first=static_cast<u64>((static_cast<u128>(threshold)+tq-1)/tq);
        if(first<=full)sat_full=full-first+1;
      }
      const bool sat_tail=tail&&threshold<=static_cast<__int128>(s.q);
      saturated_tiles=add128(sat_full,sat_tail?1:0,"attention saturated query tiles");
      saturated_rows=add128(mul128(sat_full,tq,"attention saturated rows"),sat_tail?tail:0,"attention saturated rows");
      if(tail_k){tail_sum=mul128(saturated_tiles,tail_k,"attention saturated tail tokens");tail_count=saturated_tiles;}
      if(tail_k){weighted_tail=mul128(saturated_rows,tail_k,"attention weighted tail tokens");weighted_tail_blocks=saturated_rows;}
      const u128 before=static_cast<__int128>(s.ks)>static_cast<__int128>(s.qs)
          ?static_cast<u128>(static_cast<__int128>(s.ks)-static_cast<__int128>(s.qs)):0;
      out.active_rows=before>=s.q?0:static_cast<u128>(s.q)-before;
    }
    if(tail_sum>out.prefix||(out.prefix-tail_sum)%tk)
      fail("mapping_invariant: causal KV prefix does not decompose into real tiles");
    out.blocks=add128((out.prefix-tail_sum)/tk,tail_count,"attention causal KV blocks");
    if(weighted_tail>out.slots||(out.slots-weighted_tail)%tk)
      fail("mapping_invariant: causal score slots do not decompose into real tiles");
    out.row_blocks=add128((out.slots-weighted_tail)/tk,weighted_tail_blocks,"attention row/KV block visits");
    return out;
  };
  for (u64 tq:tq_opts) for (u64 tk:tk_opts) {
    Metrics m; auto p=new_plan(hw,task);
    u128 slots=0, valid=0, qbytes=0, kvbytes=0, outbytes=0, queryrows=0, prefixes=0;
    u128 output_tasks=0,active_queryrows=0,row_block_visits=0,tile_block_visits=0,logical=0,qk_qbytes=0;
    u128 q_packed=0,k_packed=0,v_packed=0,q_quant_elements=0,k_quant_elements=0,v_quant_elements=0;
    u128 qk_kbytes=0,pv_vbytes=0,av_probbytes=0,q_scale_tiles=0,k_scale_tiles=0,v_scale_tiles=0;
    for (const Span& s:spans) {
      u128 sl=0, pr=0;
      const u128 kv_sum=sum_kv_lengths(s);
      const u128 qtiles=ceildiv(s.q,tq);
      const u128 all_kv_tiles=progression_tiles(s,tk).first;
      // Materialized score/primitive paths operate on the complete rectangle.
      // Only fused causal attention skips future KV blocks.
      const bool materialized=full||mode!="both";
      const auto causal_stat=causal&&!materialized?causal_counts(s,tq,tk):AttentionTileCounts{};
      if (materialized) { sl=mul128(s.q,kv_sum,"materialized attention score slots");pr=mul128(qtiles,kv_sum,"materialized attention KV reads"); }
      else if (causal) { sl=causal_stat.slots; pr=causal_stat.prefix; }
      else { sl=mul128(s.q,kv_sum,"noncausal score slots");pr=mul128(qtiles,kv_sum,"noncausal attention KV reads"); }
      const u128 request_heads=mul128(s.count,qh,"attention heads");
      const u128 score_slots=mul128(sl,qh,"attention score heads");
      const u128 score_valid=mul128(valid_pairs(s,causal),qh,"attention valid heads");
      const u128 query_count=mul128(s.q,request_heads,"query rows");
      output_tasks=add128(output_tasks,mul128(qtiles,request_heads,"attention output tasks"),"attention output tasks");
      const u128 active_rows=causal&&!materialized?causal_stat.active_rows:mul128(s.q,s.count,"attention active rows");
      active_queryrows=add128(active_queryrows,mul128(active_rows,qh,"attention active rows"),"attention active rows");
      const u128 row_blocks=causal&&!materialized?causal_stat.row_blocks:mul128(s.q,all_kv_tiles,"attention row/KV block visits");
      const u128 tile_blocks=causal&&!materialized?causal_stat.blocks:mul128(qtiles,all_kv_tiles,"attention tile/KV block visits");
      row_block_visits=add128(row_block_visits,mul128(row_blocks,qh,"attention row/KV block visits"),"attention row/KV block visits");
      tile_block_visits=add128(tile_block_visits,mul128(tile_blocks,qh,"attention tile/KV block visits"),"attention tile/KV block visits");
      slots=add128(slots,score_slots,"attention slots");
      valid=add128(valid,score_valid,"valid score pairs");
      queryrows=add128(queryrows,query_count,"query rows");
      prefixes=add128(prefixes,mul128(pr,qh,"GQA KV prefix reads"),"GQA KV prefix reads");

      const u128 qtile_raw=tile_grid_bytes(s.q,tq,hd,hd,dtype_q);
      const u128 qtile_i4=tile_grid_bytes(s.q,tq,hd,hd,"int4");
      const u128 q_reuses=materialized&&mode!="av"?all_kv_tiles:1;
      const u128 q_head_copies=materialized&&mode!="av"?qh:request_heads;
      if(mode!="av"){
        const u128 qread=mul128(mul128(qtile_raw,q_head_copies,"attention Q bytes"),q_reuses,"attention Q rereads");
        qbytes=add128(qbytes,qread,"Q bytes");
        if(materialized)qk_qbytes=add128(qk_qbytes,qread,"full QK Q rereads");
        q_packed=add128(q_packed,mul128(mul128(qtile_i4,q_head_copies,"attention packed Q"),q_reuses,"attention packed Q rereads"),"attention packed Q");
        q_quant_elements=add128(q_quant_elements,mul128(mul128(mul128(s.q,hd,"attention Q conversion"),q_head_copies,"attention Q conversion"),q_reuses,"attention Q conversion rereads"),"attention Q conversion");
      }
      const u64 feature_tile=std::min<u64>(hd,64);
      const u128 full_k_raw=dtype_k=="int4"?packed_kv_grid(s,tk,hd,hd):bytes(mul128(kv_sum,hd,"full K elements"),dtype_k);
      const u128 full_v_raw=dtype_k=="int4"?packed_kv_grid(s,tk,hd,feature_tile):bytes(mul128(kv_sum,hd,"full V elements"),dtype_k);
      const u128 fused_k_raw=dtype_k=="int4"?
          (causal?packed_causal_prefix(s,tq,tk,hd,hd,pr):mul128(packed_kv_grid(s,tk,hd,hd),qtiles,"attention packed K rereads")):
          bytes(mul128(pr,hd,"fused K elements"),dtype_k);
      const u128 fused_v_raw=dtype_k=="int4"?
          (causal?packed_causal_prefix(s,tq,tk,hd,feature_tile,pr):mul128(packed_kv_grid(s,tk,hd,feature_tile),qtiles,"attention packed V rereads")):
          bytes(mul128(pr,hd,"fused V elements"),dtype_k);
      const u128 full_k_read=mul128(mul128(full_k_raw,qtiles,"full K query tiles"),qh,"full K heads");
      const u128 full_v_read=mul128(mul128(full_v_raw,qtiles,"full V query tiles"),qh,"full V heads");
      const u128 fused_k_read=mul128(fused_k_raw,qh,"fused K heads");
      const u128 fused_v_read=mul128(fused_v_raw,qh,"fused V heads");
      const u128 span_kread=mode=="av"?0:(materialized?full_k_read:fused_k_read);
      const u128 span_vread=mode=="qk"?0:(materialized?full_v_read:fused_v_read);
      qk_kbytes=add128(qk_kbytes,span_kread,"attention QK K reads");
      pv_vbytes=add128(pv_vbytes,span_vread,"attention PV V reads");
      kvbytes=add128(kvbytes,add128(span_kread,span_vread,"attention K/V reads"),"attention K/V reads");
      if(mode!="av")k_quant_elements=add128(k_quant_elements,mul128(mul128(pr,hd,"attention K conversions"),qh,"attention K conversions"),"attention K conversions");
      if(mode!="qk")v_quant_elements=add128(v_quant_elements,mul128(mul128(pr,hd,"attention V conversions"),qh,"attention V conversions"),"attention V conversions");
      if(int4path){
        const u128 packed_k_one=causal&&!materialized?packed_causal_prefix(s,tq,tk,hd,hd,pr):mul128(packed_kv_grid(s,tk,hd,hd),qtiles,"attention packed K input");
        const u128 packed_v_one=causal&&!materialized?packed_causal_prefix(s,tq,tk,hd,feature_tile,pr):mul128(packed_kv_grid(s,tk,hd,feature_tile),qtiles,"attention packed V input");
        if(mode!="av"){
          k_packed=add128(k_packed,mul128(packed_k_one,qh,"attention packed K heads"),"attention packed K");
        }
        if(mode!="qk"){
          v_packed=add128(v_packed,mul128(packed_v_one,qh,"attention packed V heads"),"attention packed V");
        }
        if(dtype_q!="int4"&&mode!="av")q_scale_tiles=add128(q_scale_tiles,mul128(mul128(qtiles,q_head_copies,"attention Q scales"),q_reuses,"attention Q scale rereads"),"attention Q scales");
        if(dtype_k!="int4"&&mode!="av")k_scale_tiles=add128(k_scale_tiles,mul128(tile_blocks,qh,"attention K head scales"),"attention K scales");
        if(dtype_k!="int4"&&mode!="qk")v_scale_tiles=add128(v_scale_tiles,mul128(tile_blocks,qh,"attention V head scales"),"attention V scales");
      }
      if (mode!="qk")outbytes=add128(outbytes,mul128(tile_grid_bytes(s.q,tq,hd,hd,dtype_o),request_heads,"attention tiled output bytes"),"attention output bytes");
      if(materialized&&mode=="av")av_probbytes=add128(av_probbytes,mul128(mul128(sl,4,"AV probability bytes"),qh,"AV probability bytes"),"AV probability bytes");

      // Logical bytes are unique source tensors. Physical rereads, score
      // materialization, and conversion buffers stay in their own counters.
      const u128 qlogical=mul128(bytes(mul128(s.q,hd,"logical Q elements"),dtype_q),request_heads,"logical Q bytes");
      const u128 kvlogical=mul128(sum_storage_tensor_bytes(s,hd,dtype_k),mul128(kvh,2,"logical KV tensors"),"logical KV bytes");
      const u128 outlogical=mul128(bytes(mul128(s.q,hd,"logical output elements"),dtype_o),request_heads,"logical output bytes");
      if(mode=="both")logical=add128(logical,add128(add128(qlogical,kvlogical,"logical attention tensors"),outlogical,"logical attention tensors"),"logical attention tensors");
      else if(mode=="qk")logical=add128(logical,add128(qlogical,mul128(bytes(sl,"fp32"),qh,"logical score bytes"),"logical QK tensors"),"logical QK tensors");
      else logical=add128(logical,add128(add128(mul128(bytes(sl,"fp32"),qh,"logical probability bytes"),mul128(sum_storage_tensor_bytes(s,hd,dtype_k),kvh,"logical V bytes"),"logical AV tensors"),outlogical,"logical AV tensors"),"logical AV tensors");
    }
    slots=mul128(slots,executions,"attention executions");
    valid=mul128(valid,executions,"attention source executions");
    queryrows=mul128(queryrows,executions,"attention query executions");
    qbytes=mul128(qbytes,executions,"attention Q passes");
    kvbytes=mul128(kvbytes,executions,"attention KV passes");
    outbytes=mul128(outbytes,executions,"attention output passes");
    prefixes=mul128(prefixes,executions,"attention KV prefixes");
    logical=mul128(logical,executions,"attention logical source tensors");
    qk_qbytes=mul128(qk_qbytes,executions,"full QK query rereads");
    qk_kbytes=mul128(qk_kbytes,executions,"full QK key reads");
    pv_vbytes=mul128(pv_vbytes,executions,"full PV value reads");
    av_probbytes=mul128(av_probbytes,executions,"AV probability reads");
    q_packed=mul128(q_packed,executions,"attention packed Q bytes");
    k_packed=mul128(k_packed,executions,"attention packed K bytes");
    v_packed=mul128(v_packed,executions,"attention packed V bytes");
    q_quant_elements=mul128(q_quant_elements,executions,"attention Q conversion elements");
    k_quant_elements=mul128(k_quant_elements,executions,"attention K conversion elements");
    v_quant_elements=mul128(v_quant_elements,executions,"attention V conversion elements");
    q_scale_tiles=mul128(q_scale_tiles,executions,"attention Q scale tiles");
    k_scale_tiles=mul128(k_scale_tiles,executions,"attention K scale tiles");
    v_scale_tiles=mul128(v_scale_tiles,executions,"attention V scale tiles");
    tile_block_visits=mul128(tile_block_visits,executions,"attention KV tile visits");
    row_block_visits=mul128(row_block_visits,executions,"attention row/KV tile visits");
    active_queryrows=mul128(active_queryrows,executions,"attention active query rows");
    const u64 active=std::max<u64>(1,std::min(ncore,as_u64(output_tasks,"attention output tasks")));
    const u128 matrix_factor=mode=="both"?4:2;
    const u128 matops=mul128(matrix_factor,mul128(hd,slots,"attention matrix ops"),"attention matrix ops");
    m.source_mops=mul128(matrix_factor,mul128(hd,valid,"source attention ops"),"source attention ops");
    m.executed_mops=matops;
    const u128 online_rows=row_block_visits>=active_queryrows?row_block_visits-active_queryrows:0;
    const u128 online_eq=full?0:mul128(online_rows,add128(mul128(2,hd,"attention online state eqops"),4,"attention online state eqops"),"attention online state eqops");
    const u128 eq=mode!="both"?0:add128(add128(mul128(slots,13,"attention softmax eqops"),mul128(queryrows,hd,"attention normalization eqops"),"attention Vec eqops"),online_eq,"attention Vec eqops");
    const u128 source_eq=mode=="both"?add128(mul128(valid,13,"attention source Vec ops"),mul128(queryrows,hd,"attention source normalization"),"attention source Vec ops"):0;
    m.source_vops=source_eq;
    const u128 qk_ops=mode=="av"?0:mul128(2,mul128(hd,slots,"attention QK ops"),"attention QK ops");
    const u128 pv_ops=mode=="qk"?0:mul128(2,mul128(hd,slots,"attention PV ops"),"attention PV ops");
    const u128 int4_input_bytes=int4path?add128(add128(q_packed,k_packed,"attention packed QK bytes"),v_packed,"attention packed inputs"):0;
    m.int4bytes=int4_input_bytes;
    const u128 quant_input_elements=int4path?
        add128(dtype_q=="int4"?0:q_quant_elements,
               add128(dtype_k=="int4"?0:k_quant_elements,dtype_k=="int4"?0:v_quant_elements,"attention quant input elements"),"attention quant input elements"):
        add128(dtype_q=="fp32"?0:q_quant_elements,
               add128(dtype_k=="fp32"?0:k_quant_elements,dtype_k=="fp32"?0:v_quant_elements,"attention FP32 conversion elements"),"attention FP32 conversion elements");
    const u128 qk_acc_elements=mode=="av"?0:mul128(slots,ceildiv(hd,64),"attention QK partial conversions");
    const u128 pv_acc_elements=mode=="qk"?0:mul128(row_block_visits,hd,"attention PV partial conversions");
    const u128 acc_convert_elements=int4path?add128(qk_acc_elements,pv_acc_elements,"attention accumulator conversions"):0;
    const u128 qk_acc_adds=int4path?mul128(slots,ceildiv(hd,64)-1,"attention QK partial accumulation"):0;
    const u128 pv_first=mul128(active_queryrows,hd,"attention PV first partial");
    const u128 pv_acc_adds=int4path&&pv_acc_elements>pv_first?pv_acc_elements-pv_first:0;
    const u128 acc_adds=int4path?add128(qk_acc_adds,pv_acc_adds,"attention accumulator additions"):0;
    const u128 output_conversion=(mode=="qk"||dtype_o=="fp32")?0:mul128(queryrows,hd,"attention output conversions");
    const u128 accumulator_vec_ops=int4path?add128(mul128(acc_convert_elements,static_cast<u64>(kappa_acc),"attention accumulator Vec ops"),acc_adds,"attention accumulator Vec ops"):0;
    const u128 cast_vec_ops=output_conversion;
    const u128 vector_matrix_ops=int4path?0:matops;
    auto input_conversion_count=[&](const std::string& dt,u128 elements)->u128 {
      return int4path?(dt=="int4"?0:elements):(dt=="fp32"?0:elements);
    };
    const u128 q_input_conversion=input_conversion_count(dtype_q,q_quant_elements);
    const u128 k_input_conversion=input_conversion_count(dtype_k,k_quant_elements);
    const u128 v_input_conversion=input_conversion_count(dtype_k,v_quant_elements);
    const u128 qk_input_conversion=add128(q_input_conversion,k_input_conversion,"attention QK input conversions");
    const u128 pv_input_conversion=v_input_conversion;
    const u128 qk_quant_vec=int4path?mul128(qk_input_conversion,static_cast<u64>(kappa_quant),"attention QK quant Vec ops"):qk_input_conversion;
    const u128 pv_quant_vec=int4path?mul128(pv_input_conversion,static_cast<u64>(kappa_quant),"attention PV quant Vec ops"):pv_input_conversion;
    const u128 qk_acc_vec=int4path?add128(mul128(qk_acc_elements,static_cast<u64>(kappa_acc),"attention QK accumulator Vec ops"),qk_acc_adds,"attention QK accumulator Vec ops"):0;
    const u128 pv_acc_vec=int4path?add128(mul128(pv_acc_elements,static_cast<u64>(kappa_acc),"attention PV accumulator Vec ops"),pv_acc_adds,"attention PV accumulator Vec ops"):0;
    const u128 conversion_elements=int4path?add128(add128(quant_input_elements,acc_convert_elements,"attention conversion elements"),output_conversion,"attention conversion elements"):add128(add128(qk_input_conversion,pv_input_conversion,"attention conversion elements"),output_conversion,"attention conversion elements");
    m.conversion_elements=conversion_elements;
    const u128 input_conversion_ops=int4path?mul128(quant_input_elements,static_cast<u64>(kappa_quant),"attention quantization Vec ops"):add128(qk_input_conversion,pv_input_conversion,"attention input conversion Vec ops");
    const bool staged_full=full||mode!="both";
    const u128 scoreb=mul128(slots,4,"score bytes");
    const u128 probb=mode=="qk"?0:((mode=="av"||staged_full)?mul128(slots,4,"probability bytes"):0);
    const u128 feature_tiles=ceildiv(hd,std::min<u64>(hd,64));
    const u128 fp32_output_bytes=mul128(mul128(queryrows,hd,"attention FP32 output"),4,"attention FP32 output");
    const u128 cast_bytes=dtype_o=="fp32"?0:fp32_output_bytes;
    const u128 qk_read=mode=="av"?0:add128(qk_qbytes,qk_kbytes,"attention QK reads");
    const u128 qk_write=mode=="av"?0:scoreb;
    const u128 soft_read=staged_full&&mode=="both"?scoreb:0;
    const u128 soft_write=staged_full&&mode=="both"?probb:0;
    const u128 full_pv_read=add128(pv_vbytes,mul128(probb,feature_tiles,"full PV probability rereads"),"full PV reads");
    const u128 pv_read=mode=="qk"?0:(mode=="av"?add128(pv_vbytes,mul128(av_probbytes,feature_tiles,"AV probability rereads"),"AV reads"):full_pv_read);
    const u128 pv_write=mode=="qk"?0:(staged_full&&mode=="both"?(cast_bytes?cast_bytes:outbytes):outbytes);
    const u128 cast_read=staged_full&&mode=="both"?cast_bytes:0;
    const u128 cast_write=staged_full&&mode=="both"?outbytes:0;
    const u128 fused_read=staged_full?0:add128(qbytes,kvbytes,"fused attention reads");
    const u128 fused_write=staged_full?0:outbytes;
    const u128 totalread=staged_full&&mode=="both"?add128(add128(qk_read,soft_read,"full attention reads"),add128(pv_read,cast_read,"full attention reads"),"full attention reads"):
        (mode=="qk"?qk_read:(mode=="av"?pv_read:fused_read));
    const u128 totalwrite=staged_full&&mode=="both"?add128(add128(qk_write,soft_write,"full attention writes"),add128(pv_write,cast_write,"full attention writes"),"full attention writes"):
        (mode=="qk"?qk_write:(mode=="av"?pv_write:fused_write));
    const u128 sb=mode=="qk"?scoreb:(staged_full&&mode=="both"?add128(add128(scoreb,probb,"materialized score/probability workspace"),cast_bytes,"materialized attention workspace"):0);
    m.workspace=sb;m.score_bytes=(mode=="qk"||(staged_full&&mode=="both"))?scoreb:0;m.probability_bytes=probb;
    m.score_slots=slots;m.valid_pairs=valid;m.logical=logical;
    m.hbm_read=0;m.hbm_write=0;
    const u64 rounds=std::max<u64>(1,ceildiv(as_u64(output_tasks,"attention output tasks"),active));
    const u64 waves=as_u64(std::max<u128>(1,staged_full?add128(output_tasks,ceildiv(as_u64(slots,"attention score slots"),std::max<u64>(1,active*256)),"full attention waves"):row_block_visits),"attention waves");
    auto emit_stage=[&](const std::string& name,u128 read,u128 write,u128 c_read,u128 c_write,
                        u128 i4,u128 vec,u64 use_active,u64 frames,double hbm_eta){
      Flow load,compute,store;
      add_hbm(load,hw,topo,read,use_active,false,hbm_eta);
      add_bytes_work(load,SRAM_WRITE,read,"attention load landing bytes");
      add_bytes_work(compute,SRAM_READ,c_read,"attention compute SRAM reads");
      add_bytes_work(compute,SRAM_WRITE,c_write,"attention compute SRAM writes");
      add_bytes_work(compute,LOCAL,add128(c_read,c_write,"attention LOCAL bytes"),"attention LOCAL bytes");
      compute.int4=i4;compute.vec=vec;compute.compute_s=compute_time(hw,i4,vec,use_active);
      add_bytes_work(store,SRAM_READ,write,"attention store SRAM reads");
      add_hbm(store,hw,topo,write,use_active,true,hbm_eta);
      append_epoch(*p,hw,std::move(load),std::move(compute),std::move(store),use_active,frames,name);
      m.hbm_read=add128(m.hbm_read,read,"attention HBM reads");m.hbm_write=add128(m.hbm_write,write,"attention HBM writes");
      m.sram_write=add128(m.sram_write,add128(read,c_write,"attention SRAM writes"),"attention SRAM writes");
      m.sram_read=add128(m.sram_read,add128(c_read,write,"attention SRAM reads"),"attention SRAM reads");
      m.local=add128(m.local,add128(c_read,c_write,"attention LOCAL bytes"),"attention LOCAL bytes");
      m.int4ops=add128(m.int4ops,i4,"attention INT4 ops");m.vecops=add128(m.vecops,vec,"attention Vec ops");
      m.output_rounds=add128(m.output_rounds,frames,"attention output rounds");m.waves=add128(m.waves,frames,"attention waves");
      m.active_cores=std::max(m.active_cores,use_active);
    };
    // Only a storage-to-INT4 conversion creates a second operand buffer.
    // Native INT4 already arrived through the HBM load ledger and is consumed
    // directly from SRAM; charging it again as a generated pack would invent
    // a write and reread for a conversion that did not happen.
    const u128 qk_packed_bytes=int4path?add128(
        dtype_q=="int4"?0:q_packed,dtype_k=="int4"?0:k_packed,
        "attention QK generated INT4 inputs"):0;
    const u128 pv_packed_bytes=int4path&&dtype_k!="int4"?v_packed:0;
    const u128 qk_scale_bytes=int4path?mul128(add128(q_scale_tiles,k_scale_tiles,"attention QK scale tiles"),4,"attention QK scales"):0;
    const u128 pv_scale_bytes=int4path?mul128(v_scale_tiles,4,"attention PV scales"):0;
    const u128 qk_partial_bytes=int4path?mul128(qk_acc_elements,4,"attention QK INT32 partials"):0;
    const u128 pv_partial_bytes=int4path?mul128(pv_acc_elements,4,"attention PV INT32 partials"):0;
    const u128 qk_fp32_input_bytes=!int4path?mul128(qk_input_conversion,4,"attention QK FP32 conversion buffers"):0;
    const u128 pv_fp32_input_bytes=!int4path?mul128(pv_input_conversion,4,"attention PV FP32 conversion buffers"):0;
    const u128 qk_vec=int4path?add128(qk_quant_vec,qk_acc_vec,"attention QK Vec ops"):add128(qk_ops,qk_input_conversion,"attention QK Vec ops");
    const u128 pv_vec=int4path?add128(pv_quant_vec,pv_acc_vec,"attention PV Vec ops"):add128(pv_ops,pv_input_conversion,"attention PV Vec ops");
    const u128 soft_eq=eq;
    if(staged_full&&mode=="both"){
      // Full materialization is four ordered primitives. Each stage owns its
      // HBM boundary and its actual SRAM landing, consumption, and writeback.
      const u128 qk_c_read=add128(add128(qk_read,qk_packed_bytes,"QK reads"),add128(qk_scale_bytes,add128(qk_partial_bytes,qk_fp32_input_bytes,"QK scratch reads"),"QK reads"),"QK reads");
      const u128 qk_c_write=add128(add128(qk_write,qk_packed_bytes,"QK writes"),add128(qk_scale_bytes,add128(qk_partial_bytes,qk_fp32_input_bytes,"QK scratch writes"),"QK writes"),"QK writes");
      emit_stage("full_score_qk",qk_read,qk_write,qk_c_read,qk_c_write,int4path?qk_ops:0,qk_vec,active,rounds,eta(hw,"eta_hbm_matrix"));
      emit_stage("full_score_softmax",soft_read,soft_write,soft_read,soft_write,0,soft_eq,active,
                 std::max<u64>(1,ceildiv(as_u64(slots,"softmax slots"),std::max<u64>(1,active*256))),eta(hw,"eta_hbm_vector"));
      const u128 pv_c_read=add128(add128(pv_read,pv_packed_bytes,"PV reads"),add128(pv_scale_bytes,add128(pv_partial_bytes,pv_fp32_input_bytes,"PV scratch reads"),"PV reads"),"PV reads");
      const u128 pv_c_write=add128(add128(pv_write,pv_packed_bytes,"PV writes"),add128(pv_scale_bytes,add128(pv_partial_bytes,pv_fp32_input_bytes,"PV scratch writes"),"PV writes"),"PV writes");
      emit_stage("full_score_pv",pv_read,pv_write,pv_c_read,pv_c_write,int4path?pv_ops:0,pv_vec,active,rounds,eta(hw,"eta_hbm_matrix"));
      if(cast_bytes)emit_stage("full_score_output_cast",cast_read,cast_write,cast_read,cast_write,0,cast_vec_ops,active,
          std::max<u64>(1,ceildiv(as_u64(queryrows*hd,"output cast elements"),std::max<u64>(1,active*256))),eta(hw,"eta_hbm_vector"));
    }else if(staged_full&&mode=="qk"){
      const u128 c_read=add128(add128(qk_read,qk_packed_bytes,"QK reads"),add128(qk_scale_bytes,add128(qk_partial_bytes,qk_fp32_input_bytes,"QK scratch reads"),"QK reads"),"QK reads");
      const u128 c_write=add128(add128(qk_write,qk_packed_bytes,"QK writes"),add128(qk_scale_bytes,add128(qk_partial_bytes,qk_fp32_input_bytes,"QK scratch writes"),"QK writes"),"QK writes");
      emit_stage("attention_qk_materialized",qk_read,qk_write,c_read,c_write,int4path?qk_ops:0,qk_vec,active,rounds,eta(hw,"eta_hbm_matrix"));
    }else if(staged_full&&mode=="av"){
      const u128 c_read=add128(add128(pv_read,pv_packed_bytes,"AV reads"),add128(pv_scale_bytes,add128(pv_partial_bytes,add128(pv_fp32_input_bytes,cast_bytes,"AV scratch reads"),"AV scratch reads"),"AV reads"),"AV reads");
      const u128 c_write=add128(add128(add128(pv_write,cast_bytes,"AV writes"),pv_packed_bytes,"AV writes"),add128(pv_scale_bytes,pv_partial_bytes,"AV scratch writes"),"AV writes");
      emit_stage("attention_av_materialized",pv_read,pv_write,c_read,c_write,int4path?pv_ops:0,add128(pv_vec,cast_vec_ops,"AV Vec ops"),active,rounds,eta(hw,"eta_hbm_matrix"));
    }else{
      const u128 score_bytes=mul128(scoreb,2,"fused score/probability scratch");
      const u128 state_unit=add128(mul128(hd,4,"online state vector"),8,"online max/sum state");
      const u128 state_read=mul128(online_rows,state_unit,"online state reads");
      const u128 state_write=add128(mul128(active_queryrows,state_unit,"online state initialization"),mul128(online_rows,state_unit,"online state updates"),"online state writes");
      const u128 partial_bytes=add128(qk_partial_bytes,pv_partial_bytes,"fused partial traffic");
      const u128 scale_bytes=add128(qk_scale_bytes,pv_scale_bytes,"fused scale traffic");
      const u128 converted_input_bytes=int4path
          ?add128(qk_packed_bytes,pv_packed_bytes,"fused generated INT4 buffers")
          :add128(qk_fp32_input_bytes,pv_fp32_input_bytes,"fused FP32 conversion buffers");
      const u128 c_read=add128(add128(fused_read,converted_input_bytes,"fused converted matrix input reads"),add128(scale_bytes,add128(partial_bytes,add128(score_bytes,state_read,"fused scratch reads"),"fused scratch reads"),"fused scratch reads"),"fused compute reads");
      const u128 c_write=add128(add128(fused_write,converted_input_bytes,"fused converted matrix input writes"),add128(scale_bytes,add128(partial_bytes,add128(score_bytes,state_write,"fused scratch writes"),"fused scratch writes"),"fused scratch writes"),"fused compute writes");
      const u128 vec=int4path?add128(add128(eq,input_conversion_ops,"fused quant Vec ops"),add128(accumulator_vec_ops,cast_vec_ops,"fused accumulator Vec ops"),"fused Vec ops"):
          add128(add128(eq,vector_matrix_ops,"strict Vec attention ops"),add128(add128(qk_input_conversion,pv_input_conversion,"strict input conversions"),cast_vec_ops,"strict conversions"),"strict Vec attention ops");
      emit_stage("fused_attention",fused_read,fused_write,c_read,c_write,int4path?matops:0,vec,active,rounds,eta(hw,"eta_hbm_vector"));
      m.waves=waves;
    }

    const u64 tile_q=std::min<u64>(tq,maxq),tile_k=std::min<u64>(tk,maxkv),feature_tile=std::min<u64>(hd,64);
    const u128 metadata=64;
    const u128 raw_q=mode=="av"?0:tile_bytes(tile_q,hd,dtype_q);
    const u128 raw_k=mode=="av"?0:tile_bytes(tile_k,hd,dtype_k);
    const u128 raw_v=mode=="qk"?0:tile_bytes(tile_k,hd,dtype_k);
    const u128 raw_v_feature=mode=="qk"?0:tile_bytes(tile_k,feature_tile,dtype_k);
    auto converted_tile=[&](const std::string& dt,u64 rows,u64 cols)->u128 {
      if(int4path){
        if(dt=="int4")return 0;
        return add128(tile_bytes(rows,cols,"int4"),4,"attention converted INT4 tile and scale");
      }
      return dt=="fp32"?0:tile_bytes(rows,cols,"fp32");
    };
    const u128 q_converted_live=mode=="av"?0:converted_tile(dtype_q,tile_q,hd);
    const u128 k_converted_live=mode=="av"?0:converted_tile(dtype_k,tile_k,hd);
    const u128 v_converted_live=mode=="qk"?0:converted_tile(dtype_k,tile_k,feature_tile);
    const u128 score_tile=mul128(mul128(tile_q,tile_k,"attention score tile"),4,"attention score tile");
    const u128 probability_tile=score_tile;
    const u128 qk_partial_live=int4path&&mode!="av"
        ?mul128(mul128(mul128(tile_q,tile_k,"QK partial tile"),ceildiv(hd,64),"QK feature partials"),4,"QK partial tile"):0;
    const u128 pv_partial_live=mode!="qk"
        ?mul128(mul128(tile_q,feature_tile,"PV output tile"),int4path?12:4,"PV partial/accumulator tile"):0;
    const u128 online_live=mode=="both"&&!staged_full
        ?mul128(tile_q,add128(mul128(hd,4,"online output tile"),8,"online state tile"),"online state tile"):0;
    const u128 qk_peak=mode=="av"?0:add128(
        add128(raw_q,raw_k,"QK raw live tiles"),
        add128(add128(q_converted_live,k_converted_live,"QK converted live tiles"),
               add128(qk_partial_live,add128(score_tile,metadata,"QK score output tile"),"QK live scratch"),
               "QK live scratch"),"QK live peak");
    const u128 softmax_peak=mode=="both"&&staged_full
        ?add128(add128(score_tile,probability_tile,"softmax score/probability tiles"),
                add128(mul128(tile_q,8,"softmax row statistics"),metadata,"softmax live metadata"),
                "softmax stage peak"):0;
    const u128 pv_output_live=(staged_full&&mode=="both"&&cast_bytes)
        ?tile_bytes(tile_q,hd,"fp32")
        :(cast_bytes?tile_bytes(tile_q,hd,"fp32"):tile_bytes(tile_q,hd,dtype_o));
    const u128 pv_peak=mode=="qk"?0:add128(
        add128(raw_v_feature,v_converted_live,"PV raw/converted V tiles"),
        add128(add128(probability_tile,pv_partial_live,"PV probability/partial tiles"),
               add128(pv_output_live,metadata,"PV output tile and metadata"),"PV live scratch"),
        "PV stage peak");
    const u128 cast_peak=staged_full&&mode=="both"&&cast_bytes
        ?add128(add128(tile_bytes(tile_q,hd,"fp32"),tile_bytes(tile_q,hd,dtype_o),"output cast tiles"),metadata,"output cast peak")
        :0;
    const u128 fused_peak=add128(
        add128(add128(raw_q,raw_k,"fused raw QK tiles"),raw_v,"fused raw QKV tiles"),
        add128(add128(add128(q_converted_live,k_converted_live,"fused converted QK tiles"),v_converted_live,"fused converted QKV tiles"),
               add128(add128(mul128(mul128(tile_q,tile_k,"fused score cells"),8,"fused score/probability tiles"),qk_partial_live,"fused QK scratch"),
                      add128(pv_partial_live,add128(online_live,metadata,"fused state and metadata"),"fused PV scratch"),"fused scratch"),
               "fused live buffers"),"fused peak");
    m.sram_peak=staged_full
        ?std::max(std::max(qk_peak,softmax_peak),std::max(pv_peak,cast_peak))
        :fused_peak;
    if(m.hbm_read!=totalread||m.hbm_write!=totalwrite)
      fail("mapping_invariant: attention stage HBM bytes do not match its access ledger");
    m.active_cores=std::max(m.active_cores,active);
    m.score_slots=slots;m.valid_pairs=valid;m.score_bytes=(mode=="qk"||(staged_full&&mode=="both"))?scoreb:0;m.probability_bytes=probb;
    m.logical=logical;
    m.template_id=mode=="both"?(staged_full?"full_score_materialization_v2":"fused_q_head_shard_v2"):"attention_"+mode+"_materialized_v2";
    m.assumptions={"engineering_assumption","GQA_head_mapping","absolute_position_causal_mask","not_run_granularity_validation"};
    m.tm=tq; m.tn=tk; m.tk=tk;
    if (static_cast<long double>(as_u64(m.sram_peak,"attention SRAM peak")) > perf(hw,"rho_sram",0.9)*dd(hw,"core","sram_bytes_per_core")) continue;
    finalize_summary(*p,m,hw); p->summary["template_id"]=m.template_id; p->summary["dataflow_id"]=staged_full?"full_score_materialization_v2":"fused_q_head_shard_v2";
    p->summary["executed_matrix_ops"]=as_u64(m.executed_mops,"attention ops"); p->summary["source_matrix_ops"]=as_u64(m.source_mops,"attention source ops");
    p->summary["recomputed_matrix_ops"]=as_u64(m.executed_mops>=m.source_mops?m.executed_mops-m.source_mops:0,"attention recompute"); p->summary["recompute_matrix_ops"]=p->summary["recomputed_matrix_ops"];
    p->summary["executed_score_slots"]=as_u64(slots,"attention slots"); p->summary["valid_score_pairs"]=as_u64(valid,"attention pairs");
    p->summary["score_bytes"]=as_u64(m.score_bytes,"score bytes"); p->summary["probability_bytes"]=as_u64(m.probability_bytes,"probability bytes");
    p->summary["tm"]=tq; p->summary["tn"]=tk; p->summary["tk"]=tk;
    p->fingerprint=plan_to_json(*p).at("fingerprint").get<std::string>(); result.push_back(std::move(p));
  }
  return result;
}

}  // namespace

std::vector<std::shared_ptr<Plan>> enumerate_mappings(
    const Hardware& hardware, const nlohmann::json& task) {
  if (!task.is_object() || !task.contains("op_kind")) fail("task: normalized op_kind is required");
  const std::string op=gets(task,"op_kind");
  const u64 nx=du(hardware,"die","core_columns_nx"), ny=du(hardware,"die","core_rows_ny");
  const Topology topo=topology(hardware,nx,ny);
  if (op=="gemm") return enumerate_gemm(hardware,task,topo,shape_u(task,"m"),shape_u(task,"n"),shape_u(task,"k"));
  if (op=="attention" || op=="attention_region") {
    const std::string fusion=gets(task,"fusion_policy","auto");
    std::vector<std::shared_ptr<Plan>> result;
    if (fusion!="materialized") {
      auto fused=enumerate_attention(hardware,task,topo,false);
      result.insert(result.end(),fused.begin(),fused.end());
    }
    if (fusion!="fused") {
      auto materialized=enumerate_attention(hardware,task,topo,true);
      result.insert(result.end(),materialized.begin(),materialized.end());
    }
    return result;
  }

  const u64 M=shape_u(task,"m",1), H=shape_u(task,"hidden",1), F=shape_u(task,"ffn",H), V=shape_u(task,"vocab",H);
  const u64 layers=getu(task,"layer_instances",1), passes=getu(task,"weight_passes",1);
  const u64 ncore=du(hardware,"die","core_count");
  const u64 active=std::max<u64>(1,std::min(ncore,M));
  const std::string ad=dtype(task,"activations"), wd=dtype(task,"weights"), od=dtype(task,"output");
  const u128 elems=mul128(M,H,"vector elements");
  auto p=new_plan(hardware,task); Metrics m; m.assumptions={"engineering_assumption","fixed_equivalent_operation_template"};

  if (op=="norm" || op=="residual" || op=="mlp_activation" || op=="scale_mask" || op=="softmax" || op=="qkv_rope") {
    u128 count=elems, eq=0, extra=0;
    if (op=="norm") {
      const std::string norm=gets(task.at("tensor_shapes"),"norm_type","rmsnorm");
      if(norm!="rmsnorm"&&norm!="layernorm")fail("unsupported_op_kind: Norm variant "+norm);
      eq=mul128(norm=="rmsnorm"?6:8,count,"Norm eqops");
      extra=bytes(mul128(H,norm=="rmsnorm"?1:2,"Norm parameters"),dtype(task,"parameters"));
    }
    else if (op=="residual") { eq=count; extra=bytes(elems,ad); }
    else if (op=="mlp_activation") {
      const std::string variant=gets(task.at("tensor_shapes"),"mlp_variant","swiglu");
      if(variant!="swiglu"&&variant!="gelu")fail("unsupported_op_kind: activation variant "+variant);
      count=mul128(M,F,"activation elements");
      eq=mul128(variant=="swiglu"?16:15,count,"activation eqops");
      if(variant=="swiglu")extra=bytes(count,ad);
    }
    else if (op=="scale_mask"||op=="softmax") {
      u128 slots=0;
      const u64 heads=shape_u(task,"q_heads",1);
      for(const auto& s:spans_of(task)) {
        const u128 lengths=add128(mul128(s.kv,s.count,"score KV lengths"),mul128(s.kv_stride,mul128(s.count,s.count-1,"score KV progression")/2,"score KV progression"),"score KV lengths");
        slots=add128(slots,mul128(mul128(s.q,lengths,"score rectangles"),heads,"score heads"),"score count");
      }
      count=slots;eq=mul128(op=="scale_mask"?2:10,count,"score Vec eqops");
      if(dtype(task,"scores","fp32")!="fp32")fail("unsupported_dtype: materialized score transforms require fp32");
    }
    else if (op=="qkv_rope") {
      const u64 qh=shape_u(task,"q_heads",1), kvh=shape_u(task,"kv_heads",1), hd=shape_u(task,"head_dim",H);
      const u64 nout=as_u64(add128(mul128(qh,hd,"QKV output"),mul128(2,mul128(kvh,hd,"QKV output"),"QKV output"),"QKV output"),"QKV output");
      p->task["tensor_shapes"]["m"]=M; p->task["tensor_shapes"]["n"]=nout; p->task["tensor_shapes"]["k"]=H;
      p->task["op_kind"]="gemm";
      auto plans=enumerate_gemm(hardware,p->task,topo,M,nout,H);
      const auto& shape=task.at("tensor_shapes");
      const bool has_rope=gets(shape,"position_encoding","rope")!="none";
      const u128 executions=mul128(layers,passes,"QKV/RoPE executions");
      const u128 rope_one=mul128(M,mul128(add128(qh,kvh,"RoPE heads"),hd,"RoPE elements"),"RoPE elements");
      const u128 rope_count=has_rope?mul128(rope_one,executions,"RoPE elements"):0;
      const u128 rope_ops=mul128(3,rope_count,"RoPE eqops");
      for (auto& q:plans) {
        q->task=task;
        const auto gemm_summary=q->summary;
        if(has_rope) {
          // A fixed materialized transform keeps the Q/K readiness boundary:
          // QKV writes its outputs before RoPE reads/updates Q and K. V is
          // untouched. It is not a free summary-only Vec operation.
          auto rope=new_plan(hardware,task); Metrics r;
          const u128 payload=mul128(bytes(rope_one,od),executions,"RoPE payload");
          const u128 parameters=mul128(bytes(mul128(2,mul128(M,hd,"RoPE angles"),"RoPE angles"),dtype(task,"parameters")),executions,"RoPE parameter reads");
          add_vector(*rope,r,hardware,topo,rope_count,rope_ops,payload,payload,active,"rope_materialized_v2",parameters);
          if(static_cast<long double>(as_u64(r.sram_peak,"RoPE SRAM peak"))>perf(hardware,"rho_sram",0.9)*dd(hardware,"core","sram_bytes_per_core"))continue;
          finalize_summary(*rope,r,hardware);
          for(auto epoch:rope->epochs) {
            for(std::size_t i=0;i<epoch.activities.size();++i)
              epoch.activities[i].id=q->epochs.size()*3+i+1;
            q->epochs.push_back(std::move(epoch));
          }
          for(const char* key:{"source_vec_ops","vec_equivalent_ops","hbm_read_bytes_total","hbm_write_bytes_total","sram_read_bytes","sram_write_bytes","local_bus_bytes","output_round_count","consumption_wave_count","wave_count"})
            q->summary[key]=as_u64(add128(getu(q->summary,key),getu(rope->summary,key),std::string("QKV/RoPE ")+key),std::string("QKV/RoPE ")+key);
          // Q/K materialization is physical traffic inside this source region.
          // Its logical boundary adds only the cosine/sine parameter tensor.
          q->summary["logical_tensor_bytes"]=as_u64(add128(getu(q->summary,"logical_tensor_bytes"),parameters,"QKV/RoPE logical parameters"),"QKV/RoPE logical bytes");
          q->summary["physical_transfer_bytes"]=as_u64(add128(getu(q->summary,"hbm_read_bytes_total"),getu(q->summary,"hbm_write_bytes_total"),"QKV/RoPE physical bytes"),"QKV/RoPE physical bytes");
          q->summary["sram_peak_bytes_per_core"]=std::max(getu(q->summary,"sram_peak_bytes_per_core"),getu(rope->summary,"sram_peak_bytes_per_core"));
          q->summary["primitive_kernels"]=nlohmann::json::array({gemm_summary,rope->summary});
        }
        q->summary["kind"]="qkv_rope";
        q->summary["template_id"]=has_rope?"qkv_materialized_rope_v2":"qkv_projection_v2";
        q->summary["rope_eqops"]=as_u64(rope_ops,"RoPE eqops");
        q->summary["position_encoding"]=has_rope?"rope":"none";
        q->fingerprint.clear();
        q->fingerprint=plan_to_json(*q).at("fingerprint").get<std::string>();
      }
      plans.erase(std::remove_if(plans.begin(),plans.end(),[](const auto& q){return q->summary.value("kind",std::string())!="qkv_rope";}),plans.end());
      return plans;
    }
    const u128 executions=mul128(layers,passes,"vector executions");
    const bool score_transform=op=="scale_mask"||op=="softmax";
    const u128 inb=mul128(bytes(count,score_transform?"fp32":ad),executions,"vector input bytes");
    const u128 outb=mul128(bytes(count,score_transform?"fp32":od),executions,"vector output bytes");
    count=mul128(count,executions,"layer vector elements"); eq=mul128(eq,executions,"layer vector operations");
    extra=mul128(extra,executions,"vector extra input bytes");
    add_vector(*p,m,hardware,topo,count,eq,inb,outb,active,op+"_vec_v2",extra);
  } else if (op=="mlp_up" || op=="mlp_down" || op=="o_proj" || op=="logits") {
    u64 N=H,K=H;
    if(op=="mlp_up"){
      const std::string variant=gets(task.at("tensor_shapes"),"mlp_variant","swiglu");
      if(variant!="swiglu"&&variant!="gelu")fail("unsupported_op_kind: MLP variant "+variant);
      N=variant=="swiglu"?as_u64(mul128(2,F,"SwiGLU up/gate width"),"SwiGLU up/gate width"):F;K=H;
    } else if(op=="mlp_down"){N=H;K=F;} else if(op=="logits"){N=V;K=H;}
    const auto ts=options(M,{1,16,64,256}); const auto ns=options(N,{64,256,1024,4096}); const auto ks=options(K,{64,256,1024,4096});
    std::vector<std::shared_ptr<Plan>> result;
    for(bool wr:{false,true})for(u64 tm:ts)for(u64 tn:ns)for(u64 tk:ks){
      auto q=new_plan(hardware,task); Metrics z;z.assumptions={"engineering_assumption","bounded_tile_candidates"};
      add_gemm_epoch(*q,z,hardware,topo,M,N,K,ad,wd,od,layers,passes,GemmChoice{tm,tn,tk,wr},op+"_gemm_v2");
      if(static_cast<long double>(as_u64(z.sram_peak,"SRAM peak"))>perf(hardware,"rho_sram",0.9)*dd(hardware,"core","sram_bytes_per_core"))continue;
      finalize_summary(*q,z,hardware);q->summary["kind"]=op;q->summary["template_id"]=op+"_"+z.dataflow;q->summary["dataflow_id"]=z.dataflow;q->summary["tm"]=tm;q->summary["tn"]=tn;q->summary["tk"]=tk;
      q->fingerprint=plan_to_json(*q).at("fingerprint").get<std::string>();result.push_back(std::move(q));
    }
    return result;
  } else if (op=="qk" || op=="av") {
    auto task2=task;task2["op_kind"]="attention_region";
    auto att=enumerate_attention(hardware,task2,topo,true,op);
    for(auto& q:att){q->task=task;q->summary["kind"]=op;q->summary["template_id"]=op+"_"+q->summary.value("template_id",std::string());q->fingerprint.clear();q->fingerprint=plan_to_json(*q).at("fingerprint").get<std::string>();}
    return att;
  } else if (op=="ffn_region") {
    const std::string policy=gets(task,"fusion_policy","auto");
    const std::string variant=gets(task.at("tensor_shapes"),"mlp_variant","swiglu");
    if(variant!="swiglu"&&variant!="gelu")fail("unsupported_op_kind: FFN variant "+variant);
    const bool swiglu=variant=="swiglu";
    std::vector<std::shared_ptr<Plan>> result;
    // Materialized SwiGLU is represented as four fixed primitive epochs. Its
    // temporary is counted in HBM and SRAM; it is never expanded by tile count.
    if(policy!="fused") {
      auto q=new_plan(hardware,task);Metrics z;z.template_id="materialized_swiglu_v2";z.assumptions={"engineering_assumption","materialized_intermediate"};
      const std::string wdt=dtype(task,"weights"), xdt=dtype(task,"activations"), ydt=dtype(task,"output");
      const std::string middle_dtype=dtype(task,"activations");
      auto selected_gemm=[&](u64 n,u64 k,const std::string& input_dtype,const std::string& output_dtype) {
        auto child=task;
        child["op_kind"]="gemm";
        child["tensor_shapes"]={{"m",M},{"n",n},{"k",k}};
        child["storage_dtype"]["activations"]=input_dtype;
        child["storage_dtype"]["output"]=output_dtype;
        const auto choices=enumerate_gemm(hardware,child,topo,M,n,k);
        if(choices.empty())fail("unsupported_workspace: materialized FFN GEMM has no feasible candidate");
        using Key=std::tuple<u64,u128,u64,u64,std::string,u64,u64,u64,u64>;
        Key best_key{}; GemmChoice best{}; bool first=true;
        for(const auto& choice:choices) {
          const u64 latency=estimate_latency_ns(hardware,choice);
          const auto& summary=choice->summary;
          Key key{latency,add128(getu(summary,"hbm_read_bytes_total"),getu(summary,"hbm_write_bytes_total"),"FFN child HBM"),getu(summary,"hbm_workspace_peak_bytes"),getu(summary,"sram_peak_bytes_per_core"),gets(summary,"template_id"),getu(summary,"tm"),getu(summary,"tn"),getu(summary,"tk"),getu(summary,"th")};
          if(first||key<best_key) {
            best_key=std::move(key);first=false;
            best={getu(summary,"tm"),getu(summary,"tn"),getu(summary,"tk"),gets(summary,"dataflow_id")=="weight_resident_v2"};
          }
        }
        return best;
      };
      // The identical up/gate shapes share one deterministic child choice;
      // down is independently selected. No candidate Cartesian product.
      const GemmChoice c=selected_gemm(F,H,xdt,middle_dtype);
      const GemmChoice down=selected_gemm(H,F,middle_dtype,ydt);
      add_gemm_epoch(*q,z,hardware,topo,M,F,H,xdt,wdt,middle_dtype,layers,passes,c,"ffn_up_v2");
      if(swiglu)add_gemm_epoch(*q,z,hardware,topo,M,F,H,xdt,wdt,middle_dtype,layers,passes,c,"ffn_gate_v2");
      const u128 middle_one=mul128(M,F,"FFN intermediate");
      const u128 executions=mul128(layers,passes,"FFN executions");
      const u128 middle=mul128(middle_one,executions,"FFN intermediate work");
      const u128 middle_bytes=mul128(bytes(middle_one,middle_dtype),executions,"FFN intermediate bytes");
      // up/gate writes already belong to their GEMM store activities. The
      // activation reuses one intermediate allocation; only two tensors are
      // live at once, independent of sequential layer/pass repetitions.
      z.workspace=std::max(z.workspace,mul128(bytes(middle_one,middle_dtype),swiglu?2:1,"FFN workspace"));
      const u128 actops=mul128(swiglu?16:15,middle,"FFN activation eqops");
      add_vector(*q,z,hardware,topo,middle,actops,mul128(middle_bytes,swiglu?2:1,"FFN activation reads"),middle_bytes,active,"ffn_activation_v2");
      add_gemm_epoch(*q,z,hardware,topo,M,H,F,middle_dtype,wdt,ydt,layers,passes,down,"ffn_down_v2");
      finalize_summary(*q,z,hardware);q->summary["kind"]=op;q->summary["template_id"]=swiglu?"materialized_swiglu_v2":"materialized_gelu_v2";q->summary["fusion_policy"]="materialized";q->summary["recomputed_matrix_ops"]=0;q->summary["recompute_matrix_ops"]=0;
      const u128 logical_one=add128(add128(bytes(mul128(M,H,"FFN source input"),xdt),mul128(swiglu?3:2,bytes(mul128(H,F,"FFN source weights"),wdt),"FFN source weights"),"FFN source tensors"),bytes(mul128(M,H,"FFN source output"),ydt),"FFN source tensors");
      q->summary["logical_tensor_bytes"]=as_u64(mul128(logical_one,executions,"FFN source tensors"),"FFN source tensors");
      q->fingerprint=plan_to_json(*q).at("fingerprint").get<std::string>();result.push_back(std::move(q));
    }
    if(policy!="materialized"&&policy!="none") {
      if(!swiglu) {
        if(policy=="fused") fail("unsupported_mapping: fused_output_shard_v2 supports SwiGLU only");
      } else {
        // There are at most 4*4*2 output-shard candidates. For each, use the
        // first SRAM-feasible tk in descending order, as required by §6.5.
        const auto tms=options(M,{1,16,64,256});
        const auto tfs=options(F,{16,64,256,1024});
        const auto ths=options(H,{64,256});
        auto tk_order=options(H,{64,256,1024,4096});
        std::reverse(tk_order.begin(),tk_order.end());
        const u64 ncore=du(hardware,"die","core_count");
        const u128 executions=mul128(layers,passes,"FFN executions");
        const std::string execution_profile=gets(task,"execution_profile");
        const bool i4=execution_profile=="int4_matrix_vec_fp32_assumed_v1";
        const bool fp32vec=execution_profile=="fp32_vec_matrix_assumed_v1";
        constexpr u64 fused_metadata_bytes=64;
        auto live_tile_bytes=[&](u64 rows,u64 cols,const std::string& dt) {
          const u128 raw=tile_bytes(rows,cols,dt);
          if(i4&&dt!="int4")
            return add128(raw,add128(tile_bytes(rows,cols,"int4"),4,"FFN quant tile scale"),
                          "FFN raw and INT4 tile buffers");
          if(fp32vec&&dt!="fp32")
            return add128(raw,tile_bytes(rows,cols,"fp32"),
                          "FFN raw and FP32 tile buffers");
          return raw;
        };
        const u128 mh=mul128(M,H,"FFN tensor elements");
        const u128 hf=mul128(H,F,"FFN tensor elements");
        const u128 source_mops_one=mul128(6,mul128(mh,F,"FFN source matrix ops"),
                                          "FFN source matrix ops");
        const u128 source_vops_one=mul128(16,mul128(M,F,"FFN source activation"),
                                          "FFN source activation");
        for(u64 tm:tms) for(u64 tf:tfs) for(u64 th:ths) {
          const u64 Tm=ceildiv(M,tm), Tf=ceildiv(F,tf), Th=ceildiv(H,th);
          const u64 max_m=std::min(M,tm), max_f=std::min(F,tf), max_h=std::min(H,th);
          const u128 upgate_state_elems=mul128(2,mul128(max_m,max_f,"FFN live up/gate state"),
                                                "FFN live up/gate state");
          const u128 down_state_elems=mul128(max_m,max_h,"FFN live down state");
          u64 tk=0;
          u128 peak=0;
          std::vector<u64> tk_probes;
          nlohmann::json peak_components;
          for(const u64 candidate_tk:tk_order) {
            tk_probes.push_back(candidate_tk);
            const u64 max_k=std::min(H,candidate_tk);
            const u128 input_slots=mul128(2,live_tile_bytes(max_m,max_k,ad),
                                          "FFN double input slots");
            const u128 upgate_weights=mul128(4,live_tile_bytes(max_k,max_f,wd),
                                             "FFN up/gate weight tiles");
            const u128 down_weights=mul128(2,live_tile_bytes(max_f,max_h,wd),
                                           "FFN down weight tiles");
            const u128 upgate_fp32=mul128(upgate_state_elems,4,"FFN up/gate FP32 state");
            const u128 upgate_partial=i4?upgate_fp32:0;
            const u128 activation=i4
                ?(ad=="int4"?add128(tile_bytes(max_m,max_f,"int4"),4,
                                    "FFN activation scale")
                             :live_tile_bytes(max_m,max_f,ad))
                :(fp32vec?tile_bytes(max_m,max_f,"fp32")
                         :tile_bytes(max_m,max_f,ad));
            const u128 down_fp32=mul128(down_state_elems,4,"FFN down FP32 accumulator");
            const u128 down_partial=i4?down_fp32:0;
            const u128 pending_output=mul128(
                2,add128(tile_bytes(max_m,max_h,od),od=="int4"?4:0,
                         "FFN pending output tile"),
                "FFN pending output slots");
            const u128 metadata=fused_metadata_bytes;
            u128 candidate_peak=0;
            candidate_peak=add128(candidate_peak,input_slots,"FFN live SRAM");
            candidate_peak=add128(candidate_peak,upgate_weights,"FFN live SRAM");
            candidate_peak=add128(candidate_peak,down_weights,"FFN live SRAM");
            candidate_peak=add128(candidate_peak,upgate_fp32,"FFN live SRAM");
            candidate_peak=add128(candidate_peak,upgate_partial,"FFN live SRAM");
            candidate_peak=add128(candidate_peak,activation,"FFN live SRAM");
            candidate_peak=add128(candidate_peak,down_fp32,"FFN live SRAM");
            candidate_peak=add128(candidate_peak,down_partial,"FFN live SRAM");
            candidate_peak=add128(candidate_peak,pending_output,"FFN live SRAM");
            candidate_peak=add128(candidate_peak,metadata,"FFN live SRAM");
            const long double capacity=perf(hardware,"rho_sram",0.9)*
                static_cast<long double>(du(hardware,"core","sram_bytes_per_core"));
            if(static_cast<long double>(candidate_peak)>capacity) continue;
            tk=candidate_tk;peak=candidate_peak;
            peak_components={{"input_slots",as_u64(input_slots,"FFN input slots")},
                             {"up_gate_weight_tiles",as_u64(upgate_weights,"FFN up/gate weights")},
                             {"down_weight_tile",as_u64(down_weights,"FFN down weights")},
                             {"up_gate_fp32_state",as_u64(upgate_fp32,"FFN up/gate state")},
                             {"up_gate_int32_partial",as_u64(upgate_partial,"FFN up/gate partial")},
                             {"activation_buffer",as_u64(activation,"FFN activation buffer")},
                             {"down_fp32_accumulator",as_u64(down_fp32,"FFN down accumulator")},
                             {"down_int32_partial",as_u64(down_partial,"FFN down partial")},
                             {"pending_output",as_u64(pending_output,"FFN pending output")},
                             {"metadata",as_u64(metadata,"FFN metadata")}};
            break;
          }
          if(!tk) continue;
          const u64 Tk=ceildiv(H,tk);
          const u64 active=std::max<u64>(1,std::min(ncore,as_u64(mul128(Tm,Th,"FFN output cohorts"),
                                                                    "FFN output cohorts")));
          Metrics z;
          z.template_id="fused_output_shard_v2";z.dataflow="fused_output_shard_v2";
          z.tm=tm;z.tf=tf;z.th=th;z.tk=tk;z.active_cores=active;
          z.assumptions={"engineering_assumption",
                         "fused_output_shard_recomputes_up_gate_across_H_shards",
                         "fused_metadata_64B_per_core_reserve"};
          z.source_mops=mul128(source_mops_one,executions,"FFN source matrix ops");
          const u128 mhf=mul128(mh,F,"FFN matrix ops");
          const u128 up_gate_exec=mul128(4,mul128(mhf,Th,"FFN recomputed up/gate ops"),
                                         "FFN recomputed up/gate ops");
          const u128 down_exec=mul128(2,mhf,"FFN down ops");
          z.executed_mops=mul128(add128(up_gate_exec,down_exec,"FFN executed matrix ops"),
                                 executions,"FFN executed matrix ops");
          z.source_vops=mul128(source_vops_one,executions,"FFN source Vec ops");
          const u128 activation_ops=mul128(source_vops_one,mul128(Th,executions,
                                                    "FFN executed activation"),
                                             "FFN executed activation");

          // Exact tiled byte totals keep every odd INT4 tile's final nibble.
          const u128 input_raw_one=mul128(tile_grid_bytes(M,tm,H,tk,ad),mul128(Tf,Th,
                                          "FFN input repeats"),"FFN input bytes");
          const u128 upgate_raw_one=mul128(tile_grid_bytes(H,tk,F,tf,wd),
                                            mul128(2,mul128(Tm,Th,"FFN weight repeats"),
                                                   "FFN up/gate weight repeats"),
                                            "FFN up/gate weight bytes");
          const u128 down_raw_one=mul128(tile_grid_bytes(F,tf,H,th,wd),Tm,
                                          "FFN down weight bytes");
          const u128 output_one=tile_grid_bytes(M,tm,H,th,od);
          const u128 activation_raw_one=mul128(tile_grid_bytes(M,tm,F,tf,
                                                fp32vec?"fp32":ad),Th,
                                                "FFN activation buffer bytes");
          z.hbm_read=mul128(add128(add128(input_raw_one,upgate_raw_one,"FFN HBM reads"),
                                   down_raw_one,"FFN HBM reads"),executions,"FFN HBM reads");
          z.hbm_write=mul128(output_one,executions,"FFN HBM writes");
          z.logical=mul128(add128(add128(bytes(mh,ad),mul128(3,bytes(hf,wd),
                                  "FFN logical weights"),"FFN logical tensors"),
                                  bytes(mh,od),"FFN logical tensors"),
                           executions,"FFN logical bytes");
          z.workspace=0;z.sram_peak=peak;

          const u128 int4_input_one=mul128(tile_grid_bytes(M,tm,H,tk,"int4"),
                                           mul128(Tf,Th,"FFN INT4 input repeats"),
                                           "FFN INT4 input bytes");
          const u128 int4_upgate_one=mul128(tile_grid_bytes(H,tk,F,tf,"int4"),
                                             mul128(2,mul128(Tm,Th,"FFN INT4 weights"),
                                                    "FFN INT4 weights"),"FFN INT4 weights");
          const u128 int4_down_one=mul128(tile_grid_bytes(F,tf,H,th,"int4"),Tm,
                                           "FFN INT4 down weights");
          const u128 int4_activation_one=mul128(tile_grid_bytes(M,tm,F,tf,"int4"),Th,
                                                 "FFN INT4 activation");
          const u128 input_quant_elements_one=(i4&&ad!="int4")
              ?mul128(mul128(M,H,"FFN input conversion"),mul128(Tf,Th,
                     "FFN input conversion repeats"),"FFN input conversion"):0;
          const u128 upgate_quant_elements_one=(i4&&wd!="int4")
              ?mul128(mul128(2,hf,"FFN up/gate conversion"),
                      mul128(Tm,Th,"FFN weight conversion repeats"),
                      "FFN up/gate conversion"):0;
          const u128 down_quant_elements_one=(i4&&wd!="int4")
              ?mul128(hf,Tm,"FFN down conversion"):0;
          const u128 activation_quant_elements_one=i4
              ?mul128(mul128(M,F,"FFN activation conversion"),Th,
                      "FFN activation conversion"):0;
          const u128 output_quant_elements=od=="int4"?mh:0;
          const u128 output_bf16_conversion_elements=
              (od!="fp32"&&od!="int4")?mh:0;
          const u128 quant_conversions_one=add128(
              add128(input_quant_elements_one,upgate_quant_elements_one,
                     "FFN quantization elements"),
              add128(add128(down_quant_elements_one,activation_quant_elements_one,
                            "FFN quantization elements"),
                     output_quant_elements,"FFN quantization elements"),
              "FFN quantization elements");
          const u128 input_fp32_elements_one=(fp32vec&&ad!="fp32")
              ?mul128(mul128(M,H,"FFN FP32 input conversion"),
                      mul128(Tf,Th,"FFN FP32 input repeats"),
                      "FFN FP32 input conversion"):0;
          const u128 upgate_fp32_elements_one=(fp32vec&&wd!="fp32")
              ?mul128(mul128(2,hf,"FFN FP32 up/gate conversion"),
                      mul128(Tm,Th,"FFN FP32 weight repeats"),
                      "FFN FP32 up/gate conversion"):0;
          const u128 down_fp32_elements_one=(fp32vec&&wd!="fp32")
              ?mul128(hf,Tm,"FFN FP32 down conversion"):0;
          const u128 fp32_conversions_one=add128(
              add128(input_fp32_elements_one,upgate_fp32_elements_one,
                     "FFN FP32 conversions"),
              down_fp32_elements_one,"FFN FP32 conversions");
          const u128 conversions_one=add128(
              add128(quant_conversions_one,fp32_conversions_one,
                     "FFN conversion elements"),
              output_bf16_conversion_elements,"FFN conversion elements");
          const u128 quant_conversion_elements=mul128(quant_conversions_one,executions,
                                                      "FFN quantization elements");
          const u128 fp32_conversion_elements=mul128(fp32_conversions_one,executions,
                                                      "FFN FP32 conversion elements");
          const u128 output_bf16_conversion_count=mul128(output_bf16_conversion_elements,
                                                          executions,
                                                          "FFN output conversion elements");
          z.conversion_elements=mul128(conversions_one,executions,"FFN conversion elements");
          const u128 input_quant_bytes_one=(i4&&ad!="int4")
              ?mul128(int4_input_one,executions,"FFN input quant buffer"):0;
          const u128 upgate_quant_bytes_one=(i4&&wd!="int4")
              ?mul128(int4_upgate_one,executions,"FFN up/gate quant buffer"):0;
          const u128 down_quant_bytes_one=(i4&&wd!="int4")
              ?mul128(int4_down_one,executions,"FFN down quant buffer"):0;
          const u128 activation_quant_bytes_one=i4
              ?mul128(int4_activation_one,executions,"FFN activation quant buffer"):0;
          const u128 input_scale_tiles=(i4&&ad!="int4")
              ?mul128(mul128(Tm,Tk,"FFN input scale tiles"),mul128(Tf,Th,
                       "FFN input scale repeats"),"FFN input scale tiles"):0;
          const u128 upgate_scale_tiles=(i4&&wd!="int4")
              ?mul128(mul128(2,mul128(Tm,Th,"FFN weight scale repeats"),
                             "FFN up/gate scale repeats"),
                      mul128(Tk,Tf,"FFN up/gate scale tiles"),"FFN up/gate scale tiles"):0;
          const u128 down_scale_tiles=(i4&&wd!="int4")
              ?mul128(mul128(Tm,Tf,"FFN down scale tiles"),Th,
                      "FFN down scale tiles"):0;
          const u128 activation_scale_tiles=i4
              ?mul128(mul128(Tm,Tf,"FFN activation scale tiles"),Th,
                      "FFN activation scale tiles"):0;
          const u128 output_scale_tiles=od=="int4"
              ?mul128(Tm,Th,"FFN output scale tiles"):0;
          const u128 scale_bytes=mul128(mul128(add128(add128(input_scale_tiles,
                                      upgate_scale_tiles,"FFN scale tiles"),
                                      add128(add128(down_scale_tiles,activation_scale_tiles,
                                                    "FFN scale tiles"),
                                             output_scale_tiles,
                                             "FFN scale tiles"),"FFN scale tiles"),
                                      executions,"FFN scale tiles"),4,"FFN scale bytes");
          const u128 quant_buffer_bytes=add128(add128(input_quant_bytes_one,
                                             upgate_quant_bytes_one,"FFN quant buffers"),
                                      add128(down_quant_bytes_one,activation_quant_bytes_one,
                                             "FFN quant buffers"),"FFN quant buffers");
          const u128 quant_write_bytes=add128(quant_buffer_bytes,scale_bytes,
                                               "FFN quantized buffer writes");
          const u128 input_fp32_one=(fp32vec&&ad!="fp32")
              ?mul128(tile_grid_bytes(M,tm,H,tk,"fp32"),mul128(Tf,Th,
                       "FFN FP32 input repeats"),"FFN FP32 input bytes"):0;
          const u128 upgate_fp32_one=(fp32vec&&wd!="fp32")
              ?mul128(tile_grid_bytes(H,tk,F,tf,"fp32"),
                      mul128(2,mul128(Tm,Th,"FFN FP32 weight repeats"),
                             "FFN FP32 weight repeats"),"FFN FP32 up/gate bytes"):0;
          const u128 down_fp32_one=(fp32vec&&wd!="fp32")
              ?mul128(tile_grid_bytes(F,tf,H,th,"fp32"),Tm,
                      "FFN FP32 down weight bytes"):0;
          const u128 fp32_conversion_buffer_bytes=mul128(
              add128(add128(input_fp32_one,upgate_fp32_one,
                            "FFN FP32 conversion buffers"),
                     down_fp32_one,"FFN FP32 conversion buffers"),
              executions,"FFN FP32 conversion buffers");

          const u128 input_pe_one=i4?(ad=="int4"?input_raw_one:int4_input_one):
              (fp32vec&&ad!="fp32"?input_fp32_one:input_raw_one);
          const u128 upgate_pe_one=i4?(wd=="int4"
              ?mul128(tile_grid_bytes(H,tk,F,tf,"int4"),
                      mul128(2,mul128(Tm,Th,"FFN PE weight repeats"),
                             "FFN PE weight repeats"),"FFN PE weights")
              :int4_upgate_one):
              (fp32vec&&wd!="fp32"?upgate_fp32_one:upgate_raw_one);
          const u128 down_pe_one=i4?int4_down_one:
              (fp32vec&&wd!="fp32"?down_fp32_one:down_raw_one);
          const u128 activation_pe_one=i4?int4_activation_one:activation_raw_one;
          z.int4bytes=i4?mul128(add128(add128(int4_input_one,int4_upgate_one,
                                               "FFN INT4 cache loads"),
                                       add128(int4_activation_one,int4_down_one,
                                              "FFN INT4 cache loads"),
                                       "FFN INT4 cache loads"),executions,
                                  "FFN INT4 input bytes"):0;
          const u128 int4_operand_read_bytes=i4
              ?mul128(add128(add128(mul128(2,input_pe_one,"FFN INT4 A operands"),
                                    upgate_pe_one,"FFN INT4 operand reads"),
                             add128(activation_pe_one,down_pe_one,
                                    "FFN INT4 operand reads"),
                             "FFN INT4 operand reads"),executions,
                      "FFN INT4 operand reads"):0;

          const u128 upgate_update_elems=mul128(mul128(2,mul128(M,F,"FFN up/gate updates"),
                                                  "FFN up/gate updates"),
                                                mul128(Th,Tk,"FFN up/gate updates"),
                                                "FFN up/gate updates");
          const u128 down_update_elems=mul128(mul128(M,H,"FFN down updates"),
                                               Tf,
                                               "FFN down updates");
          const u128 updates=mul128(add128(upgate_update_elems,down_update_elems,
                                            "FFN accumulator updates"),executions,
                                   "FFN accumulator updates");
          const u128 partial_bytes=i4?mul128(updates,4,"FFN INT32 partial bytes"):0;
          const u128 accumulator_write_bytes=mul128(updates,4,"FFN accumulator writes");
          const u128 upgate_accum_reads=mul128(mul128(mul128(2,mul128(M,F,
                                                    "FFN up/gate accumulator reads"),
                                                    "FFN up/gate accumulator reads"),
                                                    Th,"FFN up/gate accumulator reads"),
                                                    Tk-1,"FFN up/gate accumulator reads");
          const u128 down_accum_reads=mul128(mul128(M,H,
                                                "FFN down accumulator reads"),
                                                Tf-1,"FFN down accumulator reads");
          const u128 accumulator_read_bytes=mul128(mul128(add128(upgate_accum_reads,
                                                    down_accum_reads,
                                                    "FFN accumulator reads"),
                                                    executions,"FFN accumulator reads"),
                                                    4,"FFN accumulator read bytes");
          const u128 activation_read_bytes=mul128(mul128(mul128(2,mul128(M,F,
                                                  "FFN activation source reads"),
                                                  "FFN activation source reads"),
                                                  Th,"FFN activation source reads"),
                                                  mul128(executions,4,
                                                         "FFN activation source reads"),
                                                  "FFN activation source reads");
          const u128 conversion_source_reads=add128(
              add128(((i4&&ad!="int4")||(fp32vec&&ad!="fp32"))
                         ?mul128(input_raw_one,executions,
                                 "FFN input conversion source reads"):0,
                     ((i4&&wd!="int4")||(fp32vec&&wd!="fp32"))
                         ?mul128(add128(upgate_raw_one,down_raw_one,
                               "FFN weight conversion source reads"),executions,
                               "FFN weight conversion source reads"):0,
                     "FFN conversion source reads"),
              (i4&&ad!="int4")?mul128(activation_raw_one,executions,
                       "FFN activation conversion source reads"):0,
              "FFN conversion source reads");
          const u128 output_conversion_read=od=="fp32"?0:
              mul128(mul128(mh,4,"FFN output conversion source"),executions,
                     "FFN output conversion source");
          const u128 output_conversion_write=od=="fp32"?0:z.hbm_write;
          const u128 pe_operand_reads=mul128(add128(add128(mul128(2,input_pe_one,
                                                     "FFN input PE reads"),upgate_pe_one,
                                                     "FFN PE operand reads"),
                                            add128(activation_pe_one,down_pe_one,
                                                   "FFN PE operand reads"),
                                            "FFN PE operand reads"),executions,
                                      "FFN PE operand reads");
          const u128 compute_sram_read=add128(add128(conversion_source_reads,
                                                pe_operand_reads,"FFN SRAM reads"),
                                         add128(add128(partial_bytes,accumulator_read_bytes,
                                                       "FFN SRAM reads"),
                                                add128(activation_read_bytes,
                                                       output_conversion_read,
                                                       "FFN SRAM reads"),
                                                "FFN SRAM reads"),"FFN SRAM reads");
          const u128 activation_write_bytes=(i4&&ad=="int4")?0:
              mul128(activation_raw_one,executions,"FFN activation writes");
          const u128 compute_sram_write=add128(add128(add128(quant_write_bytes,
                                                fp32_conversion_buffer_bytes,
                                                "FFN conversion buffer writes"),
                                                add128(partial_bytes,accumulator_write_bytes,
                                                       "FFN SRAM writes"),
                                                "FFN SRAM writes"),
                                         add128(activation_write_bytes,
                                                output_conversion_write,
                                                "FFN SRAM writes"),"FFN SRAM writes");
          const u128 output_bytes=z.hbm_write;
          z.sram_read=add128(compute_sram_read,output_bytes,"FFN total SRAM reads");
          z.sram_write=add128(z.hbm_read,compute_sram_write,"FFN total SRAM writes");
          z.local=add128(compute_sram_read,compute_sram_write,"FFN LOCAL bytes");

          u128 vecops=activation_ops;
          if(i4) {
            vecops=add128(vecops,mul128(quant_conversion_elements,
                         static_cast<u64>(perf(hardware,"kappa_quant_eqops",4)),
                         "FFN quantization eqops"),"FFN Vec ops");
            const u128 partial_elements=updates;
            vecops=add128(vecops,mul128(partial_elements,
                         static_cast<u64>(perf(hardware,"kappa_acc_convert_eqops",2)),
                         "FFN accumulator conversion eqops"),"FFN Vec ops");
            const u128 upgate_adds=mul128(mul128(mul128(2,mul128(M,F,
                                                  "FFN up/gate accumulator adds"),
                                                  "FFN up/gate accumulator adds"),
                                                  Th,"FFN up/gate accumulator adds"),
                                                  Tk-1,"FFN up/gate accumulator adds");
            const u128 down_adds=mul128(mul128(M,H,"FFN down accumulator adds"),
                                         Tf-1,"FFN down accumulator adds");
            vecops=add128(vecops,mul128(add128(upgate_adds,down_adds,
                         "FFN accumulator add eqops"),executions,
                         "FFN accumulator add eqops"),"FFN Vec ops");
          } else {
            vecops=add128(vecops,z.executed_mops,"FFN Vec matrix ops");
            vecops=add128(vecops,fp32_conversion_elements,"FFN FP32 conversion Vec ops");
            if(od=="int4")
              vecops=add128(vecops,mul128(quant_conversion_elements,
                           static_cast<u64>(perf(hardware,"kappa_quant_eqops",4)),
                           "FFN output quantization eqops"),"FFN Vec ops");
          }
          vecops=add128(vecops,output_bf16_conversion_count,
                        "FFN output conversion eqops");
          z.vecops=vecops;
          z.int4ops=i4?z.executed_mops:0;

          Flow l,c,s;
          add_hbm(l,hardware,topo,z.hbm_read,active,false,eta(hardware,"eta_hbm_matrix"));
          l.work[SRAM_WRITE]=static_cast<long double>(as_u64(z.hbm_read,"FFN load SRAM writes"));
          c.work[SRAM_READ]=static_cast<long double>(as_u64(compute_sram_read,"FFN compute SRAM reads"));
          c.work[SRAM_WRITE]=static_cast<long double>(as_u64(compute_sram_write,"FFN compute SRAM writes"));
          c.work[LOCAL]=static_cast<long double>(as_u64(z.local,"FFN LOCAL bytes"));
          if(i4)c.int4=z.executed_mops;
          c.vec=vecops;
          c.compute_s=compute_time(hardware,c.int4,c.vec,active);
          s.work[SRAM_READ]=static_cast<long double>(as_u64(output_bytes,"FFN store SRAM reads"));
          add_hbm(s,hardware,topo,output_bytes,active,true,eta(hardware,"eta_hbm_matrix"));
          const u64 waves=as_u64(mul128(ceildiv(mul128(Tm,Th,"FFN output cohorts"),
                                                   ncore),
                                      mul128(Tf,Tk,"FFN waves"),"FFN waves"),
                                  "FFN waves");
          auto q=new_plan(hardware,task);
          z.output_rounds=ceildiv(mul128(Tm,Th,"FFN output cohorts"),ncore);
          append_epoch(*q,hardware,std::move(l),std::move(c),std::move(s),active,
                       as_u64(z.output_rounds,"FFN output rounds"),"fused_ffn");
          z.waves=waves;
          finalize_summary(*q,z,hardware);
          q->summary["kind"]=op;q->summary["template_id"]="fused_output_shard_v2";
          q->summary["fusion_policy"]="fused";
          q->summary["recomputed_matrix_ops"]=as_u64(z.executed_mops-z.source_mops,
                                                       "FFN recomputed matrix ops");
          q->summary["recompute_matrix_ops"]=q->summary["recomputed_matrix_ops"];
          q->summary["fused_intermediate_hbm_bytes"]=0;
          q->summary["fused_quantized_buffer_bytes"]=as_u64(quant_buffer_bytes,
                                                             "FFN quant buffers");
          q->summary["fused_conversion_scale_bytes"]=as_u64(scale_bytes,"FFN scale bytes");
          q->summary["fused_fp32_conversion_buffer_bytes"]=
              as_u64(fp32_conversion_buffer_bytes,"FFN FP32 conversion buffers");
          q->summary["int4_operand_read_bytes"]=
              as_u64(int4_operand_read_bytes,"FFN INT4 operand reads");
          q->summary["fused_metadata_reserve_bytes_per_core"]=fused_metadata_bytes;
          q->summary["fused_tk_capacity_probe_order"]=tk_probes;
          q->summary["fused_sram_peak_components_bytes_per_core"]=peak_components;
          q->summary["fused_activation_eqops"]=as_u64(activation_ops,
                                                       "FFN activation eqops");
          q->fingerprint=plan_to_json(*q).at("fingerprint").get<std::string>();
          result.push_back(std::move(q));
        }
      }
    }
    return result;
  } else {
    fail("unsupported_op_kind: " + op);
  }
  finalize_summary(*p,m,hardware);p->fingerprint=plan_to_json(*p).at("fingerprint").get<std::string>();return {p};
}

}  // namespace coarse_die::detail
