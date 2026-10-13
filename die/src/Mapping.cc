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
      slots = add128(slots, mul128(tail, len, "attention tail slots"), "attention slots");
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
  const u64 row0 = total64 / 2, row1 = total64 - row0;
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
  a.credit_delta = {{as_double(f.credit[0], "input credit delta"),
                     as_double(f.credit[1], "output credit delta")}};
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
    push_epoch(plan,hw,lb,cb,Flow{},active,1,1,1,0,0,name+"_first_overlap");
    push_epoch(plan,hw,Flow{},add_flow(cb,cr),sb,active,1,1,1,1,1,name+"_last_overlap");
    push_epoch(plan,hw,Flow{},Flow{},add_flow(sb,sr),active,1,0,1,0,1,name+"_store_last");
    return;
  }
  push_epoch(plan,hw,lb,Flow{},Flow{},active,1,1,0,0,0,name+"_load_first");
  push_epoch(plan,hw,lb,cb,Flow{},active,1,1,1,0,0,name+"_first_overlap");
  push_epoch(plan,hw,add_flow(scale_flow(lb,W-2),lr),scale_flow(cb,W-2),scale_flow(sb,W-2),
             active,W-2,W-2,W-2,1,1,name+"_interior_repeat");
  push_epoch(plan,hw,Flow{},add_flow(cb,cr),sb,active,1,1,1,1,1,name+"_last_overlap");
  push_epoch(plan,hw,Flow{},Flow{},add_flow(sb,sr),active,1,0,0,0,1,name+"_store_last");
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
  const u128 quant_a_elems = ad == "int4" || !int4_path ? 0 : mul128(mul128(M,K,"A quant"),mul128(mul128(tN,passes,"A quant repeats"),layers,"A quant repeats"),"A quant");
  const u128 quant_w_elems = wd == "int4" || !int4_path ? 0 : mul128(mul128(K,N,"W quant"),mul128(mul128(wr?1:tM,passes,"W quant repeats"),layers,"W quant repeats"),"W quant");
  const u128 conversions = add128(quant_a_elems,quant_w_elems,"conversion elements");
  m.conversion_elements = add128(m.conversion_elements,conversions,"conversion elements");
  const u128 a_i4 = mul128(tile_grid_bytes(M,tm,K,tk,"int4"),tN,"A INT4 bytes");
  const u128 w_i4 = mul128(tile_grid_bytes(K,tk,N,tn,"int4"),wr?1:tM,"W INT4 bytes");
  const u128 i4total = int4_path ? add128(mul128(a_i4,executions,"A INT4 bytes"),mul128(w_i4,mul128(layers,mul128(passes,wr?1:tM,"W INT4 repeats"),"W INT4 repeats"),"W INT4 bytes"),"INT4 input bytes") : 0;
  m.int4bytes = add128(m.int4bytes,i4total,"INT4 input bytes");
  const u128 scale_a = quant_a_elems ? mul128(mul128(tM,tK,"A scale tiles"),tN,"A scale tiles") : 0;
  const u128 scale_w = quant_w_elems ? mul128(mul128(ceildiv(K,tk),ceildiv(N,tn),"W scale tiles"),wr?1:tM,"W scale tiles") : 0;
  const u128 scales = mul128(add128(mul128(scale_a,passes,"A scale passes"),mul128(scale_w,passes,"W scale passes"),"scale tiles"),mul128(layers,4,"scale bytes"),"scale bytes");
  const u128 quant_buf = int4_path ? add128(i4total,scales,"quant buffers") : 0;
  const u128 partial = int4_path ? mul128(mul128(mul128(M,N,"partial"),tK,"partial K"),mul128(executions,4,"partial bytes"),"partial bytes") : 0;
  const u128 accum_write = int4_path ? mul128(mul128(mul128(M,N,"accumulator"),tK,"accumulator K"),mul128(executions,4,"accumulator bytes"),"accumulator bytes") : 0;
  const u128 accum_read = int4_path ? mul128(mul128(mul128(M,N,"accumulator"),tK-1,"accumulator reuse"),mul128(executions,4,"accumulator bytes"),"accumulator read") : 0;
  const u128 consumed_inputs = int4_path
      ? add128(mul128(a_i4,executions,"A compute reads"),
               mul128(w_i4,mul128(executions,tM,"W compute reads"),"W compute reads"),
               "matrix SRAM reads")
      : read_total;
  const u128 sram_r = add128(add128(consumed_inputs,partial,"matrix reads"),accum_read,"SRAM reads");
  const u128 sram_w = add128(add128(quant_buf,partial,"SRAM writes"),add128(accum_write,c_total,"SRAM writes"),"SRAM writes");
  const u128 local = add128(sram_r,sram_w,"LOCAL traffic");
  m.sram_read = add128(m.sram_read,sram_r,"SRAM reads"); m.sram_write = add128(m.sram_write,sram_w,"SRAM writes"); m.local = add128(m.local,local,"LOCAL traffic");

  u128 vecops = 0;
  u128 peops = 0;
  if (int4_path) {
    peops = layer_ops;
    vecops = add128(vecops,mul128(conversions,static_cast<u64>(perf(hw,"kappa_quant_eqops",4)),"quant eqops"),"GEMM Vec ops");
    const u128 converted = mul128(mul128(mul128(M,N,"acc convert"),tK,"acc convert"),executions,"acc convert");
    vecops = add128(vecops,mul128(converted,static_cast<u64>(perf(hw,"kappa_acc_convert_eqops",2)),"acc convert eqops"),"GEMM Vec ops");
    vecops = add128(vecops,mul128(mul128(mul128(M,N,"acc add"),tK-1,"acc add"),executions,"acc add"),"GEMM Vec ops");
    if (od != "fp32") vecops = add128(vecops,mul128(mul128(M,N,"output conversion"),executions,"output conversion"),"GEMM Vec ops");
  } else {
    vecops = layer_ops;
  }
  m.int4ops = add128(m.int4ops,peops,"INT4 ops"); m.vecops = add128(m.vecops,vecops,"Vec eqops");
  m.source_vops = add128(m.source_vops,vecops,"source Vec ops");
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
  const u128 a_tile = int4_path && ad!="int4" ? add128(a_exec_tile,a_raw_tile,"A raw and quantized slots") : std::max(a_exec_tile,a_raw_tile);
  const u128 w_tile = int4_path && wd!="int4" ? add128(w_exec_tile,w_raw_tile,"W raw and quantized slots") : std::max(w_exec_tile,w_raw_tile);
  const u128 accum = int4_path ? mul128(mul128(max_m,max_n,"acc tile"),12,"accumulator and partial") : mul128(mul128(max_m,max_n,"acc tile"),4,"accumulator tile");
  const u128 output = tile_bytes(static_cast<u64>(max_m),static_cast<u64>(max_n),od);
  const u128 peak = wr ? add128(tile_bytes(K,static_cast<u64>(max_n),int4_path?"int4":"fp32"),add128(mul128(2,a_tile,"WR input slots"),add128(accum,output,"WR live buffers"),"WR live buffers"),"WR peak")
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
  m.sram_read = add128(m.sram_read,sramr,"SRAM read bytes"); m.sram_write = add128(m.sram_write,sramw,"SRAM write bytes");
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
      const u128 rows=extra_input/hidden;
      tile_extra=mul128(tile_elems,rows,"Norm parameter tile");
    }
  }
  const u128 tile_peak=add128(add128(tile_input,tile_output,"vector live tensors"),
      add128(mul128(tile_elems,4,"vector FP32 working tile"),tile_extra,"vector parameters"),"vector peak");
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
  u64 maxq=1,maxkv=1;for(const auto& span:spans){maxq=std::max(maxq,span.q);maxkv=std::max(maxkv,span.kv);}
  const auto tq_opts=full?std::vector<u64>{16}:options(maxq,{16,64,256,1024});
  const auto tk_opts=full?std::vector<u64>{64}:options(maxkv,{16,64,256,1024});
  for (u64 tq:tq_opts) for (u64 tk:tk_opts) {
    Metrics m; auto p=new_plan(hw,task);
    u128 slots=0, valid=0, qbytes=0, kvbytes=0, outbytes=0, queryrows=0, prefixes=0;
    for (const Span& s:spans) {
      u128 sl=0, pr=0;
      if (causal) { auto pair=prefix_slots_one(s,tq,tk); sl=pair.first; pr=pair.second; }
      else {
        const u128 qtiles=ceildiv(s.q,tq);
        const u128 ktiles=ceildiv(s.kv,tk);
        sl=mul128(mul128(mul128(qtiles,tq,"noncausal score tiles"),mul128(ktiles,tk,"noncausal score tiles"),"noncausal score tiles"),s.count,"noncausal score tiles");
        pr=mul128(mul128(qtiles,s.kv,"noncausal KV tile reads"),s.count,"noncausal KV tile reads");
      }
      const u128 request_heads=mul128(s.count,qh,"attention heads");
      const u128 qelem=mul128(mul128(s.q,hd,"Q elements"),request_heads,"Q elements");
      const u128 score_slots=mul128(sl,qh,"attention score heads");
      const u128 score_valid=mul128(valid_pairs(s,causal),qh,"attention valid heads");
      const u128 query_count=mul128(s.q,request_heads,"query rows");
      slots=add128(slots,score_slots,"attention slots");
      valid=add128(valid,score_valid,"valid score pairs");
      queryrows=add128(queryrows,query_count,"query rows");
      prefixes=add128(prefixes,mul128(pr,qh,"GQA KV prefix reads"),"GQA KV prefix reads");
      if (mode!="av") qbytes=add128(qbytes,bytes(qelem,dtype_q),"Q bytes");
      const u128 kv_tensor_count=mode=="both"?2:1;
      const u128 key_elems=mul128(mul128(pr,hd,"KV elements"),mul128(kv_tensor_count,qh,"GQA KV reads"),"KV reads");
      kvbytes=add128(kvbytes,bytes(key_elems,dtype_k),"KV bytes");
      if (mode!="qk") outbytes=add128(outbytes,bytes(qelem,dtype_o),"attention output bytes");
    }
    slots=mul128(slots,executions,"attention executions");
    valid=mul128(valid,executions,"attention source executions");
    queryrows=mul128(queryrows,executions,"attention query executions");
    qbytes=mul128(qbytes,executions,"attention Q passes");
    kvbytes=mul128(kvbytes,executions,"attention KV passes");
    outbytes=mul128(outbytes,executions,"attention output passes");
    prefixes=mul128(prefixes,executions,"attention KV prefixes");
    const u64 active=std::max<u64>(1,std::min(ncore,as_u64(queryrows,"attention query tasks")));
    const u128 matrix_factor=mode=="both"?4:2;
    const u128 matops=mul128(matrix_factor,mul128(hd,slots,"attention matrix ops"),"attention matrix ops");
    m.source_mops=mul128(matrix_factor,mul128(hd,valid,"source attention ops"),"source attention ops");
    m.executed_mops=matops;
    const u128 eq=mode=="qk"?0:(mode=="av"?slots:add128(mul128(slots,13,"attention softmax/conversion eqops"),mul128(queryrows,hd,"attention normalization eqops"),"attention Vec eqops"));
    const u128 source_eq=mode=="qk"?0:(mode=="av"?valid:add128(mul128(valid,13,"attention source Vec ops"),mul128(queryrows,hd,"attention source normalization"),"attention source Vec ops"));
    m.source_vops=source_eq; m.vecops=eq;
    const u128 scoreb=mul128(slots,4,"score bytes");
    const u128 probb=(mode=="av"||full)?mul128(slots,4,"probability bytes"):0;
    const u128 totalread=add128(add128(qbytes,kvbytes,"attention input bytes"),mode=="av"||full?probb:0,"attention input bytes");
    const u128 totalwrite=add128(add128(outbytes,mode=="qk"||full?scoreb:0,"attention score writes"),full?probb:0,"attention output writes");
    m.hbm_read=totalread; m.hbm_write=totalwrite;
    const u128 sb=full?add128(scoreb,probb,"materialized score/probability workspace"):0;
    m.workspace=sb; m.score_bytes=(mode=="qk"||full)?scoreb:0; m.probability_bytes=probb;
    m.score_slots=slots; m.valid_pairs=valid;
    const u128 sramr=add128(totalread,mode=="qk"||full?scoreb:0,"attention SRAM reads");
    const u128 sramw=add128(totalwrite,full?0:mul128(queryrows,4,"online state"),"attention SRAM writes");
    m.sram_read=sramr; m.sram_write=sramw; m.local=add128(sramr,sramw,"attention LOCAL");
    Flow l,c,s;
    add_hbm(l,hw,topo,totalread,active,false,eta(hw,"eta_hbm_vector")); l.work[SRAM_WRITE]=static_cast<long double>(as_u64(totalread,"attention load"));
    c.work[SRAM_READ]=static_cast<long double>(as_u64(sramr,"attention SRAM read"));
    c.work[SRAM_WRITE]=static_cast<long double>(as_u64(sramw,"attention SRAM write"));
    c.work[LOCAL]=static_cast<long double>(as_u64(m.local,"attention local"));
    const bool int4path=gets(task,"execution_profile")=="int4_matrix_vec_fp32_assumed_v1";
    if (int4path) { c.int4=matops; c.vec=eq; } else c.vec=add128(eq,matops,"attention Vec ops");
    c.compute_s=compute_time(hw,c.int4,c.vec,active);
    add_hbm(s,hw,topo,totalwrite,active,true,eta(hw,"eta_hbm_vector")); s.work[SRAM_READ]=static_cast<long double>(as_u64(totalwrite,"attention store"));
    const u64 rounds=std::max<u64>(1,ceildiv(as_u64(queryrows,"attention rows"),active));
    const u128 wave128=mul128(rounds,ceildiv(as_u64(slots,"attention slots"),std::max<u64>(1,as_u64(mul128(tq,tk,"attention tile area"),"attention tile area"))),"attention waves");
    const u64 waves=std::max<u64>(1,as_u64(wave128,"attention waves"));
    append_epoch(*p,hw,std::move(l),std::move(c),std::move(s),active,rounds,mode=="both"?(full?"full_score_materialization":"fused_attention"):"attention_"+mode);
    m.output_rounds=rounds; m.waves=waves; m.active_cores=active;
    m.sram_peak=add128(add128(mul128(mul128(tq,hd,"Q tile"),4,"Q tile bytes"),mul128(mul128(tk,hd,"KV tile"),4,"KV tile bytes"),"attention live tensors"),mul128(mul128(tq,tk,"score tile"),4,"score tile bytes"),"attention peak");
    if(mode=="av") m.sram_peak=add128(mul128(mul128(tk,hd,"V tile"),4,"V tile bytes"),mul128(mul128(tq,tk,"probability tile"),4,"probability tile bytes"),"AV tile peak");
    if(mode=="qk") m.sram_peak=add128(add128(mul128(mul128(tq,hd,"Q tile"),4,"Q tile bytes"),mul128(mul128(tk,hd,"K tile"),4,"K tile bytes"),"QK input tiles"),mul128(mul128(tq,tk,"score tile"),4,"score tile bytes"),"QK tile peak");
    m.template_id=mode=="both"?(full?"full_score_materialization_v2":"fused_q_head_shard_v2"):"attention_"+mode+"_v2";
    m.assumptions={"engineering_assumption","GQA_head_mapping","absolute_position_causal_mask","not_run_granularity_validation"};
    m.logical=add128(totalread,totalwrite,"attention logical bytes");
    m.tm=tq; m.tn=tk; m.tk=tk;
    if (static_cast<long double>(as_u64(m.sram_peak,"attention SRAM peak")) > perf(hw,"rho_sram",0.9)*dd(hw,"core","sram_bytes_per_core")) continue;
    finalize_summary(*p,m,hw); p->summary["template_id"]=m.template_id; p->summary["dataflow_id"]=full?"full_score_materialization_v2":"fused_q_head_shard_v2";
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
    if (op=="norm") { eq=mul128(6,count,"Norm eqops"); extra=bytes(mul128(H,2,"Norm parameters"),dtype(task,"parameters")); }
    else if (op=="residual") { eq=count; extra=bytes(elems,ad); }
    else if (op=="mlp_activation") { count=mul128(M,F,"activation elements"); eq=mul128(16,count,"activation eqops"); }
    else if (op=="scale_mask") { const auto ss=spans_of(task); u128 slots=0; for(const auto& s:ss) slots=add128(slots,mul128(mul128(s.q,s.kv,"score count"),s.count,"score count"),"score count"); count=slots; eq=mul128(2,count,"scale eqops"); }
    else if (op=="softmax") { const auto ss=spans_of(task); u128 slots=0; for(const auto& s:ss) slots=add128(slots,mul128(mul128(s.q,s.kv,"score count"),s.count,"score count"),"score count"); count=slots; eq=mul128(10,count,"softmax eqops"); }
    else if (op=="qkv_rope") {
      const u64 qh=shape_u(task,"q_heads",1), kvh=shape_u(task,"kv_heads",1), hd=shape_u(task,"head_dim",H);
      const u64 nout=as_u64(add128(mul128(qh,hd,"QKV output"),mul128(2,mul128(kvh,hd,"QKV output"),"QKV output"),"QKV output"),"QKV output");
      p->task["tensor_shapes"]["m"]=M; p->task["tensor_shapes"]["n"]=nout; p->task["tensor_shapes"]["k"]=H;
      p->task["op_kind"]="gemm";
      auto plans=enumerate_gemm(hardware,p->task,topo,M,nout,H);
      for (auto& q:plans) { q->task=task; q->summary["kind"]="qkv_rope"; q->summary["template_id"]="qkv_rope_"+q->summary.value("dataflow_id",std::string()); q->summary["rope_eqops"]=as_u64(mul128(3,mul128(M,mul128(qh+kvh,hd,"RoPE elements"),"RoPE elements"),"RoPE eqops"),"RoPE eqops"); q->fingerprint=plan_to_json(*q).at("fingerprint").get<std::string>(); }
      return plans;
    }
    count=mul128(count,layers,"layer vector elements"); eq=mul128(eq,layers,"layer vector operations");
    const u128 inb=bytes(count,ad), outb=bytes(count,od);
    add_vector(*p,m,hardware,topo,count,eq,inb,outb,active,op+"_vec_v2",extra);
  } else if (op=="mlp_up" || op=="mlp_down" || op=="o_proj" || op=="logits") {
    u64 N=H,K=H;
    if(op=="mlp_up"){N=F;K=H;} else if(op=="mlp_down"){N=H;K=F;} else if(op=="logits"){N=V;K=H;}
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
    auto att=enumerate_attention(hardware,task2,topo,false);
    for(auto& q:att){q->task=task;q->summary["kind"]=op;q->summary["template_id"]=op+"_"+q->summary.value("template_id",std::string());q->fingerprint=plan_to_json(*q).at("fingerprint").get<std::string>();}
    return att;
  } else if (op=="ffn_region") {
    const std::string policy=gets(task,"fusion_policy","auto");
    std::vector<std::shared_ptr<Plan>> result;
    // Materialized SwiGLU is represented as four fixed primitive epochs. Its
    // temporary is counted in HBM and SRAM; it is never expanded by tile count.
    if(policy!="fused") {
      auto q=new_plan(hardware,task);Metrics z;z.template_id="materialized_swiglu_v2";z.assumptions={"engineering_assumption","materialized_intermediate"};
      const std::string wdt=dtype(task,"weights"), xdt=dtype(task,"activations"), ydt=dtype(task,"output");
      GemmChoice c{std::min<u64>(16,M),std::min<u64>(64,F),std::min<u64>(64,H),false};
      add_gemm_epoch(*q,z,hardware,topo,M,F,H,xdt,wdt,ydt,layers,passes,c,"ffn_up_v2");
      add_gemm_epoch(*q,z,hardware,topo,M,F,H,xdt,wdt,ydt,layers,passes,c,"ffn_gate_v2");
      const u128 middle=mul128(mul128(M,F,"FFN intermediate"),layers,"FFN intermediate");
      const u128 middle_bytes=bytes(middle,ydt); z.hbm_write=add128(z.hbm_write,mul128(middle_bytes,2,"FFN materialized outputs"),"FFN HBM writes");
      z.workspace=std::max(z.workspace,mul128(middle_bytes,2,"FFN workspace"));
      const u128 actops=mul128(16,middle,"FFN activation eqops");
      add_vector(*q,z,hardware,topo,middle,actops,mul128(middle_bytes,2,"FFN activation reads"),middle_bytes,active,"ffn_activation_v2");
      add_gemm_epoch(*q,z,hardware,topo,M,H,F,ydt,wdt,ydt,layers,passes,GemmChoice{std::min<u64>(16,M),std::min<u64>(64,H),std::min<u64>(64,F),false},"ffn_down_v2");
      finalize_summary(*q,z,hardware);q->summary["kind"]=op;q->summary["template_id"]="materialized_swiglu_v2";q->summary["fusion_policy"]="materialized";q->summary["recomputed_matrix_ops"]=0;q->summary["recompute_matrix_ops"]=0;
      q->fingerprint=plan_to_json(*q).at("fingerprint").get<std::string>();result.push_back(std::move(q));
    }
    if(policy!="materialized"&&policy!="none") {
      // Bounded fused-output-shard candidates; no core/tile arrays are built.
      const auto tms=options(M,{1,16,64,256});const auto tfs=options(F,{16,64,256,1024});const auto ths=options(H,{64,256});const auto tks=options(H,{64,256,1024,4096});
      for(u64 tm:tms)for(u64 tf:tfs)for(u64 th:ths)for(u64 tk:tks){
        const u64 Th=ceildiv(H,th),Tf=ceildiv(F,tf),Tm=ceildiv(M,tm),Tk=ceildiv(H,tk);
        const u64 ncore=du(hardware,"die","core_count");const u64 a=std::max<u64>(1,std::min(ncore,ceildiv(as_u64(mul128(Tm,Th,"FFN tasks"),"FFN tasks"),ncore)));
        Metrics z;z.template_id="fused_output_shard_v2";z.dataflow="fused_output_shard_v2";z.tm=tm;z.tf=tf;z.th=th;z.tk=tk;z.active_cores=a;z.assumptions={"engineering_assumption","fused_output_shard_recomputes_up_gate_across_H_shards"};
        const u128 src=mul128(6,mul128(mul128(M,H,"FFN source ops"),F,"FFN source ops"),"FFN source ops");
        const u128 exe=add128(mul128(4,mul128(mul128(mul128(M,H,"FFN executed ops"),F,"FFN executed ops"),Th,"FFN recompute"),"FFN executed ops"),mul128(2,mul128(mul128(M,H,"FFN executed ops"),F,"FFN executed ops"),"FFN executed ops"),"FFN executed ops");
        z.source_mops=mul128(src,layers,"FFN source ops");z.executed_mops=mul128(exe,layers,"FFN executed ops");
        z.source_vops=mul128(mul128(16,mul128(M,F,"FFN activation"),"FFN activation"),mul128(Th,layers,"FFN activation"),"FFN activation");z.vecops=z.source_vops;
        const u128 input=mul128(mul128(bytes(mul128(M,H,"FFN input"),ad),Tf,"FFN input passes"),Th,"FFN input shards");
        const u128 weights=add128(mul128(bytes(mul128(H,F,"FFN weights"),wd),mul128(2,mul128(Tm,Th,"FFN weight repeats"),"FFN weight repeats"),"FFN up/gate weights"),mul128(bytes(mul128(F,H,"FFN down weights"),wd),Tm,"FFN down weights"),"FFN weights");
        const u128 output=bytes(mul128(M,H,"FFN output"),od);
        z.hbm_read=add128(input,weights,"FFN HBM reads");z.hbm_write=output;z.logical=add128(add128(bytes(mul128(M,H,"FFN logical input"),ad),bytes(mul128(H,F,"FFN logical weights"),wd),"FFN logical"),output,"FFN logical");
        z.sram_peak=add128(mul128(2,mul128(tm,th,"FFN input slots"),"FFN input slots"),add128(mul128(mul128(2,mul128(tm,tf,"FFN up state"),"FFN state"),4,"FFN state bytes"),mul128(mul128(tm,th,"FFN output"),8,"FFN output/accum"),"FFN live"),"FFN peak");
        if(static_cast<long double>(as_u64(z.sram_peak,"FFN SRAM peak"))>perf(hardware,"rho_sram",0.9)*dd(hardware,"core","sram_bytes_per_core"))continue;
        Flow l,c,s;add_hbm(l,hardware,topo,z.hbm_read,a,false,eta(hardware,"eta_hbm_matrix"));l.work[SRAM_WRITE]=static_cast<long double>(as_u64(z.hbm_read,"FFN load"));
        c.work[SRAM_READ]=static_cast<long double>(as_u64(z.hbm_read,"FFN SRAM read"));c.work[SRAM_WRITE]=static_cast<long double>(as_u64(output,"FFN SRAM write"));c.work[LOCAL]=static_cast<long double>(as_u64(add128(z.hbm_read,output,"FFN local"),"FFN local"));
        const bool i4=gets(task,"execution_profile")=="int4_matrix_vec_fp32_assumed_v1";if(i4)c.int4=z.executed_mops;else c.vec=z.executed_mops;c.vec=add128(c.vec,z.vecops,"FFN Vec");c.compute_s=compute_time(hardware,c.int4,c.vec,a);
        s.work[SRAM_READ]=static_cast<long double>(as_u64(output,"FFN store"));add_hbm(s,hardware,topo,output,a,true,eta(hardware,"eta_hbm_matrix"));
        const u64 waves=as_u64(mul128(ceildiv(mul128(Tm,Th,"FFN output cohorts"),ncore),mul128(Tf,ceildiv(H,tk),"FFN waves"),"FFN waves"),"FFN waves");
        auto q=new_plan(hardware,task);z.output_rounds=ceildiv(mul128(Tm,Th,"FFN tasks"),ncore);append_epoch(*q,hardware,std::move(l),std::move(c),std::move(s),a,as_u64(z.output_rounds,"FFN output rounds"),"fused_ffn");z.waves=waves;
        finalize_summary(*q,z,hardware);q->summary["kind"]=op;q->summary["template_id"]="fused_output_shard_v2";q->summary["fusion_policy"]="fused";q->summary["source_matrix_ops"]=as_u64(z.source_mops,"FFN source ops");q->summary["executed_matrix_ops"]=as_u64(z.executed_mops,"FFN exec ops");q->summary["recomputed_matrix_ops"]=as_u64(z.executed_mops-z.source_mops,"FFN recompute");q->summary["recompute_matrix_ops"]=q->summary["recomputed_matrix_ops"];q->summary["tm"]=tm;q->summary["tf"]=tf;q->summary["th"]=th;q->summary["tk"]=tk;
        q->fingerprint=plan_to_json(*q).at("fingerprint").get<std::string>();result.push_back(std::move(q));
      }
    }
    return result;
  } else {
    fail("unsupported_op_kind: " + op);
  }
  finalize_summary(*p,m,hardware);p->fingerprint=plan_to_json(*p).at("fingerprint").get<std::string>();return {p};
}

}  // namespace coarse_die::detail
