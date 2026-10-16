#include "coarse_die/Plan.hh"
#include "GemmTailBudget.hh"
#include "EquivalentWork.hh"
#include "RegionDelay.hh"
#include "GemmOSGeometry.hh"
#include "GemmOSResourceFacts.hh"
#include "ElementwiseGeometry.hh"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

namespace coarse_die::detail {
namespace {

using u64 = std::uint64_t;
using u128 = unsigned __int128;
using EqWork = equivalent_work::Work;

[[noreturn]] void fail(const std::string& message) {
  throw std::invalid_argument(message);
}

u128 add128(u128 a, u128 b, std::string_view field) {
  constexpr u128 max = ~u128{0};
  if (b > max - a) fail("overflow: " + std::string(field));
  return a + b;
}

u128 mul128(u128 a, u128 b, std::string_view field) {
  constexpr u128 max = ~u128{0};
  if (a && b > max / a) fail("overflow: " + std::string(field));
  return a * b;
}

EqWork add128(const EqWork& a, const EqWork& b, std::string_view) {
  return a + b;
}

EqWork weighted_ops(u128 elements, double coefficient, std::string_view) {
  return EqWork::weighted(elements, coefficient);
}

nlohmann::json eqops_json(const EqWork& value) {
  const u64 whole = value.checked_u64_whole();
  if (value.integral()) return whole;
  return static_cast<double>(value.as_long_double());
}

EqWork summary_eqops(const nlohmann::json& summary) {
  if (summary.contains("vec_equivalent_ops_fraction"))
    return EqWork::from_parts(summary.at("vec_equivalent_ops_whole").get<u64>(),
                             equivalent_fraction_from_json(summary.at("vec_equivalent_ops_fraction")));
  return EqWork(summary.at("vec_equivalent_ops").get<u64>());
}

void publish_eqops(nlohmann::json& summary, const EqWork& value) {
  summary["vec_equivalent_ops"] = eqops_json(value);
  if (!value.integral()) {
    summary["equivalent_work_schema"] = "dyadic_equivalent_work_v1";
    summary["vec_equivalent_ops_whole"] = value.checked_u64_whole();
    summary["vec_equivalent_ops_fraction"] = equivalent_fraction_to_json(value.fraction());
  } else {
    summary.erase("equivalent_work_schema");
    summary.erase("vec_equivalent_ops_whole");
    summary.erase("vec_equivalent_ops_fraction");
  }
}

u64 as_u64(u128 value, std::string_view field) {
  if (value > std::numeric_limits<u64>::max())
    fail("overflow: " + std::string(field) + " exceeds uint64 output range");
  return static_cast<u64>(value);
}

double as_double(long double value, std::string_view field) {
  if (!std::isfinite(value) || value < 0.0L ||
      value > static_cast<long double>(std::numeric_limits<double>::max()))
    fail("overflow: " + std::string(field) + " cannot be represented as a finite double");
  return static_cast<double>(value);
}

double as_signed_double(long double value, std::string_view field) {
  const long double limit =
      static_cast<long double>(std::numeric_limits<double>::max());
  if (!std::isfinite(value) || value < -limit || value > limit)
    fail("overflow: " + std::string(field) + " cannot be represented as a finite double");
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
  u64 q = 0, kv = 0, count = 1, batch = 1;
  u64 qs = 0, ks = 0, kv_stride = 0, qs_stride = 0;
};

std::vector<Span> spans_of(const nlohmann::json& task) {
  std::vector<Span> result;
  auto it = task.find("spans");
  if (it == task.end()) return result;
  if (!it->is_array()) fail("task.spans must be an array");
  for (const auto& j : *it) {
    Span s{getu(j, "q_tokens"), getu(j, "kv_tokens"), getu(j, "count", 1),
           getu(j, "batch", 1),
           getu(j, "q_start_pos"), getu(j, "kv_start_pos"),
           getu(j, "kv_stride"), getu(j, "q_start_stride")};
    if (!s.q || !s.kv || !s.count || !s.batch) fail("task.spans: q_tokens, kv_tokens, count and batch must be positive");
    if (s.count > 1 && s.q != 1 &&
        !(s.kv >= s.q && s.ks == 0 && s.qs == s.kv - s.q && s.qs_stride == s.kv_stride))
      fail("unsupported_mapping: compact attention span progression is not a supported causal run");
    result.push_back(s);
  }
  return result;
}

u128 attention_sum_kv_lengths(const Span& s) {
  return add128(mul128(s.count,s.kv,"attention KV span sum"),
      mul128(s.kv_stride,mul128(s.count,s.count-1,"attention KV span sum")/2,
             "attention KV span sum"),"attention KV span sum");
}

// Full score materialization charges each query row against the complete KV
// length declared for its request, independent of causal masking and tile size.
u128 attention_span_rectangle_slots(const Span& s) {
  return mul128(s.q,attention_sum_kv_lengths(s),
                "materialized attention score slots");
}

u128 attention_span_score_slots(const Span& s,u64 qh) {
  return mul128(mul128(attention_span_rectangle_slots(s),s.batch,
                       "attention batch score slots"),qh,
                "attention score heads");
}

u128 attention_span_query_rows(const Span& s,u64 qh) {
  const u128 request_heads=mul128(mul128(s.count,s.batch,
      "attention heads"),qh,"attention heads");
  return mul128(s.q,request_heads,"query rows");
}

std::pair<u128,u128> attention_materialized_counts(
    const std::vector<Span>& spans,u64 qh) {
  u128 score_slots=0,query_rows=0;
  for(const auto& s:spans) {
    score_slots=add128(score_slots,attention_span_score_slots(s,qh),
                       "attention slots");
    query_rows=add128(query_rows,attention_span_query_rows(s,qh),
                      "attention query rows");
  }
  return {score_slots,query_rows};
}

u128 attention_workspace_from_counts(u128 score_slots,u128 query_rows,
                                     u64 head_dim,
                                     const std::string& output_dtype,
                                     const std::string& mode,
                                     bool staged_full) {
  // The counts describe one materialized invocation of the task. Compact
  // spans and batch are part of that invocation; layer_instances and
  // weight_passes are sequential executions whose scratch can be reused.
  const u128 score_bytes=mul128(score_slots,4,"score bytes");
  if(mode=="qk") return score_bytes;
  if(mode!="both"||!staged_full) return 0;
  const u128 probability_bytes=score_bytes;
  const u128 cast_bytes=output_dtype=="fp32"?0:
      bytes(mul128(query_rows,head_dim,"attention FP32 output"),"fp32");
  return add128(add128(score_bytes,probability_bytes,
                       "materialized score/probability workspace"),
                cast_bytes,"materialized attention workspace");
}

u128 ffn_workspace_from_middle_elements(u128 middle_elements,
                                        const std::string& middle_dtype,
                                        bool swiglu) {
  return mul128(bytes(middle_elements,middle_dtype),swiglu?2:1,
                "FFN workspace");
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
  for (const auto& s : spans) total = add128(total,
      mul128(prefix_slots_one(s, tq, tk).first, s.batch, "attention batch slots"),
      "attention slots");
  return total;
}

struct AxisDist {
  u64 nx = 1, ny = 1, active = 1;
  bool point = false;
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
  if(d.point) return std::abs((xaxis ? d.ex : d.ey)-point);
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
  u128 int4 = 0;
  EqWork vec;
  long double compute_s = 0;
  // Additional per-Core serial service bound, distinct from COMPUTE_GATE.
  long double private_s = 0;
  std::array<long double, 2> credit{};
};

struct Metrics {
  u128 source_mops = 0, executed_mops = 0, source_vops = 0;
  EqWork vecops;
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

void route_bytes_to(Flow& flow, const Hardware& hw, const Topology& topo,
                 u64 row, u64 nbytes, const AxisDist& d, bool writing) {
  if (!nbytes) return;
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

void route_bytes(Flow& flow, const Hardware& hw, const Topology& topo,
                 u64 row, u64 nbytes, u64 active_cores, bool writing) {
  route_bytes_to(flow,hw,topo,row,nbytes,
      destination(active_cores,topo.nx,topo.ny),writing);
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
                    std::string_view field) {
  flow.work[resource] += static_cast<long double>(as_u64(amount, field));
}

double eta(const Hardware& hw, const char* key) {
  const double value = perf(hw, key, 1.0);
  if (!(value > 0.0) || value > 1.0) fail(std::string("invalid_hardware: ") + key + " must be in (0,1]");
  return value;
}

double compute_time(const Hardware& hw, u128 int4ops, const EqWork& vecops,
                    u64 active) {
  const long double ppe = dd(hw, "core", "P_int4_per_core_ops_per_s");
  const long double pvec = dd(hw, "core", "P_vec_per_core_ops_per_s");
  if (!(ppe > 0) || !(pvec > 0) || !active) fail("invalid_hardware: compute rates must be positive");
  const long double tpe = static_cast<long double>(int4ops) /
      (ppe * active * eta(hw, "eta_pe"));
  const long double tv = vecops.as_long_double() /
      (pvec * active * eta(hw, "eta_vec"));
  return as_double(tpe + tv, "compute seconds");
}

void seal_activity(Activity& a, Flow& f, const Hardware& hw, u64 active) {
  a.hbm_bytes = {};
  for (std::size_t row = 0; row < 2; ++row)
    for (std::size_t dir = 0; dir < 2; ++dir)
      a.hbm_bytes[row][dir] = as_u64(f.hbm[row][dir], "activity HBM bytes");
  a.int4_ops = as_u64(f.int4, "activity INT4 ops");
  a.vec_eqops = f.vec.checked_u64_whole();
  a.vec_eqops_fraction = f.vec.fraction();
  a.private_time_s = as_double(f.compute_s, "activity private time");
  const double local_one_core = active ? hw.rates[LOCAL] *
      (static_cast<double>(active) / std::max<u64>(1, du(hw,"die","core_count",1))) : 0;
  const double sr_one_core = active ? hw.rates[SRAM_READ] *
      (static_cast<double>(active) / std::max<u64>(1, du(hw,"die","core_count",1))) : 0;
  const double sw_one_core = active ? hw.rates[SRAM_WRITE] *
      (static_cast<double>(active) / std::max<u64>(1, du(hw,"die","core_count",1))) : 0;
  long double private_time = std::max<long double>(a.private_time_s,f.private_s);
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
  r.int4=add128(a.int4,b.int4,"activity op split");r.vec=add128(a.vec,b.vec,"activity op split");r.compute_s=a.compute_s+b.compute_s;r.private_s=a.private_s+b.private_s;
  return r;
}

Flow scale_flow(const Flow& a,u64 count) {
  Flow r;
  for(std::size_t i=0;i<resource_count;++i) r.work[i]=a.work[i]*count;
  for(std::size_t row=0;row<2;++row)for(std::size_t dir=0;dir<2;++dir)r.hbm[row][dir]=mul128(a.hbm[row][dir],count,"repeated activity bytes");
  r.int4=mul128(a.int4,count,"repeated INT4 ops");r.vec=a.vec.multiply(count);r.compute_s=a.compute_s*count;r.private_s=a.private_s*count;
  return r;
}

std::pair<Flow,Flow> divide_flow(const Flow& a,u64 count) {
  if(!count) fail("mapping: zero frame count");
  Flow base;
  for(std::size_t i=0;i<resource_count;++i) base.work[i]=a.work[i]/count;
  for(std::size_t row=0;row<2;++row)for(std::size_t dir=0;dir<2;++dir)base.hbm[row][dir]=a.hbm[row][dir]/count;
  const auto vec_split=a.vec.divide(count);
  base.int4=a.int4/count;base.vec=vec_split.first;base.compute_s=a.compute_s/count;base.private_s=a.private_s/count;
  Flow remainder;
  for(std::size_t i=0;i<resource_count;++i) remainder.work[i]=a.work[i]-base.work[i]*count;
  for(std::size_t row=0;row<2;++row)for(std::size_t dir=0;dir<2;++dir)remainder.hbm[row][dir]=a.hbm[row][dir]-base.hbm[row][dir]*count;
  remainder.int4=a.int4-base.int4*count;remainder.vec=vec_split.second;remainder.compute_s=a.compute_s-base.compute_s*count;remainder.private_s=a.private_s-base.private_s*count;
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

// Keep physical IO totals exact while extracting the true first K load and
// final output drain. Routing retains the selected aggregate destination
// profile; only readiness units change. This is not a per-wave interpreter.
Flow subtract_io_flow(const Flow& total, const Flow& part) {
  Flow rest;
  if(part.private_s>total.private_s)
    fail("mapping_invariant: IO prefix exceeds PRIVATE service");
  rest.private_s=total.private_s-part.private_s;
  if (total.int4 || part.int4 || total.vec.as_long_double() || part.vec.as_long_double() ||
      total.compute_s || part.compute_s)
    fail("mapping_invariant: IO partition contains computation");
  for (std::size_t r=0;r<resource_count;++r) {
    const long double value=total.work[r]-part.work[r];
    if (value < -1e-12L*std::max(1.0L,total.work[r]))
      fail("mapping_invariant: IO prefix exceeds total resource work");
    rest.work[r]=std::max(0.0L,value);
  }
  for (std::size_t row=0;row<2;++row) for(std::size_t dir=0;dir<2;++dir) {
    if (part.hbm[row][dir]>total.hbm[row][dir])
      fail("mapping_invariant: IO prefix exceeds total payload");
    rest.hbm[row][dir]=total.hbm[row][dir]-part.hbm[row][dir];
  }
  return rest;
}

Flow subtract_compute_flow(const Flow& total,const Flow& prefix) {
  Flow rest;
  for(std::size_t r=0;r<resource_count;++r) {
    const long double value=total.work[r]-prefix.work[r];
    if(value<0) fail("mapping_invariant: compute prefix exceeds resource total");
    rest.work[r]=value;
  }
  if(prefix.int4>total.int4 || prefix.compute_s>total.compute_s || prefix.private_s>total.private_s)
    fail("mapping_invariant: compute prefix exceeds total work");
  rest.int4=total.int4-prefix.int4;
  rest.compute_s=total.compute_s-prefix.compute_s;
  rest.private_s=total.private_s-prefix.private_s;
  auto whole=total.vec.whole();
  long double fraction=total.vec.fraction()-prefix.vec.fraction();
  if(whole<prefix.vec.whole()) fail("mapping_invariant: Vec prefix exceeds total");
  whole-=prefix.vec.whole();
  if(fraction<0) {
    if(whole==0) fail("mapping_invariant: Vec prefix exceeds total fraction");
    --whole; fraction+=1;
  }
  rest.vec=EqWork::from_parts(whole,fraction);
  const auto check=rest.vec+prefix.vec;
  if(check.whole()!=total.vec.whole() || check.fraction()!=total.vec.fraction())
    fail("mapping_invariant: Vec prefix partition loses precision");
  return rest;
}

void append_os_service_curve(Plan& plan, const Hardware& hw, const Topology& topo,
    const gemm_os_geometry::GemmOSGeometry& geometry,
    const gemm_tail_budget::Input& budget, Flow load, Flow compute, Flow store,
    u64 executions, u64 active, const std::string& name) {
  const u64 inputs=as_u64(mul128(mul128(geometry.output_tile_count(),
      geometry.k_tiles(),"OS K consumption units"),executions,"OS K consumption units"),
      "OS K consumption units");
  const u64 outputs=as_u64(mul128(geometry.output_tile_count(),executions,
      "OS output block units"),"OS output block units");
  const auto first=geometry.cohort(0);
  const auto last=geometry.cohort(geometry.output_round_count()-1);
  const auto first_ledger=geometry.wave_ledger(0);
  const auto last_ledger=geometry.wave_ledger(geometry.consumption_wave_count()-1);
  const u64 first_units=first.active_cores, drain_units=last.active_cores;
  const u64 first_compute_input_units=as_u64(mul128(first_units,geometry.k_tiles(),
      "OS first output K consumption units"),"OS first output K consumption units");
  gemm_os_geometry::Ledger first_cohort;
  for(std::size_t i=0;i<geometry.class_count();++i)
    gemm_os_geometry::detail::add(first_cohort,gemm_os_geometry::detail::scale(
        geometry.access_class(i).per_consumption,
        geometry.class_consumption_count_in_cohort(0,i)));
  const bool only_first=outputs==first_units;
  if(!only_first && outputs<first_units+drain_units)
    fail("mapping_invariant: OS prefix and drain output cohorts overlap");
  for(const auto units:{inputs,outputs,inputs-first_units,outputs-drain_units,
                         first_compute_input_units,inputs-first_compute_input_units,
                         only_first ? u64{0} : outputs-first_units-drain_units})
    if(static_cast<long double>(static_cast<double>(units))!=static_cast<long double>(units))
      fail("overflow: OS cumulative credit endpoint is not exactly representable");
  Flow first_load, cohort_load, final_store;
  add_hbm(first_load,hw,topo,first_ledger.hbm_read_bytes,active,false,eta(hw,"eta_hbm_matrix"));
  add_bytes_work(first_load,SRAM_WRITE,first_ledger.hbm_read_bytes,"OS first K load");
  add_hbm(cohort_load,hw,topo,first_cohort.hbm_read_bytes,active,false,eta(hw,"eta_hbm_matrix"));
  add_bytes_work(cohort_load,SRAM_WRITE,first_cohort.hbm_read_bytes,"OS first cohort load");
  add_hbm(final_store,hw,topo,last_ledger.hbm_write_bytes,active,true,eta(hw,"eta_hbm_matrix"));
  add_bytes_work(final_store,SRAM_READ,last_ledger.hbm_write_bytes,"OS final output drain");
  Flow prefix_load=subtract_io_flow(cohort_load,first_load);
  Flow remaining_load=subtract_io_flow(load,cohort_load);
  Flow early_store=subtract_io_flow(store,final_store);

  // A bounded class sum extracts the first whole-K cohort's real work. Its
  // terminal event releases the first actual output blocks; there is no
  // fitted latency or a delayed ramp pretending to be the first output.
  const auto shape=gemm_tail_budget::detail::shape(budget,true);
  long double first_compute=0;
  for (std::size_t i=0;i<geometry.class_count();++i) {
    const auto& cls=geometry.access_class(i);
    if (geometry.class_output_count_in_range(i,first.q_begin,first.q_end)==0) continue;
    first_compute=std::max(first_compute,
        gemm_tail_budget::detail::per_core_tile_seconds(budget,shape,cls.m,cls.n,true));
  }
  Flow prefix_compute;
  prefix_compute.work[SRAM_READ]=static_cast<long double>(first_cohort.compute_sram_read_bytes);
  prefix_compute.work[SRAM_WRITE]=static_cast<long double>(first_cohort.compute_sram_write_bytes);
  prefix_compute.work[LOCAL]=static_cast<long double>(first_cohort.local_bytes);
  prefix_compute.compute_s=first_compute;
  prefix_compute.int4=first_cohort.matrix_int4_ops;
  prefix_compute.vec=first_cohort.vec_equivalent_ops;
  // In the one-cohort case reuse the exact aggregate, avoiding a second
  // floating representation of the same COMPUTE_GATE endpoint.
  if(only_first) prefix_compute=compute;
  Flow remaining_compute=only_first ? Flow{} : subtract_compute_flow(compute,prefix_compute);
  const double first_output_progress=as_double(prefix_compute.compute_s/compute.compute_s,
      "OS first output progress");
  if (!(first_output_progress>0) || first_output_progress>1)
    fail("mapping_invariant: invalid OS first output work threshold");

  const auto begin=plan.epochs.size();
  const auto description_begin=service_description_count(plan);
  const auto append=[&](const std::string& suffix, Flow l, Flow c, Flow st,
      std::array<double,2> initial, u64 service_active) -> Epoch& {
    Epoch epoch; epoch.name=name+suffix;
    epoch.credit_initial=initial;
    // One K unit is one actual output tile's K consumption; one output unit
    // is one actual output tile. Bytes never enter either credit dimension.
    epoch.credit_capacity={{2.0*active,2.0*active}};
    for (auto item : {std::pair<Role,Flow*>{Role::Load,&l},
                      {Role::Compute,&c},{Role::Store,&st}}) {
      Activity a; a.role=item.first; a.id=plan.epochs.size()*3+epoch.activities.size()+1;
      seal_activity(a,*item.second,hw,service_active);
      a.credit_curve={{0,{{0,0}}},{1,a.credit_delta}};
      if (std::any_of(a.work.begin(),a.work.end(),[](double x){return x>0;}))
        epoch.activities.push_back(std::move(a));
    }
    plan.epochs.push_back(std::move(epoch));
    return plan.epochs.back();
  };
  first_load.credit={{static_cast<long double>(first_units),0}};
  append("_load_first_k",first_load,{}, {},{{0,0}},active);
  prefix_load.credit={{static_cast<long double>(first_compute_input_units-first_units),0}};
  prefix_compute.credit={{-static_cast<long double>(first_compute_input_units),0}};
  append("_first_output_work",prefix_load,prefix_compute,{},
         {{static_cast<double>(first_units),0}},first.active_cores);
  if(!only_first) {
    remaining_load.credit={{static_cast<long double>(inputs-first_compute_input_units),0}};
    remaining_compute.credit={{-static_cast<long double>(inputs-first_compute_input_units),
        static_cast<long double>(outputs-first_units-drain_units)}};
    early_store.credit={{0,-static_cast<long double>(outputs-drain_units)}};
    append("_cumulative_service",remaining_load,remaining_compute,early_store,
           {{0,static_cast<double>(first_units)}},active);
  }
  final_store.credit={{0,-static_cast<long double>(drain_units)}};
  append("_store_last",{}, {},final_store,{{0,static_cast<double>(drain_units)}},last.active_cores);
  const auto descriptions=service_description_count(plan)-description_begin;
  if(descriptions>8)fail("mapping_invariant: OS curve exceeds primitive description budget");
  plan.summary["service_curves"].push_back({
      {"model","analytical_cumulative_os_v3"},{"epoch_begin",begin},
      {"epoch_count",plan.epochs.size()-begin},
      {"description_count",descriptions},{"description_budget",descriptions<=5 ? 5 : 8},
      {"input_unit","one_output_tile_K_consumption"},{"output_unit","one_output_tile"},
      {"input_units",inputs},{"output_units",outputs},
      {"input_capacity_units",2*active},{"output_capacity_units",2*active},
      {"first_load_units",first_units},{"first_output_input_units",first_compute_input_units},
      {"first_output_input_bytes",as_u64(first_cohort.hbm_read_bytes,"OS first output input bytes")},
      {"final_drain_units",drain_units},
      {"first_output_compute_progress",std::min(1.0,first_output_progress)},
      {"first_output_compute_work_s",as_double(first_compute,"OS first output work")},
      {"first_output_release","first_cohort_compute_terminal"},
      {"first_output_epoch",begin+1},
      {"final_output_release","compute_terminal_before_final_drain"},
      {"readiness_approximation","aggregate_piecewise_affine_not_discrete_tile_readiness"}});
}

std::vector<u64> options(u64 dim, std::initializer_list<u64> choices) {
  if (!dim) fail("unsupported_shape: dimensions must be positive");
  std::set<u64> result;
  for (u64 value : choices) result.insert(std::min(dim, value));
  return {result.begin(), result.end()};
}

// Exact static per-core GEMM live-storage peak. Keep this calculation pure so
// candidate enumeration can reject over-capacity tiles before constructing
// a Plan, Geometry, or its analytical service curves.
u128 gemm_sram_peak(u64 M, u64 N, u64 K, u64 tm, u64 tn, u64 tk,
                    const std::string& ad, const std::string& wd,
                    const std::string& od, const std::string& profile,
                    bool wr) {
  if (!M || !N || !K || !tm || !tn || !tk)
    fail("unsupported_shape: GEMM SRAM peak requires positive dimensions and tiles");
  const bool int4_path = profile == "int4_matrix_vec_fp32_assumed_v1";
  const std::string execution_dtype = int4_path ? "int4" : "fp32";
  const bool convert_a = ad != execution_dtype, convert_w = wd != execution_dtype;
  const u64 tK = ceildiv(K, tk);
  const u64 max_m = std::min(M, tm), max_n = std::min(N, tn), max_k = std::min(K, tk);
  const u128 a_exec_tile = tile_bytes(max_m, max_k, int4_path ? "int4" : "fp32");
  const u128 w_exec_tile = tile_bytes(max_k, max_n, int4_path ? "int4" : "fp32");
  const u128 a_raw_tile = tile_bytes(max_m, max_k, ad);
  const u128 w_raw_tile = tile_bytes(max_k, max_n, wd);
  const u128 a_tile = convert_a
      ? add128(add128(a_exec_tile, a_raw_tile, "A raw and converted slots"),
               int4_path ? 4 : 0, "A conversion scale")
      : a_exec_tile;
  const u128 w_tile = convert_w
      ? add128(add128(w_exec_tile, w_raw_tile, "W raw and converted slots"),
               int4_path ? 4 : 0, "W conversion scale")
      : w_exec_tile;
  const u128 accum = mul128(mul128(max_m, max_n, "acc tile"),
                             int4_path ? 8 : 4,
                             "accumulator and partial");
  const u128 output = mul128(2,
      add128(tile_bytes(max_m, max_n, od), od == "int4" ? 4 : 0,
             "output tile and scale"),
      "pending output slots");
  if (wr) {
    const u128 resident_weight = add128(
        tile_grid_bytes(K, tk, max_n, max_n, execution_dtype),
        convert_w
            ? add128(tile_grid_bytes(K, tk, max_n, max_n, wd),
                     int4_path ? mul128(tK, 4, "WR resident scales") : 0,
                     "WR conversion buffers")
            : 0,
        "WR resident weight");
    return add128(resident_weight,
        add128(mul128(2, a_tile, "WR input slots"),
               add128(accum, output, "WR live buffers"), "WR live buffers"),
        "WR peak");
  }
  return add128(mul128(2, add128(a_tile, w_tile, "OS tile inputs"),
                       "OS double inputs"),
                add128(accum, output, "OS live outputs"), "OS peak");
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
  u128 sram_r=0, sram_w=0, local=0, peops=0, conversion_count=0;
  EqWork vecops;
  std::optional<gemm_os_geometry::GemmOSGeometry> os_geometry;
  if (!wr) {
    // The same bounded per-K facts will drive the OS readiness lifecycle.
    // Aggregate physical work here comes from those facts; repeated layers
    // and passes multiply the single invocation ledger exactly once.
    using Storage = gemm_os_access_counter::StorageDType;
    const auto storage = [](const std::string& name) {
      if (name=="fp32") return Storage::Fp32;
      if (name=="bf16") return Storage::Bf16;
      if (name=="int4") return Storage::Int4;
      if (name=="int32") return Storage::Int32;
      fail("unsupported_dtype: GEMM OS storage "+name);
    };
    gemm_os_geometry::Input input;
    input.M=M; input.N=N; input.K=K;
    input.tm=tm; input.tn=tn; input.tk=tk; input.core_count=ncore;
    input.a_storage=storage(ad); input.w_storage=storage(wd);
    input.output_storage=storage(od);
    input.matrix_path=int4_path ? gemm_os_access_counter::MatrixPath::Int4PE
                               : gemm_os_access_counter::MatrixPath::StrictFp32Vec;
    input.kappa_quant=perf(hw,"kappa_quant_eqops",4);
    input.kappa_acc=perf(hw,"kappa_acc_convert_eqops",2);
    os_geometry.emplace(input);
    const auto& ledger=os_geometry->total_ledger();
    if (mul128(ledger.hbm_read_bytes,executions,"OS HBM reads")!=read_total ||
        mul128(ledger.hbm_write_bytes,executions,"OS HBM writes")!=c_total ||
        mul128(ledger.source_math_ops,executions,"OS source operations")!=layer_ops)
      fail("mapping_invariant: GEMM OS access facts disagree with source payload/ops");
    sram_r=mul128(ledger.compute_sram_read_bytes,executions,"OS Compute SRAM reads");
    sram_w=mul128(ledger.compute_sram_write_bytes,executions,"OS Compute SRAM writes");
    local=mul128(ledger.local_bytes,executions,"OS LOCAL traffic");
    peops=mul128(ledger.matrix_int4_ops,executions,"OS INT4 operations");
    vecops=ledger.vec_equivalent_ops.multiply(as_u64(executions,"OS executions"));
    conversion_count=mul128(ledger.conversion_elements,executions,"OS conversions");
  } else {
    // WR's resident weights have a different conversion lifetime. Keep its
    // established ledger until it has its own per-consumption facts source.
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
    sram_r=add128(add128(add128(consumed_inputs,conversion_reads,"matrix source reads"),add128(partial,accum_read,"matrix state reads"),"matrix reads"),add128(scale_reads,output_cast_reads,"matrix conversion reads"),"SRAM reads");
    sram_w=add128(add128(quant_buf,partial,"matrix writes"),add128(accum_write,add128(output_cast_elements?c_total:0,output_scales,"output conversion writes"),"matrix writes"),"SRAM writes");
    local = add128(sram_r,sram_w,"LOCAL traffic");
    if (int4_path) {
      peops = layer_ops;
      vecops = add128(vecops,weighted_ops(conversions,perf(hw,"kappa_quant_eqops",4),"quant eqops"),"GEMM Vec ops");
      const u128 converted = mul128(mul128(mul128(M,N,"acc convert"),tK,"acc convert"),executions,"acc convert");
      vecops = add128(vecops,weighted_ops(converted,perf(hw,"kappa_acc_convert_eqops",2),"acc convert eqops"),"GEMM Vec ops");
      vecops = add128(vecops,mul128(mul128(mul128(M,N,"acc add"),tK-1,"acc add"),executions,"acc add"),"GEMM Vec ops");
      if (od != "fp32") vecops = add128(vecops,weighted_ops(output_cast_elements,od=="int4"?perf(hw,"kappa_quant_eqops",4):1,"output conversion"),"GEMM Vec ops");
      conversion_count=add128(add128(conversions,converted,"GEMM conversions"),output_cast_elements,"GEMM conversions");
    } else {
      // BF16/INT32 inputs are converted once per loaded tile. INT4 decoding
      // uses the explicit template cost; FP32 operands require no conversion.
      vecops=add128(layer_ops,add128(weighted_ops(quant_a_elems,ad=="int4"?perf(hw,"kappa_quant_eqops",4):1,"A FP32 conversion"),weighted_ops(quant_w_elems,wd=="int4"?perf(hw,"kappa_quant_eqops",4):1,"W FP32 conversion"),"FP32 conversions"),"GEMM Vec ops");
      vecops=add128(vecops,weighted_ops(output_cast_elements,od=="int4"?perf(hw,"kappa_quant_eqops",4):1,"output conversion"),"output conversion");
      conversion_count=add128(conversions,output_cast_elements,"GEMM conversions");
    }
  }
  m.sram_read = add128(m.sram_read,add128(sram_r,c_total,"GEMM store SRAM read"),"SRAM reads");
  m.sram_write = add128(m.sram_write,add128(sram_w,read_total,"GEMM load SRAM write"),"SRAM writes");
  m.local = add128(m.local,local,"LOCAL traffic");
  m.conversion_elements=add128(m.conversion_elements,conversion_count,"conversion elements");
  m.int4ops = add128(m.int4ops,peops,"INT4 ops"); m.vecops = add128(m.vecops,vecops,"Vec eqops");
  // Quantization, accumulator and storage conversion are execution work;
  // the source GEMM graph has matrix operations, not these Vec operations.
  gemm_tail_budget::Input budget_input;
  budget_input.M=M; budget_input.N=N; budget_input.K=K;
  budget_input.tm=tm; budget_input.tn=tn; budget_input.tk=tk;
  budget_input.ncore=ncore;
  budget_input.dataflow=wr ? gemm_tail_budget::Dataflow::WeightResident
                           : gemm_tail_budget::Dataflow::OutputStationary;
  budget_input.matrix_path=int4_path ? gemm_tail_budget::MatrixPath::Int4PE
                                     : gemm_tail_budget::MatrixPath::StrictFp32Vec;
  budget_input.p_int4_per_core_ops_s=dd(hw,"core","P_int4_per_core_ops_per_s");
  budget_input.p_vec_per_core_eqops_s=dd(hw,"core","P_vec_per_core_ops_per_s");
  budget_input.eta_pe=eta(hw,"eta_pe");
  budget_input.eta_vec=eta(hw,"eta_vec");
  // Use the same real coefficients for the action ledger and tail budget.
  const auto quant_eqops=[&]() {
    return static_cast<long double>(perf(hw,"kappa_quant_eqops",4));
  };
  if(convert_a) {
    budget_input.a_input_conversion_eqops_per_element=int4_path || ad=="int4"
        ? quant_eqops() : 1.0L;
  }
  if(convert_w) {
    budget_input.w_input_conversion_eqops_per_element=int4_path || wd=="int4"
        ? quant_eqops() : 1.0L;
  }
  if(int4_path)
    budget_input.int32_acc_conversion_eqops_per_result=static_cast<long double>(
        perf(hw,"kappa_acc_convert_eqops",2));
  // INT4 output packing has the same cost on both matrix execution paths.
  if(od!="fp32") {
    budget_input.output_cast_eqops_per_element=od=="int4"
        ? quant_eqops() : 1.0L;
  }
  const auto cohort_budget=gemm_tail_budget::calculate(budget_input);
  // The cohort helper covers one MxNxK invocation; layers*passes is already
  // represented in the operation ledger above and multiplies this duration
  // exactly once here. The append_epoch macro distribution remains unchanged.
  const long double total_compute_s=cohort_budget.serial_slowest_compute_gate_seconds*
      static_cast<long double>(executions);
  const double comp_s=as_double(total_compute_s,"GEMM cohort compute seconds");

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
  if(cohort_budget.output_round_count!=as_u64(rounds,"GEMM output rounds"))
    fail("mapping_invariant: GEMM cohort budget output rounds disagree with scheduler");
  const u128 wave = mul128(rounds,tK,"GEMM consumption waves");
  if (os_geometry) {
    append_os_service_curve(plan,hw,topo,*os_geometry,budget_input,
        std::move(load),std::move(compute),std::move(store),
        as_u64(executions,"OS executions"),active,label);
    m.assumptions.push_back("analytical_cumulative_os_v3_fixed_bounded_description_budget");
  } else {
    append_epoch(plan,hw,std::move(load),std::move(compute),std::move(store),active,
        as_u64(rounds,"GEMM output rounds"),label);
  }
  const u128 peak = gemm_sram_peak(M,N,K,tm,tn,tk,ad,wd,od,
                                    gets(plan.task,"execution_profile"),wr);
  m.sram_peak = std::max(m.sram_peak,peak);
  m.output_rounds = add128(m.output_rounds,rounds,"output rounds"); m.waves = add128(m.waves,wave,"consumption waves");
  m.active_cores = active; m.tm=tm; m.tn=tn; m.tk=tk; m.dataflow=wr?"weight_resident_v2":"output_stationary_v2"; m.template_id=label;
}

void finalize_summary(Plan& p, const Metrics& m, const Hardware& hw,
                      bool ranking_only=false) {
  // Internal GEMM trials still check every published scalar's range, even
  // when only rank keys need JSON storage. Public Plans always emit all fields.
  const auto emit=[&](const char* key,u128 value,std::string_view field,bool rank=false) {
    const u64 checked=as_u64(value,field);
    if(!ranking_only || rank)p.summary[key]=checked;
  };
  p.summary["kind"] = gets(p.task,"op_kind");
  p.summary["template_id"] = m.template_id;
  p.summary["dataflow_id"] = m.dataflow;
  emit("source_matrix_ops",m.source_mops,"source matrix ops");
  emit("executed_matrix_ops",m.executed_mops,"executed matrix ops");
  emit("recomputed_matrix_ops",m.executed_mops >= m.source_mops ? m.executed_mops-m.source_mops : 0,"recomputed matrix ops");
  if(!ranking_only)p.summary["recompute_matrix_ops"] = p.summary["recomputed_matrix_ops"];
  emit("executed_int4_ops",m.int4ops,"executed INT4 ops");
  emit("int4_input_bytes",m.int4bytes,"INT4 input bytes");
  emit("source_vec_ops",m.source_vops,"source Vec ops");
  if(!ranking_only)publish_eqops(p.summary,m.vecops);
  else {
    (void)eqops_json(m.vecops);
    if(!m.vecops.integral())(void)equivalent_fraction_to_json(m.vecops.fraction());
  }
  emit("conversion_elements",m.conversion_elements,"conversion elements");
  emit("logical_tensor_bytes",m.logical,"logical tensor bytes");
  emit("hbm_read_bytes_total",m.hbm_read,"HBM read total",true);
  emit("hbm_write_bytes_total",m.hbm_write,"HBM write total",true);
  emit("physical_transfer_bytes",add128(m.hbm_read,m.hbm_write,"HBM total"),"HBM total");
  emit("hbm_workspace_peak_bytes",m.workspace,"HBM workspace",true);
  emit("sram_peak_bytes_per_core",m.sram_peak,"SRAM peak",true);
  const auto sram_capacity=du(hw,"core","sram_bytes_per_core");
  if(!ranking_only)p.summary["sram_capacity_bytes_per_core"]=sram_capacity;
  emit("sram_read_bytes",m.sram_read,"SRAM reads");
  emit("sram_write_bytes",m.sram_write,"SRAM writes");
  emit("local_bus_bytes",m.local,"LOCAL bytes");
  emit("output_round_count",m.output_rounds,"output rounds");
  emit("consumption_wave_count",m.waves,"consumption waves");
  if(!ranking_only) {
    p.summary["wave_count"] = p.summary["consumption_wave_count"];
    p.summary["active_cores"] = m.active_cores;
  }
  p.summary["tm"] = m.tm; p.summary["tn"] = m.tn; p.summary["tk"] = m.tk;
  p.summary["th"] = m.th;
  if(!ranking_only)p.summary["tf"] = m.tf;
  emit("executed_score_slots",m.score_slots,"executed score slots");
  emit("valid_score_pairs",m.valid_pairs,"valid score pairs");
  emit("score_bytes",m.score_bytes,"score bytes");
  emit("probability_bytes",m.probability_bytes,"probability bytes");
  if(!ranking_only) {
    p.summary["granularity_validation"] = "not_run";
    p.summary["mapping_assumptions"] = m.assumptions;
    p.summary["activity_units"] = "work/hbm_bytes/ops are totals over this epoch; repeat_count is audit metadata only; signed credit_delta is total over progress [0,1]";
  }
  if (p.summary.contains("service_curves")) {
    const auto& curves=p.summary.at("service_curves");
    std::string revision=curves.at(0).at("model").get<std::string>();
    for(const auto& curve:curves)
      if(curve.at("model")!=revision) revision="analytical_cumulative_mixed_v1";
    const auto descriptions=service_description_count(p);
    if(!ranking_only) {
      p.summary["service_model_revision"]=revision;
      p.summary["service_description_count"]=descriptions;
      p.summary["activity_units"]=revision=="analytical_cumulative_os_v3"
        ? "physical work/bytes/ops are epoch totals; input credits count output-tile K consumptions; output credits count output tiles; credit_curve overrides affine delta*p; final output released only at compute terminal"
        : "physical work/bytes/ops are epoch totals; each service_curves entry defines separate input/output credit units; credit_curve overrides affine delta*p; final output released only at compute terminal";
    }
  }
  if(!ranking_only)
    p.summary["routing_evidence"]={{"profile",gets(hw.input.at("performance_profile"),"timing_topology_profile")},
        {"status","engineering_assumption"}};
  else (void)gets(hw.input.at("performance_profile"),"timing_topology_profile");
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
  // Remaining legacy vector templates stream bounded per-core tiles.
  // Norm has its own explicit row-resident allocation and ownership below.
  const u128 tile_elems = std::min<u128>(elements,256);
  const u128 tile_input = elements ? (mul128(input_bytes,tile_elems,"vector input tile") + elements - 1) / elements : 0;
  const u128 tile_output = elements ? (mul128(output_bytes,tile_elems,"vector output tile") + elements - 1) / elements : 0;
  u128 tile_extra = elements ? (mul128(extra_input,tile_elems,"vector extra tile") + elements - 1) / elements : 0;
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

// Row-resident Norm: a row is one consumption/output unit. Statistics and
// normalization are serial work inside Compute; no row output is exposed
// before its full Compute terminal. Input storage is reread in SRAM for the
// application pass. Gamma/beta never broadcast for free between Cores.
void add_norm_rows(Plan& plan, Metrics& m, const Hardware& hw,
    const Topology& topo,u64 rows,u64 hidden,u64 executions,
    const std::string& ad,const std::string& pd,const std::string& od,
    bool layernorm) {
  const u64 active=std::min(rows,du(hw,"die","core_count"));
  const u64 rounds=ceildiv(rows,active);
  const u64 total_rounds=as_u64(mul128(rounds,executions,"Norm rounds"),"Norm rounds");
  const u64 total_rows=as_u64(mul128(rows,executions,"Norm rows"),"Norm rows");
  const u64 last_rows=rows-(rounds-1)*active;
  const u64 slots=total_rounds==1 ? 1 : 2;
  const u128 xrow=bytes(hidden,ad),yrow=bytes(hidden,od);
  const u64 parameters=layernorm ? 2 : 1;
  const u128 prow=mul128(parameters,bytes(hidden,pd),"Norm separately packed parameters");
  const u128 pconverted=mul128(mul128(parameters,hidden,"Norm parameter elements"),4,"Norm FP32 parameters");
  const u128 work_tile=mul128(std::min<u64>(hidden,256),4,"Norm FP32 tile");
  const u128 base_peak=add128(add128(mul128(slots,add128(xrow,yrow,"Norm row slots"),"Norm row slots"),
      work_tile,"Norm row working set"),80,"Norm reduction state and metadata");
  const long double capacity=perf(hw,"rho_sram",0.9)*dd(hw,"core","sram_bytes_per_core");
  // Source data expands in-place backwards into the allocated FP32 parameter
  // buffer. Reuse is within one region only. Repeated-region boundaries have
  // no production timing representation yet and keep a diagnostic body.
  const bool cached=executions==1 && static_cast<long double>(add128(base_peak,pconverted,"Norm cache peak"))<=capacity;
  const u128 parameter_buffer=cached ? pconverted : mul128(parameters,work_tile,"Norm streamed parameter tiles");
  const u128 peak=add128(base_peak,parameter_buffer,"Norm SRAM peak");
  if(static_cast<long double>(peak)>capacity)
    fail("unsupported_workspace: Norm row-resident buffers require "+std::to_string(as_u64(peak,"Norm required SRAM"))+
        " bytes per Core, exceeding configured usable SRAM; a two-pass HBM row template is not implemented");
  for(u64 units:{total_rounds,total_rounds-1,total_rounds>1 ? total_rounds-2 : 0})
    if(static_cast<long double>(static_cast<double>(units))!=static_cast<long double>(units))
      fail("overflow: Norm cumulative row credit is not exactly representable");
  const auto private_bytes=[&](u128 n,std::size_t resource) {
    const long double rate=hw.rates[resource]/du(hw,"die","core_count");
    if(!(rate>0))fail("invalid_hardware: Norm per-Core byte rate is zero");
    return static_cast<long double>(as_u64(n,"Norm serial bytes"))/rate;
  };
  // Fixed uniform prefix plus final partial prefix, with invocation-local RR
  // ownership. No row/Core loop and no assumption that a partial cohort has
  // the full grid's aggregate private throughput.
  const auto io=[&](u64 count,u64 repeats,u128 row_bytes,bool write) {
    Flow f;
    const auto group=[&](u64 cores,u128 payload) {
      if(!payload)return;
      const u64 n=as_u64(payload,"Norm route group");
      for(std::size_t row=0;row<2;++row) {
        const u64 amount=n/2+(row==0 ? n%2 : 0);
        f.hbm[row][write ? 1 : 0]=add128(f.hbm[row][write ? 1 : 0],amount,"Norm HBM bytes");
        f.work[row==0 ? HBM0 : HBM1]+=static_cast<long double>(amount)/eta(hw,"eta_hbm_vector");
        route_bytes_to(f,hw,topo,row,amount,destination(cores,topo.nx,topo.ny),write);
      }
      add_bytes_work(f,write ? SRAM_READ : SRAM_WRITE,payload,"Norm IO SRAM bytes");
    };
    group(active,mul128(mul128(count/active,active,"Norm uniform rows"),row_bytes,"Norm uniform bytes"));
    if(count%active)group(count%active,mul128(count%active,row_bytes,"Norm partial bytes"));
    f=scale_flow(f,repeats);
    f.private_s=private_bytes(mul128(mul128(ceildiv(count,active),repeats,"Norm serial IO rows"),row_bytes,"Norm serial IO bytes"),
        write ? SRAM_READ : SRAM_WRITE);
    return f;
  };
  const auto conversion=[&](u128 n,const std::string& dt) {
    return dt=="fp32" ? EqWork{} : weighted_ops(n,dt=="int4" ? perf(hw,"kappa_quant_eqops",4) : 1,"Norm conversion");
  };
  const u128 row_elements=hidden;
  const u128 parameter_elements=mul128(parameters,hidden,"Norm parameter elements");
  const EqWork row_conversion=conversion(mul128(2,row_elements,"Norm two SRAM input visits"),ad)+
      conversion(row_elements,od)+(cached ? EqWork{} : conversion(parameter_elements,pd));
  const EqWork row_ops=EqWork(mul128(layernorm ? 8 : 6,row_elements,"Norm row math"))+row_conversion;
  const EqWork cache_ops=cached ? conversion(parameter_elements,pd) : EqWork{};
  // Converted input tiles are produced and consumed for each of the two
  // SRAM passes. The output cast uses the same FP32 working tile.
  const u128 xconverted=ad=="fp32" ? 0 : mul128(8,row_elements,"Norm converted input bytes");
  const u128 yconverted=od=="fp32" ? 0 : mul128(4,row_elements,"Norm output cast bytes");
  const u128 row_read=add128(add128(mul128(2,xrow,"Norm input SRAM visits"),xconverted,"Norm input SRAM reads"),
      add128(pconverted,yconverted,"Norm parameter/output reads"),"Norm row SRAM reads");
  const u128 row_write=add128(add128(xconverted,bytes(hidden,"fp32"),"Norm compute writes"),
      od=="fp32" ? 0 : yrow,"Norm output cast writes");
  const u128 stream_conversion_read=!cached && pd!="fp32" ? prow : 0;
  const u128 stream_conversion_write=!cached && pd!="fp32" ? pconverted : 0;
  const u128 cache_read=cached && pd!="fp32" ? prow : 0;
  const u128 cache_write=cached && pd!="fp32" ? pconverted : 0;
  const auto compute=[&](u64 count,u64 repeats,u64 serial_rows,bool first) {
    Flow f;
    if(!cached)f=io(count,repeats,prow,false);
    const u128 n=mul128(count,repeats,"Norm compute rows");
    const u128 cr=add128(mul128(n,add128(row_read,stream_conversion_read,"Norm compute read row"),"Norm compute reads"),
        first ? mul128(active,cache_read,"Norm first conversion reads") : 0,"Norm compute reads");
    const u128 cw=add128(mul128(n,add128(row_write,stream_conversion_write,"Norm compute write row"),"Norm compute writes"),
        first ? mul128(active,cache_write,"Norm first conversion writes") : 0,"Norm compute writes");
    add_bytes_work(f,SRAM_READ,cr,"Norm compute SRAM reads");
    add_bytes_work(f,SRAM_WRITE,cw,"Norm compute SRAM writes");
    add_bytes_work(f,LOCAL,add128(cr,cw,"Norm compute LOCAL bytes"),"Norm compute LOCAL bytes");
    f.vec=row_ops.multiply(as_u64(n,"Norm operation rows"))+(first ? cache_ops.multiply(active) : EqWork{});
    f.compute_s=compute_time(hw,0,row_ops.multiply(serial_rows)+(first ? cache_ops : EqWork{}),1);
    const auto sr=add128(row_read,stream_conversion_read,"Norm private read row");
    const auto sw=add128(add128(row_write,stream_conversion_write,"Norm private write row"),cached ? 0 : prow,"Norm private writes with stream landing");
    const auto row_time=std::max({static_cast<long double>(compute_time(hw,0,row_ops,1)),
        private_bytes(sr,SRAM_READ),private_bytes(sw,SRAM_WRITE),
        private_bytes(add128(sr,add128(row_write,stream_conversion_write,"Norm LOCAL writes"),"Norm private LOCAL"),LOCAL)});
    f.private_s=row_time*serial_rows;
    if(first)f.private_s+=std::max({static_cast<long double>(compute_time(hw,0,cache_ops,1)),
        private_bytes(cache_read,SRAM_READ),private_bytes(cache_write,SRAM_WRITE),private_bytes(add128(cache_read,cache_write,"Norm cache LOCAL"),LOCAL)});
    return f;
  };
  Flow load=io(rows,executions,xrow,false);
  Flow first_load=io(active,1,xrow,false);
  if(cached) {
    const Flow parameter_load=io(active,1,prow,false);
    load=add_flow(load,parameter_load);first_load=add_flow(first_load,parameter_load);
  }
  Flow comp=compute(rows,executions,total_rounds,true);
  Flow first_comp=compute(active,1,1,true);
  Flow store=io(rows,executions,yrow,true),last_store=io(last_rows,1,yrow,true);
  const auto begin=plan.epochs.size(),descriptions_begin=service_description_count(plan);
  const bool prefix_drain=executions==1 && rounds>=3;
  const bool two_cohort=executions==1 && rounds==2;
  const auto append=[&](const char* suffix,Flow l,Flow c,Flow s,std::array<double,2> initial) {
    Epoch e;e.name=std::string("norm_row_resident_v1")+suffix;e.credit_initial=initial;
    e.credit_capacity={{static_cast<double>(slots),1}};
    for(auto item:{std::pair<Role,Flow*>{Role::Load,&l},{Role::Compute,&c},{Role::Store,&s}}) {
      Activity a;a.role=item.first;a.id=plan.epochs.size()*3+e.activities.size()+1;
      seal_activity(a,*item.second,hw,active);a.credit_curve={{0,{{0,0}}},{1,a.credit_delta}};
      if(std::any_of(a.work.begin(),a.work.end(),[](double v){return v>0;}))e.activities.push_back(std::move(a));
    }
    plan.epochs.push_back(std::move(e));
  };
  first_load.credit={{1,0}};first_comp.credit={{-1,0}};
  append("_first_row_load",first_load,{}, {},{{0,0}});
  if(prefix_drain || two_cohort) {
    Flow next_load=prefix_drain ? io(active,1,xrow,false) : subtract_io_flow(load,first_load);
    next_load.credit={{1,0}};
    append("_first_compute_and_next_load",next_load,first_comp,{},{{1,0}});
    Flow last_compute=compute(last_rows,1,1,false);
    if(prefix_drain) {
      Flow remaining_load=subtract_io_flow(subtract_io_flow(load,first_load),next_load);
      Flow middle_compute=subtract_compute_flow(subtract_compute_flow(comp,first_comp),last_compute);
      // Uncached parameters travel with Compute; preserve their typed HBM
      // payload in both prefix partitions, independently of resource work.
      for(std::size_t row=0;row<2;++row)for(std::size_t direction=0;direction<2;++direction)
        middle_compute.hbm[row][direction]=comp.hbm[row][direction]-first_comp.hbm[row][direction]-last_compute.hbm[row][direction];
      Flow penultimate_store=io(active,1,yrow,true);
      Flow prefix_store=subtract_io_flow(subtract_io_flow(store,last_store),penultimate_store);
      remaining_load.credit={{static_cast<long double>(rounds-3),0}};
      middle_compute.credit={{-static_cast<long double>(rounds-2),static_cast<long double>(rounds-3)}};
      prefix_store.credit={{0,-static_cast<long double>(rounds-2)}};
      append("_middle_with_reserved_tail",remaining_load,middle_compute,prefix_store,{{1,1}});
      plan.epochs.back().credit_capacity={{1,1}};
      last_compute.credit={{-1,0}};penultimate_store.credit={{0,-1}};
      append("_last_compute_and_penultimate_store",{},last_compute,penultimate_store,{{1,1}});
    } else {
      Flow first_store=subtract_io_flow(store,last_store);
      last_compute.credit={{-1,0}};first_store.credit={{0,-1}};
      append("_last_compute_and_first_store",{},last_compute,first_store,{{1,1}});
    }
  } else append("_first_row_compute",{},first_comp,{},{{1,0}});
  if(total_rounds>1 && !prefix_drain && !two_cohort) {
    Flow l=subtract_io_flow(load,first_load),c=subtract_compute_flow(comp,first_comp),s=subtract_io_flow(store,last_store);
    // Unlike elementwise Compute, streamed gamma/beta carry real HBM payload.
    for(std::size_t row=0;row<2;++row)for(std::size_t direction=0;direction<2;++direction) {
      if(first_comp.hbm[row][direction]>comp.hbm[row][direction])
        fail("mapping_invariant: Norm first Compute exceeds total payload");
      c.hbm[row][direction]=comp.hbm[row][direction]-first_comp.hbm[row][direction];
    }
    l.credit={{static_cast<long double>(total_rounds-1),0}};
    c.credit={{-static_cast<long double>(total_rounds-1),static_cast<long double>(total_rounds-2)}};
    s.credit={{0,-static_cast<long double>(total_rounds-1)}};
    append("_bounded_remaining_rows",l,c,s,{{0,1}});
  }
  last_store.credit={{0,-1}};
  append("_last_row_store",{}, {},last_store,{{0,1}});
  const u64 descriptions=service_description_count(plan)-descriptions_begin;
  if(descriptions>(prefix_drain ? 16 : 8))fail("mapping_invariant: Norm description budget exceeded");
  plan.summary["service_curves"].push_back({{"model",prefix_drain ? "analytical_cumulative_norm_rows_v2_bounded_prefix_drain" :
      two_cohort ? "analytical_cumulative_norm_rows_v2_two_cohort_prefetch" : "analytical_cumulative_norm_rows_v1"},
      {"epoch_begin",begin},{"epoch_count",plan.epochs.size()-begin},{"description_count",descriptions},
      {"description_budget",descriptions<=5 ? 5 : descriptions<=8 ? 8 : 16},{"input_unit","one_complete_input_row_at_most_loaded_core"},
      {"output_unit","one_visible_normalized_row_at_most_loaded_core"},{"input_units",total_rounds},{"output_units",total_rounds},
      {"allocated_input_slots_per_core",slots},{"allocated_output_slots_per_core",slots},
      {"reserved_last_output_slots_per_core",total_rounds>1 ? 1 : 0},
      {"first_output_release","first_row_compute_terminal_after_complete_row_input"},
      {"first_output_epoch",begin+1},{"first_output_activity_id",plan.epochs.at(begin+1).activities.back().id},
      {"readiness_approximation",prefix_drain ? "fixed_prefetch_middle_reserved_last_input_and_penultimate_output" :
          two_cohort ? "last_row_prefetch_during_first_compute" : "remaining_rows_affine_with_withheld_final_output"}});
  if(prefix_drain) {
    auto& audit=plan.summary["service_curves"].back();
    audit.erase("reserved_last_output_slots_per_core");
    audit["reserved_last_input_slots_per_core"]=1;
    audit["reserved_penultimate_output_slots_per_core"]=1;
    audit["reserved_input_terminal_transfer_units"]=1;
    audit["reserved_input_terminal_transfer_epoch"]=begin+3;
  }
  plan.summary["norm_row_ledger"]={{"mapping","invocation_local_row_id_mod_core_count"},
      {"parameter_cache",cached ? "complete_fp32_gamma_beta_per_consumer_core" : "streamed_each_row"},
      {"parameter_hbm_loads",cached ? active : total_rows},{"parameter_vectors",parameters},
      {"parameter_bytes_per_load",as_u64(prow,"Norm parameter payload")},
      {"input_hbm_passes",1},{"input_sram_passes",2},{"row_input_bytes",as_u64(xrow,"Norm row input")},
      {"row_output_bytes",as_u64(yrow,"Norm row output")},{"parameter_buffer_bytes_per_core",as_u64(parameter_buffer,"Norm parameter buffer")},
      {"sram_peak_bytes_per_core",as_u64(peak,"Norm SRAM peak")},{"reduction_state_bytes_per_core",16},
      {"metadata_bytes_per_core",64},{"multi_pass_hbm_rows","not_implemented"}};
  if(executions>1) {
    plan.summary["prediction_status"]="not_available";
    plan.summary["prediction_scope"]="body_timing_only_norm_region_barriers_unmodeled";
  }
  const u128 xb=mul128(total_rows,xrow,"Norm input bytes"),yb=mul128(total_rows,yrow,"Norm output bytes");
  const u128 pb=mul128(cached ? active : total_rows,prow,"Norm parameter bytes");
  m.logical=add128(m.logical,add128(add128(xb,yb,"Norm logical tensors"),mul128(executions,prow,"Norm logical parameters"),"Norm logical tensors"),"logical bytes");
  m.hbm_read=add128(m.hbm_read,add128(xb,pb,"Norm reads"),"HBM reads");m.hbm_write=add128(m.hbm_write,yb,"HBM writes");
  m.source_vops=add128(m.source_vops,mul128(mul128(total_rows,hidden,"Norm math elements"),layernorm ? 8 : 6,"Norm math"),"source Vec ops");
  m.vecops=m.vecops+comp.vec;
  m.conversion_elements=add128(m.conversion_elements,
      add128(mul128(total_rows,add128(ad=="fp32" ? 0 : 2*row_elements,od=="fp32" ? 0 : row_elements,"Norm input/output conversions"),"Norm input/output conversions"),
          pd=="fp32" ? 0 : mul128(cached ? active : total_rows,parameter_elements,"Norm parameter conversions"),"Norm conversions"),"conversion elements");
  for(const auto* f:{&load,&comp,&store}) {
    m.sram_read=add128(m.sram_read,static_cast<u128>(f->work[SRAM_READ]),"Norm SRAM reads");
    m.sram_write=add128(m.sram_write,static_cast<u128>(f->work[SRAM_WRITE]),"Norm SRAM writes");
    m.local=add128(m.local,static_cast<u128>(f->work[LOCAL]),"Norm LOCAL bytes");
  }
  m.sram_peak=std::max(m.sram_peak,peak);m.output_rounds=add128(m.output_rounds,total_rounds,"Norm rounds");
  m.waves=add128(m.waves,total_rounds,"Norm waves");m.active_cores=active;m.template_id="norm_row_resident_v1";
  m.assumptions.push_back("complete_row_resident_statistics_then_apply_with_per_core_parameter_ownership");
}

// Elementwise kernels have no row reduction or K accumulation. Keep real
// full/tail blocks and per-invocation packing; never split 257 as 128+129.
void add_elementwise(Plan& plan, Metrics& m, const Hardware& hw,
    const Topology& topo, u64 count, u64 executions, u64 input_streams,
    u64 ops_per_element, const std::string& ad, const std::string& od,
    const std::string& label) {
  const elementwise_geometry::Geometry g(count,executions,du(hw,"die","core_count"));
  const auto input_bytes=[&](u64 n) {
    return mul128(bytes(n,ad),input_streams,"elementwise input streams");
  };
  const u128 in=mul128(input_bytes(count),executions,"elementwise input bytes");
  const u128 out=mul128(bytes(count,od),executions,"elementwise output bytes");
  const u128 ops=mul128(g.total_elements,ops_per_element,"elementwise operations");
  const u64 max_tile=std::min<u64>(count,256);
  // This template allocates two raw input and two pending-output slots per
  // Core plus one FP32 working block. Slot bytes use actual storage packing,
  // independently for each input tensor, including an odd INT4 tail.
  const u64 slots=g.total_rounds==1 ? 1 : 2;
  const bool single=g.total_blocks==g.first_blocks;
  const bool separate_two_cohort_drain=executions==1 && g.rounds==2;
  const bool separate_prefix_drain=executions==1 && g.rounds>=3;
  const bool per_core_fifo=!single && !separate_two_cohort_drain && !separate_prefix_drain;
  const u128 peak=add128(add128(mul128(slots,input_bytes(max_tile),"elementwise input slots"),
      mul128(slots,bytes(max_tile,od),"elementwise output slots"),"elementwise slots"),
      mul128(max_tile,4,"elementwise FP32 work block"),"elementwise SRAM peak");
  if(static_cast<long double>(as_u64(peak,"elementwise SRAM peak"))>
      perf(hw,"rho_sram",0.9)*dd(hw,"core","sram_bytes_per_core"))
    fail("unsupported_workspace: elementwise double buffers exceed per-Core SRAM");
  for(const auto units:{g.total_blocks,g.first_blocks,g.last_blocks,
                        g.total_blocks-g.first_blocks,g.total_blocks-g.last_blocks,
                        g.total_blocks==g.first_blocks ? u64{0} :
                            g.total_blocks-g.first_blocks-g.last_blocks,
                        g.total_rounds,g.total_rounds-1,
                        per_core_fifo || separate_prefix_drain ? g.total_rounds-2 : u64{0},
                        separate_prefix_drain ? g.total_rounds-3 : u64{0}})
    if(static_cast<long double>(static_cast<double>(units))!=static_cast<long double>(units))
      fail("overflow: elementwise cumulative credit endpoint is not exactly representable");

  const bool tail_only_last=g.tail && g.last_blocks==1;
  const u128 full_rounds=mul128(g.rounds-(tail_only_last ? 1 : 0),executions,"elementwise full cohorts");
  const u128 tail_rounds=tail_only_last ? executions : 0;
  const auto serial_bytes=[&](const std::string& dt) {
    return add128(mul128(bytes(256,dt),full_rounds,"elementwise full service"),
        mul128(bytes(g.tail,dt),tail_rounds,"elementwise tail service"),"elementwise serial bytes");
  };
  const auto private_bytes_time=[&](u128 n,std::size_t resource) {
    const long double rate=hw.rates[resource]/static_cast<long double>(du(hw,"die","core_count"));
    if(!(rate>0)) fail("invalid_hardware: elementwise per-Core service rate is zero");
    return static_cast<long double>(as_u64(n,"elementwise serial service bytes"))/rate;
  };
  const auto io_for=[&](u64 full,u64 tail,u64 repeats,bool writing,u128 serial) {
    Flow f;
    // Byte-weighted RR destinations are a complete prefix repeated uniformly,
    // a remaining full-block prefix, and at most one tail point. Three groups
    // suffice regardless of Core/block/repeat count, including second moments
    // for uniform-source hops and exact point distances for explicit ingress.
    const auto group=[&](u128 n,const AxisDist& d) {
      const u64 total=as_u64(n,"elementwise IO group");
      const std::array<u64,2> rows{{total/2+total%2,total/2}};
      for(std::size_t row=0;row<2;++row) {
        const auto value=rows[row];
        f.hbm[row][writing ? 1 : 0]=add128(f.hbm[row][writing ? 1 : 0],value,"elementwise HBM row");
        f.work[row==0 ? HBM0 : HBM1]+=static_cast<long double>(value)/eta(hw,"eta_hbm_vector");
        route_bytes_to(f,hw,topo,row,value,d,writing);
      }
      add_bytes_work(f,writing ? SRAM_READ : SRAM_WRITE,n,"elementwise IO SRAM");
    };
    const auto block_bytes=[&](u64 n){return writing ? bytes(n,od) : input_bytes(n);};
    const u64 cycles=full/g.active,remaining=full%g.active;
    if(cycles) group(mul128(block_bytes(256),mul128(cycles,g.active,"elementwise uniform blocks"),
        "elementwise uniform bytes"),destination(g.active,topo.nx,topo.ny));
    if(remaining) group(mul128(block_bytes(256),remaining,"elementwise full prefix bytes"),
        destination(remaining,topo.nx,topo.ny));
    if(tail) {
      AxisDist d=destination(1,topo.nx,topo.ny);d.point=true;
      d.ex=remaining%topo.nx;d.ey=remaining/topo.nx;
      d.ex2=d.ex*d.ex;d.ey2=d.ey*d.ey;
      d.xright=d.ex>=topo.nx/2;d.xleft=1-d.xright;
      d.yhigh=d.ey>=topo.ny/2;d.ylow=1-d.yhigh;
      group(block_bytes(tail),d);
    }
    // Each invocation's storage allocation restarts the documented row stripe.
    f=scale_flow(f,repeats);
    f.private_s=private_bytes_time(serial,writing ? SRAM_READ : SRAM_WRITE);
    return f;
  };
  const auto tile_private=[&](u64 n) {
    const u128 read=input_bytes(n),write=bytes(n,od);
    return std::max({static_cast<long double>(compute_time(hw,0,mul128(n,ops_per_element,
        "elementwise block operations"),1)),private_bytes_time(read,SRAM_READ),
        private_bytes_time(write,SRAM_WRITE),
        private_bytes_time(add128(read,write,"elementwise block LOCAL"),LOCAL)});
  };
  const auto compute_for=[&](u128 read,u128 write,u64 elements,u64 serial_elements,
      u128 full,u128 tail) {
    Flow f;
    add_bytes_work(f,SRAM_READ,read,"elementwise Compute SRAM reads");
    add_bytes_work(f,SRAM_WRITE,write,"elementwise Compute SRAM writes");
    add_bytes_work(f,LOCAL,add128(read,write,"elementwise LOCAL"),"elementwise LOCAL");
    f.vec=mul128(elements,ops_per_element,"elementwise Compute operations");
    // Cohorts run serially; each cohort's longest real block defines its
    // compute gate. Totals/A would undercharge a partial cohort.
    f.compute_s=compute_time(hw,0,mul128(serial_elements,ops_per_element,
        "elementwise serial operations"),1);
    f.private_s=full*tile_private(256)+tail*tile_private(g.tail);
    return f;
  };
  const u64 first_max=g.full_blocks ? 256 : g.tail;
  Flow load=io_for(g.full_blocks,g.tail,executions,false,
      mul128(serial_bytes(ad),input_streams,"elementwise input service"));
  Flow store=io_for(g.full_blocks,g.tail,executions,true,serial_bytes(od));
  Flow compute=compute_for(in,out,g.total_elements,g.serial_elements,full_rounds,tail_rounds);
  Flow first_load=io_for(std::min(g.full_blocks,g.first_blocks),
      g.first_blocks>g.full_blocks ? g.tail : 0,1,false,input_bytes(first_max));
  Flow first_compute=single ? compute : compute_for(input_bytes(g.first_elements),
      bytes(g.first_elements,od),g.first_elements,first_max,g.full_blocks ? 1 : 0,g.full_blocks ? 0 : 1);
  Flow final_store=single ? store : io_for(g.last_blocks-(g.tail!=0),g.tail,1,true,
      bytes(tail_only_last ? g.tail : 256,od));
  const auto begin=plan.epochs.size(), description_begin=service_description_count(plan);
  // RR Core counts can differ. Aggregate free slots cannot be borrowed by a
  // busier Core: L/C have n_i-1 remaining blocks there. Bound their maximum
  // (T-1)*(pL-pC) by two slots. Visible outputs use the most-loaded Core's
  // 1+(T-2)*pC-(T-1)*pS in [0,1], reserving the other slot for withheld Last.
  // Every other Core's visible occupancy is a convex combination of this
  // value and 1-pS; hence it too is in [0,1]. No per-Core runtime state is used.
  const bool core_units=per_core_fifo || separate_prefix_drain;
  const u64 first_credit=core_units ? 1 : g.first_blocks;
  const u64 last_credit=core_units ? 1 : g.last_blocks;
  const u64 credit_blocks=core_units ? g.total_rounds : g.total_blocks;
  const u64 input_capacity=separate_prefix_drain ? 1 : per_core_fifo ? 2 : slots*g.active;
  const u64 output_capacity=core_units ? 1 : slots*g.active;
  const auto append=[&](const char* suffix,Flow l,Flow c,Flow s,
      std::array<double,2> initial,u64 active) {
    Epoch e; e.name=label+suffix; e.credit_initial=initial;
    e.credit_capacity={{static_cast<double>(input_capacity),static_cast<double>(output_capacity)}};
    for(auto item:{std::pair<Role,Flow*>{Role::Load,&l},{Role::Compute,&c},{Role::Store,&s}}) {
      Activity a; a.role=item.first; a.id=plan.epochs.size()*3+e.activities.size()+1;
      seal_activity(a,*item.second,hw,active);
      a.credit_curve={{0,{{0,0}}},{1,a.credit_delta}};
      if(std::any_of(a.work.begin(),a.work.end(),[](double v){return v>0;}))
        e.activities.push_back(std::move(a));
    }
    plan.epochs.push_back(std::move(e));
  };
  first_load.credit={{static_cast<long double>(first_credit),0}};
  append("_load_first",first_load,{}, {},{{0,0}},g.active);
  first_compute.credit={{-static_cast<long double>(first_credit),0}};
  if(separate_prefix_drain) {
    // Fixed prefix, aggregate middle and drain, independent of cohort count.
    // All R1 middle cohorts are full on every active Core. Reserve one input
    // slot for Last and one output slot for Penultimate while the middle
    // consumes/produces the remaining slots. No wave cursor is introduced.
    Flow second_load=first_load;
    append("_first_output_work_and_second_load",second_load,first_compute,{},{{1,0}},g.active);
    // First and Second may coexist in both physical input slots. Only the
    // aggregate middle below reserves a slot for the independently held Last.
    plan.epochs.back().credit_capacity[0]=2;
    const u64 last_max=tail_only_last ? g.tail : 256;
    Flow last_compute=compute_for(input_bytes(g.last_elements),bytes(g.last_elements,od),
        g.last_elements,last_max,tail_only_last ? 0 : 1,tail_only_last ? 1 : 0);
    Flow penultimate_store=io_for(g.active,0,1,true,bytes(256,od));
    Flow remaining_load=subtract_io_flow(subtract_io_flow(load,first_load),second_load);
    Flow middle_compute=subtract_compute_flow(subtract_compute_flow(compute,first_compute),last_compute);
    Flow prefix_store=subtract_io_flow(subtract_io_flow(store,final_store),penultimate_store);
    remaining_load.credit={{static_cast<long double>(g.rounds-3),0}};
    middle_compute.credit={{-static_cast<long double>(g.rounds-2),
        static_cast<long double>(g.rounds-3)}};
    prefix_store.credit={{0,-static_cast<long double>(g.rounds-2)}};
    append("_middle_service_with_reserved_tail",remaining_load,middle_compute,prefix_store,{{1,1}},g.active);
    // The preceding Load terminal materializes reserved Last, omitted from
    // middle credit. Transfer that one slot explicitly into the drain epoch.
    last_compute.credit={{-1,0}};
    penultimate_store.credit={{0,-1}};
    append("_last_compute_and_penultimate_store",{},last_compute,penultimate_store,{{1,1}},g.active);
  } else if(separate_two_cohort_drain) {
    // Fixed first/last cohorts: the second input slot supports this bounded
    // prefetch during First Compute. Last Compute then has independent
    // progress, so it can finish before First Store without a fluid load tie.
    Flow last_load=subtract_io_flow(load,first_load);
    Flow last_compute=subtract_compute_flow(compute,first_compute);
    Flow first_store=subtract_io_flow(store,final_store);
    last_load.credit={{static_cast<long double>(g.last_blocks),0}};
    append("_first_output_work_and_last_load",last_load,first_compute,{},
        {{static_cast<double>(g.first_blocks),0}},g.active);
    last_compute.credit={{-static_cast<long double>(g.last_blocks),0}};
    first_store.credit={{0,-static_cast<long double>(g.first_blocks)}};
    // Keep First Store's full active prefix. Last Compute retains its own
    // longest-Core compute/PRIVATE bounds for a partial final cohort.
    append("_last_compute_and_first_store",{},last_compute,first_store,
        {{static_cast<double>(g.last_blocks),static_cast<double>(g.first_blocks)}},g.active);
  } else {
    append("_first_output_work",{},first_compute,{},
        {{static_cast<double>(first_credit),0}},g.active);
  }
  if(per_core_fifo) {
    Flow remaining_load=subtract_io_flow(load,first_load);
    Flow remaining_compute=subtract_compute_flow(compute,first_compute);
    Flow early_store=subtract_io_flow(store,final_store);
    remaining_load.credit={{static_cast<long double>(credit_blocks-first_credit),0}};
    remaining_compute.credit={{-static_cast<long double>(credit_blocks-first_credit),
        static_cast<long double>(credit_blocks-first_credit-last_credit)}};
    early_store.credit={{0,-static_cast<long double>(credit_blocks-last_credit)}};
    append("_cumulative_service",remaining_load,remaining_compute,early_store,
        {{0,static_cast<double>(first_credit)}},g.active);
  }
  final_store.credit={{0,-static_cast<long double>(last_credit)}};
  append("_store_last",{}, {},final_store,{{0,static_cast<double>(last_credit)}},g.last_blocks);
  const auto descriptions=service_description_count(plan)-description_begin;
  if(descriptions>(separate_prefix_drain ? 16 : 8))
    fail("mapping_invariant: elementwise description budget exceeded");
  const auto& first_epoch=plan.epochs.at(begin+1);
  const auto first_compute_activity=std::find_if(first_epoch.activities.begin(),
      first_epoch.activities.end(),[](const Activity& a){return a.role==Role::Compute;});
  if(first_compute_activity==first_epoch.activities.end())
    fail("mapping_invariant: first elementwise Compute is absent");
  plan.summary["service_curves"].push_back({
      {"model",separate_prefix_drain ? "analytical_cumulative_elementwise_v5_bounded_prefix_drain" :
          per_core_fifo ? "analytical_cumulative_elementwise_v4_per_core_fifo" :
          separate_two_cohort_drain ? "analytical_cumulative_elementwise_v3_two_cohort_prefetch" :
          "analytical_cumulative_elementwise_v1"},{"epoch_begin",begin},
      {"epoch_count",plan.epochs.size()-begin},{"description_count",descriptions},
      {"description_budget",descriptions<=5 ? 5 : descriptions<=8 ? 8 : 16},
      {"input_unit",core_units ? "one_input_block_at_the_most_loaded_core" :
          "one_elementwise_input_block_all_streams"},
      {"output_unit",core_units ? "one_visible_output_block_at_the_most_loaded_core" :
          "one_elementwise_output_block"},
      {"input_units",credit_blocks},{"output_units",credit_blocks},
      {"block_elements",256},{"blocks_per_invocation",g.blocks},
      {"full_blocks_per_invocation",g.full_blocks},{"tail_elements",g.tail},
      {"executions",executions},{"input_streams",input_streams},
      {"first_load_units",first_credit},{"first_load_elements",g.first_elements},
      {"first_output_input_bytes",as_u64(input_bytes(g.first_elements),"elementwise first input")},
      {"first_output_compute_work_s",as_double(first_compute.compute_s,"elementwise first work")},
      {"first_output_release","first_cohort_compute_activity_terminal"},{"first_output_epoch",begin+1},
      {"first_output_activity_id",first_compute_activity->id},
      {"final_drain_units",last_credit},
      {"input_capacity_units",input_capacity},{"output_capacity_units",output_capacity},
      {"allocated_input_slots_per_core",slots},{"allocated_output_slots_per_core",slots},
      {"sram_allocation","fixed_max_block_slots_plus_one_fp32_work_block"},
      {"mapping","invocation_local_round_robin_256_element_blocks"},
      {"routing","byte_weighted_full_prefix_and_tail_point_v1"},
      {"hbm_row_striping","input_streams_concatenated_per_block_invocation_restarts_row0"},
      {"readiness_approximation",separate_prefix_drain
          ? "fixed_prefetch_middle_affine_reserved_last_input_and_penultimate_output"
          : separate_two_cohort_drain
          ? "last_input_prefetch_during_first_compute_epoch_barrier_before_first_store"
          : "aggregate_bulk_affine_including_invocation_boundaries"}});
  if(core_units) {
    auto& audit=plan.summary["service_curves"].back();
    audit["physical_input_blocks"]=g.total_blocks;
    audit["physical_output_blocks"]=g.total_blocks;
    if(separate_prefix_drain) {
      audit["reserved_last_input_slots_per_core"]=1;
      audit["reserved_penultimate_output_slots_per_core"]=1;
      audit["reserved_input_terminal_transfer_units"]=1;
      audit["reserved_input_terminal_transfer_epoch"]=begin+3;
      audit["input_capacity_scope"]="middle_slot_excludes_reserved_last;prefix_prefetch_uses_two_slots";
    } else audit["reserved_last_output_slots_per_core"]=1;
  }
  m.logical=add128(m.logical,add128(in,out,"elementwise logical bytes"),"logical bytes");
  m.hbm_read=add128(m.hbm_read,in,"elementwise reads");
  m.hbm_write=add128(m.hbm_write,out,"elementwise writes");
  m.source_vops=add128(m.source_vops,ops,"elementwise math");
  m.vecops=add128(m.vecops,ops,"elementwise Vec work");
  m.sram_read=add128(m.sram_read,add128(in,out,"elementwise SRAM"),"SRAM read");
  m.sram_write=add128(m.sram_write,add128(in,out,"elementwise SRAM"),"SRAM write");
  m.local=add128(m.local,add128(in,out,"elementwise LOCAL"),"LOCAL");
  m.sram_peak=std::max(m.sram_peak,peak);
  m.output_rounds=add128(m.output_rounds,g.total_rounds,"elementwise rounds");
  m.waves=add128(m.waves,g.total_rounds,"elementwise waves");
  m.active_cores=g.active; m.template_id=label;
  m.assumptions.push_back("fixed_256_elementwise_blocks_exact_storage_packing_v1");
}

std::shared_ptr<Plan> make_gemm_mapping(const Hardware& hw,const nlohmann::json& task,
    const Topology& topo,u64 M,u64 N,u64 K,const std::string& ad,
    const std::string& wd,const std::string& od,u64 layers,u64 passes,
    const GemmChoice& choice,bool seal_fingerprint,bool ranking_only) {
  auto p=new_plan(hw,task);Metrics m;
  m.assumptions={"engineering_assumption","bounded_tile_candidates","exact_per_tile_INT4_tail_packing"};
  add_gemm_epoch(*p,m,hw,topo,M,N,K,ad,wd,od,layers,passes,choice,
      choice.wr ? "weight_resident_v2" : "output_stationary_v2");
  finalize_summary(*p,m,hw,ranking_only);
  p->summary["template_id"]="gemm_"+m.dataflow;
  p->summary["dataflow_id"]=m.dataflow;
  if(!ranking_only) {
    p->summary["source_matrix_ops"]=as_u64(m.source_mops,"source matrix ops");
    p->summary["executed_matrix_ops"]=as_u64(m.executed_mops,"executed matrix ops");
    p->summary["recomputed_matrix_ops"]=0;p->summary["recompute_matrix_ops"]=0;
  }
  p->summary["tm"]=choice.tm;p->summary["tn"]=choice.tn;p->summary["tk"]=choice.tk;
  if(seal_fingerprint)p->fingerprint=plan_to_json(*p).at("fingerprint").get<std::string>();
  return p;
}

std::vector<std::shared_ptr<Plan>> enumerate_gemm(const Hardware& hw,
                                                   const nlohmann::json& task,
                                                   const Topology& topo,
                                                   u64 M,u64 N,u64 K,
                                                   bool seal_fingerprints = true,
                                                   bool ranking_only = false) {
  const auto ts=options(M,{1,16,64,256}); const auto ns=options(N,{64,256,1024,4096});
  const auto ks=options(K,{64,256,1024,4096});
  const u64 layers=getu(task,"layer_instances",1), passes=getu(task,"weight_passes",1);
  const std::string ad=dtype(task,"activations"), wd=dtype(task,"weights"), od=dtype(task,"output");
  const std::string profile=gets(task,"execution_profile");
  const u64 ncore=du(hw,"die","core_count");
  const double cap=perf(hw,"rho_sram",0.9)*dd(hw,"core","sram_bytes_per_core");
  std::vector<std::shared_ptr<Plan>> result;
  for (bool wr : {false,true}) for (u64 tm:ts) for (u64 tn:ns) for (u64 tk:ks) {
    const u128 peak=gemm_sram_peak(M,N,K,tm,tn,tk,ad,wd,od,profile,wr);
    if (static_cast<long double>(as_u64(peak,"SRAM peak")) > cap) continue;
    result.push_back(make_gemm_mapping(hw,task,topo,M,N,K,ad,wd,od,layers,passes,
        GemmChoice{tm,tn,tk,wr},seal_fingerprints,ranking_only));
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
  const double kappa_quant=perf(hw,"kappa_quant_eqops",4);
  const double kappa_acc=perf(hw,"kappa_acc_convert_eqops",2);
  auto sum_kv_lengths=[](const Span& s)->u128 {
    return attention_sum_kv_lengths(s);
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
      const u128 batch_kv_tiles=mul128(all_kv_tiles,s.batch,"attention batch KV tiles");
      const u128 request_count=mul128(s.count,s.batch,"attention batch requests");
      // Materialized score/primitive paths operate on the complete rectangle.
      // Only fused causal attention skips future KV blocks.
      const bool materialized=full||mode!="both";
      const auto causal_stat=causal&&!materialized?causal_counts(s,tq,tk):AttentionTileCounts{};
      if (materialized) { sl=attention_span_rectangle_slots(s);pr=mul128(qtiles,kv_sum,"materialized attention KV reads"); }
      else if (causal) { sl=causal_stat.slots; pr=causal_stat.prefix; }
      else { sl=mul128(s.q,kv_sum,"noncausal score slots");pr=mul128(qtiles,kv_sum,"noncausal attention KV reads"); }
      const u128 request_heads=mul128(request_count,qh,"attention heads");
      const u128 score_slots=materialized?attention_span_score_slots(s,qh):
          mul128(mul128(sl,s.batch,"attention batch score slots"),qh,"attention score heads");
      const u128 score_valid=mul128(mul128(valid_pairs(s,causal),s.batch,"attention batch valid pairs"),qh,"attention valid heads");
      const u128 query_count=attention_span_query_rows(s,qh);
      output_tasks=add128(output_tasks,mul128(qtiles,request_heads,"attention output tasks"),"attention output tasks");
      const u128 active_rows_one=causal&&!materialized?causal_stat.active_rows:mul128(s.q,s.count,"attention active rows");
      const u128 active_rows=mul128(active_rows_one,s.batch,"attention batch active rows");
      active_queryrows=add128(active_queryrows,mul128(active_rows,qh,"attention active rows"),"attention active rows");
      const u128 row_blocks_one=causal&&!materialized?causal_stat.row_blocks:mul128(s.q,all_kv_tiles,"attention row/KV block visits");
      const u128 tile_blocks_one=causal&&!materialized?causal_stat.blocks:mul128(qtiles,all_kv_tiles,"attention tile/KV block visits");
      const u128 row_blocks=mul128(row_blocks_one,s.batch,"attention batch row/KV block visits");
      const u128 tile_blocks=mul128(tile_blocks_one,s.batch,"attention batch tile/KV block visits");
      row_block_visits=add128(row_block_visits,mul128(row_blocks,qh,"attention row/KV block visits"),"attention row/KV block visits");
      tile_block_visits=add128(tile_block_visits,mul128(tile_blocks,qh,"attention tile/KV block visits"),"attention tile/KV block visits");
      slots=add128(slots,score_slots,"attention slots");
      valid=add128(valid,score_valid,"valid score pairs");
      queryrows=add128(queryrows,query_count,"query rows");
      prefixes=add128(prefixes,mul128(mul128(pr,s.batch,"attention batch KV prefixes"),qh,"GQA KV prefix reads"),"GQA KV prefix reads");

      const u128 qtile_raw=tile_grid_bytes(s.q,tq,hd,hd,dtype_q);
      const u128 qtile_i4=tile_grid_bytes(s.q,tq,hd,hd,"int4");
      const u128 q_reuses=materialized&&mode!="av"?batch_kv_tiles:1;
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
      const u128 full_k_read=mul128(mul128(mul128(full_k_raw,qtiles,"full K query tiles"),qh,"full K heads"),s.batch,"full K batch");
      const u128 full_v_read=mul128(mul128(mul128(full_v_raw,qtiles,"full V query tiles"),qh,"full V heads"),s.batch,"full V batch");
      const u128 fused_k_read=mul128(mul128(fused_k_raw,qh,"fused K heads"),s.batch,"fused K batch");
      const u128 fused_v_read=mul128(mul128(fused_v_raw,qh,"fused V heads"),s.batch,"fused V batch");
      const u128 span_kread=mode=="av"?0:(materialized?full_k_read:fused_k_read);
      const u128 span_vread=mode=="qk"?0:(materialized?full_v_read:fused_v_read);
      qk_kbytes=add128(qk_kbytes,span_kread,"attention QK K reads");
      pv_vbytes=add128(pv_vbytes,span_vread,"attention PV V reads");
      kvbytes=add128(kvbytes,add128(span_kread,span_vread,"attention K/V reads"),"attention K/V reads");
      if(mode!="av")k_quant_elements=add128(k_quant_elements,mul128(mul128(mul128(pr,s.batch,"attention batch K prefixes"),hd,"attention K conversions"),qh,"attention K conversions"),"attention K conversions");
      if(mode!="qk")v_quant_elements=add128(v_quant_elements,mul128(mul128(mul128(pr,s.batch,"attention batch V prefixes"),hd,"attention V conversions"),qh,"attention V conversions"),"attention V conversions");
      if(int4path){
        const u128 packed_k_one=causal&&!materialized?packed_causal_prefix(s,tq,tk,hd,hd,pr):mul128(packed_kv_grid(s,tk,hd,hd),qtiles,"attention packed K input");
        const u128 packed_v_one=causal&&!materialized?packed_causal_prefix(s,tq,tk,hd,feature_tile,pr):mul128(packed_kv_grid(s,tk,hd,feature_tile),qtiles,"attention packed V input");
        if(mode!="av"){
          k_packed=add128(k_packed,mul128(mul128(packed_k_one,qh,"attention packed K heads"),s.batch,"attention packed K batch"),"attention packed K");
        }
        if(mode!="qk"){
          v_packed=add128(v_packed,mul128(mul128(packed_v_one,qh,"attention packed V heads"),s.batch,"attention packed V batch"),"attention packed V");
        }
        if(dtype_q!="int4"&&mode!="av")q_scale_tiles=add128(q_scale_tiles,mul128(mul128(qtiles,q_head_copies,"attention Q scales"),q_reuses,"attention Q scale rereads"),"attention Q scales");
        if(dtype_k!="int4"&&mode!="av")k_scale_tiles=add128(k_scale_tiles,mul128(tile_blocks,qh,"attention K head scales"),"attention K scales");
        if(dtype_k!="int4"&&mode!="qk")v_scale_tiles=add128(v_scale_tiles,mul128(tile_blocks,qh,"attention V head scales"),"attention V scales");
      }
      if (mode!="qk")outbytes=add128(outbytes,mul128(tile_grid_bytes(s.q,tq,hd,hd,dtype_o),request_heads,"attention tiled output bytes"),"attention output bytes");
      if(materialized&&mode=="av")av_probbytes=add128(av_probbytes,mul128(mul128(mul128(sl,s.batch,"AV batch score slots"),4,"AV probability bytes"),qh,"AV probability bytes"),"AV probability bytes");

      // Logical bytes are unique source tensors. Physical rereads, score
      // materialization, and conversion buffers stay in their own counters.
      const u128 qlogical=mul128(bytes(mul128(s.q,hd,"logical Q elements"),dtype_q),request_heads,"logical Q bytes");
      const u128 kvlogical=mul128(mul128(sum_storage_tensor_bytes(s,hd,dtype_k),s.batch,"logical KV batch"),mul128(kvh,2,"logical KV tensors"),"logical KV bytes");
      const u128 outlogical=mul128(bytes(mul128(s.q,hd,"logical output elements"),dtype_o),request_heads,"logical output bytes");
      if(mode=="both")logical=add128(logical,add128(add128(qlogical,kvlogical,"logical attention tensors"),outlogical,"logical attention tensors"),"logical attention tensors");
      else if(mode=="qk")logical=add128(logical,add128(qlogical,mul128(bytes(mul128(sl,s.batch,"logical score batch"),"fp32"),qh,"logical score bytes"),"logical QK tensors"),"logical QK tensors");
      else logical=add128(logical,add128(add128(mul128(bytes(mul128(sl,s.batch,"logical probability batch"),"fp32"),qh,"logical probability bytes"),mul128(mul128(sum_storage_tensor_bytes(s,hd,dtype_k),s.batch,"logical V batch"),kvh,"logical V bytes"),"logical AV tensors"),outlogical,"logical AV tensors"),"logical AV tensors");
    }
    // Save the live workspace for one invocation before scaling total
    // physical work across sequential layers/passes.
    const u128 workspace_slots=slots;
    const u128 workspace_queryrows=queryrows;
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
    const EqWork accumulator_vec_ops=int4path?add128(weighted_ops(acc_convert_elements,kappa_acc,"attention accumulator Vec ops"),acc_adds,"attention accumulator Vec ops"):0;
    const EqWork cast_vec_ops=weighted_ops(output_conversion,dtype_o=="int4"?kappa_quant:1,"attention output conversion");
    const u128 vector_matrix_ops=int4path?0:matops;
    auto input_conversion_count=[&](const std::string& dt,u128 elements)->u128 {
      return int4path?(dt=="int4"?0:elements):(dt=="fp32"?0:elements);
    };
    const u128 q_input_conversion=input_conversion_count(dtype_q,q_quant_elements);
    const u128 k_input_conversion=input_conversion_count(dtype_k,k_quant_elements);
    const u128 v_input_conversion=input_conversion_count(dtype_k,v_quant_elements);
    const u128 qk_input_conversion=add128(q_input_conversion,k_input_conversion,"attention QK input conversions");
    const u128 pv_input_conversion=v_input_conversion;
    const EqWork qk_quant_vec=add128(
        weighted_ops(q_input_conversion,int4path||dtype_q=="int4"?kappa_quant:1,"attention Q conversion"),
        weighted_ops(k_input_conversion,int4path||dtype_k=="int4"?kappa_quant:1,"attention K conversion"),
        "attention QK conversion Vec ops");
    const EqWork pv_quant_vec=weighted_ops(pv_input_conversion,
        int4path||dtype_k=="int4"?kappa_quant:1,"attention V conversion");
    const EqWork qk_acc_vec=int4path?add128(weighted_ops(qk_acc_elements,kappa_acc,"attention QK accumulator Vec ops"),qk_acc_adds,"attention QK accumulator Vec ops"):0;
    const EqWork pv_acc_vec=int4path?add128(weighted_ops(pv_acc_elements,kappa_acc,"attention PV accumulator Vec ops"),pv_acc_adds,"attention PV accumulator Vec ops"):0;
    const u128 conversion_elements=int4path?add128(add128(quant_input_elements,acc_convert_elements,"attention conversion elements"),output_conversion,"attention conversion elements"):add128(add128(qk_input_conversion,pv_input_conversion,"attention conversion elements"),output_conversion,"attention conversion elements");
    m.conversion_elements=conversion_elements;
    const EqWork input_conversion_ops=add128(qk_quant_vec,pv_quant_vec,"attention input conversion Vec ops");
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
    const u128 cast_write=staged_full&&mode=="both"&&cast_bytes?outbytes:0;
    const u128 fused_read=staged_full?0:add128(qbytes,kvbytes,"fused attention reads");
    const u128 fused_write=staged_full?0:outbytes;
    const u128 totalread=staged_full&&mode=="both"?add128(add128(qk_read,soft_read,"full attention reads"),add128(pv_read,cast_read,"full attention reads"),"full attention reads"):
        (mode=="qk"?qk_read:(mode=="av"?pv_read:fused_read));
    const u128 totalwrite=staged_full&&mode=="both"?add128(add128(qk_write,soft_write,"full attention writes"),add128(pv_write,cast_write,"full attention writes"),"full attention writes"):
        (mode=="qk"?qk_write:(mode=="av"?pv_write:fused_write));
    const u128 sb=attention_workspace_from_counts(
        workspace_slots,workspace_queryrows,hd,dtype_o,mode,staged_full);
    m.workspace=sb;m.score_bytes=(mode=="qk"||(staged_full&&mode=="both"))?scoreb:0;m.probability_bytes=probb;
    m.score_slots=slots;m.valid_pairs=valid;m.logical=logical;
    m.hbm_read=0;m.hbm_write=0;
    const u64 rounds=std::max<u64>(1,ceildiv(as_u64(output_tasks,"attention output tasks"),active));
    const u64 waves=as_u64(std::max<u128>(1,staged_full?add128(output_tasks,ceildiv(as_u64(slots,"attention score slots"),std::max<u64>(1,active*256)),"full attention waves"):row_block_visits),"attention waves");
    auto emit_stage=[&](const std::string& name,u128 read,u128 write,u128 c_read,u128 c_write,
                        u128 i4,EqWork vec,u64 use_active,u64 frames,double hbm_eta){
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
    const EqWork qk_vec=int4path?add128(qk_quant_vec,qk_acc_vec,"attention QK Vec ops"):add128(qk_ops,qk_quant_vec,"attention QK Vec ops");
    const EqWork pv_vec=int4path?add128(pv_quant_vec,pv_acc_vec,"attention PV Vec ops"):add128(pv_ops,pv_quant_vec,"attention PV Vec ops");
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
      const EqWork vec=int4path?add128(add128(eq,input_conversion_ops,"fused quant Vec ops"),add128(accumulator_vec_ops,cast_vec_ops,"fused accumulator Vec ops"),"fused Vec ops"):
          add128(add128(eq,vector_matrix_ops,"strict Vec attention ops"),add128(input_conversion_ops,cast_vec_ops,"strict conversions"),"strict Vec attention ops");
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
      fail("mapping_invariant: attention stage HBM bytes do not match its access ledger: mode="+mode+
           " full="+std::to_string(full)+" actual="+std::to_string(as_u64(m.hbm_read,"attention reads"))+"/"+
           std::to_string(as_u64(m.hbm_write,"attention writes"))+" expected="+
           std::to_string(as_u64(totalread,"attention reads"))+"/"+std::to_string(as_u64(totalwrite,"attention writes")));
    m.active_cores=std::max(m.active_cores,active);
    m.score_slots=slots;m.valid_pairs=valid;m.score_bytes=(mode=="qk"||(staged_full&&mode=="both"))?scoreb:0;m.probability_bytes=probb;
    m.logical=logical;
    m.template_id=mode=="both"?(staged_full?"full_score_materialization_v3":"fused_q_head_shard_v2"):
        (mode=="qk"?"attention_qk_materialized_v3":"attention_av_materialized_v2");
    m.assumptions={"engineering_assumption","GQA_head_mapping","absolute_position_causal_mask","not_run_granularity_validation"};
    m.tm=tq; m.tn=tk; m.tk=tk;
    if (static_cast<long double>(as_u64(m.sram_peak,"attention SRAM peak")) > perf(hw,"rho_sram",0.9)*dd(hw,"core","sram_bytes_per_core")) continue;
    finalize_summary(*p,m,hw); p->summary["template_id"]=m.template_id;
    if(mode=="qk"||(mode=="both"&&staged_full))
      p->summary["workspace_model_revision"]="workspace_lifetimes_v2";
    p->summary["dataflow_id"]=mode=="both"
        ?(staged_full?"full_score_materialization_v3":"fused_q_head_shard_v2")
        :(mode=="qk"?"full_score_materialization_v3":"full_score_materialization_v2");
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

std::uint64_t workspace_capacity_upper_bound(
    const Hardware& hardware,const nlohmann::json& task) {
  const std::string op=gets(task,"op_kind");
  if(op=="attention"||op=="attention_region"||op=="qk"||op=="av") {
    const auto spans=spans_of(task);
    if(spans.empty()) fail("task.spans: attention requires at least one span");
    const u64 qh=shape_u(task,"q_heads",shape_u(task,"q_heads_local",1));
    const u64 kvh=shape_u(task,"kv_heads",shape_u(task,"kv_heads_local",1));
    const u64 hd=shape_u(task,"head_dim",1);
    if(!qh||!kvh||qh%kvh)
      fail("unsupported_head_mapping: standard GQA requires q_heads divisible by kv_heads");
    const std::string output_dtype=dtype(task,"output");
    if(dtype(task,"scores","fp32")!="fp32"||
       dtype(task,"accumulator","fp32")!="fp32")
      fail("unsupported_dtype: attention score and accumulator workspaces are fixed FP32");

    const std::string mode=op=="qk"?"qk":(op=="av"?"av":"both");
    if(mode=="av") return 0;
    if((op=="attention"||op=="attention_region")&&
       gets(task,"fusion_policy","auto")=="fused") return 0;
    const auto counts=attention_materialized_counts(spans,qh);
    const u128 peak=attention_workspace_from_counts(
        counts.first,counts.second,hd,output_dtype,mode,true);
    return as_u64(peak,"HBM workspace");
  }
  if(op=="ffn_region") {
    const std::string variant=gets(task.at("tensor_shapes"),"mlp_variant","swiglu");
    const bool swiglu=variant=="swiglu";
    const std::string fusion=gets(task,"fusion_policy","auto");
    if(fusion=="fused"&&!swiglu)
      fail("unsupported_mapping: fused_output_shard_v2 supports SwiGLU only");
    if(fusion=="fused") return 0;
    const u64 M=shape_u(task,"m",1);
    const u64 H=shape_u(task,"hidden",1);
    const u64 F=shape_u(task,"ffn",H);
    const u128 middle=mul128(M,F,"FFN intermediate");
    const u128 peak=ffn_workspace_from_middle_elements(
        middle,dtype(task,"activations"),swiglu);
    return as_u64(peak,"HBM workspace");
  }
  (void)hardware;
  return 0;
}

std::array<Activity, 3> gemm_os_single_core_resource_facts(
    const Hardware& hardware, const gemm_os_access_counter::Input& input) {
  if (du(hardware,"die","core_count")!=1 ||
      du(hardware,"die","core_columns_nx")!=1 ||
      du(hardware,"die","core_rows_ny")!=1)
    fail("unsupported_mapping: GEMM OS resource facts require a single-Core grid");
  const auto ledger=gemm_os_access_counter::count(input);
  const Topology topo=topology(hardware,1,1);
  const double hbm_eta=eta(hardware,"eta_hbm_matrix");
  Flow load, compute, store;
  add_hbm(load,hardware,topo,ledger.hbm_read_bytes,1,false,hbm_eta);
  add_bytes_work(load,SRAM_WRITE,ledger.load_sram_write_bytes,"OS Load SRAM writes");
  add_bytes_work(compute,SRAM_READ,ledger.compute_sram_read_bytes,"OS Compute SRAM reads");
  add_bytes_work(compute,SRAM_WRITE,ledger.compute_sram_write_bytes,"OS Compute SRAM writes");
  add_bytes_work(compute,LOCAL,ledger.local_bytes,"OS Compute LOCAL bytes");
  compute.int4=ledger.matrix_int4_ops;
  compute.vec=ledger.vec_equivalent_ops;
  compute.compute_s=compute_time(hardware,compute.int4,compute.vec,1);
  add_bytes_work(store,SRAM_READ,ledger.store_sram_read_bytes,"OS Store SRAM reads");
  add_hbm(store,hardware,topo,ledger.hbm_write_bytes,1,true,hbm_eta);
  std::array<Activity,3> result;
  for(std::size_t stage=0;stage<result.size();++stage) {
    result[stage].id=stage+1;
    result[stage].role=stage==0?Role::Load:(stage==1?Role::Compute:Role::Store);
  }
  seal_activity(result[0],load,hardware,1);
  seal_activity(result[1],compute,hardware,1);
  seal_activity(result[2],store,hardware,1);
  return result;
}

static std::vector<std::shared_ptr<Plan>> enumerate_mappings_impl(
    const Hardware& hardware, const nlohmann::json& task,
    bool defer_top_level_gemm_fingerprints) {
  if (!task.is_object() || !task.contains("op_kind")) fail("task: normalized op_kind is required");
  const std::string op=gets(task,"op_kind");
  const u64 nx=du(hardware,"die","core_columns_nx"), ny=du(hardware,"die","core_rows_ny");
  const Topology topo=topology(hardware,nx,ny);
  if (op=="gemm") return enumerate_gemm(
      hardware,task,topo,shape_u(task,"m"),shape_u(task,"n"),shape_u(task,"k"),
      !defer_top_level_gemm_fingerprints,defer_top_level_gemm_fingerprints);
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
        const u128 span_slots=mul128(mul128(s.q,lengths,"score rectangles"),heads,"score heads");
        slots=add128(slots,mul128(span_slots,s.batch,"score batch"),"score count");
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
          for(const char* key:{"source_vec_ops","hbm_read_bytes_total","hbm_write_bytes_total","sram_read_bytes","sram_write_bytes","local_bus_bytes","output_round_count","consumption_wave_count","wave_count"})
            q->summary[key]=as_u64(add128(getu(q->summary,key),getu(rope->summary,key),std::string("QKV/RoPE ")+key),std::string("QKV/RoPE ")+key);
          publish_eqops(q->summary,summary_eqops(q->summary)+summary_eqops(rope->summary));
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
    if(op=="norm") {
      add_norm_rows(*p,m,hardware,topo,M,H,as_u64(executions,"Norm executions"),
          ad,dtype(task,"parameters"),od,
          gets(task.at("tensor_shapes"),"norm_type","rmsnorm")=="layernorm");
    } else if(op=="residual"||op=="mlp_activation") {
      add_elementwise(*p,m,hardware,topo,as_u64(count,"elementwise elements per invocation"),
          as_u64(executions,"elementwise executions"),extra ? 2 : 1,
          as_u64(eq/count,"elementwise ops per element"),ad,od,op+"_vec_v3");
    } else {
      const bool score_transform=op=="scale_mask"||op=="softmax";
      const u128 inb=mul128(bytes(count,score_transform?"fp32":ad),executions,"vector input bytes");
      const u128 outb=mul128(bytes(count,score_transform?"fp32":od),executions,"vector output bytes");
      count=mul128(count,executions,"layer vector elements"); eq=mul128(eq,executions,"layer vector operations");
      extra=mul128(extra,executions,"vector extra input bytes");
      add_vector(*p,m,hardware,topo,count,eq,inb,outb,active,op+"_vec_v2",extra);
    }
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
      z.workspace=std::max(z.workspace,
          ffn_workspace_from_middle_elements(middle_one,middle_dtype,swiglu));
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

          EqWork vecops=activation_ops;
          if(i4) {
            vecops=add128(vecops,weighted_ops(quant_conversion_elements,
                         perf(hardware,"kappa_quant_eqops",4),
                         "FFN quantization eqops"),"FFN Vec ops");
            const u128 partial_elements=updates;
            vecops=add128(vecops,weighted_ops(partial_elements,
                         perf(hardware,"kappa_acc_convert_eqops",2),
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
            const u128 weight_fp32_elements_one=add128(upgate_fp32_elements_one,
                down_fp32_elements_one,"FFN FP32 weight conversions");
            const EqWork input_fp32_ops=weighted_ops(
                mul128(input_fp32_elements_one,executions,"FFN FP32 input conversions"),
                ad=="int4"?perf(hardware,"kappa_quant_eqops",4):1,
                "FFN FP32 input decode ops");
            const EqWork weight_fp32_ops=weighted_ops(
                mul128(weight_fp32_elements_one,executions,"FFN FP32 weight conversions"),
                wd=="int4"?perf(hardware,"kappa_quant_eqops",4):1,
                "FFN FP32 weight decode ops");
            vecops=add128(vecops,add128(input_fp32_ops,weight_fp32_ops,
                "FFN FP32 conversions"),"FFN FP32 conversion Vec ops");
            if(od=="int4")
              vecops=add128(vecops,weighted_ops(quant_conversion_elements,
                           perf(hardware,"kappa_quant_eqops",4),
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

namespace {

struct RegionReadyOverheads {
  double launch=0,setup=0;
  u64 instances=0,required=0,applied=0;
  std::size_t first_hbm=0;
  bool has_hbm=false,complete=false;
  region_delay::Result per_instance,total;
};

std::optional<RegionReadyOverheads> region_ready_overheads(
    const Hardware& hw,const Plan& p) {
  RegionReadyOverheads f;
  f.launch=perf(hw,"launch_ns_per_macro",0);
  f.setup=perf(hw,"memory_setup_ns_per_macro",0);
  if(f.launch==0 && f.setup==0)return std::nullopt;
  f.instances=as_u64(mul128(getu(p.task,"layer_instances",1),
      getu(p.task,"weight_passes",1),"region overhead instances"),"region overhead instances");
  f.first_hbm=p.epochs.size();
  for(std::size_t e=0;e<p.epochs.size()&&f.first_hbm==p.epochs.size();++e)
    for(const auto& a:p.epochs[e].activities)
      for(const auto& row:a.hbm_bytes)
        if(row[0]||row[1])f.first_hbm=e;
  f.has_hbm=f.first_hbm!=p.epochs.size();
  const bool independent_first_load=f.has_hbm && f.first_hbm==0 && std::all_of(
      p.epochs.front().activities.begin(),p.epochs.front().activities.end(),
      [](const Activity& a){return a.role==Role::Load;});
  const double applicable_setup=f.has_hbm ? f.setup : 0;
  if(f.launch==0 && applicable_setup==0)return std::nullopt;
  f.per_instance=region_delay::sum(f.launch,applicable_setup,1);
  f.total=region_delay::sum(f.launch,applicable_setup,f.instances);
  f.required=f.total.ceil;
  // Aggregated repeats do not retain each invocation's idle windows. Keep
  // their previous body timing, but explicitly withhold a complete estimate.
  // Charging R*delay at the front would alter real competition/readiness.
  f.complete=f.instances==1 && (!f.has_hbm||f.setup==0||independent_first_load);
  f.applied=f.complete ? f.per_instance.ceil : 0;
  return f;
}

nlohmann::json region_ready_overhead_audit(const RegionReadyOverheads& f) {
  nlohmann::json audit={{"model","region_ready_overheads_v1"},{"instances",f.instances},
      {"launch_ns_per_instance",f.launch},{"memory_setup_ns_per_hbm_instance",f.setup},
      {"hbm_instances",f.has_hbm ? f.instances : 0},
      {"required_total_delay_ns_whole",f.total.whole},
      {"required_total_delay_ns_ceil",f.required},
      {"modeled_instances",f.complete ? f.instances : 0},
      {"applied_ready_delay_ns",f.applied},
      {"timing_status",f.complete ? "modeled_integer_ns_ready_gate" : "not_available"},
      {"scope","one_complete_source_region_not_internal_kernel_or_tile"},
      {"quantization","ceil_combined_region_delay_once_to_public_integer_ns"}};
  // Timing needs only exact integer floor/ceil. Optional scalar audit
  // fractions must not reject legal binary64 inputs spanning many exponents.
  try {
    const auto per_instance=EqWork::weighted(1,f.launch)+
        EqWork::weighted(1,f.has_hbm ? f.setup : 0);
    const auto total=per_instance.multiply(f.instances);
    if(!total.integral())audit["required_total_delay_ns_fraction"]=equivalent_fraction_to_json(total.fraction());
    if(f.complete) {
      const double error=per_instance.integral() ? 0.0 :
          as_double(1-per_instance.fraction(),"region delay quantization");
      if(error<1)audit["quantization_error_ns"]=error;
      else audit["quantization_error_ns_interval"]=
          {{"lower_inclusive",0},{"upper_exclusive",1}};
    }
  } catch(const std::overflow_error&) {
    // The exact expression is retained by the two input coefficients,
    // HBM-instance count and multiplier. No arbitrary precision state.
    audit["fraction_representation"]="exact_binary64_input_terms_times_instances";
    if(f.complete)audit["quantization_error_ns_interval"]=
        {{"lower_inclusive",0},{"upper_exclusive",1}};
  }
  if(f.has_hbm)audit["first_hbm_epoch"]=f.first_hbm;
  if(!f.complete)audit["limitation"]=f.instances>1 ? "aggregate_bulk_does_not_retain_per_instance_ready_windows" :
      "first_HBM_is_not_an_independent_initial_Load_epoch";
  audit["ready_gate_description_count"]=f.applied ? 1 : 0;
  return audit;
}

// Apply region-level nonservice time only after composing internal kernels.
// The supported first HBM visit is an independent Load-first epoch; the
// existing Plan ready gate needs no new activity, resource, or state.
void apply_region_ready_overheads(const Hardware& hw,
    std::vector<std::shared_ptr<Plan>>& plans,bool defer_gemm_diagnostics) {
  for(auto& p:plans) {
    const auto facts=region_ready_overheads(hw,*p);
    if(!facts)continue;  // Preserve zero-overhead wire/hash exactly.
    const auto& f=*facts;
    if(p->launch_ns!=0)fail("mapping_invariant: region ready delay already applied");
    p->launch_ns=f.applied;
    if(!f.complete) {
      p->summary["prediction_status"]="not_available";
      p->summary["prediction_scope"]="body_timing_only_fixed_region_overheads_unmodeled";
    }
    p->summary["service_description_count"]=service_description_count(*p);
    if(f.applied && p->summary.contains("service_curves")) {
      for(auto& curve:p->summary["service_curves"]) {
        if(curve.at("epoch_begin")!=0)continue;
        const u64 descriptions=getu(curve,"description_count")+1;
        if(descriptions>16)fail("mapping_invariant: region ready gate exceeds curve budget");
        curve["description_count"]=descriptions;
        curve["description_budget"]=descriptions<=5 ? 5 : descriptions<=8 ? 8 : 16;
        curve["region_ready_gate_descriptions"]=1;
        break;
      }
    }
    // Internal GEMM trials keep all timing, availability, and budget checks.
    // Only the winner's diagnostic JSON and fingerprint need materializing.
    const bool defer=defer_gemm_diagnostics && gets(p->task,"op_kind")=="gemm";
    if(!defer)p->summary["fixed_region_overheads"]=region_ready_overhead_audit(f);
    p->fingerprint.clear();
    if(!defer)p->fingerprint=plan_to_json(*p).at("fingerprint").get<std::string>();
  }
}

}  // namespace

void finalize_region_overhead_audit(const Hardware& hw,Plan& plan) {
  const auto facts=region_ready_overheads(hw,plan);
  if(!facts)return;
  if(plan.launch_ns!=facts->applied)
    fail("mapping_invariant: final region delay differs from candidate timing");
  plan.summary["fixed_region_overheads"]=region_ready_overhead_audit(*facts);
}

std::vector<std::shared_ptr<Plan>> enumerate_mappings(
    const Hardware& hardware, const nlohmann::json& task) {
  auto result=enumerate_mappings_impl(hardware,task,false);
  apply_region_ready_overheads(hardware,result,false);
  return result;
}

std::vector<std::shared_ptr<Plan>> enumerate_mappings_for_compile(
    const Hardware& hardware, const nlohmann::json& task) {
  auto result=enumerate_mappings_impl(hardware,task,true);
  apply_region_ready_overheads(hardware,result,true);
  return result;
}

std::shared_ptr<Plan> materialize_gemm_compile_candidate(const Hardware& hardware,
                                                       const Plan& candidate) {
  const auto& task=candidate.task;
  if(gets(task,"op_kind")!="gemm")fail("mapping_invariant: compact materialization requires GEMM");
  const auto dataflow=gets(candidate.summary,"dataflow_id");
  if(dataflow!="weight_resident_v2" && dataflow!="output_stationary_v2")
    fail("mapping_invariant: unknown compact GEMM dataflow");
  const GemmChoice choice{getu(candidate.summary,"tm"),getu(candidate.summary,"tn"),
      getu(candidate.summary,"tk"),dataflow=="weight_resident_v2"};
  const Topology topo=topology(hardware,du(hardware,"die","core_columns_nx"),
      du(hardware,"die","core_rows_ny"));
  auto full=make_gemm_mapping(hardware,task,topo,shape_u(task,"m"),shape_u(task,"n"),shape_u(task,"k"),
      dtype(task,"activations"),dtype(task,"weights"),dtype(task,"output"),
      getu(task,"layer_instances",1),getu(task,"weight_passes",1),choice,false,false);
  std::vector<std::shared_ptr<Plan>> one{full};
  apply_region_ready_overheads(hardware,one,true);
  if(full->task!=candidate.task || full->hardware_fingerprint!=candidate.hardware_fingerprint ||
      full->launch_ns!=candidate.launch_ns ||
      full->summary.value("prediction_status",std::string("ok_assumed"))!=
          candidate.summary.value("prediction_status",std::string("ok_assumed")))
    fail("mapping_invariant: materialized GEMM candidate identity changed");
  for(const char* key:{"hbm_read_bytes_total","hbm_write_bytes_total","hbm_workspace_peak_bytes",
      "sram_peak_bytes_per_core","template_id","tm","tn","tk","th"})
    if(full->summary.at(key)!=candidate.summary.at(key))
      fail(std::string("mapping_invariant: materialized GEMM rank key changed: ")+key);
  return full;
}

}  // namespace coarse_die::detail
