#include "coarse_die/Plan.hh"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <limits>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>

namespace coarse_die {
namespace {

constexpr char kHex[] = "0123456789abcdef";
constexpr std::array<std::uint32_t, 64> kShaK = {
    0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u,
    0x3956c25bu, 0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u,
    0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u,
    0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u, 0xc19bf174u,
    0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu,
    0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau,
    0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u,
    0xc6e00bf3u, 0xd5a79147u, 0x06ca6351u, 0x14292967u,
    0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu, 0x53380d13u,
    0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u,
    0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u,
    0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u,
    0x19a4c116u, 0x1e376c08u, 0x2748774cu, 0x34b0bcb5u,
    0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu, 0x682e6ff3u,
    0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u,
    0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u};

std::uint32_t rotr(std::uint32_t x, unsigned n) {
  return (x >> n) | (x << (32 - n));
}

std::string sha256(const std::string& input) {
  std::vector<std::uint8_t> bytes(input.begin(), input.end());
  const std::uint64_t bit_length =
      static_cast<std::uint64_t>(bytes.size()) * std::uint64_t{8};
  bytes.push_back(0x80);
  while ((bytes.size() % 64) != 56) bytes.push_back(0);
  for (int shift = 56; shift >= 0; shift -= 8) {
    bytes.push_back(static_cast<std::uint8_t>(bit_length >> shift));
  }

  std::array<std::uint32_t, 8> h = {
      0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
      0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u};
  for (std::size_t offset = 0; offset < bytes.size(); offset += 64) {
    std::array<std::uint32_t, 64> w{};
    for (std::size_t i = 0; i < 16; ++i) {
      const std::size_t p = offset + i * 4;
      w[i] = (static_cast<std::uint32_t>(bytes[p]) << 24) |
             (static_cast<std::uint32_t>(bytes[p + 1]) << 16) |
             (static_cast<std::uint32_t>(bytes[p + 2]) << 8) |
             static_cast<std::uint32_t>(bytes[p + 3]);
    }
    for (std::size_t i = 16; i < 64; ++i) {
      const std::uint32_t s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^
                                (w[i - 15] >> 3);
      const std::uint32_t s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^
                                (w[i - 2] >> 10);
      w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }
    std::uint32_t a = h[0], b = h[1], c = h[2], d = h[3];
    std::uint32_t e = h[4], f = h[5], g = h[6], hh = h[7];
    for (std::size_t i = 0; i < 64; ++i) {
      const std::uint32_t s1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
      const std::uint32_t ch = (e & f) ^ ((~e) & g);
      const std::uint32_t temp1 = hh + s1 + ch + kShaK[i] + w[i];
      const std::uint32_t s0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
      const std::uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
      const std::uint32_t temp2 = s0 + maj;
      hh = g;
      g = f;
      f = e;
      e = d + temp1;
      d = c;
      c = b;
      b = a;
      a = temp1 + temp2;
    }
    h[0] += a;
    h[1] += b;
    h[2] += c;
    h[3] += d;
    h[4] += e;
    h[5] += f;
    h[6] += g;
    h[7] += hh;
  }
  std::ostringstream out;
  out << std::hex << std::setfill('0');
  for (std::uint32_t word : h) out << std::setw(8) << word;
  return out.str();
}

const char* role_name(Role role) {
  switch (role) {
    case Role::Load: return "load";
    case Role::Compute: return "compute";
    case Role::Store: return "store";
    case Role::Endpoint: return "endpoint";
  }
  throw std::invalid_argument("plan: invalid activity role");
}

Role parse_role(const nlohmann::json& j, const std::string& path) {
  if (!j.is_string()) throw std::invalid_argument(path + ": expected string");
  const std::string value = j.get<std::string>();
  if (value == "load") return Role::Load;
  if (value == "compute") return Role::Compute;
  if (value == "store") return Role::Store;
  if (value == "endpoint") return Role::Endpoint;
  throw std::invalid_argument(path + ": unsupported role " + value);
}

void reject_unknown(const nlohmann::json& j,
                    std::initializer_list<const char*> allowed,
                    const std::string& path) {
  if (!j.is_object()) throw std::invalid_argument(path + ": expected object");
  for (auto it = j.begin(); it != j.end(); ++it) {
    bool found = false;
    for (const char* key : allowed) {
      if (it.key() == key) {
        found = true;
        break;
      }
    }
    if (!found) throw std::invalid_argument(path + ": unknown key " + it.key());
  }
}

const nlohmann::json& require_member(const nlohmann::json& j,
                                    const char* key,
                                    const std::string& path) {
  auto it = j.find(key);
  if (it == j.end()) throw std::invalid_argument(path + ": missing " + key);
  return *it;
}

std::uint64_t require_u64(const nlohmann::json& j,
                          const std::string& path) {
  if (j.is_number_unsigned()) return j.get<std::uint64_t>();
  if (j.is_number_integer()) {
    const auto value = j.get<std::int64_t>();
    if (value >= 0) return static_cast<std::uint64_t>(value);
  }
  throw std::invalid_argument(path + ": expected a non-negative integer");
}

double require_finite_nonnegative(const nlohmann::json& j,
                                 const std::string& path,
                                 bool allow_negative = false) {
  if (!j.is_number()) throw std::invalid_argument(path + ": expected number");
  const double value = j.get<double>();
  if (!std::isfinite(value) || (!allow_negative && value < 0.0)) {
    throw std::invalid_argument(path + ": expected finite non-negative number");
  }
  return value;
}

double require_finite_number(const nlohmann::json& j,
                             const std::string& path) {
  if (!j.is_number()) throw std::invalid_argument(path + ": expected number");
  const double value = j.get<double>();
  if (!std::isfinite(value))
    throw std::invalid_argument(path + ": expected finite number");
  return value;
}

void require_progress(double progress) {
  if (!std::isfinite(progress) || progress < 0.0 || progress > 1.0)
    throw std::invalid_argument("credit progress: expected a finite value in [0,1]");
}

nlohmann::json make_payload(const Plan& plan) {
  bool fractional = false;
  bool curved = false;
  nlohmann::json epochs = nlohmann::json::array();
  for (const Epoch& epoch : plan.epochs) {
    nlohmann::json activities = nlohmann::json::array();
    for (const Activity& activity : epoch.activities) {
      nlohmann::json work = nlohmann::json::array();
      for (double value : activity.work) work.push_back(value);
      nlohmann::json bytes = nlohmann::json::array();
      for (const auto& row : activity.hbm_bytes) {
        bytes.push_back(nlohmann::json::array({row[0], row[1]}));
      }
      activities.push_back({
          {"id", activity.id},
          {"role", role_name(activity.role)},
          {"work", std::move(work)},
          {"hbm_bytes", std::move(bytes)},
          {"int4_ops", activity.int4_ops},
          {"vec_eqops", activity.vec_eqops},
          {"canonical_q", activity.canonical_q},
          {"private_time_s", activity.private_time_s},
          {"credit_delta", {activity.credit_delta[0], activity.credit_delta[1]}}});
      if (activity.vec_eqops_fraction != 0.0) {
        activities.back()["vec_eqops_fraction"] = equivalent_fraction_to_json(activity.vec_eqops_fraction);
        fractional = true;
      }
      if (!activity.credit_curve.empty()) {
        nlohmann::json curve = nlohmann::json::array();
        for (const CreditPoint& point : activity.credit_curve) {
          curve.push_back({{"progress", point.progress},
                           {"cumulative", {point.cumulative[0],
                                            point.cumulative[1]}}});
        }
        activities.back()["credit_curve"] = std::move(curve);
        curved = true;
      }
    }
    epochs.push_back({
        {"name", epoch.name},
        {"activities", std::move(activities)},
        {"credit_initial", {epoch.credit_initial[0], epoch.credit_initial[1]}},
        {"credit_capacity", {epoch.credit_capacity[0], epoch.credit_capacity[1]}},
        {"repeat_count", epoch.repeat_count}});
  }
  const char* schema = curved ? "coarse_die_plan_v4"
                              : (fractional ? "coarse_die_plan_v3"
                                            : "coarse_die_plan_v2");
  return {{"schema", schema},
          {"task", plan.task},
          {"summary", plan.summary},
          {"hardware_fingerprint", plan.hardware_fingerprint},
          {"launch_ns", plan.launch_ns},
          {"epochs", std::move(epochs)}};
}

void validate_plan(const Plan& plan) {
  if (!plan.task.is_object()) throw std::invalid_argument("plan.task: expected object");
  if (!plan.summary.is_object()) throw std::invalid_argument("plan.summary: expected object");
  if (plan.hardware_fingerprint.empty()) {
    throw std::invalid_argument("plan.hardware_fingerprint: empty");
  }
  if (plan.epochs.size() > 64) {
    throw std::invalid_argument("plan.epochs: exceeds 64 epoch limit");
  }
  std::set<std::uint64_t> ids;
  for (std::size_t ei = 0; ei < plan.epochs.size(); ++ei) {
    const Epoch& epoch = plan.epochs[ei];
    const std::string path = "plan.epochs[" + std::to_string(ei) + "]";
    if (epoch.name.empty()) throw std::invalid_argument(path + ".name: empty");
    if (epoch.activities.size() > 3) {
      throw std::invalid_argument(path + ".activities: exceeds three activities");
    }
    if (epoch.repeat_count == 0) {
      throw std::invalid_argument(path + ".repeat_count: must be >= 1");
    }
    for (std::size_t c = 0; c < 2; ++c) {
      if (!std::isfinite(epoch.credit_initial[c]) ||
          !std::isfinite(epoch.credit_capacity[c]) ||
          epoch.credit_initial[c] < 0.0 ||
          epoch.credit_capacity[c] < epoch.credit_initial[c]) {
        throw std::invalid_argument(path + ": invalid finite-buffer credit");
      }
    }
    for (std::size_t ai = 0; ai < epoch.activities.size(); ++ai) {
      const Activity& activity = epoch.activities[ai];
      const std::string ap = path + ".activities[" + std::to_string(ai) + "]";
      if (!ids.insert(activity.id).second) {
        throw std::invalid_argument(ap + ".id: duplicate activity id");
      }
      if (!std::isfinite(activity.vec_eqops_fraction) ||
          activity.vec_eqops_fraction < 0.0 || activity.vec_eqops_fraction >= 1.0 ||
          (activity.vec_eqops == std::numeric_limits<std::uint64_t>::max() &&
           activity.vec_eqops_fraction != 0.0)) {
        throw std::invalid_argument(ap + ".vec_eqops_fraction: expected a remainder in [0,1) within uint64 work range");
      }
      (void)planned_vec_eqops(activity);
      for (std::size_t r = 0; r < resource_count; ++r) {
        if (!std::isfinite(activity.work[r]) || activity.work[r] < 0.0) {
          throw std::invalid_argument(ap + ".work: expected finite non-negative values");
        }
      }
      if (!std::isfinite(activity.canonical_q) || activity.canonical_q <= 0.0) {
        throw std::invalid_argument(ap + ".canonical_q: expected finite positive value");
      }
      if (!std::isfinite(activity.private_time_s) || activity.private_time_s < 0.0) {
        throw std::invalid_argument(ap + ".private_time_s: expected finite non-negative value");
      }
      for (double delta : activity.credit_delta) {
        if (!std::isfinite(delta)) {
          throw std::invalid_argument(ap + ".credit_delta: expected finite values");
        }
      }
      detail::validate_credit_curve(activity);
    }
  }
  if (has_credit_curves(plan) && service_description_count(plan) > 64) {
    throw std::invalid_argument(
        "plan.service_description_count: exceeds 64 description limit");
  }
}

}  // namespace

namespace detail {

void validate_credit_curve(const Activity& activity) {
  const auto& curve = activity.credit_curve;
  if (curve.empty()) return;
  if (curve.size() < 2 || curve.size() > 17) {
    throw std::invalid_argument(
        "activity.credit_curve: expected 2..17 points");
  }
  for (double delta : activity.credit_delta) {
    if (!std::isfinite(delta)) {
      throw std::invalid_argument(
          "activity.credit_curve: credit_delta must be finite");
    }
  }
  if (curve.front().progress != 0.0 ||
      curve.front().cumulative != std::array<double, 2>{0.0, 0.0}) {
    throw std::invalid_argument(
        "activity.credit_curve: first point must be (0,[0,0])");
  }
  if (curve.back().progress != 1.0 ||
      curve.back().cumulative != activity.credit_delta) {
    throw std::invalid_argument(
        "activity.credit_curve: last point must be (1,credit_delta)");
  }

  for (std::size_t i = 0; i < curve.size(); ++i) {
    const CreditPoint& point = curve[i];
    if (!std::isfinite(point.progress) || point.progress < 0.0 ||
        point.progress > 1.0) {
      throw std::invalid_argument(
          "activity.credit_curve.progress: expected a finite value in [0,1]");
    }
    for (double value : point.cumulative) {
      if (!std::isfinite(value)) {
        throw std::invalid_argument(
            "activity.credit_curve.cumulative: expected finite values");
      }
    }
    if (i == 0) continue;

    const CreditPoint& previous = curve[i - 1];
    const double dp = point.progress - previous.progress;
    if (!(dp > 0.0) || !std::isfinite(dp)) {
      throw std::invalid_argument(
          "activity.credit_curve.progress: points must be strictly increasing");
    }
    for (std::size_t channel = 0; channel < 2; ++channel) {
      const double delta = activity.credit_delta[channel];
      const double before = previous.cumulative[channel];
      const double after = point.cumulative[channel];
      if ((delta > 0.0 && after < before) ||
          (delta < 0.0 && after > before) ||
          (delta == 0.0 && (before != 0.0 || after != 0.0))) {
        throw std::invalid_argument(
            "activity.credit_curve.cumulative: not monotone with credit_delta");
      }
      // The public slope API returns double. Reject a valid finite endpoint
      // pair whose segment derivative cannot be represented by that API.
      if (!std::isfinite((after - before) / dp)) {
        throw std::invalid_argument(
            "activity.credit_curve: segment slope is not finite");
      }
    }
  }
}

}  // namespace detail

namespace detail {

void validate_plan_structure(const Plan& plan) {
  validate_plan(plan);
}

}  // namespace detail

std::string stable_hash(const std::string& text) { return sha256(text); }

std::array<double, 2> credit_at(const Activity& activity, double progress) {
  require_progress(progress);
  detail::validate_credit_curve(activity);
  if (activity.credit_curve.empty()) {
    std::array<double, 2> value{};
    for (std::size_t channel = 0; channel < value.size(); ++channel) {
      if (!std::isfinite(activity.credit_delta[channel])) {
        throw std::invalid_argument("activity.credit_delta: expected finite values");
      }
      value[channel] = activity.credit_delta[channel] * progress;
    }
    return value;
  }

  const auto& curve = activity.credit_curve;
  const auto upper = std::upper_bound(
      curve.begin(), curve.end(), progress,
      [](double value, const CreditPoint& point) {
        return value < point.progress;
      });
  if (upper == curve.begin()) return curve.front().cumulative;
  if (upper == curve.end()) return curve.back().cumulative;
  const CreditPoint& lower_point = *(upper - 1);
  if (lower_point.progress == progress) return lower_point.cumulative;
  const CreditPoint& upper_point = *upper;
  const double fraction = (progress - lower_point.progress) /
                         (upper_point.progress - lower_point.progress);
  std::array<double, 2> value{};
  for (std::size_t channel = 0; channel < value.size(); ++channel) {
    const double slope_delta = upper_point.cumulative[channel] -
                               lower_point.cumulative[channel];
    value[channel] = lower_point.cumulative[channel] + fraction * slope_delta;
    if (!std::isfinite(value[channel])) {
      throw std::overflow_error("activity.credit_curve: interpolation overflow");
    }
  }
  return value;
}

std::array<double, 2> credit_slope(const Activity& activity, double progress) {
  require_progress(progress);
  detail::validate_credit_curve(activity);
  if (activity.credit_curve.empty()) {
    for (double delta : activity.credit_delta) {
      if (!std::isfinite(delta)) {
        throw std::invalid_argument("activity.credit_delta: expected finite values");
      }
    }
    return activity.credit_delta;
  }

  const auto& curve = activity.credit_curve;
  const auto upper = std::upper_bound(
      curve.begin(), curve.end(), progress,
      [](double value, const CreditPoint& point) {
        return value < point.progress;
      });
  const std::size_t lower_index = upper == curve.end()
      ? curve.size() - 2
      : static_cast<std::size_t>(upper - curve.begin() - 1);
  const CreditPoint& lower_point = curve[lower_index];
  const CreditPoint& upper_point = curve[lower_index + 1];
  const double dp = upper_point.progress - lower_point.progress;
  std::array<double, 2> slope{};
  for (std::size_t channel = 0; channel < slope.size(); ++channel) {
    slope[channel] = (upper_point.cumulative[channel] -
                      lower_point.cumulative[channel]) / dp;
  }
  return slope;
}

double next_credit_breakpoint(const Activity& activity, double progress) {
  require_progress(progress);
  detail::validate_credit_curve(activity);
  if (activity.credit_curve.empty()) return 1.0;
  const auto next = std::upper_bound(
      activity.credit_curve.begin(), activity.credit_curve.end(), progress,
      [](double value, const CreditPoint& point) {
        return value < point.progress;
      });
  return next == activity.credit_curve.end() ? 1.0 : next->progress;
}

std::size_t service_description_count(const Plan& plan) {
  std::size_t count=0;
  for(const auto& epoch:plan.epochs) {
    std::size_t pieces=0;
    for(const auto& activity:epoch.activities) {
      const auto n=activity.credit_curve.empty() ? 1 : activity.credit_curve.size()-1;
      if(n>std::numeric_limits<std::size_t>::max()-pieces)
        throw std::overflow_error("plan.service_description_count: size overflow");
      pieces+=n;
    }
    pieces=std::max<std::size_t>(1,pieces);
    if(pieces>std::numeric_limits<std::size_t>::max()-count)
      throw std::overflow_error("plan.service_description_count: size overflow");
    count+=pieces;
  }
  return count;
}

nlohmann::json equivalent_fraction_to_json(long double fraction) {
  static_assert(std::numeric_limits<long double>::radix == 2 &&
                std::numeric_limits<long double>::digits <= 64,
                "fractional work requires a binary scalar with at most 64 significand bits");
  if (!std::isfinite(fraction) || fraction <= 0 || fraction >= 1)
    throw std::invalid_argument("plan.vec_eqops_fraction: expected a fraction in (0,1)");
  int exponent = 0;
  const long double mantissa = std::frexp(fraction, &exponent);
  constexpr int digits = std::numeric_limits<long double>::digits;
  std::uint64_t significand = static_cast<std::uint64_t>(std::ldexp(mantissa,digits));
  exponent -= digits;
  while ((significand & 1) == 0) { significand >>= 1; ++exponent; }
  return {{"significand",significand},{"exponent",exponent}};
}

long double equivalent_fraction_from_json(const nlohmann::json& value) {
  reject_unknown(value,{"significand","exponent"},"plan.vec_eqops_fraction");
  const auto significand=require_u64(require_member(value,"significand","fraction"),"fraction.significand");
  const auto& raw_exponent=require_member(value,"exponent","fraction");
  if (!raw_exponent.is_number_integer())
    throw std::invalid_argument("plan.vec_eqops_fraction.exponent: expected integer");
  // Parse before narrowing so malicious large exponents never wrap.
  if (raw_exponent.is_number_unsigned() && raw_exponent.get<std::uint64_t>() > 0)
    throw std::invalid_argument("plan.vec_eqops_fraction.exponent: must be negative");
  const auto exponent=raw_exponent.get<std::int64_t>();
  constexpr int smallest=std::numeric_limits<long double>::min_exponent-
                         std::numeric_limits<long double>::digits;
  if (!significand || (significand & 1)==0 || exponent < smallest || exponent >= 0)
    throw std::invalid_argument("plan.vec_eqops_fraction: invalid canonical dyadic value");
  const long double fraction=std::ldexp(static_cast<long double>(significand),static_cast<int>(exponent));
  if (!std::isfinite(fraction) || fraction <= 0 || fraction >= 1 ||
      equivalent_fraction_to_json(fraction) != value)
    throw std::invalid_argument("plan.vec_eqops_fraction: unrepresentable or noncanonical fraction");
  return fraction;
}

nlohmann::json plan_to_json(const Plan& plan) {
  validate_plan(plan);
  nlohmann::json payload = make_payload(plan);
  const std::string computed = stable_hash(payload.dump());
  if (!plan.fingerprint.empty() && plan.fingerprint != computed) {
    throw std::invalid_argument("plan.fingerprint: does not match canonical payload");
  }
  payload["fingerprint"] = computed;
  return payload;
}

std::shared_ptr<const Plan> plan_from_json(const nlohmann::json& j) {
  reject_unknown(j,
                {"schema", "task", "summary", "hardware_fingerprint",
                 "launch_ns", "epochs", "fingerprint"},
                "plan");
  const auto schema = require_member(j, "schema", "plan");
  if (!schema.is_string() || (schema.get<std::string>() != "coarse_die_plan_v2" &&
                             schema.get<std::string>() != "coarse_die_plan_v3" &&
                             schema.get<std::string>() != "coarse_die_plan_v4")) {
    throw std::invalid_argument(
        "plan.schema: expected coarse_die_plan_v2, coarse_die_plan_v3, or coarse_die_plan_v4");
  }
  const std::string schema_name = schema.get<std::string>();
  const bool fractional_schema = schema_name == "coarse_die_plan_v3" ||
                                 schema_name == "coarse_die_plan_v4";
  const bool curve_schema = schema_name == "coarse_die_plan_v4";
  bool parsed_curve = false;
  auto plan = std::make_shared<Plan>();
  plan->task = require_member(j, "task", "plan");
  plan->summary = require_member(j, "summary", "plan");
  const auto& hfp = require_member(j, "hardware_fingerprint", "plan");
  if (!hfp.is_string()) throw std::invalid_argument("plan.hardware_fingerprint: expected string");
  plan->hardware_fingerprint = hfp.get<std::string>();
  plan->launch_ns = require_u64(require_member(j, "launch_ns", "plan"), "plan.launch_ns");
  const auto& epoch_array = require_member(j, "epochs", "plan");
  if (!epoch_array.is_array()) throw std::invalid_argument("plan.epochs: expected array");
  if (epoch_array.size() > 64) throw std::invalid_argument("plan.epochs: exceeds 64 epoch limit");
  for (std::size_t ei = 0; ei < epoch_array.size(); ++ei) {
    const std::string ep = "plan.epochs[" + std::to_string(ei) + "]";
    const auto& je = epoch_array[ei];
    reject_unknown(je, {"name", "activities", "credit_initial", "credit_capacity", "repeat_count"}, ep);
    Epoch epoch;
    const auto& name = require_member(je, "name", ep);
    if (!name.is_string()) throw std::invalid_argument(ep + ".name: expected string");
    epoch.name = name.get<std::string>();
    const auto& activities = require_member(je, "activities", ep);
    if (!activities.is_array() || activities.size() > 3) {
      throw std::invalid_argument(ep + ".activities: expected array of at most three activities");
    }
    const auto& initial = require_member(je, "credit_initial", ep);
    const auto& capacity = require_member(je, "credit_capacity", ep);
    if (!initial.is_array() || initial.size() != 2 || !capacity.is_array() || capacity.size() != 2) {
      throw std::invalid_argument(ep + ": credits must each have exactly two entries");
    }
    for (std::size_t c = 0; c < 2; ++c) {
      epoch.credit_initial[c] = require_finite_nonnegative(initial[c], ep + ".credit_initial");
      epoch.credit_capacity[c] = require_finite_nonnegative(capacity[c], ep + ".credit_capacity");
    }
    epoch.repeat_count = require_u64(require_member(je, "repeat_count", ep), ep + ".repeat_count");
    for (std::size_t ai = 0; ai < activities.size(); ++ai) {
      const std::string ap = ep + ".activities[" + std::to_string(ai) + "]";
      const auto& ja = activities[ai];
      reject_unknown(ja, {"id", "role", "work", "hbm_bytes", "int4_ops", "vec_eqops", "vec_eqops_fraction", "canonical_q", "private_time_s", "credit_delta", "credit_curve"}, ap);
      Activity activity;
      activity.id = require_u64(require_member(ja, "id", ap), ap + ".id");
      activity.role = parse_role(require_member(ja, "role", ap), ap + ".role");
      const auto& work = require_member(ja, "work", ap);
      if (!work.is_array() || work.size() != resource_count) {
        throw std::invalid_argument(ap + ".work: expected 13 resource entries");
      }
      for (std::size_t r = 0; r < resource_count; ++r) {
        activity.work[r] = require_finite_nonnegative(work[r], ap + ".work");
      }
      const auto& bytes = require_member(ja, "hbm_bytes", ap);
      if (!bytes.is_array() || bytes.size() != 2) {
        throw std::invalid_argument(ap + ".hbm_bytes: expected 2 rows");
      }
      for (std::size_t row = 0; row < 2; ++row) {
        if (!bytes[row].is_array() || bytes[row].size() != 2) {
          throw std::invalid_argument(ap + ".hbm_bytes: expected row/read-write pairs");
        }
        activity.hbm_bytes[row][0] = require_u64(bytes[row][0], ap + ".hbm_bytes");
        activity.hbm_bytes[row][1] = require_u64(bytes[row][1], ap + ".hbm_bytes");
      }
      activity.int4_ops = require_u64(require_member(ja, "int4_ops", ap), ap + ".int4_ops");
      activity.vec_eqops = require_u64(require_member(ja, "vec_eqops", ap), ap + ".vec_eqops");
      if (ja.contains("vec_eqops_fraction")) {
        if (!fractional_schema) throw std::invalid_argument(ap + ": fractional work requires Plan schema v3");
        activity.vec_eqops_fraction = equivalent_fraction_from_json(ja.at("vec_eqops_fraction"));
        if (activity.vec_eqops_fraction == 0.0)
          throw std::invalid_argument(ap + ": zero fractional remainder must be omitted");
      }
      activity.canonical_q = require_finite_nonnegative(require_member(ja, "canonical_q", ap), ap + ".canonical_q");
      activity.private_time_s = require_finite_nonnegative(require_member(ja, "private_time_s", ap), ap + ".private_time_s");
      const auto& delta = require_member(ja, "credit_delta", ap);
      if (!delta.is_array() || delta.size() != 2) {
        throw std::invalid_argument(ap + ".credit_delta: expected two entries");
      }
      for (std::size_t c = 0; c < 2; ++c) {
        activity.credit_delta[c] = require_finite_nonnegative(delta[c], ap + ".credit_delta", true);
      }
      if (ja.contains("credit_curve")) {
        if (!curve_schema) {
          throw std::invalid_argument(
              ap + ".credit_curve: requires Plan schema v4");
        }
        const auto& curve = ja.at("credit_curve");
        if (!curve.is_array() || curve.size() < 2 || curve.size() > 17) {
          throw std::invalid_argument(
              ap + ".credit_curve: expected array of 2..17 points");
        }
        activity.credit_curve.reserve(curve.size());
        for (std::size_t pi = 0; pi < curve.size(); ++pi) {
          const std::string pp = ap + ".credit_curve[" +
                                 std::to_string(pi) + "]";
          const auto& point_json = curve[pi];
          reject_unknown(point_json, {"progress", "cumulative"}, pp);
          CreditPoint point;
          point.progress = require_finite_number(
              require_member(point_json, "progress", pp), pp + ".progress");
          const auto& cumulative =
              require_member(point_json, "cumulative", pp);
          if (!cumulative.is_array() || cumulative.size() != 2) {
            throw std::invalid_argument(
                pp + ".cumulative: expected two entries");
          }
          for (std::size_t channel = 0; channel < 2; ++channel) {
            point.cumulative[channel] = require_finite_number(
                cumulative[channel], pp + ".cumulative");
          }
          activity.credit_curve.push_back(point);
        }
        parsed_curve = true;
      }
      epoch.activities.push_back(std::move(activity));
    }
    plan->epochs.push_back(std::move(epoch));
  }
  if (curve_schema && !parsed_curve) {
    throw std::invalid_argument(
        "plan.schema: coarse_die_plan_v4 requires a nonempty credit_curve");
  }
  const auto& fingerprint = require_member(j, "fingerprint", "plan");
  if (!fingerprint.is_string()) throw std::invalid_argument("plan.fingerprint: expected string");
  plan->fingerprint = fingerprint.get<std::string>();
  validate_plan(*plan);
  const std::string computed = stable_hash(make_payload(*plan).dump());
  if (plan->fingerprint != computed) {
    throw std::invalid_argument("plan.fingerprint: canonical payload hash mismatch");
  }
  return plan;
}

}  // namespace coarse_die
