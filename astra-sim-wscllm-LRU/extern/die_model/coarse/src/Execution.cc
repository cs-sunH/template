#include "coarse_die/Execution.hh"

#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <limits>
#include <numeric>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <tuple>
#include <unordered_map>
#include <utility>

namespace coarse_die {
namespace {

constexpr long double kNsPerSecond = 1.0e9L;
constexpr long double kTimeEpsilonNs = 1.0e-15L;
constexpr double kProgressTolerance = 2.0e-13;
constexpr double kCreditTolerance = 2.0e-10;
constexpr std::size_t kMaxPlanEpochs = 64;
constexpr std::uint64_t kMaxAdvanceEvents = 1000000;

const std::array<const char*, resource_count> kResourceNames = {
    "COMPUTE_GATE", "SRAM_READ", "SRAM_WRITE", "LOCAL", "HBM0",
    "HBM1", "INJECT0", "INJECT1", "XPOS", "XNEG", "YPOS", "YNEG",
    "HOP"};

bool same_owner(const Owner& lhs, const Owner& rhs) {
    return lhs.request_id == rhs.request_id && lhs.node_id == rhs.node_id &&
           lhs.execution_id == rhs.execution_id &&
           lhs.generation == rhs.generation;
}

std::string role_name(Role role) {
  switch (role) {
    case Role::Load:
      return "load";
    case Role::Compute:
      return "compute";
    case Role::Store:
      return "store";
    case Role::Endpoint:
      return "endpoint";
  }
  return "unknown";
}

constexpr std::array<const char*, served_category_count> kServedCategoryNames = {
    "comp", "comm_send", "comm_recv", "restore", "pool", "die"};

std::string decimal_measure(long double value);

ServedCategory plan_served_category(const Plan& plan) {
  const auto& task = plan.task;
  if (!task.is_object() ||
      task.value("schema", std::string()) != "coarse_die_endpoint_v2") {
    if (plan.summary.is_object() &&
        plan.summary.value("kind", std::string()) == "endpoint") {
      throw std::runtime_error(
          "invalid_endpoint: endpoint plan is missing its canonical endpoint task");
    }
    return ServedCategory::Comp;
  }
  if (!task.contains("endpoint") || !task.at("endpoint").is_object()) {
    throw std::runtime_error(
        "invalid_endpoint: endpoint plan is missing its canonical endpoint task");
  }
  const auto& endpoint = task.at("endpoint");
  const std::string owner_kind = endpoint.at("owner_kind").get<std::string>();
  if (owner_kind == "comm") {
    const std::string direction = endpoint.at("direction").get<std::string>();
    if (direction == "read") return ServedCategory::CommSend;
    if (direction == "write") return ServedCategory::CommRecv;
    throw std::runtime_error(
        "invalid_endpoint: COMM owner direction must be read or write");
  }
  if (owner_kind == "restore") return ServedCategory::Restore;
  if (owner_kind == "pool") return ServedCategory::Pool;
  if (owner_kind == "die") return ServedCategory::Die;
  throw std::runtime_error("invalid_endpoint: unsupported owner_kind in runtime plan");
}

nlohmann::json hbm_matrix_json(
    const std::array<std::array<long double, 2>, 2>& matrix,
    bool exact) {
  nlohmann::json result = nlohmann::json::array();
  for (std::size_t row = 0; row < 2; ++row) {
    nlohmann::json directions = nlohmann::json::array();
    for (std::size_t direction = 0; direction < 2; ++direction) {
      directions.push_back(exact
          ? nlohmann::json(decimal_measure(matrix[row][direction]))
          : nlohmann::json(static_cast<double>(matrix[row][direction])));
    }
    result.push_back(std::move(directions));
  }
  return result;
}

nlohmann::json served_categories_snapshot_json(
    const std::array<ServedCounters, served_category_count>& counters) {
  nlohmann::json result = nlohmann::json::object();
  for (std::size_t index = 0; index < served_category_count; ++index) {
    const ServedCounters& item = counters[index];
    result[kServedCategoryNames[index]] = {
        {"hbm_served_bytes_by_row_direction_exact",
         hbm_matrix_json(item.hbm_bytes, true)},
        {"int4_ops_exact_decimal", decimal_measure(item.int4_ops)},
        {"vec_eqops_exact_decimal", decimal_measure(item.vec_eqops)}};
  }
  return result;
}

nlohmann::json hardware_raw_denominators(const Hardware& hardware) {
  nlohmann::json result = {
      {"hardware_fingerprint", hardware.fingerprint},
      {"peak_int4_ops_per_s", nullptr},
      {"peak_vec_fp32_eq_ops_per_s", nullptr},
      {"hbm_row_peak_Bps", nullptr}};
  const auto found = hardware.derived.find("hardware_summary");
  if (found == hardware.derived.end() || !found->is_object()) return result;
  const auto& raw = *found;
  for (const char* key : {"peak_int4_ops_per_s",
                          "peak_vec_fp32_eq_ops_per_s"}) {
    const auto value = raw.find(key);
    if (value != raw.end() && value->is_number()) result[key] = *value;
  }
  const auto rows = raw.find("hbm_row_peak_Bps");
  if (rows != raw.end() && rows->is_array() && rows->size() == 2) {
    result["hbm_row_peak_Bps"] = *rows;
  }
  result["available"] = result.at("peak_int4_ops_per_s").is_number() &&
                         result.at("peak_vec_fp32_eq_ops_per_s").is_number() &&
                         result.at("hbm_row_peak_Bps").is_array();
  return result;
}

std::string long_double_text(long double value) {
  std::ostringstream stream;
  stream << std::setprecision(std::numeric_limits<long double>::max_digits10)
         << std::scientific << value;
  return stream.str();
}

std::string decimal_measure(long double value) {
  if (!std::isfinite(value)) return "nonfinite";
  std::ostringstream stream;
  stream << std::fixed << std::setprecision(18) << value;
  std::string text = stream.str();
  const auto dot = text.find('.');
  if (dot != std::string::npos) {
    while (!text.empty() && text.back() == '0') text.pop_back();
    if (!text.empty() && text.back() == '.') text.pop_back();
  }
  if (text.empty() || text == "-0") return "0";
  // Preserve the existing short form when it is exact. Very small fractional
  // work and extended-precision sums need enough digits to survive snapshots.
  char* end = nullptr;
  if (std::strtold(text.c_str(), &end) != value || !end || *end != '\0')
    return long_double_text(value);
  return text;
}

std::uint64_t parse_u64(const nlohmann::json& value, const char* field) {
  if (value.is_number_unsigned()) return value.get<std::uint64_t>();
  if (value.is_number_integer()) {
    const auto signed_value = value.get<std::int64_t>();
    if (signed_value >= 0) return static_cast<std::uint64_t>(signed_value);
  }
  if (value.is_string()) {
    const std::string text = value.get<std::string>();
    if (text.empty()) throw std::invalid_argument(std::string("invalid_endpoint: ") + field);
    std::uint64_t result = 0;
    const auto parsed = std::from_chars(text.data(), text.data() + text.size(), result);
    if (parsed.ec == std::errc{} && parsed.ptr == text.data() + text.size()) {
      return result;
    }
  }
  throw std::invalid_argument(std::string("invalid_endpoint: ") + field +
                              " must be an unsigned 64-bit integer or decimal string");
}

double parse_finite_nonnegative(const nlohmann::json& value,
                                const std::string& field,
                                bool allow_zero = true) {
  if (!value.is_number()) {
    throw std::invalid_argument("invalid_input: " + field + " must be numeric");
  }
  const double result = value.get<double>();
  if (!std::isfinite(result) || result < 0.0 || (!allow_zero && result == 0.0)) {
    throw std::invalid_argument("invalid_input: " + field +
                                " must be finite and " +
                                (allow_zero ? "nonnegative" : "positive"));
  }
  return result;
}

nlohmann::json owner_json(const Owner& owner) {
  return {{"request_id", owner.request_id},
          {"node_id", owner.node_id},
          {"execution_id", owner.execution_id},
          {"generation", owner.generation},
          {"key", owner.key()}};
}

Owner parse_owner(const nlohmann::json& value) {
  Owner result;
  result.request_id = parse_u64(value.at("request_id"), "owner.request_id");
  result.node_id = parse_u64(value.at("node_id"), "owner.node_id");
  result.execution_id = parse_u64(value.at("execution_id"), "owner.execution_id");
  result.generation = parse_u64(value.at("generation"), "owner.generation");
  if (result.generation == 0) {
    throw std::invalid_argument("invalid_snapshot: owner generation must be positive");
  }
  return result;
}

bool resource_enabled(const Hardware& hardware, std::size_t resource) {
  if (resource >= resource_count) return false;
  const auto& resources = hardware.derived.at("resources");
  if (!resources.is_array() || resources.size() != resource_count) {
    throw std::invalid_argument("invalid_hardware: derived.resources must contain 13 entries");
  }
  const auto& descriptor = resources.at(resource);
  if (descriptor.at("resource_id").get<std::size_t>() != resource ||
      descriptor.at("id").get<std::string>() != kResourceNames[resource]) {
    throw std::invalid_argument("invalid_hardware: resource descriptor order mismatch");
  }
  return descriptor.at("enabled").get<bool>();
}

void validate_hardware(const Hardware& hardware) {
  if (hardware.fingerprint.empty()) {
    throw std::invalid_argument("invalid_hardware: missing fingerprint");
  }
  const auto& resources = hardware.derived.at("resources");
  if (!resources.is_array() || resources.size() != resource_count) {
    throw std::invalid_argument("invalid_hardware: derived.resources must contain 13 entries");
  }
  for (std::size_t i = 0; i < resource_count; ++i) {
    const auto& descriptor = resources.at(i);
    if (descriptor.at("resource_id").get<std::size_t>() != i ||
        descriptor.at("id").get<std::string>() != kResourceNames[i]) {
      throw std::invalid_argument("invalid_hardware: resource descriptor order mismatch");
    }
    const double rate = hardware.rates[i];
    if (!std::isfinite(rate) || rate < 0.0) {
      throw std::invalid_argument("invalid_hardware: resource rate must be finite and nonnegative");
    }
    if (descriptor.at("rate").get<double>() != rate) {
      throw std::invalid_argument("invalid_hardware: rates and derived resource rates differ");
    }
    if (!descriptor.at("enabled").get<bool>() && rate != 0.0) {
      throw std::invalid_argument("invalid_hardware: disabled resources must have zero rate");
    }
  }
}

void validate_plan(const Hardware& hardware, const Plan& plan,
                   bool require_fingerprint = true) {
  if (plan.hardware_fingerprint != hardware.fingerprint) {
    throw std::invalid_argument("incompatible_fingerprint: plan hardware fingerprint mismatch");
  }
  if (plan.epochs.size() > kMaxPlanEpochs) {
    throw std::invalid_argument("unsupported_plan: more than 64 runtime epochs");
  }
  if (require_fingerprint && plan.fingerprint.empty()) {
    throw std::invalid_argument("invalid_plan: missing fingerprint");
  }
  std::set<std::pair<std::size_t, std::uint64_t>> activity_ids;
  for (std::size_t epoch_index = 0; epoch_index < plan.epochs.size(); ++epoch_index) {
    const Epoch& epoch = plan.epochs[epoch_index];
    if (epoch.activities.size() > 3) {
      throw std::invalid_argument("unsupported_plan: an epoch may contain at most three concurrent activities");
    }
    if (epoch.repeat_count == 0) {
      throw std::invalid_argument("invalid_plan: repeat_count must be positive");
    }
    for (std::size_t buffer = 0; buffer < 2; ++buffer) {
      const double capacity = epoch.credit_capacity[buffer];
      const double initial = epoch.credit_initial[buffer];
      if (!std::isfinite(capacity) || capacity < 0.0 || !std::isfinite(initial) ||
          initial < 0.0 || initial > capacity + kCreditTolerance) {
        throw std::invalid_argument("invalid_plan: invalid initial credit or capacity");
      }
    }
    for (const Activity& activity : epoch.activities) {
      if (!activity_ids.emplace(epoch_index, activity.id).second) {
        throw std::invalid_argument("invalid_plan: duplicate activity id in epoch");
      }
      bool any_work = false;
      for (std::size_t resource = 0; resource < resource_count; ++resource) {
        const double work = activity.work[resource];
        if (!std::isfinite(work) || work < 0.0) {
          throw std::invalid_argument("invalid_plan: activity work must be finite and nonnegative");
        }
        if (work > 0.0) {
          any_work = true;
          if (!resource_enabled(hardware, resource)) {
            throw std::invalid_argument("invalid_hardware: activity uses a disabled resource");
          }
          if (hardware.rates[resource] <= 0.0) {
            throw std::invalid_argument("invalid_hardware: active work requires positive resource capacity");
          }
        }
      }
      if (any_work && (!std::isfinite(activity.canonical_q) || activity.canonical_q <= 0.0)) {
        throw std::invalid_argument("invalid_plan: active activity requires positive canonical_q");
      }
      if (!std::isfinite(activity.private_time_s) || activity.private_time_s < 0.0) {
        throw std::invalid_argument("invalid_plan: private_time_s must be finite and nonnegative");
      }
      if (activity.private_time_s > 0.0 && !std::isfinite(1.0 / activity.private_time_s)) {
        throw std::invalid_argument("invalid_plan: private-time rate bound overflows");
      }
      if (!std::isfinite(activity.vec_eqops_fraction) ||
          activity.vec_eqops_fraction < 0.0 || activity.vec_eqops_fraction >= 1.0 ||
          (activity.vec_eqops == std::numeric_limits<std::uint64_t>::max() &&
           activity.vec_eqops_fraction != 0.0)) {
        throw std::invalid_argument("invalid_plan: fractional Vec work must be a remainder in [0,1) within uint64 work range");
      }
      (void)planned_vec_eqops(activity);
      for (double delta : activity.credit_delta) {
        if (!std::isfinite(delta)) {
          throw std::invalid_argument("invalid_plan: credit_delta must be finite");
        }
      }
      for (std::size_t row = 0; row < 2; ++row) {
        for (std::size_t direction = 0; direction < 2; ++direction) {
          if (activity.hbm_bytes[row][direction] != 0 &&
              activity.work[row == 0 ? HBM0 : HBM1] <= 0.0) {
            throw std::invalid_argument("invalid_plan: HBM row payload has no corresponding row service work");
          }
        }
      }
    }
  }
}

std::array<double, resource_count> json_resource_array(const nlohmann::json& value,
                                                       const std::string& field) {
  std::array<double, resource_count> result{};
  if (!value.is_array() || value.size() != resource_count) {
    throw std::invalid_argument("invalid_snapshot: " + field + " must have 13 values");
  }
  for (std::size_t i = 0; i < resource_count; ++i) {
    result[i] = parse_finite_nonnegative(value.at(i), field + "[" + std::to_string(i) + " ]");
  }
  return result;
}

nlohmann::json resource_array_json(const std::array<double, resource_count>& values) {
  nlohmann::json result = nlohmann::json::array();
  for (double value : values) result.push_back(value);
  return result;
}

nlohmann::json credit_json(const std::array<double, 2>& values,
                           const std::array<double, 2>& capacities) {
  nlohmann::json result = nlohmann::json::array();
  for (std::size_t i = 0; i < 2; ++i) {
    result.push_back({{"index", i}, {"value", values[i]}, {"capacity", capacities[i]}});
  }
  return result;
}

long double min_positive(long double current, long double candidate) {
  if (!std::isfinite(candidate) || candidate <= kTimeEpsilonNs) return current;
  return candidate < current ? candidate : current;
}

struct EndpointSpec {
  std::uint64_t endpoint_id = 0;
  std::uint64_t transaction_id = 0;
  std::uint64_t initial_delay_ns = 0;
  std::string owner_kind;
  std::string direction;
  std::string path_class;
  std::uint64_t physical_bytes = 0;
  std::array<double, 2> row_fraction{0.5, 0.5};
  bool uniform_default = true;
  double eta_hbm = 1.0;
  bool has_hbm = false;
  bool has_route = false;
  bool internal_route = false;
  std::array<double, resource_count> route_work{};
  nlohmann::json join_token;
};

std::size_t parse_resource_id(const nlohmann::json& key) {
  if (key.is_number_integer() || key.is_number_unsigned()) {
    const auto index = key.get<std::size_t>();
    if (index < resource_count) return index;
  }
  if (key.is_string()) {
    const auto name = key.get<std::string>();
    for (std::size_t i = 0; i < resource_count; ++i) {
      if (name == kResourceNames[i]) return i;
    }
    std::size_t index = 0;
    const auto parsed = std::from_chars(name.data(), name.data() + name.size(), index);
    if (parsed.ec == std::errc{} && parsed.ptr == name.data() + name.size() &&
        index < resource_count) {
      return index;
    }
  }
  throw std::invalid_argument("invalid_endpoint: unknown route resource id");
}

void parse_route_work(const nlohmann::json& route,
                      std::array<double, resource_count>& work) {
  if (!route.contains("resource_work")) {
    throw std::invalid_argument("invalid_endpoint: route.resource_work is required");
  }
  const auto& values = route.at("resource_work");
  if (values.is_array()) {
    if (values.size() != resource_count) {
      throw std::invalid_argument("invalid_endpoint: route.resource_work array must have 13 values");
    }
    for (std::size_t i = 0; i < resource_count; ++i) {
      work[i] = parse_finite_nonnegative(values.at(i), "route.resource_work[" + std::to_string(i) + "]");
    }
  } else if (values.is_object()) {
    std::array<bool, resource_count> seen{};
    for (auto it = values.begin(); it != values.end(); ++it) {
      const std::size_t index = parse_resource_id(nlohmann::json(it.key()));
      if (seen[index]) {
        throw std::invalid_argument("invalid_endpoint: duplicate route resource id");
      }
      seen[index] = true;
      work[index] = parse_finite_nonnegative(it.value(), "route.resource_work." + it.key());
    }
  } else {
    throw std::invalid_argument("invalid_endpoint: route.resource_work must be a 13-value array or object");
  }
}

EndpointSpec parse_endpoint(const nlohmann::json& endpoint, const Hardware& hardware) {
  if (!endpoint.is_object()) {
    throw std::invalid_argument("invalid_endpoint: endpoint must be a JSON object");
  }
  EndpointSpec result;
  result.endpoint_id = parse_u64(endpoint.at("endpoint_id"), "endpoint_id");
  result.transaction_id = parse_u64(endpoint.at("transaction_id"), "transaction_id");
  if (endpoint.contains("initial_delay_ns")) {
    result.initial_delay_ns =
        parse_u64(endpoint.at("initial_delay_ns"), "initial_delay_ns");
  }
  result.owner_kind = endpoint.at("owner_kind").get<std::string>();
  result.direction = endpoint.at("direction").get<std::string>();
  result.path_class = endpoint.at("path_class").get<std::string>();
  result.physical_bytes = parse_u64(endpoint.at("physical_bytes"), "physical_bytes");
  if (result.endpoint_id == 0 || result.transaction_id == 0) {
    throw std::invalid_argument("invalid_endpoint: endpoint_id and transaction_id must be positive");
  }
  if (result.owner_kind != "comm" && result.owner_kind != "restore" &&
      result.owner_kind != "pool" && result.owner_kind != "die") {
    throw std::invalid_argument("invalid_endpoint: owner_kind must be comm, restore, pool, or die");
  }
  if (result.direction != "read" && result.direction != "write") {
    throw std::invalid_argument("invalid_endpoint: direction must be read or write");
  }
  result.join_token = endpoint.value("join_token", nlohmann::json());

  if (endpoint.contains("row_fraction")) {
    result.uniform_default = false;
    const auto& fractions = endpoint.at("row_fraction");
    if (!fractions.is_array() || fractions.size() != 2) {
      throw std::invalid_argument("invalid_endpoint: row_fraction must have two values");
    }
    result.row_fraction[0] = parse_finite_nonnegative(fractions.at(0), "row_fraction[0]");
    result.row_fraction[1] = parse_finite_nonnegative(fractions.at(1), "row_fraction[1]");
    const double sum = result.row_fraction[0] + result.row_fraction[1];
    if (!std::isfinite(sum) || std::abs(sum - 1.0) > 1.0e-9) {
      throw std::invalid_argument("invalid_endpoint: row_fraction must sum to one");
    }
    result.row_fraction[0] /= sum;
    result.row_fraction[1] /= sum;
  }

  const bool core_path = result.path_class == "hbm_to_core" || result.path_class == "core_to_hbm";
  const bool edge_path = result.path_class == "hbm_to_edge" || result.path_class == "edge_to_hbm";
  const bool pool_path = result.path_class == "remote_pool_only";
  const bool noc_path = result.path_class == "noc_only";
  if (!core_path && !edge_path && !pool_path && !noc_path) {
    throw std::invalid_argument("invalid_endpoint: unsupported path_class");
  }
  if ((result.path_class == "hbm_to_core" || result.path_class == "hbm_to_edge") &&
      result.direction != "read") {
    throw std::invalid_argument("invalid_endpoint: hbm_to_* requires direction=read");
  }
  if ((result.path_class == "core_to_hbm" || result.path_class == "edge_to_hbm") &&
      result.direction != "write") {
    throw std::invalid_argument("invalid_endpoint: *_to_hbm requires direction=write");
  }

  result.has_hbm = core_path || edge_path;
  if (result.has_hbm) {
    if (!endpoint.contains("eta_hbm")) {
      throw std::invalid_argument("invalid_endpoint: local HBM paths require eta_hbm");
    }
    result.eta_hbm = parse_finite_nonnegative(endpoint.at("eta_hbm"), "eta_hbm", false);
    if (result.eta_hbm > 1.0) {
      throw std::invalid_argument("invalid_endpoint: eta_hbm must be at most one");
    }
  }

  if (core_path) {
    if (!endpoint.contains("route") || !endpoint.at("route").is_object()) {
      throw std::invalid_argument("missing_endpoint_route: Core endpoints require explicit route work");
    }
    result.has_route = true;
    parse_route_work(endpoint.at("route"), result.route_work);
  } else if (edge_path) {
    if (!endpoint.contains("route") || !endpoint.at("route").is_object() ||
        !endpoint.at("route").contains("internal_route") ||
        !endpoint.at("route").at("internal_route").is_boolean()) {
      throw std::invalid_argument("invalid_endpoint: HBM edge path requires explicit internal_route boolean");
    }
    result.internal_route = endpoint.at("route").at("internal_route").get<bool>();
    if (result.internal_route) {
      result.has_route = true;
      parse_route_work(endpoint.at("route"), result.route_work);
    } else if (endpoint.at("route").contains("resource_work")) {
      throw std::invalid_argument("invalid_endpoint: internal_route=false cannot carry resource_work");
    }
  } else if (noc_path) {
    if (!endpoint.contains("route") || !endpoint.at("route").is_object()) {
      throw std::invalid_argument("invalid_endpoint: noc_only path requires explicit route work");
    }
    result.has_route = true;
    parse_route_work(endpoint.at("route"), result.route_work);
  } else if (pool_path && endpoint.contains("route")) {
    throw std::invalid_argument("invalid_endpoint: remote_pool_only has no local route work");
  }

  for (std::size_t i = 0; i < resource_count; ++i) {
    if (result.route_work[i] > 0.0 && !resource_enabled(hardware, i)) {
      throw std::invalid_argument("invalid_hardware: endpoint route uses a disabled resource");
    }
  }
  return result;
}

std::uint64_t row_bytes(std::uint64_t total, double fraction);

Activity endpoint_activity(const EndpointSpec& spec, bool legacy_uniform = false) {
  Activity activity;
  activity.id = spec.endpoint_id;
  activity.role = Role::Endpoint;
  activity.hbm_bytes = {};
  activity.int4_ops = 0;
  activity.vec_eqops = 0;
  activity.credit_delta = {};
  activity.private_time_s = 0.0;
  const std::size_t direction_index = spec.direction == "read" ? 0 : 1;
  if (spec.has_hbm) {
    const std::uint64_t row0 = spec.uniform_default && !legacy_uniform
        ? spec.physical_bytes / 2 + spec.physical_bytes % 2
        : row_bytes(spec.physical_bytes, spec.row_fraction[0]);
    const std::uint64_t row1 = spec.physical_bytes - row0;
    const std::array<std::uint64_t, 2> bytes{row0, row1};
    for (std::size_t row = 0; row < 2; ++row) {
      activity.hbm_bytes[row][direction_index] = bytes[row];
      const std::size_t resource = row == 0 ? HBM0 : HBM1;
      activity.work[resource] = static_cast<double>(static_cast<long double>(bytes[row]) /
                                                    static_cast<long double>(spec.eta_hbm));
    }
  }
  for (std::size_t resource = 0; resource < resource_count; ++resource) {
    activity.work[resource] += spec.route_work[resource];
  }
  activity.canonical_q = canonical_basis(activity, static_cast<double>(spec.physical_bytes));
  return activity;
}

std::uint64_t row_bytes(std::uint64_t total, double fraction) {
  const long double raw = static_cast<long double>(total) * static_cast<long double>(fraction);
  if (raw <= 0.0L) return 0;
  if (raw >= static_cast<long double>(total)) return total;
  return static_cast<std::uint64_t>(std::floor(raw));
}

enum class LpStatus { Optimal, Infeasible, Unbounded, PivotBudget };
constexpr double kLpEpsilon = 1.0e-10;

struct LpResult {
  LpStatus status = LpStatus::Infeasible;
  double objective = 0.0;
  std::vector<double> primal;
  std::vector<double> dual_rows;
};

class DenseSimplex {
 public:
  DenseSimplex(const std::vector<std::vector<double>>& input_a,
               const std::vector<double>& input_b,
               const std::vector<double>& objective,
               std::size_t& pivot_count)
      : m_(static_cast<int>(input_b.size())),
        n_(static_cast<int>(objective.size())),
        basis_(m_),
        nonbasis_(n_ + 1),
        tableau_(m_ + 2, std::vector<double>(n_ + 2, 0.0)),
        pivot_count_(pivot_count) {
    if (input_a.size() != input_b.size()) {
      throw std::invalid_argument("numerical_failure: LP row count mismatch");
    }
    for (int i = 0; i < m_; ++i) {
      if (input_a[static_cast<std::size_t>(i)].size() != static_cast<std::size_t>(n_)) {
        throw std::invalid_argument("numerical_failure: LP column count mismatch");
      }
      double scale = 0.0;
      for (double value : input_a[static_cast<std::size_t>(i)]) {
        if (!std::isfinite(value)) throw std::invalid_argument("numerical_failure: nonfinite LP coefficient");
        scale = std::max(scale, std::abs(value));
      }
      if (!std::isfinite(input_b[static_cast<std::size_t>(i)])) {
        throw std::invalid_argument("numerical_failure: nonfinite LP bound");
      }
      if (scale == 0.0) scale = 1.0;
      for (int j = 0; j < n_; ++j) {
        tableau_[i][j] = input_a[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] / scale;
      }
      tableau_[i][n_] = -1.0;
      tableau_[i][n_ + 1] = input_b[static_cast<std::size_t>(i)] / scale;
      if (!std::isfinite(tableau_[i][n_ + 1])) {
        throw std::invalid_argument("numerical_failure: LP row scaling overflow");
      }
      basis_[i] = n_ + i;
    }
    for (int j = 0; j < n_; ++j) {
      if (!std::isfinite(objective[static_cast<std::size_t>(j)])) {
        throw std::invalid_argument("numerical_failure: nonfinite LP objective");
      }
      nonbasis_[j] = j;
      tableau_[m_][j] = -objective[static_cast<std::size_t>(j)];
    }
    nonbasis_[n_] = -1;
    tableau_[m_ + 1][n_] = 1.0;
  }

  LpResult solve() {
    LpResult result;
    int row = 0;
    for (int i = 1; i < m_; ++i) {
      if (tableau_[i][n_ + 1] < tableau_[row][n_ + 1] - kEps ||
          (std::abs(tableau_[i][n_ + 1] - tableau_[row][n_ + 1]) <= kEps &&
           basis_[i] < basis_[row])) {
        row = i;
      }
    }
    if (m_ > 0 && tableau_[row][n_ + 1] < -kEps) {
      if (!pivot(row, n_)) {
        result.status = LpStatus::PivotBudget;
        return result;
      }
      const auto phase1 = simplex(1);
      if (phase1 == LpStatus::PivotBudget) {
        result.status = phase1;
        return result;
      }
      if (phase1 == LpStatus::Unbounded || tableau_[m_ + 1][n_ + 1] < -kEps ||
          std::abs(tableau_[m_ + 1][n_ + 1]) > 1.0e-7) {
        result.status = LpStatus::Infeasible;
        return result;
      }
      for (int i = 0; i < m_; ++i) {
        if (basis_[i] != -1) continue;
        int column = -1;
        for (int j = 0; j <= n_; ++j) {
          if (column == -1 || tableau_[i][j] < tableau_[i][column] - kEps ||
              (std::abs(tableau_[i][j] - tableau_[i][column]) <= kEps &&
               nonbasis_[j] < nonbasis_[column])) {
            column = j;
          }
        }
        if (column >= 0 && std::abs(tableau_[i][column]) > kEps) {
          if (!pivot(i, column)) {
            result.status = LpStatus::PivotBudget;
            return result;
          }
        }
      }
    }
    const auto phase2 = simplex(2);
    if (phase2 != LpStatus::Optimal) {
      result.status = phase2;
      return result;
    }
    result.status = LpStatus::Optimal;
    result.objective = tableau_[m_][n_ + 1];
    result.primal.assign(static_cast<std::size_t>(n_), 0.0);
    for (int i = 0; i < m_; ++i) {
      if (basis_[i] >= 0 && basis_[i] < n_) {
        result.primal[static_cast<std::size_t>(basis_[i])] = tableau_[i][n_ + 1];
      }
    }
    result.dual_rows.assign(static_cast<std::size_t>(m_), 0.0);
    for (int i = 0; i < m_; ++i) {
      const int slack_id = n_ + i;
      for (int j = 0; j <= n_; ++j) {
        if (nonbasis_[j] == slack_id) {
          result.dual_rows[static_cast<std::size_t>(i)] = std::max(0.0, tableau_[m_][j]);
          break;
        }
      }
    }
    return result;
  }

 private:
  static constexpr double kEps = kLpEpsilon;
  int m_ = 0;
  int n_ = 0;
  std::vector<int> basis_;
  std::vector<int> nonbasis_;
  std::vector<std::vector<double>> tableau_;
  std::size_t& pivot_count_;
  bool budget_exceeded_ = false;

  bool pivot(int row, int column) {
    if (++pivot_count_ > 256) {
      budget_exceeded_ = true;
      return false;
    }
    const double pivot_value = tableau_[row][column];
    if (!std::isfinite(pivot_value) || std::abs(pivot_value) <= kEps) return false;
    const double inverse = 1.0 / pivot_value;
    for (int i = 0; i < m_ + 2; ++i) {
      if (i == row) continue;
      for (int j = 0; j < n_ + 2; ++j) {
        if (j == column) continue;
        tableau_[i][j] -= tableau_[row][j] * tableau_[i][column] * inverse;
      }
    }
    for (int j = 0; j < n_ + 2; ++j) {
      if (j != column) tableau_[row][j] *= inverse;
    }
    for (int i = 0; i < m_ + 2; ++i) {
      if (i != row) tableau_[i][column] *= -inverse;
    }
    tableau_[row][column] = inverse;
    std::swap(basis_[row], nonbasis_[column]);
    return true;
  }

  LpStatus simplex(int phase) {
    const int objective_row = phase == 1 ? m_ + 1 : m_;
    for (;;) {
      int entering = -1;
      for (int column = 0; column <= n_; ++column) {
        if (phase == 2 && nonbasis_[column] == -1) continue;
        if (entering == -1 || tableau_[objective_row][column] <
                                  tableau_[objective_row][entering] - kEps ||
            (std::abs(tableau_[objective_row][column] -
                      tableau_[objective_row][entering]) <= kEps &&
             nonbasis_[column] < nonbasis_[entering])) {
          entering = column;
        }
      }
      if (entering == -1 || tableau_[objective_row][entering] >= -kEps) {
        return LpStatus::Optimal;
      }
      int leaving = -1;
      for (int row = 0; row < m_; ++row) {
        if (tableau_[row][entering] <= kEps) continue;
        if (leaving == -1) {
          leaving = row;
          continue;
        }
        const double lhs = tableau_[row][n_ + 1] / tableau_[row][entering];
        const double rhs = tableau_[leaving][n_ + 1] / tableau_[leaving][entering];
        if (lhs < rhs - kEps ||
            (std::abs(lhs - rhs) <= kEps && basis_[row] < basis_[leaving])) {
          leaving = row;
        }
      }
      if (leaving == -1) return LpStatus::Unbounded;
      if (!pivot(leaving, entering)) {
        return budget_exceeded_ ? LpStatus::PivotBudget : LpStatus::Infeasible;
      }
    }
  }
};

}  // namespace

void recompute_credit(Execution& execution);

std::string Owner::key() const {
  // request_id is tracing metadata. Fairness belongs to the outer execution,
  // so splitting it across requests cannot mint extra owner weight.
  return std::to_string(node_id) + ":" + std::to_string(execution_id) + ":" +
         std::to_string(generation);
}

Runtime::Runtime(Hardware hardware)
    : Runtime(std::make_shared<const Hardware>(std::move(hardware))) {}

Runtime::Runtime(std::shared_ptr<const Hardware> hardware)
    : hardware_(std::move(hardware)) {
  if (!hardware_) throw std::invalid_argument("invalid_hardware: null shared hardware");
  validate_hardware(*hardware_);
  capacities_ = hardware_->rates;
}

std::uint64_t Runtime::submit(std::shared_ptr<const Plan> plan, Owner owner,
                              std::uint64_t now_ns) {
  return submit_impl(std::move(plan), owner, now_ns, true);
}

std::uint64_t Runtime::submit_impl(std::shared_ptr<const Plan> plan, Owner owner,
                                   std::uint64_t now_ns,
                                   bool require_fingerprint) {
  if (!plan) throw std::invalid_argument("invalid_plan: null plan");
  if (owner.generation == 0) throw std::invalid_argument("invalid_owner: generation must be positive");
  validate_plan(*hardware_, *plan, require_fingerprint);
  if (now_ns < now_ns_) throw std::invalid_argument("invalid_time: submit time moved backwards");
  if (plan->launch_ns > std::numeric_limits<std::uint64_t>::max() - now_ns) {
    throw std::overflow_error("overflow: ready time exceeds uint64");
  }
  advance_to(now_ns);
  if (next_handle_ == 0 || next_handle_ == std::numeric_limits<std::uint64_t>::max()) {
    throw std::overflow_error("overflow: execution handle space exhausted");
  }
  Execution execution;
  execution.handle = next_handle_;
  execution.issue_ns = now_ns;
  execution.ready_ns = now_ns + plan->launch_ns;
  execution.owner = owner;
  execution.served_category = plan_served_category(*plan);
  execution.plan = std::move(plan);
  execution.epoch = 0;
  for (std::size_t epoch = 0; epoch < execution.plan->epochs.size(); ++epoch) {
    for (const Activity& definition : execution.plan->epochs[epoch].activities) {
      execution.activities.push_back(LiveActivity{definition, epoch, 0.0, 0.0, 0.0, false});
    }
  }
  if (!execution.plan->epochs.empty()) {
    execution.credit = execution.plan->epochs.front().credit_initial;
    execution.credit_capacity = execution.plan->epochs.front().credit_capacity;
  }
  const auto handle = execution.handle;
  // A solver error must not leave an execution whose handle was never
  // returned to its caller. advance_to(now_ns) above remains committed: it
  // services the pre-existing world before this attempted arrival. Preserve
  // only the bounded mutable allocation/accounting state, never copy Plans
  // or the Runtime's JSON/cache state on this hot path.
  struct SavedRate { LiveActivity* live; double rate, alpha; };
  std::vector<SavedRate> saved_rates;
  for (auto& entry : executions_) {
    for (auto& live : entry.second.activities)
      saved_rates.push_back({&live, live.rate, live.alpha});
  }
  const auto saved_hbm = hbm_served_;
  const auto saved_categories = served_by_category_;
  const auto saved_events = internal_events_;
  const auto saved_allocations = allocation_updates_;
  const auto saved_generation = generation_;
  const auto inserted = executions_.emplace(handle, std::move(execution));
  if (!inserted.second)
    throw std::runtime_error("invalid_execution: execution handle already exists");
  try {
    settle();
    allocate();
  } catch (...) {
    executions_.erase(inserted.first);
    for (const auto& saved : saved_rates) {
      saved.live->rate = saved.rate;
      saved.live->alpha = saved.alpha;
    }
    hbm_served_ = saved_hbm;
    served_by_category_ = saved_categories;
    internal_events_ = saved_events;
    allocation_updates_ = saved_allocations;
    generation_ = saved_generation;
    throw;
  }
  ++next_handle_;
  return handle;
}

std::uint64_t Runtime::submit_endpoint(const nlohmann::json& endpoint_json,
                                       Owner owner, std::uint64_t now_ns) {
  if (owner.generation == 0) throw std::invalid_argument("invalid_owner: generation must be positive");
  const EndpointSpec spec = parse_endpoint(endpoint_json, *hardware_);
  if (endpoint_registry_.count(spec.endpoint_id) != 0) {
    throw std::invalid_argument("duplicate_endpoint_id: endpoint_id is already live in this runtime");
  }
  const auto transaction = transaction_owners_.find(spec.transaction_id);
  if (transaction != transaction_owners_.end() && !same_owner(transaction->second.first, owner)) {
    throw std::invalid_argument("transaction_owner_conflict: transaction_id already belongs to another owner");
  }

  Activity activity = endpoint_activity(spec);

  auto plan = std::make_shared<Plan>();
  plan->task = {{"schema", "coarse_die_endpoint_v2"}, {"endpoint", endpoint_json}};
  plan->summary = {{"kind", "endpoint"},
                   {"endpoint_id", spec.endpoint_id},
                   {"transaction_id", spec.transaction_id},
                   {"owner_kind", spec.owner_kind},
                   {"direction", spec.direction},
                   {"path_class", spec.path_class},
                   {"physical_bytes", spec.physical_bytes},
                   {"row_fraction", {spec.row_fraction[0], spec.row_fraction[1]}},
                   {"allocation_evidence", spec.uniform_default ? "uniform_default" : "explicit_row_fraction"},
                   {"eta_hbm", spec.has_hbm ? nlohmann::json(spec.eta_hbm) : nlohmann::json(nullptr)},
                   {"route_charged", spec.has_route},
                   {"internal_route", spec.internal_route},
                   {"join_token", spec.join_token}};
  // Keep the canonical summary/fingerprint byte-compatible for old endpoint
  // JSON that omitted this optional field; explicit values are normalized
  // here while the original JSON remains in task.endpoint for snapshots.
  if (endpoint_json.contains("initial_delay_ns")) {
    plan->summary["initial_delay_ns"] = spec.initial_delay_ns;
  }
  plan->hardware_fingerprint = hardware_->fingerprint;
  plan->launch_ns = spec.initial_delay_ns;
  Epoch epoch;
  epoch.name = "endpoint";
  epoch.activities.push_back(activity);
  plan->epochs.push_back(std::move(epoch));
  // Snapshot serialization validates the same canonical payload hash used by
  // compiled Plans. Hashing only task JSON made endpoint plans impossible to
  // import after export, even though their runtime state was complete.
  plan->fingerprint = plan_to_json(*plan).at("fingerprint").get<std::string>();

  // Allocate both ownership records before accepting live work. A map
  // allocation failure after submit used to strand an unreturned handle.
  const auto endpoint_entry = endpoint_registry_.emplace(
      spec.endpoint_id, spec.transaction_id);
  auto transaction_entry = transaction_owners_.end();
  bool new_transaction = false;
  try {
    auto result = transaction_owners_.emplace(
        spec.transaction_id, std::make_pair(owner, std::size_t{0}));
    transaction_entry = result.first;
    new_transaction = result.second;
    if (transaction_entry->second.second == std::numeric_limits<std::size_t>::max())
      throw std::overflow_error("overflow: transaction endpoint count exhausted");
    const std::uint64_t handle = submit(plan, owner, now_ns);
    ++transaction_entry->second.second;
    return handle;
  } catch (...) {
    endpoint_registry_.erase(endpoint_entry.first);
    if (new_transaction) transaction_owners_.erase(transaction_entry);
    throw;
  }
}

void Runtime::settle() {
  for (auto& [handle, execution] : executions_) {
    (void)handle;
    if (execution.terminal || time_until(execution.ready_ns) > kTimeEpsilonNs) continue;
    std::size_t guard = 0;
    while (!execution.terminal && guard++ <= execution.plan->epochs.size() + execution.activities.size() + 1) {
      if (execution.epoch >= execution.plan->epochs.size()) {
        execution.terminal = true;
        execution.cancelled = false;
        execution.finish_ns = ceil_exact_time();
        for (LiveActivity& live : execution.activities) live.rate = 0.0;
        break;
      }
      const Epoch& epoch = execution.plan->epochs[execution.epoch];
      execution.credit_capacity = epoch.credit_capacity;
      ServedCounters& categorized = served_by_category_[
          static_cast<std::size_t>(execution.served_category)];
      if (execution.epoch == 0 && execution.activities.empty()) {
        execution.credit = epoch.credit_initial;
      }
      for (LiveActivity& live : execution.activities) {
        if (live.epoch_index != execution.epoch || live.done) continue;
        bool has_service = false;
        for (double work : live.definition.work) has_service = has_service || work > 0.0;
        const bool complete = !has_service || live.progress >= 1.0 - kProgressTolerance;
        if (complete) {
          live.progress = 1.0;
          for (std::size_t row = 0; row < 2; ++row) {
            for (std::size_t direction = 0; direction < 2; ++direction) {
              const long double planned = live.definition.hbm_bytes[row][direction];
              const long double correction = planned - live.hbm_served[row][direction];
              live.hbm_served[row][direction] = planned;
              execution.hbm_served[row][direction] += correction;
              hbm_served_[row][direction] += correction;
              categorized.hbm_bytes[row][direction] += correction;
            }
          }
          const long double int4_correction =
              static_cast<long double>(live.definition.int4_ops) - live.int4_served;
          live.int4_served = live.definition.int4_ops;
          execution.int4_served += int4_correction;
          categorized.int4_ops += int4_correction;
          const long double vec_correction =
              planned_vec_eqops(live.definition) - live.vec_served;
          live.vec_served = planned_vec_eqops(live.definition);
          execution.vec_served += vec_correction;
          categorized.vec_eqops += vec_correction;
          live.done = true;
          live.rate = 0.0;
          continue;
        }
      }
      recompute_credit(execution);
      const bool complete = std::all_of(
          execution.activities.begin(), execution.activities.end(),
          [&execution](const LiveActivity& live) {
            return live.epoch_index != execution.epoch || live.done;
          });
      if (!complete) break;
      ++execution.epoch;
      if (execution.epoch < execution.plan->epochs.size()) {
        execution.credit = execution.plan->epochs[execution.epoch].credit_initial;
        execution.credit_capacity = execution.plan->epochs[execution.epoch].credit_capacity;
      }
      ++internal_events_;
    }
    if (!execution.terminal && guard > execution.plan->epochs.size() + execution.activities.size() + 1) {
      throw std::runtime_error("numerical_failure: instantaneous epoch closure did not converge");
    }
  }
}

void Runtime::allocate() {
  struct Ref {
    Execution* execution = nullptr;
    LiveActivity* live = nullptr;
    std::uint64_t handle = 0;
    std::string owner_key;
    std::size_t first_resource = 0;
    double q = 0.0;
    double alpha = 0.0;
    std::size_t multiplicity = 1;
    std::vector<LiveActivity*> mirrors;
  };

  for (auto& [handle, execution] : executions_) {
    (void)handle;
    for (LiveActivity& live : execution.activities) {
      live.rate = 0.0;
      live.alpha = 0.0;
    }
  }
  ++allocation_updates_;
  ++generation_;

  std::vector<Ref> refs;
  std::map<std::string, long double> owner_q;
  for (auto& [handle, execution] : executions_) {
    (void)handle;
    if (execution.terminal || time_until(execution.ready_ns) > kTimeEpsilonNs) continue;
    for (LiveActivity& live : execution.activities) {
      if (live.epoch_index != execution.epoch || live.done) continue;
      bool has_service = false;
      for (double work : live.definition.work) has_service = has_service || work > 0.0;
      if (!has_service) continue;
      const double q = live.definition.canonical_q;
      if (!std::isfinite(q) || q <= 0.0) {
        throw std::runtime_error("numerical_failure: active activity has invalid canonical_q");
      }
      std::size_t first_resource = 0;
      while (first_resource < resource_count && live.definition.work[first_resource] <= 0.0) {
        ++first_resource;
      }
      refs.push_back(Ref{&execution, &live, handle, execution.owner.key(),
                         first_resource, q, 0.0});
    }
  }
  if (refs.empty()) return;
  if (refs.size() > 96) {
    for (Ref& ref : refs) ref.live->rate = 0.0;
    throw std::runtime_error("numerical_failure: active activity count exceeds 3x32 solver bound");
  }
  // Complete, exactly identical owner blocks can share LP variables. The
  // convex feasible set is invariant under their permutation, so weighted
  // lexicographic max-min assigns them identical rates. Only physical rows
  // gain multiplicity; each owner's PRIVATE/credit constraints and alpha are
  // retained. No progress/phase epsilon is used. Owners split across multiple
  // executions remain on the full path to preserve their Q denominator.
  const bool multiple_owners = std::any_of(refs.begin(), refs.end(),
      [&](const Ref& ref) { return ref.owner_key != refs.front().owner_key; });
  if (multiple_owners) {
    // Refs are collected contiguously per execution before the stable LP
    // sort. Store block ranges instead of allocating two maps and one vector
    // for each owner on every arbitration update.
    struct Block { std::size_t begin, end; };
    std::vector<Block> blocks;
    blocks.reserve(refs.size());
    for (std::size_t begin = 0; begin < refs.size();) {
      std::size_t end = begin + 1;
      while (end < refs.size() && refs[end].handle == refs[begin].handle) ++end;
      blocks.push_back({begin, end});
      begin = end;
    }
    const auto same_fair_owner = [](const Ref& a, const Ref& b) {
      const auto& lhs = a.execution->owner;
      const auto& rhs = b.execution->owner;
      return std::tie(lhs.node_id, lhs.execution_id, lhs.generation) ==
             std::tie(rhs.node_id, rhs.execution_id, rhs.generation);
    };
    const auto boundary_mask = [](const Execution& execution) {
      unsigned mask = 0;
      for (std::size_t b = 0; b < 2; ++b) {
        const double tolerance = kCreditTolerance *
            std::max(1.0, execution.credit_capacity[b]);
        if (execution.credit[b] <= tolerance) mask |= 1u << (2 * b);
        if (execution.credit[b] >= execution.credit_capacity[b] - tolerance)
          mask |= 2u << (2 * b);
      }
      return mask;
    };
    const auto equivalent_blocks = [&](const Block& lhs, const Block& rhs) {
      if (lhs.end - lhs.begin != rhs.end - rhs.begin ||
          boundary_mask(*refs[lhs.begin].execution) !=
          boundary_mask(*refs[rhs.begin].execution)) return false;
      for (std::size_t i = 0; i < lhs.end - lhs.begin; ++i) {
        const auto& a = refs[lhs.begin + i].live->definition;
        const auto& b = refs[rhs.begin + i].live->definition;
        if (a.id != b.id || a.role != b.role || a.work != b.work ||
            a.canonical_q != b.canonical_q ||
            a.private_time_s != b.private_time_s ||
            a.credit_delta != b.credit_delta) return false;
      }
      return true;
    };
    std::vector<std::size_t> representatives;
    std::vector<bool> merged(refs.size(), false);
    for (std::size_t index = 0; index < blocks.size(); ++index) {
      const Block& block = blocks[index];
      bool split_owner = false;
      for (std::size_t other = 0; other < blocks.size(); ++other) {
        if (other != index && same_fair_owner(refs[block.begin],
                                            refs[blocks[other].begin])) {
          split_owner = true;
          break;
        }
      }
      if (split_owner) continue;
      const auto match = std::find_if(representatives.begin(), representatives.end(),
          [&](std::size_t candidate) {
            return equivalent_blocks(blocks[candidate], block);
          });
      if (match == representatives.end()) {
        representatives.push_back(index);
        continue;
      }
      const Block& representative_block = blocks[*match];
      for (std::size_t i = 0; i < block.end - block.begin; ++i) {
        Ref& representative = refs[representative_block.begin + i];
        if (representative.mirrors.empty())
          representative.mirrors.reserve(blocks.size() - 1);
        ++representative.multiplicity;
        representative.mirrors.push_back(refs[block.begin + i].live);
        merged[block.begin + i] = true;
      }
    }
    std::vector<Ref> reduced;
    reduced.reserve(refs.size());
    for (std::size_t i = 0; i < refs.size(); ++i)
      if (!merged[i]) reduced.push_back(std::move(refs[i]));
    refs = std::move(reduced);
  }
  const auto publish = [](Ref& ref, double rate, double alpha) {
    ref.live->rate = rate;
    ref.live->alpha = alpha;
    for (auto* mirror : ref.mirrors) {
      mirror->rate = rate;
      mirror->alpha = alpha;
    }
  };
  if (refs.size() == 1) {
    // The one-variable max-min LP has an exact closed solution: its sole
    // activity receives the smallest physical/private bound, unless a signed
    // credit boundary blocks that direction. This is the same feasible set
    // as the general solver, without allocating a simplex tableau.
    Ref& ref=refs.front();
    const auto& activity=ref.live->definition;
    double rate=std::numeric_limits<double>::infinity();
    for(std::size_t r=0;r<resource_count;++r)
      if(activity.work[r]>0.0)rate=std::min(rate,
          capacities_[r]/activity.work[r]/static_cast<double>(ref.multiplicity));
    if(activity.private_time_s>0.0)rate=std::min(rate,1.0/activity.private_time_s);
    for(std::size_t b=0;b<2;++b) {
      const double tolerance=kCreditTolerance*std::max(1.0,ref.execution->credit_capacity[b]);
      const double credit=ref.execution->credit[b];
      if((credit<=tolerance&&activity.credit_delta[b]<0.0)||
         (credit>=ref.execution->credit_capacity[b]-tolerance&&activity.credit_delta[b]>0.0))
        rate=0.0;
    }
    if(!std::isfinite(rate)||rate<0.0)
      throw std::runtime_error("numerical_failure: activity has no finite progress rate bound");
    publish(ref, rate, rate>0.0?1.0:0.0);
    return;
  }
  std::sort(refs.begin(), refs.end(), [](const Ref& lhs, const Ref& rhs) {
    return std::tie(lhs.first_resource, lhs.execution->owner.node_id,
                    lhs.execution->owner.execution_id, lhs.execution->owner.generation,
                    lhs.live->definition.id, lhs.handle) <
           std::tie(rhs.first_resource, rhs.execution->owner.node_id,
                    rhs.execution->owner.execution_id, rhs.execution->owner.generation,
                    rhs.live->definition.id, rhs.handle);
  });

  struct Constraint {
    std::vector<double> a;
    double b = 0.0;
  };
  const std::size_t count = refs.size();
  std::vector<Constraint> base;

  for (std::size_t resource = 0; resource < resource_count; ++resource) {
    Constraint c{std::vector<double>(count, 0.0), capacities_[resource]};
    bool used = false;
    for (std::size_t i = 0; i < count; ++i) {
      const double work = refs[i].live->definition.work[resource];
      if (work > 0.0) {
        c.a[i] = work * static_cast<double>(refs[i].multiplicity);
        used = true;
      }
    }
    if (used) base.push_back(std::move(c));
  }
  for (std::size_t i = 0; i < count; ++i) {
    const double private_time = refs[i].live->definition.private_time_s;
    if (private_time <= 0.0) continue;
    Constraint c{std::vector<double>(count, 0.0), 1.0 / private_time};
    c.a[i] = 1.0;
    base.push_back(std::move(c));
  }

  std::map<std::uint64_t, Execution*> active_executions;
  for (const Ref& ref : refs) active_executions.emplace(ref.handle, ref.execution);
  for (const auto& [execution_handle, execution] : active_executions) {
    (void)execution_handle;
    for (std::size_t buffer = 0; buffer < 2; ++buffer) {
      const double credit = execution->credit[buffer];
      const double capacity = execution->credit_capacity[buffer];
      const double tolerance = kCreditTolerance * std::max(1.0, capacity);
      const bool at_low = credit <= tolerance;
      const bool at_high = credit >= capacity - tolerance;
      if (!at_low && !at_high) continue;
      const auto add_boundary = [&](bool lower_boundary) {
        Constraint c{std::vector<double>(count, 0.0), 0.0};
        bool used = false;
        for (std::size_t i = 0; i < count; ++i) {
          if (refs[i].execution != execution) continue;
          const double delta = refs[i].live->definition.credit_delta[buffer];
          const double coefficient = lower_boundary ? -delta : delta;
          if (coefficient != 0.0) {
            c.a[i] = coefficient;
            used = true;
          }
        }
        if (used) base.push_back(std::move(c));
      };
      if (at_low) add_boundary(true);
      if (at_high) add_boundary(false);
    }
  }

  std::size_t pivot_count = 0;
  std::vector<bool> excluded(count, false);
  std::vector<bool> frozen(count, false);
  std::vector<double> solved(count, 0.0);
  auto failure = [&](const std::string& message) -> void {
    for (Ref& ref : refs) ref.live->rate = 0.0;
    throw std::runtime_error("numerical_failure: " + message);
  };
  auto solve_lp = [&](const std::vector<Constraint>& rows,
                      const std::vector<double>& objective) -> LpResult {
    std::vector<std::vector<double>> a;
    std::vector<double> b;
    a.reserve(rows.size());
    b.reserve(rows.size());
    for (const Constraint& row : rows) {
      a.push_back(row.a);
      b.push_back(row.b);
    }
    DenseSimplex solver(a, b, objective, pivot_count);
    return solver.solve();
  };

  std::size_t total_tiers = 0;
  for (std::size_t restart = 0; restart <= count; ++restart) {
    owner_q.clear();
    std::size_t participating = 0;
    for (std::size_t i = 0; i < count; ++i) {
      if (excluded[i]) continue;
      ++participating;
      owner_q[refs[i].owner_key] += refs[i].q;
    }
    if (participating == 0) break;
    for (const auto& [owner, total] : owner_q) {
      (void)owner;
      if (!(total > 0.0L) || !std::isfinite(total)) {
        failure("owner canonical weight is invalid");
      }
    }
    for (std::size_t i = 0; i < count; ++i) {
      if (excluded[i]) {
        refs[i].alpha = 0.0;
        frozen[i] = true;
        solved[i] = 0.0;
        continue;
      }
      const long double total = owner_q.at(refs[i].owner_key);
      refs[i].alpha = static_cast<double>(static_cast<long double>(refs[i].q) / total);
      if (!std::isfinite(refs[i].alpha) || refs[i].alpha <= 0.0) {
        failure("owner weight underflowed the simplex precision");
      }
      frozen[i] = false;
      solved[i] = 0.0;
    }

    double z_scale = std::numeric_limits<double>::infinity();
    std::vector<double> variable_scale(count, 1.0);
    bool has_positive_upper_bound = false;
    for (std::size_t i = 0; i < count; ++i) {
      if (excluded[i]) continue;
      double upper_x = std::numeric_limits<double>::infinity();
      for (std::size_t resource = 0; resource < resource_count; ++resource) {
        const double work = refs[i].live->definition.work[resource];
        if (work <= 0.0) continue;
        const double coefficient = work / refs[i].q *
            static_cast<double>(refs[i].multiplicity);
        if (!std::isfinite(coefficient) || coefficient <= 0.0) {
          failure("resource coefficient is outside simplex precision");
        }
        upper_x = std::min(upper_x, capacities_[resource] / coefficient);
      }
      const double private_time = refs[i].live->definition.private_time_s;
      if (private_time > 0.0) {
        upper_x = std::min(upper_x, refs[i].q / private_time);
      }
      if (!std::isfinite(upper_x) || upper_x < 0.0) {
        failure("activity has no finite canonical service bound");
      }
      if (upper_x > 0.0) {
        has_positive_upper_bound = true;
        const double activity_scale = upper_x / refs[i].alpha;
        if (!std::isfinite(activity_scale)) failure("fairness scale is nonfinite");
        z_scale = std::min(z_scale, activity_scale);
      }
    }
    if (!has_positive_upper_bound) z_scale = 1.0;
    if (!std::isfinite(z_scale) || z_scale < 0.0) {
      failure("fairness scale is nonfinite");
    }

    // x_i = Q_i*r_i and alpha_i = Q_i/owner_Q, so the same fairness
    // transform is r_i = (z_scale/owner_Q)*y_i and its tier is y_i >= z.
    // Assemble physical/private/credit rows in progress-rate coordinates
    // first. Computing (delta/Q_i)*(alpha_i*z_scale) separately leaves
    // unequal coefficients for one owner's equal signed credit deltas;
    // cancellation during simplex pivots can amplify that discrepancy.
    // One shared progress scale per owner keeps those equalities exact.
    for (std::size_t i = 0; i < count; ++i) {
      if (excluded[i]) continue;
      variable_scale[i] = static_cast<double>(
          static_cast<long double>(z_scale) / owner_q.at(refs[i].owner_key));
      if (!std::isfinite(variable_scale[i]) || variable_scale[i] <= 0.0) {
        failure("fairness variable scale is outside simplex precision");
      }
    }

    std::vector<Constraint> normalized_base = base;
    for (Constraint& row : normalized_base) {
      for (std::size_t i = 0; i < count; ++i) {
        row.a[i] *= variable_scale[i];
      }
    }

    bool restart_with_fewer_activities = false;
    double minimum_tier = 0.0;
    while (std::any_of(frozen.begin(), frozen.end(), [](bool value) { return !value; })) {
      if (++total_tiers > count * 2) failure("lexicographic max-min exceeded activity bound");
      std::vector<Constraint> rows = normalized_base;
      std::vector<std::size_t> minimum_rows(count, std::numeric_limits<std::size_t>::max());
      for (std::size_t i = 0; i < count; ++i) {
        if (excluded[i]) {
          Constraint upper{std::vector<double>(count + 1, 0.0), 0.0};
          upper.a[i] = 1.0;
          rows.push_back(std::move(upper));
        } else if (frozen[i]) {
          Constraint upper{std::vector<double>(count + 1, 0.0), solved[i]};
          upper.a[i] = 1.0;
          rows.push_back(std::move(upper));
          Constraint lower{std::vector<double>(count + 1, 0.0), -solved[i]};
          lower.a[i] = -1.0;
          rows.push_back(std::move(lower));
        } else {
          Constraint minimum{std::vector<double>(count + 1, 0.0), 0.0};
          minimum.a[i] = -1.0;
          minimum.a[count] = 1.0;
          minimum_rows[i] = rows.size();
          rows.push_back(std::move(minimum));
        }
      }

      // Base rows were built with count columns. Add the shared tier variable.
      for (std::size_t r = 0; r < normalized_base.size(); ++r) rows[r].a.resize(count + 1, 0.0);
      std::vector<double> objective(count + 1, 0.0);
      objective[count] = 1.0;
      const LpResult tier = solve_lp(rows, objective);
      if (tier.status == LpStatus::PivotBudget) failure("simplex exceeded global 256-pivot budget");
      if (tier.status != LpStatus::Optimal || tier.primal.size() != count + 1 ||
          !std::isfinite(tier.objective)) {
        failure("signed-credit max-min LP failed");
      }
      const double reported_z = std::max(0.0, tier.primal[count]);
      if (reported_z < minimum_tier -
              kLpEpsilon * std::max(1.0, minimum_tier)) {
        failure("lexicographic minimum decreased beyond simplex precision");
      }
      // A simplex-feasible point can still fall just below the prior tier
      // within its existing epsilon. Preserve the exact monotone minimum
      // when recording a new bottleneck; never relax the credit boundary.
      const double normalized_z = std::max(minimum_tier, reported_z);
      minimum_tier = normalized_z;
      const double z = normalized_z * z_scale;
      if (!std::isfinite(z)) failure("lexicographic objective overflowed");
      std::vector<std::size_t> freeze_indices;
      for (std::size_t i = 0; i < count; ++i) {
        if (excluded[i] || frozen[i]) continue;
        const std::size_t row = minimum_rows[i];
        if (row < tier.dual_rows.size() && tier.dual_rows[row] > 1.0e-8) {
          freeze_indices.push_back(i);
        }
      }

      if (freeze_indices.empty()) {
        // A degenerate optimal dual can hide a forced minimum. Check each
        // activity against the same feasible minimum face, in stable order.
        std::vector<Constraint> face = normalized_base;
        for (std::size_t i = 0; i < count; ++i) {
          if (excluded[i]) {
            Constraint upper{std::vector<double>(count, 0.0), 0.0};
            upper.a[i] = 1.0;
            face.push_back(std::move(upper));
          } else if (frozen[i]) {
            Constraint upper{std::vector<double>(count, 0.0), solved[i]};
            upper.a[i] = 1.0;
            face.push_back(std::move(upper));
            Constraint lower{std::vector<double>(count, 0.0), -solved[i]};
            lower.a[i] = -1.0;
            face.push_back(std::move(lower));
          } else {
            Constraint minimum{std::vector<double>(count, 0.0), -normalized_z};
            minimum.a[i] = -1.0;
            face.push_back(std::move(minimum));
          }
        }
        for (std::size_t i = 0; i < count; ++i) {
          if (excluded[i] || frozen[i]) continue;
          std::vector<double> maximize(count, 0.0);
          maximize[i] = 1.0;
          const LpResult individual = solve_lp(face, maximize);
          if (individual.status == LpStatus::PivotBudget) failure("simplex exceeded global 256-pivot budget");
          if (individual.status != LpStatus::Optimal || individual.primal.size() != count) {
            failure("lexicographic bottleneck identification LP failed");
          }
          const double floor = normalized_z;
          const double tolerance = 1.0e-8 * std::max(1.0e-12, std::abs(floor));
          if (individual.primal[i] <= floor + tolerance) freeze_indices.push_back(i);
        }
      }
      if (freeze_indices.empty()) failure("lexicographic max-min found no forced activity");
      if (z == 0.0) {
        bool removed = false;
        for (std::size_t i : freeze_indices) {
          if (!excluded[i]) {
            excluded[i] = true;
            frozen[i] = true;
            solved[i] = 0.0;
            removed = true;
          }
        }
        if (!removed) failure("zero-rate bottleneck could not be removed from fairness set");
        restart_with_fewer_activities = true;
        break;
      }
      for (std::size_t i : freeze_indices) {
        frozen[i] = true;
        solved[i] = normalized_z;
      }
    }
    if (restart_with_fewer_activities) continue;

    for (std::size_t i = 0; i < count; ++i) {
      if (excluded[i]) {
        publish(refs[i], 0.0, 0.0);
        continue;
      }
      if (!std::isfinite(solved[i]) || solved[i] < -1.0e-10) failure("nonfinite lexicographic solution");
      publish(refs[i], std::max(0.0, solved[i] * variable_scale[i]),
              refs[i].alpha);
    }
    return;
  }
  for (std::size_t i = 0; i < count; ++i) {
    publish(refs[i], 0.0, 0.0);
  }
}

void Runtime::integrate(long double delta_ns) {
  if (delta_ns <= 0.0L) return;
  const long double whole = std::floor(delta_ns);
  if (!std::isfinite(whole) || whole < 0.0L || whole >= std::ldexp(1.0L, 64))
    throw std::overflow_error("overflow: integrated time exceeds uint64 nanoseconds");
  const auto integer_delta = static_cast<std::uint64_t>(whole);
  const long double fractional = exact_ns_ + (delta_ns - whole);
  const auto carry = static_cast<std::uint64_t>(std::floor(fractional));
  if (integer_delta > std::numeric_limits<std::uint64_t>::max() - exact_origin_ns_ ||
      carry > std::numeric_limits<std::uint64_t>::max() - exact_origin_ns_ - integer_delta)
    throw std::overflow_error("overflow: integrated time exceeds uint64 nanoseconds");
  const auto next_origin = exact_origin_ns_ + integer_delta + carry;
  const long double next_fraction = fractional - carry;
  if (next_origin == std::numeric_limits<std::uint64_t>::max() &&
      next_fraction > kTimeEpsilonNs)
    throw std::overflow_error("overflow: fractional event exceeds uint64 nanoseconds");
  const long double seconds = delta_ns / kNsPerSecond;
  std::array<std::set<std::string>, 2> row_owners;
  bool compute_busy = false;
  bool hbm_busy = false;
  bool any_row_shared = false;
  std::array<bool, 2> row_busy{};

  for (auto& [handle, execution] : executions_) {
    (void)handle;
    if (execution.terminal || time_until(execution.ready_ns) > kTimeEpsilonNs) continue;
    ServedCounters& categorized = served_by_category_[
        static_cast<std::size_t>(execution.served_category)];
    for (LiveActivity& live : execution.activities) {
      if (live.epoch_index != execution.epoch || live.done || live.rate <= 0.0) continue;
      long double delta_progress = static_cast<long double>(live.rate) * seconds;
      if (delta_progress <= 0.0L) continue;
      const long double remaining = 1.0L - static_cast<long double>(live.progress);
      if (delta_progress >= remaining - 1.0e-14L) delta_progress = remaining;
      if (delta_progress <= 0.0L) continue;
      const double next_progress = std::min(1.0, live.progress + static_cast<double>(delta_progress));
      delta_progress = static_cast<long double>(next_progress) - live.progress;
      live.progress = next_progress;

      for (std::size_t row = 0; row < 2; ++row) {
        for (std::size_t direction = 0; direction < 2; ++direction) {
          const auto bytes = live.definition.hbm_bytes[row][direction];
          const long double served = static_cast<long double>(bytes) * delta_progress;
          live.hbm_served[row][direction] += served;
          execution.hbm_served[row][direction] += served;
          hbm_served_[row][direction] += served;
          categorized.hbm_bytes[row][direction] += served;
        }
      }
      const long double served_int4 = static_cast<long double>(live.definition.int4_ops) * delta_progress;
      const long double served_vec = planned_vec_eqops(live.definition) * delta_progress;
      live.int4_served += served_int4;
      live.vec_served += served_vec;
      execution.int4_served += served_int4;
      execution.vec_served += served_vec;
      categorized.int4_ops += served_int4;
      categorized.vec_eqops += served_vec;

      for (std::size_t resource = 0; resource < resource_count; ++resource) {
        if (live.definition.work[resource] <= 0.0) continue;
        const long double service_rate = static_cast<long double>(live.definition.work[resource]) * live.rate;
        if (service_rate <= 0.0L) continue;
        if (resource == COMPUTE_GATE) compute_busy = true;
        if (resource == HBM0) {
          hbm_busy = true;
          row_busy[0] = true;
          row_owners[0].insert(execution.owner.key());
        } else if (resource == HBM1) {
          hbm_busy = true;
          row_busy[1] = true;
          row_owners[1].insert(execution.owner.key());
        }
      }
    }
    recompute_credit(execution);
  }
  if (compute_busy) compute_busy_ns_ += delta_ns;
  if (hbm_busy) hbm_busy_ns_ += delta_ns;
  for (std::size_t row = 0; row < 2; ++row) {
    if (row_busy[row]) row_busy_ns_[row] += delta_ns;
    if (row_owners[row].size() >= 2) {
      any_row_shared = true;
      row_shared_ns_[row] += delta_ns;
    }
  }
  if (any_row_shared) hbm_shared_ns_ += delta_ns;
  exact_origin_ns_ = next_origin;
  exact_ns_ = next_fraction;
  now_ns_ = ceil_exact_time();
}

long double Runtime::time_until(std::uint64_t absolute_ns) const {
  const long double integer_delta = absolute_ns >= exact_origin_ns_
      ? static_cast<long double>(absolute_ns - exact_origin_ns_)
      : -static_cast<long double>(exact_origin_ns_ - absolute_ns);
  return integer_delta - exact_ns_;
}

std::uint64_t Runtime::ceil_exact_time() const {
  if (exact_ns_ <= kTimeEpsilonNs) return exact_origin_ns_;
  if (exact_origin_ns_ == std::numeric_limits<std::uint64_t>::max())
    throw std::overflow_error("overflow: fractional event exceeds uint64 nanoseconds");
  return exact_origin_ns_ + 1;
}

void Runtime::set_exact_time(std::uint64_t absolute_ns) {
  exact_origin_ns_ = absolute_ns;
  exact_ns_ = 0.0L;
  now_ns_ = absolute_ns;
}

void recompute_credit(Execution& execution) {
  if (execution.epoch >= execution.plan->epochs.size()) return;
  const Epoch& epoch = execution.plan->epochs[execution.epoch];
  std::array<long double, 2> credit{epoch.credit_initial[0], epoch.credit_initial[1]};
  for (const LiveActivity& live : execution.activities) {
    if (live.epoch_index != execution.epoch) continue;
    for (std::size_t buffer = 0; buffer < 2; ++buffer) {
      credit[buffer] += static_cast<long double>(live.definition.credit_delta[buffer]) *
                        static_cast<long double>(live.progress);
    }
  }
  execution.credit_capacity = epoch.credit_capacity;
  for (std::size_t buffer = 0; buffer < 2; ++buffer) {
    const long double capacity = epoch.credit_capacity[buffer];
    const long double tolerance = static_cast<long double>(kCreditTolerance) *
                                  std::max(1.0L, capacity);
    if (!std::isfinite(credit[buffer]) || credit[buffer] < -tolerance ||
        credit[buffer] > capacity + tolerance) {
      throw std::runtime_error("numerical_failure: credit crossed a finite buffer boundary");
    }
    if (std::abs(credit[buffer]) <= tolerance) credit[buffer] = 0.0L;
    if (std::abs(credit[buffer] - capacity) <= tolerance) credit[buffer] = capacity;
    execution.credit[buffer] = static_cast<double>(credit[buffer]);
  }
}

void Runtime::advance_to(std::uint64_t absolute_ns) {
  if (absolute_ns < now_ns_) throw std::invalid_argument("invalid_time: advance_to moved backwards");
  if (absolute_ns == now_ns_ && time_until(absolute_ns) <= kTimeEpsilonNs) return;
  std::uint64_t events = 0;
  while (time_until(absolute_ns) > kTimeEpsilonNs) {
    long double next_delta = std::numeric_limits<long double>::infinity();
    for (const auto& [handle, execution] : executions_) {
      (void)handle;
      if (execution.terminal) continue;
      if (time_until(execution.ready_ns) > kTimeEpsilonNs) {
        next_delta = std::min(next_delta, time_until(execution.ready_ns));
        continue;
      }
      for (const LiveActivity& live : execution.activities) {
        if (live.epoch_index != execution.epoch || live.done || live.rate <= 0.0) continue;
        const long double remain = std::max(0.0L, 1.0L - static_cast<long double>(live.progress));
        if (remain > 0.0L) {
          next_delta = min_positive(next_delta, remain / live.rate * kNsPerSecond);
        }
      }
      long double credit_rate[2]{0.0L, 0.0L};
      for (const LiveActivity& live : execution.activities) {
        if (live.epoch_index != execution.epoch || live.done || live.rate <= 0.0) continue;
        for (std::size_t buffer = 0; buffer < 2; ++buffer) {
          credit_rate[buffer] += static_cast<long double>(live.definition.credit_delta[buffer]) * live.rate;
        }
      }
      for (std::size_t buffer = 0; buffer < 2; ++buffer) {
        const long double rate = credit_rate[buffer];
        const long double credit = execution.credit[buffer];
        const long double capacity = execution.credit_capacity[buffer];
        if (rate > 0.0L && credit < capacity - kCreditTolerance) {
          next_delta = min_positive(next_delta, (capacity - credit) / rate * kNsPerSecond);
        } else if (rate < 0.0L && credit > kCreditTolerance) {
          next_delta = min_positive(next_delta, credit / (-rate) * kNsPerSecond);
        }
      }
    }

    const long double remaining_to_target = time_until(absolute_ns);
    if (!std::isfinite(next_delta) || next_delta > remaining_to_target + kTimeEpsilonNs) {
      struct SavedActivityState {
        std::size_t epoch = 0;
        bool terminal = false;
        std::vector<bool> done;
      };
      std::map<std::uint64_t, SavedActivityState> before_settle;
      for (const auto& [handle, execution] : executions_) {
        SavedActivityState state;
        state.epoch = execution.epoch;
        state.terminal = execution.terminal;
        for (const LiveActivity& live : execution.activities) state.done.push_back(live.done);
        before_settle.emplace(handle, std::move(state));
      }
      integrate(remaining_to_target);
      set_exact_time(absolute_ns);
      settle();
      bool membership_changed = false;
      for (const auto& [handle, execution] : executions_) {
        const auto before = before_settle.find(handle);
        if (before == before_settle.end() || before->second.epoch != execution.epoch ||
            before->second.terminal != execution.terminal ||
            before->second.done.size() != execution.activities.size()) {
          membership_changed = true;
          break;
        }
        for (std::size_t i = 0; i < execution.activities.size(); ++i) {
          if (before->second.done[i] != execution.activities[i].done) {
            membership_changed = true;
            break;
          }
        }
        if (membership_changed) break;
      }
      if (membership_changed) allocate();
      break;
    }
    if (next_delta <= kTimeEpsilonNs) {
      throw std::runtime_error("numerical_failure: nonpositive internal event interval");
    }
    integrate(next_delta);
    ++events;
    ++internal_events_;
    if (events > kMaxAdvanceEvents) {
      throw std::runtime_error("numerical_failure: advance_to exceeded event budget");
    }
    settle();
    allocate();
  }
  set_exact_time(absolute_ns);
}

void Runtime::cancel(std::uint64_t handle, std::uint64_t now_ns) {
  if (now_ns < now_ns_) throw std::invalid_argument("invalid_time: cancel time moved backwards");
  if (executions_.count(handle) == 0) throw std::out_of_range("stale_handle: unknown execution handle");
  advance_to(now_ns);
  auto found = executions_.find(handle);
  if (found == executions_.end()) throw std::out_of_range("stale_handle: unknown execution handle");
  Execution& execution = found->second;
  if (execution.terminal) return;
  execution.terminal = true;
  execution.cancelled = true;
  execution.finish_ns = now_ns;
  for (LiveActivity& live : execution.activities) live.rate = 0.0;
  ++internal_events_;
  settle();
  allocate();
}

void Runtime::update_capacity(std::size_t resource, double rate,
                              std::uint64_t now_ns) {
  if (resource >= resource_count) throw std::invalid_argument("invalid_resource: index outside [0,12]");
  if (!std::isfinite(rate) || rate < 0.0) throw std::invalid_argument("invalid_capacity: rate must be finite and nonnegative");
  if (rate > hardware_->rates[resource]) {
    throw std::invalid_argument("invalid_capacity: rate exceeds the hardware peak for this resource");
  }
  if (!resource_enabled(*hardware_, resource) && rate != 0.0) {
    throw std::invalid_argument("invalid_capacity: cannot enable a hardware-disabled resource");
  }
  if (now_ns < now_ns_) throw std::invalid_argument("invalid_time: capacity update moved backwards");
  advance_to(now_ns);
  capacities_[resource] = rate;
  ++internal_events_;
  allocate();
}

std::optional<std::uint64_t> Runtime::next_event() const {
  long double delta = std::numeric_limits<long double>::infinity();
  for (const auto& [handle, execution] : executions_) {
    (void)handle;
    if (execution.terminal) continue;
    if (time_until(execution.ready_ns) > kTimeEpsilonNs) {
      delta = std::min(delta, time_until(execution.ready_ns));
      continue;
    }
    for (const LiveActivity& live : execution.activities) {
      if (live.epoch_index != execution.epoch || live.done || live.rate <= 0.0) continue;
      const long double remain = std::max(0.0L, 1.0L - static_cast<long double>(live.progress));
      if (remain > 0.0L) delta = min_positive(delta, remain / live.rate * kNsPerSecond);
    }
    long double credit_rate[2]{0.0L, 0.0L};
    for (const LiveActivity& live : execution.activities) {
      if (live.epoch_index != execution.epoch || live.done || live.rate <= 0.0) continue;
      for (std::size_t buffer = 0; buffer < 2; ++buffer) {
        credit_rate[buffer] += static_cast<long double>(live.definition.credit_delta[buffer]) * live.rate;
      }
    }
    for (std::size_t buffer = 0; buffer < 2; ++buffer) {
      const long double rate = credit_rate[buffer];
      const long double credit = execution.credit[buffer];
      const long double capacity = execution.credit_capacity[buffer];
      if (rate > 0.0L && credit < capacity - kCreditTolerance) {
        delta = min_positive(delta, (capacity - credit) / rate * kNsPerSecond);
      } else if (rate < 0.0L && credit > kCreditTolerance) {
        delta = min_positive(delta, credit / (-rate) * kNsPerSecond);
      }
    }
  }
  if (!std::isfinite(delta)) return std::nullopt;
  long double rounded = std::ceil(delta - kTimeEpsilonNs);
  if (rounded < 1.0L) rounded = 1.0L;
  if (rounded >= std::ldexp(1.0L, 64))
    throw std::overflow_error("overflow: next event exceeds uint64 nanoseconds");
  const auto integer_delta = static_cast<std::uint64_t>(rounded);
  if (integer_delta > std::numeric_limits<std::uint64_t>::max() - now_ns_) {
    throw std::overflow_error("overflow: next event exceeds uint64 nanoseconds");
  }
  return now_ns_ + integer_delta;
}

nlohmann::json Runtime::query_progress(std::uint64_t handle) const {
  const auto found = executions_.find(handle);
  if (found == executions_.end()) throw std::out_of_range("stale_handle: unknown execution handle");
  const Execution& execution = found->second;
  nlohmann::json activities = nlohmann::json::array();
  for (const LiveActivity& live : execution.activities) {
    nlohmann::json remaining = nlohmann::json::array();
    nlohmann::json planned = nlohmann::json::array();
    nlohmann::json served_work = nlohmann::json::array();
    nlohmann::json activity_hbm_served = nlohmann::json::array();
    nlohmann::json activity_hbm_served_exact = nlohmann::json::array();
    for (std::size_t resource = 0; resource < resource_count; ++resource) {
      planned.push_back(live.definition.work[resource]);
      const double served = live.definition.work[resource] * live.progress;
      served_work.push_back(served);
      remaining.push_back(std::max(0.0, live.definition.work[resource] - served));
    }
    for (std::size_t row = 0; row < 2; ++row) {
      nlohmann::json directions = nlohmann::json::array();
      nlohmann::json exact_directions = nlohmann::json::array();
      for (std::size_t direction = 0; direction < 2; ++direction) {
        directions.push_back(static_cast<double>(live.hbm_served[row][direction]));
        exact_directions.push_back(decimal_measure(live.hbm_served[row][direction]));
      }
      activity_hbm_served.push_back(std::move(directions));
      activity_hbm_served_exact.push_back(std::move(exact_directions));
    }
    std::string state;
    if (live.done) state = "done";
    else if (execution.cancelled) state = "cancelled";
    else if (execution.terminal) state = "not_run";
    else if (live.epoch_index > execution.epoch || time_until(execution.ready_ns) > 0.0L) state = "blocked";
    else if (live.epoch_index < execution.epoch) state = "done";
    else if (live.rate > 0.0) state = "running";
    else state = "blocked";
    activities.push_back({{"activity_id", live.definition.id},
                          {"epoch_index", live.epoch_index},
                          {"role", role_name(live.definition.role)},
                          {"status", state},
                          {"progress", live.progress},
                          {"rate", live.rate},
                          {"rate_unit", "progress_per_second"},
                          {"alpha", live.alpha},
                          {"canonical_q", live.definition.canonical_q},
                          {"planned_work", planned},
                          {"served_work", served_work},
                          {"remaining_work", remaining},
                          {"hbm_payload_bytes", live.definition.hbm_bytes},
                          {"hbm_served_bytes", activity_hbm_served},
                          {"hbm_served_bytes_exact", activity_hbm_served_exact},
                          {"int4_ops", std::to_string(live.definition.int4_ops)},
                          {"vec_eqops", std::to_string(live.definition.vec_eqops)},
                          {"int4_served_ops", static_cast<double>(live.int4_served)},
                          {"int4_served_ops_exact", decimal_measure(live.int4_served)},
                          {"vec_served_eqops", static_cast<double>(live.vec_served)},
                          {"vec_served_eqops_exact", decimal_measure(live.vec_served)},
                          {"credit_delta", live.definition.credit_delta},
                          {"done", live.done}});
    if (live.definition.vec_eqops_fraction != 0.0) {
      activities.back()["vec_eqops"] = decimal_measure(planned_vec_eqops(live.definition));
      activities.back()["vec_eqops_whole"] = std::to_string(live.definition.vec_eqops);
      activities.back()["vec_eqops_fraction"] = equivalent_fraction_to_json(live.definition.vec_eqops_fraction);
    }
  }
  const std::string state = execution.cancelled ? "cancelled"
                           : execution.terminal ? "finished"
                           : time_until(execution.ready_ns) > 0.0L ? "queued"
                                                           : "running";
  const std::uint64_t elapsed = execution.terminal
      ? execution.finish_ns - execution.issue_ns
      : now_ns_ - execution.issue_ns;
  long double compute_gate_served = 0.0L;
  for (const LiveActivity& live : execution.activities) {
    compute_gate_served += static_cast<long double>(live.definition.work[COMPUTE_GATE]) *
                           static_cast<long double>(live.progress);
  }
  nlohmann::json hbm_served = nlohmann::json::array();
  nlohmann::json hbm_served_exact = nlohmann::json::array();
  for (std::size_t row = 0; row < 2; ++row) {
    nlohmann::json direction = nlohmann::json::array();
    nlohmann::json direction_exact = nlohmann::json::array();
    for (std::size_t kind = 0; kind < 2; ++kind) {
      direction.push_back(static_cast<double>(execution.hbm_served[row][kind]));
      direction_exact.push_back(decimal_measure(execution.hbm_served[row][kind]));
    }
    hbm_served.push_back(std::move(direction));
    hbm_served_exact.push_back(std::move(direction_exact));
  }
  nlohmann::json served = {
      {"hbm_bytes_by_row_direction", hbm_served},
      {"hbm_bytes_by_row_direction_exact", hbm_served_exact},
      {"int4_ops", static_cast<double>(execution.int4_served)},
      {"int4_ops_exact_decimal", decimal_measure(execution.int4_served)},
      {"vec_eqops", static_cast<double>(execution.vec_served)},
      {"vec_eqops_exact_decimal", decimal_measure(execution.vec_served)}};
  served["compute_gate_work"] = static_cast<double>(compute_gate_served);
  served["compute_gate_work_exact"] = decimal_measure(compute_gate_served);
  return {{"handle", execution.handle},
          {"owner", owner_json(execution.owner)},
          {"state", state},
          {"issue_ns", execution.issue_ns},
          {"ready_ns", execution.ready_ns},
          {"finish_ns", execution.terminal ? nlohmann::json(execution.finish_ns) : nlohmann::json(nullptr)},
          {"elapsed_ns", elapsed},
          {"elapsed_ns_exact", std::to_string(elapsed)},
          {"epoch_index", execution.epoch},
          {"generation", generation_},
          {"plan_fingerprint", execution.plan->fingerprint},
          {"hardware_fingerprint", execution.plan->hardware_fingerprint},
          {"credit", credit_json(execution.credit, execution.credit_capacity)},
          {"activities", std::move(activities)},
          {"served", std::move(served)},
          {"hbm_served_bytes", hbm_served},
          {"hbm_served_bytes_exact", hbm_served_exact},
          {"compute_gate_served_work", static_cast<double>(compute_gate_served)},
          {"compute_gate_served_work_exact", decimal_measure(compute_gate_served)},
          {"hbm_direction_order", {"read", "write"}},
          {"terminal", execution.terminal},
          {"cancelled", execution.cancelled}};
}

std::optional<std::uint64_t> Runtime::terminal_finish_ns(
    std::uint64_t handle) const {
  const auto found = executions_.find(handle);
  if (found == executions_.end()) {
    throw std::out_of_range("stale_handle: unknown execution handle");
  }
  if (!found->second.terminal) return std::nullopt;
  return found->second.finish_ns;
}

nlohmann::json Runtime::counters() const {
  std::set<std::string> active_owners;
  std::set<std::string> hbm_owners;
  std::set<std::string> row_owners[2];
  std::size_t active_activities = 0;
  for (const auto& [handle, execution] : executions_) {
    (void)handle;
    if (execution.terminal || time_until(execution.ready_ns) > 0.0L) continue;
    bool owner_active = false;
    bool owner_hbm = false;
    for (const LiveActivity& live : execution.activities) {
      if (live.epoch_index != execution.epoch || live.done || live.rate <= 0.0) continue;
      ++active_activities;
      owner_active = true;
      if (live.definition.work[HBM0] > 0.0) {
        row_owners[0].insert(execution.owner.key());
        owner_hbm = true;
      }
      if (live.definition.work[HBM1] > 0.0) {
        row_owners[1].insert(execution.owner.key());
        owner_hbm = true;
      }
    }
    if (owner_active) active_owners.insert(execution.owner.key());
    if (owner_hbm) hbm_owners.insert(execution.owner.key());
  }
  nlohmann::json hbm_served = nlohmann::json::array();
  nlohmann::json hbm_served_exact = nlohmann::json::array();
  for (std::size_t row = 0; row < 2; ++row) {
    nlohmann::json direction = nlohmann::json::array();
    nlohmann::json direction_exact = nlohmann::json::array();
    for (std::size_t kind = 0; kind < 2; ++kind) {
      direction.push_back(static_cast<double>(hbm_served_[row][kind]));
      direction_exact.push_back(decimal_measure(hbm_served_[row][kind]));
    }
    hbm_served.push_back(std::move(direction));
    hbm_served_exact.push_back(std::move(direction_exact));
  }
  nlohmann::json served_by_category = nlohmann::json::object();
  for (std::size_t index = 0; index < served_category_count; ++index) {
    const ServedCounters& item = served_by_category_[index];
    served_by_category[kServedCategoryNames[index]] = {
        {"hbm_served_bytes_by_row_direction",
         hbm_matrix_json(item.hbm_bytes, false)},
        {"hbm_served_bytes_by_row_direction_exact",
         hbm_matrix_json(item.hbm_bytes, true)},
        {"int4_ops", static_cast<double>(item.int4_ops)},
        {"int4_ops_exact_decimal", decimal_measure(item.int4_ops)},
        {"vec_eqops", static_cast<double>(item.vec_eqops)},
        {"vec_eqops_exact_decimal", decimal_measure(item.vec_eqops)}};
  }
  return {{"now_ns", now_ns_},
          {"capacity", resource_array_json(capacities_)},
          {"allocation_updates", allocation_updates_},
          {"internal_events", internal_events_},
          {"active_activities", active_activities},
          {"active_owners", active_owners.size()},
          {"active_hbm_owners", hbm_owners.size()},
          {"hbm_busy_ns", static_cast<double>(hbm_busy_ns_)},
          {"hbm_busy_ns_exact", decimal_measure(hbm_busy_ns_)},
          {"hbm_shared_ns", static_cast<double>(hbm_shared_ns_)},
          {"hbm_shared_ns_exact", decimal_measure(hbm_shared_ns_)},
          {"compute_busy_ns", static_cast<double>(compute_busy_ns_)},
          {"compute_busy_ns_exact", decimal_measure(compute_busy_ns_)},
          {"row_busy_ns", {static_cast<double>(row_busy_ns_[0]), static_cast<double>(row_busy_ns_[1])}},
          {"row_busy_ns_exact", {decimal_measure(row_busy_ns_[0]), decimal_measure(row_busy_ns_[1])}},
          {"row_shared_ns", {static_cast<double>(row_shared_ns_[0]), static_cast<double>(row_shared_ns_[1])}},
          {"row_shared_ns_exact", {decimal_measure(row_shared_ns_[0]), decimal_measure(row_shared_ns_[1])}},
          {"row_shared_owners", {row_owners[0].size(), row_owners[1].size()}},
          {"hbm_served_bytes_by_row_direction", hbm_served},
          {"hbm_served_bytes_by_row_direction_exact", hbm_served_exact},
          {"hbm_direction_order", {"read", "write"}},
          {"hardware_raw", hardware_raw_denominators(*hardware_)},
          {"served_by_category", std::move(served_by_category)},
          {"accounting_history_complete", accounting_history_complete_},
          {"typed_totals_history_complete", accounting_history_complete_},
          {"served_category_totals_semantics",
           accounting_history_complete_
               ? "complete_lifetime_totals"
               : "known_lower_bound_history_incomplete"},
          {"legacy_unclassified_hbm_bytes_by_row_direction",
           hbm_matrix_json(legacy_unclassified_hbm_, false)},
          {"legacy_unclassified_hbm_bytes_by_row_direction_exact",
           hbm_matrix_json(legacy_unclassified_hbm_, true)},
          {"legacy_unclassified_typed_totals_unknown",
           !accounting_history_complete_},
          {"endpoint_registry_size", endpoint_registry_.size()},
          {"transaction_registry_size", transaction_owners_.size()},
          {"runtime_generation", generation_}};
}

nlohmann::json Runtime::export_snapshot() const {
  nlohmann::json executions = nlohmann::json::array();
  for (const auto& [handle, execution] : executions_) {
    nlohmann::json activities = nlohmann::json::array();
    for (const LiveActivity& live : execution.activities) {
      nlohmann::json activity_hbm = nlohmann::json::array();
      for (std::size_t row = 0; row < 2; ++row) {
        nlohmann::json directions = nlohmann::json::array();
        for (std::size_t direction = 0; direction < 2; ++direction) {
          directions.push_back(long_double_text(live.hbm_served[row][direction]));
        }
        activity_hbm.push_back(std::move(directions));
      }
      activities.push_back({{"activity_id", live.definition.id},
                            {"epoch_index", live.epoch_index},
                            {"progress", live.progress},
                            {"rate", live.rate},
                            {"alpha", live.alpha},
                            {"done", live.done},
                            {"hbm_served", std::move(activity_hbm)},
                            {"int4_served", long_double_text(live.int4_served)},
                            {"vec_served", long_double_text(live.vec_served)}});
    }
    nlohmann::json hbm_served = nlohmann::json::array();
    for (std::size_t row = 0; row < 2; ++row) {
      nlohmann::json direction = nlohmann::json::array();
      for (std::size_t kind = 0; kind < 2; ++kind) {
      direction.push_back(long_double_text(execution.hbm_served[row][kind]));
      }
      hbm_served.push_back(std::move(direction));
    }
    executions.push_back({{"handle", handle},
                          {"issue_ns", execution.issue_ns},
                          {"ready_ns", execution.ready_ns},
                          {"finish_ns", execution.terminal ? nlohmann::json(execution.finish_ns) : nlohmann::json(nullptr)},
                          {"owner", owner_json(execution.owner)},
                          {"plan", plan_to_json(*execution.plan)},
                          {"epoch_index", execution.epoch},
                          {"credit", execution.credit},
                          {"credit_capacity", execution.credit_capacity},
                          {"activities", std::move(activities)},
                          {"hbm_served", std::move(hbm_served)},
                          {"int4_served", long_double_text(execution.int4_served)},
                          {"vec_served", long_double_text(execution.vec_served)},
                          {"terminal", execution.terminal},
                          {"cancelled", execution.cancelled}});
  }
  nlohmann::json endpoint_registry = nlohmann::json::array();
  for (const auto& [endpoint_id, transaction_id] : endpoint_registry_) {
    endpoint_registry.push_back({{"endpoint_id", endpoint_id}, {"transaction_id", transaction_id}});
  }
  nlohmann::json transactions = nlohmann::json::array();
  for (const auto& [transaction_id, record] : transaction_owners_) {
    transactions.push_back({{"transaction_id", transaction_id},
                            {"owner", owner_json(record.first)},
                            {"refcount", record.second}});
  }
  nlohmann::json global_hbm_served = nlohmann::json::array();
  for (std::size_t row = 0; row < 2; ++row) {
    nlohmann::json directions = nlohmann::json::array();
    for (std::size_t direction = 0; direction < 2; ++direction) {
      directions.push_back(long_double_text(hbm_served_[row][direction]));
    }
    global_hbm_served.push_back(std::move(directions));
  }
  return {{"schema", "coarse_die_snapshot_v3"},
          {"snapshot_version", 3},
          {"hardware_fingerprint", hardware_->fingerprint},
          {"now_ns", now_ns_},
          // Public calls end at an exact integer tick. Legacy fractional v2/v3
          // imports retain the precision their original absolute scalar had.
          {"exact_ns", long_double_text(static_cast<long double>(exact_origin_ns_) + exact_ns_)},
          {"next_handle", next_handle_},
          {"runtime_generation", generation_},
          {"allocation_updates", allocation_updates_},
          {"internal_events", internal_events_},
          {"capacities", resource_array_json(capacities_)},
          {"stats", {{"hbm_busy_ns", long_double_text(hbm_busy_ns_)},
                      {"hbm_shared_ns", long_double_text(hbm_shared_ns_)},
                      {"compute_busy_ns", long_double_text(compute_busy_ns_)},
                      {"row_busy_ns", {long_double_text(row_busy_ns_[0]), long_double_text(row_busy_ns_[1])}},
                      {"row_shared_ns", {long_double_text(row_shared_ns_[0]), long_double_text(row_shared_ns_[1])}},
                      {"hbm_served", global_hbm_served},
                      {"served_by_category",
                       served_categories_snapshot_json(served_by_category_)},
                      {"legacy_unclassified_hbm",
                       hbm_matrix_json(legacy_unclassified_hbm_, true)},
                      {"accounting_history_complete",
                       accounting_history_complete_},
                      {"typed_totals_history_complete",
                       accounting_history_complete_}}},
          {"executions", std::move(executions)},
          {"endpoint_registry", std::move(endpoint_registry)},
          {"transaction_owners", std::move(transactions)},
          {"event_order", "completion_credit_cancel_arrival_v2"},
          {"rounding", "exact_subns_internal_ceil_public_ns_v2"}};
}

std::unique_ptr<Runtime> Runtime::import_snapshot(const Hardware& hardware,
                                                  const nlohmann::json& snapshot) {
  validate_hardware(hardware);
  const bool legacy_v2_snapshot = snapshot.is_object() &&
      snapshot.value("schema", std::string()) == "coarse_die_snapshot_v2" &&
      snapshot.value("snapshot_version", 0U) == 2U;
  const bool categorized_v3_snapshot = snapshot.is_object() &&
      snapshot.value("schema", std::string()) == "coarse_die_snapshot_v3" &&
      snapshot.value("snapshot_version", 0U) == 3U;
  if (!legacy_v2_snapshot && !categorized_v3_snapshot) {
    throw std::invalid_argument(
        "incompatible_schema: expected coarse_die_snapshot_v3 or legacy coarse_die_snapshot_v2");
  }
  if (snapshot.value("hardware_fingerprint", std::string()) != hardware.fingerprint) {
    throw std::invalid_argument("incompatible_fingerprint: snapshot hardware fingerprint mismatch");
  }
  if (snapshot.value("event_order", std::string()) != "completion_credit_cancel_arrival_v2" ||
      snapshot.value("rounding", std::string()) != "exact_subns_internal_ceil_public_ns_v2") {
    throw std::invalid_argument("incompatible_schema: event-order or rounding version mismatch");
  }

  auto runtime = std::make_unique<Runtime>(hardware);
  runtime->now_ns_ = parse_u64(snapshot.at("now_ns"), "snapshot.now_ns");
  auto parse_long_double = [](const nlohmann::json& value, const char* field) {
    if (!value.is_string()) throw std::invalid_argument(std::string("invalid_snapshot: ") + field);
    const std::string text = value.get<std::string>();
    char* end = nullptr;
    const long double parsed = std::strtold(text.c_str(), &end);
    if (end != text.c_str() + text.size() || !std::isfinite(parsed) || parsed < 0.0L) {
      throw std::invalid_argument(std::string("invalid_snapshot: ") + field);
    }
    return parsed;
  };
  const long double legacy_exact = parse_long_double(snapshot.at("exact_ns"), "exact_ns");
  if (legacy_exact >= std::ldexp(1.0L, 64))
    throw std::invalid_argument("invalid_snapshot: exact_ns exceeds uint64 nanoseconds");
  const long double whole_exact = std::floor(legacy_exact);
  runtime->exact_origin_ns_ = static_cast<std::uint64_t>(whole_exact);
  runtime->exact_ns_ = legacy_exact - whole_exact;
  if (runtime->time_until(runtime->now_ns_) < -kTimeEpsilonNs ||
      runtime->ceil_exact_time() != runtime->now_ns_) {
    throw std::invalid_argument("invalid_snapshot: exact_ns is inconsistent with public now_ns");
  }
  runtime->next_handle_ = parse_u64(snapshot.at("next_handle"), "snapshot.next_handle");
  runtime->generation_ = parse_u64(snapshot.at("runtime_generation"), "snapshot.runtime_generation");
  runtime->allocation_updates_ = parse_u64(snapshot.at("allocation_updates"), "snapshot.allocation_updates");
  runtime->internal_events_ = parse_u64(snapshot.at("internal_events"), "snapshot.internal_events");
  runtime->capacities_ = json_resource_array(snapshot.at("capacities"), "capacities");
  for (std::size_t i = 0; i < resource_count; ++i) {
    if (!resource_enabled(hardware, i) && runtime->capacities_[i] != 0.0) {
      throw std::invalid_argument("invalid_snapshot: disabled resource has nonzero capacity");
    }
    if (runtime->capacities_[i] > hardware.rates[i]) {
      throw std::invalid_argument("invalid_snapshot: capacity exceeds the hardware peak");
    }
  }
  const auto& stats = snapshot.at("stats");
  runtime->hbm_busy_ns_ = parse_long_double(stats.at("hbm_busy_ns"), "stats.hbm_busy_ns");
  runtime->hbm_shared_ns_ = parse_long_double(stats.at("hbm_shared_ns"), "stats.hbm_shared_ns");
  runtime->compute_busy_ns_ = parse_long_double(stats.at("compute_busy_ns"), "stats.compute_busy_ns");
  const auto& row_busy = stats.at("row_busy_ns");
  if (!row_busy.is_array() || row_busy.size() != 2) throw std::invalid_argument("invalid_snapshot: row_busy_ns");
  for (std::size_t row = 0; row < 2; ++row) {
    runtime->row_busy_ns_[row] = parse_long_double(row_busy.at(row), "stats.row_busy_ns");
  }
  const auto& row_shared = stats.at("row_shared_ns");
  if (!row_shared.is_array() || row_shared.size() != 2) throw std::invalid_argument("invalid_snapshot: row_shared_ns");
  for (std::size_t row = 0; row < 2; ++row) {
    runtime->row_shared_ns_[row] = parse_long_double(row_shared.at(row), "stats.row_shared_ns");
  }
  const auto& global_served = stats.at("hbm_served");
  if (!global_served.is_array() || global_served.size() != 2) throw std::invalid_argument("invalid_snapshot: hbm_served");
  for (std::size_t row = 0; row < 2; ++row) {
    if (!global_served.at(row).is_array() || global_served.at(row).size() != 2) {
      throw std::invalid_argument("invalid_snapshot: hbm_served row shape");
    }
    for (std::size_t direction = 0; direction < 2; ++direction) {
      runtime->hbm_served_[row][direction] = parse_long_double(global_served.at(row).at(direction), "stats.hbm_served");
    }
  }

  auto parse_hbm_matrix = [&](const nlohmann::json& value,
                              const std::string& field) {
    std::array<std::array<long double, 2>, 2> matrix{};
    if (!value.is_array() || value.size() != 2) {
      throw std::invalid_argument("invalid_snapshot: " + field + " must have two rows");
    }
    for (std::size_t row = 0; row < 2; ++row) {
      if (!value.at(row).is_array() || value.at(row).size() != 2) {
        throw std::invalid_argument("invalid_snapshot: " + field + " row shape");
      }
      for (std::size_t direction = 0; direction < 2; ++direction) {
        matrix[row][direction] = parse_long_double(
            value.at(row).at(direction), field.c_str());
      }
    }
    return matrix;
  };

  if (categorized_v3_snapshot) {
    const auto& categorized = stats.at("served_by_category");
    if (!categorized.is_object()) {
      throw std::invalid_argument(
          "invalid_snapshot: served_by_category must be an object");
    }
    for (std::size_t index = 0; index < served_category_count; ++index) {
      const auto& item = categorized.at(kServedCategoryNames[index]);
      ServedCounters& target = runtime->served_by_category_[index];
      target.hbm_bytes = parse_hbm_matrix(
          item.at("hbm_served_bytes_by_row_direction_exact"),
          std::string("stats.served_by_category.") +
              kServedCategoryNames[index] + ".hbm_served_bytes_by_row_direction_exact");
      target.int4_ops = parse_long_double(
          item.at("int4_ops_exact_decimal"), "stats.served_by_category.int4_ops");
      target.vec_eqops = parse_long_double(
          item.at("vec_eqops_exact_decimal"), "stats.served_by_category.vec_eqops");
      if (index != static_cast<std::size_t>(ServedCategory::Comp) &&
          (target.int4_ops != 0.0L || target.vec_eqops != 0.0L)) {
        throw std::invalid_argument(
            "invalid_snapshot: endpoint category carries COMP operations");
      }
    }
    runtime->legacy_unclassified_hbm_ = parse_hbm_matrix(
        stats.at("legacy_unclassified_hbm"),
        "stats.legacy_unclassified_hbm");
    if (!stats.at("accounting_history_complete").is_boolean() ||
        !stats.at("typed_totals_history_complete").is_boolean()) {
      throw std::invalid_argument(
          "invalid_snapshot: accounting completeness markers must be boolean");
    }
    runtime->accounting_history_complete_ =
        stats.at("accounting_history_complete").get<bool>();
    if (stats.at("typed_totals_history_complete").get<bool>() !=
        runtime->accounting_history_complete_) {
      throw std::invalid_argument(
          "invalid_snapshot: typed and overall accounting completeness differ");
    }
  }

  const auto& execution_values = snapshot.at("executions");
  if (!execution_values.is_array()) throw std::invalid_argument("invalid_snapshot: executions must be array");
  for (const auto& value : execution_values) {
    const std::uint64_t handle = parse_u64(value.at("handle"), "execution.handle");
    if (handle == 0 || runtime->executions_.count(handle) != 0) {
      throw std::invalid_argument("invalid_snapshot: duplicate or zero execution handle");
    }
    auto plan = plan_from_json(value.at("plan"));
    if (!plan) throw std::invalid_argument("invalid_snapshot: plan could not be decoded");
    validate_plan(hardware, *plan);
    Execution execution;
    execution.handle = handle;
    execution.issue_ns = parse_u64(value.at("issue_ns"), "execution.issue_ns");
    execution.ready_ns = parse_u64(value.at("ready_ns"), "execution.ready_ns");
    execution.finish_ns = value.at("finish_ns").is_null() ? 0 : parse_u64(value.at("finish_ns"), "execution.finish_ns");
    execution.owner = parse_owner(value.at("owner"));
    execution.plan = std::move(plan);
    execution.served_category = plan_served_category(*execution.plan);
    execution.epoch = value.at("epoch_index").get<std::size_t>();
    execution.terminal = value.at("terminal").get<bool>();
    execution.cancelled = value.at("cancelled").get<bool>();
    if (execution.issue_ns > runtime->now_ns_ || execution.ready_ns < execution.issue_ns ||
        execution.ready_ns - execution.issue_ns != execution.plan->launch_ns ||
        execution.epoch > execution.plan->epochs.size() ||
        (execution.terminal && (execution.finish_ns < execution.issue_ns || execution.finish_ns > runtime->now_ns_)) ||
        (execution.terminal != !value.at("finish_ns").is_null()) ||
        (execution.cancelled && !execution.terminal)) {
      throw std::invalid_argument("invalid_snapshot: execution lifecycle times are inconsistent");
    }
    if (!execution.terminal && execution.epoch >= execution.plan->epochs.size()) {
      throw std::invalid_argument("invalid_snapshot: nonterminal execution has no remaining epoch");
    }
    for (std::size_t epoch = 0; epoch < execution.plan->epochs.size(); ++epoch) {
      for (const Activity& definition : execution.plan->epochs[epoch].activities) {
        execution.activities.push_back(LiveActivity{definition, epoch, 0.0, 0.0, 0.0, false});
      }
    }
    const auto& activity_states = value.at("activities");
    if (!activity_states.is_array() || activity_states.size() != execution.activities.size()) {
      throw std::invalid_argument("invalid_snapshot: activity state count does not match plan");
    }
    for (std::size_t i = 0; i < execution.activities.size(); ++i) {
      const auto& state = activity_states.at(i);
      LiveActivity& live = execution.activities[i];
      if (parse_u64(state.at("activity_id"), "activity.activity_id") != live.definition.id ||
          state.at("epoch_index").get<std::size_t>() != live.epoch_index) {
        throw std::invalid_argument("invalid_snapshot: activity identity differs from plan");
      }
      live.progress = state.at("progress").get<double>();
      live.rate = parse_finite_nonnegative(state.at("rate"), "activity.rate");
      live.alpha = parse_finite_nonnegative(state.at("alpha"), "activity.alpha");
      live.done = state.at("done").get<bool>();
      if (!std::isfinite(live.progress) || live.progress < 0.0 || live.progress > 1.0 ||
          (live.done && live.progress != 1.0) ||
          (live.epoch_index < execution.epoch && !live.done) ||
          (live.epoch_index > execution.epoch && (live.done || live.progress != 0.0))) {
        throw std::invalid_argument("invalid_snapshot: activity progress/done state is inconsistent");
      }
      if (execution.terminal && !execution.cancelled && !live.done) {
        throw std::invalid_argument("invalid_snapshot: completed execution has unfinished activity");
      }
      if (execution.terminal && live.rate != 0.0) {
        throw std::invalid_argument("invalid_snapshot: terminal execution has nonzero activity rate");
      }
      const auto& activity_hbm = state.at("hbm_served");
      if (!activity_hbm.is_array() || activity_hbm.size() != 2) {
        throw std::invalid_argument("invalid_snapshot: activity hbm_served shape");
      }
      for (std::size_t row = 0; row < 2; ++row) {
        if (!activity_hbm.at(row).is_array() || activity_hbm.at(row).size() != 2) {
          throw std::invalid_argument("invalid_snapshot: activity hbm_served row shape");
        }
        for (std::size_t direction = 0; direction < 2; ++direction) {
          live.hbm_served[row][direction] = parse_long_double(
              activity_hbm.at(row).at(direction), "activity.hbm_served");
          const long double planned = live.definition.hbm_bytes[row][direction];
          const long double expected = live.done ? planned : planned * live.progress;
          const long double tolerance = 2.0e-9L * std::max(1.0L, planned);
          if (live.hbm_served[row][direction] > planned + tolerance ||
              std::abs(live.hbm_served[row][direction] - expected) > tolerance) {
            throw std::invalid_argument("invalid_snapshot: activity hbm_served differs from progress");
          }
        }
      }
      live.int4_served = parse_long_double(state.at("int4_served"), "activity.int4_served");
      live.vec_served = parse_long_double(state.at("vec_served"), "activity.vec_served");
      const long double expected_int4 = live.done ? live.definition.int4_ops
          : static_cast<long double>(live.definition.int4_ops) * live.progress;
      const long double expected_vec = live.done ? planned_vec_eqops(live.definition)
          : planned_vec_eqops(live.definition) * live.progress;
      if (std::abs(live.int4_served - expected_int4) > 2.0e-9L * std::max(1.0L, expected_int4) ||
          std::abs(live.vec_served - expected_vec) > 2.0e-9L * std::max(1.0L, expected_vec)) {
        throw std::invalid_argument("invalid_snapshot: activity operation counters differ from progress");
      }
    }
    if (!value.at("credit").is_array() || value.at("credit").size() != 2 ||
        !value.at("credit_capacity").is_array() || value.at("credit_capacity").size() != 2) {
      throw std::invalid_argument("invalid_snapshot: credit shape");
    }
    for (std::size_t i = 0; i < 2; ++i) {
      execution.credit[i] = parse_finite_nonnegative(value.at("credit").at(i), "execution.credit");
      execution.credit_capacity[i] = parse_finite_nonnegative(value.at("credit_capacity").at(i), "execution.credit_capacity");
    }
    const auto& hbm_served = value.at("hbm_served");
    if (!hbm_served.is_array() || hbm_served.size() != 2) throw std::invalid_argument("invalid_snapshot: execution hbm_served shape");
    for (std::size_t row = 0; row < 2; ++row) {
      if (!hbm_served.at(row).is_array() || hbm_served.at(row).size() != 2) {
        throw std::invalid_argument("invalid_snapshot: execution hbm_served row shape");
      }
      for (std::size_t direction = 0; direction < 2; ++direction) {
        execution.hbm_served[row][direction] = parse_long_double(hbm_served.at(row).at(direction), "execution.hbm_served");
      }
    }
    execution.int4_served = parse_long_double(value.at("int4_served"), "execution.int4_served");
    execution.vec_served = parse_long_double(value.at("vec_served"), "execution.vec_served");
    std::array<std::array<long double, 2>, 2> activity_hbm_sum{};
    long double activity_int4_sum = 0.0L;
    long double activity_vec_sum = 0.0L;
    for (const LiveActivity& live : execution.activities) {
      for (std::size_t row = 0; row < 2; ++row) {
        for (std::size_t direction = 0; direction < 2; ++direction) {
          activity_hbm_sum[row][direction] += live.hbm_served[row][direction];
        }
      }
      activity_int4_sum += live.int4_served;
      activity_vec_sum += live.vec_served;
    }
    for (std::size_t row = 0; row < 2; ++row) {
      for (std::size_t direction = 0; direction < 2; ++direction) {
        const long double scale = std::max(1.0L, activity_hbm_sum[row][direction]);
        if (std::abs(execution.hbm_served[row][direction] - activity_hbm_sum[row][direction]) >
            2.0e-9L * scale) {
          throw std::invalid_argument("invalid_snapshot: execution HBM total differs from activity totals");
        }
      }
    }
    if (std::abs(execution.int4_served - activity_int4_sum) >
            2.0e-9L * std::max(1.0L, activity_int4_sum) ||
        std::abs(execution.vec_served - activity_vec_sum) >
            2.0e-9L * std::max(1.0L, activity_vec_sum)) {
      throw std::invalid_argument("invalid_snapshot: execution operation totals differ from activity totals");
    }
    if (execution.epoch < execution.plan->epochs.size()) {
      const Epoch& current = execution.plan->epochs[execution.epoch];
      for (std::size_t i = 0; i < 2; ++i) {
        if (std::abs(execution.credit_capacity[i] - current.credit_capacity[i]) > kCreditTolerance ||
            execution.credit[i] > execution.credit_capacity[i] + kCreditTolerance) {
          throw std::invalid_argument("invalid_snapshot: current credit capacity differs from plan");
        }
      }
      Execution check = execution;
      recompute_credit(check);
      for (std::size_t i = 0; i < 2; ++i) {
        if (std::abs(check.credit[i] - execution.credit[i]) > kCreditTolerance * std::max(1.0, execution.credit_capacity[i])) {
          throw std::invalid_argument("invalid_snapshot: credit differs from activity progress");
        }
      }
    }
    runtime->executions_.emplace(handle, std::move(execution));
  }
  if (runtime->next_handle_ == 0 ||
      (!runtime->executions_.empty() && runtime->next_handle_ <= runtime->executions_.rbegin()->first)) {
    throw std::invalid_argument("invalid_snapshot: next_handle is not monotonic");
  }

  const auto& endpoint_values = snapshot.at("endpoint_registry");
  if (!endpoint_values.is_array()) throw std::invalid_argument("invalid_snapshot: endpoint_registry must be array");
  for (const auto& value : endpoint_values) {
    const auto id = parse_u64(value.at("endpoint_id"), "endpoint_registry.endpoint_id");
    const auto transaction = parse_u64(value.at("transaction_id"), "endpoint_registry.transaction_id");
    if (id == 0 || transaction == 0 || !runtime->endpoint_registry_.emplace(id, transaction).second) {
      throw std::invalid_argument("invalid_snapshot: duplicate or zero endpoint registry entry");
    }
  }
  const auto& transaction_values = snapshot.at("transaction_owners");
  if (!transaction_values.is_array()) throw std::invalid_argument("invalid_snapshot: transaction_owners must be array");
  for (const auto& value : transaction_values) {
    const auto id = parse_u64(value.at("transaction_id"), "transaction.transaction_id");
    const Owner owner = parse_owner(value.at("owner"));
    const auto refcount = value.at("refcount").get<std::size_t>();
    if (id == 0 || refcount == 0 || !runtime->transaction_owners_.emplace(id, std::make_pair(owner, refcount)).second) {
      throw std::invalid_argument("invalid_snapshot: duplicate or invalid transaction registry entry");
    }
  }
  std::map<std::uint64_t, std::size_t> expected_refs;
  std::map<std::uint64_t, Owner> expected_owners;
  std::map<std::uint64_t, std::uint64_t> expected_endpoints;
  for (const auto& [handle, execution] : runtime->executions_) {
    (void)handle;
    if (!execution.plan->task.is_object() ||
        execution.plan->task.value("schema", std::string()) != "coarse_die_endpoint_v2" ||
        !execution.plan->task.contains("endpoint")) continue;
    const auto& endpoint = execution.plan->task.at("endpoint");
    const EndpointSpec endpoint_spec = parse_endpoint(endpoint, hardware);
    if (execution.plan->launch_ns != endpoint_spec.initial_delay_ns ||
        execution.plan->epochs.size() != 1 ||
        execution.plan->epochs.front().activities.size() != 1) {
      throw std::invalid_argument("invalid_snapshot: endpoint plan structure is inconsistent");
    }
    const auto evidence = execution.plan->summary.find("allocation_evidence");
    // Prior v2 snapshots used floor(total/2) for an implicit uniform split.
    // Preserve their frozen work on import; new endpoint plans carry evidence
    // and use the design's row0 remainder rule.
    const bool legacy_uniform = evidence == execution.plan->summary.end();
    if (!legacy_uniform && (!evidence->is_string() ||
        evidence->get<std::string>() != (endpoint_spec.uniform_default
            ? "uniform_default" : "explicit_row_fraction"))) {
      throw std::invalid_argument("invalid_snapshot: endpoint allocation evidence differs from endpoint JSON");
    }
    const Activity expected_activity = endpoint_activity(endpoint_spec, legacy_uniform);
    const Activity& actual_activity = execution.plan->epochs.front().activities.front();
    if (actual_activity.id != expected_activity.id || actual_activity.role != Role::Endpoint ||
        actual_activity.hbm_bytes != expected_activity.hbm_bytes ||
        std::abs(actual_activity.canonical_q - expected_activity.canonical_q) >
            1.0e-10 * std::max(1.0, expected_activity.canonical_q)) {
      throw std::invalid_argument("invalid_snapshot: endpoint activity identity or payload differs from endpoint JSON");
    }
    for (std::size_t resource = 0; resource < resource_count; ++resource) {
      const double expected = expected_activity.work[resource];
      const double actual = actual_activity.work[resource];
      if (std::abs(actual - expected) > 1.0e-10 * std::max(1.0, expected)) {
        throw std::invalid_argument("invalid_snapshot: endpoint path work differs from endpoint JSON");
      }
    }
    const auto id = parse_u64(endpoint.at("endpoint_id"), "endpoint.endpoint_id");
    const auto transaction = parse_u64(endpoint.at("transaction_id"), "endpoint.transaction_id");
    if (runtime->endpoint_registry_.count(id) == 0 || runtime->endpoint_registry_.at(id) != transaction) {
      throw std::invalid_argument("invalid_snapshot: endpoint registry does not match live endpoint plan");
    }
    if (!expected_endpoints.emplace(id, transaction).second) {
      throw std::invalid_argument("invalid_snapshot: endpoint id belongs to multiple executions");
    }
    ++expected_refs[transaction];
    auto found_owner = expected_owners.find(transaction);
    if (found_owner != expected_owners.end() && !same_owner(found_owner->second, execution.owner)) {
      throw std::invalid_argument("invalid_snapshot: transaction spans different owners");
    }
    expected_owners[transaction] = execution.owner;
  }
  if (expected_refs.size() != runtime->transaction_owners_.size() ||
      expected_endpoints.size() != runtime->endpoint_registry_.size()) {
    throw std::invalid_argument("invalid_snapshot: endpoint/transaction registry cardinality mismatch");
  }
  for (const auto& [endpoint, transaction] : runtime->endpoint_registry_) {
    const auto expected = expected_endpoints.find(endpoint);
    if (expected == expected_endpoints.end() || expected->second != transaction) {
      throw std::invalid_argument("invalid_snapshot: endpoint registry contains an unowned id");
    }
  }
  for (const auto& [transaction, refs] : expected_refs) {
    const auto record = runtime->transaction_owners_.find(transaction);
    if (record == runtime->transaction_owners_.end() || record->second.second != refs ||
        !same_owner(record->second.first, expected_owners.at(transaction))) {
      throw std::invalid_argument("invalid_snapshot: transaction registry ownership/refcount mismatch");
    }
  }

  std::array<ServedCounters, served_category_count> retained_execution_totals{};
  for (const auto& [handle, execution] : runtime->executions_) {
    (void)handle;
    ServedCounters& target = retained_execution_totals[
        static_cast<std::size_t>(execution.served_category)];
    for (std::size_t row = 0; row < 2; ++row) {
      for (std::size_t direction = 0; direction < 2; ++direction) {
        target.hbm_bytes[row][direction] +=
            execution.hbm_served[row][direction];
      }
    }
    target.int4_ops += execution.int4_served;
    target.vec_eqops += execution.vec_served;
  }

  constexpr long double kCounterTolerance = 1.0e-10L;
  const auto close_enough = [](long double left, long double right) {
    return std::abs(left - right) <= kCounterTolerance *
        std::max({1.0L, std::abs(left), std::abs(right)});
  };
  if (legacy_v2_snapshot) {
    // v2 did not persist category totals. Recover only work still represented
    // by execution records; preserve the remainder of the old global HBM
    // counters as explicitly unclassified history. Typed totals for destroyed
    // work cannot be recovered and stay marked incomplete.
    runtime->served_by_category_ = retained_execution_totals;
    runtime->accounting_history_complete_ = false;
    for (std::size_t row = 0; row < 2; ++row) {
      for (std::size_t direction = 0; direction < 2; ++direction) {
        long double known = 0.0L;
        for (const ServedCounters& item : retained_execution_totals) {
          known += item.hbm_bytes[row][direction];
        }
        const long double total = runtime->hbm_served_[row][direction];
        const long double unknown = total - known;
        if (unknown < -kCounterTolerance * std::max(1.0L, total)) {
          throw std::invalid_argument(
              "invalid_snapshot: legacy HBM total is below retained execution work");
        }
        runtime->legacy_unclassified_hbm_[row][direction] =
            std::max(0.0L, unknown);
      }
    }
  } else {
    long double unclassified_hbm_total = 0.0L;
    for (std::size_t row = 0; row < 2; ++row) {
      for (std::size_t direction = 0; direction < 2; ++direction) {
        long double categorized_total = 0.0L;
        for (std::size_t index = 0; index < served_category_count; ++index) {
          const long double actual = runtime->served_by_category_[index]
                                         .hbm_bytes[row][direction];
          const long double retained = retained_execution_totals[index]
                                          .hbm_bytes[row][direction];
          if (actual + kCounterTolerance * std::max(1.0L, retained) <
              retained) {
            throw std::invalid_argument(
                "invalid_snapshot: categorized HBM total is below retained executions");
          }
          categorized_total += actual;
        }
        const long double unclassified =
            runtime->legacy_unclassified_hbm_[row][direction];
        if (!close_enough(categorized_total + unclassified,
                          runtime->hbm_served_[row][direction])) {
          throw std::invalid_argument(
              "invalid_snapshot: categorized and unclassified HBM totals do not conserve global served bytes");
        }
        unclassified_hbm_total += unclassified;
      }
    }
    for (std::size_t index = 0; index < served_category_count; ++index) {
      const ServedCounters& actual = runtime->served_by_category_[index];
      const ServedCounters& retained = retained_execution_totals[index];
      if (actual.int4_ops + kCounterTolerance *
              std::max(1.0L, retained.int4_ops) < retained.int4_ops ||
          actual.vec_eqops + kCounterTolerance *
              std::max(1.0L, retained.vec_eqops) < retained.vec_eqops) {
        throw std::invalid_argument(
            "invalid_snapshot: categorized operation total is below retained executions");
      }
    }
    if (runtime->accounting_history_complete_ &&
        unclassified_hbm_total > kCounterTolerance) {
      throw std::invalid_argument(
          "invalid_snapshot: complete accounting history has unclassified HBM bytes");
    }
  }

  // Saved rates are diagnostic only. Rebuild them from the frozen live set.
  for (auto& [handle, execution] : runtime->executions_) {
    (void)handle;
    for (LiveActivity& live : execution.activities) {
      live.rate = 0.0;
      live.alpha = 0.0;
    }
  }
  const std::uint64_t saved_generation = runtime->generation_;
  const std::uint64_t saved_allocations = runtime->allocation_updates_;
  const std::uint64_t saved_events = runtime->internal_events_;
  runtime->settle();
  if (runtime->internal_events_ != saved_events) {
    throw std::invalid_argument("invalid_snapshot: saved state was not settled at its event boundary");
  }
  runtime->allocate();
  runtime->generation_ = saved_generation;
  runtime->allocation_updates_ = saved_allocations;
  return runtime;
}

bool Runtime::drained() const {
  return std::all_of(executions_.begin(), executions_.end(),
                     [](const auto& entry) { return entry.second.terminal; });
}

void Runtime::destroy_execution(std::uint64_t handle) {
  auto found = executions_.find(handle);
  if (found == executions_.end()) throw std::out_of_range("stale_handle: unknown execution handle");
  if (!found->second.terminal) throw std::runtime_error("busy: cancel active execution before destroy");
  const auto& task = found->second.plan->task;
  if (task.is_object() && task.value("schema", std::string()) == "coarse_die_endpoint_v2" &&
      task.contains("endpoint")) {
    const auto& endpoint = task.at("endpoint");
    const auto endpoint_id = parse_u64(endpoint.at("endpoint_id"), "endpoint.endpoint_id");
    const auto transaction_id = parse_u64(endpoint.at("transaction_id"), "endpoint.transaction_id");
    endpoint_registry_.erase(endpoint_id);
    auto transaction = transaction_owners_.find(transaction_id);
    if (transaction != transaction_owners_.end()) {
      if (transaction->second.second > 1) {
        --transaction->second.second;
      } else {
        transaction_owners_.erase(transaction);
      }
    }
  }
  executions_.erase(found);
}

std::vector<std::uint64_t> Runtime::execution_handles() const {
  std::vector<std::uint64_t> result;
  result.reserve(executions_.size());
  for (const auto& [handle, execution] : executions_) {
    (void)execution;
    result.push_back(handle);
  }
  return result;
}

namespace {

struct StandaloneEstimateRun {
  std::uint64_t finish_ns = 0;
  nlohmann::json progress;
  nlohmann::json runtime_counters;
};

StandaloneEstimateRun drive_submitted_standalone_estimate(
    Runtime& runtime, std::uint64_t handle, bool capture_report) {
  std::size_t event_guard = 0;
  while (!runtime.drained()) {
    const auto next = runtime.next_event();
    if (!next) {
      throw std::runtime_error("invalid_hardware: standalone plan has no positive service rate");
    }
    if (*next <= runtime.now_ns()) {
      throw std::runtime_error("numerical_failure: next_event did not advance time");
    }
    runtime.advance_to(*next);
    if (++event_guard > kMaxAdvanceEvents) {
      throw std::runtime_error("numerical_failure: estimate_alone exceeded event budget");
    }
  }

  // Read the terminal time without constructing the large public progress
  // object. Candidate ranking needs only this typed value; public standalone
  // estimates still collect the full report once.
  const auto finish = runtime.terminal_finish_ns(handle);
  if (!finish) {
    throw std::runtime_error(
        "numerical_failure: standalone plan drained without finishing");
  }
  StandaloneEstimateRun result;
  result.finish_ns = *finish;
  if (capture_report) {
    result.progress = runtime.query_progress(handle);
    if (result.progress.at("state").get<std::string>() != "finished" ||
        parse_u64(result.progress.at("finish_ns"), "finish_ns") != *finish) {
      throw std::runtime_error(
          "numerical_failure: standalone terminal report is inconsistent");
    }
    result.runtime_counters = runtime.counters();
  }
  return result;
}

StandaloneEstimateRun drive_standalone_estimate(
    std::shared_ptr<const Hardware> hardware,
    std::shared_ptr<const Plan> plan, bool capture_report) {
  Runtime runtime(std::move(hardware));
  const std::uint64_t handle = runtime.submit(
      std::move(plan), Owner{1, 1, 1, 1}, 0);
  return drive_submitted_standalone_estimate(runtime, handle, capture_report);
}

}  // namespace

namespace detail {

std::uint64_t estimate_candidate_latency_ns(
    std::shared_ptr<const Hardware> hardware,
    std::shared_ptr<const Plan> plan) {
  if (!plan) throw std::invalid_argument("invalid_plan: null candidate plan");
  validate_plan_structure(*plan);
  Runtime runtime(std::move(hardware));
  const auto handle = runtime.submit_impl(
      std::move(plan), Owner{1, 1, 1, 1}, 0, false);
  return drive_submitted_standalone_estimate(runtime, handle, false).finish_ns;
}

std::uint64_t estimate_latency_ns(const Hardware& hardware,
                                  std::shared_ptr<const Plan> plan) {
  return estimate_latency_ns(std::make_shared<const Hardware>(hardware),
                             std::move(plan));
}

std::uint64_t estimate_latency_ns(std::shared_ptr<const Hardware> hardware,
                                  std::shared_ptr<const Plan> plan) {
  return drive_standalone_estimate(std::move(hardware), std::move(plan), false)
      .finish_ns;
}

}  // namespace detail

nlohmann::json estimate_alone(const Hardware& hardware, const Plan& plan) {
  const nlohmann::json serialized_plan = plan_to_json(plan);
  auto shared_plan = std::make_shared<const Plan>(plan);
  StandaloneEstimateRun estimate = drive_standalone_estimate(
      std::make_shared<const Hardware>(hardware), std::move(shared_plan), true);
  const std::uint64_t finish = estimate.finish_ns;
  nlohmann::json result = plan.summary.is_object() ? plan.summary : nlohmann::json::object();
  result["status"] = "ok_assumed";
  result["estimated_latency_ns"] = finish;
  result["hardware_fingerprint"] = hardware.fingerprint;
  result["plan_fingerprint"] = serialized_plan.at("fingerprint");
  const auto& performance = hardware.input.is_object()
      ? hardware.input.value("performance_profile", nlohmann::json::object())
      : nlohmann::json::object();
  const auto profile_string = [&](const char* key) -> std::string {
    if (performance.is_object()) {
      const auto it = performance.find(key);
      if (it != performance.end() && it->is_string()) return it->get<std::string>();
    }
    return "unspecified";
  };
  const auto claimed_calibration = profile_string("calibration_status");
  const auto claimed_granularity = profile_string("granularity_validation");
  const auto profile_calibration_revision = profile_string("calibration_revision");
  result["calibration_revision"] = profile_calibration_revision == "unspecified"
      ? nlohmann::json(nullptr) : nlohmann::json(profile_calibration_revision);
  result["hardware_source_revision"] = hardware.derived.is_object()
      ? hardware.derived.value("source_revision", nlohmann::json(nullptr))
      : nlohmann::json(nullptr);
  // Input strings are retained as claims for audit only. This engine has no
  // signed calibration or granularity-validation manifest to verify them.
  result["calibration_status"] = "assumed";
  result["calibration_claimed_status"] = claimed_calibration;
  result["calibration_source_flags"] = {
      {"performance_profile_present", performance.is_object() && !performance.empty()},
      {"calibration_revision_present", profile_calibration_revision != "unspecified"},
      {"certificate_manifest_present", false},
      {"verified", false}};
  result["topology_identity"] = {
      {"profile", profile_string("timing_topology_profile")},
      {"status", "assumed"},
      {"physical_validation", "unverified"}};
  result["routing_identity"] = {
      {"profile", profile_string("routing_profile")},
      {"status", "assumed"}};
  result["pipeline_identity"] = {
      {"profile", profile_string("pipeline_profile")},
      {"status", "assumed"}};
  result["granularity_identity"] = {
      {"schema", has_fractional_work(plan) ? "coarse_die_plan_v3" : "coarse_die_plan_v2"},
      {"epoch_count", plan.epochs.size()},
      {"activity_limit_per_epoch", 3},
      {"repeat_count_semantics", "audit_metadata_only"},
      {"claimed_validation", claimed_granularity},
      {"validation_status", "not_run"},
      {"heldout_status", "not_available"},
      {"ranking_status", "not_tested"}};
  if (has_fractional_work(plan)) {
    result["plan_schema"] = "coarse_die_plan_v3";
    result["equivalent_work_schema"] = "dyadic_equivalent_work_v1";
  }
  result["granularity_validation"] = "not_run";
  result["heldout_status"] = "not_available";
  result["ranking_status"] = "not_tested";
  result["derived_hardware"] = hardware.derived;
  static constexpr const char* resource_names[resource_count] = {
      "COMPUTE_GATE", "SRAM_READ", "SRAM_WRITE", "LOCAL", "HBM0",
      "HBM1", "INJECT0", "INJECT1", "XPOS", "XNEG", "YPOS",
      "YNEG", "HOP"};
  std::array<long double, resource_count> aggregate_work{};
  for (const Epoch& epoch : plan.epochs) {
    for (const Activity& activity : epoch.activities) {
      for (std::size_t resource = 0; resource < resource_count; ++resource) {
        aggregate_work[resource] += static_cast<long double>(activity.work[resource]);
      }
    }
  }
  nlohmann::json resource_domains = nlohmann::json::array();
  nlohmann::json capacity_vector = nlohmann::json::array();
  nlohmann::json work_vector = nlohmann::json::array();
  for (std::size_t resource = 0; resource < resource_count; ++resource) {
    if (!std::isfinite(aggregate_work[resource]) ||
        aggregate_work[resource] > std::numeric_limits<double>::max()) {
      throw std::overflow_error("estimate_alone: aggregate resource work overflow");
    }
    const double work = static_cast<double>(aggregate_work[resource]);
    capacity_vector.push_back(hardware.rates[resource]);
    work_vector.push_back(work);
    std::string unit = "unknown";
    bool enabled = hardware.rates[resource] > 0.0;
    if (hardware.derived.is_object() && hardware.derived.contains("resources") &&
        hardware.derived["resources"].is_array() &&
        hardware.derived["resources"].size() == resource_count &&
        hardware.derived["resources"][resource].is_object()) {
      const auto& descriptor = hardware.derived["resources"][resource];
      if (descriptor.contains("unit") && descriptor["unit"].is_string())
        unit = descriptor["unit"].get<std::string>();
      if (descriptor.contains("enabled") && descriptor["enabled"].is_boolean())
        enabled = descriptor["enabled"].get<bool>();
    }
    resource_domains.push_back({
        {"resource_id", resource}, {"id", resource_names[resource]},
        {"enabled", enabled}, {"unit", unit},
        {"capacity_per_s", hardware.rates[resource]}, {"planned_work", work}});
  }
  result["resource_domains"] = std::move(resource_domains);
  result["domain_capacities"] = std::move(capacity_vector);
  result["domain_work"] = std::move(work_vector);
  result["execution"] = std::move(estimate.progress);
  result["runtime_counters"] = std::move(estimate.runtime_counters);
  return result;
}

}  // namespace coarse_die
